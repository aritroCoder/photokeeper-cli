// Run every signal on images.
// Build: cmake -B build && cmake --build build      Usage: build/photokeeper [-j WORKERS] [--eval [MANIFEST]] photo.jpg dir ...
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <mutex>
#include <optional>
#include <queue>
#include <sstream>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "pipeline_c/pipeline.h"

namespace fs = std::filesystem;

// Blocking queue that pops the smallest (image, stage) first, like Python's queue.PriorityQueue.
template <class T>
struct PriorityQueue {
    std::priority_queue<T, std::vector<T>, std::greater<T>> q;
    std::mutex m;
    std::condition_variable cv;
    void push(T v) {
        {
            std::lock_guard<std::mutex> lock(m);
            q.push(std::move(v));
        }
        cv.notify_one();
    }
    T pop() {
        std::unique_lock<std::mutex> lock(m);
        cv.wait(lock, [&] { return !q.empty(); });
        T v = q.top();
        q.pop();
        return v;
    }
};

struct Task {
    size_t image;
    int stage;  // 0 decodes paths[image]; k >= 1 runs SIGNALS[k - 1] on the decoded thumbnail
    std::function<double()> job;
    bool operator>(const Task& o) const { return std::tie(image, stage) > std::tie(o.image, o.stage); }
};

struct Msg {
    size_t image;
    int stage;  // 0 means the decode failed, the only message that image will send
    double value;
    bool operator>(const Msg& o) const { return std::tie(image, stage) > std::tie(o.image, o.stage); }
};

// Call on_result(index, features) as images finish, in any order. features is empty if decoding failed,
// otherwise NaN marks a signal that threw.
//
// One priority queue of (image index, stage) tasks. Loads are queued up front; an image's signal
// tasks are queued when its decode finishes and sort ahead of later loads, so decoded thumbnails
// in flight stay near the worker count no matter how many paths are given.
static void analyze_all(const std::vector<std::string>& paths, int workers,
                        const std::function<void(size_t, std::optional<pk::Features>)>& on_result) {
    PriorityQueue<Task> tasks;
    PriorityQueue<Msg> out;
    std::atomic<bool> stop{false};
    for (size_t i = 0; i < paths.size(); ++i) tasks.push({i, 0, {}});

    auto work = [&] {
        while (true) {
            Task t = tasks.pop();
            if (stop) return;
            if (t.stage == 0) {
                cv::Mat bgr, gray;
                try {
                    bgr = pk::load(paths[t.image]);
                } catch (const cv::Exception&) {  // undecodable
                }
                if (bgr.empty()) {
                    out.push({t.image, 0, NAN});
                    continue;
                }
                cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
                for (int k = 0; k < pk::N; ++k) tasks.push({t.image, k + 1, [=] { return pk::SIGNALS[k](bgr, gray); }});
            } else {
                double value;
                try {
                    value = t.job();
                } catch (const cv::Exception&) {  // OpenCV rejected the pixels
                    value = NAN;
                }
                out.push({t.image, t.stage, value});
            }
        }
    };

    std::vector<std::thread> threads;
    for (int w = 0; w < workers; ++w) threads.emplace_back(work);
    std::map<size_t, std::pair<int, pk::Features>> results;  // signals received so far, values
    for (size_t finished = 0; finished < paths.size();) {
        Msg m = out.pop();
        if (m.stage == 0) {
            ++finished;
            on_result(m.image, std::nullopt);
            continue;
        }
        auto& [count, features] = results[m.image];
        features[m.stage - 1] = m.value;
        if (++count == pk::N) {
            ++finished;
            on_result(m.image, features);
            results.erase(m.image);
        }
    }
    stop = true;  // one wake-up item per worker, each exits on stop
    for (size_t w = 0; w < threads.size(); ++w) tasks.push({SIZE_MAX, 0, {}});
    for (auto& t : threads) t.join();
}

// Classify every image in the manifest; print the confusion matrix plus per-source accuracy.
// Undecodable images count as 'delete'.
static void run_eval(const std::string& manifest, int workers) {
    fs::path root = fs::path(manifest).parent_path();
    std::ifstream in(manifest);
    std::string line;
    std::getline(in, line);                          // header: path,label,source,location
    std::vector<std::array<std::string, 3>> rows;    // path, label, source
    while (std::getline(in, line)) {
        std::stringstream ss(line);
        std::array<std::string, 3> r;
        for (auto& field : r) std::getline(ss, field, ',');  // shortcut: no quoted fields, enough for dataset/manifest.csv
        rows.push_back(r);
    }
    std::vector<std::string> paths;
    for (const auto& r : rows) paths.push_back((root / r[0]).string());
    std::vector<std::string> predicted(rows.size());
    analyze_all(paths, workers, [&](size_t i, std::optional<pk::Features> f) {
        predicted[i] = f && pk::keep(*f) ? "keep" : "delete";
    });
    std::map<std::pair<std::string, std::string>, int> matrix;
    std::map<std::string, std::pair<int, int>> by_source;  // hits, total
    for (size_t i = 0; i < rows.size(); ++i) {
        const auto& [path, label, source] = rows[i];
        matrix[{label, predicted[i]}]++;
        by_source[source].first += label == predicted[i];
        by_source[source].second++;
    }
    std::printf("%20s %6s %7s\n", "actual \\ predicted", "keep", "delete");
    for (const char* actual : {"keep", "delete"})
        std::printf("%20s %6d %7d\n", actual, matrix[{actual, "keep"}], matrix[{actual, "delete"}]);
    std::printf("accuracy %.3f\n", (matrix[{"keep", "keep"}] + matrix[{"delete", "delete"}]) / double(rows.size()));
    for (const auto& [source, hits] : by_source) std::printf("  %-36s %.2f\n", source.c_str(), hits.first / double(hits.second));
}

int main(int argc, char** argv) {
    int workers = std::thread::hardware_concurrency() ? std::thread::hardware_concurrency() : 4;
    std::string manifest;
    std::vector<std::string> paths;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if ((a == "-j" || a == "--workers") && i + 1 < argc) {
            workers = std::stoi(argv[++i]);
        } else if (a == "--eval") {
            manifest = i + 1 < argc && argv[i + 1][0] != '-' ? argv[++i] : "dataset/manifest.csv";
        } else if (fs::is_directory(a)) {
            std::vector<std::string> found;
            for (const auto& e : fs::recursive_directory_iterator(a))
                if (e.is_regular_file() && cv::haveImageReader(e.path().string())) found.push_back(e.path().string());
            std::sort(found.begin(), found.end());
            paths.insert(paths.end(), found.begin(), found.end());
        } else {
            paths.push_back(a);
        }
    }
    if (!manifest.empty()) {
        run_eval(manifest, workers);
        return 0;
    }
    auto t0 = std::chrono::steady_clock::now();
    analyze_all(paths, workers, [&](size_t i, std::optional<pk::Features> f) {
        if (!f) {
            std::printf("error\t%s\tcannot decode\n", paths[i].c_str());
            return;
        }
        std::string line = (pk::keep(*f) ? "keep\t" : "delete\t") + paths[i];
        for (int k = 0; k < pk::N; ++k) line += (k ? " " : "\t") + std::string(pk::SIGNAL_NAMES[k]) + "=" + std::to_string((*f)[k]);
        std::puts(line.c_str());
    });
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    std::printf("%zu images, %d workers: %.0fms total, %.1fms/img\n", paths.size(), workers, ms, ms / std::max<size_t>(paths.size(), 1));
}
