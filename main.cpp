// Run every signal on images.
// Build: cmake -B build && cmake --build build      Usage: build/photokeeper [-j WORKERS] [--eval [MANIFEST]] photo.jpg dir ...
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/imgcodecs.hpp>

#include "pipeline_c/pipeline.h"

namespace fs = std::filesystem;

// Each worker takes an image end to end, so decoded thumbnails in flight never exceed the worker count.
static void for_each_parallel(size_t n, int workers, const std::function<void(size_t)>& fn) {
    std::atomic<size_t> next{0};
    std::vector<std::thread> pool;
    for (int w = 0; w < workers; ++w)
        pool.emplace_back([&] {
            for (size_t i; (i = next++) < n;) fn(i);
        });
    for (auto& t : pool) t.join();
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
    std::vector<std::string> predicted(rows.size());
    for_each_parallel(rows.size(), workers, [&](size_t i) {
        cv::Mat img = pk::load((root / rows[i][0]).string());
        predicted[i] = !img.empty() && pk::keep(pk::analyze(img)) ? "keep" : "delete";
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
    std::mutex out;
    for_each_parallel(paths.size(), workers, [&](size_t i) {
        cv::Mat img = pk::load(paths[i]);
        std::string line;
        if (img.empty()) {
            line = "error\t" + paths[i] + "\tcannot decode";
        } else {
            pk::Features f = pk::analyze(img);
            line = (pk::keep(f) ? "keep\t" : "delete\t") + paths[i];
            for (int k = 0; k < pk::N; ++k) line += (k ? " " : "\t") + std::string(pk::SIGNAL_NAMES[k]) + "=" + std::to_string(f[k]);
        }
        std::lock_guard<std::mutex> lock(out);
        std::puts(line.c_str());
    });
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    std::printf("%zu images, %d workers: %.0fms total, %.1fms/img\n", paths.size(), workers, ms, ms / std::max<size_t>(paths.size(), 1));
}
