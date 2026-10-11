// Synthetic sanity checks. Run: build/test_pipeline
#undef NDEBUG
#include <cassert>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <thread>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "../pipeline_c/pipeline.h"

int main() {
    cv::theRNG() = cv::RNG(0);
    cv::Mat photo(240, 320, CV_8UC3);
    cv::randu(photo, cv::Scalar::all(0), cv::Scalar::all(256));
    cv::GaussianBlur(photo, photo, {}, 2);
    cv::Mat screenshot(240, 320, CV_8UC3, cv::Scalar::all(255));
    cv::rectangle(screenshot, {10, 10}, {310, 60}, cv::Scalar::all(40), -1);
    for (int y = 100; y < 230; y += 25)
        cv::putText(screenshot, "Lorem ipsum dolor sit amet", {10, y}, cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar::all(0), 2);
    cv::Mat gray_photo, gray_shot, blurred;
    cv::cvtColor(photo, gray_photo, cv::COLOR_BGR2GRAY);
    cv::cvtColor(screenshot, gray_shot, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gray_photo, blurred, {}, 6);

    assert(pk::palette_flatness(screenshot) > 0.9 && 0.9 > pk::palette_flatness(photo));
    assert(pk::axis_edge_ratio(gray_shot) > pk::axis_edge_ratio(gray_photo));
    assert(pk::blur(gray_photo) > pk::blur(blurred));
    assert(pk::clipped(cv::Mat::zeros(240, 320, CV_8U)) == 1.0);
    assert(pk::text_area_ratio(screenshot) > 0.1 && 0.1 > pk::text_area_ratio(photo));
    assert(pk::faces(photo) == 0);

    // thumbnails land at SIDE regardless of source size or format (reduced decode must not undershoot)
    std::string dir = (std::filesystem::temp_directory_path() / "pk_test").string();
    std::filesystem::create_directory(dir);
    for (cv::Size size : {cv::Size(375, 500), cv::Size(900, 1200), cv::Size(3000, 4000)})
        for (const char* ext : {".jpg", ".png", ".tiff"}) {
            cv::Mat big;
            cv::resize(photo, big, size);
            std::string path = dir + "/img" + ext;
            cv::imwrite(path, big);
            cv::Mat loaded = pk::load(path);
            assert(std::max(loaded.rows, loaded.cols) >= pk::SIDE);
            assert(std::max(loaded.rows, loaded.cols) < 2 * pk::SIDE || size.height < 2 * pk::SIDE);
            assert(std::max(pk::thumbnail(loaded).rows, pk::thumbnail(loaded).cols) == pk::SIDE);
        }
    std::filesystem::remove_all(dir);

    pk::Features shot = pk::analyze(screenshot);
    assert(pk::keep(pk::analyze(photo)) && !pk::keep(shot));
    shot[5] = NAN;  // a failed signal is neutral, the rest still says delete
    assert(!pk::keep(shot));

    // detectors are per thread: concurrent calls agree with the serial verdicts
    std::vector<std::thread> threads;
    for (int t = 0; t < 4; ++t)
        threads.emplace_back([&] { assert(pk::keep(pk::analyze(photo)) && !pk::keep(pk::analyze(screenshot))); });
    for (auto& t : threads) t.join();
    std::puts("ok");
}
