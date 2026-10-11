#pragma once
#include <array>
#include <string>

#include <opencv2/core.hpp>

namespace pk {

constexpr int SIDE = 320;  // long edge of the working thumbnail; every signal runs on it
constexpr int N = 6;
extern const char* const SIGNAL_NAMES[N];
using Features = std::array<double, N>;  // in SIGNAL_NAMES order; NaN marks a signal that threw

// Signals. bgr is 8-bit BGR, gray its single-channel version. The two detectors are shared behind locks.
int faces(const cv::Mat& bgr);
double palette_flatness(const cv::Mat& bgr);
double axis_edge_ratio(const cv::Mat& gray);
double blur(const cv::Mat& gray);
double clipped(const cv::Mat& gray);
double text_area_ratio(const cv::Mat& bgr);

cv::Mat load(const std::string& path);  // decode with codec-side downscale, long edge stays >= SIDE; empty if undecodable
cv::Mat thumbnail(const cv::Mat& bgr);  // shrink so the long edge is at most SIDE
Features analyze(const cv::Mat& bgr);   // every signal on thumbnail(bgr)
bool keep(const Features& f);           // false means delete

}  // namespace pk
