#pragma once
#include <array>
#include <string>

#include <opencv2/core.hpp>

namespace pk {

constexpr int SIDE = 320;  // long edge of the working thumbnail; every signal runs on it
constexpr int N = 5;
extern const char* const SIGNAL_NAMES[N];
using Features = std::array<double, N>;  // in SIGNAL_NAMES order; NaN marks a signal that threw
extern std::string MODELS;  // directory of the two ONNX files; set before the first detector call (see pk.h)

// Signals. bgr is 8-bit BGR, gray its single-channel version. The two detectors are shared behind locks.
int faces(const cv::Mat& bgr);
double palette_flatness(const cv::Mat& bgr);
double axis_edge_ratio(const cv::Mat& gray);
double clipped(const cv::Mat& gray);
double text_area_ratio(const cv::Mat& bgr);

// signal table in SIGNAL_NAMES order; doubles as task priority within an image: decisive and cheap first, slowest last
using Signal = double (*)(const cv::Mat& bgr, const cv::Mat& gray);
extern const Signal SIGNALS[N];

cv::Mat load(const std::string& path);  // decode to a SIDE-px thumbnail; empty if undecodable
cv::Mat thumbnail(const cv::Mat& bgr);  // shrink so the long edge is at most SIDE
Features analyze(const cv::Mat& bgr);   // every signal on thumbnail(bgr)
bool keep(const Features& f);           // false means delete

}  // namespace pk
