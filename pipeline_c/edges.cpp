#include <opencv2/imgproc.hpp>

#include "pipeline.h"

namespace pk {

constexpr int EDGE_MIN = 64;  // |gx| + |gy| above this is an edge pixel

// Fraction of edge pixels whose gradient is near-horizontal or near-vertical
// (weaker component under 1/5 of the stronger). UI and text are boxy, photos are not.
double axis_edge_ratio(const cv::Mat& gray) {
    cv::Mat gx, gy;
    cv::Sobel(gray, gx, CV_16S, 1, 0);
    cv::Sobel(gray, gy, CV_16S, 0, 1);
    gx = cv::abs(gx);
    gy = cv::abs(gy);
    cv::Mat edge = gx + gy > EDGE_MIN;
    cv::Mat aligned = cv::min(gx, gy) * 5 < cv::max(gx, gy);
    int n = cv::countNonZero(edge);
    return n ? cv::countNonZero(edge & aligned) / double(n) : 0.0;
}

}  // namespace pk
