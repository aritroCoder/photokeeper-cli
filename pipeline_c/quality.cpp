#include <opencv2/imgproc.hpp>

#include "pipeline.h"

namespace pk {

// Fraction of near-black or near-white pixels, high means pocket shot or blown out.
double clipped(const cv::Mat& gray) {
    cv::Mat mid;
    cv::inRange(gray, cv::Scalar(8), cv::Scalar(247), mid);
    return 1 - cv::countNonZero(mid) / double(gray.total());
}

}  // namespace pk
