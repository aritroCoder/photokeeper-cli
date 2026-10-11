#include <opencv2/imgproc.hpp>

#include "pipeline.h"

namespace pk {

// Variance of the Laplacian, low means out of focus or motion blurred.
double blur(const cv::Mat& gray) {
    cv::Mat lap;
    cv::Laplacian(gray, lap, CV_16S);
    cv::Scalar mean, stddev;
    cv::meanStdDev(lap, mean, stddev);
    return stddev[0] * stddev[0];
}

// Fraction of near-black or near-white pixels, high means pocket shot or blown out.
double clipped(const cv::Mat& gray) {
    cv::Mat mid;
    cv::inRange(gray, cv::Scalar(8), cv::Scalar(247), mid);
    return 1 - cv::countNonZero(mid) / double(gray.total());
}

}  // namespace pk
