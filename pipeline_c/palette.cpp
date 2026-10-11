#include <algorithm>
#include <numeric>

#include "pipeline.h"

namespace pk {

constexpr int TOP = 16;  // how many most-common colors count as "the palette"

// Fraction of pixels in the TOP most common colors after 4-bit/channel quantization.
// Screenshots and documents sit near 1, natural photos much lower.
double palette_flatness(const cv::Mat& bgr) {
    std::array<int, 4096> counts{};
    for (int y = 0; y < bgr.rows; ++y)
        for (const cv::Vec3b *p = bgr.ptr<cv::Vec3b>(y), *end = p + bgr.cols; p != end; ++p)
            ++counts[((*p)[0] >> 4) << 8 | ((*p)[1] >> 4) << 4 | ((*p)[2] >> 4)];
    std::nth_element(counts.begin(), counts.end() - TOP, counts.end());
    return std::accumulate(counts.end() - TOP, counts.end(), 0.0) / double(bgr.total());
}

}  // namespace pk
