#include "pipeline.h"

#include <algorithm>
#include <cmath>

#include <opencv2/imgproc.hpp>

namespace pk {

const char* const SIGNAL_NAMES[N] = {"faces", "palette_flatness", "axis_edge_ratio", "clipped", "text_area_ratio"};
std::string MODELS = PK_MODELS;

const Signal SIGNALS[N] = {
    [](const cv::Mat& bgr, const cv::Mat&) { return double(faces(bgr)); },
    [](const cv::Mat& bgr, const cv::Mat&) { return palette_flatness(bgr); },
    [](const cv::Mat&, const cv::Mat& gray) { return axis_edge_ratio(gray); },
    [](const cv::Mat&, const cv::Mat& gray) { return clipped(gray); },
    [](const cv::Mat& bgr, const cv::Mat&) { return text_area_ratio(bgr); },
};

cv::Mat thumbnail(const cv::Mat& bgr) {
    double scale = double(SIDE) / std::max(bgr.rows, bgr.cols);
    if (scale >= 1) return bgr;
    cv::Mat out;
    cv::resize(bgr, out, {cvRound(bgr.cols * scale), cvRound(bgr.rows * scale)}, 0, 0, cv::INTER_AREA);
    return out;
}

Features analyze(const cv::Mat& full) {
    cv::Mat bgr = thumbnail(full), gray;
    cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
    Features f;
    for (int i = 0; i < N; ++i) {
        try {
            f[i] = SIGNALS[i](bgr, gray);
        } catch (const cv::Exception&) {  // OpenCV rejected the pixels; keep() treats NaN as neutral
            f[i] = NAN;
        }
    }
    return f;
}

static double clip(double x) { return std::min(std::max(x, 0.0), 1.0); }

// per signal: map to a keep-likeness in [0, 1], and its weight; weights sum to 1.
// Ramps come from the keep/delete quantiles on dataset/. Text carries 0.4 so a caption can outvote the
// photo-like features of a meme; a logistic fit on the same features reaches 93.5%, these give 93.4%
// (screenshots/documents 99.5%, photos 92%, memes 84%). Weights and threshold interact: text 0.4 at
// threshold 0.5 lets screenshots through, so change them as a pair against `--eval`.
// A blur signal (variance of Laplacian, weight 0.03) was dropped: dataset/ has no blurry class and
// removing it did not change the eval; the other weights were rescaled by 1/0.97.
static const struct {
    double (*norm)(double);
    double weight;
} FEATURES[N] = {
    {[](double v) { return std::min(v, 1.0); }, 0.082},
    {[](double v) { return clip((0.9 - v) / 0.4); }, 0.196},
    {[](double v) { return clip((0.6 - v) / 0.4); }, 0.196},
    {[](double v) { return clip(1 - v / 0.3); }, 0.113},
    {[](double v) { return clip(1 - v / 0.05); }, 0.413},
};
constexpr double KEEP_THRESHOLD = 0.55;

bool keep(const Features& f) {
    double score = 0;
    for (int i = 0; i < N; ++i) score += FEATURES[i].weight * (std::isnan(f[i]) ? 0.5 : FEATURES[i].norm(f[i]));
    return score >= KEEP_THRESHOLD;
}

}  // namespace pk
