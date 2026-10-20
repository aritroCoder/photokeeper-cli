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
// Ramps come from the keep/delete quantiles on dataset/. Tuned for delete recall, since the app's review
// screen is where false positives get rescued: weights from a search minimising keeps flagged at 97.5%
// delete recall (held-out halves agree), threshold at the knee where recall stops rising. On `--eval`:
// delete recall 98.0%, precision 87.5% (was 94.8% / 94.2% at the accuracy-tuned 0.55). The misses left
// are memes whose caption the text detector does not find; no threshold catches those cheaply.
// Weights and threshold interact, so change them as a pair against `--eval`.
// A blur signal (variance of Laplacian) was dropped: dataset/ has no blurry class and it moved nothing.
static const struct {
    double (*norm)(double);
    double weight;
} FEATURES[N] = {
    {[](double v) { return std::min(v, 1.0); }, 0.03},
    {[](double v) { return clip((0.9 - v) / 0.4); }, 0.10},
    {[](double v) { return clip((0.6 - v) / 0.4); }, 0.19},
    {[](double v) { return clip(1 - v / 0.3); }, 0.13},
    {[](double v) { return clip(1 - v / 0.05); }, 0.55},
};
constexpr double KEEP_THRESHOLD = 0.78;
constexpr double SCREENSHOT_EDGES = 0.6;  // axis_edge_ratio where its ramp saturates

bool keep(const Features& f) {
    if (f[2] >= SCREENSHOT_EDGES) return false;  // a textless screenshot otherwise scores like a photo
    double score = 0;
    for (int i = 0; i < N; ++i) score += FEATURES[i].weight * (std::isnan(f[i]) ? 0.5 : FEATURES[i].norm(f[i]));
    return score >= KEEP_THRESHOLD;
}

}  // namespace pk
