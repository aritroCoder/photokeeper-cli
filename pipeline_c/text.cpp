#include <cmath>
#include <mutex>

#include <opencv2/dnn.hpp>

#include "pipeline.h"

namespace pk {

static double polygon_area(const std::vector<cv::Point>& pts) {  // shoelace; contourArea lives in different headers across OpenCV 4/5
    double s = 0;
    for (size_t i = 0; i < pts.size(); ++i) s += pts[i].cross(pts[(i + 1) % pts.size()]);
    return std::abs(s) / 2;
}

// Fraction of the image covered by detected text boxes.
double text_area_ratio(const cv::Mat& bgr) {
    static auto det = [] {
        cv::dnn::TextDetectionModel_DB d(PK_MODELS "/text_detection_en_ppocrv3_2023may.onnx");
        // 320 instead of the model's native 736: coarse boxes are enough for an area ratio and run ~5x faster.
        d.setInputParams(1 / 255.0, {320, 320}, {122.67891434, 116.66876762, 104.00698793});
        return d;
    }();
    static std::mutex lock;  // DNN buffers are not reentrant
    std::vector<std::vector<cv::Point>> quads;
    {
        std::lock_guard<std::mutex> guard(lock);
        det.detect(bgr, quads);
    }
    double area = 0;
    for (const auto& q : quads) area += polygon_area(q);
    return area / double(bgr.total());
}

}  // namespace pk
