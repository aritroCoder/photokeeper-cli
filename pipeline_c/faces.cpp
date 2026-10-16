#include <mutex>

#include <opencv2/objdetect.hpp>

#include "pipeline.h"

namespace pk {

// Number of detected faces. Any face is a strong keep signal.
int faces(const cv::Mat& bgr) {
    static auto det = cv::FaceDetectorYN::create(MODELS + "/face_detection_yunet_2023mar.onnx", "", {320, 320});
    static std::mutex lock;  // detector is not reentrant
    std::lock_guard<std::mutex> guard(lock);
    det->setInputSize(bgr.size());
    cv::Mat found;
    det->detect(bgr, found);
    return found.rows;
}

}  // namespace pk
