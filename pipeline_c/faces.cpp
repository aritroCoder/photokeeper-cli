#include <opencv2/objdetect.hpp>

#include "pipeline.h"

namespace pk {

// Number of detected faces. Any face is a strong keep signal.
int faces(const cv::Mat& bgr) {
    thread_local auto det = cv::FaceDetectorYN::create(PK_MODELS "/face_detection_yunet_2023mar.onnx", "", {320, 320});
    det->setInputSize(bgr.size());
    cv::Mat found;
    det->detect(bgr, found);
    return found.rows;
}

}  // namespace pk
