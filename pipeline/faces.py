import threading
from pathlib import Path

import cv2

MODEL = Path(__file__).resolve().parent.parent / "models" / "face_detection_yunet_2023mar.onnx"
_det = cv2.FaceDetectorYN.create(str(MODEL), "", (320, 320))
_lock = threading.Lock()  # detector is not reentrant


def count(bgr):
    """Number of detected faces. Any face is a strong keep signal."""
    with _lock:
        _det.setInputSize((bgr.shape[1], bgr.shape[0]))
        _, found = _det.detect(bgr)
    return 0 if found is None else len(found)
