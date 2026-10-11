import threading
from pathlib import Path

import cv2

MODEL = Path(__file__).resolve().parent.parent / "models" / "text_detection_en_ppocrv3_2023may.onnx"
_det = cv2.dnn.TextDetectionModel_DB(str(MODEL))
# 320 instead of the model's native 736: coarse boxes are enough for an area ratio and run ~5x faster.
_det.setInputParams(1 / 255, (320, 320), (122.67891434, 116.66876762, 104.00698793))
_lock = threading.Lock()  # DNN buffers are not reentrant


def area_ratio(bgr):
    """Fraction of the image covered by detected text boxes."""
    with _lock:
        quads, _ = _det.detect(bgr)
    h, w = bgr.shape[:2]
    return sum(cv2.contourArea(q) for q in quads) / (w * h)
