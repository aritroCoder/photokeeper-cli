import cv2
import numpy as np

EDGE_MIN = 64  # |gx| + |gy| above this is an edge pixel


def axis_aligned_ratio(gray):
    """Fraction of edge pixels whose gradient is near-horizontal or near-vertical
    (weaker component under 1/5 of the stronger). UI and text are boxy, photos are not."""
    gx = np.abs(cv2.Sobel(gray, cv2.CV_16S, 1, 0))
    gy = np.abs(cv2.Sobel(gray, cv2.CV_16S, 0, 1))
    edge = gx + gy > EDGE_MIN
    aligned = np.minimum(gx, gy) * 5 < np.maximum(gx, gy)
    n = np.count_nonzero(edge)
    return np.count_nonzero(edge & aligned) / n if n else 0.0
