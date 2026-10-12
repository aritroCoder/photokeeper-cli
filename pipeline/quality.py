import cv2
import numpy as np


def clipped(gray):
    """Fraction of near-black or near-white pixels, high means pocket shot or blown out."""
    return 1 - np.count_nonzero(cv2.inRange(gray, 8, 247)) / gray.size
