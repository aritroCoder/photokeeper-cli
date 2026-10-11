import cv2
import numpy as np


def blur(gray):
    """Variance of the Laplacian, low means out of focus or motion blurred."""
    return float(cv2.Laplacian(gray, cv2.CV_16S).var())


def clipped(gray):
    """Fraction of near-black or near-white pixels, high means pocket shot or blown out."""
    return 1 - np.count_nonzero(cv2.inRange(gray, 8, 247)) / gray.size
