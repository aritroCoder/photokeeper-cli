import numpy as np

TOP = 16  # how many most-common colors count as "the palette"


def flatness(bgr):
    """Fraction of pixels in the TOP most common colors after 4-bit/channel quantization.
    Screenshots and documents sit near 1, natural photos much lower."""
    q = bgr >> 4
    idx = (q[..., 0].astype(np.uint16) << 8) | (q[..., 1] << 4) | q[..., 2]
    counts = np.bincount(idx.ravel(), minlength=4096)
    return np.partition(counts, -TOP)[-TOP:].sum() / idx.size
