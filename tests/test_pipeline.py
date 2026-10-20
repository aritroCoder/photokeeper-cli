"""Synthetic sanity checks. Run: uv run python -m tests.test_pipeline"""

import tempfile
import threading
from pathlib import Path

import cv2
import numpy as np

from main import analyze, analyze_all, classify_image_features, load
from pipeline import edges, faces, palette, quality, text

rng = np.random.default_rng(0)
photo = cv2.GaussianBlur(rng.integers(0, 256, (240, 320, 3), dtype=np.uint8), (0, 0), 2)
screenshot = np.full((240, 320, 3), 255, np.uint8)
cv2.rectangle(screenshot, (10, 10), (310, 60), (40, 40, 40), -1)
for i, y in enumerate(range(100, 230, 25)):
    cv2.putText(screenshot, "Lorem ipsum dolor sit amet", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
gray_photo, gray_shot = (cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in (photo, screenshot))

assert palette.flatness(screenshot) > 0.9 > palette.flatness(photo)
assert edges.axis_aligned_ratio(gray_shot) > edges.axis_aligned_ratio(gray_photo)
assert quality.clipped(np.zeros((240, 320), np.uint8)) == 1.0
assert text.area_ratio(screenshot) > 0.1 > text.area_ratio(photo)
assert faces.count(photo) == 0

with tempfile.TemporaryDirectory() as d:
    paths = [str(Path(d) / f"{i}.png") for i in range(8)] + [str(Path(d) / "bad.jpg")]
    for p in paths[:8]:
        cv2.imwrite(p, photo if p.endswith(("0.png", "2.png", "4.png", "6.png")) else screenshot)
    Path(paths[-1]).write_bytes(b"not an image")

    # thumbnails land at SIDE regardless of source size (reduced decode must not undershoot)
    for size in ((375, 500), (900, 1200), (3000, 4000)):
        cv2.imwrite(paths[0], cv2.resize(photo, size))
        assert max(load(paths[0])[0].shape) == 320, size
    cv2.imwrite(paths[0], photo)

    results = dict(analyze_all(paths, workers=4))
    assert all(results[p] == analyze(p) for p in paths[:8])
    assert isinstance(results[paths[-1]], Exception)
    assert [classify_image_features(results[p]) for p in paths[:2]] == ["keep", "delete"]
    assert classify_image_features({**results[paths[1]], "text_area_ratio": ValueError()}) == "delete"
    assert classify_image_features({**results[paths[0]], "axis_edge_ratio": 0.7}) == "delete"  # edges force delete

    # a failing signal yields its error in place and never swallows another image's result
    real = text.area_ratio

    def flaky(bgr):
        if palette.flatness(bgr) > 0.9:
            raise ValueError("boom")
        return real(bgr)

    text.area_ratio = flaky
    results = dict(analyze_all(paths, workers=4))
    text.area_ratio = real
    assert len(results) == len(paths)
    assert all(isinstance(results[p]["text_area_ratio"], ValueError) for p in paths[1:8:2])
    assert all(results[p] == analyze(p) for p in paths[0:8:2])

    # breaking out early stops and joins the workers
    before = threading.active_count()
    gen = analyze_all(paths * 20, workers=4)
    next(gen)
    gen.close()
    assert threading.active_count() == before
print("ok")
