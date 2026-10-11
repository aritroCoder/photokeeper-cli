"""Run every signal on images. Usage: uv run main.py [-j WORKERS] photo.jpg ..."""

import argparse
import csv
import os
import queue
import threading
import time
from collections import Counter
from functools import partial
from pathlib import Path

import cv2
import numpy as np
import pillow_heif
from PIL import Image

from pipeline import edges, faces, palette, quality, text

pillow_heif.register_heif_opener()  # lets Image.open read HEIC headers

SIDE = 320  # long edge of the working thumbnail; every signal runs on it
WORKERS = os.cpu_count() or 4
# codec-side downscale factors; JPEG skips the inverse DCT for dropped frequencies, so this is far cheaper than resize
REDUCED = {1: cv2.IMREAD_COLOR, 2: cv2.IMREAD_REDUCED_COLOR_2, 4: cv2.IMREAD_REDUCED_COLOR_4, 8: cv2.IMREAD_REDUCED_COLOR_8}

SIGNALS = (  # order doubles as task priority within an image: decisive and cheap first, slowest last
    ("faces", lambda bgr, gray: faces.count(bgr)),
    ("palette_flatness", lambda bgr, gray: palette.flatness(bgr)),
    ("axis_edge_ratio", lambda bgr, gray: edges.axis_aligned_ratio(gray)),
    ("blur", lambda bgr, gray: quality.blur(gray)),
    ("clipped", lambda bgr, gray: quality.clipped(gray)),
    ("text_area_ratio", lambda bgr, gray: text.area_ratio(bgr)),
)


def load_heif(path):
    img = pillow_heif.open_heif(path, bgr_mode=True)
    img = img[img.primary_index]
    # Phone HEICs embed a ~320px thumbnail; decoding it is ~20x cheaper than the HEVC main image.
    if img.info["thumbnails"]:
        img = img.get_thumbnail(0)
    bgr = np.asarray(img)
    return cv2.cvtColor(bgr, cv2.COLOR_BGRA2BGR) if bgr.shape[2] == 4 else bgr


def load(path):
    """Decode to a SIDE-px thumbnail, returned as (bgr, gray)."""
    with Image.open(path) as im:  # header only, no pixels decoded
        long_edge, fmt = max(im.size), im.format
    if fmt == "HEIF":
        bgr = load_heif(path)  # OpenCV wheels ship without libheif
    else:
        factor = 8 if long_edge >= 8 * SIDE else 4 if long_edge >= 4 * SIDE else 2 if long_edge >= 2 * SIDE else 1
        bgr = cv2.imread(path, REDUCED[factor])
        if bgr is None:
            raise ValueError(f"OpenCV cannot decode {path}")
    h, w = bgr.shape[:2]
    scale = SIDE / max(h, w)
    if scale < 1:
        bgr = cv2.resize(bgr, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    return bgr, cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def analyze(path):
    thumb = load(path)
    return {name: fn(*thumb) for name, fn in SIGNALS}


def clip(x):
    return min(max(x, 0.0), 1.0)


# per signal: map to a keep-likeness in [0, 1], and its weight; weights sum to 1.
# Ramps come from the keep/delete quantiles on dataset/. Text carries 0.4 so a caption can outvote the
# photo-like features of a meme; a logistic fit on the same features reaches 93.5%, these give 93.2%
# (screenshots/documents 99.5%, photos 92%, memes 83%). Weights and threshold interact: text 0.4 at
# threshold 0.5 lets screenshots through, so change them as a pair against `--eval`.
FEATURES = {
    "faces": (lambda v: min(v, 1), 0.08),
    "palette_flatness": (lambda v: clip((0.9 - v) / 0.4), 0.19),
    "axis_edge_ratio": (lambda v: clip((0.6 - v) / 0.4), 0.19),
    "blur": (lambda v: clip(v / 300), 0.03),
    "clipped": (lambda v: clip(1 - v / 0.3), 0.11),
    "text_area_ratio": (lambda v: clip(1 - v / 0.05), 0.40),
}
KEEP_THRESHOLD = 0.55


def classify_image_features(result):
    """'keep' or 'delete' from an analyze() dict. A signal that failed (Exception value) counts as neutral."""
    score = sum(w * (0.5 if isinstance(result[k], Exception) else norm(result[k])) for k, (norm, w) in FEATURES.items())
    return "keep" if score >= KEEP_THRESHOLD else "delete"


def analyze_all(paths, workers=WORKERS):
    """Yield (path, result) as images finish, in any order. result is the Exception if decoding
    failed, otherwise a dict whose values are numbers or the Exception that signal raised.

    One priority queue of (image index, stage) tasks. Loads are queued up front; an image's signal
    tasks are queued when its decode finishes and sort ahead of later loads, so decoded thumbnails
    in flight stay near the worker count no matter how many paths are given."""
    tasks = queue.PriorityQueue()
    out = queue.Queue()
    stop = threading.Event()
    for i, path in enumerate(paths):
        tasks.put((i, 0, path, partial(load, path)))

    def work():
        while True:
            i, stage, name, job = tasks.get()
            if stop.is_set():
                return
            try:
                value = job()
            except (OSError, ValueError, cv2.error) as e:  # unreadable, undecodable, or OpenCV rejected the pixels
                out.put((i, None if stage == 0 else name, e))
                continue
            if stage == 0:
                for stage, (name, fn) in enumerate(SIGNALS, 1):
                    tasks.put((i, stage, name, partial(fn, *value)))
            else:
                out.put((i, name, value))

    threads = [threading.Thread(target=work, daemon=True) for _ in range(workers)]
    for t in threads:
        t.start()
    try:
        results, finished = {}, 0
        while finished < len(paths):
            i, name, value = out.get()
            if name is None:  # decode failed, so this is the only message this image will send
                finished += 1
                yield paths[i], value
                continue
            res = results.setdefault(i, {})
            res[name] = value
            if len(res) == len(SIGNALS):
                finished += 1
                yield paths[i], results.pop(i)
    finally:  # also runs on early break or close(); one wake-up item per worker, each exits on stop
        stop.set()
        for _ in threads:
            tasks.put((float("inf"), 0, "", None))
        for t in threads:
            t.join()


def run_eval(manifest="dataset/manifest.csv", workers=WORKERS):
    """Classify every image in the manifest; print and return the confusion matrix
    {(actual, predicted): count} plus per-source accuracy. Undecodable images count as 'delete'."""
    root = Path(manifest).parent
    with open(manifest, newline="") as f:
        rows = list(csv.DictReader(f))
    predicted = {
        path: "delete" if isinstance(result, Exception) else classify_image_features(result)
        for path, result in analyze_all([str(root / r["path"]) for r in rows], workers)
    }
    matrix = Counter()
    by_source = {}
    for r in rows:
        hit = r["label"] == predicted[str(root / r["path"])]
        matrix[r["label"], predicted[str(root / r["path"])]] += 1
        by_source.setdefault(r["source"], []).append(hit)
    print(f"{'actual \\ predicted':>20s} {'keep':>6s} {'delete':>7s}")
    for actual in ("keep", "delete"):
        print(f"{actual:>20s} {matrix[actual, 'keep']:6d} {matrix[actual, 'delete']:7d}")
    print(f"accuracy {(matrix['keep', 'keep'] + matrix['delete', 'delete']) / len(rows):.3f}")
    for source, hits in by_source.items():
        print(f"  {source:36s} {sum(hits) / len(hits):.2f}")
    return matrix


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-j", "--workers", type=int, default=WORKERS)
    ap.add_argument("--eval", nargs="?", const="dataset/manifest.csv", metavar="MANIFEST", help="run the eval instead")
    ap.add_argument("paths", nargs="*", help="image files or directories (searched recursively)")
    args = ap.parse_args()
    if args.eval:
        run_eval(args.eval, args.workers)
        raise SystemExit
    exts = Image.registered_extensions()  # every format Pillow knows, HEIC included
    paths = [
        str(f)
        for p in args.paths
        for f in (sorted(Path(p).rglob("*")) if os.path.isdir(p) else [Path(p)])
        if f.suffix.lower() in exts
    ]
    t = time.perf_counter()
    for path, result in analyze_all(paths, args.workers):
        verdict = result if isinstance(result, Exception) else classify_image_features(result)
        print(f"{verdict}\t{path}\t{result}")
    dt = (time.perf_counter() - t) * 1000
    print(f"{len(paths)} images, {args.workers} workers: {dt:.0f}ms total, {dt / max(len(paths), 1):.1f}ms/img")
