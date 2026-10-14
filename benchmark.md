# Benchmark

PhotoKeeper against the tools people reach for when cleaning a photo library, on the labelled
eval set in `dataset/` (1000 images: 400 keep, 600 delete). One table, then what each candidate
actually does, so the comparison is read for what it is.

Machine: WSL2 on Linux, 24 logical cores, 7 GB RAM. Measured 2026-10-12 with `/usr/bin/time`
(wall, user CPU, peak RSS of the whole process, model load included). OpenCV 4.10.0, built from
source for PhotoKeeper (static, pthreads, only the five modules it uses; see Reproducing).

| | PhotoKeeper (C++) | PhotoKeeper (Python) | CLIP ViT-B/32 zero-shot | Czkawka 12.0.2 `image` | Blur-only threshold |
|---|---|---|---|---|---|
| total time | **9.5 s** | 12.5 s | 51.3 s | 1.5 s | (same run as ours) |
| ms / img | **9.5** | 12.5 | 51.3 | 1.5 | |
| peak RSS | **171 MB** | 212 MB | 1895 MB | 589 MB | |
| CPU time | 105 s | 96 s | 277 s | 26 s | |
| threads | 8 workers + OpenCV pthreads | 8 workers + OpenCV pthreads | 12 (torch default) | 24 | |
| accuracy | **0.934** | **0.934** | 0.916 | n/a | 0.389 |
| F1 (delete) | **0.945** | **0.945** | 0.930 | n/a | 0.035 |
| F1 (keep) | **0.917** | **0.917** | 0.896 | n/a | 0.553 |
| macro F1 | **0.931** | **0.931** | 0.913 | n/a | 0.294 |
| precision / recall (delete) | 0.942 / 0.948 | 0.942 / 0.948 | 0.934 / 0.925 | n/a | 0.333 / 0.018 |
| confusion (kk, kd, dk, dd) | 365, 35, 31, 569 | 365, 35, 31, 569 | 361, 39, 45, 555 | | 378, 22, 589, 11 |
| model size on disk | 2.6 MB | 2.6 MB | ~600 MB | 0 | 0 |

Positive class for precision/recall/F1 is `delete`, so a false positive is a keeper wrongly
deleted. Confusion order: keep→keep, keep→delete, delete→keep, delete→delete.

The two PhotoKeeper columns were measured back to back in a later session (2026-10-14), so they
compare to each other but not directly to the other columns' timings. In that same session the
C++ binary linked against Ubuntu's `libopencv-dev` (shared, TBB, every module and codec enabled)
took 11.8 s / 156 s CPU / 235 MB, slower and larger than Python; the OpenCV build, not the
language, is what moves these numbers.

### What "peak RSS" measures

Peak resident memory of the whole process over the full 1000-image run, as reported by
`/usr/bin/time %M`. No candidate loads the dataset into memory; images stream through, so the peak
is models plus however many decoded images are in flight at once (PhotoKeeper: one 320 px
thumbnail per worker, 8 workers; CLIP: a batch of 32 at 224 px; Czkawka: one image per thread,
24 threads). Peak is the right number for "will it fit on the device", but it scales with
parallelism, so here is the same measurement for a single image, which is what the app will do
when the user opens one photo:

| | PhotoKeeper (C++) | PhotoKeeper (Python) | CLIP ViT-B/32 | Czkawka |
|---|---|---|---|---|
| process start, no images (models loaded) | **8.5 MB** | 71 MB | 542 MB | 11 MB |
| 1 image | **51 MB** | 103 MB | 915 MB | 33 MB |
| 1000 images, one in flight | **59 MB** (1 worker) | 117 MB (1 worker) | 972 MB (batch 1) | 44 MB (1 thread) |
| 1000 images, parallel (table above) | **171 MB** (8 workers) | 212 MB (8 workers) | 1895 MB (batch 32) | 589 MB (24 threads) |

For PhotoKeeper the floor is the two detectors (~42 MB once both have run) on top of an 8.5 MB
process; each extra worker adds ~16 MB. CLIP's floor is the model (542 MB before any image), and
one image adds ~370 MB of activations. Czkawka has no model; its memory is almost entirely
per-thread decode buffers, which is why it is the smallest at one thread and the second largest
at 24. These are peaks, not averages: RSS does not shrink between images, so the average over a
run is within a few MB of the peak. The one-in-flight runs also give the serial speed: PhotoKeeper
18 s (C++) and 22 s (Python) in the 10-14 session, CLIP 76 s and Czkawka 11 s in a loaded earlier
session, so not comparable to the main table's idle-host timings; memory is unaffected by load.

## The dataset

`dataset/manifest.csv` labels each image `keep` or `delete` by source: Rico (Android screenshots)
and RVL-CDIP (scanned documents) are delete, hateful_memes (image macros) are delete, Flickr30k
and Unsplash-lite (photographs) are keep. See `dataset/build.py` on `main` for how it was built.

**Caveat that matters:** PhotoKeeper's feature ramps and weights were tuned on these same 1000
images (the keep/delete quantiles). Its numbers are in-sample. Every other candidate saw none of
them. Treat the accuracy columns as "how it does on the data we tuned on" until a held-out split
exists; a logistic fit on the same features reached 93.5% in-sample, so expect a few points' drop
out of sample, not a collapse.

## Candidates

### PhotoKeeper

What it does: one `keep`/`delete` verdict per image from five cheap signals on a 320 px thumbnail:
face count (YuNet, 233 KB ONNX), palette flatness, axis-aligned edge ratio, clipped-pixel
fraction, and text area ratio (PP-OCRv3 DB detector, 2.4 MB ONNX). A weighted sum with hand-set
ramps decides. Everything runs on-device through OpenCV; see `pipeline_c/`. An earlier sixth
signal, variance-of-Laplacian blur at 3% weight, was removed after measuring that dropping it
left the eval unchanged (0.933 → 0.934).

What it does not do: find duplicates or near-duplicates, group bursts, rate aesthetics, or
understand content beyond "has faces" and "has text". A sharp, well-exposed photo of nothing
interesting is a keep.

Run: `./build/photokeeper -j 8 dataset` (the scored output) or `./build/photokeeper --eval`.

### PhotoKeeper (Python)

What it is: the original implementation on the `main` branch (`main.py`, `pipeline/`), which the
C++ port follows line for line: same models, same signals, ramps and weights, same priority-queue
scheduler. It runs on the `opencv-python-headless` 5.0.0 wheel, itself a static minimal OpenCV.

Why it is here: to show what the port bought. Verdicts agree on all 1000 images and four of the
five signals match to 1e-6; `text_area_ratio` differs by up to 0.03 on a few scanned documents,
which is the OpenCV 5 vs 4 DNN path, not the code. Python is only ~30% slower because every heavy
call (decode, resize, Sobel, both detectors) runs inside OpenCV with the GIL released, so Python is
glue. Its memory gap is the interpreter and numpy: 71 MB before the first image against 8.5 MB for
the static C++ binary. The port's main value is still the deployment story (one native library for
Flutter on four platforms, no interpreter).

Run: `uv run main.py -j 8 dataset` or `uv run main.py --eval` on `main`.

### CLIP ViT-B/32, zero-shot

What it is: OpenAI's general image-text model (`openai/clip-vit-base-patch32`, ~600 MB), run on
CPU via PyTorch 2.14 and transformers 5.19. Each image is embedded and compared against nine
prompts; the verdict is whichever side has the closest prompt:

- keep: "a photo", "a photograph of a person", "a photo of a landscape", "a snapshot taken with a
  phone camera"
- delete: "a screenshot of a phone app", "a scanned document", "a meme with text", "a blurry
  photo", "a dark underexposed photo"

Why it is here: it is the "just use a foundation model" baseline. It is also the only candidate
that produces a keep/delete verdict, so it is the only one with F1 numbers.

What it does that we do not: understands content. It can be told new categories by editing a
prompt; we would need a new signal and re-tuning.

What it does not do: run cheaply. 5x our time, 8x our memory, 230x our model size, on a desktop
CPU; on a phone that gap widens and the model has to ship inside the app. It is also not tuned
to this task at all, so the accuracy comparison is unfair to it: better prompts or the ViT-L/14
model would likely close or flip the F1 gap. They would not close the cost gap.

Run: `clip_eval.py` in the bench scratch directory (see Reproducing).

### Czkawka 12.0.2, `image` mode

What it is: the best-known open-source duplicate finder (Rust, GUI and CLI). `image` mode computes
one perceptual hash per image (dHash, 16x16 by default) and groups images within a Hamming
distance. Run here with `-H` (cache off) and all 24 threads.

Why it is here: when people say "I already have a tool for cleaning photos", this is usually what
they mean, so it is the speed and memory reference everyone will ask about.

What it does that we do not: finds exact and near-exact copies, resized or recompressed versions,
and (optionally) mirrored or rotated copies, across a whole archive.

What it does not do: classify. It has no notion of screenshot, document, meme, blur or exposure,
so it cannot produce a keep/delete verdict and has no F1 here. On this set it grouped 33 images as
"similar", including a Rico screenshot paired with an Unsplash photograph; perceptual hashes key on
luminance layout, not content. It is 7x faster because hashing is a much smaller job than ours,
and it still used 2.4x our memory.

Run: `czkawka_cli image -d <abs path>/dataset -H -f out.txt`

### Blur-only threshold

What it is: variance of the Laplacian on the grey thumbnail, keep if ≥ 100. The measure comes
from Pech-Pacheco et al., "Diatom autofocusing in brightfield microscopy: a comparative study",
ICPR 2000 ([doi:10.1109/ICPR.2000.903548](https://doi.org/10.1109/ICPR.2000.903548)), listed as
LAPV in Pertuz et al.'s survey of 36 focus measures, "Analysis of focus measure operators for
shape-from-focus", Pattern Recognition 2013
([doi:10.1016/j.patcog.2012.11.011](https://doi.org/10.1016/j.patcog.2012.11.011)). It became the
default blur check in OpenCV code through Rosebrock's 2015 tutorial
["Blur detection with OpenCV"](https://pyimagesearch.com/2015/09/07/blur-detection-with-opencv/),
which is one line (`cv2.Laplacian(gray, cv2.CV_64F).var()`), ships a default threshold of 100, and
is what Stack Overflow and GitHub blur detectors cite. The threshold here is that default.
Computed from PhotoKeeper's `blur` output when it still had that signal, so no separate timing.

Why it is here: to show the floor of a sharpness-only cleaner. It gets 39%, worse than always
answering `delete` (60%), because screenshots and scanned documents are the sharpest images in any
library. PhotoKeeper carried this as a sixth signal at 3% weight and dropped it for the same
reason: on this data it moves nothing. It would only earn its place with a labelled blurry-photo
class in the dataset.
Rosebrock's own follow-up notes the method "required significant manual tuning" and proposes an
FFT measure instead; neither fixes the screenshot problem, which is about content, not focus.

## Not benchmarked, and why

- **Google Photos** (storage management: blurry, screenshots, large videos; similar-photo cleanup),
  **Apple Photos** (Utilities → Duplicates, exact and near-exact only), **Samsung Gallery**
  (duplicate suggestions). Built into the platforms, partly cloud-side, no way to run them on a
  fixed dataset or measure their memory. These are the real competition for users; they cannot be
  put in this table.
- **CleanMy®Phone** (ex-Gemini Photos), **Clever Cleaner**, **Dupie** (Apple Vision on iOS,
  MobileNetV3 on Android), **AI Cleaner**, **Hyper Cleaner**, **Roomier**, **Meerabel**. Closed
  mobile apps with overlapping features (screenshots, blurry, "photos of text", similar). Only
  measurable on a phone with Xcode Instruments or Android Studio Profiler, against the same photo
  library, once our app exists.
- **Swype**, **PhotoSwooper** (open source, Android), **Slidebox**. Swipe-to-delete UIs with no
  classifier; the UX bar, not a benchmark target.
- **digiKam Image Quality Sorter** (blur, noise, exposure, compression, aesthetic DNN). The
  closest open-source peer on the quality axis, but it needs a GUI session and a database to run;
  not attempted here.
- **Narrative Select, AfterShoot, FilterPixel, Lightroom AI culling.** Pro photographers' culling
  tools with heavy ML, closed, desktop. Different users.
- **CLIP ViT-L/14.** Would be the stronger CLIP baseline; needs 3+ GB on a 7 GB VM and was skipped
  for machine-safety reasons, not relevance.

## Reproducing

Scripts and raw outputs live outside the repo in `~/.cache/pkbench`: `clip_eval.py` (prompts
above, batch 32), `score.py` (metrics from a `path verdict` file), `czkawka_cli` binary from the
12.0.2 GitHub release, and the `*_pred.txt` / `*_time.txt` outputs behind the table. PhotoKeeper's
predictions come from `./build/photokeeper -j 8 dataset`, scored with the same `score.py`.

The OpenCV behind the C++ numbers is 4.10.0 from the source tarball, configured with
`-DBUILD_SHARED_LIBS=OFF -DBUILD_LIST=core,imgproc,imgcodecs,dnn,objdetect -DWITH_TBB=OFF
-DWITH_OPENMP=OFF -DWITH_IPP=OFF -DWITH_OPENCL=OFF -DBUILD_JPEG=ON -DBUILD_PNG=ON -DBUILD_TIFF=ON
-DBUILD_ZLIB=ON -DBUILD_PROTOBUF=ON` and every other `WITH_*` codec and backend off, then
`cmake -S . -B build -DOpenCV_DIR=<install>/lib/cmake/opencv4`. The result links six system
libraries (libc, libstdc++ and friends) against 180 for Ubuntu's `libopencv-dev`, and it is the
configuration the mobile builds will use anyway.

Pinned versions: OpenCV 4.10.0 (source build above), Python 3.12 with
`opencv-python-headless` 5.0.0, Czkawka 12.0.2, `openai/clip-vit-base-patch32`,
transformers 5.19.0, torch 2.14.1+cpu.
