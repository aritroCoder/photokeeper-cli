# Photokeeper - Photo organizer CLI

Photokeeper is a CLI tool that can take one or many images (ideally from your gallery) and suggest whether each one of the image should be kept or worth deleting. Most of our gallery is filled with random screenshots, document photos, forwarded memes and so many other junk, Photokeeper aims to find and filter them out for you so you can take a call whether to delete them or not.

This version of photokeeper is written in C++. A python version is also available in the other branch.

## Build and run locally
First you need to clone the repo. Once the repo is cloned, get into the repo and run (make sure build-essentials is installed in linux/mac, for windows I recommend using WSL and follow linux steps)
```
cmake --build build 
```
Then run the cli:
```
./build/photokeeper album/             # classify all the images in the given folder (searched recursively)
./build/photokeeper photo.jpg          # classify a single image as keep or delete
./build/photokeeper --eval             # accuracy on dataset/manifest.csv (if present, not committed with code)
./build/photokeeper -h                 # show help
```

## Performance
Photokeeper is designed for edge devices, which means it should be able to run on a 5 year old android phone as well as a brand new desktop setup. So performance is extremely important for this case. Although there is no perfect one on one open/close source alternatives that match with this cli, we still picked the closest ones to match with CLIP ViT, Czkawka, and blur threshhold. The candidates are discussed in detail below. The results are:

Machine: WSL2 on Linux, 24 logical cores, 7 GB RAM. Measured with `/usr/bin/time` (wall, user CPU, peak RSS of the whole process, model load included). OpenCV 4.10.0.

Benchmark data: Labelled eval set with 1000 images: 400 keep, 600 delete. This eval set is provided seperately.

| | PhotoKeeper (C++) | CLIP ViT-B/32 zero-shot | Czkawka 12.0.2 `image` | Blur-only threshold |
|---|---|---|---|---|
| total time | **10.5 s** | 51.3 s | 1.5 s | (same run as ours) |
| ms / img | **10.5** | 51.3 | 1.5 | |
| peak RSS | **242 MB** | 1895 MB | 589 MB | |
| accuracy | **0.934** | 0.916 | n/a | 0.389 |
| F1 (delete) | **0.945** | 0.930 | n/a | 0.035 |
| F1 (keep) | **0.917** | 0.896 | n/a | 0.553 |
| macro F1 | **0.931** | 0.913 | n/a | 0.294 |
| precision / recall (delete) | 0.942 / 0.948 | 0.934 / 0.925 | n/a | 0.333 / 0.018 |
| model size on disk | 2.6 MB | ~600 MB | 0 | 0 |

Note: Positive class for precision/recall/F1 is `delete`, so a false positive is a keeper wrongly deleted.

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

### Not benchmarked, and why

- **Google Photos** (storage management: blurry, screenshots, large videos; similar-photo cleanup),
  **Apple Photos** (Utilities → Duplicates, exact and near-exact only), **Samsung Gallery**
  (duplicate suggestions). Built into the platforms, partly cloud-side, no way to run them on a
  fixed dataset or measure their memory. These are the real competition for users; they cannot be
  put in this table. Also these do not process your images on device so cannot be measured properly.
- **CleanMyPhone**, **Clever Cleaner**, **Dupie**, **AI Cleaner**, **Hyper Cleaner**, **Roomier**, **Meerabel**. Closed
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

## How does it work

Photokeeper is built with edge devices in mind. It is highly optimized to process a large number of photo with very minimal CPU or memory overhead. We will get into performance details later.

Instead of a single classifier, photokeeper uses a series of pipeline steps, where each of the step extract a specific feature. It uses a set of 6 features to then determine whether the photo should be kept or deleted.

### The pipeline
1. **Edge ratio**: The percentage of pixels in the image with horizontal or vertical gradients. We use sobel filter to compute gradients. Too much vertical or horizontal edges mean it's a boxy image, which is common for screenshots and text heavy clicks.
2. **Pallete flatness**: It measures the variety of colors in the image. Lower variety usually means documents or textual matter (delete candidate) while natural photos are vibrant with different shades of color.
3. **Faces**: Number of detected faces. Human faces are a strong signal that a photo is worth keeping. Uses the lightweight and fast YuNet model.
4. **Extreme-tone ratio**: Fraction of near-black or near-white pixels, high means scanned documents or screenshots.
5. **Text area ratio**: Uses the PP-OCRv3 model which is super lightweight and ultrafast to detect the presence of text in the image. Then it calculates the ratio of image area with text area (from detected bounding box coordinates). This only works on English text for now.

Its uses a weighted vote to combine each of these pipeline step outputs. Each signal is turned into a keep-likeness score between 0 and 1. The scores are multiplied by their weights (which sum to 1) and added up. If the total is above the threshhold the photo is kept, otherwise it's marked for deletion.
