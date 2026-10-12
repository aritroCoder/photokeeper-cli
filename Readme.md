# Photokeeper - Extremely fast photo classification CLI tool

Photokeeper is a CLI tool that can take one or many images (ideally from your gallery) and suggest whether each one of the image should be kept or worth deleting. Most of our gallery is filled with random screenshots, document photos, forwarded memes and so many other junk, Photokeeper aims to find and filter them out for you so you can take a call whether to delete them or not.

This version of photokeeper is written in C++. A python version is also available in the other branch.

## How does it work

Photokeeper is built with edge devices in mind. It is highly optimized to process a large number of photo with very minimal CPU or memory overhead. We will get into performance details later.

Instead of a single classifier, photokeeper uses a series of pipeline steps, where each of the step extract a specific feature. It uses a set of 6 features to then determine whether the photo should be kept or deleted.

### The pipeline
1. **Edge ratio**: The percentage of pixels in the image with horizontal or vertical gradients. We use sobel filter to compute gradients. Too much vertical or horizontal edges mean it's a boxy image, which is common for screenshots and text heavy clicks.
2. **Pallete flatness**: It measures the variety of colors in the image. Lower variety usually means documents or textual matter (delete candidate) while natural photos are vibrant with different shades of color.
3. **faces**: Number of detected faces. Human faces are a strong signal that a photo is worth keeping.
