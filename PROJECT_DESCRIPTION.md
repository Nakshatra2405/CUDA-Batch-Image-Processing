# Project Description: CUDA Batch Image Processing

## Purpose
This project demonstrates GPU acceleration for image processing over a directory of many images. It uses custom CUDA C++ kernels and supports grayscale conversion, Gaussian blur, and Sobel edge detection.

## Dataset
A deterministic Python generator creates 120 256x256 RGB P6 PPM images. The generator can create a larger dataset by changing `--count`, `--width`, and `--height`.

## GPU algorithms
- Grayscale: one CUDA thread per pixel; converts RGB with the weighted luminance formula 0.299R + 0.587G + 0.114B.
- Gaussian blur: one CUDA thread per output pixel; uses a 3x3 Gaussian-like kernel with weights summing to 16.
- Sobel edge detection: one CUDA thread per output pixel; computes horizontal and vertical gradients and clips their magnitude to [0,255].

## Evaluation
The executable measures CPU reference processing time, CUDA kernel time with CUDA events, and the per-image end-to-end GPU pipeline time. It writes a CSV row for each processed image.

## Results to fill after execution
- GPU model: [fill from program output]
- Number and resolution of images: [fill in]
- Operation: [fill in]
- Total elapsed time: [fill in]
- CPU reference processing sum: [fill in]
- CUDA kernel sum: [fill in]
- Observations: [explain whether kernel time is small compared with data movement and file I/O]

## Limitations
The sample implementation processes images sequentially, reuses no device allocations between images, and supports P6 PPM only. These choices keep the example portable and readable; streams and reusable buffers are possible extensions.

## Lessons learned
[Write what you observed from the actual run, including any build/runtime problems, timing differences, and the effect of image size.]
