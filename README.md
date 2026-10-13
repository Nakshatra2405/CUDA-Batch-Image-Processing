# CUDA Batch Image Processing

A CUDA C++ project that processes a directory containing many P6 PPM images using custom GPU kernels. It supports grayscale conversion, a 3x3 Gaussian blur, and Sobel edge detection. A CPU reference implementation and CSV timing output are included for comparison.

## Requirements

- NVIDIA GPU with a compatible CUDA driver/runtime
- CUDA Toolkit (`nvcc`)
- CMake 3.18+ and a C++17 compiler
- Python 3 for generating a deterministic sample dataset
- Optional: Pillow for contact-sheet previews

> Google Colab: choose **Runtime → Change runtime type → T4 GPU** (or another GPU), then verify that `nvidia-smi` and `nvcc --version` work. Some Colab images may not include the CUDA compiler; if `nvcc` is missing, use a CUDA-enabled runtime/image that includes the toolkit rather than trying to install an arbitrary toolkit version over the runtime.

## Build

```bash
cmake -S . -B build
cmake --build build -j
```

Alternatively, compile directly:

```bash
nvcc -std=c++17 -O2 -lineinfo src/main.cu -o cuda_image_processor
```

The CMake route is recommended. If your compiler does not accept `CUDA_ARCHITECTURES native`, change that property in `CMakeLists.txt` to the architecture supported by your GPU (for example, `75` for T4 or `80` for A100).

## Generate sample data

The generator creates 120 deterministic 256x256 PPM images using only Python's standard library:

```bash
python3 scripts/generate_dataset.py --output data/sample_images --count 120 --width 256 --height 256
```

PPM is used to keep this sample project independent of OpenCV and image-codec development headers. You can convert other images to PPM using ImageMagick or Pillow if desired.

## Run

```bash
./build/cuda_image_processor \
  --input data/sample_images \
  --output output \
  --operation edge \
  --batch-size 32
```

Operations:
- `grayscale`: RGB-to-grayscale conversion.
- `blur`: 3x3 Gaussian blur with weights `[[1,2,1],[2,4,2],[1,2,1]] / 16`.
- `edge`: Sobel edge magnitude, written as a grayscale RGB image.

Example for each operation:

```bash
./build/cuda_image_processor --input data/sample_images --output output_gray --operation grayscale --batch-size 32
./build/cuda_image_processor --input data/sample_images --output output_blur --operation blur --batch-size 32
./build/cuda_image_processor --input data/sample_images --output output_edge --operation edge --batch-size 32
```

The `--batch-size` option controls progress reporting and the reported number of batch groups. This starter implementation launches one image at a time; it does **not** hold an entire batch on the GPU simultaneously. Be transparent about this in your report.

## Outputs and benchmarking

- Processed PPM files, named `<input>_<operation>.ppm`
- `output/benchmark.csv`: per-image CPU reference time, CUDA kernel time, and end-to-end per-image pipeline time.
- Terminal summary: GPU name, operation, image count, progress, aggregate timings, and batch-group count.

Timing notes:
- CUDA kernel time is measured with CUDA events around the kernel launch.
- End-to-end time includes image I/O, CPU reference processing, allocation, transfers, and output writing, so it is not a pure GPU-versus-CPU speed comparison.
- For a fairer CPU/GPU comparison, compare the `cpu_ms` column with `gpu_kernel_ms` cautiously: they measure different scopes (CPU full operation versus GPU kernel only). Report this caveat.
- The GPU path includes host-to-device and device-to-host transfers; `total_gpu_ms` is the per-image wall time for the GPU pipeline, although this current implementation also computes the CPU reference in the same per-image loop.

## Google Colab quick start

1. Upload/extract this repository ZIP in Colab or clone your public GitHub repository.
2. Enable a GPU runtime.
3. Run:

```bash
!nvidia-smi
!nvcc --version
!python3 scripts/generate_dataset.py --output data/sample_images --count 120 --width 256 --height 256
!cmake -S . -B build
!cmake --build build -j2
!./build/cuda_image_processor --input data/sample_images --output output --operation edge --batch-size 32
!wc -l output/benchmark.csv
!ls output | head
```

If `nvcc` is not found, this runtime does not have the CUDA compiler installed. Use a runtime/container with the CUDA Toolkit. Do not claim the project executed until the CUDA build and run have succeeded.

Optional visual preview:

```bash
!pip -q install pillow
!python3 scripts/preview_results.py --input data/sample_images --output output --save results/contact_sheet.png
```

## Project design

Each pixel is independent for grayscale conversion. Blur and Sobel use a 2D grid of 16x16 threads; each thread computes one output pixel while clamping neighbor coordinates at image borders. The Sobel kernel computes horizontal and vertical gradients and clips the gradient magnitude to the 8-bit range.

## Limitations and future work

- The loader currently supports binary P6 PPM files with max channel value 255.
- Images are processed sequentially on the GPU. A next step is pinned host memory, CUDA streams, and multi-image batching.
- This baseline allocates and frees device buffers for each image; reusing allocations would reduce overhead.
- Results and timings vary by GPU, driver, clock state, and runtime.
- No speedup is guaranteed for small images because launch and transfer overhead can dominate.

## Suggested repository layout

```text
src/main.cu
scripts/generate_dataset.py
scripts/preview_results.py
data/sample_images/
output/
results/
CMakeLists.txt
run.sh
README.md
```

## What to submit

1. Public GitHub repository URL.
2. Screenshot/log showing GPU name and one run that processes all 120 images in a single invocation.
3. `output/benchmark.csv` and a screenshot/contact sheet showing sample input/output pairs.
4. A short report explaining the kernels, dataset size, benchmark method, limitations, and what you learned.

Do not commit hundreds of generated images unless needed. Include a small sample or use the generator to recreate the dataset, and commit a representative output/log plus your benchmark CSV after you have actually run it.
