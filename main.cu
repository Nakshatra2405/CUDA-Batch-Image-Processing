#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

#define CUDA_CHECK(call)                                                        \
  do {                                                                          \
    cudaError_t error = (call);                                                  \
    if (error != cudaSuccess) {                                                  \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": "      \
                << cudaGetErrorString(error) << std::endl;                       \
      std::exit(EXIT_FAILURE);                                                   \
    }                                                                            \
  } while (0)

struct Image {
  int width = 0;
  int height = 0;
  std::vector<uint8_t> pixels;  // Interleaved RGB, 3 bytes per pixel.
};

__global__ void GrayscaleKernel(const uint8_t* input, uint8_t* output,
                                int pixel_count) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= pixel_count) return;
  const int base = idx * 3;
  const float gray = 0.299f * input[base] + 0.587f * input[base + 1] +
                     0.114f * input[base + 2];
  const uint8_t value = static_cast<uint8_t>(gray + 0.5f);
  output[base] = value;
  output[base + 1] = value;
  output[base + 2] = value;
}

__global__ void BlurKernel(const uint8_t* input, uint8_t* output,
                           int width, int height) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;
  const int weights[3][3] = {{1, 2, 1}, {2, 4, 2}, {1, 2, 1}};
  const int denominator = 16;
  const int out_base = (y * width + x) * 3;

  for (int channel = 0; channel < 3; ++channel) {
    int sum = 0;
    for (int ky = -1; ky <= 1; ++ky) {
      for (int kx = -1; kx <= 1; ++kx) {
        const int sx = min(max(x + kx, 0), width - 1);
        const int sy = min(max(y + ky, 0), height - 1);
        const int in_base = (sy * width + sx) * 3 + channel;
        sum += weights[ky + 1][kx + 1] * input[in_base];
      }
    }
    output[out_base + channel] =
        static_cast<uint8_t>((sum + denominator / 2) / denominator);
  }
}

__global__ void SobelKernel(const uint8_t* input, uint8_t* output,
                            int width, int height) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;

  const int gx_weights[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
  const int gy_weights[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};
  int gx = 0;
  int gy = 0;
  for (int ky = -1; ky <= 1; ++ky) {
    for (int kx = -1; kx <= 1; ++kx) {
      const int sx = min(max(x + kx, 0), width - 1);
      const int sy = min(max(y + ky, 0), height - 1);
      const int base = (sy * width + sx) * 3;
      const int gray = (299 * input[base] + 587 * input[base + 1] +
                        114 * input[base + 2]) / 1000;
      gx += gx_weights[ky + 1][kx + 1] * gray;
      gy += gy_weights[ky + 1][kx + 1] * gray;
    }
  }
  const int magnitude = min(255, static_cast<int>(sqrtf(
      static_cast<float>(gx * gx + gy * gy))));
  const int out_base = (y * width + x) * 3;
  output[out_base] = static_cast<uint8_t>(magnitude);
  output[out_base + 1] = static_cast<uint8_t>(magnitude);
  output[out_base + 2] = static_cast<uint8_t>(magnitude);
}

std::string ReadToken(std::istream& input) {
  std::string token;
  while (input >> token) {
    if (!token.empty() && token[0] == '#') {
      std::string ignored;
      std::getline(input, ignored);
      continue;
    }
    return token;
  }
  throw std::runtime_error("Unexpected end of PPM header.");
}

Image LoadPpm(const fs::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("Cannot open input image: " + path.string());
  const std::string magic = ReadToken(input);
  if (magic != "P6") throw std::runtime_error("Only binary P6 PPM images are supported: " + path.string());
  Image image;
  image.width = std::stoi(ReadToken(input));
  image.height = std::stoi(ReadToken(input));
  const int max_value = std::stoi(ReadToken(input));
  if (image.width <= 0 || image.height <= 0 || max_value != 255)
    throw std::runtime_error("Unsupported PPM dimensions or max value: " + path.string());
  input.get();  // Consume the single whitespace character after max value.
  image.pixels.resize(static_cast<size_t>(image.width) * image.height * 3);
  input.read(reinterpret_cast<char*>(image.pixels.data()),
             static_cast<std::streamsize>(image.pixels.size()));
  if (!input) throw std::runtime_error("Incomplete PPM pixel data: " + path.string());
  return image;
}

void SavePpm(const fs::path& path, const Image& image) {
  std::ofstream output(path, std::ios::binary);
  if (!output) throw std::runtime_error("Cannot write output image: " + path.string());
  output << "P6\n" << image.width << " " << image.height << "\n255\n";
  output.write(reinterpret_cast<const char*>(image.pixels.data()),
               static_cast<std::streamsize>(image.pixels.size()));
  if (!output) throw std::runtime_error("Failed while writing: " + path.string());
}

Image ProcessCpu(const Image& input, const std::string& operation) {
  Image output = input;
  const int width = input.width;
  const int height = input.height;
  const auto at = [&](int x, int y, int c) -> int {
    x = std::max(0, std::min(width - 1, x));
    y = std::max(0, std::min(height - 1, y));
    return input.pixels[(static_cast<size_t>(y) * width + x) * 3 + c];
  };

  if (operation == "grayscale") {
    for (int i = 0; i < width * height; ++i) {
      const int base = i * 3;
      const uint8_t gray = static_cast<uint8_t>(
          0.299f * input.pixels[base] + 0.587f * input.pixels[base + 1] +
          0.114f * input.pixels[base + 2] + 0.5f);
      output.pixels[base] = output.pixels[base + 1] =
          output.pixels[base + 2] = gray;
    }
  } else if (operation == "blur") {
    const int weights[3][3] = {{1, 2, 1}, {2, 4, 2}, {1, 2, 1}};
    for (int y = 0; y < height; ++y) for (int x = 0; x < width; ++x)
      for (int c = 0; c < 3; ++c) {
        int sum = 0;
        for (int ky = -1; ky <= 1; ++ky) for (int kx = -1; kx <= 1; ++kx)
          sum += weights[ky + 1][kx + 1] * at(x + kx, y + ky, c);
        output.pixels[(static_cast<size_t>(y) * width + x) * 3 + c] =
            static_cast<uint8_t>((sum + 8) / 16);
      }
  } else if (operation == "edge") {
    const int gxw[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    const int gyw[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};
    for (int y = 0; y < height; ++y) for (int x = 0; x < width; ++x) {
      int gx = 0, gy = 0;
      for (int ky = -1; ky <= 1; ++ky) for (int kx = -1; kx <= 1; ++kx) {
        const int r = at(x + kx, y + ky, 0);
        const int g = at(x + kx, y + ky, 1);
        const int b = at(x + kx, y + ky, 2);
        const int gray = (299 * r + 587 * g + 114 * b) / 1000;
        gx += gxw[ky + 1][kx + 1] * gray;
        gy += gyw[ky + 1][kx + 1] * gray;
      }
      const uint8_t value = static_cast<uint8_t>(std::min(
          255, static_cast<int>(std::sqrt(static_cast<float>(gx * gx + gy * gy)))));
      const size_t base = (static_cast<size_t>(y) * width + x) * 3;
      output.pixels[base] = output.pixels[base + 1] = output.pixels[base + 2] = value;
    }
  }
  return output;
}

struct Options {
  fs::path input_dir;
  fs::path output_dir;
  std::string operation = "edge";
  int batch_size = 32;
};

Options ParseArgs(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--input" && i + 1 < argc) options.input_dir = argv[++i];
    else if (arg == "--output" && i + 1 < argc) options.output_dir = argv[++i];
    else if (arg == "--operation" && i + 1 < argc) options.operation = argv[++i];
    else if (arg == "--batch-size" && i + 1 < argc) options.batch_size = std::stoi(argv[++i]);
    else if (arg == "--help") {
      std::cout << "Usage: ./cuda_image_processor --input DIR --output DIR "
                   "[--operation grayscale|blur|edge] [--batch-size N]\n";
      std::exit(EXIT_SUCCESS);
    } else {
      throw std::runtime_error("Unknown or incomplete argument: " + arg);
    }
  }
  if (options.input_dir.empty() || options.output_dir.empty())
    throw std::runtime_error("--input and --output are required. Use --help.");
  if (options.operation != "grayscale" && options.operation != "blur" &&
      options.operation != "edge")
    throw std::runtime_error("--operation must be grayscale, blur, or edge.");
  if (options.batch_size < 1) throw std::runtime_error("--batch-size must be positive.");
  return options;
}

int main(int argc, char** argv) {
  try {
    const Options options = ParseArgs(argc, argv);
    if (!fs::is_directory(options.input_dir))
      throw std::runtime_error("Input path is not a directory: " + options.input_dir.string());
    fs::create_directories(options.output_dir);
    std::vector<fs::path> paths;
    for (const auto& entry : fs::directory_iterator(options.input_dir)) {
      if (entry.is_regular_file() && entry.path().extension() == ".ppm")
        paths.push_back(entry.path());
    }
    std::sort(paths.begin(), paths.end());
    if (paths.empty()) throw std::runtime_error("No .ppm files found in input directory.");

    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) throw std::runtime_error("No CUDA-capable GPU detected.");
    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));
    CUDA_CHECK(cudaSetDevice(0));

    std::cout << "CUDA Batch Image Processing\n"
              << "GPU: " << properties.name << "\n"
              << "Operation: " << options.operation << "\n"
              << "Input images: " << paths.size() << "\n"
              << "Configured batch size: " << options.batch_size << "\n";

    std::ofstream csv(options.output_dir / "benchmark.csv");
    csv << "filename,width,height,cpu_ms,gpu_kernel_ms,total_gpu_ms,output_file\n";
    double cpu_total_ms = 0.0, gpu_kernel_total_ms = 0.0, wall_total_ms = 0.0;
    const auto all_start = std::chrono::steady_clock::now();

    for (size_t i = 0; i < paths.size(); ++i) {
      const auto image_start = std::chrono::steady_clock::now();
      const Image input = LoadPpm(paths[i]);
      const int width = input.width, height = input.height;
      const int pixel_count = width * height;
      const size_t bytes = input.pixels.size();

      const auto cpu_start = std::chrono::steady_clock::now();
      const Image cpu_output = ProcessCpu(input, options.operation);
      const auto cpu_end = std::chrono::steady_clock::now();
      const double cpu_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();

      uint8_t *device_input = nullptr, *device_output = nullptr;
      CUDA_CHECK(cudaMalloc(&device_input, bytes));
      CUDA_CHECK(cudaMalloc(&device_output, bytes));
      CUDA_CHECK(cudaMemcpy(device_input, input.pixels.data(), bytes, cudaMemcpyHostToDevice));

      cudaEvent_t kernel_start, kernel_stop;
      CUDA_CHECK(cudaEventCreate(&kernel_start));
      CUDA_CHECK(cudaEventCreate(&kernel_stop));
      const dim3 block(16, 16);
      const dim3 grid((width + block.x - 1) / block.x,
                      (height + block.y - 1) / block.y);
      CUDA_CHECK(cudaEventRecord(kernel_start));
      if (options.operation == "grayscale") {
        const int threads = 256;
        GrayscaleKernel<<<(pixel_count + threads - 1) / threads, threads>>>(
            device_input, device_output, pixel_count);
      } else if (options.operation == "blur") {
        BlurKernel<<<grid, block>>>(device_input, device_output, width, height);
      } else {
        SobelKernel<<<grid, block>>>(device_input, device_output, width, height);
      }
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaEventRecord(kernel_stop));
      CUDA_CHECK(cudaEventSynchronize(kernel_stop));
      float kernel_ms = 0.0f;
      CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, kernel_start, kernel_stop));

      Image output = input;
      CUDA_CHECK(cudaMemcpy(output.pixels.data(), device_output, bytes, cudaMemcpyDeviceToHost));
      const fs::path output_path = options.output_dir / (paths[i].stem().string() + "_" + options.operation + ".ppm");
      SavePpm(output_path, output);

      const auto image_end = std::chrono::steady_clock::now();
      const double total_ms = std::chrono::duration<double, std::milli>(image_end - image_start).count();
      cpu_total_ms += cpu_ms;
      gpu_kernel_total_ms += kernel_ms;
      wall_total_ms += total_ms;
      csv << paths[i].filename().string() << "," << width << "," << height << ","
          << std::fixed << std::setprecision(4) << cpu_ms << "," << kernel_ms << ","
          << total_ms << "," << output_path.filename().string() << "\n";

      if ((i + 1) % options.batch_size == 0 || i + 1 == paths.size())
        std::cout << "Progress: " << (i + 1) << "/" << paths.size() << " images processed\n";

      CUDA_CHECK(cudaEventDestroy(kernel_start));
      CUDA_CHECK(cudaEventDestroy(kernel_stop));
      CUDA_CHECK(cudaFree(device_input));
      CUDA_CHECK(cudaFree(device_output));
    }

    const auto all_end = std::chrono::steady_clock::now();
    const double total_elapsed_ms = std::chrono::duration<double, std::milli>(all_end - all_start).count();
    csv.close();
    std::cout << std::fixed << std::setprecision(3)
              << "\nCompleted: " << paths.size() << " images\n"
              << "CPU processing sum (ms): " << cpu_total_ms << "\n"
              << "GPU kernel sum (ms): " << gpu_kernel_total_ms << "\n"
              << "Per-image pipeline sum (ms): " << wall_total_ms << "\n"
              << "End-to-end elapsed (ms): " << total_elapsed_ms << "\n"
              << "Benchmark CSV: " << (options.output_dir / "benchmark.csv") << "\n"
              << "Processed batch groups: " << ((paths.size() + options.batch_size - 1) / options.batch_size) << "\n";
    CUDA_CHECK(cudaDeviceSynchronize());
  } catch (const std::exception& error) {
    std::cerr << "Error: " << error.what() << "\n";
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
