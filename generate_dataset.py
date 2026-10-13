#!/usr/bin/env python3
"""Generate deterministic P6 PPM images without third-party dependencies."""
import argparse
import math
import os
import random

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/sample_images")
    parser.add_argument("--count", type=int, default=120)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    random.seed(2026)
    for index in range(args.count):
        path = os.path.join(args.output, f"image_{index:04d}.ppm")
        with open(path, "wb") as f:
            f.write(f"P6\n{args.width} {args.height}\n255\n".encode("ascii"))
            pixels = bytearray()
            phase = index * 0.17
            for y in range(args.height):
                for x in range(args.width):
                    r = int(127.5 + 127.5 * math.sin(x * 0.035 + phase))
                    g = int(127.5 + 127.5 * math.sin(y * 0.041 + phase * 0.7))
                    b = int(127.5 + 127.5 * math.sin((x + y) * 0.023 + phase * 1.3))
                    noise = random.randrange(-8, 9)
                    pixels.extend((max(0, min(255, r + noise)),
                                   max(0, min(255, g + noise)),
                                   max(0, min(255, b + noise))))
            f.write(pixels)
    print(f"Generated {args.count} PPM images ({args.width}x{args.height}) in {args.output}")

if __name__ == "__main__":
    main()
