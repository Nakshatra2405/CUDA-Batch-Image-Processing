#!/usr/bin/env python3
"""Create a contact sheet from input and output PPM images (requires Pillow)."""
import argparse
import glob
import os
from PIL import Image, ImageDraw

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/sample_images")
    parser.add_argument("--output", default="output")
    parser.add_argument("--save", default="results/contact_sheet.png")
    args = parser.parse_args()
    input_paths = sorted(glob.glob(os.path.join(args.input, "*.ppm")))[:6]
    if not input_paths:
        raise SystemExit("No input PPM files found.")
    output_paths = []
    for path in input_paths:
        stem = os.path.splitext(os.path.basename(path))[0]
        candidates = glob.glob(os.path.join(args.output, stem + "_*.ppm"))
        if candidates:
            output_paths.append(candidates[0])
    if not output_paths:
        raise SystemExit("No output PPM files found. Run the CUDA executable first.")
    thumbs = []
    for path in input_paths:
        im = Image.open(path).convert("RGB")
        im.thumbnail((220, 220))
        thumbs.append((im.copy(), os.path.basename(path)))
    for path in output_paths:
        im = Image.open(path).convert("RGB")
        im.thumbnail((220, 220))
        thumbs.append((im.copy(), os.path.basename(path)))
    columns = 3
    cell_w, cell_h = 240, 250
    rows = (len(thumbs) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * cell_w, rows * cell_h), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (im, label) in enumerate(thumbs):
        x = (i % columns) * cell_w
        y = (i // columns) * cell_h
        sheet.paste(im, (x + (cell_w - im.width)//2, y + 5))
        draw.text((x + 8, y + 228), label[:34], fill="black")
    os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
    sheet.save(args.save)
    print(f"Saved preview to {args.save}")

if __name__ == "__main__":
    main()
