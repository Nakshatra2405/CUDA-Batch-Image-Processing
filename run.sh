#!/usr/bin/env bash
set -euo pipefail
INPUT="${1:-data/sample_images}"
OUTPUT="${2:-output}"
OPERATION="${3:-edge}"
BATCH_SIZE="${4:-32}"

mkdir -p "$OUTPUT"
./build/cuda_image_processor \
  --input "$INPUT" \
  --output "$OUTPUT" \
  --operation "$OPERATION" \
  --batch-size "$BATCH_SIZE"
