#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/opt/anaconda3/envs/ee/bin/python}"

OUTPUT_DIR="kalman/random_pixel_datasets"
START_DATE="2012-01-01"
END_DATE="2019-12-31"
CHUNK_SIZE="50"
TILE_SCALE="4"
SEED="42"
SAMPLE_SIZE="10"
TRAIN_ASSET="users/Khash/Kalman_Global_1000_Points_Training"
TEST_ASSET="users/Khash/Kalman_Global_1000_Points_Test"
TRAIN_ID_PROPERTY=""
TEST_ID_PROPERTY=""

"${PYTHON_BIN}" "${SCRIPT_DIR}/kalman/generate_random_pixel_datasets.py" \
  --output-dir "${OUTPUT_DIR}" \
  --start-date "${START_DATE}" \
  --end-date "${END_DATE}" \
  --chunk-size "${CHUNK_SIZE}" \
  --tile-scale "${TILE_SCALE}" \
  --seed "${SEED}" \
  --sample-size "${SAMPLE_SIZE}" \
  --train-asset "${TRAIN_ASSET}" \
  --test-asset "${TEST_ASSET}" \
  --train-id-property "${TRAIN_ID_PROPERTY}" \
  --test-id-property "${TEST_ID_PROPERTY}"
