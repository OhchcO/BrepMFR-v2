#!/usr/bin/env bash
# Sample MFRInstSeg and convert the selected train/val models to BrepMFR BIN.
#
# Intended shell: Git Bash on Windows.  Edit only the configuration block
# below for a different sampling ratio, seed, environment, or data location.

set -Eeuo pipefail

# ----------------------------- configuration -----------------------------
PYTHON_EXE="G:/ProgramData/anaconda3/envs/aagmspenv/python.exe"
PROJECT_ROOT="D:/Code/BrepMFR"
DATASET_ROOT="D:/Code/Datasets/MFRInstSeg"
SUBSET_ROOT="${DATASET_ROOT}/subsets/train_val_1pct_balanced"
OUTPUT_ROOT="D:/Code/Datasets/MFRInstSeg_BrepMFR"

EXTRACTOR_FILE="D:/Code/BrepMFR-g/data_set_line/AAGExtractor.py"
SCHEMA_FILE="D:/Code/BrepMFR-g/data_set_line/schema.json"

# 0.01 means 1%.  The ratio is applied to train and val independently.
RATIO="0.01"
SEED="2026"
SWAP_ATTEMPTS="20000"

# Set to 1 to rebuild existing BIN files, or leave at 0 to keep valid files.
OVERWRITE="0"

# Leave empty for all selected models.  For a smoke test, use e.g. LIMIT="1".
LIMIT=""
# -------------------------------------------------------------------------

SAMPLER="${PROJECT_ROOT}/tools/sample_mfrinstseg_1pct.py"
CONVERTER="${PROJECT_ROOT}/tools/mfrinstseg_to_bin.py"

if [[ ! -f "${PYTHON_EXE}" ]]; then
    echo "Python executable not found: ${PYTHON_EXE}" >&2
    exit 2
fi
if [[ ! -f "${SAMPLER}" || ! -f "${CONVERTER}" ]]; then
    echo "BrepMFR preprocessing scripts were not found under: ${PROJECT_ROOT}/tools" >&2
    exit 2
fi

echo "[1/2] Selecting balanced train/val subsets (ratio=${RATIO}, seed=${SEED})"
"${PYTHON_EXE}" "${SAMPLER}" \
    --dataset-root "${DATASET_ROOT}" \
    --output-dir "${SUBSET_ROOT}" \
    --ratio "${RATIO}" \
    --seed "${SEED}" \
    --swap-attempts "${SWAP_ATTEMPTS}"

converter_args=(
    --dataset-root "${DATASET_ROOT}"
    --subset-root "${SUBSET_ROOT}"
    --output-root "${OUTPUT_ROOT}"
    --extractor-file "${EXTRACTOR_FILE}"
    --schema-file "${SCHEMA_FILE}"
    --split all
)

if [[ "${OVERWRITE}" == "1" ]]; then
    converter_args+=(--overwrite)
fi
if [[ -n "${LIMIT}" ]]; then
    converter_args+=(--limit "${LIMIT}")
fi

echo "[2/2] Converting selected STEP files to BIN"
"${PYTHON_EXE}" "${CONVERTER}" "${converter_args[@]}"

echo "Preparation completed."
echo "BIN dataset: ${OUTPUT_ROOT}"
echo "Train manifest: ${OUTPUT_ROOT}/train.txt"
echo "Validation manifest: ${OUTPUT_ROOT}/val.txt"
echo "Failure report: ${OUTPUT_ROOT}/conversion_failures.json"
