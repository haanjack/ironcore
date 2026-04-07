#!/usr/bin/env bash
# Copyright (c) 2025-2026 Jaegeun Han
# SPDX-License-Identifier: Apache-2.0
#
# Single-configuration Nsight Systems profiling wrapper.
#
# Usage:
#   bash scripts/profile_nsys.sh <config_yaml> <profile_name> [ngpu]
#
# Examples:
#   bash scripts/profile_nsys.sh configs/profile/base.yaml baseline 2
#   bash scripts/profile_nsys.sh configs/profile/base.yaml attn_flash 1
#
# The script will:
#   1. Create output directory logs/profile/<profile_name>/
#   2. Run nsys profile around torchrun with the given config
#   3. Export .nsys-rep and .jsonl for analysis
#
# Prerequisites:
#   - nsys (Nsight Systems CLI) must be on PATH
#   - CUDA-capable GPUs must be available

set -euo pipefail

# --- Arguments ---
CONFIG_PATH="${1:?Usage: $0 <config_yaml> <profile_name> [ngpu]}"
PROFILE_NAME="${2:?Usage: $0 <config_yaml> <profile_name> [ngpu]}"
NGPU="${3:-2}"

# --- Paths ---
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_DIR="${PROJECT_ROOT}/logs/profile/${PROFILE_NAME}"

# --- Validate ---
if ! command -v nsys &>/dev/null; then
    echo "ERROR: nsys not found. Install Nsight Systems or add to PATH." >&2
    exit 1
fi

if [ ! -f "${CONFIG_PATH}" ]; then
    # Try relative to project root
    if [ -f "${PROJECT_ROOT}/${CONFIG_PATH}" ]; then
        CONFIG_PATH="${PROJECT_ROOT}/${CONFIG_PATH}"
    else
        echo "ERROR: Config file not found: ${CONFIG_PATH}" >&2
        exit 1
    fi
fi

# --- Setup ---
mkdir -p "${OUTPUT_DIR}"

echo "=========================================="
echo "Nsight Systems Profile: ${PROFILE_NAME}"
echo "=========================================="
echo "  Config:     ${CONFIG_PATH}"
echo "  GPUs:       ${NGPU}"
echo "  Output dir: ${OUTPUT_DIR}"
echo ""

# --- Profile ---
# ironcore's profiler starts at step 2 and ends at step 4 (set in base.yaml),
# so nsys captures the full run and ironcore's NVTX markers annotate the
# interesting iterations.
#
# Traces: nvtx (ironcore markers), cudnn, cuda, cublas, osrt
# --cuda-memory-usage: track allocations for memory analysis
# --stats: print summary statistics after profiling
# --force-overwrite: allow re-running without manual cleanup
nsys profile \
    --trace=nvtx,cudnn,cuda,cublas,osrt \
    --cuda-memory-usage=true \
    --output="${OUTPUT_DIR}/profile_${PROFILE_NAME}" \
    --force-overwrite=true \
    --stats=true \
    torchrun \
        --nproc_per_node="${NGPU}" \
        -m ironcore train \
        --config "${CONFIG_PATH}"

# --- Export jsonl for automated analysis ---
NSYS_REP="${OUTPUT_DIR}/profile_${PROFILE_NAME}.nsys-rep"
if [ -f "${NSYS_REP}" ]; then
    echo ""
    echo "Exporting to SQLite for analysis..."
    nsys export \
        --type=sqlite \
        --output="${OUTPUT_DIR}/profile_${PROFILE_NAME}.sqlite" \
        --force-overwrite=true \
        "${NSYS_REP}" 2>/dev/null || true
    echo "  -> ${OUTPUT_DIR}/profile_${PROFILE_NAME}.sqlite"
fi

echo ""
echo "=========================================="
echo "Profile complete: ${PROFILE_NAME}"
echo "  Report: ${NSYS_REP}"
echo "=========================================="
