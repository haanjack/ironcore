#!/usr/bin/env bash
# Copyright (c) 2025-2026 Jaegeun Han
# SPDX-License-Identifier: Apache-2.0
#
# Multi-axis profiling sweep using Nsight Systems.
# Runs profile_nsys.sh across all configuration axes, producing one
# profile per configuration for comparative analysis.
#
# Usage:
#   bash scripts/profile_sweep.sh [--axes AXIS1,AXIS2,...] [--ngpu N]
#
# Examples:
#   bash scripts/profile_sweep.sh                         # all axes, 2 GPUs
#   bash scripts/profile_sweep.sh --axes attn,fsdp        # specific axes only
#   bash scripts/profile_sweep.sh --ngpu 1 --axes actckpt # single GPU, one axis
#
# Available axes: attn, tp, moe, fsdp, actckpt, kvcache
#
# Output structure:
#   logs/profile/
#   ├── attn_standard/   (profile_attn_standard.nsys-rep, .sqlite)
#   ├── attn_flash/
#   ├── tp_standard/
#   ├── tp_async/
#   ├── moe_no_ep/
#   ├── moe_ep2/
#   ├── fsdp_off/
#   ├── fsdp_on/
#   ├── actckpt_off/
#   ├── actckpt_on/
#   ├── kvcache_off/
#   └── kvcache_on/
#
# Prerequisites:
#   - 2 CUDA-capable GPUs (for TP and EP axes)
#   - nsys on PATH

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROFILE_SCRIPT="${SCRIPT_DIR}/profile_nsys.sh"
BASE_CONFIG="${PROJECT_ROOT}/configs/profile/base.yaml"
OVERLAY_DIR="${PROJECT_ROOT}/configs/profile"

# --- Parse arguments ---
NGPU=2
AXES="attn,tp,moe,fsdp,actckpt,kvcache"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --axes)   AXES="$2"; shift 2 ;;
        --ngpu)   NGPU="$2"; shift 2 ;;
        --help|-h)
            head -30 "$0" | grep '^#' | sed 's/^# \?//'
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

# --- Axis definitions ---
# Each axis is a pair of (profile_name, overlay_config)
# The overlay config is merged on top of base.yaml by passing both to ironcore
declare -A AXIS_CONFIGS

# Attention: standard vs flash
AXIS_CONFIGS[attn_standard]="attn_standard.yaml"
AXIS_CONFIGS[attn_flash]="attn_flash.yaml"

# TP communication: standard vs async (requires 2 GPUs)
AXIS_CONFIGS[tp_standard]="tp_standard.yaml"
AXIS_CONFIGS[tp_async]="tp_async.yaml"

# MoE: no EP vs EP=2 (requires 2 GPUs)
AXIS_CONFIGS[moe_no_ep]="moe_no_ep.yaml"
AXIS_CONFIGS[moe_ep2]="moe_ep2.yaml"

# FSDP: off vs on
AXIS_CONFIGS[fsdp_off]="fsdp_off.yaml"
AXIS_CONFIGS[fsdp_on]="fsdp_on.yaml"

# Activation checkpointing: off vs on
AXIS_CONFIGS[actckpt_off]="actckpt_off.yaml"
AXIS_CONFIGS[actckpt_on]="actckpt_on.yaml"

# KV cache in eval: off vs on
AXIS_CONFIGS[kvcache_off]="kvcache_off.yaml"
AXIS_CONFIGS[kvcache_on]="kvcache_on.yaml"

# --- Map axis names to profile names ---
declare -A AXIS_TO_PROFILES
AXIS_TO_PROFILES[attn]="attn_standard attn_flash"
AXIS_TO_PROFILES[tp]="tp_standard tp_async"
AXIS_TO_PROFILES[moe]="moe_no_ep moe_ep2"
AXIS_TO_PROFILES[fsdp]="fsdp_off fsdp_on"
AXIS_TO_PROFILES[actckpt]="actckpt_off actckpt_on"
AXIS_TO_PROFILES[kvcache]="kvcache_off kvcache_on"

# --- Build list of profiles to run ---
PROFILES_TO_RUN=()
IFS=',' read -ra REQUESTED_AXES <<< "${AXES}"
for axis in "${REQUESTED_AXES[@]}"; do
    axis="$(echo "$axis" | xargs)"  # trim whitespace
    if [[ -z "${AXIS_TO_PROFILES[$axis]+x}" ]]; then
        echo "ERROR: Unknown axis '${axis}'. Available: attn, tp, moe, fsdp, actckpt, kvcache" >&2
        exit 1
    fi
    for profile in ${AXIS_TO_PROFILES[$axis]}; do
        PROFILES_TO_RUN+=("$profile")
    done
done

# --- GPU requirement checks ---
GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l || echo 0)
echo "=========================================="
echo "Profiling Sweep"
echo "=========================================="
echo "  GPUs available: ${GPU_COUNT}"
echo "  GPUs per run:   ${NGPU}"
echo "  Axes:           ${AXES}"
echo "  Profiles:       ${#PROFILES_TO_RUN[@]}"
echo ""

if [[ "${GPU_COUNT}" -lt "${NGPU}" ]]; then
    echo "WARNING: Only ${GPU_COUNT} GPUs available but ${NGPU} requested." >&2
    echo "         TP and EP profiles may fail." >&2
    echo ""
fi

# --- Create merged config and run each profile ---
MERGED_DIR=$(mktemp -d)
trap 'rm -rf "${MERGED_DIR}"' EXIT

PASSED=0
FAILED=0
SKIPPED=0

for profile_name in "${PROFILES_TO_RUN[@]}"; do
    overlay="${AXIS_CONFIGS[$profile_name]}"
    overlay_path="${OVERLAY_DIR}/${overlay}"

    if [[ ! -f "${overlay_path}" ]]; then
        echo "SKIP: ${profile_name} (overlay not found: ${overlay})"
        ((SKIPPED++))
        continue
    fi

    # Determine GPU count for this profile
    profile_ngpu="${NGPU}"

    # Some profiles require specific GPU counts
    case "${profile_name}" in
        tp_*|moe_ep2)
            if [[ "${GPU_COUNT}" -lt 2 ]]; then
                echo "SKIP: ${profile_name} (requires 2 GPUs, only ${GPU_COUNT} available)"
                ((SKIPPED++))
                continue
            fi
            profile_ngpu=2
            ;;
    esac

    # Merge base + overlay into a temporary config
    # ironcore supports --config with YAML, so we merge manually
    merged_config="${MERGED_DIR}/${profile_name}.yaml"
    python3 -c "
import yaml, sys
base = yaml.safe_load(open('${BASE_CONFIG}'))
overlay = yaml.safe_load(open('${overlay_path}'))

def deep_merge(base, overlay):
    for k, v in overlay.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_merge(base[k], v)
        else:
            base[k] = v

deep_merge(base, overlay)
# Update output dir and profile name
base.setdefault('profiler', {})['output_dir'] = './logs/profile/${profile_name}/'
base.setdefault('profiler', {})['name'] = '${profile_name}'
yaml.dump(base, open('${merged_config}', 'w'), default_flow_style=False)
" 2>/dev/null

    echo "------------------------------------------"
    echo "Running: ${profile_name} (${profile_ngpu} GPUs)"
    echo "------------------------------------------"

    if bash "${PROFILE_SCRIPT}" "${merged_config}" "${profile_name}" "${profile_ngpu}"; then
        echo "  -> PASSED"
        ((PASSED++))
    else
        echo "  -> FAILED"
        ((FAILED++))
    fi
    echo ""
done

# --- Summary ---
echo "=========================================="
echo "Sweep Summary"
echo "=========================================="
echo "  Passed:  ${PASSED}"
echo "  Failed:  ${FAILED}"
echo "  Skipped: ${SKIPPED}"
echo "  Total:   ${#PROFILES_TO_RUN[@]}"
echo ""
echo "Results in: logs/profile/"
echo "=========================================="

if [[ "${FAILED}" -gt 0 ]]; then
    exit 1
fi
