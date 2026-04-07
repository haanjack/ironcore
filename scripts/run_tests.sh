#!/bin/bash
# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0
#
# Comprehensive test runner for ironcore test suite.
# Runs unit, integration, regression tests, and multi-GPU tests (if available).
#
# Usage:
#   ./run_tests.sh              # Run all tests
#   ./run_tests.sh --quick      # Skip multi-GPU tests
#   ./run_tests.sh --gpu-only   # Only run multi-GPU tests
#   ./run_tests.sh --help       # Show help

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
QUICK_MODE=false
GPU_ONLY=false
RUN_RLVR=false
RUN_PROFILER=false
NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l || echo 0)
PYTHON="python"
PYTEST="pytest"
TORCHRUN="torchrun"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --quick|-q)
            QUICK_MODE=true
            shift
            ;;
        --gpu-only|-g)
            GPU_ONLY=true
            shift
            ;;
        --rlvr)
            RUN_RLVR=true
            shift
            ;;
        --profiler)
            RUN_PROFILER=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --quick, -q      Skip multi-GPU tests"
            echo "  --gpu-only, -g   Only run multi-GPU tests"
            echo "  --rlvr           Run RLVR tests (requires API key, local only)"
            echo "  --profiler       Run profiler tests"
            echo "  --help, -h       Show this help message"
            echo ""
            echo "Detected GPUs: $NUM_GPUS"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Function to print section headers
print_header() {
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

# Function to print results
print_result() {
    local passed=$1
    local failed=$2
    local skipped=$3
    local duration=$4

    if [ "$failed" -eq 0 ]; then
        echo -e "${GREEN}✓ PASSED${NC} - ${passed} tests passed, ${skipped} skipped (${duration})"
    else
        echo -e "${RED}✗ FAILED${NC} - ${failed} tests failed, ${passed} passed (${duration})"
    fi
}

# Track overall results
TOTAL_PASSED=0
TOTAL_FAILED=0
TOTAL_SKIPPED=0
START_TIME=$(date +%s)

# Check if GPU-only mode
if [ "$GPU_ONLY" = true ]; then
    print_header "Multi-GPU Tests Only"
else
    # Run standard pytest tests
    print_header "Running Unit + Integration + Regression Tests"

    UNIT_START=$(date +%s)
    PYTEST_OUTPUT=$($PYTEST tests/ -v --tb=short 2>&1 || true)
    UNIT_END=$(date +%s)
    UNIT_DURATION=$((UNIT_END - UNIT_START))

    # Parse pytest output for results (default to 0 if no match)
    PASSED=$(echo "$PYTEST_OUTPUT" | grep -oP '\d+(?= passed)' | tail -1)
    PASSED=${PASSED:-0}
    FAILED=$(echo "$PYTEST_OUTPUT" | grep -oP '\d+(?= failed)' | tail -1)
    FAILED=${FAILED:-0}
    SKIPPED=$(echo "$PYTEST_OUTPUT" | grep -oP '\d+(?= skipped)' | tail -1)
    SKIPPED=${SKIPPED:-0}

    TOTAL_PASSED=$((TOTAL_PASSED + PASSED))
    TOTAL_FAILED=$((TOTAL_FAILED + FAILED))
    TOTAL_SKIPPED=$((TOTAL_SKIPPED + SKIPPED))

    print_result "$PASSED" "$FAILED" "$SKIPPED" "${UNIT_DURATION}s"

    # Exit early if tests failed
    if [ "$FAILED" -gt 0 ]; then
        echo ""
        echo -e "${RED}Standard tests failed. Skipping multi-GPU tests.${NC}"
        echo "$PYTEST_OUTPUT" | tail -50
        exit 1
    fi
fi

# Check if we should skip multi-GPU tests
if [ "$QUICK_MODE" = true ]; then
    echo ""
    echo -e "${YELLOW}Skipping multi-GPU tests (--quick mode)${NC}"
elif [ "$NUM_GPUS" -lt 2 ]; then
    echo ""
    echo -e "${YELLOW}Skipping multi-GPU tests (need 2+ GPUs, found $NUM_GPUS)${NC}"
else
    print_header "Running Multi-GPU Tests (TP=2)"

    GPU_START=$(date +%s)
    GPU_PASSED=0
    GPU_FAILED=0
    GPU_SKIPPED=0

    # Test files that require torchrun with 2 GPUs
    GPU_TEST_FILES=(
        "tests/integration/kvcache/test_kv_cache_tp2.py"
        "tests/integration/attention/test_chunked_tp.py"
        "tests/integration/attention/test_flash_attention_cache.py"
        "tests/integration/lora/test_lora_checkpoint.py"
        "tests/multi_gpu/test_ep_multi_gpu.py"
        "tests/multi_gpu/test_grad_norm_multi_gpu.py"
        "tests/multi_gpu/test_all_to_all_ep.py"
        "tests/multi_gpu/test_distributed_optimizer_checkpoint.py"
    )

    for test_file in "${GPU_TEST_FILES[@]}"; do
        if [ -f "$test_file" ]; then
            echo -e "\n${YELLOW}Running: $test_file${NC}"

            TEST_OUTPUT=$($TORCHRUN --nproc_per_node=2 -m pytest "$test_file" -v --tb=short 2>&1 || true)

            # Parse results
            T_PASSED=$(echo "$TEST_OUTPUT" | grep -oP '\d+(?= passed)' | tail -1)
            T_PASSED=${T_PASSED:-0}
            T_FAILED=$(echo "$TEST_OUTPUT" | grep -oP '\d+(?= failed)' | tail -1)
            T_FAILED=${T_FAILED:-0}
            T_SKIPPED=$(echo "$TEST_OUTPUT" | grep -oP '\d+(?= skipped)' | tail -1)
            T_SKIPPED=${T_SKIPPED:-0}

            GPU_PASSED=$((GPU_PASSED + T_PASSED))
            GPU_FAILED=$((GPU_FAILED + T_FAILED))
            GPU_SKIPPED=$((GPU_SKIPPED + T_SKIPPED))

            if [ "$T_FAILED" -gt 0 ]; then
                echo -e "${RED}  ✗ Failed: $T_FAILED tests${NC}"
                echo "$TEST_OUTPUT" | grep -E "(FAILED|ERROR)" | head -10
            else
                echo -e "${GREEN}  ✓ Passed: $T_PASSED tests${NC}"
            fi
        else
            echo -e "${YELLOW}  Skipping: $test_file (not found)${NC}"
        fi
    done

    GPU_END=$(date +%s)
    GPU_DURATION=$((GPU_END - GPU_START))

    TOTAL_PASSED=$((TOTAL_PASSED + GPU_PASSED))
    TOTAL_FAILED=$((TOTAL_FAILED + GPU_FAILED))
    TOTAL_SKIPPED=$((TOTAL_SKIPPED + GPU_SKIPPED))

    echo ""
    print_result "$GPU_PASSED" "$GPU_FAILED" "$GPU_SKIPPED" "${GPU_DURATION}s"
fi

# RLVR Tests
if [ "$RUN_RLVR" = true ]; then
    print_header "Running RLVR Tests (requires API key)"
    RLVR_START=$(date +%s)
    
    # Run only rlvr tests
    PYTEST_OUTPUT=$($PYTEST tests/ -v -m rlvr --tb=short 2>&1 || true)
    RLVR_END=$(date +%s)
    RLVR_DURATION=$((RLVR_END - RLVR_START))

    PASSED=$(echo "$PYTEST_OUTPUT" | grep -oP '\d+(?= passed)' | tail -1)
    PASSED=${PASSED:-0}
    FAILED=$(echo "$PYTEST_OUTPUT" | grep -oP '\d+(?= failed)' | tail -1)
    FAILED=${FAILED:-0}
    SKIPPED=$(echo "$PYTEST_OUTPUT" | grep -oP '\d+(?= skipped)' | tail -1)
    SKIPPED=${SKIPPED:-0}

    TOTAL_PASSED=$((TOTAL_PASSED + PASSED))
    TOTAL_FAILED=$((TOTAL_FAILED + FAILED))
    TOTAL_SKIPPED=$((TOTAL_SKIPPED + SKIPPED))

    print_result "$PASSED" "$FAILED" "$SKIPPED" "${RLVR_DURATION}s"
    
    if [ "$FAILED" -gt 0 ]; then
        echo -e "${RED}RLVR tests failed.${NC}"
        echo "$PYTEST_OUTPUT" | tail -50
    fi
fi

# Profiler Tests
if [ "$RUN_PROFILER" = true ]; then
    print_header "Running Profiler Tests"
    PROF_START=$(date +%s)
    
    # Run profiler directory explicitly
    PYTEST_OUTPUT=$($PYTEST tests/unit/profiler/ -v --tb=short 2>&1 || true)
    PROF_END=$(date +%s)
    PROF_DURATION=$((PROF_END - PROF_START))

    PASSED=$(echo "$PYTEST_OUTPUT" | grep -oP '\d+(?= passed)' | tail -1)
    PASSED=${PASSED:-0}
    FAILED=$(echo "$PYTEST_OUTPUT" | grep -oP '\d+(?= failed)' | tail -1)
    FAILED=${FAILED:-0}
    SKIPPED=$(echo "$PYTEST_OUTPUT" | grep -oP '\d+(?= skipped)' | tail -1)
    SKIPPED=${SKIPPED:-0}

    TOTAL_PASSED=$((TOTAL_PASSED + PASSED))
    TOTAL_FAILED=$((TOTAL_FAILED + FAILED))
    TOTAL_SKIPPED=$((TOTAL_SKIPPED + SKIPPED))

    print_result "$PASSED" "$FAILED" "$SKIPPED" "${PROF_DURATION}s"
    
    if [ "$FAILED" -gt 0 ]; then
        echo -e "${RED}Profiler tests failed.${NC}"
        echo "$PYTEST_OUTPUT" | tail -50
    fi
fi

# Final summary
END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))

print_header "Final Summary"
echo -e "Total Tests: ${GREEN}$TOTAL_PASSED${NC} passed, ${RED}$TOTAL_FAILED${NC} failed, ${YELLOW}$TOTAL_SKIPPED${NC} skipped"
echo -e "Total Duration: ${TOTAL_DURATION}s"

if [ "$TOTAL_FAILED" -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓ All tests passed!${NC}"
    exit 0
else
    echo ""
    echo -e "${RED}✗ Some tests failed.${NC}"
    exit 1
fi
