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
#   ./run_tests.sh --e2e        # Run expensive e2e/rlvr smoke tests (2 GPUs, ~10 min)
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
RUN_E2E=false
RUN_PROFILER=false
NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l || echo 0)
PYTHON="python"
PYTEST="pytest"

# Dynamic port allocation to avoid EADDRINUSE when tests run in parallel
get_free_port() {
    python3 -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()"
}

# Shared torchrun file lists (DIST_TEST_FILES_NP1, DIST_TEST_FILES_NP2) — see
# that file's header for why this is factored out.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./distributed_test_files.sh
source "$SCRIPT_DIR/distributed_test_files.sh"

# Run a pytest invocation whose exit code isn't swallowed by `|| true`, and
# treat abnormal termination (crash, segfault, collection error — anything
# other than "ran cleanly" or "some tests failed") or a 0/0/0 result as a
# failure. Without this, a host-environment crash before pytest's summary
# line prints looks identical to "0 tests passed" and exits 0.
# Sets PYTEST_OUTPUT, PASSED, FAILED, SKIPPED, PYTEST_HEALTHY (0=ok, 1=bad).
run_pytest_capture() {
    set +e
    PYTEST_OUTPUT=$("$@" 2>&1)
    PYTEST_EXIT=$?
    set -e

    PASSED=$(echo "$PYTEST_OUTPUT" | grep -oP '\d+(?= passed)' | tail -1)
    PASSED=${PASSED:-0}
    FAILED=$(echo "$PYTEST_OUTPUT" | grep -oP '\d+(?= failed)' | tail -1)
    FAILED=${FAILED:-0}
    SKIPPED=$(echo "$PYTEST_OUTPUT" | grep -oP '\d+(?= skipped)' | tail -1)
    SKIPPED=${SKIPPED:-0}

    PYTEST_HEALTHY=0
    if [ "$PYTEST_EXIT" -ne 0 ] && [ "$PYTEST_EXIT" -ne 1 ]; then
        echo -e "${RED}pytest exited abnormally (code $PYTEST_EXIT) — crash or collection error, not a normal test failure.${NC}"
        PYTEST_HEALTHY=1
    elif [ "$((PASSED + FAILED + SKIPPED))" -eq 0 ]; then
        echo -e "${RED}0 tests reported (passed+failed+skipped=0) — treating as a failure.${NC}"
        PYTEST_HEALTHY=1
    fi
    if [ "$PYTEST_HEALTHY" -ne 0 ]; then
        FAILED=$((FAILED > 0 ? FAILED : 1))
    fi
}

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
        --e2e|--rlvr)
            RUN_E2E=true
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
            echo "  --e2e, --rlvr    Run expensive e2e smoke tests (rlvr+e2e markers, 2 GPUs, ~10 min)"
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
    run_pytest_capture $PYTEST tests/ -v --tb=short
    UNIT_END=$(date +%s)
    UNIT_DURATION=$((UNIT_END - UNIT_START))

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

# Local copies so --quick/insufficient-GPU handling below doesn't mutate the
# shared arrays sourced from distributed_test_files.sh.
INTEGRATION_NP1_FILES=("${DIST_TEST_FILES_NP1[@]}")
INTEGRATION_NP2_FILES=("${DIST_TEST_FILES_NP2[@]}")

# Check if we should skip multi-GPU tests
if [ "$QUICK_MODE" = true ]; then
    echo ""
    echo -e "${YELLOW}Skipping multi-GPU tests (--quick mode)${NC}"
    INTEGRATION_NP2_FILES=()
elif [ "$NUM_GPUS" -lt 2 ]; then
    echo ""
    echo -e "${YELLOW}Skipping multi-GPU tests (need 2+ GPUs, found $NUM_GPUS)${NC}"
    INTEGRATION_NP2_FILES=()
fi

if [ ${#INTEGRATION_NP1_FILES[@]} -gt 0 ] || [ ${#INTEGRATION_NP2_FILES[@]} -gt 0 ]; then
    print_header "Running Integration Tests (torchrun)"

    GPU_START=$(date +%s)
    GPU_PASSED=0
    GPU_FAILED=0
    GPU_SKIPPED=0

    # Run one torchrun-launched pytest file, accumulating into GPU_{PASSED,FAILED,SKIPPED}.
    run_torchrun_file() {
        local nproc=$1
        local test_file=$2

        if [ ! -f "$test_file" ]; then
            echo -e "${YELLOW}  Skipping: $test_file (not found)${NC}"
            return
        fi

        echo -e "\n${YELLOW}Running: $test_file (nproc_per_node=$nproc)${NC}"
        run_pytest_capture torchrun --nproc_per_node="$nproc" --master_port="$(get_free_port)" -m pytest "$test_file" -v --tb=short

        GPU_PASSED=$((GPU_PASSED + PASSED))
        GPU_FAILED=$((GPU_FAILED + FAILED))
        GPU_SKIPPED=$((GPU_SKIPPED + SKIPPED))

        if [ "$FAILED" -gt 0 ]; then
            echo -e "${RED}  ✗ Failed: $FAILED tests${NC}"
            echo "$PYTEST_OUTPUT" | grep -E "(FAILED|ERROR)" | head -10
        else
            echo -e "${GREEN}  ✓ Passed: $PASSED tests${NC}"
        fi
    }

    # Run single-GPU (torchrun-launched) integration tests
    for test_file in "${INTEGRATION_NP1_FILES[@]}"; do
        run_torchrun_file 1 "$test_file"
    done

    # Run 2-GPU integration + multi_gpu tests
    for test_file in "${INTEGRATION_NP2_FILES[@]}"; do
        run_torchrun_file 2 "$test_file"
    done

    GPU_END=$(date +%s)
    GPU_DURATION=$((GPU_END - GPU_START))

    TOTAL_PASSED=$((TOTAL_PASSED + GPU_PASSED))
    TOTAL_FAILED=$((TOTAL_FAILED + GPU_FAILED))
    TOTAL_SKIPPED=$((TOTAL_SKIPPED + GPU_SKIPPED))

    echo ""
    print_result "$GPU_PASSED" "$GPU_FAILED" "$GPU_SKIPPED" "${GPU_DURATION}s"
fi

# E2E / RLVR smoke tests (opt-in, ~10 min, 2 GPUs)
if [ "$RUN_E2E" = true ]; then
    print_header "Running E2E Smoke Tests (rlvr+e2e, ~10 min, 2 GPUs)"
    E2E_START=$(date +%s)

    run_pytest_capture $PYTEST tests/ -v -m "e2e" --tb=short
    E2E_END=$(date +%s)
    E2E_DURATION=$((E2E_END - E2E_START))

    TOTAL_PASSED=$((TOTAL_PASSED + PASSED))
    TOTAL_FAILED=$((TOTAL_FAILED + FAILED))
    TOTAL_SKIPPED=$((TOTAL_SKIPPED + SKIPPED))

    print_result "$PASSED" "$FAILED" "$SKIPPED" "${E2E_DURATION}s"

    if [ "$FAILED" -gt 0 ]; then
        echo -e "${RED}E2E tests failed.${NC}"
        echo "$PYTEST_OUTPUT" | tail -50
    fi
fi

# Profiler Tests
if [ "$RUN_PROFILER" = true ]; then
    print_header "Running Profiler Tests"
    PROF_START=$(date +%s)
    
    # Run profiler directory explicitly
    run_pytest_capture $PYTEST tests/unit/profiler/ -v --tb=short
    PROF_END=$(date +%s)
    PROF_DURATION=$((PROF_END - PROF_START))

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
