#!/bin/bash
# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0
#
# Local test runner script for ironcore
# Usage:
#   ./scripts/run_tests.sh              # Run all tests
#   ./scripts/run_tests.sh unit         # Run unit tests only
#   ./scripts/run_tests.sh integration  # Run integration tests only
#   ./scripts/run_tests.sh multi_gpu    # Run multi-GPU tests only

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== ironcore Test Runner ===${NC}"
echo ""

run_unit_tests() {
    echo -e "${YELLOW}Running Unit Tests...${NC}"
    pytest tests/unit/ --tb=short -q
    echo ""
}

run_regression_tests() {
    echo -e "${YELLOW}Running Regression Tests...${NC}"
    pytest tests/regression/ --tb=short -q
    echo ""
}

run_integration_tests() {
    echo -e "${YELLOW}Running Integration Tests...${NC}"
    pytest tests/integration/ \
        --ignore=tests/integration/kvcache/test_kv_cache_tp2.py \
        --ignore=tests/integration/parallelism/ \
        --tb=short -q
    echo ""
}

run_multi_gpu_tests() {
    echo -e "${YELLOW}Running Multi-GPU Tests (requires 2 GPUs)...${NC}"
    if command -v torchrun &> /dev/null; then
        torchrun --nproc_per_node=2 -m pytest tests/multi_gpu/ --tb=short -q
    else
        echo -e "${RED}torchrun not found. Please install torch with distributed support.${NC}"
        exit 1
    fi
    echo ""
}

run_all_tests() {
    run_unit_tests
    run_regression_tests
    run_integration_tests
    echo -e "${GREEN}Note: Multi-GPU tests require explicit 'multi_gpu' argument${NC}"
}

# Main logic
case "${1:-all}" in
    unit)
        run_unit_tests
        ;;
    regression)
        run_regression_tests
        ;;
    integration)
        run_integration_tests
        ;;
    multi_gpu|multigpu|gpu)
        run_multi_gpu_tests
        ;;
    all)
        run_all_tests
        ;;
    *)
        echo "Unknown test type: $1"
        echo "Usage: $0 [unit|regression|integration|multi_gpu|all]"
        exit 1
        ;;
esac

echo -e "${GREEN}=== Tests Complete ===${NC}"
