#!/bin/bash
set -e
source "$(dirname "$0")/_ci_setup.sh"
ci_install_package
pytest tests/ -m "cuda and not mp and not e2e" --tb=short -v -q
