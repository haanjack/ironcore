#!/bin/bash
set -e
source "$(dirname "$0")/_ci_setup.sh"
ci_install_package
pytest tests/ -m "e2e" -v --tb=short
