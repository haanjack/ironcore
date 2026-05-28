#!/bin/bash
set -e
pip install -e ".[dev]" -q
pytest tests/ -m "cuda and not mp and not e2e" --tb=short -v -q
