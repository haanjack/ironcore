#!/bin/bash
set -e
pip install -e ".[dev]" -q
pytest tests/ -m "e2e" -v --tb=short
