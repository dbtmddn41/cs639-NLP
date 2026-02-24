#!/usr/bin/env bash
# Reproduce BPE tokenizer training results.
# Usage: bash run.sh
# Requires uv (https://github.com/astral-sh/uv) and Python 3.12.

set -e

cd "$(dirname "$0")"

uv run --no-project --python 3.12 --with regex tokenizer_hw.py
