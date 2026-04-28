#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p .toolchain-gcc10
ln -sfn /usr/bin/gcc-10 .toolchain-gcc10/gcc
ln -sfn /usr/bin/g++-10 .toolchain-gcc10/g++
ln -sfn /usr/bin/gcc-10 .toolchain-gcc10/cc
ln -sfn /usr/bin/g++-10 .toolchain-gcc10/c++
export PATH="$PWD/.toolchain-gcc10:$PATH"
export CC=gcc
export CXX=g++
export DISABLE_MULTIPROCESSING=1
source .venv/bin/activate
python run.py --task "$1"
