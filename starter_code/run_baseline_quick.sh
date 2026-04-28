#!/usr/bin/env bash
set -euo pipefail
cd /home/sallen/jittor-pointcloud-denoise/starter_code
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
