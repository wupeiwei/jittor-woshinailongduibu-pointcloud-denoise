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
rm -rf /home/sallen/.cache/jittor/jt1.3.10/g++13.3.0/py3.12.3 || true
source .venv/bin/activate
printf '== compiler ==\n'
which gcc; gcc --version | head -1
which g++; g++ --version | head -1
printf '\n== jittor smoke ==\n'
python - <<'PY'
import os, shutil
print('which g++', shutil.which('g++'))
import jittor as jt
print('jittor', jt.__version__)
jt.flags.use_cuda = 1
a = jt.ones((2,3))
print((a+2).numpy())
PY
printf '\n== py compile starter ==\n'
python -m py_compile run.py evaluate.py src/data/*.py src/model/*.py src/system/*.py
