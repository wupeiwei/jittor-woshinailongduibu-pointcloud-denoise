#!/usr/bin/env python3
import os
import platform
import shutil
import subprocess
import sys

print('python:', sys.version.replace('\n', ' '))
print('platform:', platform.platform())
print('CC:', os.environ.get('CC'))
print('CXX:', os.environ.get('CXX'))
print('gcc:', shutil.which('gcc'))
print('g++:', shutil.which('g++'))
for cmd in [['gcc','--version'], ['g++','--version']]:
    try:
        print('$', ' '.join(cmd))
        print(subprocess.check_output(cmd, text=True).splitlines()[0])
    except Exception as e:
        print('failed:', e)

try:
    import jittor as jt
    jt.flags.use_cuda = 1
    x = jt.ones((2, 3))
    print('jittor:', jt.__version__)
    print('jittor_cuda_ok:', x.numpy().shape)
except Exception as e:
    print('jittor_check_failed:', repr(e))
    raise

try:
    out = subprocess.check_output(['nvidia-smi', '-L'], text=True)
    print('nvidia-smi -L:')
    print(out.strip())
except Exception as e:
    print('nvidia-smi failed:', e)
