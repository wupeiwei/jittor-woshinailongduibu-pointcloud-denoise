from pathlib import Path
import numpy as np
files = list(Path('results_quick').rglob('denoised.npy'))
print('denoised files', len(files))
for f in files:
    a = np.load(f)
    print(f, a.shape, a.dtype, float(a.min()), float(a.max()))
