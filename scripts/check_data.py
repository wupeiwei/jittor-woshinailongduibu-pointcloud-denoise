#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', default='dataset_train')
    p.add_argument('--test-root', default='dataset_test_noisy')
    p.add_argument('--train-list', default='starter_code/datalist/train.txt')
    p.add_argument('--limit', type=int, default=5)
    args = p.parse_args()

    data_root = Path(args.data_root)
    test_root = Path(args.test_root)
    train_list = Path(args.train_list)
    print('data_root:', data_root.resolve(), 'exists=', data_root.exists())
    print('test_root:', test_root.resolve(), 'exists=', test_root.exists())
    print('train_list:', train_list.resolve(), 'exists=', train_list.exists())

    if train_list.exists():
        ids = [x.strip() for x in train_list.read_text().splitlines() if x.strip()]
        print('train_list entries:', len(ids))
        ok = 0
        missing = []
        for x in ids[:max(args.limit, 0)]:
            rel = Path(x)
            if rel.parts and rel.parts[0] == 'shapenet':
                f = data_root / rel / 'models' / 'model_normalized.obj'
            else:
                f = data_root / 'shapenet' / rel / 'models' / 'model_normalized.obj'
            if f.exists():
                ok += 1
            else:
                missing.append(str(f))
        print(f'train obj sample ok: {ok}/{min(len(ids), args.limit)}')
        if missing:
            print('missing samples:')
            for m in missing[:10]: print(' ', m)

    noisy = sorted(test_root.glob('shapenet/*/*/noisy.npy'))
    print('test noisy.npy count:', len(noisy))
    for f in noisy[:args.limit]:
        arr = np.load(f, mmap_mode='r')
        print(' ', f, arr.shape, arr.dtype)


if __name__ == '__main__':
    main()
