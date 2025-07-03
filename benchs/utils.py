import sys
import struct
import numpy as np

## Read for numpy array
def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()

def fvecs_read(fname):
    return ivecs_read(fname).view('float32')

# Read for memory map
def mmap_ivecs(fname):
    a = np.memmap(fname, dtype='int32', mode='r')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:]

def mmap_fvecs(fname):
    x = np.memmap(fname, dtype='int32', mode='r')
    d = x[0]
    return x.view('float32').reshape(-1, d + 1)[:, 1:]

def mmap_bvecs(fname):
    x = np.memmap(fname, dtype='uint8', mode='r')
    d = x[:4].view('int32')[0]
    return x.reshape(-1, d + 4)[:, 4:]

## Convert memory map to numpy array
def sanitize(x):
    return np.ascontiguousarray(x.astype('float32'))

## Save numpy array to vecs format
def save_vecs(fname, data):
    dim = data.shape[1]
    with open(fname, 'wb') as f:
        for vec in data:
            f.write(np.int32(dim).tobytes())
            f.write(vec.tobytes())

#########################

def load_data(prefix, dataset, *, ntrain=65536):
    print(f"Loading {dataset}...", end='', file=sys.stderr)
    basedir = f'{prefix}/{dataset}/'
    xb = mmap_fvecs(f'{basedir}/{dataset}_base.fvecs')
    xq = mmap_fvecs(f'{basedir}/{dataset}_query.fvecs')
    gt = ivecs_read(f'{basedir}/{dataset}_groundtruth.ivecs')
    xb = sanitize(xb)
    xt = sanitize(xb[:ntrain])
    xq = sanitize(xq)
    print("done", file=sys.stderr)
    return xb, xq, xt, gt

def matrix_recall(ids, gt, len):
    return np.sum(ids == gt[:, :len], axis=1).sum() / float(ids.shape[0])
