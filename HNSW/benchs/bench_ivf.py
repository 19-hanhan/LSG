import time
import sys
sys.path.insert(0, '/home/wanghongya/hanhan/faiss-1.5.0/python')
sys.path.insert(0, '/home/wanghongya/hanhan/faiss-1.5.0/benchs')
import faiss
import numpy as np
from datasets import sanitize
from datasets import fvecs_read
from datasets import load_sift10K
from datasets import load_sift1M
from datasets import load_bigann
from datasets import load_deep
from datasets import index_add_part
from datasets import load_audio
from datasets import load_glove
from datasets import load_glove2m
from datasets import load_imageNet
from datasets import load_random_gaussian
from datasets import load_word2vec

part_dataset = ['bigann', 'deep']

dataset = "imageNet"
dbsize = 1000

# 加载数据
print(f"load data: dataset[{dataset}]")
if dataset == "sift10K":
    xb, xq, xt, gt = load_sift10K()
    nb = xb.shape[0]
elif dataset == "audio":
    xb, xq, xt, gt = load_audio()
    nb = xb.shape[0]
elif dataset == "sift1M":
    xb, xq, xt, gt = load_sift1M()
    nb = xb.shape[0]
elif dataset == "glove1M":
    xb, xq, xt, gt = load_glove()
    nb = xb.shape[0]
elif dataset == "glove2M":
    xb, xq, xt, gt = load_glove2m()
    nb = xb.shape[0]
elif dataset == "random_gaussian":
    xb, xq, xt, gt = load_random_gaussian()
    nb = xb.shape[0]
elif dataset == "imageNet":
    xb, xq, xt, gt = load_imageNet()
    nb = xb.shape[0]
elif dataset == "word2vec":
    xb, xq, xt, gt = load_word2vec()
    nb = xb.shape[0]
elif dataset == "bigann":
    xb_map, xq, xt, gt = load_bigann(dbsize)
    nb = dbsize * 1000 * 1000
elif dataset == "deep":
    xb_map, xq, xt, gt = load_deep(dbsize)
    nb = dbsize * 1000 * 1000
else:
    print("dataset not exist")
    exit()
nq, d = xq.shape
print("nq: [%d], d: [%d]" % (nq, d))


index = faiss.IndexFlatL2(d)
if dataset in part_dataset:
    index.add(sanitize(xb_map[:nb]))
else:
    index.add(xb)

faiss.omp_set_num_threads(32)
t1 = time.time()
D, I = index.search(xq, 10)
t2 = time.time()
recall10 = (I[:, :10] == gt[:, :10]).sum() / (float(nq) * 10)

print(f"ivf clac use time {(t2 - t1):.3f} s, recall@10 = {recall10}")