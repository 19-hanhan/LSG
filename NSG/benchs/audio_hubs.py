# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import time
import sys
import numpy as np
# sys.path.append('/home/wanghongya/bab/faiss-1.7.1/benchs')
sys.path.append('/home/wanghongya/wys/faiss171/python')

import faiss


from datasets import load_sift1M
from datasets import load_bigann
from datasets import load_deep
from datasets import load_gist
from datasets import load_glove
from datasets import load_sun
from datasets import load_audio
from datasets import load_random
from datasets import load_imageNet
from datasets import load_tiny5m


k = int(sys.argv[1])
m = int(sys.argv[2])

r1 = float(sys.argv[3])
r2 = float(sys.argv[4])

nb1 = int(sys.argv[5])
nb2 = int(sys.argv[6])

todo = sys.argv[2:]

print("load data")

# xb, xq, xt, gt = load_sift1M()
# xb, xq, xt, gt = load_glove()
# xb, xq, xt, gt = load_deep()
# xb, xq, xt, gt = load_gist()
# xb, xq, xt, gt = load_sun()
xb, xq, xt, gt = load_audio()
# xb, xq, xt, gt = load_imageNet()
# xb, xq, xt, gt = load_random()
# xb, xq, xt, gt = load_crawl()
# xb, xq, xt, gt = load_tiny5m()



print("load load_audio")
print(k,m,r1,r2,nb1,nb2)
# xb, xq, xt, gt = load_sift1M()


nq, d = xq.shape

if todo == []:
    todo = 'hnsw hnsw_sq ivf ivf_hnsw_quantizer kmeans kmeans_hnsw nsg'.split()

times = []
recall = []
efSearch = []

def evaluate(index):
    # for timing with a single core
    # faiss.omp_set_num_threads(1)

    # t0 = time.time()
    # # D, I = index.search(xq, k)
    # D, I = index.search_enhence_with_hubs(xq,k,xb.shape[0])
    # t1 = time.time()
    D = np.empty((xq.shape[0], k), dtype=np.float32)
    I = np.empty((xq.shape[0], k), dtype=np.int64)

    t0 = time.time()
    index.search_enhence_with_hubs(xq.shape[0], faiss.swig_ptr(xq), k, faiss.swig_ptr(D), faiss.swig_ptr(I),xb.shape[0])
    t1 = time.time()

    missing_rate = (I == -1).sum() / float(k * nq)
    recall_at_1 = (I == gt[:, :1]).sum() / float(nq)
    print("\t %7.3f ms per query, R@1 %.4f, missing rate %.4f" % (
        (t1 - t0) * 1000.0 / nq, recall_at_1, missing_rate))

    efSearch.append(index.nsg.search_L)
    recall.append(float(format(recall_at_1, '.4f')))
    times.append(float(format((t1-t0)*1000.0/nq, '.4f')))
    return recall_at_1
if 'nsg' in todo:

    print("Testing NSG Flat")

    index = faiss.IndexNSGFlat(d, m)
    index.build_type = 1
    # training is not needed

    # this is the default, higher is more accurate and slower to
    # construct

    print("add")
    # to see progress
    index.verbose = True
    index.add(xb)

    ratios = np.empty((1, 2), dtype=np.float32)
    nb_nbors_per_level = np.empty((1, 2), dtype=np.int32)

    ratios[0][0]=r1
    ratios[0][1]=r2

    nb_nbors_per_level[0][0]=nb1
    nb_nbors_per_level[0][1]=nb2

    index.combine_index_with_hot_hubs_enhence(xb.shape[0],2,faiss.swig_ptr(ratios),
    faiss.swig_ptr(nb_nbors_per_level))

    print("search")
    times = []
    recall = []
    efSearch = []
    for search_L in 100,200,300,400,500,600,700,800,900,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000:
        print("search_L", search_L, end=' ')
        index.nsg.search_L = search_L
        r = evaluate(index)
        if r == 1:
            break
    print(times)
    print(recall)
    print(efSearch)



if 'hnsw' in todo:

    print("Testing HNSW Flat")

    index = faiss.IndexHNSWFlat(d, m)

    # training is not needed

    # this is the default, higher is more accurate and slower to
    # construct
    index.hnsw.efConstruction = 40

    print("add")
    # to see progress
    index.verbose = True
    index.add(xb)

    print("search")
    for efSearch in 16, 32, 64, 128, 256:
        for bounded_queue in [True, False]:
            print("efSearch", efSearch, "bounded queue", bounded_queue, end=' ')
            index.hnsw.search_bounded_queue = bounded_queue
            index.hnsw.efSearch = efSearch
            evaluate(index)

if 'hnsw_sq' in todo:

    print("Testing HNSW with a scalar quantizer")
    # also set M so that the vectors and links both use 128 bytes per
    # entry (total 256 bytes)
    index = faiss.IndexHNSWSQ(d, faiss.ScalarQuantizer.QT_8bit, 16)

    print("training")
    # training for the scalar quantizer
    index.train(xt)

    # this is the default, higher is more accurate and slower to
    # construct
    index.hnsw.efConstruction = 40

    print("add")
    # to see progress
    index.verbose = True
    index.add(xb)

    print("search")
    for efSearch in 16, 32, 64, 128, 256:
        print("efSearch", efSearch, end=' ')
        index.hnsw.efSearch = efSearch
        evaluate(index)

if 'ivf' in todo:

    print("Testing IVF Flat (baseline)")
    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFFlat(quantizer, d, 16384)
    index.cp.min_points_per_centroid = 5   # quiet warning

    # to see progress
    index.verbose = True

    print("training")
    index.train(xt)

    print("add")
    index.add(xb)

    print("search")
    for nprobe in 1, 4, 16, 64, 256:
        print("nprobe", nprobe, end=' ')
        index.nprobe = nprobe
        evaluate(index)

if 'ivf_hnsw_quantizer' in todo:

    print("Testing IVF Flat with HNSW quantizer")
    quantizer = faiss.IndexHNSWFlat(d, m)
    index = faiss.IndexIVFFlat(quantizer, d, 16384)
    index.cp.min_points_per_centroid = 5   # quiet warning
    index.quantizer_trains_alone = 2

    # to see progress
    index.verbose = True

    print("training")
    index.train(xt)

    print("add")
    index.add(xb)

    print("search")
    quantizer.hnsw.efSearch = 64
    for nprobe in 1, 4, 16, 64, 256:
        print("nprobe", nprobe, end=' ')
        index.nprobe = nprobe
        evaluate(index)

# Bonus: 2 kmeans tests

if 'kmeans' in todo:
    print("Performing kmeans on sift1M database vectors (baseline)")
    clus = faiss.Clustering(d, 16384)
    clus.verbose = True
    clus.niter = 10
    index = faiss.IndexFlatL2(d)
    clus.train(xb, index)


if 'kmeans_hnsw' in todo:
    print("Performing kmeans on sift1M using HNSW assignment")
    clus = faiss.Clustering(d, 16384)
    clus.verbose = True
    clus.niter = 10
    index = faiss.IndexHNSWFlat(d, m)
    # increase the default efSearch, otherwise the number of empty
    # clusters is too high.
    index.hnsw.efSearch = 128
    clus.train(xb, index)

