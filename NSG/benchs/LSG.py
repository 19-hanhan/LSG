# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import sys
import os

import faiss

import time
import numpy as np
import pandas as pd

# faiss.omp_set_num_threads(8)

# LSG-NICDM
def clac_nicdm_avgdis_ivf(ls_k = 10,sr = 1/1000):
    print(f"clac k{ls_k} by ivf")
    index = faiss.IndexFlatL2(d)
    samples = max(int(sr * xb.shape[0]),10000)
    if(dataset_name == 'spacev'): samples = max(samples,100000) # the biggest datasets use more samples
    rnd_rows = np.random.choice(xb.shape[0], samples, replace=False)
    index.add(xb[rnd_rows])
    t0 = time.time()
    avgdis = []
    D, I = index.search(xb, ls_k)
    t1 = time.time()
    avgdis.extend(np.mean(D, axis=1))
    print(f"ivf get knn use time {t1 - t0}s")
    return np.array(avgdis)

start_time = time.time()

dataset_name = 'sift1M'

if(dataset_name == 'sift1M'):
    from datasets import load_sift1M
    xb, xq, xt, gt = load_sift1M()

if(dataset_name == 'glove'):
    from datasets import load_glove
    xb, xq, xt, gt = load_glove()

if(dataset_name == 'glove2M'):
    from datasets import load_glove2m
    xb, xq, gt = load_glove2m()

if(dataset_name == 'word2vec'):
    from datasets import load_word2vec
    xb, xq, xt, gt = load_word2vec()

if(dataset_name == 'deep10M'):
    from datasets import load_deep
    xb, xq, xt, gt = load_deep()

if(dataset_name == 'gist'):
    from datasets import load_gist
    xb, xq, xt, gt = load_gist()

if(dataset_name == 'bigann' or dataset_name == 'sift1B'):
    from datasets import load_bigann
    xb, xq, xt, gt = load_bigann()

if(dataset_name == 'sun'):
    from datasets import load_sun
    xb, xq, xt, gt = load_sun()

if(dataset_name == 'audio'):
    from datasets import load_audio
    xb, xq, xt, gt = load_audio()

if(dataset_name == 'imageNet'):
    from datasets import load_imageNet
    xb, xq, xt, gt = load_imageNet()

if(dataset_name == 'random'):
    from datasets import load_random
    xb, xq, xt, gt = load_random()

if(dataset_name == 'tiny5M'):
    from datasets import load_tiny5m
    xb, xq, xt, gt = load_tiny5m()

if(dataset_name == 'trevi'):
    from datasets import load_trevi
    xb, xq, gt = load_trevi()

if(dataset_name == 'enron'):
    from datasets import load_enron
    xb, xq, gt = load_enron()

if(dataset_name == 'crawl'):
    from datasets import load_crawl
    xb, xq, gt = load_crawl()

if(dataset_name == 'spacev'):
    from datasets import load_spacev
    xb, xq, gt = load_spacev()

if(dataset_name == 'msong'):
    from datasets import load_msong
    xb, xq, gt = load_msong()

nq, d = xq.shape

print(xb.shape)

print("Testing NSG Flat")
time_start = time.time()

index = faiss.IndexNSGFlat(d, 30)

index.build_type = 1

# to see progress
index.verbose = True

# nsg-lsg
# LSG init
avgdis = clac_nicdm_avgdis_ivf(20) # 20 is the normal value
index.set_nicdm_distance(faiss.swig_ptr(avgdis), 1.0)

print("add")
index.add(xb)

time_build = time.time()

print('build_time:',time_build-time_start)

def evaluate(index):
    k = 10 # recall@k
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

print("search")
times = []
recall = []
efSearch = []
for search_L in 100,200,300,400,500,600,700,800,900,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000: # normal
# for search_L in 10,20,30,40,50,60,70,80,90,100,125,150,175,200,225,250,275,300,350,400,450,500: # msong
    print("search_L", search_L, end=' ')
    index.nsg.search_L = search_L
    r = evaluate(index)
    if r == 1:
        break

print(efSearch)
print(times)
print(recall)

# The number of visit points during the search process
with open('count.txt', 'r') as file:count = [int(line.strip()) for line in file.readlines()]

time_end = time.time()

data = { 'efSearch':efSearch , 'recall':recall , 'times':times, 'count':count }

total_times = time_end-time_start
print('total_time:',total_times)

# save result to csv 
# pd.DataFrame(data).to_csv('/your path/'+dataset_name+'_lsg.csv', index=False)

os.remove('count.txt') # Del tmp file
