import time
import sys
sys.path.insert(0, '/home/wanghongya/hanhan/faiss-1.5.0/python')
sys.path.insert(0, '/home/wanghongya/hanhan/faiss-1.5.0/benchs')
import faiss
import numpy as np
from datasets import load_sift1M
from datasets import load_sift10K
from datasets import load_audio
from datasets import load_glove
from datasets import load_glove2m
from datasets import load_imageNet
from datasets import load_random_gaussian
from datasets import load_word2vec
from datasets import load_tiny80m

# 参数：k、m、efc、maxInDegree、splitRate、r1、r2、nb1、nb2
para = sys.argv
INF = int(1e9 + 7)
# para = [None, 'tiny80M', 5, 300, 10, 0.001]
ef = [10, 20, 30, 40, 50, 60, 70, 80, 90 
      , 100, 200, 300, 400, 500, 600, 700, 800, 900
      , 1000, 2000, 3000, 4000, 5000, 7000, 9000, 10000, 20000
      ]

# 加载输入参数
dataset = str(para[1])
k = 10
m = int(para[2])
efConstruction = int(para[3])
maxInDegree = INF
splitRate = 1.0
r1 = 0.01
r2 = 0.03
nb1 = 4
nb2 = 2
nicdm_k = int(para[4])
dis_method = 'NICDM'
alpha = 1.0
sr = float(para[5])
# python benchs/bench_hnswflat.py glove1M 5 300 10

# 加载数据
print(f"load data: dataset[{dataset}],m[{m}],efc[{efConstruction}],nicdm_k[{nicdm_k}],dis_method[{dis_method}],alpha[{alpha}]")
if dataset == "sift10K":
    xb, xq, xt, gt = load_sift10K()
elif dataset == "audio":
    xb, xq, xt, gt = load_audio()
elif dataset == "sift1M":
    xb, xq, xt, gt = load_sift1M()
elif dataset == "glove1M":
    xb, xq, xt, gt = load_glove()
elif dataset == "glove2M":
    xb, xq, xt, gt = load_glove2m()
elif dataset == "random_gaussian":
    xb, xq, xt, gt = load_random_gaussian()
elif dataset == "imageNet":
    xb, xq, xt, gt = load_imageNet()
elif dataset == "word2vec":
    xb, xq, xt, gt = load_word2vec()
elif dataset == "tiny80M":
    xb, xq, xt, gt = load_tiny80m()
else:
    print("dataset not exist")
    exit()
nq, d = xq.shape
nb = xb.shape[0]
print("nq:%d d:%d" %(nq, d))

def evaluate_hnswflat(index): 
    faiss.omp_set_num_threads(1)
    index.hnsw.resetCount()
    t0 = time.time()
    D, I = index.search(xq, k)
    t1 = time.time()
    computer_count = index.hnsw.get_computer_count()
    recall_at_1 = (I == gt[:, :1]).sum() / float(nq)
    # print("\t %7.3f ms per query, R@1: [%.4f], dis per count:[%d]" % 
    #       ((t1 - t0) * 1000.0 / nq, recall_at_1, computer_count // nq))
    print("%d;%.3f;%.4f;%d" % (index.hnsw.efSearch,
          (t1 - t0) * 1000.0 / nq, recall_at_1, computer_count // nq))
    return recall_at_1
        
def evaluate_hnswflat_hot(index):
    faiss.omp_set_num_threads(1)
    index.hnsw.resetCount()
    D = np.empty((xq.shape[0], k), dtype=np.float32)
    I = np.empty((xq.shape[0], k), dtype=np.int64)
    t0 = time.time()
    index.search_with_hot_hubs_enhence(xq.shape[0], faiss.swig_ptr(xq), k, faiss.swig_ptr(D), faiss.swig_ptr(I),xb.shape[0])
    t1 = time.time()
    computer_count = index.hnsw.get_computer_count()
    recall_at_1 = (I == gt[:, :1]).sum() / float(nq)
    # print("\t %7.3f ms per query, R@1: [%.4f], dis count:[%d]" % 
    #       ((t1 - t0) * 1000.0 / nq, recall_at_1, computer_count))
    print("%d;%.3f;%.4f;%d" % (index.hnsw.efSearch,
          (t1 - t0) * 1000.0 / nq, recall_at_1, computer_count // nq))
    return recall_at_1


# 构建NICDM搜索图获得knn
# def clac_nicdm_avgdis_hnsw():
#     print(f"add search graph: xb{xb.shape} k{nicdm_k} efs{qefs}")
#     rate = 0.01
#     index = faiss.IndexHNSWFlat(d, m)
#     index.hnsw.efConstruction = int(efConstruction * rate)
#     index.hnsw.maxInDegree = maxInDegree
#     index.hnsw.splitRate = splitRate
#     index.verbose = True
#     index.hnsw.search_bounded_queue = False
#     index.hnsw.search_mode = 0
#     index.dis_method = 'L2'
#     rnd_rows = np.random.choice(xb.shape[0], int(rate * xb.shape[0]), replace=False)
#     index.add(xb[rnd_rows])
#     index.hnsw.efSearch = qefs
#     t0 = time.time()
#     faiss.omp_set_num_threads(32)
#     D, I = index.search(xb, nicdm_k)
#     print(f"hnsw get knn use time {time.time() - t0}s")
#     return np.mean(D, axis=1)

# ivf计算knn均值
def clac_nicdm_avgdis_ivf():
    print(f"clac k{nicdm_k} by ivf")
    index = faiss.IndexFlatL2(d)
    rnd_rows = np.random.choice(xb.shape[0], int(sr * xb.shape[0]), replace=False)
    index.add(xb[rnd_rows])
    t0 = time.time()
    faiss.omp_set_num_threads(32)
    D, I = index.search(xb, nicdm_k)
    print(f"ivf get knn use time {time.time() - t0}s")
    return np.mean(D, axis=1)

# 构图
print(f"add {dis_method} HNSW Flat")
index = faiss.IndexHNSWFlat(d, m)
index.hnsw.efConstruction = efConstruction
index.hnsw.maxInDegree = maxInDegree
index.hnsw.splitRate = splitRate
index.verbose = True
index.hnsw.search_bounded_queue = False
index.hnsw.search_mode = 0
index.dis_method = dis_method
if dis_method == 'NICDM':
    avgdis = clac_nicdm_avgdis_ivf()
    # avgdis = clac_nicdm_avgdis_hnsw()
    index.set_nicdm_distance(faiss.swig_ptr(avgdis), alpha)
# faiss.omp_set_num_threads(32)
index.add(xb)

# 搜索结果
index.dis_method = 'L2'
print(f"efsearch;per_query;R@{k};per_count")
for efSearch in ef:
    # print("efSearch", efSearch, end=' ')
    index.hnsw.efSearch = efSearch
    if evaluate_hnswflat(index) == 1.0:
        break

# 计算、获取热点
ratios = np.empty((1, 2), dtype=np.float32)
nb_nbors_per_level = np.empty((1, 2), dtype=np.int32)
ratios[0][0] = r1
ratios[0][1] = r2
nb_nbors_per_level[0][0] = nb1
nb_nbors_per_level[0][1] = nb2
index.hnsw.getHotSet(nb, 2, faiss.swig_ptr(ratios))

index.hnsw.statichotpercent(nb)
index.combine_index_with_hot_hubs_enhence(nb, 2, faiss.swig_ptr(ratios), 0, 0,
    faiss.swig_ptr(nb_nbors_per_level))
index.hnsw.statcihotlinknums(nb)

print("evaluate_hnswflat_hot")
print(f"efsearch;per_query;R@{k};per_count")
for efSearch in ef:
    # print("efSearch", efSearch, end=' ')
    index.hnsw.efSearch = efSearch
    if evaluate_hnswflat_hot(index) == 1.0:
        break