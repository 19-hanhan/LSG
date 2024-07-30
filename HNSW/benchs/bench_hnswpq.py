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

# global
INF = int(1e9 + 7)
re = [1, 10, 100, 500] # recall的输出值
ef = [#4, 8, 12, 16, 32, 64, 128, 256, 
      500, 800, 1000, 2000, 3000, 4000, 
      5000, 7000, 9000, 11000, 13000, 15000, 
      17000, 20000] # efsearch的输出值
part_dataset = ['bigann', 'deep']


# 加载输入参数
para = sys.argv
# para = [None, 'glove1M', 5, 300, 20, 0.01]
# para = [None, 'bigann', 5, 300, 16, 0.0001, 1000]
dataset = str(para[1])
k = 500
m = int(para[2])
efConstruction = int(para[3])
maxInDegree = INF
splitRate = 1.0
r1 = 0.01
r2 = 0.01
nb1 = 4
nb2 = 2
pq_m = int(para[4])
dis_method = 'NICDM'
ls_k = 10
alpha = 1.0
sr = float(para[5]) # sampling rate
if dataset in part_dataset:
    dbsize = int(para[6])
    step = int(1e6)
    print(f"load data: dbsize[{dbsize}],step[{step}]")
# python benchs/bench_hnswpq.py sift1M 5 300 16 10
# python benchs/bench_hnswpq.py bigann 5 300 16 10 1000

# 加载数据
print(f"load data: dataset[{dataset}],m[{m}],efc[{efConstruction}],dis_method[{dis_method}],ls_k[{ls_k}],alpha[{alpha}],sampling_rate[{sr}]")
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

def evaluate_hnswpq(index):
    faiss.omp_set_num_threads(1)
    index.hnsw.resetCount()
    t0 = time.time()
    D, I = index.search(xq, k)
    t1 = time.time()
    recalls = {i: (I[:, :i] == gt[:, :1]).sum() / float(nq) for i in re}
    computer_count = index.hnsw.get_computer_count()
    missing_rate = (I == -1).sum() / float(k * nq)
    # print(f"\t {((t1 - t0) * 1000.0 / nq):.4f} ms per query, "
    #       f"R@: {recalls}, dis per count:[{computer_count // nq}], "
    #       f"missing rate:[{missing_rate:.4f}]")
    print(f"{((t1 - t0) * 1000.0 / nq):.4f};{recalls[1]};"
          f"{recalls[10]};{recalls[100]};{recalls[500]};"
          f"{computer_count // nq};{missing_rate:.4f}")
        
def evaluate_hnswpq_hot(index):
    faiss.omp_set_num_threads(1)
    index.hnsw.resetCount()
    D = np.empty((xq.shape[0], k), dtype=np.float32)
    I = np.empty((xq.shape[0], k), dtype=np.int64)
    t0 = time.time()
    index.search_with_hot_hubs_enhence(xq.shape[0], 
            faiss.swig_ptr(xq), k, faiss.swig_ptr(D), 
            faiss.swig_ptr(I),nb)
    t1 = time.time()
    recalls = {i: (I[:, :i] == gt[:, :1]).sum() / float(nq) for i in re}
    computer_count = index.hnsw.get_computer_count()
    missing_rate = (I == -1).sum() / float(k * nq)
    # print(f"\t {((t1 - t0) * 1000.0 / nq):.4f} ms per query, "
    #       f"R@: {recalls}, dis per count:[{computer_count // nq}], "
    #       f"missing rate:[{missing_rate:.4f}]")
    print(f"{((t1 - t0) * 1000.0 / nq):.4f};{recalls[1]};"
          f"{recalls[10]};{recalls[100]};{recalls[500]};"
          f"{computer_count // nq};{missing_rate:.4f}")

def search_without_hot(index):
    print("evaluate_hnswpq")
    print("efsearch;per query;recall@1;recall@10;recall@100;recall@500;clac count;missing rate")
    for efSearch in ef:
        # print("efSearch", efSearch, end=' ')
        print(efSearch, end=';')
        index.hnsw.efSearch = efSearch
        evaluate_hnswpq(index)

def search_hot(index):
    # 计算、获取热点
    ratios = np.empty((1, 2), dtype=np.float32)
    nb_nbors_per_level = np.empty((1, 2), dtype=np.int32)
    # ratios.fill(0.0001)
    # nb_nbors_per_level.fill(0)
    ratios[0][0] = r1
    ratios[0][1] = r2
    nb_nbors_per_level[0][0] = nb1
    nb_nbors_per_level[0][1] = nb2
    index.hnsw.getHotSet(nb, 2, faiss.swig_ptr(ratios))
    
    index.hnsw.statichotpercent(nb)
    index.combine_index_with_hot_hubs_enhence(nb, 2, faiss.swig_ptr(ratios), 0, 0,
        faiss.swig_ptr(nb_nbors_per_level))
    index.hnsw.statcihotlinknums(nb)

    print("evaluate_hnswpq_hot")
    print("efsearch;per query;recall@1;recall@10;recall@100;recall@500;clac count;missing rate")
    for efSearch in ef:
        # print("efSearch", efSearch, end=' ')
        print(efSearch, end=';')
        index.hnsw.efSearch = efSearch
        evaluate_hnswpq_hot(index)
    
# ivf计算knn均值
def clac_nicdm_avgdis_ivf():
    print(f"clac k{ls_k} by ivf")
    index = faiss.IndexFlatL2(d)
    if dataset in part_dataset:
        index.add(sanitize(xb_map[:int(nb * sr)]))
    else:
        rnd_rows = np.random.choice(xb.shape[0], int(sr * xb.shape[0]), replace=False)
        index.add(xb[rnd_rows])
    t0 = time.time()
    faiss.omp_set_num_threads(32)
    avgdis = []
    if dataset in part_dataset:
        end = nb
        step = nb // 10
        for i in range(0, end, step):
            t1 = time.time()
            D, I = index.search(sanitize(xb_map[i: min(end, i + step)]), ls_k)
            t2 = time.time()
            avgdis.extend(np.mean(D, axis=1))
            print(f"ivf clac {int(i/step+1)}/10 in {(t2 - t1):.3f} s")
        t3 = time.time()
        print(f"ivf get knn total use time {t3 - t0}s")
    else:
        D, I = index.search(xb, ls_k)
        t3 = time.time()
        avgdis.extend(np.mean(D, axis=1))
        print(f"ivf get knn use time {t3 - t0}s")
    return np.array(avgdis)
    
# 从文件读knn
def read_knn(filename):
    t0 = time.time()
    avgdis = np.mean(fvecs_read(filename)[:, :ls_k], axis=1)
    t1 = time.time()
    print(f"ivf read knn from {filename} use time {t1 - t0}s")
    return avgdis
    
def main():
    # 构造
    print("Testing HNSWPQ")
    index = faiss.IndexHNSWPQ(d, pq_m, m)#三个参数分别代表（向量维度，pq的分块数，图的边数）
    index.verbose = False
    index.storage.verbose = False
    index.hnsw.efConstruction = efConstruction
    index.hnsw.maxInDegree = maxInDegree
    index.hnsw.splitRate = splitRate
    index.hnsw.search_bounded_queue = False
    index.hnsw.search_mode = 0
    index.dis_method = dis_method
    if dis_method == 'NICDM':
        # avgdis = read_knn(f'/home/wanghongya/hanhan/data/sampling100_knn/{dataset}_k1000.fvecs')
        avgdis = clac_nicdm_avgdis_ivf()
        # avgdis = clac_nicdm_avgdis_hnsw()
        index.set_nicdm_distance(faiss.swig_ptr(avgdis), alpha)

    # 训练质心
    print("train, size", xt.shape)
    t0 = time.time()
    faiss.omp_set_num_threads(32)
    index.train(xt)
    t1 = time.time()
    print("  train in %.3f s" % (t1 - t0))

    # 添加节点
    print("add, size", nb)
    t0 = time.time()
    faiss.omp_set_num_threads(32)
    if dataset in part_dataset:
        index_add_part(index, xb_map, nb, step=step)
    else:
        index.add(xb)
    t1 = time.time()
    print("  add in total %.2f min" % ((t1 - t0) / 60))

    index.dis_method = 'L2'
    search_without_hot(index)
    search_hot(index)
    
if __name__ == '__main__':
    t0 = time.time()
    main()
    print(f"(efc={efConstruction}, maxInDegree={maxInDegree}) "
            f"use time {((time.time() - t0) / 60):.2f} min\n")