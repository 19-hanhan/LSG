import time
import sys
sys.path.insert(0, '/home/wanghongya/hanhan/faiss-1.5.0/python')
sys.path.insert(0, '/home/wanghongya/hanhan/faiss-1.5.0/benchs')
import faiss
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from datasets import fvecs_read
from datasets import load_sift10K
from datasets import load_sift1M
from datasets import load_audio
from datasets import load_glove
from datasets import load_imageNet
from datasets import load_word2vec
from datasets import load_trevi
from datasets import load_random_gaussian

para = sys.argv
para = [None, 'sift1M', 0.01]
dataset = str(para[1])
rate = float(para[2])
nicdm_k = 10
basedir = '/home/wanghongya/hanhan/'


print(f"dataset[{dataset}]; rate[{rate}]; nicdm_k[{nicdm_k}]")

print("load data")
if dataset == 'glove1M':
    xb, xq, xt, gt = load_glove()
elif dataset == 'sift1M':
    xb, xq, xt, gt = load_sift1M()
elif dataset == 'imageNet':
    xb, xq, xt, gt = load_imageNet()
elif dataset == 'word2vec':
    xb, xq, xt, gt = load_word2vec()
elif dataset == 'random_gaussian':
    xb, xq, xt, gt = load_random_gaussian()
elif dataset == 'trevi':
    xb, xq, xt, gt = load_trevi()
else:
    exit()

rnd_rows = np.random.choice(xb.shape[0], int(rate * xb.shape[0]), replace=False)
xb_sampling = xb[rnd_rows]

def clac_nicdm_avgdis_ivf(bu, se):
    print(f"clac k{nicdm_k} by ivf{bu.shape}")
    index = faiss.IndexFlatL2(bu.shape[1])
    index.add(bu)
    t0 = time.time()
    faiss.omp_set_num_threads(32)
    D, I = index.search(se, nicdm_k)
    print(f"ivf get knn use time {time.time() - t0}s")
    return np.mean(D, axis=1)

# 从文件读knn
def read_knn(filename):
    t0 = time.time()
    avgdis = np.mean(fvecs_read(filename)[:, :nicdm_k], axis=1)
    t1 = time.time()
    print(f"ivf read knn from {filename} use time {t1 - t0}s")
    return avgdis

t0 = time.time()
avgdis = clac_nicdm_avgdis_ivf(xb_sampling, xb)
avgdis2 = clac_nicdm_avgdis_ivf(xb, xb)
# avgdis = read_knn(basedir + f'data/sampling1_knn/{dataset}_k1000.fvecs')
# avgdis2 = read_knn(basedir + f'data/sampling100_knn/{dataset}_k1000.fvecs')
print(f"get avg knn use time {time.time() - t0}s")

# 绘制密度图
t0 = time.time()
plt.figure(figsize=(8, 6))
sns.kdeplot(avgdis, color='blue', linewidth=2, label='knn_sampling')
sns.kdeplot(avgdis2, color='red', linewidth=2, label='knn')
print(f"plot use time {time.time() - t0}s")

# 添加标题和标签
plt.title(f'Density Plot of {dataset} k{nicdm_k} average')
plt.xlabel(f'k{nicdm_k} average')
plt.ylabel('Density')

# 添加图例
plt.legend()

plt.savefig(basedir + f'faiss-1.5.0/log/Density_{dataset}_{rate}.png')

# # 初始化 PCA 模型
# pca = PCA(n_components=2)

# # 对数据进行 PCA 降维
# xb_pca1 = pca.fit_transform(xb)
# xb_pca2 = pca.fit_transform(xb_sampling)

# # 绘制可视化图表
# plt.figure(figsize=(8, 6))
# plt.scatter(xb_pca1[:, 0], xb_pca1[:, 1], s=6, color='blue', label=f'{dataset}')
# plt.scatter(xb_pca2[:, 0], xb_pca2[:, 1], s=6, color='red', label=f'{dataset}_sampling')
# plt.title(f'PCA Visualization of {dataset}')
# # plt.xlabel('Principal Component 1')
# # plt.ylabel('Principal Component 2')
# plt.legend()
# plt.grid(True)
# plt.savefig(f'pca_{dataset}.png')
# # plt.show()