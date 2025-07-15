import sys
import time
sys.path.insert(0, './python')
import faiss
import numpy as np
from argparse import ArgumentParser
from utils import load_data, matrix_recall

## global
efs = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000]
efs_big = [6000, 7000, 8000, 9000 , 10000, 20000]

parser = ArgumentParser(description="Example script: "
                        "\n\tpython benchs/bench_ls_hnsw.py --datadir /data1/data --dataset sift1M --m 5 --use_ls")

## Required parameters
parser.add_argument("--datadir", type=str, required=True, help="The root location of your dataset folder")
parser.add_argument("--dataset", type=str, required=True, help="Dataset name (Folder name must match file prefix)")

## Optional parameters
parser.add_argument("--m", type=int, default=5, help="Neighborhood size for building the graph (default: 5)")
parser.add_argument("--efc", type=int, default=300, help="EfConstruction for building the graph (default: 300)")
parser.add_argument("--query_k", type=int, default=10, help="Number of results to query, caculate recall_k@1 (default: 10)")
parser.add_argument("--slow_test", action='store_true', help="Run slow test with big efSearch (default: False)")
parser.add_argument("--use_ls", action='store_true', help="Use local scaling distance for build (default: False)")
parser.add_argument("--ls_k", type=int, default=10, help="Number of neighbor for neighbor radius (default: 10)")
parser.add_argument("--ls_sample_num", type=int, default=None, help="Number of samples for neighbor radius (default: num_base * 0.01)")
parser.add_argument("--ls_alpha", type=float, default=1.0, help="Smoothness parameter for local scaling (default: 1.0)")

## generate para
args = parser.parse_args()
hnsw_stats = faiss.cvar.hnsw_stats
if args.slow_test:
    efs.extend(efs_big)

## load dataset
xb, xq, xt, gt = load_data(args.datadir, args.dataset)
nq, d = xq.shape
nb = xb.shape[0]
print("nq: [%d], d: [%d]" % (nq, d))

## set default para
args.ls_sample_num = nb // 100

print(f"para: datadir[{args.datadir}],dataset[{args.dataset}],m[{args.m}],efc[{args.efc}]")
if (args.use_ls):
    print(f"\tls_para: use_ls[{args.use_ls}],ls_k[{args.ls_k}],ls_sample_num[{args.ls_sample_num}],ls_alpha[{args.ls_alpha}]")

def generate_neighbor_radius():
    print(f"clac neighbor radius by FlatL2")
    index = faiss.IndexFlatL2(d)
    rnd_rows = np.random.choice(xb.shape[0], args.ls_sample_num, replace=False)
    index.add(xb[rnd_rows])
    t0 = time.time()
    D, I = index.search(xb, args.ls_k)
    t1 = time.time()
    print(f"FlatL2 get knn use time {t1 - t0}s")
    return np.mean(D ** 0.5, axis=1)

## set build para
index = faiss.IndexHNSWFlat(d, args.m)
index.verbose = True
index.hnsw.efConstruction = args.efc
if (args.use_ls):
    index.set_local_scaling(args.use_ls)
    nbr_radius = generate_neighbor_radius()
    index.set_neighbor_radius(faiss.swig_ptr(nbr_radius))
    index.set_ls_alpha(args.ls_alpha)

## build index
t0 = time.time()
index.add(xb)
t1 = time.time()
print("Graph construction time(s)", t1 - t0)

## search hnsw index
index.set_local_scaling(False)
print(f"efsearch;QPS;Recall@1;avg-dist")
for efSearch in efs:
    index.hnsw.efSearch = efSearch
    hnsw_stats.reset()
    faiss.omp_set_num_threads(16)
    t0 = time.time()
    D, I = index.search(xq, args.query_k)
    t1 = time.time()
    qps = round(nq/(t1-t0), 2)
    recall_at_1 = matrix_recall(I, gt, 1)
    avg_dist = hnsw_stats.ndis // nq
    print("%d;%.2f;%.4f;%d" % (efSearch, qps, recall_at_1, avg_dist))
    if recall_at_1 == 1.0:
        break
