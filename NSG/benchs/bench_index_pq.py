# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import print_function
import faiss
from datasets import load_sift1M, evaluate
import time

xb, xq, xt, gt = load_sift1M()
#nq:向量个数
#d:维数
nq, d = xq.shape

#k为查找候选队列，要求大于@N
k = 128

#nbits:子空间向量编码位数（2^nbits=子空间质心数）
for nbits in 4, 6, 8:
    #int子空间数
    index = faiss.IndexPQ(d, 8, nbits)
    #xt训练集
    tt0=time.time()
    index.train(xt)
    tt1=time.time()
    print("\t %7.3f ms train time" % ((tt1 - tt0) * 1000.0 ))

    #x被查找集
    ta0=time.time()
    index.add(xb)
    ta1=time.time()
    print("\t %7.3f ms train time" % ((ta1 - ta0) * 1000.0 ))

    #xq:测试集
    #gt:对照集（答案）
    #k：查找队列（堆结构）
    t, r = evaluate(index, xq, gt, k)
    #print("\t %7.3f ms per query, R@1 %.4f" % (t, r[1]))
    print("\t %7.3f ms per query, R@1 %.4f,R@10 %.4f,R@100 %.4f" % (t, r[1],r[10],r[100]))
    del index
