# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import print_function
import sys
import time
import numpy as np


def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()

def ivecs_mmap(fname):
    a = np.memmap(fname, dtype='int32', mode='r')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:]

def fvecs_mmap(fname):
    return ivecs_mmap(fname).view('float32')

def bvecs_mmap(fname):
    x = np.memmap(fname, dtype='uint8', mode='r')
    d = x[:4].view('int32')[0]
    return x.reshape(-1, d + 4)[:, 4:]
    
def fvecs_read(fname):
    return ivecs_read(fname).view('float32')

def mmap_fvecs(fname):
    x = np.memmap(fname, dtype='int32', mode='r')
    d = x[0]
    return x.view('float32').reshape(-1, d + 1)[:, 1:]

def mmap_bvecs(fname):
    x = np.memmap(fname, dtype='uint8', mode='r')
    d = x[:4].view('int32')[0]
    return x.reshape(-1, d + 4)[:, 4:]

def sanitize(x):
    """ convert array to a c-contiguous float array """
    return np.ascontiguousarray(x.astype('float32'))

def read_i8bin(filename, start_idx=0, chunk_size=None):
    with open(filename, "rb") as f:
        nvecs, dim = np.fromfile(f, count=2, dtype=np.int32)
        nvecs = (nvecs - start_idx) if chunk_size is None else chunk_size
        arr = np.fromfile(f, count=nvecs * dim, dtype=np.int8, offset=start_idx * 4 * dim)
    return arr.reshape(nvecs, dim)

def read_u8bin(filename, start_idx=0, chunk_size=None):
    """ Read *.ibin file that contains int32 vectors
    Args:
        :param filename (str): path to *.ibin file
        :param start_idx (int): start reading vectors from this index
        :param chunk_size (int): number of vectors to read.
                                 If None, read all vectors
    Returns:
        Array of int32 vectors (numpy.ndarray)
    """
    with open(filename, "rb") as f:
        nvecs, dim = np.fromfile(f, count=2, dtype=np.int32)
        nvecs = (nvecs - start_idx) if chunk_size is None else chunk_size
        # f.seek(4+4)
        arr = np.fromfile(f, count=nvecs * dim, dtype=np.uint8,
                          offset=start_idx * 4 * dim)
    return arr.reshape(nvecs, dim).astype(np.float32)

def knn_result_read(fname):
    with open(fname, "rb") as f:
        nq, k = np.fromfile(f, count=2, dtype=np.uint32)
        # nvecs = (nvecs - start_idx) if chunk_size is None else chunk_size
        # f.seek(4+4)
        arr = np.fromfile(f, count=nq * k, dtype=np.uint32,
                          offset=0)
    return arr.reshape(nq, k)
def range_result_read(fname):
    """ read the range search result file format """
    f = open(fname, "rb")
    nq, total_res = np.fromfile(f, count=2, dtype="int32")
    nres = np.fromfile(f, count=nq, dtype="int32")
    assert nres.sum() == total_res
    I = np.fromfile(f, count=total_res, dtype="int32")
    D = np.fromfile(f, count=total_res, dtype="float32")
    return  I.reshape(nq, 1)

simdir = '/path/'

def load_SimSearchNet(s=1):
    print("Loading SimSearchNet...", end='', file=sys.stderr)
    basedir = '/data/wanghongya/SimSearchNet/'
    # xt = fvecs_read(basedir + "query.learn.50M.fbin")
    # print(xt.shape)
    million=1000000
    xb = read_u8bin(basedir + "FB_ssnpp_database.u8bin",0,s*million)
    xq = read_u8bin(basedir + "FB_ssnpp_public_queries.u8bin",0,1000)
    gt = ivecs_read(basedir+'SimSearchNet%dm_groundtruth.ivecs'%s)[:1000,:]
    print("done", file=sys.stderr)
    return xb, xq,gt
    # return xb, xq


def load_text_to_image():
    print("Loading text_to_image...", end='', file=sys.stderr)
    basedir = '/path/dataset/Text-to-Image/'
    xt = fvecs_read(basedir + "query.learn.50M.fbin")
    print(xt.shape)
    xb = fvecs_read(basedir + "ukbench_base.fvecs")
    xq = fvecs_read(basedir + "ukbench_query.fvecs")
    gt = ivecs_read(basedir + "ukbench_groundtruth.ivecs")
    print("done", file=sys.stderr)
    return xb, xq, xt, gt

def load_spacev():
    print("Loading spacev...", end='', file=sys.stderr)
    basedir = '/path/dataset/spacev/'
    xb = mmap_fvecs(basedir+'spacev10m_base.fvecs')
    xq = mmap_fvecs(basedir + 'spacev10m_query.fvecs')
    xb = sanitize(xb[:])
    xq = sanitize(xq[:])
    gt = ivecs_read(basedir + 'spacev10m_groundtruth.ivecs')
    return xb, xq, gt

def load_sift1M():
    print("Loading sift1M...", end='', file=sys.stderr)
    basedir = '/path/sift1M/'
    # basedir = '/mnt/d/github/sift1M/'
    xt = fvecs_read(basedir + "sift_learn.fvecs")
    xb = fvecs_read(basedir + "sift_base.fvecs")
    xq = fvecs_read(basedir + "sift_query.fvecs")
    gt = ivecs_read(basedir + "sift_groundtruth.ivecs")
    print("done", file=sys.stderr)

    return xb, xq, xt, gt
def load_deep():
    # simdir = '/path/'
    print("load deep.....")
    # basedir = simdir + 'deep1b/'
    basedir = '/path/deep1b/'
    
    dbsize = 10
    xb = mmap_fvecs('/data/wanghongya/data_disk/deep1B_base.fvecs')
    xq = mmap_fvecs(basedir + 'deep1B_queries.fvecs')
    # xt = mmap_fvecs(basedir + 'deep1B_learn.fvecs')
    # trim xb to correct size
    xb = sanitize(xb[:dbsize * 1000 * 1000])
    # xt = sanitize(xt[:500000])
    xq = sanitize(xq[:10000])
    gt = ivecs_read(basedir + 'deep%dM1_groundtruth.ivecs' % dbsize)
  
    return xb, xq, gt
def load_deep1m():
    # simdir = '/path/'
    print("load deep1m.....")
    # basedir = simdir + 'deep1b/'
    basedir = '/path/deep1b/'
    
    dbsize = 1
    xb = mmap_fvecs(basedir+'100w.fvecs')
    xq = mmap_fvecs(basedir + 'deep1B_queries.fvecs')
    xt = mmap_fvecs(basedir + 'deep1B_learn.fvecs')
    # trim xb to correct size
    xb = sanitize(xb[:])
    # xt = sanitize(xt[:500000])
    xq = sanitize(xq[:])
    gt = ivecs_read(basedir + 'deep%dM_groundtruth.ivecs' % dbsize)
  
    return xb, xq, xt, gt
def load_sift10K():
    print("Loading sift10K...", end='', file=sys.stderr)
    # basedir = simdir + '1M/sift10K/'
    basedir = '/path/dataset/sift10K/'
    xt = fvecs_read(basedir + "siftsmall_learn.fvecs")
    xb = fvecs_read(basedir + "siftsmall_base.fvecs")
    xq = fvecs_read(basedir + "siftsmall_query.fvecs")
    gt = ivecs_read(basedir + "siftsmall_groundtruth.ivecs")
    print("done", file=sys.stderr)

    return xb, xq, xt, gt
def load_msong():
    print("Loading msong...", end='', file=sys.stderr)
    # basedir = simdir + '1M/sift10K/'
    basedir = '/path/dataset/millionSong/'
    # xt = fvecs_read(basedir + "siftsmall_learn.fvecs")
    xb = fvecs_read(basedir + "millionSong_base.fvecs")
    xq = fvecs_read(basedir + "millionSong_query.fvecs")
    gt = ivecs_read(basedir + "millionSong_groundtruth.ivecs")
    print("done", file=sys.stderr)

    return xb, xq, gt
def load_bigann():
    # print("Loading bigann...", end='', file=sys.stderr)
    # basedir = '/mnt/d/github/sift1B/'
    basedir = simdir + 'sift1B/'

    dbsize = 100
    xb = bvecs_mmap(basedir + '1milliard.p1.siftbin')
    xq = bvecs_mmap(basedir + 'queries.bvecs')
    xt = bvecs_mmap(basedir + 'learn.bvecs')
    # trim xb to correct size
    xb = sanitize(xb[:dbsize * 1000 * 1000])
    xt = sanitize(xt[:250000])
    xq = sanitize(xq)
    gt = ivecs_read(basedir + 'gnd/idx_%dM.ivecs' % dbsize)

    return xb, xq, xt, gt
def load_tiny5m():
    print("load tiny5m")
    basedir='/data/wanghongya/tiny5m/'
    xb = mmap_fvecs(basedir + 'tiny5m_base.fvecs')
    xq = mmap_fvecs(basedir + 'tiny5m_query.fvecs')
    xt = mmap_fvecs(basedir + 'tiny5m_query.fvecs')
    xb = sanitize(xb[:])
    xq = sanitize(xq[:])
    gt = ivecs_read(basedir + "tiny5m_groundtruth.ivecs")
    return xb, xq, xt, gt

# def load_deep10m():
#     # simdir = '/path/'
#     print("load deep10m.....")
#     # basedir = simdir + 'deep1b/'
#     basedir = '/path/deep1b/'
    
#     dbsize = 10
#     xb = mmap_fvecs(basedir + 'deep10M.fvecs')
#     xq = mmap_fvecs(basedir + 'deep1B_queries.fvecs')
#     xt = mmap_fvecs(basedir + 'deep1M/deep_learn500k.fvecs')
#     # trim xb to correct size
#     xb = sanitize(xb[:])
#     xt = sanitize(xt[:])
#     xq = sanitize(xq[:])
#     gt = ivecs_read(basedir + 'deep%dM_groundtruth.ivecs' % dbsize)
#     return xb, xq, xt, gt

def load_deep():
    # simdir = '/path/'
    print("load deep.....")
    # basedir = simdir + 'deep1b/'
    basedir = '/data/wanghongya/Deep/'
    dbsize = 10
    xb = mmap_fvecs(basedir + 'deep1B_base.fvecs')
    xq = mmap_fvecs(basedir + 'deep1B_queries.fvecs')
    xt = mmap_fvecs(basedir + 'deep1B_learn.fvecs')
    # trim xb to correct size
    xb = sanitize(xb[:dbsize * 1000 * 1000])
    xt = sanitize(xt[:500000])
    xq = sanitize(xq[:10000])
    gt = ivecs_read(basedir + 'deep%dM_groundtruth.ivecs' % dbsize)
    return xb, xq, xt, gt

# ukbench 1M
def load_ukbench():
    # simdir = '/path/'
    print("load ukbench.....")
    # basedir = simdir + 'deep1b/'
    basedir = '/path/dataset/ukbench/'
    
    dbsize = 1
    xb = mmap_fvecs(basedir + 'ukbench_base.fvecs')
    xq = mmap_fvecs(basedir + 'ukbench_query.fvecs')
    # trim xb to correct size
    xb = sanitize(xb[:])
    # # xt = sanitize(xt[:100000])
    xq = sanitize(xq[:])
    gt = ivecs_read(basedir + 'ukbench_groundtruth.ivecs')
    return xb, xq, gt
# trevi 1M
def load_trevi():
    # simdir = '/path/'
    print("load trevi.....")
    # basedir = simdir + 'deep1b/'
    basedir = '/path/dataset/trevi/'
    
    dbsize = 1
    xb = mmap_fvecs(basedir + 'trevi_base.fvecs')
    xq = mmap_fvecs(basedir + 'trevi_query.fvecs')
    # trim xb to correct size
    xb = sanitize(xb[:])
    # xt = sanitize(xt[:100000])
    xq = sanitize(xq[:])
    gt = ivecs_read(basedir + 'trevi_groundtruth.ivecs')

    return xb, xq, gt

# enron 
def load_enron():
    print("load enron.....")
    basedir = '/path/dataset/enron/'
    
    dbsize = 1
    xb = mmap_fvecs(basedir + 'enron_base.fvecs')
    xq = mmap_fvecs(basedir + 'enron_query.fvecs')
    # trim xb to correct size
    xb = sanitize(xb[:])
    # xt = sanitize(xt[:100000])
    xq = sanitize(xq[:])
    gt = ivecs_read(basedir + 'enron_groundtruth.ivecs')

    return xb, xq, gt

# crawl 
def load_crawl():
    print("load crawl.....")
    basedir = '/path/dataset/crawl/'
    
    dbsize = 1
    xb = mmap_fvecs(basedir + 'crawl_base.fvecs')
    xq = mmap_fvecs(basedir + 'crawl_query.fvecs')
    xb = sanitize(xb[:])
    xq = sanitize(xq[:])
    gt = ivecs_read(basedir + 'crawl_groundtruth.ivecs')

    return xb, xq, gt

def load_notre():
    # simdir = '/path/'
    print("load notre.....")
    # basedir = simdir + 'deep1b/'
    basedir = '/path/dataset/notre/'
    
    dbsize = 1
    xb = mmap_fvecs(basedir + 'notre_base.fvecs')
    xq = mmap_fvecs(basedir + 'notre_query.fvecs')
    # xt = mmap_fvecs(basedir + 'gist_learn.fvecs')
    # trim xb to correct size
    xb = sanitize(xb[:])
    # xt = sanitize(xt[:100000])
    xq = sanitize(xq[:])
    gt = ivecs_read(basedir + 'notre_groundtruth.ivecs')

    return xb, xq, gt
def load_mnist():
    # simdir = '/path/'
    print("load mnist.....")
    # basedir = simdir + 'deep1b/'
    basedir = '/path/dataset/MNIST/'
    
    dbsize = 1
    xb = mmap_fvecs(basedir + 'MNIST_base.fvecs')
    xq = mmap_fvecs(basedir + 'MNIST_query.fvecs')
    # xt = mmap_fvecs(basedir + 'gist_learn.fvecs')
    # trim xb to correct size
    xb = sanitize(xb[:])
    # xt = sanitize(xt[:100000])
    xq = sanitize(xq[:])
    gt = ivecs_read(basedir + 'MNIST_groundtruth.ivecs')

    return xb, xq, gt
# nuswide 1M
def load_nuswide():
    # simdir = '/path/'
    print("load nuswide.....")
    # basedir = simdir + 'deep1b/'
    basedir = '/path/dataset/nuswide/'
    
    dbsize = 1
    xb = mmap_fvecs(basedir + 'nuswide_base.fvecs')
    xq = mmap_fvecs(basedir + 'nuswide_query.fvecs')
    # xt = mmap_fvecs(basedir + 'gist_learn.fvecs')
    # trim xb to correct size
    xb = sanitize(xb[:])
    # xt = sanitize(xt[:])
    xq = sanitize(xq[:])
    gt = ivecs_read(basedir + 'nuswide_groundtruth.ivecs')

    return xb, xq, gt
# gist 1M
def load_gist():
    # simdir = '/path/'
    print("load gist.....")
    # basedir = simdir + 'deep1b/'
    basedir = '/path/dataset/gist/'
    
    dbsize = 1
    xb = mmap_fvecs(basedir + 'gist_base.fvecs')
    xq = mmap_fvecs(basedir + 'gist_query.fvecs')
    xt = mmap_fvecs(basedir + 'gist_query.fvecs')
    # trim xb to correct size
    xb = sanitize(xb[:dbsize * 1000 * 1000])
    # xt = sanitize(xt[:100000])
    xq = sanitize(xq[:1000])
    gt = ivecs_read(basedir + 'gist_groundtruth.ivecs')

    return xb, xq, xt, gt


# glove 1M
def load_glove():
    # simdir = '/path/'
    # basedir = simdir + 'deep1b/'

    basedir = '/path/dataset/glove/'
    
    dbsize = 1
    xb = mmap_fvecs(basedir + 'glove_base.fvecs')
    xq = mmap_fvecs(basedir + 'glove_query.fvecs')
    xt = mmap_fvecs(basedir + 'glove_query.fvecs')
    # trim xb to correct size
    print(xb.shape[0])
    xb = sanitize(xb[:xb.shape[0]+1])
    xt = sanitize(xt[:1000])
    xq = sanitize(xq[:10000])
    gt = ivecs_read(basedir + 'glove_groundtruth.ivecs')
    print(xq.shape[0])
    return xb, xq, xt, gt

def load_glove2m():
    print("load glove2M.....", end='', file=sys.stderr)
    basedir='/path/dataset/glove2.2m/'
    xb = mmap_fvecs(basedir + 'glove2.2m_base.fvecs')
    xq = mmap_fvecs(basedir + 'glove2.2m_query.fvecs')
    xb = sanitize(xb[:])
    xq = sanitize(xq[:])
    gt = ivecs_read(basedir + "glove2.2m_groundtruth.ivecs")
    print("done", file=sys.stderr)
    return xb,xq,gt
    
def load_audio():
    print("Loading audio...", end='', file=sys.stderr)
    basedir = '/path/dataset/audio/'
    xt = fvecs_read(basedir + "audio_base.fvecs")
    xb = fvecs_read(basedir + "audio_base.fvecs")
    xq = fvecs_read(basedir + "audio_query.fvecs")
    gt = ivecs_read(basedir + "audio_groundtruth.ivecs")
    print("done", file=sys.stderr)
    return xb, xq, xt, gt

def load_sun():
    print("Loading sun...", end='', file=sys.stderr)
    basedir = '/path/dataset/sun/'
    xt = fvecs_read(basedir + "sun_base.fvecs")
    xb = fvecs_read(basedir + "sun_base.fvecs")
    xq = fvecs_read(basedir + "sun_query.fvecs")
    gt = ivecs_read(basedir + "sun_groundtruth.ivecs")
    print("done", file=sys.stderr)
    return xb, xq, xt, gt

def load_random():
    print("Loading random...", end='', file=sys.stderr)
    basedir = '/path/dataset/random/'
    xt = fvecs_read(basedir + "random_base.fvecs")
    xb = fvecs_read(basedir + "random_base.fvecs")
    xq = fvecs_read(basedir + "random_query.fvecs")
    gt = ivecs_read(basedir + "random_groundtruth.ivecs")
    print("done", file=sys.stderr)
    return xb, xq, xt, gt

def load_imageNet():
    print("Loading imageNet...", end='', file=sys.stderr)
    basedir = '/path/dataset/imageNet/'
    xt = fvecs_read(basedir + "imageNet_base.fvecs")
    xb = fvecs_read(basedir + "imageNet_base.fvecs")
    xq = fvecs_read(basedir + "imageNet_query.fvecs")
    gt = ivecs_read(basedir + "imageNet_groundtruth.ivecs")
    print("done", file=sys.stderr)
    return xb, xq, xt, gt

def load_word2vec():
    print("Loading word2vec...", end='', file=sys.stderr)
    basedir = simdir + 'dataset/word2vec/'
    xb = mmap_fvecs(basedir + 'word2vec_base.fvecs')
    xt = np.zeros((0, 0))
    xq = mmap_fvecs(basedir + 'word2vec_query.fvecs')
    xb = sanitize(xb[:])
    xq = sanitize(xq[:])
    gt = ivecs_read(basedir + "word2vec_groundtruth.ivecs")
    print("done", file=sys.stderr)
    return xb, xq, xt, gt

def evaluate(index, xq, gt, k):
    nq = xq.shape[0]
    t0 = time.time()
    D, I = index.search(xq, k)  # noqa: E741
    t1 = time.time()

    recalls = {}
    i = 1
    while i <= k:
        recalls[i] = (I[:, :i] == gt[:, :1]).sum() / float(nq)
        i *= 10

    return (t1 - t0) * 1000.0 / nq, recalls
