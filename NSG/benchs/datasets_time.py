# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import time
import sys
import numpy as np
import faiss

# dataset_name = str(sys.argv[1])
# print(dataset_name)

dataset_name = 'word2vec'

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

if(dataset_name == 'spacev'):
    from datasets import load_spacev
    xb, xq, gt = load_spacev()

if(dataset_name == 'msong'):
    from datasets import load_msong
    xb, xq, gt = load_msong()

nq, d = xq.shape

print(xb.shape)

# ivf计算knn均值(用于NICDM)
def clac_nicdm_avgdis_ivf(ls_k = 10,sr = 1/1000):
    print(f"clac k{ls_k} by ivf")
    index = faiss.IndexFlatL2(d)

    t0 = time.time()
    samples = max(int(sr * xb.shape[0]),10000) # 至少1w个采样点保证效果
    if dataset_name =="spacev":samples = 100000
    rnd_rows = np.random.choice(xb.shape[0], samples, replace=False)

    t1 = time.time()
    print(f">>>sample use time {t1 - t0}s")

    index.add(xb[rnd_rows])
    avgdis = []
    D, I = index.search(xb, ls_k)
    t2 = time.time()
    avgdis.extend(np.mean(D, axis=1))
    print(f">>>ivf get knn use time {t2 - t1}s")
    return np.array(avgdis)

for m in 10,32: # m=R

    time_start = time.time()

    print("Testing NSG Flat")

    index = faiss.IndexNSGFlat(d, m) # 默认使用L2

    # 使用nndescent
    index.build_type = 1

    # to see progress
    index.verbose = True


    # # 计算均值
    # avgdis = clac_nicdm_avgdis_ivf(m)
    # # 设定NICDM参数
    # index.set_nicdm_distance(faiss.swig_ptr(avgdis), 1.0)
    # print(">>>LSG")

    print("add")
    index.add(xb)

    time_build = time.time()

    print('>>>datasets:', dataset_name, 'R=', m, 'time:', time_build-time_start)
    # pd.DataFrame(data).to_csv('/root/faiss-1.7.1/csv/nsg_'+ str(m) +'.csv', index=False) 
    # pd.DataFrame(data).to_csv('/home/wanghongya/wys/faiss171/csv/time/nsg_lsg_'+ str(m) +'.csv', index=False)
# sift1M
# LSG
# 1
# >>>sample use time 3.814697265625e-06s
# >>>ivf get knn use time 47.355977296829224s
# >>>datasets: sift1M R= 10 time: 460.05829334259033
# >>>sample use time 1.2636184692382812e-05s
# >>>ivf get knn use time 52.78076672554016s
# >>>datasets: sift1M R= 32 time: 745.2748568058014
# 2
# >>>sample use time 4.76837158203125e-06s
# >>>ivf get knn use time 46.83645749092102s
# >>>datasets: sift1M R= 10 time: 475.6684639453888
# >>>sample use time 7.867813110351562e-06s
# >>>ivf get knn use time 84.1277973651886s
# >>>datasets: sift1M R= 32 time: 848.6620225906372
# 3
# >>>sample use time 2.2649765014648438e-05s
# >>>ivf get knn use time 45.730366468429565s
# >>>datasets: sift1M R= 10 time: 454.72987699508667
# >>>sample use time 6.9141387939453125e-06s
# >>>ivf get knn use time 54.14461135864258s
# >>>datasets: sift1M R= 32 time: 739.4379889965057
# NSG
# 1
# >>>datasets: sift1M R= 10 time: 583.4098088741302
# >>>datasets: sift1M R= 32 time: 800.9857409000397
# 2
# >>>datasets: sift1M R= 10 time: 569.6864361763
# >>>datasets: sift1M R= 32 time: 792.3938703536987
# GK = 2R(nndescent_S=10)
# LSG
# 1
# >>>datasets: sift1M R= 10 time: 580.2421591281891
# >>>datasets: sift1M R= 32 time: 951.889622926712
# 2
# >>>datasets: sift1M R= 10 time: 584.1076486110687
# >>>datasets: sift1M R= 32 time: 955.8835260868073
# NSG
# >>>datasets: sift1M R= 10 time: 671.178421497345
# >>>datasets: sift1M R= 32 time: 974.1870095729828
# nndescent_S = 2R
# LSG
# 1
# >>>datasets: sift1M R= 10 time: 785.5176010131836
# >>>datasets: sift1M R= 32 time: 2711.8387196063995
# 2
# >>>datasets: sift1M R= 10 time: 788.6003260612488
# >>>datasets: sift1M R= 32 time: 2761.3757066726685
# NSG
# >>>datasets: sift1M R= 10 time: 880.2004153728485
# >>>datasets: sift1M R= 32 time: 2800.8524544239044
# nndescent_L = GK + 200
# NSG
# >>>datasets: sift1M R= 10 time: 1058.7882266044617
# >>>datasets: sift1M R= 32 time: 1102.4739861488342
# LSG
# >>>sample use time 0.01546478271484375s
# >>>ivf get knn use time 45.28146409988403s
# >>>datasets: sift1M R= 10 time: 950.3627142906189
# >>>sample use time 0.03762626647949219s
# >>>ivf get knn use time 53.79279112815857s
# >>>datasets: sift1M R= 32 time: 979.9510254859924
# L = R + 100
# LSG
# >>>sample use time 0.0199277400970459s
# >>>ivf get knn use time 47.769267559051514s
# >>>datasets: sift1M R= 10 time: 542.1526291370392
# >>>sample use time 0.040822505950927734s
# >>>ivf get knn use time 55.36714482307434s
# >>>datasets: sift1M R= 32 time: 922.85791015625
# NSG
# 1
# >>>datasets: sift1M R= 10 time: 647.8336207866669
# >>>datasets: sift1M R= 32 time: 916.7767090797424
# 2
# >>>datasets: sift1M R= 10 time: 636.0893685817719
# >>>datasets: sift1M R= 32 time: 967.9963207244873
# gist
# LSG
# >>>sample use time 3.5762786865234375e-06s
# >>>ivf get knn use time 111.72757863998413s
# >>>datasets: gist R= 10 time: 2208.886072397232
# 内存不足
# trevi
# LSG
# 1
# >>>sample use time 3.337860107421875e-06s
# >>>ivf get knn use time 36.28860688209534s
# >>>datasets: trevi R= 10 time: 982.5050137042999
# >>>sample use time 9.059906005859375e-06s
# >>>ivf get knn use time 42.04379940032959s
# >>>datasets: trevi R= 32 time: 1313.178823709488
# 2
# >>>sample use time 3.814697265625e-06s
# >>>ivf get knn use time 35.73884034156799s
# >>>datasets: trevi R= 10 time: 983.4604392051697
# >>>sample use time 8.344650268554688e-06s
# >>>ivf get knn use time 40.82480454444885s
# >>>datasets: trevi R= 32 time: 1329.102661371231
# NSG
# 1
# >>>datasets: trevi R= 10 time: 1652.9915897846222
# >>>datasets: trevi R= 32 time: 1942.7147517204285
# 2
# >>>datasets: trevi R= 10 time: 1626.9617307186127
# >>>datasets: trevi R= 32 time: 1889.7293176651
# word2vec
# LSG
# 1
# >>>sample use time 3.814697265625e-06s
# >>>ivf get knn use time 81.9984700679779s
# >>>datasets: word2vec R= 10 time: 967.5311872959137
# >>>sample use time 8.344650268554688e-06s
# >>>ivf get knn use time 69.03756785392761s
# >>>datasets: word2vec R= 32 time: 1317.5703806877136
# 2
# >>>sample use time 3.337860107421875e-06s
# >>>ivf get knn use time 58.72439622879028s
# >>>datasets: word2vec R= 10 time: 910.8050994873047
# >>>sample use time 7.62939453125e-06s
# >>>ivf get knn use time 70.75710129737854s
# >>>datasets: word2vec R= 32 time: 1281.5098812580109
# 3
# >>>sample use time 3.0994415283203125e-06s
# >>>ivf get knn use time 57.435614585876465s
# >>>datasets: word2vec R= 10 time: 911.1217129230499
# >>>sample use time 7.62939453125e-06s
# >>>ivf get knn use time 69.89625144004822s
# >>>datasets: word2vec R= 32 time: 1275.6563901901245
# NSG
# 1
# >>>datasets: word2vec R= 10 time: 1794.0220046043396
# >>>datasets: word2vec R= 32 time: 1597.0613996982574
# 2
# >>>datasets: word2vec R= 10 time: 1799.4555945396423
# >>>datasets: word2vec R= 32 time: 1609.4862978458405
# enron
# LSG 
# 1
# >>>sample use time 3.814697265625e-06s
# >>>ivf get knn use time 13.216959238052368s
# >>>datasets: enron R= 10 time: 328.9756760597229
# >>>sample use time 8.58306884765625e-06s
# >>>ivf get knn use time 16.188858032226562s
# >>>datasets: enron R= 32 time: 439.02552604675293
# 2
# >>>sample use time 3.814697265625e-06s
# >>>ivf get knn use time 13.405209064483643s
# >>>datasets: enron R= 10 time: 374.01030349731445
# >>>sample use time 8.821487426757812e-06s
# >>>ivf get knn use time 16.015538454055786s
# >>>datasets: enron R= 32 time: 440.58397245407104
# NSG
# 1
# >>>datasets: enron R= 10 time: 460.14540791511536
# >>>datasets: enron R= 32 time: 580.4002816677094
# 2
# >>>datasets: enron R= 10 time: 463.27882194519043
# >>>datasets: enron R= 32 time: 582.6949827671051
# msong
# NSG
# 1
# >>>datasets: msong R= 10 time: 1488.0255918502808
# >>>datasets: msong R= 32 time: 1765.3913445472717
# 2
# >>>datasets: msong R= 10 time: 1485.345368385315
# >>>datasets: msong R= 32 time: 1803.749588251114
# LSG
# 1
# >>>sample use time 5.9604644775390625e-06s
# >>>ivf get knn use time 75.08318853378296s
# >>>datasets: msong R= 10 time: 968.4411051273346
# >>>sample use time 1.1682510375976562e-05s
# >>>ivf get knn use time 84.9713339805603s
# >>>datasets: msong R= 32 time: 1432.8558015823364
# 2
# >>>sample use time 4.291534423828125e-06s
# >>>ivf get knn use time 76.06951880455017s
# >>>datasets: msong R= 10 time: 970.9237289428711
# >>>sample use time 9.298324584960938e-06s
# >>>ivf get knn use time 82.42426323890686s
# >>>datasets: msong R= 32 time: 1431.488051891327