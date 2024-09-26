#!/bin/bash
# 编译
make -C build -j faiss && make -C build -j swigfaiss && cd build/faiss/python && python setup.py develop && cd ../../.. 
# 执行 Python 脚本
# python benchs/datasets_time.py

# python benchs/datasets_hubs.py 10 10 0.01 0.02 4 2 sift1M nsg
# python benchs/datasets_hubs.py 10 32 0.01 0.02 4 2 sift1M nsg

# efsearch小精度
# python benchs/datasets_hubs.py 10 10 0.01 0.02 4 2 msong nsg
# python benchs/datasets_hubs.py 10 32 0.01 0.02 4 2 msong nsg

# python benchs/datasets_hubs.py 10 10 0.01 0.02 4 2 gist nsg
# python benchs/datasets_hubs.py 10 32 0.01 0.02 4 2 gist nsg

# python benchs/datasets_hubs.py 10 10 0.01 0.02 4 2 enron nsg
# python benchs/datasets_hubs.py 10 32 0.01 0.02 4 2 enron nsg

# python benchs/datasets_hubs.py 10 10 0.01 0.02 4 2 imageNet nsg
# python benchs/datasets_hubs.py 10 32 0.01 0.02 4 2 imageNet nsg

# 数据集较大
# python benchs/datasets_hubs.py 10 10 0.01 0.02 4 2 tiny5M nsg
# python benchs/datasets_hubs.py 10 32 0.01 0.02 4 2 tiny5M nsg

# python benchs/datasets_hubs.py 10 10 0.01 0.02 4 2 glove2M nsg
# python benchs/datasets_hubs.py 10 32 0.01 0.02 4 2 glove2M nsg

# python benchs/datasets_hubs.py 10 10 0.01 0.02 4 2 trevi nsg
# python benchs/datasets_hubs.py 10 32 0.01 0.02 4 2 trevi nsg

# python benchs/datasets_hubs.py 10 10 0.01 0.02 4 2 word2vec nsg
# python benchs/datasets_hubs.py 10 32 0.01 0.02 4 2 word2vec nsg

# 数据集10M，提高采样数
python benchs/datasets_hubs.py 10 10 0.01 0.02 4 2 spacev nsg
python benchs/datasets_hubs.py 10 32 0.01 0.02 4 2 spacev nsg


