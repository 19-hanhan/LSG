#!/bin/bash
# 编译
make -C build -j faiss && make -C build -j swigfaiss && cd build/faiss/python && python setup.py develop && cd ../../.. 
# 执行 Python 脚本
# python benchs/sift_hubs.py 10 5 0.01 0.02 4 2 nsg


