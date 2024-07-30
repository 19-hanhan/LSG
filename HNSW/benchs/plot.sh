export PATH=/home/CPC/hanhan/anaconda3/bin:$PATH
faiss=/home/CPC/hanhan/faiss-1.5.0/benchs

python -u $faiss/plot.py trevi 0.1
# python -u $faiss/plot.py sift1M 0.01
# python -u $faiss/plot.py glove1M 0.01
# python -u $faiss/plot.py imageNet 0.01
# python -u $faiss/plot.py word2vec 0.01
# python -u $faiss/plot.py random_gaussian 0.01