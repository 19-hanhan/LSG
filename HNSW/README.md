# HNSW

This project is based on Ubuntu 18.04.

## Project Build:

Install the following dependencies:

```bash
# Install Anaconda; a Python environment is required
sudo apt install make g++ swig libopenblas-dev
```

### Build
```bash
./configure --without-cuda --with-python=python3.8
make && make py
```

## Running the LSG Scheme

### Preparing the Project Data Files

The `benchs/datasets.py` file defines the data loading functions for each dataset. You can modify these functions to adjust the dataset reading scheme. The experiment uses datasets in `fvecs` and `bvecs` file formats.

### Running LSG (HNSW)

The test code for the basic LSG scheme is written in `benchs/bench_hnswflat.py`. You can use it by passing parameters as follows:

```bash
python -u $faiss/bench_hnswflat.py sift1M 5 300 10 1
```

The five parameters correspond to: dataset name, number of neighbors M, efc, size of the LSG knn, and sampling ratio sr.

### Running LSG + PQ

The test code for the LSG + PQ scheme is written in `benchs/bench_hnswpq.py`. You can use it by passing parameters as follows:

```bash
python -u $faiss/bench_hnswpq.py sift1M 5 300 16 0.01
```

The five parameters correspond to: dataset name, number of neighbors M, efc, size of the LSG knn, and sampling ratio sr.