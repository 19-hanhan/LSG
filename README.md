# LSG

**Paper: Boosting Accuracy and Efficiency for Vector Retrieval with  Local Scaling Graph [ICDE 2025]**

Citations to our paper are highly encouraged if you find LSG is useful and/or use it in your systems. 

> Hongya Wang, WenLong Wu, Cong Luo, Aobei Bian, Chunguang Meng, Yishuo Wu, Ji Sun. Boosting Accuracy and Efficiency for Vector Retrieval with Local Scaling Graph.  ICDE 2025

This project code is based on version 1.5.0 of [facebook faiss](https://github.com/facebookresearch/faiss.git).

## Installation Environment

### Python Environment

You can install conda environment via Anaconda or Miniforge. If you don't have the above environment, you can use the following command:

```bash
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh
```

### System environment

The environment used by the developer team is Ubuntu>=18.04. Install the following dependencies:

```bash
sudo apt install make g++ swig libopenblas-dev -y
```

The version of g++ needs to be set to `g++-11`.

### Build Project

The `--with-python` parameter in the precompilation process needs to be filled in with the Python version in your current environment, such as `--with-python=python[version]`.

```bash
bash ./configure --without-cuda --with-python=python3.12
make -j && make py -j
```

## Running the LSG Scheme

### Preparing the Project Data Files

You can download relevant datasets online, such as `sift1M` or `glove1M`.

The experiment uses datasets in `fvecs` and `bvecs` file formats, and the folder is named `{dataset}`, and the file is named `{dataset}_base.fvecs`, etc. As follows:

```bash
sift1M
 | sift1M_base.fvecs
 | sift1M_query.fvecs
 | sift1M_groundtruth.ivecs
glove1M
 | glove1M_base.fvecs
 | glove1M_query.fvecs
 | glove1M_groundtruth.ivecs
```

### Parameters of LSG

In the `benchs/bench_ls_hnsw.py` file, there are several parameters that can be used to adjust the operation of the script. You can find them at the beginning of the file.

You can view the help message by running: `python benchs/bench_ls_hnsw.py -h`

**Required parameters:**

- `datadir`: The root location of your dataset folder

- `dataset`: Dataset name (Folder name must match file prefix)

**Adjustable parameters:**

- `m`: Neighborhood size for building the graph (default: 5)

- `efc`: EfConstruction for building the graph (default: 300)

- `query_k`: Number of results to query, caculate recall_k@1 (default: 10)

- `use_ls`: Use local scaling distance for build (default: False)

- `slow_test`: Run slow test with big efSearch (default: False)

- `ls_k`: Number of neighbor for neighbor radius (default: 10)

- `ls_nsample_num`: Number of samples for neighbor radius (default: num_base * 0.01)

- `ls_alpha`: Smoothness parameter for local scaling (default: 1.0)

### Running LSG Script

The test code for the basic LSG scheme is written in `benchs/bench_ls_hnsw.py`. You can use it by following command:

```bash
# if you want test hnsw
python benchs/bench_ls_hnsw.py --datadir /data1/data --dataset sift1M

# if you want test lsg
python benchs/bench_ls_hnsw.py --datadir /data1/data --dataset sift1M --use_ls

# if you want higher recall
python benchs/bench_ls_hnsw.py --datadir /data1/data --dataset sift1M --use_ls --slow_test
```

## Authors

The main authors of Local Scaling Method are:

- [HongYa Wang]() (hywang@dhu.edu.cn) proposed and analyzed the formula for Local Scaling.

- [WenLong Wu](https://github.com/19-hanhan) (hanhan_708@163.com) initiated the project and wrote its implementation.