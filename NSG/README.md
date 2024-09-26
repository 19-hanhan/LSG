# LSG IN NSG
The experimental code is based on Faiss 1.7.1, and the relevant configuration can refer to Faiss.
We modified part of the code to add our LSG method.
# Installing
## Prepare
Prepare Anaconda, Python, make, swig, blas, g++ and other necessary tools.
## Precompile
Enter the faiss-lsg folder.
```
cmake -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=ON -DBUILD_SHARED_LIBS=ON -DBUILD_TESTING=OFF -DBLA_VENDOR=Intel10_64_dyn -DMKL_LIBRARIES=/your anaconda path/anaconda3/lib/ -B build .
```
## Make
```
make -C build -j faiss && make -C build -j swigfaiss && cd build/faiss/python && python setup.py develop && cd ../../..
```
## Example
Our example code is located in /benchs/LSG.py, but before running it, you will need to do some preparation.
### Datasets
Please prepare the datasets you need , then modify the /benchs/datasets.py file for successful use.
### Environment variables
You need to confirm your environment variables. If there is an error, you can use the following command
```
export PYTHONPATH=/path to/faiss-lsg/build/python:$PYTHONPATH
```
```
export LD_LIBRARY_PATH=/path to/faiss-lsg/build/faiss:$PYTHONPATH
```
# Other
You can search for "nsg-lsg" to review some key modifications, which demonstrate how our LSG method is easily embedded into the existing graph-based approach.
# Bug
please use "import faiss" before "import pandas"