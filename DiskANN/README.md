# DiskANN

This project is based on Ubuntu 18.04.

## Project Build:

Install the following dependencies:

```bash
sudo apt install make cmake g++ libaio-dev libgoogle-perftools-dev clang-format libboost-all-dev libmkl-full-dev
```

### Build
```bash
mkdir build && cd build && cmake -DCMAKE_BUILD_TYPE=Release .. && make -j 
```

## Running the LSG Scheme

For DiskANN-specific parameters, refer to the documentation in `workflows/in_memory_index.md` and `workflows/SSD_index.md`.

### Preparing Project Data Files

If the experimental data is in `fvecs` format, DiskANN requires it to be converted to a custom `fbin` format. You can use the built-in methods for conversion.

To validate the accuracy of the queries, you also need to compute the ground truth (gt) of the queries. The built-in method `compute_groundtruth` uses concurrent linear search to obtain the results.

Here is the specific code, where the variables can be replaced with custom data files and output paths:

```bash
./apps/utils/fvecs_to_bin float $data_file $data_path
./apps/utils/fvecs_to_bin float $query_file $query_path
./apps/utils/compute_groundtruth  --data_type float --dist_fn $dis_method --base_file $data_path --query_file $query_path --gt_file $gt_path --K $K
```

The query file usage is the same as in DiskANN. To run the LSG method, we also need a sample dataset file and the knn file for the sample (equivalent to the sample's gt). Here is the specific code, where `sr_file` can be sampled from the `fvecs` dataset file:

```bash
./apps/utils/fvecs_to_bin float $sr_file $sr_path
./apps/utils/compute_groundtruth  --data_type float --dist_fn $dis_method --base_file $sr_path --query_file $data_path --gt_file $sr_knn_path --K $K
```

This setup can be executed via the `opt/prepare.sh` script, which corresponds to the above code for reference.

### Running LSG + Vamana In-Memory Graph Construction

We have integrated the LSG scheme into DiskANN's in-memory graph construction method. The method is invoked the same way as in DiskANN, with two additional parameters, `$sr_knn_path` and `$ls_alpha`, representing the sample dataset knn file path and an adjustable LSG parameter (usually set to 1.0).

```bash
./apps/build_memory_index  --data_type float --dist_fn $build_dis_method --knn_path $sr_knn_path --data_path $data_path --index_path_prefix $index_path_prefix -T $T -R $R -L $L -a $ls_alpha --alpha 1.2

./apps/search_memory_index  --data_type float --dist_fn $search_dis_method --index_path_prefix $index_path_prefix --query_file $query_path  --gt_file $gt_path --result_path $res_path_prefix -K $recall_K -T $T -L 10 20 30 40 50 60 80 100
```

This setup can be executed via the `opt/memory.sh` script, which corresponds to the above code for reference.

### Running LSG + Vamana Sharded Graph Construction

Similar to the in-memory method, the sharded graph construction is also based on DiskANN. The parameters `$sr_knn_path` and `$ls_alpha` are the same as in the in-memory method.

```bash
./apps/build_disk_index --data_type float --dist_fn $build_dis_method --knn_path $sr_knn_path --data_path $data_path --index_path_prefix $index_path_prefix -R $R -L $L -B $B -M $M -a $ls_alpha -T $T --QD $QD

./apps/search_disk_index  --data_type float --dist_fn $search_dis_method --index_path_prefix $index_path_prefix --query_file $query_path --gt_file $gt_path --result_path $res_path_prefix -K $recall_K -T $T -L 10 20 30 40 50 60 80 100 --num_nodes_to_cache 10000
```

This setup can be executed via the `opt/disk.sh` script, which corresponds to the above code for reference.