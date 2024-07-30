DiskANN=/home/hanhan/Projects/diskann
# data=/home/hanhan/data
data=/data1/data
build=$DiskANN/build
dataset=$1 dis_method=l2 K=100 sr=$2
data_file=$data/$dataset/${dataset}_base.fvecs
query_file=$data/$dataset/${dataset}_query.fvecs
if [ "$sr" == '1' ]; then
    sr_file=$data_file
else
    sr_file=$data/$dataset/${dataset}_base_sr${sr}.fvecs
fi
srlog_path=$DiskANN/log/${dataset}_sr${sr}_K${K}.log

data_path=$data/$dataset/${dataset}_base.fbin
query_path=$data/$dataset/${dataset}_query.fbin
if [ "$sr" == '1' ]; then
    sr_path=$data_path
else
    sr_path=$data/$dataset/${dataset}_base_sr${sr}.fbin
fi
gt_path=$data/$dataset/${dataset}_gt${K}
sr_knn_path=$data/$dataset/${dataset}_knn${K}_sr${sr}

cd $build && make -j

# 生成l2计算标准数据
# ./apps/utils/fvecs_to_bin float $data_file $data_path
# ./apps/utils/fvecs_to_bin float $query_file $query_path
# ./apps/utils/compute_groundtruth  --data_type float --dist_fn $dis_method \
#     --base_file $data_path --query_file $query_path --gt_file $gt_path --K $K

# 生成LS avgknn数据（需要调整采样数据则修改sr）
# ./apps/utils/fvecs_to_bin float $sr_file $sr_path
./apps/utils/compute_groundtruth  --data_type float --dist_fn $dis_method  \
    --base_file $sr_path --query_file $data_path --gt_file $sr_knn_path --K $K \
    2>&1 | tee $srlog_path
