DiskANN=/home/hanhan/Projects/diskann
data=/data1/data
build=$DiskANN/build
# dataset=sift1M build_dis_method=ls search_dis_method=l2 R=64 L=100 B=0.016 K=1
dataset=$1 build_dis_method=$2 search_dis_method=l2 R=$3 L=$4 B=1 QD=$5 M=$6 gt_K=100 recall_K=1 T=16 ls_alpha=1.0 
if [ "$build_dis_method" == 'ls' ]; then
    sr=$7
    info="${dataset}_${build_dis_method}_R${R}_L${L}_QD${QD}_M${M}_sr${sr}"
else
    info="${dataset}_${build_dis_method}_R${R}_L${L}_QD${QD}_M${M}"
fi

# output=$data/$info # HDD
output=$DiskANN/log/$info # SSD
log_file=$DiskANN/log/$info.log
data_path=$data/$dataset/${dataset}_base.fbin
query_path=$data/$dataset/${dataset}_query.fbin
gt_path=$data/$dataset/${dataset}_gt${gt_K}
sr_knn_path=$data/$dataset/${dataset}_knn100_sr${sr}
index_path_prefix=$output/index/$dataset
res_path_prefix=$output/res/$dataset

cd $build && make -j
mkdir $output $output/index $output/res && > $log_file
echo '-----build-----' 2>&1 | tee -a $log_file
./apps/build_disk_index --data_type float --dist_fn $build_dis_method --knn_path $sr_knn_path \
    --data_path $data_path --index_path_prefix $index_path_prefix \
    -R $R -L $L -B $B -M $M -a $ls_alpha -T $T --QD $QD \
    2>&1 | tee -a $log_file
echo '-----search-----' 2>&1 | tee -a $log_file
./apps/search_disk_index  --data_type float --dist_fn $search_dis_method \
    --index_path_prefix $index_path_prefix --query_file $query_path --gt_file $gt_path --result_path $res_path_prefix \
    -K $recall_K -T $T \
    -L 10 20 30 40 50 60 80 100 200 300 400 500 600 800 1000 2000 3000 4000 5000 6000 8000 10000\
    --num_nodes_to_cache 10000 2>&1 | tee -a $log_file
rm -rf $output