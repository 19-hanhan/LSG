/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include <faiss/IndexNSG.h>

#include <omp.h>

#include <cinttypes>
#include <memory>

#include <faiss/IndexFlat.h>
#include <faiss/IndexNNDescent.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/Heap.h>
#include <faiss/utils/distances.h>

#include<iostream>
#include<fstream>

namespace faiss {

using idx_t = Index::idx_t;
using namespace nsg;

/**************************************************************
 * IndexNSG implementation
 **************************************************************/

IndexNSG::IndexNSG(int d, int R, MetricType metric)
        : Index(d, metric),
          nsg(R),
          own_fields(false),
          storage(nullptr),
          is_built(false),
          GK(64),
          // GK(200),
          build_type(0) {
            // 设定nndescent参数
            nndescent_S = 10;
            nndescent_R = 100;
            nndescent_L = GK + 50;
            nndescent_iter = 10; 
            }

// build_type(默认方式)
IndexNSG::IndexNSG(Index* storage, int R)
        : Index(storage->d, storage->metric_type),
          nsg(R),
          own_fields(false),
          storage(storage),
          is_built(false),
          // 初始KNN图参数
          GK(R), // 邻居数
          build_type(1) {
            nndescent_S = 10; // 随机选取的初始邻居数
            nndescent_R = 100; // 搜索深度
            nndescent_L = GK + 50; // 候选集
            nndescent_iter = 10; // 迭代次数
            }

IndexNSG::~IndexNSG() {
    if (own_fields) {
        delete storage;
    }
}

void IndexNSG::train(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(
            storage,
            "Please use IndexNSGFlat (or variants) instead of IndexNSG directly");
    // nsg structure does not require training
    storage->train(n, x);
    is_trained = true;
}

void IndexNSG::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels) const

{
    FAISS_THROW_IF_NOT_MSG(
            storage,
            "Please use IndexNSGFlat (or variants) instead of IndexNSG directly");

    int L = std::max(nsg.search_L, (int)k); // in case of search L = -1
    idx_t check_period = InterruptCallback::get_period_hint(d * L);

    for (idx_t i0 = 0; i0 < n; i0 += check_period) {
        idx_t i1 = std::min(i0 + check_period, n);

#pragma omp parallel
        {
            VisitedTable vt(ntotal);

            DistanceComputer* dis = storage_distance_computer(storage);
            ScopeDeleter1<DistanceComputer> del(dis);

#pragma omp for
            for (idx_t i = i0; i < i1; i++) {
                idx_t* idxi = labels + i * k;
                float* simi = distances + i * k;
                dis->set_query(x + i * d);

                maxheap_heapify(k, simi, idxi);
                nsg.search(*dis, k, idxi, simi, vt);
                maxheap_reorder(k, simi, idxi);

                vt.advance();
            }
        }
        InterruptCallback::check();
    }

    if (metric_type == METRIC_INNER_PRODUCT) {
        // we need to revert the negated distances
        for (size_t i = 0; i < k * n; i++) {
            distances[i] = -distances[i];
        }
    }
}

// 用于设定NICDM参数
void IndexNSG::set_nicdm_distance(float* x, float y) {
    // 设定DistanceComputer的NICDM参数
    storage->set_nicdm_distance(x,y);
}

void IndexNSG::search_enhence_with_hubs(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,int xb_size) const

{
    FAISS_THROW_IF_NOT_MSG(
            storage,
            "Please use IndexNSGFlat (or variants) instead of IndexNSG directly");

    // 分成多组以实现并行搜索
    int L = std::max(nsg.search_L, (int)k); // in case of search L = -1
    idx_t check_period = InterruptCallback::get_period_hint(d * L);

    // 只是创建文件，然后关闭，等待写入
    std::ofstream tempfile("temp.txt");
    tempfile.close(); 

    for (idx_t i0 = 0; i0 < n; i0 += check_period) {
        idx_t i1 = std::min(i0 + check_period, n);

// #pragma omp parallel
        {
            // 这里有个大坑，该 ntotal 是 Index基类继承变量
            // 而NSG 本身也有一个ntotal，
            // 下边热点方法中使用的vt可能会越界
             
            VisitedTable vt(ntotal);

            //第3个距离计算器，用于NSG的搜索
            DistanceComputer* dis = storage_distance_computer(storage);
            ScopeDeleter1<DistanceComputer> del(dis);

// #pragma omp for
            for (idx_t i = i0; i < i1; i++) {
                idx_t* idxi = labels + i * k;
                float* simi = distances + i * k;
                // 设置查询向量
                dis->set_query(x + i * d);
                // 初始化最大堆
                maxheap_heapify(k, simi, idxi);
                // 搜索
                nsg.search_enhence_with_hubs(*dis, k, idxi, simi, vt, xb_size);
                // 重新排序堆中的元素,处理结果
                maxheap_reorder(k, simi, idxi);
                // 更新访问记录表
                vt.advance();
            }
        }
        InterruptCallback::check();
    }

    // 读取数据
    std::ifstream infile("temp.txt");
    if (!infile.is_open()) {
        std::cerr << "无法打开文件" << std::endl;
    }
    std::vector<int> data;
    int value;
    while (infile >> value) {
        data.push_back(value);
    }
    infile.close();
    std::remove("temp.txt");

    // 访问点统计计算，写入到大文件
    std::ofstream outfile("count.txt",std::ios::app);
    int64_t count=0; //在data较大时，普通Int可能会溢出
    for (const int& val : data) {
        count += val; // double(val)/n 与 double(val/n)不同
    }
    count = count/n;
    outfile << count << std::endl;
    outfile.close(); 

    if (metric_type == METRIC_INNER_PRODUCT) {
        // we need to revert the negated distances
        for (size_t i = 0; i < k * n; i++) {
            distances[i] = -distances[i];
        }
    }
}


// 如果已经有了创建好的nndescent图，则可以直接构建，不需要使用add 构建
void IndexNSG::build(idx_t n, const float* x, idx_t* knn_graph, int GK) {
    FAISS_THROW_IF_NOT_MSG(
            storage,
            "Please use IndexNSGFlat (or variants) instead of IndexNSG directly");
    FAISS_THROW_IF_NOT_MSG(
            !is_built && ntotal == 0, "The IndexNSG is already built");

    storage->add(n, x);
    ntotal = storage->ntotal;

    // check the knn graph
    check_knn_graph(knn_graph, n, GK);

    const nsg::Graph<idx_t> knng(knn_graph, n, GK);
    nsg.build(storage, n, knng, verbose);
    is_built = true;
}

// 没有初始knn图则新建nndescent图
void IndexNSG::add(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(
            storage,
            "Please use IndexNSGFlat (or variants) "
            "instead of IndexNSG directly");
    FAISS_THROW_IF_NOT(is_trained);

    FAISS_THROW_IF_NOT_MSG(
            !is_built && ntotal == 0,
            "NSG does not support incremental addition");

    std::vector<idx_t> knng;
    if (verbose) {
        printf("IndexNSG::add %zd vectors\n", size_t(n));
    }

    // 构建近似KNN图

    // 暴力搜索方法构建 KNN 图
    if (build_type == 0) { // build with brute force search

        if (verbose) {
            printf("  Build knn graph with brute force search on storage index\n");
        }

        storage->add(n, x);
        ntotal = storage->ntotal;
        FAISS_THROW_IF_NOT(ntotal == n);

        knng.resize(ntotal * (GK + 1));
        storage->assign(ntotal, x, knng.data(), GK + 1); // 寻找k近邻方法

        // Remove itself
        // - For metric distance, we just need to remove the first neighbor
        // - But for non-metric, e.g. inner product, we need to check
        // - each neighbor
        if (storage->metric_type == METRIC_INNER_PRODUCT) {
            for (idx_t i = 0; i < ntotal; i++) {
                int count = 0;
                for (int j = 0; j < GK + 1; j++) {
                    idx_t id = knng[i * (GK + 1) + j];
                    if (id != i) {
                        knng[i * GK + count] = id;
                        count += 1;
                    }
                    if (count == GK) {
                        break;
                    }
                }
            }
        } else {
            for (idx_t i = 0; i < ntotal; i++) {
                memmove(knng.data() + i * GK,
                        knng.data() + i * (GK + 1) + 1,
                        GK * sizeof(idx_t));
            }
        }

    } 
    // 使用 NNDescent 方法构建 KNN 图。
    else if (build_type == 1) { // build with NNDescent
        IndexNNDescent index(storage, GK);
        index.nndescent.S = nndescent_S;
        index.nndescent.R = nndescent_R;
        index.nndescent.L = std::max(nndescent_L, GK + 50);
        index.nndescent.iter = nndescent_iter;
        // 默认参数如下：
        // GK(64),
        // build_type(1) {
        // nndescent_S = 10;
        // nndescent_R = 100;
        // nndescent_L = GK + 50;
        // nndescent_iter = 10;
        if (verbose) {
            printf("  Build knn graph with NNdescent S=%d R=%d L=%d niter=%d\n",
                   index.nndescent.S,
                   index.nndescent.R,
                   index.nndescent.L,
                   index.nndescent.iter);
        }

        // prevent IndexNSG from deleting the storage
        index.own_fields = false;

        auto start = std::chrono::high_resolution_clock::now();
        // nndescent构图
        index.add(n, x);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;
        std::cout<<"nndescent构图运行时间:"<<duration.count()<<std::endl;

        // storage->add is already implicit called in IndexNSG.add
        ntotal = storage->ntotal;
        FAISS_THROW_IF_NOT(ntotal == n);

        // 最小为64+50
        knng.resize(ntotal * GK);

        // cast from idx_t to int
        const int* knn_graph = index.nndescent.final_graph.data();
#pragma omp parallel for
        for (idx_t i = 0; i < ntotal * GK; i++) {
            knng[i] = knn_graph[i];
        }
    } else {
        FAISS_THROW_MSG("build_type should be 0 or 1");
    }

    if (verbose) {
        printf("  Check the knn graph\n");
    }

    // check the knn graph
    check_knn_graph(knng.data(), n, GK);

    if (verbose) {
        printf("  nsg building\n");
    }

    // 根据nndescent图创建nsg图
    const nsg::Graph<idx_t> knn_graph(knng.data(), n, GK);

    nsg.build(storage, n, knn_graph, verbose);
    is_built = true;
}


// 统计每层邻居
void avgOutdegres(std::vector<std::unordered_map<idx_t,std::vector<idx_t>>>& hot_hubs){
    // 热点反向邻居个数统计
    std::map<int,int> hubs_distr;
    for (int i = 0; i < hot_hubs.size(); ++i)
    {
        auto each_level = hot_hubs[i];
        int mm = -1,mn = INT_MAX;
        double avg = 0; 
        size_t sum = 0;
        // a.first 为热点id， a.second 为热点反向邻居集合
        for (auto a:each_level)
        {
            // 热点a，有a.second.size()个反向邻居的点个数+1
            hubs_distr[a.second.size()]++;
            sum += a.second.size();
            mm = std::max(mm ,(int)a.second.size());
            mn = std::min(mn, (int)a.second.size());
        }
        avg = (double)sum/(each_level.size()*1.0);
        printf("第%d层热点最大反向邻居个数为：%d,最小邻居个数为：%d,平均邻居个数为：%0.3f\n", i, mm, mn ,avg);
    }
}

// 入度分布统计
void IndexNSG::Indegree_sta(int n){
    std::map<int,int> indegrees;
    for (int i = 0; i < ntotal; i++) {
        int size = 0;
        while (size < nsg.R && nsg.final_graph->at(i, size) != -1) {
            // i的第size个邻居
            int v1 = nsg.final_graph->at(i,size);
            if (v1<0 || v1>n)
                break;
            indegrees[v1]++;
            size += 1;
        }
    }
    // printf("0层平均出度为 ： %0.3f\t,邻居占满比例：%0.3f\n", sum/(n*1.0),cnt/(n*1.0));

    std::ofstream out("./out_data/sift1m_hubs_24.csv",std::ofstream::app);
    for (auto a:indegrees)
    {   
        out<<a.second<<std::endl;

    }
    out.close();

    

    // printf("----------------------------\n");

}

// 将热点索引（直接增强旧索引）
void IndexNSG::combine_index_with_hot_hubs_enhence(idx_t n,int len_ratios,
        const float* ht_hbs_ratios,const int* nb_nbors_per_level){
    
    Indegree_sta(n);
    double t0  = getmillisecs();
    std::vector<float> ratios;
    std::vector<int> nb_reverse_neighbors;
    idx_t tmp = n;
    // printf("ok!!\n");
    for (int i = 0; i < len_ratios; i++)
    {   
        // 扩充之后的容量
        tmp += n*ht_hbs_ratios[i]*nb_nbors_per_level[i];// 按热点等级分配热点新增邻居个数（若三层：3*m ,2*m ,1*m ,m为0层邻居数）
        // 热点比例
        ratios.push_back(ht_hbs_ratios[i]);
        // 每种比例下添加反向边个数
        nb_reverse_neighbors.push_back(nb_nbors_per_level[i]);
    }
    tmp += 5; // 防止下取整
    nsg.ntotal = tmp;
    ntotal = tmp;
    // 统计热点,将多层次热点及与他们相连的邻居放到hot_hubs中
    std::vector<std::unordered_map<idx_t,std::vector<idx_t>>> hot_hubs(len_ratios);
    // printf("len_ratios%d\t,ratios0%f\t,ratios1%f\n",len_ratios,ratios[0],ratios[1]);
    nsg.find_hot_hubs_enhence(hot_hubs,n,ratios);    
    // 为nsg扩容（指针数组只能使用复制形式扩容）
    // copy constructor
    // Graph(const Graph& g) : Graph(g.N, g.K) {
    //     memcpy(data, g.data, N * K * sizeof(node_t));
    // }
    // printf("final_graph[0][0]:%d\t,final_graph[100][0]:%d\n",(int)nsg.final_graph->at(n-1,0),(int)nsg.final_graph->at(100,0));
    auto final_graph_tmp = std::make_shared<nsg::Graph<int>>(nsg.ntotal, nsg.final_graph->K);
    std::fill_n(final_graph_tmp->data, nsg.ntotal * nsg.final_graph->K, -1);
    memcpy(final_graph_tmp->data, nsg.final_graph->data, n * nsg.final_graph->K * sizeof(int));
    nsg.final_graph = final_graph_tmp;

    nsg.add_new_reverse_link_end_enhence(hot_hubs,n,nb_reverse_neighbors);
    // for(int j=n;j<ntotal;j++){
    //     for (int i = 0; i < nsg.final_graph->K; ++i)
    //     {
    //         printf("final_graph[0][0]:%d\n",(int)nsg.final_graph->at(j,i));
    //     } 
    // }
    Indegree_sta(n);
    avgOutdegres(hot_hubs);
    
}





void IndexNSG::reset() {
    nsg.reset();
    storage->reset();
    ntotal = 0;
    is_built = false;
}

void IndexNSG::reconstruct(idx_t key, float* recons) const {
    storage->reconstruct(key, recons);
}


// 检查knng 图是否合法（id越界，或者自环）
void IndexNSG::check_knn_graph(const idx_t* knn_graph, idx_t n, int K) const {
    idx_t total_count = 0;

#pragma omp parallel for reduction(+ : total_count)
    for (idx_t i = 0; i < n; i++) {
        int count = 0;
        for (int j = 0; j < K; j++) {
            idx_t id = knn_graph[i * K + j];
            if (id < 0 || id >= n || id == i) {
                count += 1;
            }
        }
        total_count += count;
    }

    if (total_count > 0) {
        fprintf(stderr,
                "WARNING: the input knn graph "
                "has %" PRId64 " invalid entries\n",
                total_count);
    }
    FAISS_THROW_IF_NOT_MSG(
            total_count < n / 10,
            "There are too much invalid entries in the knn graph. "
            "It may be an invalid knn graph.");
}

/**************************************************************
 * IndexNSGFlat implementation
 **************************************************************/

IndexNSGFlat::IndexNSGFlat() {
    is_trained = true;
}

IndexNSGFlat::IndexNSGFlat(int d, int R, MetricType metric)
        : IndexNSG(new IndexFlat(d, metric), R) {
        //: IndexNSG(new IndexFlat(d, METRIC_NICDM), R) {
    own_fields = true;
    is_trained = true;
}

} // namespace faiss
