/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD+Patents license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include "IndexHNSW.h"


#include <cstdlib>
#include <cassert>
#include <cstring>
#include <cstdio>
#include <cmath>
#include <omp.h>
#include <functional>
#include <bits/stdc++.h>

#include <unordered_set>
#include <queue>

#include <iostream>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdint.h>

#ifdef __SSE__
#include <immintrin.h>
#endif

#include "utils.h"
#include "Heap.h"
#include "FaissAssert.h"
#include "IndexFlat.h"
#include "IndexIVFPQ.h"

float* AVGDIS;
float alpha;

extern "C" {

/* declare BLAS functions, see http://www.netlib.org/clapack/cblas/ */

int sgemm_ (const char *transa, const char *transb, FINTEGER *m, FINTEGER *
            n, FINTEGER *k, const float *alpha, const float *a,
            FINTEGER *lda, const float *b, FINTEGER *
            ldb, float *beta, float *c, FINTEGER *ldc);

}

namespace faiss {

using idx_t = Index::idx_t;
using MinimaxHeap = HNSW::MinimaxHeap;
using storage_idx_t = HNSW::storage_idx_t;
using NodeDistCloser = HNSW::NodeDistCloser;
using NodeDistFarther = HNSW::NodeDistFarther;
using DistanceComputer = HNSW::DistanceComputer;

HNSWStats hnsw_stats;

/**************************************************************
 * add / search blocks of descriptors
 **************************************************************/

namespace {

// 若hnsw.levels.size == ntotal preset_levels = true;
// n0 为ntotal
void hnsw_add_vertices(IndexHNSW &index_hnsw,
                       size_t n0,
                       size_t n, const float *x,
                       bool verbose,
                       bool preset_levels = false) {
    HNSW & hnsw = index_hnsw.hnsw;
    size_t ntotal = n0 + n;
    double t0 = getmillisecs();
    if (verbose) {
        printf("hnsw_add_vertices: adding %ld elements on top of %ld "
               "(preset_levels=%d)\n",
               n, n0, int(preset_levels));
    }

    // prepare_level_tab 产生最大层次函数
    // int max_level = hnsw.prepare_level_tab(n, preset_levels);
    int max_level = hnsw.prepare_level_tab(n, preset_levels);

    if (verbose) {
        printf("  max_level = %d\n", max_level);
    }

/*    
    OpenMP中的互斥锁函数：

　　void omp_init_lock(omp_lock *) 初始化互斥器

　　void omp_destroy_lock(omp_lock *)销毁互斥器

　　void omp_set_lock(omp_lock *)获得互斥器

　　void omp_unset_lock(omp_lock *)释放互斥器

　　bool omp_test_lock(omp_lock *)试图获得互斥器，如果获得成功返回true，否则返回false
*/
    std::vector<omp_lock_t> locks(ntotal);
    for(int i = 0; i < ntotal; i++)
        omp_init_lock(&locks[i]);


    // add vectors from highest to lowest level
    std::vector<int> hist;
    std::vector<int> order(n);

    { // make buckets with vectors of the same level

        // build histogram
        for (int i = 0; i < n; i++) {
            // pt_id 表示第i个节点
            storage_idx_t pt_id = i + n0;
            // 获取当前节点的最高层
            int pt_level = hnsw.levels[pt_id] - 1;
            // 如果当前节点层次 大于hist中的最大值，扩展
            while (pt_level >= hist.size())
                hist.push_back(0);
            // 相当于计数器，记录每个层次的节点个数
            hist[pt_level] ++;
        }

        // accumulate
        /*  
            hist.size() 为层次数，
            offsets为前缀和数组，hist的前缀值
            offsets[1]代表hist[0]中的点数
            offsets[2]代表hist[0]+hist[1]中的点数
        */ 
        std::vector<int> offsets(hist.size() + 1, 0);
        for (int i = 0; i < hist.size() - 1; i++) {
            offsets[i + 1] = offsets[i] + hist[i];
        }

        // bucket sort
        for (int i = 0; i < n; i++) {
            storage_idx_t pt_id = i + n0;
            int pt_level = hnsw.levels[pt_id] - 1;
            // order 存放的为所有（层）的点的位置，包含了重复的点
            order[offsets[pt_level]++] = pt_id;
        }
    }


    { // perform add
        RandomGenerator rng2(789);

        int i1 = n;

        for (int pt_level = hist.size() - 1; pt_level >= 0; pt_level--) {
            int i0 = i1 - hist[pt_level];

            if (verbose) {
                printf("Adding %d elements at level %d\n",
                       i1 - i0, pt_level);
            }

            // random permutation to get rid of dataset order bias
            // 随机替换当前层次中的点位置
            for (int j = i0; j < i1; j++)
                std::swap(order[j], order[j + rng2.rand_int(i1 - j)]);

#pragma omp parallel
            {
                VisitedTable vt (ntotal);

                DistanceComputer *dis = index_hnsw.get_distance_computer();
                ScopeDeleter1<DistanceComputer> del(dis);
                int prev_display = verbose && omp_get_thread_num() == 0 ? 0 : -1;

#pragma omp  for schedule(dynamic)
                for (int i = i0; i < i1; i++) {
                    storage_idx_t pt_id = order[i];
                    dis->set_query (x + (pt_id - n0) * dis->d, pt_id);

                    hnsw.add_with_locks(*dis, pt_level, pt_id, locks, vt);

                    if (prev_display >= 0 && i - i0 > prev_display + 10000) {
                        prev_display = i - i0;
                        printf("  %d / %d\r", i - i0, i1 - i0);
                        fflush(stdout);
                    }
                }
            }
            i1 = i0;
        }
        FAISS_ASSERT(i1 == 0);
    }
    if (verbose) {
        printf("Done in %.3f ms\n", getmillisecs() - t0);
    }

    for(int i = 0; i < ntotal; i++) {
        omp_destroy_lock(&locks[i]);
    }
}


}  // namespace




/**************************************************************
 * IndexHNSW implementation
 **************************************************************/

IndexHNSW::IndexHNSW(int d, int M):
    Index(d, METRIC_L2),
    hnsw(M),
    own_fields(false),
    storage(nullptr),
    reconstruct_from_neighbors(nullptr){
    printf("hanhan 1.5.0 HNSW\n");
}

IndexHNSW::IndexHNSW(Index *storage, int M):
    Index(storage->d, METRIC_L2),
    hnsw(M),
    own_fields(false),
    storage(storage),
    reconstruct_from_neighbors(nullptr){
    printf("hanhan 1.5.0 HNSW\n");
}

IndexHNSW::~IndexHNSW() {
    if (own_fields) {
        delete storage;
    }
}

void IndexHNSW::train(idx_t n, const float* x)
{
    // hnsw structure does not require training
    storage->train (n, x);
    is_trained = true;
}

void IndexHNSW::search (idx_t n, const float *x, idx_t k,
                        float *distances, idx_t *labels) const

{

#pragma omp parallel
    {
        VisitedTable vt (ntotal);
        DistanceComputer *dis = get_distance_computer();
        ScopeDeleter1<DistanceComputer> del(dis);
        size_t nreorder = 0;

#pragma omp for
        for(idx_t i = 0; i < n; i++) {
            idx_t * idxi = labels + i * k;
            float * simi = distances + i * k;
            dis->set_query(x + i * d);

            maxheap_heapify (k, simi, idxi);
            hnsw.search(*dis, k, idxi, simi, vt);

            maxheap_reorder (k, simi, idxi);

            if (reconstruct_from_neighbors &&
                reconstruct_from_neighbors->k_reorder != 0) {
                int k_reorder = reconstruct_from_neighbors->k_reorder;
                if (k_reorder == -1 || k_reorder > k) k_reorder = k;

                nreorder += reconstruct_from_neighbors->compute_distances(
                       k_reorder, idxi, x + i * d, simi);

                // sort top k_reorder
                maxheap_heapify (k_reorder, simi, idxi, simi, idxi, k_reorder);
                maxheap_reorder (k_reorder, simi, idxi);
            }
        }
#pragma omp critical
        {
            hnsw_stats.nreorder += nreorder;
        }
    }


}





void IndexHNSW::search1 (idx_t n, float *x, float *y ,float *distances, idx_t *labels,
    float *distances1, idx_t *labels1,IndexHNSW* index1,IndexHNSW* index2,idx_t k)

{

    // #pragma omp parallel
    {
        VisitedTable vt (ntotal);
        DistanceComputer *dis = index1->get_distance_computer();
        ScopeDeleter1<DistanceComputer> del(dis);


        VisitedTable vt1 (ntotal);
        // dis的获取需要 通过storage（也就是传入的index），因此要通过调用的方式获取dis
        DistanceComputer *dis1 = index2->get_distance_computer();
        ScopeDeleter1<DistanceComputer> del1(dis1);

        size_t nreorder = 0;
        float cnt=0;
        float cnt1=0;
        float cnt2=0;
// #pragma omp for
        for(idx_t i = 0; i < n; i++) {
            idx_t * idxi = labels + i * k;
            float * simi = distances + i * k;
            dis->set_query(x + i * d);

            maxheap_heapify (k, simi, idxi);
            // index2所需变量
            idx_t * idxi1 = labels1 + i * k;
            float * simi1 = distances1 + i * k;
            dis1->set_query(y + i * d);

            maxheap_heapify (k, simi1, idxi1);

            //hnsw.search_rt_array(*dis, k, idxi, simi,vt,vct);
            index1->hnsw.search_rt_array(*dis, k, idxi, simi,vt,index1->vct);
            index2->hnsw.search_rt_array(*dis1, k, idxi1, simi1,vt1,index2->vct);
            /*if ( i==0 )
            {
                for (int i = 0; i < index1->vct.size(); ++i)
                {
                    std::cout<<index1->vct[i]<<std::endl;

                }

                std::cout<<":::"<<std::endl;  

                for (int i = 0; i < index2->vct.size(); ++i)
                {
                    std::cout<<index2->vct[i]<<std::endl;
                    
                }
            }*/
            //std::cout<<index2->vct.size()<<"::"<<index1->vct.size()<<std::endl;
            /*cnt1+=index1->vct.size();
            cnt2+=index2->vct.size();
            for (int i = 0; i < index1->vct.size(); ++i)
            {
                for (int j = 0; j < index2->vct.size(); ++j)
                {
                    if(index1->vct[i]==index2->vct[j])
                    {
                        cnt++;
                        break;
                    }
                }
            }*/

            index2->vct.resize(0);
            index1->vct.resize(0);
  
            //maxheap_reorder (k, simi, idxi);
            /*
            if (reconstruct_from_neighbors &&
                reconstruct_from_neighbors->k_reorder != 0) {
                int k_reorder = reconstruct_from_neighbors->k_reorder;
                if (k_reorder == -1 || k_reorder > k) k_reorder = k;

                nreorder += reconstruct_from_neighbors->compute_distances(
                       k_reorder, idxi, x + i * d, simi);

                // sort top k_reorder
                maxheap_heapify (k_reorder, simi, idxi, simi, idxi, k_reorder);
                maxheap_reorder (k_reorder, simi, idxi);
            }
            */
        }
        //std::cout<<cnt<<"：："<<cnt1<<"::"<<cnt2<<"::"<<cnt*2/(cnt1+cnt2)<<std::endl;
// #pragma omp critical
        {
            hnsw_stats.nreorder += nreorder;
        }
    }

}


// 将划分index合并
void IndexHNSW::combine_index_with_division(IndexHNSW& index,
        IndexHNSW& index1,IndexHNSW& index2,unsigned n){
    // printf("ok!\n");
    index.hnsw.prepare_level_tab2(n, true);
    index.ntotal = n;
    index.f_storage=index1.storage;
    index.s_storage=index2.storage;
    for (int i = 0; i < n; ++i)
    {
        // 三个索引中的
        size_t begin, end;
        index1.hnsw.neighbor_range(i, 0, &begin, &end);
        size_t begin1, end1;
        index2.hnsw.neighbor_range(i, 0, &begin1, &end1);
        // 合并后的索引
        size_t begin2, end2;
        index.hnsw.neighbor_range(i, 0, &begin2, &end2);
        int m1=0,m2=0;
        for (int j = begin; j < end; ++j)
        {   
            idx_t v1=index1.hnsw.neighbors[j];
            if(v1>=0&&v1<n)
                m1++;
        }
        for (int j = begin1; j < end1; ++j)
        {   
            idx_t v1=index2.hnsw.neighbors[j];
            if(v1>=0&&v1<n)
                m2++;
        }
        int cnt1=m1,cnt2=m2;
        int M = index.hnsw.nb_neighbors(0);
        // 索引1中的邻居相对于索引2会多减一个
        if(m1+m2>M-1){
            // 上取整
            cnt1=m1-((m1+m2-M+1)*m1+m1+m2-1)/(m1+m2);
            // 下取整
            cnt2=m2-(m1+m2-M+1)*m2/(m1+m2);
        }
        // 合并两个邻居，index1前往后，索引2从后往前
        while(cnt1){
            index.hnsw.neighbors[begin2++]=index1.hnsw.neighbors[begin++];
            cnt1--;
        }
        while(cnt2){
            index.hnsw.neighbors[--end2]=index2.hnsw.neighbors[begin1++];
            cnt2--;
        }
        // 索引1的末尾处为分隔符-2位置
        index.hnsw.neighbors[begin2]=-2;
    }
    /*printf("132%d\n",index.hnsw.nb_neighbors(0));
    for (int i = 0; i < 10; ++i)
    {
        size_t begin, end;
        index1.hnsw.neighbor_range(i, 0, &begin, &end);
        size_t begin1, end1;
        index1.hnsw.neighbor_range(i, 0, &begin1, &end1);
        size_t begin2, end2;
        index.hnsw.neighbor_range(i, 0, &begin2, &end2);
        for (int j = begin; j < end; ++j)
        {
            idx_t v1=index1.hnsw.neighbors[j];
            if(v1<0||v1>n)
                break;
            printf("1 :%d ", v1);
        }
        printf("\n");
        for (int j = begin2; j < end1; ++j)
        {
            idx_t v1=index2.hnsw.neighbors[j];
            if(v1<0||v1>n)
                break;
            printf("2 :%d ", v1);
        }
        printf("\n");
        for (int j = begin2; j < end2; ++j)
        {
            size_t v1=index.hnsw.neighbors[j];
            printf("3 :%d ", v1);
        }
        printf("\n");
    }*/
}


void IndexHNSW::final_top_100(float* simi,idx_t* idxi,
            float*simi1,idx_t* idxi1,
            float*simi2,idx_t* idxi2,
            DistanceComputer& dis1,
            DistanceComputer& dis2,
            idx_t k) const{
    std::priority_queue<std::pair<float,idx_t>> top_100_pq;
    // 防止索引1索引2处理重复元素
    std::unordered_set<idx_t> common_elements;
    // 遍历数组
    /*for (int i = 0; i < k; ++i)
    {
        if(idxi1[i]<0)
            break;
        common_elements.insert(idxi1[i]);
        float d0 = simi1[i];
        d0 += dis2(idxi1[i]);
        
        if(top_100_pq.size()<100||(!top_100_pq.empty()&&top_100_pq.top().first>d0)){
            top_100_pq.emplace(std::make_pair(d0,idxi1[i]));
            if (top_100_pq.size()>100)
            {
                top_100_pq.pop();
            }
        }
    }

    for (int i = 0; i < k; ++i)
    {
        if(common_elements.find(idxi2[i])!=common_elements.end())
            continue;
        if(idxi2[i]<0)
            break;
        float d0 = simi2[i];
        d0 += dis1(idxi2[i]);
        
        if(top_100_pq.size()<100||(!top_100_pq.empty()&&top_100_pq.top().first>d0)){
            top_100_pq.emplace(std::make_pair(d0,idxi2[i]));
            if (top_100_pq.size()>100)
            {
                top_100_pq.pop();
            }
        }
    }*/
    // 计算出所有距离
    std::vector<std::pair<float,idx_t>> res;
    // printf("ok1\n");
    for (int i = 0; i < k; ++i)
    {
        if (idxi1[i]<0)
            break;
        // common_elements.insert(idxi1[i]);
        float d0 = simi1[i] + dis2(idxi1[i]); 
        res.emplace_back(d0,idxi1[i]);
    }

    //printf("ok2\n");
    for (int i = 0; i < k; ++i)
    {
        //printf("a\n");
        if (idxi2[i]<0)
            break;
        //printf("b\n");
        /*if(common_elements.find(idxi2[i])!=common_elements.end())
            continue;*/
        // printf("dfL:%ld\n",idxi2[] );
        float d0 = simi2[i] + dis1(idxi2[i]);   
        //printf("c\n");
        res.emplace_back(d0,idxi2[i]);
        //printf("d\n");
    }
    //printf("ok3\n");
    std::partial_sort(std::begin(res), std::begin(res) + 1, std::end(res));

    /*for (int i = 0; i < 1; ++i)
    {
        // 防止res中的数量不足k个
        if(i+1>res.size())
            idxi[i]=-1;
        idxi[i]=res[i].second;
    }*/
    idxi[0] = res[0].second; 
    //printf("ok4\n");
/*    // 将100个最近邻存到idxt，simi中
    int i=0;
    while(!top_100_pq.empty()){
        idxi[i]=top_100_pq.top().second;
        simi[i]=top_100_pq.top().first;
        top_100_pq.pop();
        i++;
    }*/
}


void IndexHNSW::combine_search(
        idx_t n,
        const float* x,
        const float* y,
        idx_t k,
        float* distances,
        idx_t* labels,
        float* distances1,
        idx_t* labels1,
        float* distances2,
        idx_t* labels2) const

{
    FAISS_THROW_IF_NOT(k > 0);

    FAISS_THROW_IF_NOT_MSG(
            storage,
            "Please use IndexHNSWFlat (or variants) instead of IndexHNSW directly");
    size_t n1 = 0, n2 = 0, n3 = 0, ndis = 0, nreorder = 0;
    idx_t check_period = 10000;
/*    idx_t RS=n*k;
    float distances1[RS];
    float distances2[RS];
    idx_t labels1[RS];
    idx_t labels2[RS];*/
    RandomGenerator rng3(789456);
    for (idx_t i0 = 0; i0 < n; i0 += check_period) {
        idx_t i1 = std::min(i0 + check_period, n);

//#pragma omp parallel
        {
            VisitedTable vt1(ntotal);
            VisitedTable vt2(ntotal);
            
            DistanceComputer* dis1 = get_distance_computer(f_storage);
            DistanceComputer* dis2 = get_distance_computer(s_storage);
            ScopeDeleter1<DistanceComputer> del1(dis1),del2(dis2);
//#pragma omp for reduction(+ : n1, n2, n3, ndis, nreorder)
            for (idx_t i = i0; i < i1; i++) {
                idx_t* idxi = labels + i * k;
                float* simi = distances + i * k;
                idx_t* idxi1 = labels1 + i * k;
                float* simi1 = distances1 + i * k;
                idx_t* idxi2 = labels2 + i * k;
                float* simi2 = distances2 + i * k;
                dis1->set_query(x + i * d);
                dis2->set_query(y + i * d);
                // 经测试单索引搜索没有问题
                hnsw.combine_search(*dis1, k, idxi1, simi1, 0,vt1,rng3);
                hnsw.combine_search(*dis2, k, idxi2, simi2,1, vt2,rng3);
                final_top_100(simi,idxi,
                                simi1,idxi1,
                                simi2,idxi2,
                                *dis1,
                                *dis2,
                                k);
                // 经测试时间影响可忽略
                maxheap_reorder(100, simi, idxi);   

                /*n1 += stats.n1;
                n2 += stats.n2;
                n3 += stats.n3;
                ndis += stats.ndis;
                nreorder += stats.nreorder;
                maxheap_reorder(k, simi, idxi);

                if (reconstruct_from_neighbors &&
                    reconstruct_from_neighbors->k_reorder != 0) {
                    int k_reorder = reconstruct_from_neighbors->k_reorder;
                    if (k_reorder == -1 || k_reorder > k)
                        k_reorder = k;

                    nreorder += reconstruct_from_neighbors->compute_distances(
                            k_reorder, idxi, x + i * d, simi);

                    // sort top k_reorder
                    maxheap_heapify(
                            k_reorder, simi, idxi, simi, idxi, k_reorder);
                    maxheap_reorder(k_reorder, simi, idxi);
                }*/
            }
        }
    }

    if (metric_type == METRIC_INNER_PRODUCT) {
        // we need to revert the negated distances
        for (size_t i = 0; i < k * n; i++) {
            distances[i] = -distances[i];
        }
    }
}


void IndexHNSW::set_nicdm_distance(float* x, float y) {
    AVGDIS = x;
    alpha = y;
}


void IndexHNSW::add(idx_t n, const float *x)
{
    FAISS_THROW_IF_NOT(is_trained);
    int n0 = ntotal;
    // storage ==  indexflat
    storage->add(n, x);
    ntotal = storage->ntotal;
    // this 表示当前调用add方法的对象 -- IndexHNSW对象
    // hnsw.levels.size() == ntotal ： bool类型
    hnsw_add_vertices (*this, n0, n, x, verbose,
                       hnsw.levels.size() == ntotal);
}


// 将热点索引（直接增强旧索引）
void IndexHNSW::combine_index_with_hot_hubs_enhence(idx_t n,int len_ratios,
        const float* ht_hbs_ratios,int find_hubs_mode,
        int find_neighbors_mode, const int* nb_nbors_per_level){
    // 每个点对应的类别

    // 节点数量+热点数量 , 放入vector中存储
    aod_level();//统计平均出度和邻居占满比例
    double t0  = getmillisecs();
    std::vector<float> ratios;
    std::vector<int> nb_reverse_neighbors ;
    idx_t tmp = n;
    // printf("ok!!\n");
    for (int i = 0; i < len_ratios; i++)
    { 
        // printf("第%d层比例%0.4f， 第%d层个数%d\n",i,ht_hbs_ratios[i],i ,nb_nbors_per_level[i]);
        // 按热点等级分配热点新增邻居个数（若三层：3*m ,2*m ,1*m ,m为0层邻居数）
        tmp += n*ht_hbs_ratios[i]*nb_nbors_per_level[i];
        ratios.push_back(ht_hbs_ratios[i]);
        nb_reverse_neighbors.push_back(nb_nbors_per_level[i]);
    }
    // 防止下取整
    tmp += 5;

    hnsw.prepare_level_tab4(n,tmp);
    ntotal = tmp;
    // 统计热点,将多层次热点及与他们相连的邻居放到hot_hubs中
    std::vector<std::unordered_map<idx_t,std::vector<idx_t>>> hot_hubs(len_ratios);
    hnsw.find_hot_hubs_enhence(hot_hubs,n,ratios,find_hubs_mode,
            find_neighbors_mode);

    // 距离计算器
    DistanceComputer* dis = get_distance_computer(storage);
    ScopeDeleter1<DistanceComputer> del(dis);

    hnsw.add_new_reverse_link_end_enhence(hot_hubs,n,*dis,nb_reverse_neighbors);


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

    // 热点反向边个数 统计输出
/*    for (auto a:hubs_distr)
    {   
        printf("%d\t", a.first);
    }

    printf("----------------------------\n");

    for (auto a:hubs_distr)
    {   
        printf("%d\t", a.second);
    }

    printf("----------------------------\n");*/

    aod_level();

/*    for (int i = 0; i < tmp; ++i)
    {
        size_t begin,end;
        hnsw.neighbor_range(i, 0, &begin, &end);
        for (int j = begin; j < end; ++j)
        {
            printf("%d\t",hnsw.neighbors[j]);
        }
        printf("\n");
    }*/

}


void IndexHNSW::aod_level(){
    
    size_t sum = 0;
    idx_t n = hnsw.levels.size();
    idx_t tot = hnsw.offsets.size();
    idx_t cnt = 0;
    int out_degree = 0;
    std::map<int,int> out_degrees;

    for (int i = 0; i < tot; ++i)
    {
        out_degree = 0;
        size_t begin,end;
        hnsw.neighbor_range(i,0,&begin,&end);
        size_t j;
        for ( j = begin; j < end; ++j)
        {
            if (hnsw.neighbors[j]<0||hnsw.neighbors[j]>=n)
            {
                break;
            }
            sum++;
            out_degree++;
        }
        if (i<n)
        {   
            if (j==end) 
                cnt++;
            out_degrees[out_degree]++;
        }

    }

    printf("0层平均出度为 ： %0.3f\t,邻居占满比例：%0.3f\n", sum/(n*1.0),cnt/(n*1.0));

/*    for (auto a:out_degrees)
    {   
        printf("%d\t", a.first);
    }

    printf("----------------------------\n");

    for (auto a:out_degrees)
    {   
        printf("%d\t", a.second);
    }

    printf("----------------------------\n");*/


}




void IndexHNSW::search_with_hot_hubs_enhence(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,size_t xb_size) const

{
    FAISS_THROW_IF_NOT(k > 0);

    FAISS_THROW_IF_NOT_MSG(
            storage,
            "Please use IndexHNSWFlat (or variants) instead of IndexHNSW directly");
    size_t n1 = 0, n2 = 0, n3 = 0, ndis = 0, nreorder = 0;
    idx_t check_period = 10000;
    for (idx_t i0 = 0; i0 < n; i0 += check_period) {
        idx_t i1 = std::min(i0 + check_period, n);

//#pragma omp parallel
        {
            VisitedTable vt(ntotal);
            RandomGenerator rng3(789456);
            DistanceComputer* dis = get_distance_computer();
            ScopeDeleter1<DistanceComputer> del(dis);
//#pragma omp for reduction(+ : n1, n2, n3, ndis, nreorder)
            for (idx_t i = i0; i < i1; i++) {
                idx_t* idxi = labels + i * k;
                float* simi = distances + i * k;
                dis->set_query(x + i * d);
                // printf("ok4\n");
                maxheap_heapify(k, simi, idxi);
                hnsw.search_with_hot_hubs_enhence(*dis, k, idxi, simi, vt,xb_size,rng3);
                // printf("ok5\n");
                maxheap_reorder(k, simi, idxi);
/*                printf("%f\t,%f\t,%d\t,%d\n",
                                    simi[0],simi[1],
                                    idxi[0],idxi[1]);*/
            }
        }
    }
}



// random_v2
void IndexHNSW::search_with_hot_hubs_enhence_random(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,size_t xb_size) const

{
    FAISS_THROW_IF_NOT(k > 0);

    FAISS_THROW_IF_NOT_MSG(
            storage,
            "Please use IndexHNSWFlat (or variants) instead of IndexHNSW directly");
    size_t n1 = 0, n2 = 0, n3 = 0, ndis = 0, nreorder = 0;
    idx_t check_period = 10000;
    for (idx_t i0 = 0; i0 < n; i0 += check_period) {
        idx_t i1 = std::min(i0 + check_period, n);

//#pragma omp parallel
        {
            VisitedTable vt(ntotal);
            RandomGenerator rng3(789456);
            DistanceComputer* dis = get_distance_computer(storage);
            ScopeDeleter1<DistanceComputer> del(dis);
//#pragma omp for reduction(+ : n1, n2, n3, ndis, nreorder)
            for (idx_t i = i0; i < i1; i++) {
                idx_t* idxi = labels + i * k;
                float* simi = distances + i * k;
                dis->set_query(x + i * d);
                // printf("ok4\n");
                maxheap_heapify(k, simi, idxi);
                hnsw.search_with_hot_hubs_enhence_random(*dis, k, idxi, simi, vt,xb_size,rng3);
                // printf("ok5\n");
                maxheap_reorder(k, simi, idxi);
/*                printf("%f\t,%f\t,%d\t,%d\n",
                                    simi[0],simi[1],
                                    idxi[0],idxi[1]);*/
            }
        }
    }
}

// pars3 为 对应关系 , 将子图的
void subTranferToAll(std::vector<std::unordered_map<idx_t,std::vector<idx_t>>>& hot_hubs,
    std::vector<std::unordered_map<idx_t,std::vector<idx_t>>>& sub_hot_hubs,
    idx_t* idxScript){
    // 遍历每一层
    for(int i=0;i<hot_hubs.size();i++){
        auto& all = hot_hubs[i];
        auto& sub = sub_hot_hubs[i];
        int sum = 0;
        // 遍历当层每一个子图热点，形成全局热点加入到全局热点中
        for (auto& b:sub)
        {
            idx_t subHubID = b.first;
            std::vector<idx_t>& nbs = b.second;
            // 找到对应的全局邻居
            idx_t allHubID = idxScript[subHubID];
            sum += nbs.size();
            for (idx_t j = 0; j < nbs.size(); ++j)
            {
                // printf("子图:%ld,对应id:%ld ：\n",nbs[j],idxScript[nbs[j]]);
                hot_hubs[i][allHubID].push_back(idxScript[nbs[j]]);    
            }
        } 
    }
}

void vctReset(std::vector<std::unordered_map<idx_t,std::vector<idx_t>>>& sub_hot_hubs){
    for (int i = 0; i < sub_hot_hubs.size(); ++i)
    {
        sub_hot_hubs[i].clear();
    }
}

// 统计每层论据
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





// 子图选择热点
void IndexHNSW::enhence_index_with_subIndex_hubs(idx_t n,int len_ratios,
        const float* ht_hbs_ratios,int find_hubs_mode,
        int find_neighbors_mode, const int* nb_nbors_per_level,
        IndexHNSW& ihf1,IndexHNSW& ihf2,IndexHNSW& ihf3,IndexHNSW& ihf4,IndexHNSW& ihf5,
        idx_t* idxScript1,idx_t* idxScript2,
        idx_t* idxScript3,idx_t* idxScript4,idx_t* idxScript5){
    // for (int i = 0; i < 5; ++i)
    // {
    //     printf("IndexHNSW：%ld\n", idxScript1[i]);
    // }
/*void IndexHNSW::enhence_index_with_subIndex_hubs(idx_t n,int len_ratios,
    const float* ht_hbs_ratios,int find_hubs_mode,
    int find_neighbors_mode, const int* nb_nbors_per_level,
    IndexHNSW& ihf1,IndexHNSW& ihf2,IndexHNSW& ihf3,
    idx_t* idxScript1,idx_t* idxScript2,
    idx_t* idxScript3){*/
// void IndexHNSW::enhence_index_with_subIndex_hubs(idx_t n,int len_ratios,
//         const float* ht_hbs_ratios,int find_hubs_mode,
//         int find_neighbors_mode, const int* nb_nbors_per_level,
//         IndexHNSW& ihf1,
//         idx_t* idxScript1){

    // 节点数量+热点数量 , 放入vector中存储
    aod_level();
    double t0  = getmillisecs();
    std::vector<float> ratios;
    std::vector<int> nb_reverse_neighbors ;
    idx_t tmp = n;
    // printf("ok!!\n");
    for (int i = 0; i < len_ratios; i++)
    { 
        // printf("第%d层比例%0.4f， 第%d层个数%d\n",i,ht_hbs_ratios[i],i ,nb_nbors_per_level[i]);
        // 按热点等级分配热点新增邻居个数（若三层：3*m ,2*m ,1*m ,m为0层邻居数）
        tmp += n*ht_hbs_ratios[i]*nb_nbors_per_level[i];
        ratios.push_back(ht_hbs_ratios[i]);
        nb_reverse_neighbors.push_back(nb_nbors_per_level[i]);
    }
    // 防止下取整
    tmp += 5;

    // 分配空间（如果并行需要修改 ，levels中分配每个点的添加位置）
    hnsw.prepare_level_tab4(n,tmp);
    ntotal = tmp;
    // 统计热点,将多层次热点及与他们相连的邻居放到hot_hubs中
    std::vector<std::unordered_map<idx_t,std::vector<idx_t>>> hot_hubs(len_ratios);
    std::vector<std::unordered_map<idx_t,std::vector<idx_t>>> sub_hot_hubs(len_ratios);
    ihf1.hnsw.find_hot_hubs_enhence_subIndex(sub_hot_hubs,ihf1.ntotal,ratios,find_hubs_mode,
            find_neighbors_mode);
    ihf1.aod_level();
    avgOutdegres(sub_hot_hubs);
    subTranferToAll(hot_hubs,sub_hot_hubs,idxScript1);
    vctReset(sub_hot_hubs);
    ihf2.hnsw.find_hot_hubs_enhence_subIndex(sub_hot_hubs,ihf2.ntotal,ratios,find_hubs_mode,
            find_neighbors_mode);
    ihf2.aod_level();
    avgOutdegres(sub_hot_hubs);
    subTranferToAll(hot_hubs,sub_hot_hubs,idxScript2);
    vctReset(sub_hot_hubs);
    ihf3.hnsw.find_hot_hubs_enhence_subIndex(sub_hot_hubs,ihf3.ntotal,ratios,find_hubs_mode,
            find_neighbors_mode);
    ihf3.aod_level();
    avgOutdegres(sub_hot_hubs);
    subTranferToAll(hot_hubs,sub_hot_hubs,idxScript3);
    vctReset(sub_hot_hubs);
    ihf4.hnsw.find_hot_hubs_enhence_subIndex(sub_hot_hubs,ihf4.ntotal,ratios,find_hubs_mode,
            find_neighbors_mode);
    ihf4.aod_level();
    avgOutdegres(sub_hot_hubs);
    subTranferToAll(hot_hubs,sub_hot_hubs,idxScript4);
    vctReset(sub_hot_hubs);
    ihf5.hnsw.find_hot_hubs_enhence_subIndex(sub_hot_hubs,ihf5.ntotal,ratios,find_hubs_mode,
            find_neighbors_mode);
    ihf5.aod_level();
    avgOutdegres(sub_hot_hubs);
    subTranferToAll(hot_hubs,sub_hot_hubs,idxScript5);
    vctReset(sub_hot_hubs);
    printf("ok123!!!\n");

    // 距离计算器
    DistanceComputer* dis = get_distance_computer(storage);
    ScopeDeleter1<DistanceComputer> del(dis);

    hnsw.add_links_to_hubs(hot_hubs,n);
    hnsw.add_new_reverse_link_end_enhence(hot_hubs,n,*dis,nb_reverse_neighbors);

    aod_level();
    avgOutdegres(hot_hubs);
}


// cur 表示当前子图开始位置，len表示子图结点个数， x乱序数据
void get_hubs_and_reverse_links(int cur, int len, const idx_t* x,
    std::vector<std::unordered_map<idx_t,std::vector<idx_t>>>& hot_hubs,
    float ratios ){
    // 选择固定比例的热点，每个热点选择20000
    int cnts = (int)len*ratios; // 热点数量
    idx_t bg = cur + cnts; // 热点反向边开始位置
    idx_t ed = bg;
    auto& hot_hub = hot_hubs[0];
    // printf("cnt == %d,cur == %d\t,len == %d\t,\n",cnts,cur,len);
    for (int i = cur; i < ed; ++i) // 为每个热点找出热点邻居，邻居之间不重叠
    {   // printf("x[i] = %ld\n",x[i]);
        for (int j = 0; j < 200; ++j)
        {
            hot_hub[x[i]].push_back(x[bg+j]);
        }
        bg += 200;
    }
}


// 子图中随机选取热点，随机选取热点邻居
// cls 类别数目
void IndexHNSW::complete_random_hot_hubs_enhence(idx_t n,const idx_t* x,int len_ratios,
        const float* ht_hbs_ratios,const int* nb_nbors_per_level,int cls){

    // 节点数量+热点数量 , 放入vector中存储
    aod_level();
    double t0  = getmillisecs();
    std::vector<float> ratios;
    std::vector<int> nb_reverse_neighbors ;
    idx_t tmp = n;
    for (int i = 0; i < len_ratios; i++)
    { 
        tmp += n*ht_hbs_ratios[i]*nb_nbors_per_level[i];
        ratios.push_back(ht_hbs_ratios[i]);
        nb_reverse_neighbors.push_back(nb_nbors_per_level[i]);
    }
    // 防止下取整
    tmp += 5;

    // 分配空间（如果并行需要修改 ，levels中分配每个点的添加位置）
    hnsw.prepare_level_tab4(n,tmp);
    ntotal = tmp;
    // 统计热点,将多层次热点及与他们相连的邻居放到hot_hubs中
    // std::vector<std::unordered_map<idx_t,std::vector<idx_t>>> hot_hubs(len_ratios); // 变为全局变量;
    // 寻找热点反向边
    hnsw.hot_hubs.resize(len_ratios);

    idx_t len = n/cls; // 数据集划分为cls类，每类有len个数据点
    idx_t cur = 0;
    for (int i = 0; i < cls; ++i)
    {   printf("%d\n",i);
        get_hubs_and_reverse_links(cur,len,x,hnsw.hot_hubs,ratios[0]);
        cur += len;
    }
    
    // printf("ok123!!!\n"); 

    DistanceComputer* dis = get_distance_computer(storage); // 距离计算器
    ScopeDeleter1<DistanceComputer> del(dis);

    hnsw.add_links_to_hubs(hnsw.hot_hubs,n); // 将反向边连向热点
    hnsw.add_new_reverse_link_end_enhence(hnsw.hot_hubs,n,*dis,nb_reverse_neighbors);

    aod_level();
    avgOutdegres(hnsw.hot_hubs);
}





void IndexHNSW::reset()
{
    hnsw.reset();
    storage->reset();
    ntotal = 0;
}

void IndexHNSW::reconstruct (idx_t key, float* recons) const
{
    storage->reconstruct(key, recons);
}

void IndexHNSW::shrink_level_0_neighbors(int new_size)
{
#pragma omp parallel
    {
        DistanceComputer *dis = get_distance_computer();
        ScopeDeleter1<DistanceComputer> del(dis);

#pragma omp for
        for (idx_t i = 0; i < ntotal; i++) {

            size_t begin, end;
            hnsw.neighbor_range(i, 0, &begin, &end);

            std::priority_queue<NodeDistFarther> initial_list;

            for (size_t j = begin; j < end; j++) {
                int v1 = hnsw.neighbors[j];
                if (v1 < 0) break;
                initial_list.emplace(dis->symmetric_dis(i, v1), v1);

                // initial_list.emplace(qdis(v1), v1);
            }

            std::vector<NodeDistFarther> shrunk_list;
            // HNSW::shrink_neighbor_list(*dis, initial_list,
            //                            shrunk_list, new_size,100000);

            for (size_t j = begin; j < end; j++) {
                if (j - begin < shrunk_list.size())
                    hnsw.neighbors[j] = shrunk_list[j - begin].id;
                else
                    hnsw.neighbors[j] = -1;
            }
        }
    }

}

void IndexHNSW::search_level_0(
    idx_t n, const float *x, idx_t k,
    const storage_idx_t *nearest, const float *nearest_d,
    float *distances, idx_t *labels, int nprobe,
    int search_type) const
{

    storage_idx_t ntotal = hnsw.levels.size();
// #pragma omp parallel
    {
        DistanceComputer *qdis = get_distance_computer();
        ScopeDeleter1<DistanceComputer> del(qdis);

        VisitedTable vt (ntotal);

// #pragma omp for
        for(idx_t i = 0; i < n; i++) {
            idx_t * idxi = labels + i * k;
            float * simi = distances + i * k;

            qdis->set_query(x + i * d);
            maxheap_heapify (k, simi, idxi);

            if (search_type == 1) {

                int nres = 0;

                for(int j = 0; j < nprobe; j++) {
                    storage_idx_t cj = nearest[i * nprobe + j];

                    if (cj < 0) break;

                    if (vt.get(cj)) continue;

                    int candidates_size = std::max(hnsw.efSearch, int(k));
                    MinimaxHeap candidates(candidates_size);

                    candidates.push(cj, nearest_d[i * nprobe + j]);

                    nres = hnsw.search_from_candidates(
                      *qdis, k, idxi, simi,
                      candidates, vt, 0, nres
                    );
                }
            } else if (search_type == 2) {

                int candidates_size = std::max(hnsw.efSearch, int(k));
                candidates_size = std::max(candidates_size, nprobe);

                MinimaxHeap candidates(candidates_size);
                for(int j = 0; j < nprobe; j++) {
                    storage_idx_t cj = nearest[i * nprobe + j];

                    if (cj < 0) break;
                    candidates.push(cj, nearest_d[i * nprobe + j]);
                }
                hnsw.search_from_candidates(
                  *qdis, k, idxi, simi,
                  candidates, vt, 0
                );

            }
            vt.advance();

            maxheap_reorder (k, simi, idxi);

        }
    }


}


void IndexHNSW::static_in_degree_by_direct(idx_t n){
    DistanceComputer *qdis = get_distance_computer();
    hnsw.static_in_degree_by_direct(n,*qdis);

}


void IndexHNSW::createKnn(idx_t n,int k){
    for (idx_t i = 0; i < ntotal; i++) {
        if((int)i%5000==0){
            printf("%d\n",i);
        }
        DistanceComputer *qdis = get_distance_computer();
        float vec[d];
        storage->reconstruct(i, vec);
        qdis->set_query(vec);

        std::priority_queue<NodeDistCloser> initial_list;

        hnsw.getKnn(*qdis,i,ntotal,k,initial_list);
        std::priority_queue<NodeDistFarther> initial_list_Father;

        std::vector<NodeDistFarther> shrunk_list;
        while(!initial_list.empty()){
            initial_list_Father.emplace(initial_list.top().d,initial_list.top().id);
            //printf("id:%d\tdis:%f\n",initial_list.top().id,initial_list.top().d);
            initial_list.pop();
        }
        while(!initial_list_Father.empty()){
            shrunk_list.push_back(initial_list_Father.top());
            initial_list_Father.pop();
        }
        // printf("\n");
        // HNSW::shrink_neighbor_list(*qdis, initial_list_Father, shrunk_list, k);

        size_t begin, end;
        hnsw.neighbor_range(i, 0, &begin, &end);

        for (size_t j = begin; j < end; j++) {
            if (j - begin < shrunk_list.size())
                hnsw.neighbors[j] = shrunk_list[j - begin].id;
            else
                hnsw.neighbors[j] = -1;
        }

            
    }
    printf("构建KNN完毕\n");
}




void IndexHNSW::init_level_0_from_knngraph(
       int k, const float *D, const idx_t *I)
{
    int dest_size = hnsw.nb_neighbors (0);

#pragma omp parallel for
    for (idx_t i = 0; i < ntotal; i++) {
        DistanceComputer *qdis = get_distance_computer();
        float vec[d];
        storage->reconstruct(i, vec);
        qdis->set_query(vec);

        std::priority_queue<NodeDistFarther> initial_list;

        for (size_t j = 0; j < k; j++) {
            int v1 = I[i * k + j];
            if (v1 == i) continue;
            if (v1 < 0) break;
            initial_list.emplace(D[i * k + j], v1);
        }

        std::vector<NodeDistFarther> shrunk_list;
        // HNSW::shrink_neighbor_list(*qdis, initial_list, shrunk_list, dest_size,100000);

        size_t begin, end;
        hnsw.neighbor_range(i, 0, &begin, &end);

        for (size_t j = begin; j < end; j++) {
            if (j - begin < shrunk_list.size())
                hnsw.neighbors[j] = shrunk_list[j - begin].id;
            else
                hnsw.neighbors[j] = -1;
        }
    }
}



void IndexHNSW::init_level_0_from_entry_points(
          int n, const storage_idx_t *points,
          const storage_idx_t *nearests)
{

    std::vector<omp_lock_t> locks(ntotal);
    for(int i = 0; i < ntotal; i++)
        omp_init_lock(&locks[i]);

#pragma omp parallel
    {
        VisitedTable vt (ntotal);

        DistanceComputer *dis = get_distance_computer();
        ScopeDeleter1<DistanceComputer> del(dis);
        float vec[storage->d];

#pragma omp  for schedule(dynamic)
        for (int i = 0; i < n; i++) {
            storage_idx_t pt_id = points[i];
            storage_idx_t nearest = nearests[i];
            storage->reconstruct (pt_id, vec);
            dis->set_query (vec);

            

            hnsw.add_links_starting_from(*dis, pt_id,
                                         nearest, (*dis)(nearest),
                                         0, locks.data(), vt,0);

            if (verbose && i % 10000 == 0) {
                printf("  %d / %d\r", i, n);
                fflush(stdout);
            }
        }
    }
    if (verbose) {
        printf("\n");
    }

    for(int i = 0; i < ntotal; i++)
        omp_destroy_lock(&locks[i]);
}

void IndexHNSW::reorder_links()
{
    int M = hnsw.nb_neighbors(0);

#pragma omp parallel
    {
        std::vector<float> distances (M);
        std::vector<size_t> order (M);
        std::vector<storage_idx_t> tmp (M);
        DistanceComputer *dis = get_distance_computer();
        ScopeDeleter1<DistanceComputer> del(dis);

#pragma omp for
        for(storage_idx_t i = 0; i < ntotal; i++) {

            size_t begin, end;
            hnsw.neighbor_range(i, 0, &begin, &end);

            for (size_t j = begin; j < end; j++) {
                storage_idx_t nj = hnsw.neighbors[j];
                if (nj < 0) {
                    end = j;
                    break;
                }
                distances[j - begin] = dis->symmetric_dis(i, nj);
                tmp [j - begin] = nj;
            }

            fvec_argsort (end - begin, distances.data(), order.data());
            for (size_t j = begin; j < end; j++) {
                hnsw.neighbors[j] = tmp[order[j - begin]];
            }
        }

    }
}


void IndexHNSW::link_singletons()
{
    printf("search for singletons\n");

    std::vector<bool> seen(ntotal);

    for (size_t i = 0; i < ntotal; i++) {
        size_t begin, end;
        hnsw.neighbor_range(i, 0, &begin, &end);
        for (size_t j = begin; j < end; j++) {
            storage_idx_t ni = hnsw.neighbors[j];
            if (ni >= 0) seen[ni] = true;
        }
    }

    int n_sing = 0, n_sing_l1 = 0;
    std::vector<storage_idx_t> singletons;
    for (storage_idx_t i = 0; i < ntotal; i++) {
        if (!seen[i]) {
            singletons.push_back(i);
            n_sing++;
            if (hnsw.levels[i] > 1)
                n_sing_l1++;
        }
    }

    printf("  Found %d / %ld singletons (%d appear in a level above)\n",
           n_sing, ntotal, n_sing_l1);

    std::vector<float>recons(singletons.size() * d);
    for (int i = 0; i < singletons.size(); i++) {

        FAISS_ASSERT(!"not implemented");

    }


}


namespace {


// storage that explicitly reconstructs vectors before computing distances
struct GenericDistanceComputer: DistanceComputer {

    const Index & storage;
    std::vector<float> buf;
    const float *q;

    GenericDistanceComputer(const Index & storage): storage(storage)
    {
        d = storage.d;
        buf.resize(d * 2);
    }

    float operator () (storage_idx_t i) override
    {   
        // IndexFlat中的方法，memcpy
        storage.reconstruct(i, buf.data());
        return fvec_L2sqr(q, buf.data(), d);
    }

    float symmetric_dis(storage_idx_t i, storage_idx_t j) override
    {
        storage.reconstruct(i, buf.data());
        storage.reconstruct(j, buf.data() + d);
        return fvec_L2sqr(buf.data() + d, buf.data(), d);
    }

    void set_query(const float *x, storage_idx_t idx = -1) override {
        q = x;
    }

};


}  // namespace

DistanceComputer * IndexHNSW::get_distance_computer () const
{
    return new GenericDistanceComputer (*storage);
}

// 自己添加
DistanceComputer * IndexHNSW::get_distance_computer (Index* storage) const
{
    return new GenericDistanceComputer (*storage);
}


/**************************************************************
 * ReconstructFromNeighbors implementation
 **************************************************************/


ReconstructFromNeighbors::ReconstructFromNeighbors(
             const IndexHNSW & index, size_t k, size_t nsq):
    index(index), k(k), nsq(nsq) {
    M = index.hnsw.nb_neighbors(0);
    FAISS_ASSERT(k <= 256);
    code_size = k == 1 ? 0 : nsq;
    ntotal = 0;
    d = index.d;
    FAISS_ASSERT(d % nsq == 0);
    dsub = d / nsq;
    k_reorder = -1;
}

void ReconstructFromNeighbors::reconstruct(storage_idx_t i, float *x, float *tmp) const
{


    const HNSW & hnsw = index.hnsw;
    size_t begin, end;
    hnsw.neighbor_range(i, 0, &begin, &end);

    if (k == 1 || nsq == 1) {
        const float * beta;
        if (k == 1) {
            beta = codebook.data();
        } else {
            int idx = codes[i];
            beta = codebook.data() + idx * (M + 1);
        }

        float w0 = beta[0]; // weight of image itself
        index.storage->reconstruct(i, tmp);

        for (int l = 0; l < d; l++)
            x[l] = w0 * tmp[l];

        for (size_t j = begin; j < end; j++) {

            storage_idx_t ji = hnsw.neighbors[j];
            if (ji < 0) ji = i;
            float w = beta[j - begin + 1];
            index.storage->reconstruct(ji, tmp);
            for (int l = 0; l < d; l++)
                x[l] += w * tmp[l];
        }
    } else if (nsq == 2) {
        int idx0 = codes[2 * i];
        int idx1 = codes[2 * i + 1];

        const float *beta0 = codebook.data() +  idx0 * (M + 1);
        const float *beta1 = codebook.data() + (idx1 + k) * (M + 1);

        index.storage->reconstruct(i, tmp);

        float w0;

        w0 = beta0[0];
        for (int l = 0; l < dsub; l++)
            x[l] = w0 * tmp[l];

        w0 = beta1[0];
        for (int l = dsub; l < d; l++)
            x[l] = w0 * tmp[l];

        for (size_t j = begin; j < end; j++) {
            storage_idx_t ji = hnsw.neighbors[j];
            if (ji < 0) ji = i;
            index.storage->reconstruct(ji, tmp);
            float w;
            w = beta0[j - begin + 1];
            for (int l = 0; l < dsub; l++)
                x[l] += w * tmp[l];

            w = beta1[j - begin + 1];
            for (int l = dsub; l < d; l++)
                x[l] += w * tmp[l];
        }
    } else {
        const float *betas[nsq];
        {
            const float *b = codebook.data();
            const uint8_t *c = &codes[i * code_size];
            for (int sq = 0; sq < nsq; sq++) {
                betas[sq] = b + (*c++) * (M + 1);
                b += (M + 1) * k;
            }
        }

        index.storage->reconstruct(i, tmp);
        {
            int d0 = 0;
            for (int sq = 0; sq < nsq; sq++) {
                float w = *(betas[sq]++);
                int d1 = d0 + dsub;
                for (int l = d0; l < d1; l++) {
                    x[l] = w * tmp[l];
                }
                d0 = d1;
            }
        }

        for (size_t j = begin; j < end; j++) {
            storage_idx_t ji = hnsw.neighbors[j];
            if (ji < 0) ji = i;

            index.storage->reconstruct(ji, tmp);
            int d0 = 0;
            for (int sq = 0; sq < nsq; sq++) {
                float w = *(betas[sq]++);
                int d1 = d0 + dsub;
                for (int l = d0; l < d1; l++) {
                    x[l] += w * tmp[l];
                }
                d0 = d1;
            }
        }
    }
}

void ReconstructFromNeighbors::reconstruct_n(storage_idx_t n0,
                                             storage_idx_t ni,
                                             float *x) const
{
#pragma omp parallel
    {
        std::vector<float> tmp(index.d);
#pragma omp for
        for (storage_idx_t i = 0; i < ni; i++) {
            reconstruct(n0 + i, x + i * index.d, tmp.data());
        }
    }
}

size_t ReconstructFromNeighbors::compute_distances(
    size_t n, const idx_t *shortlist,
    const float *query, float *distances) const
{
    std::vector<float> tmp(2 * index.d);
    size_t ncomp = 0;
    for (int i = 0; i < n; i++) {
        if (shortlist[i] < 0) break;
        reconstruct(shortlist[i], tmp.data(), tmp.data() + index.d);
        distances[i] = fvec_L2sqr(query, tmp.data(), index.d);
        ncomp++;
    }
    return ncomp;
}

void ReconstructFromNeighbors::get_neighbor_table(storage_idx_t i, float *tmp1) const
{
    const HNSW & hnsw = index.hnsw;
    size_t begin, end;
    hnsw.neighbor_range(i, 0, &begin, &end);
    size_t d = index.d;

    index.storage->reconstruct(i, tmp1);

    for (size_t j = begin; j < end; j++) {
        storage_idx_t ji = hnsw.neighbors[j];
        if (ji < 0) ji = i;
        index.storage->reconstruct(ji, tmp1 + (j - begin + 1) * d);
    }

}


/// called by add_codes
void ReconstructFromNeighbors::estimate_code(
       const float *x, storage_idx_t i, uint8_t *code) const
{

    // fill in tmp table with the neighbor values
    float *tmp1 = new float[d * (M + 1) + (d * k)];
    float *tmp2 = tmp1 + d * (M + 1);
    ScopeDeleter<float> del(tmp1);

    // collect coordinates of base
    get_neighbor_table (i, tmp1);

    for (size_t sq = 0; sq < nsq; sq++) {
        int d0 = sq * dsub;

        {
            FINTEGER ki = k, di = d, m1 = M + 1;
            FINTEGER dsubi = dsub;
            float zero = 0, one = 1;

            sgemm_ ("N", "N", &dsubi, &ki, &m1, &one,
                    tmp1 + d0, &di,
                    codebook.data() + sq * (m1 * k), &m1,
                    &zero, tmp2, &dsubi);
        }

        float min = HUGE_VAL;
        int argmin = -1;
        for (size_t j = 0; j < k; j++) {
            float dis = fvec_L2sqr(x + d0, tmp2 + j * dsub, dsub);
            if (dis < min) {
                min = dis;
                argmin = j;
            }
        }
        code[sq] = argmin;
    }

}

void ReconstructFromNeighbors::add_codes(size_t n, const float *x)
{
    if (k == 1) { // nothing to encode
        ntotal += n;
        return;
    }
    codes.resize(codes.size() + code_size * n);
#pragma omp parallel for
    for (int i = 0; i < n; i++) {
        estimate_code(x + i * index.d, ntotal + i,
                      codes.data() + (ntotal + i) * code_size);
    }
    ntotal += n;
    FAISS_ASSERT (codes.size() == ntotal * code_size);
}


/**************************************************************
 * IndexHNSWFlat implementation
 **************************************************************/
//inner 内积距离计算器

namespace {


struct FlatInnerPrductDis: DistanceComputer {
    Index::idx_t nb;
    const float *q;
    const float *b;
    size_t ndis;

    float operator () (storage_idx_t i) override
    {
        ndis++;
        return -(fvec_inner_product(q, b + i * d, d));
    }

    float symmetric_dis(storage_idx_t i, storage_idx_t j) override
    {
        return -(fvec_inner_product(b + j * d, b + i * d, d));
    }


    FlatInnerPrductDis(const IndexFlatIP & storage, const float *q = nullptr):
        q(q)
    {
        nb = storage.ntotal;
        d = storage.d;
        b = storage.xb.data();
        ndis = 0;
    }

    void set_query(const float *x, storage_idx_t idx = -1) override {
        q = x;
    }

    virtual ~FlatInnerPrductDis () {
#pragma omp critical
        {
            hnsw_stats.ndis += ndis;
        }
    }
};


}  // namespace

namespace {

struct FlatL2Dis: DistanceComputer {
    Index::idx_t nb;
    const float *q;
    const float *b;
    size_t ndis;

    float operator () (storage_idx_t i) override
    {
        ndis++;
        return (fvec_L2sqr(q, b + i * d, d));
    }

    float symmetric_dis(storage_idx_t i, storage_idx_t j) override
    {
        return (fvec_L2sqr(b + j * d, b + i * d, d));
    }


    FlatL2Dis(const IndexFlatL2 & storage, const float *q = nullptr):
        q(q)
    {
        nb = storage.ntotal;
        d = storage.d;
        b = storage.xb.data();
        ndis = 0;
    }

    void set_query(const float *x, storage_idx_t idx = -1) override {
        q = x;
    }

    virtual ~FlatL2Dis () {
#pragma omp critical
        {
            hnsw_stats.ndis += ndis;
        }
    }
};


}  // namespace

namespace {


struct NICDMDis: DistanceComputer {
    Index::idx_t nb; // 100M
    const float *q;
    storage_idx_t idx_q;
    const float *b; // ->8 8 8 8 8 8, q-b
    size_t ndis;

    float operator () (storage_idx_t i) override // 和query节点比
    {
        ndis++;
        // printf("operator, i = %d\n", i);
        return fvec_nicdm(idx_q, i);
    }

    float symmetric_dis(storage_idx_t i, storage_idx_t j) override // 直接算两个输入之间的
    {
        // printf("symmetric_dis, i = %d, j = %d\n", i, j);
        return fvec_nicdm(i, j);
    }

    float fvec_nicdm(storage_idx_t i, storage_idx_t j) {
        float* gt = AVGDIS;
        float l2sqr = fvec_L2sqr(b + i * d, b + j * d, d);
        return ~idx_q ? l2sqr / powf(sqrtf(gt[i] * gt[j]), alpha) : l2sqr;
    }

    NICDMDis(const IndexFlatL2 & storage, const float *q = nullptr):
        q(q)
    {
        nb = storage.ntotal;
        d = storage.d;
        b = storage.xb.data();
        ndis = 0;
    }

    void set_query(const float *x, storage_idx_t idx = -1) override {
        q = x;
        idx_q = idx;
    }

    virtual ~NICDMDis () {
#pragma omp critical
        {
            hnsw_stats.ndis += ndis;
        }
    }
};


}  // namespace

IndexHNSWFlat::IndexHNSWFlat()
{
    is_trained = true;
}


IndexHNSWFlat::IndexHNSWFlat(int d, int M):
    IndexHNSW(new IndexFlatL2(d), M)
{
    own_fields = true;
    is_trained = true;
}
// IndexHNSWFlat::IndexHNSWFlat(int d, int M):
//     IndexHNSW(new IndexFlatIP(d), M)
// {
//     // printf("内积距离计算器\n");
//     own_fields = true;
//     is_trained = true;
// }


//计算距离
DistanceComputer * IndexHNSWFlat::get_distance_computer () const
{
    // return new FlatInnerPrductDis (*dynamic_cast<IndexFlatIP*> (storage));

    // std::cout << dis_method << "距离计算" << std::endl;
    if (dis_method == "NICDM") return new NICDMDis(*dynamic_cast<IndexFlatL2*> (storage));
    return new FlatL2Dis (*dynamic_cast<IndexFlatL2*> (storage));
}




/**************************************************************
 * IndexHNSWPQ implementation
 **************************************************************/


namespace {


struct PQDis: DistanceComputer {
    Index::idx_t nb;
    const uint8_t *codes;
    size_t code_size;
    const ProductQuantizer & pq;
    const float *sdc;
    std::vector<float> precomputed_table;
    size_t ndis;

    float operator () (storage_idx_t i) override
    {
        const uint8_t *code = codes + i * code_size;
        const float *dt = precomputed_table.data();
        float accu = 0;
        for (int j = 0; j < pq.M; j++) {
            accu += dt[*code++];
            dt += 256;
        }
        ndis++;
        return accu;
    }

    float symmetric_dis(storage_idx_t i, storage_idx_t j) override
    {
        const float * sdci = sdc;
        float accu = 0;
        const uint8_t *codei = codes + i * code_size;
        const uint8_t *codej = codes + j * code_size;

        for (int l = 0; l < pq.M; l++) {
            accu += sdci[(*codei++) + (*codej++) * 256];
            sdci += 256 * 256;
        }
        return accu;
    }

    PQDis(const IndexPQ& storage, const float* /*q*/ = nullptr)
        : pq(storage.pq) {
      precomputed_table.resize(pq.M * pq.ksub);
      nb = storage.ntotal;
      d = storage.d;
      codes = storage.codes.data();
      code_size = pq.code_size;
      FAISS_ASSERT(pq.ksub == 256);
      FAISS_ASSERT(pq.sdc_table.size() == pq.ksub * pq.ksub * pq.M);
      sdc = pq.sdc_table.data();
      ndis = 0;
    }

    void set_query(const float *x, storage_idx_t idx = -1) override {
        pq.compute_distance_table(x, precomputed_table.data());
    }

    virtual ~PQDis () {
#pragma omp critical
        {
            hnsw_stats.ndis += ndis;
        }
    }
};


}  // namespace


namespace {


struct PQNICDMDis: DistanceComputer {
    Index::idx_t nb;
    const uint8_t *codes;
    size_t code_size;
    const ProductQuantizer & pq;
    const float *sdc;
    std::vector<float> precomputed_table;
    size_t ndis;
    storage_idx_t idx_q;

    float operator () (storage_idx_t i) override
    {
        float* gt = AVGDIS;
        const uint8_t *code = codes + i * code_size;
        const float *dt = precomputed_table.data();
        float accu = 0;
        for (int j = 0; j < pq.M; j++) {
            accu += dt[*code++];
            dt += 256;
        }
        ndis++;
        return ~idx_q ? accu / powf(sqrtf(gt[idx_q] * gt[i]), alpha) : accu;
    }

    float symmetric_dis(storage_idx_t i, storage_idx_t j) override
    {
        float* gt = AVGDIS;
        const float * sdci = sdc;
        float accu = 0;
        const uint8_t *codei = codes + i * code_size;
        const uint8_t *codej = codes + j * code_size;

        for (int l = 0; l < pq.M; l++) {
            accu += sdci[(*codei++) + (*codej++) * 256];
            sdci += 256 * 256;
        }
        return ~idx_q ? accu / powf(sqrtf(gt[i] * gt[j]), alpha) : accu;
    }

    PQNICDMDis(const IndexPQ& storage, const float* /*q*/ = nullptr)
        : pq(storage.pq) {
      precomputed_table.resize(pq.M * pq.ksub);
      nb = storage.ntotal;
      d = storage.d;
      codes = storage.codes.data();
      code_size = pq.code_size;
      FAISS_ASSERT(pq.ksub == 256);
      FAISS_ASSERT(pq.sdc_table.size() == pq.ksub * pq.ksub * pq.M);
      sdc = pq.sdc_table.data();
      ndis = 0;
    }

    void set_query(const float *x, storage_idx_t idx = -1) override {
        pq.compute_distance_table(x, precomputed_table.data());
        idx_q = idx;
    }

    virtual ~PQNICDMDis () {
#pragma omp critical
        {
            hnsw_stats.ndis += ndis;
        }
    }
};


}  // namespace


IndexHNSWPQ::IndexHNSWPQ() {}

IndexHNSWPQ::IndexHNSWPQ(int d, int pq_m, int M):
    IndexHNSW(new IndexPQ(d, pq_m, 8), M)
{
    own_fields = true;
    is_trained = false;
}

void IndexHNSWPQ::train(idx_t n, const float* x)
{
    IndexHNSW::train (n, x);
    (dynamic_cast<IndexPQ*> (storage))->pq.compute_sdc_table();
}



DistanceComputer * IndexHNSWPQ::get_distance_computer () const
{
    if (dis_method == "NICDM") return new PQNICDMDis (*dynamic_cast<IndexPQ*> (storage));
    return new PQDis (*dynamic_cast<IndexPQ*> (storage));
}


/**************************************************************
 * IndexHNSWSQ implementation
 **************************************************************/


namespace {


struct SQDis: DistanceComputer {
    Index::idx_t nb;
    const uint8_t *codes;
    size_t code_size;
    const ScalarQuantizer & sq;
    const float *q;
    ScalarQuantizer::DistanceComputer * dc;

    float operator () (storage_idx_t i) override
    {
        const uint8_t *code = codes + i * code_size;

        return dc->compute_distance (q, code);
    }

    float symmetric_dis(storage_idx_t i, storage_idx_t j) override
    {
        const uint8_t *codei = codes + i * code_size;
        const uint8_t *codej = codes + j * code_size;
        return dc->compute_code_distance (codei, codej);
    }

    SQDis(const IndexScalarQuantizer& storage, const float* /*q*/ = nullptr)
        : sq(storage.sq) {
      nb = storage.ntotal;
      d = storage.d;
      codes = storage.codes.data();
      code_size = sq.code_size;
      dc = sq.get_distance_computer();
    }

    void set_query(const float *x, storage_idx_t idx = -1) override {
        q = x;
    }

    virtual ~SQDis () {
        delete dc;
    }
};


}  // namespace


IndexHNSWSQ::IndexHNSWSQ(int d, ScalarQuantizer::QuantizerType qtype, int M):
    IndexHNSW (new IndexScalarQuantizer (d, qtype), M)
{
    own_fields = true;
}

IndexHNSWSQ::IndexHNSWSQ() {}

DistanceComputer * IndexHNSWSQ::get_distance_computer () const
{
    return new SQDis (*dynamic_cast<IndexScalarQuantizer*> (storage));
}




/**************************************************************
 * IndexHNSW2Level implementation
 **************************************************************/



IndexHNSW2Level::IndexHNSW2Level(Index *quantizer, size_t nlist, int m_pq, int M):
    IndexHNSW (new Index2Layer (quantizer, nlist, m_pq), M)
{
    own_fields = true;
    is_trained = false;
}

IndexHNSW2Level::IndexHNSW2Level() {}


namespace {


struct Distance2Level: DistanceComputer {

    const Index2Layer & storage;
    std::vector<float> buf;
    const float *q;

    const float *pq_l1_tab, *pq_l2_tab;

    Distance2Level(const Index2Layer & storage): storage(storage)
    {
        d = storage.d;
        FAISS_ASSERT(storage.pq.dsub == 4);
        pq_l2_tab = storage.pq.centroids.data();
        buf.resize(2 * d);
    }

    float symmetric_dis(storage_idx_t i, storage_idx_t j) override
    {
        storage.reconstruct(i, buf.data());
        storage.reconstruct(j, buf.data() + d);
        return fvec_L2sqr(buf.data() + d, buf.data(), d);
    }

    void set_query(const float *x, storage_idx_t idx = -1) override {
        q = x;
    }
};


// well optimized for xNN+PQNN
struct DistanceXPQ4: Distance2Level {

    int M, k;

    DistanceXPQ4(const Index2Layer & storage):
        Distance2Level (storage)
    {
        const IndexFlat *quantizer =
            dynamic_cast<IndexFlat*> (storage.q1.quantizer);

        FAISS_ASSERT(quantizer);
        M = storage.pq.M;
        pq_l1_tab = quantizer->xb.data();
    }

    float operator () (storage_idx_t i) override
    {
#ifdef __SSE__
        const uint8_t *code = storage.codes.data() + i * storage.code_size;
        long key = 0;
        memcpy (&key, code, storage.code_size_1);
        code += storage.code_size_1;

        // walking pointers
        const float *qa = q;
        const __m128 *l1_t = (const __m128 *)(pq_l1_tab + d * key);
        const __m128 *pq_l2_t = (const __m128 *)pq_l2_tab;
        __m128 accu = _mm_setzero_ps();

        for (int m = 0; m < M; m++) {
            __m128 qi = _mm_loadu_ps(qa);
            __m128 recons = l1_t[m] + pq_l2_t[*code++];
            __m128 diff = qi - recons;
            accu += diff * diff;
            pq_l2_t += 256;
            qa += 4;
        }

        accu = _mm_hadd_ps (accu, accu);
        accu = _mm_hadd_ps (accu, accu);
        return  _mm_cvtss_f32 (accu);
#else
        FAISS_THROW_MSG("not implemented for non-x64 platforms");
#endif
    }

};

// well optimized for 2xNN+PQNN
struct Distance2xXPQ4: Distance2Level {

    int M_2, mi_nbits;

    Distance2xXPQ4(const Index2Layer & storage):
        Distance2Level (storage)
    {
        const MultiIndexQuantizer *mi =
            dynamic_cast<MultiIndexQuantizer*> (storage.q1.quantizer);

        FAISS_ASSERT(mi);
        FAISS_ASSERT(storage.pq.M % 2 == 0);
        M_2 = storage.pq.M / 2;
        mi_nbits = mi->pq.nbits;
        pq_l1_tab = mi->pq.centroids.data();
    }

    float operator () (storage_idx_t i) override
    {
        const uint8_t *code = storage.codes.data() + i * storage.code_size;
        long key01 = 0;
        memcpy (&key01, code, storage.code_size_1);
        code += storage.code_size_1;
#ifdef __SSE__

        // walking pointers
        const float *qa = q;
        const __m128 *pq_l1_t = (const __m128 *)pq_l1_tab;
        const __m128 *pq_l2_t = (const __m128 *)pq_l2_tab;
        __m128 accu = _mm_setzero_ps();

        for (int mi_m = 0; mi_m < 2; mi_m++) {
            long l1_idx = key01 & ((1L << mi_nbits) - 1);
            const __m128 * pq_l1 = pq_l1_t + M_2 * l1_idx;

            for (int m = 0; m < M_2; m++) {
                __m128 qi = _mm_loadu_ps(qa);
                __m128 recons = pq_l1[m] + pq_l2_t[*code++];
                __m128 diff = qi - recons;
                accu += diff * diff;
                pq_l2_t += 256;
                qa += 4;
            }
            pq_l1_t += M_2 << mi_nbits;
            key01 >>= mi_nbits;
        }
        accu = _mm_hadd_ps (accu, accu);
        accu = _mm_hadd_ps (accu, accu);
        return  _mm_cvtss_f32 (accu);
#else
        FAISS_THROW_MSG("not implemented for non-x64 platforms");
#endif
    }

};


}  // namespace


DistanceComputer * IndexHNSW2Level::get_distance_computer () const
{
    const Index2Layer *storage2l =
        dynamic_cast<Index2Layer*>(storage);

    if (storage2l) {
#ifdef __SSE__

        const MultiIndexQuantizer *mi =
            dynamic_cast<MultiIndexQuantizer*> (storage2l->q1.quantizer);

        if (mi && storage2l->pq.M % 2 == 0 && storage2l->pq.dsub == 4) {
            return new Distance2xXPQ4(*storage2l);
        }

        const IndexFlat *fl =
            dynamic_cast<IndexFlat*> (storage2l->q1.quantizer);

        if (fl && storage2l->pq.dsub == 4) {
            return new DistanceXPQ4(*storage2l);
        }
#endif
    }

    // IVFPQ and cases not handled above
    return new GenericDistanceComputer (*storage);

}


namespace {


// same as search_from_candidates but uses v
// visno -> is in result list
// visno + 1 -> in result list + in candidates
int search_from_candidates_2(const HNSW & hnsw,
                             DistanceComputer & qdis, int k,
                             idx_t *I, float * D,
                             MinimaxHeap &candidates,
                             VisitedTable &vt,
                             int level, int nres_in = 0)
{
    int nres = nres_in;
    int ndis = 0;
    for (int i = 0; i < candidates.size(); i++) {
        idx_t v1 = candidates.ids[i];
        FAISS_ASSERT(v1 >= 0);
        vt.visited[v1] = vt.visno + 1;
    }

    int nstep = 0;

    while (candidates.size() > 0) {
        float d0 = 0;
        int v0 = candidates.pop_min(&d0);

        size_t begin, end;
        hnsw.neighbor_range(v0, level, &begin, &end);

        for (size_t j = begin; j < end; j++) {
            int v1 = hnsw.neighbors[j];
            if (v1 < 0) break;
            if (vt.visited[v1] == vt.visno + 1) {
                // nothing to do
            } else {
                ndis++;
                float d = qdis(v1);
                candidates.push(v1, d);

                // never seen before --> add to heap
                if (vt.visited[v1] < vt.visno) {
                    if (nres < k) {
                        faiss::maxheap_push (++nres, D, I, d, v1);
                    } else if (d < D[0]) {
                        faiss::maxheap_pop (nres--, D, I);
                        faiss::maxheap_push (++nres, D, I, d, v1);
                    }
                }
                vt.visited[v1] = vt.visno + 1;
            }
        }

        nstep++;
        if (nstep > hnsw.efSearch) {
            break;
        }
    }

    if (level == 0) {
#pragma omp critical
        {
            hnsw_stats.n1 ++;
            if (candidates.size() == 0)
                hnsw_stats.n2 ++;
        }
    }


    return nres;
}


}  // namespace

void IndexHNSW2Level::search (idx_t n, const float *x, idx_t k,
                              float *distances, idx_t *labels) const
{
    if (dynamic_cast<const Index2Layer*>(storage)) {
        IndexHNSW::search (n, x, k, distances, labels);

    } else { // "mixed" search

        const IndexIVFPQ *index_ivfpq =
            dynamic_cast<const IndexIVFPQ*>(storage);

        int nprobe = index_ivfpq->nprobe;

        long * coarse_assign = new long [n * nprobe];
        ScopeDeleter<long> del (coarse_assign);
        float * coarse_dis = new float [n * nprobe];
        ScopeDeleter<float> del2 (coarse_dis);

        index_ivfpq->quantizer->search (n, x, nprobe, coarse_dis, coarse_assign);

        index_ivfpq->search_preassigned (
            n, x, k, coarse_assign, coarse_dis, distances, labels, false);

#pragma omp parallel
        {
            VisitedTable vt (ntotal);
            DistanceComputer *dis = get_distance_computer();
            ScopeDeleter1<DistanceComputer> del(dis);

            int candidates_size = hnsw.upper_beam;
            MinimaxHeap candidates(candidates_size);

#pragma omp for
            for(idx_t i = 0; i < n; i++) {
                idx_t * idxi = labels + i * k;
                float * simi = distances + i * k;
                dis->set_query(x + i * d);

                // mark all inverted list elements as visited

                for (int j = 0; j < nprobe; j++) {
                    idx_t key = coarse_assign[j + i * nprobe];
                    if (key < 0) break;
                    size_t list_length = index_ivfpq->get_list_size (key);
                    const idx_t * ids = index_ivfpq->invlists->get_ids (key);

                    for (int jj = 0; jj < list_length; jj++) {
                        vt.set (ids[jj]);
                    }
                }

                candidates.clear();
                // copy the upper_beam elements to candidates list

                int search_policy = 2;

                if (search_policy == 1) {

                    for (int j = 0 ; j < hnsw.upper_beam && j < k; j++) {
                        if (idxi[j] < 0) break;
                        candidates.push (idxi[j], simi[j]);
                        // search_from_candidates adds them back
                        idxi[j] = -1;
                        simi[j] = HUGE_VAL;
                    }

                    // reorder from sorted to heap
                    maxheap_heapify (k, simi, idxi, simi, idxi, k);

                    hnsw.search_from_candidates(
                      *dis, k, idxi, simi,
                      candidates, vt, 0, k
                    );

                    vt.advance();

                } else if (search_policy == 2) {

                    for (int j = 0 ; j < hnsw.upper_beam && j < k; j++) {
                        if (idxi[j] < 0) break;
                        candidates.push (idxi[j], simi[j]);
                    }

                    // reorder from sorted to heap
                    maxheap_heapify (k, simi, idxi, simi, idxi, k);

                    search_from_candidates_2 (
                        hnsw, *dis, k, idxi, simi,
                        candidates, vt, 0, k);
                    vt.advance ();
                    vt.advance ();

                }

                maxheap_reorder (k, simi, idxi);
            }
        }
    }


}


void IndexHNSW2Level::flip_to_ivf ()
{
    Index2Layer *storage2l =
        dynamic_cast<Index2Layer*>(storage);

    FAISS_THROW_IF_NOT (storage2l);

    IndexIVFPQ * index_ivfpq =
        new IndexIVFPQ (storage2l->q1.quantizer,
                        d, storage2l->q1.nlist,
                        storage2l->pq.M, 8);
    index_ivfpq->pq = storage2l->pq;
    index_ivfpq->is_trained = storage2l->is_trained;
    index_ivfpq->precompute_table();
    index_ivfpq->own_fields = storage2l->q1.own_fields;
    storage2l->transfer_to_IVFPQ(*index_ivfpq);
    index_ivfpq->make_direct_map (true);

    storage = index_ivfpq;
    delete storage2l;

}


}  // namespace faiss