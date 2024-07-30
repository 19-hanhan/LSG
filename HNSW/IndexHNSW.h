/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD+Patents license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#pragma once

#include <vector>
#include <functional>

#include "HNSW.h"
#include "IndexFlat.h"
#include "IndexPQ.h"
#include "IndexScalarQuantizer.h"
#include "utils.h"


namespace faiss {

using DistanceComputer = HNSW::DistanceComputer;

struct IndexHNSW;

struct ReconstructFromNeighbors {
    typedef Index::idx_t idx_t;
    typedef HNSW::storage_idx_t storage_idx_t;

    const IndexHNSW & index;
    size_t M; // number of neighbors
    size_t k; // number of codebook entries
    size_t nsq; // number of subvectors
    size_t code_size;
    int k_reorder; // nb to reorder. -1 = all

    std::vector<float> codebook; // size nsq * k * (M + 1)

    std::vector<uint8_t> codes; // size ntotal * code_size
    size_t ntotal;
    size_t d, dsub; // derived values

    explicit ReconstructFromNeighbors(const IndexHNSW& index,
                                      size_t k=256, size_t nsq=1);

    /// codes must be added in the correct order and the IndexHNSW
    /// must be populated and sorted
    void add_codes(size_t n, const float *x);

    size_t compute_distances(size_t n, const idx_t *shortlist,
                             const float *query, float *distances) const;

    /// called by add_codes
    void estimate_code(const float *x, storage_idx_t i, uint8_t *code) const;

    /// called by compute_distances
    void reconstruct(storage_idx_t i, float *x, float *tmp) const;

    void reconstruct_n(storage_idx_t n0, storage_idx_t ni, float *x) const;

    /// get the M+1 -by-d table for neighbor coordinates for vector i
    void get_neighbor_table(storage_idx_t i, float *out) const;

};


/** The HNSW index is a normal random-access index with a HNSW
 * link structure built on top */

struct IndexHNSW : Index {

    typedef HNSW::storage_idx_t storage_idx_t;

    std::string dis_method;

    // the link strcuture
    HNSW hnsw;

    // add，默认标准搜索
    int search_mode;

    // the sequential storage
    bool own_fields;
    Index *storage;
    Index* f_storage;
    Index* s_storage;

    std::vector<idx_t> vct;

    ReconstructFromNeighbors *reconstruct_from_neighbors;

    explicit IndexHNSW (int d = 0, int M = 32);
    explicit IndexHNSW (Index *storage, int M = 32);

    ~IndexHNSW() override;

    // get a DistanceComputer object for this kind of storage
    virtual HNSW::DistanceComputer *get_distance_computer() const = 0;

    HNSW::DistanceComputer *get_distance_computer(Index* storage) const ;

    void add(idx_t n, const float *x) override;

    /// Trains the storage if needed
    void train(idx_t n, const float* x) override;

    /// entry point for search
    void search (idx_t n, const float *x, idx_t k,
                 float *distances, idx_t *labels) const override;

    void search1 (idx_t n, float *x, float *y ,
        float *distances, idx_t *labels,float *distances1, idx_t *labels1,
        IndexHNSW* index1,IndexHNSW* index2,idx_t k) ;

    // 填充AVGDIS
    void set_nicdm_distance(float* x, float y);

    // 索引合并
    void combine_index_with_division(IndexHNSW& index,
        IndexHNSW& index1,IndexHNSW& index2,unsigned n);

    // 查询结果合并
    void final_top_100(float* simi,idx_t* idxi,
            float*simi1,idx_t* idxi1,
            float*simi2,idx_t* idxi2,
            DistanceComputer& dis1,
            DistanceComputer& dis2,
            idx_t k) const;
    // 搜索
    void combine_search(
            idx_t n,
            const float* x,
            const float* y,
            idx_t k,
            float* distances,
            idx_t* labels,
            float* distances1,
            idx_t* labels1,
            float* distances2,
            idx_t* labels2) const;

    /*
    *
    * n:向量个数
    * len_ratios:热点等级个数
    * ht_hbs_ratios：热点比例数组
    * find_hubs_mode:热点寻找方式
    * find_neighbors_mode：热点邻居寻找方式
    * nb_reverse_neighbors ： 热点反向边个数
    *
    */
    void combine_index_with_hot_hubs_enhence(idx_t n,int len_ratios,
            const float* ht_hbs_ratios,int find_hubs_mode,
            int find_neighbors_mode, const int* max_rvs_nb_per_level);

    void search_with_hot_hubs_enhence(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,size_t xb_size) const;

    void enhence_index_with_subIndex_hubs(idx_t n,int len_ratios,
        const float* ht_hbs_ratios,int find_hubs_mode,
        int find_neighbors_mode, const int* nb_nbors_per_level,
        IndexHNSW& ihf1,IndexHNSW& ihf2,IndexHNSW& ihf3,IndexHNSW& ihf4,IndexHNSW& ihf5,
        idx_t* idxScript1,idx_t* idxScript2,
        idx_t* idxScript3,idx_t* idxScript4,idx_t* idxScript5);
/*    void enhence_index_with_subIndex_hubs(idx_t n,int len_ratios,
    const float* ht_hbs_ratios,int find_hubs_mode,
    int find_neighbors_mode, const int* nb_nbors_per_level,
    IndexHNSW& ihf1,IndexHNSW& ihf2,IndexHNSW& ihf3,
    idx_t* idxScript1,idx_t* idxScript2,
    idx_t* idxScript3);*/
    // void enhence_index_with_subIndex_hubs(idx_t n,int len_ratios,
    //     const float* ht_hbs_ratios,int find_hubs_mode,
    //     int find_neighbors_mode, const int* nb_nbors_per_level,
    //     IndexHNSW& ihf1,
    //     idx_t* idxScript1);


    // 新加热点搜索方法，重启
    void search_with_hot_hubs_enhence_random(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,size_t xb_size) const;
    
    // 随机热点，随机反向边，热点搜索方法改变，加入重启
    void complete_random_hot_hubs_enhence(idx_t n,const idx_t *x,int len_ratios,
        const float* ht_hbs_ratios,const int* nb_nbors_per_level,int cls);


    void aod_level();

    void reconstruct(idx_t key, float* recons) const override;

    void reset () override;

    void shrink_level_0_neighbors(int size);

    /** Perform search only on level 0, given the starting points for
     * each vertex.
     *
     * @param search_type 1:perform one search per nprobe, 2: enqueue
     *                    all entry points
     */
    void search_level_0(idx_t n, const float *x, idx_t k,
                        const storage_idx_t *nearest, const float *nearest_d,
                        float *distances, idx_t *labels, int nprobe = 1,
                        int search_type = 1) const;

    /// alternative graph building
    void init_level_0_from_knngraph(
                        int k, const float *D, const idx_t *I);

    /// alternative graph building
    void init_level_0_from_entry_points(
                        int npt, const storage_idx_t *points,
                        const storage_idx_t *nearests);

    // reorder links from nearest to farthest
    void reorder_links();

    void link_singletons();
    void createKnn(idx_t n,int k);
    
    void static_in_degree_by_direct(idx_t n);

    // void test_inner_prodect();
};



/** Flat index topped with with a HNSW structure to access elements
 *  more efficiently.
 */

struct IndexHNSWFlat : IndexHNSW {
    IndexHNSWFlat();
    IndexHNSWFlat(int d, int M);
    HNSW::DistanceComputer *
      get_distance_computer() const override;
};

/** PQ index topped with with a HNSW structure to access elements
 *  more efficiently.
 */
struct IndexHNSWPQ : IndexHNSW {
    IndexHNSWPQ();
    IndexHNSWPQ(int d, int pq_m, int M);
    void train(idx_t n, const float* x) override;
    HNSW::DistanceComputer *
      get_distance_computer() const override;
};

/** SQ index topped with with a HNSW structure to access elements
 *  more efficiently.
 */
struct IndexHNSWSQ : IndexHNSW {
    IndexHNSWSQ();
    IndexHNSWSQ(int d, ScalarQuantizer::QuantizerType qtype, int M);
    HNSW::DistanceComputer *
      get_distance_computer() const override;
};

/** 2-level code structure with fast random access
 */
struct IndexHNSW2Level : IndexHNSW {
    IndexHNSW2Level();
    IndexHNSW2Level(Index *quantizer, size_t nlist, int m_pq, int M);
    HNSW::DistanceComputer *
      get_distance_computer() const override;
    void flip_to_ivf();

    /// entry point for search
    void search (idx_t n, const float *x, idx_t k,
                 float *distances, idx_t *labels) const override;

};


}  // namespace faiss
