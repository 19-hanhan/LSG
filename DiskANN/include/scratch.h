// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <vector>

#include "boost_dynamic_bitset_fwd.h"
// #include "boost/dynamic_bitset.hpp"
#include "tsl/robin_set.h"
#include "tsl/robin_map.h"
#include "tsl/sparse_map.h"

#include "aligned_file_reader.h"
#include "abstract_scratch.h"
#include "neighbor.h"
#include "defaults.h"
#include "concurrent_queue.h"

namespace diskann
{
template <typename T> class PQScratch;

//
// AbstractScratch space for in-memory index based search
//
template <typename T> class InMemQueryScratch : public AbstractScratch<T>
{
  public:
    ~InMemQueryScratch();
    InMemQueryScratch(uint32_t search_l, uint32_t indexing_l, uint32_t r, uint32_t maxc, size_t dim, size_t aligned_dim,
                      size_t alignment_factor, bool init_pq_scratch = false);
    void resize_for_new_L(uint32_t new_search_l);
    void clear();

    inline uint32_t get_L()
    {
        return _L;
    }
    inline uint32_t get_R()
    {
        return _R;
    }
    inline uint32_t get_maxc()
    {
        return _maxc;
    }
    inline T *aligned_query()
    {
        return this->_aligned_query_T;
    }
    inline PQScratch<T> *pq_scratch()
    {
        return this->_pq_scratch;
    }
    inline std::vector<Neighbor> &pool()
    {
        return _pool;
    }
    inline NeighborPriorityQueue &best_l_nodes()
    {
        return _best_l_nodes;
    }
    inline std::vector<float> &occlude_factor()
    {
        return _occlude_factor;
    }
    inline tsl::robin_set<uint32_t> &inserted_into_pool_rs()
    {
        return _inserted_into_pool_rs;
    }
    inline boost::dynamic_bitset<> &inserted_into_pool_bs()
    {
        return *_inserted_into_pool_bs;
    }
    inline std::vector<uint32_t> &id_scratch()
    {
        return _id_scratch;
    }
    inline std::vector<float> &dist_scratch()
    {
        return _dist_scratch;
    }
    inline tsl::robin_set<uint32_t> &expanded_nodes_set()
    {
        return _expanded_nodes_set;
    }
    inline std::vector<Neighbor> &expanded_nodes_vec()
    {
        return _expanded_nghrs_vec;
    }
    inline std::vector<uint32_t> &occlude_list_output()
    {
        return _occlude_list_output;
    }
    // hanhan avgknn
    void set_location(int l)
    {
        location = l;
    }
    int get_location()
    {
        return location;
    }

  private:
    uint32_t _L;
    uint32_t _R;
    uint32_t _maxc;
    int location;

    // _pool stores all neighbors explored from best_L_nodes.
    // Usually around L+R, but could be higher.
    // Initialized to 3L+R for some slack, expands as needed.
    // _pool 存储从 best_L_nodes 探索的所有邻居。
    // 通常在 L+R 左右，但也可能更高。
    // 初始化为 3L+R 以获得一些松弛，根据需要进行扩展。
    std::vector<Neighbor> _pool;

    // _best_l_nodes is reserved for storing best L entries
    // Underlying storage is L+1 to support inserts
    // _best_l_nodes 保留用于存储最佳 L 条目
    // 底层存储为L+1，支持插入
    NeighborPriorityQueue _best_l_nodes;

    // _occlude_factor.size() >= pool.size() in occlude_list function
    // _pool is clipped to maxc in occlude_list before affecting _occlude_factor
    // _occlude_factor is initialized to maxc size
    // occlude_list 函数中的 _occlude_factor.size() >= pool.size()
    // _pool 在影响 _occlude_factor 之前被剪裁到 occlude_list 中的 maxc
    // _occlude_factor 初始化为 maxc 大小
    std::vector<float> _occlude_factor;

    // Capacity initialized to 20L
    // 容量初始化为20L
    tsl::robin_set<uint32_t> _inserted_into_pool_rs;

    // Use a pointer here to allow for forward declaration of dynamic_bitset
    // in public headers to avoid making boost a dependency for clients
    // of DiskANN.
    // 此处使用指针以允许动态_bitset 的前向声明
    // 在公共标头中以避免让 boost 成为客户端的依赖项
    // DiskANN 的。
    boost::dynamic_bitset<> *_inserted_into_pool_bs;

    // _id_scratch.size() must be > R*GRAPH_SLACK_FACTOR for iterate_to_fp
    // 对于 iterate_to_fp，_id_scratch.size() 必须 > R*GRAPH_SLACK_FACTOR
    std::vector<uint32_t> _id_scratch;

    // _dist_scratch must be > R*GRAPH_SLACK_FACTOR for iterate_to_fp
    // _dist_scratch should be at least the size of id_scratch
    // 对于 iterate_to_fp，_dist_scratch 必须 > R*GRAPH_SLACK_FACTOR
    // _dist_scratch 应该至少是 id_scratch 的大小
    std::vector<float> _dist_scratch;

    //  Buffers used in process delete, capacity increases as needed
    // 进程删除时使用的缓冲区，容量根据需要增加
    tsl::robin_set<uint32_t> _expanded_nodes_set;
    std::vector<Neighbor> _expanded_nghrs_vec;
    std::vector<uint32_t> _occlude_list_output;
};

//
// AbstractScratch space for SSD index based search
//

template <typename T> class SSDQueryScratch : public AbstractScratch<T>
{
  public:
    T *coord_scratch = nullptr; // MUST BE AT LEAST [sizeof(T) * data_dim]

    char *sector_scratch = nullptr; // MUST BE AT LEAST [MAX_N_SECTOR_READS * SECTOR_LEN]
    size_t sector_idx = 0;          // index of next [SECTOR_LEN] scratch to use

    tsl::robin_set<size_t> visited;
    NeighborPriorityQueue retset;
    std::vector<Neighbor> full_retset;

    SSDQueryScratch(size_t aligned_dim, size_t visited_reserve);
    ~SSDQueryScratch();

    void reset();
};

template <typename T> class SSDThreadData
{
  public:
    SSDQueryScratch<T> scratch;
    IOContext ctx;

    SSDThreadData(size_t aligned_dim, size_t visited_reserve);
    void clear();
};

//
// Class to avoid the hassle of pushing and popping the query scratch.
//
template <typename T> class ScratchStoreManager
{
  public:
    ScratchStoreManager(ConcurrentQueue<T *> &query_scratch) : _scratch_pool(query_scratch)
    {
        _scratch = query_scratch.pop();
        while (_scratch == nullptr)
        {
            query_scratch.wait_for_push_notify();
            _scratch = query_scratch.pop();
        }
    }
    T *scratch_space()
    {
        return _scratch;
    }

    ~ScratchStoreManager()
    {
        _scratch->clear();
        _scratch_pool.push(_scratch);
        _scratch_pool.push_notify_all();
    }

    void destroy()
    {
        while (!_scratch_pool.empty())
        {
            auto scratch = _scratch_pool.pop();
            while (scratch == nullptr)
            {
                _scratch_pool.wait_for_push_notify();
                scratch = _scratch_pool.pop();
            }
            delete scratch;
        }
    }

  private:
    T *_scratch;
    ConcurrentQueue<T *> &_scratch_pool;
    ScratchStoreManager(const ScratchStoreManager<T> &);
    ScratchStoreManager &operator=(const ScratchStoreManager<T> &);
};
} // namespace diskann
