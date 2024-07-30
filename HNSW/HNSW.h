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
#include <unordered_set>
#include <queue>
#include <unordered_map>

#include <omp.h>

#include "Index.h"
#include "FaissAssert.h"
#include "utils.h"



namespace faiss {


/** Implementation of the Hierarchical Navigable Small World
 * datastructure.
 *
 * Efficient and robust approximate nearest neighbor search using
 * Hierarchical Navigable Small World graphs
 *
 *  Yu. A. Malkov, D. A. Yashunin, arXiv 2017
 *
 * This implmentation is heavily influenced by the NMSlib
 * implementation by Yury Malkov and Leonid Boystov
 * (https://github.com/searchivarius/nmslib)
 *
 * The HNSW object stores only the neighbor link structure, see
 * IndexHNSW below for the full index object.
 */


struct VisitedTable;


struct HNSW {
  /// internal storage of vectors (32 bits: this is expensive)  
  typedef int storage_idx_t;

  /// Faiss results are 64-bit
  typedef Index::idx_t idx_t;

  typedef std::pair<float, storage_idx_t> Node;

  /** The HNSW structure does not store vectors, it only accesses
   * them through this class.
   *
   * Functions are guaranteed to be be accessed only from 1 thread. */
  struct DistanceComputer {
    idx_t d;

    /// called before computing distances
    virtual void set_query(const float *x, storage_idx_t idx = -1) = 0;
 
    /// compute distance of vector i to current query
    virtual float operator () (storage_idx_t i) = 0;

    /// compute distance between two stored vectors
    virtual float symmetric_dis(storage_idx_t i, storage_idx_t j) = 0;

    virtual ~DistanceComputer() {}
  };


  /** Heap structure that allows fast
     */
    struct MinimaxHeap {
        int n;
        int k;
        int nvalid;

        std::vector<storage_idx_t> ids;
        std::vector<float> dis;
        typedef faiss::CMax<float, storage_idx_t> HC;

        explicit MinimaxHeap(int n) : n(n), k(0), nvalid(0), ids(n), dis(n) {}

        void push(storage_idx_t i, float v);

        float max() const;

        int size() const;

        void clear();

        int pop_min(float* vmin_out = nullptr);

        int count_below(float thresh);
    };


  /// to sort pairs of (id, distance) from nearest to fathest or the reverse
  struct NodeDistCloser {
    float d;
    int id;
    NodeDistCloser(float d, int id): d(d), id(id) {}
    bool operator < (const NodeDistCloser &obj1) const { return d < obj1.d; }
  };

  struct NodeDistFarther {
    float d;
    int id;
    NodeDistFarther(float d, int id): d(d), id(id) {}
    NodeDistFarther(){}
    bool operator < (const NodeDistFarther &obj1) const { return d > obj1.d; }
  };

  //存储热点：
  // std::vector<std::unordered_set<idx_t>> hotset;



  /// assignment probability to each layer (sum=1)
  std::vector<double> assign_probas;

  /// number of neighbors stored per layer (cumulative), should not
  /// be changed after first add
  std::vector<int> cum_nneighbor_per_level;

  /// level of each vector (base level = 1), size = ntotal
  // 存放每个节点的最高层
  std::vector<int> levels;

  /// offsets[i] is the offset in the neighbors array where vector i is stored
  /// size ntotal + 1
  // 存放每个节点在neighbors中的偏移量
  // e.g.: offsets[i] 表示i为节点在neighbor中的起始位置
  std::vector<size_t> offsets;

  /// neighbors[offsets[i]:offsets[i+1]] is the list of neighbors of vector i
  /// for all levels. this is where all storage goes.
  // offsets[i] 表示所有i节点在所有层次邻居中的位置
  std::vector<storage_idx_t> neighbors;

  /// entry point in the search structure (one of the points with maximum level
  storage_idx_t entry_point;

  faiss::RandomGenerator rng;

  /// maximum level
  int max_level;

  /// expansion factor at construction time
  // 构建时搜索邻居参数，相当于搜索时的efsearch
  int efConstruction;

  /// expansion factor at search time
  int efSearch;
  
  /// number of entry points in levels > 0.
  int upper_beam;


  //最大入度
  int maxInDegree;

  float splitRate;
  // 双索引
  HNSW* index1;
  HNSW* index2;

  /// use bounded queue during exploration
  bool search_bounded_queue = false;

  /// hot_hubs for searching phase
  std::vector<std::unordered_map<idx_t,std::vector<idx_t>>> hot_hubs;

  // methods that initialize the tree sizes

  /// initialize the assign_probas and cum_nneighbor_per_level to
  /// have 2*M links on level 0 and M links on levels > 0
  void set_default_probas(int M, float levelMult);

  /// set nb of neighbors for this level (before adding anything)
  void set_nb_neighbors(int level_no, int n);

  // methods that access the tree sizes

  /// nb of neighbors for this level
  int nb_neighbors(int layer_no) const;

  /// cumumlative nb up to (and excluding) this level
  int cum_nb_neighbors(int layer_no) const;

  /// range of entries in the neighbors table of vertex no at layer_no
  void neighbor_range(idx_t no, int layer_no,
                      size_t * begin, size_t * end) const;

  /// only mandatory parameter: nb of neighbors
  explicit HNSW(int M = 32);

  /// pick a random level for a new point
  int random_level();

  /// add n random levels to table (for debugging...)
  void fill_with_random_links(size_t n);

  void add_links_starting_from(DistanceComputer& ptdis,
                               storage_idx_t pt_id,
                               storage_idx_t nearest,
                               float d_nearest,
                               int level,
                               omp_lock_t *locks,
                               VisitedTable &vt,
                               int temp_n);


  /** add point pt_id on all levels <= pt_level and build the link
   * structure for them. */
  void add_with_locks(DistanceComputer& ptdis, int pt_level, int pt_id,
                      std::vector<omp_lock_t>& locks,
                      VisitedTable& vt);

  // 寻找热点，并将热点对应的邻居放入hot_hubs
  void find_hot_hubs(std::vector<std::unordered_set<idx_t>>& ses,
      idx_t n, std::vector<float>& ratios);
  /// 邻居的邻居寻找热点
  /*
  * find_hubs_mod = 0 表示使用全局反向边统计热点
  * find_hubs_mod = 1 表示使用邻居的邻居反向边统计热点
  * find_hubs_mod = 2 表示使用聚类邻居反向边统计热点
  * find_hubs_mod = 3 表示使用全局与聚类相结合的方式寻找反向边
  */
  void find_hot_hubs_with_neighbors(std::vector<std::unordered_set<idx_t>>& ses,
    idx_t n, std::vector<float>& ratios);


  /// 聚类寻找热点  ，clsf 为每个结点对应的类
  void find_hot_hubs_with_classfication(std::vector<std::unordered_set<idx_t>>& ses,
    idx_t n, std::vector<float>& ratios,std::vector<idx_t>& clsf);

  /*
  *
  * hot_hubs存放多层次热点，及其候选邻居；
  * n:原始节点数； 
  * len_ratios:等级个数,ratios：等级比例
  * find_hubs_mode：寻找热点方式
  * find_neighbors_mode：热点邻居选择方式
  * nb_reverse_neighbors：添加热点邻居的个数
  *
  */
  void find_hot_hubs_enhence(std::vector<std::unordered_map<idx_t,std::vector<idx_t>>>& hot_hubs,
        idx_t n, std::vector<float>& ratios, int find_hubs_mode,
        int find_neighbors_mode);
  /*
  *
  * ses存放多层次热点
  * hot_hub存放多层次热点，及其候选邻居；
  * n:原始节点数； 
  * find_neighbors_mode: 选择候选邻居的方式 
  * clsf:每个点所对应的类别
  */
  void hot_hubs_new_neighbors(std::vector<std::unordered_set<idx_t>>& ses,
        std::vector<std::unordered_map<idx_t,std::vector<idx_t>>>& hot_hubs,
        idx_t n,int find_neighbors_mode);


  void fromNearestToFurther(int nb_reverse_nbs_level,std::vector<idx_t>& new_neibor,
                DistanceComputer& dis,idx_t hot_hub);

  void shink_reverse_neighbors(int nb_reverse_nbs_level,std::vector<idx_t>& new_neibor,
                DistanceComputer& dis,idx_t hot_hub,size_t n);

  // 将该结点的新邻居添加到索引末尾
  void add_new_reverse_link_end_enhence(
        std::vector<std::unordered_map<idx_t,std::vector<idx_t>>>& hot_hubs,
        size_t n,DistanceComputer& dis,std::vector<int>& nb_reverse_neighbors);

  // 将热点的邻居连向热点
void add_links_to_hubs(
        std::vector<std::unordered_map<idx_t,std::vector<idx_t>>>& hot_hubs,
        size_t n);

  // 子图寻找热点
  void find_hot_hubs_enhence_subIndex(std::vector<std::unordered_map<idx_t,std::vector<idx_t>>>& hot_hubs,
        idx_t n, std::vector<float>& ratios, int find_hubs_mode,
        int find_neighbors_mode);
  // 子图根据热点选择热点反向边
  void hot_hubs_new_neighbors_subIndex(std::vector<std::unordered_set<idx_t>>& ses,
        std::vector<std::unordered_map<idx_t,std::vector<idx_t>>>& hot_hubs,
        idx_t n,int find_neighbors_mode);

  int search_from_candidates(DistanceComputer& qdis, int k,
                             idx_t *I, float *D,
                             MinimaxHeap& candidates,
                             VisitedTable &vt,
                             int level, int nres_in = 0) const;

  std::priority_queue<Node> search_from(const Node& node,
                                        DistanceComputer& qdis,
                                        int ef,
                                        VisitedTable *vt) const;

  // 通过添加一个优先队列，获取K个最近邻
  std::priority_queue<Node> search_from_addk(const Node& node,
                                        DistanceComputer& qdis,
                                        int ef,
                                        VisitedTable *vt,int k) const;
  
  // 通过从candidate中获取访问的K个最近邻
  std::priority_queue<Node> search_from_addk_v2(const Node& node,
                                        DistanceComputer& qdis,
                                        int ef,
                                        VisitedTable *vt,int k) const;

  void search_rt_array(DistanceComputer& qdis, int k,
                  idx_t *I, float *D,
                  VisitedTable& vt,std::vector<idx_t> &vct) const;

  std::priority_queue<Node> search_from_add_vct(
                  const Node& node,
                  DistanceComputer& qdis,
                  int ef,
                  VisitedTable *vt,std::vector<idx_t> &vct) const;


  void similarity(HNSW * index1,HNSW * index2,int nb);

  void setIndex(HNSW * index1,HNSW * index2);

  /*std::priority_queue<Node> search_from_two_index(DistanceComputer& qdis,
                                        int ef,
                                        VisitedTable *vt,int k) const;*/

  /// search interface
  void search(DistanceComputer& qdis, int k,
              idx_t *I, float *D,
              VisitedTable& vt) const;

  void search_custom(DistanceComputer& qdis, int k,
              idx_t *I, float *D,
              VisitedTable& vt,int search_mode) const;

  // 搜索第一个或者是第二个
  // fos==0 第一个,fos==1第二个
  void combine_search(
          DistanceComputer& qdis,
          int k,
          idx_t* I,
          float* D,
          int fos,
          VisitedTable& vt,
          RandomGenerator& rng3) const;

  void search_from_candidates_combine(
      DistanceComputer& qdis,
      int k,
      idx_t* I,
      float* D,
      MinimaxHeap& candidates,
      VisitedTable& vt,
      int level,
      int nres_in ,int fos) const;

  std::priority_queue<Node> search_from_candidate_unbounded_combine(
          const Node& node,
          DistanceComputer& qdis,
          int ef,
          VisitedTable* vt,
          int fos) const;
  
  std::priority_queue<HNSW::Node> search_from_candidate_unbounded_hot_hubs_enhence(
        const Node& node,
        DistanceComputer& qdis,
        int ef,
        VisitedTable* vt,
        idx_t n) const;

    // 普通热点搜索
    void search_from_candidates_hot_hubs(
        DistanceComputer& qdis,
        int k,
        idx_t* I,
        float* D,
        MinimaxHeap& candidates,
        VisitedTable& vt,
        int level,
        int nres_in,unsigned n) const;
  /// 普通热点搜索
  void search_with_hot_hubs_enhence(
          DistanceComputer& qdis,
          int k,
          idx_t* I,
          float* D,
          VisitedTable& vt,size_t n,RandomGenerator& rng3) const;

  // 重启搜索
  std::priority_queue<HNSW::Node> search_from_candidate_unbounded_hot_hubs_enhence_random(
        const Node& node,
        DistanceComputer& qdis,
        int ef,
        VisitedTable* vt,
        idx_t n) const;
  std::priority_queue<HNSW::Node> search_from_candidate_unbounded_hot_hubs_enhence_random_v2(
        const Node& node,
        DistanceComputer& qdis,
        int ef,
        VisitedTable* vt,
        idx_t n) const ;
  std::priority_queue<HNSW::Node> search_from_candidate_unbounded_hot_hubs_enhence_random_v3(
        const Node& node,
        DistanceComputer& qdis,
        int ef,
        VisitedTable* vt,
        idx_t n) const ;


  // 热点重启搜索
  void search_with_hot_hubs_enhence_random(
            DistanceComputer& qdis,
            int k,
            idx_t* I,
            float* D,
            VisitedTable& vt,size_t n,RandomGenerator& rng3) const;



  void reset();

  void clear_neighbor_tables(int level);
  void print_neighbor_stats(int level) const;

  int prepare_level_tab(size_t n, bool preset_levels = false);
  int prepare_level_tab2(size_t n, bool preset_levels = false);
  int prepare_level_tab3(HNSW& index1, size_t n, size_t tmp, bool preset_levels = false);
  // 为热点邻居在索引末尾分配空间
  void prepare_level_tab4(idx_t n, idx_t tmp);

  void shrink_neighbor_list(
    DistanceComputer& qdis,
    std::priority_queue<NodeDistFarther>& input,
    std::vector<NodeDistFarther>& output,
    int max_size,int split,int level);
  void getSearchDis();
  void getSearchDisFlat();
  void resetCount();
  
  void getHotSet(idx_t n,int len_ratios, const float* ht_hbs_ratios);
  void getHothubsSearchCount();
  void initHothubSearchmp();
  //统计热点占比，比例较少的热点从边的角度占比多少，以及热点联通了多少的点。（都是与普通的点进行对比比较）
  void statichotpercent(idx_t n);
  void statcihotlinknums(idx_t n);
  void getKnn(DistanceComputer& qdis,idx_t id,idx_t n,int k,std::priority_queue<NodeDistCloser>& initial_list);
  void staticKNNHot(idx_t n, int len_ratios,const float* ht_hbs_ratios);
  void find_inNode_Hnsw(idx_t n,int len_ratios,const float* ht_hbs_ratios);
  void find_inNode_Hot(idx_t n,int len_ratios,const float* ht_hbs_ratios);

  //构图过程统计热点
  // std::vector<idx_t> in_degree;
  //过程中统计入度查询热点
  // void init_in_vector(idx_t n);
  void static_in_degree_by_construction(idx_t n);

  //分析热点方向边的方向
  //裁边前和裁边后分析
  void static_in_degree_by_direct(idx_t n,DistanceComputer& qdis);
  void print_time(idx_t n);


  void print_nb_in_degree(idx_t n,int m,std::string dataName,std::string _type);
  void print_nb_in_degree_hot(idx_t n,int m,std::string dataName,std::string _type);

  long long get_computer_count();
  std::priority_queue<HNSW::Node> search_from_candidate_unbounded_hot_hubs_enhence_s(
        MinimaxHeap& candidates,
        DistanceComputer& qdis,
        int ef,
        VisitedTable* vt,
        idx_t n) const;

};


/**************************************************************
 * Auxiliary structures
 **************************************************************/

/// set implementation optimized for fast access.
struct VisitedTable {
  std::vector<uint8_t> visited;
  int visno;

  explicit VisitedTable(int size)
    : visited(size), visno(1) {}

  /// set flog #no to true
  void set(int no) {
    visited[no] = visno;
  }

  /// get flag #no
  bool get(int no) const {
    return visited[no] == visno;
  }

  /// reset all flags to false
  void advance() {
    visno++;
    if (visno == 250) {
      // 250 rather than 255 because sometimes we use visno and visno+1
      memset(visited.data(), 0, sizeof(visited[0]) * visited.size());
      visno = 1;
    }
  }
};


struct HNSWStats {
  size_t n1, n2, n3;
  size_t ndis;
  size_t nreorder;
  bool view;

  HNSWStats() {
    reset();
  }

  void reset() {
    n1 = n2 = n3 = 0;
    ndis = 0;
    nreorder = 0;
    view = false;
  }
};

// global var that collects them all
extern HNSWStats hnsw_stats;


}  // namespace faiss

