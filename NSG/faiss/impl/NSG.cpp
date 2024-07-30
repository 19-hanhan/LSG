/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include <faiss/impl/NSG.h>

#include <algorithm>
#include <memory>
#include <mutex>
#include <stack>

#include <unordered_set>
#include <queue>
#include <unordered_map>
#include<map>

#include <iostream>
#include<fstream>

#include <faiss/impl/AuxIndexStructures.h>

namespace faiss {

namespace nsg {

namespace {

// It needs to be smaller than 0
constexpr int EMPTY_ID = -1;

/* Wrap the distance computer into one that negates the
   distances. This makes supporting INNER_PRODUCE search easier */

struct NegativeDistanceComputer : DistanceComputer {
    using idx_t = Index::idx_t;

    /// owned by this
    DistanceComputer* basedis;

    explicit NegativeDistanceComputer(DistanceComputer* basedis)
            : basedis(basedis) {}

    void set_query(const float* x) override {
        basedis->set_query(x);
    }

    /// compute distance of vector i to current query
    float operator()(idx_t i) override {
        return -(*basedis)(i);
    }

    /// compute distance between two stored vectors
    float symmetric_dis(idx_t i, idx_t j) override {
        return -basedis->symmetric_dis(i, j);
    }

    ~NegativeDistanceComputer() override {
        delete basedis;
    }
};

} // namespace

DistanceComputer* storage_distance_computer(const Index* storage) {
    // 内积（余弦）
    if (storage->metric_type == METRIC_INNER_PRODUCT) {
        printf("METRIC_INNER_PRODUCT");
        return new NegativeDistanceComputer(storage->get_distance_computer());
    } else {
        // 常规距离
        return storage->get_distance_computer();
    }
}

} // namespace nsg

using namespace nsg;

using LockGuard = std::lock_guard<std::mutex>;

struct Neighbor {
    int id;
    float distance;
    bool flag;

    Neighbor() = default;
    Neighbor(int id, float distance, bool f)
            : id(id), distance(distance), flag(f) {}

    inline bool operator<(const Neighbor& other) const {
        return distance < other.distance;
    }
};

struct Node {
    int id;
    float distance;

    Node() = default;
    Node(int id, float distance) : id(id), distance(distance) {}

    inline bool operator<(const Node& other) const {
        return distance < other.distance;
    }
};

inline int insert_into_pool(Neighbor* addr, int K, Neighbor nn) {
    // find the location to insert
    int left = 0, right = K - 1;
    if (addr[left].distance > nn.distance) {
        memmove(&addr[left + 1], &addr[left], K * sizeof(Neighbor));
        addr[left] = nn;
        return left;
    }
    if (addr[right].distance < nn.distance) {
        addr[K] = nn;
        return K;
    }
    while (left < right - 1) {
        int mid = (left + right) / 2;
        if (addr[mid].distance > nn.distance) {
            right = mid;
        } else {
            left = mid;
        }
    }
    // check equal ID

    while (left > 0) {
        if (addr[left].distance < nn.distance) {
            break;
        }
        if (addr[left].id == nn.id) {
            return K + 1;
        }
        left--;
    }
    if (addr[left].id == nn.id || addr[right].id == nn.id) {
        return K + 1;
    }
    memmove(&addr[right + 1], &addr[right], (K - right) * sizeof(Neighbor));
    addr[right] = nn;
    return right;
}

// 在indexNSG中调用该构造方法传入R，最早来自于flat中传入
// NSG构图参数
NSG::NSG(int R) : R(R), rng(0x0903) {
    L = R + 32;  // 相当于efcon
    C = R + 100; // 相当于canditate大小
    search_L = 16; // 最小search_L = 16
    ntotal = 0;
    is_built = false;
    srand(0x1998);
}

// NSG::NSG(int R) : R(R), rng(0x0903) {
//     // L = R + 32;  // 相当于efcon
//     L = 10;
//     C = 10; // 相当于canditate大小
//     // C = 500;
//     search_L = 10; // 最小search_L = 16
//     ntotal = 0;
//     is_built = false;
//     srand(0x1998);
// }

void NSG::search(
        DistanceComputer& dis,
        int k,
        idx_t* I,
        float* D,
        VisitedTable& vt) const {
    FAISS_THROW_IF_NOT(is_built);
    FAISS_THROW_IF_NOT(final_graph);

    int pool_size = std::max(search_L, k); // efsearch
    std::vector<Neighbor> retset;
    std::vector<Node> tmp;
    search_on_graph<false>(
            *final_graph, dis, vt, enterpoint, pool_size, retset, tmp);

    std::partial_sort(
            retset.begin(), retset.begin() + k, retset.begin() + pool_size);

    for (size_t i = 0; i < k; i++) {
        I[i] = retset[i].id;
        D[i] = retset[i].distance;
    }
}


// 搜索
void NSG::search_enhence_with_hubs(
        DistanceComputer& dis,
        int k,
        idx_t* I,
        float* D,
        VisitedTable& vt,int xb_size) const {
    FAISS_THROW_IF_NOT(is_built);
    FAISS_THROW_IF_NOT(final_graph);
    int pool_size = std::max(search_L, k);
    std::vector<Neighbor> retset;
    std::vector<Node> tmp;

    search_on_graph_enhence_with_hubs<false>(
            *final_graph, dis, vt, enterpoint, pool_size, retset, tmp,xb_size);

    std::partial_sort(
            retset.begin(), retset.begin() + k, retset.begin() + pool_size);

    for (size_t i = 0; i < k; i++) {
        I[i] = retset[i].id;
        D[i] = retset[i].distance;
    }
}

// 构图
void NSG::build(
        Index* storage,
        idx_t n,
        const nsg::Graph<idx_t>& knn_graph,
        bool verbose) {
    FAISS_THROW_IF_NOT(!is_built && ntotal == 0);

    // 构图参数
    if (verbose) {
        printf("NSG::build R=%d, L=%d, C=%d\n", R, L, C);
    }

    ntotal = n;
    init_graph(storage, knn_graph); // 寻找入口点ep

    std::vector<int> degrees(n, 0);
    {
        nsg::Graph<Node> tmp_graph(n, R);

        // 从knn->nsg核心代码
        link(storage, knn_graph, tmp_graph, verbose); // KNN图存于Graph中的*data

        // 需要final_graph初始化的时候分配热点空间，共享指针
        final_graph = std::make_shared<nsg::Graph<int>>(n, R);
        std::fill_n(final_graph->data, n * R, EMPTY_ID);

#pragma omp parallel for
        for (int i = 0; i < n; i++) {
            int cnt = 0;
            for (int j = 0; j < R; j++) {
                int id = tmp_graph.at(i, j).id;
                if (id != EMPTY_ID) {
                    final_graph->at(i, cnt) = id;
                    cnt += 1;
                }
                degrees[i] = cnt;
            }
        }
    }
    

    // 生成树结构，保证联通性
    int num_attached = tree_grow(storage, degrees);
    printf("Attached nodes: %d\n", num_attached);

    // 检查图结构
    check_graph();
    is_built = true;

    if (verbose) {
        int max = 0, min = 1e6;
        double avg = 0;

        for (int i = 0; i < n; i++) {
            int size = 0;
            while (size < R && final_graph->at(i, size) != EMPTY_ID) {
                size += 1;
            }
            max = std::max(size, max);
            min = std::min(size, min);
            avg += size;
        }

        avg = avg / n;
        printf("Degree Statistics: Max = %d, Min = %d, Avg = %lf\n",
               max,
               min,
               avg);
        // printf("Attached nodes: %d\n", num_attached);
    }
}

void NSG::reset() {
    final_graph.reset();
    ntotal = 0;
    is_built = false;
}

void NSG::init_graph(Index* storage, const nsg::Graph<idx_t>& knn_graph) {
    int d = storage->d;
    int n = storage->ntotal;

    std::unique_ptr<float[]> center(new float[d]);
    std::unique_ptr<float[]> tmp(new float[d]);
    std::fill_n(center.get(), d, 0.0f);

    for (int i = 0; i < n; i++) {
        storage->reconstruct(i, tmp.get());
        for (int j = 0; j < d; j++) {
            center[j] += tmp[j];
        }
    }

    for (int i = 0; i < d; i++) {
        center[i] /= n;
    }

    std::vector<Neighbor> retset;
    std::vector<Node> tmpset;

    // random initialize navigating point
    int ep = rng.rand_int(n);
    std::unique_ptr<DistanceComputer> dis(storage_distance_computer(storage));

    dis->set_query(center.get());
    VisitedTable vt(ntotal);

    // Do not collect the visited nodes
    search_on_graph<false>(knn_graph, *dis, vt, ep, L, retset, tmpset);

    // set enterpoint
    enterpoint = retset[0].id;
}

// fullset 存放访问过的点
template <bool collect_fullset, class index_t>
void NSG::search_on_graph(
        const nsg::Graph<index_t>& graph,
        DistanceComputer& dis,
        VisitedTable& vt,
        int ep,
        int pool_size,
        std::vector<Neighbor>& retset,
        std::vector<Node>& fullset)
        const {
    RandomGenerator gen(0x1234);
    retset.resize(pool_size + 1);
    std::vector<int> init_ids(pool_size);

    // 将ep 的邻居添加到候选集中
    int num_ids = 0;
    for (int i = 0; i < init_ids.size() && i < graph.K; i++) {
        int id = (int)graph.at(ep, i);
        if (id < 0 || id >= ntotal) {
            continue;
        }

        init_ids[i] = id;
        vt.set(id);
        num_ids += 1;
    }

    // 没有填满，随机加入
    while (num_ids < pool_size) {
        int id = gen.rand_int(ntotal);
        if (vt.get(id)) {
            continue;
        }

        init_ids[num_ids] = id;
        num_ids++;
        vt.set(id);
    }

    // 将初始结点添加到fullset 和 retset 中
    for (int i = 0; i < init_ids.size(); i++) {
        int id = init_ids[i];

        float dist = dis(id);
        retset[i] = Neighbor(id, dist, true);

        if (collect_fullset) {
            fullset.emplace_back(retset[i].id, retset[i].distance);
        }
    }

    std::sort(retset.begin(), retset.begin() + pool_size); // 升序排序，只排序前pool_size个

    // 寻找pool_size个最近邻
    int k = 0;
    while (k < pool_size) {
        int updated_pos = pool_size;

        if (retset[k].flag) {
            retset[k].flag = false;
            int n = retset[k].id;

            for (int m = 0; m < graph.K; m++) { // 遍历 候选点 在 knn 图中 K 个邻居
                int id = (int)graph.at(n, m);
                if (id < 0 || id > ntotal || vt.get(id)) {
                    continue;
                }
                vt.set(id);

                float dist = dis(id);
                Neighbor nn(id, dist, true); 
                if (collect_fullset) {
                    fullset.emplace_back(id, dist);
                }

                if (dist >= retset[pool_size - 1].distance) { // 该点距离大于候选集中点的最大距离
                    continue;
                }

                int r = insert_into_pool(retset.data(), pool_size, nn); // 将nn插入retset中，返回插入位置，插入失败则返回 K + 1

                updated_pos = std::min(updated_pos, r); 
            }
        }

        k = (updated_pos <= k) ? updated_pos : (k + 1); // 成功插入会导致重新搜索 
    }
}


template <bool collect_fullset, class index_t>
void NSG::search_on_graph_enhence_with_hubs(
        const nsg::Graph<index_t>& graph,
        DistanceComputer& dis,
        VisitedTable& vt,
        int ep,
        int pool_size,
        std::vector<Neighbor>& retset,
        std::vector<Node>& fullset,int xb_size) const {
    RandomGenerator gen(0x1234);
    retset.resize(pool_size + 1);
    std::vector<int> init_ids(pool_size);

    int visited_count = 0;

    // printf("search_on_graph_enhence_with_hubs_phase1..begin\n");
    // 将ep 的邻居添加到候选集中
    int num_ids = 0;
    for (int i = 0; i < init_ids.size() && i < graph.K; i++) {
        int id = (int)graph.at(ep, i);
        if (id < 0 || id >= xb_size) {
            continue;
        }

        init_ids[i] = id;
        vt.set(id);
        num_ids += 1;
        visited_count++;  // 更新访问计数器
    }

    // 没有填满，随机加入
    while (num_ids < pool_size) {
        int id = gen.rand_int(xb_size);
        if (vt.get(id)) {
            continue;
        }

        init_ids[num_ids] = id;
        num_ids++;
        vt.set(id);
        visited_count++;  // 更新访问计数器
    }
    // printf("search_on_graph_enhence_with_hubs_phase1..end\n");
    // 将初始结点添加到fullset 和 retset 中
    for (int i = 0; i < init_ids.size(); i++) {
        int id = init_ids[i];
        // printf("id = %d\n",id);
        float dist = dis(id);
        // printf("dist = %f\n",dist);
        retset[i] = Neighbor(id, dist, true);
        // printf("retset[i] = %d\n",retset.size());
        if (collect_fullset) {
            fullset.emplace_back(retset[i].id, retset[i].distance);
        }
    }

    std::sort(retset.begin(), retset.begin() + pool_size);

    

    // printf("search_on_graph_enhence_with_hubs_phase2..begin\n");

    int k = 0;
    while (k < pool_size) {
        int updated_pos = pool_size;

        if (retset[k].flag) {
            retset[k].flag = false;
            int n = retset[k].id;
            // printf("search_on_graph_enhence_with_hubs_phase3..begin\n");
            for (int m = 0; m < graph.K; m++) {
                // printf("ok111:%d\n",n);
                int id = (int)graph.at(n, m);
                // printf("ok222:%d\n",id);
                // 没有元素或者已经访问过该元素（为了好看这样写不好吧！😊）
                // 小心 vt 越界(根据index设置vt长度，而不是nsg中的ntotal，修改vt长度)
                if (vt.get(id)) {
                    continue;
                }
                // printf("ok333:%d\n",n);
                if (id < 0)
                    break;

                // 热点，开始一个新的行搜索
                if (id >= xb_size) {
                    n = id;
                    m = -1;
                    continue;
                }
                // printf("ok444:%d\n",n);

                vt.set(id);
                visited_count++;  // 更新访问计数器

                float dist = dis(id);
                Neighbor nn(id, dist, true);
                if (collect_fullset) {
                    fullset.emplace_back(id, dist);
                }
                // printf("ok555:%d\n",n);

                if (dist >= retset[pool_size - 1].distance) {
                    continue;
                }

                int r = insert_into_pool(retset.data(), pool_size, nn);

                updated_pos = std::min(updated_pos, r);
            }
            // printf("search_on_graph_enhence_with_hubs_phase3..end\n");
        }

        k = (updated_pos <= k) ? updated_pos : (k + 1);
    }
    // printf("search_on_graph_enhence_with_hubs_phase2..end\n");

    // 输出总访问节点数
    // printf("Total visited nodes:%d\n",visited_count);
    // 打开文件用于写入
    std::ofstream outfile("temp.txt",std::ios::app);
    if (!outfile.is_open()) {
        std::cerr << "无法打开文件" << std::endl;
    }else{
        outfile << visited_count << std::endl;
    }
    // 关闭文件
    outfile.close();
}


void NSG::link(
        Index* storage,
        const nsg::Graph<idx_t>& knn_graph,
        nsg::Graph<Node>& graph,
        bool /* verbose */) {

#pragma omp parallel
    {
        std::unique_ptr<float[]> vec(new float[storage->d]);

        std::vector<Node> pool;
        std::vector<Neighbor> tmp;

        VisitedTable vt(ntotal);

        //第2个距离计算器，用于NSG的构图和裁边
        std::unique_ptr<DistanceComputer> dis(
                storage_distance_computer(storage));

#pragma omp for schedule(dynamic, 100)
        for (int i = 0; i < ntotal; i++) {
            // 根据索引 i 重建存储向量，并将其结果存储在由 vec 管理的数组中
            // 获取原始向量存入vec中
            storage->reconstruct(i, vec.get());

            // 设置基准点，用于计算距离
            dis->set_query(vec.get());
            //dis->set_query_idx(vec.get(),i); 
            
            // Collect the visited nodes into pool
            // 获取候选结点
            search_on_graph<true>(
                    knn_graph, *dis, vt, enterpoint, L, tmp, pool);

            // 裁边
            sync_prune(i, pool, *dis, vt, knn_graph, graph);

            pool.clear();
            tmp.clear();
            vt.advance();
        }


    } // omp parallel

    std::vector<std::mutex> locks(ntotal);
#pragma omp parallel
    {
        std::unique_ptr<DistanceComputer> dis(
                storage_distance_computer(storage));

#pragma omp for schedule(dynamic, 100)
        for (int i = 0; i < ntotal; ++i) {
            add_reverse_links(i, locks, *dis, graph);
        }
    } // omp parallel
}

void NSG::sync_prune(
        int q,
        std::vector<Node>& pool,
        DistanceComputer& dis,
        VisitedTable& vt,
        const nsg::Graph<idx_t>& knn_graph,
        nsg::Graph<Node>& graph) {
    for (int i = 0; i < knn_graph.K; i++) {
        int id = knn_graph.at(q, i);
        if (id < 0 || id >= ntotal || vt.get(id)) {
            continue;
        }

        float dist = dis.symmetric_dis(q, id);
        pool.emplace_back(id, dist);
    }

    std::sort(pool.begin(), pool.end());

    std::vector<Node> result;

    int start = 0;
    if (pool[start].id == q) {
        start++;
    }
    result.push_back(pool[start]);

    while (result.size() < R && (++start) < pool.size() && start < C) { //R是最终保留的邻居数，pool.size = L是跳数，C是访问次数
        auto& p = pool[start]; // 遍历候选集pool
        bool occlude = false;
        for (int t = 0; t < result.size(); t++) {
            // 防止重复
            if (p.id == result[t].id) {
                occlude = true;
                break;
            }
            // 裁边：60度角
            float djk = dis.symmetric_dis(result[t].id, p.id);
            if (djk < p.distance /* dik */) {
                occlude = true;
                break;
            }
        }
        if (!occlude) { // 符合条件加入结果集
            result.push_back(p);
        }
    }

    // 将邻居数量缩小 , 预留位置添加反向边
    int r1 = R/2;

    for (size_t i = 0; i < R; i++) {
        if (i < result.size() && i < r1) {
            graph.at(q, i).id = result[i].id;
            graph.at(q, i).distance = result[i].distance;
        } else {
            graph.at(q, i).id = EMPTY_ID;
        }
    }
}

/*
void NSG::add_reverse_links(
        int q,
        std::vector<std::mutex>& locks,
        DistanceComputer& dis,
        nsg::Graph<Node>& graph) {
    for (size_t i = 0; i < R; i++) {
        if (graph.at(q, i).id == EMPTY_ID) {
            break;
        }

        Node sn(q, graph.at(q, i).distance);
        int des = graph.at(q, i).id;

        std::vector<Node> tmp_pool;
        int dup = 0;
        {
            LockGuard guard(locks[des]);
            for (int j = 0; j < R; j++) {
                if (graph.at(des, j).id == EMPTY_ID) {
                    break;
                }
                if (q == graph.at(des, j).id) {
                    dup = 1;
                    break;
                }
                tmp_pool.push_back(graph.at(des, j));
            }
        }

        // 已经建立双向边
        if (dup) {
            continue;
        }

        tmp_pool.push_back(sn);
        if (tmp_pool.size() > R) { // 竞争裁边
            std::vector<Node> result;
            int start = 0;
            std::sort(tmp_pool.begin(), tmp_pool.end());
            result.push_back(tmp_pool[start]);

            while (result.size() < R && (++start) < tmp_pool.size()) {
                auto& p = tmp_pool[start];
                bool occlude = false;

                for (int t = 0; t < result.size(); t++) {
                    if (p.id == result[t].id) {
                        occlude = true;
                        break;
                    }
                    float djk = dis.symmetric_dis(result[t].id, p.id);
                    if (djk < p.distance ) {
                        occlude = true;
                        break;
                    }
                }

                if (!occlude) {
                    result.push_back(p);
                }
            }

            {
                LockGuard guard(locks[des]);
                for (int t = 0; t < result.size(); t++) {
                    graph.at(des, t) = result[t];
                }
            }

        } else { // 有多余空间
            LockGuard guard(locks[des]);
            for (int t = 0; t < R; t++) {
                if (graph.at(des, t).id == EMPTY_ID) {
                    graph.at(des, t) = sn;
                    break;
                }
            }
        }
    }
}
*/



void NSG::add_reverse_links(
        int q,
        std::vector<std::mutex>& locks,
        DistanceComputer& dis,
        nsg::Graph<Node>& graph) {
        for (size_t i = 0; i < R; i++) {
        if (graph.at(q, i).id == EMPTY_ID) {
            break;
        }

        Node sn(q, graph.at(q, i).distance);
        int des = graph.at(q, i).id;

        std::vector<Node> tmp_pool;
        int dup = 0;
        {
            LockGuard guard(locks[des]);
            for (int j = 0; j < R; j++) {
                if (graph.at(des, j).id == EMPTY_ID) {
                    break;
                }
                if (q == graph.at(des, j).id) {
                    dup = 1;
                    break;
                }
                tmp_pool.push_back(graph.at(des, j));
            }
        }

        // 已经建立双向边
        if (dup) {
            continue;
        }

        float rt = 2;
        tmp_pool.push_back(sn);
        // printf("tmp_pool.size = %d\n",tmp_pool.size());
        if (tmp_pool.size() > R) { // 竞争裁边
            std::vector<Node> result;
            int start = 0; 
            // 将后续结点放入，并不放入前边结点
            int len = tmp_pool.size();
            // printf("r= %d\n",len);
            tmp_pool.erase(tmp_pool.begin(), tmp_pool.end() - (len - R/rt));
            // printf("r2= %d\n",(int)tmp_pool.size());
            std::sort(tmp_pool.begin(), tmp_pool.end());
            result.push_back(tmp_pool[start]);

            while (result.size() < R-R/rt && (++start) < tmp_pool.size()) {
                auto& p = tmp_pool[start];
                bool occlude = false;

                for (int t = 0; t < result.size(); t++) {
                    if (p.id == result[t].id) {
                        occlude = true;
                        break;
                    }
                    float djk = dis.symmetric_dis(result[t].id, p.id);
                    if (djk < p.distance /* dik */) {
                        occlude = true;
                        break;
                    }
                }

                if (!occlude) {
                    result.push_back(p);
                }
            }

            // printf("r2= %d\n",(int)tmp_pool.size());

            {
                // printf("ok\n");
                // printf("pre = %d, prun = %d\n",graph.at(des, R/rt - 1).id, (int)result.size());
                LockGuard guard(locks[des]);
                for (int t = 0; t < result.size(); t++) {  
                    graph.at(des, t + R/rt) = result[t]; // 前边部分不变，后边放入新的内容
                }
            }

        } else { // 有多余空间
            LockGuard guard(locks[des]);
            for (int t = 0; t < R; t++) {
                if (graph.at(des, t).id == EMPTY_ID) {
                    graph.at(des, t) = sn;
                    break;
                }
            }
        }
    }
}

int NSG::tree_grow(Index* storage, std::vector<int>& degrees) {
    int root = enterpoint;
    VisitedTable vt(ntotal);
    VisitedTable vt2(ntotal);

    int num_attached = 0;
    int cnt = 0;
    while (true) {
        cnt = dfs(vt, root, cnt);
        if (cnt >= ntotal) {
            break;
        }

        root = attach_unlinked(storage, vt, vt2, degrees);
        vt2.advance();
        num_attached += 1;
    }

    return num_attached;
}

int NSG::dfs(VisitedTable& vt, int root, int cnt) const {
    int node = root;
    std::stack<int> stack;
    stack.push(root);

    if (!vt.get(root)) {
        cnt++;
    }
    vt.set(root);

    while (!stack.empty()) {
        int next = EMPTY_ID;
        for (int i = 0; i < R; i++) {
            int id = final_graph->at(node, i);
            if (id != EMPTY_ID && !vt.get(id)) {
                next = id;
                break;
            }
        }

        if (next == EMPTY_ID) {
            stack.pop();
            if (stack.empty()) {
                break;
            }
            node = stack.top();
            continue;
        }
        node = next;
        vt.set(node);
        stack.push(node);
        cnt++;
    }

    return cnt;
}

int NSG::attach_unlinked(
        Index* storage,
        VisitedTable& vt,
        VisitedTable& vt2,
        std::vector<int>& degrees) {
    /* NOTE: This implementation is slightly different from the original paper.
     *
     * Instead of connecting the unlinked node to the nearest point in the
     * spanning tree which will increase the maximum degree of the graph and
     * also make the graph hard to maintain, this implementation links the
     * unlinked node to the nearest node of which the degree is smaller than R.
     * It will keep the degree of all nodes to be no more than `R`.
     */

    // find one unlinked node
    int id = EMPTY_ID;
    for (int i = 0; i < ntotal; i++) {
        if (!vt.get(i)) {
            id = i;
            break;
        }
    }

    if (id == EMPTY_ID) {
        return EMPTY_ID; // No Unlinked Node
    }

    std::vector<Neighbor> tmp;
    std::vector<Node> pool;

    std::unique_ptr<DistanceComputer> dis(storage_distance_computer(storage));
    std::unique_ptr<float[]> vec(new float[storage->d]);

    storage->reconstruct(id, vec.get());
    dis->set_query(vec.get());

    // Collect the visited nodes into pool
    search_on_graph<true>(
            *final_graph, *dis, vt2, enterpoint, search_L, tmp, pool);

    std::sort(pool.begin(), pool.end());

    int node;
    bool found = false;
    for (int i = 0; i < pool.size(); i++) {
        node = pool[i].id;
        if (degrees[node] < R && node != id) {
            found = true;
            break;
        }
    }

    // randomly choice annother node
    if (!found) {
        do {
            node = rng.rand_int(ntotal);
            if (vt.get(node) && degrees[node] < R && node != id) {
                found = true;
            }
        } while (!found);
    }

    int pos = degrees[node];
    final_graph->at(node, pos) = id; // replace
    degrees[node] += 1;

    return node;
}

void NSG::check_graph() const {
#pragma omp parallel for
    for (int i = 0; i < ntotal; i++) {
        for (int j = 0; j < R; j++) {
            int id = final_graph->at(i, j);
            FAISS_THROW_IF_NOT(id < ntotal && (id >= 0 || id == EMPTY_ID));
        }
    }
}


// 将不同层次的热点及其邻居放入hot_hubs
void NSG::find_hot_hubs_enhence(std::vector<std::unordered_map<idx_t,std::vector<idx_t>>>& hot_hubs,
        idx_t n, std::vector<float>& ratios){
  
    // 存放寻找的热点
    std::vector<std::unordered_set<idx_t>> ses(ratios.size());
    // 将热点放入ses
    find_hot_hubs(ses, n, ratios);
    // 找热点的反向邻居放入hot_hubs
    hot_hubs_new_neighbors(ses,hot_hubs,n);
}


// 通过unordered_map统计每个结点反向连接个数
void NSG::find_hot_hubs(std::vector<std::unordered_set<idx_t>>& ses,
    idx_t n, std::vector<float>& ratios){
    
    // 统计每个结点的入度
    std::unordered_map<idx_t,idx_t> ma;
    for (int i = 0; i < n; i++) {
        int size = 0;
        while (size < R && final_graph->at(i, size) != EMPTY_ID) {
            // i的第size个邻居
            int v1 = final_graph->at(i,size);
            if (v1<0 || v1>n)
                break;
            ma[v1]++;
            size += 1;
        }
    }

    // 频率为first , second:结点编号
    typedef std::pair<int,idx_t> pii;
    std::vector<pii> heat_degrees;
    // 按照热点的热度从大到小排序
    for(auto a : ma){
        heat_degrees.push_back(pii(-a.second,a.first));
    }
    std::sort(heat_degrees.begin(),heat_degrees.end());

    // printf("heat_degrees.size():%d\n",heat_degrees.size());
    
    // 存放不同等级的热点
    int cur=0;
    for (int i = 0; i < ratios.size(); ++i)
    {
      int nb_ratios = n*ratios[i];

      for (int j = cur; j < cur+nb_ratios; ++j)
      {
        // printf("热度: %d\t,热点id:%d\n", heat_degrees[j].first,heat_degrees[j].second);
        ses[i].insert(heat_degrees[j].second);
      }
      cur+=nb_ratios;
    }

}


// 通过unordered_map统计每个结点反向连接个数
void NSG::hot_hubs_new_neighbors(std::vector<std::unordered_set<idx_t>>& ses,
        std::vector<std::unordered_map<idx_t,std::vector<idx_t>>>& hot_hubs,
        idx_t n){
    printf("OK1\n");

    for (int i = 0; i < n; i++) {
        int size = 0;
        while (size < R && final_graph->at(i, size) != EMPTY_ID) {
            // i的第size个邻居
            int v1 = final_graph->at(i,size);
            if (v1<0 || v1>n)
                break;
            // 如果v1为k层的热点，该点作为第k层热点v1的候选邻居
            for (int k = 0; k < ses.size(); ++k)
            {
              if(ses[k].find(v1)!=ses[k].end())
                hot_hubs[k][v1].push_back(i);
            }
            size += 1;
        }
    }
}

// 添加热点反向边到热点
void NSG::add_new_reverse_link_end_enhence(
        std::vector<std::unordered_map<idx_t,std::vector<idx_t>>>& hot_hubs,
        size_t n,std::vector<int>& nb_reverse_neighbors){
        
        // 反向邻居入度
        std::map<int,int> reverseNbsIndegrees;

        size_t cur_pos=n;
        int nb_level = hot_hubs.size();
        for (int ii = 0; ii < nb_level; ii++)
        {
            // 第ii层的热点及其邻居，类型：unordered_map<idx_t,std::vector<idx_t>>
            auto each_level = hot_hubs[ii];
            for(auto hh : each_level){
                // 找到该热点，将反向边插入到后续位置上
                int h_id = hh.first;
                // printf("hh.first:%d ,hh.second.size()%d\n",hh.first,hh.second.size());
                // 将该点hh.first邻居放入se中，防止重复插入
                std::unordered_set<idx_t> se;
                int size = 0;
                while (size < R && final_graph->at(h_id, size) != EMPTY_ID) {
                    // i的第size个邻居
                    int v1 = final_graph->at(h_id,size);
                    if (v1<0 || v1>n)
                        break;
                    se.insert(v1);
                    size += 1;
                }

                auto new_neibor = hh.second;
                // todo：按照距离排序
                // fromNearestToFurther(nb_reverse_neighbors[ii],new_neibor,dis,hh.first);
                // todo : 热点反向边裁边
                // shink_reverse_neighbors(nb_reverse_neighbors[ii],new_neibor,dis,hh.first,n);
                // todo : 按照入度较小排序添加反向边
                // add_reverse_link_by_indegrees(nb_reverse_neighbors[ii],new_neibor,dis,hh.first,n);
                // todo : 按照入度较小出度较大的方式添加反向边
                // add_reverse_link_by_indegrees_and_outdegrees(nb_reverse_neighbors[ii],new_neibor,dis,hh.first,n);
                // todo : 按照热点比例添加热点
                // add_reverse_link_by_pctg(nb_reverse_neighbors[ii],new_neibor,dis,hh.first,n);
                // int m = cum_nneighbor_per_level[1];
                int m = R;
                // 指向热点邻居的指针
                int p=0;
                int size1 = 0;
                while (size1 < R && p < new_neibor.size()) {
                    // i的第size个邻居
                    int v1 = final_graph->at(h_id,size1);
                    if(se.find(new_neibor[p])!=se.end()){
                      p++;
                      continue;
                    }
                    if(v1 < 0)
                        final_graph->at(h_id,size1)=new_neibor[p++];
                    size1 += 1;
                }

                // 记录热点邻居上一行下标
                int pre=h_id;
                // 旧位置全部占用，但是热点邻居还没有放置完毕
                // m每个位置只能分配m-1个数据，一个位置存放指针
                while (p < new_neibor.size() && p<nb_reverse_neighbors[ii]*(m-1))
                {
                    // 新位置所处空间位置
                    // 将新位置的第一个元素，存放旧位置最后位置的元素
                    // printf("cur_pos :%d,end-1: %d\n",cur_pos,hnsw.neighbors[end-1]);
                    final_graph->at(cur_pos,0) = final_graph->at(pre,R-1);
                    
                    /*printf("%d\t,%d\n",cur_pos,hnsw.neighbors[end-1])*/
                    // 就位置的末尾指向新位置
                    final_graph->at(pre,R-1) = cur_pos;
                    // 跳过第一个位置，已经防止了上一行的内容
                    int size2 = 1;
                    while (size2 < R && p < new_neibor.size() && p<nb_reverse_neighbors[ii]*(m-1)) {
                        // i的第size个邻居
                        final_graph->at(cur_pos,size2) = new_neibor[p++];
                        size2 += 1;
                    }
                    pre=cur_pos;
                    cur_pos++;
                }
            }
        }
}


} // namespace faiss
