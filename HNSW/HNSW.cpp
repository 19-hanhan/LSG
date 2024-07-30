/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD+Patents license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include "HNSW.h"
#include<iostream>
#include<fstream>
#include<algorithm>
#include<math.h>
#include<unordered_map>
#include<map>
#include<set>
#include<deque>
#include<random>
#include<atomic>
#include<mutex>

namespace faiss {

using idx_t = Index::idx_t;
using DistanceComputer = HNSW::DistanceComputer;

/**************************************************************
 * HNSW structure implementation
 **************************************************************/

// 获取当前层中的邻居数目
int HNSW::nb_neighbors(int layer_no) const
{
  return cum_nneighbor_per_level[layer_no + 1] -
    cum_nneighbor_per_level[layer_no];
}

// 
void HNSW::set_nb_neighbors(int level_no, int n)
{
  FAISS_THROW_IF_NOT(levels.size() == 0);
  int cur_n = nb_neighbors(level_no);
  for (int i = level_no + 1; i < cum_nneighbor_per_level.size(); i++) {
    cum_nneighbor_per_level[i] += n - cur_n;
  }
}

int HNSW::cum_nb_neighbors(int layer_no) const
{
  return cum_nneighbor_per_level[layer_no];
}

// void HNSW::neigh(Index* ind, idx_t no){
//   for (int i = 0; i < count; ++i)
//   {
//     neighbor_range(0,0,begin,end);

//   }
// }

void HNSW::neighbor_range(idx_t no, int layer_no,
                          size_t * begin, size_t * end) const
{
  // offsets中存放的是当前节点no 元素所有邻居在neighbors中的位置
  // cum_nb_neighbors中存放的是截止到当前层邻居的总和 
  // cum_nb_neighbors(layer_no)表示layer层以上邻居总和
  // cum_nb_neighbors(layer_no+1)表示layer-1层以上邻居总和
  // end - begin就是当前节点（通过offsets找到）在当前层邻居（通过cum_nb_neighbors）的位置
  size_t o = offsets[no];
  *begin = o + cum_nb_neighbors(layer_no);
  *end = o + cum_nb_neighbors(layer_no + 1);
}



long long computer_count=0;
std::atomic<size_t>* in_degree;
std::atomic<size_t>* re_in_degree;
size_t time_n=0;
HNSW::HNSW(int M) : rng(12345) {
  set_default_probas(M,1/log(M));
  // set_default_probas(M,0);
  max_level = -1;
  entry_point = -1;
  efSearch = 16;
  efConstruction = 40;
  upper_beam = 1;
  offsets.push_back(0);
  hot_hubs.resize(1);
  in_degree = (std::atomic<size_t>*)malloc(1e9 * sizeof(std::atomic<size_t>));
  re_in_degree = (std::atomic<size_t>*)malloc(1e9 * sizeof(std::atomic<size_t>));
}


int HNSW::random_level()
{
  double f = rng.rand_float();
  // could be a bit faster with bissection
  for (int level = 0; level < assign_probas.size(); level++) {
    if (f < assign_probas[level]) {
      return level;
    }
    f -= assign_probas[level];
  }
  // happens with exponentially low probability
  return assign_probas.size() - 1;
}

void HNSW::set_default_probas(int M, float levelMult)
{
  int nn = 0;
  cum_nneighbor_per_level.push_back (0);
  for (int level = 0; ;level++) {
    float proba = exp(-level / levelMult) * (1 - exp(-1 / levelMult));
    if (proba < 1e-9) break;
    assign_probas.push_back(proba);
    nn += level == 0 ? 2*M  : M;
    // cum_nneighbor_per_level中内容： 0 M 2M 3M 5M
    cum_nneighbor_per_level.push_back (nn);
  }
}

// neighbor_tables 就是存放存放图结构的表，实际上是一个一维表neighbors
void HNSW::clear_neighbor_tables(int level)
{
  for (int i = 0; i < levels.size(); i++) {
    size_t begin, end;
    neighbor_range(i, level, &begin, &end);
    for (size_t j = begin; j < end; j++) {
      neighbors[j] = -1;
    }
  }
}


void HNSW::reset() {
  max_level = -1;
  entry_point = -1;
  offsets.clear();
  offsets.push_back(0);
  levels.clear();
  neighbors.clear();
}

void HNSW::print_neighbor_stats(int level) const
{
  FAISS_THROW_IF_NOT (level < cum_nneighbor_per_level.size());
  printf("stats on level %d, max %d neighbors per vertex:\n",
         level, nb_neighbors(level));
  size_t tot_neigh = 0, tot_common = 0, tot_reciprocal = 0, n_node = 0;
#pragma omp parallel for reduction(+: tot_neigh) reduction(+: tot_common) \
  reduction(+: tot_reciprocal) reduction(+: n_node)
  for (int i = 0; i < levels.size(); i++) {
    if (levels[i] > level) {
      n_node++;
      size_t begin, end;
      neighbor_range(i, level, &begin, &end);
      std::unordered_set<int> neighset;
      for (size_t j = begin; j < end; j++) {
        if (neighbors [j] < 0) break;
        neighset.insert(neighbors[j]);
      }
      int n_neigh = neighset.size();
      int n_common = 0;
      int n_reciprocal = 0;
      for (size_t j = begin; j < end; j++) {
        storage_idx_t i2 = neighbors[j];
        if (i2 < 0) break;
        FAISS_ASSERT(i2 != i);
        size_t begin2, end2;
        neighbor_range(i2, level, &begin2, &end2);
        for (size_t j2 = begin2; j2 < end2; j2++) {
          storage_idx_t i3 = neighbors[j2];
          if (i3 < 0) break;
          if (i3 == i) {
            n_reciprocal++;
            continue;
          }
          if (neighset.count(i3)) {
            neighset.erase(i3);
            n_common++;
          }
        }
      }
      tot_neigh += n_neigh;
      tot_common += n_common;
      tot_reciprocal += n_reciprocal;
    }
  }
  float normalizer = n_node;
  printf("   nb of nodes at that level %ld\n", n_node);
  printf("   neighbors per node: %.2f (%ld)\n",
         tot_neigh / normalizer, tot_neigh);
  printf("   nb of reciprocal neighbors: %.2f\n", tot_reciprocal / normalizer);
  printf("   nb of neighbors that are also neighbor-of-neighbors: %.2f (%ld)\n",
         tot_common / normalizer, tot_common);
}


void HNSW::fill_with_random_links(size_t n)
{
  int max_level = prepare_level_tab(n);
  RandomGenerator rng2(456);

  for (int level = max_level - 1; level >= 0; level++) {
    std::vector<int> elts;
    for (int i = 0; i < n; i++) {
      if (levels[i] > level) {
        elts.push_back(i);
      }
    }
    printf ("linking %ld elements in level %d\n",
            elts.size(), level);

    if (elts.size() == 1) continue;

    for (int ii = 0; ii < elts.size(); ii++) {
      int i = elts[ii];
      size_t begin, end;
      neighbor_range(i, 0, &begin, &end);
      for (size_t j = begin; j < end; j++) {
        int other = 0;
        do {
          other = elts[rng2.rand_int(elts.size())];
        } while(other == i);

        neighbors[j] = other;
      }
    }
  }
}


int HNSW::prepare_level_tab(size_t n, bool preset_levels)
{
  size_t n0 = offsets.size() - 1;

  if (preset_levels) {
    FAISS_ASSERT (n0 + n == levels.size());
  } else {
    FAISS_ASSERT (n0 == levels.size());
    for (int i = 0; i < n; i++) {
      int pt_level = random_level();
      levels.push_back(pt_level + 1);
    }
  }

  int max_level = 0;
  for (int i = 0; i < n; i++) {
    int pt_level = levels[i + n0] - 1;
    if (pt_level > max_level) max_level = pt_level;
    offsets.push_back(offsets.back() +
                      cum_nb_neighbors(pt_level + 1));
    neighbors.resize(offsets.back(), -1);
  }

  return max_level;
}


/// 只保留0层
int HNSW::prepare_level_tab2(size_t n, bool preset_levels) {
/*    // n0 为已添加点的id（0->n0）
    size_t n0 = offsets.size() - 1;
    // printf("offsets.size():%d\n",offsets.size());
    if (preset_levels) {
        FAISS_ASSERT(n0 + n == levels.size());
    } else {
        FAISS_ASSERT(n0 == levels.size());
        for (int i = 0; i < n; i++) {
            // int pt_level = random_level();
            int pt_level = 0;
            levels.push_back(pt_level + 1);
        }
        
        
    }*/
    // 全部位于为0层
    levels.resize(n,1);
    int max_level = 0;
    for (int i = 0; i < n; i++) {
        /*int pt_level = levels[i + n0] - 1;
        if (pt_level > max_level)
            max_level = pt_level;*/
        int pt_level = 0;
        // 记录每个点在neighbors中的位置
        offsets.push_back(offsets.back() + cum_nb_neighbors(pt_level + 1));
        neighbors.resize(offsets.back(), -1);
    }

    return max_level;
}


// 新索引复制旧索引配置
int HNSW::prepare_level_tab3(HNSW& hnsw1, size_t n, size_t tmp, bool preset_levels) {
  // 复制层次关系
  levels.clear();
  for (size_t i = 0; i < n; i++) {
    levels.push_back(hnsw1.levels[i]);
  }

  int max_level = 0;
  for (size_t i = 0; i < n; i++) {
    // levels中存放的值>=1,因此需要-1计算层次
    int pt_level = levels[i] - 1;
    if (pt_level > max_level) max_level = pt_level;
    // 记录每个点在neighbors中的位置
    offsets.push_back(offsets.back() + cum_nb_neighbors(pt_level + 1));
    neighbors.resize(offsets.back(), -1);
  }

    // 新申请空间只包含0层
    for (size_t i = n; i < tmp; ++i)
    {
      int pt_level = 0;
      // 记录每个点在neighbors中的位置
      offsets.push_back(offsets.back() + cum_nb_neighbors(pt_level + 1));
      neighbors.resize(offsets.back(), -1);
    }
  }


// 索引增强(在原始索引后边申请新的空间,层次为0层)
  void HNSW::prepare_level_tab4(idx_t n, idx_t tmp)
{

  // 新申请空间只包含0层，放在索引后边
    for (idx_t i = n; i < tmp; ++i)
    {
      int pt_level = 0;
      // 记录每个点在neighbors中的位置
      offsets.push_back(offsets.back() + cum_nb_neighbors(pt_level + 1));
      neighbors.resize(offsets.back(), -1);
    }

}
float kPi=3.14159265358979323846264;
float threshold = std::cos(75.0 / 180.0 * kPi);





//   // float dist_v2_q=v2.d;
//   // // printf("%f\n",dist_v2_q);
//   // float cos_v1_v2=(dist_v2_q+dist_v1_q-dist_v1_v2)/2/sqrt(dist_v2_q*dist_v1_q);
//   // if (threshold < cos_v1_v2) {
//   //   good = false;
    
//   //   break;
//   // }
//   // printf("%f\n",threshold);
/** Enumerate vertices from farthest to nearest from query, keep a
 * neighbor only if there is no previous neighbor that is closer to
 * that vertex than the query.
 */
// input : 小根堆,存放候选节点，output存放裁后节点
// std::default_random_engine generator(time(0));
// std::uniform_real_distribution<double> distribution(0.0, 1.0);
std::vector<std::pair<int,int>> shrink_In_degree;
std::mutex g_mutex;
// void HNSW::shrink_neighbor_list(
//   DistanceComputer& qdis,
//   std::priority_queue<NodeDistFarther>& input,
//   std::vector<NodeDistFarther>& output,
//   int max_size,int split)
// {
//   typedef std::pair<int,int> pii;
//   std::vector<pii> temp_in_degree;
//   if(input.size()==max_size+1){
//     while (input.size() > 0) {
      
//       NodeDistFarther v1 = input.top();
//       input.pop();
//       float dist_v1_q = v1.d;
//       bool good = true;
//       // 与output中以存在节点相比较
//       // if(in_degree[v1.id]<max_size/8){
//       // }
//       // else{
//       // for (NodeDistFarther v2 : output) {
//       //   float dist_v1_v2 = qdis.symmetric_dis(v2.id, v1.id);
//       //   // q_v2 为已存在节点，若v1_q距离小于v1_v2距离，则不保存v1作为q的邻居（舍弃三角形较长边）
//       //   if (dist_v1_v2 < dist_v1_q) {
//       //     good = false;
//       //     break;
//       //   }
//       // }
      
//       if(output.size()!=0){
//         if(in_degree[v1.id]>=split){
//           good=false;
//         }
//         else{
//           double randomNumber = distribution(generator);
//           if(randomNumber<0.6){
//             good=false;
//           }
//         }
//       }
//       if (good) {
//         output.push_back(v1);
//         if (output.size() >= max_size) {
//           break;
//         }
//       }
//     }
//   }
//   else{
//     while (input.size() > 0) {
//       NodeDistFarther v1 = input.top();
//       input.pop();
//       float dist_v1_q = v1.d;
//       bool good = true;
//       // 与output中以存在节点相比较
//       // for (NodeDistFarther v2 : output) {
//       //   float dist_v1_v2 = qdis.symmetric_dis(v2.id, v1.id);
//       //   // q_v2 为已存在节点，若v1_q距离小于v1_v2距离，则不保存v1作为q的邻居（舍弃三角形较长边）
//       //   if (dist_v1_v2 < dist_v1_q) {
//       //     good = false;
//       //     // temp_in_degree.push_back(pii(in_degree[v1.id],v1.id));
//       //     break;
//       //   }
//       // }
//       if(output.size()!=0){
//         if(in_degree[v1.id]>=split){
//           good=false;
//         }
//         else{
//           double randomNumber = distribution(generator);
//           if(randomNumber<0.6){
//             good=false;
//           }
//         }
//       }

//       if (good) {
//         output.push_back(v1);
//         if (output.size() >= int(max_size)) {
//           break;
//         }
//       }
//     }
//   }
//   // g_mutex.lock();
//   // shrink_In_degree.insert(shrink_In_degree.end(),temp_in_degree.begin(),temp_in_degree.end());
//   // g_mutex.unlock();
  
// }

void HNSW::shrink_neighbor_list(
  DistanceComputer& qdis,
  std::priority_queue<NodeDistFarther>& input,
  std::vector<NodeDistFarther>& output,
  int max_size,int split,int level)
{
  if(input.size()==max_size+1){
    while (input.size() > 0) {
      NodeDistFarther v1 = input.top();
      input.pop();
      float dist_v1_q = v1.d;
      bool good = true;
      // 与output中以存在节点相比较
      // if(in_degree[v1.id]<max_size/8){
      // }
      // else{
      for (NodeDistFarther v2 : output) {
        float dist_v1_v2 = qdis.symmetric_dis(v2.id, v1.id);
        // q_v2 为已存在节点，若v1_q距离小于v1_v2距离，则不保存v1作为q的邻居（舍弃三角形较长边）
        if (dist_v1_v2 < dist_v1_q) {
          good = false;
          break;
        }

        // 控制夹角
        // float a = v1.d, b = v2.d, c = qdis.symmetric_dis(v2.id, v1.id);
        // float cosc = (a*a + b*b - c*c) / 2*a*b;
        // if (cosc > cos(angle * M_PI / 180)) { // 角度小于angle则抛弃
        //   good = false;
        //   break;
        // }
      }
   
      if (good) {
        output.push_back(v1);
        if (output.size() >= max_size) {
          break;
        }
      }
    }
  }
  else{
    while (input.size() > 0) {
      NodeDistFarther v1 = input.top();
      input.pop();
      float dist_v1_q = v1.d;
      bool good = true;
      // 与output中以存在节点相比较
      // if(in_degree[v1.id]<max_size/8){
      // }
      // else{
      for (NodeDistFarther v2 : output) {
        float dist_v1_v2 = qdis.symmetric_dis(v2.id, v1.id);
        // q_v2 为已存在节点，若v1_q距离小于v1_v2距离，则不保存v1作为q的邻居（舍弃三角形较长边）
        if (dist_v1_v2 < dist_v1_q) {
          good = false;
          break;
        }

        // 控制夹角
        // float a = v1.d, b = v2.d, c = qdis.symmetric_dis(v2.id, v1.id);
        // float cosc = (a*a + b*b - c*c) / 2*a*b;
        // if (cosc > cos(angle * M_PI / 180)) { // 角度小于angle则抛弃
        //   good = false;
        //   break;
        // }
      }
   
      if (good) {
        output.push_back(v1);
        if (output.size() >= max_size) {
          break;
        }
        
      }
    }
  }
  
  
}
//根据最近距离判断，如果有邻居连接了就不要连接了
// void HNSW::shrink_neighbor_list(
//   DistanceComputer& qdis,
//   std::priority_queue<NodeDistFarther>& input,
//   std::vector<NodeDistFarther>& output,
//   int max_size,int split,int level)
// {

//   if(input.size()==max_size+1){
//     typedef std::pair<NodeDistFarther,int> pii;
//     std::unordered_map<int,pii> id_in_count;
//     NodeDistFarther _first=input.top();
//     while(!input.empty()){
//       pii p;
//       p.first=input.top();
//       p.second=0;
//       id_in_count[input.top().id]=p;
//       input.pop();
//     }

//     for(auto temp_id:id_in_count){
//       size_t begin, end;
//       neighbor_range(temp_id.first, level, &begin, &end);
//       for(int i=begin;i<end;i++){
//         size_t n_id=neighbors[i];
//         if(n_id==-1) break;
//         if(id_in_count.find(n_id)!=id_in_count.end())
//           id_in_count[n_id].second++;
//       }
//     }
//     std::vector<pii> result;
//     for(auto temp_id:id_in_count){
//       result.push_back(temp_id.second);
//     }
//     std::sort(result.begin(),result.end(),[](const pii&a,const pii&b){
//       return a.second<b.second;
//     });
//     output.push_back(_first);
    
//     for(auto temp:result){
      
//       if(temp.first.id==_first.id){
//         continue;
//       }
//       if(temp.second>=1){
//         break;
//       }
//       // if(in_degree[temp.first.id]>=split) continue;
//       output.push_back(temp.first);
//       if (output.size() >= max_size) {
//         break;
//       }
//     }
    
//   }
//   else{
//     while (input.size() > 0) {
//       NodeDistFarther v1 = input.top();
//       input.pop();
//       float dist_v1_q = v1.d;
//       bool good = true;
//       // 与output中以存在节点相比较
//       // if(in_degree[v1.id]<max_size/8){
//       // }
//       // else{
//       // for (NodeDistFarther v2 : output) {
//       //   float dist_v1_v2 = qdis.symmetric_dis(v2.id, v1.id);
//       //   // q_v2 为已存在节点，若v1_q距离小于v1_v2距离，则不保存v1作为q的邻居（舍弃三角形较长边）
//       //   if (dist_v1_v2 < dist_v1_q) {
//       //     good = false;
//       //     break;
//       //   }
//       // }
//       if(output.size()!=0){
//         if(in_degree[v1.id]>=split){
//           good=false;
//         }
//       }
//       if (good) {
//         output.push_back(v1);
//         if (output.size() >= max_size/2) {
//           break;
//         }
//       }
//     }
//   }
  
  
// }


// void HNSW::shrink_neighbor_list( 
//   DistanceComputer& qdis,
//   std::priority_queue<NodeDistFarther>& input,
//   std::vector<NodeDistFarther>& output,
//   int max_size)
// {

//   if(input.size()==max_size+1){
//     std::pair<int,int> pii(-1,-1);
//     NodeDistFarther _first=input.top();
//     input.pop();

//     while (input.size() > 0) {
//       NodeDistFarther v1 = input.top();
//       input.pop();
//       float dist_v1_q = v1.d;
//       bool good = true;
     
//       // // 与output中以存在节点相比较
//       // if(output.size()!=0){
        
//       // }
//       // else{
//         // for (NodeDistFarther v2 : output) {
//         //   float dist_v1_v2 = qdis.symmetric_dis(v2.id, v1.id);
//         //   // q_v2 为已存在节点，若v1_q距离小于v1_v2距离，则不保存v1作为q的邻居（舍弃三角形较长边）
//         //   if (dist_v1_v2 < dist_v1_q) {
//         //     good = false;
//         //     break;
//         //   }
//         // }
//       // }
      
//       if (good) {
//         if (output.size() >= max_size-1) {
//           if(in_degree[v1.id]<pii.first){
//             output[pii.second]=v1;
//           }
//           break;
//         }
//         output.push_back(v1);
//         int du=in_degree[v1.id];
//         if(du>pii.first){
//           pii.first=du;
//           pii.second=output.size()-1;
//         }
        
        
        
//       }
//     }
//     output.push_back(_first);
//   }
//   else{
//     while (input.size() > 0) {
//       NodeDistFarther v1 = input.top();
//       input.pop();
//       float dist_v1_q = v1.d;
//       bool good = true;
//       // 与output中以存在节点相比较
      
//       for (NodeDistFarther v2 : output) {
      
//         float dist_v1_v2 = qdis.symmetric_dis(v2.id, v1.id);
//         // q_v2 为已存在节点，若v1_q距离小于v1_v2距离，则不保存v1作为q的邻居（舍弃三角形较长边）
//         if (dist_v1_v2 < dist_v1_q) {
//           good = false;
//           break;
//         }
        
//       }
//       if (good) {
//         output.push_back(v1);
//         if (output.size() >= max_size) {
//           break;
//         }
//       }
//     }
//   }
//     // while (input.size() > 0) {
//     //   NodeDistFarther v1 = input.top();
//     //   input.pop();
//     //   float dist_v1_q = v1.d;
//     //   bool good = true;
//     //   // 与output中以存在节点相比较
//     //   for (NodeDistFarther v2 : output) {
//     //     float dist_v1_v2 = qdis.symmetric_dis(v2.id, v1.id);
//     //     // q_v2 为已存在节点，若v1_q距离小于v1_v2距离，则不保存v1作为q的邻居（舍弃三角形较长边）
//     //     if (dist_v1_v2 < dist_v1_q) {
//     //       good = false;
//     //       break;
//     //     }
//     //   }
//     //   if (good) {
//     //     output.push_back(v1);
//     //     if (output.size() >= max_size) {
//     //       break;
//     //     }
//     //   }
//     // }




// }



// void HNSW::shrink_neighbor_list( 
//   DistanceComputer& qdis,
//   std::priority_queue<NodeDistFarther>& input,
//   std::vector<NodeDistFarther>& output,
//   int max_size)
// {

//   std::unordered_set<idx_t> pre;
//   std::vector<std::pair<int,NodeDistFarther>> _pair;
//   while (input.size() > 0) {
//     NodeDistFarther v1 = input.top();
//     input.pop();
//     float dist_v1_q = v1.d;
//     bool good = true;
//     // 与output中以存在节点相比较
//     for (NodeDistFarther v2 : output) {
//       float dist_v1_v2 = qdis.symmetric_dis(v2.id, v1.id);
//       // q_v2 为已存在节点，若v1_q距离小于v1_v2距离，则不保存v1作为q的邻居（舍弃三角形较长边）
//       if (dist_v1_v2 < dist_v1_q) {
//         good = false;
//         break;
//       }
//     }
//     _pair.push_back(std::pair<int,NodeDistFarther>(in_degree[v1.id],v1));
//     if (good) {
//       output.push_back(v1);
//       pre.insert(v1.id);
//       if (output.size() >= max_size) {
//         break;
//       }
//     }
//   }
//   if(output.size()==max_size) return;
//   while (input.size() > 0){
//     NodeDistFarther v1 = input.top();
//     input.pop();
//     _pair.push_back(std::pair<int,NodeDistFarther>(in_degree[v1.id],v1));

//   }
//   std::sort(_pair.begin(),_pair.end(), 
//   [](const std::pair<int,NodeDistFarther>& a, const std::pair<int,NodeDistFarther>& b){
//     if(a.first==b.first){
//       return a.second.d < b.second.d; 
//     }
//     else{
//       return a.first<b.first;
//     } 
    
//     }); 
//   int i=0;
//   for(auto temp:_pair){
//     NodeDistFarther v1 = temp.second;
//     if(pre.find(v1.id)==pre.end()){
      
      
//       output.push_back(v1);
//       if (output.size() >= max_size) {
//         return;
//       }
//       i++;
//       if(i==8){
//         return;
//       }
      
//     }
//   }

// }


namespace {


using storage_idx_t = HNSW::storage_idx_t;
using NodeDistCloser = HNSW::NodeDistCloser;
using NodeDistFarther = HNSW::NodeDistFarther;


/**************************************************************
 * Addition subroutines
 **************************************************************/

std::atomic<int> flag;
void shrink_neighbor_list(
  DistanceComputer& qdis,
  std::priority_queue<NodeDistCloser>& resultSet1,
  int max_size,storage_idx_t id,HNSW& hnsw,int level)
{
    if (resultSet1.size() < max_size) {
        return;
    }
    std::priority_queue<NodeDistFarther> resultSet;
    std::vector<NodeDistFarther> returnlist;

    // 将大根堆中元素放到小根堆中裁边，结果存入returnlist中
    while (resultSet1.size() > 0) {
      if(in_degree[resultSet1.top().id]<hnsw.maxInDegree){
        resultSet.emplace(resultSet1.top().d, resultSet1.top().id);
      }
      resultSet1.pop();
    }

    hnsw.shrink_neighbor_list(qdis, resultSet, returnlist, max_size,0,level);

    for (NodeDistFarther curen2 : returnlist) {
        resultSet1.emplace(curen2.d, curen2.id);
    }

}
// void shrink_neighbor_list(
//   DistanceComputer& qdis,
//   std::priority_queue<NodeDistCloser>& resultSet1,
//   int max_size,storage_idx_t id,HNSW& hnsw,int level)
// {
//     if (resultSet1.size() < max_size) {
//         return;
//     }
    
//     std::priority_queue<NodeDistFarther> resultSet;
//     std::vector<NodeDistFarther> returnlist;
    
//     // if(resultSet1.size()==400&&max_size==32){
//     //   flag++;
//     // }
//     // // 将大根堆中元素放到小根堆中裁边，结果存入returnlist中
//     // if(max_size==32&&(flag==1||flag==5000||flag==100000||flag==200000||flag==300000||flag==500000||flag==800000)){
//     //   flag++;
//     int split=0;
//     std::vector<int> input_id;
//     if(resultSet1.size()==max_size+1){
//       //反向边的处理
//       while (resultSet1.size() > 0) {
//         storage_idx_t re_id=resultSet1.top().id;
//         resultSet.emplace(resultSet1.top().d, re_id);
//         split+=re_in_degree[re_id];
//         input_id.push_back(re_id);
//         resultSet1.pop();
//       }
//       split/=max_size+1;
//     }
//     else{
//       //正向边的处理
//       typedef std::pair<int,idx_t> pii;
//       std::vector<pii> heat_degrees;
//       // 按照热点的热度排序
//       while (resultSet1.size() > 0) {
//               // if(in_degree[resultSet1.top().id]<88888888){
//         resultSet.emplace(resultSet1.top().d, resultSet1.top().id);
//         heat_degrees.push_back(pii(-in_degree[resultSet1.top().id],resultSet1.top().id));
//               // }
//         input_id.push_back(resultSet1.top().id);
//         resultSet1.pop();
//       }
//       std::sort(heat_degrees.begin(),heat_degrees.end());  
//       for(int i=0;i<heat_degrees.size()*0.05;i++){
//         split+=(-heat_degrees[i].first);
//       }
//       split/=(heat_degrees.size()*0.05);
//     }
//     re_in_degree[id]=split;
 
//     hnsw.shrink_neighbor_list(qdis, resultSet, returnlist, max_size,split,level);

//     for (NodeDistFarther curen2 : returnlist) {
//         resultSet1.emplace(curen2.d, curen2.id);
//     }

// }
/// remove neighbors from the list to make it smaller than max_size
// void shrink_neighbor_list(
//   DistanceComputer& qdis,
//   std::priority_queue<NodeDistCloser>& resultSet1,
//   int max_size,storage_idx_t id)
// {
//     if (resultSet1.size() < max_size) {
//         return;
//     }
    
//     std::priority_queue<NodeDistFarther> resultSet;
//     std::vector<NodeDistFarther> returnlist;
    
//     // if(resultSet1.size()==400&&max_size==32){
//     //   flag++;
//     // }
//     // // 将大根堆中元素放到小根堆中裁边，结果存入returnlist中
//     // if(max_size==32&&(flag==1||flag==5000||flag==100000||flag==200000||flag==300000||flag==500000||flag==800000)){
//     //   flag++;


//     typedef std::pair<int,idx_t> pii;
//     std::vector<pii> heat_degrees;
//     // 按照热点的热度排序
//     while (resultSet1.size() > 0) {
//             // if(in_degree[resultSet1.top().id]<88888888){
//       resultSet.emplace(resultSet1.top().d, resultSet1.top().id);
//       heat_degrees.push_back(pii(-in_degree[resultSet1.top().id],resultSet1.top().id));
//             // }
//       resultSet1.pop();
//     }
//     std::sort(heat_degrees.begin(),heat_degrees.end());  
//     int split=0;
//     // else{
//     //   while (resultSet1.size() > 0) {
//     //           // if(in_degree[resultSet1.top().id]<88888888){
//     //             resultSet.emplace(resultSet1.top().d, resultSet1.top().id);
//     //           // }
//     //           resultSet1.pop();
//     //       }
//     // }
//     // printf("headt_degree%d",int(heat_degrees.size()*0.04));
//     for(int i=0;i<heat_degrees.size()*0.04;i++){
//       split+=(-heat_degrees[i].first);
//     }
//     split/=(heat_degrees.size()*0.04);
//     // if(split!=32)
//       // printf("split%d",split);
//     // if(max_size==12){
//     //   split/=2;
//     // }
//     HNSW::shrink_neighbor_list(qdis, resultSet, returnlist, max_size,split);

//     for (NodeDistFarther curen2 : returnlist) {
//         resultSet1.emplace(curen2.d, curen2.id);
//     }

// }


/// add a link between two elements, possibly shrinking the list
/// of links to make room for it.
void add_link(HNSW& hnsw,
              DistanceComputer& qdis,
              storage_idx_t src, storage_idx_t dest,
              int level)
{
  size_t begin, end;
  hnsw.neighbor_range(src, level, &begin, &end);
  if (hnsw.neighbors[end - 1] == -1) {
    // there is enough room, find a slot to add it
    size_t i = end;
    while(i > begin) {
      if (hnsw.neighbors[i - 1] != -1) break;
      i--;
    }
    hnsw.neighbors[i] = dest;
    // hnsw.in_degree[dest]++;
    //限制入度
  
    if(level==0)
      in_degree[dest]++;
    // if(level==1)
    // in_degree2[dest]++;
    return;
  }

  // otherwise we let them fight out which to keep

  // copy to resultSet...
  std::priority_queue<NodeDistCloser> resultSet;
  resultSet.emplace(qdis.symmetric_dis(src, dest), dest);
 
  for (size_t i = begin; i < end; i++) { // HERE WAS THE BUG
    storage_idx_t neigh = hnsw.neighbors[i];
    resultSet.emplace(qdis.symmetric_dis(src, neigh), neigh);
    
    // hnsw.in_degree[neigh]--;
    //限制入度
    if(level==0)
      in_degree[neigh]--;
  }

  shrink_neighbor_list(qdis, resultSet, end - begin,src,hnsw,level);

  // ...and back
  size_t i = begin;
  while (resultSet.size()) {
    // hnsw.in_degree[resultSet.top().id]++;
    //限制入度
    if(level==0)
      in_degree[resultSet.top().id]++;
    // if(level==1)
    //   in_degree2[resultSet.top().id]++;
    hnsw.neighbors[i++] = resultSet.top().id;
    resultSet.pop();
  }

  // they may have shrunk more than just by 1 element
  while(i < end) {
    hnsw.neighbors[i++] = -1;
  }
}
//修改动态调整efc
/// search neighbors on a single level, starting from an entry point
void search_neighbors_to_add(
  HNSW& hnsw,
  DistanceComputer& qdis,
  std::priority_queue<NodeDistCloser>& results,
  int entry_point,
  float d_entry_point,
  int level,
  VisitedTable &vt,int temp_n)
{
  int expan_efc=0;
  std::priority_queue<NodeDistCloser> hot_result;
  // if((float)temp_n/(float)hnsw.levels.size()<0.5){
    
  // }
  // else{
  //   expan_efc=(int)((float)temp_n/(float)hnsw.levels.size()*300);

  // }
  // expan_efc=(int)((float)temp_n/(float)hnsw.levels.size()*200);
  // top is nearest candidate（小根堆）
  std::priority_queue<NodeDistFarther> candidates;

  NodeDistFarther ev(d_entry_point, entry_point);
  candidates.push(ev);
  int flag=1;
  // std::pair<float,storage_idx_t> in_degree_nn(0,-1);
  if(level==0){
    if(in_degree[entry_point]<hnsw.maxInDegree){
      results.emplace(d_entry_point, entry_point);
    }
    else{
      hot_result.emplace(d_entry_point, entry_point);
    }
  }
  else{
      results.emplace(d_entry_point, entry_point);
  }
  
  // else{
  //     in_degree_nn.first=d_entry_point;
  //     in_degree_nn.second=entry_point;
  // }
  
  vt.set(entry_point);
  

  while (!candidates.empty()) {
    // get nearest
    const NodeDistFarther &currEv = candidates.top();

    if (!results.empty()&&currEv.d > results.top().d) {
      break;
    }
    int currNode = currEv.id;
    candidates.pop();

    // loop over neighbors
    size_t begin, end;
    hnsw.neighbor_range(currNode, level, &begin, &end);
    for(size_t i = begin; i < end; i++) {
      storage_idx_t nodeId = hnsw.neighbors[i];
      if (nodeId < 0) break;
      if (vt.get(nodeId)) continue;
      vt.set(nodeId);

      float dis = qdis(nodeId);
      NodeDistFarther evE1(dis, nodeId);

      if (results.size() < hnsw.efConstruction ||
          results.top().d > dis) {
            candidates.emplace(dis, nodeId);
            //限制入度  64    2M 32
            if(level==0){
              if(in_degree[nodeId]<hnsw.maxInDegree){
                results.emplace(dis, nodeId);
              }
              else{
                hot_result.emplace(dis, nodeId);
              }
            }
            else{
              results.emplace(dis, nodeId);
            }
            
            // else{
            //   if(in_degree_nn.second==-1||in_degree_nn.first>dis){
            //     in_degree_nn.first=dis;
            //     in_degree_nn.second=nodeId;
            //   }
            // }
        
        
        if (results.size() > hnsw.efConstruction) {
          results.pop();
        }
      }
    }
  }
  // if(in_degree_nn.second!=-1){
  if(!hot_result.empty())
  {
    results.push(hot_result.top());
    if (results.size() > hnsw.efConstruction) {
        results.pop();
    }
  }
    
  //   results.emplace(in_degree_nn.first,in_degree_nn.second);
  //   if (results.size() > hnsw.efConstruction+expan_efc) {
  //     results.pop();
  //   }
  // }
  vt.advance();
}


/**************************************************************
 * Searching subroutines
 **************************************************************/

/// greedily update a nearest vector at a given level
void greedy_update_nearest(const HNSW& hnsw,
                           DistanceComputer& qdis,
                           int level,
                           storage_idx_t& nearest,
                           float& d_nearest)
{
  for(;;) {
    storage_idx_t prev_nearest = nearest;

    size_t begin, end;
    hnsw.neighbor_range(nearest, level, &begin, &end);
    // printf("ok3");
    for(size_t i = begin; i < end; i++) {
      // printf("ok5");
      storage_idx_t v = hnsw.neighbors[i];
      
      if (v < 0) break;
      // storage 存放原始向量为n个
      float dis = qdis(v);
      computer_count++;
      // printf("ok7\n");
      if (dis < d_nearest) {
        nearest = v;
        d_nearest = dis;
      }
    }
    // printf("ok4");
    if (nearest == prev_nearest) {
      return;
    }
  }
}


/// greedily update a nearest vector at a given level
void greedy_update_nearest2(const HNSW& hnsw,
                           DistanceComputer& qdis,
                           int level,
                           storage_idx_t& nearest,
                           float& d_nearest,idx_t n)
{
  for(;;) {
    storage_idx_t prev_nearest = nearest;

    size_t begin, end;
    hnsw.neighbor_range(nearest, level, &begin, &end);
    // printf("ok3");
    for(size_t i = begin; i < end; i++) {
      // printf("ok5");
      storage_idx_t v = hnsw.neighbors[i];
      
      if (v < 0 || v>=n) break;
      // storage 存放原始向量为n个
      float dis = qdis(v);
      // printf("ok7\n");
      if (dis < d_nearest) {
        nearest = v;
        d_nearest = dis;
      }
    }
    // printf("ok4");
    if (nearest == prev_nearest) {
      return;
    }
  }
}

}  // namespace


// // 初始化构图入度数组
// void HNSW::init_in_vector(idx_t ntotal){
//   for(int i=0;i<ntotal;i++){
//     in_degree.push_back(0);
//   }
//   // in_degree.resize(ntotal);
// }

void HNSW::static_in_degree_by_construction(idx_t ntotal){
  std::vector<int> in_degree_real(ntotal,0);
  for (size_t i = 0; i < ntotal; ++i)
  {
    size_t begin, end;
    neighbor_range(i, 0, &begin, &end);
    for (size_t j = begin; j < end ;j++) {
      idx_t v1 = neighbors[j];
      if(v1<0||v1>ntotal)
          break;
      in_degree_real[v1]++;
    }
  }
  int diff=0;
  int total=0;
  for(int i=0;i<ntotal;i++){
    if(in_degree[i]!=in_degree_real[i]){
      diff++;
    }
    total+=in_degree[i];
  }
  printf("diff 平均入度:%f\n",(float)total/(float)ntotal);
  printf("diff:%d\n",diff);
  long long s_in_degree=0;
  long long len_10=shrink_In_degree.size()*0.05;
  long long count=0;
  int r_in_diff=0;
  for(int i =0;i<ntotal;i++){
    // if(re_in_degree[i]!=in_degree[i]){
    //   r_in_diff++;
    // }
    // int d=re_in_degree[i];
    // printf("%d\t",d);
    r_in_diff+=re_in_degree[i];
  }
  printf("mytest_pjunrudu%lf\n\n",(float)r_in_diff/(float)ntotal);
  
}





/// Finds neighbors and builds links with them, starting from an entry
/// point. The own neighbor list is assumed to be locked.
/*
  ptdis 距离计算器
  pt_id 当前点id
  nearest 贪心搜索找到的最近邻（ep）
  d_nearest 最近邻的距离
  level 添加的层次
  locks 锁对象数组
  vt 标记数组
*/

void HNSW::add_links_starting_from(DistanceComputer& ptdis,
                                   storage_idx_t pt_id,
                                   storage_idx_t nearest,
                                   float d_nearest,
                                   int level,
                                   omp_lock_t *locks,
                                   VisitedTable &vt,
                                   int temp_n)
{
  std::priority_queue<NodeDistCloser> link_targets;

  // 层内贪心搜索，获取efconstruction个候选节点，efconstruction相当于搜索中控制link_target
  search_neighbors_to_add(*this, ptdis, link_targets, nearest, d_nearest,
                          level, vt,temp_n);

  // but we can afford only this many neighbors
  // M当前层中允许的邻居数
  int M = nb_neighbors(level);

  // 裁边
  ::faiss::shrink_neighbor_list(ptdis, link_targets, M,pt_id,*this,level);
  
  // 添加双向链接
  while (!link_targets.empty()) {
    int other_id = link_targets.top().id;

    omp_set_lock(&locks[other_id]);
    add_link(*this, ptdis, other_id, pt_id, level);
    omp_unset_lock(&locks[other_id]);
    // 添加反向边（如果邻居多了shrink_neighbor_list）

    add_link(*this, ptdis, pt_id, other_id, level);
 

    link_targets.pop();
  }
}


/**************************************************************
 * Building, parallel
 **************************************************************/
//统计顺序关系
std::vector<std::pair<int,idx_t>> time_in_degree;
int id_time=0;
std::vector<std::unordered_set<idx_t>> hotset;

// hnsw 中构图的关键代码
void HNSW::add_with_locks(DistanceComputer& ptdis, int pt_level, int pt_id,
                          std::vector<omp_lock_t>& locks,
                          VisitedTable& vt)
{
  //  greedy search on upper levels
  int temp_n=0;
  storage_idx_t nearest;
#pragma omp critical
  {
    nearest = entry_point;

    time_n++;
    temp_n=time_n;
    if (nearest == -1) {
      max_level = pt_level;
      entry_point = pt_id;
    }  
    time_in_degree.push_back(std::pair<int,idx_t>(id_time++,pt_id));

  }

  if (nearest < 0) {
    return;
  }

  omp_set_lock(&locks[pt_id]);

  int level = max_level; // level at which we start adding neighbors
  float d_nearest = ptdis(nearest);

  // 上层执行greedy_search（找到进入level的ep）
  for(; level > pt_level; level--) {
    greedy_update_nearest(*this, ptdis, level, nearest, d_nearest);
  }

  // 从pt到0层
  for(; level >= 0; level--) {
    add_links_starting_from(ptdis, pt_id, nearest, d_nearest,
                            level, locks.data(), vt,temp_n);
  }

  omp_unset_lock(&locks[pt_id]);

  if (pt_level > max_level) {
    max_level = pt_level;
    entry_point = pt_id;
  }
}

void HNSW::print_time(idx_t n){
    std::unordered_map<idx_t,idx_t> ma;
    std::unordered_map<idx_t,idx_t> out_degree;
    int total_in_degree=0;
    for (size_t i = 0; i < n; ++i)
    {
        size_t begin, end;
        neighbor_range(i, 0, &begin, &end);
        for (size_t j = begin; j < end ;j++) {
            idx_t v1 = neighbors[j];
            if(v1<0||v1>n)
                break;
            ma[v1]++;
            out_degree[i]++;
            total_in_degree++;
        }
    }
    int gap=0.1*n;
    int _count=0;
    int temp_in_degree=0;
    int temp_out_degree=0;
    int hot_count=0;
    for(int i=0;i<n;i++){
      if(_count==gap){
        _count=0;
        printf("平均入度:%f\t热点数量%d\t平均出度：%f\n",float(temp_in_degree)/float(gap),hot_count,float(temp_out_degree)/float(gap));
        temp_in_degree=0;
        temp_out_degree=0;
        hot_count=0;
      }
      _count++;
      idx_t id=time_in_degree[i].second;
      if(ma.find(id)!=ma.end()){
        temp_in_degree+=ma[id];
      }
      if(out_degree.find(id)!=out_degree.end()){
        temp_out_degree+=out_degree[id];
      }
      if(hotset[0].find(id)!=hotset[0].end()||hotset[1].find(id)!=hotset[1].end()){
        hot_count++;
      }
    }
    
}
// 通过unordered_map统计每个结点反向连接个数
void HNSW::hot_hubs_new_neighbors(std::vector<std::unordered_set<idx_t>>& ses,
        std::vector<std::unordered_map<idx_t,std::vector<idx_t>>>& hot_hubs,
        idx_t n,int find_neighbors_mode){
  printf("OK1\n");
  // 选择热点反向边作为热点新增的候选邻居
  if (find_neighbors_mode==0)
  {
    for (size_t i = 0; i < n; ++i)
    {
        size_t begin, end;
        neighbor_range(i, 0, &begin, &end);
        for (size_t j = begin; j < end ;j++) {
            int v1 = neighbors[j];
            if(v1<0||v1>n)
                break;
            // 如果v1为k层的热点，该点作为第k层热点v1的候选邻居 添加反向边的过程
            for (int k = 0; k < ses.size(); ++k)
            {
              if(ses[k].find(v1)!=ses[k].end())
                hot_hubs[k][v1].push_back(i);
            }
        }
    }
  } 
  else if (find_neighbors_mode == 1)
  {
    // for (size_t i = 0; i < n; ++i)
    // {
    //     size_t begin, end;
    //     neighbor_range(i, 0, &begin, &end);
    //     for (size_t j = begin; j < end ;j++) {
    //         int v1 = neighbors[j];
    //         if(v1<0||v1>n)
    //             break;
    //         // 如果v1为k层的热点并且v1和i属于同一类-->
    //         // 该点作为第k层热点v1的候选邻居
    //         for (int k = 0; k < ses.size(); ++k)
    //         {
    //           if(ses[k].find(v1)!=ses[k].end() && clsf[i] == clsf[v1])
    //             hot_hubs[k][v1].push_back(i);
    //         }
    //     }
    // }
  } 
  printf("OK2\n");
  // todo: find_neighbors_mode (其他热点邻居选择方式)

}




// 子图根据热点选择热点反向边
void HNSW::hot_hubs_new_neighbors_subIndex(std::vector<std::unordered_set<idx_t>>& ses,
        std::vector<std::unordered_map<idx_t,std::vector<idx_t>>>& hot_hubs,
        idx_t n,int find_neighbors_mode){
  printf("OK1\n");
  // 选择热点反向边作为热点新增的候选邻居
  if (find_neighbors_mode==0)
  {
    for (size_t i = 0; i < n; ++i)
    {
        // printf("%ld\n",i);
        size_t begin, end;
        neighbor_range(i, 0, &begin, &end);
        for (size_t j = begin; j < end ;j++) {
            // printf("neighbors.size():%ld\t,%ld\n",neighbors.size(),j);
            int v1 = neighbors[j];
            if(v1<0||v1>n)
                break;
            // 如果v1为k层的热点，该点作为第k层热点v1的候选邻居
            for (int k = 0; k < ses.size(); ++k)
            {
              if(ses[k].find(v1)!=ses[k].end())
                hot_hubs[k][v1].push_back(i);
            }
        }
    }
  } 
  printf("OK2\n");
}

//存储原始hnsw入度
std::unordered_map<idx_t,idx_t> indu;

// 通过unordered_map统计每个结点反向连接个数
void HNSW::find_hot_hubs(std::vector<std::unordered_set<idx_t>>& ses,
    idx_t n, std::vector<float>& ratios){
    std::unordered_map<idx_t,idx_t> ma;
    for (size_t i = 0; i < n; ++i)
    {
        size_t begin, end;
        neighbor_range(i, 0, &begin, &end);
        for (size_t j = begin; j < end ;j++) {
            idx_t v1 = neighbors[j];
            if(v1<0||v1>n)
                break;
            ma[v1]++;
        }
    }
    indu=ma;
    // 频率为first , second:结点编号
    typedef std::pair<int,idx_t> pii;
    std::vector<pii> heat_degrees;
    // 按照热点的热度排序
    for(auto a : ma){
        heat_degrees.push_back(pii(-a.second,a.first));
    }
    std::sort(heat_degrees.begin(),heat_degrees.end());

    // printf("heat_degrees.size():%d\n",heat_degrees.size());
    // 存放不同等级的热点
    idx_t cur=0;
    for (idx_t i = 0; i < ratios.size(); ++i)
    {
      idx_t nb_ratios = n*ratios[i];

      for (idx_t j = cur; j < cur+nb_ratios; ++j)
      {
        // printf("热度: %d\t,热点id:%d\n", heat_degrees[j].first,heat_degrees[j].second);
        ses[i].insert(heat_degrees[j].second);
      }
      
      cur+=nb_ratios;
    }
}


// 在邻居的邻居范围内寻找热点
void HNSW::find_hot_hubs_with_neighbors(std::vector<std::unordered_set<idx_t>>& ses,
    idx_t n, std::vector<float>& ratios){
    std::unordered_map<idx_t,idx_t> ma;
    for (size_t i = 0; i < n; ++i)
    {
        size_t begin, end;
        neighbor_range(i, 0, &begin, &end);
        // i 的 邻居
        for (size_t j = begin; j < end ;j++) {
            idx_t v1 = neighbors[j];
            if(v1<0||v1>n)
                break;
            size_t begin1, end1;
            neighbor_range(v1, 0, &begin1, &end1);
            // i邻居的邻居：如果i的邻居的邻居指向i，ma[i]++;
            for (size_t ii = begin1; ii < end1 ;ii++) {
                idx_t v2 = neighbors[ii];
                if(v2<0||v2>n)
                    break;
                size_t begin2, end2;
                neighbor_range(v2, 0, &begin2, &end2);
                for (size_t iii = begin2; iii < end2 ;iii++) {
                  idx_t v3 = neighbors[iii];
                  if(v3<0||v3>n)
                      break;
                  if (v3==i)
                  {
                    ma[i]++;    
                  }
                }  
            }  
        }
    }

    // 频率为first , second:结点编号
    typedef std::pair<int,idx_t> pii;
    std::vector<pii> heat_degrees;
    // 按照热点的热度排序
    for(auto a : ma){
        heat_degrees.push_back(pii(-a.second,a.first));
    }
    std::sort(heat_degrees.begin(),heat_degrees.end());

    // printf("heat_degrees.size():%d\n",heat_degrees.size());
    // 存放不同等级的热点
    idx_t cur=0;
    for (idx_t i = 0; i < ratios.size(); ++i)
    {
      idx_t nb_ratios = n*ratios[i];

      for (idx_t j = cur; j < cur+nb_ratios; ++j)
      {
        // printf("热度: %d\t,热点id:%d\n", heat_degrees[j].first,heat_degrees[j].second);
        ses[i].insert(heat_degrees[j].second);
      }
      
      cur+=nb_ratios;
    }
}


// 全局热点与分类热点相似度
void hubs_similarity(std::vector<std::unordered_set<idx_t>>& ses1,
    std::vector<std::unordered_set<idx_t>>& ses2){

  idx_t cnt = 0;
  idx_t sum = 0;
  for (int i = 0; i < ses1.size(); ++i)
  {
    // 获得第一种方法的第i层热点
    auto se1 = ses1[i];
    sum += se1.size();
    for (int j = 0; j < ses2.size(); ++j)
    {
      // 获得第二种方法第j层热点
      auto se2 = ses2[j];
      for (auto a : se2)
      {
        // 说明热点重合
        if (se1.find(a) != se1.end())
        {
          cnt++;
        }
      }
    }
  }
  printf("热点重合个数： %ld, 热点总个数: %ld, 热点重合率 ：%lf \n", cnt,sum,(cnt*1.0)/sum);
}

 
/*
 * 分类方法从每一类中获取邻居固定比例的热点
 * ma 中存放所有点的反向邻居
 * ratios 存放每一层所占的比例
 * clsf 存放每一类的所属类别
 */
void get_hubs_by_ratios(idx_t n, std::unordered_map<idx_t,idx_t>& ma ,
    std::vector<float>& ratios,std::vector<idx_t>& clsf){
    int k = 1;
    // 统计每一类的个数
    std::vector<int> numsofcls(k);
    for (int i = 0; i < n; ++i)
    {
      numsofcls[clsf[i]]++;
    }

/*    for (int i = 0; i < k; ++i)
    {
      printf("第%d结点数目：%d\n",i,numsofcls[i]);
    }*/


        
    // 结点总比例
    float sum_ratios = 0.0;
    for (int i = 0; i < ratios.size(); ++i)
    {
      sum_ratios += ratios[i];
    }

    typedef std::pair<idx_t,idx_t> pii;
    // 每一类的top_cnts 放入priority_queue
    std::vector<std::priority_queue<pii>> vct(k);
    for(auto a : ma){
      // cls：获取类别，cnts：每一类固定比例下对应的点数
      int cls = clsf[a.first];
      int cnts = numsofcls[cls] * sum_ratios ; 
      if(vct[cls].size() < cnts || -a.second < vct[cls].top().first){
        vct[cls].push(pii(-a.second,a.first));
        if (vct[cls].size() > cnts)
        {
          vct[cls].pop();
        }
      }
    }

    // 将固定比例的热点以及反向邻居个数重新放入ma中
    ma.clear();
    for (int i = 0; i < k; ++i)
    {
      while(!vct[i].empty()){
        ma[vct[i].top().second] = -vct[i].top().first ;
        vct[i].pop();
      }
    }
}


// 聚类寻找热点方法，clsf中存放每个点对应类别
void HNSW::find_hot_hubs_with_classfication(std::vector<std::unordered_set<idx_t>>& ses,
    idx_t n, std::vector<float>& ratios,std::vector<idx_t>& clsf){

    // 统计所有结点的反向边个数
    std::unordered_map<idx_t,idx_t> ma;
    for (size_t i = 0; i < n; ++i)
    {
        size_t begin, end;
        neighbor_range(i, 0, &begin, &end);
        // i 的 邻居
        for (size_t j = begin; j < end ;j++) {
            idx_t v1 = neighbors[j];
            if(v1<0||v1>n)
                break;
            // 说明两者属于同一类，也就是说i是j的反向边
            if (clsf[i] == clsf[v1])
            {
              ma[v1]++;
            }
        }
    }

    // 将热点分类，每类取固定的比例然后放入ma中
    get_hubs_by_ratios(n,ma,ratios,clsf);
    printf("ma.size():%d\n",ma.size());

    // 频率为first , second:结点编号
    typedef std::pair<int,idx_t> pii;
    std::vector<pii> heat_degrees;
    // 按照热点的热度排序
    for(auto a : ma){
        heat_degrees.push_back(pii(-a.second,a.first));
    }
    std::sort(heat_degrees.begin(),heat_degrees.end());

    // 热点分层
    idx_t cur=0;
    for (idx_t i = 0; i < ratios.size(); ++i)
    {
      idx_t nb_ratios = n*ratios[i];

      for (idx_t j = cur; j < cur+nb_ratios; ++j)
      {
        ses[i].insert(heat_degrees[j].second);
      }
      
      cur+=nb_ratios;
    }
}


// 将不同层次的热点及其邻居放入hot_hubs
void HNSW::getHotSet(idx_t n, int len_ratios,const float* ht_hbs_ratios){
    std::vector<float> ratios;
    std::vector<std::unordered_set<idx_t>> ses(len_ratios);
    // printf("ok!!\n");
    for (int i = 0; i < len_ratios; i++)
    { 
        ratios.push_back(ht_hbs_ratios[i]);
    }
  find_hot_hubs(ses,n,ratios);
  hotset=ses;
  // for(int i=0;i<hotset.size();i++){
  //   std::unordered_set<idx_t>::iterator it;
  //   for(it=hotset[i].beg`in();it!=hotset[i].end();it++)
  //     printf("hotset%d\t",*it);
  // }
  // //输出所有点的入度
  // std::unordered_map<idx_t,idx_t> ma;
  // for (size_t i = 0; i < n; ++i)
  // {
  //     size_t begin, end;
  //     neighbor_range(i, 0, &begin, &end);
  //     if(ma.find(i)==ma.end()) ma[i]=0;
  //     for (size_t j = begin; j < end ;j++) {
  //         idx_t v1 = neighbors[j];
  //         if(v1<0||v1>n)
  //             break;
  //         ma[v1]++;
  //     }
  // }
  // indu=ma;
  // // 频率为first , second:结点编号
  // typedef std::pair<int,idx_t> pii;
  // std::vector<pii> heat_degrees;
  // // 按照热点的热度排序
  // for(auto a : ma){
  //     heat_degrees.push_back(pii(a.first,a.second));
  // }
  // std::sort(heat_degrees.begin(),heat_degrees.end());
  // std::string dir="/home/wanghongya/lc/efsearch_time/";
  // dir+="mate_glove";//knn中hnsw热点含量
  // dir+=".tsv";
  // std::ofstream file(dir);
  

  // if(file){
  //   file<<"id"<<"\t"<<"in_degree"<<"\n";
  //   for(int i=0;i<heat_degrees.size();i++){
  //     // if(hotset[0].find(heat_degrees[i].first)!=hotset[0].end()){
  //     //   file<<heat_degrees[i].first<<"\t"<<3<<"\n";
  //     // }
  //     // else if(hotset[1].find(heat_degrees[i].first)!=hotset[1].end()){
  //     //   file<<heat_degrees[i].first<<"\t"<<2<<"\n";

  //     // }
  //     // else{
  //     //   if(heat_degrees[i].second<5){
  //     //     file<<heat_degrees[i].first<<"\t"<<0<<"\n";

  //     //   }
  //     //   else{
  //     //     file<<heat_degrees[i].first<<"\t"<<1<<"\n";
  //     //   }
        
  //     // }
  //     file<<heat_degrees[i].first<<"\t"<<heat_degrees[i].second<<"\n";
      
  //   }
  // }
  // file.close();
}
//存储热点邻居的信息
std::vector<std::vector<int>> nb_in_degree;
void HNSW::print_nb_in_degree(idx_t n,int m,std::string dataName,std::string _type){
  std::unordered_set<idx_t> hot;
  hot.insert(hotset[0].begin(),hotset[0].end());
  hot.insert(hotset[1].begin(),hotset[1].end());
  std::unordered_map<idx_t,idx_t> ma;
  for (size_t i = 0; i < n; ++i)
  {
      if(ma.find(i)==ma.end()) ma[i]=0;
      size_t begin, end;
      neighbor_range(i, 0, &begin, &end);
      for (size_t j = begin; j < end ;j++) {
          idx_t v1 = neighbors[j];
          if(v1<0||v1>n)
              break;
          ma[v1]++;
      }
  }


  indu=ma;
  // 频率为first , second:结点编号
  typedef std::pair<int,idx_t> pii;
  std::vector<pii> heat_degrees;
  // 按照热点的热度排序
  for(auto a : ma){
      heat_degrees.push_back(pii(-a.second,a.first));
  }
  std::sort(heat_degrees.begin(),heat_degrees.end());
  //1.入度， 2.id  3.邻居数量， 4. 邻居平均入度 5 热点数量
  //修正，1.入度 2.id 3.邻居数量 4.邻居平均入度 5 热点数量 6 双向边数量 7 双向热点数量 8 邻居平均
  

  for(int i=0;i<heat_degrees.size();i++){
    pii node=heat_degrees[i];
    std::vector<int> temp(7,0);
    //1.入度;
    temp[0]=node.first*-1;
    //2id
    temp[1]=node.second;
    int nb_count=0;
    int in_degree=0;
    int hot_count=0;
    int double_edge=0;
    int double_edge_hot=0;
    size_t begin, end;
    neighbor_range(node.second, 0, &begin, &end);
    for (size_t j = begin; j < end ;j++) {
        idx_t v1 = neighbors[j];
        if(v1<0||v1>n)
            break;
        nb_count++;
        if(hot.find(v1)!=hot.end()){
          hot_count++;
        }
        in_degree+=ma[v1];

        size_t rbegin,rend;
        neighbor_range(v1, 0, &rbegin, &rend);
        for(size_t r=rbegin;r<rend;r++){
          idx_t rv=neighbors[r];
          if(rv<0||rv>n)
            break;
          if(rv==node.second){
            double_edge++;
            if(hot.find(v1)!=hot.end()){
                double_edge_hot++;
            }
          }
          
        }


    }
    //3邻居数量
    temp[2]=nb_count;
    temp[3]=in_degree/nb_count;
    //热点数量
    temp[4]=hot_count;
  //修正，1.入度 2.id 3.邻居数量 4.邻居平均入度 5 热点数量 6 双向边数量 7 双向热点数量 8 邻居中热点比例 9 邻居中双向边比例 10 邻居中热点双向边比例
    //双向边 float(temp[5])/float(temp[2])
    temp[5]=double_edge;
    //双向边中热点的比例 float(temp[6])/float(temp[2])
    temp[6]=double_edge_hot;
    nb_in_degree.push_back(temp);
  }
  std::string dir="/home/wanghongya/lc/in_degree_ablation/"+dataName+"/";

  dir+="nb_in_degree_"+std::to_string(m)+"m_"+_type;//knn中hnsw热点含量
  dir+=".csv";
  std::ofstream file(dir);
  if(file){
    for(int i=0;i<nb_in_degree.size();i++){
      std::vector<int> temp=nb_in_degree[i];
      file<<temp[0]<<","<<temp[1]<<","<<temp[2]<<","<<temp[3]<<","<<temp[4]<<","<<temp[5]<<","<<temp[6]<<","<<float(temp[4])/float(temp[2])<<","<<float(temp[5])/float(temp[2])<<","<<float(temp[6])/float(temp[2])<<"\n";
    }
  }
  file.close();
  
  
  printf("输出入度状态ok！");


}


//存储热点邻居_hot的信息
std::vector<std::vector<int>> nb_in_degree_hot;
void HNSW::print_nb_in_degree_hot(idx_t n,int m,std::string dataName,std::string _type){
  std::unordered_set<idx_t> hot;
  int diff=0;
  hot.insert(hotset[0].begin(),hotset[0].end());
  hot.insert(hotset[1].begin(),hotset[1].end());
  std::unordered_map<idx_t,idx_t> ma;
  for (size_t i = 0; i < n; ++i)
  {
      std::unordered_set<size_t> nb_set;
      int nb_count_set=0;
      if(ma.find(i)==ma.end()) ma[i]=0;
      size_t begin, end;
      neighbor_range(i, 0, &begin, &end);
      // for (size_t j = begin; j < end ;j++) {
      //     idx_t v1 = neighbors[j];
      //     if(v1<0||v1>n)
      //         break;
      //     ma[v1]++;
      // }
      size_t j = begin;
      while (j < end) {
        int v1 = neighbors[j];
        if (v1 < 0) {
            break;
        }
        // 遇到>=n 搜索末尾位置
        // 将指针指向下一组邻居结点
        if (v1 >= n)
        {
            size_t begin1,end1;
            neighbor_range(v1,0,&begin1,&end1);
            // printf("v1:%d,begin:%ld,end:%ld,begin1:%ld,end1:%ld\n",v1,begin,end,begin1,end1);
            j=begin1;
            end=end1;
            continue;
        }
        nb_count_set++;
        nb_set.insert(v1);
        
        ma[v1]++;
        j++;
    }
    if(nb_count_set!=nb_set.size()){
      diff++;
    }
  }
  printf("重复的邻居数量！%d\n",diff);

  // indu=ma;
  // 频率为first , second:结点编号
  typedef std::pair<int,idx_t> pii;
  std::vector<pii> heat_degrees;
  // 按照热点的热度排序
  for(auto a : ma){
      heat_degrees.push_back(pii(-a.second,a.first));
  }
  std::sort(heat_degrees.begin(),heat_degrees.end());
  //1.入度， 2.id  3.邻居数量， 4. 邻居平均入度 5 热点数量
  //修正，1.入度 2.id 3.邻居数量 4.邻居平均入度 5 热点数量 6 双向边数量 7 双向热点数量 8 邻居平均
  

  for(int i=0;i<heat_degrees.size();i++){
    pii node=heat_degrees[i];
    std::vector<int> temp(7,0);
    //1.入度;
    temp[0]=node.first*-1;
     //2id
    temp[1]=node.second;
    int nb_count=0;
    int in_degree=0;
    int hot_count=0;
    int double_edge=0;
    int double_edge_hot=0;
    size_t begin, end;
    neighbor_range(node.second, 0, &begin, &end);
    size_t j = begin;
    while (j < end) {
      idx_t v1 = neighbors[j];

      if (v1 < 0) {
          break;
      }
      // 遇到>=n 搜索末尾位置
      // 将指针指向下一组邻居结点
      if (v1 >= n)
      {
          size_t begin1,end1;
          neighbor_range(v1,0,&begin1,&end1);
          // printf("v1:%d,begin:%ld,end:%ld,begin1:%ld,end1:%ld\n",v1,begin,end,begin1,end1);
          j=begin1;
          end=end1;
          continue;
      }
      nb_count++;
      if(hot.find(v1)!=hot.end()){
        hot_count++;
      }
      in_degree+=ma[v1];

      size_t rbegin,rend;
      neighbor_range(v1, 0, &rbegin, &rend);
      size_t r=rbegin;
      while (r < rend) {
        idx_t rv=neighbors[r];
        if (rv < 0) {
            break;
        }
        // 遇到>=n 搜索末尾位置
        // 将指针指向下一组邻居结点
        if (rv >= n)
        {
            size_t rbegin1,rend1;
            neighbor_range(rv,0,&rbegin1,&rend1);
            // printf("v1:%d,begin:%ld,end:%ld,begin1:%ld,end1:%ld\n",v1,begin,end,begin1,end1);
            r=rbegin1;
            rend=rend1;
            continue;
        }

        if(rv==node.second){
          double_edge++;
          if(hot.find(v1)!=hot.end()){
              double_edge_hot++;
          }
          // break;
        }
        r++;
      }

      j++;
    }
    //3邻居数量
    temp[2]=nb_count;
    temp[3]=in_degree/nb_count;
    //热点数量
    temp[4]=hot_count;
  //修正，1.入度 2.id 3.邻居数量 4.邻居平均入度 5 热点数量 6 双向边数量 7 双向热点数量 8 邻居中热点比例 9 邻居中双向边比例 10 邻居中热点双向边比例
    //双向边 float(temp[5])/float(temp[2])
    temp[5]=double_edge;
    //双向边中热点的比例 float(temp[6])/float(temp[2])
    temp[6]=double_edge_hot;
    nb_in_degree_hot.push_back(temp);
  }
  std::string dir="/home/wanghongya/lc/in_degree_ablation/"+dataName+"/";

  dir+="nb_in_degree_"+std::to_string(m)+"m_"+_type;//knn中hnsw热点含量
  dir+=".csv";
  std::ofstream file(dir);
  if(file){
    for(int i=0;i<nb_in_degree_hot.size();i++){
      std::vector<int> temp=nb_in_degree_hot[i];
      file<<temp[0]<<","<<temp[1]<<","<<temp[2]<<","<<temp[3]<<","<<temp[4]<<","<<temp[5]<<","<<temp[6]<<","<<float(temp[4])/float(temp[2])<<","<<float(temp[5])/float(temp[2])<<","<<float(temp[6])/float(temp[2])<<"\n";
    }
  }
  file.close();
  
  
  printf("输出入度状态ok！");


}





void HNSW::static_in_degree_by_direct(idx_t n,DistanceComputer& qdis){
  std::unordered_set<idx_t> hot;
  hot.insert(hotset[0].begin(),hotset[0].end());
  hot.insert(hotset[1].begin(),hotset[1].end());
  typedef std::pair<idx_t,idx_t> pii;
  std::unordered_map<idx_t,pii> mp;
  std::unordered_map<idx_t,std::priority_queue<NodeDistFarther>> mp_pq;

  for (size_t i:hot)
  {
    size_t begin, end;
    neighbor_range(i, 0, &begin, &end);
    size_t j = begin;
    //如果是热点将热点的邻居放入hotnbnums
    while (j < end) {
      int v1 = neighbors[j];

      if (v1 < 0) {
          break;
      }
      // 遇到>=n 搜索末尾位置
      // 将指针指向下一组邻居结点
      if (v1 >= n)
      {
          size_t begin1,end1;
          neighbor_range(v1,0,&begin1,&end1);
          // printf("v1:%d,begin:%ld,end:%ld,begin1:%ld,end1:%ld\n",v1,begin,end,begin1,end1);
          j=begin1;
          end=end1;
          continue;
      }
      mp_pq[i].emplace(qdis.symmetric_dis(i, v1),v1);
      j++;
    }
    
    //不是热点放入nomalnbnums
  }

  for(auto i:hot){
      //剪裁前：
      mp[i].first=mp_pq[i].size();
      //剪裁后：
      std::vector<NodeDistFarther> returnlist;
      // shrink_neighbor_list(qdis, mp_pq[i], returnlist, 100000000,100000);
      mp[i].second=returnlist.size();
    }
    int nums=0;
    for(auto i:mp){
      nums++;
      if(i.second.first>1000)
        printf("id:%d\t剪裁前：%d\t剪裁后：%d\t ",i.first,i.second.first,i.second.second);

    }
    printf("\n总热点数量：%d\n",nums);


}



void HNSW::staticKNNHot(idx_t n, int len_ratios,const float* ht_hbs_ratios){
  std::unordered_set<idx_t> hotsetHnsw;
  std::unordered_set<idx_t> hotsetKnn;
  std::unordered_set<idx_t> hotsetTotal;

  std::vector<float> ratios;
  std::vector<std::unordered_set<idx_t>> ses(len_ratios);
    // printf("ok!!\n");
  //存储原始indu
  std::unordered_map<idx_t,idx_t> induHnsw=indu;
  std::unordered_map<idx_t,idx_t> induKnn;
  for (int i = 0; i < len_ratios; i++)
  { 
    ratios.push_back(ht_hbs_ratios[i]);
  }
  find_hot_hubs(ses,n,ratios);
  induKnn=indu;
  for(int i=0;i<hotset.size();i++)
    hotsetHnsw.insert(hotset[i].begin(),hotset[i].end());
  for(int i=0;i<hotset.size();i++)
    hotsetKnn.insert(ses[i].begin(),ses[i].end());
  hotsetTotal.insert(hotsetHnsw.begin(),hotsetHnsw.end());
  hotsetTotal.insert(hotsetKnn.begin(),hotsetKnn.end());
  float result=(float)(hotsetHnsw.size()+hotsetKnn.size()-hotsetTotal.size())/(float)hotsetHnsw.size();
  printf("hotsetHnsw数量：%d\thotsetKnn数量：%d\t二者交集数量:%d\t交集占比：%f\n",hotsetHnsw.size(),hotsetKnn.size(),hotsetTotal.size(),result);


  typedef std::pair<int,idx_t> pii;
  std::vector<pii> heat_degrees;
    // 按照热点的热度排序
  for(auto a : hotsetKnn){
    heat_degrees.push_back(pii(-induKnn[a],a));
  }
  std::sort(heat_degrees.begin(),heat_degrees.end());
  std::string dir="/home/wanghongya/lc/knn/";
  dir+="audioKnn_hnsw";//knn中hnsw热点含量
  dir+=".csv";
  std::ofstream file(dir);
  if(file){
    for(int i=0;i<heat_degrees.size();i++){
      pii tempNode=heat_degrees[i];
      file<<tempNode.second<<","<<-1*tempNode.first<<","<<induHnsw[tempNode.second]<<",";
      // printf("id:%d\thnsw入度:%d\tknn入度:%d\t是否knn热点(1为knn热点)",tempNode.second,-1*tempNode.first,induKnn[tempNode.second]);
      if(hotsetHnsw.find(tempNode.second)!=hotsetHnsw.end()){
        file<<1<<"\n";
        // printf("1\n");
      }
      else{
        file<<0<<"\n";
        // printf("0\n");
      }
    }
  }
  file.close();
  heat_degrees.clear();
    // 按照热点的热度排序
  for(auto a : hotsetHnsw){
    heat_degrees.push_back(pii(-induHnsw[a],a));
  }
  std::sort(heat_degrees.begin(),heat_degrees.end());
  dir="/home/wanghongya/lc/knn/";
  dir+="audioKnn_knn";//hnsw中knn热点含量
  dir+=".csv";
  std::ofstream file1(dir);
  if(file1){
    for(int i=0;i<heat_degrees.size();i++){
      pii tempNode=heat_degrees[i];
      file1<<tempNode.second<<","<<-1*tempNode.first<<","<<induKnn[tempNode.second]<<",";
      // printf("id:%d\thnsw入度:%d\tknn入度:%d\t是否knn热点(1为knn热点)",tempNode.second,-1*tempNode.first,induKnn[tempNode.second]);
      if(hotsetKnn.find(tempNode.second)!=hotsetKnn.end()){
        file1<<1<<"\n";
        // printf("1\n");
      }
      else{
        file1<<0<<"\n";
        // printf("0\n");
      }
    }
  }
  
  file1.close();

  std::unordered_map<idx_t,idx_t> staticKnnIndu;
  for(auto a:induKnn){
    staticKnnIndu[a.second]++;
  }
  std::vector<pii> induAll;
  // 按照热点的热度排序
  for(auto a : staticKnnIndu){
    induAll.push_back(pii(a.first,a.second));
  }
  std::sort(induAll.begin(),induAll.end());
  dir="/home/wanghongya/lc/knn/";
  dir+="audioKnn_indu";//hnsw中knn热点含量
  dir+=".csv";
  std::ofstream file2(dir);
  if(file2){
    for(int i=0;i<induAll.size();i++){
      file2<<induAll[i].first<<","<<induAll[i].second<<"\n";
    }
  }
  file2.close();

  // for(int i=0;i<heat_degrees.size();i++){
  //   pii tempNode=heat_degrees[i];
  //   printf("id:%d\thnsw入度:%d\tknn入度:%d\t是否knn热点(1为knn热点)",tempNode.second,-1*tempNode.first,induKnn[tempNode.second]);
  //   if(hotsetKnn.find(tempNode.second)!=hotsetKnn.end()){
  //     printf("1\n");
  //   }
  //   else{
  //     printf("0\n");
  //   }
  // }

}

void writeCsv(std::vector<std::pair<idx_t,idx_t>> csv,std::string dir){
  std::ofstream file1(dir);
  if(file1){
      for(idx_t i=0;i<csv.size();i++){
        file1 <<csv[i].first<<","<<csv[i].second<< "\n" ;
      }
  }
  file1.close();
}
void writeCsv_v(std::vector<std::vector<idx_t>> csv,std::string dir){
  std::ofstream file1(dir);
  if(file1){
      for(idx_t i=0;i<csv.size();i++){
        for(int j=0;j<csv[i].size();j++){
          if(j!=csv[i].size()-1){
            file1<<csv[i][j]<<",";
          }
          else{
            file1<<csv[i][j]<<"\n";
          }
        }
      }
  }
  file1.close();
}
// std::vector<std::pair<int,int>> getDiff(std::unordered_map<int,int> oldIn,std::unordered_map<int,int> newIn){

// }


// std::vector<std::pair<idx_t,std::vector<idx_t>>> inNodesHnsw;
// std::vector<std::pair<idx_t,std::vector<idx_t>>> inNodeHot;


//存储原始节点每个点的入度
std::unordered_map<idx_t,idx_t> node_indu;
//存储原始入度数量分布
std::unordered_map<idx_t,idx_t> allInNums_g;
//存储热点
std::unordered_set<idx_t> inNode_hot;


void HNSW::find_inNode_Hot(idx_t n,int len_ratios,const float* ht_hbs_ratios){
  std::unordered_map<idx_t,idx_t> indu_all_hot;
  for (size_t i = 0; i < n; ++i)
  {
    size_t begin, end;
    neighbor_range(i, 0, &begin, &end);
    size_t j = begin;
    //如果是热点将热点的邻居放入hotnbnums
    while (j < end) {
      int v1 = neighbors[j];

      if (v1 < 0) {
          break;
      }
      // 遇到>=n 搜索末尾位置
      // 将指针指向下一组邻居结点
      if (v1 >= n)
      {
          size_t begin1,end1;
          neighbor_range(v1,0,&begin1,&end1);
          // printf("v1:%d,begin:%ld,end:%ld,begin1:%ld,end1:%ld\n",v1,begin,end,begin1,end1);
          j=begin1;
          end=end1;
          continue;
      }
      indu_all_hot[v1]++;
      j++;
    }
    //不是热点放入nomalnbnums
  }

  //node_indu原始的每个节点入度
  std::unordered_map<idx_t,idx_t> same_nums;
  for (size_t i = 0; i < n; ++i)
  {
    if(indu_all_hot.find(i)!=indu_all_hot.end()&&node_indu.find(i)!=node_indu.end()){
      if(indu_all_hot[i]==node_indu[i]){
        same_nums[indu_all_hot[i]]++;
      }
    }
  }
  std::unordered_map<idx_t,idx_t> indu_all_hot_nums;
  idx_t temp0=0;
  for(auto m:indu_all_hot){
    indu_all_hot_nums[m.second]++;
    temp0++;
  }
  indu_all_hot_nums[0]=n-temp0;
  typedef std::pair<idx_t,idx_t> pii;
  std::vector<pii> indu_all_hot_vector;
  for(auto m:indu_all_hot_nums){
    indu_all_hot_vector.push_back(pii(m.first,m.second));
  }
  std::sort(indu_all_hot_vector.begin(),indu_all_hot_vector.end(),[](pii a,pii b){return a.first<b.first;});
  std::string dirbase="/home/wanghongya/lc/in_degree_all/sift1m/";
  std::string dir1=dirbase;
  // dir+=std::to_string(efSearch);
  dir1+="hot10m_induAll.csv";
  writeCsv(indu_all_hot_vector,dir1);
  // std::vector<std::vector<idx_t>> indu_all_hot_diff;
  // std::set<idx_t> diff_set;
  // for(auto m:indu_all_hot_nums){
  //   diff_set.insert(m.first);
  // }
  // for(auto m:allInNums_g){
  //   diff_set.insert(m.first);
  // }
  // for(auto m:diff_set){
  //   if(indu_all_hot_nums.find(m)==indu_all_hot_nums.end()){
  //     indu_all_hot_nums[m]=0;
  //   }
  //   if(allInNums_g.find(m)==allInNums_g.end()){
  //     allInNums_g[m]=0;
  //   }
  // }
  // for(auto m:diff_set){
  //   //第一个为id， 第二个为原始，第三个为hot  第二个为hnsw入度和hot入度的不同，第三个为交集大小（即不变的数目），第四个为降低多少（hnsw原始入度-交集大小），第五个为有多少点新增上来（hot数量-原始-降低）
  //   std::vector<idx_t> temp;
  //   temp.push_back(m);
  //   temp.push_back(allInNums_g[m]);//原始
  //   temp.push_back(indu_all_hot_nums[m]);//hot
  //   temp.push_back(indu_all_hot_nums[m]-allInNums_g[m]);//不同
  //   if(same_nums.find(m)!=same_nums.end()){
  //     temp.push_back(same_nums[m]);
  //   }
  //   else{
  //     temp.push_back(0);
  //     same_nums[m]=0;
  //   }
  //   temp.push_back(allInNums_g[m]-same_nums[m]);//降低
  //   temp.push_back(indu_all_hot_nums[m]-allInNums_g[m]+allInNums_g[m]-same_nums[m]);//增加
    
  //   indu_all_hot_diff.push_back(temp);
  // }

  // std::string dir2=dirbase;
  // // dir+=std::to_string(efSearch);
  // dir2+="indu_all_diff.csv";
  // writeCsv_v(indu_all_hot_diff,dir2);
  

}

//统计每一个点的入度邻居，存储在inNodesHnsw中
void HNSW::find_inNode_Hnsw(idx_t n,int len_ratios,const float* ht_hbs_ratios){
  std::unordered_map<idx_t,idx_t> ma;
  for (size_t i = 0; i < n; ++i)
  {
      size_t begin, end;
      neighbor_range(i, 0, &begin, &end);
      for (size_t j = begin; j < end ;j++) {
          idx_t v1 = neighbors[j];
          if(v1<0||v1>n)
              break;
          ma[v1]++;
      }
  }
  node_indu=ma;
  typedef std::pair<idx_t,idx_t> pii;
  std::vector<pii> inNodesHnsw;
  std::unordered_map<idx_t,idx_t> allInNums;
  idx_t all0=0;
  for(auto a:ma){
    inNodesHnsw.push_back(pii(a.first,a.second));
    allInNums[a.second]++;
    all0++;
  }
  allInNums[0]=n-all0;
  std::sort(inNodesHnsw.begin(),inNodesHnsw.end(),[](pii a,pii b){return a.second>b.second;});
  idx_t lastHot=0;
  for(int i=0;i<len_ratios;i++){
    lastHot+=ht_hbs_ratios[i]*n;
  }
  //热点导致的入度
  std::unordered_map<idx_t,idx_t> hotIn;
  //普通点导致的入度
  std::unordered_map<idx_t,idx_t> normalIn;
  //存储热点
  std::unordered_set<idx_t> hotSet;
  
  for(idx_t i=0;i<n;i++){
    if(i<=lastHot)
      hotSet.insert(inNodesHnsw[i].first);
    
  }
  
  inNode_hot=hotSet;
  allInNums_g=allInNums;
  for (size_t i = 0; i < n; ++i)
  {
      size_t begin, end;
      neighbor_range(i, 0, &begin, &end);
      if(hotSet.find(i)!=hotSet.end()){
        for (size_t j = begin; j < end ;j++) {
          idx_t v1 = neighbors[j];
          if(v1<0||v1>n)
              break;
          hotIn[v1]++;
        }
      }
      else{
        for (size_t j = begin; j < end ;j++) {
          idx_t v1 = neighbors[j];
          if(v1<0||v1>n)
              break;
          normalIn[v1]++;
        }
      }
  }
  //存储每个点入度的map，first为入度，second为数量；
  std::unordered_map<idx_t,idx_t> hotInNums;
  std::unordered_map<idx_t,idx_t> normalInNums;

  for(auto m:normalIn){
    normalInNums[m.second]++;
  }
  for(auto m:hotIn){
    hotInNums[m.second]++;
  }
  typedef std::pair<idx_t,idx_t> p;
  //std::vector<p> hotInVec;
  std::vector<p> normalInVec;
  std::vector<p> allInVec;
  std::vector<p> all_normal_diff;
  std::unordered_set<idx_t> diff_set;
  int temp0=0;
  // for(auto m:hotInNums){
  //   hotInVec.push_back(p(m.first,m.second));
  // }
  for(auto m:normalInNums){
    normalInVec.push_back(p(m.first,m.second));
    temp0+=m.second;
    diff_set.insert(m.first);
  }
  int indu0=n-temp0;
  normalInNums[0]=indu0;
  for(auto m:allInNums){
    allInVec.push_back(p(m.first,m.second));
    diff_set.insert(m.first);
  }
  for(auto m:diff_set){
    if(normalInNums.find(m)==normalInNums.end()){
      normalInNums[m]=0;
    }
    if(allInNums.find(m)==allInNums.end()){
      allInNums[m]=0;
    }
  }
  for(auto m:allInNums){
    all_normal_diff.push_back(p(m.first,m.second-normalInNums[m.first]));
  }

  
  normalInVec.insert(normalInVec.begin(),p(0,indu0));


  std::sort(allInVec.begin(),allInVec.end(),[](p a,p b){return a.first<b.first;});
  //std::sort(hotInVec.begin(),hotInVec.end(),[](p a,p b){return a.first<b.first;});
  std::sort(normalInVec.begin(),normalInVec.end(),[](p a,p b){return a.first<b.first;});
  std::sort(all_normal_diff.begin(),all_normal_diff.end(),[](p a,p b){return a.first<b.first;});
  std::string dirbase="/home/wanghongya/lc/in_degree_all/sift1m/";
  
  // std::string dir1=dirbase;
  // // dir+=std::to_string(efSearch);
  // dir1+="induAll.csv";
  writeCsv(allInVec,dirbase+"base10m_induAll.csv");
  // for(int temp=0;temp<allInVec.size();temp++){
  //   printf("%d\t%d\n",allInVec[temp].first,allInVec[temp].second);
  // }
  // std::string dir2="/home/wanghongya/lc/test/";
  // // dir+=std::to_string(efSearch);
  // dir2+="induHot.csv";
  // std::ofstream file2(dir2);
  // writeCsv(hotInVec,dir2);
  // std::string dir3="induNormal.csv";
  // // dir+=std::to_string(efSearch);
  // writeCsv(normalInVec,dirbase+dir3);

  // std::string dir4="induDiff.csv";
  // // dir+=std::to_string(efSearch);
  // writeCsv(all_normal_diff,dirbase+dir4);


}

void HNSW::statichotpercent(idx_t n){
  int nomal=0;
  int hot=0;
  std::unordered_map<idx_t,idx_t> ma;

  for (size_t i = 0; i < n; ++i)
  {
      size_t begin, end;
      neighbor_range(i, 0, &begin, &end);
      for (size_t j = begin; j < end ;j++) {
          idx_t v1 = neighbors[j];
          if(v1<0||v1>n)
              break;
          ma[v1]++;
      }
  }
  indu=ma;
  // 频率为first , second:结点编号
  typedef std::pair<int,idx_t> pii;
  std::vector<pii> heat_degrees;
  // 按照热点的热度排序
  for(auto a : ma){
    if(hotset[0].find(a.first)!=hotset[0].end()||hotset[1].find(a.first)!=hotset[1].end()){
        hot+=a.second;
    }
    else{
      nomal+=a.second;
    }
    heat_degrees.push_back(pii(-a.second,a.first));
  }
  std::sort(heat_degrees.begin(),heat_degrees.end());
  printf("热点入度：%d\t普通点入度：%d\t热点入度占比：%.2f\n",hot,nomal,(float)hot/(float)(hot+nomal));

  //输出所有点的入度
  //统计分析修正后的入度

  std::unordered_set<idx_t> hotnbnums;
  std::unordered_set<idx_t> nomalnbnums;
  for (size_t i = 0; i < n; ++i)
  {
    size_t begin, end;
    neighbor_range(i, 0, &begin, &end);
    //如果是热点将热点的邻居放入hotnbnums
    if(hotset[0].find(i)!=hotset[0].end()||hotset[1].find(i)!=hotset[1].end()){
        for (size_t j = begin; j < end ;j++) {
          idx_t v1 = neighbors[j];
          if(v1<0||v1>n)
              break;
          if(hotnbnums.find(v1)==hotnbnums.end()){
          hotnbnums.insert(v1);
        }
      }
    }
    //不是热点放入nomalnbnums
    else{
      for (size_t j = begin; j < end ;j++) {
          idx_t v1 = neighbors[j];
          if(v1<0||v1>n)
              break;
          if(nomalnbnums.find(v1)==nomalnbnums.end()){
          nomalnbnums.insert(v1);
        }
      }
    }
  }
  hot=0;
  nomal=0;
  int total=0;
  hot=hotnbnums.size();
  nomal=nomalnbnums.size();
  nomalnbnums.insert(hotnbnums.begin(),hotnbnums.end());
  total=nomalnbnums.size();
  printf("总结点数：%d\t热点连接节点数量：%d\t普通点连接节点数量：%d\t二者连接总节点数量：%d\t热点链接占比：%.2f\t普通点连接占比%.2f\n",n,hot,nomal,total,(float)hot/(float)n,(float)nomal/(float)n);
 
}



void HNSW::statcihotlinknums(idx_t n){
  std::unordered_set<idx_t> hotnbnums;
  std::unordered_set<idx_t> nomalnbnums;
  int hotlink=0;
  int nomallink=0;
  for (size_t i = 0; i < n; ++i)
  {
    size_t begin, end;
    neighbor_range(i, 0, &begin, &end);
    size_t j = begin;
    //如果是热点将热点的邻居放入hotnbnums
    if(hotset[0].find(i)!=hotset[0].end()||hotset[1].find(i)!=hotset[1].end()){
      while (j < end) {
        int v1 = neighbors[j];

        if (v1 < 0) {
            break;
        }
        // 遇到>=n 搜索末尾位置
        // 将指针指向下一组邻居结点
        if (v1 >= n)
        {
            size_t begin1,end1;
            neighbor_range(v1,0,&begin1,&end1);
            // printf("v1:%d,begin:%ld,end:%ld,begin1:%ld,end1:%ld\n",v1,begin,end,begin1,end1);
            j=begin1;
            end=end1;
            continue;
        }
        if(hotset[0].find(v1)!=hotset[0].end()||hotset[1].find(v1)!=hotset[1].end()){
          hotlink++;
        }
        else{
          nomallink++;
        }
        if(hotnbnums.find(v1)==hotnbnums.end()){
          hotnbnums.insert(v1);
        }
        j++;
      }
    }
    //不是热点放入nomalnbnums
    else{
      while (j < end) {
        int v1 = neighbors[j];

        if (v1 < 0) {
            break;
        }
        // 遇到>=n 搜索末尾位置
        // 将指针指向下一组邻居结点
        if (v1 >= n)
        {
            size_t begin1,end1;
            neighbor_range(v1,0,&begin1,&end1);
            // printf("v1:%d,begin:%ld,end:%ld,begin1:%ld,end1:%ld\n",v1,begin,end,begin1,end1);
            j=begin1;
            end=end1;
            continue;
        }
        if(hotset[0].find(v1)!=hotset[0].end()||hotset[1].find(v1)!=hotset[1].end()){
          hotlink++;
        }
        else{
          nomallink++;
        }
        if(nomalnbnums.find(v1)==nomalnbnums.end()){
          nomalnbnums.insert(v1);
        }
        j++;
      }
    }

  }
  int hot=0;
  int nomal=0;
  int total=0;
  hot=hotnbnums.size();
  nomal=nomalnbnums.size();
  nomalnbnums.insert(hotnbnums.begin(),hotnbnums.end());
  total=nomalnbnums.size();
  printf("热点入度：%d\t普通点入度：%d\t热点入度占比%.2f\n",hotlink,nomallink,(float)hotlink/(float)(hotlink+nomallink));
  printf("总结点数：%d\t热点连接节点数量：%d\t普通点连接节点数量：%d\t二者连接总节点数量：%d\t热点链接占比：%.2f\t普通点连接占比%.2f\n",n,hot,nomal,total,(float)hot/(float)n,(float)nomal/(float)n);
}

//获取knn个最近邻
void HNSW::getKnn(DistanceComputer& qdis,idx_t id,idx_t n,int k,std::priority_queue<NodeDistCloser>& initial_list){
  for (idx_t i = 0; i < n; i++){
    if(id==i) continue;
    float dis=qdis(i);
    NodeDistCloser tempNode(dis,i);
    if(initial_list.size()<k||initial_list.top().d>dis){
      initial_list.emplace(tempNode);
      if(initial_list.size()>k) initial_list.pop();
    }
  }
}



void HNSW::find_hot_hubs_enhence(std::vector<std::unordered_map<idx_t,std::vector<idx_t>>>& hot_hubs,
        idx_t n, std::vector<float>& ratios, int find_hubs_mode,
        int find_neighbors_mode){
  
  // 存放寻找的热点
  std::vector<std::unordered_set<idx_t>> ses(ratios.size());

  // 按照入度寻找热点,放入se中
  if(find_hubs_mode==0){
    find_hot_hubs(ses, n, ratios);
  }else if (find_hubs_mode == 1)
  {
    // 根据邻居的邻居方式统计热点
    find_hot_hubs_with_neighbors(ses,n,ratios);
  }else if (find_hubs_mode == 2)
  {
    // 聚类之后，在每类中利用反向邻居个数统计热点
    // find_hot_hubs_with_classfication(ses,n,ratios,clsf);    
  }/*else if (find_hubs_mode == 3)
  {
    //两种方式结合的热点选择方法
    std::vector<std::unordered_set<idx_t>> ses1(ratios.size());
    std::vector<std::unordered_set<idx_t>> ses2(ratios.size());
    std::vector<std::unordered_map<idx_t,std::vector<idx_t>>>& hot_hubs1;
    std::vector<std::unordered_map<idx_t,std::vector<idx_t>>>& hot_hubs2;
    find_hot_hubs(ses1, n, ratios);
    find_hot_hubs_with_classfication(ses2,n,ratios,clsf);
    hot_hubs_new_neighbors(ses1,hot_hubs1,n,0,clsf);
    hot_hubs_new_neighbors(ses2,hot_hubs2,n,1,clsf);
    // 合并两个map，并去重
    // todo : 比例如何分配？ 需要从中选取ratios比例的热点
    // 如何选择
    for(auto a : hot_hubs1){
      ma[a.first] = a.second;
    }

    for(auto a : hot_hubs2){
      if (ma.find(a.first) == ma.end())
      {
        ma[a.first] = a.second;      
      }
    }

  }*/
  // todo : 其他搜索方式

  // 全局与分类相似度计算
/*  std::vector<std::unordered_set<idx_t>> ses1(ratios.size());
  std::vector<std::unordered_set<idx_t>> ses2(ratios.size());
  find_hot_hubs(ses1, n, ratios);
  
  find_hot_hubs_with_classfication(ses2,n,ratios,clsf);
  
  hubs_similarity(ses1,ses2);
*/
  

  // 拿到热点(ses中存放)去寻找热点邻居

  if (find_hubs_mode != 3)
  {
    hot_hubs_new_neighbors(ses,hot_hubs,n,find_neighbors_mode);
  }
  
/*  printf("hot_hubs.size(): %d\n", hot_hubs.size());
  printf("ses.size() %d\n",ses.size());
  auto se=ses[0];
  for(auto a:se){
    printf("热点：%d\n", a);
  }

  auto ma = hot_hubs[0];
  for(auto a : ma){
    for(auto b:a.second){
      printf("热点反向边 ： %d\n", b);
    }
  }
*/ 
}


// 子图寻找热点及其反向边
void HNSW::find_hot_hubs_enhence_subIndex(std::vector<std::unordered_map<idx_t,std::vector<idx_t>>>& hot_hubs,
        idx_t n, std::vector<float>& ratios, int find_hubs_mode,
        int find_neighbors_mode){
  
  // 存放寻找的热点
  std::vector<std::unordered_set<idx_t>> ses(ratios.size());

  // 按照入度寻找热点,放入se中
  if(find_hubs_mode==0){
    find_hot_hubs(ses, n, ratios);
  }

  // 拿到热点(ses中存放)去寻找热点邻居
  hot_hubs_new_neighbors_subIndex(ses,hot_hubs,n,find_neighbors_mode);

  
/*  printf("hot_hubs.size(): %d\n", hot_hubs.size());
  printf("ses.size() %d\n",ses.size());
  auto se=ses[0];
  for(auto a:se){
    printf("热点：%d\n", a);
  }

  auto ma = hot_hubs[0];
  for(auto a : ma){
    for(auto b:a.second){
      printf("热点反向边 ： %d\n", b);
    }
  }
*/ 
}







 
/*
* 为热点添加反向边添加到索引末尾
* 首先，将原始位置填满
* 如果填满将最后一个位置指向索引末尾位置，将剩余邻居填入该位置
*/
void HNSW::fromNearestToFurther(int nb_reverse_nbs_level,std::vector<idx_t>& new_neibor,
                DistanceComputer& dis,idx_t hot_hub){
  std::vector<std::pair<float,idx_t>> v;
  for (auto a : new_neibor)
  {
      // printf("%f\t", dis.symmetric_dis(hot_hub,a));
      v.push_back(std::make_pair(dis.symmetric_dis(hot_hub,a),a));
  }
  // printf("----------------\n");
  std::sort(v.begin(),v.end());
  // std::reverse(v.begin(),v.end());

/*  for (int i = 0; i < v.size(); ++i)
  {
    printf("%f\t", v[i].first );
  }

  printf("----------------\n");*/

  // 0 层分配10个，其他层分配2*M个
  int cnt = nb_reverse_nbs_level;
  cnt = cnt>=10 ? 10 : 2;
  // 0层最大邻居数
  int m = cum_nneighbor_per_level[1];

  // 防止不够new_neibor中不够cnt*m
  new_neibor.resize(std::min((int)new_neibor.size(),cnt*m));

  int nn=v.size();
  std::shuffle(v.begin(), v.begin() + nn/2, std::mt19937{ std::random_device{}() });
  std::shuffle(v.begin()+nn/2, v.begin() + nn, std::mt19937{ std::random_device{}() });
  // 全部放入new_neighbor
  /*for (int i = 0; i < new_neibor.size(); ++i)
  {
      new_neibor[i]=v[i].second;
  }*/

  // todo: 从近距离（前一半）中选择4/5*2 M,从远距离(后一半)中选取1/5*2M;

/*  for (int i = 0; i < nn; ++i)
  {
    printf("%f\t", v[i].first );
  }
  printf("----------------\n");*/

  int cur=0;
  for (int i = 0; i < nn&&cur<new_neibor.size()*4/5; ++i)
  {
    new_neibor[cur++] = v[i].second;
  }

  for (int i = nn/2; i < nn&&cur<new_neibor.size(); ++i)
  {
    new_neibor[cur++] = v[i].second;
  }

  new_neibor.resize(cur);

}


void HNSW::shink_reverse_neighbors(int nb_reverse_nbs_level,std::vector<idx_t>& new_neibor,
                DistanceComputer& dis,idx_t hot_hub,size_t n){
  
  // 将反向邻居按照距离放入小根堆
  std::priority_queue<NodeDistFarther> input;
  std::vector<NodeDistFarther> returnlist;
  std::vector<idx_t> tmp;
  for (auto a : new_neibor)
  {
      // first:与热点距离， second：热点反向邻居
      input.emplace(dis.symmetric_dis(hot_hub,a), (int)a);
      tmp.push_back(a);
  }

  // 0 层分配10个，其他层分配m个
  int cnt = nb_reverse_nbs_level;
  cnt = cnt>=10 ? 10 : 1;
  // 0层最大邻居数
  int m = cum_nneighbor_per_level[1];

  std::unordered_set<idx_t> vis;
  // 将已存在的正向邻居加入
  size_t begin, end;
  neighbor_range(hot_hub, 0, &begin, &end);
  for (size_t i = begin; i < end; i++)
  {
    idx_t v1 = neighbors[i];
    if(v1 < 0||v1 > n)
      break;
    // 类型不一致，NodeDistFarther数据超过int会出问题
    returnlist.push_back(NodeDistFarther(dis.symmetric_dis(hot_hub,v1),(int)v1));
    vis.insert(v1);
  }

  // 原始邻居长度
  int len = returnlist.size();
  // HNSW::shrink_neighbor_list(dis, input, returnlist, (cnt+2)*m,100000);
  // 重新组建new_neighbor
  new_neibor.resize(0);
  // 可以放满就放裁边的结果，放不满就放随机邻居
  for (int i = len; i < returnlist.size(); ++i)
  {
    new_neibor.push_back((idx_t)returnlist[i].id);
    vis.insert((idx_t)returnlist[i].id);
  }

  // todo:将new_neibor空位插入随机邻居(加够m)
  for (int i = 0; i < tmp.size()&&new_neibor.size()<cnt*m; ++i)
  {
    if (vis.find(tmp[i])==vis.end())
    {
      vis.insert(tmp[i]);
      new_neibor.push_back(tmp[i]);
    }
  }

}


void HNSW::add_new_reverse_link_end_enhence(
        std::vector<std::unordered_map<idx_t,std::vector<idx_t>>>& hot_hubs,
        size_t n,DistanceComputer& dis,std::vector<int>& nb_reverse_neighbors){
  
        size_t cur_pos=n;
        int nb_level = hot_hubs.size();
        for (int ii = 0; ii < nb_level; ii++)
        {
          // 第ii层的热点及其邻居，类型：unordered_map<idx_t,std::vector<idx_t>>
          auto each_level = hot_hubs[ii];
          for(auto hh : each_level){
              // 找到该热点，将反向边插入到后续位置上
              size_t begin, end;
              neighbor_range(hh.first, 0, &begin, &end);
              // printf("hh.first:%d ,hh.second.size()%d\n",hh.first,hh.second.size());
              // 将该点hh.first邻居放入ma中，防止重复插入
              std::unordered_set<idx_t> se;
              for (size_t i = begin; i < end; i++)
              {
                  idx_t v1 = neighbors[i];
                  if(v1 < 0||v1 > n)
                      break;
                  se.insert(v1);
              }

              auto new_neibor = hh.second;
              // todo：按照距离排序
              // fromNearestToFurther(nb_reverse_neighbors[ii],new_neibor,dis,hh.first);
              // todo : 热点反向边裁边
              // shink_reverse_neighbors(nb_reverse_neighbors[ii],new_neibor,dis,hh.first,n);
              // 指向热点邻居的指针
              int m = cum_nneighbor_per_level[1];
              int p=0;
              // 将新邻居插入到后边,添加cnt个邻居（5，10，15，20，25）
              int cnt=0;
              for (size_t j = begin; 
                  j < end && p < new_neibor.size() /*&& cnt < 15*/;) {
                  int v1 = neighbors[j];
                  // 如果该邻居已经存在，就不用重复添加该邻居
                  if(se.find(new_neibor[p])!=se.end()){
                      p++;
                      continue;
                  }
                  if(v1 < 0)
                      neighbors[j]=new_neibor[p++];
                  j++;
              }

              size_t pre=end;
              // 旧位置全部占用，但是热点邻居还没有放置完毕
              // m每个位置只能分配m-1个数据，一个位置存放指针
              while (p < new_neibor.size() && p<nb_reverse_neighbors[ii]*(m-1))
              {
                  // 新位置所处空间位置
                  size_t begin1, end1;
                  neighbor_range(cur_pos, 0, &begin1, &end1);
                  // 将新位置的第一个元素，存放旧位置最后位置的元素
                  // printf("cur_pos :%d,end-1: %d\n",cur_pos,hnsw.neighbors[end-1]);
                  neighbors[begin1] = neighbors[pre-1]; 

                  /*printf("%d\t,%d\n",cur_pos,hnsw.neighbors[end-1])*/
                  // 就位置的末尾指向新位置
                  neighbors[pre-1] = cur_pos;

                  cur_pos++;
                  // 新位置都是-1
                  for (size_t k = begin1+1;
                        k < end1 && p < new_neibor.size()&& p<nb_reverse_neighbors[ii]*(m-1);)
                  {
                      if(se.find(new_neibor[p])!=se.end()){
                        p++;
                        continue;
                      }
                      neighbors[k++]=new_neibor[p++];
                  }
                  pre=end1;
              }
          }
      }
 

}


// 将热点的邻居连向热点
void HNSW::add_links_to_hubs(
        std::vector<std::unordered_map<idx_t,std::vector<idx_t>>>& hot_hubs,
        size_t n){

  // 热点层次遍历
  for (int l = 0; l < hot_hubs.size(); ++l)
  {
      // 第i层热点遍历
      for(auto& hub : hot_hubs[l]){
        // 热点id
        idx_t hubID = hub.first;
        std:std::vector<idx_t> nbs = hub.second;

        // 遍历热点邻居连向热点
        for (int i = 0; i < nbs.size(); ++i)
        {
          idx_t cur = nbs[i];
          // 邻居的邻居是否已经满 ？ 满去掉最后一个连接，不满直接添加到后边
          size_t begin,end;
          neighbor_range(cur,0,&begin,&end);
          // 不满
          if (neighbors[end-1]==-1)
          {
            for (size_t i = begin; i < end; ++i)
            {
              if (neighbors[i] == -1)
              {
                neighbors[i] = hubID;
                break;
              }
            }
          }else {
            neighbors[end-1] = hubID;
          }
        }
      }
  }

}



/** Do a BFS on the candidates list */

int HNSW::search_from_candidates(
  DistanceComputer& qdis, int k,
  idx_t *I, float *D,
  MinimaxHeap& candidates,
  VisitedTable& vt,
  int level, int nres_in) const
{
  int nres = nres_in;
  int ndis = 0;

  for (int i = 0; i < candidates.size(); i++) {
    idx_t v1 = candidates.ids[i];
    float d = candidates.dis[i];
    FAISS_ASSERT(v1 >= 0);
    if (nres < k) {
      faiss::maxheap_push(++nres, D, I, d, v1);
    } else if (d < D[0]) {
      faiss::maxheap_pop(nres--, D, I);
      faiss::maxheap_push(++nres, D, I, d, v1);
    }
    vt.set(v1);
  }

  int nstep = 0;

  while (candidates.size() > 0) {
    float d0 = 0;

    int v0 = candidates.pop_min(&d0);

    size_t begin, end;
    neighbor_range(v0, level, &begin, &end);

    for (size_t j = begin; j < end; j++) {
      int v1 = neighbors[j];
      if (v1 < 0) break;
      if (vt.get(v1)) {
        continue;
      }
      vt.set(v1);
      ndis++;
      float d = qdis(v1);
      if (nres < k) {
        faiss::maxheap_push(++nres, D, I, d, v1);
      } else if (d < D[0]) {
        faiss::maxheap_pop(nres--, D, I);
        faiss::maxheap_push(++nres, D, I, d, v1);
      }
      candidates.push(v1, d);
    }

    nstep++;
    // efsearch 控制搜索结束
    if (nstep > efSearch) {
      break;
    }
  }

  if (level == 0) {
// #pragma omp critical
    {
      hnsw_stats.n1 ++;
      if (candidates.size() == 0) {
        hnsw_stats.n2 ++;
      }
      hnsw_stats.n3 += ndis;
    }
  }
  faiss::maxheap_reorder (k, D, I);
  D[0]=ndis;
  D[1]=nstep;

  return nres;
}



/*
* 在search_from_candidates的基础上，修改了候选队列的数据结构
* 由原本的线性扫描换为优先队列
*/
/*
template<typename T>
using MinHeap = std::priority_queue<T, std::vector<T>, std::greater<T>>;
  // 小根堆，存放候选节点

int HNSW::search_from_candidates_optimize(
  DistanceComputer& qdis, int k,
  idx_t *I, float *D,
  MinimaxHeap& candidates,
  VisitedTable& vt,
  int level, int nres_in) const
{
  int nres = nres_in;
  int ndis = 0;

  float d0 = 0;
  int v0 = candidates.pop_min(&d0);

  MinHeap<Node> candidate_set;
  candidate_set.emplace(d0, v0);

  int nstep = 0;

  while (candidate_set.size() > 0) {
    float d0;
    storage_idx_t v0;
    std::tie(d0, v0) = candidate_set.top();

    candidate_set.pop();
    
    size_t begin, end;
    neighbor_range(v0, level, &begin, &end);

    for (size_t j = begin; j < end; j++) {
      int v1 = neighbors[j];
      if (v1 < 0) break;
      if (vt.get(v1)) {
        continue;
      }
      vt.set(v1);
      ndis++;
      float d = qdis(v1);
      if (nres < k) {
        faiss::maxheap_push(++nres, D, I, d, v1);
      } else if (d < D[0]) {
        faiss::maxheap_pop(nres--, D, I);
        faiss::maxheap_push(++nres, D, I, d, v1);
      }
      candidate_set.emplace(d, v1);
    }

    nstep++;
    // efsearch 控制搜索结束
    if (nstep > efSearch) {
      break;
    }
  }

  if (level == 0) {
// #pragma omp critical
    {
      hnsw_stats.n1 ++;
      if (candidates.size() == 0) {
        hnsw_stats.n2 ++;
      }
      hnsw_stats.n3 += ndis;
    }
  }

  D[0]=ndis;
  D[1]=nstep;

  return nres;
}
*/
/**************************************************************
 * Searching
 **************************************************************/
//统计距离：
int _count=0;

void HNSW::resetCount(){
    
    _count=0;
    computer_count=0;
}

long long HNSW::get_computer_count(){
  return computer_count;
}
std::vector<std::vector<std::pair<int,float>>> disresult(10000,std::vector<std::pair<int,float>>()); //存储距离和层次
std::vector<std::vector<int>> isHothubResult(10000,std::vector<int>());//存储时否热点
std::vector<std::vector<int>> typeHothubResult(10000,std::vector<int>());//存储每一个点的种类
std::vector<std::vector<int>> discountResult(10000,std::vector<int>());//存储每一个点的计算次数

void HNSW::getSearchDis(){
    std::string dir="/home/wanghongya/lc/test/";
    dir+=std::to_string(efSearch);
    dir+=".csv";
    std::ofstream file(dir);
    if(file){
        for(int i=0;i<disresult.size();i++){
          //输出距离计算次数（第一行）
            for(int j=0;j<discountResult[i].size();j++){
                if(j!=discountResult[i].size()-1){
                    file <<discountResult[i][j] << "," ;
                }
                else{
                    file <<discountResult[i][j] <<"\n";
                }
            }
            //输出每个点的特征（第二列）
            for(int j=0;j<typeHothubResult[i].size();j++){
                if(j!=typeHothubResult[i].size()-1){
                    file <<typeHothubResult[i][j] << "," ;
                }
                else{
                    file <<typeHothubResult[i][j] <<"\n";
                }
            }
            //输出是否是热点（第三列）
            for(int j=0;j<isHothubResult[i].size();j++){
                if(j!=isHothubResult[i].size()-1){
                    file <<isHothubResult[i][j] << "," ;
                }
                else{
                    file <<isHothubResult[i][j] <<"\n";
                }
            }
            //输出层次（第四列）
            for(int j=0;j<disresult[i].size();j++){
                if(j!=disresult[i].size()-1){
                    file <<disresult[i][j].first << "," ;
                }
                else{
                    file <<disresult[i][j].first <<"\n";
                }
            }
            //输出距离（第五列）
            for(int j=0;j<disresult[i].size();j++){
                if(j!=disresult[i].size()-1){
                    file <<disresult[i][j].second << "," ;
                }
                else{
                    file <<disresult[i][j].second <<"\n";
                }
            }
        }
    }
    file.close();
    
    int times=0;
    int minTimes=INT_MAX;
    int maxTimes=0;
    for(int i=0;i<disresult.size();i++){
        minTimes=std::min(minTimes,(int)disresult[i].size());
        maxTimes=std::max(maxTimes,(int)disresult[i].size());
        times+=(int)disresult[i].size();
    }
    printf("bounded:%d\tefsearch:%d\tminTimes:%d\tmaxTimes:%d\taverageTimes:%f\n",search_bounded_queue,efSearch,minTimes,maxTimes,times/(float)disresult.size());
    int dtimes=0;
    int dminTimes=INT_MAX;
    int dmaxTimes=0;
    for(int i=0;i<discountResult.size();i++){
      int discounttemp=0;
      for(int j=0;j<discountResult[i].size();j++){
        discounttemp+=discountResult[i][j];
      }
      dminTimes=std::min(dminTimes,(int)discounttemp);
      dmaxTimes=std::max(dmaxTimes,(int)discounttemp);
      dtimes+=(int)discounttemp;
    }
    printf("bounded:%d\tefsearch:%d\tminCounts:%d\tmaxCounts:%d\taverageCounts:%f\n",search_bounded_queue,efSearch,dminTimes,dmaxTimes,dtimes/(float)discountResult.size());

}
void HNSW::getSearchDisFlat(){
    std::string dir="/home/wanghongya/lc/testFlat/";
    dir+=std::to_string(efSearch);
    dir+=".csv";
    std::ofstream file(dir);
    if(file){
        for(int i=0;i<disresult.size();i++){
          //输出距离计算次数（第一行）
            for(int j=0;j<discountResult[i].size();j++){
                if(j!=discountResult[i].size()-1){
                    file <<discountResult[i][j] << "," ;
                }
                else{
                    file <<discountResult[i][j] <<"\n";
                }
            }
            for(int j=0;j<isHothubResult[i].size();j++){
                if(j!=isHothubResult[i].size()-1){
                    file <<isHothubResult[i][j] << "," ;
                }
                else{
                    file <<isHothubResult[i][j] <<"\n";
                }
            }
            for(int j=0;j<disresult[i].size();j++){
                if(j!=disresult[i].size()-1){
                    file <<disresult[i][j].first << "," ;
                }
                else{
                    file <<disresult[i][j].first <<"\n";
                }
            }
            for(int j=0;j<disresult[i].size();j++){
                if(j!=disresult[i].size()-1){
                    file <<disresult[i][j].second << "," ;
                }
                else{
                    file <<disresult[i][j].second <<"\n";
                }
            }
        }
    }
    file.close();
    
    int times=0;
    int minTimes=INT_MAX;
    int maxTimes=0;
    for(int i=0;i<disresult.size();i++){
        minTimes=std::min(minTimes,(int)disresult[i].size());
        maxTimes=std::max(maxTimes,(int)disresult[i].size());
        times+=(int)disresult[i].size();
    }
    printf("bounded:%d\tefsearch:%d\tminTimes:%d\tmaxTimes:%d\taverageTimes:%f\n",search_bounded_queue,efSearch,minTimes,maxTimes,times/(float)disresult.size());
    int dtimes=0;
    int dminTimes=INT_MAX;
    int dmaxTimes=0;
    for(int i=0;i<discountResult.size();i++){
      int discounttemp=0;
      for(int j=0;j<discountResult[i].size();j++){
        discounttemp+=discountResult[i][j];
      }
      dminTimes=std::min(dminTimes,(int)discounttemp);
      dmaxTimes=std::max(dmaxTimes,(int)discounttemp);
      dtimes+=(int)discounttemp;
    }
    printf("bounded:%d\tefsearch:%d\tminCounts:%d\tmaxCounts:%d\taverageCounts:%f\n",search_bounded_queue,efSearch,dminTimes,dmaxTimes,dtimes/(float)discountResult.size());
}

//距离统计

// int stop_count_base=0;
template<typename T>
using MaxHeap = std::priority_queue<T, std::vector<T>, std::less<T>>;
template<typename T>
using MinHeap = std::priority_queue<T, std::vector<T>, std::greater<T>>;
//修改此处打印输出距离
MaxHeap<HNSW::Node> HNSW::search_from(
  const Node& node,
  DistanceComputer& qdis,
  int ef,
  VisitedTable *vt) const
{
  // 大根堆，存放结果
  MaxHeap<Node> top_candidates;
  // 小根堆，存放候选节点
  MinHeap<Node> candidate_set;

  top_candidates.push(node);
  candidate_set.push(node);

  vt->set(node.second);

  float lower_bound = node.first;

  // 记录访问点的个数
  storage_idx_t ndis=0;

  while (!candidate_set.empty()) {
    float d0;
    storage_idx_t v0;
    std::tie(d0, v0) = candidate_set.top();

    // 当前候选中最小，大于结果集中的最大
    if (d0 > lower_bound) {
      break;
    }

    candidate_set.pop();

    size_t begin, end;
    neighbor_range(v0, 0, &begin, &end);

    for (size_t j = begin; j < end; ++j) {
      int v1 = neighbors[j];

      if (v1 < 0) {
        break;
      }
      if (vt->get(v1)) {
        continue;
      }

      vt->set(v1);

      float d1 = qdis(v1);
      computer_count++;
      if (top_candidates.top().first > d1 || top_candidates.size() < ef) {
        candidate_set.emplace(d1, v1);
        top_candidates.emplace(d1, v1);

        if (top_candidates.size() > ef) {
          top_candidates.pop();
        }
        // 更新结果集中的最大
        lower_bound = top_candidates.top().first;
      }

    }
  }
  // 传递访问点的个数：寻找当前队列中的距离最大值，然后push(lower_bound+1,ndis)
  // 如果top_candidates为空会报错
  FAISS_THROW_IF_NOT_MSG(!top_candidates.empty()," top_candidates is empty！");
  lower_bound = top_candidates.top().first;
  top_candidates.emplace(lower_bound+1,ndis);
  return top_candidates;
}
// MaxHeap<HNSW::Node> HNSW::search_from(
//   const Node& node,
//   DistanceComputer& qdis,
//   int ef,
//   VisitedTable *vt) const
// {
//   MaxHeap<Node> top_candidates;
//   MinHeap<Node> candidate_set;
//   MaxHeap<Node> hot_top_candidates;
//   int flag=1;
//   if(in_degree[node.second]<88888888){
//     top_candidates.push(node);
//   }
//   else{
//     flag=0;
//     hot_top_candidates.push(node);
//   }
//   candidate_set.push(node);
//   vt->set(node.second);

//   float lower_bound = node.first;

//   while (!candidate_set.empty()) {
//     float d0;
//     storage_idx_t v0;
//     std::tie(d0, v0) = candidate_set.top();
//     if (flag&&d0 > lower_bound) {
//       break;
//     }
//     candidate_set.pop();
//     size_t begin, end;
//     neighbor_range(v0, 0, &begin, &end);
//     for (size_t j = begin; j < end; ++j) {
//       int v1 = neighbors[j];
//       if (v1 < 0) {
//         break;
//       }
//       if (vt->get(v1)) {
//         continue;
//       }
//       vt->set(v1);
//       float d1 = qdis(v1);
//       computer_count++;
//       if ((!top_candidates.empty()&&top_candidates.top().first > d1) || top_candidates.size() < ef) {
//         candidate_set.emplace(d1, v1);
//         if(in_degree[v1]<88888888){
//           top_candidates.emplace(d1, v1);
//         }
//         else{
//           hot_top_candidates.emplace(d1, v1);
//         }
//         if (top_candidates.size() > ef) {
//           top_candidates.pop();
//         }
//         if(!top_candidates.empty()){
//           flag=1;
//           lower_bound = top_candidates.top().first;
//         }
          
//       }
//     }
//   }

//   while(!hot_top_candidates.empty()){
//     top_candidates.push(hot_top_candidates.top());
//     hot_top_candidates.pop();
//     if (top_candidates.size() > ef) {
//       top_candidates.pop();
//     }
//   }
  
//   return top_candidates;
// }
// MaxHeap<HNSW::Node> HNSW::search_from(
//   const Node& node,
//   DistanceComputer& qdis,
//   int ef,
//   VisitedTable *vt) const
// {
//   MaxHeap<Node> top_candidates;
//   MinHeap<Node> candidate_set;
//   MaxHeap<Node> hot_top_candidates;
//   int flag=1;
//   if(in_degree[node.second]<88888888){
//     top_candidates.push(node);
//   }
//   else{
//     flag=0;
//     hot_top_candidates.push(node);
//   }
//   // top_candidates.push(node);
//   candidate_set.push(node);
//   // //距离统计
//   // std::unordered_map<idx_t,int> mp;
//   // std::vector<int> isHothub;
//   // std::vector<int> discount;
//   // int _times=0;
//   // mp[node.second]=_times++;
//   // std::vector<std::pair<int,float>> distemp;
//   // //距离统计
//   vt->set(node.second);

//   float lower_bound = node.first;

//   while (!candidate_set.empty()) {
//     float d0;
//     storage_idx_t v0;
//     std::tie(d0, v0) = candidate_set.top();

//     if (d0 > lower_bound) {
      
//       // printf("%ld\t%d\t",v0_in_degree,candidate_set.size());
//       // stop_count_base++;
//       break;
//     }

//     candidate_set.pop();

//     size_t begin, end;
//     neighbor_range(v0, 0, &begin, &end);
//     // //距离统计
//     // distemp.push_back(std::pair<int,float>(mp[v0],d0));
//     // _times=mp[v0]+1;
//     // bool ishotbool=false;
//     // for(int hb=0;hb<hotset.size();hb++){
//     //   if(hotset[hb].find(v0)!=hotset[hb].end()){
//     //     ishotbool=true;
//     //     break;
//     //   }
//     // }
//     //     //bool ishotbool=neighbors[end-1]>=n?true:false;//修正热点的判断，将热点的存储起来
//     // isHothub.push_back(ishotbool);
//     // //距离统计
//     // int discounttemp=0;
//     for (size_t j = begin; j < end; ++j) {
//       int v1 = neighbors[j];

//       if (v1 < 0) {
//         break;
//       }
//       if (vt->get(v1)) {
//         continue;
//       }

//       vt->set(v1);

//       float d1 = qdis(v1);
//       computer_count++;
//       // discounttemp++;
//       if (top_candidates.top().first > d1 || top_candidates.size() < ef) {
//         candidate_set.emplace(d1, v1);
//         // if(in_degree[node.second]<88888888){
//           top_candidates.emplace(d1, v1);
//         // }
//         // else{
//         //   hot_top_candidates.emplace(d1, v1);
//         // }

//         // top_candidates.emplace(d1, v1);
//         // //距离统计
//         // mp[v1]=_times;
//         // //距离统计
//         if (top_candidates.size() > ef) {
//           top_candidates.pop();
//         }

//         lower_bound = top_candidates.top().first;
//       }
//     }
//     // discount.push_back(discounttemp);
//   }
//   // //距离统计
//   // isHothubResult[_count]=isHothub;
//   // discountResult[_count]=discount;
//   // disresult[_count++]=distemp;
//   // //距离统计
//   // while(!hot_top_candidates.empty()){
//   //   top_candidates.push(hot_top_candidates.top());
//   //   hot_top_candidates.pop();
//   //   if (top_candidates.size() > ef) {
//   //     top_candidates.pop();
//   //   }
//   // }
  
//   return top_candidates;
// }



// 在search_from 的基础上解决k对unbounded的影响

template<typename T>
using MaxHeap = std::priority_queue<T, std::vector<T>, std::less<T>>;
template<typename T>
using MinHeap = std::priority_queue<T, std::vector<T>, std::greater<T>>;

MaxHeap<HNSW::Node> HNSW::search_from_addk(
  const Node& node,
  DistanceComputer& qdis,
  int ef,
  VisitedTable *vt ,int k) const
{
  // 大根堆，存放结果
  MaxHeap<Node> top_candidates;
  // 小根堆，存放候选节点
  MinHeap<Node> candidate_set;
  // 大根堆存放返回结果
  MaxHeap<Node> results;

  top_candidates.push(node);
  candidate_set.push(node);
  results.push(node);

  vt->set(node.second);

  float lower_bound = node.first;

  // 记录访问点的个数
  storage_idx_t ndis=0;

  while (!candidate_set.empty()) {
    float d0;
    storage_idx_t v0;
    std::tie(d0, v0) = candidate_set.top();

    // 当前候选中最小，大于结果集中的最大
    if (d0 > lower_bound) {
      break;
    }

    candidate_set.pop();

    size_t begin, end;
    neighbor_range(v0, 0, &begin, &end);

    for (size_t j = begin; j < end; ++j) {
      int v1 = neighbors[j];

      if (v1 < 0) {
        break;
      }
      if (vt->get(v1)) {
        continue;
      }

      vt->set(v1);

      float d1 = qdis(v1);
      ndis++;
      if (top_candidates.top().first > d1 || top_candidates.size() < ef) {
        candidate_set.emplace(d1, v1);
        top_candidates.emplace(d1, v1);
        if (top_candidates.size() > ef) {
            top_candidates.pop();
        }

        // 将放入top_candidates中的数据，放到result中存储（目的：top_candidates大小不变的情况下，还能取到k个值）。
        if (results.top().first > d1 || results.size() < k)
            results.emplace(d1,v1);

        if(results.size()>k)
            results.pop();

        lower_bound = top_candidates.top().first;
      }

    }
  }
  // 传递访问点的个数：寻找当前队列中的距离最大值，然后push(lower_bound+1,ndis)
  // 如果top_candidates为空会报错
  FAISS_THROW_IF_NOT_MSG(!top_candidates.empty()," top_candidates is empty！");
  lower_bound = results.top().first;
  results.emplace(lower_bound+1,ndis);
  return results;
}



// 在search_from 的基础上解决k对unbounded的影响，通过set优化
// 用unordered_set存储top_candidate中元素，然后将candidate中元素和candidate_set_pop元素添加到top_candidate中

template<typename T>
using MaxHeap = std::priority_queue<T, std::vector<T>, std::less<T>>;
template<typename T>
using MinHeap = std::priority_queue<T, std::vector<T>, std::greater<T>>;

MaxHeap<HNSW::Node> HNSW::search_from_addk_v2(
  const Node& node,
  DistanceComputer& qdis,
  int ef,
  VisitedTable *vt ,int k) const
{
  // 大根堆，存放结果
  MaxHeap<Node> top_candidates;
  // 小根堆，存放候选节点
  MinHeap<Node> candidate_set;
  //存放candidate_set弹出元素
  std::vector<Node> candidate_set_pop;

  std::unordered_set<idx_t> top_candidates_set;

  top_candidates.push(node);
  candidate_set.push(node);
  top_candidates_set.emplace(node.second);

  vt->set(node.second);

  float lower_bound = node.first;

  // 记录访问点的个数
  storage_idx_t ndis=0;

  while (!candidate_set.empty()) {
    float d0;
    storage_idx_t v0;
    std::tie(d0, v0) = candidate_set.top();

    // 将candidate_set弹出元素暂存
    candidate_set_pop.push_back(candidate_set.top());
    // 当前候选中最小，大于结果集中的最大
    if (d0 > lower_bound) {
      break;
    }

    candidate_set.pop();

    size_t begin, end;
    neighbor_range(v0, 0, &begin, &end);

    for (size_t j = begin; j < end; ++j) {
      int v1 = neighbors[j];

      if (v1 < 0) {
        break;
      }
      if (vt->get(v1)) {
        continue;
      }

      vt->set(v1);

      float d1 = qdis(v1);
      ndis++;
      if (top_candidates.top().first > d1 || top_candidates.size() < ef) {
        candidate_set.emplace(d1, v1);
        top_candidates.emplace(d1, v1);
        top_candidates_set.emplace(v1);

        if (top_candidates.size() > ef) {
            candidate_set_pop.push_back(top_candidates.top());
            top_candidates_set.erase(top_candidates.top().second);
            top_candidates.pop();
        }

        lower_bound = top_candidates.top().first;
      }

    }
  }


  //first : add to top_candidate
  int n = candidate_set_pop.size();
  for (int i = 0; i < n ; ++i)
  {
      if (!top_candidates_set.count(candidate_set_pop[i].second))
      {
          /*top_candidates.emplace(candidate_set_pop[i]);
          top_candidates_set.emplace(candidate_set_pop[i].second);*/
          candidate_set.emplace(candidate_set_pop[i]);
      }
      
  }

  // second : add to top_candidate
  while(!candidate_set.empty() && top_candidates.size()<k){
      if (!top_candidates_set.count(candidate_set.top().second))
      {
          top_candidates.emplace(candidate_set.top());
          top_candidates_set.emplace(candidate_set.top().second);
      }
      candidate_set.pop();
  }

  lower_bound = top_candidates.top().first;

  top_candidates.emplace(lower_bound+1,ndis);

  return top_candidates;
}





// upper_beam == 1 greedy_update_nearest寻找初始节点
// upper_beam ！= 1 search_from_candidate寻找多个初始节点
void HNSW::search(DistanceComputer& qdis, int k,
                  idx_t *I, float *D,
                  VisitedTable& vt) const
{
  if (upper_beam == 1) {

    //  greedy search on upper levels
    storage_idx_t nearest = entry_point;
    float d_nearest = qdis(nearest);

    for(int level = max_level; level >= 1; level--) {
      greedy_update_nearest(*this, qdis, level, nearest, d_nearest);
    }

    int ef = std::max(efSearch, k);
    //printf("search 开始\n");
    MaxHeap<Node> top_candidates = search_from(Node(d_nearest, nearest), qdis, ef, &vt);
    while (top_candidates.size() > k) {
      top_candidates.pop();
    }

    int nres = 0;
    int totle=0;
    while (!top_candidates.empty()) {
      float d;
      storage_idx_t label;
      std::tie(d, label) = top_candidates.top();
      faiss::maxheap_push(++nres, D, I, d, label);
      // if(top_candidates.size()<=10){
      //   int top_indu=in_degree[top_candidates.top().second];
      //   totle+=top_indu;
      //   printf("%d\t",top_indu);
      // }
      top_candidates.pop();
    }
    // printf("%d\n",totle);


    // MinimaxHeap candidates(candidates_size);

//    top_candidates.emplace(d_nearest, nearest);

    // search_from_candidates(qdis, k, I, D, candidates, vt, 0);

    // NOTE(hoss): Init at the beginning?
    vt.advance();

  } else {
    assert(false);

    int candidates_size = upper_beam;
    MinimaxHeap candidates(candidates_size);

    std::vector<idx_t> I_to_next(candidates_size);
    std::vector<float> D_to_next(candidates_size);

    int nres = 1;
    I_to_next[0] = entry_point;
    D_to_next[0] = qdis(entry_point);

    for(int level = max_level; level >= 0; level--) {

      // copy I, D -> candidates

      candidates.clear();

      for (int i = 0; i < nres; i++) {
        candidates.push(I_to_next[i], D_to_next[i]);
      }

      if (level == 0) {
        nres = search_from_candidates(qdis, k, I, D, candidates, vt, 0);
      } else  {
        nres = search_from_candidates(
          qdis, candidates_size,
          I_to_next.data(), D_to_next.data(),
          candidates, vt, level
        );
      }
      vt.advance();
    }
  }
}



template<typename T>
using MaxHeap = std::priority_queue<T, std::vector<T>, std::less<T>>;
template<typename T>
using MinHeap = std::priority_queue<T, std::vector<T>, std::greater<T>>;

MaxHeap<HNSW::Node> HNSW::search_from_add_vct(
    const Node& node,
    DistanceComputer& qdis,
    int ef,
    VisitedTable *vt,std::vector<idx_t> &vct) const
{
  // 大根堆，存放结果
  MaxHeap<Node> top_candidates;
  // 小根堆，存放候选节点
  MinHeap<Node> candidate_set;

  top_candidates.push(node);
  candidate_set.push(node);

  vt->set(node.second);

  float lower_bound = node.first;

  // 记录访问点的个数
  storage_idx_t ndis=0;

  while (!candidate_set.empty()) {
    float d0;
    storage_idx_t v0;
    std::tie(d0, v0) = candidate_set.top();

    // 当前候选中最小，大于结果集中的最大
    if (d0 > lower_bound) {
      break;
    }

    candidate_set.pop();

    size_t begin, end;
    neighbor_range(v0, 0, &begin, &end);

    for (size_t j = begin; j < end; ++j) {
      int v1 = neighbors[j];

      if (v1 < 0) {
        break;
      }
      if (vt->get(v1)) {
        continue;
      }

      vt->set(v1);

      // 访问点记录
      vct.push_back(v1);

      float d1 = qdis(v1);
      ndis++;
      if (top_candidates.top().first > d1 || top_candidates.size() < ef) {
        candidate_set.emplace(d1, v1);
        top_candidates.emplace(d1, v1);

        if (top_candidates.size() > ef) {
          top_candidates.pop();
        }
        // 更新结果集中的最大
        lower_bound = top_candidates.top().first;
      }

    }
  }
  // 传递访问点的个数：寻找当前队列中的距离最大值，然后push(lower_bound+1,ndis)
  // 如果top_candidates为空会报错
  FAISS_THROW_IF_NOT_MSG(!top_candidates.empty()," top_candidates is empty！");
  lower_bound = top_candidates.top().first;
  top_candidates.emplace(lower_bound+1,ndis);
  return top_candidates;
}


// upper_beam == 1 greedy_update_nearest寻找初始节点
// upper_beam ！= 1 search_from_candidate寻找多个初始节点
void HNSW::search_rt_array(DistanceComputer& qdis, int k,
                  idx_t *I, float *D,
                  VisitedTable& vt,std::vector<idx_t> &vct) const
{
  if (upper_beam == 1) {

    //  greedy search on upper levels
    storage_idx_t nearest = entry_point;
    float d_nearest = qdis(nearest);

    for(int level = max_level; level >= 1; level--) {
      greedy_update_nearest(*this, qdis, level, nearest, d_nearest);
    }

    int ef = std::max(efSearch, 10);

    // search_mode==0 标准的unbounded；search_mode==1，添加k属性的unbounded
    
    MaxHeap<Node> top_candidates = search_from_add_vct(Node(d_nearest, nearest), qdis, ef, &vt,vct);

    // MaxHeap<Node> top_candidates = search_from_addk(Node(d_nearest, nearest), qdis, ef, &vt,k);
    // 将第一个元素(记录访问点信息)取出
    top_candidates.pop();
    while (top_candidates.size() > k) {
      top_candidates.pop();
    }
    int nres = 0;
    while (!top_candidates.empty()) {
      float d;
      storage_idx_t label;
      std::tie(d, label) = top_candidates.top();
      faiss::maxheap_push(++nres, D, I, d, label);
      top_candidates.pop();
    }
    // 结果重排序
    faiss::maxheap_reorder (k, D, I);
    vt.advance();

  }

}



/*

// 修改的search
// upper_beam == 1 greedy_update_nearest寻找初始节点
// upper_beam ！= 1 search_from_candidate寻找多个初始节点
void HNSW::search(DistanceComputer& qdis, int k,
                  idx_t *I, float *D,
                  VisitedTable& vt) const
{
    HNSWStats stats;
    // 上层传过来最近邻节点为1个
    if (upper_beam == 1) {
        //  greedy search on upper levels
        storage_idx_t nearest = entry_point;
        float d_nearest = qdis(nearest);

        for (int level = max_level; level >= 1; level--) {
            greedy_update_nearest(*this, qdis, level, nearest, d_nearest);
        }

        int ef = std::max(efSearch, k);

        // candidates为efsearch k最大值
        MinimaxHeap candidates(ef);

        candidates.push(nearest, d_nearest);

        search_from_candidates(qdis, k, I, D, candidates, vt, 0);

        vt.advance();
  } 

}
*/


// upper_beam == 1 greedy_update_nearest寻找初始节点
// upper_beam ！= 1 search_from_candidate寻找多个初始节点
void HNSW::search_custom(DistanceComputer& qdis, int k,
                  idx_t *I, float *D,
                  VisitedTable& vt,int search_mode) const
{
  if (upper_beam == 1) {

    //  greedy search on upper levels
    storage_idx_t nearest = entry_point;
    float d_nearest = qdis(nearest);

    for(int level = max_level; level >= 1; level--) {
      greedy_update_nearest(*this, qdis, level, nearest, d_nearest);
    }

    int ef = std::max(efSearch, 10);

    // search_mode==0 标准的unbounded；search_mode==1，添加k属性的unbounded
    MaxHeap<Node> top_candidates;
    if(search_mode == 0)
        top_candidates = search_from(Node(d_nearest, nearest), qdis, ef, &vt);
    else if (search_mode == 1)
    {
      top_candidates = search_from_addk_v2(Node(d_nearest, nearest), qdis, ef, &vt,k);
    }
    else if (search_mode == 2)
    {
      //top_candidates = search_from_two_index(qdis, ef, &vt,k);
    }
        
    // 将第一个元素取出
    float d0;
    storage_idx_t ndis;
    std::tie(d0, ndis) = top_candidates.top();
    top_candidates.pop();

    while (top_candidates.size() > k) {
      top_candidates.pop();
    }


    int nres = 0;
    while (!top_candidates.empty()) {
      float d;
      storage_idx_t label;
      std::tie(d, label) = top_candidates.top();
      faiss::maxheap_push(++nres, D, I, d, label);
      top_candidates.pop();
    }
    // 结果重排序
    faiss::maxheap_reorder (k, D, I);
    D[0]=ndis;
    vt.advance();

  } else {

    assert(false);

    int candidates_size = upper_beam;
    MinimaxHeap candidates(candidates_size);

    std::vector<idx_t> I_to_next(candidates_size);
    std::vector<float> D_to_next(candidates_size);

    int nres = 1;
    I_to_next[0] = entry_point;
    D_to_next[0] = qdis(entry_point);

    for(int level = max_level; level >= 0; level--) {

      // copy I, D -> candidates

      candidates.clear();

      for (int i = 0; i < nres; i++) {
        candidates.push(I_to_next[i], D_to_next[i]);
      }

      if (level == 0) {
        nres = search_from_candidates(qdis, k, I, D, candidates, vt, 0);
      } else  {
        nres = search_from_candidates(
          qdis, candidates_size,
          I_to_next.data(), D_to_next.data(),
          candidates, vt, level
        );
      }
      vt.advance();
    }
  }

}

/** Do a BFS on the candidates list */
/** Do a BFS on the candidates list */

void HNSW::search_from_candidates_combine(
        DistanceComputer& qdis,
        int k,
        idx_t* I,
        float* D,
        MinimaxHeap& candidates,
        VisitedTable& vt,
        int level,
        int nres_in,int fos) const {
  int nres = nres_in;
  int ndis = 0;

  for (int i = 0; i < candidates.size(); i++) {
    idx_t v1 = candidates.ids[i];
    float d = candidates.dis[i];
    FAISS_ASSERT(v1 >= 0);
    if (nres < k) {
      faiss::maxheap_push(++nres, D, I, d, v1);
    } else if (d < D[0]) {
      faiss::maxheap_pop(nres--, D, I);
      faiss::maxheap_push(++nres, D, I, d, v1);
    }
    vt.set(v1);
  }

  int nstep = 0;

  while (candidates.size() > 0) {
    float d0 = 0;
    int v0 = candidates.pop_min(&d0);

    size_t begin, end;
    neighbor_range(v0, level, &begin, &end);
    if (!fos)
    {  
        for (size_t j = begin; j < end; j++) {
            int v1 = neighbors[j];
            if (v1 < 0)
                break;
            if (vt.get(v1)) {
                continue;
            }
            vt.set(v1);
            ndis++;
            float d = qdis(v1);

            if (nres < k) {
              faiss::maxheap_push(++nres, D, I, d, v1);
            } else if (d < D[0]) {
              faiss::maxheap_pop(nres--, D, I);
              faiss::maxheap_push(++nres, D, I, d, v1);
            }
            candidates.push(v1, d);
        }
    }else {
        for (size_t j = end-1; j >=begin; j--) {
            int v1 = neighbors[j];
            if (v1 < 0)
                break;
            if (vt.get(v1)) {
                continue;
            }
            vt.set(v1);
            ndis++;
            float d = qdis(v1);
            if (nres < k) {
              faiss::maxheap_push(++nres, D, I, d, v1);
            } else if (d < D[0]) {
              faiss::maxheap_pop(nres--, D, I);
              faiss::maxheap_push(++nres, D, I, d, v1);
            }
            candidates.push(v1, d);
        }
    } // end else
    nstep++;
    if (nstep > efSearch) {
        break;
    }
    } // end while

  while (candidates.size() > 0) {
    float d0 = 0;

    int v0 = candidates.pop_min(&d0);

    size_t begin, end;
    neighbor_range(v0, level, &begin, &end);

    for (size_t j = begin; j < end; j++) {
      int v1 = neighbors[j];
      if (v1 < 0) break;
      if (vt.get(v1)) {
        continue;
      }
      vt.set(v1);
      ndis++;
      float d = qdis(v1);
      if (nres < k) {
        faiss::maxheap_push(++nres, D, I, d, v1);
      } else if (d < D[0]) {
        faiss::maxheap_pop(nres--, D, I);
        faiss::maxheap_push(++nres, D, I, d, v1);
      }
      candidates.push(v1, d);
    }

    nstep++;
    // efsearch 控制搜索结束
    if (nstep > efSearch) {
      break;
    }
  }

  faiss::maxheap_reorder (k, D, I);
}

/// standard unbounded search
std::priority_queue<HNSW::Node> HNSW::search_from_candidate_unbounded_combine(
        const Node& node,
        DistanceComputer& qdis,
        int ef,
        VisitedTable* vt,
        int fos) const {
    int ndis = 0;
    std::priority_queue<Node> top_candidates;
    std::priority_queue<Node, std::vector<Node>, std::greater<Node>> candidates;

    top_candidates.push(node);
    candidates.push(node);
    vt->set(node.second);
    size_t cnt=0;
    while (!candidates.empty()) {
        float d0;
        storage_idx_t v0;
        std::tie(d0, v0) = candidates.top();

        if (d0 > top_candidates.top().first) {
            break;
        }

        candidates.pop();
        size_t begin, end;
        neighbor_range(v0, 0, &begin, &end);
        if (fos==0)
        {
            for (size_t j = begin; j < end; j++) {
                int v1 = neighbors[j];
                if (v1 < 0 ) {
                    break;
                }
                if (vt->get(v1)) {
                    continue;
                }
                vt->set(v1);
                float d1 = qdis(v1);
                ++ndis;

                if (top_candidates.top().first > d1 || top_candidates.size() < ef) {
                    candidates.emplace(d1, v1);
                    top_candidates.emplace(d1, v1);
                    if (top_candidates.size() > ef) {
                        top_candidates.pop();
                    }
                }
            }
        }else {
            for (size_t j = end-1; j >=begin; j--) {
                int v1 = neighbors[j];

                if (v1 < 0) {
                    break;
                }
                if (vt->get(v1)) {
                    continue;
                }

                vt->set(v1);

                float d1 = qdis(v1);
                ++ndis;
                if (top_candidates.top().first > d1 || top_candidates.size() < ef) {
                    candidates.emplace(d1, v1);
                    top_candidates.emplace(d1, v1);
                    if (top_candidates.size() > ef) {
                        top_candidates.pop();
                    }
                }
            }
        } // end else
        
    }// end while

    return top_candidates;
}

void HNSW::combine_search(
        DistanceComputer& qdis,
        int k,
        idx_t* I,
        float* D,
        int fos,
        VisitedTable& vt,
        RandomGenerator& rng3) const {
    if (upper_beam == 1) {
        //  greedy search on upper levels 
        storage_idx_t nearest = rng3.rand_long()%levels.size();
        // printf("nearest : %ld\n", nearest );
        float d_nearest = qdis(nearest);

/*        for (int level = max_level; level >= 1; level--) {
            greedy_update_nearest(*this, qdis, level, nearest, d_nearest);
        }*/

        int ef = std::max(efSearch, 10);

        if (search_bounded_queue) {
            MinimaxHeap candidates(ef);

            candidates.push(nearest, d_nearest);

            search_from_candidates_combine(qdis, k, I, D, candidates, 
                vt, 0, 0 ,fos);
        } else {
            std::priority_queue<Node> top_candidates;
            top_candidates=search_from_candidate_unbounded_combine(
                        Node(d_nearest, nearest), qdis, ef, &vt,fos);
            while (top_candidates.size() > k) {
                top_candidates.pop();
            }

            int nres = 0;
            while (!top_candidates.empty()) {
                float d;
                storage_idx_t label;
                std::tie(d, label) = top_candidates.top();
                faiss::maxheap_push(++nres, D, I, d, label);
                top_candidates.pop();
            }
        }

        vt.advance();
}
}
std::unordered_map<idx_t,int> hothubSearchmp;
void HNSW::initHothubSearchmp(){
  for(int hb=0;hb<hotset.size();hb++){
    std::unordered_set<idx_t>::iterator it;
    for(it=hotset[hb].begin();it!=hotset[hb].end();it++){
      hothubSearchmp[*it]=0;
    }
  }
}
void HNSW::getHothubsSearchCount(){
  int searchCount=10000;
  typedef std::pair<int,idx_t> pii;
  std::vector<pii> times;
  // 按照热点的热度排序
  for(auto a : hothubSearchmp){
    times.push_back(pii(-a.second,a.first));
  }
  std::sort(times.begin(),times.end());
  std::string dir="/home/wanghongya/lc/hotrate/glove/";
  dir+=std::to_string(efSearch);
  dir+="indu";
  dir+=".csv";
  std::ofstream file(dir);
  if(file){
    for(int i=0;i<times.size();i++){
      file <<-1*times[i].first<<","<<times[i].second << ","<<(float)times[i].first/10000.0*-1<<","<<indu[times[i].second]<<"\n" ;
    }
  }
  file.close();
}

/// standard unbounded search
std::priority_queue<HNSW::Node> HNSW::search_from_candidate_unbounded_hot_hubs_enhence_s(
        MinimaxHeap& candidates_t,
        DistanceComputer& qdis,
        int ef,
        VisitedTable* vt,
        idx_t n) const {
    int ndis = 0;
    int nstep = 0;
    std::priority_queue<Node> top_candidates;
    std::priority_queue<Node, std::vector<Node>, std::greater<Node>> candidates;
    while(candidates_t.size()!=0){
      float d0=0;
      int v0=candidates_t.pop_min(&d0);
      vt->set(v0);
      top_candidates.emplace(d0, v0);
      candidates.emplace(d0, v0);
    }
    
    // //距离统计
    // std::unordered_map<idx_t,int> mp;
    // std::unordered_map<idx_t,int> typemp;
    // std::vector<int> isHothub;
    // std::vector<int> typeHothub;
    // std::vector<int> discount;

    // typemp[node.second]=0;
    // int _times=0;
    // mp[node.second]=_times++;
    // std::vector<std::pair<int,float>> distemp;
    // //距离统计
    
    size_t cnt=0;
    printf("wowowo\n");
    while (!candidates.empty()) {
        float d0;
        storage_idx_t v0;
        std::tie(d0, v0) = candidates.top();

        if (d0 > top_candidates.top().first) {
            printf("我是因为距离退出！\n");
            break;
        }
        nstep++;
        candidates.pop();

        size_t begin, end;
        neighbor_range(v0, 0, &begin, &end);
        // //距离统计
        // distemp.push_back(std::pair<int,float>(mp[v0],d0));
        // _times=mp[v0]+1;
        // bool ishotbool=false;
        // for(int hb=0;hb<hotset.size();hb++){
        //   if(hotset[hb].find(v0)!=hotset[hb].end()){
        //     ishotbool=true;
        //     break;
        //   }
        // }
        // //bool ishotbool=neighbors[end-1]>=n?true:false;//修正热点的判断，将热点的存储起来
        // if(ishotbool){
        //   hothubSearchmp[v0]++;
        // }
        
        // isHothub.push_back(ishotbool);
        // typemp[v0]+=ishotbool;

        // typeHothub.push_back(typemp[v0]);
        // //距离统计
        // printf("begin:%ld,end:%ld\n",begin,end);
        // 遍历v0点对应的所有邻居
        // printf("begin:%ld,end:%ld\n",begin,end);
        // 遍历v0点对应的所有邻居
        size_t j = begin;
        // int discounttemp=0;
        while (j < end) {
            int v1 = neighbors[j];

            if (v1 < 0) {
                break;
            }
            // 遇到>=n 搜索末尾位置
            // 将指针指向下一组邻居结点
            if (v1 >= n)
            {
                size_t begin1,end1;
                neighbor_range(v1,0,&begin1,&end1);
                // printf("v1:%d,begin:%ld,end:%ld,begin1:%ld,end1:%ld\n",v1,begin,end,begin1,end1);
                j=begin1;
                end=end1;
                continue;
            }
            j++;
            if (vt->get(v1)) {
                continue;
            }

            vt->set(v1);

            float d1 = qdis(v1);
            computer_count++;
            // discounttemp++;
            ++ndis;

            if (top_candidates.top().first > d1 || top_candidates.size() < ef) {
                candidates.emplace(d1, v1);
                top_candidates.emplace(d1, v1);
                // //距离统计
                // mp[v1]=_times;
                // typemp[v1]=typemp[v0];
           
                // //距离统计
                if (top_candidates.size() > ef) {
                    top_candidates.pop();
                }
            }
        }
        // //距离统计
        // discount.push_back(discounttemp);
        // //距离统计
    }
    // //距离统计
    // typeHothubResult[_count]=typeHothub;
    // isHothubResult[_count]=isHothub;
    // discountResult[_count]=discount;
    // disresult[_count++]=distemp;
    
    // //距离统计
    return top_candidates;
}

// int stop_count=0;

/// standard unbounded search
std::priority_queue<HNSW::Node> HNSW::search_from_candidate_unbounded_hot_hubs_enhence(
        const Node& node,
        DistanceComputer& qdis,
        int ef,
        VisitedTable* vt,
        idx_t n) const {
    int ndis = 0;
    int nstep = 0;
    std::priority_queue<Node> top_candidates;
    std::priority_queue<Node, std::vector<Node>, std::greater<Node>> candidates;

    top_candidates.push(node);
    candidates.push(node);
    // //距离统计
    // std::unordered_map<idx_t,int> mp;
    // std::unordered_map<idx_t,int> typemp;
    // std::vector<int> isHothub;
    // std::vector<int> typeHothub;
    // std::vector<int> discount;

    // typemp[node.second]=0;
    // int _times=0;
    // mp[node.second]=_times++;
    // std::vector<std::pair<int,float>> distemp;
    // //距离统计
    vt->set(node.second);
    size_t cnt=0;
    // printf("我开始了\n");
    while (!candidates.empty()) {
        float d0;
        storage_idx_t v0;
        std::tie(d0, v0) = candidates.top();
   
        if (d0 > top_candidates.top().first) {
            // stop_count++;
            
            break;
        }
        nstep++;
        candidates.pop();

        size_t begin, end;
        neighbor_range(v0, 0, &begin, &end);
        // //距离统计
        // distemp.push_back(std::pair<int,float>(mp[v0],d0));
        // _times=mp[v0]+1;
        // bool ishotbool=false;
        // for(int hb=0;hb<hotset.size();hb++){
        //   if(hotset[hb].find(v0)!=hotset[hb].end()){
        //     ishotbool=true;
        //     break;
        //   }
        // }
        // //bool ishotbool=neighbors[end-1]>=n?true:false;//修正热点的判断，将热点的存储起来
        // if(ishotbool){
        //   hothubSearchmp[v0]++;
        // }
        
        // isHothub.push_back(ishotbool);
        // typemp[v0]+=ishotbool;

        // typeHothub.push_back(typemp[v0]);
        // //距离统计
        // printf("begin:%ld,end:%ld\n",begin,end);
        // 遍历v0点对应的所有邻居
        // printf("begin:%ld,end:%ld\n",begin,end);
        // 遍历v0点对应的所有邻居
        size_t j = begin;
        // int discounttemp=0;
        while (j < end) {
            int v1 = neighbors[j];

            if (v1 < 0) {
                break;
            }
            // 遇到>=n 搜索末尾位置
            // 将指针指向下一组邻居结点
            if (v1 >= n)
            {
                size_t begin1,end1;
                neighbor_range(v1,0,&begin1,&end1);
                // printf("v1:%d,begin:%ld,end:%ld,begin1:%ld,end1:%ld\n",v1,begin,end,begin1,end1);
                j=begin1;
                end=end1;
                continue;
            }
            j++;
            if (vt->get(v1)) {
                continue;
            }

            vt->set(v1);

            float d1 = qdis(v1);
            computer_count++;
            // discounttemp++;
            ++ndis;

            if (top_candidates.top().first > d1 || top_candidates.size() < ef) {
                candidates.emplace(d1, v1);
                top_candidates.emplace(d1, v1);
                // //距离统计
                // mp[v1]=_times;
                // typemp[v1]=typemp[v0];
           
                // //距离统计
                if (top_candidates.size() > ef) {
                    top_candidates.pop();
                }
            }
        }
        // //距离统计
        // discount.push_back(discounttemp);
        // //距离统计
    }
    // //距离统计
    // typeHothubResult[_count]=typeHothub;
    // isHothubResult[_count]=isHothub;
    // discountResult[_count]=discount;
    // disresult[_count++]=distemp;
    
    // //距离统计
    // printf("%d\n",stop_count);
    return top_candidates;
}

void HNSW::search_from_candidates_hot_hubs(
        DistanceComputer& qdis,
        int k,
        idx_t* I,
        float* D,
        MinimaxHeap& candidates,
        VisitedTable& vt,
        int level,
        int nres_in,unsigned n) const {
    int nres = nres_in;
    int ndis = 0;
    for (int i = 0; i < candidates.size(); i++) {
        idx_t v1 = candidates.ids[i];
        float d = candidates.dis[i];
        FAISS_ASSERT(v1 >= 0);
        if (nres < k) {
            faiss::maxheap_push(++nres, D, I, d, v1);
        } else if (d < D[0]) {
            faiss::maxheap_pop(nres--, D, I);
            faiss::maxheap_push(++nres, D, I, d, v1);
        }
        vt.set(v1);
    }

    int nstep = 0;
    printf("disange");
    while (candidates.size() > 0) {
        float d0 = 0;
        int v0 = candidates.pop_min(&d0);

        size_t begin, end;
        neighbor_range(v0, level, &begin, &end);

        for (size_t j = begin; j < end; j++) {
            int v1 = neighbors[j];
            if (v1 < 0)
                break;
            if (v1 >= n)
            {
                neighbor_range(v1,0,&begin,&end);
                continue;
            }
            if (vt.get(v1)) {
                continue;
            }
            vt.set(v1);
            ndis++;
            float d = qdis(v1);
            if (nres < k) {
                faiss::maxheap_push(++nres, D, I, d, v1);
            } else if (d < D[0]) {
                faiss::maxheap_pop(nres--, D, I);
                faiss::maxheap_push(++nres, D, I, d, v1);
            }
            candidates.push(v1, d);
        }

        nstep++;
        if (nstep > efSearch) {
            break;
        }
    }
}



void HNSW::search_with_hot_hubs_enhence(
            DistanceComputer& qdis,
            int k,
            idx_t* I,
            float* D,
            VisitedTable& vt,size_t n,RandomGenerator& rng3) const{
    if (upper_beam == 1) {
        //  greedy search on upper levels 
        // 固定ep 
        storage_idx_t nearest = entry_point;
        // 随机ep（只保留0层方法）
        // storage_idx_t nearest = rng3.rand_long()%n;
        // printf("nearest : %ld\n", nearest );
        float d_nearest = qdis(nearest);

        // printf("nearest: %ld,d_nearest%f\t",nearest,d_nearest);
        for (int level = max_level; level >= 1; level--) {
            greedy_update_nearest(*this, qdis, level, nearest, d_nearest);
        }
        // printf("nearest: %ld,d_nearest%f\t",nearest,d_nearest);

        int ef = std::max(efSearch, k);

        if (search_bounded_queue) {
            MinimaxHeap candidates(ef);

            candidates.push(nearest, d_nearest);

            search_from_candidates_hot_hubs(qdis, k, I, D, candidates, 
                vt, 0, 0 ,n);
        } else {
            std::priority_queue<Node> top_candidates;
            top_candidates=search_from_candidate_unbounded_hot_hubs_enhence(
                        Node(d_nearest, nearest), qdis, ef, &vt,n);
            while (top_candidates.size() > k) {
                top_candidates.pop();
            }
            // printf("top_candidates.size():%d\n",top_candidates.size());
            int nres = 0;
            while (!top_candidates.empty()) {
                float d;
                storage_idx_t label;
                std::tie(d, label) = top_candidates.top();
                faiss::maxheap_push(++nres, D, I, d, label);
                top_candidates.pop();
            }
        }

        vt.advance();
  }
  else{

    // assert(false);

    int candidates_size = upper_beam;
    MinimaxHeap candidates(candidates_size);

    std::vector<idx_t> I_to_next(candidates_size);
    std::vector<float> D_to_next(candidates_size);

    int nres = 1;
    I_to_next[0] = entry_point;
    D_to_next[0] = qdis(entry_point);

    for(int level = max_level; level >= 0; level--) {

      // copy I, D -> candidates
      // printf("进入前\n");
      candidates.clear();

      for (int i = 0; i < nres; i++) {
        candidates.push(I_to_next[i], D_to_next[i]);
      }

      if (level == 0) {
        // printf("最低\n");

        std::priority_queue<Node> top_candidates;
        int ef = std::max(efSearch, k);

        top_candidates=search_from_candidate_unbounded_hot_hubs_enhence_s(
                    candidates, qdis, ef, &vt,n);
        while (top_candidates.size() > k) {
            top_candidates.pop();
        }
        // printf("top_candidates.size():%d\n",top_candidates.size());
        int nres = 0;
        while (!top_candidates.empty()) {
            float d;
            storage_idx_t label;
            std::tie(d, label) = top_candidates.top();
            faiss::maxheap_push(++nres, D, I, d, label);
            top_candidates.pop();
        }
      } else  {
        // printf("进入后\n");

        nres = search_from_candidates(
          qdis, candidates_size,
          I_to_next.data(), D_to_next.data(),
          candidates, vt, level
        );
      }
      vt.advance();
    }
  }
}


std::priority_queue<HNSW::Node> HNSW::search_from_candidate_unbounded_hot_hubs_enhence_random(
        const Node& node,
        DistanceComputer& qdis,
        int ef,
        VisitedTable* vt,
        idx_t n) const {
    int ndis = 0;
    int nstep = 0;
    std::priority_queue<Node> top_candidates;
    std::priority_queue<Node, std::vector<Node>, std::greater<Node>> candidates;
    // 已经遇到过且未访问的热点列表
    std::set<Node> hb_candidates; 

    // 将热点放入set
    std::unordered_set<idx_t> hbs;
    for (int i = 0; i < hot_hubs.size(); ++i)
    {
      for (auto& a:hot_hubs[i])
      {
        hbs.insert(a.first);
      }
    }

    top_candidates.push(node);
    candidates.push(node);

    vt->set(node.second);
    size_t cnt=0;
    while (!candidates.empty()) {
        float d0;
        storage_idx_t v0;
        std::tie(d0, v0) = candidates.top();

        if (d0 > top_candidates.top().first) {
            break;
        }
        nstep++;
        cnt++;
        // 防止热点被取完
        if (cnt%20==0&&!hbs.empty()) // 如果在一定步长内没有碰到热点，选择已经访问的随机热点重启
        {
            if (hb_candidates.empty()) // 如果候选为空,从所有热点中随机选取一个热点重启
            {
                auto it = hbs.begin();
                v0 = *it;
                d0 = qdis(v0);
                hbs.erase(it); // 从热点列表中删去该热点，防止重复访问

            }else{  // 如果候选不为空，从已访问中选取
                auto it=hb_candidates.begin();
                v0 = (*it).second;
                d0 = (*it).first;
                hbs.erase(v0);
                hb_candidates.erase(it);
            }
            cnt=0;
        }
        else 
          candidates.pop();

        // 如果访问列该列表中的元素，就将该列
        auto it=hb_candidates.find({d0,v0});
        if (it!=hb_candidates.end())
        {
          hb_candidates.erase(it);
          //同时从热点中删除该点防止重复加入
          hbs.erase(v0);
          cnt=0;
        }


        size_t begin, end;
        neighbor_range(v0, 0, &begin, &end);

        // printf("begin:%ld,end:%ld\n",begin,end);
        // 遍历v0点对应的所有邻居
        size_t j = begin;
        while (j < end) {
            int v1 = neighbors[j];

            if (v1 < 0) {
                break;
            }
            // 遇到>=n 搜索末尾位置
            // 将指针指向下一组邻居结点
            if (v1 >= n)
            {
                size_t begin1,end1;
                neighbor_range(v1,0,&begin1,&end1);
                // printf("v1:%d,begin:%ld,end:%ld,begin1:%ld,end1:%ld\n",v1,begin,end,begin1,end1);
                j=begin1;
                end=end1;
                continue;
            }
            j++;
            if (vt->get(v1)) {
                continue;
            }

            vt->set(v1);

            float d1 = qdis(v1);
            ++ndis;

            // 如果v1 是热点，将该热点放入hb_candidates中
            if (hbs.find(v1)!=hbs.end())
            {
              hb_candidates.emplace(d1,v1);
            }

            if (top_candidates.top().first > d1 || top_candidates.size() < ef) {
                candidates.emplace(d1, v1);
                top_candidates.emplace(d1, v1);
                if (top_candidates.size() > ef) {
                    top_candidates.pop();
                }
            }
        }
    }
    return top_candidates;
}


// 随机热点 统计热点信息
std::priority_queue<HNSW::Node> HNSW::search_from_candidate_unbounded_hot_hubs_enhence_random_v2(
        const Node& node,
        DistanceComputer& qdis,
        int ef,
        VisitedTable* vt,
        idx_t n) const {
    int ndis = 0;
    int nstep = 0;
    std::priority_queue<Node> top_candidates;
    std::priority_queue<Node, std::vector<Node>, std::greater<Node>> candidates;
    // 已经遇到过且未访问的热点列表
    std::set<Node> hb_candidates; 
    std::unordered_set<idx_t> hbs_neighbors; // 热点邻居标记
    float base = 100000000.0;
    float base2 = 1500000000.0;
    std::vector<std::string> diss;   
    printf("kaishi\n");
    // 将热点放入set
    std::unordered_set<idx_t> hbs;
    for (int i = 0; i < hot_hubs.size(); ++i)
    {
      for (auto& a:hot_hubs[i])
      {
        hbs.insert(a.first);
      }
    }

    top_candidates.push(node);
    candidates.push(node);

    vt->set(node.second);
    size_t cnt=0;
    while (!candidates.empty()) {
        float d0;
        storage_idx_t v0;
        std::tie(d0, v0) = candidates.top();

        if (d0 > top_candidates.top().first) {
            break;
        }
        nstep++;
        cnt++;
        // 防止热点被取完
        if (cnt%20==0&&!hbs.empty()) // 如果在一定步长内没有碰到热点，选择已经访问的随机热点重启
        {
            if (hb_candidates.empty()) // 如果候选为空,从所有热点中随机选取一个热点重启
            {
                auto it = hbs.begin();
                v0 = *it;
                d0 = qdis(v0);
                // hbs.erase(it); // 从热点列表中删去该热点，防止重复访问

            }else{  // 如果候选不为空，从已访问中选取
                auto it=hb_candidates.begin();
                v0 = (*it).second;
                d0 = (*it).first;
                // hbs.erase(v0);
                hb_candidates.erase(it);
            }
            cnt=0;
        }
        else 
          candidates.pop();

        // 如果访问列该列表中的元素，就将该列
        auto it=hb_candidates.find({d0,v0});
        if (it!=hb_candidates.end())
        {
          hb_candidates.erase(it);
          //同时从热点中删除该点防止重复加入
          // hbs.erase(v0);
          cnt=0;
        }

        if (hbs.find(v0)!=hbs.end())
          diss.push_back(std::to_string(base+d0));
        else if (hbs_neighbors.find(v0)!=hbs_neighbors.end())
          diss.push_back(std::to_string(base2+d0)),hbs_neighbors.erase(v0);
        else
          diss.push_back(std::to_string(d0));


        size_t begin, end;
        neighbor_range(v0, 0, &begin, &end);

        // printf("begin:%ld,end:%ld\n",begin,end);
        // 遍历v0点对应的所有邻居
        size_t j = begin;
        while (j < end) {
            int v1 = neighbors[j];

            if (v1 < 0) {
                break;
            }
            // 遇到>=n 搜索末尾位置
            // 将指针指向下一组邻居结点
            if (v1 >= n)
            {
                size_t begin1,end1;
                neighbor_range(v1,0,&begin1,&end1);
                // printf("v1:%d,begin:%ld,end:%ld,begin1:%ld,end1:%ld\n",v1,begin,end,begin1,end1);
                j=begin1;
                end=end1;
                continue;
            }
            j++;
            if (vt->get(v1)) {
                continue;
            }

            if (hbs.find(v0)!=hbs.end())
              hbs_neighbors.insert(v1);

            vt->set(v1);

            float d1 = qdis(v1);
            ++ndis;

            // 如果v1 是热点，将该热点放入hb_candidates中
            if (hbs.find(v1)!=hbs.end())
            {
              hb_candidates.emplace(d1,v1);
            }

            if (top_candidates.top().first > d1 || top_candidates.size() < ef) {
                candidates.emplace(d1, v1);
                top_candidates.emplace(d1, v1);
                if (top_candidates.size() > ef) {
                    top_candidates.pop();
                }
            }
        }
        if (hbs.find(v0)!=hbs.end())
          hbs.erase(v0); // 将该热点删除操作放到外部统一操作
    }
    // 将距离保存到文件
    std::ofstream out("./hnsw_outdatas.csv",std::ofstream::app);

    for(int i=0;i<diss.size();i++){
        out<<diss[i]<<",";
    }
    out<<std::endl;
    out.close();
    return top_candidates;
}


// 全局热点 统计热点信息
std::priority_queue<HNSW::Node> HNSW::search_from_candidate_unbounded_hot_hubs_enhence_random_v3(
        const Node& node,
        DistanceComputer& qdis,
        int ef,
        VisitedTable* vt,
        idx_t n) const {
    int ndis = 0;
    int nstep = 0;
    std::priority_queue<Node> top_candidates;
    std::priority_queue<Node, std::vector<Node>, std::greater<Node>> candidates;
    // 已经遇到过且未访问的热点列表
    std::set<Node> hb_candidates; 
    std::unordered_set<idx_t> hbs_neighbors; // 热点邻居标记
    float base = 100000000.0;
    float base2 = 1500000000.0;
    std::vector<std::string> diss;   

    // 将热点放入set
    std::unordered_set<idx_t> hbs;
    for (int i = 0; i < hot_hubs.size(); ++i)
    {
      for (auto& a:hot_hubs[i])
      {
        hbs.insert(a.first);
      }
    }

    top_candidates.push(node);
    candidates.push(node);

    vt->set(node.second);

    while (!candidates.empty()) {
        float d0;
        storage_idx_t v0;
        std::tie(d0, v0) = candidates.top();

        if (d0 > top_candidates.top().first) {
            break;
        }
        nstep++;
        candidates.pop();

        // 如果访问列该列表中的元素，就将该列
        auto it=hb_candidates.find({d0,v0});
        if (it!=hb_candidates.end())
        {
          hb_candidates.erase(it);
        }

        if (hbs.find(v0)!=hbs.end())
          diss.push_back(std::to_string(base+d0));
        else if (hbs_neighbors.find(v0)!=hbs_neighbors.end())
          diss.push_back(std::to_string(base2+d0)),hbs_neighbors.erase(v0);
        else
          diss.push_back(std::to_string(d0));


        size_t begin, end;
        neighbor_range(v0, 0, &begin, &end);

        // printf("begin:%ld,end:%ld\n",begin,end);
        // 遍历v0点对应的所有邻居
        size_t j = begin;
        while (j < end) {
            int v1 = neighbors[j];

            if (v1 < 0) {
                break;
            }
            // 遇到>=n 搜索末尾位置
            // 将指针指向下一组邻居结点
            if (v1 >= n)
            {
                size_t begin1,end1;
                neighbor_range(v1,0,&begin1,&end1);
                // printf("v1:%d,begin:%ld,end:%ld,begin1:%ld,end1:%ld\n",v1,begin,end,begin1,end1);
                j=begin1;
                end=end1;
                continue;
            }
            j++;
            if (vt->get(v1)) {
                continue;
            }

            if (hbs.find(v0)!=hbs.end())
              hbs_neighbors.insert(v1);

            vt->set(v1);

            float d1 = qdis(v1);
            ++ndis;

            // 如果v1 是热点，将该热点放入hb_candidates中
            if (hbs.find(v1)!=hbs.end())
            {
              hb_candidates.emplace(d1,v1);
            }

            if (top_candidates.top().first > d1 || top_candidates.size() < ef) {
                candidates.emplace(d1, v1);
                top_candidates.emplace(d1, v1);
                if (top_candidates.size() > ef) {
                    top_candidates.pop();
                }
            }
        }
        if (hbs.find(v0)!=hbs.end())
          hbs.erase(v0); // 将该热点删除操作放到外部统一操作
    }
    // 将距离保存到文件
    std::ofstream out("./globle_hubs.csv",std::ofstream::app);

    for(int i=0;i<diss.size();i++){
        out<<diss[i]<<",";
    }
    out<<std::endl;
    out.close();
    return top_candidates;
}





void HNSW::search_with_hot_hubs_enhence_random(
            DistanceComputer& qdis,
            int k,
            idx_t* I,
            float* D,
            VisitedTable& vt,size_t n,RandomGenerator& rng3) const{
    if (upper_beam == 1) {
        //  greedy search on upper levels 
        // 固定ep 
        storage_idx_t nearest = entry_point;
        // 随机ep（只保留0层方法）
        // storage_idx_t nearest = rng3.rand_long()%n;
        // printf("nearest : %ld\n", nearest );
        float d_nearest = qdis(nearest);

        // printf("nearest: %ld,d_nearest%f\t",nearest,d_nearest);
        for (int level = max_level; level >= 1; level--) {
            greedy_update_nearest(*this, qdis, level, nearest, d_nearest);
        }
        // printf("nearest: %ld,d_nearest%f\t",nearest,d_nearest);

        int ef = std::max(efSearch, 10);

        if (search_bounded_queue) {
            MinimaxHeap candidates(ef);

            candidates.push(nearest, d_nearest);

            search_from_candidates_hot_hubs(qdis, k, I, D, candidates, 
                vt, 0, 0 ,n);
        } else {
            std::priority_queue<Node> top_candidates;
            // printf("%d\n",hot_hubs[0].size());
            top_candidates=search_from_candidate_unbounded_hot_hubs_enhence_random_v2(
                        Node(d_nearest, nearest), qdis, ef, &vt,n);
            while (top_candidates.size() > k) {
                top_candidates.pop();
            }
            // printf("top_candidates.size():%d\n",top_candidates.size());
            int nres = 0;
            while (!top_candidates.empty()) {
                float d;
                storage_idx_t label;
                std::tie(d, label) = top_candidates.top();
                faiss::maxheap_push(++nres, D, I, d, label);
                top_candidates.pop();
            }
        }

        vt.advance();
}
}





// index 相似度比较

/*void HNSW::similarity(HNSW * index1,HNSW * index2,int nb){
  float res=0;
  float index1_sum=0,index2_sum=0;

  for (int i = 0; i <nb ; ++i)
  {
      float cnt=0;
      size_t begin1, end1;
      index1->neighbor_range(i, 0, &begin1, &end1);
      size_t begin2, end2;
      index2->neighbor_range(i, 0, &begin2, &end2);  
      //std::cout<<end1-begin1<<"：："<<end2-begin2<<std::endl;

      // index1中，访问点个数
      for (size_t x = begin1; x < end1; ++x) {
          if (index1->neighbors[x] < 0) {
            break;
          }
          index1_sum++;
      }

      // index2中，访问点个数
      for (size_t x = begin2; x < end2; ++x) {

          if (index2->neighbors[x] < 0) {
            break;
          }
          index2_sum++;
      }



      for (size_t x = begin1; x < end1; ++x) {
          if (index1->neighbors[x] < 0) {
            break;
          }
        for (size_t y = begin2; y < end2; ++y){
            if (index2->neighbors[y] < 0) {
              break;
            }
            if(index1->neighbors[x] == index2->neighbors[y]){
              cnt++;

              
              continue;
            }
                
        }
      
      }

      res+=cnt;
      
}
  std::cout<<res<<std::endl;

  std::cout<<index1_sum<<std::endl;

  std::cout<<index2_sum<<std::endl;

  std::cout<<res*2/(index1_sum+index2_sum)<<std::endl;

}*/


// 统计索引中每个点出现频率
void HNSW::similarity(HNSW * index1,HNSW * index2,int nb){
  
  std::map<size_t,size_t> ma;
  long long sum=0;
  for (int i = 0; i < nb ; ++i)
  {
      size_t begin1, end1;
      index1->neighbor_range(i, 0, &begin1, &end1);
      int cnt=5;
      for (size_t x = begin1; x < end1&&cnt; ++x) {
          if (index1->neighbors[x] < 0) {
              break;
            }
          cnt--;
          ma[index1->neighbors[x]]++;
          sum++;
      }
  }
  for (auto a:ma)
  {
    std::cout<<"sum: "<<sum<<"  id: "<<a.first<<" cnt: "<<a.second<<std::endl;
  }
}


void HNSW::setIndex(HNSW* findex,HNSW* sindex){
      index1 = findex;
      index2 = sindex;
}


void HNSW::MinimaxHeap::push(storage_idx_t i, float v) {
    // 如果放入的节点数大于最大数（n）
    if (k == n) {
        if (v >= dis[0])
            return;
        faiss::heap_pop<HC>(k--, dis.data(), ids.data());
        --nvalid;
    }
    // 不足n个直接插入
    faiss::heap_push<HC>(++k, dis.data(), ids.data(), v, i);
    ++nvalid;
}

// dis[0] 存放最大距离
float HNSW::MinimaxHeap::max() const {
    return dis[0];
}

int HNSW::MinimaxHeap::size() const {
    return nvalid;
}

void HNSW::MinimaxHeap::clear() {
    nvalid = k = 0;
}

// ids，dis为MinimaxHeap的变量，用来存储candidates中的值
int HNSW::MinimaxHeap::pop_min(float* vmin_out) {
    assert(k > 0);  
    // returns min. This is an O(n) operation
    int i = k - 1;
    // 寻找ids中第一个不为-1的值（ids为结果存放数组）
    while (i >= 0) {
        if (ids[i] != -1)
            break;
        i--;
    }
    // 如果全为-1，意味着没有元素
    if (i == -1)
        return -1;

    // 存放最小值下标
    int imin = i;
    // 存放最小值
    float vmin = dis[i];
    i--;
    // 寻找最小值
    while (i >= 0) {
        if (ids[i] != -1 && dis[i] < vmin) {
            vmin = dis[i];
            imin = i;
        }
        i--;
    }
    // 用vmin_out 记录最小值的值
    if (vmin_out)
        *vmin_out = vmin;

    // ret为最小值的id
    int ret = ids[imin];

    // 将此最小值位置置为-1，相当于弹出
    ids[imin] = -1;
    --nvalid;

    return ret;
}

// 寻找当前距离中，小于阈值thresh中值的个数的点
int HNSW::MinimaxHeap::count_below(float thresh) {
// O(N)
    int n_below = 0;
    for (int i = 0; i < k; i++) {
        if (dis[i] < thresh) {
            n_below++;
        }
    }

    return n_below;
}

} // namespace faiss


