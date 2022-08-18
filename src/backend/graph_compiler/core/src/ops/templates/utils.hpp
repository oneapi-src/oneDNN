/*******************************************************************************
 * Copyright 2020-2022 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *******************************************************************************/

#ifndef BACKEND_GRAPH_COMPILER_CORE_SRC_OPS_TEMPLATES_UTILS_HPP
#define BACKEND_GRAPH_COMPILER_CORE_SRC_OPS_TEMPLATES_UTILS_HPP

#include <algorithm>
#include <utility>
#include <vector>
#include <compiler/config/context.hpp>
#include <compiler/ir/sc_data_format.hpp>
#include <compiler/ir/sc_data_type.hpp>
#include <unordered_set>
namespace sc {
namespace ops {
template <typename T>
/**
 * This func is used in config parameter candidates merge, drops duplicate items
 * and set floor and ceil value to filter candidates.
 * @param v1 non-const vector to merge
 * @param v2 const vector to merge
 * @param floor clip min value
 * @param ceil clip max value
 * */
static std::vector<T> concat_candidate_vec(std::vector<T> v1,
  const std::vector<T> &v2, const T floor = 8, const T ceil = 4096) {
  if (v1.empty()) { return v2; }
  if (v2.empty()) { return v1; }
  std::unordered_set<T> v3(v1.begin(), v1.end());
  std::sort(v1.begin(), v1.end());
  T biggest = v1[v1.size() - 1];
  for (auto it = v2.begin(); it != v2.end(); it++) {
    if (*it < biggest * 2) { v3.insert(*it); }
  }
  v1.clear();
  v1.insert(v1.end(), v3.begin(), v3.end());
  std::sort(v1.rbegin(), v1.rend());
  while (v1.back() < floor && !v1.empty()) {
    v1.pop_back();
  }
  std::sort(v1.begin(), v1.end());
  while (v1.back() > ceil && !v1.empty()) {
    v1.pop_back();
  }
  return v1;
}

inline bool is_use_amx(const context_ptr &ctx) {
  return (ctx->machine_.cpu_flags_.fAVX512AMXBF16
           || ctx->machine_.cpu_flags_.fAVX512AMXINT8)
    && ctx->flags_.brgemm_use_amx_;
}

/**
 * filter the vector by factor and tile factor, in order to adapt to the input
 * of dnnl amx.
 * @param input a array before filter
 * @param reduce_factor factor of reduce dim, 64 for dtype
 * == s8 and 32 for dtype == bf16
 * @param tile_factor tile factor of reduce dim, 4 for dtype
 * == s8 and 2 for dtype == bf16
 * */
inline std::vector<int> filter_valid_factor_for_amx(
  const std::vector<int> &input, int reduce_factor = 1, int tile_factor = 1) {
  std::vector<int> results;
  for (auto &it : input) {
    if ((it / reduce_factor == 0 && it % reduce_factor % tile_factor == 0)
      || it % reduce_factor == 0) {
      results.push_back(it);
    }
  }
  return results;
}

inline std::pair<int, int> get_amx_reduce_and_tile_factor(
  sc_data_type_t dtype) {
  int reduce_factor = 1, tile_factor = 1;
  if (dtype.type_code_ == sc_data_etype::S8) {
    reduce_factor = 64;
    tile_factor = 4;
  } else if (dtype.type_code_ == sc_data_etype::BF16) {
    reduce_factor = 32;
    tile_factor = 2;
  }
  return std::make_pair(reduce_factor, tile_factor);
}

template <typename T>
std::vector<T> merge_vec(const std::vector<T> &a, const std::vector<T> &b) {
  std::vector<T> result(a);
  for (auto it : b) {
    result.push_back(it);
  }
  return result;
}

inline std::vector<int> get_dynamic_block_candidates(bool has_48 = true) {
  return has_48 ? std::vector<int> {16, 32, 48, 64}
                : std::vector<int> {16, 32, 64};
}
} // namespace ops
} // namespace sc

#endif
