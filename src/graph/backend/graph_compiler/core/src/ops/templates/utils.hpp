/*******************************************************************************
 * Copyright 2020-2023 Intel Corporation
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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_OPS_TEMPLATES_UTILS_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_OPS_TEMPLATES_UTILS_HPP

#include <algorithm>
#include <string>
#include <utility>
#include <vector>
#include <compiler/config/context.hpp>
#include <compiler/ir/builder.hpp>
#include <compiler/ir/builtin.hpp>
#include <compiler/ir/sc_data_format.hpp>
#include <compiler/ir/sc_data_type.hpp>
#include <compiler/ir/transform/auto_cast.hpp>
#include <compiler/ir/transform/constant_fold.hpp>
#include <runtime/trace.hpp>
#include <unordered_set>
#include <util/utils.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
expr divide_and_ceil(const expr &, const expr &);
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

inline bool is_amx_dtype(const context_ptr &ctx, const sc_data_type_t &dtype) {
  bool ret = false;
  if (ctx->use_amx()) {
    if (ctx->machine_.cpu_flags_.fAVX512AMXBF16) {
      ret |= (dtype == datatypes::bf16);
    }
    if (ctx->machine_.cpu_flags_.fAVX512AMXINT8) {
      ret |= utils::is_one_of(dtype, datatypes::u8, datatypes::s8);
    }
  }

  return ret;
}

inline bool is_vnni_low_fp(
  const context_ptr &ctx, const sc_data_type_t &dtype) {
  return dtype == datatypes::bf16;
}

inline bool no_vnni_low_fp(
  const context_ptr &ctx, const sc_data_type_t &dtype) {
  return dtype == datatypes::f16;
}

inline bool no_vnni(const context_ptr &ctx, const sc_data_type_t &dtype) {
  return dtype == datatypes::f32 || dtype == datatypes::f16;
}

inline std::vector<expr> dims_to_expr(const sc_dims &dim) {
  std::vector<expr> ret;
  for (auto i : dim) {
    ret.emplace_back(dim2unsigned(i));
  }
  return ret;
}

inline bool is_parallel_space_enough(int work_amount, int nthreads) {
  return (work_amount > 0
    && (work_amount % nthreads == 0
      || utils::divide_and_ceil(work_amount, nthreads) >= 4));
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

inline std::vector<int> get_dynamic_block_candidates() {
  return std::vector<int> {16, 32, 64};
}

inline std::vector<int> get_dynamic_batch_block_candidates() {
  return std::vector<int> {16, 32, 64, 2, 4, 8};
}

inline uint16_t vectorize_step(
  const context_ptr &ctx, sc_data_etype detype, uint16_t minv) {
  return std::min(minv, ctx->get_max_vector_lanes(detype));
}

struct trace_guard_t {
  int trace_id;
  context_ptr ctx;
  trace_guard_t(const context_ptr &ctx, const std::string &func_name)
    : ctx(ctx) {
    trace_id = register_traced_func(func_name);
    if (ctx->flags_.trace_) {
      builder::get_current_builder()->push_evaluate(
        builtin::make_trace(trace_id, 0, 0));
    }
  }
  ~trace_guard_t() {
    if (ctx->flags_.trace_) {
      builder::get_current_builder()->push_evaluate(
        builtin::make_trace(trace_id, 1, 0));
    }
  }
};

inline static std::vector<int> get_splits(const int X) {
  std::vector<int> splits;
  for (auto i = 1; i <= X; ++i) {
    if (X % i == 0) { splits.push_back(i); }
  }
  return splits;
}

inline static std::vector<int> get_sub_blocks(const int X, int factor = 2) {
  std::vector<int> sub_blocks = {1, 2, 4, 8, 12, 16, 32};
  for (size_t i = 0; i < sub_blocks.size(); i++) {
    if (sub_blocks.at(i) >= X / factor) {
      return {sub_blocks.begin(), sub_blocks.begin() + i + 1};
    }
  }
  return sub_blocks;
}

template <typename T>
inline static std::vector<int> get_blocks_if_not_satisfy(
  const int X, int floor, int ceiling, T filter) {
  auto block_list = utils::get_blocks(X, floor, ceiling);
  block_list.erase(std::remove_if(block_list.begin(), block_list.end(), filter),
    block_list.end());
  if (block_list.empty()) { block_list = utils::get_blocks(X, floor, ceiling); }
  return block_list;
}

inline expr get_balance211_length(
  const expr &n, const expr &team, const expr &idx, expr &n_start, expr &T1) {
  assert(team.isa<var>() || get_expr_as_int(team) >= 1);
  expr n1 = divide_and_ceil(n, team);
  expr n2 = do_cast_and_fold(n1 - 1);
  T1 = do_cast_and_fold(n - n2 * team);
  n_start = builder::make_select(idx <= T1, do_cast_and_fold(idx * n1),
    do_cast_and_fold(T1 * n1 + (idx - T1) * n2));
  return builder::make_select(idx < T1, n1, n2);
}

inline std::vector<int> get_os_blocks(const int ow, const int adj_os) {
  std::vector<int> factors = utils::get_factors(ow);
  std::vector<int> os_factors = utils::get_blocks(adj_os, 16);
  factors.insert(factors.end(), os_factors.begin(), os_factors.end());
  std::unordered_set<int> unique_factors(factors.begin(), factors.end());
  factors.assign(unique_factors.begin(), unique_factors.end());
  std::sort(factors.begin(), factors.end());
  return factors;
}

inline int block_split(
  const int &total_size, const int &num, int &block, int &tail_block) {
  block = utils::divide_and_ceil(total_size, num);
  tail_block = total_size % block;
  if (tail_block == 0) { tail_block = block; }
  int used_threads = utils::divide_and_ceil(total_size, block);
  return used_threads;
}

inline int get_lanes(
  const context_ptr &ctx, const int C_block, const sc_data_type_t &dtype) {
  int lanes = 1;
  if (utils::is_one_of(dtype, datatypes::s8, datatypes::u8)) {
    if (C_block / 64 && C_block % 64 == 0) {
      lanes = vectorize_step(ctx, dtype.type_code_, 64);
    } else if (C_block / 32 && C_block % 32 == 0) {
      lanes = vectorize_step(ctx, dtype.type_code_, 32);
    } else if (C_block / 16 && C_block % 16 == 0) {
      lanes = vectorize_step(ctx, dtype.type_code_, 16);
    }
  } else if (is_vnni_low_fp(ctx, dtype)) {
    if (C_block / 32 && C_block % 32 == 0) {
      lanes = vectorize_step(ctx, dtype.type_code_, 32);
    } else if (C_block / 16 && C_block % 16 == 0) {
      lanes = vectorize_step(ctx, dtype.type_code_, 16);
    }
  } else {
    if (C_block / 16 && C_block % 16 == 0) {
      lanes = vectorize_step(ctx, dtype.type_code_, 16);
    }
  }
  return lanes;
}

inline uint64_t convert_int_to_mask(const int val) {
  uint64_t mask = 0;
  for (int i = 0; i < val; ++i) {
    mask = mask << 1;
    mask |= 0x1;
  }
  return mask;
}

inline int get_minimal_lanes(const int val) {
  COMPILE_ASSERT(val <= 64,
    "expected to be less than cache line size(64), but got " << val << "!");
  if (val > 32) {
    return 64;
  } else if (val > 16) {
    return 32;
  } else {
    return 16;
  }
}

inline sc_data_type_t get_dtype(const int lanes) {
  sc_data_type_t var_dtype;
  switch (lanes) {
    case 16: {
      var_dtype = datatypes::u16;
      break;
    }
    case 32: {
      var_dtype = datatypes::u32;
      break;
    }
    case 64: {
      var_dtype = datatypes::index;
      break;
    }
    default:
      COMPILE_ASSERT(0, "expected lanes to be 16, 32, 64, but got " << lanes);
  }
  return var_dtype;
}

bool is_prefetch_debug_mode();
// emit traces of prefetched address for debugging
void trace_prefetch_for_debug(const expr &addr);
// emit traces of BRGEMM B address for debugging
void trace_brgemm_for_debug(
  const expr &Baddr, const expr &bs, const expr &N, const expr &K);
} // namespace ops
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
