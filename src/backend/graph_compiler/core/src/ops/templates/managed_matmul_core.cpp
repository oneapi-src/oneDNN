/*******************************************************************************
 * Copyright 2022-2023 Intel Corporation
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

#include "managed_matmul_core.hpp"
#include <algorithm>
#include <limits>
#include <string>
#include "../fusible/memory_movement.hpp"
#include "utils.hpp"
#include <compiler/ir/builder.hpp>
#include <compiler/ir/builtin.hpp>
#include <compiler/ir/easy_build.hpp>
#include <compiler/ir/graph/fusion_mgr.hpp>
#include <compiler/ir/transform/loop_transform.hpp>
#include <compiler/ir/transform/scope_flatten.hpp>
#include <compiler/ir/transform/tensor_shrink.hpp>
#include <ops/matmul_core.hpp>
#include <ops/templates/commit_op.hpp>
#include <runtime/config.hpp>
#include <runtime/parallel.hpp>
#include <runtime/trace.hpp>
#include <util/any_map.hpp>
#include <util/math_utils.hpp>
#include <util/reflection.hpp>

SC_MODULE(ops.managed_matmul_core)
using namespace sc::builder;
namespace sc {

using ops::managed_matmul_core_config_t;
// clang-format off
SC_CLASS(managed_matmul_core_config_t)
  SC_FIELD(M_split_num)
  SC_FIELD(N_split_num)
  SC_FIELD(M_sub_block)
  SC_FIELD(N_sub_block)
  SC_FIELD(K_sub_block)
  SC_FIELD(im_loop_order)
SC_CLASS_END();
// clang-format on

namespace ops {

config_ptr gen_managed_matmul_core_t::get_default_config(
  context_ptr ctx) const {
  auto ret = reflection::general_object_t::make<managed_matmul_core_config_t>();
  managed_matmul_core_config_t &cfg
    = *ret.unchecked_get_as<managed_matmul_core_config_t>();
  const int num_threads = runtime_config_t::get().get_num_threads();
  const int iim_block = iim_block_;
  const int iin_block = iin_block_;
  const int iik_block = iik_block_;
  bool is_int8 = utils::is_one_of(get_A_dtype(), datatypes::u8, datatypes::s8);
  bool is_f32 = get_A_dtype() == datatypes::f32;
  const int M
    = utils::divide_and_ceil(
        static_cast<int>(in_tensors_[0].get_plain_dims()[0]), iim_block)
    * iim_block;
  const int N
    = utils::divide_and_ceil(
        static_cast<int>(in_tensors_[1].get_plain_dims()[1]), iin_block)
    * iin_block;
  const int K
    = utils::divide_and_ceil(
        static_cast<int>(in_tensors_[0].get_plain_dims()[1]), iik_block)
    * iik_block;
  const int sizeofdtypeA
    = utils::get_sizeof_etype(in_tensors_[0].dtype_.as_etype());
  const int sizeofdtypeC
    = utils::get_sizeof_etype(out_tensors_[0].dtype_.as_etype());
  float cost = std::numeric_limits<float>::max();
  int split_n = 1;
  cfg.im_loop_order = 0;
  bool is_special_fm = ctx->machine_.cpu_flags_.family == 6
    && ctx->machine_.cpu_flags_.model == 143;
  for (int i = 1; i <= num_threads; i++) {
    int num_M_block = utils::divide_and_ceil(M / iim_block, num_threads / i);
    int num_N_block = utils::divide_and_ceil(N / iin_block, i);
    int num_brgemm = num_M_block * num_N_block;
    int num_core
      = std::min(i, N / iin_block) * std::min(num_threads / i, M / iim_block);
    // Cost = Shape_efficient_weight *
    // (workload_balance + divide_X_plenty) / core_utilitizaiton
    // single core gemm prefers square shape for A and B.
    // For small workload, the A and B shape is not a key problem, but the
    // num_core and num_brgemm is important to performance. Use 2048 to reduce
    // the shape weight on small shape.
    float new_cost;
    float sew = 1024 + M * i / num_threads + N / i;
    if (((K >= 1024 && is_int8 && !is_use_amx(ctx))
          || (K >= 512 && is_f32 && !is_special_fm))) {
      // Cost += empty_cores, making M_split_num * N_split_num closer to
      // num_threads
      float empty_cores = num_threads - i * (num_threads / i);
      if (((N >= 1024 && is_int8 && M <= 2 * N)
            || (N >= 256 && is_f32 && M <= 64))
        || (is_f32 && M <= 256 && N >= 1024 && K >= 1024)) {
        // give bigger splits on N when N is bigger
        new_cost
          = sew * (num_brgemm + num_threads / i / 2) / num_core + empty_cores;
      } else if (N >= 256 && is_f32 && M <= 256) {
        new_cost = sew * (num_brgemm + i + num_threads / i * 2) / num_core
          + empty_cores;
      } else {
        new_cost = sew * (num_brgemm + i / 2) / num_core + empty_cores;
      }
    } else {
      new_cost = sew * (num_brgemm + 8 * i) / num_core;
    }

    if (new_cost < cost) {
      split_n = i;
      cost = new_cost;
    }
  }
  cfg.M_split_num = num_threads / split_n;
  cfg.N_split_num = split_n;
  if (is_int8 && N <= 512 && K <= 512) {
    // for int8 datatype and small N/Ks, we prefer to give splits only on M
    // when M is small, num_threadsx1x1 split is the same as 1x1x1 split, which
    // runs on single core
    cfg.M_split_num = num_threads;
    cfg.N_split_num = 1;
  } else if (N <= 192 && K <= 192) {
    // for other datatypes, we prefer to give splits only on M with much smaller
    // N/Ks
    cfg.M_split_num = num_threads;
    cfg.N_split_num = 1;
  } else if (M == iim_block && num_threads <= 4) {
    cfg.M_split_num = 1;
    // magic number = 4096, needs to be further discussed for pretty big K
    if (K < 4096) {
      cfg.N_split_num = num_threads;
    } else {
      for (auto i : get_splits(num_threads)) {
        float new_cost
          = std::abs(float(i * i) / float(num_threads) - float(N) / float(K));
        if (new_cost < cost) {
          cfg.N_split_num = i;
          cost = new_cost;
        }
      }
    }
  } else if (K >= 8192) {
    // for really big K, we need to give splits on K
    if (M < N) {
      auto possible_splits = get_splits(cfg.M_split_num);
      if (possible_splits.size() > 2 && N / M < 3) {
        cfg.M_split_num = cfg.M_split_num / possible_splits[1];
      } else {
        cfg.M_split_num = 1;
        int K_split_num = get_splits(num_threads).at(1);
        cfg.N_split_num = num_threads / K_split_num;
      }
    } else {
      auto possible_splits = get_splits(cfg.N_split_num);
      if (possible_splits.size() > 2) {
        cfg.N_split_num = cfg.N_split_num / possible_splits[1];
      }
    }
  } else if (is_f32 && !is_special_fm && M <= 256 && N >= 256 && K >= 512) {
    // f32 special case
    // for small M, consider giving splits on big K
    if (K >= 1024 && K >= 2 * N && get_splits(num_threads).size() > 3) {
      // give bigger splits on K
      int K_split_num = get_splits(num_threads).at(2);
      cfg.N_split_num = cfg.N_split_num / K_split_num > 0
        ? cfg.N_split_num / K_split_num
        : cfg.N_split_num;
    } else if (get_splits(num_threads).size() > 2 && N >= 1024 && N != K) {
      int K_split_num = get_splits(num_threads).at(1);
      cfg.N_split_num = cfg.N_split_num / K_split_num > 0
        ? cfg.N_split_num / K_split_num
        : cfg.N_split_num;
    } else if (N == 256 && K == 512 && get_splits(num_threads).size() > 3) {
      // special requirements in dlrm shapes, will refactor logic after
      // involving graph-level loop up
      if (cfg.M_split_num >= 2) { cfg.M_split_num /= 2; }
      if (cfg.N_split_num >= 2) { cfg.N_split_num /= 2; }
    }
  } else if (M / iim_block < 2 && (N >= 16 * M || K >= 16 * M)) {
    // int8 special case
    if (is_int8 && !is_use_amx(ctx)) {
      cfg.M_split_num = 1;
      int K_split_num = 1;
      if (K >= 16 * M) {
        K_split_num = get_splits(num_threads).size() > 2
          ? get_splits(num_threads).at(1)
          : 1;
      }
      cfg.N_split_num = num_threads / K_split_num;
    }
  }
  int single_M = utils::divide_and_ceil(
                   utils::divide_and_ceil(M, iim_block), cfg.M_split_num)
    * iim_block;
  int single_N = utils::divide_and_ceil(
                   utils::divide_and_ceil(N, iin_block), cfg.N_split_num)
    * iin_block;
  int single_K = K;
  int L2_size = static_cast<int>(ctx->machine_.cpu_flags_.getDCacheSize(2));
  int single_K_threshold
    = (single_M * single_N * sizeofdtypeA < L2_size ? 2048 : 4096)
    / sizeofdtypeA;
  if (is_f32 && !is_special_fm
    && num_threads / cfg.M_split_num / cfg.N_split_num <= 2 && M >= 128) {
    // if no split is given on K axis, bigger K_sub_block is required
    single_K_threshold /= 4;
  }
  if (single_K >= single_K_threshold) {
    cfg.K_sub_block = utils::divide_and_ceil(single_K, single_K_threshold);
    int K_split_num = num_threads / cfg.M_split_num / cfg.N_split_num;
    while (K / iik_block_ / K_split_num < cfg.K_sub_block) {
      cfg.K_sub_block--;
    }
    int L2_K = utils::divide_and_ceil(
                 utils::divide_and_ceil(single_K, iik_block), cfg.K_sub_block)
      * iik_block;
    // sizeofdtypeA* (M * K) + sizeofdtypeB * (N * K) + sizeofdtypeC(M * N) <=
    // L2_size, let M == N, then
    // 2 * sizeofdtypeA * M * K + sizeofdtypeC * M * M <= L2_size
    // Then M = (sqrt((2 * sizeofdtypeA * K) ^ 2 + 4 * sizeofdtypeC *
    // L2_size) - 2 * sizeofdtypeA * K)/ (2 * sizeofdtypeC)
    int L2_MN
      = (sqrt(pow(2 * sizeofdtypeA * L2_K, 2) + 4 * sizeofdtypeC * L2_size)
          - 2 * sizeofdtypeA * L2_K)
      / (2 * sizeofdtypeC);
    cfg.M_sub_block = std::max(1, single_M / L2_MN);
    cfg.N_sub_block = std::max(1, single_N / L2_MN);
  } else {
    // sizeofdtypeA * M * K + sizeofdtypeB * N * K <= L2_size
    // let M == N, then
    // M = L2_size / (2 * sizeofdtypeA * K)
    int L2_MN = L2_size / (2 * sizeofdtypeA * single_K);
    cfg.M_sub_block = std::max(1, single_M / L2_MN);
    cfg.N_sub_block = std::max(1, single_N / L2_MN);
    cfg.K_sub_block = 1;
  }
  return std::move(ret);
}

config_ptr gen_managed_matmul_core_t::get_default_transposed_a_config(
  const context_ptr &ctx) const {
  auto ret = reflection::general_object_t::make<managed_matmul_core_config_t>();
  managed_matmul_core_config_t &cfg
    = *ret.unchecked_get_as<managed_matmul_core_config_t>();
  const int num_threads = runtime_config_t::get().get_num_threads();
  const int iim_block = iim_block_;
  const int iin_block = iin_block_;
  const int iik_block = iik_block_;
  const int ori_M = in_tensors_[0].get_plain_dims()[0];
  const int ori_N = in_tensors_[1].get_plain_dims()[1];
  const int ori_K = in_tensors_[0].get_plain_dims()[1];
  const int M = utils::rnd_up(ori_M, iim_block);
  const int N = utils::rnd_up(ori_N, iin_block);
  const int K = utils::rnd_up(ori_K, iik_block);
  const int sizeofdtypeA
    = utils::get_sizeof_etype(in_tensors_[0].dtype_.as_etype());
  const int sizeofdtypeC
    = utils::get_sizeof_etype(out_tensors_[0].dtype_.as_etype());
  float cost = std::numeric_limits<float>::max();
  int split_m = 1;
  int split_n = 1;
  cfg.im_loop_order = 1;

  if (M == iim_block) {
    cfg.M_split_num = 1;
    if (K < 512) {
      cfg.N_split_num = num_threads;
    } else {
      for (auto i : get_splits(num_threads)) {
        float new_cost
          = std::abs(float(i * i) / float(num_threads) - float(N) / float(K));
        if (new_cost < cost) {
          cfg.N_split_num = i;
          cost = new_cost;
        }
      }
    }
  } else if (N == iin_block) {
    cfg.N_split_num = 1;
    if (K < 512) {
      cfg.M_split_num = num_threads;
    } else {
      for (auto i : get_splits(num_threads)) {
        float new_cost
          = std::abs(float(i * i) / float(num_threads) - float(M) / float(K));
        if (new_cost < cost) {
          cfg.M_split_num = i;
          cost = new_cost;
        }
      }
    }
  } else if (K == iik_block) {
    return get_default_config(ctx);
  } else {
    // for small K, no need to give splits
    if (K < 512) { return get_default_config(ctx); }
    auto K_split_candidates = get_splits(num_threads);
    int split_k = K_split_candidates.at(0);
    std::vector<int> K_real_split_candidates;
    if (K_split_candidates.size() > 2) {
      K_split_candidates.pop_back();
      K_split_candidates.erase(K_split_candidates.begin());
      for (auto k : K_split_candidates) {
        // make num_threads / k able to be further split for M and N
        if (get_splits(num_threads / k).size() > 2) {
          K_real_split_candidates.push_back(k);
        }
      }
      if (K_real_split_candidates.empty()) {
        K_real_split_candidates = std::move(K_split_candidates);
      }
      while (K_real_split_candidates.size() > 4) {
        // corner case, such as num_threads = 128
        K_real_split_candidates.pop_back();
      }
      while (K_real_split_candidates.size() < 4) {
        K_real_split_candidates.push_back(K_real_split_candidates.back());
      }

      float relative_K = float(ori_K) / float(std::min(ori_M, ori_N));
      if (relative_K < 8.0) {
        split_k = K_real_split_candidates.at(0);
      } else if (relative_K >= 8.0 && relative_K <= 12.0) {
        split_k = K_real_split_candidates.at(1);
      } else if (relative_K > 12.0 && relative_K <= 32.0) {
        split_k = K_real_split_candidates.at(2);
      } else {
        split_k = K_real_split_candidates.at(3);
      }
    } else {
      SC_MODULE_WARN << "not enough split candidates under " << num_threads
                     << " threads, may lead to poor performance.";
      if (float(ori_K) / float(std::min(ori_M, ori_N)) >= 128) {
        split_k = K_split_candidates.back();
      }
    }
    for (auto i : get_splits(num_threads / split_k)) {
      int num_M_block
        = utils::divide_and_ceil(M / iim_block, num_threads / i / split_k);
      int num_N_block = utils::divide_and_ceil(N / iin_block, i);
      int num_brgemm = num_M_block * num_N_block;
      int num_core = std::min(i, N / iin_block)
        * std::min(num_threads / i / split_k, M / iim_block);

      float new_cost = 0.0;
      if (K <= 4096 || (M <= 512 && N <= 512)) {
        // For small matmul, make M_split_num / N_split_num closer to M / N
        new_cost = std::abs(num_threads * N - i * i * M * split_k);
      } else {
        // Cost = Shape_efficient_weight *
        // (workload_balance + divide_N_plenty) / core_utilitizaiton
        // single core gemm prefers square shape for A and B.
        // For small workload, the A and B shape is not a key problem, but the
        // num_core and num_brgemm is important to performance. Use 2048 to
        // reduce the shape weight on small shape.
        new_cost = (1024 + M * i * split_k / num_threads + N / i)
          * (num_brgemm + 8 * i) / num_core;
      }
      if (new_cost < cost) {
        split_n = i;
        cost = new_cost;
      }
    }
    cfg.M_split_num = num_threads / split_k / split_n;
    cfg.N_split_num = split_n;
  }

  // big M or N needs to introduce X_sub_block for better cache reuse
  if (M >= 4096 || N >= 4096) {
    int single_M
      = utils::divide_and_ceil(M / iim_block, cfg.M_split_num) * iim_block;
    int single_N
      = utils::divide_and_ceil(N / iin_block, cfg.N_split_num) * iin_block;
    int single_K = utils::divide_and_ceil(K / iik_block,
                     num_threads / cfg.M_split_num / cfg.N_split_num)
      * iik_block;
    int L2_size = static_cast<int>(ctx->machine_.cpu_flags_.getDCacheSize(2));
    int single_K_threshold
      = (single_M * single_N * sizeofdtypeA < L2_size ? 2048 : 4096)
      / sizeofdtypeA;
    if (single_K >= single_K_threshold) {
      cfg.K_sub_block = utils::divide_and_ceil(single_K, single_K_threshold);
      int L2_K = utils::divide_and_ceil(single_K / iik_block, cfg.K_sub_block)
        * iik_block;
      // sizeofdtypeA* (M * K) + sizeofdtypeB * (N * K) + sizeofdtypeC(M * N) <=
      // L2_size, let M == N, then
      // 2 * sizeofdtypeA * M * K + sizeofdtypeC * M * M <= L2_size
      // Then M = (sqrt((2 * sizeofdtypeA * K) ^ 2 + 4 * sizeofdtypeC *
      // L2_size) - 2 * sizeofdtypeA * K)/ (2 * sizeofdtypeC)
      int L2_MN
        = (sqrt(pow(2 * sizeofdtypeA * L2_K, 2) + 4 * sizeofdtypeC * L2_size)
            - 2 * sizeofdtypeA * L2_K)
        / (2 * sizeofdtypeC);
      cfg.M_sub_block = std::max(1, single_M / L2_MN);
      cfg.N_sub_block = std::max(1, single_N / L2_MN);
    } else {
      // sizeofdtypeA * M * K + sizeofdtypeB * N * K <= L2_size
      // let M == N, then
      // M = L2_size / (2 * sizeofdtypeA * K)
      int L2_MN = L2_size / (2 * sizeofdtypeA * single_K);
      cfg.M_sub_block = std::max(1, single_M / L2_MN);
      cfg.N_sub_block = std::max(1, single_N / L2_MN);
      cfg.K_sub_block = 1;
    }
    return std::move(ret);
  }

  cfg.M_sub_block = 1;
  cfg.N_sub_block = 1;
  cfg.K_sub_block = 1;
  return std::move(ret);
}

static int suggest_aligned_block(const size_t plain_X,
  const size_t default_block, size_t min = 1, size_t align = 1) {
  if (plain_X < default_block) {
    if (plain_X <= min) {
      return min;
    } else if (plain_X < align) {
      return utils::rnd_up(plain_X, min);
    } else {
      return utils::rnd_up(plain_X, align);
    }
  }
  if (plain_X % default_block == 0) {
    return utils::rnd_up(default_block, align);
  }
  size_t num_X_block = utils::divide_and_ceil(plain_X, default_block);
  return utils::rnd_up(utils::divide_and_ceil(plain_X, num_X_block), align);
}

gen_managed_matmul_core_t::gen_managed_matmul_core_t(sc_op *owner,
  std::vector<logical_tensor_t> &&ins, std::vector<logical_tensor_t> &&outs)
  : parent(owner, std::move(ins), std::move(outs)) {
  COMPILE_ASSERT(
    in_tensors_.size() == 2, "input logical tensor size should be two.");
  COMPILE_ASSERT(
    out_tensors_.size() == 1, "output logical tensor size should be one.");
  const int64_t plain_M = get_mma_plain_dims()[0];
  const int64_t plain_K = get_mma_plain_dims()[1];
  const int64_t plain_N = get_mmb_plain_dims()[1];
  const int num_threads = runtime_config_t::get().get_num_threads();

  bool is_bf16 = get_A_dtype() == datatypes::bf16;
  bool is_f32 = get_A_dtype() == datatypes::f32;
  bool is_int8 = utils::is_one_of(get_A_dtype(), datatypes::u8, datatypes::s8);
  int64_t M_block_default = 64;
  int64_t N_block_default = 64;
  int64_t K_block_default = 64;
  bool is_special_fm = get_default_context()->machine_.cpu_flags_.family == 6
    && get_default_context()->machine_.cpu_flags_.model == 143;
  if (is_f32) {
    if (is_special_fm) {
      // prefer small blocks
      M_block_default = 16;
      N_block_default = 16;
      K_block_default = 16;
    } else {
      if (plain_M <= 256) { M_block_default = 32; }
    }
  } else if (is_bf16) {
    M_block_default = 32;
    N_block_default = 32;
    K_block_default = 32;
  } else {
    assert(utils::is_one_of(get_A_dtype(), datatypes::u8, datatypes::s8));
    M_block_default
      = (plain_M >= 1024 && !is_use_amx(get_default_context())) ? 64 : 32;
    N_block_default = 64;
    K_block_default = 64;
  }
  if (plain_N <= 512 && plain_K <= 512) {
    iim_block_
      = std::max((is_f32 && !is_special_fm && plain_M >= 64 && plain_M <= 128
                   && (plain_N >= 256 || plain_K >= 256))
          ? (int64_t)8
          : (int64_t)4,
        std::min(M_block_default,
          static_cast<int64_t>(utils::divide_and_ceil(plain_M, num_threads))));
  } else {
    iim_block_ = suggest_aligned_block(plain_M, M_block_default);
  }
  iin_block_ = suggest_aligned_block(plain_N, N_block_default, 1, 16);
  // f32 small M with small even K prefers padding iik_block to align 16
  // int8 perfers padding iik_block_ to algin 16 when M <=2048
  if (((is_f32 && plain_K < 16 && plain_K % 2 == 0 && plain_M <= 128)
        || (is_int8 && plain_M < 2048))
    && !is_special_fm) {
    iik_block_ = 16;
  } else {
    iik_block_
      = suggest_aligned_block(plain_K, K_block_default, is_f32 ? 1 : 4, 16);
  }
}

bool gen_managed_matmul_core_t::is_okay_to_prefetch(
  const managed_matmul_core_config_t &config, bool is_global) {
  const int num_threads = runtime_config_t::get().get_num_threads();
  if (!in_tensors_[1].get_format().is_blocking()
    || num_threads / config.M_split_num / config.N_split_num > 1) {
    return false;
  }
  return true;
}

void gen_managed_matmul_core_t::generate_prefetcher_body_for_tensor(
  const context_ptr &ctx, const managed_matmul_core_config_t &config,
  const std::vector<expr> &func_args, const std::vector<expr> &ins,
  const std::vector<int> &indices) {
  auto lookup = func_args[0];
  auto expected = func_args[1];
  auto tid = func_args[2];
  bool is_int8 = utils::is_one_of(get_A_dtype(), datatypes::u8, datatypes::s8);
  uint64_t sizeof_dtype = utils::get_sizeof_type(get_A_dtype());
  bool is_bf16 = get_A_dtype() == datatypes::bf16;
  int N = static_cast<int>(
        utils::rnd_up(in_tensors_[1].get_plain_dims()[1], iin_block_)),
      K = static_cast<int>(
        utils::rnd_up(in_tensors_[0].get_plain_dims()[1], iik_block_));
  auto num_threads = runtime_config_t::get().get_num_threads();
  auto threads_per_group = num_threads / config.M_split_num;
  if (config.N_split_num * config.M_split_num != num_threads) {
    _if_(tid % threads_per_group >= config.N_split_num) {
      _return_(UINT64_C(0));
    }
  }
  _var_(cnt, datatypes::index);
  cnt = 0;
  expr n_idx, N_single_thr_size, X_bigger_num;
  _for_(n_s, tid % threads_per_group, tid % threads_per_group + 1) {
    N_single_thr_size = get_balance211_length(
      N / iin_block_, config.N_split_num, n_s, n_idx, X_bigger_num);
    N_single_thr_size = N_single_thr_size * iin_block_;
    n_idx = n_idx * iin_block_;
    // only K_split_num=1 for convenience
    _for_(n_b, 0, config.N_sub_block) {
      expr n_b_idx, n_b_bigger_num, k_b_idx, k_b_bigger_num;
      _var_init_(n_o_end, datatypes::s32,
        get_balance211_length(N_single_thr_size / iin_block_,
          config.N_sub_block, n_b, n_b_idx, n_b_bigger_num));
      _for_(k_b, 0, config.K_sub_block) {
        _for_(n_o, 0, n_o_end) {
          _var_init_(n_start_idx, datatypes::index,
            n_idx + n_b_idx * iin_block_
              + ((n_o + tid) % n_o_end) * iin_block_);
          _var_init_(bs, datatypes::s32,
            get_balance211_length(K / iik_block_, config.K_sub_block, k_b,
              k_b_idx, k_b_bigger_num));
          _var_init_(k_start_idx, datatypes::index, 0 + k_b_idx * iik_block_);
          _for_(i, 0, iik_block_ * iin_block_ * bs, 512 / sizeof_dtype) {
            _if_(lookup[0] == expected) { _return_(cnt); }
            cnt = cnt + 1;
            _for_(j, 0, 512 / sizeof_dtype, 64 / sizeof_dtype) {
              std::vector<expr> indices;
              if (get_A_dtype() == datatypes::f32) {
                indices = {n_start_idx / expr(iin_block_),
                  k_start_idx / expr(iik_block_), 0, i + j};
              } else {
                indices = {n_start_idx / expr(iin_block_),
                  k_start_idx / expr(iik_block_), 0, 0, i + j};
              }
              auto tptr = builder::tensor_ptr(ins[0], indices);
              builder::get_current_builder()->push_evaluate(
                make_expr<intrin_call_node>(intrin_type::prefetch,
                  std::vector<expr> {tptr}, any_map_t {{"locality", 1}}));
            }
          }
        }
      }
    }
  }
  _return_(cnt);
}

float gen_managed_matmul_core_t::get_gflop() const {
  const int64_t plain_M = get_mma_plain_dims()[0];
  const int64_t plain_K = get_mma_plain_dims()[1];
  const int64_t plain_N = get_mmb_plain_dims()[1];
  return get_a_batch_dims().empty() && get_a_batch_dims().empty()
    ? 2.f * plain_M * plain_N * plain_K / 1e9
    : 2.f * plain_M * plain_N * plain_K
      * math_utils::get_dims_product(
        get_a_batch_dims().size() > get_b_batch_dims().size()
          ? get_a_batch_dims()
          : get_b_batch_dims())
      / 1e9;
}

void gen_managed_matmul_core_t::schedule_loops(context_ptr ctx,
  const managed_matmul_core_config_t &config, stmt body,
  std::vector<for_loop> &fors) const {}

void gen_managed_matmul_core_t::single_thread_reorder_matmul_call(
  context_ptr ctx, const logical_tensor_t &ta, const logical_tensor_t &tb,
  const logical_tensor_t &tc, const managed_matmul_core_config_t &config,
  const expr &M, const expr &N, const expr &K, const expr &m_idx,
  const expr &n_idx, const expr &k_idx, const expr &A, expr &B, const expr &C,
  int dtype_block, fusion_manager *fusion, const expr &m_s, const expr &n_s,
  std::vector<int> &M_anchor_info, std::vector<int> &N_anchor_info,
  std::vector<int> &K_anchor_info, bool is_partial, const expr &k_s) const {
  expr M_sub_block = config.M_sub_block, N_sub_block = config.N_sub_block,
       K_sub_block = config.K_sub_block;
  if (config.im_loop_order == 0) {
    SC_MODULE_WARN << "Pre fusion vnni reorder in managed_matmul_core requires "
                      "N loop to be OUTER loop to minimize the repeat time of "
                      "reorder. Rectify im_loop_order==1 automatically here.";
  }
  for_loop im_k, im_m, im_n, o_im_m;
  int ori_M = static_cast<int>(ta.get_plain_dims()[0]),
      ori_K = static_cast<int>(ta.get_plain_dims()[1]),
      ori_N = static_cast<int>(tb.get_plain_dims()[1]);
  expr tid = builtin::get_thread_id_func()();
  expr B_vnni_tensor;
  _tensor_(B_vnni, get_B_dtype(),
    {utils::divide_and_ceil(ori_N, iin_block_),
      utils::divide_and_ceil(ori_K, iik_block_), iik_block_ / dtype_block,
      iin_block_, dtype_block});
  B_vnni_tensor = B_vnni;

  _for_(n_b, 0, N_sub_block) {
    _named_for_(o_im_m, m_b, 0, M_sub_block) {
      expr m_b_idx, n_b_idx, k_b_idx, m_b_bigger_num, n_b_bigger_num,
        k_b_bigger_num;
      _var_init_(m_o_end, datatypes::s32,
        get_balance211_length(
          M / iim_block_, M_sub_block, m_b, m_b_idx, m_b_bigger_num));
      _var_init_(n_o_end, datatypes::s32,
        get_balance211_length(
          N / iin_block_, N_sub_block, n_b, n_b_idx, n_b_bigger_num));
      _named_for_(im_k, k_b, 0, K_sub_block) {
        // general matmul_core loops
        _var_init_(bs, datatypes::s32,
          get_balance211_length(
            K / iik_block_, K_sub_block, k_b, k_b_idx, k_b_bigger_num));
        _var_init_(k_start_idx, datatypes::index, k_idx + k_b_idx * iik_block_);
        _named_for_(im_n, n_o, 0, n_o_end) {
          // rolling N
          _var_init_(n_start_idx, datatypes::index,
            n_idx + n_b_idx * iin_block_
              + ((n_o + tid) % n_o_end) * iin_block_);
          // do reorder here
          {
            trace_guard_t tg(ctx, "vnni_reorder_B");
            B_vnni_tensor->attr()[tensor_shrinker_attrs::should_shrink]
              = tensor_shrinker_t::shrink_info_t {
                {n_start_idx / iin_block_, k_start_idx / iik_block_, 0, 0, 0},
                {1,
                  utils::divide_and_ceil(
                    is_partial ? K_anchor_info[1] : ori_K, iik_block_),
                  iik_block_ / dtype_block, iin_block_, dtype_block},
                stmts()};
            auto commit_reorder_B
              = [&](int length_N, int length_K, const expr &ko) {
                  slice_range b_old_range
                    = {{k_start_idx + ko * iik_block_, length_K},
                      {n_start_idx, length_N}};
                  if (in_tensors_[1].get_format() == sc_data_format_t::NK()) {
                    std::swap(b_old_range[0], b_old_range[1]);
                  }
                  ops::commit_op(ctx, "reorder",
                    {tensor_slice(B, std::move(b_old_range))},
                    {tensor_slice(B_vnni_tensor,
                      {{n_start_idx / iin_block_, 1},
                        {(k_start_idx + ko * iik_block_) / iik_block_,
                          utils::divide_and_ceil(length_K, iik_block_)},
                        {0, iik_block_ / dtype_block}, {0, iin_block_},
                        {0, dtype_block}})},
                    {graph_tensor::make({ori_K, ori_N},
                      in_tensors_[1].get_format(), datatypes::bf16)},
                    {},
                    {{"out_format",
                      sc_data_format_t::NKkn2k(iik_block_, iin_block_)}});
                };

            int N_real_split = std::min(
              static_cast<int>(utils::divide_and_ceil(ori_N, iin_block_)),
              config.N_split_num);
            int K_real_split = std::min(
              static_cast<int>(utils::divide_and_ceil(ori_K, iik_block_)),
              runtime_config_t::get().get_num_threads() / config.M_split_num
                / config.N_split_num);
            // discuss reorder type, which has different input slice
            if (ori_K % iik_block_ == 0 && ori_N % iin_block_ == 0) {
              // no padding
              _for_(ko, 0, bs) { commit_reorder_B(iin_block_, iik_block_, ko); }
            } else if (ori_K % iik_block_ == 0) {
              // padding on N axis
              _for_(ko, 0, bs) {
                _if_(n_s == N_real_split - 1 && n_b == N_sub_block - 1
                  && n_o == n_o_end - 1) {
                  commit_reorder_B(ori_N % iin_block_, iik_block_, ko);
                }
                _else_ { commit_reorder_B(iin_block_, iik_block_, ko); }
              }
            } else if (ori_N % iin_block_ == 0) {
              // padding on K axis
              _for_(ko, 0, bs) {
                _if_(k_s == K_real_split - 1 && k_b == K_sub_block - 1
                  && ko == bs - 1) {
                  commit_reorder_B(iin_block_, ori_K % iik_block_, ko);
                }
                _else_ { commit_reorder_B(iin_block_, iik_block_, ko); }
              }
            } else {
              // padding on both K and N axes
              _for_(ko, 0, bs) {
                _if_(n_s == N_real_split - 1 && n_b == N_sub_block - 1
                  && n_o == n_o_end - 1) {
                  _if_(k_s == K_real_split - 1 && k_b == K_sub_block - 1
                    && ko == bs - 1) {
                    commit_reorder_B(
                      ori_N % iin_block_, ori_K % iik_block_, ko);
                  }
                  _else_ {
                    commit_reorder_B(ori_N % iin_block_, iik_block_, ko);
                  }
                }
                _else_ {
                  _if_(k_s == K_real_split - 1 && k_b == K_sub_block - 1
                    && ko == bs - 1) {
                    commit_reorder_B(iin_block_, ori_K % iik_block_, ko);
                  }
                  _else_ { commit_reorder_B(iin_block_, iik_block_, ko); }
                }
              }
            }
          }
          _named_for_(im_m, m_o, 0, m_o_end) {
            // rolling M
            _var_init_(m_start_idx, datatypes::index,
              m_idx + m_b_idx * iim_block_
                + ((m_o + tid) % m_o_end) * iim_block_);
            std::vector<expr> aidx = ta.get_format() == sc_data_format_t::MK()
              ? std::vector<expr> {m_start_idx, k_start_idx}
              : std::vector<expr> {
                m_start_idx / iim_block_, k_start_idx / iik_block_, 0, 0};
            std::vector<expr> bidx = std::vector<expr> {
              n_start_idx / iin_block_, k_start_idx / iik_block_, 0, 0, 0};
            std::vector<expr> cidx;
            if (is_partial) {
              cidx = !tc.get_format().is_blocking()
                ? std::vector<expr> {m_b_idx * iim_block_
                    + ((m_o + tid) % m_o_end) * iim_block_,
                  n_b_idx * iin_block_ + ((n_o + tid) % n_o_end) * iin_block_}
                : std::vector<expr> {m_b_idx + (m_o + tid) % m_o_end,
                  n_b_idx + (n_o + tid) % n_o_end, 0, 0};
              cidx.insert(cidx.begin(), k_s);
            } else {
              cidx = !tc.get_format().is_blocking()
                ? std::vector<expr> {m_start_idx, n_start_idx}
                : std::vector<expr> {
                  m_start_idx / iim_block_, n_start_idx / iin_block_, 0, 0};
            }
            auto LDA
              = ta.get_format() == sc_data_format_t::MK() ? ori_K : iik_block_;
            auto LDB = iin_block_;
            auto LDC = !tc.get_format().is_blocking()
              ? (is_partial ? N_anchor_info[1] : ori_N)
              : iin_block_;
            auto stride_a = ta.get_format() == sc_data_format_t::MK()
              ? iik_block_
              : iim_block_ * iik_block_;
            auto stride_b = iik_block_ * iin_block_;
            _if_(k_b == 0) {
              sc::builtin::brgemm_init_update(tensor_ptr(A, aidx),
                tensor_ptr(B_vnni_tensor, bidx), tensor_ptr(C, cidx), bs,
                iim_block_, iin_block_, iik_block_, LDA, LDB, LDC, stride_a,
                stride_b, ta.dtype_, tb.dtype_);
            }
            _else_ {
              sc::builtin::brgemm_update(tensor_ptr(A, aidx),
                tensor_ptr(B_vnni_tensor, bidx), tensor_ptr(C, cidx), bs,
                iim_block_, iin_block_, iik_block_, LDA, LDB, LDC, stride_a,
                stride_b, ta.dtype_, tb.dtype_);
            }
            if (fusion && !is_partial) {
              _if_(k_b == K_sub_block - 1) {
                fusion->create_output_fusion_anchor({tensor_slice(C,
                  !tc.get_format().is_blocking()
                    ? std::vector<std::pair<expr, expr>> {{m_start_idx,
                                                            expr(iim_block_)},
                      {n_start_idx, expr(iin_block_)}}
                    : std::vector<std::pair<expr, expr>> {
                      {m_start_idx / iim_block_, 1},
                      {n_start_idx / iin_block_, 1}, {0, expr(iim_block_)},
                      {0, expr(iin_block_)}})});
              }
            }
          }
        }
      }
      if (fusion && !is_partial) {
        // 16 cases in total
        if (M_anchor_info[1] == M_anchor_info[2]
          && N_anchor_info[1] == N_anchor_info[2]
          && M_anchor_info[1] / iim_block_ % config.M_sub_block == 0
          && N_anchor_info[1] / iin_block_ % config.N_sub_block == 0) {
          // case 1: no imbalance on single core, X_sub_block can be
          // dividedevenly
          fusion->create_output_fusion_anchor({tensor_slice(C,
            !tc.get_format().is_blocking()
              ? std::vector<std::pair<expr, expr>> {{m_idx
                                                        + m_b_idx * iim_block_,
                                                      M_anchor_info[1]
                                                        / config.M_sub_block},
                {n_idx + n_b_idx * iin_block_,
                  N_anchor_info[1] / config.N_sub_block}}
              : std::vector<std::pair<expr, expr>> {
                {(m_idx + m_b_idx * iim_block_) / expr(iim_block_),
                  M_anchor_info[1] / iim_block_ / config.M_sub_block},
                {(n_idx + n_b_idx * iin_block_) / expr(iin_block_),
                  N_anchor_info[1] / iin_block_ / config.N_sub_block},
                {0, expr(iim_block_)}, {0, expr(iin_block_)}})});
        } else {
          slice_range_list mm_multi_slice;
          // order:X_anchor_info[1] -> X_anchor_info[2]
          for (int p = 0; p < 2; p++) {
            for (int q = 0; q < 2; q++) {
              for (int i = 0; i < 2; i++) {
                for (int j = 0; j < 2; j++) {
                  if (!tc.get_format().is_blocking()) {
                    auto length_M = M_anchor_info[p + 1] / config.M_sub_block;
                    if (M_anchor_info[p + 1] / iim_block_ % config.M_sub_block
                      != 0) {
                      length_M += (1 - i) * iim_block_;
                    }
                    auto length_N = N_anchor_info[q + 1] / config.N_sub_block;
                    if (N_anchor_info[q + 1] / iin_block_ % config.N_sub_block
                      != 0) {
                      length_N += (1 - j) * iin_block_;
                    }
                    assert(length_M > 0 && length_N > 0);
                    mm_multi_slice.emplace_back(
                      slice_range {{m_idx + m_b_idx * iim_block_, length_M},
                        {n_idx + n_b_idx * iin_block_, length_N}});
                  } else {
                    auto length_M
                      = M_anchor_info[p + 1] / iim_block_ / config.M_sub_block;
                    if (M_anchor_info[p + 1] / iim_block_ % config.M_sub_block
                      != 0) {
                      length_M += 1 - i;
                    }
                    auto length_N
                      = N_anchor_info[q + 1] / iin_block_ / config.N_sub_block;
                    if (N_anchor_info[q + 1] / iin_block_ % config.N_sub_block
                      != 0) {
                      length_N += 1 - j;
                    }
                    assert(length_M > 0 && length_N > 0);
                    mm_multi_slice.emplace_back(slice_range {
                      {(m_idx + m_b_idx * iim_block_) / expr(iim_block_),
                        length_M},
                      {(n_idx + n_b_idx * iin_block_) / expr(iin_block_),
                        length_N},
                      {0, expr(iim_block_)}, {0, expr(iin_block_)}});
                  }
                }
              }
            }
          }
          auto gen_iter_anchor = [&]() {
            builder::ir_builder_t bd_helper;
            bd_helper.push_scope();
            _var_init_(anchor_iter, datatypes::index, UINT64_C(0));
            // TODO(xxx): reduce the if-else node in IR
            _if_(m_s < M_anchor_info[0]) {
              // 0-8
              _if_(n_s < N_anchor_info[0]) {
                // 0-4
                _if_(m_b < m_b_bigger_num) {
                  _if_(n_b < n_b_bigger_num) { anchor_iter = UINT64_C(0); }
                  _else_ { anchor_iter = UINT64_C(1); }
                }
                _else_ {
                  _if_(n_b < n_b_bigger_num) { anchor_iter = UINT64_C(2); }
                  _else_ { anchor_iter = UINT64_C(3); }
                }
              }
              _else_ {
                _if_(m_b < m_b_bigger_num) {
                  _if_(n_b < n_b_bigger_num) { anchor_iter = UINT64_C(4); }
                  _else_ { anchor_iter = UINT64_C(5); }
                }
                _else_ {
                  _if_(n_b < n_b_bigger_num) { anchor_iter = UINT64_C(6); }
                  _else_ { anchor_iter = UINT64_C(7); }
                }
              }
            }
            _else_ {
              _if_(n_s < N_anchor_info[0]) {
                _if_(m_b < m_b_bigger_num) {
                  _if_(n_b < n_b_bigger_num) { anchor_iter = UINT64_C(8); }
                  _else_ { anchor_iter = UINT64_C(9); }
                }
                _else_ {
                  _if_(n_b < n_b_bigger_num) { anchor_iter = UINT64_C(10); }
                  _else_ { anchor_iter = UINT64_C(11); }
                }
              }
              _else_ {
                _if_(m_b < m_b_bigger_num) {
                  _if_(n_b < n_b_bigger_num) { anchor_iter = UINT64_C(12); }
                  _else_ { anchor_iter = UINT64_C(13); }
                }
                _else_ {
                  _if_(n_b < n_b_bigger_num) { anchor_iter = UINT64_C(14); }
                  _else_ { anchor_iter = UINT64_C(15); }
                }
              }
            }
            auto scope_helper = bd_helper.pop_scope();
            return std::make_pair(anchor_iter, scope_helper);
          };
          expr anchor_iter;
          stmt scope_helper;
          std::tie(anchor_iter, scope_helper) = gen_iter_anchor();
          fusion->create_output_fusion_anchor(
            anchor_iter, C, mm_multi_slice, scope_helper);
        }
      }
    }
  }
  if (config.K_sub_block > 1) {
    im_m->attr()[stmt_attr_key::reduce_root_loop]
      = std::weak_ptr<stmt_base_t>(o_im_m.impl);
  }
}

void gen_managed_matmul_core_t::single_thread_matmul_call(
  const logical_tensor_t &ta, const logical_tensor_t &tb,
  const logical_tensor_t &tc, const managed_matmul_core_config_t &config,
  const expr &M, const expr &N, const expr &K, const expr &m_idx,
  const expr &n_idx, const expr &k_idx, const expr &A, const expr &B,
  const expr &C, int dtype_block, fusion_manager *fusion, const expr &m_s,
  const expr &n_s, std::vector<int> &M_anchor_info,
  std::vector<int> &N_anchor_info, bool is_partial, const expr &k_s) const {
  expr M_sub_block = config.M_sub_block, N_sub_block = config.N_sub_block,
       K_sub_block = config.K_sub_block;
  for_loop im_k, im_m, im_n, o_im_n;
  int ori_M = static_cast<int>(ta.get_plain_dims()[0]),
      ori_K = static_cast<int>(ta.get_plain_dims()[1]),
      ori_N = static_cast<int>(tb.get_plain_dims()[1]);
  expr tid = builtin::get_thread_id_func()();
  _for_(m_b, 0, M_sub_block) {
    _named_for_(o_im_n, n_b, 0, N_sub_block) {
      expr m_b_idx, n_b_idx, k_b_idx, m_b_bigger_num, n_b_bigger_num,
        k_b_bigger_num;
      _var_init_(m_o_end, datatypes::s32,
        get_balance211_length(
          M / iim_block_, M_sub_block, m_b, m_b_idx, m_b_bigger_num));
      _var_init_(n_o_end, datatypes::s32,
        get_balance211_length(
          N / iin_block_, N_sub_block, n_b, n_b_idx, n_b_bigger_num));
      _named_for_(im_k, k_b, 0, K_sub_block) {
        // general matmul_core loops
        _named_for_(im_m, m_o, 0, m_o_end) {
          _named_for_(im_n, n_o, 0, n_o_end) {
            // rolling M and N
            _var_init_(m_start_idx, datatypes::index,
              m_idx + m_b_idx * iim_block_
                + ((m_o + tid) % m_o_end) * iim_block_);
            _var_init_(n_start_idx, datatypes::index,
              n_idx + n_b_idx * iin_block_
                + ((n_o + tid) % n_o_end) * iin_block_);
            _var_init_(bs, datatypes::s32,
              get_balance211_length(
                K / iik_block_, K_sub_block, k_b, k_b_idx, k_b_bigger_num));
            _var_init_(
              k_start_idx, datatypes::index, k_idx + k_b_idx * iik_block_);
            // create input anchor for B if necessary
            if (fusion && in_tensors_[1].get_format().is_blocking()
              && K.isa<constant>()
              && ((get_expr_as_int(K) / iik_block_ % config.K_sub_block) == 0)
              && ori_M <= 512) {
              slice_range B_slice = {{n_start_idx / iin_block_, 1},
                {k_start_idx / iik_block_, K / iik_block_ / K_sub_block},
                {0, iik_block_ / dtype_block}, {0, iin_block_}};
              if (dtype_block > 1) { B_slice.push_back({0, dtype_block}); }
              fusion->create_input_fusion_anchor(
                {tensor_slice(B, std::move(B_slice))});
            }
            std::vector<expr> aidx = ta.get_format() == sc_data_format_t::MK()
              ? std::vector<expr> {m_start_idx, k_start_idx}
              : std::vector<expr> {
                m_start_idx / iim_block_, k_start_idx / iik_block_, 0, 0};
            std::vector<expr> bidx = dtype_block > 1
              ? std::vector<expr> {n_start_idx / iin_block_,
                k_start_idx / iik_block_, 0, 0, 0}
              : (!tb.get_format().is_blocking()
                  ? std::vector<expr> {k_start_idx, n_start_idx}
                  : std::vector<expr> {
                    n_start_idx / iin_block_, k_start_idx / iik_block_, 0, 0});
            std::vector<expr> cidx;
            if (is_partial) {
              cidx = !tc.get_format().is_blocking()
                ? std::vector<expr> {m_b_idx * iim_block_
                    + ((m_o + tid) % m_o_end) * iim_block_,
                  n_b_idx * iin_block_ + ((n_o + tid) % n_o_end) * iin_block_}
                : std::vector<expr> {m_b_idx + (m_o + tid) % m_o_end,
                  n_b_idx + (n_o + tid) % n_o_end, 0, 0};
              cidx.insert(cidx.begin(), k_s);
            } else {
              cidx = !tc.get_format().is_blocking()
                ? std::vector<expr> {m_start_idx, n_start_idx}
                : std::vector<expr> {
                  m_start_idx / iim_block_, n_start_idx / iin_block_, 0, 0};
            }
            auto LDA
              = ta.get_format() == sc_data_format_t::MK() ? ori_K : iik_block_;
            auto LDB = !tb.get_format().is_blocking() ? ori_N : iin_block_;
            auto LDC = !tc.get_format().is_blocking()
              ? (is_partial ? N_anchor_info[1] : ori_N)
              : iin_block_;
            auto stride_a = ta.get_format() == sc_data_format_t::MK()
              ? iik_block_
              : iim_block_ * iik_block_;
            auto stride_b = !tb.get_format().is_blocking()
              ? iik_block_ * ori_N
              : iik_block_ * iin_block_;
            _if_(k_b == 0) {
              sc::builtin::brgemm_init_update(tensor_ptr(A, aidx),
                tensor_ptr(B, bidx), tensor_ptr(C, cidx), bs, iim_block_,
                iin_block_, iik_block_, LDA, LDB, LDC, stride_a, stride_b,
                ta.dtype_, tb.dtype_);
            }
            _else_ {
              sc::builtin::brgemm_update(tensor_ptr(A, aidx),
                tensor_ptr(B, bidx), tensor_ptr(C, cidx), bs, iim_block_,
                iin_block_, iik_block_, LDA, LDB, LDC, stride_a, stride_b,
                ta.dtype_, tb.dtype_);
            }
            if (fusion && !is_partial) {
              _if_(k_b == K_sub_block - 1) {
                fusion->create_output_fusion_anchor({tensor_slice(C,
                  !tc.get_format().is_blocking()
                    ? std::vector<std::pair<expr, expr>> {{m_start_idx,
                                                            expr(iim_block_)},
                      {n_start_idx, expr(iin_block_)}}
                    : std::vector<std::pair<expr, expr>> {
                      {m_start_idx / iim_block_, 1},
                      {n_start_idx / iin_block_, 1}, {0, expr(iim_block_)},
                      {0, expr(iin_block_)}})});
              }
            }
          }
        }
      }
      if (fusion && !is_partial) {
        // 16 cases in total
        if (M_anchor_info[1] == M_anchor_info[2]
          && N_anchor_info[1] == N_anchor_info[2]
          && M_anchor_info[1] / iim_block_ % config.M_sub_block == 0
          && N_anchor_info[1] / iin_block_ % config.N_sub_block == 0) {
          // case 1: no imbalance on single core, X_sub_block can be
          // dividedevenly
          fusion->create_output_fusion_anchor({tensor_slice(C,
            !tc.get_format().is_blocking()
              ? std::vector<std::pair<expr, expr>> {{m_idx
                                                        + m_b_idx * iim_block_,
                                                      M_anchor_info[1]
                                                        / config.M_sub_block},
                {n_idx + n_b_idx * iin_block_,
                  N_anchor_info[1] / config.N_sub_block}}
              : std::vector<std::pair<expr, expr>> {
                {(m_idx + m_b_idx * iim_block_) / expr(iim_block_),
                  M_anchor_info[1] / iim_block_ / config.M_sub_block},
                {(n_idx + n_b_idx * iin_block_) / expr(iin_block_),
                  N_anchor_info[1] / iin_block_ / config.N_sub_block},
                {0, expr(iim_block_)}, {0, expr(iin_block_)}})});
        } else {
          slice_range_list mm_multi_slice;
          // order:X_anchor_info[1] -> X_anchor_info[2]
          for (int p = 0; p < 2; p++) {
            for (int q = 0; q < 2; q++) {
              for (int i = 0; i < 2; i++) {
                for (int j = 0; j < 2; j++) {
                  if (!tc.get_format().is_blocking()) {
                    auto length_M = M_anchor_info[p + 1] / config.M_sub_block;
                    if (M_anchor_info[p + 1] / iim_block_ % config.M_sub_block
                      != 0) {
                      length_M += (1 - i) * iim_block_;
                    }
                    auto length_N = N_anchor_info[q + 1] / config.N_sub_block;
                    if (N_anchor_info[q + 1] / iin_block_ % config.N_sub_block
                      != 0) {
                      length_N += (1 - j) * iin_block_;
                    }
                    assert(length_M > 0 && length_N > 0);
                    mm_multi_slice.emplace_back(
                      slice_range {{m_idx + m_b_idx * iim_block_, length_M},
                        {n_idx + n_b_idx * iin_block_, length_N}});
                  } else {
                    auto length_M
                      = M_anchor_info[p + 1] / iim_block_ / config.M_sub_block;
                    if (M_anchor_info[p + 1] / iim_block_ % config.M_sub_block
                      != 0) {
                      length_M += 1 - i;
                    }
                    auto length_N
                      = N_anchor_info[q + 1] / iin_block_ / config.N_sub_block;
                    if (N_anchor_info[q + 1] / iin_block_ % config.N_sub_block
                      != 0) {
                      length_N += 1 - j;
                    }
                    assert(length_M > 0 && length_N > 0);
                    mm_multi_slice.emplace_back(slice_range {
                      {(m_idx + m_b_idx * iim_block_) / expr(iim_block_),
                        length_M},
                      {(n_idx + n_b_idx * iin_block_) / expr(iin_block_),
                        length_N},
                      {0, expr(iim_block_)}, {0, expr(iin_block_)}});
                  }
                }
              }
            }
          }
          auto gen_iter_anchor = [&]() {
            builder::ir_builder_t bd_helper;
            bd_helper.push_scope();
            _var_init_(anchor_iter, datatypes::index, UINT64_C(0));
            // TODO(xxx): reduce the if-else node in IR
            _if_(m_s < M_anchor_info[0]) {
              // 0-8
              _if_(n_s < N_anchor_info[0]) {
                // 0-4
                _if_(m_b < m_b_bigger_num) {
                  _if_(n_b < n_b_bigger_num) { anchor_iter = UINT64_C(0); }
                  _else_ { anchor_iter = UINT64_C(1); }
                }
                _else_ {
                  _if_(n_b < n_b_bigger_num) { anchor_iter = UINT64_C(2); }
                  _else_ { anchor_iter = UINT64_C(3); }
                }
              }
              _else_ {
                _if_(m_b < m_b_bigger_num) {
                  _if_(n_b < n_b_bigger_num) { anchor_iter = UINT64_C(4); }
                  _else_ { anchor_iter = UINT64_C(5); }
                }
                _else_ {
                  _if_(n_b < n_b_bigger_num) { anchor_iter = UINT64_C(6); }
                  _else_ { anchor_iter = UINT64_C(7); }
                }
              }
            }
            _else_ {
              _if_(n_s < N_anchor_info[0]) {
                _if_(m_b < m_b_bigger_num) {
                  _if_(n_b < n_b_bigger_num) { anchor_iter = UINT64_C(8); }
                  _else_ { anchor_iter = UINT64_C(9); }
                }
                _else_ {
                  _if_(n_b < n_b_bigger_num) { anchor_iter = UINT64_C(10); }
                  _else_ { anchor_iter = UINT64_C(11); }
                }
              }
              _else_ {
                _if_(m_b < m_b_bigger_num) {
                  _if_(n_b < n_b_bigger_num) { anchor_iter = UINT64_C(12); }
                  _else_ { anchor_iter = UINT64_C(13); }
                }
                _else_ {
                  _if_(n_b < n_b_bigger_num) { anchor_iter = UINT64_C(14); }
                  _else_ { anchor_iter = UINT64_C(15); }
                }
              }
            }
            auto scope_helper = bd_helper.pop_scope();
            return std::make_pair(anchor_iter, scope_helper);
          };
          expr anchor_iter;
          stmt scope_helper;
          std::tie(anchor_iter, scope_helper) = gen_iter_anchor();
          fusion->create_output_fusion_anchor(
            anchor_iter, C, mm_multi_slice, scope_helper);
        }
      }
    }
  }
  if (config.K_sub_block > 1 && config.im_loop_order != 1) {
    im_n->attr()[stmt_attr_key::reduce_root_loop]
      = std::weak_ptr<stmt_base_t>(o_im_n.impl);
  }
  if (config.im_loop_order == 1) {
    im_m->reorder(im_k->body_, {im_n, im_m});
    im_m->attr()[stmt_attr_key::reduce_root_loop]
      = std::weak_ptr<stmt_base_t>(o_im_n.impl);
  }
}

/**
 * For each single thread we may deal with different size of matmuls
 * For either axis, we have following candidates:
 * 1) X_block_size
 * 2) X_ib_block_size (imbalance)
      X_block_size and X_ib_block_size are calculated by balance211 algrithm.
 Specially, X_block_size >= X_ib_block_size, and the gap is either 0 or
 iix_block_.
 * */
bool gen_managed_matmul_core_t::generate(context_ptr ctx,
  const managed_matmul_core_config_t &config, fusion_manager *fusion,
  const std::vector<expr> &inputs, const std::vector<expr> &outputs,
  std::vector<for_loop> &loops) const {
  if (!ctx->flags_.mixed_fusion_) {
    SC_MODULE_WARN << "Managed matmul core has some conflicts with old fusion "
                      "strategy, which may lead to wrong calculation.";
  }
  // Init
  int M_split_num = config.M_split_num, N_split_num = config.N_split_num;
  int num_threads = runtime_config_t::get().get_num_threads();
  int K_split_num = num_threads / M_split_num / N_split_num;
  int M_sub_block = config.M_sub_block, N_sub_block = config.N_sub_block,
      K_sub_block = config.K_sub_block, im_loop_order = config.im_loop_order;
  int M = static_cast<int>(
        utils::rnd_up(in_tensors_[0].get_plain_dims()[0], iim_block_)),
      K = static_cast<int>(
        utils::rnd_up(in_tensors_[0].get_plain_dims()[1], iik_block_)),
      N = static_cast<int>(
        utils::rnd_up(in_tensors_[1].get_plain_dims()[1], iin_block_));
  int M_block_size
    = utils::divide_and_ceil(M / iim_block_, M_split_num) * iim_block_;
  int M_ib_block_size = M / iim_block_ / M_split_num * iim_block_;
  int N_block_size
    = utils::divide_and_ceil(N / iin_block_, N_split_num) * iin_block_;
  int N_ib_block_size = N / iin_block_ / N_split_num * iin_block_;
  int K_block_size
    = utils::divide_and_ceil(K / iik_block_, K_split_num) * iik_block_;
  int K_ib_block_size = K / iik_block_ / K_split_num * iik_block_;

  if (M_ib_block_size == 0) { M_ib_block_size = M_block_size; }
  if (N_ib_block_size == 0) { N_ib_block_size = N_block_size; }
  if (K_ib_block_size == 0) { K_ib_block_size = K_block_size; }

  // M, N block num with block size equals to X_block_size
  int M_blk_num = (M - (M_block_size - iim_block_) * M_split_num) / iim_block_;
  int N_blk_num = (N - (N_block_size - iin_block_) * N_split_num) / iin_block_;
  int K_blk_num = (K - (K_block_size - iik_block_) * K_split_num) / iik_block_;

  COMPILE_ASSERT(M_block_size / iim_block_ >= M_sub_block
      && M_ib_block_size / iim_block_ >= M_sub_block,
    "bad M_sub_block given");
  COMPILE_ASSERT(N_block_size / iin_block_ >= N_sub_block
      && N_ib_block_size / iin_block_ >= N_sub_block,
    "bad N_sub_block given");
  COMPILE_ASSERT(K_block_size / iik_block_ >= K_sub_block
      && K_ib_block_size / iik_block_ >= K_sub_block,
    "bad K_sub_block given");

  int dtype_block = 1;
  auto A_dtype = get_A_dtype();
  auto B_dtype = get_B_dtype();
  if (B_dtype == datatypes::bf16) {
    dtype_block = 2;
  } else if (utils::is_one_of(B_dtype, datatypes::u8, datatypes::s8)) {
    dtype_block = 4;
  }
  if (dtype_block > 1) {
    COMPILE_ASSERT(in_tensors_[1].get_format().blocks_[2] == -1
        || in_tensors_[1].get_format().blocks_[2] == dtype_block
        || (dtype_block == 2 && in_tensors_[1].get_format().blocks_[2] == 0),
      "Wrong data format of B");
  }

  expr C = outputs[op_params_t::out_C];
  expr A = inputs[op_params_t::in_A];
  expr B = inputs[op_params_t::in_B];
  // used for anchor construction when K_split_num==1 && K_sub_block>1
  std::vector<int> M_anchor_info = {M_blk_num, M_block_size, M_ib_block_size},
                   N_anchor_info = {N_blk_num, N_block_size, N_ib_block_size},
                   K_anchor_info = {K_blk_num, K_block_size, K_ib_block_size};
  for_loop mloop;
  int M_real_split = std::min(
    static_cast<int>(utils::divide_and_ceil(M, iim_block_)), M_split_num);
  int N_real_split = std::min(
    static_cast<int>(utils::divide_and_ceil(N, iin_block_)), N_split_num);
  int K_real_split = std::min(
    static_cast<int>(utils::divide_and_ceil(K, iik_block_)), K_split_num);

  if (K_split_num == 1) {
    expr m_idx, n_idx, M_single_thr_size, N_single_thr_size, X_bigger_num;
    _named_for_(
      mloop, m_s, 0, M_real_split, 1, for_type::PARALLEL, M_split_num) {
      _for_(n_s, 0, N_real_split, 1, for_type::PARALLEL, N_split_num) {
        M_single_thr_size = get_balance211_length(
          M / iim_block_, M_split_num, m_s, m_idx, X_bigger_num);
        M_single_thr_size = M_single_thr_size * iim_block_;
        m_idx = m_idx * iim_block_;

        N_single_thr_size = get_balance211_length(
          N / iin_block_, N_split_num, n_s, n_idx, X_bigger_num);
        N_single_thr_size = N_single_thr_size * iin_block_;
        n_idx = n_idx * iin_block_;
        _for_(k_s, 0, K_split_num, 1,
          M_split_num * N_split_num == num_threads ? for_type::NORMAL
                                                   : for_type::PARALLEL,
          M_split_num * N_split_num == num_threads ? 0 : K_split_num) {
          // create input anchor for A if necessary
          if (fusion && in_tensors_[0].get_format().is_blocking()
            && M * K <= 1024 * 1024) {
            fusion->create_input_fusion_anchor({tensor_slice(A,
              {{m_idx / iim_block_,
                 utils::divide_and_ceil(M_block_size, iim_block_)},
                {0, K / iik_block_}, {0, iim_block_}, {0, iik_block_}})});
          }
          if (in_tensors_[0].get_format() == sc_data_format_t::NK()
            && A_dtype == datatypes::bf16) {
            trace_guard_t tg(ctx, "transpose_A");
            expr A_trans_tensor;
            _tensor_(A_trans, get_A_dtype(),
              {M / iim_block_, K / iik_block_, iim_block_, iik_block_});
            A_trans_tensor = A_trans;
            A_trans_tensor->attr()[tensor_shrinker_attrs::should_shrink]
              = tensor_shrinker_t::shrink_info_t {{m_idx / iim_block_, 0, 0, 0},
                {M_block_size / iim_block_, K / iik_block_, iim_block_,
                  iik_block_},
                stmts()};
            // do transpose A
            auto commit_reorder_A = [&](int length_M) {
              int length_K = in_tensors_[0].get_plain_dims()[1];
              ops::commit_op(ctx, "reorder",
                {tensor_slice(A, {{0, length_K}, {m_idx, length_M}})},
                {tensor_slice(A_trans_tensor,
                  {{m_idx / iim_block_,
                     utils::divide_and_ceil(length_M, iim_block_)},
                    {0, K / iik_block_}, {0, iim_block_}, {0, iik_block_}})},
                {graph_tensor::make(in_tensors_[0].get_plain_dims(),
                  sc_data_format_t::NK(), datatypes::bf16)},
                {},
                {{"out_format",
                  sc_data_format_t::MKmk(iim_block_, iik_block_)}});
            };
            if (in_tensors_[0].get_plain_dims()[0] % iim_block_ == 0) {
              if (M_block_size == M_ib_block_size) {
                commit_reorder_A(M_block_size);
              } else {
                _if_(m_s < M_blk_num) { commit_reorder_A(M_block_size); }
                _else_ { commit_reorder_A(M_ib_block_size); }
              }
            } else {
              // has padding on M axis
              if (M_block_size == M_ib_block_size) {
                _if_(m_s == M_real_split - 1) {
                  commit_reorder_A(M_block_size - iim_block_
                    + in_tensors_[0].get_plain_dims()[0] % iim_block_);
                }
                _else_ { commit_reorder_A(M_block_size); }
              } else {
                _if_(m_s < M_blk_num) { commit_reorder_A(M_block_size); }
                _else_ {
                  _if_(m_s == M_real_split - 1) {
                    commit_reorder_A(M_ib_block_size - iim_block_
                      + in_tensors_[0].get_plain_dims()[0] % iim_block_);
                  }
                  _else_ { commit_reorder_A(M_ib_block_size); }
                }
              }
            }
            A = A_trans_tensor;
          }
          if (utils::is_one_of(in_tensors_[1].get_format(),
                sc_data_format_t::MK(), sc_data_format_t::NK())
            && B_dtype == datatypes::bf16) {
            single_thread_reorder_matmul_call(ctx, in_tensors_[0],
              in_tensors_[1], out_tensors_[0], config, M_single_thr_size,
              N_single_thr_size, (int)utils::rnd_up(K, iik_block_), m_idx,
              n_idx, k_s, A, B, C, dtype_block, fusion, m_s, n_s, M_anchor_info,
              N_anchor_info, K_anchor_info);
          } else {
            single_thread_matmul_call(in_tensors_[0], in_tensors_[1],
              out_tensors_[0], config, M_single_thr_size, N_single_thr_size,
              (int)utils::rnd_up(K, iik_block_), m_idx, n_idx, k_s, A, B, C,
              dtype_block, fusion, m_s, n_s, M_anchor_info, N_anchor_info);
          }
        }
        if (fusion) {
          slice_range_list mm_multi_slice;
          // only 2 candidates will exist
          for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
              auto M_length = i == 0 ? M_block_size : M_ib_block_size;
              auto N_length = j == 0 ? N_block_size : N_ib_block_size;
              if (out_tensors_[0].get_format().is_blocking()) {
                mm_multi_slice.emplace_back(slice_range {
                  {m_idx / expr(iim_block_), M_length / iim_block_},
                  {n_idx / expr(iin_block_), N_length / iin_block_},
                  {0, iim_block_}, {0, iin_block_}});
              } else {
                mm_multi_slice.emplace_back(
                  slice_range {{m_idx, M_length}, {n_idx, N_length}});
              }
            }
          }
          if (M_block_size == M_ib_block_size
            && N_block_size == N_ib_block_size) {
            if (out_tensors_[0].get_format().is_blocking()) {
              fusion->create_output_fusion_anchor({tensor_slice(C,
                {{m_idx / expr(iim_block_), M_block_size / iim_block_},
                  {n_idx / expr(iin_block_), N_block_size / iin_block_},
                  {0, iim_block_}, {0, iin_block_}})});
            } else {
              fusion->create_output_fusion_anchor({tensor_slice(
                C, {{m_idx, M_block_size}, {n_idx, N_block_size}})});
            }
          } else if (M_block_size == M_ib_block_size) {
            // differnt length on N
            mm_multi_slice.pop_back();
            mm_multi_slice.pop_back();
            assert(mm_multi_slice.size() == 2);
            auto gen_iter_anchor = [&]() {
              builder::ir_builder_t bd_helper;
              bd_helper.push_scope();
              _var_init_(middle_anchor_iter, datatypes::index, UINT64_C(0));
              _if_(n_s < N_blk_num) { middle_anchor_iter = UINT64_C(0); }
              _else_ { middle_anchor_iter = UINT64_C(1); }
              auto scope_helper = bd_helper.pop_scope();
              return std::make_pair(middle_anchor_iter, scope_helper);
            };
            expr middle_anchor_iter;
            stmt scope_helper;
            std::tie(middle_anchor_iter, scope_helper) = gen_iter_anchor();
            fusion->create_output_fusion_anchor(
              middle_anchor_iter, C, mm_multi_slice, scope_helper);
          } else if (N_block_size == N_ib_block_size) {
            // different length on M
            mm_multi_slice.pop_back();
            mm_multi_slice.erase(mm_multi_slice.begin() + 1);
            assert(mm_multi_slice.size() == 2);
            auto gen_iter_anchor = [&]() {
              builder::ir_builder_t bd_helper;
              bd_helper.push_scope();
              _var_init_(middle_anchor_iter, datatypes::index, UINT64_C(0));
              _if_(m_s < M_blk_num) { middle_anchor_iter = UINT64_C(0); }
              _else_ { middle_anchor_iter = UINT64_C(1); }
              auto scope_helper = bd_helper.pop_scope();
              return std::make_pair(middle_anchor_iter, scope_helper);
            };
            expr middle_anchor_iter;
            stmt scope_helper;
            std::tie(middle_anchor_iter, scope_helper) = gen_iter_anchor();
            fusion->create_output_fusion_anchor(
              middle_anchor_iter, C, mm_multi_slice, scope_helper);
          } else {
            // different length on both M and N
            auto gen_iter_anchor = [&]() {
              builder::ir_builder_t bd_helper;
              bd_helper.push_scope();
              _var_init_(middle_anchor_iter, datatypes::index, UINT64_C(0));
              _if_(m_s < M_blk_num) {
                _if_(n_s < N_blk_num) { middle_anchor_iter = UINT64_C(0); }
                _else_ { middle_anchor_iter = UINT64_C(1); }
              }
              _else_ {
                _if_(n_s < N_blk_num) { middle_anchor_iter = UINT64_C(2); }
                _else_ { middle_anchor_iter = UINT64_C(3); }
              }
              auto scope_helper = bd_helper.pop_scope();
              return std::make_pair(middle_anchor_iter, scope_helper);
            };
            expr middle_anchor_iter;
            stmt scope_helper;
            std::tie(middle_anchor_iter, scope_helper) = gen_iter_anchor();
            fusion->create_output_fusion_anchor(
              middle_anchor_iter, C, mm_multi_slice, scope_helper);
          }
        }
      }
      // give explict anchor when N_split_num==1 to enable tensor shrink
      if (fusion && N_split_num == 1) {
        if (M_block_size == M_ib_block_size) {
          if (out_tensors_[0].get_format().is_blocking()) {
            fusion->create_output_fusion_anchor({tensor_slice(C,
              {{m_idx / expr(iim_block_), M_block_size / iim_block_},
                {0, utils::divide_and_ceil(N, iin_block_)},
                {0, expr(iim_block_)}, {0, expr(iin_block_)}})});
          } else {
            fusion->create_output_fusion_anchor(
              {tensor_slice(C, {{m_idx, M_block_size}, {0, N}})});
          }
        } else {
          slice_range_list mm_multi_slice;
          if (out_tensors_[0].get_format().is_blocking()) {
            mm_multi_slice
              = {{{m_idx / expr(iim_block_), M_block_size / iim_block_},
                   {0, utils::divide_and_ceil(N, iin_block_)},
                   {0, expr(iim_block_)}, {0, expr(iin_block_)}},
                {{m_idx / expr(iim_block_), M_ib_block_size / iim_block_},
                  {0, utils::divide_and_ceil(N, iin_block_)},
                  {0, expr(iim_block_)}, {0, expr(iin_block_)}}};
          } else {
            mm_multi_slice = {{{m_idx, M_block_size}, {0, N}},
              {{m_idx, M_ib_block_size}, {0, N}}};
          }
          auto gen_iter_anchor = [&]() {
            builder::ir_builder_t bd_helper;
            bd_helper.push_scope();
            _var_init_(outer_anchor_iter, datatypes::index, UINT64_C(0));
            _if_(m_s < M_blk_num) { outer_anchor_iter = UINT64_C(0); }
            _else_ { outer_anchor_iter = UINT64_C(1); }
            auto scope_helper = bd_helper.pop_scope();
            return std::make_pair(outer_anchor_iter, scope_helper);
          };
          expr outer_anchor_iter;
          stmt scope_helper;
          std::tie(outer_anchor_iter, scope_helper) = gen_iter_anchor();
          fusion->create_output_fusion_anchor(
            outer_anchor_iter, C, mm_multi_slice, scope_helper);
        }
      }
    }
  } else {
    // write into a temp buffer and then do reduce
    sc_dims out_tmp_buf_shape = out_tensors_[0].get_format().is_blocking()
      ? sc_dims {M_block_size / iim_block_, N_block_size / iin_block_,
        iim_block_, iin_block_}
      : sc_dims {M_block_size, N_block_size};
    out_tmp_buf_shape.insert(out_tmp_buf_shape.begin(), K_real_split);
    std::vector<expr> out_tmp_buf_shape_expr;
    out_tmp_buf_shape_expr.reserve(out_tmp_buf_shape.size());
    for (auto dim : out_tmp_buf_shape) {
      out_tmp_buf_shape_expr.emplace_back(dim2unsigned(dim));
    }
    auto out_dtype = utils::is_one_of(A_dtype, datatypes::u8, datatypes::s8)
      ? datatypes::s32
      : datatypes::f32;
    expr m_idx, n_idx, k_idx, M_single_thr_size, N_single_thr_size,
      X_bigger_num;
    _named_for_(
      mloop, m_s, 0, M_real_split, 1, for_type::PARALLEL, M_split_num) {
      _for_(n_s, 0, N_real_split, 1, for_type::PARALLEL, N_split_num) {
        _tensor_(out_tmp_buf, out_dtype, out_tmp_buf_shape_expr);
        M_single_thr_size = get_balance211_length(
          M / iim_block_, M_split_num, m_s, m_idx, X_bigger_num);
        M_single_thr_size = M_single_thr_size * iim_block_;
        m_idx = m_idx * iim_block_;

        N_single_thr_size = get_balance211_length(
          N / iin_block_, N_split_num, n_s, n_idx, X_bigger_num);
        N_single_thr_size = N_single_thr_size * iin_block_;
        n_idx = n_idx * iin_block_;

        _for_(k_s, 0, K_real_split, 1, for_type::PARALLEL, K_split_num) {
          expr K_single_thr_size;
          if (K_block_size == K_ib_block_size) {
            K_single_thr_size = K_block_size;
            k_idx = k_s * K_block_size;
          } else {
            K_single_thr_size = get_balance211_length(
              K / iik_block_, K_split_num, k_s, k_idx, X_bigger_num);
            K_single_thr_size = K_single_thr_size * iik_block_;
            k_idx = k_idx * iik_block_;
          }
          // create input anchor for A if necessary
          if (fusion && in_tensors_[0].get_format().is_blocking()
            && (K_block_size == K_ib_block_size) && M * K <= 1024 * 1024) {
            fusion->create_input_fusion_anchor({tensor_slice(A,
              {{m_idx / iim_block_,
                 utils::divide_and_ceil(M_block_size, iim_block_)},
                {k_idx / iik_block_, K_block_size / iik_block_},
                {0, iim_block_}, {0, iik_block_}})});
          }
          if (in_tensors_[0].get_format() == sc_data_format_t::NK()
            && A_dtype == datatypes::bf16) {
            trace_guard_t tg(ctx, "transpose_A");
            expr A_trans_tensor;
            _tensor_(A_trans, get_A_dtype(),
              {M / iim_block_, K / iik_block_, iim_block_, iik_block_});
            A_trans_tensor = A_trans;
            A_trans_tensor->attr()[tensor_shrinker_attrs::should_shrink]
              = tensor_shrinker_t::shrink_info_t {
                {m_idx / iim_block_, k_idx / iik_block_, 0, 0},
                {M_block_size / iim_block_, K_block_size / iik_block_,
                  iim_block_, iik_block_},
                stmts()};
            // do transpose A
            auto commit_reorder_A = [&](int length_M, int length_K) {
              ops::commit_op(ctx, "reorder",
                {tensor_slice(A, {{k_idx, length_K}, {m_idx, length_M}})},
                {tensor_slice(A_trans_tensor,
                  {{m_idx / iim_block_,
                     utils::divide_and_ceil(length_M, iim_block_)},
                    {k_idx / iik_block_,
                      utils::divide_and_ceil(length_K, iik_block_)},
                    {0, iim_block_}, {0, iik_block_}})},
                {graph_tensor::make(in_tensors_[0].get_plain_dims(),
                  sc_data_format_t::NK(), datatypes::bf16)},
                {},
                {{"out_format",
                  sc_data_format_t::MKmk(iim_block_, iik_block_)}});
            };
            auto discuss_K = [&](int length_M) {
              if (K_block_size == K_ib_block_size) {
                commit_reorder_A(length_M, K_block_size);
              } else {
                _if_(k_s < K_blk_num) {
                  commit_reorder_A(length_M, K_block_size);
                }
                _else_ { commit_reorder_A(length_M, K_ib_block_size); }
              }
            };
            auto discuss_K2 = [&](int length_M) {
              if (K_block_size == K_ib_block_size) {
                _if_(k_s == K_real_split - 1) {
                  commit_reorder_A(length_M,
                    K_block_size - iik_block_
                      + in_tensors_[0].get_plain_dims()[1] % iik_block_);
                }
                _else_ { commit_reorder_A(length_M, K_block_size); }
              } else {
                _if_(k_s < K_blk_num) {
                  commit_reorder_A(length_M, K_block_size);
                }
                _else_ {
                  _if_(k_s == K_real_split - 1) {
                    commit_reorder_A(length_M,
                      K_ib_block_size - iik_block_
                        + in_tensors_[0].get_plain_dims()[1] % iik_block_);
                  }
                  _else_ { commit_reorder_A(length_M, K_ib_block_size); }
                }
              }
            };
            auto discuss_M = [&](int length_K) {
              if (M_block_size == M_ib_block_size) {
                commit_reorder_A(M_block_size, length_K);
              } else {
                _if_(m_s < M_blk_num) {
                  commit_reorder_A(M_block_size, length_K);
                }
                _else_ { commit_reorder_A(M_ib_block_size, length_K); }
              }
            };
            if (in_tensors_[0].get_plain_dims()[0] % iim_block_ == 0
              && in_tensors_[0].get_plain_dims()[1] % iik_block_ == 0) {
              // no padding
              if (M_block_size == M_ib_block_size) {
                discuss_K(M_block_size);
              } else {
                _if_(m_s < M_blk_num) { discuss_K(M_block_size); }
                _else_ { discuss_K(M_ib_block_size); }
              }
            } else if (in_tensors_[0].get_plain_dims()[1] % iik_block_ == 0) {
              // has padding on M axis only
              if (M_block_size == M_ib_block_size) {
                _if_(m_s == M_real_split - 1) {
                  discuss_K(M_block_size - iim_block_
                    + in_tensors_[0].get_plain_dims()[0] % iim_block_);
                }
                _else_ { discuss_K(M_block_size); }
              } else {
                _if_(m_s < M_blk_num) { discuss_K(M_block_size); }
                _else_ {
                  _if_(m_s == M_real_split - 1) {
                    discuss_K(M_ib_block_size - iim_block_
                      + in_tensors_[0].get_plain_dims()[0] % iim_block_);
                  }
                  _else_ { discuss_K(M_ib_block_size); }
                }
              }
            } else if (in_tensors_[0].get_plain_dims()[0] % iim_block_ == 0) {
              // has padding on K axis only
              if (K_block_size == K_ib_block_size) {
                _if_(k_s == K_real_split - 1) {
                  discuss_M(K_block_size - iik_block_
                    + in_tensors_[0].get_plain_dims()[1] % iik_block_);
                }
                _else_ { discuss_M(K_block_size); }
              } else {
                _if_(k_s < K_blk_num) { discuss_M(K_block_size); }
                _else_ {
                  _if_(k_s == K_real_split - 1) {
                    discuss_M(K_ib_block_size - iik_block_
                      + in_tensors_[0].get_plain_dims()[1] % iik_block_);
                  }
                  _else_ { discuss_M(K_ib_block_size); }
                }
              }
            } else {
              // has padding on both M and K axes
              if (M_block_size == M_ib_block_size) {
                _if_(m_s == M_real_split - 1) {
                  discuss_K2(M_block_size - iim_block_
                    + in_tensors_[0].get_plain_dims()[0] % iim_block_);
                }
                _else_ { discuss_K2(M_block_size); }
              } else {
                _if_(m_s < M_blk_num) { discuss_K2(M_block_size); }
                _else_ {
                  _if_(m_s == M_real_split - 1) {
                    discuss_K2(M_ib_block_size - iim_block_
                      + in_tensors_[0].get_plain_dims()[0] % iim_block_);
                  }
                  _else_ { discuss_K2(M_ib_block_size); }
                }
              }
            }
            A = A_trans_tensor;
          }
          if (utils::is_one_of(in_tensors_[1].get_format(),
                sc_data_format_t::MK(), sc_data_format_t::NK())
            && B_dtype == datatypes::bf16) {
            single_thread_reorder_matmul_call(ctx, in_tensors_[0],
              in_tensors_[1], out_tensors_[0], config, M_single_thr_size,
              N_single_thr_size, K_single_thr_size, m_idx, n_idx, k_idx, A, B,
              out_tmp_buf, dtype_block, fusion, m_s, n_s, M_anchor_info,
              N_anchor_info, K_anchor_info, true, k_s);
          } else {
            single_thread_matmul_call(in_tensors_[0], in_tensors_[1],
              out_tensors_[0], config, M_single_thr_size, N_single_thr_size,
              K_single_thr_size, m_idx, n_idx, k_idx, A, B, out_tmp_buf,
              dtype_block, fusion, m_s, n_s, M_anchor_info, N_anchor_info, true,
              k_s);
          }
        }
        // do reduce here
        for_loop rm, rn;
        int lanes = 1;
        if (iin_block_ / 16 && iin_block_ % 16 == 0) {
          lanes = vectorize_step(ctx, get_C_dtype().type_code_, 16);
        }
        expr M_single_thr_num_block
          = divide_and_ceil(M_single_thr_size, iim_block_);
        expr N_single_thr_num_block
          = divide_and_ceil(N_single_thr_size, iin_block_);
        if (out_tensors_[0].get_format().is_blocking()) {
          trace_guard_t tg(ctx, "blocking_post_reduce");
          _for_(lm_ln, 0, M_single_thr_num_block * N_single_thr_num_block, 1,
            for_type::PARALLEL, K_split_num) { //
            expr lm = lm_ln / N_single_thr_num_block;
            expr ln = lm_ln % N_single_thr_num_block;
            builtin::mem_zero(
              tensor_ptr(
                C, {m_idx / iim_block_ + lm, n_idx / iin_block_ + ln, 0, 0}),
              iim_block_ * iin_block_, out_dtype);
            _for_(lks, 0, K_real_split, 1) {
              _for_(lmo, 0, iim_block_) {
                _for_(lno, 0, iin_block_, lanes) {
                  C[span_t({m_idx / iim_block_ + lm, n_idx / iin_block_ + ln,
                             lmo, lno},
                    lanes)]
                    = builder::make_add(
                      C[span_t({m_idx / iim_block_ + lm,
                                 n_idx / iin_block_ + ln, lmo, lno},
                        lanes)],
                      out_tmp_buf[span_t({lks, lm, ln, lmo, lno}, lanes)]);
                }
              }
            }
            if (fusion) {
              fusion->create_output_fusion_anchor({tensor_slice(C,
                {{m_idx / expr(iim_block_) + lm, 1},
                  {n_idx / expr(iin_block_) + ln, 1}, {0, expr(iim_block_)},
                  {0, expr(iin_block_)}})});
            }
          }
        } else {
          trace_guard_t tg(ctx, "plain_post_reduce");
          builtin::dnnl_brgemm_init(tensor_ptr(C, {m_idx, n_idx}),
            M_single_thr_size, N_single_thr_size, N, out_dtype, 0);
          _for_(lm_ln, 0, M_single_thr_size * N_single_thr_size, lanes,
            for_type::PARALLEL, K_split_num) {
            expr lm = lm_ln / N_single_thr_size;
            expr ln = lm_ln % N_single_thr_size;
            _for_(lks, 0, K_real_split, 1) {
              C[span_t({m_idx + lm, n_idx + ln}, lanes)]
                = builder::make_add(C[span_t({m_idx + lm, n_idx + ln}, lanes)],
                  out_tmp_buf[span_t({lks, lm, ln}, lanes)]);
            }
          }
          if (fusion) {
            slice_range_list mm_multi_slice;
            // only 2 candidates will exist
            for (int i = 0; i < 2; i++) {
              for (int j = 0; j < 2; j++) {
                auto M_length = i == 0 ? M_block_size : M_ib_block_size;
                auto N_length = j == 0 ? N_block_size : N_ib_block_size;
                if (out_tensors_[0].get_format().is_blocking()) {
                  mm_multi_slice.emplace_back(slice_range {
                    {m_idx / expr(iim_block_), M_length / iim_block_},
                    {n_idx / expr(iin_block_), N_length / iin_block_},
                    {0, iim_block_}, {0, iin_block_}});
                } else {
                  mm_multi_slice.emplace_back(
                    slice_range {{m_idx, M_length}, {n_idx, N_length}});
                }
              }
            }
            if (M_block_size == M_ib_block_size
              && N_block_size == N_ib_block_size) {
              if (out_tensors_[0].get_format().is_blocking()) {
                fusion->create_output_fusion_anchor({tensor_slice(C,
                  {{m_idx / expr(iim_block_), M_block_size / iim_block_},
                    {n_idx / expr(iin_block_), N_block_size / iin_block_},
                    {0, iim_block_}, {0, iin_block_}})});
              } else {
                fusion->create_output_fusion_anchor({tensor_slice(
                  C, {{m_idx, M_block_size}, {n_idx, N_block_size}})});
              }
            } else if (M_block_size == M_ib_block_size) {
              // differnt length on N
              mm_multi_slice.pop_back();
              mm_multi_slice.pop_back();
              assert(mm_multi_slice.size() == 2);
              auto gen_iter_anchor = [&]() {
                builder::ir_builder_t bd_helper;
                bd_helper.push_scope();
                _var_init_(inner_anchor_iter, datatypes::index, UINT64_C(0));
                _if_(n_s < N_blk_num) { inner_anchor_iter = UINT64_C(0); }
                _else_ { inner_anchor_iter = UINT64_C(1); }
                auto scope_helper = bd_helper.pop_scope();
                return std::make_pair(inner_anchor_iter, scope_helper);
              };
              expr inner_anchor_iter;
              stmt scope_helper;
              std::tie(inner_anchor_iter, scope_helper) = gen_iter_anchor();
              fusion->create_output_fusion_anchor(
                inner_anchor_iter, C, mm_multi_slice, scope_helper);
            } else if (N_block_size == N_ib_block_size) {
              // different length on M
              mm_multi_slice.pop_back();
              mm_multi_slice.erase(mm_multi_slice.begin() + 1);
              assert(mm_multi_slice.size() == 2);
              auto gen_iter_anchor = [&]() {
                builder::ir_builder_t bd_helper;
                bd_helper.push_scope();
                _var_init_(inner_anchor_iter, datatypes::index, UINT64_C(0));
                _if_(m_s < M_blk_num) { inner_anchor_iter = UINT64_C(0); }
                _else_ { inner_anchor_iter = UINT64_C(1); }
                auto scope_helper = bd_helper.pop_scope();
                return std::make_pair(inner_anchor_iter, scope_helper);
              };
              expr inner_anchor_iter;
              stmt scope_helper;
              std::tie(inner_anchor_iter, scope_helper) = gen_iter_anchor();
              fusion->create_output_fusion_anchor(
                inner_anchor_iter, C, mm_multi_slice, scope_helper);
            } else {
              // different length on both M and N
              auto gen_iter_anchor = [&]() {
                builder::ir_builder_t bd_helper;
                bd_helper.push_scope();
                _var_init_(inner_anchor_iter, datatypes::index, UINT64_C(0));
                _if_(m_s < M_blk_num) {
                  _if_(n_s < N_blk_num) { inner_anchor_iter = UINT64_C(0); }
                  _else_ { inner_anchor_iter = UINT64_C(1); }
                }
                _else_ {
                  _if_(n_s < N_blk_num) { inner_anchor_iter = UINT64_C(2); }
                  _else_ { inner_anchor_iter = UINT64_C(3); }
                }
                auto scope_helper = bd_helper.pop_scope();
                return std::make_pair(inner_anchor_iter, scope_helper);
              };
              expr inner_anchor_iter;
              stmt scope_helper;
              std::tie(inner_anchor_iter, scope_helper) = gen_iter_anchor();
              fusion->create_output_fusion_anchor(
                inner_anchor_iter, C, mm_multi_slice, scope_helper);
            }
          }
        }
      }
      // give explict anchor when N_split_num==1 to enable tensor shrink
      if (fusion && N_split_num == 1) {
        if (M_block_size == M_ib_block_size) {
          if (out_tensors_[0].get_format().is_blocking()) {
            fusion->create_output_fusion_anchor({tensor_slice(C,
              {{m_idx / expr(iim_block_), M_block_size / iim_block_},
                {0, utils::divide_and_ceil(N, iin_block_)},
                {0, expr(iim_block_)}, {0, expr(iin_block_)}})});
          } else {
            fusion->create_output_fusion_anchor(
              {tensor_slice(C, {{m_idx, M_block_size}, {0, N}})});
          }
        } else {
          slice_range_list mm_multi_slice;
          if (out_tensors_[0].get_format().is_blocking()) {
            mm_multi_slice
              = {{{m_idx / expr(iim_block_), M_block_size / iim_block_},
                   {0, utils::divide_and_ceil(N, iin_block_)},
                   {0, expr(iim_block_)}, {0, expr(iin_block_)}},
                {{m_idx / expr(iim_block_), M_ib_block_size / iim_block_},
                  {0, utils::divide_and_ceil(N, iin_block_)},
                  {0, expr(iim_block_)}, {0, expr(iin_block_)}}};
          } else {
            mm_multi_slice = {{{m_idx, M_block_size}, {0, N}},
              {{m_idx, M_ib_block_size}, {0, N}}};
          }
          auto gen_iter_anchor = [&]() {
            builder::ir_builder_t bd_helper;
            bd_helper.push_scope();
            _var_init_(outer_anchor_iter, datatypes::index, UINT64_C(0));
            _if_(m_s < M_blk_num) { outer_anchor_iter = UINT64_C(0); }
            _else_ { outer_anchor_iter = UINT64_C(1); }
            auto scope_helper = bd_helper.pop_scope();
            return std::make_pair(outer_anchor_iter, scope_helper);
          };
          expr outer_anchor_iter;
          stmt scope_helper;
          std::tie(outer_anchor_iter, scope_helper) = gen_iter_anchor();
          fusion->create_output_fusion_anchor(
            outer_anchor_iter, C, mm_multi_slice, scope_helper);
        }
      }
    }
  }

  mloop->attr()[stmt_attr_key::parallel_merge_loop_granularity] = iim_block_;
  mloop->attr()[stmt_attr_key::parallel_loop_balanced]
    = M_block_size == M_ib_block_size;

  mloop->attr()[stmt_attr_key::loop_axis_hint]
    = bound_axis {{0}, {1}, {-1}, {0}, {1}};
  loops = {mloop};
  return true;
}
} // namespace ops
} // namespace sc
