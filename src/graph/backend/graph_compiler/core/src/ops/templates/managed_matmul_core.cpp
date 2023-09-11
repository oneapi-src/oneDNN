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
#include <atomic>
#include <cmath>
#include <limits>
#include <string>
#include "../fusible/memory_movement.hpp"
#include "utils.hpp"
#include <compiler/ir/builder.hpp>
#include <compiler/ir/builtin.hpp>
#include <compiler/ir/easy_build.hpp>
#include <compiler/ir/graph/dynamic_internal_info.hpp>
#include <compiler/ir/graph/fusion_mgr.hpp>
#include <compiler/ir/graph/utils.hpp>
#include <compiler/ir/pass/ir_copy.hpp>
#include <compiler/ir/transform/dyn_tsr_transform.hpp>
#include <compiler/ir/transform/index2var.hpp>
#include <compiler/ir/transform/loop_transform.hpp>
#include <compiler/ir/transform/scope_flatten.hpp>
#include <compiler/ir/transform/tensor_shrink.hpp>
#include <ops/matmul_core.hpp>
#include <ops/templates/commit_op.hpp>
#include <runtime/config.hpp>
#include <runtime/dynamic_dispatch/ops/config.hpp>
#include <runtime/parallel.hpp>
#include <runtime/trace.hpp>
#include <util/any_map.hpp>
#include <util/math_utils.hpp>
#include <util/reflection.hpp>

SC_MODULE(ops.managed_matmul_core)
using namespace dnnl::impl::graph::gc::builder;
namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
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

bool gen_managed_matmul_core_t::is_valid_config(
  const context_ptr &ctx, const managed_matmul_core_config_t &config) const {
  auto num_threads = runtime_config_t::get().get_num_threads();
  if (config.M_split_num * config.N_split_num > num_threads) { return false; }
  if (config.M_sub_block <= 0 || config.N_sub_block <= 0
    || config.K_sub_block <= 0) {
    return false;
  }
  const int M = static_cast<int>(
    utils::rnd_up(in_tensors_[0].get_plain_dims()[0], iim_block_));
  const int K = static_cast<int>(
    utils::rnd_up(in_tensors_[0].get_plain_dims()[1], iik_block_));
  const int N = static_cast<int>(
    utils::rnd_up(in_tensors_[1].get_plain_dims()[1], iin_block_));
  int M_block_size
    = utils::divide_and_ceil(M / iim_block_, config.M_split_num) * iim_block_;
  int M_ib_block_size = M / iim_block_ / config.M_split_num * iim_block_;
  int N_block_size
    = utils::divide_and_ceil(N / iin_block_, config.N_split_num) * iin_block_;
  int N_ib_block_size = N / iin_block_ / config.N_split_num * iin_block_;
  int K_block_size = utils::divide_and_ceil(K / iik_block_,
                       num_threads / config.N_split_num / config.M_split_num)
    * iik_block_;
  int K_ib_block_size = K / iik_block_
    / (num_threads / config.N_split_num / config.M_split_num) * iik_block_;
  if (M_ib_block_size == 0) { M_ib_block_size = M_block_size; }
  if (N_ib_block_size == 0) { N_ib_block_size = N_block_size; }
  if (K_ib_block_size == 0) { K_ib_block_size = K_block_size; }

  if (M_block_size / iim_block_ < config.M_sub_block
    || M_ib_block_size / iim_block_ < config.M_sub_block) {
    return false;
  }
  if (N_block_size / iin_block_ < config.N_sub_block
    || N_ib_block_size / iin_block_ < config.N_sub_block) {
    return false;
  }
  if (K_block_size / iik_block_ < config.K_sub_block
    || K_ib_block_size / iik_block_ < config.K_sub_block) {
    return false;
  }
  return true;
}

bool is_prefetch_debug_mode() {
  auto &cfg = runtime_config_t::get();
  if (cfg.trace_mode_ == 1
    && utils::string_endswith(cfg.trace_out_path_, "pref.log")) {
    return true;
  }
  return false;
}

void trace_prefetch_for_debug(const expr &addr) {
  if (!is_prefetch_debug_mode()) { return; }
  static auto trace_id = register_traced_func("pref");
  builder::get_current_builder()->push_evaluate(builtin::make_trace(trace_id,
    builder::make_cast(datatypes::s32,
      builder::make_reinterpret(addr, datatypes::index) >> UINT64_C(32)),
    builder::make_cast(
      datatypes::s32, builder::make_reinterpret(addr, datatypes::index))));
}

void trace_brgemm_for_debug(
  const expr &Baddr, const expr &bs, const expr &N, const expr &K) {
  if (!is_prefetch_debug_mode()) { return; }
  static auto trace_id = register_traced_func("brg");
  builder::get_current_builder()->push_evaluate(builtin::make_trace(trace_id,
    builder::make_cast(datatypes::s32,
      builder::make_reinterpret(Baddr, datatypes::index) >> UINT64_C(32)),
    builder::make_cast(
      datatypes::s32, builder::make_reinterpret(Baddr, datatypes::index))));
  builder::get_current_builder()->push_evaluate(builtin::make_trace(trace_id, 0,
    bs * N * K
      * static_cast<int>(
        utils::get_sizeof_type(Baddr->dtype_.get_pointer_element()))));
}

config_ptr_vec gen_managed_matmul_core_t::get_dynamic_config_candidates(
  const context_ptr &ctx) const {
  config_ptr_vec ret;
  int num_threads = runtime_config_t::get().get_num_threads();
  auto M_split_candidates = get_splits(num_threads);
  auto N_split_candidates = get_splits(num_threads);
  // todo: add more candidates in following prs.
  std::vector<int> MNK_sub_candidates = {1, 4};
  for (auto &M_split_num : M_split_candidates) {
    for (auto &N_split_num : N_split_candidates) {
      if (num_threads % (M_split_num * N_split_num) == 0) {
        for (auto &M_sub_block : MNK_sub_candidates) {
          for (auto &N_sub_block : MNK_sub_candidates) {
            // for (auto &K_sub_block : MNK_sub_candidates) {
            auto gcfg = reflection::general_object_t::make<
              managed_matmul_core_config_t>();
            managed_matmul_core_config_t &cfg
              = *gcfg.unchecked_get_as<managed_matmul_core_config_t>();
            cfg.M_split_num = M_split_num;
            cfg.N_split_num = N_split_num;
            cfg.M_sub_block = M_sub_block;
            cfg.N_sub_block = N_sub_block;
            // Currently we only support K_sub_block == 1, otherwise have
            // correctness issue.
            cfg.K_sub_block = 1;
            cfg.im_loop_order = 0;
            ret.emplace_back(std::move(gcfg));
            // }
          }
        }
      }
    }
  }
  return ret;
}

std::vector<uint64_t> gen_managed_matmul_core_t::convert_config_to_keys(
  const config_ptr &config) const {
  managed_matmul_core_config_t &cfg
    = *config.unchecked_get_as<managed_matmul_core_config_t>();
  std::vector<uint64_t> keys = {static_cast<uint64_t>(cfg.M_split_num),
    static_cast<uint64_t>(cfg.N_split_num),
    static_cast<uint64_t>(cfg.M_sub_block),
    static_cast<uint64_t>(cfg.N_sub_block),
    static_cast<uint64_t>(cfg.K_sub_block),
    static_cast<uint64_t>(cfg.im_loop_order)};
  return keys;
}

config_ptr gen_managed_matmul_core_t::get_default_config(
  context_ptr ctx) const {
  auto ret = reflection::general_object_t::make<managed_matmul_core_config_t>();
  managed_matmul_core_config_t &cfg
    = *ret.unchecked_get_as<managed_matmul_core_config_t>();
  if (is_dynamic()) {
    cfg.M_split_num = 1;
    cfg.N_split_num = 1;
    cfg.M_sub_block = 1;
    cfg.N_sub_block = 1;
    cfg.K_sub_block = 1;
    cfg.im_loop_order = 0;
    return std::move(ret);
  }
  const int num_threads = runtime_config_t::get().get_num_threads();
  const int iim_block = iim_block_;
  const int iin_block = iin_block_;
  const int iik_block = iik_block_;
  bool is_int8 = utils::is_one_of(get_A_dtype(), datatypes::u8, datatypes::s8);
  bool is_f32 = get_A_dtype() == datatypes::f32;
  bool no_vnni_low_fp = ops::no_vnni_low_fp(ctx, get_A_dtype());
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
  get_managed_matmul_config(ctx->machine_, cfg.M_split_num, cfg.N_split_num,
    cfg.M_sub_block, cfg.N_sub_block, cfg.K_sub_block, cfg.im_loop_order, M, N,
    K, iim_block, iin_block, iik_block, sizeofdtypeA, sizeofdtypeC, is_int8,
    is_f32 || no_vnni_low_fp, owner_->is_dynamic());
  return std::move(ret);
}

config_ptr gen_managed_matmul_core_t::get_default_post_rd_config(
  const context_ptr &ctx) const {
  // mmm + post reduce_on_N's config
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
  bool is_int8 = utils::is_one_of(get_A_dtype(), datatypes::u8, datatypes::s8);
  bool is_vnni_low_fp = ops::is_vnni_low_fp(ctx, get_A_dtype());
  const int sizeofdtypeA
    = utils::get_sizeof_etype(in_tensors_[0].dtype_.as_etype());
  const int sizeofdtypeC
    = utils::get_sizeof_etype(out_tensors_[0].dtype_.as_etype());
  bool is_special_fm = ctx->machine_.cpu_flags_.is_spr_like();

  // should discuss int8 and f32
  if ((M < 4096 && !is_vnni_low_fp) || M / iim_block_ < num_threads) {
    return get_default_config(ctx);
  }

  // enable to commit the anchor inside m_o
  cfg.M_split_num = num_threads;
  cfg.N_split_num = 1;
  cfg.N_sub_block = 1;
  cfg.im_loop_order = 0;

  int single_M
    = utils::divide_and_ceil(M / iim_block, cfg.M_split_num) * iim_block;
  int single_N
    = utils::divide_and_ceil(N / iin_block, cfg.N_split_num) * iin_block;
  int single_K = K;
  int L2_size = static_cast<int>(ctx->machine_.cpu_flags_.getDCacheSize(2));
  int single_K_threshold
    = (single_M * single_N * sizeofdtypeA < L2_size ? 2048 : 4096)
    / sizeofdtypeA;
  if (single_K >= single_K_threshold) {
    cfg.K_sub_block = (is_vnni_low_fp || is_int8 || K <= 1024)
      ? 1
      : utils::divide_and_ceil(single_K, single_K_threshold);
    cfg.K_sub_block = std::min(K / iik_block_, cfg.K_sub_block);
    // K is rounded up by iik_block_, so K / iik_block_ is always non-zero
    int L2_K = utils::divide_and_ceil(
                 utils::divide_and_ceil(single_K, iik_block), cfg.K_sub_block)
      * iik_block;
    // sizeofdtypeA* (M * K) + sizeofdtypeB * (N * K) + sizeofdtypeC * (M *
    // N)  <= L2_size, Then M = (L2_size - sizeofdtypeB * (N
    // * K)) / (sizeofdtypeA * K + sizeofdtypeC * N)
    int L2_MN = (L2_size - sizeofdtypeA * N * L2_K)
      / (sizeofdtypeA * L2_K + sizeofdtypeC * N);
    cfg.M_sub_block = L2_MN <= iim_block ? single_M / iim_block
                                         : std::max(1, single_M / L2_MN);
    while (cfg.M_sub_block > 1
      && (single_M / iim_block < cfg.M_sub_block
        || (M / iim_block % cfg.M_split_num > 0
          && M / iim_block / cfg.M_split_num < cfg.M_sub_block))) {
      cfg.M_sub_block--;
    }

  } else {
    // sizeofdtypeA * M * K + sizeofdtypeB * N * K <= L2_size, then
    // M = (L2_size - sizeofdtypeA * N * K) / (sizeofdtypeA * K)
    int L2_MN
      = (L2_size - sizeofdtypeA * N * single_K) / (sizeofdtypeA * single_K);
    cfg.M_sub_block = L2_MN <= iim_block ? single_M / iim_block
                                         : std::max(1, single_M / L2_MN);
    while (cfg.M_sub_block > 1
      && (single_M / iim_block < cfg.M_sub_block
        || (M / iim_block % cfg.M_split_num > 0
          && M / iim_block / cfg.M_split_num < cfg.M_sub_block))) {
      cfg.M_sub_block--;
    }
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
          = std::fabs(float(i * i) / float(num_threads) - float(N) / float(K));
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
          = std::fabs(float(i * i) / float(num_threads) - float(M) / float(K));
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
        new_cost = std::fabs(num_threads * N - i * i * M * split_k);
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
      COMPILE_ASSERT(L2_MN > 0, "Bad L2_MN. Is cache size correctly fetched?");
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

static int64_t get_bf16_M_block_default(
  const int64_t plain_M, const int num_threads) {
  int64_t M_block_default = 64;
  std::vector<int64_t> iim_block_candidates = {32, 64, 96};
  for (auto i : iim_block_candidates) {
    auto num_block = utils::divide_and_ceil(plain_M, i) / num_threads;
    // magic number 2
    if (plain_M % i == 0 && num_block > 2) { M_block_default = i; }
  }
  return M_block_default;
}

static int64_t get_bf16_N_block_default(const int64_t plain_N) {
  int64_t N_block_default = 64;
  std::vector<int64_t> iim_block_candidates = {64, 96};
  for (auto i : iim_block_candidates) {
    // magic number 2
    if (plain_N % i == 0 && plain_N / i > 2) { N_block_default = i; }
  }
  return N_block_default;
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
  bool is_vnni_low_fp
    = ops::is_vnni_low_fp(get_default_context(), get_A_dtype());
  bool no_vnni_low_fp
    = ops::no_vnni_low_fp(get_default_context(), get_A_dtype());
  bool is_f32 = get_A_dtype() == datatypes::f32;
  bool is_int8 = utils::is_one_of(get_A_dtype(), datatypes::u8, datatypes::s8);
  int64_t M_block_default = 64;
  int64_t N_block_default = 64;
  int64_t K_block_default = 64;
  // if true, run on spr, emr, gnr
  bool is_spr_like = get_default_context()->machine_.cpu_flags_.is_spr_like();
  // if true, run on skx, clx, cpx, icx
  bool is_skx_like = get_default_context()->machine_.cpu_flags_.is_skx_like();
  bool is_dynamic = is_dynamic_dim(plain_M) || is_dynamic_dim(plain_N)
    || is_dynamic_dim(plain_K);
  if (is_f32 || no_vnni_low_fp) {
    if (is_spr_like) {
      // prefer small blocks
      if (plain_M <= 4096) {
        M_block_default = 16;
        N_block_default = 16;
        K_block_default = 16;
      }
    } else {
      if (plain_M <= 256) { M_block_default = 32; }
    }
  } else if (is_vnni_low_fp) {
    if (plain_M > 16384 && plain_N >= 1024 && plain_K >= 768) {
      M_block_default = get_bf16_M_block_default(plain_M, num_threads);
      N_block_default = get_bf16_N_block_default(plain_N);
      K_block_default = 64;
    } else {
      M_block_default = 32;
      N_block_default = 32;
      K_block_default = 32;
    }
  } else {
    bool is_amx = get_default_context()->use_amx();
    assert(utils::is_one_of(get_A_dtype(), datatypes::u8, datatypes::s8));
    if (plain_M <= 1024 || (plain_M / num_threads / 64 < 8 && is_amx)) {
      M_block_default = 32;
    }
    // in amx, single core small M perfers using 128
    N_block_default
      = (num_threads == 1 && plain_M <= 12 && is_amx && plain_N >= 512) ? 128
                                                                        : 64;
    K_block_default
      = (num_threads == 1 && plain_M <= 12 && is_amx && plain_K >= 512) ? 128
                                                                        : 64;
  }
  if (!is_dynamic) {
    if (plain_N <= 512 && plain_K <= 512) {
      iim_block_ = std::max(
        ((is_f32 || no_vnni_low_fp) && is_skx_like && plain_M >= 64
          && plain_M <= 128 && (plain_N >= 256 || plain_K >= 256))
          ? (int64_t)8
          : (int64_t)4,
        std::min(M_block_default,
          static_cast<int64_t>(utils::divide_and_ceil(plain_M, num_threads))));
    } else {
      iim_block_ = suggest_aligned_block(plain_M, M_block_default);
    }
  } else if (!is_dynamic_dim(plain_M)) {
    iim_block_ = get_matmul_dyn_cfg_single(plain_M, true);
  }
  if (!is_dynamic) {
    iin_block_ = suggest_aligned_block(plain_N, N_block_default, 1, 16);
  } else if (!is_dynamic_dim(plain_N)) {
    iin_block_ = get_matmul_dyn_cfg_single(plain_N);
  }
  if (!is_dynamic) {
    if (is_f32 || no_vnni_low_fp) {
      // f32 small M with small even K prefers padding iik_block to align 16
      if (plain_K < 16 && plain_K % 2 == 0 && plain_M <= 128 && is_skx_like) {
        iik_block_ = 16;
      } else {
        iik_block_ = suggest_aligned_block(plain_K, K_block_default, 1, 16);
      }
    } else if (is_int8) {
      // vnni-int8 perfers padding iik_block_ to algin 16 when M <=2048
      if (plain_M < 2048 && !get_default_context()->use_amx()) {
        iik_block_ = 16;
      } else {
        iik_block_ = suggest_aligned_block(plain_K, K_block_default, 4, 16);
      }
    } else {
      iik_block_ = suggest_aligned_block(plain_K, K_block_default, 4, 16);
    }
  } else if (!is_dynamic_dim(plain_K)) {
    iik_block_ = get_matmul_dyn_cfg_single(plain_K);
  }
}

bool gen_managed_matmul_core_t::is_okay_to_prefetch(const context_ptr &ctx,
  const managed_matmul_core_config_t &config, bool is_global) {
  const int num_threads = runtime_config_t::get().get_num_threads();
  if (ctx->flags_.opt_level_ < sc_opt_level::lv3) { return false; }
  if (!in_tensors_[1].get_format().is_blocking()) { return false; }
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
  bool is_vnni_low_fp = ops::is_vnni_low_fp(ctx, get_A_dtype());
  int N = static_cast<int>(
        utils::rnd_up(in_tensors_[1].get_plain_dims()[1], iin_block_)),
      K = static_cast<int>(
        utils::rnd_up(in_tensors_[0].get_plain_dims()[1], iik_block_));
  auto num_threads = runtime_config_t::get().get_num_threads();
  // N_split_num
  auto threads_per_group = num_threads / config.M_split_num;
  int K_split_num = num_threads / config.M_split_num / config.N_split_num;
  if (config.N_split_num * config.M_split_num != num_threads
    && K_split_num <= 1) {
    _if_(tid % threads_per_group >= config.N_split_num) {
      _return_(UINT64_C(0));
    }
  }
  _var_(cnt, datatypes::index);
  cnt = 0;
  expr n_idx, k_idx, N_single_thr_size, K_single_thr_size, X_bigger_num;
  if (K_split_num == 1) {
    _for_(n_s, tid % threads_per_group, tid % threads_per_group + 1) {
      N_single_thr_size = get_balance211_length(
        N / iin_block_, config.N_split_num, n_s, n_idx, X_bigger_num);
      N_single_thr_size = N_single_thr_size * iin_block_;
      n_idx = n_idx * iin_block_;
      _for_(n_b, 0, config.N_sub_block) {
        expr n_b_idx, n_b_bigger_num, k_b_idx, k_b_bigger_num;
        _var_init_(n_o_end, datatypes::index,
          builder::make_cast(datatypes::index,
            get_balance211_length(N_single_thr_size / iin_block_,
              config.N_sub_block, n_b, n_b_idx, n_b_bigger_num)));
        _for_(k_b, 0, config.K_sub_block) {
          _for_(n_o, 0, n_o_end) {
            _var_init_(n_start_idx, datatypes::index,
              n_idx + n_b_idx * iin_block_
                + ((n_o + tid) % n_o_end) * iin_block_);
            _var_init_(bs, datatypes::index,
              builder::make_cast(datatypes::index,
                get_balance211_length(K / iik_block_, config.K_sub_block, k_b,
                  k_b_idx, k_b_bigger_num)));
            _var_init_(k_start_idx, datatypes::index, 0 + k_b_idx * iik_block_);
            _for_(i, 0, iik_block_ * iin_block_ * bs, 512 / sizeof_dtype) {
              _if_(lookup[0] == expected && !is_prefetch_debug_mode()) {
                _return_(cnt);
              }
              cnt = cnt + 1;
              _for_(j, 0,
                builder::make_min(
                  iik_block_ * iin_block_ * bs, 512 / sizeof_dtype),
                64 / sizeof_dtype) {
                std::vector<expr> B_indices;
                if (get_A_dtype() == datatypes::f32) {
                  B_indices = {n_start_idx / expr(iin_block_),
                    k_start_idx / expr(iik_block_), 0, i + j};
                } else {
                  B_indices = {n_start_idx / expr(iin_block_),
                    k_start_idx / expr(iik_block_), 0, 0, i + j};
                }
                auto tptr = builder::tensor_ptr(ins[0], B_indices);
                trace_prefetch_for_debug(tptr);
                builder::get_current_builder()->push_evaluate(
                  make_expr<intrin_call_node>(intrin_type::prefetch,
                    std::vector<expr> {tptr}, any_map_t {{"locality", 1}}));
              }
            }
          }
        }
      }
    }
  } else {
    _for_(n_s, tid % (config.N_split_num * K_split_num) / K_split_num,
      tid % (config.N_split_num * K_split_num) / K_split_num + 1) {
      N_single_thr_size = get_balance211_length(
        N / iin_block_, config.N_split_num, n_s, n_idx, X_bigger_num);
      N_single_thr_size = N_single_thr_size * iin_block_;
      n_idx = n_idx * iin_block_;
      _for_(k_s, tid % (config.N_split_num * K_split_num) % K_split_num,
        tid % (config.N_split_num * K_split_num) % K_split_num + 1) {
        K_single_thr_size = get_balance211_length(
          K / iik_block_, K_split_num, k_s, k_idx, X_bigger_num);
        K_single_thr_size = K_single_thr_size * iik_block_;
        k_idx = k_idx * iik_block_;
        _for_(n_b, 0, config.N_sub_block) {
          expr n_b_idx, n_b_bigger_num, k_b_idx, k_b_bigger_num;
          _var_init_(n_o_end, datatypes::index,
            builder::make_cast(datatypes::index,
              get_balance211_length(N_single_thr_size / iin_block_,
                config.N_sub_block, n_b, n_b_idx, n_b_bigger_num)));
          _for_(k_b, 0, config.K_sub_block) {
            _for_(n_o, 0, n_o_end) {
              _var_init_(n_start_idx, datatypes::index,
                n_idx + n_b_idx * iin_block_
                  + ((n_o + tid) % n_o_end) * iin_block_);
              _var_init_(bs, datatypes::index,
                builder::make_cast(datatypes::index,
                  get_balance211_length(K_single_thr_size / iik_block_,
                    config.K_sub_block, k_b, k_b_idx, k_b_bigger_num)));
              _var_init_(
                k_start_idx, datatypes::index, k_idx + k_b_idx * iik_block_);
              _for_(i, 0, iik_block_ * iin_block_ * bs, 512 / sizeof_dtype) {
                _if_(lookup[0] == expected && !is_prefetch_debug_mode()) {
                  _return_(cnt);
                }
                cnt = cnt + 1;

                _for_(j, 0,
                  builder::make_min(
                    iik_block_ * iin_block_ * bs, 512 / sizeof_dtype),
                  64 / sizeof_dtype) {
                  std::vector<expr> B_indices;
                  if (get_A_dtype() == datatypes::f32) {
                    B_indices = {n_start_idx / expr(iin_block_),
                      k_start_idx / expr(iik_block_), 0, i + j};
                  } else {
                    B_indices = {n_start_idx / expr(iin_block_),
                      k_start_idx / expr(iik_block_), 0, 0, i + j};
                  }
                  auto tptr = builder::tensor_ptr(ins[0], B_indices);
                  trace_prefetch_for_debug(tptr);
                  builder::get_current_builder()->push_evaluate(
                    make_expr<intrin_call_node>(intrin_type::prefetch,
                      std::vector<expr> {tptr}, any_map_t {{"locality", 1}}));
                }
              }
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
  std::vector<int> &K_anchor_info, bool is_partial, const expr &k_s,
  bool is_dynamic) const {
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
  expr tid = builder::make_get_group_thread_id(-1);
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
      _var_init_(m_o_end, datatypes::index,
        get_balance211_length(
          M / iim_block_, M_sub_block, m_b, m_b_idx, m_b_bigger_num));
      _var_init_(n_o_end, datatypes::index,
        get_balance211_length(
          N / iin_block_, N_sub_block, n_b, n_b_idx, n_b_bigger_num));
      _named_for_(im_k, k_b, 0, K_sub_block) {
        // general matmul_core loops
        _var_init_(bs, datatypes::index,
          builder::make_cast(datatypes::index,
            get_balance211_length(
              K / iik_block_, K_sub_block, k_b, k_b_idx, k_b_bigger_num)));
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
              auto eval = builtin::brgemm_init_update(tensor_ptr(A, aidx),
                tensor_ptr(B_vnni_tensor, bidx), tensor_ptr(C, cidx), bs,
                iim_block_, iin_block_, iik_block_, LDA, LDB, LDC, stride_a,
                stride_b, ta.dtype_, tb.dtype_);
            }
            _else_ {
              builtin::brgemm_update(tensor_ptr(A, aidx),
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

void gen_managed_matmul_core_t::dynamic_single_thread_matmul_call(
  const managed_matmul_core_config_t &config,
  const std::vector<expr> &buffer_args, const expr &m_s, const expr &n_s,
  const expr &k_s, int K_split_num, expr &iim_block, expr &iin_block,
  expr &iik_block) const {
  COMPILE_ASSERT(single_core_func_param_.defined(),
    "Single core function parameter should be defined first!");
  auto bld = builder::get_current_builder();
  std::vector<expr> extra_args {static_cast<uint64_t>(config.M_split_num),
    static_cast<uint64_t>(config.N_split_num),
    static_cast<uint64_t>(K_split_num),
    static_cast<uint64_t>(config.M_sub_block),
    static_cast<uint64_t>(config.N_sub_block),
    static_cast<uint64_t>(config.K_sub_block), m_s, n_s, k_s, iim_block,
    iin_block, iik_block};
  extra_args.insert(extra_args.begin(), buffer_args.begin(), buffer_args.end());
  auto single_core_call
    = make_expr<call_node>(single_core_func_param_, extra_args);
  bld->push_evaluate(single_core_call);
}

void gen_managed_matmul_core_t::single_thread_matmul_call(
  const context_ptr &ctx, sc_graph_t &graph, const logical_tensor_t &ta,
  const logical_tensor_t &tb, const logical_tensor_t &tc,
  const managed_matmul_core_config_t &config, const expr &M, const expr &N,
  const expr &K, const expr &m_idx, const expr &n_idx, const expr &k_idx,
  const expr &A, const expr &B, const expr &C, int dtype_block,
  fusion_manager *fusion, const expr &m_s, const expr &n_s,
  std::vector<int> &M_anchor_info, std::vector<int> &N_anchor_info,
  bool is_partial, const expr &k_s, bool is_dynamic,
  const expr &N_block_size_expr) const {
  expr M_sub_block = config.M_sub_block, N_sub_block = config.N_sub_block,
       K_sub_block = config.K_sub_block;
  for_loop im_k, im_m, im_n, o_im_n;
  int ori_M = static_cast<int>(ta.get_plain_dims()[0]),
      ori_K = static_cast<int>(ta.get_plain_dims()[1]),
      ori_N = static_cast<int>(tb.get_plain_dims()[1]);
  expr tid = builder::make_get_group_thread_id(-1);
  _for_(m_b, 0, M_sub_block) {
    _named_for_(o_im_n, n_b, 0, N_sub_block) {
      expr m_b_idx, n_b_idx, k_b_idx, m_b_bigger_num, n_b_bigger_num,
        k_b_bigger_num;
      _var_init_(m_o_end, datatypes::index,
        get_balance211_length(
          M / iim_block_, M_sub_block, m_b, m_b_idx, m_b_bigger_num));
      _var_init_(n_o_end, datatypes::index,
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
            _var_init_(bs, datatypes::index,
              builder::make_cast(datatypes::index,
                get_balance211_length(
                  K / iik_block_, K_sub_block, k_b, k_b_idx, k_b_bigger_num)));
            _var_init_(
              k_start_idx, datatypes::index, k_idx + k_b_idx * iik_block_);
            // create input anchor for B if necessary
            if (fusion && in_tensors_[1].get_format().is_blocking()
              && K.isa<constant>()
              && ((get_expr_as_int(K) / iik_block_ % config.K_sub_block) == 0)
              && !is_dynamic_dim(ori_M) && ori_M <= 512) {
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
            expr LDA = ta.get_format() == sc_data_format_t::MK()
              ? graph.dim_to_expr(ori_K)
              : expr(iik_block_);
            expr LDB = !tb.get_format().is_blocking() ? graph.dim_to_expr(ori_N)
                                                      : expr(iin_block_);
            expr LDC = !tc.get_format().is_blocking()
              ? (is_partial ? N_block_size_expr : graph.dim_to_expr(ori_N))
              : iin_block_;
            expr stride_a = ta.get_format() == sc_data_format_t::MK()
              ? iik_block_
              : iim_block_ * iik_block_;
            expr stride_b = !tb.get_format().is_blocking()
              ? iik_block_ * graph.dim_to_expr(ori_N)
              : iik_block_ * iin_block_;
            trace_brgemm_for_debug(
              tensor_ptr(B, bidx), bs, iin_block_, iik_block_);

            _if_(k_b == 0) {
              auto eval = builtin::brgemm_init_update(tensor_ptr(A, aidx),
                tensor_ptr(B, bidx), tensor_ptr(C, cidx), bs, iim_block_,
                iin_block_, iik_block_, LDA, LDB, LDC, stride_a, stride_b,
                ta.dtype_, tb.dtype_);
            }
            _else_ {
              builtin::brgemm_update(tensor_ptr(A, aidx), tensor_ptr(B, bidx),
                tensor_ptr(C, cidx), bs, iim_block_, iin_block_, iik_block_,
                LDA, LDB, LDC, stride_a, stride_b, ta.dtype_, tb.dtype_);
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
          if (fusion && !is_partial && config.N_split_num == 1
            && config.N_sub_block == 1 && config.im_loop_order == 0) {
            _if_(k_b == K_sub_block - 1) {
              fusion->create_output_fusion_anchor({tensor_slice(C,
                !tc.get_format().is_blocking()
                  ? std::vector<std::pair<expr,
                    expr>> {{m_idx + m_b_idx * iim_block_
                                + ((m_o + tid) % m_o_end) * iim_block_,
                              expr(iim_block_)},
                    {0, utils::rnd_up(ori_N, iin_block_)}}
                  : std::vector<std::pair<expr, expr>> {
                    {(m_idx + m_b_idx * iim_block_
                       + ((m_o + tid) % m_o_end) * iim_block_)
                        / iim_block_,
                      1},
                    {0, utils::divide_and_ceil(ori_N, iin_block_)},
                    {0, expr(iim_block_)}, {0, expr(iin_block_)}})});
            }
          }
        }
      }
      if (fusion && !is_dynamic && !is_partial) {
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
  if (!is_dynamic && config.K_sub_block > 1 && config.im_loop_order != 1) {
    im_n->attr()[stmt_attr_key::reduce_root_loop]
      = std::weak_ptr<stmt_base_t>(o_im_n.impl);
  }
  if (config.im_loop_order == 1) {
    im_m->reorder(im_k->body_, {im_n, im_m});
    im_m->attr()[stmt_attr_key::reduce_root_loop]
      = std::weak_ptr<stmt_base_t>(o_im_n.impl);
  }
}

std::vector<expr> gen_managed_matmul_core_t::get_extra_args_from_func(
  const func_t &f) const {
  return std::vector<expr>(f->params_.begin() + 3, f->params_.end());
}

func_t gen_managed_matmul_core_t::get_single_core_func(context_ptr ctx,
  const managed_matmul_core_config_t &config, fusion_manager *fusion,
  const std::vector<expr> &inputs, const std::vector<expr> &outputs,
  std::vector<for_loop> &loops) const {
  auto &graph = owner_->get_owner_graph();
  auto &ta = in_tensors_[0];
  auto &tb = in_tensors_[1];
  auto &tc = out_tensors_[0];
  int ori_M = static_cast<int>(ta.get_plain_dims()[0]),
      ori_K = static_cast<int>(ta.get_plain_dims()[1]),
      ori_N = static_cast<int>(tb.get_plain_dims()[1]);
  COMPILE_ASSERT(
    !is_dynamic_dim(ori_K), "Currently we don't support dynamic on K");
  int dtype_block = 1;
  auto B_dtype = get_B_dtype();
  bool is_B_vnni_low_fp = ops::is_vnni_low_fp(ctx, B_dtype);
  if (is_B_vnni_low_fp) {
    dtype_block = 2;
  } else if (utils::is_one_of(B_dtype, datatypes::u8, datatypes::s8)) {
    dtype_block = 4;
  }
  ir_builder_t bld;
  static std::atomic<int> func_idx = {0};
  expr partial_C;
  sc_brgemm_attrs_t brg_attrs;
  _function_(datatypes::boolean, managed_matmul_single_core, {outputs[0]},
    {inputs[0]}, {inputs[1]}, _arg_("M_split_num", datatypes::index),
    _arg_("N_split_num", datatypes::index),
    _arg_("K_split_num", datatypes::index),
    _arg_("M_sub_block", datatypes::index),
    _arg_("N_sub_block", datatypes::index),
    _arg_("K_sub_block", datatypes::index), _arg_("m_s", datatypes::index),
    _arg_("n_s", datatypes::index), _arg_("k_s", datatypes::index),
    _arg_("iim_block", datatypes::s32, {1}),
    _arg_("iin_block", datatypes::s32, {1}),
    _arg_("iik_block", datatypes::s32, {1})) {
    _bind_(C, A, B, M_split_num, N_split_num, K_split_num, M_sub_block,
      N_sub_block, K_sub_block, m_s, n_s, k_s, iim_block, iin_block, iik_block);
    expr m_idx_, n_idx_, k_idx_, K_single_thr_size_, X_bigger_num;
    auto ori_M_expr = graph.dim_to_expr(ori_M);
    auto ori_N_expr = graph.dim_to_expr(ori_N);
    auto ori_K_expr = graph.dim_to_expr(ori_K);
    auto M = divide_and_ceil(ori_M_expr, iim_block_) * iim_block_;
    auto N = divide_and_ceil(ori_N_expr, iin_block_) * iin_block_;
    auto K = divide_and_ceil(ori_K_expr, iik_block_) * iik_block_;
    // Postpone calculation time of M_single_thr_size/N_single_thr_size and
    // m_idx/n_idx.
    iim_block[0] = iim_block_;
    iin_block[0] = iin_block_;
    iik_block[0] = iik_block_;
    expr tid = builtin::get_thread_id_func()();
    k_idx_ = 0;
    K_single_thr_size_ = get_balance211_length(
      K / iik_block_, K_split_num, k_s, k_idx_, X_bigger_num);
    K_single_thr_size_ = K_single_thr_size_ * iik_block_;
    k_idx_ = k_idx_ * iik_block_;
    expr M_block_size_expr
      = divide_and_ceil(M / iim_block_, M_split_num) * iim_block_;
    expr N_block_size_expr
      = divide_and_ceil(N / iin_block_, N_split_num) * iin_block_;
    for_loop o_im_n, im_k, im_m, im_n;
    auto K_real_split = builder::make_min(divide_and_ceil(K, iik_block_),
      runtime_config_t::get().get_num_threads() / M_split_num / N_split_num);
    auto N_real_split
      = builder::make_min(divide_and_ceil(N, iin_block_), N_split_num);
    bool input_plain = ta.get_format() == sc_data_format_t::MK();
    _if_(m_s < divide_and_ceil(M, iim_block_)
      && n_s < divide_and_ceil(N, iin_block_)
      && k_s < divide_and_ceil(K, iik_block_)) {
      _var_init_(M_single_thr_size, datatypes::index,
        get_balance211_length(
          M / iim_block_, M_split_num, m_s, m_idx_, X_bigger_num)
          * iim_block_);
      _var_init_(m_idx, datatypes::index, m_idx_ * iim_block_);

      _var_init_(N_single_thr_size, datatypes::index,
        get_balance211_length(
          N / iin_block_, N_split_num, n_s, n_idx_, X_bigger_num)
          * iin_block_);
      _var_init_(n_idx, datatypes::index, n_idx_ * iin_block_);
      _var_init_(K_single_thr_size, datatypes::index,
        !is_partial_
          ? do_cast_and_fold(divide_and_ceil(K, iik_block_) * iik_block_)
          : K_single_thr_size_);
      _for_(m_b, 0, M_sub_block) {
        _named_for_(o_im_n, n_b, 0, N_sub_block) {
          expr m_b_idx, n_b_idx, k_b_idx, m_b_bigger_num, n_b_bigger_num,
            k_b_bigger_num;
          _var_init_(m_o_end, datatypes::index,
            get_balance211_length(M_single_thr_size / iim_block_, M_sub_block,
              m_b, m_b_idx, m_b_bigger_num));
          _var_init_(n_o_end, datatypes::index,
            get_balance211_length(N_single_thr_size / iin_block_, N_sub_block,
              n_b, n_b_idx, n_b_bigger_num));
          auto partial_c_shape
            = std::vector<expr> {K_real_split, M_block_size_expr / iim_block_,
              N_block_size_expr / iin_block_, iim_block_, iin_block_};
          partial_C = copy_attr(*(C.get()),
            builder::make_tensor(
              "partial_out", partial_c_shape, out_tensors_[0].dtype_));
          partial_C->attr().set(attr_keys::plain_dims,
            std::vector<expr> {ori_M_expr, ori_N_expr, ori_K_expr});
          auto C_tptr = is_partial_ ? partial_C : C;
          _named_for_(im_k, k_b, 0, K_sub_block) {
            expr K_tail_cond = (k_b == K_sub_block - 1
              && k_s == K_real_split - 1 && K != ori_K_expr);
            _var_init_(bs, datatypes::index,
              builder::make_cast(datatypes::index,
                get_balance211_length(K_single_thr_size / iik_block_,
                  K_sub_block, k_b, k_b_idx, k_b_bigger_num)));
            // bs floor, tail is 1.
            expr K_tail;
            _var_init_(
              k_start_idx, datatypes::index, k_idx_ + k_b_idx * iik_block_);
            if (input_plain) {
              bs = builder::make_select(
                K_tail_cond, do_cast_and_fold(bs - 1), bs);
              K_tail = is_dynamic_dim(ori_K)
                ? ori_K_expr - k_start_idx - bs * iik_block_
                : ori_K % iik_block_;
              brg_attrs[brgemm::attr_key::K_range_upper_bound] = iik_block_;
              if (!is_dynamic_dim(ori_K)) {
                brg_attrs[brgemm::attr_key::K_range_tail_value]
                  = ori_K % iik_block_;
              }
            }
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
                std::vector<expr> aidx = input_plain
                  ? std::vector<expr> {m_start_idx, k_start_idx}
                  : std::vector<expr> {
                    m_start_idx / iim_block_, k_start_idx / iik_block_, 0, 0};
                std::vector<expr> bidx = dtype_block > 1
                  ? std::vector<expr> {n_start_idx / iin_block_,
                    k_start_idx / iik_block_, 0, 0, 0}
                  : (!tb.get_format().is_blocking()
                      ? std::vector<expr> {k_start_idx, n_start_idx}
                      : std::vector<expr> {n_start_idx / iin_block_,
                        k_start_idx / iik_block_, 0, 0});
                std::vector<expr> partial_cidx, full_cidx;
                partial_cidx = !tc.get_format().is_blocking()
                  ? std::vector<expr> {m_b_idx * iim_block_
                      + ((m_o + tid) % m_o_end) * iim_block_,
                    n_b_idx * iin_block_ + ((n_o + tid) % n_o_end) * iin_block_}
                  : std::vector<expr> {m_b_idx + (m_o + tid) % m_o_end,
                    n_b_idx + (n_o + tid) % n_o_end, 0, 0};
                partial_cidx.insert(partial_cidx.begin(), k_s);
                full_cidx = !tc.get_format().is_blocking()
                  ? std::vector<expr> {m_start_idx, n_start_idx}
                  : std::vector<expr> {
                    m_start_idx / iim_block_, n_start_idx / iin_block_, 0, 0};
                auto partial_C_ptr = tensor_ptr(C_tptr, partial_cidx);
                auto full_C_ptr = tensor_ptr(C_tptr, full_cidx);
                expr LDA = input_plain ? ori_K_expr : expr(iik_block_);
                expr LDB = !tb.get_format().is_blocking() ? ori_N_expr
                                                          : expr(iin_block_);
                expr partial_LDC = !tc.get_format().is_blocking()
                  ? do_cast_and_fold(
                    divide_and_ceil(N / iin_block_, N_split_num) * iin_block_)
                  : iin_block_;
                expr full_LDC
                  = !tc.get_format().is_blocking() ? ori_N_expr : iin_block_;
                expr stride_a
                  = input_plain ? iik_block_ : iim_block_ * iik_block_;
                expr stride_b = !tb.get_format().is_blocking()
                  ? iik_block_ * ori_N_expr
                  : iik_block_ * iin_block_;
                expr m_block = iim_block_, n_block = iin_block_,
                     k_block = iik_block_;
                if (input_plain) {
                  if (is_dynamic_dim(ori_M) || ori_M % iim_block_) {
                    m_block = builder::make_select(
                      m_start_idx + iim_block_ < ori_M_expr, iim_block_,
                      builder::make_cast(
                        datatypes::s32, ori_M_expr - m_start_idx));
                    brg_attrs[brgemm::attr_key::M_range_upper_bound]
                      = iim_block_;
                    if (!is_dynamic_dim(ori_M)) {
                      brg_attrs[brgemm::attr_key::M_range_tail_value]
                        = ori_M % iim_block_;
                    }
                  }
                  if (is_dynamic_dim(ori_N) || ori_N % iin_block_) {
                    n_block = builder::make_select(
                      n_start_idx + iin_block_ < ori_N_expr, iin_block_,
                      builder::make_cast(
                        datatypes::s32, ori_N_expr - n_start_idx));
                    brg_attrs[brgemm::attr_key::N_range_upper_bound]
                      = iin_block_;
                    if (!is_dynamic_dim(ori_N)) {
                      brg_attrs[brgemm::attr_key::N_range_tail_value]
                        = ori_N % iin_block_;
                    }
                    partial_LDC->attr().set("skip_shrink_check", true);
                  }
                }
                auto call_init_update_brgemm
                  = [&](const expr &real_bs, const expr &k_block,
                      const std::vector<expr> &real_aidx,
                      const std::vector<expr> &real_bidx) {
                      if (is_partial_) {
                        builtin::brgemm_init_update(tensor_ptr(A, real_aidx),
                          tensor_ptr(B, real_bidx), partial_C_ptr, real_bs,
                          m_block, n_block, k_block, LDA, LDB, partial_LDC,
                          stride_a, stride_b, ta.dtype_, tb.dtype_, brg_attrs);
                      } else {
                        builtin::brgemm_init_update(tensor_ptr(A, real_aidx),
                          tensor_ptr(B, real_bidx), full_C_ptr, real_bs,
                          m_block, n_block, k_block, LDA, LDB, full_LDC,
                          stride_a, stride_b, ta.dtype_, tb.dtype_, brg_attrs);
                      }
                    };
                auto call_update_brgemm
                  = [&](const expr &real_bs, const expr &k_block,
                      const std::vector<expr> &real_aidx,
                      const std::vector<expr> &real_bidx) {
                      if (is_partial_) {
                        builtin::brgemm_update(tensor_ptr(A, real_aidx),
                          tensor_ptr(B, real_bidx), partial_C_ptr, real_bs,
                          m_block, n_block, k_block, LDA, LDB, partial_LDC,
                          stride_a, stride_b, ta.dtype_, tb.dtype_, brg_attrs);
                      } else {
                        builtin::brgemm_update(tensor_ptr(A, real_aidx),
                          tensor_ptr(B, real_bidx), full_C_ptr, real_bs,
                          m_block, n_block, k_block, LDA, LDB, full_LDC,
                          stride_a, stride_b, ta.dtype_, tb.dtype_, brg_attrs);
                      }
                    };
                _if_(bs > 0) {
                  _if_(k_b == 0) {
                    call_init_update_brgemm(bs, iik_block_, aidx, bidx);
                  }
                  _else_ { call_update_brgemm(bs, iik_block_, aidx, bidx); }
                }
                if (input_plain) {
                  auto k_tail_idx = k_start_idx + bs * iik_block_;
                  std::vector<expr> tail_aidx = {m_start_idx, k_tail_idx};
                  std::vector<expr> tail_bidx = dtype_block > 1
                    ? std::vector<expr> {n_start_idx / iin_block_,
                      k_tail_idx / iik_block_, 0, 0, 0}
                    : (!tb.get_format().is_blocking()
                        ? std::vector<expr> {k_tail_idx, n_start_idx}
                        : std::vector<expr> {n_start_idx / iin_block_,
                          k_tail_idx / iik_block_, 0, 0});
                  _if_(K_tail_cond) {
                    _if_(k_b == 0 && bs == 0) {
                      call_init_update_brgemm(1, K_tail, tail_aidx, tail_bidx);
                    }
                    _else_ {
                      call_update_brgemm(1, K_tail, tail_aidx, tail_bidx);
                    }
                  }
                }

                if (fusion && !is_partial_) {
                  _if_(k_b == K_sub_block - 1) {
                    fusion->create_output_fusion_anchor({tensor_slice(C,
                      !tc.get_format().is_blocking()
                        ? std::vector<std::pair<expr, expr>> {{m_start_idx,
                                                                expr(m_block)},
                          {n_start_idx, expr(n_block)}}
                        : std::vector<std::pair<expr, expr>> {
                          {m_start_idx / iim_block_, 1},
                          {n_start_idx / iin_block_, 1}, {0, expr(iim_block_)},
                          {0, expr(iin_block_)}})});
                  }
                }
              }
            }
          }
        }
      }
    }
    // todo: K_sub_block partial or not may need combinated with format
    // dispatch(outer dispatch key) to dispatch.(inner impl in format dispatch
    // key not impl internal dispatch key.); Find the fusion strategy with
    // K_sub_block > 1.
    _return_(true);
  }
  if (is_partial_) {
    managed_matmul_single_core->params_[0] = partial_C;
    managed_matmul_single_core->decl_->params_[0] = partial_C;
  }
  managed_matmul_single_core->name_ += "_" + std::to_string(func_idx++);
  managed_matmul_single_core->decl_->name_ = managed_matmul_single_core->name_;
  return managed_matmul_single_core;
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
  sc_graph_t &graph = owner_->get_owner_graph();
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
  expr M_expr
    = divide_and_ceil(
        graph.dim_to_expr(in_tensors_[0].get_plain_dims()[0]), iim_block_)
    * iim_block_;
  expr N_expr
    = divide_and_ceil(
        graph.dim_to_expr(in_tensors_[1].get_plain_dims()[1]), iin_block_)
    * iin_block_;
  expr K_expr
    = divide_and_ceil(
        graph.dim_to_expr(in_tensors_[0].get_plain_dims()[1]), iik_block_)
    * iik_block_;
  int M_block_size
    = utils::divide_and_ceil(M / iim_block_, M_split_num) * iim_block_;
  expr M_block_size_expr
    = divide_and_ceil(M_expr / iim_block_, M_split_num) * iim_block_;
  int M_ib_block_size = M / iim_block_ / M_split_num * iim_block_;
  int N_block_size
    = utils::divide_and_ceil(N / iin_block_, N_split_num) * iin_block_;
  expr N_block_size_expr
    = divide_and_ceil(N_expr / iin_block_, N_split_num) * iin_block_;
  int N_ib_block_size = N / iin_block_ / N_split_num * iin_block_;
  int K_block_size
    = utils::divide_and_ceil(K / iik_block_, K_split_num) * iik_block_;
  expr K_block_size_expr
    = divide_and_ceil(K_expr / iik_block_, K_split_num) * iik_block_;
  int K_ib_block_size = K / iik_block_ / K_split_num * iik_block_;

  if (M_ib_block_size == 0) { M_ib_block_size = M_block_size; }
  if (N_ib_block_size == 0) { N_ib_block_size = N_block_size; }
  if (K_ib_block_size == 0) { K_ib_block_size = K_block_size; }

  // M, N block num with block size equals to X_block_size
  int M_blk_num = (M - (M_block_size - iim_block_) * M_split_num) / iim_block_;
  int N_blk_num = (N - (N_block_size - iin_block_) * N_split_num) / iin_block_;
  int K_blk_num = (K - (K_block_size - iik_block_) * K_split_num) / iik_block_;

  bool is_dynamic = owner_->is_dynamic();
  if (!is_dynamic) {
    COMPILE_ASSERT(M_block_size / iim_block_ >= M_sub_block
        && M_ib_block_size / iim_block_ >= M_sub_block && M_sub_block >= 1,
      "bad M_sub_block given");
    COMPILE_ASSERT(N_block_size / iin_block_ >= N_sub_block
        && N_ib_block_size / iin_block_ >= N_sub_block && N_sub_block >= 1,
      "bad N_sub_block given");
    COMPILE_ASSERT(K_block_size / iik_block_ >= K_sub_block
        && K_ib_block_size / iik_block_ >= K_sub_block && K_sub_block >= 1,
      "bad K_sub_block given");
  }
  int dtype_block = 1;
  auto A_dtype = get_A_dtype();
  auto B_dtype = get_B_dtype();
  const int sizeofdtypeA
    = utils::get_sizeof_etype(in_tensors_[0].dtype_.as_etype());
  bool is_A_vnni_low_fp = ops::is_vnni_low_fp(ctx, A_dtype);
  bool is_B_vnni_low_fp = ops::is_vnni_low_fp(ctx, B_dtype);
  if (is_B_vnni_low_fp) {
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
  expr M_real_split = is_dynamic
    ? M_split_num
    : do_cast_and_fold(
      builder::make_min(divide_and_ceil(M_expr, iim_block_), M_split_num));
  expr N_real_split = is_dynamic
    ? N_split_num
    : do_cast_and_fold(
      builder::make_min(divide_and_ceil(N_expr, iin_block_), N_split_num));
  expr K_real_split = is_dynamic
    ? K_split_num
    : do_cast_and_fold(
      builder::make_min(divide_and_ceil(K_expr, iik_block_), K_split_num));

  if (K_split_num == 1) {
    expr m_idx, n_idx, M_single_thr_size, N_single_thr_size, X_bigger_num;
    expr iim_block_tsr, iin_block_tsr, iik_block_tsr;
    _named_for_(
      mloop, m_s, 0, M_real_split, 1, for_type::PARALLEL, M_split_num) {
      _for_(n_s, 0, N_real_split, 1, for_type::PARALLEL, N_split_num) {
        M_single_thr_size = get_balance211_length(
          M_expr / iim_block_, M_split_num, m_s, m_idx, X_bigger_num);
        M_single_thr_size = M_single_thr_size * iim_block_;
        m_idx = m_idx * iim_block_;

        N_single_thr_size = get_balance211_length(
          N_expr / iin_block_, N_split_num, n_s, n_idx, X_bigger_num);
        N_single_thr_size = N_single_thr_size * iin_block_;
        n_idx = n_idx * iin_block_;
        _for_(k_s, 0, K_split_num, 1,
          M_split_num * N_split_num == num_threads ? for_type::NORMAL
                                                   : for_type::PARALLEL,
          M_split_num * N_split_num == num_threads ? 0 : K_split_num) {
          // create input anchor for A if necessary
          if (!is_dynamic) {
            if (fusion && in_tensors_[0].get_format().is_blocking()
              && (M * K * sizeofdtypeA <= 1024 * 1024
                || K * sizeofdtypeA <= 1024)) {
              fusion->create_input_fusion_anchor({tensor_slice(A,
                {{m_idx / iim_block_,
                   utils::divide_and_ceil(M_block_size, iim_block_)},
                  {0, K / iik_block_}, {0, iim_block_}, {0, iik_block_}})});
            }
            if (in_tensors_[0].get_format() == sc_data_format_t::NK()
              && is_A_vnni_low_fp) {
              trace_guard_t tg(ctx, "transpose_A");
              expr A_trans_tensor;
              _tensor_(A_trans, get_A_dtype(),
                {M / iim_block_, K / iik_block_, iim_block_, iik_block_});
              A_trans_tensor = A_trans;
              A_trans_tensor->attr()[tensor_shrinker_attrs::should_shrink]
                = tensor_shrinker_t::shrink_info_t {
                  {m_idx / iim_block_, 0, 0, 0},
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
                    sc_data_format_t::NK(), A_dtype)},
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
          } else {
            _tensor_(iim_block_tmp, datatypes::s32, {1});
            _tensor_(iin_block_tmp, datatypes::s32, {1});
            _tensor_(iik_block_tmp, datatypes::s32, {1});
            iim_block_tsr = iim_block_tmp;
            iin_block_tsr = iin_block_tmp;
            iik_block_tsr = iik_block_tmp;
          }
          if (!is_dynamic
            && utils::is_one_of(in_tensors_[1].get_format(),
              sc_data_format_t::MK(), sc_data_format_t::NK())
            && is_B_vnni_low_fp) {
            single_thread_reorder_matmul_call(ctx, in_tensors_[0],
              in_tensors_[1], out_tensors_[0], config, M_single_thr_size,
              N_single_thr_size, (int)utils::rnd_up(K, iik_block_), m_idx,
              n_idx, k_s, A, B, C, dtype_block, fusion, m_s, n_s, M_anchor_info,
              N_anchor_info, K_anchor_info, false, k_s, is_dynamic);
          } else if (!is_dynamic) {
            single_thread_matmul_call(ctx, graph, in_tensors_[0],
              in_tensors_[1], out_tensors_[0], config, M_single_thr_size,
              N_single_thr_size, (int)utils::rnd_up(K, iik_block_), m_idx,
              n_idx, k_s, A, B, C, dtype_block, fusion, m_s, n_s, M_anchor_info,
              N_anchor_info, false, expr(), is_dynamic, N_block_size_expr);
          } else {
            assert(owner_->info_.internal_info_);
            auto buffer_args = outputs;
            buffer_args.insert(buffer_args.end(), inputs.begin(), inputs.end());
            dynamic_single_thread_matmul_call(config, buffer_args, m_s, n_s,
              k_s, K_split_num, iim_block_tsr, iin_block_tsr, iik_block_tsr);
          }
        }
        if (fusion && !is_dynamic) {
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
      if (fusion) {
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
    std::vector<expr> out_tmp_buf_shape_expr
      = out_tensors_[0].get_format().is_blocking()
      ? std::vector<expr> {K_real_split, M_block_size_expr / iim_block_,
        N_block_size_expr / iin_block_, iim_block_, iin_block_}
      : std::vector<expr> {K_real_split, M_block_size_expr, N_block_size_expr};
    if (is_dynamic) {
      out_tmp_buf_shape_expr = std::vector<expr> {K_real_split,
        divide_and_ceil(M_expr, M_split_num) * 2,
        divide_and_ceil(N_expr, N_split_num) * 2};
    };
    auto out_dtype = utils::is_one_of(A_dtype, datatypes::u8, datatypes::s8)
      ? datatypes::s32
      : datatypes::f32;
    expr m_idx, n_idx, k_idx, M_single_thr_size, N_single_thr_size,
      X_bigger_num;
    // for single core func query
    expr iim_block_tsr, iin_block_tsr, iik_block_tsr;
    expr iim_block = iim_block_, iin_block = iin_block_, iik_block = iik_block_;
    _named_for_(
      mloop, m_s, 0, M_real_split, 1, for_type::PARALLEL, M_split_num) {
      _for_(n_s, 0, N_real_split, 1, for_type::PARALLEL, N_split_num) {
        _tensor_(out_tmp_buf, out_dtype, out_tmp_buf_shape_expr);
        // fake plain dims to pass down.
        out_tmp_buf->attr().set(attr_keys::plain_dims,
          std::vector<expr> {
            graph.dim_to_expr(in_tensors_[0].get_plain_dims()[0]),
            graph.dim_to_expr(in_tensors_[1].get_plain_dims()[1]),
            graph.dim_to_expr(in_tensors_[0].get_plain_dims()[1])});
        M_single_thr_size = get_balance211_length(
          M_expr / iim_block_, M_split_num, m_s, m_idx, X_bigger_num);
        M_single_thr_size = M_single_thr_size * iim_block_;
        m_idx = m_idx * iim_block_;

        N_single_thr_size = get_balance211_length(
          N_expr / iin_block_, N_split_num, n_s, n_idx, X_bigger_num);
        N_single_thr_size = N_single_thr_size * iin_block_;
        n_idx = n_idx * iin_block_;
        if (is_dynamic) {
          _tensor_(iim_block_tmp, datatypes::s32, {1});
          _tensor_(iin_block_tmp, datatypes::s32, {1});
          _tensor_(iik_block_tmp, datatypes::s32, {1});
          iim_block_tsr = iim_block_tmp;
          iin_block_tsr = iin_block_tmp;
          iik_block_tsr = iik_block_tmp;
          iim_block = iim_block_tmp[0];
          iin_block = iin_block_tmp[0];
          iik_block = iik_block_tmp[0];
          iim_block->attr().set(attr_keys::no_index2var, true);
          iin_block->attr().set(attr_keys::no_index2var, true);
          iik_block->attr().set(attr_keys::no_index2var, true);
        }
        _for_(k_s, 0, K_real_split, 1, for_type::PARALLEL, K_split_num) {
          expr K_single_thr_size;
          if (!is_dynamic && K_block_size == K_ib_block_size) {
            K_single_thr_size = K_block_size;
            k_idx = k_s * K_block_size;
          } else {
            K_single_thr_size = get_balance211_length(
              K_expr / iik_block_, K_split_num, k_s, k_idx, X_bigger_num);
            K_single_thr_size = K_single_thr_size * iik_block_;
            k_idx = k_idx * iik_block_;
          }
          // create input anchor for A if necessary
          if (!is_dynamic) {
            if (fusion && in_tensors_[0].get_format().is_blocking()
              && (K_block_size == K_ib_block_size) && M * K <= 1024 * 1024) {
              fusion->create_input_fusion_anchor({tensor_slice(A,
                {{m_idx / iim_block_,
                   utils::divide_and_ceil(M_block_size, iim_block_)},
                  {k_idx / iik_block_, K_block_size / iik_block_},
                  {0, iim_block_}, {0, iik_block_}})});
            }
            if (in_tensors_[0].get_format() == sc_data_format_t::NK()
              && is_A_vnni_low_fp) {
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
                    sc_data_format_t::NK(), A_dtype)},
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
          }
          if (!is_dynamic
            && utils::is_one_of(in_tensors_[1].get_format(),
              sc_data_format_t::MK(), sc_data_format_t::NK())
            && is_B_vnni_low_fp) {
            single_thread_reorder_matmul_call(ctx, in_tensors_[0],
              in_tensors_[1], out_tensors_[0], config, M_single_thr_size,
              N_single_thr_size, K_single_thr_size, m_idx, n_idx, k_idx, A, B,
              out_tmp_buf, dtype_block, fusion, m_s, n_s, M_anchor_info,
              N_anchor_info, K_anchor_info, true, k_s);
          } else if (!is_dynamic) {
            single_thread_matmul_call(ctx, graph, in_tensors_[0],
              in_tensors_[1], out_tensors_[0], config, M_single_thr_size,
              N_single_thr_size, K_single_thr_size, m_idx, n_idx, k_idx, A, B,
              out_tmp_buf, dtype_block, fusion, m_s, n_s, M_anchor_info,
              N_anchor_info, true, k_s, is_dynamic, N_block_size_expr);
          } else {
            assert(owner_->info_.internal_info_);
            auto buffer_args = outputs;
            buffer_args.insert(buffer_args.end(), inputs.begin(), inputs.end());
            buffer_args[0] = out_tmp_buf;
            buffer_args[0]->attr().set(attr_keys::always_trans, true);
            dynamic_single_thread_matmul_call(config, buffer_args, m_s, n_s,
              k_s, K_split_num, iim_block_tsr, iin_block_tsr, iik_block_tsr);
          }
        }
        // do reduce here
        for_loop rm, rn;
        int lanes = 1;
        if (is_dynamic || (iin_block_ / 16 && iin_block_ % 16 == 0)) {
          lanes = vectorize_step(ctx, get_C_dtype().type_code_, 16);
        }
        if (is_dynamic) {
          M_expr = divide_and_ceil(
                     graph.dim_to_expr(in_tensors_[0].get_plain_dims()[0]),
                     iim_block)
            * iim_block;
          N_expr = divide_and_ceil(
                     graph.dim_to_expr(in_tensors_[1].get_plain_dims()[1]),
                     iin_block)
            * iin_block;
          K_expr = divide_and_ceil(
                     graph.dim_to_expr(in_tensors_[0].get_plain_dims()[1]),
                     iik_block)
            * iik_block;
          M_single_thr_size = get_balance211_length(
            M_expr / iim_block, M_split_num, m_s, m_idx, X_bigger_num);
          M_single_thr_size = M_single_thr_size * iim_block;
          m_idx = m_idx * iim_block;
          N_single_thr_size = get_balance211_length(
            N_expr / iin_block, N_split_num, n_s, n_idx, X_bigger_num);
          N_single_thr_size = N_single_thr_size * iin_block;
          n_idx = n_idx * iin_block;
          M_block_size_expr
            = divide_and_ceil(M_expr / iim_block, M_split_num) * iim_block;
          N_block_size_expr
            = divide_and_ceil(N_expr / iin_block, N_split_num) * iin_block;
        }
        expr M_single_thr_num_block
          = divide_and_ceil(M_single_thr_size, iim_block);
        expr N_single_thr_num_block
          = divide_and_ceil(N_single_thr_size, iin_block);
        if (out_tensors_[0].get_format().is_blocking()) {
          trace_guard_t tg(ctx, "blocking_post_reduce");
          for_loop reduce_loop;
          _named_for_(reduce_loop, lm_ln, 0,
            M_single_thr_num_block * N_single_thr_num_block, 1,
            for_type::PARALLEL, K_split_num) { //
            _var_init_(lm, datatypes::index, lm_ln / N_single_thr_num_block);
            _var_init_(ln, datatypes::index, lm_ln % N_single_thr_num_block);
            _var_init_(m_idx_1, datatypes::index, m_idx / iim_block);
            _var_init_(n_idx_1, datatypes::index, n_idx / iin_block);
            expr real_C = C, real_out_tmp_buf = out_tmp_buf;
            if (is_dynamic) {
              real_C = tensor_ptr(C,
                std::vector<expr>(C.checked_as<tensor>()->dims_.size(), 0),
                {divide_and_ceil(M_expr, iim_block),
                  divide_and_ceil(N_expr, iin_block), iim_block, iin_block});
              real_out_tmp_buf = tensor_ptr(out_tmp_buf,
                std::vector<expr>(
                  out_tmp_buf.get().checked_as<tensor>()->dims_.size(), 0),
                {K_real_split, M_block_size_expr / iim_block,
                  N_block_size_expr / iin_block, iim_block, iin_block});
            }
            builtin::mem_zero(
              tensor_ptr(real_C, {m_idx_1 + lm, n_idx_1 + ln, 0, 0}),
              iim_block * iin_block, out_dtype);
            _for_(lks, 0, K_real_split, 1) {
              if (!is_dynamic) {
                _for_(lmo, 0, iim_block) {
                  _for_(lno, 0, iin_block, lanes) {
                    C[span_t({m_idx / iim_block + lm, n_idx / iin_block + ln,
                               lmo, lno},
                      lanes)]
                      = builder::make_add(
                        C[span_t({m_idx / iim_block + lm,
                                   n_idx / iin_block + ln, lmo, lno},
                          lanes)],
                        out_tmp_buf[span_t({lks, lm, ln, lmo, lno}, lanes)]);
                  }
                }
              } else {
                _if_(lks < divide_and_ceil(K_expr, iik_block)) {
                  _for_(lmo, 0, iim_block) {
                    _for_(lno, 0, iin_block, lanes) {
                      real_C[span_t(
                        {m_idx_1 + lm, n_idx_1 + ln, lmo, lno}, lanes)]
                        = builder::make_add(
                          real_C[span_t(
                            {m_idx_1 + lm, n_idx_1 + ln, lmo, lno}, lanes)],
                          real_out_tmp_buf[span_t(
                            {lks, lm, ln, lmo, lno}, lanes)]);
                    }
                  }
                }
              }
            }
            if (fusion && !is_dynamic) {
              fusion->create_output_fusion_anchor({tensor_slice(C,
                {{m_idx / expr(iim_block) + lm, 1},
                  {n_idx / expr(iin_block) + ln, 1}, {0, expr(iim_block)},
                  {0, expr(iin_block)}})});
            }
          }
          reduce_loop->attr()["dont_prefetch"] = true;
        } else {
          trace_guard_t tg(ctx, "plain_post_reduce");
          builtin::dnnl_brgemm_init(tensor_ptr(C, {m_idx, n_idx}),
            builder::make_cast(datatypes::s32, M_single_thr_size),
            builder::make_cast(datatypes::s32, N_single_thr_size), N, out_dtype,
            0);
          _for_(lm_ln, 0, M_single_thr_size * N_single_thr_size, lanes,
            for_type::PARALLEL, K_split_num) {
            expr lm = lm_ln / N_single_thr_size;
            expr ln = lm_ln % N_single_thr_size;
            _for_(lks, 0, K_real_split, 1) {
              if (!is_dynamic) {
                C[span_t({m_idx + lm, n_idx + ln}, lanes)] = builder::make_add(
                  C[span_t({m_idx + lm, n_idx + ln}, lanes)],
                  out_tmp_buf[span_t({lks, lm, ln}, lanes)]);
              } else {
                _if_(lks < divide_and_ceil(K_expr, iik_block)) {
                  C[span_t({m_idx + lm, n_idx + ln}, lanes)]
                    = builder::make_add(
                      C[span_t({m_idx + lm, n_idx + ln}, lanes)],
                      out_tmp_buf[span_t({lks, lm, ln}, lanes)]);
                }
              }
            }
          }
          if (fusion && !is_dynamic) {
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
      if (fusion && !is_dynamic && N_split_num == 1) {
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
    = bound_axis {{0}, {1}, {}, {0}, {1}};
  loops = {mloop};
  return true;
}
} // namespace ops
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
