/*******************************************************************************
 * Copyright 2022 Intel Corporation
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
#include "utils.hpp"
#include <compiler/ir/builder.hpp>
#include <compiler/ir/easy_build.hpp>
#include <compiler/ir/graph/fusion_mgr.hpp>
#include <compiler/ir/transform/auto_cast.hpp>
#include <compiler/ir/transform/constant_fold.hpp>
#include <compiler/ir/transform/loop_transform.hpp>
#include <compiler/ir/transform/scope_flatten.hpp>
#include <microkernel/builtin.hpp>
#include <ops/matmul_core.hpp>
#include <runtime/config.hpp>
#include <runtime/parallel.hpp>
#include <util/any_map.hpp>
#include <util/math_utils.hpp>
#include <util/reflection.hpp>

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

template <typename T>
static std::vector<T> concat_vec(
  const std::vector<T> &a, const std::vector<T> &b) {
  std::vector<T> result(a);
  for (const T &it : b) {
    result.push_back(it);
  }
  return result;
}

static std::vector<int> get_splits(const int X) {
  std::vector<int> splits;
  for (auto i = 1; i <= X; ++i) {
    if (X % i == 0) { splits.push_back(i); }
  }
  return splits;
}

static expr divide_and_ceil(const expr &v, const expr &d) {
  return constant_folder_t()(auto_caster_t()((v + d - 1) / d)).remove_const();
}

static expr get_balance211_length(
  const expr &n, const expr &team, const expr &idx, expr &n_start, expr &T1) {
  assert(get_expr_as_int(team) >= 1);
  expr n1 = divide_and_ceil(n, team);
  expr n2 = n1 - 1;
  T1 = n - n2 * team;
  n_start
    = builder::make_select(idx <= T1, idx * n1, T1 * n1 + (idx - T1) * n2);
  return builder::make_select(idx < T1, n1, n2);
}

static void get_blocks_and_ib_blocks(const int X, const int X_split_num,
  const int ix_block, int &X_block_size, int &X_ib_block_size) {
  if (utils::divide_and_ceil(X, X_block_size) < (size_t)X_split_num
    && X_block_size > ix_block) {
    X_block_size -= ix_block;
  }
  // M, N, K imbalance block size
  X_ib_block_size = X - X_block_size * X_split_num <= 0
    ? X - X_block_size * (X_split_num - 1)
    : X_block_size + ix_block;
  if (X_ib_block_size < 0) {
    // cannot use all the threads
    X_ib_block_size = X - X / X_block_size * X_block_size;
  }
  if (X_ib_block_size == 0) { X_ib_block_size = X_block_size; }
}

config_ptr gen_managed_matmul_core_t::get_default_config(
  context_ptr ctx) const {
  auto ret = reflection::general_object_t::make<managed_matmul_core_config_t>();
  managed_matmul_core_config_t &cfg
    = *ret.unchecked_get_as<managed_matmul_core_config_t>();
  const int num_threads = runtime_config_t::get().get_num_threads();
  const auto splits = get_splits(runtime_config_t::get().get_num_threads());
  const int iim_block = iim_block_;
  const int iin_block = iin_block_;
  const int iik_block = iik_block_;
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
  if (M * N / iim_block / iin_block >= num_threads) {
    for (auto i : splits) {
      int num_M_block = utils::divide_and_ceil(M / iim_block, num_threads / i);
      int num_N_block = utils::divide_and_ceil(N / iin_block, i);
      int num_brgemm = num_M_block * num_N_block;
      int num_core
        = std::min(i, N / iin_block) * std::min(num_threads / i, M / iim_block);
      // Cost = Shape_efficient_weight *
      // (workload_balance + divide_N_plenty) / core_utilitizaiton
      // single core gemm prefers square shape for A and B.
      // For small workload, the A and B shape is not a key problem, but the
      // num_core and num_brgemm is important to performance. Use 2048 to reduce
      // the shape weight on small shape.
      float new_cost = (1024 + M * i / num_threads + N / i)
        * (num_brgemm + 8 * i) / num_core;
      if (new_cost < cost) {
        split_n = i;
        cost = new_cost;
      }
    }
    cfg.M_split_num = num_threads / split_n;
    cfg.N_split_num = split_n;
    int single_M = utils::divide_and_ceil(
                     utils::divide_and_ceil(M, iim_block), cfg.M_split_num)
      * iim_block;
    int single_N = utils::divide_and_ceil(
                     utils::divide_and_ceil(N, iin_block), cfg.N_split_num)
      * iin_block;
    int single_K = K;
    // TODO(zhennan): Query L2 cache size from hardware
    int L2_size = 1024 * 1024 * 2;
    int single_K_threshold
      = (single_M * single_N * sizeofdtypeA < L2_size ? 2048 : 4096)
      / sizeofdtypeA;
    if (single_K >= single_K_threshold) {
      cfg.K_sub_block = utils::divide_and_ceil(single_K, single_K_threshold);
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
  } else {
    // TODO(Zhennan): Need further improvement here.
    for (auto &split : splits) {
      if (split <= static_cast<int>(sqrt(num_threads))) { split_n = split; }
    }
    cfg.M_split_num = num_threads / split_n;
    cfg.N_split_num = split_n;
    cfg.M_sub_block = 1;
    cfg.N_sub_block = 1;
    cfg.K_sub_block = 1;
    cfg.im_loop_order = 1;
  }
  return std::move(ret);
}

gen_managed_matmul_core_t::gen_managed_matmul_core_t(sc_op *owner,
  std::vector<logical_tensor_t> &&ins, std::vector<logical_tensor_t> &&outs)
  : parent(owner, std::move(ins), std::move(outs)) {
  COMPILE_ASSERT(
    in_tensors_.size() == 2, "input logical tensor size should be two.");
  COMPILE_ASSERT(
    out_tensors_.size() == 1, "output logical tensor size should be one.");

  iim_block_ = 32;
  iin_block_
    = utils::is_one_of(get_A_dtype(), datatypes::u8, datatypes::s8) ? 64 : 32;
  iik_block_ = iin_block_;
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

void gen_managed_matmul_core_t::single_thread_matmul_call(
  const logical_tensor_t &ta, const logical_tensor_t &tb,
  const logical_tensor_t &tc, const managed_matmul_core_config_t &config,
  const expr &M, const expr &N, const expr &K, const expr &m_idx,
  const expr &n_idx, const expr &k_idx, const expr &A, const expr &B,
  const expr &C, int dtype_block, fusion_manager *fusion, int im_loop_order,
  const expr &m_s, const expr &n_s, std::vector<int> &M_anchor_info,
  std::vector<int> &N_anchor_info, bool is_partial, const expr &k_s) const {
  expr M_sub_block = config.M_sub_block, N_sub_block = config.N_sub_block,
       K_sub_block = config.K_sub_block;
  for_loop im_k, im_m, im_n, o_im_n;
  int ori_M = static_cast<int>(ta.get_plain_dims()[0]),
      ori_K = static_cast<int>(ta.get_plain_dims()[1]),
      ori_N = static_cast<int>(tb.get_plain_dims()[1]);
  _var_init_(tid, datatypes::s32, builtin::get_thread_id_func()());

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
            std::vector<expr> aidx = !ta.get_format().is_blocking()
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
            std::vector<expr> cidx = !tc.get_format().is_blocking()
              ? std::vector<expr> {m_start_idx, n_start_idx}
              : std::vector<expr> {
                m_start_idx / iim_block_, n_start_idx / iin_block_, 0, 0};
            if (is_partial) { cidx.insert(cidx.begin(), k_s); }
            auto LDA = !ta.get_format().is_blocking() ? ori_K : iik_block_;
            auto LDB = !tb.get_format().is_blocking() ? ori_N : iin_block_;
            auto LDC = !tc.get_format().is_blocking() ? ori_N : iin_block_;
            auto stride_a = !ta.get_format().is_blocking()
              ? iim_block_
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
          && M_anchor_info[1] % config.M_sub_block == 0
          && N_anchor_info[1] % config.N_sub_block == 0) {
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
          _var_init_(anchor_iter, datatypes::index, UINT64_C(0));
          // TODO(xxx): reduce the if-else node in IR
          _if_(m_s < config.M_split_num - M_anchor_info[0] || m_s == 0) {
            // 0-8
            _if_(n_s < config.N_split_num - N_anchor_info[0] || n_s == 0) {
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
            _if_(n_s < config.N_split_num - N_anchor_info[0] || n_s == 0) {
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
          fusion->create_iterated_fusion_anchor(anchor_iter, C, mm_multi_slice);
        }
      }
    }
  }
  if (config.K_sub_block > 1 && im_loop_order != 1) {
    im_n->attr()[stmt_attr_key::reduce_root_loop]
      = std::weak_ptr<stmt_base_t>(o_im_n.impl);
  }
  if (im_loop_order == 1) {
    im_m->reorder(im_k->body_, {im_n, im_m});
    im_m->attr()[stmt_attr_key::reduce_root_loop]
      = std::weak_ptr<stmt_base_t>(o_im_n.impl);
  }
}

/**
 * For each single thread we may deal with different size of matmuls
 * Take M axis as example, we have three following candidates:
 * 1) M_block_size
      M_block_size is used for all matmul shapes. It can be divided by 32.
      For instance, if M = 1792 with M_split_num=28, we have M_block_size=32
 * 2) M_ib_block_size (imbalance)
      M_ib_block_size is used for some specific shapes. For M=2240,
 M_split_num=28, we may choose96 as M_block_size. However, such a block size
 will leave 4 cores with no workload. So we need to choose 64 as M_block_size
 instead. Thus, 14 cores will process an M of 64, another 14 cores will process
 96, which is defined as M_ib_block_size.
 * 3) tail_M
      tail_M will be used when M cannot be divided by 32. For instance, if
 M=1791, M_split_num=28, we have tail_M=63. Note that tail_M will be round up
 into a number that is divisible by 32. This value is either M_block_size or
 M_ib_block_size.
 * */
bool gen_managed_matmul_core_t::generate(context_ptr ctx,
  const managed_matmul_core_config_t &config, fusion_manager *fusion,
  const std::vector<expr> &inputs, const std::vector<expr> &outputs,
  std::vector<for_loop> &loops) const {
  // Init
  int M_split_num = config.M_split_num, N_split_num = config.N_split_num;
  int num_threads = runtime_config_t::get().get_num_threads();
  int K_split_num = num_threads / M_split_num / N_split_num;
  COMPILE_ASSERT(
    num_threads % (M_split_num * N_split_num) == 0, "wrong split nums!");
  int M_sub_block = config.M_sub_block, N_sub_block = config.N_sub_block,
      K_sub_block = config.K_sub_block, im_loop_order = config.im_loop_order;
  int M = static_cast<int>(in_tensors_[0].get_plain_dims()[0]),
      K = static_cast<int>(in_tensors_[0].get_plain_dims()[1]),
      N = static_cast<int>(in_tensors_[1].get_plain_dims()[1]);
  int M_block_size
    = utils::divide_and_ceil(utils::divide_and_ceil(M, M_split_num), iim_block_)
    * iim_block_;
  int N_block_size
    = utils::divide_and_ceil(utils::divide_and_ceil(N, N_split_num), iin_block_)
    * iin_block_;
  int K_block_size
    = utils::divide_and_ceil(utils::divide_and_ceil(K, K_split_num), iik_block_)
    * iik_block_;
  // make sure that each thread has workload
  int M_ib_block_size, N_ib_block_size, K_ib_block_size;
  get_blocks_and_ib_blocks(
    M, M_split_num, iim_block_, M_block_size, M_ib_block_size);
  get_blocks_and_ib_blocks(
    N, N_split_num, iin_block_, N_block_size, N_ib_block_size);
  get_blocks_and_ib_blocks(
    K, K_split_num, iik_block_, K_block_size, K_ib_block_size);
  // update X_block_size and X_ib_block_size to minimize their gaps
  if (M_block_size >= iim_block_ * 2) {
    int M_new_ib_block_size, M_new_block_size = M_block_size - iim_block_;
    get_blocks_and_ib_blocks(
      M, M_split_num, iim_block_, M_new_block_size, M_new_ib_block_size);
    if (std::abs(M_block_size - M_ib_block_size)
      > std::abs(M_new_block_size - M_new_ib_block_size)) {
      M_block_size = M_new_block_size;
      M_ib_block_size = M_new_ib_block_size;
    }
  }
  if (N_block_size >= iin_block_ * 2) {
    int N_new_ib_block_size, N_new_block_size = N_block_size - iin_block_;
    get_blocks_and_ib_blocks(
      N, N_split_num, iin_block_, N_new_block_size, N_new_ib_block_size);
    if (std::abs(N_block_size - N_ib_block_size)
      > std::abs(N_new_block_size - N_new_ib_block_size)) {
      N_block_size = N_new_block_size;
      N_ib_block_size = N_new_ib_block_size;
    }
  }
  if (K_block_size >= iik_block_ * 2) {
    int K_new_ib_block_size, K_new_block_size = K_block_size - iik_block_;
    get_blocks_and_ib_blocks(
      K, K_split_num, iik_block_, K_new_block_size, K_new_ib_block_size);
    if (std::abs(K_block_size - K_ib_block_size)
      > std::abs(K_new_block_size - K_new_ib_block_size)) {
      K_block_size = K_new_block_size;
      K_ib_block_size = K_new_ib_block_size;
    }
  }
  // M, N, K imbalance block num
  int M_ib_num = M - M_block_size * M_split_num < 0
    ? 1
    : utils::divide_and_ceil(M - M_block_size * M_split_num, iim_block_);
  int N_ib_num = N - N_block_size * N_split_num < 0
    ? 1
    : utils::divide_and_ceil(N - N_block_size * N_split_num, iin_block_);
  int K_ib_num = K - K_block_size * K_split_num < 0
    ? 1
    : utils::divide_and_ceil(K - K_block_size * K_split_num, iik_block_);
  int tail_M = M_ib_num <= 1 ? M_ib_block_size
                             : M_ib_block_size
      - (M_ib_num * M_ib_block_size + (M_split_num - M_ib_num) * M_block_size
        - M);
  int tail_N = N_ib_num <= 1 ? N_ib_block_size
                             : N_ib_block_size
      - (N_ib_num * N_ib_block_size + (N_split_num - N_ib_num) * N_block_size
        - N);
  int tail_K = K_ib_num <= 1 ? K_ib_block_size
                             : K_ib_block_size
      - (K_ib_num * K_ib_block_size + (K_split_num - K_ib_num) * K_block_size
        - K);
  assert(M_ib_num >= 0 && M_ib_num <= M_split_num && N_ib_num >= 0
    && N_ib_num <= N_split_num && K_ib_num >= 0 && K_ib_num <= K_split_num);

  M_ib_block_size = utils::rnd_up(M_ib_block_size, iim_block_);
  N_ib_block_size = utils::rnd_up(N_ib_block_size, iin_block_);
  K_ib_block_size = utils::rnd_up(K_ib_block_size, iik_block_);
  tail_M = utils::rnd_up(tail_M, iim_block_);
  tail_N = utils::rnd_up(tail_N, iin_block_);
  tail_K = utils::rnd_up(tail_K, iik_block_);

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
        || in_tensors_[1].get_format().blocks_[2] == dtype_block,
      "Wrong data format of B");
  }

  expr C = outputs[op_params_t::out_C];
  expr A = inputs[op_params_t::in_A];
  expr B = inputs[op_params_t::in_B];
  // used for anchor construction when K_split_num==1 && K_sub_block>1
  std::vector<int> M_anchor_info = {M_ib_num, M_block_size, M_ib_block_size},
                   N_anchor_info = {N_ib_num, N_block_size, N_ib_block_size};
  for_loop mloop;
  if (K_split_num == 1) {
    expr m_idx, n_idx, k_idx, M_single_thr_size, N_single_thr_size;
    _named_for_(
      mloop, m_s, 0, M_split_num, 1, for_type::PARALLEL, M_split_num) {
      _for_(n_s, 0, N_split_num, 1,
        M_split_num == num_threads ? for_type::NORMAL : for_type::PARALLEL,
        M_split_num == num_threads ? 0 : N_split_num) {
        if (M_block_size == M_ib_block_size) {
          M_single_thr_size = M_block_size;
          m_idx = m_s * M_block_size;
        } else {
          if (M - M_block_size * M_split_num <= 0) {
            // cannot use all the cores due to small shapes
            m_idx = m_s * M_block_size;
          } else {
            m_idx = builder::make_select(m_s < M_split_num - M_ib_num,
              m_s * M_block_size,
              (M_split_num - M_ib_num) * M_block_size
                + (m_s + M_ib_num - M_split_num) * M_ib_block_size);
          }
          if (tail_M != M_ib_block_size) {
            // has tail and imbalance
            M_single_thr_size
              = builder::make_select(m_s < M_split_num - M_ib_num, M_block_size,
                builder::make_select(
                  m_s == M_split_num - 1, tail_M, M_ib_block_size));
          } else {
            if (M - M_block_size * M_split_num <= 0) {
              // cannot use all the cores due to small shapes
              M_single_thr_size = builder::make_select(
                m_s < M / M_block_size, M_block_size, M_ib_block_size);
            } else {
              M_single_thr_size = builder::make_select(
                m_s < M_split_num - M_ib_num, M_block_size, M_ib_block_size);
            }
          }
        }
        if (N_block_size == N_ib_block_size) {
          N_single_thr_size = N_block_size;
          n_idx = n_s * N_block_size;
        } else {
          if (N - N_block_size * N_split_num <= 0) {
            // cannot use all the cores due to small shapes
            n_idx = n_s * N_block_size;
          } else {
            n_idx = builder::make_select(n_s < N_split_num - N_ib_num,
              n_s * N_block_size,
              (N_split_num - N_ib_num) * N_block_size
                + (n_s + N_ib_num - N_split_num) * N_ib_block_size);
          }

          if (tail_N != N_ib_block_size) {
            // has tail and imbalance
            N_single_thr_size
              = builder::make_select(n_s < N_split_num - N_ib_num, N_block_size,
                builder::make_select(
                  n_s == N_split_num - 1, tail_N, N_ib_block_size));
          } else {
            if (N - N_block_size * N_split_num <= 0) {
              // cannot use all the cores due to small shapes
              N_single_thr_size = builder::make_select(
                n_s < N / N_block_size, N_block_size, N_ib_block_size);
            } else {
              N_single_thr_size = builder::make_select(
                n_s < N_split_num - N_ib_num, N_block_size, N_ib_block_size);
            }
          }
        }
        _for_(k_s, 0, K_split_num, 1,
          M_split_num * N_split_num == num_threads ? for_type::NORMAL
                                                   : for_type::PARALLEL,
          M_split_num * N_split_num == num_threads ? 0 : K_split_num) {
          _if_(m_idx < (uint64_t)M && n_idx < (uint64_t)N) {
            single_thread_matmul_call(in_tensors_[0], in_tensors_[1],
              out_tensors_[0], config, M_single_thr_size, N_single_thr_size,
              (int)utils::rnd_up(K, iik_block_), m_idx, n_idx, k_s, A, B, C,
              dtype_block, fusion, im_loop_order, m_s, n_s, M_anchor_info,
              N_anchor_info);
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
          _var_init_(middle_anchor_iter, datatypes::index, UINT64_C(0));
          if (M_block_size == M_ib_block_size
            && N_block_size == N_ib_block_size) {
            _if_(m_idx < (uint64_t)M && n_idx < (uint64_t)N) {
              if (out_tensors_[0].get_format().is_blocking()) {
                fusion->create_output_fusion_anchor({tensor_slice(C,
                  {{m_idx / expr(iim_block_), M_block_size / iim_block_},
                    {n_idx / expr(iin_block_), N_block_size / iin_block_},
                    {0, iim_block_}, {0, iin_block_}})});
              } else {
                fusion->create_output_fusion_anchor({tensor_slice(
                  C, {{m_idx, M_block_size}, {n_idx, N_block_size}})});
              }
            }
          } else if (M_block_size == M_ib_block_size) {
            // differnt length on N
            mm_multi_slice.pop_back();
            mm_multi_slice.pop_back();
            assert(mm_multi_slice.size() == 2);
            _if_(n_s < config.N_split_num - N_ib_num || n_s == 0) {
              middle_anchor_iter = UINT64_C(0);
            }
            _else_ { middle_anchor_iter = UINT64_C(1); }
            _if_(m_idx < (uint64_t)M && n_idx < (uint64_t)N) {
              fusion->create_iterated_fusion_anchor(
                middle_anchor_iter, C, mm_multi_slice);
            }
          } else if (N_block_size == N_ib_block_size) {
            // different length on M
            mm_multi_slice.pop_back();
            mm_multi_slice.erase(mm_multi_slice.begin() + 1);
            assert(mm_multi_slice.size() == 2);
            _if_(m_s < config.M_split_num - M_ib_num || m_s == 0) {
              middle_anchor_iter = UINT64_C(0);
            }
            _else_ { middle_anchor_iter = UINT64_C(1); }
            _if_(m_idx < (uint64_t)M && n_idx < (uint64_t)N) {
              fusion->create_iterated_fusion_anchor(
                middle_anchor_iter, C, mm_multi_slice);
            }
          } else {
            // different length on both M and N
            _if_(m_s < config.M_split_num - M_ib_num || m_s == 0) {
              _if_(n_s < config.N_split_num - N_ib_num || n_s == 0) {
                middle_anchor_iter = UINT64_C(0);
              }
              _else_ { middle_anchor_iter = UINT64_C(1); }
            }
            _else_ {
              _if_(n_s < config.N_split_num - N_ib_num || n_s == 0) {
                middle_anchor_iter = UINT64_C(2);
              }
              _else_ { middle_anchor_iter = UINT64_C(3); }
            }
            _if_(m_idx < (uint64_t)M && n_idx < (uint64_t)N) {
              fusion->create_iterated_fusion_anchor(
                middle_anchor_iter, C, mm_multi_slice);
            }
          }
        }
      }
      if (fusion) {
        _if_(m_idx < (uint64_t)M) {
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
            _var_init_(outer_anchor_iter, datatypes::index, UINT64_C(0));
            _if_(m_s < config.M_split_num - M_ib_num || m_s == 0) {
              outer_anchor_iter = UINT64_C(0);
            }
            _else_ { outer_anchor_iter = UINT64_C(1); }
            fusion->create_iterated_fusion_anchor(
              outer_anchor_iter, C, mm_multi_slice);
          }
        }
      }
    }
  } else {
    // write into a temp buffer and then do reduce
    auto out_tmp_buf_shape = out_tensors_[0].get_blocking_dims();
    out_tmp_buf_shape.insert(out_tmp_buf_shape.begin(), (sc_dim)K_split_num);
    std::vector<expr> out_tmp_buf_shape_expr;
    out_tmp_buf_shape_expr.reserve(out_tmp_buf_shape.size());
    for (auto dim : out_tmp_buf_shape) {
      out_tmp_buf_shape_expr.emplace_back(dim2unsigned(dim));
    }
    auto out_dtype = utils::is_one_of(A_dtype, datatypes::u8, datatypes::s8)
      ? datatypes::s32
      : datatypes::f32;
    expr m_idx, n_idx, k_idx, M_single_thr_size, N_single_thr_size;
    _tensor_(out_tmp_buf, out_dtype, out_tmp_buf_shape_expr);
    _named_for_(
      mloop, m_s, 0, M_split_num, 1, for_type::PARALLEL, M_split_num) {
      _for_(n_s, 0, N_split_num, 1, for_type::PARALLEL, N_split_num) {
        if (M_block_size == M_ib_block_size) {
          M_single_thr_size = M_block_size;
          m_idx = m_s * M_block_size;
        } else {
          if (M - M_block_size * M_split_num <= 0) {
            // cannot use all the cores due to small shapes
            m_idx = m_s * M_block_size;
          } else {
            m_idx = builder::make_select(m_s < M_split_num - M_ib_num,
              m_s * M_block_size,
              (M_split_num - M_ib_num) * M_block_size
                + (m_s + M_ib_num - M_split_num) * M_ib_block_size);
          }
          if (tail_M != M_ib_block_size) {
            // has tail and imbalance
            M_single_thr_size
              = builder::make_select(m_s < M_split_num - M_ib_num, M_block_size,
                builder::make_select(
                  m_s == M_split_num - 1, tail_M, M_ib_block_size));
          } else {
            if (M - M_block_size * M_split_num <= 0) {
              // cannot use all the cores due to small shapes
              M_single_thr_size = builder::make_select(
                m_s < M / M_block_size, M_block_size, M_ib_block_size);
            } else {
              M_single_thr_size = builder::make_select(
                m_s < M_split_num - M_ib_num, M_block_size, M_ib_block_size);
            }
          }
        }
        if (N_block_size == N_ib_block_size) {
          N_single_thr_size = N_block_size;
          n_idx = n_s * N_block_size;
        } else {
          if (N - N_block_size * N_split_num <= 0) {
            // cannot use all the cores due to small shapes
            n_idx = n_s * N_block_size;
          } else {
            n_idx = builder::make_select(n_s < N_split_num - N_ib_num,
              n_s * N_block_size,
              (N_split_num - N_ib_num) * N_block_size
                + (n_s + N_ib_num - N_split_num) * N_ib_block_size);
          }

          if (tail_N != N_ib_block_size) {
            // has tail and imbalance
            N_single_thr_size
              = builder::make_select(n_s < N_split_num - N_ib_num, N_block_size,
                builder::make_select(
                  n_s == N_split_num - 1, tail_N, N_ib_block_size));
          } else {
            if (N - N_block_size * N_split_num <= 0) {
              // cannot use all the cores due to small shapes
              N_single_thr_size = builder::make_select(
                n_s < N / N_block_size, N_block_size, N_ib_block_size);
            } else {
              N_single_thr_size = builder::make_select(
                n_s < N_split_num - N_ib_num, N_block_size, N_ib_block_size);
            }
          }
        }

        _for_(k_s, 0, K_split_num, 1, for_type::PARALLEL, K_split_num) { //
          expr K_single_thr_size;
          if (K_block_size == K_ib_block_size) {
            K_single_thr_size = K_block_size;
            k_idx = k_s * K_block_size;
          } else {
            if (K - K_block_size * K_split_num <= 0) {
              // cannot use all the cores due to small shapes
              k_idx = k_s * K_block_size;
            } else {
              k_idx = builder::make_select(k_s < K_split_num - K_ib_num,
                k_s * K_block_size,
                (K_split_num - K_ib_num) * K_block_size
                  + (k_s + K_ib_num - K_split_num) * K_ib_block_size);
            }
            if (tail_K != K_ib_block_size) {
              // has tail and imbalance
              K_single_thr_size = builder::make_select(
                k_s < K_split_num - K_ib_num, K_block_size,
                builder::make_select(
                  k_s == K_split_num - 1, tail_K, K_ib_block_size));
            } else {
              if (K - K_block_size * K_split_num <= 0) {
                // cannot use all the cores due to small shapes
                K_single_thr_size = builder::make_select(
                  k_s < K / K_block_size, K_block_size, K_ib_block_size);
              } else {
                K_single_thr_size = builder::make_select(
                  k_s < K_split_num - K_ib_num, K_block_size, K_ib_block_size);
              }
            }
          }
          _if_(
            m_idx < (uint64_t)M && n_idx < (uint64_t)N && k_idx < (uint64_t)K) {
            single_thread_matmul_call(in_tensors_[0], in_tensors_[1],
              out_tensors_[0], config, M_single_thr_size, N_single_thr_size,
              K_single_thr_size, m_idx, n_idx, k_idx, A, B, out_tmp_buf,
              dtype_block, fusion, im_loop_order, m_s, n_s, M_anchor_info,
              N_anchor_info, true, k_s);
          }
        }
        // do reduce here
        for_loop rm, rn;
        // since the blockings are ix_block_ and plain shapes must be divided by
        // ix_block_, we can use lanes=16 directly
        int lanes = 16;
        assert(iim_block_ % 16 == 0);
        assert(iin_block_ % 16 == 0);
        expr M_single_thr_num_block
          = divide_and_ceil(M_single_thr_size, iim_block_);
        expr N_single_thr_num_block
          = divide_and_ceil(N_single_thr_size, iin_block_);
        if (out_tensors_[0].get_format().is_blocking()) {
          _for_(lm_ln, 0, M_single_thr_num_block * N_single_thr_num_block, 1,
            for_type::PARALLEL, K_split_num) { //
            expr lm = lm_ln / N_single_thr_num_block;
            expr ln = lm_ln % N_single_thr_num_block;
            _if_(m_idx < (uint64_t)M && n_idx < (uint64_t)N) {
              builtin::mem_zero(
                tensor_ptr(
                  C, {m_idx / iim_block_ + lm, n_idx / iin_block_ + ln, 0, 0}),
                iim_block_ * iin_block_, out_dtype);
              _for_(lks, 0, K_split_num, 1) {
                _for_(lmo, 0, iim_block_) {
                  _for_(lno, 0, iin_block_, lanes) {
                    C[span_t({m_idx / iim_block_ + lm, n_idx / iin_block_ + ln,
                               lmo, lno},
                      lanes)]
                      = builder::make_add(
                        C[span_t({m_idx / iim_block_ + lm,
                                   n_idx / iin_block_ + ln, lmo, lno},
                          lanes)],
                        out_tmp_buf[span_t({lks, m_idx / iim_block_ + lm,
                                             n_idx / iin_block_ + ln, lmo, lno},
                          lanes)]);
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
          }
        } else {
          _if_(m_idx < (uint64_t)M && n_idx < (uint64_t)N) {
            builtin::dnnl_brgemm_init(tensor_ptr(C, {m_idx, n_idx}),
              M_single_thr_size, N_single_thr_size, N, out_dtype, 0);
          }
          _for_(lm_ln, 0, M_single_thr_size * N_single_thr_size, lanes,
            for_type::PARALLEL, K_split_num) {
            expr lm = lm_ln / N_single_thr_size;
            expr ln = lm_ln % N_single_thr_size;
            _for_(lks, 0, K_split_num, 1) {
              _if_(m_idx < (uint64_t)M && n_idx < (uint64_t)N) {
                C[span_t({m_idx + lm, n_idx + ln}, lanes)] = builder::make_add(
                  C[span_t({m_idx + lm, n_idx + ln}, lanes)],
                  out_tmp_buf[span_t({lks, m_idx + lm, n_idx + ln}, lanes)]);
              }
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
            _var_init_(inner_anchor_iter, datatypes::index, UINT64_C(0));
            if (M_block_size == M_ib_block_size
              && N_block_size == N_ib_block_size) {
              _if_(m_idx < (uint64_t)M && n_idx < (uint64_t)N) {
                if (out_tensors_[0].get_format().is_blocking()) {
                  fusion->create_output_fusion_anchor({tensor_slice(C,
                    {{m_idx / expr(iim_block_), M_block_size / iim_block_},
                      {n_idx / expr(iin_block_), N_block_size / iin_block_},
                      {0, iim_block_}, {0, iin_block_}})});
                } else {
                  fusion->create_output_fusion_anchor({tensor_slice(
                    C, {{m_idx, M_block_size}, {n_idx, N_block_size}})});
                }
              }
            } else if (M_block_size == M_ib_block_size) {
              // differnt length on N
              mm_multi_slice.pop_back();
              mm_multi_slice.pop_back();
              assert(mm_multi_slice.size() == 2);
              _if_(n_s < config.N_split_num - N_ib_num || n_s == 0) {
                inner_anchor_iter = UINT64_C(0);
              }
              _else_ { inner_anchor_iter = UINT64_C(1); }
              _if_(m_idx < (uint64_t)M && n_idx < (uint64_t)N) {
                fusion->create_iterated_fusion_anchor(
                  inner_anchor_iter, C, mm_multi_slice);
              }
            } else if (N_block_size == N_ib_block_size) {
              // different length on M
              mm_multi_slice.pop_back();
              mm_multi_slice.erase(mm_multi_slice.begin() + 1);
              assert(mm_multi_slice.size() == 2);
              _if_(m_s < config.M_split_num - M_ib_num || m_s == 0) {
                inner_anchor_iter = UINT64_C(0);
              }
              _else_ { inner_anchor_iter = UINT64_C(1); }
              _if_(m_idx < (uint64_t)M && n_idx < (uint64_t)N) {
                fusion->create_iterated_fusion_anchor(
                  inner_anchor_iter, C, mm_multi_slice);
              }
            } else {
              // different length on both M and N
              _if_(m_s < config.M_split_num - M_ib_num || m_s == 0) {
                _if_(n_s < config.N_split_num - N_ib_num || n_s == 0) {
                  inner_anchor_iter = UINT64_C(0);
                }
                _else_ { inner_anchor_iter = UINT64_C(1); }
              }
              _else_ {
                _if_(n_s < config.N_split_num - N_ib_num || n_s == 0) {
                  inner_anchor_iter = UINT64_C(2);
                }
                _else_ { inner_anchor_iter = UINT64_C(3); }
              }
              _if_(m_idx < (uint64_t)M && n_idx < (uint64_t)N) {
                fusion->create_iterated_fusion_anchor(
                  inner_anchor_iter, C, mm_multi_slice);
              }
            }
          }
        }
      }
      if (fusion) {
        _if_(m_idx < (uint64_t)M) {
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
            _var_init_(outer_anchor_iter, datatypes::index, UINT64_C(0));
            _if_(m_s < config.M_split_num - M_ib_num || m_s == 0) {
              outer_anchor_iter = UINT64_C(0);
            }
            _else_ { outer_anchor_iter = UINT64_C(1); }
            fusion->create_iterated_fusion_anchor(
              outer_anchor_iter, C, mm_multi_slice);
          }
        }
      }
    }
  }
  if (M_split_num == num_threads) {
    mloop->attr()[stmt_attr_key::parallel_merge_loop] = true;
  }
  loops = {};
  return true;
}
} // namespace ops
} // namespace sc
