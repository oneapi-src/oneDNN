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

#include "matmul_core.hpp"
#include <algorithm>
#include <string>
#include "utils.hpp"
#include <compiler/ir/builder.hpp>
#include <compiler/ir/easy_build.hpp>
#include <compiler/ir/graph/fusion_mgr.hpp>
#include <compiler/ir/transform/loop_transform.hpp>
#include <compiler/ir/transform/scope_flatten.hpp>
#include <microkernel/builtin.hpp>
#include <ops/matmul_core.hpp>
#include <runtime/config.hpp>
#include <runtime/parallel.hpp>
#include <util/any_map.hpp>
#include <util/math_utils.hpp>

using namespace sc::builder;
namespace sc {
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

// check if cfg is valid for lower library, fallback to a default value if is
// not valid
static void inline validate_cfg(
  matmul_core_config_t &cfg, bool is_amx, sc_data_type_t dtype) {
  if (!is_amx) return;
  int rd_block = dtype == datatypes::bf16
    ? 32
    : utils::is_one_of(dtype, datatypes::u8, datatypes::s8) ? 64 : -1;
  if (rd_block == -1) return;
  int rdb = cfg.K_block / rd_block;
  int rdb_tail = cfg.K_block % rd_block;
  if (rdb > 0 && rdb_tail) {
    cfg.K_block = utils::rnd_dn(cfg.K_block, rd_block);
  }
  int dtype_block = (rd_block == 32 ? 2 : 4);
  if (rdb_tail % dtype_block) {
    cfg.K_block = utils::rnd_up(cfg.K_block, dtype_block);
  }
}
// default cfg for bmm bases on below priori knowledge(based on tests on MHA)
// 1. if M % x == 0 (32 <= x <64), x as m_blk usually performs better than
// 32/64, if there are multiple x to choose, even number is better than odd
// number
// 2. in mha case, due to post fusion exp can cost a lot, first bmm's Nblk
// shouldn't be padded too much to reduce overhead and shall be rnd_up or rnd_dn
// to 16x for f32/s32, and should be close to M_blk, rnd_dn(M_blk, 16) is fine
// 3. smaller M*K should use 32 as k_blk, threshold is still testing

std::shared_ptr<void> gen_matmul_core_t::get_default_config(
  context_ptr ctx) const {
  // todo:(xianhang) take into consideration num thread information from
  // threadpool
  auto ret = std::make_shared<matmul_core_config_t>();
  const bool is_amx = is_use_amx(ctx);
  const bool is_int8
    = utils::is_one_of(get_in_dtypes(0), datatypes::u8, datatypes::s8);
  const bool is_bf16 = get_in_dtypes(0) == datatypes::bf16;
  const int max_block = 64;
  const int min_block = 32;
  matmul_core_config_t &cfg = *ret;
  auto A_plain_dims = get_mma_plain_dims();
  std::vector<int> possible_blks;
  cfg.K_block = 64;
  if (in_tensors_[0].get_format().is_blocking()) {
    cfg.M_block = in_tensors_[0].get_format().blocks_[0];
    cfg.K_block = in_tensors_[0].get_format().blocks_[1];
    if (!get_a_batch_dims().empty() || !get_b_batch_dims().empty()) {
      cfg.N_block = 64;
      // Safe Guard: avoid K_block % rbd != 0
      validate_cfg(cfg, is_amx, get_in_dtypes(0));
      return ret;
    }
  } else {
    assert(A_plain_dims.size() == 2);
    int M = static_cast<int>(A_plain_dims[0]);
    int K = static_cast<int>(A_plain_dims[1]);
    if (get_a_batch_dims().empty() && get_b_batch_dims().empty()) {
      // matmul2d default config
      cfg.M_block = std::min(M, 64);
      cfg.K_block = std::min(K, 64);
    } else {
      // bmm default config
      cfg.M_block = 0;
      for (int m = max_block; m >= min_block; m--) {
        if (M % m == 0) { possible_blks.emplace_back(m); }
      }
      for (const auto &blk : possible_blks) {
        if (blk % 2 == 0) {
          cfg.M_block = blk;
          break;
        }
      }
      if (cfg.M_block == 0) {
        if (!possible_blks.empty()) {
          cfg.M_block = possible_blks.front();
        } else {
          if (M > 64) {
            int ceil64_M = static_cast<int>(sc::utils::rnd_up(M, 64));
            int ceil48_M = static_cast<int>(sc::utils::rnd_up(M, 48));
            int ceil32_M = static_cast<int>(sc::utils::rnd_up(M, 32));
            int pad64_M = ceil64_M - M;
            int pad48_M = ceil48_M - M;
            int pad32_M = ceil32_M - M;
            if (!(is_amx && is_bf16)) {
              cfg.M_block = pad48_M >= pad64_M ? (pad64_M > pad32_M ? 32 : 64)
                                               : (pad48_M >= pad32_M ? 32 : 48);
            } else {
              cfg.M_block = pad32_M >= pad64_M ? 64 : 32;
            }
          } else {
            cfg.M_block = std::min(M, 64);
          }
        }
      }
    }
  }
  auto B_plain_dims = get_mmb_plain_dims();
  if (in_tensors_[1].get_format().is_blocking()) {
    assert(in_tensors_[1].get_format().blocks_[0] == cfg.K_block);
    cfg.N_block = in_tensors_[1].get_format().blocks_[1];
  } else {
    assert(B_plain_dims.size() == 2);
    int N = static_cast<int>(B_plain_dims[1]);
    int K = static_cast<int>(B_plain_dims[0]);
    if (get_a_batch_dims().empty() && get_b_batch_dims().empty()) {
      // matmul2d default config
      assert(A_plain_dims[1] == B_plain_dims[0]);
      cfg.N_block = std::min(N, 64);
      cfg.K_block = std::min(K, 64);
    } else {
      // bmm default config
      assert(cfg.M_block > 0);

      int ceil64_N = static_cast<int>(sc::utils::rnd_up(N, 64));
      int ceil48_N = static_cast<int>(sc::utils::rnd_up(N, 48));
      int ceil32_N = static_cast<int>(sc::utils::rnd_up(N, 32));
      int pad64_N = ceil64_N - N;
      int pad48_N = ceil48_N - N;
      int pad32_N = ceil32_N - N;
      if (!(is_amx && is_bf16)) {
        cfg.N_block = pad48_N >= pad64_N ? (pad64_N > pad32_N ? 32 : 64)
                                         : (pad48_N >= pad32_N ? 32 : 48);
      } else {
        cfg.N_block = pad32_N >= pad64_N ? 64 : 32;
      }
      if (N < 32) { cfg.N_block = 16; }
    }
  }
  validate_cfg(cfg, is_amx, get_in_dtypes(0));
  return ret;
}

gen_matmul_core_t::gen_matmul_core_t(
  std::vector<logical_tensor_t> &&ins, std::vector<logical_tensor_t> &&outs)
  : parent(std::move(ins), std::move(outs)) {
  COMPILE_ASSERT(
    in_tensors_.size() == 2, "input logical tensor size should be two.");
  COMPILE_ASSERT(
    out_tensors_.size() == 1, "output logical tensor size should be one.");
}

float gen_matmul_core_t::get_gflop() const {
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

void gen_matmul_core_t::get_and_check_blocks(const logical_tensor_t &ta,
  const logical_tensor_t &tb, const matmul_core_config_t &config,
  int &M_num_blocks, int &K_num_blocks, int &M_block, int &K_block,
  int &N_block, int &B_K_num_blocks, int &N_num_blocks) {
  const sc_dims &A_dims = ta.get_blocking_dims();
  const sc_dims &B_dims = tb.get_blocking_dims();
  size_t pds_a = ta.get_plain_dims().size();
  size_t pds_b = tb.get_plain_dims().size();
  const int plain_M = ta.get_plain_dims()[pds_a - 2];
  const int plain_K = ta.get_plain_dims()[pds_a - 1];
  const int plain_B_K = tb.get_plain_dims()[pds_b - 2];
  const int plain_N = tb.get_plain_dims()[pds_b - 1];
  bool is_config_set
    = config.M_block != 0 && config.K_block != 0 && config.N_block != 0;

  if (!ta.get_format().is_blocking()) {
    COMPILE_ASSERT(is_config_set, "config must be set with plain input.");
    M_block = config.M_block;
    K_block = config.K_block;
  } else {
    M_block = ta.get_format().blocks_[0];
    K_block = ta.get_format().blocks_[1];
  }
  N_block = tb.get_format().blocks_[1];
  assert(K_block == tb.get_format().blocks_[0]);
  COMPILE_ASSERT(!is_config_set
      || (M_block == config.M_block && K_block == config.K_block
        && N_block == config.N_block),
    "Unmatched config with input format");

  M_num_blocks = utils::divide_and_ceil(plain_M, M_block);
  K_num_blocks = utils::divide_and_ceil(plain_K, K_block);
  B_K_num_blocks = utils::divide_and_ceil(plain_B_K, K_block);
  N_num_blocks = utils::divide_and_ceil(plain_N, N_block);

  COMPILE_ASSERT(plain_K == plain_B_K,
    "K in A and B are not match, got " << plain_K << " v.s. " << plain_B_K);
  COMPILE_ASSERT(
    K_num_blocks == B_K_num_blocks, "A and B num blocks of K are not equal.");
}

void gen_matmul_core_t::get_brgemm_and_fusion_params(const logical_tensor_t &ta,
  const logical_tensor_t &tb, const logical_tensor_t &tc,
  std::vector<expr> &aidx, std::vector<expr> &bidx, std::vector<expr> &cidx,
  int &stride_a, int &stride_b, std::vector<std::pair<expr, expr>> &fidx1,
  std::vector<std::pair<expr, expr>> &fidx2) {
  if (ta.get_format().format_code_.is_batch_format()
    || tb.get_format().format_code_.is_batch_format()) {
    assert(tc.get_format().format_code_.is_batch_format());
    return;
  }
  bool flag = false;
  int bds = ta.get_plain_dims().size() - 2;
  std::vector<expr> aidx_ = aidx;
  std::vector<expr> bidx_ = bidx;
  std::vector<expr> cidx_ = cidx;
  std::vector<std::pair<expr, expr>> fidx1_ = fidx1;
  std::vector<std::pair<expr, expr>> fidx2_ = fidx2;
  int batch_idx = 0;
  for (uint8_t i = 0; i < (uint8_t)ta.get_plain_dims().size(); i++) {
    if (ta.get_format().format_code_.get(i)
      == ta.get_format().format_code_.get(ta.get_plain_dims().size())) {
      aidx_[i] = aidx[bds];
    } else if (ta.get_format().format_code_.get(i)
      == ta.get_format().format_code_.get(ta.get_plain_dims().size() + 1)) {
      aidx_[i] = aidx[bds + 1];
      flag = true;
      continue;
    } else {
      aidx_[i] = aidx[batch_idx];
      batch_idx++;
    }
    if (flag) { stride_a *= ta.get_blocking_dims()[i]; }
  }
  aidx.swap(aidx_);

  flag = false;
  batch_idx = 0;
  for (uint8_t i = 0; i < (uint8_t)tb.get_plain_dims().size(); i++) {
    if (tb.get_format().format_code_.get(i)
      == tb.get_format().format_code_.get(tb.get_plain_dims().size())) {
      bidx_[i] = bidx[bds + 1];
      flag = true;
      continue;
    } else if (tb.get_format().format_code_.get(i)
      == tb.get_format().format_code_.get(tb.get_plain_dims().size() + 1)) {
      bidx_[i] = bidx[bds];
    } else {
      bidx_[i] = bidx[batch_idx];
      batch_idx++;
    }
    if (flag) { stride_b *= tb.get_blocking_dims()[i]; }
  }
  bidx.swap(bidx_);

  batch_idx = 0;
  for (uint8_t i = 0; i < (uint8_t)tc.get_plain_dims().size(); i++) {
    if (tc.get_format().format_code_.get(i)
      == tc.get_format().format_code_.get(tc.get_plain_dims().size())) {
      cidx_[i] = cidx[bds];
      fidx1_[i] = fidx1[bds];
      fidx2_[i] = fidx2[bds];
    } else if (tc.get_format().format_code_.get(i)
      == tc.get_format().format_code_.get(tc.get_plain_dims().size() + 1)) {
      cidx_[i] = cidx[bds + 1];
      fidx1_[i] = fidx1[bds + 1];
      fidx2_[i] = fidx2[bds + 1];
    } else {
      cidx_[i] = cidx[batch_idx];
      fidx1_[i] = fidx1[batch_idx];
      fidx2_[i] = fidx2[batch_idx];
      batch_idx++;
    }
  }
  cidx.swap(cidx_);
  fidx1.swap(fidx1_);
  fidx2.swap(fidx2_);
}

void gen_matmul_core_t::schedule_loops(context_ptr ctx,
  const matmul_core_config_t &config, stmt body,
  std::vector<for_loop> &fors) const {
  if (get_a_batch_dims().empty() && get_b_batch_dims().empty()) {
    for_loop lm_c = fors.at(fors.size() - 2), ln_c = fors.back();
    auto lmn = lm_c->fuse(ln_c);
  } else {
    size_t bs = std::max(get_a_batch_dims().size(), get_b_batch_dims().size());
    for (size_t i = 1; i < bs; i++) {
      fors[0] = fors[0]->fuse(fors[i]);
    }
    stmts matmul_body = fors[0]->body_.static_as<stmts>();
    scope_flatten(matmul_body, -1);
    fors[0]->fuse(matmul_body->seq_[0].static_as<for_loop>());
  }
}

bool gen_matmul_core_t::generate(context_ptr ctx,
  const matmul_core_config_t &config, fusion_manager *fusion,
  const std::vector<expr> &inputs, const std::vector<expr> &outputs,
  std::vector<for_loop> &loops) const {
  // matmul_core will support plain format as input in the future
  COMPILE_ASSERT(!in_tensors_[0].get_format().is_plain()
      && !in_tensors_[1].get_format().is_plain(),
    "matmul_core should input blocking format");

  // Init
  auto A_dtype = get_A_dtype(), B_dtype = get_B_dtype();
  int M_block = 0, K_block = 0, N_block = 0, M_num_blocks = 0, N_num_blocks = 0,
      K_num_blocks = 0, B_K_num_blocks = 0;

  get_and_check_blocks(in_tensors_[0], in_tensors_[1], config, M_num_blocks,
    K_num_blocks, M_block, K_block, N_block, B_K_num_blocks, N_num_blocks);

  bool is_config_set
    = config.M_block != 0 && config.K_block != 0 && config.N_block != 0;
  COMPILE_ASSERT(!is_config_set
      || (M_block == config.M_block && K_block == config.K_block
        && N_block == config.N_block),
    "Unmatched config with input format");

  int dtype_block = 1;
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

  for_loop lm_c, ln_c;

  expr C = outputs[op_params_t::out_C];
  expr A = inputs[op_params_t::in_A];
  expr B = inputs[op_params_t::in_B];
  auto batch_dims = get_a_batch_dims().size() > get_b_batch_dims().size()
    ? get_a_batch_dims()
    : get_b_batch_dims();
  auto batch_dims_size = batch_dims.size();
  auto small_batch_dims_size
    = get_a_batch_dims().size() > get_b_batch_dims().size()
    ? get_b_batch_dims().size()
    : get_a_batch_dims().size();
  std::vector<expr> idxs, idxs_small;
  std::vector<for_loop> batch_loops;
  std::vector<for_range_simulator_t> ranges;
  idxs.resize(batch_dims_size);
  idxs_small.resize(small_batch_dims_size);
  batch_loops.resize(batch_dims_size);
  ranges.reserve(batch_dims_size);
  for (size_t i = 0; i < batch_dims_size; i++) {
    ranges.emplace_back(sc::builder::range(batch_loops[i], expr(0),
      expr(dim2unsigned(batch_dims[i])), expr(1), for_type::PARALLEL));
  }
  if (!batch_dims.empty()) {
    _nested_for_(std::move(ranges)) {
      std::vector<std::pair<expr, expr>> batch_tensor_slice_ranges, fidx1,
        fidx2;
      for (size_t i = 0; i < batch_dims_size; i++) {
        idxs[i] = _0_nested_for.get_var();
        idxs[i].checked_as<var>()->name_
          = std::string("idx") + std::to_string(i);
        batch_tensor_slice_ranges.emplace_back(idxs[i], 1);
      }
      idxs_small
        = {idxs.begin() + batch_dims_size - small_batch_dims_size, idxs.end()};
      _named_for_(lm_c, m_o, 0, M_num_blocks) {
        _named_for_(ln_c, n_o, 0, N_num_blocks) {
          std::vector<expr> aidx = concat_vec(
            get_a_batch_dims().size() > get_b_batch_dims().size() ? idxs
                                                                  : idxs_small,
            std::vector<expr> {m_o, 0, 0, 0});
          std::vector<expr> bidx = concat_vec(
            get_a_batch_dims().size() > get_b_batch_dims().size() ? idxs_small
                                                                  : idxs,
            std::vector<expr> {n_o, 0, 0, 0});
          std::vector<expr> cidx
            = concat_vec(idxs, std::vector<expr> {m_o, n_o, 0, 0});
          fidx1 = concat_vec(batch_tensor_slice_ranges,
            {{m_o, 1}, {n_o, 1}, {0, M_block}, {0, N_block}});
          fidx2 = concat_vec(batch_tensor_slice_ranges,
            {{m_o, 1}, {0, N_num_blocks}, {0, M_block}, {0, N_block}});

          if (dtype_block > 1) bidx.emplace_back(0);
          int stride_a = M_block * K_block,
              stride_b = (int)utils::divide_and_ceil(K_block, dtype_block)
            * dtype_block * N_block;

          get_brgemm_and_fusion_params(in_tensors_[0], in_tensors_[1],
            out_tensors_[0], aidx, bidx, cidx, stride_a, stride_b, fidx1,
            fidx2);

          sc::builtin::brgemm_init_update_allow_fusion(tensor_ptr(A, aidx),
            tensor_ptr(B, bidx), tensor_ptr(C, cidx), K_num_blocks, M_block,
            N_block, K_block, K_block, N_block, N_block, stride_a, stride_b,
            A_dtype, B_dtype);

          // this is the gemm output
          if (fusion) {
            std::vector<tensor_slice> fusion_inputs
              = {tensor_slice(C, std::vector<std::pair<expr, expr>>(fidx1))};
            fusion->create_output_fusion_anchor(fusion_inputs);
          }
        }
        if (fusion) {
          std::vector<tensor_slice> fusion_inputs
            = {tensor_slice(C, std::vector<std::pair<expr, expr>>(fidx2))};
          fusion->create_output_fusion_anchor(fusion_inputs);
        }
      }
      if (fusion) {
        fusion->create_output_fusion_anchor({tensor_slice(C,
          concat_vec(batch_tensor_slice_ranges,
            {{0, M_num_blocks}, {0, N_num_blocks}, {0, M_block},
              {0, N_block}}))});
      }
    }
  } else {
    _named_for_(lm_c, m_o, 0, M_num_blocks, 1, for_type::PARALLEL) {
      _named_for_(ln_c, n_o, 0, N_num_blocks) {
        sc::builtin::brgemm_init_update_allow_fusion(
          tensor_ptr(A, std::vector<expr> {m_o, 0, 0, 0}),
          tensor_ptr(B,
            dtype_block > 1 ? std::vector<expr> {n_o, 0, 0, 0, 0}
                            : std::vector<expr> {n_o, 0, 0, 0}),
          tensor_ptr(C, std::vector<expr> {m_o, n_o, 0, 0}), K_num_blocks,
          M_block, N_block, K_block, K_block, N_block, N_block,
          M_block * K_block,
          (int)utils::divide_and_ceil(K_block, dtype_block) * dtype_block
            * N_block,
          A_dtype, B_dtype);

        // this is the gemm output
        if (fusion) {
          fusion->create_output_fusion_anchor({tensor_slice(
            C, {{m_o, 1}, {n_o, 1}, {0, M_block}, {0, N_block}})});
        }
      }
      // this is the gemm output
      if (fusion
        && (ctx->bwise_fusion_
          || M_num_blocks >= runtime_config_t::get().threads_per_instance_)) {
        fusion->create_output_fusion_anchor({tensor_slice(
          C, {{m_o, 1}, {0, N_num_blocks}, {0, M_block}, {0, N_block}})});
      }
    }
  }

  loops = concat_vec(batch_loops, {lm_c, ln_c});
  return true;
}
} // namespace ops
} // namespace sc
