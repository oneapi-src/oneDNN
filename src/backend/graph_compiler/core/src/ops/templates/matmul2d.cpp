/*******************************************************************************
 * Copyright 2020-2021 Intel Corporation
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

#include "matmul2d.hpp"
#include <algorithm>
#include <utility>
#include "utils.hpp"
#include <compiler/ir/builder.hpp>
#include <compiler/ir/easy_build.hpp>
#include <compiler/ir/graph/fusion_mgr.hpp>
#include <compiler/ir/transform/scope_flatten.hpp>
#include <compiler/ir/transform/tensor_shrink.hpp>
#include <microkernel/builtin.hpp>
#include <runtime/config.hpp>
#include <util/any_map.hpp>
#include <util/utils.hpp>

using namespace sc::builder;
namespace sc {
namespace ops {

static bool is_valid_num_tiles(int K_num_tile, int K_num_blocks) {
  return K_num_tile <= K_num_blocks && K_num_blocks % K_num_tile == 0;
}

std::shared_ptr<void> gen_matmul2d_t::get_default_config(
  context_ptr ctx) const {
  auto ret = std::make_shared<matmul2d_config_t>();
  matmul2d_config_t &cfg = *ret;
  auto A_blocking_dims = get_A_blocking_dims();
  auto A_plain_dims = get_A_plain_dims();
  if (A_blocking_dims.size() == 4) {
    cfg.M_block = A_blocking_dims[2];
    cfg.K_block = A_blocking_dims[3];
  } else {
    int M = A_plain_dims[0];
    int K = A_plain_dims[1];
    cfg.M_block = std::min(M, 64);
    cfg.K_block = std::min(K, 64);
  }
  auto B_blocking_dims = get_B_blocking_dims();
  auto B_plain_dims = get_B_plain_dims();
  if (B_blocking_dims.size() == 4) {
    assert(cfg.K_block == B_blocking_dims[2]);
    cfg.K_block = B_blocking_dims[2];
    cfg.N_block = B_blocking_dims[3];
  } else {
    assert(A_plain_dims[1] == B_plain_dims[0]);
    int N = B_plain_dims[1];
    int K = B_plain_dims[0];
    cfg.N_block = std::min(N, 64);
    cfg.K_block = std::min(K, 64);
  }

  cfg.num_tile_k = 1;
  return ret;
}

gen_matmul2d_t::gen_matmul2d_t(
  std::vector<logical_tensor_t> &&ins, std::vector<logical_tensor_t> &&outs)
  : parent(std::move(ins), std::move(outs)) {
  COMPILE_ASSERT(
    in_tensors_.size() == 2, "input logical tensor size should be two.");
  COMPILE_ASSERT(
    out_tensors_.size() == 1, "output logical tensor size should be one.");
}

float gen_matmul2d_t::get_gflop() const {
  const int plain_M = get_A_plain_dims()[0];
  const int plain_K = get_A_plain_dims()[1];
  const int plain_N = get_B_plain_dims()[1];
  return 2.f * plain_M * plain_N * plain_K / 1e9;
}

void gen_matmul2d_t::schedule_loops(context_ptr ctx,
  const matmul2d_config_t &config, stmt body,
  std::vector<for_loop> &fors) const {
  for_loop lm_c = fors.at(0), ln_c = fors.at(1), l_k_t = fors.at(2),
           ln_o_r = fors.at(3), ln_i_r = fors.at(4);
  // Schedule
  bool s8s8_compensation = ctx->machine_.cpu_flags_.fAVX512VNNI
    && in_tensors_[0].dtype_ == datatypes::s8
    && (!ctx->flags_.brgemm_use_amx_
      || (ctx->flags_.brgemm_use_amx_
        && (!ctx->machine_.cpu_flags_.fAVX512AMXINT8 || config.K_block < 4)));
  if (s8s8_compensation) ln_o_r->fuse(ln_i_r);
  auto lmn = lm_c->fuse(ln_c);
  if (config.num_tile_k == 1) {
    l_k_t->unroll(0, body);
    scope_flatten(body, -1);
  }
}

bool gen_matmul2d_t::is_valid_config(
  const context_ptr &ctx, const matmul2d_config_t &config) const {
  const int K_num_blocks
    = sc_data_format_t::get_blocking_shapes(get_A_plain_dims(),
      sc_data_format_t::MKmk(config.M_block, config.K_block))[1];
  auto K_num_tile = config.num_tile_k;
  if (K_num_tile > K_num_blocks || K_num_blocks % K_num_tile != 0) {
    return false;
  }
  return true;
}

bool gen_matmul2d_t::generate(context_ptr ctx, const matmul2d_config_t &config,
  fusion_manager *fusion, const std::vector<expr> &inputs,
  const std::vector<expr> &outputs, std::vector<for_loop> &loops) const {
  COMPILE_ASSERT(!in_tensors_[0].get_format().is_plain()
      && !in_tensors_[1].get_format().is_plain(),
    "matmul2d should input blocking format");

  // Init
  int M = 0, N = 0, K = 0;
  const sc_dims &A_blocking_dims = get_A_blocking_dims();
  const sc_dims &B_blocking_dims = get_B_blocking_dims();
  auto A_dtype = get_A_dtype(), B_dtype = get_B_dtype();
  auto C_dtype = get_C_dtype();
  COMPILE_ASSERT(
    get_A_plain_dims().size() == 2 && get_B_plain_dims().size() == 2,
    "Expect data_plain_dims and B_plain_dims are both 2, but got "
      << get_A_plain_dims().size() << " and " << get_B_plain_dims().size());
  const int plain_M = get_A_plain_dims()[0];
  const int plain_K = get_A_plain_dims()[1];
  const int plain_B_K = get_B_plain_dims()[0];
  const int plain_N = get_B_plain_dims()[1];
  // The blocking size from inputs, outputs
  int M_block = config.M_block, K_block = config.K_block,
      N_block = config.N_block;
  int M_num_blocks = 0, N_num_blocks = 0, K_num_blocks = 0, B_K_num_blocks = 0;

  int last_M_block = 0, last_N_block = 0;
  int dtype_block = 1;
  if (B_dtype == datatypes::bf16) {
    dtype_block = 2;
  } else if (utils::is_one_of(B_dtype, datatypes::u8, datatypes::s8)) {
    dtype_block = 4;
  }
  std::tie(M_num_blocks, K_num_blocks)
    = std::tie(A_blocking_dims[0], A_blocking_dims[1]);
  std::tie(N_num_blocks, B_K_num_blocks)
    = std::tie(B_blocking_dims[0], B_blocking_dims[1]);
  COMPILE_ASSERT(plain_K == plain_B_K,
    "K in A and B are not match, got " << plain_K << " v.s. " << plain_B_K);
  COMPILE_ASSERT(
    K_num_blocks == B_K_num_blocks, "A and B num blocks of K are not equal.");
  if (dtype_block > 1) {
    COMPILE_ASSERT(
      B_blocking_dims[4] == -1 || B_blocking_dims[4] == dtype_block,
      "Wrong data format of B");
  }

  // TODO(xxx): return an empty func if K_num_tile value is invalid as tune
  // space builder doesn't support nested spliting for a given dimension
  // currently.
  auto K_num_tile = config.num_tile_k;
  if (K_num_tile > K_num_blocks || K_num_blocks % K_num_tile != 0) {
    return false;
  }
  auto K_tile = K_num_blocks / K_num_tile;

  sc_dims C_dims = sc_dims {M_num_blocks, N_num_blocks, M_block, N_block};
  // whether we need special compensation for microkernel.
  bool s8s8_compensation = ctx->machine_.cpu_flags_.fAVX512VNNI
    && A_dtype == datatypes::s8
    && (!ctx->flags_.brgemm_use_amx_
      || (ctx->flags_.brgemm_use_amx_
        && (!ctx->machine_.cpu_flags_.fAVX512AMXINT8 || config.K_block < 4)));
  for_loop lm_c, ln_c, l_k_t, ln_o_r, ln_i_r;

  expr C = outputs[op_params_t::out_C];
  expr A = inputs[op_params_t::in_A];
  expr B = inputs[op_params_t::in_B];
  expr compen0;
  if (s8s8_compensation) {
    _tensor_(compen_buf, C_dtype, {N_num_blocks, N_block});
    compen0 = compen_buf;
    builtin::brgemm_init(compen0, N_num_blocks, N_block, N_block, C_dtype, 0);
    _named_for_(ln_o_r, n_o, 0, N_num_blocks, 1, for_type::PARALLEL) {
      _named_for_(ln_i_r, n_i, 0, N_block) {
        _for_(ck_o, 0, B_K_num_blocks) {
          _for_(ck_i, 0, utils::divide_and_ceil(K_block, 4)) {
            _for_(ck_b, 0, 4) {
              compen0[{n_o, n_i}]
                = compen0[{n_o, n_i}] + B[{n_o, ck_o, ck_i, n_i, ck_b}];
            }
          }
        }
        compen0[{n_o, n_i}] = compen0[{n_o, n_i}] * 128;
      }
    }
  }

  _named_for_(l_k_t, k_o, 0, K_num_tile) {
    _named_for_(lm_c, m_o, 0, M_num_blocks, 1, for_type::PARALLEL) {
      _named_for_(ln_c, n_o, 0, N_num_blocks) {
        _if_(k_o == 0) {
          sc::builtin::brgemm_init_update(
            tensor_ptr(A, std::vector<expr> {m_o, 0, 0, 0}),
            tensor_ptr(B,
              dtype_block > 1 ? std::vector<expr> {n_o, 0, 0, 0, 0}
                              : std::vector<expr> {n_o, 0, 0, 0}),
            tensor_ptr(C, std::vector<expr> {m_o, n_o, 0, 0}), K_tile, M_block,
            N_block, K_block, K_block, N_block, N_block, M_block * K_block,
            (int)utils::divide_and_ceil(K_block, dtype_block) * dtype_block
              * N_block,
            A_dtype, B_dtype);
        }
        _else_ {
          sc::builtin::brgemm_update(tensor_ptr(A, {m_o, k_o * K_tile, 0, 0}),
            tensor_ptr(B,
              dtype_block > 1 ? std::vector<expr> {n_o, k_o * K_tile, 0, 0, 0}
                              : std::vector<expr> {n_o, k_o * K_tile, 0, 0}),
            tensor_ptr(C, std::vector<expr> {m_o, n_o, 0, 0}), K_tile, M_block,
            N_block, K_block, K_block, N_block, N_block, M_block * K_block,
            (int)utils::divide_and_ceil(K_block, dtype_block) * dtype_block
              * N_block,
            A_dtype, B_dtype);
        }

        // this is for s8s8 vnni compensation
        if (s8s8_compensation) {
          uint32_t lanes = 1;
          if (N_block / 16 && N_block % 16 == 0) {
            lanes
              = std::min(16U, ctx->get_max_vector_lanes(C_dtype.type_code_));
          }
          _if_(k_o == K_num_tile - 1) {
            _for_(m_i, 0, M_block) {
              _for_(n_i, 0, N_block, (int)lanes) {
                C[span_t({m_o, n_o, m_i, n_i}, lanes)]
                  = C[span_t({m_o, n_o, m_i, n_i}, lanes)]
                  - compen0[span_t({n_o, n_i}, lanes)];
              }
            }
          }
        }

        // this is the gemm output
        if (fusion) {
          fusion->create_output_fusion_anchor({tensor_slice(
            C, {{m_o, 1}, {n_o, 1}, {0, M_block}, {0, N_block}})});
        }
      }
      // this is the gemm output
      if (fusion
        && M_num_blocks >= runtime_config_t::get().threads_per_instance_) {
        fusion->create_output_fusion_anchor({tensor_slice(
          C, {{m_o, 1}, {0, N_num_blocks}, {0, M_block}, {0, N_block}})});
      }
    }
  }
  loops = {lm_c, ln_c, l_k_t, ln_o_r, ln_i_r};
  return true;
}
} // namespace ops
} // namespace sc
