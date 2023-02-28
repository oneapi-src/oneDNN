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

#include "conv_bwd.hpp"
#include <memory>
#include <utility>
#include <vector>
#include "utils.hpp"
#include <compiler/ir/builder.hpp>
#include <compiler/ir/builtin.hpp>
#include <compiler/ir/easy_build.hpp>
#include <compiler/ir/graph/fusion_mgr.hpp>
#include <util/any_map.hpp>
#include <util/reflection.hpp>
#include <util/utils.hpp>

using namespace dnnl::impl::graph::gc::builder;
namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

using ops::conv_bwd_data_config_t;
// clang-format off
SC_CLASS(conv_bwd_data_config_t)
  SC_FIELD(K_block)
  SC_FIELD(C_block)
  SC_FIELD(tile_d)
  SC_FIELD(tile_p)
  SC_FIELD(tile_q)
  SC_FIELD(loop_sched)
SC_CLASS_END();
// clang-format on

using ops::conv_bwd_weight_config_t;
// clang-format off
SC_CLASS(conv_bwd_weight_config_t)
  SC_FIELD(K_block)
  SC_FIELD(C_block)
  SC_FIELD(N_block)
  SC_FIELD(tile_p)
  SC_FIELD(tile_q)
  SC_FIELD(num_tile_n)
  SC_FIELD(loop_sched)
SC_CLASS_END();
// clang-format on

using ops::nested_conv_bwd_data_config_t;
// clang-format off
SC_CLASS(nested_conv_bwd_data_config_t)
  SC_FIELD(bs_threads)
  SC_FIELD(spatial_threads)
  SC_FIELD(ic_threads)
  SC_FIELD(bs_num_blocks)
  SC_FIELD(spatial_num_blocks)
  SC_FIELD(ic_num_blocks)
  SC_FIELD(oc_num_blocks)
SC_CLASS_END();
// clang-format on

using ops::nested_conv_bwd_weight_config_t;
// clang-format off
SC_CLASS(nested_conv_bwd_weight_config_t)
  SC_FIELD(oc_threads)
  SC_FIELD(ic_threads)
  SC_FIELD(bs_threads)
  SC_FIELD(oh_threads)
  SC_FIELD(od_threads)
  SC_FIELD(oc_num_blocks)
  SC_FIELD(ic_num_blocks)
  SC_FIELD(bs_num_blocks)
  SC_FIELD(oh_num_blocks)
  SC_FIELD(od_num_blocks)
  SC_FIELD(ow_num_blocks)
SC_CLASS_END();
// clang-format on

namespace ops {

config_ptr gen_conv_bwd_t::get_default_config(context_ptr ctx) const {
  auto ret = reflection::general_object_t::make<conv_bwd_data_config_t>();
  conv_bwd_data_config_t &cfg = *ret.unchecked_get_as<conv_bwd_data_config_t>();
  const auto weight_dim = get_weight_dims();
  if (weight_dim[0] % 32 == 0) {
    cfg.K_block = 32;
  } else {
    cfg.K_block = weight_dim[0];
  }
  if (weight_dim[1] % 32 == 0) {
    cfg.C_block = 32;
  } else {
    cfg.C_block = weight_dim[1];
  }
  cfg.tile_p = 1;
  cfg.tile_q = get_output_dims()[3];
  cfg.loop_sched = 1;
  return std::move(ret);
}

gen_conv_bwd_t::gen_conv_bwd_t(sc_op *owner, const sc_dims &stride,
  const sc_dims &padding, std::vector<logical_tensor_t> &&ins,
  std::vector<logical_tensor_t> &&outs)
  : parent(owner, std::move(ins), std::move(outs))
  , stride_(stride)
  , padding_(padding) {
  COMPILE_ASSERT(
    in_tensors_.size() == 2, "input logical tensor size should be two.");
  COMPILE_ASSERT(
    out_tensors_.size() == 1, "output logical tensor size should be two.");
}

float gen_conv_bwd_t::get_gflop() const {
  float result = 0.0;
  /* implement */
  result = (float)get_input_dims()[0]
    * (2.0 * (float)get_weight_dims()[0] * (float)get_weight_dims()[2]
        * (float)get_weight_dims()[3]
      - 1)
    * (float)get_output_dims()[2] * (float)get_output_dims()[3]
    * (float)get_weight_dims()[1] / (float)1e9;
  return result;
}

void gen_conv_bwd_t::schedule_loops(context_ptr ctx,
  const conv_bwd_data_config_t &config, stmt body,
  std::vector<for_loop> &fors) const {
  for_loop ln = fors.at(0), lc = fors.at(1), lp = fors.at(2);
  auto loop_sched = config.loop_sched;
  if (loop_sched == 1) {
    auto ln_c = ln->fuse(lc);
    auto ln_c_p = ln_c->fuse(lp);
  }
}

bool gen_conv_bwd_t::generate(context_ptr ctx,
  const conv_bwd_data_config_t &config, fusion_manager *fusion,
  const std::vector<expr> &inputs, const std::vector<expr> &outputs,
  std::vector<for_loop> &loops) const {
  // Init

  int padding_h = padding_[0], padding_w = padding_[0];
  if (padding_.size() == 2) { padding_w = padding_[1]; }
  int stride_h = stride_[0], stride_w = stride_[0];
  if (stride_.size() == 2) { stride_w = stride_[1]; }

  int N = get_input_dims()[0];
  int K = get_weight_dims()[0], C = get_weight_dims()[1],
      P = get_input_dims()[2], Q = get_input_dims()[3];
  int R = get_weight_dims()[2], S = get_weight_dims()[3],
      H = get_output_dims()[2] + 2 * padding_h,
      W = get_output_dims()[3] + 2 * padding_w;
  int K_block = config.K_block, C_block = config.C_block,
      tile_p = config.tile_p, tile_q = config.tile_q;
  int K_num_block = K / K_block, C_num_block = C / C_block;
  bool loop_sched = config.loop_sched;
  auto dtype = get_dtype();
  assert((K % K_block == 0) && "todo: padding K_block");
  assert((C % C_block == 0) && "todo: padding C_block");
  assert((P % tile_p == 0) && "wrong tile p");
  assert((Q % tile_q == 0) && "wrong tile q");
  // assert((get_input_dims()[1] == C) && "wrong input and weight");

  // define input, weight and output block
  sc_dims input_dims = {N, C / C_block, H, W, C_block};
  sc_dims weight_dims = {K / K_block, C / C_block, R, S, C_block, K_block};
  sc_dims output_dims = {N, K / K_block, P, Q, K_block};

  // define compute
  for_loop ln, lc, lp;
  expr del_input = outputs.at(op_params_t::out_del_input),
       output = inputs.at(op_params_t::in_fwd_output),
       weight = inputs.at(op_params_t::in_weight);
  {
    _tensor_(
      tr_weight, dtype, {C_num_block, K_num_block, R, S, K_block, C_block});
    _for_(c_o, 0, C_num_block) {
      _for_(k_o, 0, K_num_block) {
        _for_(r, 0, R) {
          _for_(s, 0, S) {
            _for_(k_i, 0, K_block) {
              _for_(c_i, 0, C_block) {
                tr_weight[{c_o, k_o, R - 1 - r, S - 1 - s, k_i, c_i}]
                  = weight[{k_o, c_o, r, s, c_i, k_i}];
              }
            }
          }
        }
      }
    }
    if (R == 1 && S == 1) {
      assert(padding_h == 0 && padding_w == 0 && "1*1 conv has no padding");
      if (stride_h == 1 && stride_w == 1) {
        _named_for_(ln, n, 0, N, 1, for_type::PARALLEL) {
          _named_for_(lc, c_o, 0, C_num_block) {
            _named_for_(lp, p_o, 0, P / tile_p) {
              builtin::brgemm_init_update(
                tensor_ptr(output, {n, 0, p_o * tile_p, 0, 0}),
                tensor_ptr(tr_weight, {c_o, 0, 0, 0, 0, 0}),
                tensor_ptr(del_input, {n, c_o, (p_o * tile_p), 0, 0}),
                K_num_block, tile_p * Q, C_block, K_block, K_block, C_block,
                C_block, P * Q * K_block, K_block * C_block, dtype, dtype);
            }
          }
        }
      } else {
        _named_for_(ln, n, 0, N, 1, for_type::PARALLEL) {
          _named_for_(lc, c_o, 0, C_num_block) {
            _named_for_(lp, p_o, 0, P / tile_p) {
              _for_(q_o, 0, Q / tile_q) {
                _for_(p_i, 0, tile_p) {
                  builtin::brgemm_init_update(
                    tensor_ptr(
                      output, {n, 0, p_o * tile_p + p_i, q_o * tile_q, 0}),
                    tensor_ptr(tr_weight, {c_o, 0, 0, 0, 0, 0}),
                    tensor_ptr(del_input,
                      {n, c_o, (p_o * tile_p + p_i) * stride_h,
                        q_o * tile_q * stride_w, 0}),
                    K_num_block, tile_q, C_block, K_block, K_block, C_block,
                    C_block * stride_w, P * Q * K_block, K_block * C_block,
                    dtype, dtype);
                }
              }
            }
          }
        }
      }
    } else {
      _named_for_(ln, n, 0, N, 1, for_type::PARALLEL) {
        _named_for_(lc, c_o, 0, C_num_block) {
          _named_for_(lp, p_o, 0, P / tile_p) {
            builtin::brgemm_init(
              tensor_ptr(del_input, {n, c_o, p_o * tile_p * stride_h, 0, 0}),
              (tile_p * stride_w + R - 1) * (Q * stride_w + S - 1), C_block,
              C_block, dtype, expr(0));
          }
        }
      }
      _named_for_(ln, n, 0, N, 1, for_type::PARALLEL) {
        _named_for_(lc, c_o, 0, C_num_block) {
          _named_for_(lp, p_o, 0, P / tile_p) {
            _for_(q_o, 0, Q / tile_q) {
              _for_(p_i, 0, tile_p) {
                _for_(r, 0, R) {
                  _for_(s, 0, S) {
                    builtin::brgemm_update(
                      tensor_ptr(
                        output, {n, 0, p_o * tile_p + p_i, q_o * tile_q, 0}),
                      tensor_ptr(tr_weight, {c_o, 0, r, s, 0, 0}),
                      tensor_ptr(del_input,
                        {n, c_o, (p_o * tile_p + p_i) * stride_h + (R - r - 1),
                          (q_o * tile_q) * stride_w + (S - s - 1), 0}),
                      K_num_block, tile_q, C_block, K_block, K_block, C_block,
                      C_block * stride_w, P * Q * K_block,
                      R * S * K_block * C_block, dtype, dtype);
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  loops = {ln, lc, lp};
  return true;
}

} // namespace ops
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
