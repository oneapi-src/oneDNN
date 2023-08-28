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

#include "convNxN_backprop_data.hpp"
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
namespace ops {

config_ptr gen_convNxN_backprop_data::get_default_config(
  context_ptr ctx) const {
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
  cfg.tile_d = 1;
  cfg.tile_p = 1;
  cfg.tile_q = get_input_dims()[3];
  cfg.loop_sched = 1;
  return std::move(ret);
}

gen_convNxN_backprop_data::gen_convNxN_backprop_data(sc_op *owner,
  const sc_dims &stride, const sc_dims &padding,
  std::vector<logical_tensor_t> &&ins, std::vector<logical_tensor_t> &&outs)
  : parent(owner, std::move(ins), std::move(outs))
  , stride_(stride)
  , padding_(padding) {
  COMPILE_ASSERT(
    in_tensors_.size() == 2, "input logical tensor size should be two.");
  COMPILE_ASSERT(
    out_tensors_.size() == 1, "output logical tensor size should be two.");
  ndims_ = get_input_dims().size();
  const bool is_3d = (ndims_ == 5);
  COMPILE_ASSERT(!is_3d, "conv_bwd_data NxN kernel does not support 3D conv.");
}

float gen_convNxN_backprop_data::get_gflop() const {
  const int D = 1;
  const int H = get_input_dims()[2];
  const int W = get_input_dims()[3];
  const int C = get_input_dims()[1];
  const int K = get_output_dims()[1];
  const int N = get_input_dims()[0];
  const int KD = 1;
  const int KH = get_weight_dims()[2];
  const int KW = get_weight_dims()[3];
  float result = 2.0f * N * K * C * KD * KH * KW * D * H * W / (float)1e9;
  return result;
}

void gen_convNxN_backprop_data::schedule_loops(context_ptr ctx,
  const conv_bwd_data_config_t &config, stmt body,
  std::vector<for_loop> &fors) const {
  for_loop ln = fors.at(0), lc = fors.at(1);
  auto loop_sched = config.loop_sched;
  if (loop_sched == 1) { auto ln_c = ln->fuse(lc); }
}

bool gen_convNxN_backprop_data::generate(context_ptr ctx,
  const conv_bwd_data_config_t &config, fusion_manager *fusion,
  const std::vector<expr> &inputs, const std::vector<expr> &outputs,
  std::vector<for_loop> &loops) const {
  // initialize paddings and strides on fwd_input
  int padding_h = padding_[0], padding_w = padding_[0];
  if (padding_.size() > 1) { padding_w = padding_[1]; }
  int stride_h = stride_[0], stride_w = stride_[0];
  if (stride_.size() > 1) { stride_w = stride_[1]; }

  bool has_padding = (padding_h > 0) && (padding_w > 0);
  bool has_stride = (stride_h > 1) && (stride_w > 1);

  int N = get_input_dims()[0];
  // P, Q is the height, width of grad input
  int K = get_weight_dims()[0], C = get_weight_dims()[1],
      P = get_input_dims()[2], Q = get_input_dims()[3];
  int R = get_weight_dims()[2], S = get_weight_dims()[3],
      H = get_output_dims()[2], W = get_output_dims()[3];
  int K_block = config.K_block, C_block = config.C_block,
      tile_p = config.tile_p, tile_q = config.tile_q;
  int K_num_block = utils::divide_and_ceil(K, K_block),
      C_num_block = utils::divide_and_ceil(C, C_block);
  bool loop_sched = config.loop_sched;
  auto dtype = get_dtype();
  bool is_vnni_low_fp = ops::is_vnni_low_fp(ctx, dtype);
  int dtype_block = is_vnni_low_fp ? 2 : 1;
  int padded_K_block
    = utils::divide_and_ceil(K_block, dtype_block) * dtype_block;
  COMPILE_ASSERT(
    R != 1 && S != 1, "gen_convNxN_backprop_data only supports R!=1 and S!=1");
  bool is_out_blocking = out_tensors_[0].get_format().is_blocking();

  // define compute
  for_loop ln, lc, lp;
  expr del_input = outputs.at(op_params_t::out_del_input),
       del_output = inputs.at(op_params_t::in_fwd_output),
       weight = inputs.at(op_params_t::in_weight);
  {
    if (!has_padding) {
      _named_for_(ln, n, 0, N, 1, for_type::PARALLEL) {
        _named_for_(lc, c_o, 0, C_num_block) {
          if (!is_out_blocking) {
            builtin::dnnl_brgemm_init(
              tensor_ptr(del_input, {n, 0, 0, c_o * C_block}), H * W, C_block,
              C, datatypes::f32, 0);
          } else {
            builtin::mem_zero(tensor_ptr(del_input, {n, c_o, 0, 0, 0}),
              H * W * C_block, datatypes::f32);
          }
          // lp cannot be parallelled due to potential race condition
          _named_for_(lp, p_o, 0, P / tile_p) {
            _for_(q_o, 0, Q / tile_q) {
              _for_(p_i, 0, tile_p) {
                _for_(r, 0, R) {
                  _for_(s, 0, S) {
                    std::vector<expr> del_input_index, del_output_index;
                    expr LDA, LDC, stride_a;
                    if (!is_out_blocking) {
                      del_output_index
                        = {n, p_o * tile_p + p_i, q_o * tile_q, 0};
                      LDA = K;
                      stride_a = K_block;
                      del_input_index = {n, (p_o * tile_p + p_i) * stride_h + r,
                        (q_o * tile_q) * stride_w + s, c_o * C_block};
                      LDC = C * stride_w;
                      LDC->attr().set("N_axis", std::vector<size_t> {3});
                    } else {
                      del_output_index
                        = {n, 0, p_o * tile_p + p_i, q_o * tile_q, 0};
                      LDA = K_block;
                      stride_a = P * Q * K_block;
                      del_input_index
                        = {n, c_o, (p_o * tile_p + p_i) * stride_h + r,
                          (q_o * tile_q) * stride_w + s, 0};
                      LDC = C_block * stride_w;
                    }
                    LDC->attr().set("stride_w", stride_w);
                    builtin::brgemm_update(
                      tensor_ptr(del_output, del_output_index),
                      tensor_ptr(weight,
                        dtype_block > 1
                          ? std::vector<expr> {c_o, 0, r, s, 0, 0, 0}
                          : std::vector<expr> {c_o, 0, r, s, 0, 0}),
                      tensor_ptr(del_input, del_input_index), K_num_block,
                      tile_q, C_block, K_block, LDA, C_block, LDC, stride_a,
                      R * S * padded_K_block * C_block, dtype, dtype);
                  }
                }
              }
            }
          }
          if (fusion) {
            if (!is_out_blocking) {
              fusion->create_output_fusion_anchor({tensor_slice(del_input,
                {{n, 1}, {0, H}, {0, W}, {c_o * C_block, C_block}})});
            } else {
              fusion->create_output_fusion_anchor({tensor_slice(
                del_input, {{n, 1}, {c_o, 1}, {0, H}, {0, W}, {0, C_block}})});
            }
          }
        }
      }
    } else {
      // padding cases
      _named_for_(ln, n, 0, N, 1, for_type::PARALLEL) {
        _named_for_(lc, c_o, 0, C_num_block) {
          if (!is_out_blocking) {
            builtin::dnnl_brgemm_init(
              tensor_ptr(del_input, {n, 0, 0, c_o * C_block}), H * W, C_block,
              C, datatypes::f32, 0);
          } else {
            builtin::mem_zero(tensor_ptr(del_input, {n, c_o, 0, 0, 0}),
              H * W * C_block, datatypes::f32);
          }
          _named_for_(lp, p_o, 0, P / tile_p) {
            _for_(q_o, 0, Q / tile_q) {
              _for_(p_i, 0, tile_p) {
                _for_(r, 0, R) {
                  _for_(s, 0, S) {
                    _if_((p_o * tile_p + p_i) * stride_h + r >= padding_h
                      && (p_o * tile_p + p_i) * stride_h + r < H + padding_h) {
                      // blocking or non-blocking variables
                      std::vector<expr> del_input_index, del_output_index;
                      expr LDA, LDC, stride_a;
                      if (!is_out_blocking) {
                        LDA = K;
                        stride_a = K_block;
                        LDC = C * stride_w;
                        LDC->attr().set("N_axis", std::vector<size_t> {3});
                      } else {
                        LDA = K_block;
                        stride_a = P * Q * K_block;
                        LDC = C_block * stride_w;
                      }
                      LDC->attr().set("stride_w", stride_w);
                      _var_(valid_q_cnt, datatypes::s32);
                      // q_left and q_right are w.r.t index on del_input after
                      // padding
                      _var_(q_left, datatypes::index);
                      _var_(q_right, datatypes::index);
                      q_left = (q_o * tile_q) * stride_w + s;
                      q_right = ((q_o + 1) * tile_q - 1) * stride_w + s;
                      _if_(q_right >= padding_w && q_left < W + padding_w) {
                        _if_(q_left >= padding_w && q_right < W + padding_w) {
                          if (!is_out_blocking) {
                            del_output_index
                              = {n, p_o * tile_p + p_i, q_o * tile_q, 0};
                            del_input_index = {n,
                              (p_o * tile_p + p_i) * stride_h + r - padding_h,
                              (q_o * tile_q) * stride_w + s - padding_w,
                              c_o * C_block};
                          } else {
                            del_output_index
                              = {n, 0, p_o * tile_p + p_i, q_o * tile_q, 0};
                            del_input_index = {n, c_o,
                              (p_o * tile_p + p_i) * stride_h + r - padding_h,
                              (q_o * tile_q) * stride_w + s - padding_w, 0};
                          }
                          builtin::brgemm_update(
                            tensor_ptr(del_output, del_output_index),
                            tensor_ptr(weight,
                              dtype_block > 1
                                ? std::vector<expr> {c_o, 0, r, s, 0, 0, 0}
                                : std::vector<expr> {c_o, 0, r, s, 0, 0}),
                            tensor_ptr(del_input, del_input_index), K_num_block,
                            tile_q, C_block, K_block, LDA, C_block, LDC,
                            stride_a, R * S * padded_K_block * C_block, dtype,
                            dtype);
                        }
                        _else_ {
                          _if_(q_left >= padding_w) {
                            // same as valid_q_cnt = divide_and_ceil(W +
                            // padding_w - q_left, stride_w);
                            valid_q_cnt = builder::make_cast(datatypes::s32,
                              (W + padding_w - q_left + stride_w - 1)
                                / stride_w);
                            if (!is_out_blocking) {
                              del_output_index
                                = {n, p_o * tile_p + p_i, q_o * tile_q, 0};
                              del_input_index = {n,
                                (p_o * tile_p + p_i) * stride_h + r - padding_h,
                                q_left - padding_w, c_o * C_block};
                            } else {
                              del_output_index
                                = {n, 0, p_o * tile_p + p_i, q_o * tile_q, 0};
                              del_input_index = {n, c_o,
                                (p_o * tile_p + p_i) * stride_h + r - padding_h,
                                q_left - padding_w, 0};
                            }
                            builtin::brgemm_update(
                              tensor_ptr(del_output, del_output_index),
                              tensor_ptr(weight,
                                dtype_block > 1
                                  ? std::vector<expr> {c_o, 0, r, s, 0, 0, 0}
                                  : std::vector<expr> {c_o, 0, r, s, 0, 0}),
                              tensor_ptr(del_input, del_input_index),
                              K_num_block, valid_q_cnt, C_block, K_block, LDA,
                              C_block, LDC, stride_a,
                              R * S * padded_K_block * C_block, dtype, dtype);
                          }
                          _if_(q_right < W + padding_w) {
                            _var_(valid_q_cnt_idx, datatypes::index);
                            valid_q_cnt_idx
                              = (q_right - padding_w + 1 + stride_w - 1)
                              / stride_w;
                            valid_q_cnt = builder::make_cast(
                              datatypes::s32, valid_q_cnt_idx);
                            _var_(q_left_valid, datatypes::index);
                            q_left_valid
                              = q_right - (valid_q_cnt_idx - 1) * stride_w;
                            if (!is_out_blocking) {
                              del_output_index = {n, p_o * tile_p + p_i,
                                q_o * tile_q + tile_q - valid_q_cnt, 0};
                              del_input_index = {n,
                                (p_o * tile_p + p_i) * stride_h + r - padding_h,
                                q_left_valid - padding_w, c_o * C_block};
                            } else {
                              del_output_index = {n, 0, p_o * tile_p + p_i,
                                q_o * tile_q + tile_q - valid_q_cnt, 0};
                              del_input_index = {n, c_o,
                                (p_o * tile_p + p_i) * stride_h + r - padding_h,
                                q_left_valid - padding_w, 0};
                            }
                            builtin::brgemm_update(
                              tensor_ptr(del_output, del_output_index),
                              tensor_ptr(weight,
                                dtype_block > 1
                                  ? std::vector<expr> {c_o, 0, r, s, 0, 0, 0}
                                  : std::vector<expr> {c_o, 0, r, s, 0, 0}),
                              tensor_ptr(del_input, del_input_index),
                              K_num_block, valid_q_cnt, C_block, K_block, LDA,
                              C_block, LDC, stride_a,
                              R * S * padded_K_block * C_block, dtype, dtype);
                          }
                          _if_(q_left < padding_w && q_right >= W + padding_w) {
                            _var_(q_offset_left, datatypes::index);
                            _var_(q_offset_right, datatypes::index);
                            // DC(padding_w - q_left, stride_w)
                            q_offset_left
                              = (padding_w - q_left + stride_w - 1) / stride_w;
                            // DC(q_right - W - padding_w + 1, stride_w)
                            q_offset_right
                              = (q_right - W - padding_w + stride_w) / stride_w;
                            valid_q_cnt = builder::make_cast(datatypes::s32,
                              tile_q - q_offset_left - q_offset_right);
                            _var_(q_left_valid, datatypes::index);
                            q_left_valid = q_left + q_offset_left * stride_w;
                            if (!is_out_blocking) {
                              del_output_index = {n, p_o * tile_p + p_i,
                                q_o * tile_q + q_offset_left, 0};
                              del_input_index = {n,
                                (p_o * tile_p + p_i) * stride_h + r - padding_h,
                                q_left_valid - padding_w, c_o * C_block};
                            } else {
                              del_output_index = {n, 0, p_o * tile_p + p_i,
                                q_o * tile_q + q_offset_left, 0};
                              del_input_index = {n, c_o,
                                (p_o * tile_p + p_i) * stride_h + r - padding_h,
                                q_left_valid - padding_w, 0};
                            }
                            builtin::brgemm_update(
                              tensor_ptr(del_output, del_output_index),
                              tensor_ptr(weight,
                                dtype_block > 1
                                  ? std::vector<expr> {c_o, 0, r, s, 0, 0, 0}
                                  : std::vector<expr> {c_o, 0, r, s, 0, 0}),
                              tensor_ptr(del_input, del_input_index),
                              K_num_block, valid_q_cnt, C_block, K_block, LDA,
                              C_block, LDC, stride_a,
                              R * S * padded_K_block * C_block, dtype, dtype);
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
          }
          if (fusion) {
            if (!is_out_blocking) {
              fusion->create_output_fusion_anchor({tensor_slice(del_input,
                {{n, 1}, {0, H}, {0, W}, {c_o * C_block, C_block}})});
            } else {
              fusion->create_output_fusion_anchor({tensor_slice(
                del_input, {{n, 1}, {c_o, 1}, {0, H}, {0, W}, {0, C_block}})});
            }
          }
        }
      }
    }
  }
  loops = {ln, lc};
  lp->attr().set(stmt_attr_key::no_loop_fuse, true);
  return true;
}
} // namespace ops
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
