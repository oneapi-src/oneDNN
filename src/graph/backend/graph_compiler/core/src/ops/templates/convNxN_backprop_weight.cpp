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

#include "convNxN_backprop_weight.hpp"
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

#include <algorithm>
#include <string>

using namespace dnnl::impl::graph::gc::builder;
namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace ops {
static int calculate_q_reduce(int Q, bool is_vnni_low_fp) {
  int Q_padded = Q % 2 == 0 ? Q : Q + 1;
  int Q_reduce = is_vnni_low_fp ? Q_padded : Q;
  return Q_reduce;
}

config_ptr gen_convNxN_backprop_weight::get_default_config(
  context_ptr ctx) const {
  auto ret = reflection::general_object_t::make<conv_bwd_weight_config_t>();
  conv_bwd_weight_config_t &cfg
    = *ret.unchecked_get_as<conv_bwd_weight_config_t>();
  sc_dim ndims = get_grad_dims().size();
  int C = static_cast<int>(get_data_dims()[1]);
  int K = static_cast<int>(get_grad_dims()[1]);
  int N = static_cast<int>(get_data_dims()[0]);
  int P = static_cast<int>(get_grad_dims()[ndims - 2]);
  int Q = static_cast<int>(get_grad_dims()[ndims - 1]);
  bool is_vnni_low_fp = ops::is_vnni_low_fp(ctx, get_dtype());
  int Q_reduce = calculate_q_reduce(Q, is_vnni_low_fp);

  // while K and C are small, use N_block = 16; K_block = K; C_block = C
  bool large_spatial = (P >= 56 && Q >= 56);
  if (large_spatial && N % 16 == 0) {
    cfg.N_block = 16;
  } else if (N % 32 == 0) {
    cfg.N_block = 32;
  } else {
    cfg.N_block = N;
  }

  if (K % 64 == 0 && !large_spatial) {
    cfg.K_block = 64;
  } else {
    cfg.K_block = K;
  }
  if (C % 64 == 0 && !large_spatial) {
    cfg.C_block = 64;
  } else {
    cfg.C_block = C;
  }
  cfg.tile_p = 1;
  if (type_ == REDUCE_W) {
    cfg.tile_q = Q_reduce;
  } else {
    cfg.tile_q = Q;
  }
  cfg.loop_sched = 1;
  return std::move(ret);
}

gen_convNxN_backprop_weight::gen_convNxN_backprop_weight(sc_op *owner,
  const sc_dims &stride, const sc_dims &padding,
  std::vector<logical_tensor_t> &&ins, std::vector<logical_tensor_t> &&outs,
  generator_type_t type)
  : parent(owner, std::move(ins), std::move(outs))
  , stride_(stride)
  , padding_(padding)
  , type_(type) {
  COMPILE_ASSERT(
    in_tensors_.size() == 2, "input logical tensor size should be two.");
  COMPILE_ASSERT(
    out_tensors_.size() == 1, "output logical tensor size should be two.");
  ndims_ = get_data_dims().size();
  const bool is_3d = (ndims_ == 5);
  COMPILE_ASSERT(
    !is_3d, "conv_bwd_weight NxN kernel does not support 3D conv.");
}

float gen_convNxN_backprop_weight::get_gflop() const {
  const int D = 1;
  const int P = get_grad_dims()[2];
  const int Q = get_grad_dims()[3];
  const int C = get_data_dims()[1];
  const int K = get_grad_dims()[1];
  const int N = get_data_dims()[0];
  const int KD = 1;
  const int KH = get_output_dims()[2];
  const int KW = get_output_dims()[3];
  float result = 2.0f * N * K * C * KD * KH * KW * D * P * Q / (float)1e9;
  return result;
}

void gen_convNxN_backprop_weight::schedule_loops(context_ptr ctx,
  const conv_bwd_weight_config_t &config, stmt body,
  std::vector<for_loop> &fors) const {
  COMPILE_ASSERT(
    type_ != generator_type_t::UNDEF, "Generator shall have an explicit type.");
  if (fors.empty()) { return; }
  auto loop_sched = config.loop_sched;
  if (type_ == generator_type_t::REDUCE_N) {
    COMPILE_ASSERT(fors.size() == 9 || fors.size() == 5,
      "number of for_loops not satisfying reduce N condition.");
    if (fors.size() == 9) {
      for_loop ln = fors.at(0), lk = fors.at(1), lc = fors.at(2),
               lr = fors.at(3), ls = fors.at(4);
      for_loop rlko = fors.at(5), rlco = fors.at(6), rlr = fors.at(7),
               rls = fors.at(8);
      if (loop_sched == 1) {
        // brgemm part
        auto ln_k = ln->fuse(lk);
        auto ln_k_c = ln_k->fuse(lc);
        auto ln_k_c_r = ln_k_c->fuse(lr);
        auto ln_k_c_r_s = ln_k_c_r->fuse(ls);
        // reduce add part
        auto rlk_c = rlko->fuse(rlco);
        auto rlk_c_r = rlk_c->fuse(rlr);
        auto rlk_c_r_s = rlk_c_r->fuse(rls);
      }
    } else {
      for_loop ln = fors.at(0), lk = fors.at(1), lc = fors.at(2),
               lr = fors.at(3), ls = fors.at(4);
      if (loop_sched == 1) {
        // brgemm part
        auto ln_k = ln->fuse(lk);
        auto ln_k_c = ln_k->fuse(lc);
        auto ln_k_c_r = ln_k_c->fuse(lr);
        auto ln_k_c_r_s = ln_k_c_r->fuse(ls);
      }
    }
  } else {
    for_loop ln = fors.at(0), lk = fors.at(1), lc = fors.at(2);
    if (loop_sched == 1) {
      auto ln_k = ln->fuse(lk);
      auto ln_k_c = ln_k->fuse(lc);
    }
  }
}

bool gen_convNxN_backprop_weight::generate(context_ptr ctx,
  const conv_bwd_weight_config_t &config, fusion_manager *fusion,
  const std::vector<expr> &inputs, const std::vector<expr> &outputs,
  std::vector<for_loop> &loops) const {
  COMPILE_ASSERT(
    type_ != generator_type_t::UNDEF, "Generator shall have an explicit type.");
  if (type_ == generator_type_t::REDUCE_N) {
    return generate_reduce_N(ctx, config, fusion, inputs, outputs, loops);
  } else {
    return generate_reduce_W(ctx, config, fusion, inputs, outputs, loops);
  }
  return true;
}

bool gen_convNxN_backprop_weight::generate_reduce_N(const context_ptr &ctx,
  const conv_bwd_weight_config_t &config, fusion_manager *fusion,
  const std::vector<expr> &inputs, const std::vector<expr> &outputs,
  std::vector<for_loop> &loops) const {
  // Init

  int padding_h = padding_[0], padding_w = padding_[0];
  if (padding_.size() == 2) { padding_w = padding_[1]; }
  int stride_h = stride_[0], stride_w = stride_[0];
  if (stride_.size() == 2) { stride_w = stride_[1]; }

  bool has_padding = (padding_h > 0 || padding_w > 0);
  bool has_stride = (stride_h > 1 || stride_w > 1);

  int N = get_data_dims()[0], C = get_data_dims()[1], H = get_data_dims()[2],
      W = get_data_dims()[3];
  int K = get_grad_dims()[1];
  int P = get_grad_dims()[2], Q = get_grad_dims()[3];
  int R = get_output_dims()[2], S = get_output_dims()[3];
  int K_block = config.K_block, C_block = config.C_block;
  int tile_n = config.N_block, tile_q = config.tile_q;
  int K_num_block = utils::divide_and_ceil(K, K_block),
      C_num_block = utils::divide_and_ceil(C, C_block);
  int N_num_block = utils::divide_and_ceil(N, tile_n);
  int Q_num_tile = utils::divide_and_ceil(Q, tile_q);

  COMPILE_ASSERT(Q % tile_q == 0, "Q should be divisible by tile_q.");
  bool loop_sched = config.loop_sched;
  auto dtype = get_dtype();
  bool is_vnni_low_fp = ops::is_vnni_low_fp(ctx, dtype);
  int dtype_block = is_vnni_low_fp ? 2 : 1;
  // calculate the correct p_o value for brgemm init
  std::vector<int> p_offset(R, 0);
  for (int r = 0; r < R; ++r) {
    for (int p = 0; p < P; ++p) {
      if (p * stride_h + r >= padding_h) {
        p_offset[r] = p;
        break;
      }
    }
  }

  // define compute
  for_loop ln, lc, lk, lr, ls, lp;
  for_loop rlco, rlko, rlr, rls, rlk;
  expr del_weight = outputs.at(op_params_t::out_del_weight),
       data = inputs.at(op_params_t::in_data),
       output = inputs.at(op_params_t::in_fwd_output);
  auto filter_dims
    = out_tensors_[op_params_t::out_del_weight].get_blocking_dims();
  std::vector<expr> del_weight_tmp_buf_shape;
  del_weight_tmp_buf_shape.reserve(filter_dims.size());
  for (auto dim : filter_dims) {
    del_weight_tmp_buf_shape.emplace_back(dim2unsigned(dim));
  }
  del_weight_tmp_buf_shape[0] = del_weight_tmp_buf_shape[0] * N_num_block;
  {
    _tensor_(del_weight_tmp_buf, datatypes::f32, del_weight_tmp_buf_shape);
    _named_for_(ln, n_o, 0, N_num_block, 1, for_type::PARALLEL) {
      _named_for_(lk, k_o, 0, K_num_block) {
        _named_for_(lc, c_o, 0, C_num_block) {
          // p_o here shall not be merged; set as temp.no_loop_fuse in the end
          _named_for_(lp, p_o, 0, P) {
            _if_(p_o * stride_h + R > padding_h
              && p_o * stride_h < H + padding_h) {
              _for_(q_o, 0, Q_num_tile) {
                _named_for_(lr, r, 0, R) {
                  _if_(p_o * stride_h + r >= padding_h
                    && p_o * stride_h + r < H + padding_h) {
                    _named_for_(ls, s, 0, S) {
                      // q_o * tile_q * stride_w + s - padding_w
                      // (q_o * tile_q + tile_q - 1) * stride_w + s - padding_w
                      _var_(q_start, datatypes::s32);
                      _var_(q_end, datatypes::s32);
                      _var_(q_start_ofb_cnt, datatypes::s32); // out of bound
                      _var_(q_end_ofb_cnt, datatypes::s32); // out of bound
                      q_start = builder::make_cast(datatypes::s32, q_o) * tile_q
                          * stride_w
                        + builder::make_cast(datatypes::s32, s) - padding_w;
                      q_end = (builder::make_cast(datatypes::s32, q_o) * tile_q
                                + tile_q - 1)
                          * stride_w
                        + builder::make_cast(datatypes::s32, s) - padding_w;
                      // divide and ceil (-q_start, stride_w)
                      q_start_ofb_cnt = (0 - q_start + stride_w - 1) / stride_w;
                      // divide and ceil (q_end - (W - 1), stride_w)
                      q_end_ofb_cnt
                        = (q_end - (W - 1) + stride_w - 1) / stride_w;
                      q_start_ofb_cnt = builder::make_max(0, q_start_ofb_cnt);
                      q_end_ofb_cnt = builder::make_max(0, q_end_ofb_cnt);
                      _var_(q_start_valid, datatypes::s32);
                      q_start_valid = q_start + stride_w * q_start_ofb_cnt;
                      _var_(Q_batch_size, datatypes::s32);
                      Q_batch_size = tile_q - q_start_ofb_cnt - q_end_ofb_cnt;
                      _var_(output_q_start, datatypes::s32);
                      output_q_start = q_start_ofb_cnt
                        + builder::make_cast(datatypes::s32, q_o) * tile_q;
                      // TODO(yifei): double check the condition here
                      // to deal with padding case more carefully
                      trace_guard_t trg(ctx, "brgemm");
                      _if_(p_o * stride_h + r >= padding_h
                        && (p_o == 0 || (p_o - 1) * stride_h + r < padding_h)
                        // p_o == p_offset[r]  // cannot write like this
                        && q_o == 0) {
                        builtin::brgemm_init_update(
                          tensor_ptr(
                            output, {n_o, k_o, p_o, output_q_start, 0, 0}),
                          tensor_ptr(data,
                            dtype_block > 1 ? std::vector<expr> {n_o, c_o,
                              p_o * stride_h + r - padding_h, q_start_valid, 0,
                              0, 0}
                                            : std::vector<expr> {n_o, c_o,
                                              p_o * stride_h + r - padding_h,
                                              q_start_valid, 0, 0}),
                          tensor_ptr(
                            N_num_block > 1 ? del_weight_tmp_buf : del_weight,
                            {n_o * K_num_block + k_o, c_o, r, s, 0, 0}),
                          Q_batch_size, K_block, C_block, tile_n, tile_n,
                          C_block, C_block, K_block * tile_n,
                          stride_w * C_block
                            * static_cast<int>(
                              utils::divide_and_ceil(tile_n, dtype_block))
                            * dtype_block,
                          dtype, dtype);
                      }
                      _else_ {
                        builtin::brgemm_update(
                          tensor_ptr(
                            output, {n_o, k_o, p_o, output_q_start, 0, 0}),
                          tensor_ptr(data,
                            dtype_block > 1 ? std::vector<expr> {n_o, c_o,
                              p_o * stride_h + r - padding_h, q_start_valid, 0,
                              0, 0}
                                            : std::vector<expr> {n_o, c_o,
                                              p_o * stride_h + r - padding_h,
                                              q_start_valid, 0, 0}),
                          tensor_ptr(
                            N_num_block > 1 ? del_weight_tmp_buf : del_weight,
                            {n_o * K_num_block + k_o, c_o, r, s, 0, 0}),
                          Q_batch_size, K_block, C_block, tile_n, tile_n,
                          C_block, C_block, K_block * tile_n,
                          stride_w * C_block
                            * static_cast<int>(
                              utils::divide_and_ceil(tile_n, dtype_block))
                            * dtype_block,
                          dtype, dtype);
                      }
                    }
                  }
                }
              }
            }
          }
          if (N_num_block == 1 && fusion) {
            fusion->create_output_fusion_anchor({tensor_slice(del_weight,
              {{k_o, 1}, {c_o, 1}, {0, R}, {0, S}, {0, K_block},
                {0, C_block}})});
          }
        }
      }
    }
    if (N_num_block > 1) {
      int lanes = 1;
      if (C_block / 16 && C_block % 16 == 0) {
        lanes = vectorize_step(ctx, out_tensors_[0].dtype_.type_code_, 16);
      }
      // KC(D)RSkc
      trace_guard_t trg(ctx, "final_reduce");
      _named_for_(rlko, l_k_o, 0, K_num_block, 1, for_type::PARALLEL) {
        _named_for_(rlco, l_c_o, 0, C_num_block, 1) {
          _named_for_(rlr, l_r, 0, R, 1) {
            _named_for_(rls, l_s, 0, S, 1) {
              builtin::mem_zero(
                tensor_ptr(del_weight, {l_k_o, l_c_o, l_r, l_s, 0, 0}),
                C_block * K_block, datatypes::f32);
              _named_for_(rlk, l_k, 0, K_block, 1) {
                _for_(l_c, 0, C_block, lanes) {
                  _for_(l_n, 0, N_num_block, 1) {
                    del_weight[span_t(
                      {l_k_o, l_c_o, l_r, l_s, l_k, l_c}, lanes)]
                      = builder::make_add(
                        del_weight[span_t(
                          {l_k_o, l_c_o, l_r, l_s, l_k, l_c}, lanes)],
                        del_weight_tmp_buf[span_t({l_n * K_num_block + l_k_o,
                                                    l_c_o, l_r, l_s, l_k, l_c},
                          lanes)]);
                  }
                }
              }
              if (fusion) {
                fusion->create_output_fusion_anchor({tensor_slice(del_weight,
                  {{l_k_o, 1}, {l_c_o, 1}, {l_r, 1}, {l_s, 1}, {0, K_block},
                    {0, C_block}})});
              }
            }
          }
        }
      }
      loops = {ln, lk, lc, lr, ls, rlko, rlco, rlr, rls};
    } else {
      loops = {ln, lk, lc, lr, ls};
    }
  }
  lp->attr().set(stmt_attr_key::no_loop_fuse, true);
  return true;
}

bool gen_convNxN_backprop_weight::generate_reduce_W(const context_ptr &ctx,
  const conv_bwd_weight_config_t &config, fusion_manager *fusion,
  const std::vector<expr> &inputs, const std::vector<expr> &outputs,
  std::vector<for_loop> &loops) const {
  // Init
  int padding_h = padding_[0], padding_w = padding_[0];
  if (padding_.size() == 2) { padding_w = padding_[1]; }
  int stride_h = stride_[0], stride_w = stride_[0];
  if (stride_.size() == 2) { stride_w = stride_[1]; }

  bool has_padding = (padding_h > 0 || padding_w > 0);
  bool has_stride = (stride_h > 1 || stride_w > 1);

  int N = get_data_dims()[0], C = get_data_dims()[1], H = get_data_dims()[2],
      W = get_data_dims()[3];
  int K = get_grad_dims()[1];
  int P = get_grad_dims()[2], Q = get_grad_dims()[3];
  int R = get_output_dims()[3], S = get_output_dims()[4];
  int K_block = config.K_block, C_block = config.C_block,
      tile_p = config.tile_p, tile_q = config.tile_q;
  int K_num_block = utils::divide_and_ceil(K, K_block),
      C_num_block = utils::divide_and_ceil(C, C_block);
  bool loop_sched = config.loop_sched;
  auto dtype = get_dtype();
  bool is_vnni_low_fp = ops::is_vnni_low_fp(ctx, dtype);
  int dtype_block = is_vnni_low_fp ? 2 : 1;
  int Q_reduce = calculate_q_reduce(Q, is_vnni_low_fp);
  int Q_block = Q_reduce / tile_q;

  COMPILE_ASSERT(P % tile_p == 0, "P should be divisible by tile_p.");
  COMPILE_ASSERT(Q_reduce % tile_q == 0, "Q should be divisible by tile_q.");

  sc_dims input_blocking = in_tensors_[0].get_blocking_dims();
  sc_dims dst_blocking = in_tensors_[1].get_blocking_dims();
  sc_dims weight_blocking = out_tensors_[0].get_blocking_dims();

  int data_tmp_inner = tile_q + (S - 1) / stride_w;
  int data_tmp_outer = tile_p + (R - 1) / stride_h;
  data_tmp_outer *= stride_h;

  // define compute
  for_loop ln, lc, lk;
  expr del_weight = outputs.at(op_params_t::out_del_weight),
       data = inputs.at(op_params_t::in_data),
       output = inputs.at(op_params_t::in_fwd_output);
  {
    // computation performed on on plain format
    // weight: KCRSck
    // data: NHWC --> pack: C_block x W
    // output_delta: NHWK --> pack: W x K_block
    if (!has_padding) {
      _named_for_(ln, n, 0, N, 1, for_type::PARALLEL) {
        _named_for_(lk, k_o, 0, K_num_block) {
          _named_for_(lc, c_o, 0, C_num_block) {
            _for_(p_o, 0, P / tile_p) {
              _for_(q_o, 0, Q_reduce / tile_q) {
                _tensor_(output_tmp, dtype,
                  dtype_block > 1
                    ? std::vector<expr> {tile_p, tile_q / 2, K_block, 2}
                    : std::vector<expr> {tile_p, tile_q, K_block});
                _for_(p_i, 0, tile_p) {
                  _for_(q_i, 0, tile_q) {
                    // q_idx = q_o * tile_q + q_i
                    _for_(k_i, 0, K_block) {
                      if (dtype_block > 1) {
                        _if_(q_o * tile_q + q_i >= Q) {
                          output_tmp[{p_i, q_i / 2, k_i, q_i % 2}]
                            = builder::make_constant({0.0f}, dtype);
                        }
                        _else_ {
                          output_tmp[{p_i, q_i / 2, k_i, q_i % 2}]
                            = output[{n, p_o * tile_p + p_i, q_o * tile_q + q_i,
                              k_o * K_block + k_i}];
                        }
                      } else {
                        output_tmp[{p_i, q_i, k_i}]
                          = output[{n, p_o * tile_p + p_i, q_o * tile_q + q_i,
                            k_o * K_block + k_i}];
                      }
                    }
                  }
                }
                _tensor_(data_tmp, dtype,
                  {data_tmp_outer, C_block, stride_w, data_tmp_inner});
                _for_(p_i, 0, data_tmp_outer) {
                  _for_(q_i, 0, data_tmp_inner) {
                    _for_(sw, 0, stride_w) {
                      _for_(c_i, 0, C_block) {
                        _if_((q_o * tile_q + q_i) * stride_w + sw >= W) {
                          data_tmp[{p_i, c_i, sw, q_i}]
                            = builder::make_constant({0.0f}, dtype);
                        }
                        _else_ {
                          data_tmp[{p_i, c_i, sw, q_i}]
                            = data[{n, p_o * tile_p * stride_h + p_i,
                              (q_o * tile_q + q_i) * stride_w + sw,
                              c_o * C_block + c_i}];
                        }
                      }
                    }
                  }
                }
                _for_(r, 0, R) {
                  _for_(sw, 0, stride_w) {
                    // non const loop range
                    _for_(s, sw, S, stride_w) { // var, begin, end, step
                      _if_(n == 0 && p_o == 0 && q_o == 0) {
                        builtin::brgemm_init_update(
                          tensor_ptr(data_tmp, {r, 0, sw, s / stride_w}),
                          tensor_ptr(output_tmp,
                            dtype_block > 1 ? std::vector<expr> {0, 0, 0, 0}
                                            : std::vector<expr> {0, 0, 0}),
                          tensor_ptr(del_weight, {n, k_o, c_o, r, s, 0, 0}),
                          tile_p, C_block, K_block, tile_q,
                          data_tmp_inner * stride_w, K_block, K_block,
                          C_block * data_tmp_inner * stride_w * stride_h,
                          tile_q * K_block, dtype, dtype);
                      }
                      _else_ {
                        builtin::brgemm_update(
                          tensor_ptr(data_tmp, {r, 0, sw, s / stride_w}),
                          tensor_ptr(output_tmp,
                            dtype_block > 1 ? std::vector<expr> {0, 0, 0, 0}
                                            : std::vector<expr> {0, 0, 0}),
                          tensor_ptr(del_weight, {n, k_o, c_o, r, s, 0, 0}),
                          tile_p, C_block, K_block, tile_q,
                          data_tmp_inner * stride_w, K_block, K_block,
                          C_block * data_tmp_inner * stride_w * stride_h,
                          tile_q * K_block, dtype, dtype);
                      }
                      if (fusion) {
                        fusion->create_output_fusion_anchor(
                          {tensor_slice(del_weight,
                            {{n, 1}, {k_o, 1}, {c_o, 1}, {r, 1}, {s, 1},
                              {0, C_block}, {0, K_block}})});
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
    loops = {ln, lk, lc};
  }
  return true;
}

} // namespace ops
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
