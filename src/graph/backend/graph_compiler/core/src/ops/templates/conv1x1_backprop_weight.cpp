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

#include "conv1x1_backprop_weight.hpp"
#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "utils.hpp"
#include <compiler/ir/builder.hpp>
#include <compiler/ir/builtin.hpp>
#include <compiler/ir/easy_build.hpp>
#include <compiler/ir/graph/fusion_mgr.hpp>
#include <compiler/ir/graph/graph.hpp>
#include <compiler/ir/transform/tensor_shrink.hpp>
#include <runtime/config.hpp>
#include <util/any_map.hpp>
#include <util/reflection.hpp>
#include <util/utils.hpp>
using namespace dnnl::impl::graph::gc::builder;
namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace ops {

config_ptr gen_conv1x1_backprop_weight_t::get_default_config(
  context_ptr ctx) const {
  if (type_ == generator_type_t::REDUCE_ALL) {
    auto ret = reflection::general_object_t::make<conv_bwd_weight_config_t>();
    conv_bwd_weight_config_t &cfg
      = *ret.unchecked_get_as<conv_bwd_weight_config_t>();
    int N = static_cast<int>(get_data_dims()[0]);
    int C = static_cast<int>(get_data_dims()[1]);
    int K = static_cast<int>(get_grad_input_dims()[1]);
    int P = static_cast<int>(get_grad_input_dims()[ndims_ - 2]);
    int Q = static_cast<int>(get_grad_input_dims()[ndims_ - 1]);

    // temporarily deal with first stage
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
    cfg.loop_sched = 1;
    cfg.num_tile_n = 1;
    cfg.tile_q = 1;
    return std::move(ret);
  } else {
    auto ret = reflection::general_object_t::make<conv_bwd_weight_config_t>();
    conv_bwd_weight_config_t &cfg
      = *ret.unchecked_get_as<conv_bwd_weight_config_t>();
    int N = static_cast<int>(get_data_dims()[0]);
    int C = static_cast<int>(get_data_dims()[1]);
    int K = static_cast<int>(get_grad_input_dims()[1]);
    int P = static_cast<int>(get_grad_input_dims()[ndims_ - 2]);
    int Q = static_cast<int>(get_grad_input_dims()[ndims_ - 1]);
    int num_threads = runtime_config_t::get().get_num_threads();

    // temporarily deal with first stage
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
    cfg.loop_sched = 1;
    cfg.num_tile_n = 1;
    cfg.tile_q = 1;
    cfg.loop_sched = 1; // split NxHxW
    if ((K / cfg.K_block) * (C / cfg.C_block) < num_threads * 2) {
      // ouput chanel and input chanel cannot provide enough parallisiem
      cfg.loop_sched = 1; // split NxHxW
    } else {
      cfg.loop_sched = 2; // don't split reduce axis
    }
    return std::move(ret);
  }
}

gen_conv1x1_backprop_weight_t::gen_conv1x1_backprop_weight_t(sc_op *owner,
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
    out_tensors_.size() == 1, "output logical tensor size should be one.");
  ndims_ = get_data_dims().size();
  const bool is_3d = (ndims_ == 5);
  COMPILE_ASSERT(is_3d
      ? utils::is_one_of(static_cast<int>(padding_.size()), 1, 3)
      : utils::is_one_of(static_cast<int>(padding_.size()), 1, 2),
    "wrong padding dims, should be 1, 2 or 3, but got " << padding_.size()
                                                        << ".");
  COMPILE_ASSERT(is_3d
      ? utils::is_one_of(static_cast<int>(stride_.size()), 1, 3)
      : utils::is_one_of(static_cast<int>(stride_.size()), 1, 2),
    "wrong stride dims, should be 1, 2 or 3, but got " << stride_.size()
                                                       << ".");
}

float gen_conv1x1_backprop_weight_t::get_gflop() const {
  float result = 0.0;
  bool is_3d = ndims_ == 5;
  int stride_d = is_3d ? stride_[0] : 1, stride_h = stride_[0],
      stride_w = stride_[0];
  if (stride_.size() > 1) {
    if (is_3d) { stride_d = stride_[ndims_ - 5]; }
    stride_h = stride_[ndims_ - 4];
    stride_w = stride_[ndims_ - 3];
  }
  const int D = !is_3d
    ? 1
    : (stride_d > 1 ? get_grad_input_dims()[2] : get_data_dims()[2]);
  const int H = stride_h > 1 ? get_grad_input_dims()[ndims_ - 2]
                             : get_data_dims()[ndims_ - 2];
  const int W = stride_w > 1 ? get_grad_input_dims()[ndims_ - 1]
                             : get_data_dims()[ndims_ - 1];
  const int C = get_data_dims()[1];
  const int K = get_grad_input_dims()[1];
  const int N = get_data_dims()[0];
  result = 2.f * N * C * D * W * H * K / 1e9;
  return result;
}

void gen_conv1x1_backprop_weight_t::schedule_loops(context_ptr ctx,
  const conv_bwd_weight_config_t &config, stmt body,
  std::vector<for_loop> &fors) const {
  if (fors.empty()) { return; }
  auto loop_sched = config.loop_sched;
  if (type_ == generator_type_t::REDUCE_N) {
    for_loop ln = fors.at(0), lk = fors.at(1), lc = fors.at(2),
             rlko = fors.at(3), rlco = fors.at(4);
    if (config.loop_sched == 1) {
      auto ln_k = ln->fuse(lk);
      auto ln_kc = ln_k->fuse(lc);
      auto rlk_c = rlko->fuse(rlco);
    }
  } else if (type_ == generator_type_t::REDUCE_ALL) {
    for_loop ln = fors.at(0), lk = fors.at(1), lc = fors.at(2);
    for_loop lnt = fors.at(3), lnpq = fors.at(4);
    for_loop rlko = fors.at(5), rlco = fors.at(6), rlc = fors.at(7);
    if (config.loop_sched == 1) {
      auto ln_k = ln->fuse(lk);
      auto ln_kc = ln_k->fuse(lc);
      auto lnt_pq = lnt->fuse(lnpq);
      auto lrkc = rlko->fuse(rlco);
      auto lrkcc = lrkc->fuse(rlc);
    }
  } else if (type_ == generator_type_t::REDUCE_ALL2) {
    for_loop lk = fors.at(0), lc = fors.at(1), ln = fors.at(2), ld = fors.at(3),
             lp = fors.at(4), rlko = fors.at(5), rlco = fors.at(6);
    if (config.loop_sched == 1) {
      lk->reorder(body, {ln, lk, lc, ld, lp});
      ln->fuse(lk)->fuse(lc)->fuse(ld)->fuse(lp);
      rlko->fuse(rlco);
    } else if (config.loop_sched == 2) {
      lk->fuse(lc);
      ln->attr().set(stmt_attr_key::no_loop_fuse, true);
      ld->attr().set(stmt_attr_key::no_loop_fuse, true);
      lp->attr().set(stmt_attr_key::no_loop_fuse, true);
    }
  }
}

bool gen_conv1x1_backprop_weight_t::generate(context_ptr ctx,
  const conv_bwd_weight_config_t &config, fusion_manager *fusion,
  const std::vector<expr> &inputs, const std::vector<expr> &outputs,
  std::vector<for_loop> &loops) const {
  COMPILE_ASSERT(
    type_ != generator_type_t::UNDEF, "Generator shall have an explicit type.");
  if (type_ == generator_type_t::REDUCE_N) {
    return generate_reduce_N(ctx, config, fusion, inputs, outputs, loops);
  } else if (type_ == generator_type_t::REDUCE_ALL) {
    return generate_reduce_ALL(ctx, config, fusion, inputs, outputs, loops);
  } else {
    return generate_reduce_ALL2(ctx, config, fusion, inputs, outputs, loops);
  }
}

bool gen_conv1x1_backprop_weight_t::generate_reduce_N(const context_ptr &ctx,
  const conv_bwd_weight_config_t &config, fusion_manager *fusion,
  const std::vector<expr> &inputs, const std::vector<expr> &outputs,
  std::vector<for_loop> &loops) const {
  // Init
  bool is_3d = ndims_ == 5;
  int padding_d = is_3d ? padding_[0] : 0, stride_d = is_3d ? stride_[0] : 1;
  int padding_h = padding_[0], padding_w = padding_[0];
  if (padding_.size() > 1) {
    if (is_3d) { padding_d = padding_[ndims_ - 5]; }
    padding_h = padding_[ndims_ - 4];
    padding_w = padding_[ndims_ - 3];
  }
  int stride_h = stride_[0], stride_w = stride_[0];
  if (stride_.size() > 1) {
    if (is_3d) { stride_d = stride_[ndims_ - 5]; }
    stride_h = stride_[ndims_ - 4];
    stride_w = stride_[ndims_ - 3];
  }

  int N = get_data_dims()[0], C = get_data_dims()[1],
      H = get_data_dims()[ndims_ - 2], W = get_data_dims()[ndims_ - 1];
  int O = is_3d ? get_grad_input_dims()[2] : 1,
      D = is_3d ? get_data_dims()[2] : 1;
  int P = get_grad_input_dims()[ndims_ - 2],
      Q = get_grad_input_dims()[ndims_ - 1];
  int K = get_grad_input_dims()[1];
  int K_block = config.K_block, C_block = config.C_block,
      N_block = config.N_block;
  int K_num_block = utils::divide_and_ceil(K, K_block),
      C_num_block = utils::divide_and_ceil(C, C_block),
      N_num_block = utils::divide_and_ceil(N, N_block);
  int N_tile = N_num_block / config.num_tile_n;
  bool loop_sched = config.loop_sched;
  auto dtype = get_dtype();
  bool is_vnni_low_fp = ops::is_vnni_low_fp(ctx, dtype);
  int dtype_block = is_vnni_low_fp ? 2 : 1;
  std::vector<int> p_locations, q_locations, d_locations;
  for (int i = 0; i < H + 2 * padding_h; i++) {
    if (i * stride_h >= padding_h && i * stride_h < H + padding_h) {
      p_locations.push_back(i * stride_h);
    }
  }
  for (int i = 0; i < W + 2 * padding_w; i++) {
    if (i * stride_w >= padding_w && i * stride_w < W + padding_w) {
      q_locations.push_back(i * stride_w);
    }
  }
  for (int i = 0; i < D + 2 * padding_d; i++) {
    if (i * stride_d >= padding_d && i * stride_d < D + padding_d) {
      d_locations.push_back(i * stride_d);
    }
  }
  int P_num_block = static_cast<int>(p_locations.size()),
      Q_num_block = static_cast<int>(q_locations.size()),
      D_num_block = is_3d ? static_cast<int>(d_locations.size()) : 1;
  int padded_h_num
    = static_cast<int>(utils::divide_and_ceil(padding_h, stride_h)),
    padded_d_num
    = static_cast<int>(utils::divide_and_ceil(padding_d, stride_d)),
    padded_w_num
    = static_cast<int>(utils::divide_and_ceil(padding_w, stride_w));

  // define compute
  for_loop ln, lc, lk, ld, lp;
  for_loop rlco, rlko, rlk;
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
          _named_for_(ld, d_o, 0, D_num_block) {
            _named_for_(lp, p_o, 0, P_num_block) {
              std::vector<expr> output_idx
                = {n_o, k_o, p_o + padded_h_num, padded_w_num, 0, 0},
                data_idx
                = {n_o, c_o, p_o * stride_h + p_locations[0] - padding_h,
                  q_locations[0] - padding_w, 0, 0},
                del_weight_tmp_idx = {n_o * K_num_block + k_o, c_o, 0, 0, 0, 0};
              if (is_3d) {
                output_idx.insert(output_idx.begin() + 2, d_o + padded_d_num);
                data_idx.insert(data_idx.begin() + 2,
                  d_o * stride_d + d_locations[0] - padding_d);
                del_weight_tmp_idx.emplace_back(expr(0));
              }
              if (dtype_block > 1) { data_idx.emplace_back(expr(0)); }
              _if_(d_o == 0 && p_o == 0) {
                builtin::brgemm_init_update(tensor_ptr(output, output_idx),
                  tensor_ptr(data, data_idx),
                  tensor_ptr(del_weight_tmp_buf, del_weight_tmp_idx),
                  Q_num_block, K_block, C_block, N_block, N_block, C_block,
                  C_block, K_block * N_block,
                  stride_w * C_block
                    * (int)utils::divide_and_ceil(N_block, dtype_block)
                    * dtype_block,
                  dtype, dtype);
              }
              _else_ {
                builtin::brgemm_update(tensor_ptr(output, output_idx),
                  tensor_ptr(data, data_idx),
                  tensor_ptr(del_weight_tmp_buf, del_weight_tmp_idx),
                  Q_num_block, K_block, C_block, N_block, N_block, C_block,
                  C_block, K_block * N_block,
                  stride_w * C_block
                    * (int)utils::divide_and_ceil(N_block, dtype_block)
                    * dtype_block,
                  dtype, dtype);
              }
            }
          }
        }
      }
    }
    int lanes = 1;
    if (C_block / 16 && C_block % 16 == 0) {
      lanes = vectorize_step(ctx, out_tensors_[0].dtype_.type_code_, 16);
    }
    // KC(D)RSkc
    _named_for_(rlko, l_k_o, 0, K_num_block, 1, for_type::PARALLEL) {
      _named_for_(rlco, l_c_o, 0, C_num_block, 1) {
        builtin::mem_zero(
          tensor_ptr(del_weight,
            is_3d ? std::vector<expr> {l_k_o, l_c_o, 0, 0, 0, 0, 0}
                  : std::vector<expr> {l_k_o, l_c_o, 0, 0, 0, 0}),
          C_block * K_block, datatypes::f32);
        _named_for_(rlk, l_k, 0, K_block, 1) {
          _for_(l_c, 0, C_block, lanes) {
            _for_(l_n, 0, N_num_block, 1) {
              std::vector<expr> del_weight_idx = {l_k_o, l_c_o, 0, 0, l_k, l_c},
                                del_weight_tmp_idx
                = {l_n * K_num_block + l_k_o, l_c_o, 0, 0, l_k, l_c};
              if (is_3d) {
                del_weight_idx.insert(del_weight_idx.begin() + 2, expr(0));
                del_weight_tmp_idx.insert(
                  del_weight_tmp_idx.begin() + 2, expr(0));
              }
              del_weight[span_t(del_weight_idx, lanes)]
                = builder::make_add(del_weight[span_t(del_weight_idx, lanes)],
                  del_weight_tmp_buf[span_t(del_weight_tmp_idx, lanes)]);
            }
          }
        }
        if (fusion) {
          if (is_3d) {
            fusion->create_output_fusion_anchor({tensor_slice(del_weight,
              {{l_k_o, 1}, {l_c_o, 1}, {0, 1}, {0, 1}, {0, 1}, {0, K_block},
                {0, C_block}})});
          } else {
            fusion->create_output_fusion_anchor({tensor_slice(del_weight,
              {{l_k_o, 1}, {l_c_o, 1}, {0, 1}, {0, 1}, {0, K_block},
                {0, C_block}})});
          }
        }
      }
    }
    loops = {ln, lk, lc, rlko, rlco};
  }
  ld->attr().set(stmt_attr_key::no_loop_fuse, true);
  lp->attr().set(stmt_attr_key::no_loop_fuse, true);
  return true;
}

bool gen_conv1x1_backprop_weight_t::generate_reduce_ALL(const context_ptr &ctx,
  const conv_bwd_weight_config_t &config, fusion_manager *fusion,
  const std::vector<expr> &inputs, const std::vector<expr> &outputs,
  std::vector<for_loop> &loops) const {
  // Init
  bool is_3d = ndims_ == 5;
  int padding_d = is_3d ? padding_[0] : 0, stride_d = is_3d ? stride_[0] : 1;
  int padding_h = padding_[0], padding_w = padding_[0];
  if (padding_.size() > 1) {
    if (is_3d) { padding_d = padding_[ndims_ - 5]; }
    padding_h = padding_[ndims_ - 4];
    padding_w = padding_[ndims_ - 3];
  }
  int stride_h = stride_[0], stride_w = stride_[0];
  if (stride_.size() > 1) {
    if (is_3d) { stride_d = stride_[ndims_ - 5]; }
    stride_h = stride_[ndims_ - 4];
    stride_w = stride_[ndims_ - 3];
  }

  int C = get_data_dims()[1], K = get_grad_input_dims()[1];
  int N = get_grad_input_dims()[0], D = is_3d ? get_data_dims()[ndims_ - 3] : 1;
  int H = get_data_dims()[ndims_ - 2], W = get_data_dims()[ndims_ - 1];
  std::vector<int> p_locations, q_locations, d_locations;
  for (int i = 0; i < H + 2 * padding_h; i++) {
    if (i * stride_h >= padding_h && i * stride_h < H + padding_h) {
      p_locations.push_back(i * stride_h);
    }
  }
  for (int i = 0; i < W + 2 * padding_w; i++) {
    if (i * stride_w >= padding_w && i * stride_w < W + padding_w) {
      q_locations.push_back(i * stride_w);
    }
  }
  for (int i = 0; i < D + 2 * padding_d; i++) {
    if (i * stride_d >= padding_d && i * stride_d < D + padding_d) {
      d_locations.push_back(i * stride_d);
    }
  }
  int padded_h_num
    = static_cast<int>(utils::divide_and_ceil(padding_h, stride_h)),
    padded_d_num
    = static_cast<int>(utils::divide_and_ceil(padding_d, stride_d)),
    padded_w_num
    = static_cast<int>(utils::divide_and_ceil(padding_w, stride_w));
  // the effective numbers in computation
  int O = static_cast<int>(d_locations.size()),
      P = static_cast<int>(p_locations.size()),
      Q = static_cast<int>(q_locations.size());
  int NPQ = N * P * Q;
  if (is_3d) { NPQ *= O; }
  int PQ = P * Q;
  int OPQ = is_3d ? PQ * O : PQ;

  // use num_tile_n to represent NPQ_num_tile
  // use tile_p to represent the NPQ_block (reduce block)
  // use tile_q to cut NPQ_num_tile
  int NPQ_num_tile = config.num_tile_n, NPQ_block = config.tile_p,
      tile_q = config.tile_q;
  int NPQ_num_block = utils::divide_and_ceil(NPQ, NPQ_block);
  int NPQ_tile = NPQ_num_block / NPQ_num_tile;
  COMPILE_ASSERT(NPQ_num_block % NPQ_num_tile == 0,
    "bad config with num_tile_n = " << NPQ_num_tile
                                    << ", tile_p = " << NPQ_block
                                    << ", NPQ_num_block = " << NPQ_num_block);
  COMPILE_ASSERT(NPQ_num_tile % tile_q == 0,
    "bad config with num_tile_n = " << NPQ_num_tile << ", tile_q = " << tile_q);
  int K_block = config.K_block, C_block = config.C_block;
  int K_num_block = utils::divide_and_ceil(K, K_block),
      C_num_block = utils::divide_and_ceil(C, C_block);
  bool loop_sched = config.loop_sched;
  auto dtype = get_dtype();
  bool is_vnni_low_fp = ops::is_vnni_low_fp(ctx, dtype);
  int dtype_block = is_vnni_low_fp ? 2 : 1;
  int NPQ_block_pad
    = (dtype_block > 1 && NPQ_block % 2) ? NPQ_block + 1 : NPQ_block;

  // define compute
  for_loop ln, lc, lk, lnt, lnpq, rlko, rlco, rlc;
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
  del_weight_tmp_buf_shape[0]
    = del_weight_tmp_buf_shape[0] * NPQ_num_tile / tile_q;
  {
    _tensor_(del_weight_tmp_buf, datatypes::f32, del_weight_tmp_buf_shape);
    _named_for_(ln, n_o, 0, NPQ_num_tile / tile_q, 1, for_type::PARALLEL) {
      _named_for_(lk, k_o, 0, K_num_block) {
        _named_for_(lc, c_o, 0, C_num_block) {
          _for_(qq, 0, tile_q) {
            _tensor_(data_tmp, dtype, {NPQ_tile, C_block, NPQ_block_pad});
            _tensor_(output_tmp, dtype,
              dtype_block > 1
                ? std::vector<expr> {NPQ_tile, NPQ_block_pad / dtype_block,
                  K_block, dtype_block}
                : std::vector<expr> {NPQ_tile, NPQ_block, K_block});
            _named_for_(lnt, nt_i, 0, NPQ_tile) {
              _named_for_(lnpq, npq_i, 0, NPQ_block_pad) {
                expr npq_idx
                  = ((n_o * tile_q + qq) * NPQ_tile + nt_i) * NPQ_block + npq_i;
                _for_(c_i, 0, C_block) {
                  _if_(npq_i >= NPQ_block) {
                    data_tmp[{nt_i, c_i, npq_i}]
                      = make_expr<constant_node>(float(0.0), dtype);
                  }
                  _else_ {
                    if (is_3d) {
                      data_tmp[{nt_i, c_i, npq_i}] = data[{npq_idx / OPQ,
                        npq_idx % OPQ / PQ * stride_d + d_locations[0]
                          - padding_d,
                        npq_idx % OPQ % PQ / Q * stride_h + p_locations[0]
                          - padding_h,
                        npq_idx % OPQ % PQ % Q * stride_w + q_locations[0]
                          - padding_w,
                        c_o * C_block + c_i}];
                    } else {
                      data_tmp[{nt_i, c_i, npq_i}] = data[{npq_idx / PQ,
                        npq_idx % PQ / Q * stride_h + p_locations[0]
                          - padding_h,
                        npq_idx % PQ % Q * stride_w + q_locations[0]
                          - padding_w,
                        c_o * C_block + c_i}];
                    }
                  }
                }
                _for_(k_i, 0, K_block) {
                  if (dtype_block == 1) {
                    if (is_3d) {
                      output_tmp[{nt_i, npq_i, k_i}] = output[{npq_idx / OPQ,
                        npq_idx % OPQ / PQ + padded_d_num,
                        npq_idx % OPQ % PQ / Q + padded_h_num,
                        npq_idx % OPQ % PQ % Q + padded_w_num,
                        k_o * K_block + k_i}];
                    } else {
                      output_tmp[{nt_i, npq_i, k_i}] = output[{npq_idx / PQ,
                        npq_idx % PQ / Q + padded_h_num,
                        npq_idx % PQ % Q + padded_w_num, k_o * K_block + k_i}];
                    }
                  } else {
                    _if_(npq_i >= NPQ_block) {
                      output_tmp[{nt_i, npq_i / 2, k_i, npq_i % 2}]
                        = make_expr<constant_node>(float(0.0), dtype);
                    }
                    _else_ {
                      if (is_3d) {
                        output_tmp[{nt_i, npq_i / 2, k_i, npq_i % 2}] = output[{
                          npq_idx / OPQ, npq_idx % OPQ / PQ + padded_d_num,
                          npq_idx % OPQ % PQ / Q + padded_h_num,
                          npq_idx % OPQ % PQ % Q + padded_w_num,
                          k_o * K_block + k_i}];
                      } else {
                        output_tmp[{nt_i, npq_i / 2, k_i, npq_i % 2}] = output[{
                          npq_idx / PQ, npq_idx % PQ / Q + padded_h_num,
                          npq_idx % PQ % Q + padded_w_num,
                          k_o * K_block + k_i}];
                      }
                    }
                  }
                }
              }
            }
            _if_(qq == 0) {
              builtin::brgemm_init_update(tensor_ptr(data_tmp, {0, 0, 0}),
                tensor_ptr(output_tmp,
                  dtype_block > 1 ? std::vector<expr> {0, 0, 0, 0}
                                  : std::vector<expr> {0, 0, 0}),
                tensor_ptr(del_weight_tmp_buf,
                  is_3d ? std::vector<expr> {n_o * K_num_block + k_o, c_o, 0, 0,
                    0, 0, 0}
                        : std::vector<expr> {n_o * K_num_block + k_o, c_o, 0, 0,
                          0, 0}),
                NPQ_tile, C_block, K_block, NPQ_block_pad, NPQ_block_pad,
                K_block, K_block, C_block * NPQ_block_pad,
                K_block
                  * (int)utils::divide_and_ceil(NPQ_block_pad, dtype_block)
                  * dtype_block,
                dtype, dtype);
            }
            _else_ {
              builtin::brgemm_update(tensor_ptr(data_tmp, {0, 0, 0}),
                tensor_ptr(output_tmp,
                  dtype_block > 1 ? std::vector<expr> {0, 0, 0, 0}
                                  : std::vector<expr> {0, 0, 0}),
                tensor_ptr(del_weight_tmp_buf,
                  is_3d ? std::vector<expr> {n_o * K_num_block + k_o, c_o, 0, 0,
                    0, 0, 0}
                        : std::vector<expr> {n_o * K_num_block + k_o, c_o, 0, 0,
                          0, 0}),
                NPQ_tile, C_block, K_block, NPQ_block_pad, NPQ_block_pad,
                K_block, K_block, C_block * NPQ_block_pad,
                K_block
                  * (int)utils::divide_and_ceil(NPQ_block_pad, dtype_block)
                  * dtype_block,
                dtype, dtype);
            }
          }
        }
      }
    }
    int lanes = 1;
    if (K_block / 16 && K_block % 16 == 0) {
      lanes = vectorize_step(ctx, out_tensors_[0].dtype_.type_code_, 16);
    }
    // KC(D)RSck
    _named_for_(rlko, l_k_o, 0, K_num_block, 1, for_type::PARALLEL) {
      _named_for_(rlco, l_c_o, 0, C_num_block, 1) {
        builtin::mem_zero(
          tensor_ptr(del_weight,
            is_3d ? std::vector<expr> {l_k_o, l_c_o, 0, 0, 0, 0, 0}
                  : std::vector<expr> {l_k_o, l_c_o, 0, 0, 0, 0}),
          C_block * K_block, datatypes::f32);
        _named_for_(rlc, l_c, 0, C_block, 1) {
          _for_(l_k, 0, K_block, lanes) {
            _for_(l_n, 0, NPQ_num_tile / tile_q, 1) {
              std::vector<expr> del_weight_idx = {l_k_o, l_c_o, 0, 0, l_c, l_k},
                                del_weight_tmp_idx
                = {l_n * K_num_block + l_k_o, l_c_o, 0, 0, l_c, l_k};
              if (is_3d) {
                del_weight_idx.insert(del_weight_idx.begin() + 2, expr(0));
                del_weight_tmp_idx.insert(
                  del_weight_tmp_idx.begin() + 2, expr(0));
              }
              del_weight[span_t(del_weight_idx, lanes)]
                = builder::make_add(del_weight[span_t(del_weight_idx, lanes)],
                  del_weight_tmp_buf[span_t(del_weight_tmp_idx, lanes)]);
            }
          }
        }
        if (fusion) {
          if (is_3d) {
            fusion->create_output_fusion_anchor({tensor_slice(del_weight,
              {{l_k_o, 1}, {l_c_o, 1}, {0, 1}, {0, 1}, {0, 1}, {0, C_block},
                {0, K_block}})});
          } else {
            fusion->create_output_fusion_anchor({tensor_slice(del_weight,
              {{l_k_o, 1}, {l_c_o, 1}, {0, 1}, {0, 1}, {0, C_block},
                {0, K_block}})});
          }
        }
      }
    }
    loops = {ln, lk, lc, lnt, lnpq, rlko, rlco, rlc};
  }
  return true;
}

bool gen_conv1x1_backprop_weight_t::generate_reduce_ALL2(const context_ptr &ctx,
  const conv_bwd_weight_config_t &config, fusion_manager *fusion,
  const std::vector<expr> &inputs, const std::vector<expr> &outputs,
  std::vector<for_loop> &loops) const {
  // Init
  bool is_3d = ndims_ == 5;
  int num_threads = runtime_config_t::get().get_num_threads();
  int padding_d = is_3d ? padding_[0] : 0, stride_d = is_3d ? stride_[0] : 1;
  int padding_h = padding_[0], padding_w = padding_[0];
  if (padding_.size() > 1) {
    if (is_3d) { padding_d = padding_[ndims_ - 5]; }
    padding_h = padding_[ndims_ - 4];
    padding_w = padding_[ndims_ - 3];
  }
  int stride_h = stride_[0], stride_w = stride_[0];
  if (stride_.size() > 1) {
    if (is_3d) { stride_d = stride_[ndims_ - 5]; }
    stride_h = stride_[ndims_ - 4];
    stride_w = stride_[ndims_ - 3];
  }

  int N = get_data_dims()[0], IC = get_data_dims()[1],
      IH = get_data_dims()[ndims_ - 2], IW = get_data_dims()[ndims_ - 1];
  int OD = is_3d ? get_grad_input_dims()[2] : 1,
      ID = is_3d ? get_data_dims()[2] : 1;
  int OH = get_grad_input_dims()[ndims_ - 2],
      OW = get_grad_input_dims()[ndims_ - 1];
  int OC = get_grad_input_dims()[1];
  int OC_block = config.K_block, IC_block = config.C_block,
      N_block = config.N_block;
  int OC_num_block = utils::divide_and_ceil(OC, OC_block),
      IC_num_block = utils::divide_and_ceil(IC, IC_block),
      N_num_block = utils::divide_and_ceil(N, N_block);
  int N_tile = N_num_block / config.num_tile_n;
  bool loop_sched = config.loop_sched;
  auto dtype = get_dtype();
  bool is_vnni_low_fp = ops::is_vnni_low_fp(ctx, dtype);
  int dtype_block = is_vnni_low_fp ? 2 : 1;
  std::vector<int> oh_locations, ow_locations, d_locations;
  for (int i = 0; i < IH + 2 * padding_h; i++) {
    if (i * stride_h >= padding_h && i * stride_h < IH + padding_h) {
      oh_locations.push_back(i * stride_h);
    }
  }
  for (int i = 0; i < IW + 2 * padding_w; i++) {
    if (i * stride_w >= padding_w && i * stride_w < IW + padding_w) {
      ow_locations.push_back(i * stride_w);
    }
  }
  for (int i = 0; i < ID + 2 * padding_d; i++) {
    if (i * stride_d >= padding_d && i * stride_d < ID + padding_d) {
      d_locations.push_back(i * stride_d);
    }
  }
  int OH_num_block = static_cast<int>(oh_locations.size()),
      OW_num_block = static_cast<int>(ow_locations.size()),
      D_num_block = is_3d ? static_cast<int>(d_locations.size()) : 1;
  int padded_h_num
    = static_cast<int>(utils::divide_and_ceil(padding_h, stride_h)),
    padded_d_num
    = static_cast<int>(utils::divide_and_ceil(padding_d, stride_d)),
    padded_w_num
    = static_cast<int>(utils::divide_and_ceil(padding_w, stride_w));

  // define compute
  for_loop ln, lc, lk, ld, lp;
  for_loop rlco, rlko, rlk;
  expr out = outputs.at(op_params_t::out_del_weight),
       data_input = inputs.at(op_params_t::in_data),
       grad_input = inputs.at(op_params_t::in_fwd_output);
  auto filter_dims
    = out_tensors_[op_params_t::out_del_weight].get_blocking_dims();
  std::vector<expr> out_tmp_buf_shape;
  out_tmp_buf_shape.reserve(filter_dims.size());
  for (auto dim : filter_dims) {
    out_tmp_buf_shape.emplace_back(dim2unsigned(dim));
  }
  out_tmp_buf_shape.insert(out_tmp_buf_shape.begin(), num_threads);

  {
    _tensor_(out_tmp_buf, datatypes::f32, out_tmp_buf_shape);
    expr real_out_tmp_buf = out_tmp_buf;
    slice_range data_input_slice_in_range, data_input_slice_out_range,
      grad_input_slice_in_range, grad_input_slice_out_range;

    if (config.loop_sched == 1) {
      // initialize tensor
      trace_guard_t trg(ctx, "zero_init");
      _for_(tid, 0, num_threads, 1, for_type::PARALLEL) {
        auto size = 1UL;
        for (auto dim : filter_dims) {
          size *= dim2unsigned(dim);
        }
        builtin::mem_zero(tensor_ptr(out_tmp_buf,
                            is_3d ? std::vector<expr> {tid, 0, 0, 0, 0, 0, 0, 0}
                                  : std::vector<expr> {tid, 0, 0, 0, 0, 0, 0}),
          size, datatypes::f32);
      }
    } else {
      real_out_tmp_buf = out;
    }
    {
      // core computation
      _named_for_(lk, oc, 0, OC_num_block, 1, for_type::PARALLEL) {
        _named_for_(lc, ic, 0, IC_num_block) {
          _named_for_(ln, bs, 0, N_num_block) {
            _named_for_(ld, d, 0, D_num_block) {
              _named_for_(lp, oh, 0, OH_num_block) {
                _var_init_(
                  tid, datatypes::s32, builder::make_get_group_thread_id(-1));
                std::vector<expr> grad_input_idx
                  = {bs, oc, oh + padded_h_num, padded_w_num, 0, 0},
                  data_input_idx
                  = {bs, ic, oh * stride_h + oh_locations[0] - padding_h,
                    ow_locations[0] - padding_w, 0, 0},
                  out_tmp_idx = {tid, oc, ic, 0, 0, 0, 0};
                if (config.loop_sched == 2) {
                  out_tmp_idx = {oc, ic, 0, 0, 0, 0};
                }
                if (is_3d) {
                  grad_input_idx.insert(
                    grad_input_idx.begin() + 2, d + padded_d_num);
                  data_input_idx.insert(data_input_idx.begin() + 2,
                    d * stride_d + d_locations[0] - padding_d);
                  out_tmp_idx.emplace_back(expr(0));
                }
                if (dtype_block > 1) { data_input_idx.emplace_back(expr(0)); }
                trace_guard_t trg(ctx, "brgemm");
                if (config.loop_sched == 2) {
                  _if_(d == 0 && oh == 0 && bs == 0) {
                    builtin::brgemm_init_update(
                      tensor_ptr(grad_input, grad_input_idx),
                      tensor_ptr(data_input, data_input_idx),
                      tensor_ptr(real_out_tmp_buf, out_tmp_idx), OW_num_block,
                      OC_block, IC_block, N_block, N_block, IC_block, IC_block,
                      OC_block * N_block,
                      stride_w * IC_block
                        * (int)utils::divide_and_ceil(N_block, dtype_block)
                        * dtype_block,
                      dtype, dtype);
                  }
                  _else_ {
                    builtin::brgemm_update(
                      tensor_ptr(grad_input, grad_input_idx),
                      tensor_ptr(data_input, data_input_idx),
                      tensor_ptr(real_out_tmp_buf, out_tmp_idx), OW_num_block,
                      OC_block, IC_block, N_block, N_block, IC_block, IC_block,
                      OC_block * N_block,
                      stride_w * IC_block
                        * (int)utils::divide_and_ceil(N_block, dtype_block)
                        * dtype_block,
                      dtype, dtype);
                  }
                } else {
                  builtin::brgemm_update(tensor_ptr(grad_input, grad_input_idx),
                    tensor_ptr(data_input, data_input_idx),
                    tensor_ptr(real_out_tmp_buf, out_tmp_idx), OW_num_block,
                    OC_block, IC_block, N_block, N_block, IC_block, IC_block,
                    OC_block * N_block,
                    stride_w * IC_block
                      * (int)utils::divide_and_ceil(N_block, dtype_block)
                      * dtype_block,
                    dtype, dtype);
                }
              }
            }
          }
          if (config.loop_sched == 2 && fusion) {
            auto loop = ln->fuse(ld)->fuse(lp);
            loop->attr().set(stmt_attr_key::no_loop_fuse, true);
            trace_guard_t trg(ctx, "post_fusion");
            fusion->create_output_fusion_anchor({tensor_slice(real_out_tmp_buf,
              {{oc, 1}, {ic, 1}, {0, 1}, {0, 1}, {0, OC_block},
                {0, IC_block}})});
          }
        }
      }
    }
    int lanes = 1;
    if (IC_block / 16 && IC_block % 16 == 0) {
      lanes = vectorize_step(ctx, get_dtype().type_code_, 16);
    }
    if (config.loop_sched == 1) {
      // final reduce
      _named_for_(rlko, l_oc, 0, OC_num_block, 1, for_type::PARALLEL) {
        _named_for_(rlco, l_ic, 0, IC_num_block, 1) {
          {
            trace_guard_t trg(ctx, "final_reduce");
            builtin::mem_zero(
              tensor_ptr(out,
                is_3d ? std::vector<expr> {l_oc, l_ic, 0, 0, 0, 0, 0}
                      : std::vector<expr> {l_oc, l_ic, 0, 0, 0, 0}),
              IC_block * OC_block, datatypes::f32);
            _named_for_(rlk, l_k, 0, OC_block, 1) {
              _for_(l_c, 0, IC_block, lanes) {
                _for_(l_tid, 0, num_threads, 1) {
                  std::vector<expr> out_idx = {l_oc, l_ic, 0, 0, l_k, l_c},
                                    out_tmp_idx
                    = {l_tid, l_oc, l_ic, 0, 0, l_k, l_c};
                  if (is_3d) {
                    out_idx.insert(out_idx.begin() + 2, expr(0));
                    out_tmp_idx.insert(out_tmp_idx.begin() + 2, expr(0));
                  }
                  out[span_t(out_idx, lanes)]
                    = builder::make_add(out[span_t(out_idx, lanes)],
                      real_out_tmp_buf[span_t(out_tmp_idx, lanes)]);
                }
              }
            }
          }
          if (fusion) {
            trace_guard_t trg(ctx, "post_fusion");
            if (is_3d) {
              fusion->create_output_fusion_anchor({tensor_slice(out,
                {{l_oc, 1}, {l_ic, 1}, {0, 1}, {0, 1}, {0, 1}, {0, OC_block},
                  {0, IC_block}})});
            } else {
              fusion->create_output_fusion_anchor({tensor_slice(out,
                {{l_oc, 1}, {l_ic, 1}, {0, 1}, {0, 1}, {0, OC_block},
                  {0, IC_block}})});
            }
          }
        }
      }
    }
    loops = {lk, lc, ln, ld, lp, rlko, rlco};
  }
  return true;
}
} // namespace ops
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
