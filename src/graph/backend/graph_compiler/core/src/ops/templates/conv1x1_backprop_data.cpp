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

#include "conv1x1_backprop_data.hpp"
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

config_ptr gen_conv1x1_backprop_data_t::get_default_config(
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

  cfg.tile_d = 1;
  cfg.tile_p = 1;
  if ((padding_h || padding_w || padding_d)
    && ((stride_h > 1) || (stride_w > 1) || (stride_d > 1))) {
    cfg.tile_q = 1;
  } else if (stride_w > 1) {
    cfg.tile_q = get_input_dims()[ndims_ - 1];
  } else if (padding_w > 0) {
    cfg.tile_q = get_output_dims()[ndims_ - 1];
  } else {
    cfg.tile_q = get_output_dims()[ndims_ - 1];
  }
  cfg.loop_sched = 1;
  return std::move(ret);
}

gen_conv1x1_backprop_data_t::gen_conv1x1_backprop_data_t(sc_op *owner,
  const sc_dims &stride, const sc_dims &padding,
  std::vector<logical_tensor_t> &&ins, std::vector<logical_tensor_t> &&outs)
  : parent(owner, std::move(ins), std::move(outs))
  , stride_(stride)
  , padding_(padding) {
  COMPILE_ASSERT(
    in_tensors_.size() == 2, "input logical tensor size should be two.");
  COMPILE_ASSERT(
    out_tensors_.size() == 1, "output logical tensor size should be one.");
  ndims_ = get_input_dims().size();
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

float gen_conv1x1_backprop_data_t::get_gflop() const {
  float result = 0.0;
  bool is_3d = ndims_ == 5;
  int stride_d = is_3d ? stride_[0] : 1, stride_h = stride_[0],
      stride_w = stride_[0];
  if (stride_.size() > 1) {
    stride_h = stride_[ndims_ - 4];
    stride_w = stride_[ndims_ - 3];
  }
  const int D
    = !is_3d ? 1 : (stride_d > 1 ? get_input_dims()[2] : get_output_dims()[2]);
  const int H = stride_h > 1 ? get_input_dims()[ndims_ - 2]
                             : get_output_dims()[ndims_ - 2];
  const int W = stride_w > 1 ? get_input_dims()[ndims_ - 1]
                             : get_output_dims()[ndims_ - 1];
  const int C = get_input_dims()[1];
  const int K = get_output_dims()[1];
  const int N = get_input_dims()[0];

  result = 2.f * N * C * D * W * H * K / 1e9;
  return result;
}

void gen_conv1x1_backprop_data_t::schedule_loops(context_ptr ctx,
  const conv_bwd_data_config_t &config, stmt body,
  std::vector<for_loop> &fors) const {
  for_loop ln = fors.at(0), lc = fors.at(1), ld = fors.at(2), lp = fors.at(3);
  auto loop_sched = config.loop_sched;
  if (loop_sched == 1) {
    int stride_d = (ndims_ == 5) ? stride_[0] : 1;
    auto ln_c = ln->fuse(lc);
    auto ln_c_d = ln_c->fuse(ld);
    if (stride_d == 1) { auto ln_c_d_p = ln_c_d->fuse(lp); }
  }
}

bool gen_conv1x1_backprop_data_t::generate(context_ptr ctx,
  const conv_bwd_data_config_t &config, fusion_manager *fusion,
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

  int N = get_input_dims()[0];
  // O, P, Q is the depth, height, width of grad input
  int K = get_weight_dims()[0], C = get_weight_dims()[1],
      P = get_input_dims()[ndims_ - 2], Q = get_input_dims()[ndims_ - 1];
  int H = get_output_dims()[ndims_ - 2], W = get_output_dims()[ndims_ - 1];
  int O = is_3d ? get_input_dims()[2] : 0, D = is_3d ? get_output_dims()[2] : 0;
  int K_block = config.K_block, C_block = config.C_block,
      tile_d = config.tile_d, tile_p = config.tile_p, tile_q = config.tile_q;
  int K_num_block = utils::divide_and_ceil(K, K_block),
      C_num_block = utils::divide_and_ceil(C, C_block);
  bool loop_sched = config.loop_sched;
  bool is_out_blocking = out_tensors_[0].get_format().is_blocking();
  auto dtype = get_dtype();
  bool is_vnni_low_fp = ops::is_vnni_low_fp(ctx, dtype);
  int dtype_block = is_vnni_low_fp ? 2 : 1;

  if (is_3d) {
    COMPILE_ASSERT(get_weight_dims()[2] == 1 && get_weight_dims()[3] == 1
        && get_weight_dims()[4] == 1,
      "gen_conv1x1_backprop_data_t kernels==1");
  } else {
    COMPILE_ASSERT(get_weight_dims()[2] == 1 && get_weight_dims()[3] == 1,
      "gen_conv1x1_backprop_data_t kernels==1");
  }

  // define compute
  for_loop ln, lc, ld, lp;
  expr del_input = outputs.at(op_params_t::out_del_input),
       output = inputs.at(op_params_t::in_fwd_output),
       weight = inputs.at(op_params_t::in_weight);
  {
    // for conv1x1 cases that have both non-one stride and none-zero padding
    if ((padding_w > 0 || padding_h > 0 || padding_d > 0)
      && (stride_w > 1 || stride_h > 1 || stride_d > 1)) {
      assert(tile_d == 1 && tile_p == 1 && tile_q == 1);
      int C_shift_d = padding_d > 0
        ? (padding_d > stride_d
            ? (stride_d == 1 ? 0 : stride_d - padding_d % stride_d)
            : stride_d - padding_d)
        : 0;
      int C_shift_h = padding_h > 0
        ? (padding_h > stride_h
            ? (stride_h == 1 ? 0 : stride_h - padding_h % stride_h)
            : stride_h - padding_h)
        : 0;
      int C_shift_w = padding_w > 0
        ? (padding_w > stride_w
            ? (stride_w == 1 ? 0 : stride_w - padding_w % stride_w)
            : stride_w - padding_w)
        : 0;
      C_shift_d = C_shift_d < 0 ? 0 : C_shift_d;
      C_shift_h = C_shift_h < 0 ? 0 : C_shift_h;
      C_shift_w = C_shift_w < 0 ? 0 : C_shift_w;
      int A_shift_d = padding_d > 0 ? (1 + (padding_d - 1) / stride_d) : 0;
      int A_shift_h = padding_h > 0 ? (1 + (padding_h - 1) / stride_h) : 0;
      int A_shift_w = padding_w > 0 ? (1 + (padding_w - 1) / stride_w) : 0;

      _named_for_(ln, n, 0, N, 1, for_type::PARALLEL) {
        _named_for_(lc, c_o, 0, C_num_block) {
          int P_num_block = 0, Q_num_block = 0, D_num_block = is_3d ? 0 : 1;
          for (int i = 0; i < O; i++) {
            if (i * stride_d > padding_d - 1 && i * stride_d < D + padding_d) {
              D_num_block++;
            }
          }
          for (int i = 0; i < P; i++) {
            if (i * stride_h > padding_h - 1 && i * stride_h < H + padding_h) {
              P_num_block++;
            }
          }
          for (int i = 0; i < Q; i++) {
            if (i * stride_w > padding_w - 1 && i * stride_w < W + padding_w) {
              Q_num_block++;
            }
          }
          _named_for_(ld, d_o, 0, D_num_block) {
            _named_for_(lp, p_o, 0, P_num_block) {
              _for_(q_o, 0, Q_num_block) {
                expr LDA, LDC, stride_a;
                std::vector<expr> a_idx, b_idx, c_idx;
                if (in_tensors_[0].get_format().is_blocking()) {
                  LDA = K_block;
                  stride_a = is_3d ? O * P * Q * K_block : P * Q * K_block;
                  a_idx = is_3d
                    ? std::vector<expr> {n, 0, d_o * tile_d + A_shift_d,
                      p_o * tile_p + A_shift_h, q_o * tile_q + A_shift_w, 0}
                    : std::vector<expr> {n, 0, p_o * tile_p + A_shift_h,
                      q_o * tile_q + A_shift_w, 0};
                } else {
                  LDA = K;
                  stride_a = K_block;
                  a_idx = is_3d
                    ? std::vector<expr> {n, d_o * tile_d + A_shift_d,
                      p_o * tile_p + A_shift_h, q_o * tile_q + A_shift_w, 0}
                    : std::vector<expr> {
                      n, p_o * tile_p + A_shift_h, q_o * tile_q + A_shift_w, 0};
                }
                b_idx = std::vector<expr> {c_o, 0, 0, 0, 0, 0};
                if (is_3d) b_idx.emplace_back(0);
                if (dtype_block > 1) b_idx.emplace_back(0);
                if (is_out_blocking) {
                  LDC = C_block * stride_w;
                  c_idx = is_3d ? std::vector<expr> {n, c_o,
                            d_o * tile_d * stride_d + C_shift_d,
                            p_o * tile_p * stride_h + C_shift_h,
                            q_o * tile_q * stride_w + C_shift_w, 0}
                                : std::vector<expr> {n, c_o,
                                  p_o * tile_p * stride_h + C_shift_h,
                                  q_o * tile_q * stride_w + C_shift_w, 0};
                } else {
                  LDC = C * stride_w;
                  LDC->attr().set("N_axis",
                    is_3d ? std::vector<size_t> {4} : std::vector<size_t> {3});
                  c_idx = is_3d
                    ? std::vector<expr> {n, d_o * tile_d * stride_d + C_shift_d,
                      p_o * tile_p * stride_h + C_shift_h,
                      q_o * tile_q * stride_w + C_shift_w, c_o * C_block}
                    : std::vector<expr> {n, p_o * tile_p * stride_h + C_shift_h,
                      q_o * tile_q * stride_w + C_shift_w, c_o * C_block};
                }
                LDC->attr().set("stride_w", stride_w);
                builtin::brgemm_init_update(tensor_ptr(output, a_idx),
                  tensor_ptr(weight, b_idx), tensor_ptr(del_input, c_idx),
                  K_num_block, tile_p * tile_q, C_block, K_block, LDA, C_block,
                  LDC, stride_a,
                  (int)utils::divide_and_ceil(K_block, dtype_block)
                    * dtype_block * C_block,
                  dtype, dtype);

                if (is_3d) {
                  if (fusion) {
                    if (!is_out_blocking) {
                      fusion->create_output_fusion_anchor({tensor_slice(
                        del_input,
                        {{n, 1}, {d_o * tile_d * stride_d + C_shift_d, tile_d},
                          {p_o * tile_p * stride_h + C_shift_h, tile_p},
                          {q_o * tile_q * stride_w + C_shift_w, tile_q},
                          {c_o * C_block, C_block}})});
                    } else {
                      fusion->create_output_fusion_anchor(
                        {tensor_slice(del_input,
                          {{n, 1}, {c_o, 1},
                            {d_o * tile_d * stride_d + C_shift_d, tile_d},
                            {p_o * tile_p * stride_h + C_shift_h, tile_p},
                            {q_o * tile_q * stride_w + C_shift_w, tile_q},
                            {0, C_block}})});
                    }
                  }
                } else {
                  if (fusion) {
                    if (!is_out_blocking) {
                      fusion->create_output_fusion_anchor({tensor_slice(
                        del_input,
                        {{n, 1}, {p_o * tile_p * stride_h + C_shift_h, tile_p},
                          {q_o * tile_q * stride_w + C_shift_w, tile_q},
                          {c_o * C_block, C_block}})});
                    } else {
                      fusion->create_output_fusion_anchor(
                        {tensor_slice(del_input,
                          {{n, 1}, {c_o, 1},
                            {p_o * tile_p * stride_h + C_shift_h, tile_p},
                            {q_o * tile_q * stride_w + C_shift_w, tile_q},
                            {0, C_block}})});
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
    // for conv1x1 cases that has either non-one stride or none-zero padding
    else {
      _named_for_(ln, n, 0, N, 1, for_type::PARALLEL) {
        _named_for_(lc, c_o, 0, C_num_block) {
          int D_num_block
            = !is_3d ? 1 : (stride_d > 1 ? O / tile_d : D / tile_d),
            P_num_block = stride_h > 1 ? P / tile_p : H / tile_p,
            Q_num_block = stride_w > 1 ? Q / tile_q : W / tile_q;
          _named_for_(ld, d_o, 0, D_num_block) {
            _named_for_(lp, p_o, 0, P_num_block) {
              _for_(q_o, 0, Q_num_block) {
                if (is_3d) {
                  // init brgemm when having strided writing, this enables the
                  // locations which did not take part in convolution (due to
                  // stride) to be 0, rather than NAN or some other values
                  // exited in memory
                  if (is_out_blocking && stride_w > 1) {
                    // consider cases that stride_x is not divisible by X
                    _for_(s_d, 0, stride_d) {
                      // different from 2d init process, we need to initilize
                      // one by one on D axis, considering the discontinuity
                      // casued by stride_d
                      std::vector<expr> del_input_index
                        = {n, c_o, d_o * tile_d * stride_d + s_d,
                          p_o * tile_p * stride_h, q_o * tile_q * stride_w, 0};
                      _if_(d_o * tile_d * stride_d + s_d < expr(D)) {
                        _if_(q_o == expr(Q_num_block - 1)) {
                          _if_(p_o < expr(P_num_block - 1)) {
                            builtin::dnnl_brgemm_init(
                              tensor_ptr(del_input, del_input_index),
                              tile_p * stride_h,
                              C_block
                                * (W - (Q_num_block - 1) * tile_q * stride_w),
                              C_block
                                * (W - (Q_num_block - 1) * tile_q * stride_w),
                              datatypes::f32, 0);
                          }
                          _else_ {
                            builtin::dnnl_brgemm_init(
                              tensor_ptr(del_input, del_input_index),
                              H - (P_num_block - 1) * tile_p * stride_h,
                              C_block
                                * (W - (Q_num_block - 1) * tile_q * stride_w),
                              C_block
                                * (W - (Q_num_block - 1) * tile_q * stride_w),
                              datatypes::f32, 0);
                          }
                        }
                        _else_ {
                          _if_(p_o < expr(P_num_block - 1)) {
                            builtin::dnnl_brgemm_init(
                              tensor_ptr(del_input, del_input_index),
                              tile_p * stride_h, C_block * tile_q * stride_w,
                              C_block * tile_q * stride_w, datatypes::f32, 0);
                          }
                          _else_ {
                            builtin::dnnl_brgemm_init(
                              tensor_ptr(del_input, del_input_index),
                              H - (P_num_block - 1) * tile_p * stride_h,
                              C_block * tile_q * stride_w,
                              C_block * tile_q * stride_w, datatypes::f32, 0);
                          }
                        }
                      }
                    }
                  } else if (!is_out_blocking && stride_w > 1) {
                    _for_(s_d, 0, stride_d) {
                      std::vector<expr> del_input_index = {n,
                        d_o * tile_d * stride_d + s_d, p_o * tile_p * stride_h,
                        q_o * tile_q * stride_w, c_o * C_block};
                      _if_(d_o * tile_d * stride_d + s_d < expr(D)) {
                        // here we are forcing Q_num_block == 1 to make P, Q
                        // axis continuous
                        COMPILE_ASSERT(Q_num_block == 1,
                          "Q_num_block must be 1 in non-blocking conv bwd data "
                          "kernel (strided case).");
                        expr LDC = C;
                        LDC->attr().set("plain_init", true);
                        _if_(p_o < expr(P_num_block - 1)) {
                          builtin::dnnl_brgemm_init(
                            tensor_ptr(del_input, del_input_index),
                            tile_p * stride_h * W, C_block, LDC, datatypes::f32,
                            0);
                        }
                        _else_ {
                          builtin::dnnl_brgemm_init(
                            tensor_ptr(del_input, del_input_index),
                            (H - (P_num_block - 1) * tile_p * stride_h) * W,
                            C_block, LDC, datatypes::f32, 0);
                        }
                      }
                    }
                  }
                  expr LDA, LDC, stride_a;
                  std::vector<expr> a_idx, c_idx;
                  if (in_tensors_[0].get_format().is_blocking()) {
                    LDA = K_block;
                    stride_a = O * P * Q * K_block;
                    a_idx = {n, 0, d_o * tile_d + padding_d,
                      p_o * tile_p + padding_h, q_o * tile_q + padding_w, 0};
                  } else {
                    LDA = K;
                    stride_a = K_block;
                    a_idx = {n, d_o * tile_d + padding_d,
                      p_o * tile_p + padding_h, q_o * tile_q + padding_w, 0};
                  }
                  if (is_out_blocking) {
                    LDC = C_block * stride_w;
                    c_idx = {n, c_o, d_o * tile_d * stride_d,
                      p_o * tile_p * stride_h, q_o * tile_q * stride_w, 0};
                  } else {
                    LDC = C * stride_w;
                    LDC->attr().set("N_axis", std::vector<size_t> {4});
                    c_idx
                      = {n, d_o * tile_d * stride_d, p_o * tile_p * stride_h,
                        q_o * tile_q * stride_w, c_o * C_block};
                  }
                  LDC->attr().set("stride_w", stride_w);
                  builtin::brgemm_init_update(tensor_ptr(output, a_idx),
                    tensor_ptr(weight,
                      dtype_block > 1
                        ? std::vector<expr> {c_o, 0, 0, 0, 0, 0, 0, 0}
                        : std::vector<expr> {c_o, 0, 0, 0, 0, 0, 0}),
                    tensor_ptr(del_input, c_idx), K_num_block, tile_p * tile_q,
                    C_block, K_block, LDA, C_block, LDC, stride_a,
                    (int)utils::divide_and_ceil(K_block, dtype_block)
                      * dtype_block * C_block,
                    dtype, dtype);
                  if (fusion && W % stride_w == 0 && H % stride_h == 0
                    && D % stride_d == 0) {
                    // when D(H, W) is not devisible by stride_d(stride_h,
                    // stride_w), there will be non-constant value on
                    // tensor_slice, we need to put the fusion anchor outside
                    // this loop
                    if (!is_out_blocking) {
                      fusion->create_output_fusion_anchor(
                        {tensor_slice(del_input,
                          std::vector<std::pair<expr, expr>> {{n, 1},
                            {d_o * tile_d * stride_d, tile_d * stride_d},
                            {p_o * tile_p * stride_h, tile_p * stride_h},
                            {q_o * tile_q * stride_w, tile_q * stride_w},
                            {c_o * C_block, C_block}})});
                    } else {
                      fusion->create_output_fusion_anchor(
                        {tensor_slice(del_input,
                          std::vector<std::pair<expr, expr>> {{n, 1}, {c_o, 1},
                            {d_o * tile_d * stride_d, tile_d * stride_d},
                            {p_o * tile_p * stride_h, tile_p * stride_h},
                            {q_o * tile_q * stride_w, tile_q * stride_w},
                            {0, C_block}})});
                    }
                  }
                } else {
                  // init brgemm when having strided writing
                  if (is_out_blocking && stride_w > 1) {
                    // consider cases that stride_x is not divisible by X
                    std::vector<expr> del_input_index = {n, c_o,
                      p_o * tile_p * stride_h, q_o * tile_q * stride_w, 0};
                    _if_(q_o == expr(Q_num_block - 1)) {
                      _if_(p_o < expr(P_num_block - 1)) {
                        builtin::dnnl_brgemm_init(
                          tensor_ptr(del_input, del_input_index),
                          tile_p * stride_h,
                          C_block * (W - (Q_num_block - 1) * tile_q * stride_w),
                          C_block * (W - (Q_num_block - 1) * tile_q * stride_w),
                          datatypes::f32, 0);
                      }
                      _else_ {
                        builtin::dnnl_brgemm_init(
                          tensor_ptr(del_input, del_input_index),
                          H - (P_num_block - 1) * tile_p * stride_h,
                          C_block * (W - (Q_num_block - 1) * tile_q * stride_w),
                          C_block * (W - (Q_num_block - 1) * tile_q * stride_w),
                          datatypes::f32, 0);
                      }
                    }
                    _else_ {
                      _if_(p_o < expr(P_num_block - 1)) {
                        builtin::dnnl_brgemm_init(
                          tensor_ptr(del_input, del_input_index),
                          tile_p * stride_h, C_block * tile_q * stride_w,
                          C_block * tile_q * stride_w, datatypes::f32, 0);
                      }
                      _else_ {
                        builtin::dnnl_brgemm_init(
                          tensor_ptr(del_input, del_input_index),
                          H - (P_num_block - 1) * tile_p * stride_h,
                          C_block * tile_q * stride_w,
                          C_block * tile_q * stride_w, datatypes::f32, 0);
                      }
                    }
                  } else if (!is_out_blocking && stride_w > 1) {
                    std::vector<expr> del_input_index
                      = {n, p_o * tile_p * stride_h, q_o * tile_q * stride_w,
                        c_o * C_block};
                    // here we are forcing Q_num_block == 1 to make P, Q axis
                    // continuous
                    COMPILE_ASSERT(Q_num_block == 1,
                      "Q_num_block must be 1 in non-blocking conv bwd data "
                      "kernel (strided case).");
                    expr LDC = C;
                    LDC->attr().set("plain_init", true);
                    _if_(p_o < expr(P_num_block - 1)) {
                      builtin::dnnl_brgemm_init(
                        tensor_ptr(del_input, del_input_index),
                        tile_p * stride_h * W, C_block, LDC, datatypes::f32, 0);
                    }
                    _else_ {
                      builtin::dnnl_brgemm_init(
                        tensor_ptr(del_input, del_input_index),
                        (H - (P_num_block - 1) * tile_p * stride_h) * W,
                        C_block, LDC, datatypes::f32, 0);
                    }
                  }
                  // consider plain/blocking format
                  expr LDA, LDC, stride_a;
                  std::vector<expr> a_idx, c_idx;
                  if (in_tensors_[0].get_format().is_blocking()) {
                    LDA = K_block;
                    stride_a = P * Q * K_block;
                    a_idx = {n, 0, p_o * tile_p + padding_h,
                      q_o * tile_q + padding_w, 0};
                  } else {
                    LDA = K;
                    stride_a = K_block;
                    a_idx = {
                      n, p_o * tile_p + padding_h, q_o * tile_q + padding_w, 0};
                  }
                  if (is_out_blocking) {
                    LDC = C_block * stride_w;
                    c_idx = {n, c_o, p_o * tile_p * stride_h,
                      q_o * tile_q * stride_w, 0};
                  } else {
                    LDC = C * stride_w;
                    LDC->attr().set("N_axis", std::vector<size_t> {3});
                    c_idx = {n, p_o * tile_p * stride_h,
                      q_o * tile_q * stride_w, c_o * C_block};
                  }
                  LDC->attr().set("stride_w", stride_w);

                  builtin::brgemm_init_update(tensor_ptr(output, a_idx),
                    tensor_ptr(weight,
                      dtype_block > 1
                        ? std::vector<expr> {c_o, 0, 0, 0, 0, 0, 0}
                        : std::vector<expr> {c_o, 0, 0, 0, 0, 0}),
                    tensor_ptr(del_input, c_idx), K_num_block, tile_p * tile_q,
                    C_block, K_block, LDA, C_block, LDC, stride_a,
                    (int)utils::divide_and_ceil(K_block, dtype_block)
                      * dtype_block * C_block,
                    dtype, dtype);
                  if (fusion && W % stride_w == 0 && H % stride_h == 0) {
                    // when H(W) is not devisible by stride_h(stride_w), there
                    // will be non-constant value on tensor_slice, we need to
                    // put the fusion anchor outside this loop
                    if (!is_out_blocking) {
                      fusion->create_output_fusion_anchor(
                        {tensor_slice(del_input,
                          std::vector<std::pair<expr, expr>> {{n, 1},
                            {p_o * tile_p * stride_h, tile_p * stride_h},
                            {q_o * tile_q * stride_w, tile_q * stride_w},
                            {c_o * C_block, C_block}})});
                    } else {
                      fusion->create_output_fusion_anchor(
                        {tensor_slice(del_input,
                          std::vector<std::pair<expr, expr>> {{n, 1}, {c_o, 1},
                            {p_o * tile_p * stride_h, tile_p * stride_h},
                            {q_o * tile_q * stride_w, tile_q * stride_w},
                            {0, C_block}})});
                    }
                  }
                }
              }
              if (fusion && H % stride_h == 0) {
                // when H is not devisible by stride_h, there will be
                // non-constant value on tensor_slice, we need to put the fusion
                // anchor outside this loop
                if (is_3d) {
                  if (D % stride_d == 0) {
                    if (!is_out_blocking) {
                      fusion->create_output_fusion_anchor(
                        {tensor_slice(del_input,
                          {{n, 1}, {d_o * tile_d * stride_d, tile_d * stride_d},
                            {p_o * tile_p * stride_h, tile_p * stride_h},
                            {0, W}, {c_o * C_block, C_block}})});
                    } else {
                      fusion->create_output_fusion_anchor(
                        {tensor_slice(del_input,
                          {{n, 1}, {c_o, 1},
                            {d_o * tile_d * stride_d, tile_d * stride_d},
                            {p_o * tile_p * stride_h, tile_p * stride_h},
                            {0, W}, {0, C_block}})});
                    }
                  }
                } else {
                  if (!is_out_blocking) {
                    fusion->create_output_fusion_anchor({tensor_slice(del_input,
                      {{n, 1}, {p_o * tile_p * stride_h, tile_p * stride_h},
                        {0, W}, {c_o * C_block, C_block}})});
                  } else {
                    fusion->create_output_fusion_anchor({tensor_slice(del_input,
                      {{n, 1}, {c_o, 1},
                        {p_o * tile_p * stride_h, tile_p * stride_h}, {0, W},
                        {0, C_block}})});
                  }
                }
              }
            }
            if (fusion && is_3d && D % stride_d == 0) {
              if (!is_out_blocking) {
                fusion->create_output_fusion_anchor({tensor_slice(del_input,
                  {{n, 1}, {d_o * tile_d * stride_d, tile_d * stride_d}, {0, H},
                    {0, W}, {c_o * C_block, C_block}})});
              } else {
                fusion->create_output_fusion_anchor({tensor_slice(del_input,
                  {{n, 1}, {c_o, 1},
                    {d_o * tile_d * stride_d, tile_d * stride_d}, {0, H},
                    {0, W}, {0, C_block}})});
              }
            }
          }
          if (fusion) {
            if (is_3d) {
              if (!is_out_blocking) {
                fusion->create_output_fusion_anchor({tensor_slice(del_input,
                  {{n, 1}, {0, D}, {0, H}, {0, W}, {c_o * C_block, C_block}})});
              } else {
                fusion->create_output_fusion_anchor({tensor_slice(del_input,
                  {{n, 1}, {c_o, 1}, {0, D}, {0, H}, {0, W}, {0, C_block}})});
              }
            } else {
              if (!is_out_blocking) {
                fusion->create_output_fusion_anchor({tensor_slice(del_input,
                  {{n, 1}, {0, H}, {0, W}, {c_o * C_block, C_block}})});
              } else {
                fusion->create_output_fusion_anchor({tensor_slice(del_input,
                  {{n, 1}, {c_o, 1}, {0, H}, {0, W}, {0, C_block}})});
              }
            }
          }
        }
      }
    }
  }

  loops = {ln, lc, ld, lp};
  return true;
}

} // namespace ops
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
