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

#include "nested_conv1x1_backprop_data.hpp"
#include <algorithm>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "utils.hpp"
#include <compiler/ir/builder.hpp>
#include <compiler/ir/builtin.hpp>
#include <compiler/ir/easy_build.hpp>
#include <compiler/ir/graph/fusion_mgr.hpp>
#include <runtime/config.hpp>
#include <util/any_map.hpp>
#include <util/math_utils.hpp>
#include <util/reflection.hpp>
#include <util/utils.hpp>

using namespace dnnl::impl::graph::gc::builder;
namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace ops {

static void get_blocks_and_ib_blocks(const int X, const int X_split_num,
  const int ix_block, int &X_block_size, int &X_ib_block_size) {
  if (utils::divide_and_ceil(X, X_block_size) < (size_t)X_split_num
    && X_block_size > ix_block) {
    X_block_size -= ix_block;
  }
  // BS, N, K imbalance block size
  X_ib_block_size = X - X_block_size * X_split_num <= 0
    ? X - X_block_size * (X_split_num - 1)
    : X_block_size + ix_block;
  if (X_ib_block_size < 0) {
    // cannot use all the threads
    X_ib_block_size = X - X / X_block_size * X_block_size;
  }
  if (X_ib_block_size == 0) { X_ib_block_size = X_block_size; }
}

static void compute_single_thr_and_idx(const int X, const int X_split_num,
  const int X_block_size, const int X_ib_num, const int tail_X,
  const int X_ib_block_size, const expr &x_s, expr &X_single_thr_size,
  expr &x_idx) {
  if (X_block_size == X_ib_block_size) {
    X_single_thr_size = X_block_size;
    x_idx = x_s * X_block_size;
  } else {
    if (X - X_block_size * X_split_num <= 0) {
      // cannot use all the cores due to small shapes
      x_idx = x_s * X_block_size;
    } else {
      x_idx
        = builder::make_select(x_s < X_split_num - X_ib_num, x_s * X_block_size,
          (X_split_num - X_ib_num) * X_block_size
            + (x_s + X_ib_num - X_split_num) * X_ib_block_size);
    }
    if (tail_X != X_ib_block_size) {
      // has tail and imbalance
      X_single_thr_size = builder::make_select(x_s < X_split_num - X_ib_num,
        X_block_size,
        builder::make_select(x_s == X_split_num - 1, tail_X, X_ib_block_size));
    } else {
      if (X - X_block_size * X_split_num <= 0) {
        // cannot use all the cores due to small shapes
        X_single_thr_size = builder::make_select(
          x_s < X / X_block_size, X_block_size, X_ib_block_size);
      } else {
        X_single_thr_size = builder::make_select(
          x_s < X_split_num - X_ib_num, X_block_size, X_ib_block_size);
      }
    }
  }
}

config_ptr gen_nested_conv1x1_backprop_data_t::get_default_config(
  context_ptr ctx) const {
  auto ret
    = reflection::general_object_t::make<nested_conv_bwd_data_config_t>();
  nested_conv_bwd_data_config_t &cfg
    = *ret.unchecked_get_as<nested_conv_bwd_data_config_t>();
  const int num_threads = runtime_config_t::get().get_num_threads();
  const int im_bs_block = im_bs_block_;
  const int im_ow_block = im_ow_block_;
  const int im_ic_block = im_ic_block_;
  const int im_oc_block = im_oc_block_;

  // Assume padding = 0
  bool is_3d = ndims_ == 5;
  int stride_d = is_3d ? stride_[0] : 1;
  int stride_h = stride_[0], stride_w = stride_[0];
  if (stride_.size() > 1) {
    if (is_3d) { stride_d = stride_[ndims_ - 5]; }
    stride_h = stride_[ndims_ - 4];
    stride_w = stride_[ndims_ - 3];
  }
  bool has_stride = stride_h > 1 || stride_w > 1;

  const int D
    = !is_3d ? 1 : (stride_d > 1 ? get_input_dims()[2] : get_output_dims()[2]);
  const int H = stride_h > 1 ? get_input_dims()[ndims_ - 2]
                             : get_output_dims()[ndims_ - 2];
  const int W = stride_w > 1 ? get_input_dims()[ndims_ - 1]
                             : get_output_dims()[ndims_ - 1];
  const int IC = get_weight_dims()[1];
  const int OC = get_weight_dims()[0];
  const int BS = get_input_dims()[0];
  const int OS = D * H * W;

  const int sizeofdtypeA
    = utils::get_sizeof_etype(in_tensors_[0].dtype_.as_etype());
  const int sizeofdtypeC
    = utils::get_sizeof_etype(out_tensors_[0].dtype_.as_etype());
  float cost = std::numeric_limits<float>::max();
  auto split_s_list = get_splits(num_threads);
  int split_s = 1;
  for (int64_t j = split_s_list.size() - 1; j >= 0; --j) {
    int i = split_s_list[j];
    int num_BS_block
      = utils::divide_and_ceil(BS / im_bs_block, num_threads / i);
    int num_S_block = utils::divide_and_ceil(OS / im_ow_block, i);
    int num_brgemm = num_BS_block * num_S_block;
    int num_core = std::min(i, OS / im_ow_block)
      * std::min(num_threads / i, BS / im_bs_block);
    // Cost = Shape_efficient_weight *
    // (workload_balance + divide_N_plenty) / core_utilitizaiton
    // single core gemm prefers square shape for A and B.
    // For small workload, the A and B shape is not a key problem, but the
    // num_core and num_brgemm is important to performance. Use 2048 to reduce
    // the shape weight on small shape.
    float new_cost = (1024 + BS * i / float(num_threads) + OS / float(i))
      * (num_brgemm + 8 * i) / float(num_core);
    if (new_cost < cost) {
      split_s = i;
      cost = new_cost;
    }
  }
  cfg.bs_threads = num_threads / split_s;
  cfg.spatial_threads = split_s;
  cfg.ic_threads = 1;
  // when spatial size is small, and has enough IC split space, give splits on
  // BS and IC
  if (OS < 64 && IC > 256) {
    auto possible_factors = get_splits(num_threads);
    for (int64_t i = possible_factors.size() - 1; i >= 0; --i) {
      if (BS >= possible_factors[i]) {
        cfg.bs_threads = possible_factors[i];
        cfg.ic_threads = num_threads / possible_factors[i];
        cfg.spatial_threads = 1;
        break;
      }
    }
  }

  int single_BS = utils::divide_and_ceil(BS, cfg.bs_threads) * im_bs_block;
  int single_S = utils::divide_and_ceil(
                   utils::divide_and_ceil(OS, im_ow_block), cfg.spatial_threads)
    * im_ow_block;
  int single_IC = IC, single_OC = OC;
  int L2_size = static_cast<int>(ctx->machine_.cpu_flags_.getDCacheSize(2));
  int single_C_threshold
    = (single_BS * single_S * sizeofdtypeA < L2_size ? 2048 : 4096)
    / sizeofdtypeA;
  if (single_IC * single_OC >= single_C_threshold) {
    int L2_C = utils::divide_and_ceil(single_IC, im_ic_block) * im_ic_block
      * utils::divide_and_ceil(single_OC, im_oc_block) * im_oc_block;
    int L2_BS
      = (sqrt(pow(2 * sizeofdtypeA * L2_C, 2) + 4 * sizeofdtypeC * L2_size)
          - 2 * sizeofdtypeA * L2_C)
      / (2 * sizeofdtypeC);
    cfg.bs_num_blocks = std::max(1, single_BS / L2_C);
    cfg.spatial_num_blocks = std::max(1, single_S / L2_C);
  } else {
    int L2_BS_S = L2_size / (2 * sizeofdtypeA * single_IC * single_OC);
    cfg.bs_num_blocks = std::max(1, single_BS / L2_BS_S);
    cfg.spatial_num_blocks = std::max(1, single_S / L2_BS_S);
  }

  cfg.ic_num_blocks = std::max(1, IC / im_ic_block / cfg.ic_threads / 8);
  cfg.oc_num_blocks = 1;

  return std::move(ret);
}

gen_nested_conv1x1_backprop_data_t::gen_nested_conv1x1_backprop_data_t(
  sc_op *owner, const sc_dims &stride, const sc_dims &padding,
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
  const int IC = get_weight_dims()[1];
  const int OC = get_weight_dims()[0];
  const int BS = get_input_dims()[0];
  const int OS = D * H * W;

  bool is_vnni_low_fp = ops::is_vnni_low_fp(get_default_context(), get_dtype());
  bool no_vnni = ops::no_vnni(get_default_context(), get_dtype());
  bool has_stride = stride_d > 1 || stride_h > 1 || stride_w > 1;

  int64_t IC_block_default = 32;
  int64_t OC_block_default = 32;
  if (no_vnni) {
    IC_block_default = 16;
    OC_block_default = 16;
  } else if (is_vnni_low_fp) {
    IC_block_default = 32;
    OC_block_default = 32;
  } else {
    assert(utils::is_one_of(get_dtype(), datatypes::u8, datatypes::s8));
    IC_block_default = 64;
    OC_block_default = 64;
  }

  if (OS >= 256)
    im_ow_block_ = W;
  else if (OS >= 64)
    im_ow_block_ = has_stride ? W : W * 2;
  else
    im_ow_block_ = has_stride ? W : OS;
  im_ic_block_ = IC_block_default;
  im_oc_block_ = OC_block_default;
  im_bs_block_ = 1;
}

float gen_nested_conv1x1_backprop_data_t::get_gflop() const {
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

void gen_nested_conv1x1_backprop_data_t::schedule_loops(context_ptr ctx,
  const nested_conv_bwd_data_config_t &config, stmt body,
  std::vector<for_loop> &fors) const {}

void gen_nested_conv1x1_backprop_data_t::
  single_thread_conv1x1_backprop_data_call(const context_ptr &ctx,
    const logical_tensor_t &ta, const logical_tensor_t &tb,
    const logical_tensor_t &tc, const nested_conv_bwd_data_config_t &config,
    const expr &BS, const expr &OS, const expr &IC, const expr &OC,
    const expr &bs_idx, const expr &s_idx, const expr &ic_idx,
    const expr &oc_idx, const int stride_d, const int stride_h,
    const int stride_w, const expr &A, const expr &B, const expr &C,
    int dtype_block, fusion_manager *fusion, const expr &bs_s, const expr &s_s,
    const expr &ic_s, std::vector<int> &BS_anchor_info,
    std::vector<int> &S_anchor_info, std::vector<int> &IC_anchor_info,
    const bool is_out_blocking, bool is_partial, const expr &oc_s) const {
  expr BS_sub_block = config.bs_num_blocks,
       S_sub_block = config.spatial_num_blocks,
       IC_sub_block = config.ic_num_blocks, OC_sub_block = config.oc_num_blocks;
  for_loop im_oc, im_bs, im_os, im_ic, o_im_oc;
  int ori_BS = static_cast<int>(ta.get_plain_dims()[0]),
      ori_H = static_cast<int>(ta.get_plain_dims()[ndims_ - 2]),
      ori_W = static_cast<int>(ta.get_plain_dims()[ndims_ - 1]),
      ori_S = ori_H * ori_W, ori_IC = static_cast<int>(tb.get_plain_dims()[1]),
      ori_OC = static_cast<int>(tb.get_plain_dims()[0]);
  _var_init_(tid, datatypes::s32, builder::make_get_group_thread_id(-1));

  _for_(o_bs, 0, BS_sub_block) {
    _for_(o_s, 0, S_sub_block) {
      _for_(o_ic, 0, IC_sub_block) {
        expr bs_b_idx, s_b_idx, ic_b_idx, oc_b_idx, bs_b_bigger_num,
          s_b_bigger_num, ic_b_bigger_num, oc_b_bigger_num;
        _var_init_(bs_o_end, datatypes::s32,
          get_balance211_length(
            BS / im_bs_block_, BS_sub_block, o_bs, bs_b_idx, bs_b_bigger_num));
        _var_init_(s_o_end, datatypes::s32,
          get_balance211_length(
            OS / im_ow_block_, S_sub_block, o_s, s_b_idx, s_b_bigger_num));
        _var_init_(ic_o_end, datatypes::s32,
          get_balance211_length(
            IC / im_ic_block_, IC_sub_block, o_ic, ic_b_idx, ic_b_bigger_num));
        _named_for_(o_im_oc, o_oc, 0, OC_sub_block) {
          _named_for_(im_bs, i_bs, 0, bs_o_end) {
            _var_init_(bs_start_idx, datatypes::index,
              bs_idx + o_bs * BS / BS_sub_block
                + ((i_bs + tid) % bs_o_end) * im_bs_block_);
            _named_for_(im_os, i_os, 0, s_o_end) {
              _named_for_(im_ic, i_ic, 0, ic_o_end) {
                _var_init_(m_start_idx, datatypes::index,
                  s_idx + o_s * OS / S_sub_block
                    + ((i_os + tid) % s_o_end) * im_ow_block_);
                _var_init_(n_start_idx, datatypes::index,
                  ic_idx + o_ic * IC / IC_sub_block
                    + ((i_ic + tid) % ic_o_end) * im_ic_block_);
                _var_init_(num, datatypes::s32,
                  get_balance211_length(OC / im_oc_block_, OC_sub_block, o_oc,
                    oc_b_idx, oc_b_bigger_num));
                _var_init_(k_start_idx, datatypes::index, oc_idx + o_oc);
                // TODO(zhangyan): consider 3D case
                std::vector<expr> aidx = std::vector<expr> {bs_start_idx,
                  m_start_idx / ori_W, m_start_idx % ori_W, k_start_idx};
                std::vector<expr> bidx = dtype_block > 1
                  ? std::vector<expr> {n_start_idx / im_ic_block_,
                    k_start_idx / im_oc_block_ / 2, 0, 0, 0, 0, 0}
                  : !tb.get_format().is_blocking()
                  ? std::vector<expr> {k_start_idx, n_start_idx, 0, 0}
                  : std::vector<expr> {n_start_idx / im_ic_block_,
                    k_start_idx / im_oc_block_, 0, 0, 0, 0};
                std::vector<expr> cidx
                  = {bs_start_idx, m_start_idx / ori_W * stride_h,
                    m_start_idx % ori_W * stride_w * ori_W, n_start_idx};
                if (stride_w > 1 && !is_out_blocking) {
                  expr LDC = ori_IC;
                  LDC->attr().set("plain_init", true);
                  _if_(o_oc == 0) {
                    // when strides > 1, im_ow_block_ = OW
                    builtin::dnnl_brgemm_init(tensor_ptr(C, cidx),
                      stride_h * stride_w * im_ow_block_, im_ic_block_, LDC,
                      datatypes::f32, 0);
                  }
                }
                auto LDA
                  = !ta.get_format().is_blocking() ? ori_OC : im_oc_block_;
                auto LDB
                  = !tb.get_format().is_blocking() ? ori_IC : im_ic_block_;
                expr LDC
                  = (!is_out_blocking ? ori_IC : im_ic_block_) * stride_w;
                if (!tc.get_format().is_blocking())
                  LDC->attr().set("stride_w", stride_w);
                LDC->attr().set(
                  "N_axis", std::vector<size_t> {3}); // only consider 2D case
                auto stride_a = im_oc_block_;
                auto stride_b = !tb.get_format().is_blocking()
                  ? im_oc_block_ * ori_IC
                  : im_oc_block_ * im_ic_block_;
                _if_(o_oc == 0) {
                  builtin::brgemm_init_update(tensor_ptr(A, aidx),
                    tensor_ptr(B, bidx), tensor_ptr(C, cidx), num, im_ow_block_,
                    im_ic_block_, im_oc_block_, LDA, LDB, LDC, stride_a,
                    stride_b, ta.dtype_, tb.dtype_);
                }
                _else_ {
                  builtin::brgemm_update(tensor_ptr(A, aidx),
                    tensor_ptr(B, bidx), tensor_ptr(C, cidx), num, im_ow_block_,
                    im_ic_block_, im_oc_block_, LDA, LDB, LDC, stride_a,
                    stride_b, ta.dtype_, tb.dtype_);
                } // brgemm
                if (fusion && !is_partial) {
                  // TODO(zhangyan): consider more fusion cases.
                  if (ori_W % stride_w == 0 && ori_H % stride_h == 0) {
                    fusion->create_output_fusion_anchor({tensor_slice(C,
                      std::vector<std::pair<expr, expr>> {
                        {bs_start_idx, 1},
                        {m_start_idx / ori_W * stride_h, stride_h},
                        {m_start_idx % ori_W * stride_w,
                          im_ow_block_ * stride_w},
                        {n_start_idx, im_ic_block_},
                      })});
                  }
                } // if fusion
              } // im_ic
            } // im_os
          } // im_bs
        } // o_oc
        if (fusion && !is_partial) {
          // TODO(zhangyan): consider more fusion cases.
          if (BS_anchor_info[1] == BS_anchor_info[2]
            && S_anchor_info[1] == S_anchor_info[2]
            && IC_anchor_info[1] == IC_anchor_info[2]
            && BS_anchor_info[1] / im_bs_block_ % config.bs_num_blocks == 0
            && S_anchor_info[1] / im_ow_block_ % config.spatial_num_blocks == 0
            && IC_anchor_info[1] / im_ic_block_ % config.ic_num_blocks
              == 0) { // no imbalance
            if (ori_H % stride_h == 0 && ori_W % stride_w == 0) {
              fusion->create_output_fusion_anchor({tensor_slice(C,
                std::vector<std::pair<expr, expr>> {
                  {bs_idx + o_bs * BS / BS_sub_block,
                    BS_anchor_info[1] / config.bs_num_blocks},
                  {(s_idx + o_s * OS / S_sub_block) / ori_W * stride_h,
                    S_anchor_info[1] / config.spatial_num_blocks / ori_W
                      * stride_h},
                  {(s_idx + o_s * OS / S_sub_block) % ori_W * stride_w,
                    ori_W * stride_w},
                  {ic_idx + o_ic * IC / IC_sub_block,
                    IC_anchor_info[1] / config.ic_num_blocks},
                })});
            }
          }
        } // if fusion
      } // o_ic
    } // o_os
  } // o_bs
  if (config.oc_num_blocks > 1) {
    im_ic->attr()[stmt_attr_key::reduce_root_loop]
      = std::weak_ptr<stmt_base_t>(o_im_oc.impl);
  }
}

bool gen_nested_conv1x1_backprop_data_t::generate(context_ptr ctx,
  const nested_conv_bwd_data_config_t &config, fusion_manager *fusion,
  const std::vector<expr> &inputs, const std::vector<expr> &outputs,
  std::vector<for_loop> &loops) const {
  // Init OP param
  // Assume paddings = 0
  bool is_3d = ndims_ == 5;
  int stride_d = is_3d ? stride_[0] : 1;
  int stride_h = stride_[0], stride_w = stride_[0];
  if (stride_.size() > 1) {
    if (is_3d) { stride_d = stride_[ndims_ - 5]; }
    stride_h = stride_[ndims_ - 4];
    stride_w = stride_[ndims_ - 3];
  }
  int BS = get_input_dims()[0];
  int OC = get_weight_dims()[0], IC = get_weight_dims()[1];
  int OH = get_input_dims()[ndims_ - 2], OW = get_input_dims()[ndims_ - 1];
  int IH = get_output_dims()[ndims_ - 2], IW = get_output_dims()[ndims_ - 1];
  int OD = is_3d ? get_input_dims()[2] : 1,
      ID = is_3d ? get_output_dims()[2] : 1;
  int OS = OD * OH * OW, IS = ID * IH * IW;

  // Get config
  int BS_split_num = config.bs_threads, S_split_num = config.spatial_threads,
      IC_split_num = config.ic_threads;
  int num_threads = runtime_config_t::get().get_num_threads();
  int OC_split_num = num_threads / BS_split_num / S_split_num / IC_split_num;
  assert(OC_split_num == 1);
  int BS_sub_block = config.bs_num_blocks,
      S_sub_block = config.spatial_num_blocks,
      IC_sub_block = config.ic_num_blocks, OC_sub_block = config.oc_num_blocks;
  int BS_block_size = utils::divide_and_ceil(
                        utils::divide_and_ceil(BS, BS_split_num), im_bs_block_)
    * im_bs_block_;
  int S_block_size = utils::divide_and_ceil(
                       utils::divide_and_ceil(OS, S_split_num), im_ow_block_)
    * im_ow_block_;
  int IC_block_size = utils::divide_and_ceil(
                        utils::divide_and_ceil(IC, IC_split_num), im_ic_block_)
    * im_ic_block_;
  int OC_block_size = utils::divide_and_ceil(
                        utils::divide_and_ceil(OC, OC_split_num), im_oc_block_)
    * im_oc_block_;

  // make sure that each thread has workload
  int BS_ib_block_size, S_ib_block_size, IC_ib_block_size, OC_ib_block_size;
  get_blocks_and_ib_blocks(
    BS, BS_split_num, im_bs_block_, BS_block_size, BS_ib_block_size);
  get_blocks_and_ib_blocks(
    OS, S_split_num, im_ow_block_, S_block_size, S_ib_block_size);
  get_blocks_and_ib_blocks(
    IC, IC_split_num, im_ic_block_, IC_block_size, IC_ib_block_size);
  get_blocks_and_ib_blocks(
    OC, OC_split_num, im_oc_block_, OC_block_size, OC_ib_block_size);
  // update X_block_size and X_ib_block_size to minimize their gaps
  if (BS_block_size >= im_bs_block_ * 2) {
    int BS_new_ib_block_size, BS_new_block_size = BS_block_size - im_bs_block_;
    get_blocks_and_ib_blocks(
      BS, BS_split_num, im_bs_block_, BS_new_block_size, BS_new_ib_block_size);
    if (std::abs(BS_block_size - BS_ib_block_size)
      > std::abs(BS_new_block_size - BS_new_ib_block_size)) {
      BS_block_size = BS_new_block_size;
      BS_ib_block_size = BS_new_ib_block_size;
    }
  }
  if (S_block_size >= im_ow_block_ * 2) {
    int S_new_ib_block_size, S_new_block_size = S_block_size - im_ow_block_;
    get_blocks_and_ib_blocks(
      OS, S_split_num, im_ow_block_, S_new_block_size, S_new_ib_block_size);
    if (std::abs(S_block_size - S_ib_block_size)
      > std::abs(S_new_block_size - S_new_ib_block_size)) {
      S_block_size = S_new_block_size;
      S_ib_block_size = S_new_ib_block_size;
    }
  }
  if (IC_block_size >= im_ic_block_ * 2) {
    int IC_new_ib_block_size, IC_new_block_size = IC_block_size - im_ic_block_;
    get_blocks_and_ib_blocks(
      IC, IC_split_num, im_ic_block_, IC_new_block_size, IC_new_ib_block_size);
    if (std::abs(IC_block_size - IC_ib_block_size)
      > std::abs(IC_new_block_size - IC_new_ib_block_size)) {
      IC_block_size = IC_new_block_size;
      IC_ib_block_size = IC_new_ib_block_size;
    }
  }
  if (OC_block_size >= im_oc_block_ * 2) {
    int OC_new_ib_block_size, OC_new_block_size = OC_block_size - im_oc_block_;
    get_blocks_and_ib_blocks(
      OC, OC_split_num, im_oc_block_, OC_new_block_size, OC_new_ib_block_size);
    if (std::abs(OC_block_size - OC_ib_block_size)
      > std::abs(OC_new_block_size - OC_new_ib_block_size)) {
      OC_block_size = OC_new_block_size;
      OC_ib_block_size = OC_new_ib_block_size;
    }
  }
  // BS, OS, IC, OC imbalance block num
  int BS_ib_num = BS - BS_block_size * BS_split_num < 0
    ? 1
    : utils::divide_and_ceil(BS - BS_block_size * BS_split_num, im_bs_block_);
  int S_ib_num = OS - S_block_size * S_split_num < 0
    ? 1
    : utils::divide_and_ceil(OS - S_block_size * S_split_num, im_ow_block_);
  int IC_ib_num = IC - IC_block_size * IC_split_num < 0
    ? 1
    : utils::divide_and_ceil(IC - IC_block_size * IC_split_num, im_ic_block_);
  int OC_ib_num = OC - OC_block_size * OC_split_num < 0
    ? 1
    : utils::divide_and_ceil(OC - OC_block_size * OC_split_num, im_oc_block_);
  int tail_BS = BS_ib_num <= 1 ? BS_ib_block_size
                               : BS_ib_block_size
      - (BS_ib_num * BS_ib_block_size
        + (BS_split_num - BS_ib_num) * BS_block_size - BS);
  int tail_S = S_ib_num <= 1 ? S_ib_block_size
                             : S_ib_block_size
      - (S_ib_num * S_ib_block_size + (S_split_num - S_ib_num) * S_block_size
        - OS);
  int tail_IC = IC_ib_num <= 1 ? IC_ib_block_size
                               : IC_ib_block_size
      - (IC_ib_num * IC_ib_block_size
        + (IC_split_num - IC_ib_num) * IC_block_size - IC);
  int tail_OC = OC_ib_num <= 1 ? OC_ib_block_size
                               : OC_ib_block_size
      - (OC_ib_num * OC_ib_block_size
        + (OC_split_num - OC_ib_num) * OC_block_size - OC);
  assert(BS_ib_num >= 0 && BS_ib_num <= BS_split_num && S_ib_num >= 0
    && S_ib_num <= S_split_num && IC_ib_num >= 0 && IC_ib_num <= IC_split_num
    && OC_ib_num >= 0 && OC_ib_num <= OC_split_num);

  BS_ib_block_size = utils::rnd_up(BS_ib_block_size, im_bs_block_);
  S_ib_block_size = utils::rnd_up(S_ib_block_size, im_ow_block_);
  IC_ib_block_size = utils::rnd_up(IC_ib_block_size, im_ic_block_);
  OC_ib_block_size = utils::rnd_up(OC_ib_block_size, im_oc_block_);
  tail_BS = utils::rnd_up(tail_BS, im_bs_block_);
  tail_S = utils::rnd_up(tail_S, im_ow_block_);
  tail_IC = utils::rnd_up(tail_IC, im_ic_block_);
  tail_OC = utils::rnd_up(tail_OC, im_oc_block_);

  COMPILE_ASSERT(BS_block_size / im_bs_block_ >= BS_sub_block
      && BS_ib_block_size / im_bs_block_ >= BS_sub_block,
    "bad BS_sub_block given");
  COMPILE_ASSERT(S_block_size / im_ow_block_ >= S_sub_block
      && S_ib_block_size / im_ow_block_ >= S_sub_block,
    "bad S_sub_block given");
  COMPILE_ASSERT(IC_block_size / im_ic_block_ >= IC_sub_block
      && IC_ib_block_size / im_ic_block_ >= IC_sub_block,
    "bad IC_sub_block given");
  COMPILE_ASSERT(OC_block_size / im_oc_block_ >= OC_sub_block
      && OC_ib_block_size / im_oc_block_ >= OC_sub_block,
    "bad OC_sub_block given");

  bool is_out_blocking = out_tensors_[0].get_format().is_blocking();
  auto dtype = get_dtype();
  bool is_vnni_low_fp = ops::is_vnni_low_fp(ctx, dtype);
  int dtype_block = is_vnni_low_fp ? 2 : 1;

  if (is_3d) {
    COMPILE_ASSERT(get_weight_dims()[2] == 1 && get_weight_dims()[3] == 1
        && get_weight_dims()[4] == 1,
      "gen_nested_conv1x1_backprop_data_t kernels==1");
  } else {
    COMPILE_ASSERT(get_weight_dims()[2] == 1 && get_weight_dims()[3] == 1,
      "gen_nested_conv1x1_backprop_data_t kernels==1");
  }

  // define compute
  expr del_input = outputs.at(op_params_t::out_del_input),
       output = inputs.at(op_params_t::in_fwd_output),
       weight = inputs.at(op_params_t::in_weight);

  std::vector<int> BS_anchor_info
    = {BS_ib_num, BS_block_size, BS_ib_block_size},
    S_anchor_info = {S_ib_num, S_block_size, S_ib_block_size},
    IC_anchor_info = {IC_ib_num, IC_block_size, IC_ib_block_size};

  for_loop bsloop;
  int BS_real_split = std::min(
    static_cast<int>(utils::divide_and_ceil(BS, im_bs_block_)), BS_split_num);
  int S_real_split = std::min(
    static_cast<int>(utils::divide_and_ceil(OS, im_ow_block_)), S_split_num);
  int IC_real_split = std::min(
    static_cast<int>(utils::divide_and_ceil(IC, im_ic_block_)), IC_split_num);
  int OC_real_split = std::min(
    static_cast<int>(utils::divide_and_ceil(OC, im_oc_block_)), OC_split_num);

  if (OC_split_num == 1) { // no need to do reduction on OC axis
    expr bs_idx, s_idx, ic_idx, oc_idx, BS_single_thr_size, S_single_thr_size,
      IC_single_thr_size;
    _named_for_(
      bsloop, bs_s, 0, BS_real_split, 1, for_type::PARALLEL, BS_split_num) {
      _for_(s_s, 0, S_real_split, 1, for_type::PARALLEL, S_split_num) {
        _for_(ic_s, 0, IC_real_split, 1, for_type::PARALLEL, IC_split_num) {
          compute_single_thr_and_idx(BS, BS_split_num, BS_block_size, BS_ib_num,
            tail_BS, BS_ib_block_size, bs_s, BS_single_thr_size, bs_idx);
          compute_single_thr_and_idx(OS, S_split_num, S_block_size, S_ib_num,
            tail_S, S_ib_block_size, s_s, S_single_thr_size, s_idx);
          compute_single_thr_and_idx(IC, IC_split_num, IC_block_size, IC_ib_num,
            tail_IC, IC_ib_block_size, ic_s, IC_single_thr_size, ic_idx);
          _for_(oc_s, 0, OC_split_num, 1, for_type::PARALLEL, OC_split_num) {
            _if_(bs_idx < (uint64_t)BS && s_idx < (uint64_t)(OS)
              && ic_idx < (uint64_t)IC) {
              single_thread_conv1x1_backprop_data_call(ctx, in_tensors_[0],
                in_tensors_[1], out_tensors_[0], config, BS_single_thr_size,
                S_single_thr_size, IC_single_thr_size,
                (int)utils::rnd_up(OC, im_oc_block_), bs_idx, s_idx, ic_idx,
                oc_s, stride_d, stride_h, stride_w, output, weight, del_input,
                dtype_block, fusion, bs_s, s_s, ic_s, BS_anchor_info,
                S_anchor_info, IC_anchor_info, is_out_blocking);
            } // if
          } // oc_s loop
          if (fusion) {
            // TODO(zhangyan): consider more fusion cases.
            if (BS_block_size == BS_ib_block_size
              && S_block_size == S_ib_block_size
              && IC_block_size == IC_ib_block_size) {
              if (OH % stride_h == 0 && OW % stride_w) {
                fusion->create_output_fusion_anchor({tensor_slice(del_input,
                  {{bs_idx, BS_block_size},
                    {s_idx / OW * stride_h, S_block_size / OW * stride_w},
                    {s_idx % OW, OW * stride_w}, {ic_idx, IC_block_size}})});
              }
            }
          } // if fusion
        } // ic_s loop
      } // s_s loop
    } // bs_s loop
  } else {
    return false; // TODO(zhangyan): support reduction on OC axis
  }
  loops = {};
  return true;
}
} // namespace ops
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
