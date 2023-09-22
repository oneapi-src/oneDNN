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

#include "nested_conv1x1_backprop_weight.hpp"
#include <algorithm>
#include <limits>
#include <string>
#include <utility>
#include "utils.hpp"
#include <compiler/ir/builder.hpp>
#include <compiler/ir/builtin.hpp>
#include <compiler/ir/easy_build.hpp>
#include <compiler/ir/graph/fusion_mgr.hpp>
#include <compiler/ir/transform/tensor_shrink.hpp>
#include <ops/templates/commit_op.hpp>
#include <runtime/config.hpp>
#include <runtime/parallel.hpp>
#include <util/any_map.hpp>
#include <util/math_utils.hpp>
#include <util/reflection.hpp>

using namespace dnnl::impl::graph::gc::builder;
namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace ops {

config_ptr gen_nested_conv1x1_backprop_weight_t::get_default_config(
  context_ptr ctx) const {
  auto ret
    = reflection::general_object_t::make<nested_conv_bwd_weight_config_t>();
  nested_conv_bwd_weight_config_t &cfg
    = *ret.unchecked_get_as<nested_conv_bwd_weight_config_t>();
  int num_threads = runtime_config_t::get().get_num_threads();

  const int im_bs_block = im_bs_block_;
  const int im_ic_block = im_ic_block_;
  const int im_oc_block = im_oc_block_;

  bool is_3d = ndims_ == 5;
  int stride_d = is_3d ? stride_[0] : 1, stride_h = stride_[0],
      stride_w = stride_[0];
  if (stride_.size() > 1) {
    if (is_3d) { stride_d = stride_[ndims_ - 5]; }
    stride_h = stride_[ndims_ - 4];
    stride_w = stride_[ndims_ - 3];
  }
  const int D
    = !is_3d ? 1 : (stride_d > 1 ? get_grad_dims()[2] : get_data_dims()[2]);
  const int H
    = stride_h > 1 ? get_grad_dims()[ndims_ - 2] : get_data_dims()[ndims_ - 2];
  const int W
    = stride_w > 1 ? get_grad_dims()[ndims_ - 1] : get_data_dims()[ndims_ - 1];
  const int IC = get_data_dims()[1];
  const int OC = get_grad_dims()[1];
  const int BS = get_data_dims()[0];

  cfg.oc_threads = 1;
  cfg.ic_threads = 1;
  cfg.bs_threads = 1;
  cfg.od_threads = 1;
  cfg.oh_threads = 1;

  if (num_threads % 7 == 0) {
    COMPILE_ASSERT(H % 7 == 0 && W % 7 == 0,
      "Currently, only support cases with spatial size equals multiple integer "
      "of 7.")
    cfg.oh_threads = 7;
    num_threads /= cfg.oh_threads;
  }

  const int sizeofdtypeA
    = utils::get_sizeof_etype(in_tensors_[0].dtype_.as_etype());
  const int sizeofdtypeC
    = utils::get_sizeof_etype(out_tensors_[0].dtype_.as_etype());
  float cost = std::numeric_limits<float>::max();
  int split_ic = 1;
  for (int i = 1; i <= num_threads; i++) {
    int num_BS_block
      = utils::divide_and_ceil(BS / im_bs_block, num_threads / i);
    int num_IC_block = utils::divide_and_ceil(IC / im_ic_block, i);
    int num_brgemm = num_BS_block * num_IC_block;
    int num_core = std::min(i, IC / im_ic_block)
      * std::min(num_threads / i, BS / im_bs_block);
    // Cost = Shape_efficient_weight *
    // (workload_balance + divide_N_plenty) / core_utilitizaiton
    // single core gemm prefers square shape for A and B.
    // For small workload, the A and B shape is not a key problem, but the
    // num_core and num_brgemm is important to performance. Use 2048 to reduce
    // the shape weight on small shape.
    float new_cost = (1024 + BS * i / float(num_threads) + IC / float(i))
      * (num_brgemm + 8 * i) / float(num_core);
    if (new_cost < cost && IC / im_ic_block % i == 0
      && BS / im_bs_block % (num_threads / i) == 0) {
      split_ic = i;
      cost = new_cost;
    }
  }
  cfg.bs_threads = std::min(BS / im_bs_block, num_threads / split_ic);
  cfg.ic_threads = std::min(IC / im_ic_block, split_ic);
  // avoid (OC / im_oc_block) % oc_threads != 0
  auto valid_splits_oc = get_splits(OC / im_oc_block);
  for (auto split : valid_splits_oc) {
    if (split <= num_threads / cfg.bs_threads / cfg.ic_threads) {
      cfg.oc_threads = std::max(cfg.oc_threads, split);
    }
  }

  // Config single core
  cfg.oc_num_blocks = OC / cfg.oc_threads / im_oc_block / 16 >= 1
    ? OC / cfg.oc_threads / im_oc_block / 16
    : 1;
  cfg.ic_num_blocks = IC / cfg.ic_threads / im_ic_block / 16 >= 1
    ? IC / cfg.ic_threads / im_ic_block / 16
    : 1;
  cfg.bs_num_blocks = BS / cfg.bs_threads / im_bs_block >= 1
    ? BS / cfg.bs_threads / im_bs_block
    : 1;
  cfg.oh_num_blocks = H / cfg.oh_threads / 2 >= 1 ? H / cfg.oh_threads / 2 : 1;
  cfg.ow_num_blocks = W / 14 >= 1 ? W / 14 : 1;
  cfg.od_num_blocks = 1;

  return std::move(ret);
}

gen_nested_conv1x1_backprop_weight_t::gen_nested_conv1x1_backprop_weight_t(
  sc_op *owner, const sc_dims &stride, const sc_dims &padding,
  std::vector<logical_tensor_t> &&ins, std::vector<logical_tensor_t> &&outs)
  : parent(owner, std::move(ins), std::move(outs))
  , stride_(stride)
  , padding_(padding) {
  COMPILE_ASSERT(
    in_tensors_.size() == 2, "input logical tensor size should be two.");
  COMPILE_ASSERT(
    out_tensors_.size() == 1, "output logical tensor size should be one.");
  bool is_vnni_low_fp
    = ops::is_vnni_low_fp(get_default_context(), get_A_dtype());
  ndims_ = get_data_dims().size();
  bool is_3d = ndims_ == 5;
  int stride_d = is_3d ? stride_[0] : 1, stride_h = stride_[0],
      stride_w = stride_[0];
  if (stride_.size() > 1) {
    if (is_3d) { stride_d = stride_[ndims_ - 5]; }
    stride_h = stride_[ndims_ - 4];
    stride_w = stride_[ndims_ - 3];
  }
  const int D
    = !is_3d ? 1 : (stride_d > 1 ? get_grad_dims()[2] : get_data_dims()[2]);
  const int H
    = stride_h > 1 ? get_grad_dims()[ndims_ - 2] : get_data_dims()[ndims_ - 2];
  const int W
    = stride_w > 1 ? get_grad_dims()[ndims_ - 1] : get_data_dims()[ndims_ - 1];
  const int IC = get_data_dims()[1];
  const int OC = get_grad_dims()[1];
  const int N = get_data_dims()[0];
  if (is_vnni_low_fp) {
    im_oc_block_ = 32;
    im_ic_block_ = 32;
    im_bs_block_ = 32;
  } else {
    im_oc_block_ = 16;
    im_ic_block_ = 16;
    im_bs_block_ = 16;
  }
}

float gen_nested_conv1x1_backprop_weight_t::get_gflop() const {
  float result = 0.0;
  bool is_3d = ndims_ == 5;
  int stride_d = is_3d ? stride_[0] : 1, stride_h = stride_[0],
      stride_w = stride_[0];
  if (stride_.size() > 1) {
    if (is_3d) { stride_d = stride_[ndims_ - 5]; }
    stride_h = stride_[ndims_ - 4];
    stride_w = stride_[ndims_ - 3];
  }
  const int D
    = !is_3d ? 1 : (stride_d > 1 ? get_grad_dims()[2] : get_data_dims()[2]);
  const int H
    = stride_h > 1 ? get_grad_dims()[ndims_ - 2] : get_data_dims()[ndims_ - 2];
  const int W
    = stride_w > 1 ? get_grad_dims()[ndims_ - 1] : get_data_dims()[ndims_ - 1];
  const int C = get_data_dims()[1];
  const int K = get_grad_dims()[1];
  const int N = get_data_dims()[0];
  result = 2.f * N * C * D * W * H * K / 1e9;
  return result;
}

void gen_nested_conv1x1_backprop_weight_t::schedule_loops(context_ptr ctx,
  const nested_conv_bwd_weight_config_t &config, stmt body,
  std::vector<for_loop> &fors) const {}

void gen_nested_conv1x1_backprop_weight_t::forward_input_reorder_call(
  context_ptr &ctx, const expr &temp_forward_input, const expr &forward_input,
  const logical_tensor_t &input_lt, const sc_data_type_t &dtype, int bs_block,
  int ic_block, int oh_block, int ow_block, int IH, int IW,
  const expr &bs_offset, const expr &ic_offset, const expr &oh_offset,
  const expr &ow_offset, int stride_h, int stride_w) const {
  // NHWCn or NHWC --> NCHWcn
  if (stride_h > 1 || stride_w > 1) {
    trace_guard_t trg(ctx, "forward_input_reorder");
    _for_(ih_reorder, 0, oh_block) {
      _for_(iw_reorder, 0, ow_block) {
        _for_(ic_reorder, 0, ic_block) {
          _for_(ibs_reorder, 0, bs_block) {
            expr bs_idx = bs_offset + ibs_reorder;
            expr ic_idx = ic_offset + ic_reorder;
            expr h_idx = oh_offset + ih_reorder;
            expr w_idx = ow_offset + iw_reorder;
            expr input_h_idx = h_idx * stride_h;
            expr input_w_idx = w_idx * stride_w;
            std::vector<expr> tmp_input_idx {ibs_reorder / im_bs_block_,
              ic_reorder / im_ic_block_, ih_reorder, iw_reorder,
              ic_reorder % im_ic_block_, ibs_reorder % im_bs_block_};
            std::vector<expr> input_idx {
              bs_idx, input_h_idx, input_w_idx, ic_idx};
            _if_((input_h_idx >= 0 && input_h_idx < IH)
              && (input_w_idx >= 0 && input_w_idx < IW)) {
              temp_forward_input[tmp_input_idx] = forward_input[input_idx];
            }
            _else_ {
              temp_forward_input[tmp_input_idx] = builder::make_constant(
                std::vector<union_val>(0.f), sc_data_type_t(dtype.type_code_));
            }
          }
        }
      }
    }
  } else {
    // use vnni reorder when strides = 1
    // shrinked_shape
    std::vector<expr> temp_forward_input_shape_shr = {bs_block / im_bs_block_,
      ic_block / im_ic_block_, oh_block, ow_block, im_ic_block_, im_bs_block_};
    std::vector<expr> shrink_offset = std::vector<expr> {
      bs_offset / im_bs_block_, ic_offset / im_ic_block_, oh_offset, ow_offset,
      bs_offset % im_bs_block_, ic_offset % im_ic_block_};
    temp_forward_input->attr()[tensor_shrinker_attrs::should_shrink]
      = tensor_shrinker_t::shrink_info_t {
        shrink_offset, temp_forward_input_shape_shr, stmts()};
    slice_range tmp_input_slice_range
      = slice_range {{bs_offset, bs_block / im_bs_block_},
        {ic_offset, ic_block / im_ic_block_}, {oh_offset, oh_block},
        {ow_offset, ow_block}, {0, im_bs_block_}, {0, im_ic_block_}};
    ops::commit_op(ctx, "reorder",
      {tensor_slice(forward_input,
        {{bs_offset, bs_block}, {oh_offset, oh_block}, {ow_offset, ow_block},
          {ic_offset, ic_block}})},
      {tensor_slice(temp_forward_input, std::move(tmp_input_slice_range))},
      {graph_tensor::make(
        input_lt.get_plain_dims(), input_lt.get_format(), input_lt.dtype_)},
      {},
      {{"out_format", sc_data_format_t::NCHWcn(im_ic_block_, im_bs_block_)}});
  }
}

void gen_nested_conv1x1_backprop_weight_t::inner_loop_call(context_ptr &ctx,
  const expr &temp_forward_input,
  const std::vector<expr> &temp_forward_idx_non_block,
  const logical_tensor_t &delta_output_lt, const expr &delta_output,
  const expr &real_delta_weight_buf, const std::vector<expr> &temp_weight_idx,
  const sc_data_type_t &dtype, int dtype_block, int ic_block, int oc_block,
  int bs_block, int od_block, int oh_block, int ow_block, int stride_h,
  int stride_w, const expr &o_bs, const expr &o_od, const expr &o_oh,
  const expr &o_ow, const expr &obs_offset, const expr &oc_offset,
  const expr &oh_offset, const expr &ow_offset, fusion_manager *fusion,
  bool is_partial) const {
  int BS = delta_output_lt.get_plain_dims()[0];
  int OC = delta_output_lt.get_plain_dims()[1];
  int OH = delta_output_lt.get_plain_dims()[2];
  int OW = delta_output_lt.get_plain_dims()[3];
  // NPQK --> NKPQnk OR -> NKPQnk2n
  // full shape based on delta_output's reorder result
  std::vector<expr> temp_output_delta_shape_full = dtype_block > 1
    ? std::vector<expr> {BS / im_bs_block_, OC / im_oc_block_, OH, OW,
      im_bs_block_ / dtype_block, im_oc_block_, dtype_block}
    : std::vector<expr> {
      BS / im_bs_block_, OC / im_oc_block_, OH, OW, im_bs_block_, im_oc_block_};
  _tensor_(temp_output_delta, dtype, temp_output_delta_shape_full);
  _for_(i_ic, 0, ic_block / im_ic_block_) {
    // shrinked_shape
    std::vector<expr> temp_output_delta_shape_shr = dtype_block > 1
      ? std::vector<expr> {bs_block / im_bs_block_, oc_block / im_oc_block_,
        oh_block, ow_block, im_bs_block_ / dtype_block, im_oc_block_,
        dtype_block}
      : std::vector<expr> {bs_block / im_bs_block_, oc_block / im_oc_block_,
        oh_block, ow_block, im_bs_block_, im_oc_block_};
    // f32 --> vectorized; bf16 --> vnni_reorder
    std::vector<expr> shrink_offset = dtype_block > 1
      ? std::vector<expr> {obs_offset / im_bs_block_, oc_offset / im_oc_block_,
        oh_offset, ow_offset, obs_offset % im_bs_block_ / dtype_block,
        oc_offset % im_oc_block_, obs_offset % im_bs_block_ % dtype_block}
      : std::vector<expr> {obs_offset / im_bs_block_, oc_offset / im_oc_block_,
        oh_offset, ow_offset, obs_offset % im_bs_block_,
        oc_offset % im_oc_block_};

    _if_(i_ic == 0) {
      // reorder temp_output_delta
      int lanes = vectorize_step(ctx, get_B_dtype().type_code_, 32);
      if (oc_block < lanes || oc_block % lanes != 0 || dtype_block > 1) {
        lanes = 1;
      }
      temp_output_delta->attr()[tensor_shrinker_attrs::should_shrink]
        = tensor_shrinker_t::shrink_info_t {
          shrink_offset, temp_output_delta_shape_shr, stmts()};
      slice_range tmp_output_slice_range = dtype_block > 1
        ? slice_range {{obs_offset, bs_block / im_bs_block_},
          {oc_offset, oc_block / im_oc_block_}, {oh_offset, oh_block},
          {ow_offset, ow_block}, {0, im_bs_block_ / dtype_block},
          {0, im_oc_block_}, {0, dtype_block}}
        : slice_range {{obs_offset, bs_block / im_bs_block_},
          {oc_offset, oc_block / im_oc_block_}, {oh_offset, oh_block},
          {ow_offset, ow_block}, {0, im_bs_block_}, {0, im_oc_block_}};
      ops::commit_op(ctx, "reorder",
        {tensor_slice(delta_output,
          {{obs_offset, bs_block}, {oh_offset, oh_block}, {ow_offset, ow_block},
            {oc_offset, oc_block}})},
        {tensor_slice(temp_output_delta, std::move(tmp_output_slice_range))},
        {graph_tensor::make(delta_output_lt.get_plain_dims(),
          delta_output_lt.get_format(), delta_output_lt.dtype_)},
        {},
        {{"out_format",
          dtype_block > 1
            ? sc_data_format_t::NCHWnc2n(im_bs_block_, im_oc_block_)
            : sc_data_format_t::NCHWnc(im_bs_block_, im_oc_block_)}});
    }
    trace_guard_t trg(ctx, "brgemm");
    _for_(i_oc, 0, oc_block / im_oc_block_) {
      auto real_weight_idx = is_partial
        ? std::vector<expr> {temp_weight_idx[0], temp_weight_idx[1] + i_ic,
          temp_weight_idx[2] + i_oc, 0, 0, 0, 0}
        : std::vector<expr> {
          temp_weight_idx[0] + i_ic, temp_weight_idx[1] + i_oc, 0, 0, 0, 0};
      _for_(i_bs, 0, bs_block / im_bs_block_) {
        _for_(i_od, 0, od_block) {
          _for_(i_oh, 0, oh_block) {
            auto temp_output_delta_brgemm_index = dtype_block > 1
              ? std::vector<expr> {shrink_offset[0] + i_bs,
                shrink_offset[1] + i_oc, shrink_offset[2] + i_oh,
                shrink_offset[3], shrink_offset[4], shrink_offset[5],
                shrink_offset[6]}
              : std::vector<expr> {shrink_offset[0] + i_bs,
                shrink_offset[1] + i_oc, shrink_offset[2] + i_oh,
                shrink_offset[3], shrink_offset[4], shrink_offset[5]};
            _if_(o_bs == 0 && o_od == 0 && o_oh == 0 && o_ow == 0 && i_bs == 0
              && i_od == 0 && i_oh == 0) {
              // ic x bs matmul bs x oc
              builtin::brgemm_init_update(
                tensor_ptr(temp_forward_input,
                  {temp_forward_idx_non_block[0] + i_bs,
                    temp_forward_idx_non_block[1] + i_ic,
                    temp_forward_idx_non_block[2] + i_oh,
                    temp_forward_idx_non_block[3],
                    temp_forward_idx_non_block[4],
                    temp_forward_idx_non_block[5]}),
                tensor_ptr(temp_output_delta, temp_output_delta_brgemm_index),
                tensor_ptr(real_delta_weight_buf, real_weight_idx), ow_block,
                im_ic_block_, im_oc_block_, im_bs_block_, im_bs_block_,
                im_oc_block_, im_oc_block_, im_ic_block_ * im_bs_block_,
                im_oc_block_ * im_bs_block_, dtype, dtype);
            }
            _else_ {
              builtin::brgemm_update(tensor_ptr(temp_forward_input,
                                       {temp_forward_idx_non_block[0] + i_bs,
                                         temp_forward_idx_non_block[1] + i_ic,
                                         temp_forward_idx_non_block[2] + i_oh,
                                         temp_forward_idx_non_block[3],
                                         temp_forward_idx_non_block[4],
                                         temp_forward_idx_non_block[5]}),
                tensor_ptr(temp_output_delta, temp_output_delta_brgemm_index),
                tensor_ptr(real_delta_weight_buf, real_weight_idx), ow_block,
                im_ic_block_, im_oc_block_, im_bs_block_, im_bs_block_,
                im_oc_block_, im_oc_block_, im_ic_block_ * im_bs_block_,
                im_oc_block_ * im_bs_block_, dtype, dtype);
            }
          }
        }
      }
      if (fusion && !is_partial) {
        fusion->create_output_fusion_anchor({tensor_slice(real_delta_weight_buf,
          {{real_weight_idx[0], 1}, {real_weight_idx[1], 1}, {0, 1}, {0, 1},
            {0, im_ic_block_}, {0, im_oc_block_}})});
      }
    }
  }
}

bool gen_nested_conv1x1_backprop_weight_t::generate(context_ptr ctx,
  const nested_conv_bwd_weight_config_t &config, fusion_manager *fusion,
  const std::vector<expr> &inputs, const std::vector<expr> &outputs,
  std::vector<for_loop> &loops) const {
  int padding_h = padding_[0], padding_w = padding_[0];
  int padding_d = ndims_ == 5 ? padding_[0] : 0;
  if (padding_.size() > 1) {
    COMPILE_ASSERT((int)padding_.size() == ndims_ - 2,
      "padding length shall confirm with ndims.");
    padding_h = padding_[ndims_ - 4];
    padding_w = padding_[ndims_ - 3];
  }
  int stride_h = stride_[0], stride_w = stride_[0];
  int stride_d = ndims_ == 5 ? stride_[0] : 1;
  if (stride_.size() > 1) {
    COMPILE_ASSERT((int)stride_.size() == ndims_ - 2,
      "stride length shall confirm with ndims.");
    stride_h = stride_[ndims_ - 4];
    stride_w = stride_[ndims_ - 3];
  }
  bool has_padding = (padding_d > 0 || padding_h > 0 || padding_w > 0);
  bool has_stride = (stride_d > 1 || stride_h > 1 || stride_w > 1);

  COMPILE_ASSERT(!has_padding, "assert padding = 0");

  // setting dim values
  int BS = get_data_dims()[0], IC = get_data_dims()[1];
  int ID = ndims_ == 5 ? get_data_dims()[ndims_ - 3] : 1;
  int IH = get_data_dims()[ndims_ - 2], IW = get_data_dims()[ndims_ - 1];
  int OC = get_grad_dims()[1];
  int OD = ndims_ == 5 ? get_grad_dims()[ndims_ - 3] : 1;
  int OH = get_grad_dims()[2], OW = get_grad_dims()[3];
  int KD = ndims_ == 5 ? get_output_dims()[ndims_ - 3] : 1;
  int R = 1, S = 1;
  // setting configs
  int bs_threads = config.bs_threads, ic_threads = config.ic_threads,
      oc_threads = config.oc_threads, oh_threads = config.oh_threads,
      od_threads = config.od_threads;
  int oc_num_blocks
    = config.oc_num_blocks,
    ic_num_blocks = config.ic_num_blocks, bs_num_blocks = config.bs_num_blocks,
    oh_num_blocks = config.oh_num_blocks, od_num_blocks = config.od_num_blocks,
    ow_num_blocks = config.ow_num_blocks;
  bool is_partial = bs_threads * oh_threads * od_threads != 1;

  // other template related pre-compute values
  auto dtype = get_A_dtype();
  bool is_vnni_low_fp = ops::is_vnni_low_fp(ctx, dtype);
  int dtype_block = is_vnni_low_fp ? 2 : 1;
  int oc_single_core = OC / oc_threads;
  int ic_single_core = IC / ic_threads;
  int bs_single_core = BS / bs_threads;
  int od_single_core = OD / od_threads;
  int oh_single_core = OH / oh_threads;

  // define compute
  // delta_weight's blocking shape [IC / ic_block, OC / oc_block, R, S,
  // ic_block, oc_block]
  expr delta_weight = outputs.at(op_params_t::out_delta_weight),
       forward_input = inputs.at(op_params_t::in_forward_input),
       delta_output = inputs.at(op_params_t::in_delta_output);

  _tensor_(temp_delta_weight, datatypes::f32,
    {bs_threads * oh_threads * od_threads, IC / im_ic_block_, OC / im_oc_block_,
      1, 1, im_ic_block_, im_oc_block_});
  expr real_delta_weight_buf = is_partial ? temp_delta_weight : delta_weight;
  _for_(p_oc, 0, oc_threads, 1, for_type::PARALLEL, oc_threads) {
    _for_(p_ic, 0, ic_threads, 1, for_type::PARALLEL, ic_threads) {
      _for_(p_bs, 0, bs_threads, 1, for_type::PARALLEL, bs_threads) {
        _for_(p_od, 0, od_threads, 1, for_type::PARALLEL, od_threads) {
          _for_(p_oh, 0, oh_threads, 1, for_type::PARALLEL, oh_threads) {
            _for_(o_oc, 0, oc_num_blocks) {
              _for_(o_ic, 0, ic_num_blocks) {
                int oc_block = oc_single_core / oc_num_blocks;
                int ic_block = ic_single_core / ic_num_blocks;
                int bs_block = bs_single_core / bs_num_blocks;
                int od_block = od_single_core / od_num_blocks;
                int oh_block = oh_single_core / oh_num_blocks;
                int ow_block = OW / ow_num_blocks;
                expr oc_offset = p_oc * oc_single_core + o_oc * oc_block;
                expr ic_offset = p_ic * ic_single_core + o_ic * ic_block;
                _for_(o_bs, 0, bs_num_blocks) {
                  _for_(o_od, 0, od_num_blocks) {
                    _for_(o_oh, 0, oh_num_blocks) {
                      _for_(o_ow, 0, ow_num_blocks) {
                        expr obs_offset
                          = p_bs * bs_single_core + o_bs * bs_block;
                        expr oh_offset
                          = p_oh * oh_single_core + o_oh * oh_block;
                        expr ow_offset = o_ow * ow_block;
                        // start perform reorder:
                        // N[D]HWCn or N[D]HWC ->NC[D]HWcn
                        // TODO(zhangyan): consider 3D case
                        // full shape based on forward_input's reorder result
                        std::vector<expr> temp_forward_input_shape_full
                          = std::vector<expr> {BS / im_bs_block_,
                            IC / im_ic_block_, IH, IW, im_ic_block_,
                            im_bs_block_};
                        std::vector<expr> temp_forward_input_shape
                          = std::vector<expr> {bs_block / im_bs_block_,
                            ic_block / im_ic_block_, oh_block, ow_block,
                            im_ic_block_, im_bs_block_};
                        _tensor_(temp_forward_input, dtype,
                          has_stride ? temp_forward_input_shape
                                     : temp_forward_input_shape_full);
                        forward_input_reorder_call(ctx, temp_forward_input,
                          forward_input, in_tensors_[0], dtype, bs_block,
                          ic_block, oh_block, ow_block, IH, IW, obs_offset,
                          ic_offset, oh_offset, ow_offset, stride_h, stride_w);
                        std::vector<expr> temp_forward_idx_non_block
                          = has_stride
                          ? std::vector<expr> {0, 0, 0, 0, 0, 0}
                          : std::vector<expr> {obs_offset / im_bs_block_,
                            ic_offset / im_ic_block_, oh_offset, ow_offset,
                            obs_offset % im_bs_block_,
                            ic_offset % im_ic_block_};
                        std::vector<expr> temp_weight_idx = is_partial
                          ? std::vector<expr> {p_bs * oh_threads * od_threads
                              + p_od * oh_threads + p_oh,
                            ic_offset / im_ic_block_, oc_offset / im_oc_block_,
                            0, 0, im_ic_block_, im_oc_block_}
                          : std::vector<expr> {ic_offset / im_ic_block_,
                            oc_offset / im_oc_block_, 0, 0, im_ic_block_,
                            im_oc_block_};
                        inner_loop_call(ctx, temp_forward_input,
                          temp_forward_idx_non_block, in_tensors_[1],
                          delta_output, real_delta_weight_buf, temp_weight_idx,
                          dtype, dtype_block, ic_block, oc_block, bs_block,
                          od_block, oh_block, ow_block, stride_h, stride_w,
                          o_bs, o_od, o_oh, o_ow, obs_offset, oc_offset,
                          oh_offset, ow_offset, fusion, is_partial);
                      }
                    }
                  }
                }
                if (fusion && !is_partial) {
                  fusion->create_output_fusion_anchor({tensor_slice(
                    real_delta_weight_buf,
                    {{ic_offset / im_ic_block_, ic_block / im_ic_block_},
                      {oc_offset / im_oc_block_, oc_block / im_oc_block_},
                      {0, 1}, {0, 1}, {ic_offset % im_ic_block_, im_ic_block_},
                      {oc_offset % im_oc_block_, im_oc_block_}})});
                }
              }
            }
          }
        }
      }
      // final parallel reduce
      if (is_partial) {
        int lanes = vectorize_step(ctx, get_C_dtype().type_code_, 16);
        if (oc_single_core < lanes || oc_single_core % lanes != 0) {
          lanes = 1;
        }
        trace_guard_t trg(ctx, "final_reduce");
        _for_(r_parallel, 0, ic_single_core * oc_single_core / im_oc_block_, 1,
          for_type::PARALLEL, bs_threads * oh_threads * od_threads) {
          expr ic_block_idx = r_parallel % im_ic_block_;
          expr s_idx = r_parallel / im_ic_block_ % S;
          expr r_idx = r_parallel / im_ic_block_ / S % R;
          expr idx_tmp = r_parallel / im_ic_block_ / S / R;
          expr oc_outer_idx = idx_tmp % (oc_single_core / im_oc_block_);
          expr ic_outer_idx = idx_tmp / (oc_single_core / im_oc_block_)
            % (ic_single_core / im_ic_block_);
          _for_(r_reduce, 0, bs_threads * od_threads * oh_threads, 1) {
            _for_(r_oc_inner, 0, im_oc_block_, lanes) {
              std::vector<expr> delta_weight_idx {
                p_ic * ic_single_core / im_ic_block_ + ic_outer_idx,
                p_oc * oc_single_core / im_oc_block_ + oc_outer_idx, r_idx,
                s_idx, ic_block_idx, r_oc_inner};
              _if_(r_reduce == 0) {
                builtin::mem_zero(tensor_ptr(delta_weight, delta_weight_idx),
                  lanes, datatypes::f32);
              }
              std::vector<expr> temp_delta_weight_idx = delta_weight_idx;
              temp_delta_weight_idx.insert(
                temp_delta_weight_idx.begin(), r_reduce);
              delta_weight[span_t(delta_weight_idx, lanes)] = builder::make_add(
                delta_weight[span_t(delta_weight_idx, lanes)],
                temp_delta_weight[span_t(temp_delta_weight_idx, lanes)]);
            }
          }
          if (fusion) {
            fusion->create_output_fusion_anchor({tensor_slice(delta_weight,
              {{p_ic * ic_single_core / im_ic_block_ + ic_outer_idx, 1},
                {p_oc * oc_single_core / im_oc_block_ + oc_outer_idx, 1},
                {0, 1}, {0, 1}, {ic_block_idx, 1}, {0, im_oc_block_}})});
          }
        }
      }
    }
  }
  return true;
}
} // namespace ops
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
