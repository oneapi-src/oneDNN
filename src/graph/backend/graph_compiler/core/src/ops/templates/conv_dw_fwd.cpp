/*******************************************************************************
 * Copyright 2023-2024 Intel Corporation
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

#include "conv_dw_fwd.hpp"
#include <utility>
#include "utils.hpp"
#include <compiler/ir/builder.hpp>
#include <compiler/ir/builtin.hpp>
#include <compiler/ir/easy_build.hpp>
#include <compiler/ir/graph/fusion_anchor.hpp>
#include <compiler/ir/graph/mixed_partition.hpp>
#include <compiler/ir/graph/trait/configurable.hpp>
#include <ops/convolution.hpp>
#include <runtime/barrier.hpp>
#include <runtime/config.hpp>
#include <runtime/dynamic_dispatch/ops/config.hpp>
#include <runtime/dynamic_dispatch/utils.hpp>
#include <util/any_map.hpp>
#include <util/math_utils.hpp>
#include <util/reflection.hpp>
#include <util/utils.hpp>

using namespace dnnl::impl::graph::gc::builder;
namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

using ops::conv_dw_fwd_config_t;
// clang-format off
SC_CLASS(conv_dw_fwd_config_t)
  SC_FIELD(bs_threads)
  SC_FIELD(h_threads)
  SC_FIELD(w_threads)
  SC_FIELD(g_threads)
  SC_FIELD(h_block)
  SC_FIELD(w_block)
  SC_FIELD(g_block)
  SC_FIELD(im_h_block)
  SC_FIELD(im_w_block)
SC_CLASS_END();
// clang-format on

namespace ops {

config_ptr_vec gen_conv_dw_fwd_t::get_dynamic_config_candidates(
  const context_ptr &ctx) const {
  config_ptr_vec ret;
  return ret;
}

config_ptr gen_conv_dw_fwd_t::get_default_config(context_ptr ctx) const {
  auto ret = reflection::general_object_t::make<conv_dw_fwd_config_t>();
  conv_dw_fwd_config_t &cfg = *ret.unchecked_get_as<conv_dw_fwd_config_t>();

  const int num_threads = runtime_config_t::get().get_num_threads();
  auto thread_split = get_splits(num_threads);
  cfg.bs_threads = mb_ > num_threads
    ? num_threads
    : *(std::find_if(thread_split.rbegin(), thread_split.rend(),
      [&](int split) { return split == 1 || split < mb_; }));
  cfg.h_threads = num_threads / cfg.bs_threads;
  cfg.w_threads = 1;
  cfg.g_threads = 1;
  cfg.h_block = 1;
  cfg.w_block = ow_;
  cfg.g_block = groups_;
  cfg.im_h_block = 1;
  cfg.im_w_block = ow_;

  return std::move(ret);
}

gen_conv_dw_fwd_t::gen_conv_dw_fwd_t(sc_op *owner, const sc_dims &stride,
  const sc_dims &dilation, const sc_dims &pads_begin, const sc_dims &pads_end,
  std::vector<logical_tensor_t> &&ins, std::vector<logical_tensor_t> &&outs)
  : parent(owner, std::move(ins), std::move(outs)) {
  COMPILE_ASSERT(in_tensors_.size() == 2,
    "Wrong number of inputs, expected to be 2 but got " << in_tensors_.size()
                                                        << ".");
  COMPILE_ASSERT(out_tensors_.size() == 1,
    "Wrong number of output, expected to be 1 but got " << out_tensors_.size()
                                                        << ".");

  auto input_plain_dims = get_input_plain_dims();
  auto weight_plain_dims = get_weight_plain_dims();
  auto out_plain_dims = get_output_plain_dims();
  if (owner) { attrs_ = owner->attrs_; }
  COMPILE_ASSERT(
    utils::is_one_of(static_cast<int>(input_plain_dims.size()), 4, 5, 6),
    "Wrong input dims, expected to be  4D, 5D or 6D input, but got "
      << input_plain_dims.size() << "D.");
  COMPILE_ASSERT(
    utils::is_one_of(static_cast<int>(weight_plain_dims.size()), 4, 5, 6)
      && (weight_plain_dims.size() == input_plain_dims.size()),
    "Wrong weight dims, only support 4D 5D or 6D weights, but got "
      << weight_plain_dims.size() << "D.");
  COMPILE_ASSERT(
    utils::is_one_of(static_cast<int>(out_plain_dims.size()), 4, 5, 6)
      && (out_plain_dims.size() == input_plain_dims.size()),
    "Wrong output dims, only support 4D , 5D or 6D output, but got "
      << out_plain_dims.size() << "D.");

  ndims_ = input_plain_dims.size();
  is_3d_ = (ndims_ == 6);

  COMPILE_ASSERT(is_3d_
      ? utils::is_one_of(static_cast<int>(pads_begin.size()), 1, 3)
      : utils::is_one_of(static_cast<int>(pads_begin.size()), 1, 2),
    "Wrong pads_begin dims, should be 1D, 2D or 3D, but got "
      << pads_begin.size() << "D.");
  COMPILE_ASSERT(is_3d_
      ? utils::is_one_of(static_cast<int>(stride.size()), 1, 3)
      : utils::is_one_of(static_cast<int>(stride.size()), 1, 2),
    "Wrong stride dims, should be 1D, 2D or 3D, but got " << stride.size()
                                                          << "D.");
  COMPILE_ASSERT(is_3d_
      ? utils::is_one_of(static_cast<int>(dilation.size()), 1, 3)
      : utils::is_one_of(static_cast<int>(dilation.size()), 1, 2),
    "Wrong dilation dims, should be 1D, 2D or 3D, but got " << dilation.size()
                                                            << "D.");
  groups_ = static_cast<int>(attrs_.get_or_else("groups", 1));
  COMPILE_ASSERT(input_plain_dims[1] == weight_plain_dims[0]
      && input_plain_dims[1] == groups_,
    "expect input groups == weight groups, but got "
      << input_plain_dims[1] << " vs " << weight_plain_dims[0] << ".");
  COMPILE_ASSERT(
    input_plain_dims[2] == weight_plain_dims[2] && input_plain_dims[2] == 1,
    "expect input ic and weight ic equal  to 1");

  mb_ = input_plain_dims[0];
  ic_ = input_plain_dims[2];
  id_ = is_3d_ ? input_plain_dims[ndims_ - 3] : 1;
  ih_ = input_plain_dims[ndims_ - 2];
  iw_ = input_plain_dims[ndims_ - 1];
  oc_ = weight_plain_dims[1];
  kd_ = is_3d_ ? weight_plain_dims[ndims_ - 3] : 1;
  kh_ = weight_plain_dims[ndims_ - 2];
  kw_ = weight_plain_dims[ndims_ - 1];
  od_ = is_3d_ ? out_plain_dims[ndims_ - 3] : 1;
  oh_ = out_plain_dims[ndims_ - 2];
  ow_ = out_plain_dims[ndims_ - 1];
  pd_b_ = is_3d_ ? pads_begin[0] : 0;
  ph_b_ = pads_begin[0], pw_b_ = pads_begin[0];
  pd_e_ = is_3d_ ? pads_end[0] : 0;
  ph_e_ = pads_end[0], pw_e_ = pads_end[0];
  sd_ = is_3d_ ? stride[0] : 1;
  sh_ = stride[0], sw_ = stride[0];
  dd_ = is_3d_ ? dilation[0] : 1;
  dh_ = dilation[0], dw_ = dilation[0];

  if (pads_begin.size() > 1) {
    ph_b_ = pads_begin[ndims_ - 5];
    pw_b_ = pads_begin[ndims_ - 4];
  }
  if (pads_end.size() > 1) {
    ph_e_ = pads_end[ndims_ - 5];
    pw_e_ = pads_end[ndims_ - 4];
  }
  if (stride.size() > 1) {
    sh_ = stride[ndims_ - 5];
    sw_ = stride[ndims_ - 4];
  }
  if (dilation.size() > 1) {
    dh_ = dilation[ndims_ - 5];
    dw_ = dilation[ndims_ - 4];
  }
}

float gen_conv_dw_fwd_t::get_gflop() const {
  float result = (float)mb_ * groups_ * oc_ * 2.0 * ic_ * kd_ * kh_ * kw_ * od_
    * oh_ * ow_ / (float)1e9;
  return result;
}

#define CONV_ARG_LIST \
  const context_ptr &ctx, const conv_dw_fwd_config_t &config, \
    fusion_anchor_mgr_t *fusion, expr &output, const expr &input, \
    const expr &weight, std::vector<for_loop> &loops

void gen_conv_dw_fwd_t::dynamic_conv_logical_padding(CONV_ARG_LIST) const {
  COMPILE_ASSERT(false, "dynamic compute conv no padding is not supported!");
}

void gen_conv_dw_fwd_t::compute_conv_logical_padding(CONV_ARG_LIST) const {
  expr weight_k_start = 0;
  if (fusion) {
    auto mxp = fusion->get_binded_mxp();
    auto anchor = mxp->lookup_anchor_map(this->owner_, false);
    if (anchor) {
      auto slice = anchor->fsmap_.get(this->owner_->get_outputs()[0])[0];
      weight_k_start = slice.back().first;
    }
  }

  auto input_expr_dims = input.checked_as<tensor>()->dims_;
  int mb_expr_ = static_cast<uint64_t>(get_expr_as_int(input_expr_dims[0]));
  int g_expr_ = get_expr_as_int(input_expr_dims[input_expr_dims.size() - 2]);

  int bs_threads = config.bs_threads;
  int h_threads = config.h_threads;
  int w_threads = config.w_threads;
  int g_threads = config.g_threads;

  int h_block = config.h_block;
  int w_block = config.w_block;
  int g_block = g_expr_ == groups_ ? config.g_block : g_expr_;
  int im_h_block = config.im_h_block;
  int im_w_block = config.im_w_block;

  COMPILE_ASSERT(
    h_block % im_h_block == 0, "h_block % im_h_block != 0, config is invalid")
  COMPILE_ASSERT(
    w_block % im_w_block == 0, "w_block % im_w_block != 0, config is invalid")

  for_loop lpbs, lph, lpw, lpg;

  int h_num_block_pt, h_tail_num_block_pt, w_num_block_pt, w_tail_num_block_pt,
    g_num_block_pt, g_tail_num_block_pt;
  int oh_used_threads = block_split(utils::divide_and_ceil(oh_, h_block),
    h_threads, h_num_block_pt, h_tail_num_block_pt);
  int ow_used_threads = block_split(utils::divide_and_ceil(ow_, w_block),
    w_threads, w_num_block_pt, w_tail_num_block_pt);
  int g_used_threads = block_split(utils::divide_and_ceil(g_expr_, g_block),
    g_threads, g_num_block_pt, g_tail_num_block_pt);

  int LDA = sw_ * ic_ * g_expr_;
  int LDC = oc_ * g_expr_;

  auto nthreads = runtime_config_t::get().get_num_threads();
  bool parallel_space_is_enough
    = (mb_ % nthreads == 0 || utils::divide_and_ceil(mb_, nthreads) > 8);

  _named_for_(lpbs, pbs, 0, mb_expr_, 1, for_type::PARALLEL) {
    _named_for_(lph, ph, 0, oh_used_threads, 1) {
      _named_for_(lpw, pw, 0, ow_used_threads, 1) {
        _named_for_(lpg, pg, 0, g_used_threads, 1) {
          expr n = pbs;
          expr h_num_block = builder::make_select(ph < (oh_used_threads - 1),
                 h_num_block_pt, h_tail_num_block_pt),
               w_num_block = builder::make_select(pw < (ow_used_threads - 1),
                 w_num_block_pt, w_tail_num_block_pt),
               g_num_block = builder::make_select(pg < (g_used_threads - 1),
                 g_num_block_pt, g_tail_num_block_pt);
          // single core
          _for_(o_h, 0, h_num_block_pt) {
            _for_(o_w, 0, w_num_block_pt) {
              _for_(o_g, 0, g_num_block_pt) {
                expr cond
                  = o_h < h_num_block && o_w < w_num_block && o_g < g_num_block;
                _if_(cond) { // TODO(ciyong): maybe remove when it's dividable?
                  _for_(i_h, 0, h_block / im_h_block) {
                    expr h = (ph * h_num_block_pt + o_h) * h_block
                      + i_h * im_h_block;
                    _for_(i_w, 0, w_block / im_w_block) {
                      expr w = (pw * w_num_block_pt + o_w) * w_block
                        + i_w * im_w_block;
                      _if_(w < ow_) {
                        _tensor_(A_list, datatypes::pointer, {kh_ * kw_});
                        _tensor_(B_list, datatypes::pointer, {kh_ * kw_});
                        _tensor_(top_pad, datatypes::s32, {kh_ * kw_});
                        _tensor_(bottom_pad, datatypes::s32, {kh_ * kw_});
                        _for_(im_h_i, 0, im_h_block) {
                          _if_(h + im_h_i < oh_) {
                            _var_init_(cnt, datatypes::s32, 0);
                            _for_(r, 0, kh_) {
                              _for_(s, 0, kw_) {
                                auto ih = builder::make_cast(datatypes::s32,
                                            (h + im_h_i) * sh_ + r)
                                  - ph_b_;
                                auto iw = builder::make_cast(
                                            datatypes::s32, w * sw_ + s)
                                  - pw_b_;
                                _if_(ih >= 0 && ih < ih_) {
                                  top_pad[cnt] = make_select(iw < 0,
                                    divide_and_ceil((0 - iw), sw_), expr(0));
                                  bottom_pad[cnt] = make_select(
                                    iw + (im_w_block - 1) * sw_ + 1 > iw_,
                                    builder::make_cast(datatypes::s32,
                                      divide_and_ceil(
                                        (iw + (im_w_block - 1) * sw_ + 1) - iw_,
                                        sw_)),
                                    expr(0));
                                  std::vector<expr> input_pos
                                    = std::vector<expr> {n, ih, w * sw_ + s,
                                      (pg * g_num_block_pt + o_g) * g_block, 0};
                                  A_list[cnt] = tensor_ptr(input, input_pos);
                                  A_list[cnt]
                                    = builder::make_cast(datatypes::pointer,
                                      builder::make_cast(
                                        datatypes::index, A_list[cnt])
                                        - builder::make_cast(datatypes::index,
                                          pw_b_ * ic_ * g_expr_
                                            * utils::get_sizeof_type(
                                              get_input_dtype())));
                                  B_list[cnt] = tensor_ptr(weight,
                                    std::vector<expr> {r, s, 0,
                                      (pg * g_num_block_pt + o_g) * g_block
                                        + weight_k_start,
                                      0});
                                  cnt = cnt + 1;
                                }
                              }
                            }
                            std::vector<expr> output_pos
                              = std::vector<expr> {n, h + im_h_i, w,
                                (pg * g_num_block_pt + o_g) * g_block, 0};

                            sc_brgemm_attrs_t brg_attrs {
                              {brgemm::attr_key::hint_bs_group,
                                sw_ == 1 ? kw_ : 1},
                              {brgemm::attr_key::max_top_vpad,
                                utils::divide_and_ceil(pw_b_, sw_)},
                              {brgemm::attr_key::max_bottom_vpad,
                                utils::divide_and_ceil(pw_e_, sw_)}};
                            builtin::brgemm_init_list_update(A_list, B_list,
                              tensor_ptr(output, output_pos), 1, im_w_block,
                              g_block, -1, LDA, -1, LDC, 1 /*useless*/,
                              1 /*useless*/, cnt, get_input_dtype(),
                              get_weight_dtype(), brg_attrs,
                              sc_brgemm_bd_mask_t(), get_ir_zero_index(), 1,
                              top_pad, bottom_pad);

                            // im_w_block * g_block
                            create_fusion_anchor(fusion,
                              owner_->get_outputs()[0],
                              slice_range {{n, 1UL}, {h + im_h_i, 1},
                                {w, im_w_block},
                                {(pg * g_num_block_pt + o_g) * g_block,
                                  g_block},
                                {0, 1}});
                          }
                        }
                        // im_h_block * im_w_block * g_block
                        create_fusion_anchor(fusion, owner_->get_outputs()[0],
                          slice_range {{n, 1UL}, {h, im_h_block},
                            {w, im_w_block},
                            {(pg * g_num_block_pt + o_g) * g_block, g_block},
                            {0, 1}});
                      }
                    }
                    // im_h_block * w_block * g_block
                    create_fusion_anchor(fusion, owner_->get_outputs()[0],
                      slice_range {{n, 1UL}, {h, im_h_block},
                        {(pw * w_num_block_pt + o_w) * w_block, w_block},
                        {(pg * g_num_block_pt + o_g) * g_block, g_block},
                        {0, 1}});
                  }
                  // h_block * w_block * g_block
                  create_fusion_anchor(fusion, owner_->get_outputs()[0],
                    slice_range {{n, 1UL},
                      {(ph * h_num_block_pt + o_h) * h_block, h_block},
                      {(pw * w_num_block_pt + o_w) * w_block, w_block},
                      {(pg * g_num_block_pt + o_g) * g_block, g_block},
                      {0, 1}});
                }
              }
              // h_block * w_block * g_num_block_pt * g_block
              create_fusion_anchor(fusion, owner_->get_outputs()[0],
                slice_range {{n, 1UL},
                  {(ph * h_num_block_pt + o_h) * h_block, h_block},
                  {(pw * w_num_block_pt + o_w) * w_block, w_block},
                  {(pg * g_num_block_pt) * g_block, g_num_block_pt * g_block},
                  {0, 1}});
            }
            // h_block * w_num_block_pt * w_block * g_num_block_pt * g_block
            create_fusion_anchor(fusion, owner_->get_outputs()[0],
              slice_range {{n, 1UL},
                {(ph * h_num_block_pt + o_h) * h_block, h_block},
                {(pw * w_num_block_pt) * w_block, w_num_block_pt * w_block},
                {(pg * g_num_block_pt) * g_block, g_num_block_pt * g_block},
                {0, 1}});
          }
          // h_num_block_pt * h_block * w_num_block_pt * w_block *
          // g_num_block_pt * g_block
          create_fusion_anchor(fusion, owner_->get_outputs()[0],
            slice_range {{n, 1UL},
              {(ph * h_num_block_pt) * h_block, h_num_block_pt * h_block},
              {(pw * w_num_block_pt) * w_block, w_num_block_pt * w_block},
              {(pg * g_num_block_pt) * g_block, g_num_block_pt * g_block},
              {0, 1}});
        }
      }
    }
  }
  // loop axis bind(NGCHW)
  bind_loop_axis(owner_->get_outputs()[0], lpbs, 0);
  bind_loop_axis(owner_->get_outputs()[0], lph, 3);
  bind_loop_axis(owner_->get_outputs()[0], lpw, 4);
  bind_loop_axis(owner_->get_outputs()[0], lpg, 1);
  loops = {lpbs, lph, lpw, lpg};
}

void gen_conv_dw_fwd_t::compute_conv_physical_padding(CONV_ARG_LIST) const {
  expr weight_k_start = 0;
  if (fusion) {
    auto mxp = fusion->get_binded_mxp();
    auto anchor = mxp->lookup_anchor_map(this->owner_, false);
    if (anchor) {
      auto slice = anchor->fsmap_.get(this->owner_->get_outputs()[0])[0];
      weight_k_start = slice.back().first;
    }
  }
  for_loop ln, ld, lp, lg;
  const auto dtype_input = get_input_dtype();
  const auto dtype_weight = get_weight_dtype();
  const auto dtype_output = get_output_dtype();
  const int num_threads = runtime_config_t::get().get_num_threads();
  auto input_expr_dims = input.checked_as<tensor>()->dims_;
  int mb_expr_ = static_cast<uint64_t>(get_expr_as_int(input_expr_dims[0]));
  int g_expr_ = get_expr_as_int(input_expr_dims[input_expr_dims.size() - 2]);
  int g_block = g_expr_ == groups_ ? config.g_block : g_expr_;
  // no need to include groups for LDA as it's used by sub-tensor
  // instead of origin input tensor.
  int LDA = sw_ * ic_ * g_expr_;
  int LDC = oc_ * g_expr_;
  int aux_w_block_size = (config.im_w_block - 1) * sw_ + dw_ * (kw_ - 1) + 1;
  auto padding_value = attrs_.get_or_else("padding_value", 0);
  typedef enum { LEFT_PAD = 0, BOTH_PAD, RIGHT_PAD } pad_kind;

  // some shapes might have less pad than given at the end of current
  // axis
  auto get_num_pad_end = [](int ip, int k, int s, int p) {
    int remaining = (ip - k) % s;
    int num_pad_end = (remaining == 0)
      ? utils::divide_and_ceil(p, s)
      : ((p > remaining) ? utils::divide_and_ceil(p - remaining, s) : 0);
    return num_pad_end;
  };
  const int out_num_pad_top = utils::divide_and_ceil(ph_b_, sh_);
  const int out_num_pad_left = utils::divide_and_ceil(pw_b_, sw_);
  const int out_num_pad_front = is_3d_ ? utils::divide_and_ceil(pd_b_, sd_) : 0;
  const int out_num_pad_bottom
    = get_num_pad_end(ih_ + ph_b_ + ph_e_, dh_ * (kh_ - 1) + 1, sh_, ph_e_);
  const int out_num_pad_right
    = get_num_pad_end(iw_ + pw_b_ + pw_e_, dw_ * (kw_ - 1) + 1, sw_, pw_e_);
  const int out_num_pad_back = is_3d_
    ? get_num_pad_end(id_ + pd_b_ + pd_e_, dd_ * (kd_ - 1) + 1, sd_, pd_e_)
    : 0;

  const int out_nopad_top = out_num_pad_top;
  const int out_nopad_bottom = oh_ - out_num_pad_bottom - 1;
  const int out_nopad_left = out_num_pad_left;
  const int out_nopad_right = ow_ - out_num_pad_right - 1;
  const int out_nopad_front = is_3d_ ? out_num_pad_front : 0;
  const int out_nopad_back = is_3d_ ? od_ - out_num_pad_front - 1 : 1;
  const uint32_t lanes = get_lanes(ctx, g_block, dtype_input);

  // large pd and ph will be skipped for non-os-blocking approach.
  const bool large_pad = aux_w_block_size < pw_b_ || aux_w_block_size < pw_e_;

  int g_num_block = utils::divide_and_ceil(g_expr_, g_block);

  const int work_amount = mb_ * g_num_block * ow_ / config.im_w_block;
  bool reuse_aux_buffer = sh_ < (dh_ * (kh_ - 1) + 1) && g_num_block == 1
    && is_parallel_space_enough(work_amount, num_threads) && dh_ == 1;
  bool use_var_bs
    = attrs_.get_or_else("use_var_bs", true) && padding_value == 0;
  // utility function
  auto update_pad_idx =
    [](const expr &cur_o, const expr &cur_i, const int ker, const int dilation,
      const int in, const int nopad_begin, const int nopad_end,
      expr::lvalue_proxy_t &num_pad, expr::lvalue_proxy_t &pad_begin_idx,
      expr::lvalue_proxy_t &pad_end_idx, expr::lvalue_proxy_t &nopad_begin_idx,
      expr::lvalue_proxy_t &nopad_end_idx) {
      _if_((cur_o >= nopad_begin) && (cur_o <= nopad_end)) {
        num_pad = 0;
        pad_begin_idx = 0;
        pad_end_idx = 0;
        nopad_begin_idx = 0;
        nopad_end_idx = ker;
      }
      _else_ {
        _if_(cur_o < nopad_begin) {
          num_pad
            = (0 - builder::make_cast(datatypes::s32, cur_i) - 1) / dilation
            + 1;
          expr num_right_pad
            = divide_and_ceil(((builder::make_cast(datatypes::s32, cur_i)
                                + (ker - 1) * dilation + 1 - in)),
              dilation);
          pad_begin_idx = 0;
          pad_end_idx = num_pad;
          nopad_begin_idx = num_pad;
          nopad_end_idx = ker;
          _if_(num_right_pad > 0) { nopad_end_idx = ker - num_right_pad; }
        }
        _else_ {
          num_pad = divide_and_ceil(((builder::make_cast(datatypes::s32, cur_i)
                                      + (ker - 1) * dilation + 1 - in)),
            dilation);
          pad_begin_idx = ker - num_pad;
          pad_end_idx = ker;
          nopad_begin_idx = 0;
          nopad_end_idx = ker - num_pad;
        }
      }
    };

  // global tensor define
  _tensor_(pad_buffer, dtype_input, {aux_w_block_size, LDA});
  // thread shared var to hold stateful status
  _tensor_(global_aux_buffer, dtype_input,
    is_3d_ ? std::vector<expr> {num_threads, kd_, kh_, aux_w_block_size, LDA}
           : std::vector<expr> {num_threads, kh_, aux_w_block_size, LDA});
  _tensor_(global_cur_indices, datatypes::u32, {num_threads, kh_});
  _tensor_(global_init_state, datatypes::boolean, {num_threads});
  if (!use_var_bs) {
    // when not using var_bs, define a unified zero-buffer for
    // padding.
    builtin::brgemm_init(
      pad_buffer, aux_w_block_size, LDA, LDA, dtype_output, padding_value);
  }
  int outer_range
    = reuse_aux_buffer ? ow_ / config.im_w_block : oh_ / config.im_h_block;
  int inner_range
    = reuse_aux_buffer ? oh_ / config.im_h_block : ow_ / config.im_w_block;
  _named_for_(ln, n, 0, expr(mb_expr_), 1, for_type::PARALLEL) {
    _named_for_(lp, outer_var, 0, outer_range) {
      _named_for_(ld, d_o, 0, od_) {
        _var_init_(tid, datatypes::s32, builder::make_get_group_thread_id(-1));
        _tensor_(A_list, datatypes::pointer, {kd_ * kh_ * kw_});
        _tensor_(B_list, datatypes::pointer, {kd_ * kh_ * kw_});
        _var_(num_h_pad, datatypes::s32);
        _var_(h_pad_begin_idx, datatypes::index);
        _var_(h_pad_end_idx, datatypes::index);
        _var_(h_nopad_begin_idx, datatypes::index);
        _var_(h_nopad_end_idx, datatypes::index);
        _var_(num_d_pad, datatypes::s32);
        _var_(d_pad_begin_idx, datatypes::index);
        _var_(d_pad_end_idx, datatypes::index);
        _var_(d_nopad_begin_idx, datatypes::index);
        _var_(d_nopad_end_idx, datatypes::index);
        _var_(aux_buf_d, datatypes::index);
        _var_(aux_buf_h, datatypes::index);

        expr cur_od = d_o;
        auto cur_id = cur_od * sd_;
        // initialized stateful vars for each thread.
        if (reuse_aux_buffer) {
          _for_(gi, 0, kh_) {
            global_cur_indices[{tid, gi}]
              = builder::make_cast(datatypes::u32, gi);
          }
          global_init_state[tid] = true;
        }
        _for_(inner_var, 0, inner_range) {
          _var_init_(
            h_o, datatypes::index, reuse_aux_buffer ? inner_var : outer_var);
          _var_init_(
            w_o, datatypes::index, reuse_aux_buffer ? outer_var : inner_var);
          auto cur_ow_begin = w_o * config.im_w_block;
          auto cur_ow_end = cur_ow_begin + config.im_w_block - 1;
          auto cur_iw = cur_ow_begin * sw_ - pw_b_;
          _named_for_(lg, g, 0, g_num_block) {
            auto cur_g = g * g_block;
            _for_(h_i, 0, config.im_h_block) {
              auto cur_oh = h_o * config.im_h_block + h_i;
              auto cur_ih = cur_oh * sh_ - ph_b_;
              std::vector<expr> output_pos = is_3d_
                ? std::vector<expr> {n, cur_od, cur_oh, cur_ow_begin, cur_g, 0}
                : std::vector<expr> {n, cur_oh, cur_ow_begin, cur_g, 0};

              if (is_3d_) {
                update_pad_idx(cur_od, cur_id, kd_, dd_, id_, out_nopad_front,
                  out_nopad_back, num_d_pad, d_pad_begin_idx, d_pad_end_idx,
                  d_nopad_begin_idx, d_nopad_end_idx);
              }
              update_pad_idx(cur_oh, cur_ih, kh_, dh_, ih_, out_nopad_top,
                out_nopad_bottom, num_h_pad, h_pad_begin_idx, h_pad_end_idx,
                h_nopad_begin_idx, h_nopad_end_idx);

              auto zero_out_aux_buffer = [&]() {
                builtin::brgemm_init(tensor_ptr(output, output_pos),
                  config.im_w_block, g_block, LDC, dtype_output, padding_value);
              };

              auto process_tile_with_pad = [&](const expr &d_begin,
                                             const expr &d_end,
                                             const expr &h_begin,
                                             const expr &h_end,
                                             const expr &left_pad,
                                             const expr
                                               &w_block_size_without_pad,
                                             const pad_kind &kind,
                                             const expr &aux_buf_hi = 0,
                                             const bool &update_mode = false) {
                _for_(kd, d_begin, d_end) {
                  _for_(kh, h_begin, h_end) {
                    if (is_3d_) { aux_buf_d = kd - d_begin; }
                    aux_buf_h = update_mode ? aux_buf_hi : (kh - h_begin);
                    if (kind == LEFT_PAD || kind == BOTH_PAD) {
                      builtin::brgemm_init(
                        tensor_ptr(global_aux_buffer,
                          is_3d_ ? std::vector<expr> {tid, aux_buf_d, aux_buf_h,
                            0, 0}
                                 : std::vector<expr> {tid, aux_buf_h, 0, 0}),
                        builder::make_cast(datatypes::s32, left_pad), g_block,
                        LDA, dtype_input, padding_value);
                    }

                    // mapping dst to src_padded then
                    // mapping to original src to copy the
                    // origin elements.
                    _for_(aux_buf_w, left_pad, w_block_size_without_pad) {
                      _for_(k, 0, g_block, (int)lanes) {
                        if (is_3d_) {
                          global_aux_buffer[span_t(
                            {tid, aux_buf_d, aux_buf_h, aux_buf_w, k}, lanes)]
                            = input[span_t({n, cur_id + kd * d_nopad_begin_idx,
                                             cur_ih + kh * dh_,
                                             cur_iw + aux_buf_w, cur_g + k, 0},
                              lanes)];
                        } else {
                          global_aux_buffer[span_t(
                            {tid, aux_buf_h, aux_buf_w, k}, lanes)]
                            = input[span_t({n, cur_ih + kh * dh_,
                                             cur_iw + aux_buf_w, cur_g + k, 0},
                              lanes)];
                        }
                      }
                    }

                    if (kind == RIGHT_PAD || kind == BOTH_PAD) {
                      builtin::brgemm_init(
                        tensor_ptr(global_aux_buffer,
                          is_3d_ ? std::vector<expr> {tid, aux_buf_d, aux_buf_h,
                            w_block_size_without_pad, 0}
                                 : std::vector<expr> {tid, aux_buf_h,
                                   w_block_size_without_pad, 0}),
                        builder::make_cast(datatypes::s32,
                          aux_w_block_size - w_block_size_without_pad),
                        g_block, LDA, dtype_input, padding_value);
                    }

                    _for_(kw, 0, kw_) {
                      expr idx;
                      if (is_3d_) {
                        auto valid_kh
                          = (h_nopad_end_idx - h_nopad_begin_idx - 1) / dh_ + 1;
                        idx = builder::make_cast(datatypes::u32,
                          use_var_bs ? (
                            aux_buf_d * valid_kh * kw_ + aux_buf_h * kw_ + kw)
                                     : (kd * kh_ * kw_ + kh * kw_ + kw));
                      } else {
                        idx = builder::make_cast(datatypes::u32,
                          use_var_bs ? (aux_buf_h * kw_ + kw)
                                     : (kh * kw_ + kw));
                      }

                      // TODO(xxx): pack input for dilated
                      // conv
                      A_list[idx] = tensor_ptr(global_aux_buffer,
                        is_3d_
                          ? std::vector<expr> {tid, aux_buf_d, aux_buf_h,
                            kw * dw_, 0}
                          : std::vector<expr> {tid, aux_buf_h, kw * dw_, 0});
                    }
                  }
                }
              };

              auto fill_aux_buffer = [&](const expr &d_nopad_begin = 0,
                                       const expr &d_nopad_end = 1) {
                _if_(cur_ow_begin < out_nopad_left) {
                  _if_(out_nopad_right >= 0 && cur_ow_end <= out_nopad_right) {
                    // left pad only
                    expr real_l_pad
                      = 0 - builder::make_cast(datatypes::s32, cur_iw);
                    process_tile_with_pad(d_nopad_begin, d_nopad_end,
                      h_nopad_begin_idx, h_nopad_end_idx, real_l_pad,
                      aux_w_block_size, LEFT_PAD);
                  }
                  _else_ {
                    // both left and right pad
                    expr real_l_pad
                      = 0 - builder::make_cast(datatypes::s32, cur_iw);
                    expr real_r_pad = builder::make_cast(datatypes::s32,
                                        cur_iw + aux_w_block_size)
                      - iw_;
                    expr w_block_size_without_pad
                      = aux_w_block_size - real_r_pad;
                    process_tile_with_pad(d_nopad_begin, d_nopad_end,
                      h_nopad_begin_idx, h_nopad_end_idx, real_l_pad,
                      w_block_size_without_pad, BOTH_PAD);
                  }
                }
                _else_ {
                  // right pad only
                  expr real_r_pad = builder::make_cast(
                                      datatypes::s32, cur_iw + aux_w_block_size)
                    - iw_;
                  expr w_block_size_without_pad = aux_w_block_size - real_r_pad;
                  process_tile_with_pad(d_nopad_begin, d_nopad_end,
                    h_nopad_begin_idx, h_nopad_end_idx, 0,
                    w_block_size_without_pad, RIGHT_PAD);
                }
              };

              auto update_aux_buffer = [&]() {
                _tensor_(modified_indices, datatypes::index, {sh_});
                _var_(modified_idx, datatypes::index);
                _var_(actual_idx, datatypes::index);
                modified_idx = 0;
                _for_(idx, 0, kh_) {
                  expr prev_indices = global_cur_indices[{tid, idx}];
                  _if_(prev_indices < sh_) {
                    global_cur_indices[{tid, idx}] = prev_indices + kh_ - sh_;
                    modified_indices[modified_idx] = idx;
                    modified_idx = modified_idx + 1;
                  }
                  _else_ {
                    global_cur_indices[{tid, idx}] = prev_indices - sh_;
                  }
                }

                _for_(idx, 0, sh_) {
                  modified_idx = modified_indices[idx];
                  actual_idx = global_cur_indices[{tid, modified_idx}];
                  // update necessary row of sub-tensor
                  // according to actual_idx
                  _if_(cur_ow_begin < out_nopad_left) {
                    _if_(
                      out_nopad_right >= 0 && cur_ow_end <= out_nopad_right) {
                      // left pad only
                      expr real_l_pad
                        = 0 - builder::make_cast(datatypes::s32, cur_iw);
                      process_tile_with_pad(0, kd_, actual_idx, actual_idx + 1,
                        real_l_pad, aux_w_block_size, LEFT_PAD, modified_idx,
                        true);
                    }
                    _else_ {
                      // both left and right pad
                      expr real_l_pad
                        = 0 - builder::make_cast(datatypes::s32, cur_iw);
                      expr real_r_pad = builder::make_cast(datatypes::s32,
                                          cur_iw + aux_w_block_size)
                        - iw_;
                      expr w_block_size_without_pad
                        = aux_w_block_size - real_r_pad;
                      process_tile_with_pad(0, kd_, actual_idx, actual_idx + 1,
                        real_l_pad, w_block_size_without_pad, BOTH_PAD,
                        modified_idx, true);
                    }
                  }
                  _else_ {
                    // right pad only
                    expr real_r_pad = builder::make_cast(datatypes::s32,
                                        cur_iw + aux_w_block_size)
                      - iw_;
                    expr w_block_size_without_pad
                      = aux_w_block_size - real_r_pad;
                    process_tile_with_pad(0, kd_, actual_idx, actual_idx + 1, 0,
                      w_block_size_without_pad, RIGHT_PAD, modified_idx, true);
                  }
                }

                // update A_list with reusable sub-tensor
                // using cur_indices, no padding on depth or
                // height axis.
                _for_(kd, 0, kd_) {
                  _for_(kh, 0, kh_) {
                    _var_(aux_buf_idx, datatypes::index);
                    aux_buf_idx = builder::make_cast(
                      datatypes::index, global_cur_indices[{tid, kh}]);
                    _for_(kw, 0, kw_) {
                      _var_(A_idx, datatypes::u32);
                      if (is_3d_) {
                        A_idx = builder::make_cast(datatypes::u32,
                          kd * kh_ * kw_ + aux_buf_idx * kw_ + kw);
                        A_list[A_idx] = tensor_ptr(
                          global_aux_buffer, {tid, kd, kh, kw * dw_, 0});
                      } else {
                        A_idx = builder::make_cast(
                          datatypes::u32, aux_buf_idx * kw_ + kw);
                        A_list[A_idx] = tensor_ptr(
                          global_aux_buffer, {tid, kh, kw * dw_, 0});
                      }
                    }
                  }
                }
              };

              auto call_brgemm = [&](int valid_kh, int valid_kd = 1) {
                auto valid_ker_size = valid_kd * valid_kh * kw_;
                int M = config.im_w_block, K = g_block, N = g_block;
                auto hint_A_size = M * K * valid_ker_size;
                auto hint_B_size = K * N * valid_ker_size;
                auto hint_C_size = M * N;
                sc_brgemm_attrs_t brg_attrs {
                  {brgemm::attr_key::hint_bs_group, sw_ == 1 ? kw_ : 1},
                  {brgemm::attr_key::max_bs, valid_ker_size}};

                builtin::brgemm_init_list_update(A_list, B_list,
                  tensor_ptr(output, output_pos), 1, M, N, -1, sw_ * LDA, -1,
                  LDC, 1, 1, valid_ker_size, dtype_input, dtype_weight,
                  brg_attrs);
              };

              auto generate_var_bs
                = [](const std::function<void(int, int)> &func, int k, int o,
                    int s, int d, int p, int i, int valid_kd, expr &cur_pos) {
                    int valid_k;
                    auto current_builder = get_current_builder();
                    current_builder->push_scope();
                    func(k, valid_kd);
                    stmt else_stmt = current_builder->pop_scope();
                    for (auto pos = 0; pos < o; ++pos) {
                      auto pos_begin = pos * s - p;
                      valid_k = 0;
                      auto ker_pos = pos_begin;
                      for (auto ker = 0; ker < k; ker++) {
                        if (ker_pos >= 0 && ker_pos < i) { valid_k++; }
                        ker_pos += d;
                      }
                      if (valid_k < k && valid_k > 0) {
                        current_builder->push_scope();
                        func(valid_k, valid_kd);
                        auto then_stmt = current_builder->pop_scope();
                        auto cond = (cur_pos == pos);
                        else_stmt
                          = make_if_else_unattached(cond, then_stmt, else_stmt);
                      }
                    }
                    current_builder->emit(else_stmt);
                  };

              auto do_var_bs_for_2d = [&](const int kd, const int kh) {
                generate_var_bs(
                  call_brgemm, kh, oh_, sh_, dh_, ph_b_, ih_, kd, cur_oh);
              };

              if (is_3d_) {
                auto cond = large_pad
                  ? (((cur_iw + aux_w_block_size <= 0) || (cur_iw > iw_))
                    || (num_d_pad >= kd_ || num_h_pad >= kh_))
                  : (num_d_pad >= kd_ || num_h_pad >= kh_);
                _if_(cond && padding_value == 0) { zero_out_aux_buffer(); }
                _else_ {
                  // 1) fill A_list
                  if (!use_var_bs) {
                    _for_(kd, 0, kd_) {
                      // all zero feature map
                      _if_(kd >= d_pad_begin_idx && kd < d_pad_end_idx) {
                        _for_(kh, 0, kh_) {
                          _for_(kw, 0, kw_) {
                            expr idx = builder::make_cast(
                              datatypes::u32, kd * kh_ * kw_ + kh * kw_ + kw);
                            A_list[idx] = tensor_ptr(pad_buffer, {0, 0});
                          }
                        }
                      }
                      _else_ {
                        _for_(kh, h_pad_begin_idx, h_pad_end_idx) {
                          _for_(kw, 0, kw_) {
                            expr idx = builder::make_cast(
                              datatypes::u32, kd * kh_ * kw_ + kh * kw_ + kw);
                            A_list[idx] = tensor_ptr(pad_buffer, {0, 0});
                          }
                        }
                      }
                    }
                  }

                  _if_(cur_ow_begin >= out_nopad_left
                    && cur_ow_end <= out_nopad_right) {
                    // 1.1) The middle region which don't need
                    // to copy input rows but just refer to
                    // original input buffer.
                    _for_(kd, d_nopad_begin_idx, d_nopad_end_idx) {
                      _for_(kh, h_nopad_begin_idx, h_nopad_end_idx) {
                        _for_(kw, 0, kw_) {
                          auto valid_kh = h_nopad_end_idx - h_nopad_begin_idx;
                          expr idx = builder::make_cast(datatypes::u32,
                            use_var_bs
                              ? ((kd - d_nopad_begin_idx) * valid_kh * kw_
                                + (kh - h_nopad_begin_idx) * kw_ + kw)
                              : (kd * kh_ * kw_ + kh * kw_ + kw));
                          A_list[idx] = tensor_ptr(input,
                            std::vector<expr> {n, cur_id + kd * dd_,
                              cur_ih + kh * dh_, cur_iw + kw * dw_, cur_g, 0});
                        }
                      }
                    }
                  }
                  _else_ {
                    // 1.2)copy rows and do physical padding
                    if (!reuse_aux_buffer) {
                      fill_aux_buffer(d_nopad_begin_idx, d_nopad_end_idx);
                    } else {
                      _if_(num_d_pad > 0 || num_h_pad > 0
                        || global_init_state[tid]) {
                        _if_(num_d_pad == 0 && num_h_pad == 0) {
                          global_init_state[tid] = false;
                        }
                        fill_aux_buffer(d_nopad_begin_idx, d_nopad_end_idx);
                      }
                      _else_ {
                        // num_d_pad == 0 && num_h_pad == 0,
                        // reuse sub-tsr
                        update_aux_buffer();
                      }
                    }
                  }

                  // 2) fill B_list
                  if (use_var_bs) {
                    _for_(kd, d_nopad_begin_idx, d_nopad_end_idx) {
                      _for_(kh, h_nopad_begin_idx, h_nopad_end_idx) {
                        _for_(kw, 0, kw_) {
                          auto valid_kh = h_nopad_end_idx - h_nopad_begin_idx;
                          expr idx = builder::make_cast(datatypes::u32,
                            ((kd - d_nopad_begin_idx) * valid_kh * kw_
                              + (kh - h_nopad_begin_idx) * kw_ + kw));
                          B_list[idx] = tensor_ptr(weight,
                            std::vector<expr> {
                              kd, kh, kw, 0, cur_g + weight_k_start, 0});
                        }
                      }
                    }
                  } else {
                    _for_(kd, 0, kd_) {
                      _for_(kh, 0, kh_) {
                        _for_(kw, 0, kw_) {
                          expr idx = builder::make_cast(
                            datatypes::u32, kd * kh_ * kw_ + kh * kw_ + kw);
                          B_list[idx] = tensor_ptr(weight,
                            std::vector<expr> {
                              kd, kh, kw, 0, cur_g + weight_k_start, 0});
                        }
                      }
                    }
                  }

                  if (use_var_bs) {
                    // determine the exact value of var_bs for
                    // brgemm call, Ai & Bi are already
                    // fulfilled at this stage.
                    generate_var_bs(do_var_bs_for_2d, kd_, od_, sd_, dd_, pd_b_,
                      id_, kh_, cur_od);
                  } else {
                    call_brgemm(kh_, kd_);
                  }
                }
              } else {
                auto cond = large_pad
                  ? (((cur_iw + aux_w_block_size <= 0) || (cur_iw > iw_))
                    || (num_h_pad >= kh_))
                  : (num_h_pad >= kh_);
                _if_(cond && padding_value == 0) { zero_out_aux_buffer(); }
                _else_ {
                  auto fill_A_and_B_list = [&]() {
                    if (!use_var_bs) {
                      // Add zero-padding tensorptr to A_list
                      _for_(kh, h_pad_begin_idx, h_pad_end_idx) {
                        _for_(kw, 0, kw_) {
                          expr idx
                            = builder::make_cast(datatypes::u32, kh * kw_ + kw);
                          A_list[idx] = tensor_ptr(pad_buffer, {0, 0});
                        }
                      }

                      _if_(h_pad_begin_idx == 0 && h_nopad_end_idx < kh_) {
                        // Add zero-padding tensorptr to
                        // A_list
                        _for_(kh, h_nopad_end_idx, kh_) {
                          _for_(kw, 0, kw_) {
                            expr idx = builder::make_cast(
                              datatypes::u32, kh * kw_ + kw);
                            A_list[idx] = tensor_ptr(pad_buffer, {0, 0});
                          }
                        }
                      }
                    }
                    _if_(cur_ow_begin >= out_nopad_left
                      && cur_ow_end <= out_nopad_right) {
                      _for_(kh, h_nopad_begin_idx, h_nopad_end_idx) {
                        _for_(kw, 0, kw_) {
                          expr idx = builder::make_cast(datatypes::u32,
                            (use_var_bs ? (kh - h_nopad_begin_idx) : kh) * kw_
                              + kw);
                          A_list[idx] = tensor_ptr(input,
                            std::vector<expr> {n, cur_ih + kh * dh_,
                              cur_iw + kw * dw_, cur_g, 0});
                        }
                      }
                    }
                    _else_ {
                      // copy rows and do physical padding
                      if (!reuse_aux_buffer) {
                        fill_aux_buffer();
                      } else {
                        _if_(num_h_pad > 0 || global_init_state[tid]) {
                          _if_(num_h_pad == 0) {
                            global_init_state[tid] = false;
                          }
                          fill_aux_buffer();
                        }
                        _else_ { update_aux_buffer(); }
                      }
                    }

                    // 2) fill B_list
                    if (use_var_bs) {
                      _for_(kh, h_nopad_begin_idx, h_nopad_end_idx) {
                        _for_(kw, 0, kw_) {
                          expr idx = builder::make_cast(datatypes::u32,
                            (kh - h_nopad_begin_idx) * kw_ + kw);
                          auto weight_idx = std::vector<expr> {
                            kh, kw, 0UL, cur_g + weight_k_start, 0};
                          B_list[idx] = tensor_ptr(weight, weight_idx);
                        }
                      }
                    } else {
                      _for_(kh, 0, kh_) {
                        _for_(kw, 0, kw_) {
                          expr idx
                            = builder::make_cast(datatypes::u32, kh * kw_ + kw);
                          B_list[idx] = tensor_ptr(weight,
                            std::vector<expr> {
                              kh, kw, 0, cur_g + weight_k_start, 0});
                        }
                      }
                    }
                  };
                  fill_A_and_B_list();
                  if (use_var_bs) {
                    do_var_bs_for_2d(kd_, kh_);
                  } else {
                    call_brgemm(kh_);
                  }
                }
              }
              if (fusion) {
                // im_w_block * g_block
                if (is_3d_) {
                  create_fusion_anchor(fusion, owner_->get_outputs()[0],
                    slice_range {{n, 1}, {cur_od, 1}, {cur_oh, 1},
                      {cur_ow_begin, config.im_w_block}, {cur_g, g_block},
                      {0, 1}});
                } else {
                  create_fusion_anchor(fusion, owner_->get_outputs()[0],
                    slice_range {{n, 1}, {cur_oh, 1},
                      {cur_ow_begin, config.im_w_block}, {cur_g, g_block},
                      {0, 1}});
                }
              }
            }
            if (fusion) {
              // im_h_block * im_w_block * g_block
              if (is_3d_) {
                create_fusion_anchor(fusion, owner_->get_outputs()[0],
                  slice_range {{n, 1}, {cur_od, 1},
                    {h_o * config.im_h_block, config.im_h_block},
                    {cur_ow_begin, config.im_w_block}, {cur_g, g_block},
                    {0, 1}});
              } else {
                create_fusion_anchor(fusion, owner_->get_outputs()[0],
                  slice_range {{n, 1},
                    {h_o * config.im_h_block, config.im_h_block},
                    {cur_ow_begin, config.im_w_block}, {cur_g, g_block},
                    {0, 1}});
              }
            }
          }
          if (fusion) {
            // im_h_block * im_w_block * groups_
            if (is_3d_) {
              create_fusion_anchor(fusion, owner_->get_outputs()[0],
                slice_range {{n, 1}, {cur_od, 1},
                  {h_o * config.im_h_block, config.im_h_block},
                  {cur_ow_begin, config.im_w_block}, {0, g_expr_}, {0, 1}});
            } else {
              create_fusion_anchor(fusion, owner_->get_outputs()[0],
                slice_range {{n, 1},
                  {h_o * config.im_h_block, config.im_h_block},
                  {cur_ow_begin, config.im_w_block}, {0, g_expr_}, {0, 1}});
            }
          }
        }
        if (fusion) {
          auto h_start
            = reuse_aux_buffer ? expr(0UL) : outer_var * config.im_h_block;
          auto h_len = reuse_aux_buffer ? oh_ : config.im_h_block;
          auto w_start
            = reuse_aux_buffer ? outer_var * config.im_w_block : expr(0UL);
          auto w_len = reuse_aux_buffer ? config.im_w_block : ow_;
          // im_h_block * im_w_block * groups_
          if (is_3d_) {
            create_fusion_anchor(fusion, owner_->get_outputs()[0],
              slice_range {{n, 1}, {cur_od, 1}, {h_start, h_len},
                {w_start, w_len}, {0, g_expr_}, {0, 1}});
          } else {
            create_fusion_anchor(fusion, owner_->get_outputs()[0],
              slice_range {{n, 1}, {h_start, h_len}, {w_start, w_len},
                {0, g_expr_}, {0, 1}});
          }
        }
      }
      if (fusion) {
        auto h_start
          = reuse_aux_buffer ? expr(0UL) : outer_var * config.im_h_block;
        auto h_len = reuse_aux_buffer ? oh_ : config.im_h_block;
        auto w_start
          = reuse_aux_buffer ? outer_var * config.im_w_block : expr(0UL);
        auto w_len = reuse_aux_buffer ? config.im_w_block : ow_;
        // im_h_block * im_w_block * groups_
        if (is_3d_) {
          create_fusion_anchor(fusion, owner_->get_outputs()[0],
            slice_range {{n, 1}, {0, od_}, {h_start, h_len}, {w_start, w_len},
              {0, g_expr_}, {0, 1}});
        } else {
          create_fusion_anchor(fusion, owner_->get_outputs()[0],
            slice_range {{n, 1}, {h_start, h_len}, {w_start, w_len},
              {0, g_expr_}, {0, 1}});
        }
      }
    }
    if (fusion) {
      // im_h_block * im_w_block * groups_
      if (is_3d_) {
        create_fusion_anchor(fusion, owner_->get_outputs()[0],
          slice_range {
            {n, 1}, {0, od_}, {0, oh_}, {0, ow_}, {0, g_expr_}, {0, 1}});
      } else {
        create_fusion_anchor(fusion, owner_->get_outputs()[0],
          slice_range {{n, 1}, {0, oh_}, {0, ow_}, {0, g_expr_}, {0, 1}});
      }
    }
  }
  bind_loop_axis(owner_->get_outputs()[0], ln, 0);
  bind_loop_axis(owner_->get_outputs()[0], ld,
    is_3d_ ? std::vector<int> {3} : std::vector<int> {});
  bind_loop_axis(owner_->get_outputs()[0], lp, is_3d_ + reuse_aux_buffer + 3);
  bind_loop_axis(owner_->get_outputs()[0], lg, 1);
  loops = {ln, ld, lp, lg};
}

void gen_conv_dw_fwd_t::dynamic_conv_physical_padding(CONV_ARG_LIST) const {
  COMPILE_ASSERT(
    false, "dynamic compute conv with padding is not yet supported!");
}

void gen_conv_dw_fwd_t::schedule_loops(context_ptr ctx,
  const conv_dw_fwd_config_t &config, stmt body,
  std::vector<for_loop> &fors) const {
  if (!is_dynamic()) {
    COMPILE_ASSERT(static_cast<int>(fors.size()) == 4,
      "expected to have 4 for loops, but got " << fors.size() << " for loops.");
    auto lpbs = fors[0], lph = fors[1], lpw = fors[2], lpg = fors[3];
    auto outer = lpbs->fuse(lph)->fuse(lpw)->fuse(lpg);
    outer->kind_ = for_type::PARALLEL;
  }
}

bool gen_conv_dw_fwd_t::generate(context_ptr ctx,
  const conv_dw_fwd_config_t &config, fusion_anchor_mgr_t *fusion,
  const std::vector<expr> &inputs, const std::vector<expr> &outputs,
  std::vector<for_loop> &loops) const {
  COMPILE_ASSERT(inputs.size() == 2,
    "Expecting 2 inputs for conv, but got " << inputs.size() << " inputs.");
  COMPILE_ASSERT(outputs.size() == 1,
    "Expecting 1 output for conv, but got " << outputs.size() << " output.");

  auto dtypeInput = get_input_dtype();
  auto dtypeWeight = get_weight_dtype();
  auto dtypeOutput = get_output_dtype();
  if (dtypeInput == datatypes::bf16) {
    COMPILE_ASSERT((dtypeWeight == datatypes::bf16),
      "Weights should be bf16 as "
      "data, the mixed datatypes is not supported yet!");
    COMPILE_ASSERT((dtypeOutput == datatypes::f32),
      "Output should be f32 when data and weights are in bf16.");
  }
  if (dtypeInput == datatypes::f16) {
    COMPILE_ASSERT((dtypeWeight == datatypes::f16),
      "Weights should be f16 as "
      "data, the mixed datatypes is not supported yet!");
    COMPILE_ASSERT((dtypeOutput == datatypes::f32),
      "Output should be f32 when data and weights are in f16.");
  }
  if (utils::is_one_of(dtypeInput, datatypes::s8, datatypes::u8)) {
    COMPILE_ASSERT((dtypeWeight == datatypes::s8),
      "Weights should be s8 when \
            data is s8/u8, the mixed datatypes is not supported yet!");
    COMPILE_ASSERT((dtypeOutput == datatypes::s32),
      "Output should be s32 when data and weights are in "
      "s8/u8.");
  }

  if (!is_dynamic()) {
    COMPILE_ASSERT((config.im_h_block > 0) && (oh_ % config.im_h_block == 0),
      "oh should be dividable by im_h_block, but got oh="
        << oh_ << " im_h_block=" << config.im_h_block << ".");
    COMPILE_ASSERT((config.im_w_block > 0) && (ow_ % config.im_w_block == 0),
      "ow should be dividable by im_w_block, but got ow="
        << ow_ << " im_w_block=" << config.im_w_block << ".");
  } else {
    COMPILE_ASSERT(false, "dynamic conv is not supported yet!");
  }

  expr output = outputs[op_params_t::out];
  expr input = inputs[op_params_t::data];
  expr weight = inputs[op_params_t::weight];

  if (!is_3d_ && attrs_.get_or_else("padding_value", 0) == 0) {
    COMPILE_ASSERT(!is_3d_, "conv dw fwd does not support 3d yet.");
    if (is_dynamic()) {
      dynamic_conv_logical_padding(
        ctx, config, fusion, output, input, weight, loops);
    } else {
      compute_conv_logical_padding(
        ctx, config, fusion, output, input, weight, loops);
    }
  } else {
    if (is_dynamic()) {
      dynamic_conv_physical_padding(
        ctx, config, fusion, output, input, weight, loops);
    } else {
      compute_conv_physical_padding(
        ctx, config, fusion, output, input, weight, loops);
    }
  }
  return true;
}
#undef CONV_ARG_LIST

} // namespace ops
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
