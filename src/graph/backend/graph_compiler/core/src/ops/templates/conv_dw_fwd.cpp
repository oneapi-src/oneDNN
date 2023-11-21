/*******************************************************************************
 * Copyright 2023 Intel Corporation
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

  cfg.h_block = oh_;
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
    utils::is_one_of(static_cast<int>(input_plain_dims.size()), 3, 4, 5),
    "Wrong input dims, expected to be 3D, 4D or 5D input, but got "
      << input_plain_dims.size() << "D.");
  COMPILE_ASSERT(
    utils::is_one_of(static_cast<int>(weight_plain_dims.size()), 3, 4, 5)
      && (weight_plain_dims.size() == input_plain_dims.size()),
    "Wrong weight dims, only support 3D, 4D or 5D weights, but got "
      << weight_plain_dims.size() << "D.");
  COMPILE_ASSERT(
    utils::is_one_of(static_cast<int>(out_plain_dims.size()), 3, 4, 5)
      && (out_plain_dims.size() == input_plain_dims.size()),
    "Wrong output dims, only support 3D, 4D or 5D weights, but got "
      << out_plain_dims.size() << "D.");

  ndims_ = input_plain_dims.size();
  is_3d_ = (ndims_ == 5);

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
  COMPILE_ASSERT(input_plain_dims[1] / groups_ == weight_plain_dims[1],
    "expect input_plain_dims[1] / groups == weight_plain_dims[1], but got "
      << input_plain_dims[1] / groups_ << " vs " << weight_plain_dims[1]
      << ".");

  mb_ = input_plain_dims[0];
  ic_ = input_plain_dims[1] / groups_;
  id_ = is_3d_ ? input_plain_dims[2] : 1;
  ih_ = input_plain_dims[ndims_ - 2];
  iw_ = input_plain_dims[ndims_ - 1];
  oc_ = weight_plain_dims[0] / groups_;
  kd_ = is_3d_ ? weight_plain_dims[2] : 1;
  kh_ = weight_plain_dims[ndims_ - 2];
  kw_ = weight_plain_dims[ndims_ - 1];
  od_ = is_3d_ ? out_plain_dims[2] : 1;
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
    ph_b_ = pads_begin[ndims_ - 4];
    pw_b_ = pads_begin[ndims_ - 3];
  }
  if (pads_end.size() > 1) {
    ph_e_ = pads_end[ndims_ - 4];
    pw_e_ = pads_end[ndims_ - 3];
  }
  if (stride.size() > 1) {
    sh_ = stride[ndims_ - 4];
    sw_ = stride[ndims_ - 3];
  }
  if (dilation.size() > 1) {
    dh_ = dilation[ndims_ - 4];
    dw_ = dilation[ndims_ - 3];
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

void gen_conv_dw_fwd_t::dynamic_compute_conv_no_padding(CONV_ARG_LIST) const {
  COMPILE_ASSERT(false, "dynamic compute conv no padding is not supported!");
}

void gen_conv_dw_fwd_t::compute_conv_no_padding(CONV_ARG_LIST) const {
  int bs_threads = config.bs_threads;
  int h_threads = config.h_threads;
  int w_threads = config.w_threads;
  int g_threads = config.g_threads;

  int h_block = config.h_block;
  int w_block = config.w_block;
  int g_block = config.g_block;
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
  int g_used_threads = block_split(utils::divide_and_ceil(groups_, g_block),
    g_threads, g_num_block_pt, g_tail_num_block_pt);

  auto input_expr_dims = input.checked_as<tensor>()->dims_;
  auto mb_expr_ = input_expr_dims[0];

  auto LDA = sw_ * ic_ * groups_;
  auto LDC = oc_ * groups_;

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

                        _for_(im_h_i, 0, im_h_block) {
                          _if_(h + im_h_i < oh_) {
                            _for_(r, 0, kh_) {
                              _for_(s, 0, kw_) {
                                auto idx = r * kw_ + s;
                                std::vector<expr> input_pos
                                  = std::vector<expr> {n,
                                    (h + im_h_i) * sh_ + r, w * sw_ + s,
                                    (pg * g_num_block_pt + o_g) * g_block};

                                A_list[idx] = tensor_ptr(input, input_pos);
                                B_list[idx] = tensor_ptr(weight,
                                  std::vector<expr> {r, s, 0,
                                    (pg * g_num_block_pt + o_g) * g_block});
                              }
                            }
                            std::vector<expr> output_pos
                              = std::vector<expr> {n, h + im_h_i, w,
                                (pg * g_num_block_pt + o_g) * g_block};
                            sc_brgemm_attrs_t brg_attrs {
                              {brgemm::attr_key::max_bs, kw_ * kh_},
                              {brgemm::attr_key::bs_group, 1}};

                            builtin::brgemm_init_list_update(A_list, B_list,
                              tensor_ptr(output, output_pos), 1, im_w_block,
                              g_block, -1, LDA, -1, LDC, 1 /*useless*/,
                              1 /*useless*/, kh_ * kw_, get_input_dtype(),
                              get_weight_dtype(), brg_attrs);

                            // im_w_block * g_block
                            create_fusion_anchor(fusion,
                              owner_->get_outputs()[0],
                              slice_range {{n, 1UL}, {h + im_h_i, 1},
                                {w, im_w_block},
                                {(pg * g_num_block_pt + o_g) * g_block,
                                  g_block}});
                          }
                        }
                        // im_h_block * im_w_block * g_block
                        create_fusion_anchor(fusion, owner_->get_outputs()[0],
                          slice_range {{n, 1UL}, {h, im_h_block},
                            {w, im_w_block},
                            {(pg * g_num_block_pt + o_g) * g_block, g_block}});
                      }
                    }
                    // im_h_block * w_block * g_block
                    create_fusion_anchor(fusion, owner_->get_outputs()[0],
                      slice_range {{n, 1UL}, {h, im_h_block},
                        {(pw * w_num_block_pt + o_w) * w_block, w_block},
                        {(pg * g_num_block_pt + o_g) * g_block, g_block}});
                  }
                  // h_block * w_block * g_block
                  create_fusion_anchor(fusion, owner_->get_outputs()[0],
                    slice_range {{n, 1UL},
                      {(ph * h_num_block_pt + o_h) * h_block, h_block},
                      {(pw * w_num_block_pt + o_w) * w_block, w_block},
                      {(pg * g_num_block_pt + o_g) * g_block, g_block}});
                }
              }
              // h_block * w_block * g_num_block_pt * g_block
              create_fusion_anchor(fusion, owner_->get_outputs()[0],
                slice_range {{n, 1UL},
                  {(ph * h_num_block_pt + o_h) * h_block, h_block},
                  {(pw * w_num_block_pt + o_w) * w_block, w_block},
                  {(pg * g_num_block_pt) * g_block, g_num_block_pt * g_block}});
            }
            // h_block * w_num_block_pt * w_block * g_num_block_pt * g_block
            create_fusion_anchor(fusion, owner_->get_outputs()[0],
              slice_range {{n, 1UL},
                {(ph * h_num_block_pt + o_h) * h_block, h_block},
                {(pw * w_num_block_pt) * w_block, w_num_block_pt * w_block},
                {(pg * g_num_block_pt) * g_block, g_num_block_pt * g_block}});
          }
          // h_num_block_pt * h_block * w_num_block_pt * w_block *
          // g_num_block_pt * g_block
          create_fusion_anchor(fusion, owner_->get_outputs()[0],
            slice_range {{n, 1UL},
              {(ph * h_num_block_pt) * h_block, h_num_block_pt * h_block},
              {(pw * w_num_block_pt) * w_block, w_num_block_pt * w_block},
              {(pg * g_num_block_pt) * g_block, g_num_block_pt * g_block}});
        }
      }
    }
  }

  loops = {lpbs, lph, lpw, lpg};
}

void gen_conv_dw_fwd_t::compute_conv_padding(CONV_ARG_LIST) const {
  COMPILE_ASSERT(false, "compute conv with padding is not yet supported!");
}

void gen_conv_dw_fwd_t::dynamic_compute_conv_padding(CONV_ARG_LIST) const {
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
      "ow should be dividable by tile_q, but got ow="
        << ow_ << " im_w_block=" << config.im_w_block << ".");
  } else {
    COMPILE_ASSERT(false, "dynamic conv is not supported yet!");
  }

  expr output = outputs[op_params_t::out];
  expr input = inputs[op_params_t::data];
  expr weight = inputs[op_params_t::weight];

  if (pd_b_ == 0 && ph_b_ == 0 && pw_b_ == 0 && pd_e_ == 0 && ph_e_ == 0
    && pw_e_ == 0) {
    COMPILE_ASSERT(!is_3d_, "conv dw fwd does not support 3d yet.");
    if (is_dynamic()) {
      dynamic_compute_conv_no_padding(
        ctx, config, fusion, output, input, weight, loops);
    } else {
      compute_conv_no_padding(
        ctx, config, fusion, output, input, weight, loops);
    }
  } else {
    if (is_dynamic()) {
      dynamic_compute_conv_padding(
        ctx, config, fusion, output, input, weight, loops);
    } else {
      compute_conv_padding(ctx, config, fusion, output, input, weight, loops);
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
