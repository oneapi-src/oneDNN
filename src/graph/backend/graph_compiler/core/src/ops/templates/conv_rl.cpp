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
#include "conv_rl.hpp"
#include <algorithm>
#include <utility>
#include "utils.hpp"
#include <compiler/ir/builder.hpp>
#include <compiler/ir/builtin.hpp>
#include <compiler/ir/easy_build.hpp>
#include <compiler/ir/graph/binding_axis.hpp>
#include <compiler/ir/graph/fusion_anchor.hpp>
#include <runtime/config.hpp>
#include <util/any_map.hpp>
#include <util/reflection.hpp>
#include <util/utils.hpp>
using namespace dnnl::impl::graph::gc::builder;
SC_MODULE(conv_rl)

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

using ops::conv_fwd_rl_config_t;
// clang-format off
SC_CLASS(conv_fwd_rl_config_t)
  SC_FIELD(brgemm_m)
  SC_FIELD(brgemm_n)
SC_CLASS_END();
// clang-format on

namespace ops {
static constexpr int cache_line_size = 64;

config_ptr gen_conv_fwd_rl_t::get_default_config(context_ptr ctx) const {
  auto ret = reflection::general_object_t::make<conv_fwd_rl_config_t>();
  conv_fwd_rl_config_t &cfg = *ret.unchecked_get_as<conv_fwd_rl_config_t>();
  cfg.brgemm_m = ow_;
  std::vector<int> brgemm_m_candidates = {32, 16};
  bool has_proper_brgemm_m = false;
  for (auto &c : brgemm_m_candidates) {
    if (ow_ % c == 0) {
      cfg.brgemm_m = c;
      has_proper_brgemm_m = true;
      break;
    }
  }
  if (!has_proper_brgemm_m) {
    cfg.brgemm_m = utils::get_blocks(ow_, 16, 64).back();
  }
  cfg.brgemm_n = utils::get_blocks(oc_, 16).back();

  return std::move(ret);
}

std::vector<expr> gen_conv_fwd_rl_t::data_offset(const expr &N, const expr &G,
  const expr &C, const expr &D, const expr &H, const expr &W,
  const expr &C_block, const expr &c_idx) const {
  return is_group_conv_
    ? (!blocking_input_ ? std::vector<expr> {N, H, W, G, C * C_block + c_idx}
                        : std::vector<expr> {N, G, C, H, W, c_idx})
    : (!blocking_input_ ? std::vector<expr> {N, H, W, C * C_block + c_idx}
                        : std::vector<expr> {N, C, H, W, c_idx});
}

std::vector<expr> gen_conv_fwd_rl_t::output_offset(const expr &N, const expr &G,
  const expr &C, const expr &D, const expr &H, const expr &W,
  const expr &C_block, const expr &c_idx) const {
  return is_group_conv_
    ? (!blocking_output_ ? std::vector<expr> {N, H, W, G, C * C_block + c_idx}
                         : std::vector<expr> {N, G, C, H, W, c_idx})
    : (!blocking_output_ ? std::vector<expr> {N, H, W, C * C_block + c_idx}
                         : std::vector<expr> {N, C, H, W, c_idx});
}

void gen_conv_fwd_rl_t::create_anchor(fusion_anchor_mgr_t *fusion,
  const graph_tensor_ptr &output_gt, const expr &n, const int n_len,
  const expr &g, const expr &g_len, const expr &k, const int k_len,
  const expr &d, const int d_len, const expr &p, const expr &p_len,
  const expr &q, const int q_len, const int K_block) const {
  if (fusion) {
    if (is_group_conv_) {
      fusion->create_fusion_anchor(slice_map {{output_gt.get(),
        blocking_output_
          ? slice_range_list {{{n, n_len}, {g, g_len}, {k, k_len}, {p, p_len},
            {q, q_len}, {0, K_block}}}
          : slice_range_list {{{n, n_len}, {p, p_len}, {q, q_len}, {g, g_len},
            {k * K_block, k_len * K_block}}}}});

    } else {
      fusion->create_fusion_anchor(slice_map {{output_gt.get(),
        blocking_output_ ? slice_range_list {{{n, n_len}, {k, k_len},
          {p, p_len}, {q, q_len}, {0, K_block}}}
                         : slice_range_list {{{n, n_len}, {p, p_len},
                           {q, q_len}, {k * K_block, k_len * K_block}}}}});
    }
  }
}

void gen_conv_fwd_rl_t::validate_default_config(
  const context_ptr &ctx, conv_fwd_rl_config_t &cfg) const {}

float gen_conv_fwd_rl_t::get_gflop() const {
  float result
    = (float)mb_ * oc_ * 2.0 * ic_ * kh_ * kw_ * oh_ * ow_ / (float)1e9;
  return result;
}

gen_conv_fwd_rl_t::gen_conv_fwd_rl_t(sc_op *owner, const sc_dims &stride,
  const sc_dims &dilations, const sc_dims &pads_begin, const sc_dims &pads_end,
  std::vector<logical_tensor_t> &&ins, std::vector<logical_tensor_t> &&outs)
  : parent(owner, std::move(ins), std::move(outs)) {
  if (owner) { attrs_ = owner->attrs_; }
  COMPILE_ASSERT(attrs_.has_key("use_rl"), "expected to have 'use_rl' attrs");
  COMPILE_ASSERT(in_tensors_.size() == 2,
    "Wrong number of inputs, expected to be 2 but got " << in_tensors_.size()
                                                        << ".");
  COMPILE_ASSERT(out_tensors_.size() == 1,
    "Wrong number of output, expected to be 1 but got " << out_tensors_.size()
                                                        << ".");
  auto no_dilation = std::all_of(
    dilations.begin(), dilations.end(), [](const int d) { return d == 1; });
  COMPILE_ASSERT(no_dilation, "conv with dilation is not supported yet!");

  auto input_plain_dims = get_input_plain_dims();
  auto weight_plain_dims = attrs_.get<sc_dims>("origin_wei_plain_dims");
  auto out_plain_dims = get_output_plain_dims();

  ndims_ = input_plain_dims.size();
  groups_ = static_cast<int>(attrs_.get_or_else("groups", 1));
  is_group_conv_ = groups_ > 1;
  COMPILE_ASSERT(ndims_ == 4UL + is_group_conv_,
    "reduce lowering currently only support 4D input!")
  COMPILE_ASSERT(weight_plain_dims.size() == ndims_,
    "Wrong weight dims, only support 4D weights, but got "
      << weight_plain_dims.size() << "D.");

  COMPILE_ASSERT(input_plain_dims[1 + is_group_conv_]
      == weight_plain_dims[1 + is_group_conv_],
    "expect ic == kic, but got "
      << input_plain_dims[1 + is_group_conv_] << " vs "
      << weight_plain_dims[1 + is_group_conv_] << ".");

  mb_ = input_plain_dims[0];
  ic_ = input_plain_dims[1 + is_group_conv_];
  ih_ = input_plain_dims[2 + is_group_conv_];
  iw_ = input_plain_dims[3 + is_group_conv_];

  oc_ = weight_plain_dims[is_group_conv_];
  kh_ = weight_plain_dims[2 + is_group_conv_];
  kw_ = weight_plain_dims[3 + is_group_conv_];
  oh_ = out_plain_dims[2 + is_group_conv_];
  ow_ = out_plain_dims[3 + is_group_conv_];
  pt_ = pads_begin[0], pb_ = pads_begin[0];
  pl_ = pads_end[0], pr_ = pads_end[0];
  if (pads_begin.size() > 1) {
    pt_ = pads_begin[ndims_ - 4 - is_group_conv_];
    pl_ = pads_begin[ndims_ - 3 - is_group_conv_];
  }
  if (pads_end.size() > 1) {
    pb_ = pads_end[ndims_ - 4 - is_group_conv_];
    pr_ = pads_end[ndims_ - 3 - is_group_conv_];
  }

  sh_ = stride[0], sw_ = stride[0];
  if (stride.size() > 1) {
    sh_ = stride[ndims_ - 4 - is_group_conv_];
    sw_ = stride[ndims_ - 3 - is_group_conv_];
  }

  LDA_ = kh_ * ic_ * sw_;
  actual_iw_ = (ow_ - 1) * sw_ + kw_;
  actual_ih_ = (oh_ - 1) * sh_ + kh_;

  COMPILE_ASSERT(pt_ <= kh_ && pb_ <= kh_ && pl_ <= kw_ && pr_ <= kw_,
    "Not support the case of padding > filter_size!");
  int num_threads = runtime_config_t::get().get_num_threads();
  int height_threshold = kh_;
  parallel_axis_ = (mb_ >= num_threads)
    ? parallel_kind::BATCH
    : ((int)utils::divide_and_ceil(oh_, num_threads) > height_threshold
        ? parallel_kind::HEIGHT
        : parallel_kind::BATCH);

  num_brgemm_k_ = attrs_.get<int>("num_brgemm_k");
  brgemm_k_ = attrs_.get<int>("brgemm_k");
  extra_padding_ = attrs_.get<int>("extra_padding");
  int last_row_size = actual_ih_ * ic_ + extra_padding_;
  if (parallel_kind::HEIGHT == parallel_axis_) {
    last_row_size
      = ((utils::divide_and_ceil(oh_, num_threads) - 1) * sh_ + kh_) * ic_
      + extra_padding_;
  }
  aux_buf_size_ = (actual_iw_ - 1) * kh_ * ic_ + last_row_size;
  blocking_input_ = get_input_blocking_dims().size() > ndims_;
  blocking_output_ = get_output_blocking_dims().size() > ndims_;
}

bool gen_conv_fwd_rl_t::generate(context_ptr ctx,
  const conv_fwd_rl_config_t &config, fusion_anchor_mgr_t *fusion,
  const std::vector<expr> &inputs, const std::vector<expr> &outputs,
  std::vector<for_loop> &loops) const {
  COMPILE_ASSERT(inputs.size() == 2,
    "Expecting 2 inputs for conv, but got " << inputs.size() << " inputs.");
  COMPILE_ASSERT(outputs.size() == 1,
    "Expecting 1 output for conv, but got " << outputs.size() << " output.");

  int brgemm_m = config.brgemm_m;
  int brgemm_n = config.brgemm_n;
  int K_num_block = oc_ / brgemm_n;
  COMPILE_ASSERT(brgemm_n && (oc_ % brgemm_n == 0),
    "oc should be dividable by brgemm_n, but got oc=" << oc_ << " brgemm_n="
                                                      << brgemm_n << ".");
  COMPILE_ASSERT((brgemm_m > 0) && (ow_ % brgemm_m == 0),
    "oh should be dividable by brgemm_m, but got oh=" << ow_ << " brgemm_m="
                                                      << brgemm_m << ".");

  auto input_dtype = get_input_dtype();
  auto weight_dtype = get_weight_dtype();
  auto output_dtype = get_output_dtype();
  if (input_dtype == datatypes::bf16) {
    COMPILE_ASSERT((weight_dtype == datatypes::bf16),
      "Weights should be bf16 as "
      "data, the mixed datatypes is not supported yet!");
    COMPILE_ASSERT((output_dtype == datatypes::f32),
      "Output should be f32 when data and weights are in bf16.");
  }
  if (input_dtype == datatypes::f16) {
    COMPILE_ASSERT((weight_dtype == datatypes::f16),
      "Weights should be f16 as "
      "data, the mixed datatypes is not supported yet!");
    COMPILE_ASSERT((output_dtype == datatypes::f32),
      "Output should be f32 when data and weights are in f16.");
  }
  if (utils::is_one_of(input_dtype, datatypes::s8, datatypes::u8)) {
    COMPILE_ASSERT((weight_dtype == datatypes::s8),
      "Weights should be s8 when \
            data is s8/u8, the mixed datatypes is not supported yet!");
    COMPILE_ASSERT((output_dtype == datatypes::s32),
      "Output should be s32 when data and weights are in "
      "s8/u8.");
  }

  expr output = outputs[0];
  expr input = inputs[0];
  expr weight = inputs[1];

  int given_num_threads = runtime_config_t::get().get_num_threads();
  int num_threads = given_num_threads;
  if (parallel_axis_ == parallel_kind::BATCH) {
    num_threads = std::min(given_num_threads, mb_ * groups_);
    if (num_threads < given_num_threads) {
      SC_WARN
        << "The actual parallelism is less than given due to task assignment, "
        << num_threads << " vs " << given_num_threads << ".";
    }
  }
  COMPILE_ASSERT(num_threads <= given_num_threads,
    "expect num_threads <= given_num_threads, but got "
      << num_threads << " vs " << given_num_threads << ".");

  auto mb_expr = input.checked_as<tensor>()->dims_[0];
  auto lanes = static_cast<int>(
    ctx->get_max_vector_lanes(in_tensors_[0].dtype_.type_code_));

  int real_pb = std::max(actual_ih_ - pt_ - ih_, 0);
  COMPILE_ASSERT(real_pb <= pb_,
    "expect real_pb <= pb_, but got " << real_pb << ", vs " << pb_ << ".");
  int real_pr = std::max(actual_iw_ - pl_ - iw_, 0);
  COMPILE_ASSERT(real_pr <= pr_,
    "expect real_pr <= pr_, but got " << real_pr << ", vs " << pr_ << ".");
  int ih_remaining = (pt_ + ih_ + pb_ - kh_) % sh_;
  int oh_num_pb = (ih_remaining == 0)
    ? utils::divide_and_ceil(pb_, sh_)
    : ((pb_ > ih_remaining) ? utils::divide_and_ceil(pb_ - ih_remaining, sh_)
                            : 0);
  int oh_pr_idx = oh_num_pb > 0 ? (oh_ - oh_num_pb) : -1;
  auto padding_value = attrs_.get_or_else("padding_value", 0);
  auto ic_lanes
    = ic_ * utils::get_sizeof_type(get_input_dtype()) / 8 > cache_line_size
    ? 1
    : ic_;
  auto ic_real_lane
    = ic_lanes == 1 ? 1 : std::min(lanes, get_minimal_lanes(ic_lanes));
  expr ic_mask = ic_lanes == 1
    ? expr()
    : builder::make_cast(get_dtype(ic_real_lane), convert_int_to_mask(ic_));

  auto init_aux_buf = [&](const expr &aux_buf, const expr &n_o, const expr &g,
                        const expr &p, const expr &init_idx, const expr &tid) {
    // only need to copy the valid area as all the remaining padding
    // areas are already zero-out
    expr cur_pt = pt_;
    expr p_offset = 0;
    if (parallel_axis_ == parallel_kind::HEIGHT) {
      cur_pt = builder::make_min(kh_,
        builder::make_max(
          0, pt_ - builder::make_cast(datatypes::s32, init_idx * sh_)));
      p_offset = builder::make_select(p > 0, pt_, 0);
    }
    _for_(iw, pl_, real_pr > 0 ? (pl_ + iw_) : actual_iw_) {
      _for_(kh, 0, kh_ - cur_pt, 1) {
        _for_(c_i, 0, ic_, ic_lanes) {
          aux_buf[span_t({iw * kh_ * ic_ + cur_pt * ic_ + kh * ic_ + c_i},
            ic_real_lane, ic_mask)]
            = input[span_t(data_offset(n_o, g, 0, 0, p * sh_ + kh - p_offset,
                             iw - pl_, 1, c_i),
              ic_real_lane, ic_mask)];
        }
      }
    }

    if (real_pr == 0) {
      // last row handling, requires special handling for the case of
      // parallel on height axis
      _var_init_(last_row, datatypes::s32, (actual_ih_ - pt_ - real_pb));
      _var_init_(cur_pt, datatypes::s32, builder::make_select(p == 0, pt_, 0));
      _var_init_(
        p_offset, datatypes::s32, builder::make_select(p == 0, 0, pt_));
      if (parallel_axis_ == parallel_kind::HEIGHT) {
        int job1 = utils::divide_and_ceil(oh_, num_threads);
        int job2 = oh_ / num_threads;
        int threshold = (oh_ % num_threads) * job1;
        _if_(p == 0) {
          // left-most region
          last_row = ((job1 - 1) * sh_ + kh_ - pt_);
        }
        _else_ {
          _if_(p == oh_ - job2) {
            // right-most region
            last_row = ((job2 - 1) * sh_ + kh_ - real_pb);
          }
          _else_ {
            _if_(p >= threshold) { last_row = ((job2 - 1) * sh_ + kh_); }
            _else_ { last_row = ((job1 - 1) * sh_ + kh_); }
          }
        }
      }
      _for_(hi, 0, last_row, 1) {
        _for_(ci, 0, ic_, ic_lanes) {
          aux_buf[span_t(
            {(actual_iw_ - 1) * kh_ * ic_ + hi * ic_ + ci + cur_pt * ic_},
            ic_real_lane, ic_mask)]
            = input[span_t(
              data_offset(n_o, g, 0, 0, hi + init_idx * sh_ - p_offset,
                actual_iw_ - pl_ - 1, 1, ci),
              ic_real_lane, ic_mask)];
        }
      }
    }
  };

  auto update_aux_buf = [&](const expr &aux_buf, const expr &n_o, const expr &g,
                          const expr &p, const expr &init_idx) {
    for (int iw = 1; iw < pl_ + 1; ++iw) {
      builtin::brgemm_init(
        tensor_ptr(
          aux_buf, {((p - init_idx) - 1) * sh_ * ic_ + iw * kh_ * ic_}),
        1, sh_ * ic_, sh_ * ic_, get_input_dtype(), padding_value);
    }
    // need special copy for the right-end
    // for right-end with padding, but
    // different update mask according to the current positions.
    _var_init_(update_pad_lanes, datatypes::s32,
      builder::make_select(p >= oh_pr_idx && oh_num_pb > 0,
        builder::make_cast(
          datatypes::s32, builder::make_min(p * sh_ + kh_ - (pt_ + ih_), sh_)),
        0));
    _var_init_(update_copy_lanes, datatypes::s32,
      builder::make_cast(datatypes::s32,
        builder::make_max(0, sh_ - builder::make_max(0, update_pad_lanes))));
    _for_(iw, pl_ + 1,
      real_pr > 0 && oh_num_pb > 0 ? (pl_ + iw_ + 1) : actual_iw_) {
      // copy input
      _for_(h, 0, update_copy_lanes, 1) {
        _for_(c_i, 0, ic_, ic_lanes) {
          aux_buf[span_t(
            {((p - init_idx) - 1) * sh_ * ic_ + iw * kh_ * ic_ + h * ic_ + c_i},
            ic_real_lane, ic_mask)]
            = input[span_t(
              data_offset(n_o, g, 0, 0, (p - 1) * sh_ + kh_ - pt_ + h,
                iw - 1 - pl_, 1, c_i),
              ic_real_lane, ic_mask)];
        }
      }
      // zero-out
      _if_(update_pad_lanes > 0) {
        builtin::brgemm_init(tensor_ptr(aux_buf,
                               {((p - init_idx) - 1) * sh_ * ic_
                                 + iw * kh_ * ic_ + update_copy_lanes * ic_}),
          1, update_pad_lanes * ic_, update_pad_lanes * ic_, get_input_dtype(),
          padding_value);
      }
    }
  };

  auto do_compute = [&](const expr &aux_buf, const expr &n_o, const expr &g,
                      const expr &p, const expr &init_idx) {
    _for_(q, 0, ow_ / config.brgemm_m) {
      _tensor_(A_list, datatypes::pointer, {num_brgemm_k_});
      _tensor_(B_list, datatypes::pointer, {num_brgemm_k_});
      auto offset = (p - init_idx) * sw_ * ic_;
      for (int i = 0; i < num_brgemm_k_; ++i) {
        A_list[i] = tensor_ptr(aux_buf,
          {offset + q * config.brgemm_m * sw_ * kw_ * ic_ + i * brgemm_k_});
      }

      _for_(k_o, 0, K_num_block) {
        for (int i = 0; i < num_brgemm_k_; ++i) {
          // weight in KNknk format
          B_list[i] = tensor_ptr(weight, {i, g * K_num_block + k_o, 0, 0, 0});
        }
        auto out_tsr = tensor_ptr(output,
          output_offset(n_o, g, 0, 0, p, q * config.brgemm_m,
            K_num_block * config.brgemm_n, k_o * config.brgemm_n));

        const auto hint_A_size = config.brgemm_m * brgemm_k_ * num_brgemm_k_;
        const auto hint_B_size = num_brgemm_k_ * brgemm_k_ * config.brgemm_n;
        const auto hint_C_size = config.brgemm_m * config.brgemm_n;
        sc_brgemm_attrs_t brg_attrs {{brgemm::attr_key::max_bs, num_brgemm_k_},
          {brgemm::attr_key::hint_expected_A_size, hint_A_size},
          {brgemm::attr_key::hint_expected_B_size, hint_B_size},
          {brgemm::attr_key::hint_expected_C_size, hint_C_size},
          {brgemm::attr_key::use_interleave_stores, true},
          {brgemm::attr_key::use_uker, true}};
        {
          trace_guard_t trg(ctx, "brgemm");
          builtin::brgemm_init_list_update(A_list, B_list, out_tsr, 1,
            config.brgemm_m, config.brgemm_n, brgemm_k_, LDA_, config.brgemm_n,
            blocking_output_
              ? oc_
              : groups_ * oc_ /* channel last for g=1, blocking for g>1 */,
            1 /*useless*/, 1 /*useless*/, num_brgemm_k_, get_input_dtype(),
            get_weight_dtype(),
            ctx->flags_.kernel_optim_ == 1 ? brg_attrs : sc_brgemm_attrs_t());
        }

        // brgemm_m * brgemm_n
        trace_guard_t trg(ctx, "post-op fusion");
        create_fusion_anchor(fusion, owner_->get_outputs()[0],
          is_group_conv_
            ? (blocking_output_
                ? slice_range {{n_o, 1}, {g, 1}, {0, 1}, {p, 1},
                  {q * config.brgemm_m, config.brgemm_m},
                  {k_o * config.brgemm_n, config.brgemm_n}}
                : slice_range {{n_o, 1}, {p, 1},
                  {q * config.brgemm_m, config.brgemm_m}, {g, 1},
                  {k_o * config.brgemm_n, config.brgemm_n}})
            : (blocking_output_ ? slice_range {{n_o, 1}, {g, 1}, {p, 1},
                 {q * config.brgemm_m, config.brgemm_m},
                 {k_o * config.brgemm_n, config.brgemm_n}}
                                : slice_range {{n_o, 1}, {p, 1},
                                  {q * config.brgemm_m, config.brgemm_m},
                                  {(g * K_num_block + k_o) * config.brgemm_n,
                                    config.brgemm_n}}));
      }
      // brgemm_m * oc_
      create_anchor(fusion, owner_->get_outputs()[0], n_o, 1, g, 1, 0, 1, 0, 1,
        p, 1, q * config.brgemm_m, config.brgemm_m, oc_);
    }
  };

  if (parallel_kind::BATCH == parallel_axis_) {
    for_loop ln, lg, lp;
    auto input_expr_dims = input.checked_as<tensor>()->dims_;
    auto mb_expr = input_expr_dims[0];
    _named_for_(ln, n_o, 0, mb_expr, 1, for_type::PARALLEL) {
      _named_for_(lg, g, 0, groups_, 1) {
        _tensor_(aux_buf, input_dtype, {aux_buf_size_});
        builtin::brgemm_init(
          aux_buf, 1, aux_buf_size_, aux_buf_size_, input_dtype, padding_value);

        _named_for_(lp, p, 0, oh_, 1) {
          _if_(p == 0) {
            trace_guard_t trg(ctx, "init_aux");
            init_aux_buf(aux_buf, n_o, g, p, 0, 0 /*useless*/);
          }
          _else_ {
            trace_guard_t trg(ctx, "update_aux");
            update_aux_buf(aux_buf, n_o, g, p, 0);
          }
          do_compute(aux_buf, n_o, g, p, 0);
          // ow_ * oc_
          create_anchor(fusion, owner_->get_outputs()[0], n_o, 1, g, 1, 0, 1, 0,
            1, p, 1, 0, ow_, oc_);
        }
        if (mb_ * groups_ >= num_threads) {
          // oh_ * ow_ * oc_
          create_anchor(fusion, owner_->get_outputs()[0], n_o, 1, g, 1, 0, 1, 0,
            1, 0, oh_, 0, ow_, oc_);
        }
      }
      // oh_ * ow_ * oc_
      create_anchor(fusion, owner_->get_outputs()[0], n_o, 1, 0, groups_, 0, 1,
        0, 1, 0, oh_, 0, ow_, oc_);
    }
    lp->attr().set(stmt_attr_key::no_loop_fuse, true);
    bind_loop_axis(owner_->get_outputs()[0], ln, 0);
    bind_loop_axis(owner_->get_outputs()[0], lg,
      is_group_conv_ ? std::vector<int> {1} : std::vector<int> {});
    bind_loop_axis(owner_->get_outputs()[0], lp, is_group_conv_ + 2);
  } else {
    expr oh_b, oh_e;
    for_loop lt;
    expr start_idx, large_group, init_idx;
    _named_for_(lt, t, 0, num_threads, 1, for_type::PARALLEL) {
      _tensor_(aux_buf, input_dtype, {aux_buf_size_});
      _var_init_(group_size, datatypes::s32,
        get_balance211_length(oh_, num_threads, t, start_idx, large_group));
      oh_b = start_idx;
      oh_e = start_idx + group_size;
      init_idx = start_idx;
      _for_(n_o, 0, mb_, 1) {
        _for_(g, 0, groups_, 1) {
          builtin::brgemm_init(aux_buf, 1, aux_buf_size_, aux_buf_size_,
            input_dtype, padding_value);
          _for_(p, oh_b, oh_e, 1) {
            _if_(p == init_idx) {
              trace_guard_t trg(ctx, "init_aux");
              init_aux_buf(aux_buf, n_o, g, p, init_idx, t);
            }
            _else_ {
              trace_guard_t trg(ctx, "update_aux");
              update_aux_buf(aux_buf, n_o, g, p, init_idx);
            }
            do_compute(aux_buf, n_o, g, p, init_idx);
            // ow_ * oc_
            create_anchor(fusion, owner_->get_outputs()[0], n_o, 1, g, 1, 0, 1,
              0, 1, p, 1, 0, ow_, oc_);
          }
        }
      }
    }
    bind_loop_axis(owner_->get_outputs()[0], lt, is_group_conv_ + 2);
  }
  return true;
}

void gen_conv_fwd_rl_t::schedule_loops(context_ptr ctx,
  const conv_fwd_rl_config_t &config, stmt body,
  std::vector<for_loop> &fors) const {}
} // namespace ops
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
