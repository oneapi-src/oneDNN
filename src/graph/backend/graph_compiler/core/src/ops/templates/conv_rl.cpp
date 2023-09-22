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
#include "conv_rl.hpp"
#include <algorithm>
#include <utility>
#include "utils.hpp"
#include <compiler/ir/builder.hpp>
#include <compiler/ir/builtin.hpp>
#include <compiler/ir/easy_build.hpp>
#include <compiler/ir/graph/fusion_mgr.hpp>
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
  cfg.brgemm_m = oh_;
  std::vector<int> brgemm_m_candidates = {32, 16};
  bool has_proper_brgemm_m = false;
  for (auto &c : brgemm_m_candidates) {
    if (oh_ % c == 0) {
      cfg.brgemm_m = c;
      has_proper_brgemm_m = true;
      break;
    }
  }
  if (!has_proper_brgemm_m) {
    cfg.brgemm_m = utils::get_blocks(oh_, 16, 64).back();
  }
  cfg.brgemm_n = utils::get_blocks(oc_, 16).back();

  return std::move(ret);
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
  COMPILE_ASSERT(
    ndims_ == 4, "reduce lowering currently only support 4D input!")
  COMPILE_ASSERT(weight_plain_dims.size() == ndims_,
    "Wrong weight dims, only support 4D weights, but got "
      << weight_plain_dims.size() << "D.");

  groups_ = static_cast<int>(attrs_.get_or_else("groups", 1));
  COMPILE_ASSERT(input_plain_dims[1] / groups_ == weight_plain_dims[1],
    "expect input_plain_dims[1] / groups == weight_plain_dims[1], but got "
      << input_plain_dims[1] / groups_ << " vs " << weight_plain_dims[1]
      << ".");

  mb_ = input_plain_dims[0];
  ic_ = input_plain_dims[1] / groups_;
  ih_ = input_plain_dims[2];
  iw_ = input_plain_dims[3];

  oc_ = weight_plain_dims[0] / groups_;
  kh_ = weight_plain_dims[2];
  kw_ = weight_plain_dims[3];
  oh_ = out_plain_dims[2];
  ow_ = out_plain_dims[3];
  pt_ = pads_begin[0], pb_ = pads_begin[0];
  pl_ = pads_end[0], pr_ = pads_end[0];
  if (pads_begin.size() > 1) {
    pt_ = pads_begin[ndims_ - 4];
    pl_ = pads_begin[ndims_ - 3];
  }
  if (pads_end.size() > 1) {
    pb_ = pads_end[ndims_ - 4];
    pr_ = pads_end[ndims_ - 3];
  }

  sh_ = stride[0], sw_ = stride[0];
  if (stride.size() > 1) {
    sh_ = stride[ndims_ - 4];
    sw_ = stride[ndims_ - 3];
  }

  LDA_ = kw_ * ic_ * sw_;
  actual_iw_ = (ow_ - 1) * sw_ + kw_;
  actual_ih_ = (oh_ - 1) * sh_ + kh_;
  auto input_dtype = get_input_dtype();

  COMPILE_ASSERT(pt_ <= kh_ && pb_ <= kh_ && pl_ <= kw_ && pr_ <= kw_,
    "Not support the case of padding > filter_size!");
  int width_threshold = kw_;
  int num_threads = runtime_config_t::get().get_num_threads();
  parallel_axis_ = (mb_ * groups_ >= num_threads)
    ? parallel_kind::BATCH
    : ((int)utils::divide_and_ceil(ow_, num_threads) > width_threshold
        ? parallel_kind::WIDTH
        : parallel_kind::BATCH);

  num_brgemm_k_ = attrs_.get<int>("num_brgemm_k");
  brgemm_k_ = attrs_.get<int>("brgemm_k");
  extra_padding_ = attrs_.get<int>("extra_padding");
  int last_row_size
    = ((parallel_axis_ == parallel_kind::BATCH)
          ? (actual_iw_ * ic_)
          : ((utils::divide_and_ceil(ow_, num_threads) - 1) * sw_ + kw_) * ic_)
    + extra_padding_;
  aux_buf_size_ = (actual_ih_ - 1) * kw_ * ic_ + last_row_size;

  init_lanes_ = (kw_ - pl_) * ic_ * utils::get_sizeof_type(input_dtype);
  update_lanes_ = sw_ * ic_ * utils::get_sizeof_type(input_dtype);
  assert(init_lanes_ <= cache_line_size && update_lanes_ <= cache_line_size);
  init_mask_ = convert_int_to_mask(init_lanes_);
  update_mask_ = convert_int_to_mask(update_lanes_);
}

bool gen_conv_fwd_rl_t::generate(context_ptr ctx,
  const conv_fwd_rl_config_t &config, fusion_manager *fusion,
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
  COMPILE_ASSERT((brgemm_m > 0) && (oh_ % brgemm_m == 0),
    "oh should be dividable by brgemm_m, but got oh=" << oh_ << " brgemm_m="
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
  int iw_remaining = (pl_ + iw_ + pr_ - kw_) % sw_;
  int ow_num_pr = (iw_remaining == 0)
    ? utils::divide_and_ceil(pr_, sw_)
    : ((pr_ > iw_remaining) ? utils::divide_and_ceil(pr_ - iw_remaining, sw_)
                            : 0);
  int ow_pr_idx = ow_num_pr > 0 ? (ow_ - ow_num_pr) : -1;

  _tensor_(pr_pad_lanes_tsr, datatypes::index, {ow_num_pr});
  _tensor_(pr_copy_mask_tsr, datatypes::index, {ow_num_pr});
  _tensor_(pr_copy_lanes_tsr, datatypes::s32, {ow_num_pr});
  if (ow_num_pr > 0) {
    assert(ow_num_pr == ow_ - ow_pr_idx);
    for (int i = ow_pr_idx; i < ow_; ++i) {
      int pr_pad_lanes = std::min(i * sw_ + kw_ - (pl_ + iw_), sw_) * ic_;
      int pr_copy_lanes = std::max(0, sw_ * ic_ - pr_pad_lanes);
      pr_pad_lanes_tsr[i - ow_pr_idx] = pr_pad_lanes;
      pr_copy_mask_tsr[i - ow_pr_idx] = convert_int_to_mask(pr_copy_lanes);
      pr_copy_lanes_tsr[i - ow_pr_idx] = pr_copy_lanes;
    }
  }

  _tensor_(init_mask_tsr, datatypes::index, {num_threads});
  _tensor_(init_lanes_tsr, datatypes::index, {num_threads});
  if (parallel_axis_ == parallel_kind::WIDTH) {
    auto get_start_pos = [](int tid, int groups, int size) {
      int job1 = utils::divide_and_ceil(size, groups);
      int job2 = job1 - 1;
      int pos = size - job2 * groups;
      int job = tid < pos ? job1 : job2;
      int start_pos
        = tid <= pos ? (tid * job1) : pos * job1 + (tid - pos) * job2;
      return start_pos;
    };

    for (int i = 0; i < num_threads; ++i) {
      int start_iw = get_start_pos(i, num_threads, ow_) * sw_;
      int cur_pl = std::min(kw_, std::max(0, pl_ - start_iw));
      int init_lanes
        = (kw_ - cur_pl) * ic_ * utils::get_sizeof_type(input_dtype);
      init_mask_tsr[i] = convert_int_to_mask(init_lanes);
      init_lanes_tsr[i] = init_lanes;
    }
  }

  auto init_aux_buf = [&](const expr &aux_buf, const expr &n_o, const expr &g,
                        const expr &q, const expr &init_idx, const expr &tid) {
    // only need to copy the valid area as all the remaining padding
    // areas are already zero-out
    int max_lanes = kw_ * ic_ * utils::get_sizeof_type(input_dtype);
    max_lanes = std::min(lanes, get_minimal_lanes(max_lanes));
    expr init_mask_expr = builder::make_cast(get_dtype(max_lanes), init_mask_);
    expr cur_pl = pl_;
    expr q_offset = 0;
    if (parallel_axis_ == parallel_kind::WIDTH) {
      init_mask_expr
        = builder::make_cast(get_dtype(max_lanes), init_mask_tsr[tid]);
      cur_pl = builder::make_min(kw_,
        builder::make_max(
          0, pl_ - builder::make_cast(datatypes::s32, init_idx * sw_)));
      q_offset = builder::make_select(q > 0, pl_, 0);
    }

    _for_(ih, pt_, real_pb > 0 ? (pt_ + ih_) : actual_ih_) {
      aux_buf[span_t(
        {ih * kw_ * ic_ + cur_pl * ic_}, max_lanes, init_mask_expr)]
        = input[span_t(groups_ > 1
            ? std::vector<expr> {n_o, g, ih - pt_, q * sw_ - q_offset, 0}
            : std::vector<expr> {n_o, ih - pt_, q * sw_ - q_offset, 0},
          max_lanes, init_mask_expr)];
    }

    if (real_pb == 0) {
      // last row handling, requires special handling for the case of
      // parallel on width axis
      auto copy_last_row
        = [&](const int last_row, const int lanes, const int q_offset,
            const int pl = 0, const int pr = 0) {
            auto copy_with_simd = utils::rnd_dn(last_row, lanes);
            auto remainder = last_row % lanes;
            if (copy_with_simd > 0) {
              _for_(wi, 0, copy_with_simd, lanes) {
                aux_buf[span_t(
                  {(actual_ih_ - 1) * kw_ * ic_ + wi + pl * ic_}, lanes)]
                  = input[span_t(groups_ > 1
                      ? std::vector<expr> {n_o, g, actual_ih_ - pt_ - 1,
                        wi / ic_ + init_idx * sw_ - q_offset, wi % ic_}
                      : std::vector<expr> {n_o, actual_ih_ - pt_ - 1,
                        wi / ic_ + init_idx * sw_ - q_offset, wi % ic_},
                    lanes)];
              }
            }
            if (remainder > 0) {
              auto remainder_mask = convert_int_to_mask(remainder);
              aux_buf[span_t(
                {(actual_ih_ - 1) * kw_ * ic_ + copy_with_simd + pl * ic_},
                lanes, remainder_mask)]
                = input[span_t(groups_ > 1
                    ? std::vector<expr> {n_o, g, actual_ih_ - pt_ - 1,
                      copy_with_simd / ic_ + init_idx * sw_ - q_offset,
                      copy_with_simd % ic_}
                    : std::vector<expr> {n_o, actual_ih_ - pt_ - 1,
                      copy_with_simd / ic_ + init_idx * sw_ - q_offset,
                      copy_with_simd % ic_},
                  lanes, remainder_mask)];
            }
          };

      if (parallel_axis_ == parallel_kind::WIDTH) {
        int job1 = utils::divide_and_ceil(ow_, num_threads);
        int job2 = ow_ / num_threads;
        int threshold = (ow_ % num_threads) * job1;

        _if_(q == 0) {
          // left-most region
          auto last_row = ((job1 - 1) * sw_ + kw_ - pl_) * ic_;
          copy_last_row(last_row, lanes, 0, pl_, 0);
        }
        _else_ {
          _if_(q == ow_ - job2) {
            // right-most region
            auto last_row = ((job2 - 1) * sw_ + kw_ - real_pr) * ic_;
            copy_last_row(last_row, lanes, pl_, 0, real_pr);
          }
          _else_ {
            _if_(q >= threshold) {
              auto last_row = ((job2 - 1) * sw_ + kw_) * ic_;
              copy_last_row(last_row, lanes, pl_, 0, real_pr);
            }
            _else_ {
              auto last_row = ((job1 - 1) * sw_ + kw_) * ic_;
              copy_last_row(last_row, lanes, pl_, 0, 0);
            }
          }
        }
      } else {
        auto last_row = (actual_iw_ - pl_ - real_pr) * ic_;
        copy_last_row(last_row, lanes, 0, pl_, real_pr);
      }
    }
  };

  auto update_aux_buf = [&](const expr &aux_buf, const expr &n_o, const expr &g,
                          const expr &q, const expr &init_idx) {
    auto update_lanes = std::min(lanes, get_minimal_lanes(update_lanes_));
    expr update_mask_var
      = builder::make_cast(get_dtype(update_lanes), update_mask_);
    if (ow_num_pr > 0) {
      // need special copy for the right-end
      _if_(q >= ow_pr_idx) {
        // for right-end with padding, resue update_lanes, but
        // different update mask according to the current positions.
        expr mask_idx = q - ow_pr_idx;
        expr update_pad_lanes = pr_pad_lanes_tsr[mask_idx];
        expr update_copy_mask = builder::make_cast(
          get_dtype(update_lanes), pr_copy_mask_tsr[mask_idx]);
        expr update_copy_lanes = pr_copy_lanes_tsr[mask_idx];

        for (int ih = 1; ih < pt_ + 1; ++ih) {
          builtin::mem_zero(
            tensor_ptr(
              aux_buf, {((q - init_idx) - 1) * sw_ * ic_ + ih * kw_ * ic_}),
            update_lanes_, get_input_dtype());
        }

        _for_(ih, pt_ + 1, real_pb > 0 ? (pt_ + ih_ + 1) : actual_ih_) {
          // copy input
          _if_(update_copy_mask > 0) {
            aux_buf[span_t({((q - init_idx) - 1) * sw_ * ic_ + ih * kw_ * ic_},
              update_lanes, update_copy_mask)]
              = input[span_t(groups_ > 1 ? std::vector<expr> {n_o, g,
                               ih - 1 - pt_, (q - 1) * sw_ + kw_ - pl_, 0}
                                         : std::vector<expr> {n_o, ih - 1 - pt_,
                                           (q - 1) * sw_ + kw_ - pl_, 0},
                update_lanes, update_copy_mask)];
          }
          // zero-out
          _if_(update_pad_lanes > 0) {
            builtin::mem_zero(tensor_ptr(aux_buf,
                                {((q - init_idx) - 1) * sw_ * ic_
                                  + ih * kw_ * ic_ + update_copy_lanes}),
              update_pad_lanes, get_input_dtype());
          }
        }
      }
      _else_ {
        for (int ih = 1; ih < pt_ + 1; ++ih) {
          builtin::mem_zero(
            tensor_ptr(
              aux_buf, {((q - init_idx) - 1) * sw_ * ic_ + ih * kw_ * ic_}),
            update_lanes_, get_input_dtype());
        }
        _for_(ih, pt_ + 1, real_pb > 0 ? (pt_ + ih_ + 1) : actual_ih_) {
          aux_buf[span_t({((q - init_idx) - 1) * sw_ * ic_ + ih * kw_ * ic_},
            update_lanes, update_mask_var)]
            = input[span_t(groups_ > 1 ? std::vector<expr> {n_o, g,
                             ih - pt_ - 1, (q - 1) * sw_ + kw_ - pl_, 0}
                                       : std::vector<expr> {n_o, ih - pt_ - 1,
                                         (q - 1) * sw_ + kw_ - pl_, 0},
              update_lanes, update_mask_var)];
        }
      }
    } else {
      for (int ih = 1; ih < pt_ + 1; ++ih) {
        builtin::mem_zero(
          tensor_ptr(
            aux_buf, {((q - init_idx) - 1) * sw_ * ic_ + ih * kw_ * ic_}),
          update_lanes_, get_input_dtype());
      }
      _for_(ih, pt_ + 1, actual_ih_) {
        aux_buf[span_t({((q - init_idx) - 1) * sw_ * ic_ + ih * kw_ * ic_},
          update_lanes, update_mask_var)]
          = input[span_t(groups_ > 1 ? std::vector<expr> {n_o, g, ih - pt_ - 1,
                           (q - 1) * sw_ + kw_ - pl_, 0}
                                     : std::vector<expr> {n_o, ih - pt_ - 1,
                                       (q - 1) * sw_ + kw_ - pl_, 0},
            update_lanes, update_mask_var)];
      }
    }
  };

  auto do_compute = [&](const expr &aux_buf, const expr &n_o, const expr &g,
                      const expr &q, const expr &init_idx) {
    _for_(p, 0, oh_ / config.brgemm_m) {
      _tensor_(A_list, datatypes::pointer, {num_brgemm_k_});
      _tensor_(B_list, datatypes::pointer, {num_brgemm_k_});
      auto offset = (q - init_idx) * sw_ * ic_;
      for (int i = 0; i < num_brgemm_k_; ++i) {
        A_list[i] = tensor_ptr(aux_buf,
          {offset + p * config.brgemm_m * sh_ * kw_ * ic_ + i * brgemm_k_});
      }

      _for_(k_o, 0, K_num_block) {
        for (int i = 0; i < num_brgemm_k_; ++i) {
          // weight in KNknk format
          B_list[i] = tensor_ptr(weight, {i, g * K_num_block + k_o, 0, 0, 0});
        }
        auto out_tsr = tensor_ptr(output,
          groups_ > 1 ? std::vector<expr> {n_o, g, p * config.brgemm_m, q,
            k_o * config.brgemm_n}
                      : std::vector<expr> {n_o, p * config.brgemm_m, q,
                        (g * K_num_block + k_o) * config.brgemm_n});

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
            ow_ * oc_ /* channel last for g=1, blocking for g>1 */,
            1 /*useless*/, 1 /*useless*/, num_brgemm_k_, get_input_dtype(),
            get_weight_dtype(),
            ctx->flags_.kernel_optim_ == 1 ? brg_attrs : sc_brgemm_attrs_t());
        }

        if (fusion) {
          // brgemm_m * brgemm_n
          trace_guard_t trg(ctx, "post-op fusion");
          fusion->create_output_fusion_anchor({tensor_slice(output,
            groups_ > 1 ? slice_range {{n_o, 1}, {g, 1},
              {p * config.brgemm_m, config.brgemm_m}, {q, 1},
              {k_o * config.brgemm_n, config.brgemm_n}}
                        : slice_range {{n_o, 1},
                          {p * config.brgemm_m, config.brgemm_m}, {q, 1},
                          {(g * K_num_block + k_o) * config.brgemm_n,
                            config.brgemm_n}})});
        }
      }
      if (fusion) {
        // brgemm_m * oc_
        fusion->create_output_fusion_anchor({tensor_slice(output,
          groups_ > 1
            ? slice_range {{n_o, 1}, {g, 1},
              {p * config.brgemm_m, config.brgemm_m}, {q, 1}, {0, oc_}}
            : slice_range {{n_o, 1}, {p * config.brgemm_m, config.brgemm_m},
              {q, 1}, {g * oc_, oc_}})});
      }
    }
  };

  if (parallel_kind::BATCH == parallel_axis_) {
    for_loop ln, lg, lq;
    auto input_expr_dims = input.checked_as<tensor>()->dims_;
    auto mb_expr = input_expr_dims[0];
    _named_for_(ln, n_o, 0, mb_expr, 1, for_type::PARALLEL) {
      _named_for_(lg, g, 0, groups_, 1) {
        _tensor_(aux_buf, input_dtype, {aux_buf_size_});
        builtin::mem_zero(aux_buf, aux_buf_size_, input_dtype);
        _named_for_(lq, q, 0, ow_, 1) {
          _if_(q == 0) {
            trace_guard_t trg(ctx, "init_aux");
            init_aux_buf(aux_buf, n_o, g, q, 0, 0 /*useless*/);
          }
          _else_ {
            trace_guard_t trg(ctx, "update_aux");
            update_aux_buf(aux_buf, n_o, g, q, 0);
          }
          do_compute(aux_buf, n_o, g, q, 0);
          if (fusion) {
            // oh_ * oc_
            fusion->create_output_fusion_anchor({tensor_slice(output,
              groups_ > 1
                ? slice_range {{n_o, 1}, {g, 1}, {0, oh_}, {q, 1}, {0, oc_}}
                : slice_range {{n_o, 1}, {0, oh_}, {q, 1}, {g * oc_, oc_}})});
          }
        }
        if (fusion && mb_ * groups_ >= num_threads) {
          // oh_ * ow_ * oc_
          fusion->create_output_fusion_anchor({tensor_slice(output,
            groups_ > 1
              ? slice_range {{n_o, 1}, {g, 1}, {0, oh_}, {0, ow_}, {0, oc_}}
              : slice_range {{n_o, 1}, {0, oh_}, {0, ow_}, {g * oc_, oc_}})});
        }
      }
    }
    lq->attr().set(stmt_attr_key::no_loop_fuse, true);
  } else {
    expr ow_b, ow_e;
    for_loop lt;
    expr start_idx, large_group, init_idx;
    _named_for_(lt, t, 0, num_threads, 1, for_type::PARALLEL) {
      _tensor_(aux_buf, input_dtype, {aux_buf_size_});
      _var_init_(group_size, datatypes::s32,
        get_balance211_length(ow_, num_threads, t, start_idx, large_group));
      ow_b = start_idx;
      ow_e = start_idx + group_size;
      init_idx = start_idx;
      _for_(n_o, 0, mb_, 1) {
        _for_(g, 0, groups_, 1) {
          builtin::mem_zero(aux_buf, aux_buf_size_, input_dtype);
          _for_(q, ow_b, ow_e, 1) {
            _if_(q == init_idx) {
              trace_guard_t trg(ctx, "init_aux");
              init_aux_buf(aux_buf, n_o, g, q, init_idx, t);
            }
            _else_ {
              trace_guard_t trg(ctx, "update_aux");
              update_aux_buf(aux_buf, n_o, g, q, init_idx);
            }
            do_compute(aux_buf, n_o, g, q, init_idx);
            if (fusion) {
              // oh_ * oc_
              fusion->create_output_fusion_anchor({tensor_slice(output,
                groups_ > 1
                  ? slice_range {{n_o, 1}, {g, 1}, {0, oh_}, {q, 1}, {0, oc_}}
                  : slice_range {{n_o, 1}, {0, oh_}, {q, 1}, {g * oc_, oc_}})});
            }
          }
          if (fusion && mb_ * groups_ >= num_threads) {
            fusion->create_output_fusion_anchor({tensor_slice(output,
              groups_ > 1 ? slice_range {{n_o, 1}, {g, 1}, {0, oh_},
                {0, ow_e - ow_b}, {0, oc_}}
                          : slice_range {{n_o, 1}, {0, oh_}, {0, ow_e - ow_b},
                            {g * oc_, oc_}})});
          }
        }
      }
    }
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
