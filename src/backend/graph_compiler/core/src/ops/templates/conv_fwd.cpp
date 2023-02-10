/*******************************************************************************
 * Copyright 2020-2023 Intel Corporation
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

#include "conv_fwd.hpp"
#include <algorithm>
#include <functional>
#include <numeric>
#include <utility>
#include "utils.hpp"
#include <compiler/ir/builder.hpp>
#include <compiler/ir/builtin.hpp>
#include <compiler/ir/easy_build.hpp>
#include <compiler/ir/graph/fusion_mgr.hpp>
#include <compiler/ir/transform/auto_cast.hpp>
#include <compiler/ir/transform/constant_fold.hpp>
#include <compiler/ir/transform/tensor_shrink.hpp>
#include <runtime/barrier.hpp>
#include <runtime/config.hpp>
#include <unordered_set>
#include <util/any_map.hpp>
#include <util/math_utils.hpp>
#include <util/reflection.hpp>
#include <util/utils.hpp>
using namespace sc::builder;
namespace sc {

using ops::conv_fwd_config_t;
// clang-format off
SC_CLASS(conv_fwd_config_t)
  SC_FIELD(bs_threads)
  SC_FIELD(s_threads)
  SC_FIELD(oc_threads)
  SC_FIELD(s_block)
  SC_FIELD(K_block)
  SC_FIELD(C_block)
  SC_FIELD(tile_d)
  SC_FIELD(tile_p)
  SC_FIELD(tile_q)
  SC_FIELD(tile_os)
  SC_FIELD(pack_input)
  SC_FIELD(loop_sched)
SC_CLASS_END();
// clang-format on

namespace ops {
static int get_im_s_block(const context_ptr &ctx, const int &os,
  const int &default_block, const int &im_oc_block,
  const sc::any_map_t &attrs) {
  auto ret = default_block;
  auto origin_ow = dim2unsigned(attrs.get_or_else("origin_ow", sc_dim(os)));
  auto origin_oh = dim2unsigned(attrs.get_or_else("origin_oh", sc_dim(1)));
  auto s_default_block = default_block;
  if (origin_ow > 14) {
    auto L1_cache_size = ctx->machine_.cpu_flags_.getDCacheSize(1);
    // not use L1_cache too full
    s_default_block = L1_cache_size / 4 / im_oc_block;
  }
  auto s_block_list = utils::get_blocks(os, 1, s_default_block);
  s_block_list.erase(
    std::remove_if(s_block_list.begin(), s_block_list.end(),
      [&](int blk) {
        return !(origin_ow % blk == 0
          || (blk % origin_ow == 0 && origin_oh * origin_ow % blk == 0)
          || blk % (origin_oh * origin_ow) == 0);
      }),
    s_block_list.end());
  return s_block_list.back();
}

void gen_conv_fwd_t::validate_conv_fwd_default_config(
  const context_ptr &ctx, conv_fwd_config_t &cfg) const {
  bool dtype_f32 = get_input_dtype() == datatypes::f32;
  bool use_os_blocking = try_os_blocking_ && is_use_amx(ctx);
  auto K_block_list = utils::get_blocks(oc_, 16);
  auto C_block_list = utils::get_blocks(ic_, 16);
  auto tile_d_list = utils::get_factors(od_);
  auto tile_p_list = use_os_blocking
    ? std::vector<int> {-1}
    : (dtype_f32 ? std::vector<int> {1} : utils::get_factors(oh_));
  auto tile_q_list
    = use_os_blocking ? std::vector<int> {-1} : utils::get_factors(ow_);
  auto tile_os_list
    = use_os_blocking ? get_os_blocks(ow_, adj_os_) : std::vector<int> {-1};
  auto pack_input_list = (is_1x1_conv_ && (sd_ > 1 || sh_ > 1 || sw_ > 1))
    ? std::vector<int> {0, 1}
    : std::vector<int> {-1};
  auto loop_sched_list = std::vector<int> {0, 1, 2, 3};
  if (std::find(K_block_list.begin(), K_block_list.end(), cfg.K_block)
    == K_block_list.end()) {
    cfg.K_block = K_block_list.at(0);
  }
  if (std::find(C_block_list.begin(), C_block_list.end(), cfg.C_block)
    == C_block_list.end()) {
    cfg.C_block = C_block_list.at(0);
  }
  if (std::find(tile_d_list.begin(), tile_d_list.end(), cfg.tile_d)
    == tile_d_list.end()) {
    cfg.tile_d = tile_d_list.at(0);
  }
  if (std::find(tile_p_list.begin(), tile_p_list.end(), cfg.tile_p)
    == tile_p_list.end()) {
    cfg.tile_p = tile_p_list.at(0);
  }
  if (std::find(tile_q_list.begin(), tile_q_list.end(), cfg.tile_q)
    == tile_q_list.end()) {
    cfg.tile_q = tile_q_list.at(0);
  }
  if (std::find(tile_os_list.begin(), tile_os_list.end(), cfg.tile_os)
    == tile_os_list.end()) {
    cfg.tile_os = tile_os_list.back();
  }
  if (std::find(pack_input_list.begin(), pack_input_list.end(), cfg.pack_input)
    == pack_input_list.end()) {
    cfg.pack_input = pack_input_list.at(0);
  }
  if (std::find(loop_sched_list.begin(), loop_sched_list.end(), cfg.loop_sched)
    == loop_sched_list.end()) {
    cfg.loop_sched = loop_sched_list.at(0);
  }
}

config_ptr gen_conv_fwd_t::get_default_config(context_ptr ctx) const {
  auto ret = reflection::general_object_t::make<conv_fwd_config_t>();
  conv_fwd_config_t &cfg = *ret.unchecked_get_as<conv_fwd_config_t>();
  const auto nthreads = runtime_config_t::get().get_num_threads();
  auto C_block_list = utils::get_blocks(ic_, 16);
  auto K_block_list = utils::get_blocks(oc_, 16);

  auto tile_p_list = utils::get_factors(oh_);
  auto tile_q_list = utils::get_factors(ow_);
  auto dtype_size = get_weight_dtype() == datatypes::f32
    ? 4
    : (get_weight_dtype() == datatypes::bf16 ? 2 : 1);
  cfg.tile_d = 1;
  cfg.tile_os = -1;
  cfg.pack_input = (is_1x1_conv_ && (sd_ > 1 || sh_ > 1 || sw_ > 1)) ? 1 : -1;
  cfg.loop_sched = 0;
  // C_block shall only relay to ic_
  int max_ic_block = -1;
  int max_oc_block = -1;
  if (ic_ % 32 != 0) {
    cfg.C_block = ic_ > 32 ? 32 : utils::rnd_up(ic_, 4);
  } else {
    for (int i = C_block_list.size() - 1; i >= 0; i--) {
      if (C_block_list[i] <= 128) {
        max_ic_block = C_block_list[i];
        break;
      }
    }
  }
  // K block shall only relay to oc_ and apply same logic with ic to avoid
  // possibly double buffer
  if (oc_ % 32 != 0) {
    cfg.K_block = oc_ > 32 ? 32 : utils::rnd_up(oc_, 4);
  } else {
    for (int i = K_block_list.size() - 1; i >= 0; i--) {
      if (K_block_list[i] <= 128) {
        max_oc_block = K_block_list[i];
        break;
      }
    }
  }
  // large K N: adjust K_block and N_block(as gemm)
  if (oc_ * ic_ >= 512 * 512) {
    if (is_1x1_conv_) {
      max_oc_block *= 2;
    } else {
      max_ic_block *= 2;
    }
    cfg.loop_sched = 3;
  }

  // large spatial
  bool large_spatial = oh_ * ow_ >= 128 * 128;
  if (large_spatial) {
    if (is_use_amx(ctx) && get_weight_dtype() != datatypes::f32) {
      cfg.loop_sched = 2;
    } else {
      cfg.loop_sched = 0;
    }
  }

  bool parallel_space_is_enough
    = (mb_ % nthreads == 0 || utils::divide_and_ceil(mb_, nthreads) > 8);
  if (is_1x1_conv_ && (oc_ / ic_ >= 4 && oc_ >= 1024)) { max_oc_block = 128; }
  // tile_p and tile_q
  if (!is_1x1_conv_) {
    cfg.tile_p = 1;
  } else {
    if (ic_ * oc_ <= 64 * 256 || !parallel_space_is_enough) {
      cfg.tile_p = 1;
    } else {
      cfg.tile_p = tile_p_list.back();
    }
  }
  if (!is_1x1_conv_) {
    cfg.tile_q = tile_q_list.back();
    if (large_spatial) { cfg.tile_q = utils::get_blocks(ow_, 1, 32).back(); }
  } else {
    // handle large M for gemm kernel: shrink M
    if (sw_ > 1) {
      cfg.tile_q = 1;
    } else {
      cfg.tile_q = tile_q_list.back();
    }
    if (iw_ > 28 && oc_ * ic_ >= 128 * 256) {
      if (iw_ % 2 == 0) cfg.tile_q = iw_ / 2;
    }
    if (ih_ > 28 && oc_ * ic_ >= 128 * 256 && parallel_space_is_enough) {
      if (ih_ % 2 == 0) cfg.tile_p = ih_ / 2;
    }
  }
  if (get_input_dtype() == datatypes::f32) { cfg.tile_p = 1; }
  if (try_os_blocking_) {
    // if use os blocking override tile p and tile q above
    cfg.tile_os = cfg.tile_q;
    auto os_choices = get_os_blocks(ow_, adj_os_);
    std::sort(os_choices.begin(), os_choices.end());
    if (ow_ < 28 && ow_ % 16 != 0) {
      for (int i = os_choices.size() - 1; i >= 0; i--) {
        if (nthreads <= adj_os_ / os_choices[i] * mb_) {
          cfg.tile_os = os_choices[i];
          break;
        }
      }
    }
    cfg.tile_q = -1;
    cfg.tile_p = -1;
  }
  max_oc_block = std::max(
    max_oc_block, utils::get_blocks(oc_, 1, ow_ * 2 / dtype_size).back());
  max_ic_block = std::max(
    max_ic_block, utils::get_blocks(ic_, 1, ow_ * 2 / dtype_size).back());
  cfg.K_block = oc_ % 32 == 0 ? max_oc_block : oc_;
  cfg.C_block = ic_ % 32 == 0 ? max_ic_block : ic_;
  validate_conv_fwd_default_config(ctx, cfg);
  if (inverse_filter_) {
    cfg.C_block = 64;
    cfg.K_block = 64;
    cfg.tile_d = 1;
    cfg.tile_p = 1;
    cfg.tile_q = ow_;
  }
  if (use_conv1d) {
    const int num_threads = runtime_config_t::get().get_num_threads();
    auto thread_split = get_splits(num_threads);
    im_s_block_
      = get_im_s_block(ctx, oh_ * ow_, default_im_block_, im_oc_block_, attrs_);
    auto closest_split = [](int x, std::vector<int> splits) {
      int close_num = splits[0];
      for (auto split : splits) {
        if (x - split < x - close_num && x > split) { close_num = split; }
      }
      return close_num;
    };
    cfg.bs_threads
      = mb_ >= num_threads ? num_threads : closest_split(mb_, thread_split);
    cfg.s_threads = num_threads / cfg.bs_threads;
    cfg.oc_threads = 1;
    auto s_max_task_num = ow_ / im_s_block_;
    if (mb_ == 1 && s_max_task_num < num_threads) {
      auto oc_max_task_num = oc_ / im_oc_block_;
      if (oc_max_task_num == num_threads || oc_max_task_num % num_threads == 0
        || oc_max_task_num > num_threads * 8) {
        cfg.bs_threads = 1;
        cfg.oc_threads = num_threads;
        cfg.s_threads = 1;
      } else if (oc_ < 1024 && oh_ * ow_ <= 28 * 28 && num_threads % 2 == 0) {
        cfg.bs_threads = 1;
        cfg.oc_threads = num_threads / 2;
        cfg.s_threads = num_threads / 2;
      } else {
        cfg.bs_threads = 1;
        cfg.oc_threads = 1;
        cfg.s_threads = num_threads;
      }
    }
    auto ic_threads = 1;
    cfg.s_block
      = utils::divide_and_ceil(
          utils::divide_and_ceil(oh_ * ow_, im_s_block_), cfg.s_threads)
      * im_s_block_;
    cfg.K_block = utils::divide_and_ceil(
                    utils::divide_and_ceil(oc_, im_oc_block_), cfg.oc_threads)
      * im_oc_block_;
    cfg.C_block = utils::divide_and_ceil(
                    utils::divide_and_ceil(ic_, im_ic_block_), ic_threads)
      * im_ic_block_;
  }

  return std::move(ret);
}

gen_conv_fwd_t::gen_conv_fwd_t(sc_op *owner, const sc_dims &stride,
  const sc_dims &pads_begin, std::vector<logical_tensor_t> &&ins,
  std::vector<logical_tensor_t> &&outs)
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
  is_1d_ = (ndims_ == 3);

  blocking_input_ = get_input_blocking_dims().size() > ndims_;
  blocking_output_ = get_output_blocking_dims().size() > ndims_;
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
  COMPILE_ASSERT(input_plain_dims[1] == weight_plain_dims[1],
    "expect input_plain_dims[1] == weight_plain_dims[1], but got "
      << input_plain_dims[1] << " vs " << weight_plain_dims[1] << ".");

  mb_ = input_plain_dims[0];
  ic_ = input_plain_dims[1];
  id_ = is_3d_ ? input_plain_dims[2] : 1;
  ih_ = is_1d_ ? 1 : input_plain_dims[ndims_ - 2];
  iw_ = input_plain_dims[ndims_ - 1];
  oc_ = weight_plain_dims[0];
  kd_ = is_3d_ ? weight_plain_dims[2] : 1;
  kh_ = is_1d_ ? 1 : weight_plain_dims[ndims_ - 2];
  kw_ = weight_plain_dims[ndims_ - 1];
  od_ = is_3d_ ? out_plain_dims[2] : 1;
  oh_ = is_1d_ ? 1 : out_plain_dims[ndims_ - 2];
  ow_ = out_plain_dims[ndims_ - 1];
  is_1x1_conv_ = (kd_ == 1 && kh_ == 1 && kw_ == 1);
  pd_ = is_3d_ ? pads_begin[0] : 0;
  ph_ = is_1d_ ? 0 : pads_begin[0], pw_ = pads_begin[0];
  bool is_int8
    = utils::is_one_of(get_input_dtype(), datatypes::u8, datatypes::s8);
  bool is_bf16 = get_input_dtype() == datatypes::bf16;

  auto dtype_block = is_int8 ? 4 : (is_bf16 ? 2 : 1);
  const int num_threads = runtime_config_t::get().get_num_threads();
  if (owner) { attrs_ = owner->attrs_; }
  default_im_block_ = dtype_block * 64;
  if (ic_ * oc_ < 512 * 512) { default_im_block_ /= 2; }
  if (mb_ == 1 && num_threads == 4) { default_im_block_ = 64; }
  im_oc_block_ = utils::get_blocks(oc_, 1, default_im_block_).back();
  im_ic_block_ = utils::get_blocks(ic_, 1, default_im_block_).back();
  im_s_block_ = utils::get_blocks(ow_ * oh_, 1, default_im_block_).back();

  if (pads_begin.size() > 1) {
    ph_ = pads_begin[ndims_ - 4];
    pw_ = pads_begin[ndims_ - 3];
  }
  sd_ = is_3d_ ? stride[0] : 1;
  sh_ = is_1d_ ? 1 : stride[0], sw_ = stride[0];
  if (stride.size() > 1) {
    auto stride_size = stride.size();
    sh_ = stride[stride_size - 2];
    sw_ = stride[stride_size - 1];
  }

  // For non 1x1 conv and AMX platform, spatial blocking instead of row
  // blocking is used, which needs to consider the border carefully, as the
  // cross row boundary (contains padding or not) will generate useless output
  // which have to be skipped before storing.
  actual_os_ = oh_ * ow_;
  num_elems_skip_per_ow_ = ((kw_ - 1) / sw_) * sh_ + (sh_ - 1) * ow_;
  adj_os_ = std::min(actual_os_ + num_elems_skip_per_ow_ * (oh_ - 1),
    (ih_ + 2 * ph_) * (iw_ + 2 * pw_));

  // Note: os blocking is only valid for non_1x1, no pad and non 3D conv with
  // amx-int8 only so far.
  bool has_pad = (pd_ > 0) || (ph_ > 0) || (pw_ > 0);
  try_os_blocking_
    = (!is_1x1_conv_) && (!has_pad) && (!is_3d_) && is_int8 && ow_ < 28;
  if (is_1d_) {
    use_conv1d = true;
    COMPILE_ASSERT((kw_ == 1 && pw_ == 0),
      "Conv1d doesn't support padding and kernel size except 1x1.");
  }
}

float gen_conv_fwd_t::get_gflop() const {
  float result = (float)mb_ * oc_ * 2.0 * ic_ * kd_ * kh_ * kw_ * od_ * oh_
    * ow_ / (float)1e9;
  return result;
}

static expr tensor_offset(const sc_dims &dims_, const std::vector<expr> &idx) {
  COMPILE_ASSERT(dims_.size() == idx.size(),
    "The tensor of tensor_ptr has " << dims_.size() << " dimemsions, but got "
                                    << idx.size() << " indices.");
  expr offset = idx.back();
  expr dim = dim2unsigned(dims_.back());
  for (int64_t i = idx.size() - 2; i >= 0; i--) {
    offset = idx.at(i) * dim + offset;
    dim = dim2unsigned(dims_.at(i)) * dim;
  }
  return builder::make_cast(datatypes::s32, offset);
}

static int tensor_offset(const sc_dims &dims_, const std::vector<int> &idx) {
  COMPILE_ASSERT(dims_.size() == idx.size(),
    "The tensor of tensor_ptr has " << dims_.size() << " dimemsions, but got "
                                    << idx.size() << " indices.");
  int offset = idx.back();
  int dim = dims_.back();
  for (int i = idx.size() - 2; i >= 0; i--) {
    offset = idx.at(i) * dim + offset;
    dim = dims_.at(i) * dim;
  }
  return offset;
}

#define CONV_ARG_LIST \
  const context_ptr &ctx, const conv_fwd_config_t &config, \
    fusion_manager *fusion, expr &output, const expr &input, \
    const expr &weight, std::vector<for_loop> &loops, const int K_num_block, \
    const int C_num_block, const int os, const int kpack, \
    const bool use_os_blocking, const bool pack_rows, const expr &os_acc_size, \
    const std::vector<char> &os_mask

void gen_conv_fwd_t::compute_conv1d(CONV_ARG_LIST) const {
  // TODO(zhicong):
  // 1. support blocking layout
  // 2. provide better support in scc bench
  // 3. add iterated anchor

  // std::max avoid grid tuning generate bad config
  int num_threads = runtime_config_t::get().get_num_threads();
  int bs_threads = std::max(1, config.bs_threads);
  int s_threads
    = std::max(1, std::min(num_threads / bs_threads, config.s_threads));
  int oc_threads = std::max(1, num_threads / bs_threads / s_threads);
  int ic_threads = 1;
  int oc_block = config.K_block;
  int s_block = config.s_block;
  int ic_block = config.C_block;
  int im_oc_block = im_oc_block_;
  int im_ic_block = im_ic_block_;
  int im_s_block
    = get_im_s_block(ctx, oh_ * ow_, default_im_block_, im_oc_block_, attrs_);
  if (oc_block % im_oc_block != 0) { oc_block = im_oc_block; }
  COMPILE_ASSERT(oc_block % im_oc_block == 0,
    "oc_block % im_oc_block != 0, config is invalid")
  if (ic_block % im_ic_block != 0) { ic_block = im_ic_block; }
  COMPILE_ASSERT(ic_block % im_ic_block == 0,
    "ic_block % im_ic_block != 0, config is invalid")
  if (s_block % im_s_block != 0) { s_block = im_s_block; }
  COMPILE_ASSERT(
    s_block % im_s_block == 0, "s_block % im_s_block != 0, config is invalid")

  // param
  expr input_tmp = input;
  expr output_tmp = output;
  auto tinput = in_tensors_[0];
  auto tweight = in_tensors_[1];
  auto toutput = out_tensors_[0];
  const auto &input_blocking_dims = tinput.get_blocking_dims();
  const auto &weight_blocking_dims = tweight.get_blocking_dims();
  const auto &output_blocking_dims = toutput.get_blocking_dims();
  int os_ = ow_;
  for_loop lpbs, lps, lpoc, lpic, lobs, los, looc, loic, lioc, lis;

  int bs_num_block_pt, bs_tail_num_block_pt, oc_num_block_pt,
    oc_tail_num_block_pt, s_num_block_pt, s_tail_num_block_pt, ic_num_block_pt,
    ic_tail_num_block_pt;
  int bs_used_threads
    = block_split(mb_, bs_threads, bs_num_block_pt, bs_tail_num_block_pt);
  int oc_used_threads = block_split(utils::divide_and_ceil(oc_, oc_block),
    oc_threads, oc_num_block_pt, oc_tail_num_block_pt);
  int os_used_threads = block_split(utils::divide_and_ceil(os_, s_block),
    s_threads, s_num_block_pt, s_tail_num_block_pt);
  int ic_used_threads = block_split(utils::divide_and_ceil(ic_, ic_block),
    ic_threads, ic_num_block_pt, ic_tail_num_block_pt);
  auto input_expr_dims = input.checked_as<tensor>()->dims_;
  auto mb_expr_ = input_expr_dims[0];
  bool shrink_tensor = false;

  if (ic_used_threads > 1) {
    // output temp buffer
    auto out_dims = output_blocking_dims;
    out_dims[0] *= ic_used_threads;
    _tensor_(out_tmp, toutput.dtype_, dims_to_expr(out_dims));
    output_tmp = out_tmp;
  }

  auto infer_input_idx = [&](std::vector<expr> output_idx) {
    std::vector<expr> input_idx = output_idx;
    auto origin_ow = dim2unsigned(attrs_.get_or_else("origin_ow", sc_dim(ow_)));
    auto origin_iw = dim2unsigned(attrs_.get_or_else("origin_iw", sc_dim(iw_)));
    auto origin_oh = dim2unsigned(attrs_.get_or_else("origin_oh", sc_dim(oh_)));
    auto origin_ih = dim2unsigned(attrs_.get_or_else("origin_ih", sc_dim(ih_)));
    if (sh_ > 1 || sw_ > 1) {
      expr os = output_idx[1];
      expr ow = os % origin_ow;
      expr oh = os / origin_ow % origin_oh;
      expr bs = os / origin_ow / origin_oh;
      expr iw = ow * sw_;
      expr ih = oh * sh_;
      expr is = bs * origin_iw * origin_ih + ih * origin_iw + iw;
      input_idx[1] = is;
    }
    return input_idx;
  };

  _named_for_(lpbs, pbs, 0, mb_expr_, 1, for_type::PARALLEL) {
    _named_for_(lps, ps, 0, s_threads, 1) {
      _named_for_(lpoc, poc, 0, oc_threads, 1) {
        _named_for_(lpic, pic, 0, ic_threads, 1) {
          expr s_num_block = builder::make_select(ps < (os_used_threads - 1),
                 s_num_block_pt, s_tail_num_block_pt),
               oc_num_block = builder::make_select(poc < (oc_used_threads - 1),
                 oc_num_block_pt, oc_tail_num_block_pt);
          if (sh_ > 1 || sw_ > 1) {
            auto in_dims = input_blocking_dims;
            in_dims[1] = output_blocking_dims[1];
            _tensor_(in_tmp, tinput.dtype_, dims_to_expr(in_dims));
            input_tmp = in_tmp;
            if (s_threads == 1 && oc_threads == 1 && ic_threads == 1) {
              shrink_tensor = false;
            } else {
              shrink_tensor = true;
            }
          }
          _if_(ps < os_used_threads && poc < oc_used_threads
            && pic < ic_used_threads) {
            // single core
            expr ic_num_block
              = builder::make_select(pic < (ic_used_threads - 1),
                ic_num_block_pt, ic_tail_num_block_pt);
            expr n = pbs;
            _named_for_(los, o_s, 0, s_num_block_pt) {
              _named_for_(looc, o_oc, 0, oc_num_block_pt) {
                _named_for_(loic, o_ic, 0, ic_num_block_pt) {
                  expr cond = o_s < s_num_block && o_oc < oc_num_block
                    && o_ic < ic_num_block;
                  _if_(cond) {
                    _named_for_(lis, i_s, 0, s_block / im_s_block) {
                      expr s = (ps * s_num_block_pt * s_block / im_s_block
                                 + o_s * s_block / im_s_block + i_s)
                        * im_s_block;
                      _if_(s < os_) {
                        _named_for_(lioc, i_oc, 0, oc_block / im_oc_block) {
                          _tensor_(A_list, datatypes::pointer,
                            {ic_block / im_ic_block});
                          _tensor_(B_list, datatypes::pointer,
                            {ic_block / im_ic_block});

                          expr oc
                            = poc * oc_num_block_pt * oc_block / im_oc_block
                            + o_oc * oc_block / im_oc_block + i_oc;
                          _if_(oc * im_oc_block < oc_) {
                            if (sh_ > 1 || sw_ > 1) {
                              if (shrink_tensor) {
                                input_tmp
                                  ->attr()[tensor_shrinker_attrs::should_shrink]
                                  = tensor_shrinker_t::shrink_info_t {
                                    /*base*/ {n, s,
                                      (pic * ic_num_block_pt * ic_block
                                          / im_ic_block
                                        + o_ic * ic_block / im_ic_block)
                                        * im_ic_block},
                                    /*shape*/ {1, im_s_block, ic_block},
                                    /*move def*/ stmts()};
                              } else {
                                input_tmp
                                  ->attr()[tensor_shrinker_attrs::should_shrink]
                                  = tensor_shrinker_t::shrink_info_t {
                                    /*base*/ {pbs, 0, 0},
                                    /*shape*/ {1, os_, ic_block},
                                    /*move def*/ stmts()};
                              }
                            }
                            _for_(i_c, 0, ic_block / im_ic_block) {
                              expr ic
                                = pic * ic_num_block_pt * ic_block / im_ic_block
                                + o_ic * ic_block / im_ic_block + i_c;
                              _if_(ic * im_ic_block < ic_) {
                                std::vector<expr> input_pos
                                  = std::vector<expr> {n, s, ic * im_ic_block};
                                if (sh_ > 1 || sw_ > 1) {
                                  int lanes = 1;
                                  lanes = (uint32_t)(ctx->get_max_vector_lanes(
                                    tinput.dtype_.type_code_));
                                  if (im_ic_block % lanes != 0) { lanes = 1; }
                                  _if_(
                                    (o_oc == 0 && i_oc == 0) || shrink_tensor) {
                                    _for_(ii_os, 0, im_s_block, 1) {
                                      _for_(ii_ic, 0, im_ic_block, lanes) {
                                        auto out_pos = input_pos;
                                        out_pos[1] = out_pos[1] + ii_os;
                                        out_pos[2] = out_pos[2] + ii_ic;
                                        input_tmp[span_t(out_pos, lanes)]
                                          = input[span_t(
                                            infer_input_idx(out_pos), lanes)];
                                      }
                                    }
                                  }
                                }
                                A_list[i_c] = tensor_ptr(input_tmp, input_pos);
                                B_list[i_c] = tensor_ptr(weight,
                                  kpack > 1
                                    ? std::vector<expr> {oc, ic, 0, 0, 0, 0}
                                    : std::vector<expr> {oc, ic, 0, 0, 0});
                              }
                            }

                            const auto hint_A_size = im_s_block * ic_block;
                            const auto hint_B_size = im_oc_block * ic_block;
                            const auto hint_C_size = im_s_block * im_oc_block;
                            sc_brgemm_attrs_t brg_attrs {
                              {brgemm::attr_key::max_bs,
                                ic_block / im_ic_block},
                              {brgemm::attr_key::hint_expected_A_size,
                                hint_A_size},
                              {brgemm::attr_key::hint_expected_B_size,
                                hint_B_size},
                              {brgemm::attr_key::hint_expected_C_size,
                                hint_C_size},
                              {brgemm::attr_key::use_interleave_stores, true},
                              {brgemm::attr_key::use_uker, true}};

                            auto LDA = ic_;
                            auto LDB = im_oc_block;
                            auto LDC = oc_;
                            auto stride_a = 1; /*useless*/
                            auto stride_b = 1; /*useless*/
                            auto stride_c = ic_block / im_ic_block;
                            std::vector<expr> output_pos = std::vector<expr> {
                              pic * mb_ + n, s, oc * im_oc_block};
                            if (ic_num_block_pt > 1) {
                              _if_(o_ic == 0) {
                                sc::builtin::brgemm_init_list_update(A_list,
                                  B_list, tensor_ptr(output_tmp, output_pos), 1,
                                  im_s_block, im_oc_block, im_ic_block, LDA,
                                  LDB, LDC, stride_a, stride_b, stride_c,
                                  get_input_dtype(), get_weight_dtype(),
                                  brg_attrs);
                              }
                              _else_ {
                                sc::builtin::brgemm_list_update(A_list, B_list,
                                  tensor_ptr(output_tmp, output_pos), 1,
                                  im_s_block, im_oc_block, im_ic_block, LDA,
                                  LDB, LDC, stride_a, stride_b, stride_c,
                                  get_input_dtype(), get_weight_dtype(),
                                  brg_attrs);
                              }
                            } else {
                              sc::builtin::brgemm_init_list_update(A_list,
                                B_list, tensor_ptr(output_tmp, output_pos), 1,
                                im_s_block, im_oc_block, im_ic_block, LDA, LDB,
                                LDC, stride_a, stride_b, stride_c,
                                get_input_dtype(), get_weight_dtype(),
                                brg_attrs);
                            }
                            if (fusion && ic_used_threads == 1
                              && ic_num_block_pt == 1) {
                              _if_(o_ic == (ic_num_block - 1)) {
                                fusion->create_output_fusion_anchor(
                                  {tensor_slice(output,
                                    std::vector<std::pair<expr, expr>> {
                                      {n, 1UL}, {s, im_s_block},
                                      {oc * im_oc_block, im_oc_block}})});
                              }
                            }
                          }
                        }
                        if (fusion && ic_used_threads == 1
                          && ic_num_block_pt == 1
                          && oc_block * oc_used_threads == oc_) {
                          _if_(o_ic == (ic_num_block - 1)) {
                            fusion->create_output_fusion_anchor({tensor_slice(
                              output,
                              std::vector<std::pair<expr, expr>> {{n, 1UL},
                                {s, im_s_block},
                                {(poc * oc_num_block_pt * oc_block / im_oc_block
                                   + o_oc * oc_block / im_oc_block)
                                    * im_oc_block,
                                  oc_block}})});
                          }
                        }
                      }
                    }
                    if (fusion && ic_used_threads == 1 && ic_num_block_pt == 1
                      && oc_block * oc_used_threads == oc_
                      && s_block * os_used_threads == os_) {
                      _if_(o_ic == (ic_num_block - 1)) {
                        fusion->create_output_fusion_anchor(
                          {tensor_slice(output,
                            std::vector<std::pair<expr, expr>> {{n, 1UL},
                              {(ps * s_num_block_pt * s_block / im_s_block
                                 + o_s * s_block / im_s_block)
                                  * im_s_block,
                                s_block},
                              {(poc * oc_num_block_pt * oc_block / im_oc_block
                                 + o_oc * oc_block / im_oc_block)
                                  * im_oc_block,
                                oc_block}})});
                      }
                    }
                  }
                }
                // TODO(zhicong): need to use iterated anchor to support more
                // fusion opportunity
                if (false && fusion && ic_used_threads == 1
                  && oc_block * oc_used_threads == oc_
                  && s_block * os_used_threads == os_) {
                  fusion->create_output_fusion_anchor({tensor_slice(output,
                    std::vector<std::pair<expr, expr>> {{n, 1UL},
                      {(ps * s_num_block_pt * s_block / im_s_block
                         + o_s * s_block / im_s_block)
                          * im_s_block,
                        s_block},
                      {(poc * oc_num_block_pt * oc_block / im_oc_block
                         + o_oc * oc_block / im_oc_block)
                          * im_oc_block,
                        oc_block}})});
                }
              }
            }
          }
          if (fusion && oc_threads == 1 && ic_threads == 1 && s_threads == 1) {
            fusion->create_output_fusion_anchor({tensor_slice(output,
              std::vector<std::pair<expr, expr>> {
                {pbs, 1UL}, {0, os_}, {0, oc_}})});
          }
        } // final reduce
        if (fusion && oc_threads == 1 && s_threads == 1) {
          fusion->create_output_fusion_anchor({tensor_slice(output,
            std::vector<std::pair<expr, expr>> {
              {pbs, 1UL}, {0, os_}, {0, oc_}})});
        }
      }
      if (fusion && s_threads == 1) {
        fusion->create_output_fusion_anchor({tensor_slice(output,
          std::vector<std::pair<expr, expr>> {
            {pbs, 1UL}, {0, os_}, {0, oc_}})});
      }
    }
    if (fusion && mb_ > 1) {
      // when mb_ == 1, no need fuse in here or the conv is flattened conv which
      // cannot be fused in bs
      fusion->create_output_fusion_anchor({tensor_slice(output,
        std::vector<std::pair<expr, expr>> {{pbs, 1UL}, {0, os_}, {0, oc_}})});
    }
  }
  loops = {lpbs, lps, lpoc, lpic};
}

void gen_conv_fwd_t::compute_1x1_no_pack_input(CONV_ARG_LIST) const {
  assert(loops.size() == 4 && "expected to have 4 level loops!");
  for_loop &ln = loops.at(0), &lk = loops.at(1), &ld = loops.at(2),
           &lp = loops.at(3);
  auto input_expr_dims = input.checked_as<tensor>()->dims_;
  auto mb_expr_ = input_expr_dims[0];
  _named_for_(ln, n, 0, mb_expr_, 1, for_type::PARALLEL) {
    _named_for_(lk, k, 0, K_num_block) {
      _named_for_(lp, p_o, 0, oh_ / config.tile_p) {
        _named_for_(ld, d_o, 0, od_ / config.tile_d) {
          _for_(q_o, 0, ow_ / config.tile_q) {
            _for_(d_i, 0, config.tile_d) {
              _for_(p_i, 0, config.tile_p) {
                auto LDA = blocking_input_ ? sw_ * config.C_block : sw_ * ic_;
                auto LDC = blocking_output_ ? config.K_block : oc_;
                auto stride_a = blocking_input_
                  ? (is_3d_ ? id_ * ih_ * iw_ * config.C_block
                            : ih_ * iw_ * config.C_block)
                  : config.C_block;
                if (is_3d_) {
                  _tensor_(A_list, datatypes::pointer, {C_num_block});
                  _tensor_(B_list, datatypes::pointer, {C_num_block});
                  _for_(c, 0, C_num_block) {
                    std::vector<expr> input_pos = blocking_input_
                      ? std::vector<expr> {n, c,
                        (d_o * config.tile_d + d_i) * sd_,
                        (p_o * config.tile_p + p_i) * sh_,
                        q_o * config.tile_q * sw_, 0}
                      : std::vector<expr> {n, (d_o * config.tile_d + d_i) * sd_,
                        (p_o * config.tile_p + p_i) * sh_,
                        q_o * config.tile_q * sw_, c * config.C_block};
                    A_list[c] = tensor_ptr(input, input_pos);
                    B_list[c] = tensor_ptr(weight,
                      kpack > 1 ? std::vector<expr> {k, c, 0, 0, 0, 0, 0, 0}
                                : std::vector<expr> {k, c, 0, 0, 0, 0, 0});
                  }

                  const auto hint_A_size = config.tile_q * ic_;
                  const auto hint_B_size = config.K_block * ic_;
                  const auto hint_C_size = config.tile_q * config.K_block;
                  sc_brgemm_attrs_t brg_attrs {
                    {brgemm::attr_key::max_bs, C_num_block},
                    {brgemm::attr_key::hint_expected_A_size, hint_A_size},
                    {brgemm::attr_key::hint_expected_B_size, hint_B_size},
                    {brgemm::attr_key::hint_expected_C_size, hint_C_size},
                    {brgemm::attr_key::use_interleave_stores, true},
                    {brgemm::attr_key::use_uker, true}};

                  std::vector<expr> output_pos = blocking_output_
                    ? std::vector<expr> {n, k, d_o * config.tile_d + d_i,
                      p_o * config.tile_p + p_i, q_o * config.tile_q, 0}
                    : std::vector<expr> {n, d_o * config.tile_d + d_i,
                      p_o * config.tile_p + p_i, q_o * config.tile_q,
                      k * config.K_block};
                  sc::builtin::brgemm_init_list_update(A_list, B_list,
                    tensor_ptr(output, output_pos), 1, config.tile_q,
                    config.K_block, config.C_block, LDA, config.K_block, LDC,
                    1 /*useless*/, 1 /*useless*/, C_num_block,
                    get_input_dtype(), get_weight_dtype(), brg_attrs);
                } else {
                  _tensor_(A_list, datatypes::pointer, {C_num_block});
                  _tensor_(B_list, datatypes::pointer, {C_num_block});
                  _for_(c, 0, C_num_block) {
                    std::vector<expr> input_pos = blocking_input_
                      ? std::vector<expr> {n, c,
                        (p_o * config.tile_p + p_i) * sh_,
                        q_o * config.tile_q * sw_, 0}
                      : std::vector<expr> {n, (p_o * config.tile_p + p_i) * sh_,
                        q_o * config.tile_q * sw_, c * config.C_block};
                    A_list[c] = tensor_ptr(input, input_pos);
                    B_list[c] = tensor_ptr(weight,
                      kpack > 1 ? std::vector<expr> {k, c, 0, 0, 0, 0, 0}
                                : std::vector<expr> {k, c, 0, 0, 0, 0});
                  }

                  const auto hint_A_size = config.tile_q * ic_;
                  const auto hint_B_size = config.K_block * ic_;
                  const auto hint_C_size = config.tile_q * config.K_block;
                  sc_brgemm_attrs_t brg_attrs {
                    {brgemm::attr_key::max_bs, C_num_block},
                    {brgemm::attr_key::hint_expected_A_size, hint_A_size},
                    {brgemm::attr_key::hint_expected_B_size, hint_B_size},
                    {brgemm::attr_key::hint_expected_C_size, hint_C_size},
                    {brgemm::attr_key::use_interleave_stores, true},
                    {brgemm::attr_key::use_uker, true}};

                  std::vector<expr> output_pos = blocking_output_
                    ? std::vector<expr> {n, k, p_o * config.tile_p + p_i,
                      q_o * config.tile_q, 0}
                    : std::vector<expr> {n, p_o * config.tile_p + p_i,
                      q_o * config.tile_q, k * config.K_block};
                  sc::builtin::brgemm_init_list_update(A_list, B_list,
                    tensor_ptr(output, output_pos), 1, config.tile_q,
                    config.K_block, config.C_block, LDA, config.K_block, LDC,
                    1 /*useless*/, 1 /*useless*/, C_num_block,
                    get_input_dtype(), get_weight_dtype(), brg_attrs);
                }
                if (fusion) {
                  if (is_3d_) {
                    fusion->create_output_fusion_anchor({tensor_slice(output,
                      blocking_output_
                        ? slice_range {{n, 1}, {k, 1},
                          {d_o * config.tile_d + d_i, 1},
                          {p_o * config.tile_p + p_i, 1},
                          {q_o * config.tile_q, config.tile_q},
                          {0, config.K_block}}
                        : slice_range {{n, 1}, {d_o * config.tile_d + d_i, 1},
                          {p_o * config.tile_p + p_i, 1},
                          {q_o * config.tile_q, config.tile_q},
                          {k * config.K_block, config.K_block}})});
                  } else {
                    fusion->create_output_fusion_anchor({tensor_slice(output,
                      blocking_output_
                        ? slice_range {{n, 1}, {k, 1},
                          {p_o * config.tile_p + p_i, 1},
                          {q_o * config.tile_q, config.tile_q},
                          {0, config.K_block}}
                        : slice_range {{n, 1}, {p_o * config.tile_p + p_i, 1},
                          {q_o * config.tile_q, config.tile_q},
                          {k * config.K_block, config.K_block}})});
                  }
                }
              }
            }
          }

          if (fusion) {
            if (is_3d_) {
              fusion->create_output_fusion_anchor({tensor_slice(output,
                blocking_output_
                  ? slice_range {{n, 1}, {k, 1},
                    {d_o * config.tile_d, config.tile_d},
                    {p_o * config.tile_p, config.tile_p}, {0, ow_},
                    {0, config.K_block}}
                  : slice_range {{n, 1}, {d_o * config.tile_d, config.tile_d},
                    {p_o * config.tile_p, config.tile_p}, {0, ow_},
                    {k * config.K_block, config.K_block}})});
            } else {
              fusion->create_output_fusion_anchor({tensor_slice(output,
                blocking_output_
                  ? slice_range {{n, 1}, {k, 1},
                    {p_o * config.tile_p, config.tile_p}, {0, ow_},
                    {0, config.K_block}}
                  : slice_range {{n, 1}, {p_o * config.tile_p, config.tile_p},
                    {0, ow_}, {k * config.K_block, config.K_block}})});
            }
          }
        }
      }
    }
    if (fusion) {
      fusion->create_output_fusion_anchor({tensor_slice(output,
        blocking_output_
          ? slice_range {{n, 1}, {0, K_num_block}, {0, oh_}, {0, ow_},
            {0, config.K_block}}
          : slice_range {{n, 1}, {0, oh_}, {0, ow_}, {0, oc_}})});
    }
  }
}

void gen_conv_fwd_t::compute_1x1_pack_input(CONV_ARG_LIST) const {
  COMPILE_ASSERT(!is_3d_, "1x1 pack input doens't support 3D conv yet!");
  assert(loops.size() == 4 && "expected to have 4 level loops!");
  for_loop &ln = loops.at(0), &lk = loops.at(1), &ld = loops.at(2),
           &lp = loops.at(3);
  tensor input1;
  auto input_expr_dims = input.checked_as<tensor>()->dims_;
  auto mb_expr_ = input_expr_dims[0];
  int lanes = get_lanes(ctx, config.C_block, get_input_dtype());
  if (config.pack_input == 1 && (sd_ > 1 || sh_ > 1 || sw_ > 1)) {
    if (blocking_input_) {
      _tensor_(input_tmp, get_input_dtype(),
        {mb_expr_, C_num_block, oh_, ow_, config.C_block});
      _named_for_(ln, n, 0, mb_expr_, 1, for_type::PARALLEL) {
        _named_for_(lk, c_o, 0, C_num_block) {
          _named_for_(lp, p, 0, oh_) {
            _for_(q, 0, ow_) {
              _for_(c_i, 0, config.C_block, (int)lanes) {
                input_tmp[span_t({n, c_o, p, q, c_i}, lanes)]
                  = input[span_t({n, c_o, p * sh_, q * sw_, c_i}, lanes)];
              }
            }
          }
        }
      }
      auto lnk = ln->fuse(lk);
      if (C_num_block * mb_ < runtime_config_t::get().get_num_threads() * 2) {
        auto lnkp = lnk->fuse(lp);
      }
      input1 = input_tmp.static_as<tensor>();
    } else {
      _tensor_(input_tmp, get_input_dtype(), {mb_expr_, oh_, ow_, ic_});
      _named_for_(ln, n, 0, mb_expr_, 1, for_type::PARALLEL) {
        _named_for_(lp, p, 0, oh_) {
          _for_(q, 0, ow_) {
            _for_(c_i, 0, ic_, (int)lanes) {
              input_tmp[span_t({n, p, q, c_i}, lanes)]
                = input[span_t({n, p * sh_, q * sw_, c_i}, lanes)];
            }
          }
        }
      }
      ln = ln->fuse(lp);
      input1 = input_tmp.static_as<tensor>();
    }
  } else {
    input1 = input.static_as<tensor>();
  }
  _named_for_(ln, n, 0, mb_expr_, 1, for_type::PARALLEL) {
    _named_for_(lk, k, 0, K_num_block) {
      _named_for_(lp, p_o, 0, oh_ / config.tile_p) {
        auto LDA = blocking_input_ ? config.C_block : ic_;
        auto LDC = blocking_output_ ? config.K_block : oc_;
        _tensor_(A_list, datatypes::pointer, {C_num_block});
        _tensor_(B_list, datatypes::pointer, {C_num_block});
        /* fill the address list for A/B */
        _for_(c, 0, C_num_block) {
          std::vector<expr> input_pos = blocking_input_
            ? std::vector<expr> {n, c, p_o * config.tile_p, 0, 0}
            : std::vector<expr> {n, p_o * config.tile_p, 0, c * config.C_block};
          A_list[c] = tensor_ptr(input1, input_pos);
          B_list[c] = tensor_ptr(weight,
            kpack > 1 ? std::vector<expr> {k, c, 0, 0, 0, 0, 0}
                      : std::vector<expr> {k, c, 0, 0, 0, 0});
        }

        const auto hint_A_size = config.tile_p * ow_ * ic_;
        const auto hint_B_size = config.K_block * ic_;
        const auto hint_C_size = config.tile_p * ow_ * config.K_block;
        sc_brgemm_attrs_t brg_attrs {{brgemm::attr_key::max_bs, C_num_block},
          {brgemm::attr_key::hint_expected_A_size, hint_A_size},
          {brgemm::attr_key::hint_expected_B_size, hint_B_size},
          {brgemm::attr_key::hint_expected_C_size, hint_C_size},
          {brgemm::attr_key::use_interleave_stores, true},
          {brgemm::attr_key::use_uker, true}};

        std::vector<expr> output_pos = blocking_output_
          ? std::vector<expr> {n, k, p_o * config.tile_p, 0, 0}
          : std::vector<expr> {n, p_o * config.tile_p, 0, k * config.K_block};
        sc::builtin::brgemm_init_list_update(A_list, B_list,
          tensor_ptr(output, output_pos), 1, config.tile_p * ow_,
          config.K_block, config.C_block, LDA, config.K_block, LDC,
          1 /*useless*/, 1 /*useless*/, C_num_block, get_input_dtype(),
          get_weight_dtype(), brg_attrs);
        if (fusion) {
          fusion->create_output_fusion_anchor({blocking_output_
              ? tensor_slice(output,
                {{n, 1}, {k, 1}, {p_o * config.tile_p, config.tile_p}, {0, ow_},
                  {0, config.K_block}})
              : tensor_slice(output,
                {{n, 1}, {p_o * config.tile_p, config.tile_p}, {0, ow_},
                  {k * config.K_block, config.K_block}})});
        }
      }
      if (fusion) {
        fusion->create_output_fusion_anchor(
          {blocking_output_ ? tensor_slice(
             output, {{n, 1}, {k, 1}, {0, oh_}, {0, ow_}, {0, config.K_block}})
                            : tensor_slice(output,
                              {{n, 1}, {0, oh_}, {0, ow_},
                                {k * config.K_block, config.K_block}})});
      }
    }
    if (fusion) {
      fusion->create_output_fusion_anchor({tensor_slice(output,
        blocking_output_
          ? slice_range {{n, 1}, {0, K_num_block}, {0, oh_}, {0, ow_},
            {0, config.K_block}}
          : slice_range {{n, 1}, {0, oh_}, {0, ow_}, {0, oc_}})});
    }
  }
}

void gen_conv_fwd_t::compute_conv3d_no_padding(CONV_ARG_LIST) const {
  COMPILE_ASSERT((pd_ == 0 && ph_ == 0 && pw_ == 0),
    "unexpected padding in no_padding kernels!");
  assert(loops.size() == 4 && "expected to have 4 level loops!");
  for_loop &ln = loops.at(0), &lk = loops.at(1), &ld = loops.at(2),
           &lp = loops.at(3);

  auto LDA = blocking_input_ ? sw_ * config.C_block : sw_ * ic_;
  auto LDC = blocking_output_ ? config.K_block : oc_;
  auto input_expr_dims = input.checked_as<tensor>()->dims_;
  auto mb_expr_ = input_expr_dims[0];
  _named_for_(ln, n, 0, mb_expr_, 1, for_type::PARALLEL) {
    _named_for_(lk, k_o, 0, K_num_block) {
      _named_for_(ld, d_o, 0, od_ / config.tile_d) {
        _named_for_(lp, p_o, 0, oh_ / config.tile_p) {
          _tensor_(A_list, datatypes::pointer, {kd_ * kh_ * kw_ * C_num_block});
          _tensor_(B_list, datatypes::pointer, {kd_ * kh_ * kw_ * C_num_block});
          _for_(q_o, 0, ow_ / config.tile_q) {
            _for_(d_i, 0, config.tile_d) {
              _for_(p_i, 0, config.tile_p) {
                std::vector<expr> output_pos = blocking_output_
                  ? std::vector<expr> {n, k_o, d_o * config.tile_d + d_i,
                    p_o * config.tile_p + p_i, q_o * config.tile_q, 0}
                  : std::vector<expr> {n, d_o * config.tile_d + d_i,
                    p_o * config.tile_p + p_i, q_o * config.tile_q,
                    k_o * config.K_block};

                _for_(c_o, 0, C_num_block) {
                  _for_(d, 0, kd_) {
                    _for_(r, 0, kh_) {
                      _for_(s, 0, kw_) {
                        std::vector<expr> input_pos = blocking_input_
                          ? std::vector<expr> {n, c_o,
                            (d_o * config.tile_d + d_i) * sd_ + d,
                            (p_o * config.tile_p + p_i) * sh_ + r,
                            q_o * config.tile_q * sw_ + s, 0}
                          : std::vector<expr> {n,
                            (d_o * config.tile_d + d_i) * sd_ + d,
                            (p_o * config.tile_p + p_i) * sh_ + r,
                            q_o * config.tile_q * sw_ + s,
                            c_o * config.C_block};
                        auto idx
                          = c_o * kd_ * kh_ * kw_ + d * kh_ * kw_ + r * kw_ + s;
                        A_list[idx] = tensor_ptr(input, input_pos);
                        B_list[idx] = tensor_ptr(weight,
                          kpack > 1
                            ? std::vector<expr> {k_o, c_o, d, r, s, 0, 0, 0}
                            : std::vector<expr> {k_o, c_o, d, r, s, 0, 0});
                      }
                    }
                  }
                }

                const auto hint_A_size = config.tile_q * config.C_block * kd_
                  * kh_ * kw_ * C_num_block;
                const auto hint_B_size
                  = config.K_block * config.C_block * kd_ * kh_ * kw_;
                // note, the actual C_size is <= tile_os if pack_rows=true
                const auto hint_C_size = config.tile_q * config.K_block;
                sc_brgemm_attrs_t brg_attrs {
                  {brgemm::attr_key::max_bs, kd_ * kh_ * kw_ * C_num_block},
                  {brgemm::attr_key::hint_expected_A_size, hint_A_size},
                  {brgemm::attr_key::hint_expected_B_size, hint_B_size},
                  {brgemm::attr_key::hint_expected_C_size, hint_C_size},
                  {brgemm::attr_key::use_interleave_stores, true},
                  {brgemm::attr_key::use_uker, true},
                  {brgemm::attr_key::bd_mask_level, 0}};

                sc::builtin::brgemm_init_list_update(A_list, B_list,
                  tensor_ptr(output, output_pos), 1, config.tile_q,
                  config.K_block, config.C_block, LDA, config.K_block, LDC,
                  1 /*useless*/, 1 /*useless*/, kd_ * kh_ * kw_ * C_num_block,
                  get_input_dtype(), get_weight_dtype(), brg_attrs);

                if (fusion) {
                  fusion->create_output_fusion_anchor({tensor_slice(output,
                    blocking_output_
                      ? slice_range {{n, 1}, {k_o, 1},
                        {d_o * config.tile_d + d_i, 1},
                        {p_o * config.tile_p + p_i, 1},
                        {q_o * config.tile_q, config.tile_q},
                        {0, config.K_block}}
                      : slice_range {{n, 1}, {d_o * config.tile_d + d_i, 1},
                        {p_o * config.tile_p + p_i, 1},
                        {q_o * config.tile_q, config.tile_q},
                        {k_o * config.K_block, config.K_block}})});
                }
              }
            }
          }
        }
      }
    }
    if (fusion) {
      fusion->create_output_fusion_anchor({tensor_slice(output,
        blocking_output_
          ? slice_range {{n, 1}, {0, K_num_block}, {0, od_}, {0, oh_}, {0, ow_},
            {0, config.K_block}}
          : slice_range {{n, 1}, {0, od_}, {0, oh_}, {0, ow_}, {0, oc_}})});
    }
  }
}

void gen_conv_fwd_t::compute_conv_no_padding(CONV_ARG_LIST) const {
  COMPILE_ASSERT((pd_ == 0 && ph_ == 0 && pw_ == 0),
    "unexpected padding in no_padding kernels!");
  assert(loops.size() == 4 && "expected to have 4 level loops!");
  loops.emplace_back(for_loop());
  for_loop &ln = loops.at(0), &lk = loops.at(1), &ld = loops.at(2),
           &lp = loops.at(3), &lok = loops.at(4);

  auto LDA = blocking_input_ ? sw_ * config.C_block : sw_ * ic_;
  auto LDC = blocking_output_ ? config.K_block : oc_;
  auto input_expr_dims = input.checked_as<tensor>()->dims_;
  auto mb_expr_ = input_expr_dims[0];
  int oc_split = 1;
  auto nthreads = runtime_config_t::get().get_num_threads();
  bool parallel_space_is_enough
    = (mb_ % nthreads == 0 || utils::divide_and_ceil(mb_, nthreads) > 8);
  auto weight_size
    = math_utils::get_dims_product(in_tensors_[1].get_blocking_dims())
    * utils::get_sizeof_type(get_weight_dtype());
  auto L2_cache_size = ctx->machine_.cpu_flags_.getDCacheSize(2);
  if (weight_size >= L2_cache_size && parallel_space_is_enough) {
    int expected_split_num = utils::divide_and_ceil(weight_size, L2_cache_size);
    for (auto &factor : utils::get_factors(K_num_block)) {
      if (factor >= expected_split_num) {
        expected_split_num = factor;
        break;
      }
    }
    oc_split = K_num_block < expected_split_num ? 1 : expected_split_num;
  }

  _named_for_(lok, outer_k, 0, oc_split, 1, for_type::PARALLEL) {
    _named_for_(ln, n, 0, mb_expr_, 1) {
      _named_for_(lk, k_i, 0, K_num_block / oc_split) {
        expr k_o = outer_k * K_num_block / oc_split + k_i;
        if (use_os_blocking) {
          _named_for_(lp, o_o, 0, os / config.tile_os) {
            _tensor_(A_list, datatypes::pointer, {kh_ * kw_ * C_num_block});
            _tensor_(B_list, datatypes::pointer, {kh_ * kw_ * C_num_block});
            auto out_tsr = tensor_ptr(output,
              blocking_output_
                ? std::vector<expr> {n, k_o, o_o * config.tile_os / ow_,
                  o_o * config.tile_os % ow_, 0}
                : std::vector<expr> {n, o_o * config.tile_os / ow_,
                  o_o * config.tile_os % ow_, k_o * config.K_block});
            int adj_ow = ow_ + (pack_rows ? num_elems_skip_per_ow_ : 0);

            if (pack_rows) {
              if (os / config.tile_os == 1) {
                out_tsr = tensor_ptr(output,
                  blocking_output_
                    ? std::vector<expr> {n, k_o, 0, 0, 0}
                    : std::vector<expr> {n, 0, 0, k_o * config.K_block});
              } else {
                auto acc_m = os_acc_size[{o_o}];
                out_tsr = tensor_ptr(output,
                  blocking_output_
                    ? std::vector<expr> {n, k_o, acc_m / ow_, acc_m % ow_, 0}
                    : std::vector<expr> {
                      n, acc_m / ow_, acc_m % ow_, k_o * config.K_block});
              }
            }

            _for_(c_o, 0, C_num_block) {
              _for_(r, 0, kh_) {
                _for_(s, 0, kw_) {
                  auto idx = c_o * kh_ * kw_ + r * kw_ + s;
                  std::vector<expr> input_pos = blocking_input_
                    ? std::vector<expr> {n, c_o,
                      ((o_o * config.tile_os) / adj_ow) * sh_ + r,
                      ((o_o * config.tile_os) % adj_ow) * sw_ + s, 0}
                    : std::vector<expr> {n,
                      ((o_o * config.tile_os) / adj_ow) * sh_ + r,
                      ((o_o * config.tile_os) % adj_ow) * sw_ + s,
                      c_o * config.C_block};
                  A_list[idx] = tensor_ptr(input, input_pos);
                  B_list[idx] = tensor_ptr(weight,
                    kpack > 1 ? std::vector<expr> {k_o, c_o, r, s, 0, 0, 0}
                              : std::vector<expr> {k_o, c_o, r, s, 0, 0});
                }
              }
            }

            const auto hint_A_size
              = config.tile_os * config.C_block * kh_ * kw_ * C_num_block;
            const auto hint_B_size
              = config.K_block * config.C_block * kh_ * kw_ * C_num_block;
            // note, the actual C_size is <= tile_os if pack_rows=true
            const auto hint_C_size = config.tile_os * config.K_block;
            sc_brgemm_attrs_t brg_attrs {
              {brgemm::attr_key::max_bs, kh_ * kw_ * C_num_block},
              {brgemm::attr_key::hint_expected_A_size, hint_A_size},
              {brgemm::attr_key::hint_expected_B_size, hint_B_size},
              {brgemm::attr_key::hint_expected_C_size, hint_C_size},
              {brgemm::attr_key::use_interleave_stores, true},
              {brgemm::attr_key::use_uker, true},
              {brgemm::attr_key::bd_mask_level, pack_rows ? 2 : 0}};

            sc::builtin::brgemm_init_list_update(A_list, B_list, out_tsr, 1,
              config.tile_os, config.K_block, config.C_block, LDA,
              config.K_block, LDC, 1 /*useless*/, 1 /*useless*/,
              kh_ * kw_ * C_num_block, get_input_dtype(), get_weight_dtype(),
              brg_attrs, os_mask, o_o, os / config.tile_os);
            auto os_num_block = os / config.tile_os;
            if (fusion && !pack_rows) {
              fusion->create_output_fusion_anchor({tensor_slice(output,
                blocking_output_
                  ? slice_range {{n, 1}, {k_o, 1},
                    {o_o * config.tile_os / ow_, 1},
                    {o_o * config.tile_os % ow_, config.tile_os},
                    {0, config.K_block}}
                  : slice_range {{n, 1}, {o_o * config.tile_os / ow_, 1},
                    {o_o * config.tile_os % ow_, config.tile_os},
                    {k_o * config.K_block, config.K_block}})});
            } else if (fusion && oh_ % os_num_block == 0) {
              fusion->create_output_fusion_anchor({tensor_slice(output,
                blocking_output_
                  ? slice_range {{n, 1}, {k_o, 1},
                    {o_o * (oh_ / os_num_block), (oh_ / os_num_block)},
                    {0, ow_}, {0, config.K_block}}
                  : slice_range {{n, 1},
                    {o_o * (oh_ / os_num_block), (oh_ / os_num_block)},
                    {0, ow_}, {k_o * config.K_block, config.K_block}})});
            }
          }
          if (fusion) {
            // Note: slice tensor might across multi-rows with non-rectangular
            // shapes. Currently, we just promote the fusion anchor to higher
            // level of loop, which will consume larger buffer and is
            // non-optimal This can be optimized in next version of fusion
            // manager.
            fusion->create_output_fusion_anchor({tensor_slice(output,
              blocking_output_ ? slice_range {{n, 1}, {k_o, 1}, {0, oh_},
                {0, ow_}, {0, config.K_block}}
                               : slice_range {{n, 1}, {0, oh_}, {0, ow_},
                                 {k_o * config.K_block, config.K_block}})});
          }
        } else {
          _named_for_(lp, p_o, 0, oh_ / config.tile_p) {
            _tensor_(A_list, datatypes::pointer, {kh_ * kw_ * C_num_block});
            _tensor_(B_list, datatypes::pointer, {kh_ * kw_ * C_num_block});
            _for_(q_o, 0, ow_ / config.tile_q) {
              _for_(p_i, 0, config.tile_p) {
                std::vector<expr> output_pos = blocking_output_
                  ? std::vector<expr> {n, k_o, p_o * config.tile_p + p_i,
                    q_o * config.tile_q, 0}
                  : std::vector<expr> {n, p_o * config.tile_p + p_i,
                    q_o * config.tile_q, k_o * config.K_block};
                _for_(c_o, 0, C_num_block) {
                  _for_(r, 0, kh_) {
                    _for_(s, 0, kw_) {
                      auto idx = c_o * kh_ * kw_ + r * kw_ + s;
                      std::vector<expr> input_pos = blocking_input_
                        ? std::vector<expr> {n, c_o,
                          (p_o * config.tile_p + p_i) * sh_ + r,
                          q_o * config.tile_q * sw_ + s, 0}
                        : std::vector<expr> {n,
                          (p_o * config.tile_p + p_i) * sh_ + r,
                          q_o * config.tile_q * sw_ + s, c_o * config.C_block};

                      A_list[idx] = tensor_ptr(input, input_pos);
                      B_list[idx] = tensor_ptr(weight,
                        kpack > 1 ? std::vector<expr> {k_o, c_o, r, s, 0, 0, 0}
                                  : std::vector<expr> {k_o, c_o, r, s, 0, 0});
                    }
                  }
                }

                const auto hint_A_size
                  = config.tile_q * config.C_block * kh_ * kw_ * C_num_block;
                const auto hint_B_size
                  = config.K_block * config.C_block * kh_ * kw_ * C_num_block;
                const auto hint_C_size = config.tile_q * config.K_block;
                sc_brgemm_attrs_t brg_attrs {
                  {brgemm::attr_key::max_bs, kh_ * kw_ * C_num_block},
                  {brgemm::attr_key::hint_expected_A_size, hint_A_size},
                  {brgemm::attr_key::hint_expected_B_size, hint_B_size},
                  {brgemm::attr_key::hint_expected_C_size, hint_C_size},
                  {brgemm::attr_key::use_interleave_stores, true},
                  {brgemm::attr_key::use_uker, true},
                  {brgemm::attr_key::bd_mask_level, 0}};

                sc::builtin::brgemm_init_list_update(A_list, B_list,
                  tensor_ptr(output, output_pos), 1, config.tile_q,
                  config.K_block, config.C_block, LDA, config.K_block, LDC,
                  1 /*useless*/, 1 /*useless*/, kh_ * kw_ * C_num_block,
                  get_input_dtype(), get_weight_dtype(), brg_attrs);

                if (fusion) {
                  fusion->create_output_fusion_anchor({tensor_slice(output,
                    blocking_output_
                      ? slice_range {{n, 1}, {k_o, 1},
                        {p_o * config.tile_p + p_i, 1},
                        {q_o * config.tile_q, config.tile_q},
                        {0, config.K_block}}
                      : slice_range {{n, 1}, {p_o * config.tile_p + p_i, 1},
                        {q_o * config.tile_q, config.tile_q},
                        {k_o * config.K_block, config.K_block}})});
                }
              }
              if (fusion) {
                fusion->create_output_fusion_anchor({tensor_slice(output,
                  blocking_output_
                    ? slice_range {{n, 1}, {k_o, 1},
                      {p_o * config.tile_p, config.tile_p},
                      {q_o * config.tile_q, config.tile_q}, {0, config.K_block}}
                    : slice_range {{n, 1}, {p_o * config.tile_p, config.tile_p},
                      {q_o * config.tile_q, config.tile_q},
                      {k_o * config.K_block, config.K_block}})});
              }
            }
            if (fusion) {
              fusion->create_output_fusion_anchor({tensor_slice(output,
                blocking_output_
                  ? slice_range {{n, 1}, {k_o, 1},
                    {p_o * config.tile_p, config.tile_p}, {0, ow_},
                    {0, config.K_block}}
                  : slice_range {{n, 1}, {p_o * config.tile_p, config.tile_p},
                    {0, ow_}, {k_o * config.K_block, config.K_block}})});
            }
          }
          if (fusion) {
            fusion->create_output_fusion_anchor({tensor_slice(output,
              blocking_output_ ? slice_range {{n, 1}, {k_o, 1}, {0, oh_},
                {0, ow_}, {0, config.K_block}}
                               : slice_range {{n, 1}, {0, oh_}, {0, ow_},
                                 {k_o * config.K_block, config.K_block}})});
          }
        }
      }
      if (fusion) {
        fusion->create_output_fusion_anchor({tensor_slice(output,
          blocking_output_ ? slice_range {{n, 1},
            {outer_k * K_num_block / oc_split, K_num_block / oc_split},
            {0, oh_}, {0, ow_}, {0, config.K_block}}
                           : slice_range {{n, 1}, {0, oh_}, {0, ow_},
                             {outer_k * K_num_block / oc_split * config.K_block,
                               K_num_block / oc_split * config.K_block}})});
      }
    }
  }
}

void gen_conv_fwd_t::compute_conv_padding(CONV_ARG_LIST) const {
  COMPILE_ASSERT(!is_3d_, "3D conv with padding is not supported yet!");
  assert(loops.size() == 4 && "expected to have 4 level loops!");
  for_loop &ln = loops.at(0), &lk = loops.at(1), &ld = loops.at(2),
           &lp = loops.at(3);
  COMPILE_ASSERT(blocking_input_ && blocking_output_,
    "Only blocking in&out are supported so far!");

  /* to do conv 3*3 with padding */
  std::unordered_set<int> Q1;
  std::unordered_set<int> Q2;
  std::unordered_set<int> Q3;

  int H_PADDED = ih_ + 2 * ph_, W_PADDED = iw_ + 2 * pw_;
  sc_dims padded_input_dims = blocking_input_
    ? sc_dims {mb_, C_num_block, H_PADDED, W_PADDED, config.C_block}
    : sc_dims {mb_, H_PADDED, W_PADDED, ic_};

  // collect the possible values for Q_tmp
  for (int p_o = 0; p_o < oh_ / config.tile_p; p_o++) {
    for (int q_o = 0; q_o < ow_ / config.tile_q; q_o++) {
      for (int p_i = 0; p_i < config.tile_p; p_i++) {
        int x_start_offset = tensor_offset(padded_input_dims,
          blocking_input_
            ? std::vector<int> {0, 0, (p_o * config.tile_p + p_i) * sh_,
              q_o * config.tile_q * sw_, 0}
            : std::vector<int> {0, (p_o * config.tile_p + p_i) * sh_,
              q_o * config.tile_q * sw_, 0});
        int x_threshold_left = tensor_offset(padded_input_dims,
          blocking_input_
            ? std::vector<int> {0, 0, (p_o * config.tile_p + p_i) * sh_, pw_, 0}
            : std::vector<int> {0, (p_o * config.tile_p + p_i) * sh_, pw_, 0});
        int x_threshold_right = tensor_offset(padded_input_dims,
          blocking_input_
            ? std::vector<int> {0, 0, (p_o * config.tile_p + p_i) * sh_,
              W_PADDED - pw_ - 1, 0}
            : std::vector<int> {
              0, (p_o * config.tile_p + p_i) * sh_, W_PADDED - pw_ - 1, 0});
        for (int s = 0; s < kw_; s++) {
          int pad_tmp
            = (x_threshold_left - (x_start_offset + s * config.C_block))
            / config.C_block;
          if (((x_start_offset + s * config.C_block) < x_threshold_left)
            && ((x_start_offset + s * config.C_block
                  + (config.tile_q - 1) * config.C_block * sw_)
              <= x_threshold_right)) {
            int interval = (pad_tmp + sw_ - 1) / sw_;
            int Q_tmp = config.tile_q - interval;
            Q1.insert(Q_tmp);
          } else {
            if (((x_start_offset + s * config.C_block) >= x_threshold_left)
              && ((x_start_offset + s * config.C_block
                    + (config.tile_q - 1) * config.C_block * sw_)
                > x_threshold_right)) {
              int Q_tmp
                = ((x_threshold_right - (x_start_offset + s * config.C_block))
                      / config.C_block
                    + sw_)
                / sw_;
              if (Q_tmp > 0) { Q2.insert(Q_tmp); }
            } else {
              int Q_tmp
                = (pad_tmp
                    + (x_threshold_right - x_threshold_left) / config.C_block
                    + sw_)
                  / sw_
                - (pad_tmp + sw_ - 1) / sw_;
              if (Q_tmp > 0) { Q3.insert(Q_tmp); }
            }
          }
        }
      }
    }
  }
  auto input_expr_dims = input.checked_as<tensor>()->dims_;
  auto mb_expr_ = input_expr_dims[0];
  _named_for_(ln, n, 0, mb_expr_, 1, for_type::PARALLEL) {
    _named_for_(lk, k_o, 0, K_num_block) {
      _named_for_(lp, p_o, 0, oh_ / config.tile_p) {
        _tensor_(A_list, datatypes::pointer, {kh_});
        _tensor_(B_list, datatypes::pointer, {kh_});
        _for_(q_o, 0, ow_ / config.tile_q) {
          _for_(p_i, 0, config.tile_p) {
            sc::builtin::mem_zero(
              tensor_ptr(output,
                {n, k_o, p_o * config.tile_p + p_i, q_o * config.tile_q, 0}),
              config.tile_q * config.K_block, get_output_dtype());
            _for_(c_o, 0, C_num_block) {
              _var_(x_threshold_top, datatypes::s32);
              _var_(x_threshold_bottom, datatypes::s32);
              _var_(x_threshold_left, datatypes::s32);
              _var_(x_threshold_right, datatypes::s32);
              _var_(x_start_offset, datatypes::s32);
              x_threshold_top
                = tensor_offset(padded_input_dims, {0, 0, (expr)ph_, 0, 0});
              x_threshold_bottom = tensor_offset(
                padded_input_dims, {0, 0, (expr)(H_PADDED - ph_), 0, 0});
              x_threshold_left = tensor_offset(padded_input_dims,
                {0, 0, (p_o * config.tile_p + p_i) * sh_, pw_, 0});
              x_threshold_right = tensor_offset(padded_input_dims,
                {0, 0, (p_o * config.tile_p + p_i) * sh_, W_PADDED - pw_ - 1,
                  0});
              x_start_offset = tensor_offset(padded_input_dims,
                {0, 0, (p_o * config.tile_p + p_i) * sh_,
                  q_o * config.tile_q * sw_, 0});

              _var_(x_tmp_offset, datatypes::s32);
              _var_(tmp, datatypes::s32);
              _var_(cnt, datatypes::s32);
              cnt = 0;
              _var_(head, datatypes::s32);
              head = -1;
              _for_(s, 0, kw_) {
                _if_(((x_start_offset + s * config.C_block) >= x_threshold_left)
                  && ((x_start_offset + s * config.C_block
                        + (config.tile_q - 1) * config.C_block * sw_)
                    <= x_threshold_right)) {
                  cnt = cnt + 1;
                  _if_(head == -1) head = builder::make_cast(datatypes::s32, s);
                }
                _else_ {
                  _var_(pad_tmp, datatypes::s32);
                  _var_(interval, datatypes::s32);
                  _var_(Q_tmp, datatypes::s32);
                  _if_(
                    ((x_start_offset + s * config.C_block) < x_threshold_left)
                    && ((x_start_offset + s * config.C_block
                          + (config.tile_q - 1) * config.C_block * sw_)
                      <= x_threshold_right)) {
                    pad_tmp = (x_threshold_left
                                - (x_start_offset
                                  + builder::make_cast(datatypes::s32, s)
                                    * config.C_block))
                      / config.C_block;
                    interval = (pad_tmp + sw_ - 1) / sw_;
                    Q_tmp = config.tile_q - interval;
                    _if_(Q_tmp > 0) {
                      tmp = 0;
                      _for_(r, 0, kh_) {
                        x_tmp_offset = tensor_offset(padded_input_dims,
                          {0, 0, (p_o * config.tile_p + p_i) * sh_ + r,
                            q_o * config.tile_q * sw_ + s + interval * sw_, 0});
                        _if_(x_tmp_offset >= x_threshold_top
                          && x_tmp_offset < x_threshold_bottom) {
                          A_list[tmp] = tensor_ptr(input,
                            {n, c_o,
                              (p_o * config.tile_p + p_i) * sh_ + r - ph_,
                              q_o * config.tile_q * sw_ - pw_ + s
                                + interval * sw_,
                              0});
                          B_list[tmp] = tensor_ptr(weight,
                            kpack > 1
                              ? std::vector<expr> {k_o, c_o, r, s, 0, 0, 0}
                              : std::vector<expr> {k_o, c_o, r, s, 0, 0});
                          tmp = tmp + 1;
                        }
                      }
                      _if_(tmp > 0) {
                        if (Q1.size() == 1) {
                          sc::builtin::brgemm_list_update(A_list, B_list,
                            tensor_ptr(output,
                              {n, k_o, p_o * config.tile_p + p_i,
                                q_o * config.tile_q + interval, 0}),
                            1, *Q1.begin(), config.K_block, config.C_block,
                            sw_ * config.C_block, config.K_block,
                            config.K_block, config.C_block,
                            config.C_block * config.K_block, tmp,
                            get_input_dtype(), get_weight_dtype());
                        } else {
                          sc::builtin::brgemm_list_update(A_list, B_list,
                            tensor_ptr(output,
                              {n, k_o, p_o * config.tile_p + p_i,
                                q_o * config.tile_q + interval, 0}),
                            1, Q_tmp, config.K_block, config.C_block,
                            sw_ * config.C_block, config.K_block,
                            config.K_block, config.C_block,
                            config.C_block * config.K_block, tmp,
                            get_input_dtype(), get_weight_dtype());
                        }
                      }
                    }
                  }
                  _else_ {
                    _if_(((x_start_offset + s * config.C_block)
                           >= x_threshold_left)
                      && ((x_start_offset + s * config.C_block
                            + (config.tile_q - 1) * config.C_block * sw_)
                        > x_threshold_right)) {
                      Q_tmp = ((x_threshold_right
                                 - (x_start_offset
                                   + builder::make_cast(datatypes::s32, s)
                                     * config.C_block))
                                  / config.C_block
                                + sw_)
                        / sw_;
                      _if_(Q_tmp > 0) {
                        tmp = 0;
                        _for_(r, 0, kh_) {
                          x_tmp_offset = tensor_offset(padded_input_dims,
                            {0, 0, (p_o * config.tile_p + p_i) * sh_ + r,
                              q_o * config.tile_q * sw_ + s, 0});
                          _if_(x_tmp_offset >= x_threshold_top
                            && x_tmp_offset < x_threshold_bottom) {
                            A_list[tmp] = tensor_ptr(input,
                              {n, c_o,
                                (p_o * config.tile_p + p_i) * sh_ + r - ph_,
                                q_o * config.tile_q * sw_ - pw_ + s, 0});
                            B_list[tmp] = tensor_ptr(weight,
                              kpack > 1
                                ? std::vector<expr> {k_o, c_o, r, s, 0, 0, 0}
                                : std::vector<expr> {k_o, c_o, r, s, 0, 0});
                            tmp = tmp + 1;
                          }
                        }
                        _if_(tmp > 0) {
                          if (Q2.size() == 1) {
                            sc::builtin::brgemm_list_update(A_list, B_list,
                              tensor_ptr(output,
                                {n, k_o, p_o * config.tile_p + p_i,
                                  q_o * config.tile_q, 0}),
                              1, *Q2.begin(), config.K_block, config.C_block,
                              sw_ * config.C_block, config.K_block,
                              config.K_block, config.C_block,
                              config.C_block * config.K_block, tmp,
                              get_input_dtype(), get_weight_dtype());
                          } else {
                            sc::builtin::brgemm_list_update(A_list, B_list,
                              tensor_ptr(output,
                                {n, k_o, p_o * config.tile_p + p_i,
                                  q_o * config.tile_q, 0}),
                              1, Q_tmp, config.K_block, config.C_block,
                              sw_ * config.C_block, config.K_block,
                              config.K_block, config.C_block,
                              config.C_block * config.K_block, tmp,
                              get_input_dtype(), get_weight_dtype());
                          }
                        }
                      }
                    }
                    _else_ {
                      pad_tmp = (x_threshold_left
                                  - (x_start_offset
                                    + builder::make_cast(datatypes::s32, s)
                                      * config.C_block))
                        / config.C_block;
                      interval = (pad_tmp + sw_ - 1) / sw_;
                      Q_tmp = (pad_tmp
                                + (x_threshold_right - x_threshold_left)
                                  / config.C_block
                                + sw_)
                          / sw_
                        - (pad_tmp + sw_ - 1) / sw_;
                      _if_(Q_tmp > 0) {
                        tmp = 0;
                        _for_(r, 0, kh_) {
                          x_tmp_offset = tensor_offset(padded_input_dims,
                            {0, 0, (p_o * config.tile_p + p_i) * sh_ + r,
                              q_o * config.tile_q * sw_ + s + interval * sw_,
                              0});
                          _if_(x_tmp_offset >= x_threshold_top
                            && x_tmp_offset < x_threshold_bottom) {
                            A_list[tmp] = tensor_ptr(input,
                              {n, c_o,
                                (p_o * config.tile_p + p_i) * sh_ + r - ph_,
                                q_o * config.tile_q * sw_ - pw_ + s
                                  + interval * sw_,
                                0});
                            B_list[tmp] = tensor_ptr(weight,
                              kpack > 1
                                ? std::vector<expr> {k_o, c_o, r, s, 0, 0, 0}
                                : std::vector<expr> {k_o, c_o, r, s, 0, 0});
                            tmp = tmp + 1;
                          }
                        }
                        _if_(tmp > 0) {
                          if (Q3.size() == 1) {
                            sc::builtin::brgemm_list_update(A_list, B_list,
                              tensor_ptr(output,
                                {n, k_o, p_o * config.tile_p + p_i,
                                  q_o * config.tile_q + interval, 0}),
                              1, *Q3.begin(), config.K_block, config.C_block,
                              sw_ * config.C_block, config.K_block,
                              config.K_block, config.C_block,
                              config.C_block * config.K_block, tmp,
                              get_input_dtype(), get_weight_dtype());
                          } else {
                            sc::builtin::brgemm_list_update(A_list, B_list,
                              tensor_ptr(output,
                                {n, k_o, p_o * config.tile_p + p_i,
                                  q_o * config.tile_q + interval, 0}),
                              1, Q_tmp, config.K_block, config.C_block,
                              sw_ * config.C_block, config.K_block,
                              config.K_block, config.C_block,
                              config.C_block * config.K_block, tmp,
                              get_input_dtype(), get_weight_dtype());
                          }
                        }
                      }
                    }
                  }
                }
              }
              _if_(cnt > 0) {
                tmp = 0;
                _for_(r, 0, kh_) {
                  x_tmp_offset = tensor_offset(padded_input_dims,
                    {0, 0, (p_o * config.tile_p + p_i) * sh_ + r,
                      q_o * config.tile_q * sw_ + head, 0});
                  _if_(x_tmp_offset >= x_threshold_top
                    && x_tmp_offset < x_threshold_bottom) {
                    A_list[tmp] = tensor_ptr(input,
                      {n, c_o, (p_o * config.tile_p + p_i) * sh_ + r - ph_,
                        q_o * config.tile_q * sw_ - pw_ + head, 0});
                    B_list[tmp] = tensor_ptr(weight,
                      kpack > 1 ? std::vector<expr> {k_o, c_o, r, head, 0, 0, 0}
                                : std::vector<expr> {k_o, c_o, r, head, 0, 0});
                    tmp = tmp + 1;
                  }
                }
                _if_(tmp > 0) {
                  sc::builtin::brgemm_list_update(A_list, B_list,
                    tensor_ptr(output,
                      {n, k_o, p_o * config.tile_p + p_i, q_o * config.tile_q,
                        0}),
                    cnt, config.tile_q, config.K_block, config.C_block,
                    sw_ * config.C_block, config.K_block, config.K_block,
                    config.C_block, config.C_block * config.K_block, tmp,
                    get_input_dtype(), get_weight_dtype());
                }
              }
            }

            if (fusion) {
              fusion->create_output_fusion_anchor({tensor_slice(output,
                {{n, 1}, {k_o, 1}, {p_o * config.tile_p + p_i, 1},
                  {q_o * config.tile_q, config.tile_q}, {0, config.K_block}})});
            }
          }
        }
      }
    }
  }
}

void gen_conv_fwd_t::compute_conv_padding_v2(CONV_ARG_LIST) const {
  COMPILE_ASSERT(!is_3d_, "3D conv with padding is not supported yet!");
  assert(loops.size() == 4 && "expected to have 4 level loops!");
  for_loop &ln = loops.at(0), &lk = loops.at(1), &ld = loops.at(2),
           &lp = loops.at(3);

  auto LDA = blocking_input_ ? config.C_block : ic_;
  auto LDC = blocking_output_ ? config.K_block : oc_;

  int ih_padded = ih_ + 2 * ph_, iw_padded = iw_ + 2 * pw_;
  auto dtypeInput = get_input_dtype();
  auto dtypeWeight = get_weight_dtype();
  auto dtypeOutput = get_output_dtype();

  const int src_row_tile_size = (config.tile_q - 1) * sw_ + kw_;
  /** calculate the unpadded point of spatial space in output tensor
   *   +-----------------------+
   *   |p p p p ...    p p p p |
   *   |p a x x ...    x x b p |
   *   |p x x x ...    x x x p |
   *   |p x x x ...    x x x p |
   *   |p x x x ...    x x x p |
   *   |p c x x ...    x x d p |
   *   |p p p p ...    p p p p |
   *   +-----------------------+
   *  where:
   *    p: pad area
   *    x: valid area
   *    a: (y_unpad_top, y_unpad_left)
   *    b: (y_unpad_top, y_unpad_right)
   *    c: (y_unpad_bottom, y_unpad_left)
   *    d: (y_unpad_bottom, y_unpad_right)
   */

  // some shapes might not have bottom or right pad at all
  auto get_num_pad_end = [](int ip, int k, int s, int p) {
    int remaining = (ip - k) % s;
    int num_pad_end = (remaining == 0)
      ? utils::divide_and_ceil(p, s)
      : ((p > remaining) ? utils::divide_and_ceil(p - remaining, s) : 0);
    return num_pad_end;
  };
  const int dst_num_pad_top = utils::divide_and_ceil(ph_, sh_);
  const int dst_num_pad_left = utils::divide_and_ceil(pw_, sw_);
  const int dst_num_pad_bottom = get_num_pad_end(ih_padded, kh_, sh_, ph_);
  const int dst_num_pad_right = get_num_pad_end(iw_padded, kw_, sw_, pw_);

  int y_unpad_top = dst_num_pad_top;
  int y_unpad_bottom = oh_ - dst_num_pad_bottom - 1;
  int y_unpad_left = dst_num_pad_left;
  int y_unpad_right = ow_ - dst_num_pad_right - 1;

  // create a global shared zero-buffer referenced by padding
  _tensor_(pbuffer, dtypeInput, {src_row_tile_size, LDA});
  sc::builtin::mem_zero(pbuffer, src_row_tile_size * LDA, dtypeInput);

  uint32_t lanes = get_lanes(ctx, config.C_block, dtypeInput);
  auto input_expr_dims = input.checked_as<tensor>()->dims_;
  auto mb_expr_ = input_expr_dims[0];
  _named_for_(ln, n, 0, mb_expr_, 1, for_type::PARALLEL) {
    _named_for_(lk, k_o, 0, K_num_block) {
      _named_for_(lp, p_o, 0, oh_ / config.tile_p) {
        _tensor_(A_list, datatypes::pointer, {kh_ * kw_});
        _tensor_(B_list, datatypes::pointer, {kh_ * kw_});
        // create a sub-tensor with maximum size which holds all the boundary
        // that contains padding
        _tensor_(sub_tensor, dtypeInput, {kh_, src_row_tile_size, LDA});
        _var_(pad_begin_index, datatypes::index);
        _var_(pad_end_index, datatypes::index);
        _var_(unpad_begin_index, datatypes::index);
        _var_(unpad_end_index, datatypes::index);
        _var_(real_pad_left, datatypes::u32);
        _var_(real_pad_right, datatypes::u32);
        _var_(num_pad_rows, datatypes::u32);
        _var_(copy_width, datatypes::u32);

        _for_(q_o, 0, ow_ / config.tile_q) {
          _for_(p_i, 0, config.tile_p) {
            std::vector<expr> output_pos = blocking_output_
              ? std::vector<expr> {n, k_o, p_o * config.tile_p + p_i,
                q_o * config.tile_q, 0}
              : std::vector<expr> {n, p_o * config.tile_p + p_i,
                q_o * config.tile_q, k_o * config.K_block};

            sc::builtin::brgemm_init(tensor_ptr(output, output_pos),
              config.tile_q, config.K_block, LDC, dtypeOutput, 0);
            _for_(c_o, 0, C_num_block) {
              // 1) top or bottom region with padding inputs
              // 1.1) calculate the number of padding rows
              _if_(((p_o * config.tile_p + p_i) >= y_unpad_top)
                && ((p_o * config.tile_p + p_i) <= y_unpad_bottom)) {
                num_pad_rows = 0;
                pad_begin_index = 0;
                pad_end_index = 0;
                unpad_begin_index = 0;
                unpad_end_index = kh_;
              }
              _else_ {
                _if_((p_o * config.tile_p + p_i) < y_unpad_top) {
                  num_pad_rows = ph_
                    - builder::make_cast(
                        datatypes::u32, p_o * config.tile_p + p_i)
                      * sh_;
                  pad_begin_index = 0;
                  pad_end_index = num_pad_rows;
                  unpad_begin_index = num_pad_rows;
                  unpad_end_index = kh_;
                }
                _else_ {
                  num_pad_rows = builder::make_cast(
                                   datatypes::u32, p_o * config.tile_p + p_i)
                      * sh_
                    + kh_ - (ih_ + ph_);
                  pad_begin_index = kh_ - num_pad_rows;
                  pad_end_index = kh_;
                  unpad_begin_index = 0;
                  unpad_end_index = kh_ - num_pad_rows;
                }

                // 1.2) Add zero-padding tensor to A_list
                _for_(r, pad_begin_index, pad_end_index) {
                  _for_(s, 0, kw_) {
                    _var_(idx, datatypes::u32);
                    idx = builder::make_cast(datatypes::u32, r * kw_ + s);
                    A_list[idx] = tensor_ptr(pbuffer, {0, 0});
                  }
                }
              }

              // 1.3) copy sub-tensor and append to A_list
              _if_(num_pad_rows < kh_) {
                // 1.3.1) copy sub-tensor
                _if_(q_o * config.tile_q < y_unpad_left) {
                  _if_((q_o * config.tile_q + config.tile_q - 1)
                    <= y_unpad_right) {
                    // 1.3.1.1) left pad only
                    real_pad_left = pw_
                      - builder::make_cast(
                        datatypes::u32, q_o * config.tile_q * sw_);

                    // copy sub-tensor
                    _for_(i, unpad_begin_index, unpad_end_index) {
                      sc::builtin::brgemm_init(
                        tensor_ptr(sub_tensor, {i - unpad_begin_index, 0, 0}),
                        builder::make_cast(datatypes::s32, real_pad_left),
                        config.C_block, LDA, dtypeInput, 0);

                      // mapping dst to padding src, then mapping
                      // padding src to real src to get the actual elements.
                      _for_(j, real_pad_left, src_row_tile_size) {
                        _for_(k, 0, config.C_block, (int)lanes) {
                          sub_tensor[span_t(
                            {i - unpad_begin_index, j, k}, lanes)]
                            = input[blocking_input_
                                ? span_t(
                                  {n, c_o,
                                    (p_o * config.tile_p + p_i) * sh_ + i - ph_,
                                    q_o * config.tile_q * sw_ + j - pw_, k},
                                  lanes)
                                : span_t(
                                  {n,
                                    (p_o * config.tile_p + p_i) * sh_ + i - ph_,
                                    q_o * config.tile_q * sw_ + j - pw_,
                                    c_o * config.C_block + k},
                                  lanes)];
                        }
                      }
                    }

                    _for_(r, unpad_begin_index, unpad_end_index) {
                      _for_(s, 0, kw_) {
                        _var_(idx, datatypes::u32);
                        idx = builder::make_cast(datatypes::u32, r * kw_ + s);
                        A_list[idx] = tensor_ptr(
                          sub_tensor, {r - unpad_begin_index, s, 0});
                      }
                    }
                  }
                  _else_ {
                    // 1.3.1.2) both left and right pad
                    real_pad_left = pw_
                      - builder::make_cast(
                        datatypes::u32, q_o * config.tile_q * sw_);
                    real_pad_right
                      = builder::make_cast(datatypes::u32,
                          q_o * config.tile_q * sw_ + src_row_tile_size)
                      - (iw_padded - pw_);

                    copy_width
                      = src_row_tile_size - real_pad_left - real_pad_right;

                    // copy sub-tensor
                    _for_(i, unpad_begin_index, unpad_end_index) {
                      // memzero left part
                      sc::builtin::brgemm_init(
                        tensor_ptr(sub_tensor, {i - unpad_begin_index, 0, 0}),
                        builder::make_cast(datatypes::s32, real_pad_left),
                        config.C_block, LDA, dtypeInput, 0);

                      _for_(j, real_pad_left, copy_width + real_pad_left) {
                        _for_(k, 0, config.C_block, (int)lanes) {
                          // N, C, H, W, c
                          sub_tensor[span_t(
                            {i - unpad_begin_index, j, k}, lanes)]
                            = input[blocking_input_
                                ? span_t(
                                  {n, c_o,
                                    (p_o * config.tile_p + p_i) * sh_ + i - ph_,
                                    q_o * config.tile_q * sw_ + j - pw_, k},
                                  lanes)
                                : span_t(
                                  {n,
                                    (p_o * config.tile_p + p_i) * sh_ + i - ph_,
                                    q_o * config.tile_q * sw_ + j - pw_,
                                    c_o * config.C_block + k},
                                  lanes)];
                        }
                      }

                      sc::builtin::brgemm_init(
                        tensor_ptr(sub_tensor,
                          {i - unpad_begin_index, copy_width + real_pad_left,
                            0}),
                        builder::make_cast(datatypes::s32, real_pad_right),
                        config.C_block, LDA, dtypeInput, 0);
                    }

                    _for_(r, unpad_begin_index, unpad_end_index) {
                      _for_(s, 0, kw_) {
                        _var_(idx, datatypes::u32);
                        idx = builder::make_cast(datatypes::u32, r * kw_ + s);
                        A_list[idx] = tensor_ptr(
                          sub_tensor, {r - unpad_begin_index, s, 0});
                      }
                    }
                  }
                }
                _else_ {
                  _if_((q_o * config.tile_q + config.tile_q - 1)
                    <= y_unpad_right) {
                    // 1.3.1.3) not using pad buffer, use original buffer
                    _for_(r, unpad_begin_index, unpad_end_index) {
                      _for_(s, 0, kw_) {
                        _var_(idx, datatypes::u32);
                        idx = builder::make_cast(datatypes::u32, r * kw_ + s);
                        A_list[idx] = tensor_ptr(input,
                          blocking_input_
                            ? std::vector<expr> {n, c_o,
                              (p_o * config.tile_p + p_i) * sh_ + r - ph_,
                              q_o * config.tile_q * sw_ + s - pw_, 0}
                            : std::vector<expr> {n,
                              (p_o * config.tile_p + p_i) * sh_ + r - ph_,
                              q_o * config.tile_q * sw_ + s - pw_,
                              c_o * config.C_block});
                      }
                    }
                  }
                  _else_ {
                    // 1.3.1.4) right pad only
                    real_pad_right
                      = builder::make_cast(datatypes::u32,
                          q_o * config.tile_q * sw_ + src_row_tile_size)
                      - (iw_padded - pw_);
                    copy_width = src_row_tile_size - real_pad_right;
                    // copy sub-tensor

                    _for_(i, unpad_begin_index, unpad_end_index) {
                      _for_(j, 0, copy_width) {
                        _for_(k, 0, config.C_block, (int)lanes) {
                          sub_tensor[span_t(
                            {i - unpad_begin_index, j, k}, lanes)]
                            = input[blocking_input_
                                ? span_t(
                                  {n, c_o,
                                    (p_o * config.tile_p + p_i) * sh_ + i - ph_,
                                    q_o * config.tile_q * sw_ + j - pw_, k},
                                  lanes)
                                : span_t(
                                  {n,
                                    (p_o * config.tile_p + p_i) * sh_ + i - ph_,
                                    q_o * config.tile_q * sw_ + j - pw_,
                                    c_o * config.C_block + k},
                                  lanes)];
                        }
                      }
                      sc::builtin::brgemm_init(
                        tensor_ptr(
                          sub_tensor, {i - unpad_begin_index, copy_width, 0}),
                        builder::make_cast(datatypes::s32, real_pad_right),
                        config.C_block, LDA, dtypeInput, 0);
                    }

                    _for_(r, unpad_begin_index, unpad_end_index) {
                      _for_(s, 0, kw_) {
                        _var_(idx, datatypes::u32);
                        idx = builder::make_cast(datatypes::u32, r * kw_ + s);
                        A_list[idx] = tensor_ptr(
                          sub_tensor, {r - unpad_begin_index, s, 0});
                      }
                    }
                  }
                }
              }

              // Add tensor to B_list
              _for_(r, 0, kh_) {
                _for_(s, 0, kw_) {
                  _var_(idx, datatypes::u32);
                  // inverse the idx
                  if (inverse_filter_) {
                    idx = builder::make_cast(
                      datatypes::u32, kh_ * kw_ - 1 - (r * kw_ + s));
                  } else {
                    idx = builder::make_cast(datatypes::u32, r * kw_ + s);
                  }
                  B_list[idx] = tensor_ptr(weight,
                    kpack > 1 ? std::vector<expr> {k_o, c_o, r, s, 0, 0, 0}
                              : std::vector<expr> {k_o, c_o, r, s, 0, 0});
                }
              }

              const auto hint_A_size
                = config.tile_q * config.C_block * kh_ * kw_;
              const auto hint_B_size = config.K_block * config.C_block;
              const auto hint_C_size = config.tile_q * config.K_block;
              sc_brgemm_attrs_t brg_attrs {
                {brgemm::attr_key::max_bs, kh_ * kw_},
                {brgemm::attr_key::hint_expected_A_size, hint_A_size},
                {brgemm::attr_key::hint_expected_B_size, hint_B_size},
                {brgemm::attr_key::hint_expected_C_size, hint_C_size},
                {brgemm::attr_key::use_interleave_stores, true},
                {brgemm::attr_key::use_uker, true}};

              sc::builtin::brgemm_list_update(A_list, B_list,
                tensor_ptr(output, output_pos), 1, config.tile_q,
                config.K_block, config.C_block, sw_ * LDA, config.K_block, LDC,
                1, 1, kh_ * kw_, dtypeInput, dtypeWeight, brg_attrs);
            }

            if (fusion) {
              fusion->create_output_fusion_anchor({tensor_slice(output,
                blocking_output_
                  ? slice_range {{n, 1}, {k_o, 1},
                    {p_o * config.tile_p + p_i, 1},
                    {q_o * config.tile_q, config.tile_q}, {0, config.K_block}}
                  : slice_range {{n, 1}, {p_o * config.tile_p + p_i, 1},
                    {q_o * config.tile_q, config.tile_q},
                    {k_o * config.K_block, config.K_block}})});
            }
          }
        }
      }
      if (fusion) {
        fusion->create_output_fusion_anchor({tensor_slice(output,
          blocking_output_ ? slice_range {{n, 1}, {k_o, 1}, {0, oh_}, {0, ow_},
            {0, config.K_block}}
                           : slice_range {{n, 1}, {0, oh_}, {0, ow_},
                             {k_o * config.K_block, config.K_block}})});
      }
    }
    if (fusion) {
      fusion->create_output_fusion_anchor({tensor_slice(output,
        blocking_output_
          ? slice_range {{n, 1}, {0, K_num_block}, {0, oh_}, {0, ow_},
            {0, config.K_block}}
          : slice_range {{n, 1}, {0, oh_}, {0, ow_}, {0, config.K_block}})});
    }
  }
}

void gen_conv_fwd_t::schedule_loops(context_ptr ctx,
  const conv_fwd_config_t &config, stmt body,
  std::vector<for_loop> &fors) const {
  if (use_conv1d) {
    auto lpbs = fors[0], lps = fors[1], lpoc = fors[2], lpic = fors[3];
    lpbs->fuse(lps)->fuse(lpoc)->fuse(lpic);
  } else {
    COMPILE_ASSERT(
      static_cast<int>(fors.size()) == 4 || static_cast<int>(fors.size()) == 5,
      "expected to have 4 for loops, but got " << fors.size() << " for loops.");
    for_loop ln = fors.at(0), lk = fors.at(1), ld = fors.at(2), lp = fors.at(3),
             lok;
    if (fors.size() == 5) {
      lok = fors[4];
      ln = lok->fuse(ln);
    }
    auto loop_sched = config.loop_sched;
    if (loop_sched == 0) {
      // default loop order ln->lk->ld->lp
      // merge ln, lk, lp
      auto outer = ln->fuse(lk);
      if (is_3d_) { outer->fuse(ld); }
      outer = outer->fuse(lp);
      outer->kind_ = for_type::PARALLEL;
    } else if (loop_sched == 1) {
      // loop order lk->lp->ln
      // merge lk, lp, ln
      for_loop outer;
      if (is_3d_) {
        ln->reorder(body, {lk, ld, lp, ln});
        outer = lk->fuse(ld);
        outer = outer->fuse(lp);
      } else {
        ln->reorder(body, {lk, lp, ln});
        outer = lk->fuse(lp);
      }
      outer = outer->fuse(ln);
      outer->kind_ = for_type::PARALLEL;
    } else if (loop_sched == 2) {
      // loop order lp->lk->ln
      // merge lp, lk, ln
      for_loop outer;
      if (is_3d_) {
        ln->reorder(body, {ld, lp, lk, ln});
        outer = ld->fuse(lp);
        outer = outer->fuse(lk);
      } else {
        ln->reorder(body, {lp, lk, ln});
        outer = lp->fuse(lk);
      }
      outer = outer->fuse(ln);
      outer->kind_ = for_type::PARALLEL;
    } else if (loop_sched == 3) {
      // loop order lk->ln->lp
      // merge lk,ln,lp
      ln->reorder(body, {lk, ln});
      auto outer = lk->fuse(ln);
      if (is_3d_) { outer = outer->fuse(ld); }
      outer = outer->fuse(lp);
      outer->kind_ = for_type::PARALLEL;
    }
  }
}

bool gen_conv_fwd_t::generate(context_ptr ctx, const conv_fwd_config_t &config,
  fusion_manager *fusion, const std::vector<expr> &inputs,
  const std::vector<expr> &outputs, std::vector<for_loop> &loops) const {
  COMPILE_ASSERT(inputs.size() == 2,
    "Expecting 2 inputs for conv, but got " << inputs.size() << " inputs.");
  COMPILE_ASSERT(outputs.size() == 1,
    "Expecting 1 output for conv, but got " << outputs.size() << " output.");

  if (!is_3d_) {
    COMPILE_ASSERT(id_ == 1 && kd_ == 1 && od_ == 1 && config.tile_d == 1,
      "id/kd/od/tile_d should be 1 for non-3D conv, but got id="
        << id_ << ", kd=" << kd_ << ", od=" << od_
        << ", tile_d=" << config.tile_d << ".");
  }

  int K_block = config.K_block;
  int C_block = config.C_block;
  int tile_d = config.tile_d;
  int tile_p = config.tile_p;
  int tile_q = config.tile_q;
  int tile_os = config.tile_os;
  int pack_input = config.pack_input;
  int loop_sched = config.loop_sched;
  int K_num_block = oc_ / K_block;
  int C_num_block = ic_ / C_block;
  const bool use_os_blocking = try_os_blocking_ && is_use_amx(ctx);
  const bool pack_rows = use_os_blocking && (tile_os > 0 && ow_ % tile_os != 0);
  int os = actual_os_;

  COMPILE_ASSERT(K_block && (oc_ % K_block == 0),
    "oc should be dividable by K_block, but got oc=" << oc_ << " K_block="
                                                     << K_block << ".");
  COMPILE_ASSERT(C_block && (ic_ % C_block == 0),
    "ic should be dividable by C_block, but got ic=" << ic_ << " C_block="
                                                     << C_block << ".");
  COMPILE_ASSERT(tile_d && (od_ % tile_d == 0),
    "od should be dividable by tile_d, but got od=" << od_ << " tile_d="
                                                    << tile_d << ".");

  // kpack is used to determine the vnni block format
  //  +----+--------------+
  //  | 1  | FP32         |
  //  +----+--------------+
  //  | 2  | VNNI_BF16    |
  //  +----+--------------+
  //  | 4  | VNNI_INT8    |
  //  +----+--------------+
  int kpack = 1;
  auto dtypeInput = get_input_dtype();
  auto dtypeWeight = get_weight_dtype();
  auto dtypeOutput = get_output_dtype();
  if (dtypeInput == datatypes::bf16) {
    COMPILE_ASSERT((dtypeWeight == datatypes::bf16),
      "Weights should be bf16 as "
      "data, the mixed datatypes is not supported yet!");
    COMPILE_ASSERT((dtypeOutput == datatypes::f32),
      "Output should be f32 when data and weights are in bf16.");
    kpack = 2;
  }
  if (utils::is_one_of(dtypeInput, datatypes::s8, datatypes::u8)) {
    COMPILE_ASSERT((dtypeWeight == datatypes::s8),
      "Weights should be s8 when \
            data is s8/u8, the mixed datatypes is not supported yet!");
    COMPILE_ASSERT((dtypeOutput == datatypes::s32),
      "Output should be s32 when data and weights are in "
      "s8/u8.");
    kpack = 4;
  }

  std::vector<char> os_mask = {};
  expr os_acc_size = expr();
  if (pack_rows) {
    os = adj_os_;
    int os_num_block = os / tile_os;
    int adj_ow = ow_ + num_elems_skip_per_ow_;
    os_mask.resize(os);
    for (int i = 0; i < os; ++i) {
      if (i % adj_ow < ow_) {
        os_mask[i] = 1;
      } else {
        os_mask[i] = 0;
      }
    }

    _tensor_(conv_os_acc_size, datatypes::s32, {os_num_block});
    int acc_size = 0;
    int blk_size = 0;
    for (int i = 0; i < os_num_block; ++i) {
      blk_size = std::accumulate(
        os_mask.begin() + i * tile_os, os_mask.begin() + (i + 1) * tile_os, 0);
      conv_os_acc_size[i] = acc_size;
      acc_size += blk_size;
    }

    os_acc_size = conv_os_acc_size;
  }

  if (use_os_blocking) {
    COMPILE_ASSERT((tile_os > 0) && (os % tile_os == 0),
      "os should be dividable by tile_os, but got os=" << os << " tile_os="
                                                       << tile_os << ".");
  } else {
    COMPILE_ASSERT((tile_p > 0) && (oh_ % tile_p == 0),
      "oh should be dividable by tile_p, but got oh=" << oh_ << " tile_p="
                                                      << tile_p << ".");
    COMPILE_ASSERT((tile_q > 0) && (ow_ % tile_q == 0),
      "ow should be dividable by tile_q, but got ow=" << ow_ << " tile_q="
                                                      << tile_q << ".");
  }

  for_loop ln, lk, ld, lp;
  loops = {ln, lk, ld, lp};
  expr output = outputs[op_params_t::out];
  expr input = inputs[op_params_t::in_data];
  expr weight = inputs[op_params_t::in_weight];
  if (use_conv1d) {
    // no padding/stride 1x1 1d/2d
    compute_conv1d(ctx, config, fusion, output, input, weight, loops,
      K_num_block, C_num_block, os, kpack);
  } else if (is_1x1_conv_) {
    COMPILE_ASSERT(
      pd_ == 0 && ph_ == 0 && pw_ == 0, "1x1 conv doesn't support padding!");
    COMPILE_ASSERT(
      !inverse_filter_, "1x1 conv doesn't support inverse convolution.");
    if (is_3d_ || (pack_input == 0 && (sd_ > 1 || sh_ > 1 || sw_ > 1))) {
      compute_1x1_no_pack_input(ctx, config, fusion, output, input, weight,
        loops, K_num_block, C_num_block, os, kpack);
    } else {
      compute_1x1_pack_input(ctx, config, fusion, output, input, weight, loops,
        K_num_block, C_num_block, os, kpack);
    }
  } else {
    if (pd_ == 0 && ph_ == 0 && pw_ == 0) {
      COMPILE_ASSERT(!inverse_filter_,
        "conv NxN (no padding) does not support inverse convolution.");
      if (is_3d_) {
        compute_conv3d_no_padding(ctx, config, fusion, output, input, weight,
          loops, K_num_block, C_num_block, os, kpack);
      } else {
        compute_conv_no_padding(ctx, config, fusion, output, input, weight,
          loops, K_num_block, C_num_block, os, kpack, use_os_blocking,
          pack_rows, os_acc_size, os_mask);
      }
    } else {
      if (is_use_amx(ctx) && (ph_ <= kh_ && pw_ <= kw_)) {
        if (inverse_filter_) {
          SC_INFO << "inverse_filter_ used in conv padding v2.";
        }
        compute_conv_padding_v2(ctx, config, fusion, output, input, weight,
          loops, K_num_block, C_num_block, os, kpack);
      } else {
        COMPILE_ASSERT(!inverse_filter_,
          "conv padding v1 does not support inverse convolution.");
        compute_conv_padding(ctx, config, fusion, output, input, weight, loops,
          K_num_block, C_num_block, os, kpack);
      }
    }
  }
  return true;
}
#undef CONV_ARG_LIST

} // namespace ops
} // namespace sc
