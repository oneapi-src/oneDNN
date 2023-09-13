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
#include <ops/templates/conv_rl.hpp>
#include <runtime/barrier.hpp>
#include <runtime/config.hpp>
#include <unordered_set>
#include <util/any_map.hpp>
#include <util/math_utils.hpp>
#include <util/reflection.hpp>
#include <util/utils.hpp>
using namespace dnnl::impl::graph::gc::builder;
namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

using ops::conv_fwd_config_t;
// clang-format off
SC_CLASS(conv_fwd_config_t)
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

void gen_conv_fwd_t::validate_conv_fwd_default_config(
  const context_ptr &ctx, conv_fwd_config_t &cfg) const {
  bool dtype_f32 = get_input_dtype() == datatypes::f32;
  const bool use_os_blocking
    = try_os_blocking_ && ops::is_amx_dtype(ctx, get_input_dtype());
  auto K_block_list = utils::get_blocks(oc_, 16);
  auto C_block_list = utils::get_blocks(ic_, 16);
  auto tile_d_list = utils::get_factors(od_);
  auto tile_p_list
    = use_os_blocking ? std::vector<int> {-1} : utils::get_factors(oh_);
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

void gen_conv_fwd_t::adjust_config_for_parallelisem(
  const context_ptr &ctx, conv_fwd_config_t &cfg) const {
  const auto nthreads = runtime_config_t::get().get_num_threads();
  const int parallel_balance_boundary = 2 * nthreads;
  auto is_balance_parallel = [&](int K_block, int tile_p, int tile_d) {
    int work_amount = (oc_ / K_block) * (oh_ / tile_p) * (od_ / tile_d) * mb_;
    return is_parallel_space_enough(work_amount, nthreads);
  };

  const auto dtype_size = utils::get_sizeof_type(get_weight_dtype());

  if (!is_balance_parallel(cfg.K_block, cfg.tile_p, cfg.tile_d)) {
    int oc_minimum_block = utils::get_blocks(oc_, 1, 32).back();
    cfg.C_block = ic_;
    if (is_1x1_conv_ && oc_ / oc_minimum_block >= parallel_balance_boundary) {
      cfg.K_block = oc_minimum_block;
    }
    auto K_block_candidates
      = utils::get_blocks(oc_, oc_minimum_block, cfg.K_block - 1);
    auto tile_p_candidates = utils::get_blocks(oh_, 1, cfg.tile_p - 1);
    auto tile_d_candidates = utils::get_blocks(od_, 1, cfg.tile_d - 1);
    while (!is_balance_parallel(cfg.K_block, cfg.tile_p, cfg.tile_d)
      && cfg.K_block > oc_minimum_block) {
      cfg.K_block = K_block_candidates.back();
      K_block_candidates.pop_back();
    }
    while (!is_balance_parallel(cfg.K_block, cfg.tile_p, cfg.tile_d)
      && cfg.tile_p > 1) {
      cfg.tile_p = tile_p_candidates.back();
      tile_p_candidates.pop_back();
    }
    while (!is_balance_parallel(cfg.K_block, cfg.tile_p, cfg.tile_d)
      && cfg.tile_d > 1) {
      cfg.tile_d = tile_d_candidates.back();
      tile_d_candidates.pop_back();
    }
  }
}

void gen_conv_fwd_t::adjust_config_for_cache_efficiency(
  const context_ptr &ctx, conv_fwd_config_t &cfg) const {
  auto L2_cache_size = ctx->machine_.cpu_flags_.getDCacheSize(2);
  const auto dtype_size = utils::get_sizeof_type(get_weight_dtype());

  auto get_brgemm_A_size = [&](const conv_fwd_config_t &cfg) {
    return cfg.tile_q * ic_ * dtype_size;
  };
  auto get_brgemm_B_size = [&](const conv_fwd_config_t &cfg) {
    return kh_ * kw_ * ic_ * cfg.K_block * dtype_size;
  };
  auto get_brgemm_C_size = [&](const conv_fwd_config_t &cfg) {
    return cfg.tile_q * cfg.K_block * dtype_size;
  };

  if (get_brgemm_A_size(cfg) + get_brgemm_B_size(cfg) + get_brgemm_C_size(cfg)
    > L2_cache_size) {
    cfg.K_block = utils::get_blocks(oc_, 64).front();
    cfg.tile_q = utils::get_blocks(ow_, 16, 32).back();
  }

  // brgemm prefer small M if the smallest factor still larger
  if (cfg.tile_q > 150 && utils::get_blocks(ow_, 16, 32).back() > 32) {
    cfg.tile_q = utils::get_blocks(ow_, 16, 64).back();
    cfg.K_block = utils::get_blocks(oc_, 16, 512).back();
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
  bool is_vnni_low_fp = ops::is_vnni_low_fp(ctx, get_weight_dtype());
  bool no_vnni = ops::no_vnni(ctx, get_weight_dtype());
  auto dtype_size = no_vnni ? 4 : (is_vnni_low_fp ? 2 : 1);
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
      if (C_block_list[i] <= 128 && C_block_list[i] % 32 == 0) {
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
      if (K_block_list[i] <= 128 && K_block_list[i] % 32 == 0) {
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
    if ((ctx->use_amx() && get_weight_dtype() != datatypes::f32)
      || nthreads > mb_) {
      cfg.loop_sched = 2;
    } else {
      cfg.loop_sched = 0;
    }
  }

  auto L2_cache_size = ctx->machine_.cpu_flags_.getDCacheSize(2);
  if (is_1x1_conv_ && (oc_ / ic_ >= 4 && oc_ >= 1024)) { max_oc_block = 128; }
  // tile_p
  if (!is_1x1_conv_) {
    if (mb_ % nthreads == 0) {
      for (auto p_candidate : tile_p_list) {
        if (p_candidate
          >= 8 / static_cast<int>(utils::get_sizeof_type(get_weight_dtype()))) {
          cfg.tile_p = p_candidate;
          break;
        }
      }
    } else {
      // set tile_p == 1 to increase parallel space
      cfg.tile_p = 1;
    }
  } else {
    if (ow_ >= 16 || !is_parallel_space_enough(mb_ * oc_ / 64, nthreads)) {
      cfg.tile_p = 1;
    } else {
      if (oh_ >= 16) {
        cfg.tile_p = utils::get_blocks(oh_, 1, 64 / ow_).back();
      } else {
        cfg.tile_p = oh_;
      }
    }
  }
  if (get_input_dtype() == datatypes::f32 && is_1x1_conv_) { cfg.tile_p = 1; }
  // tile q
  if (!is_1x1_conv_) {
    cfg.tile_q = tile_q_list.back();
  } else {
    // handle large M for gemm kernel: shrink M
    if (sw_ > 1) {
      cfg.tile_q = 1;
    } else {
      cfg.tile_q = tile_q_list.back();
    }
  }
  if (try_os_blocking_ && ops::is_amx_dtype(ctx, get_input_dtype())) {
    // if use os blocking override tile p and tile q above
    cfg.tile_os = cfg.tile_q;
    auto os_choices = get_os_blocks(ow_, adj_os_);
    std::sort(os_choices.begin(), os_choices.end());
    if (ow_ <= 28 && ow_ % 16 != 0) {
      for (int i = os_choices.size() - 1; i >= 0; i--) {
        if (nthreads <= adj_os_ / os_choices[i] * mb_ * oc_ / 64) {
          cfg.tile_os = os_choices[i];
          break;
        }
      }
    }
    cfg.tile_q = -1;
    cfg.tile_p = -1;
  }

  bool has_pad = (pd_b_ > 0) || (ph_b_ > 0) || (pw_b_ > 0) || (pd_e_ > 0)
    || (ph_e_ > 0) || (pw_e_ > 0);
  if (is_1x1_conv_) {
    max_oc_block = std::max(
      max_oc_block, utils::get_blocks(oc_, 1, ow_ * 2 / dtype_size).back());
    cfg.K_block = oc_ % 32 == 0 ? max_oc_block : oc_;
  } else {
    // config observation: if oc is small enough, directly consuming all oc
    // would be better
    // A relative large K_block(128) is good for padding_v2 template
    auto run_on_amx = ops::is_amx_dtype(ctx, get_weight_dtype());
    bool small_oc = oc_ <= 128;
    bool using_v2_template = has_pad && (run_on_amx || is_3d_);
    if ((!run_on_amx || using_v2_template || small_oc) && !large_spatial) {
      auto default_block
        = small_oc || using_v2_template ? 128 : 128 / dtype_size;
      cfg.K_block = utils::get_blocks(oc_, 16, default_block).back();
    } else if (large_spatial) {
      cfg.K_block = utils::get_blocks(oc_, 16, 256).back();
    } else {
      // config observation: small oc block could provide a good performance for
      // conv3x3 on amx no padding case
      cfg.K_block = utils::get_blocks(oc_, 64).front();
    }
  }
  max_ic_block = std::max(
    max_ic_block, utils::get_blocks(ic_, 1, ow_ * 2 / dtype_size).back());
  cfg.C_block = ic_ % 32 == 0 ? max_ic_block : ic_;

  if (is_3d_ && has_pad) {
    if (mb_ * od_ / cfg.tile_d >= 8) {
      cfg.tile_p = utils::get_blocks(oh_, 1, 32).back();
    }
  }
  adjust_config_for_parallelisem(ctx, cfg);
  adjust_config_for_cache_efficiency(ctx, cfg);
  validate_conv_fwd_default_config(ctx, cfg);

  if (ic_ > 32 && cfg.C_block % 32 != 0) {
    // The performance will be very bad if C_bloc % 32 != 0 in convNxN
    cfg.C_block = utils::rnd_up(cfg.C_block, 64 / dtype_size);
  }
  if (oc_ > 128 && cfg.K_block % 32 != 0) {
    // The performance will be very bad if C_bloc % 32 != 0 in convNxN
    cfg.K_block = utils::rnd_up(cfg.K_block, 64 / dtype_size);
  }
  if (attrs_.get_or_else("use_rl", ops::rl_kind::NO_LOWERING)
    == ops::rl_kind::KW_LOWERING) {
    cfg.C_block = ic_;
  }

  if (inverse_filter_) {
    cfg.C_block = ic_ % 64 == 0 ? 64 : ic_;
    cfg.K_block = oc_ % 64 == 0 ? 64 : oc_;
    cfg.tile_d = 1;
    cfg.tile_p = 1;
    cfg.tile_q = ow_;
  }
  return std::move(ret);
}

gen_conv_fwd_t::gen_conv_fwd_t(sc_op *owner, const sc_dims &stride,
  const sc_dims &dilation, const sc_dims &pads_begin, const sc_dims &pads_end,
  std::vector<logical_tensor_t> &&ins, std::vector<logical_tensor_t> &&outs)
  : parent(owner, std::move(ins), std::move(outs)) {
  COMPILE_ASSERT(in_tensors_.size() == 2,
    "Wrong number of inputs, expected to be 2 but got " << in_tensors_.size()
                                                        << ".");
  COMPILE_ASSERT(out_tensors_.size() == 1,
    "Wrong number of output, expected to be 1 but got " << out_tensors_.size()
                                                        << ".");
  if (owner) { attrs_ = owner->attrs_; }
  auto use_rl = attrs_.get_or_else("use_rl", ops::rl_kind::NO_LOWERING);
  auto input_plain_dims = get_input_plain_dims();
  auto weight_plain_dims = use_rl == ops::rl_kind::KW_LOWERING
    ? attrs_.get<sc_dims>("origin_wei_plain_dims")
    : get_weight_plain_dims();
  auto out_plain_dims = get_output_plain_dims();

  ndims_ = input_plain_dims.size();
  COMPILE_ASSERT(utils::is_one_of(static_cast<int>(ndims_), 3, 4, 5),
    "Wrong input dims, expected to be 3D, 4D or 5D input, but got " << ndims_
                                                                    << "D.");
  COMPILE_ASSERT(
    utils::is_one_of(static_cast<int>(weight_plain_dims.size()), 3, 4, 5)
      && (weight_plain_dims.size() == ndims_),
    "Wrong weight dims, only support 3D, 4D or 5D weights, but got "
      << weight_plain_dims.size() << "D.");
  COMPILE_ASSERT(
    utils::is_one_of(static_cast<int>(out_plain_dims.size()), 3, 4, 5)
      && (out_plain_dims.size() == ndims_),
    "Wrong output dims, only support 3D, 4D or 5D weights, but got "
      << out_plain_dims.size() << "D.");

  is_3d_ = (ndims_ == 5);

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
  iw_ = input_plain_dims[ndims_ - 1];
  ih_ = input_plain_dims[ndims_ - 2];
  oc_ = weight_plain_dims[0] / groups_;
  kd_ = is_3d_ ? weight_plain_dims[2] : 1;
  kh_ = weight_plain_dims[ndims_ - 2];
  kw_ = weight_plain_dims[ndims_ - 1];
  od_ = is_3d_ ? out_plain_dims[2] : 1;
  oh_ = out_plain_dims[ndims_ - 2];
  ow_ = out_plain_dims[ndims_ - 1];
  is_1x1_conv_ = (kd_ == 1 && kh_ == 1 && kw_ == 1);
  pd_b_ = is_3d_ ? pads_begin[0] : 0;
  ph_b_ = pads_begin[0], pw_b_ = pads_begin[0];
  pd_e_ = is_3d_ ? pads_end[0] : 0;
  ph_e_ = pads_end[0], pw_e_ = pads_end[0];
  bool is_int8
    = utils::is_one_of(get_input_dtype(), datatypes::u8, datatypes::s8);
  bool is_bf16 = get_input_dtype() == datatypes::bf16;

  if (pads_begin.size() > 1) {
    ph_b_ = pads_begin[ndims_ - 4];
    pw_b_ = pads_begin[ndims_ - 3];
  }
  if (pads_end.size() > 1) {
    ph_e_ = pads_end[ndims_ - 4];
    pw_e_ = pads_end[ndims_ - 3];
  }
  sd_ = is_3d_ ? stride[0] : 1;
  sh_ = stride[0], sw_ = stride[0];
  if (stride.size() > 1) {
    auto stride_size = stride.size();
    sh_ = stride[stride_size - 2];
    sw_ = stride[stride_size - 1];
  }

  dd_ = is_3d_ ? dilation[0] : 1;
  dh_ = dilation[0], dw_ = dilation[0];
  auto dilation_size = dilation.size();
  if (dilation_size > 1) {
    dh_ = dilation[dilation_size - 2];
    dw_ = dilation[dilation_size - 1];
  }

  // For non 1x1 conv and AMX platform, spatial blocking instead of row
  // blocking is used, which needs to consider the border carefully, as the
  // cross row boundary (contains padding or not) will generate useless output
  // which have to be skipped before storing.
  actual_os_ = oh_ * ow_;
  num_elems_skip_per_ow_ = ((dw_ * (kw_ - 1)) / sw_) * sh_ + (sh_ - 1) * ow_;
  adj_os_ = std::min(actual_os_ + num_elems_skip_per_ow_ * (oh_ - 1),
    (ih_ + ph_b_ + ph_e_) * (iw_ + pw_b_ + pw_e_));

  // Note: os blocking is only valid for non_1x1, no pad and non 3D conv with
  // amx-int8 only so far.
  bool has_pad = (pd_b_ > 0) || (ph_b_ > 0) || (pw_b_ > 0) || (pd_e_ > 0)
    || (ph_e_ > 0) || (pw_e_ > 0);
  // TODO(zhicong): check whether to use os_blocking when sh > 1
  const auto num_threads = runtime_config_t::get().get_num_threads();
  try_os_blocking_ = (!is_1x1_conv_) && (!has_pad) && (!is_3d_) && sh_ == 1
    && ((is_int8 && ow_ <= 28) || (is_bf16 && ow_ <= 14))
    && is_parallel_space_enough(mb_ * (oc_ / 64), num_threads)
    && use_rl == ops::rl_kind::NO_LOWERING;
}

float gen_conv_fwd_t::get_gflop() const {
  float result = (float)mb_ * groups_ * oc_ * 2.0 * ic_ * kd_ * kh_ * kw_ * od_
    * oh_ * ow_ / (float)1e9;
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

static inline int get_oc_split_factor(const int mb, const int data_size,
  const int weight_size, const int L2_cache_size, const int K_num_block) {
  int oc_split = 1;
  auto nthreads = runtime_config_t::get().get_num_threads();
  if (weight_size >= L2_cache_size
    || (nthreads > mb && weight_size > data_size)) {
    // config observation: real time mode will prefer split oc first and
    // oc_split = nthreads.
    int expected_split_num
      = std::max(utils::divide_and_ceil(weight_size, L2_cache_size),
        utils::divide_and_ceil(nthreads, mb));
    for (auto &factor : utils::get_factors(K_num_block)) {
      if (factor >= expected_split_num) {
        expected_split_num = factor;
        break;
      }
    }
    oc_split = K_num_block < expected_split_num ? 1 : expected_split_num;
  }
  return oc_split;
}

static inline void create_anchor(fusion_manager *fusion, expr &output,
  const expr &n, const int n_len, const expr &k, const int k_len, const expr &d,
  const int d_len, const expr &p, const expr &p_len, const expr &q,
  const int q_len, const int K_block, const int inner_k_len,
  const bool blocking_output, const bool is_3d) {
  if (fusion) {
    if (is_3d) {
      fusion->create_output_fusion_anchor({tensor_slice(output,
        blocking_output ? slice_range {{n, n_len}, {k, k_len}, {d, d_len},
          {p, p_len}, {q, q_len}, {0, K_block}}
                        : slice_range {{n, n_len}, {d, d_len}, {p, p_len},
                          {q, q_len}, {k * K_block, inner_k_len}})});
    } else {
      fusion->create_output_fusion_anchor({tensor_slice(output,
        blocking_output ? slice_range {{n, n_len}, {k, k_len}, {p, p_len},
          {q, q_len}, {0, K_block}}
                        : slice_range {{n, n_len}, {p, p_len}, {q, q_len},
                          {k * K_block, inner_k_len}})});
    }
  }
}

#define CONV_ARG_LIST \
  const context_ptr &ctx, const conv_fwd_config_t &config, \
    fusion_manager *fusion, expr &output, const expr &input, \
    const expr &weight, std::vector<for_loop> &loops, const int K_num_block, \
    const int C_num_block, const int os, const int kpack, \
    const bool use_os_blocking, const bool pack_rows, const expr &os_acc_size, \
    const std::vector<char> &os_mask

void gen_conv_fwd_t::compute_1x1_no_pack_input(CONV_ARG_LIST) const {
  assert(loops.size() == 5 && "expected to have 5 level loops!");
  for_loop &ln = loops.at(0), &lg = loops.at(1), &lk = loops.at(2),
           &ld = loops.at(3), &lp = loops.at(4);
  auto input_expr_dims = input.checked_as<tensor>()->dims_;
  auto mb_expr_ = input_expr_dims[0];
  auto toutput = out_tensors_[0];
  auto out_fmt = toutput.get_format();
  auto oh_expr_ = oh_;
  if (!out_fmt.is_any()) {
    auto out_p2b_map = out_fmt.format_code_.collect_p2b_mapping();
    oh_expr_ = static_cast<int>(get_expr_as_int(
      output.checked_as<tensor>()->dims_[out_p2b_map[is_3d_ ? 3 : 2][0]]));
  }

  _named_for_(ln, n, 0, mb_expr_, 1, for_type::PARALLEL) {
    _named_for_(lg, g, 0, groups_) {
      _named_for_(lk, k, 0, K_num_block) {
        _named_for_(lp, p_o, 0, oh_expr_ / config.tile_p) {
          _named_for_(ld, d_o, 0, od_ / config.tile_d) {
            _for_(q_o, 0, ow_ / config.tile_q) {
              _for_(d_i, 0, config.tile_d) {
                _for_(p_i, 0, config.tile_p) {
                  auto LDA = blocking_input_ ? sw_ * config.C_block
                                             : sw_ * ic_ * groups_;
                  auto LDC = blocking_output_ ? config.K_block : oc_ * groups_;
                  auto stride_a = blocking_input_
                    ? (is_3d_ ? id_ * ih_ * iw_ * config.C_block
                              : ih_ * iw_ * config.C_block)
                    : config.C_block;
                  if (is_3d_) {
                    _tensor_(A_list, datatypes::pointer, {C_num_block});
                    _tensor_(B_list, datatypes::pointer, {C_num_block});
                    _for_(c, 0, C_num_block) {
                      std::vector<expr> input_pos = blocking_input_
                        ? std::vector<expr> {n, g * C_num_block + c,
                          (d_o * config.tile_d + d_i) * sd_,
                          (p_o * config.tile_p + p_i) * sh_,
                          q_o * config.tile_q * sw_, 0}
                        : std::vector<expr> {n,
                          (d_o * config.tile_d + d_i) * sd_,
                          (p_o * config.tile_p + p_i) * sh_,
                          q_o * config.tile_q * sw_,
                          (g * C_num_block + c) * config.C_block};
                      A_list[c] = tensor_ptr(input, input_pos);
                      B_list[c] = tensor_ptr(weight,
                        kpack > 1 ? std::vector<expr> {g * K_num_block + k, c,
                          0, 0, 0, 0, 0, 0}
                                  : std::vector<expr> {
                                    g * K_num_block + k, c, 0, 0, 0, 0, 0});
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
                      ? std::vector<expr> {n, g * K_num_block + k,
                        d_o * config.tile_d + d_i, p_o * config.tile_p + p_i,
                        q_o * config.tile_q, 0}
                      : std::vector<expr> {n, d_o * config.tile_d + d_i,
                        p_o * config.tile_p + p_i, q_o * config.tile_q,
                        (g * K_num_block + k) * config.K_block};
                    builtin::brgemm_init_list_update(A_list, B_list,
                      tensor_ptr(output, output_pos), 1, config.tile_q,
                      config.K_block, config.C_block, LDA, config.K_block, LDC,
                      1 /*useless*/, 1 /*useless*/, C_num_block,
                      get_input_dtype(), get_weight_dtype(), brg_attrs);
                  } else {
                    _tensor_(A_list, datatypes::pointer, {C_num_block});
                    _tensor_(B_list, datatypes::pointer, {C_num_block});
                    _for_(c, 0, C_num_block) {
                      std::vector<expr> input_pos = blocking_input_
                        ? std::vector<expr> {n, g * C_num_block + c,
                          (p_o * config.tile_p + p_i) * sh_,
                          q_o * config.tile_q * sw_, 0}
                        : std::vector<expr> {n,
                          (p_o * config.tile_p + p_i) * sh_,
                          q_o * config.tile_q * sw_,
                          (g * C_num_block + c) * config.C_block};
                      A_list[c] = tensor_ptr(input, input_pos);
                      B_list[c] = tensor_ptr(weight,
                        kpack > 1 ? std::vector<expr> {g * K_num_block + k, c,
                          0, 0, 0, 0, 0}
                                  : std::vector<expr> {
                                    g * K_num_block + k, c, 0, 0, 0, 0});
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
                      ? std::vector<expr> {n, g * K_num_block + k,
                        p_o * config.tile_p + p_i, q_o * config.tile_q, 0}
                      : std::vector<expr> {n, p_o * config.tile_p + p_i,
                        q_o * config.tile_q,
                        (g * K_num_block + k) * config.K_block};
                    builtin::brgemm_init_list_update(A_list, B_list,
                      tensor_ptr(output, output_pos), 1, config.tile_q,
                      config.K_block, config.C_block, LDA, config.K_block, LDC,
                      1 /*useless*/, 1 /*useless*/, C_num_block,
                      get_input_dtype(), get_weight_dtype(), brg_attrs);
                  }
                  create_anchor(fusion, output, n, 1, g * K_num_block + k, 1,
                    d_o * config.tile_d + d_i, 1, p_o * config.tile_p + p_i, 1,
                    q_o * config.tile_q, config.tile_q, config.K_block,
                    config.K_block, blocking_output_, is_3d_);
                }
                create_anchor(fusion, output, n, 1, g * K_num_block + k, 1,
                  d_o * config.tile_d + d_i, 1, p_o * config.tile_p,
                  config.tile_p, q_o * config.tile_q, config.tile_q,
                  config.K_block, config.K_block, blocking_output_, is_3d_);
              }
              if (is_3d_) {
                create_anchor(fusion, output, n, 1, g * K_num_block + k, 1,
                  d_o * config.tile_d, config.tile_d, p_o * config.tile_p,
                  config.tile_p, q_o * config.tile_q, config.tile_q,
                  config.K_block, config.K_block, blocking_output_, true);
              }
            }
            create_anchor(fusion, output, n, 1, g * K_num_block + k, 1,
              d_o * config.tile_d, config.tile_d, p_o * config.tile_p,
              config.tile_p, 0, ow_, config.K_block, config.K_block,
              blocking_output_, is_3d_);
          }
          if (is_3d_) {
            create_anchor(fusion, output, n, 1, g * K_num_block + k, 1, 0, od_,
              p_o * config.tile_p, config.tile_p, 0, ow_, config.K_block,
              config.K_block, blocking_output_, true);
          }
        }
        create_anchor(fusion, output, n, 1, g * K_num_block + k, 1, 0, od_, 0,
          oh_expr_, 0, ow_, config.K_block, config.K_block, blocking_output_,
          is_3d_);
      }
      create_anchor(fusion, output, n, 1, g * K_num_block, K_num_block, 0, od_,
        0, oh_expr_, 0, ow_, config.K_block, oc_, blocking_output_, is_3d_);
    }
    create_anchor(fusion, output, n, 1, 0, groups_ * K_num_block, 0, od_, 0,
      oh_expr_, 0, ow_, config.K_block, oc_, blocking_output_, is_3d_);
  }
}

void gen_conv_fwd_t::compute_1x1_pack_input(CONV_ARG_LIST) const {
  COMPILE_ASSERT(!is_3d_, "1x1 pack input doens't support 3D conv yet!");
  assert(loops.size() == 5 && "expected to have 5 level loops!");
  for_loop &ln = loops.at(0), &lg = loops.at(1), &lk = loops.at(2),
           &ld = loops.at(3), &lp = loops.at(4);
  tensor input1;
  auto input_expr_dims = input.checked_as<tensor>()->dims_;
  auto mb_expr_ = input_expr_dims[0];
  auto toutput = out_tensors_[0];
  auto out_fmt = toutput.get_format();
  auto oh_expr_ = oh_;
  if (!out_fmt.is_any()) {
    auto out_p2b_map = out_fmt.format_code_.collect_p2b_mapping();
    oh_expr_ = static_cast<int>(get_expr_as_int(
      output.checked_as<tensor>()->dims_[out_p2b_map[is_3d_ ? 3 : 2][0]]));
  }
  int lanes = get_lanes(ctx, config.C_block, get_input_dtype());
  if (config.pack_input == 1 && (sd_ > 1 || sh_ > 1 || sw_ > 1)) {
    trace_guard_t trg(ctx, "pack_input");
    if (blocking_input_) {
      _tensor_(input_tmp, get_input_dtype(),
        {mb_expr_, groups_ * C_num_block, oh_expr_, ow_, config.C_block});
      _named_for_(ln, n, 0, mb_expr_, 1, for_type::PARALLEL) {
        _named_for_(lg, g, 0, groups_) {
          _named_for_(lk, c_o, 0, C_num_block) {
            _named_for_(lp, p, 0, oh_expr_) {
              _for_(q, 0, ow_) {
                _for_(c_i, 0, config.C_block, (int)lanes) {
                  input_tmp[span_t(
                    {n, g * C_num_block + c_o, p, q, c_i}, lanes)]
                    = input[span_t(
                      {n, g * C_num_block + c_o, p * sh_, q * sw_, c_i},
                      lanes)];
                }
              }
            }
          }
        }
      }
      auto lnk = ln->fuse(lg);
      lnk = lnk->fuse(lk);
      if (C_num_block * mb_ < runtime_config_t::get().get_num_threads() * 2) {
        auto lnkp = lnk->fuse(lp);
      }
      input1 = input_tmp.static_as<tensor>();
    } else {
      _tensor_(input_tmp, get_input_dtype(), {mb_expr_, oh_expr_, ow_, ic_});
      _named_for_(ln, n, 0, mb_expr_, 1, for_type::PARALLEL) {
        _named_for_(lp, p, 0, oh_expr_) {
          _for_(q, 0, ow_) {
            _named_for_(lg, g, 0, groups_) {
              _for_(c_i, 0, ic_, (int)lanes) {
                input_tmp[span_t({n, p, q, g * C_num_block + c_i}, lanes)]
                  = input[span_t(
                    {n, p * sh_, q * sw_, g * C_num_block + c_i}, lanes)];
              }
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
    _named_for_(lg, g, 0, groups_) {
      _named_for_(lk, k, 0, K_num_block) {
        _named_for_(lp, p_o, 0, oh_expr_ / config.tile_p) {
          auto LDA = blocking_input_ ? config.C_block : ic_ * groups_;
          auto LDC = blocking_output_ ? config.K_block : oc_ * groups_;
          _tensor_(A_list, datatypes::pointer, {C_num_block});
          _tensor_(B_list, datatypes::pointer, {C_num_block});
          /* fill the address list for A/B */
          _for_(c, 0, C_num_block) {
            std::vector<expr> input_pos = blocking_input_
              ? std::vector<expr> {n, g * C_num_block + c, p_o * config.tile_p,
                0, 0}
              : std::vector<expr> {n, p_o * config.tile_p, 0,
                (g * C_num_block + c) * config.C_block};
            A_list[c] = tensor_ptr(input1, input_pos);
            B_list[c] = tensor_ptr(weight,
              kpack > 1
                ? std::vector<expr> {g * K_num_block + k, c, 0, 0, 0, 0, 0}
                : std::vector<expr> {g * K_num_block + k, c, 0, 0, 0, 0});
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
            ? std::vector<expr> {n, g * K_num_block + k, p_o * config.tile_p, 0,
              0}
            : std::vector<expr> {n, p_o * config.tile_p, 0,
              (g * K_num_block + k) * config.K_block};
          builtin::brgemm_init_list_update(A_list, B_list,
            tensor_ptr(output, output_pos), 1, config.tile_p * ow_,
            config.K_block, config.C_block, LDA, config.K_block, LDC,
            1 /*useless*/, 1 /*useless*/, C_num_block, get_input_dtype(),
            get_weight_dtype(), brg_attrs);
          if (fusion) {
            fusion->create_output_fusion_anchor({blocking_output_
                ? tensor_slice(output,
                  {{n, 1}, {g * K_num_block + k, 1},
                    {p_o * config.tile_p, config.tile_p}, {0, ow_},
                    {0, config.K_block}})
                : tensor_slice(output,
                  {{n, 1}, {p_o * config.tile_p, config.tile_p}, {0, ow_},
                    {(g * K_num_block + k) * config.K_block,
                      config.K_block}})});
          }
        }
        if (fusion) {
          fusion->create_output_fusion_anchor({blocking_output_
              ? tensor_slice(output,
                {{n, 1}, {g * K_num_block + k, 1}, {0, oh_expr_}, {0, ow_},
                  {0, config.K_block}})
              : tensor_slice(output,
                {{n, 1}, {0, oh_expr_}, {0, ow_},
                  {(g * K_num_block + k) * config.K_block, config.K_block}})});
        }
      }
      if (fusion) {
        fusion->create_output_fusion_anchor({tensor_slice(output,
          blocking_output_
            ? slice_range {{n, 1}, {g * K_num_block, K_num_block}, {0, oh_},
              {0, ow_}, {0, config.K_block}}
            : slice_range {
              {n, 1}, {0, oh_}, {0, ow_}, {g * K_num_block, oc_}})});
      }
    }
    if (fusion) {
      fusion->create_output_fusion_anchor({tensor_slice(output,
        blocking_output_
          ? slice_range {{n, 1}, {0, groups_ * K_num_block}, {0, oh_}, {0, ow_},
            {0, config.K_block}}
          : slice_range {{n, 1}, {0, oh_}, {0, ow_}, {0, groups_ * oc_}})});
    }
  }
}

void gen_conv_fwd_t::compute_conv3d_no_padding(CONV_ARG_LIST) const {
  COMPILE_ASSERT((pd_b_ == 0 && ph_b_ == 0 && pw_b_ == 0 && pd_e_ == 0
                   && ph_e_ == 0 && pw_e_ == 0),
    "unexpected padding in no_padding kernels!");
  assert(loops.size() == 5 && "expected to have 5 level loops!");
  for_loop &ln = loops.at(0), &lg = loops.at(1), &lk = loops.at(2),
           &ld = loops.at(3), &lp = loops.at(4);

  auto LDA = blocking_input_ ? sw_ * config.C_block : sw_ * ic_ * groups_;
  auto LDC = blocking_output_ ? config.K_block : oc_ * groups_;
  auto input_expr_dims = input.checked_as<tensor>()->dims_;
  auto mb_expr_ = input_expr_dims[0];
  _named_for_(ln, n, 0, mb_expr_, 1, for_type::PARALLEL) {
    _named_for_(lg, g, 0, groups_, 1) {
      _named_for_(lk, k_o, 0, K_num_block) {
        _named_for_(lp, p_o, 0, oh_ / config.tile_p) {
          _named_for_(ld, d_o, 0, od_ / config.tile_d) {
            _tensor_(
              A_list, datatypes::pointer, {kd_ * kh_ * kw_ * C_num_block});
            _tensor_(
              B_list, datatypes::pointer, {kd_ * kh_ * kw_ * C_num_block});
            _for_(q_o, 0, ow_ / config.tile_q) {
              _for_(d_i, 0, config.tile_d) {
                _for_(p_i, 0, config.tile_p) {
                  std::vector<expr> output_pos = blocking_output_
                    ? std::vector<expr> {n, g * K_num_block + k_o,
                      d_o * config.tile_d + d_i, p_o * config.tile_p + p_i,
                      q_o * config.tile_q, 0}
                    : std::vector<expr> {n, d_o * config.tile_d + d_i,
                      p_o * config.tile_p + p_i, q_o * config.tile_q,
                      (g * K_num_block + k_o) * config.K_block};

                  _for_(c_o, 0, C_num_block) {
                    _for_(d, 0, kd_) {
                      _for_(r, 0, kh_) {
                        _for_(s, 0, kw_) {
                          std::vector<expr> input_pos = blocking_input_
                            ? std::vector<expr> {n, g * C_num_block + c_o,
                              (d_o * config.tile_d + d_i) * sd_ + dd_ * d,
                              (p_o * config.tile_p + p_i) * sh_ + dh_ * r,
                              q_o * config.tile_q * sw_ + dw_ * s, 0}
                            : std::vector<expr> {n,
                              (d_o * config.tile_d + d_i) * sd_ + dd_ * d,
                              (p_o * config.tile_p + p_i) * sh_ + dh_ * r,
                              q_o * config.tile_q * sw_ + dw_ * s,
                              (g * C_num_block + c_o) * config.C_block};
                          auto idx = c_o * kd_ * kh_ * kw_ + d * kh_ * kw_
                            + r * kw_ + s;
                          A_list[idx] = tensor_ptr(input, input_pos);
                          B_list[idx] = tensor_ptr(weight,
                            kpack > 1
                              ? std::vector<expr> {g * K_num_block + k_o, c_o,
                                d, r, s, 0, 0, 0}
                              : std::vector<expr> {
                                g * K_num_block + k_o, c_o, d, r, s, 0, 0});
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

                  builtin::brgemm_init_list_update(A_list, B_list,
                    tensor_ptr(output, output_pos), 1, config.tile_q,
                    config.K_block, config.C_block, LDA, config.K_block, LDC,
                    1 /*useless*/, 1 /*useless*/, kd_ * kh_ * kw_ * C_num_block,
                    get_input_dtype(), get_weight_dtype(), brg_attrs);

                  if (fusion) {
                    fusion->create_output_fusion_anchor({tensor_slice(output,
                      blocking_output_
                        ? slice_range {{n, 1}, {g * K_num_block + k_o, 1},
                          {d_o * config.tile_d + d_i, 1},
                          {p_o * config.tile_p + p_i, 1},
                          {q_o * config.tile_q, config.tile_q},
                          {0, config.K_block}}
                        : slice_range {{n, 1}, {d_o * config.tile_d + d_i, 1},
                          {p_o * config.tile_p + p_i, 1},
                          {q_o * config.tile_q, config.tile_q},
                          {(g * K_num_block + k_o) * config.K_block,
                            config.K_block}})});
                  }
                }
                if (fusion) {
                  fusion->create_output_fusion_anchor({tensor_slice(output,
                    blocking_output_
                      ? slice_range {{n, 1}, {g * K_num_block + k_o, 1},
                        {d_o * config.tile_d + d_i, 1},
                        {p_o * config.tile_p, config.tile_p},
                        {q_o * config.tile_q, config.tile_q},
                        {0, config.K_block}}
                      : slice_range {{n, 1}, {d_o * config.tile_d + d_i, 1},
                        {p_o * config.tile_p, config.tile_p},
                        {q_o * config.tile_q, config.tile_q},
                        {(g * K_num_block + k_o) * config.K_block,
                          config.K_block}})});
                }
              }
              if (fusion) {
                fusion->create_output_fusion_anchor({tensor_slice(output,
                  blocking_output_
                    ? slice_range {{n, 1}, {g * K_num_block + k_o, 1},
                      {d_o * config.tile_d, config.tile_d},
                      {p_o * config.tile_p, config.tile_p},
                      {q_o * config.tile_q, config.tile_q}, {0, config.K_block}}
                    : slice_range {{n, 1}, {d_o * config.tile_d, config.tile_d},
                      {p_o * config.tile_p, config.tile_p},
                      {q_o * config.tile_q, config.tile_q},
                      {(g * K_num_block + k_o) * config.K_block,
                        config.K_block}})});
              }
            }
            if (fusion) {
              fusion->create_output_fusion_anchor({tensor_slice(output,
                blocking_output_
                  ? slice_range {{n, 1}, {g * K_num_block + k_o, 1},
                    {d_o * config.tile_d, config.tile_d},
                    {p_o * config.tile_p, config.tile_p}, {0, ow_},
                    {0, config.K_block}}
                  : slice_range {{n, 1}, {d_o * config.tile_d, config.tile_d},
                    {p_o * config.tile_p, config.tile_p}, {0, ow_},
                    {(g * K_num_block + k_o) * config.K_block,
                      config.K_block}})});
            }
          }
          if (fusion) {
            fusion->create_output_fusion_anchor({tensor_slice(output,
              blocking_output_
                ? slice_range {{n, 1}, {g * K_num_block + k_o, 1}, {0, od_},
                  {p_o * config.tile_p, config.tile_p}, {0, ow_},
                  {0, config.K_block}}
                : slice_range {{n, 1}, {0, od_},
                  {p_o * config.tile_p, config.tile_p}, {0, ow_},
                  {(g * K_num_block + k_o) * config.K_block,
                    config.K_block}})});
          }
        }
        if (fusion) {
          fusion->create_output_fusion_anchor({tensor_slice(output,
            blocking_output_
              ? slice_range {{n, 1}, {g * K_num_block + k_o, 1}, {0, od_},
                {0, oh_}, {0, ow_}, {0, config.K_block}}
              : slice_range {{n, 1}, {0, od_}, {0, oh_}, {0, ow_},
                {(g * K_num_block + k_o) * config.K_block, config.K_block}})});
        }
      }
      if (fusion) {
        fusion->create_output_fusion_anchor({tensor_slice(output,
          blocking_output_
            ? slice_range {{n, 1}, {g * K_num_block, K_num_block}, {0, od_},
              {0, oh_}, {0, ow_}, {0, config.K_block}}
            : slice_range {
              {n, 1}, {0, od_}, {0, oh_}, {0, ow_}, {g * K_num_block, oc_}})});
      }
    }
    if (fusion) {
      fusion->create_output_fusion_anchor({tensor_slice(output,
        blocking_output_ ? slice_range {{n, 1}, {0, groups_ * K_num_block},
          {0, od_}, {0, oh_}, {0, ow_}, {0, config.K_block}}
                         : slice_range {{n, 1}, {0, od_}, {0, oh_}, {0, ow_},
                           {0, groups_ * oc_}})});
    }
  }
}

void gen_conv_fwd_t::compute_conv_no_padding(CONV_ARG_LIST) const {
  COMPILE_ASSERT((pd_b_ == 0 && ph_b_ == 0 && pw_b_ == 0 && pd_e_ == 0
                   && ph_e_ == 0 && pw_e_ == 0),
    "unexpected padding in no_padding kernels!");
  assert(loops.size() == 5 && "expected to have 5 level loops!");
  loops.emplace_back(for_loop());
  for_loop &ln = loops.at(0), &lg = loops.at(1), &lk = loops.at(2),
           &ld = loops.at(3), &lp = loops.at(4), &lok = loops.at(5);
  auto use_rl = attrs_.get_or_else("use_rl", ops::rl_kind::NO_LOWERING);

  auto LDA = blocking_input_ ? sw_ * config.C_block : sw_ * ic_ * groups_;
  auto LDB = config.K_block;
  auto LDC = blocking_output_ ? config.K_block : oc_ * groups_;
  auto input_expr_dims = input.checked_as<tensor>()->dims_;
  auto mb_expr_ = input_expr_dims[0];
  auto data_size
    = math_utils::get_dims_product(in_tensors_[0].get_blocking_dims())
    * utils::get_sizeof_type(get_input_dtype());
  auto weight_size
    = math_utils::get_dims_product(in_tensors_[1].get_blocking_dims())
    * utils::get_sizeof_type(get_weight_dtype());
  auto L2_cache_size = ctx->machine_.cpu_flags_.getDCacheSize(2);
  int oc_split = get_oc_split_factor(
    mb_, data_size, weight_size, L2_cache_size, K_num_block);

  _named_for_(lok, outer_k, 0, oc_split, 1, for_type::PARALLEL) {
    _named_for_(ln, n, 0, mb_expr_, 1) {
      _named_for_(lg, g, 0, groups_, 1) {
        if (use_os_blocking) {
          _named_for_(lk, k_i, 0, K_num_block / oc_split) {
            expr k_o = outer_k * K_num_block / oc_split + k_i;
            _named_for_(lp, o_o, 0, os / config.tile_os) {
              _tensor_(A_list, datatypes::pointer, {kh_ * kw_ * C_num_block});
              _tensor_(B_list, datatypes::pointer, {kh_ * kw_ * C_num_block});
              auto out_tsr = tensor_ptr(output,
                blocking_output_
                  ? std::vector<expr> {n, g * K_num_block + k_o,
                    o_o * config.tile_os / ow_, o_o * config.tile_os % ow_, 0}
                  : std::vector<expr> {n, o_o * config.tile_os / ow_,
                    o_o * config.tile_os % ow_,
                    (g * K_num_block + k_o) * config.K_block});
              int adj_ow = ow_ + (pack_rows ? num_elems_skip_per_ow_ : 0);

              if (pack_rows) {
                if (os / config.tile_os == 1) {
                  out_tsr = tensor_ptr(output,
                    blocking_output_
                      ? std::vector<expr> {n, g * K_num_block + k_o, 0, 0, 0}
                      : std::vector<expr> {
                        n, 0, 0, (g * K_num_block + k_o) * config.K_block});
                } else {
                  auto acc_m = os_acc_size[{o_o}];
                  out_tsr = tensor_ptr(output,
                    blocking_output_
                      ? std::vector<expr> {n, g * K_num_block + k_o,
                        acc_m / ow_, acc_m % ow_, 0}
                      : std::vector<expr> {n, acc_m / ow_, acc_m % ow_,
                        (g * K_num_block + k_o) * config.K_block});
                }
              }

              _for_(c_o, 0, C_num_block) {
                _for_(r, 0, kh_) {
                  _for_(s, 0, kw_) {
                    auto idx = c_o * kh_ * kw_ + r * kw_ + s;
                    std::vector<expr> input_pos = blocking_input_
                      ? std::vector<expr> {n, g * C_num_block + c_o,
                        ((o_o * config.tile_os) / adj_ow) * sh_ + dh_ * r,
                        ((o_o * config.tile_os) % adj_ow) * sw_ + dw_ * s, 0}
                      : std::vector<expr> {n,
                        ((o_o * config.tile_os) / adj_ow) * sh_ + dh_ * r,
                        ((o_o * config.tile_os) % adj_ow) * sw_ + dw_ * s,
                        (g * C_num_block + c_o) * config.C_block};
                    A_list[idx] = tensor_ptr(input, input_pos);
                    B_list[idx] = tensor_ptr(weight,
                      kpack > 1 ? std::vector<expr> {g * K_num_block + k_o, c_o,
                        r, s, 0, 0, 0}
                                : std::vector<expr> {
                                  g * K_num_block + k_o, c_o, r, s, 0, 0});
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

              builtin::brgemm_init_list_update(A_list, B_list, out_tsr, 1,
                config.tile_os, config.K_block, config.C_block, LDA,
                config.K_block, LDC, 1 /*useless*/, 1 /*useless*/,
                kh_ * kw_ * C_num_block, get_input_dtype(), get_weight_dtype(),
                brg_attrs, os_mask, o_o, os / config.tile_os);
              auto os_num_block = os / config.tile_os;
              if (fusion && !pack_rows) {
                fusion->create_output_fusion_anchor({tensor_slice(output,
                  blocking_output_
                    ? slice_range {{n, 1}, {g * K_num_block + k_o, 1},
                      {o_o * config.tile_os / ow_, 1},
                      {o_o * config.tile_os % ow_, config.tile_os},
                      {0, config.K_block}}
                    : slice_range {{n, 1}, {o_o * config.tile_os / ow_, 1},
                      {o_o * config.tile_os % ow_, config.tile_os},
                      {(g * K_num_block + k_o) * config.K_block,
                        config.K_block}})});
              } else if (fusion && oh_ % os_num_block == 0) {
                fusion->create_output_fusion_anchor({tensor_slice(output,
                  blocking_output_
                    ? slice_range {{n, 1}, {g * K_num_block + k_o, 1},
                      {o_o * (oh_ / os_num_block), (oh_ / os_num_block)},
                      {0, ow_}, {0, config.K_block}}
                    : slice_range {{n, 1},
                      {o_o * (oh_ / os_num_block), (oh_ / os_num_block)},
                      {0, ow_},
                      {(g * K_num_block + k_o) * config.K_block,
                        config.K_block}})});
              }
            }
            if (fusion) {
              // Note: slice tensor might across multi-rows with
              // non-rectangular shapes. Currently, we just promote the fusion
              // anchor to higher level of loop, which will consume larger
              // buffer and is non-optimal This can be optimized in next
              // version of fusion manager.
              fusion->create_output_fusion_anchor({tensor_slice(output,
                blocking_output_
                  ? slice_range {{n, 1}, {g * K_num_block + k_o, 1}, {0, oh_},
                    {0, ow_}, {0, config.K_block}}
                  : slice_range {{n, 1}, {0, oh_}, {0, ow_},
                    {(g * K_num_block + k_o) * config.K_block,
                      config.K_block}})});
            }
          }
        } else {
          _named_for_(lp, p_o, 0, oh_ / config.tile_p) {
            _named_for_(lk, k_i, 0, K_num_block / oc_split) {
              expr k_o = outer_k * K_num_block / oc_split + k_i;
              int list_size = kh_ * kw_ * C_num_block;
              int num_kw = kw_, kw_step = 1;
              int brgemm_k = 1, num_brgemm_k = 1;
              int extra_padding = 0;
              if (use_rl == ops::rl_kind::KW_LOWERING) {
                brgemm_k = attrs_.get_or_else("brgemm_k", 1);
                num_brgemm_k = attrs_.get_or_else("num_brgemm_k", 1);
                extra_padding = attrs_.get_or_else("extra_padding", 0);
                list_size = num_brgemm_k;
                num_kw = num_brgemm_k / kh_;
                kw_step = brgemm_k / ic_;
              }

              _tensor_(A_list, datatypes::pointer, {list_size});
              _tensor_(B_list, datatypes::pointer, {list_size});
              _for_(q_o, 0, ow_ / config.tile_q) {
                _for_(p_i, 0, config.tile_p) {
                  std::vector<expr> output_pos = blocking_output_
                    ? std::vector<expr> {n, g * K_num_block + k_o,
                      p_o * config.tile_p + p_i, q_o * config.tile_q, 0}
                    : std::vector<expr> {n, p_o * config.tile_p + p_i,
                      q_o * config.tile_q,
                      (g * K_num_block + k_o) * config.K_block};

                  auto fill_A_and_B_list = [&]() {
                    _for_(c_o, 0, C_num_block) {
                      _for_(r, 0, kh_) {
                        _for_(s, 0, num_kw) {
                          auto idx = c_o * kh_ * num_kw + r * num_kw + s;
                          std::vector<expr> input_pos = blocking_input_
                            ? std::vector<expr> {n, g * C_num_block + c_o,
                              (p_o * config.tile_p + p_i) * sh_ + dh_ * r,
                              q_o * config.tile_q * sw_ + dw_ * s * kw_step, 0}
                            : std::vector<expr> {n,
                              (p_o * config.tile_p + p_i) * sh_ + dw_ * r,
                              q_o * config.tile_q * sw_ + dw_ * s * kw_step,
                              (g * C_num_block + c_o) * config.C_block};

                          A_list[idx] = tensor_ptr(input, input_pos);
                          B_list[idx] = tensor_ptr(weight,
                            kpack > 1
                              ? std::vector<expr> {g * K_num_block + k_o, c_o,
                                r, s * kw_step, 0, 0, 0}
                              : std::vector<expr> {g * K_num_block + k_o, c_o,
                                r, s * kw_step, 0, 0});
                        }
                      }
                    }
                  };

                  if (use_rl == ops::rl_kind::KW_LOWERING
                    && extra_padding > 0) {
                    // need extra padding for A at the last brgemm
                    _if_((p_o == oh_ / config.tile_p - 1)
                      && (p_i == config.tile_p - 1)
                      && (q_o == ow_ / config.tile_q - 1)) {
                      auto dtype_input = get_input_dtype();
                      int max_col_in_bytes = 32;
                      int ic_in_bytes
                        = ic_ * utils::get_sizeof_type(dtype_input);
                      COMPILE_ASSERT(ic_in_bytes <= max_col_in_bytes,
                        "Expect ic*dtype <=32 for kw lowering, but got "
                          << ic_in_bytes << ".");
                      COMPILE_ASSERT(extra_padding % ic_ == 0,
                        "Expect extra_padding is dividable by ic, but got "
                        "extra_padding="
                          << extra_padding << ",ic=" << ic_ << ".");
                      auto src_row_tile_orig_size
                        = (config.tile_q - 1) * sw_ + dw_ * (kw_ - 1) + 1;
                      auto src_row_tile_size
                        = src_row_tile_orig_size + extra_padding / ic_;
                      _tensor_(
                        tmp_input, dtype_input, {kh_, src_row_tile_size, ic_});

                      auto lanes = get_minimal_lanes(ic_in_bytes);
                      auto mask = convert_int_to_mask(ic_in_bytes);
                      auto mask_expr = mask != 0
                        ? builder::make_cast(get_dtype(lanes), mask)
                        : expr();

                      _for_(ih, 0, kh_) {
                        _for_(iw, 0, src_row_tile_orig_size) {
                          tmp_input[span_t({ih, iw, 0}, lanes, mask_expr)]
                            = input[blocking_input_
                                ? span_t({n, g, (oh_ - 1) * sh_ + dh_ * ih,
                                           (ow_ - config.tile_q) * sw_ + iw, 0},
                                  lanes, mask_expr)
                                : span_t({n, (oh_ - 1) * sh_ + dh_ * ih,
                                           (ow_ - config.tile_q) * sw_ + iw,
                                           g * config.C_block},
                                  lanes, mask_expr)];
                        }
                        builtin::brgemm_init(tensor_ptr(tmp_input,
                                               {ih, src_row_tile_orig_size, 0}),
                          extra_padding / ic_, ic_, ic_, dtype_input, 0);
                      }

                      _for_(r, 0, kh_) {
                        _for_(s, 0, num_kw) {
                          auto idx = r * num_kw + s;
                          A_list[idx]
                            = tensor_ptr(tmp_input, {r, s * kw_step, 0});
                          B_list[idx] = tensor_ptr(weight,
                            kpack > 1
                              ? std::vector<expr> {g * K_num_block + k_o, 0, r,
                                s * kw_step, 0, 0, 0}
                              : std::vector<expr> {g * K_num_block + k_o, 0, r,
                                s * kw_step, 0, 0});
                        }
                      }
                    }
                    _else_ { fill_A_and_B_list(); }
                  } else {
                    fill_A_and_B_list();
                  }

                  int M = config.tile_q, K = config.C_block, N = config.K_block;
                  if (use_rl == ops::rl_kind::KW_LOWERING) { K = brgemm_k; }
                  const auto hint_A_size = M * K * list_size;
                  const auto hint_B_size = K * N * list_size;
                  const auto hint_C_size = M * N;
                  sc_brgemm_attrs_t brg_attrs {
                    {brgemm::attr_key::max_bs, list_size},
                    {brgemm::attr_key::hint_expected_A_size, hint_A_size},
                    {brgemm::attr_key::hint_expected_B_size, hint_B_size},
                    {brgemm::attr_key::hint_expected_C_size, hint_C_size},
                    {brgemm::attr_key::use_interleave_stores, true},
                    {brgemm::attr_key::use_uker, true},
                    {brgemm::attr_key::bd_mask_level, 0}};

                  builtin::brgemm_init_list_update(A_list, B_list,
                    tensor_ptr(output, output_pos), 1, M, N, K, LDA, LDB, LDC,
                    1 /*useless*/, 1 /*useless*/, list_size, get_input_dtype(),
                    get_weight_dtype(), brg_attrs);

                  if (fusion) {
                    fusion->create_output_fusion_anchor({tensor_slice(output,
                      blocking_output_
                        ? slice_range {{n, 1}, {g * K_num_block + k_o, 1},
                          {p_o * config.tile_p + p_i, 1},
                          {q_o * config.tile_q, config.tile_q},
                          {0, config.K_block}}
                        : slice_range {{n, 1}, {p_o * config.tile_p + p_i, 1},
                          {q_o * config.tile_q, config.tile_q},
                          {(g * K_num_block + k_o) * config.K_block,
                            config.K_block}})});
                  }
                }
                if (fusion) {
                  fusion->create_output_fusion_anchor({tensor_slice(output,
                    blocking_output_
                      ? slice_range {{n, 1}, {g * K_num_block + k_o, 1},
                        {p_o * config.tile_p, config.tile_p},
                        {q_o * config.tile_q, config.tile_q},
                        {0, config.K_block}}
                      : slice_range {{n, 1},
                        {p_o * config.tile_p, config.tile_p},
                        {q_o * config.tile_q, config.tile_q},
                        {(g * K_num_block + k_o) * config.K_block,
                          config.K_block}})});
                }
              }
              if (fusion) {
                fusion->create_output_fusion_anchor({tensor_slice(output,
                  blocking_output_
                    ? slice_range {{n, 1}, {g * K_num_block + k_o, 1},
                      {p_o * config.tile_p, config.tile_p}, {0, ow_},
                      {0, config.K_block}}
                    : slice_range {{n, 1}, {p_o * config.tile_p, config.tile_p},
                      {0, ow_},
                      {(g * K_num_block + k_o) * config.K_block,
                        config.K_block}})});
              }
            }
            if (fusion) {
              fusion->create_output_fusion_anchor({tensor_slice(output,
                blocking_output_
                  ? slice_range {{n, 1},
                    {g * K_num_block + outer_k * K_num_block / oc_split,
                      K_num_block / oc_split},
                    {p_o * config.tile_p, config.tile_p}, {0, ow_},
                    {0, config.K_block}}
                  : slice_range {{n, 1}, {p_o * config.tile_p, config.tile_p},
                    {0, ow_},
                    {(g * K_num_block + outer_k * K_num_block / oc_split)
                        * config.K_block,
                      K_num_block / oc_split * config.K_block}})});
            }
          }
        }
      }
      if (fusion) {
        if (groups_ == 1) {
          fusion->create_output_fusion_anchor({tensor_slice(output,
            blocking_output_
              ? slice_range {{n, 1},
                {outer_k * K_num_block / oc_split, K_num_block / oc_split},
                {0, oh_}, {0, ow_}, {0, config.K_block}}
              : slice_range {{n, 1}, {0, oh_}, {0, ow_},
                {outer_k * K_num_block / oc_split * config.K_block,
                  K_num_block / oc_split * config.K_block}})});
        } else if (groups_ > 1 && oc_split == 1) {
          fusion->create_output_fusion_anchor({tensor_slice(output,
            blocking_output_
              ? slice_range {{n, 1}, {0, groups_ * K_num_block}, {0, oh_},
                {0, ow_}, {0, config.K_block}}
              : slice_range {{n, 1}, {0, oh_}, {0, ow_}, {0, groups_ * oc_}})});
        } else { /*noops as discontiguous slice for groups >1 && outer_k > 1*/
        }
      }
    }
  }
}

void gen_conv_fwd_t::compute_conv_padding(CONV_ARG_LIST) const {
  COMPILE_ASSERT(!is_3d_, "3D conv with padding is not supported yet!");
  assert(loops.size() == 5 && "expected to have 5 level loops!");
  for_loop &ln = loops.at(0), &lg = loops.at(1), &lk = loops.at(2),
           &ld = loops.at(3), &lp = loops.at(4);
  COMPILE_ASSERT(blocking_input_ && blocking_output_,
    "Only blocking in&out are supported so far!");

  /* to do conv 3*3 with padding */
  std::unordered_set<int> Q1;
  std::unordered_set<int> Q2;
  std::unordered_set<int> Q3;

  int H_PADDED = ih_ + ph_b_ + ph_e_, W_PADDED = iw_ + pw_b_ + pw_e_;
  sc_dims padded_input_dims = blocking_input_
    ? sc_dims {mb_, groups_ * C_num_block, H_PADDED, W_PADDED, config.C_block}
    : sc_dims {mb_, H_PADDED, W_PADDED, groups_ * ic_};

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
          blocking_input_ ? std::vector<int> {0, 0,
            (p_o * config.tile_p + p_i) * sh_, pw_b_, 0}
                          : std::vector<int> {
                            0, (p_o * config.tile_p + p_i) * sh_, pw_b_, 0});
        int x_threshold_right = tensor_offset(padded_input_dims,
          blocking_input_
            ? std::vector<int> {0, 0, (p_o * config.tile_p + p_i) * sh_,
              W_PADDED - pw_e_ - 1, 0}
            : std::vector<int> {
              0, (p_o * config.tile_p + p_i) * sh_, W_PADDED - pw_e_ - 1, 0});
        for (int s = 0; s < kw_; s++) {
          int pad_tmp // real_left_pad
            = (x_threshold_left - (x_start_offset + (dw_ * s) * config.C_block))
            / config.C_block;
          if (((x_start_offset + dw_ * s * config.C_block) < x_threshold_left)
            && ((x_start_offset + dw_ * s * config.C_block
                  + (config.tile_q - 1) * config.C_block * sw_)
              <= x_threshold_right)) {
            int interval = (pad_tmp + sw_ - 1) / sw_; // real_dst_pad
            int Q_tmp = config.tile_q - interval;
            Q1.insert(Q_tmp);
          } else {
            if (((x_start_offset + dw_ * s * config.C_block)
                  >= x_threshold_left)
              && ((x_start_offset + dw_ * s * config.C_block
                    + (config.tile_q - 1) * config.C_block * sw_)
                > x_threshold_right)) {
              int Q_tmp = ((x_threshold_right
                             - (x_start_offset + dw_ * s * config.C_block))
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
    _named_for_(lg, g, 0, groups_) {
      _named_for_(lk, k_o, 0, K_num_block) {
        _named_for_(lp, p_o, 0, oh_ / config.tile_p) {
          _tensor_(A_list, datatypes::pointer, {kh_});
          _tensor_(B_list, datatypes::pointer, {kh_});
          _for_(q_o, 0, ow_ / config.tile_q) {
            _for_(p_i, 0, config.tile_p) {
              builtin::mem_zero(
                tensor_ptr(output,
                  {n, g * K_num_block + k_o, p_o * config.tile_p + p_i,
                    q_o * config.tile_q, 0}),
                config.tile_q * config.K_block, get_output_dtype());
              _for_(c_o, 0, C_num_block) {
                _var_(x_threshold_top, datatypes::s32);
                _var_(x_threshold_bottom, datatypes::s32);
                _var_(x_threshold_left, datatypes::s32);
                _var_(x_threshold_right, datatypes::s32);
                _var_(x_start_offset, datatypes::s32);
                x_threshold_top
                  = tensor_offset(padded_input_dims, {0, 0, (expr)ph_b_, 0, 0});
                x_threshold_bottom = tensor_offset(
                  padded_input_dims, {0, 0, (expr)(H_PADDED - ph_e_), 0, 0});
                x_threshold_left = tensor_offset(padded_input_dims,
                  {0, 0, (p_o * config.tile_p + p_i) * sh_, pw_b_, 0});
                x_threshold_right = tensor_offset(padded_input_dims,
                  {0, 0, (p_o * config.tile_p + p_i) * sh_,
                    W_PADDED - pw_e_ - 1, 0});
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
                  _if_(((x_start_offset + dw_ * s * config.C_block)
                         >= x_threshold_left)
                    && ((x_start_offset + dw_ * s * config.C_block
                          + (config.tile_q - 1) * config.C_block * sw_)
                      <= x_threshold_right)) {
                    cnt = cnt + 1;
                    _if_(head == -1) head
                      = builder::make_cast(datatypes::s32, s);
                  }
                  _else_ {
                    _var_(pad_tmp, datatypes::s32);
                    _var_(interval, datatypes::s32);
                    _var_(Q_tmp, datatypes::s32);
                    _if_(((x_start_offset + dw_ * s * config.C_block)
                           < x_threshold_left)
                      && ((x_start_offset + dw_ * s * config.C_block
                            + (config.tile_q - 1) * config.C_block * sw_)
                        <= x_threshold_right)) {
                      pad_tmp
                        = (x_threshold_left
                            - (x_start_offset
                              + builder::make_cast(datatypes::s32, dw_ * s)
                                * config.C_block))
                        / config.C_block;
                      interval = (pad_tmp + sw_ - 1) / sw_;
                      Q_tmp = config.tile_q - interval;
                      _if_(Q_tmp > 0) {
                        tmp = 0;
                        _for_(r, 0, kh_) {
                          x_tmp_offset = tensor_offset(padded_input_dims,
                            {0, 0, (p_o * config.tile_p + p_i) * sh_ + dh_ * r,
                              q_o * config.tile_q * sw_ + dw_ * s
                                + interval * sw_,
                              0});
                          _if_(x_tmp_offset >= x_threshold_top
                            && x_tmp_offset < x_threshold_bottom) {
                            A_list[tmp] = tensor_ptr(input,
                              {n, g * C_num_block + c_o,
                                (p_o * config.tile_p + p_i) * sh_ + dh_ * r
                                  - ph_b_,
                                q_o * config.tile_q * sw_ - pw_b_ + dw_ * s
                                  + interval * sw_,
                                0});
                            B_list[tmp] = tensor_ptr(weight,
                              kpack > 1
                                ? std::vector<expr> {g * K_num_block + k_o, c_o,
                                  r, s, 0, 0, 0}
                                : std::vector<expr> {
                                  g * K_num_block + k_o, c_o, r, s, 0, 0});
                            tmp = tmp + 1;
                          }
                        }
                        _if_(tmp > 0) {
                          if (Q1.size() == 1) {
                            builtin::brgemm_list_update(A_list, B_list,
                              tensor_ptr(output,
                                {n, g * K_num_block + k_o,
                                  p_o * config.tile_p + p_i,
                                  q_o * config.tile_q + interval, 0}),
                              1, *Q1.begin(), config.K_block, config.C_block,
                              sw_ * config.C_block, config.K_block,
                              config.K_block, 1 /*useless*/, 1 /*useless*/, tmp,
                              get_input_dtype(), get_weight_dtype());
                          } else {
                            builtin::brgemm_list_update(A_list, B_list,
                              tensor_ptr(output,
                                {n, g * K_num_block + k_o,
                                  p_o * config.tile_p + p_i,
                                  q_o * config.tile_q + interval, 0}),
                              1, Q_tmp, config.K_block, config.C_block,
                              sw_ * config.C_block, config.K_block,
                              config.K_block, 1 /*useless*/, 1 /*useless*/, tmp,
                              get_input_dtype(), get_weight_dtype());
                          }
                        }
                      }
                    }
                    _else_ {
                      _if_(((x_start_offset + dw_ * s * config.C_block)
                             >= x_threshold_left)
                        && ((x_start_offset + dw_ * s * config.C_block
                              + (config.tile_q - 1) * config.C_block * sw_)
                          > x_threshold_right)) {
                        Q_tmp
                          = ((x_threshold_right
                               - (x_start_offset
                                 + builder::make_cast(datatypes::s32, dw_ * s)
                                   * config.C_block))
                                / config.C_block
                              + sw_)
                          / sw_;
                        _if_(Q_tmp > 0) {
                          tmp = 0;
                          _for_(r, 0, kh_) {
                            x_tmp_offset = tensor_offset(padded_input_dims,
                              {0, 0,
                                (p_o * config.tile_p + p_i) * sh_ + dh_ * r,
                                q_o * config.tile_q * sw_ + dw_ * s, 0});
                            _if_(x_tmp_offset >= x_threshold_top
                              && x_tmp_offset < x_threshold_bottom) {
                              A_list[tmp] = tensor_ptr(input,
                                {n, g * C_num_block + c_o,
                                  (p_o * config.tile_p + p_i) * sh_ + dh_ * r
                                    - ph_b_,
                                  q_o * config.tile_q * sw_ - pw_b_ + dw_ * s,
                                  0});
                              B_list[tmp] = tensor_ptr(weight,
                                kpack > 1
                                  ? std::vector<expr> {g * K_num_block + k_o,
                                    c_o, r, s, 0, 0, 0}
                                  : std::vector<expr> {
                                    g * K_num_block + k_o, c_o, r, s, 0, 0});
                              tmp = tmp + 1;
                            }
                          }
                          _if_(tmp > 0) {
                            if (Q2.size() == 1) {
                              builtin::brgemm_list_update(A_list, B_list,
                                tensor_ptr(output,
                                  {n, g * K_num_block + k_o,
                                    p_o * config.tile_p + p_i,
                                    q_o * config.tile_q, 0}),
                                1, *Q2.begin(), config.K_block, config.C_block,
                                sw_ * config.C_block, config.K_block,
                                config.K_block, config.C_block,
                                config.C_block * config.K_block, tmp,
                                get_input_dtype(), get_weight_dtype());
                            } else {
                              builtin::brgemm_list_update(A_list, B_list,
                                tensor_ptr(output,
                                  {n, g * K_num_block + k_o,
                                    p_o * config.tile_p + p_i,
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
                        pad_tmp
                          = (x_threshold_left
                              - (x_start_offset
                                + builder::make_cast(datatypes::s32, dw_ * s)
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
                              {0, 0,
                                (p_o * config.tile_p + p_i) * sh_ + dh_ * r,
                                q_o * config.tile_q * sw_ + dw_ * s
                                  + interval * sw_,
                                0});
                            _if_(x_tmp_offset >= x_threshold_top
                              && x_tmp_offset < x_threshold_bottom) {
                              A_list[tmp] = tensor_ptr(input,
                                {n, g * C_num_block + c_o,
                                  (p_o * config.tile_p + p_i) * sh_ + dh_ * r
                                    - ph_b_,
                                  q_o * config.tile_q * sw_ - pw_b_ + dw_ * s
                                    + interval * sw_,
                                  0});
                              B_list[tmp] = tensor_ptr(weight,
                                kpack > 1
                                  ? std::vector<expr> {g * K_num_block + k_o,
                                    c_o, r, s, 0, 0, 0}
                                  : std::vector<expr> {
                                    g * K_num_block + k_o, c_o, r, s, 0, 0});
                              tmp = tmp + 1;
                            }
                          }
                          _if_(tmp > 0) {
                            if (Q3.size() == 1) {
                              builtin::brgemm_list_update(A_list, B_list,
                                tensor_ptr(output,
                                  {n, g * K_num_block + k_o,
                                    p_o * config.tile_p + p_i,
                                    q_o * config.tile_q + interval, 0}),
                                1, *Q3.begin(), config.K_block, config.C_block,
                                sw_ * config.C_block, config.K_block,
                                config.K_block, config.C_block,
                                config.C_block * config.K_block, tmp,
                                get_input_dtype(), get_weight_dtype());
                            } else {
                              builtin::brgemm_list_update(A_list, B_list,
                                tensor_ptr(output,
                                  {n, g * K_num_block + k_o,
                                    p_o * config.tile_p + p_i,
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
                      {0, 0, (p_o * config.tile_p + p_i) * sh_ + dh_ * r,
                        q_o * config.tile_q * sw_ + dw_ * head, 0});
                    _if_(x_tmp_offset >= x_threshold_top
                      && x_tmp_offset < x_threshold_bottom) {
                      A_list[tmp] = tensor_ptr(input,
                        {n, g * C_num_block + c_o,
                          (p_o * config.tile_p + p_i) * sh_ + dh_ * r - ph_b_,
                          q_o * config.tile_q * sw_ - pw_b_ + dw_ * head, 0});
                      B_list[tmp] = tensor_ptr(weight,
                        kpack > 1 ? std::vector<expr> {g * K_num_block + k_o,
                          c_o, r, head, 0, 0, 0}
                                  : std::vector<expr> {
                                    g * K_num_block + k_o, c_o, r, head, 0, 0});
                      tmp = tmp + 1;
                    }
                  }
                  _if_(tmp > 0) {
                    builtin::brgemm_list_update(A_list, B_list,
                      tensor_ptr(output,
                        {n, g * K_num_block + k_o, p_o * config.tile_p + p_i,
                          q_o * config.tile_q, 0}),
                      cnt, config.tile_q, config.K_block, config.C_block,
                      sw_ * config.C_block, config.K_block, config.K_block,
                      dw_ * config.C_block, config.C_block * config.K_block,
                      tmp, get_input_dtype(), get_weight_dtype());
                  }
                }
              }
              if (fusion) {
                // tile_q * K_block
                fusion->create_output_fusion_anchor({tensor_slice(output,
                  {{n, 1}, {g * K_num_block + k_o, 1},
                    {p_o * config.tile_p + p_i, 1},
                    {q_o * config.tile_q, config.tile_q},
                    {0, config.K_block}})});
              }
            }
            if (fusion) {
              // tile_p * tile_q * K_block
              fusion->create_output_fusion_anchor({tensor_slice(output,
                {{n, 1}, {g * K_num_block + k_o, 1},
                  {p_o * config.tile_p, config.tile_p},
                  {q_o * config.tile_q, config.tile_q}, {0, config.K_block}})});
            }
          }
          if (fusion) {
            // tile_p * ow * K_block
            fusion->create_output_fusion_anchor({tensor_slice(output,
              {{n, 1}, {g * K_num_block + k_o, 1},
                {p_o * config.tile_p, config.tile_p}, {0, ow_},
                {0, config.K_block}})});
          }
        }
        if (fusion) {
          // oh * ow * K_block
          fusion->create_output_fusion_anchor({tensor_slice(output,
            {{n, 1}, {g * K_num_block + k_o, 1}, {0, oh_}, {0, ow_},
              {0, config.K_block}})});
        }
      }
    }
    if (fusion) {
      // oc * oh * ow * K_block
      fusion->create_output_fusion_anchor({tensor_slice(output,
        {{n, 1}, {0, groups_ * K_num_block}, {0, oh_}, {0, ow_},
          {0, config.K_block}})});
    }
  }
}

void gen_conv_fwd_t::compute_conv_padding_v2(CONV_ARG_LIST) const {
  assert(loops.size() == 5 && "expected to have 5 level loops!");
  loops.emplace_back(for_loop());
  for_loop &ln = loops.at(0), &lg = loops.at(1), &lk = loops.at(2),
           &ld = loops.at(3), &lp = loops.at(4), &lok = loops.at(5);

  // no need to include groups for LDA as it's used by sub-tensor instead of
  // origin input tensor.
  const auto LDA = blocking_input_ ? config.C_block : ic_;
  const auto LDC = blocking_output_ ? config.K_block : oc_ * groups_;

  const int id_padded = is_3d_ ? (id_ + pd_b_ + pd_e_) : 1;
  const int ih_padded = ih_ + ph_b_ + ph_e_;
  const int iw_padded = iw_ + pw_b_ + pw_e_;
  const auto dtype_input = get_input_dtype();
  const auto dtype_weight = get_weight_dtype();
  const auto dtype_output = get_output_dtype();
  const int num_threads = runtime_config_t::get().get_num_threads();
  int src_row_tile_size = (config.tile_q - 1) * sw_ + dw_ * (kw_ - 1) + 1;
  typedef enum { LEFT_PAD = 0, BOTH_PAD, RIGHT_PAD } pad_kind;

  // some shapes might have less pad than given at the end of current axis
  auto get_num_pad_end = [](int ip, int k, int s, int p) {
    int remaining = (ip - k) % s;
    int num_pad_end = (remaining == 0)
      ? utils::divide_and_ceil(p, s)
      : ((p > remaining) ? utils::divide_and_ceil(p - remaining, s) : 0);
    return num_pad_end;
  };
  const int y_num_pad_top = utils::divide_and_ceil(ph_b_, sh_);
  const int y_num_pad_left = utils::divide_and_ceil(pw_b_, sw_);
  const int y_num_pad_front = is_3d_ ? utils::divide_and_ceil(pd_b_, sd_) : 0;
  const int y_num_pad_bottom
    = get_num_pad_end(ih_padded, dh_ * (kh_ - 1) + 1, sh_, ph_e_);
  const int y_num_pad_right
    = get_num_pad_end(iw_padded, dw_ * (kw_ - 1) + 1, sw_, pw_e_);
  const int y_num_pad_back
    = is_3d_ ? get_num_pad_end(id_padded, dd_ * (kd_ - 1) + 1, sd_, pd_e_) : 0;

  const int y_unpad_top = y_num_pad_top;
  const int y_unpad_bottom = oh_ - y_num_pad_bottom - 1;
  const int y_unpad_left = y_num_pad_left;
  const int y_unpad_right = ow_ - y_num_pad_right - 1;
  const int y_unpad_front = is_3d_ ? y_num_pad_front : 0;
  const int y_unpad_back = is_3d_ ? od_ - y_num_pad_front - 1 : 1;
  const uint32_t lanes = get_lanes(ctx, config.C_block, dtype_input);

  // large pd and ph will be skipped for non-os-blocking approach.
  const bool large_pad = src_row_tile_size < pw_b_ || src_row_tile_size < pw_e_;

  const int work_amount = mb_ * K_num_block * ow_ / config.tile_q;
  bool reuse_sub_tensor = sh_ < (dh_ * (kh_ - 1) + 1) && C_num_block == 1
    && is_parallel_space_enough(work_amount, num_threads) && dh_ == 1;
  bool use_var_bs = attrs_.get_or_else("use_var_bs", true);
  auto use_rl = attrs_.get_or_else("use_rl", ops::rl_kind::NO_LOWERING);

  // TODO(xxx): fix inverse filter correctness issue when use_var_bs==true
  if (inverse_filter_) { use_var_bs = false; }

  int list_size = kd_ * kh_ * kw_;
  int brgemm_k = 1, num_brgemm_k = 1;
  int num_kw = kw_, kw_step = 1;
  int extra_padding = 0;
  if (use_rl == ops::rl_kind::KW_LOWERING) {
    // TODO(xxx): enable advanced optimization for kw_lowering
    reuse_sub_tensor = false;
    use_var_bs = false;
    brgemm_k = attrs_.get_or_else("brgemm_k", 1);
    num_brgemm_k = attrs_.get_or_else("num_brgemm_k", 1);
    extra_padding = attrs_.get_or_else("extra_padding", 0);
    list_size = num_brgemm_k * kd_;
    num_kw = num_brgemm_k / kh_;
    kw_step = brgemm_k / ic_;
    src_row_tile_size += extra_padding / ic_;
  }

  _tensor_(pbuffer, dtype_input, {src_row_tile_size, LDA});
  if (!use_var_bs) {
    // when not using var_bs, define a unified zero-buffer for padding.
    builtin::mem_zero(pbuffer, src_row_tile_size * LDA, dtype_input);
  }

  // thread shared var to hold stateful status
  _tensor_(g_sub_tensor, dtype_input,
    is_3d_ ? std::vector<expr> {num_threads, kd_, kh_, src_row_tile_size, LDA}
           : std::vector<expr> {num_threads, kh_, src_row_tile_size, LDA});
  _tensor_(g_cur_indices, datatypes::u32, {num_threads, kh_});
  _tensor_(g_init_state, datatypes::boolean, {num_threads});
  auto data_size
    = math_utils::get_dims_product(in_tensors_[0].get_blocking_dims())
    * utils::get_sizeof_type(get_input_dtype());
  auto weight_size
    = math_utils::get_dims_product(in_tensors_[1].get_blocking_dims())
    * utils::get_sizeof_type(get_weight_dtype());
  auto L2_cache_size = ctx->machine_.cpu_flags_.getDCacheSize(2);
  int oc_split = get_oc_split_factor(
    mb_, data_size, weight_size, L2_cache_size, K_num_block);

  int outer_range
    = reuse_sub_tensor ? ow_ / config.tile_q : oh_ / config.tile_p;
  int inner_range
    = reuse_sub_tensor ? oh_ / config.tile_p : ow_ / config.tile_q;
  auto input_expr_dims = input.checked_as<tensor>()->dims_;
  auto mb_expr_ = input_expr_dims[0];
  _named_for_(lok, outer_k, 0, oc_split, 1, for_type::PARALLEL) {
    _named_for_(ln, n, 0, mb_expr_, 1) {
      _named_for_(lg, g, 0, groups_) {
        _named_for_(lk, k_i, 0, K_num_block / oc_split) {
          expr k_o = outer_k * K_num_block / oc_split + k_i;
          _named_for_(lp, outer_var, 0, outer_range) {
            expr p_o, q_o;
            if (reuse_sub_tensor) {
              q_o = outer_var;
            } else {
              p_o = outer_var;
            }
            _named_for_(ld, d_o, 0, od_ / config.tile_d) {
              _var_init_(
                tid, datatypes::s32, builder::make_get_group_thread_id(-1));

              _tensor_(A_list, datatypes::pointer, {list_size});
              _tensor_(B_list, datatypes::pointer, {list_size});
              _var_(prev_indices, datatypes::u32);
              _var_(num_h_pad, datatypes::s32);
              _var_(h_pad_begin_idx, datatypes::index);
              _var_(h_pad_end_idx, datatypes::index);
              _var_(h_unpad_begin_idx, datatypes::index);
              _var_(h_unpad_end_idx, datatypes::index);
              _var_(real_l_pad, datatypes::s32);
              _var_(real_r_pad, datatypes::s32);
              _var_(copy_width, datatypes::s32);
              _var_(num_d_pad, datatypes::s32);
              _var_(d_pad_begin_idx, datatypes::index);
              _var_(d_pad_end_idx, datatypes::index);
              _var_(d_unpad_begin_idx, datatypes::index);
              _var_(d_unpad_end_idx, datatypes::index);

              _for_(d_i, 0, config.tile_d) {
                // initialized stateful vars for each thread.
                if (reuse_sub_tensor) {
                  _for_(gi, 0, kh_) {
                    g_cur_indices[{tid, gi}]
                      = builder::make_cast(datatypes::u32, gi);
                  }
                  g_init_state[tid] = true;
                }
                _for_(inner_var, 0, inner_range) {
                  if (reuse_sub_tensor) {
                    p_o = inner_var;
                  } else {
                    q_o = inner_var;
                  }
                  _for_(p_i, 0, config.tile_p) {
                    _for_(c_o, 0, C_num_block) {
                      std::vector<expr> output_pos = is_3d_
                        ? (blocking_output_
                            ? std::vector<expr> {n, g * K_num_block + k_o,
                              d_o * config.tile_d + d_i,
                              p_o * config.tile_p + p_i, q_o * config.tile_q, 0}
                            : std::vector<expr> {n, d_o * config.tile_d + d_i,
                              p_o * config.tile_p + p_i, q_o * config.tile_q,
                              (g * K_num_block + k_o) * config.K_block})
                        : (blocking_output_
                            ? std::vector<expr> {n, g * K_num_block + k_o,
                              p_o * config.tile_p + p_i, q_o * config.tile_q, 0}
                            : std::vector<expr> {n, p_o * config.tile_p + p_i,
                              q_o * config.tile_q,
                              (g * K_num_block + k_o) * config.K_block});

                      auto update_pad_idx =
                        [](const expr &cur_o, const expr &cur_i, const int ker,
                          const int pad, const int dilation, const int in,
                          const int unpad_begin, const int unpad_end,
                          expr::lvalue_proxy_t &num_pad,
                          expr::lvalue_proxy_t &pad_begin_idx,
                          expr::lvalue_proxy_t &pad_end_idx,
                          expr::lvalue_proxy_t &unpad_begin_idx,
                          expr::lvalue_proxy_t &unpad_end_idx) {
                          _if_((cur_o >= unpad_begin) && (cur_o <= unpad_end)) {
                            num_pad = 0;
                            pad_begin_idx = 0;
                            pad_end_idx = 0;
                            unpad_begin_idx = 0;
                            unpad_end_idx = ker;
                          }
                          _else_ {
                            _if_(cur_o < unpad_begin) {
                              num_pad
                                = (pad
                                    - builder::make_cast(datatypes::s32, cur_i)
                                    - 1)
                                  / dilation
                                + 1;
                              expr num_right_pad = divide_and_ceil(
                                ((builder::make_cast(datatypes::s32, cur_i)
                                  + (ker - 1) * dilation + 1 - (in + pad))),
                                dilation);
                              pad_begin_idx = 0;
                              pad_end_idx = num_pad;
                              unpad_begin_idx = num_pad;
                              unpad_end_idx = ker;
                              _if_(num_right_pad > 0) {
                                unpad_end_idx = ker - num_right_pad;
                              }
                            }
                            _else_ {
                              num_pad = divide_and_ceil(
                                ((builder::make_cast(datatypes::s32, cur_i)
                                  + (ker - 1) * dilation + 1 - (in + pad))),
                                dilation);
                              pad_begin_idx = ker - num_pad;
                              pad_end_idx = ker;
                              unpad_begin_idx = 0;
                              unpad_end_idx = ker - num_pad;
                            }
                          }
                        };

                      auto cur_d = d_o * config.tile_d + d_i;
                      auto cur_id = cur_d * sd_;
                      auto cur_p = p_o * config.tile_p + p_i;
                      auto cur_ih = cur_p * sh_;
                      auto cur_tile_begin = q_o * config.tile_q;
                      auto cur_tile_end = cur_tile_begin + config.tile_q - 1;
                      auto cur_iw = cur_tile_begin * sw_;

                      if (is_3d_) {
                        update_pad_idx(cur_d, cur_id, kd_, pd_b_, dd_, id_,
                          y_unpad_front, y_unpad_back, num_d_pad,
                          d_pad_begin_idx, d_pad_end_idx, d_unpad_begin_idx,
                          d_unpad_end_idx);
                      }
                      update_pad_idx(cur_p, cur_ih, kh_, ph_b_, dh_, ih_,
                        y_unpad_top, y_unpad_bottom, num_h_pad, h_pad_begin_idx,
                        h_pad_end_idx, h_unpad_begin_idx, h_unpad_end_idx);

                      auto zero_out_sub_tensor = [&]() {
                        builtin::brgemm_init(tensor_ptr(output, output_pos),
                          config.tile_q, config.K_block, LDC, dtype_output, 0);
                      };

                      auto process_tile_with_pad =
                        [&](const expr &d_begin, const expr &d_end,
                          const expr &h_begin, const expr &h_end,
                          const expr &left_pad, const expr &right_pad,
                          const expr &tile_size_exclude_right_pad,
                          const pad_kind &kind, const expr &sub_tsr_hi = 0,
                          const bool &update_mode = false) {
                          _for_(di, d_begin, d_end) {
                            _for_(hi, h_begin, h_end) {
                              _var_(sub_tsr_d, datatypes::index);
                              _var_(sub_tsr_h, datatypes::index);
                              if (is_3d_) { sub_tsr_d = di - d_begin; }
                              sub_tsr_h
                                = update_mode ? sub_tsr_hi : (hi - h_begin);
                              if (kind == LEFT_PAD || kind == BOTH_PAD) {
                                builtin::brgemm_init(
                                  tensor_ptr(g_sub_tensor,
                                    is_3d_ ? std::vector<expr> {tid, sub_tsr_d,
                                      sub_tsr_h, 0, 0}
                                           : std::vector<expr> {tid, sub_tsr_h,
                                             0, 0}),
                                  builder::make_cast(datatypes::s32, left_pad),
                                  config.C_block, LDA, dtype_input, 0);
                              }

                              // mapping dst to src_padded then mapping to
                              // original src to copy the origin elements.
                              _for_(j, left_pad, tile_size_exclude_right_pad) {
                                _for_(k, 0, config.C_block, (int)lanes) {
                                  if (is_3d_) {
                                    g_sub_tensor[span_t(
                                      {tid, sub_tsr_d, sub_tsr_h, j, k}, lanes)]
                                      = input[blocking_input_
                                          ? span_t({n, g * C_num_block + c_o,
                                                     cur_id + di * dd_ - pd_b_,
                                                     cur_ih + hi * dh_ - ph_b_,
                                                     cur_iw + j - pw_b_, k},
                                            lanes)
                                          : span_t(
                                            {n, cur_id + di * dd_ - pd_b_,
                                              cur_ih + hi * dh_ - ph_b_,
                                              cur_iw + j - pw_b_,
                                              (g * C_num_block + c_o)
                                                  * config.C_block
                                                + k},
                                            lanes)];
                                  } else {
                                    g_sub_tensor[span_t(
                                      {tid, sub_tsr_h, j, k}, lanes)]
                                      = input[blocking_input_
                                          ? span_t({n, g * C_num_block + c_o,
                                                     cur_ih + hi * dh_ - ph_b_,
                                                     cur_iw + j - pw_b_, k},
                                            lanes)
                                          : span_t(
                                            {n, cur_ih + hi * dh_ - ph_b_,
                                              cur_iw + j - pw_b_,
                                              (g * C_num_block + c_o)
                                                  * config.C_block
                                                + k},
                                            lanes)];
                                  }
                                }
                              }

                              if (kind == RIGHT_PAD || kind == BOTH_PAD) {
                                builtin::brgemm_init(
                                  tensor_ptr(g_sub_tensor,
                                    is_3d_ ? std::vector<expr> {tid, sub_tsr_d,
                                      sub_tsr_h, tile_size_exclude_right_pad, 0}
                                           : std::vector<expr> {tid, sub_tsr_h,
                                             tile_size_exclude_right_pad, 0}),
                                  builder::make_cast(datatypes::s32,
                                    src_row_tile_size
                                      - tile_size_exclude_right_pad),
                                  config.C_block, LDA, dtype_input, 0);
                              }

                              _for_(wi, 0, num_kw) {
                                _var_(idx, datatypes::u32);
                                if (is_3d_) {
                                  auto valid_kh
                                    = (h_unpad_end_idx - h_unpad_begin_idx - 1)
                                      / dh_
                                    + 1;
                                  idx = builder::make_cast(datatypes::u32,
                                    use_var_bs
                                      ? (sub_tsr_d * valid_kh * num_kw
                                        + sub_tsr_h * num_kw + wi)
                                      : (di * kh_ * num_kw + hi * num_kw + wi));
                                } else {
                                  idx = builder::make_cast(datatypes::u32,
                                    use_var_bs ? (sub_tsr_h * num_kw + wi)
                                               : (hi * num_kw + wi));
                                }
                                // TODO(xxx): pack input for dilated conv
                                A_list[idx] = tensor_ptr(g_sub_tensor,
                                  is_3d_ ? std::vector<expr> {tid, sub_tsr_d,
                                    sub_tsr_h, wi * dw_ * kw_step, 0}
                                         : std::vector<expr> {tid, sub_tsr_h,
                                           wi * dw_ * kw_step, 0});
                              }
                            }
                          }
                        };

                      auto fill_sub_tensor = [&](const expr &d_unpad_begin = 0,
                                               const expr &d_unpad_end = 1) {
                        _if_(cur_tile_begin < y_unpad_left) {
                          _if_(y_unpad_right >= 0
                            && cur_tile_end <= y_unpad_right) {
                            // left pad only
                            real_l_pad = pw_b_
                              - builder::make_cast(datatypes::s32, cur_iw);
                            process_tile_with_pad(d_unpad_begin, d_unpad_end,
                              h_unpad_begin_idx, h_unpad_end_idx, real_l_pad, 0,
                              src_row_tile_size, LEFT_PAD);
                          }
                          _else_ {
                            // both left and right pad
                            real_l_pad = pw_b_
                              - builder::make_cast(datatypes::s32, cur_iw);
                            real_r_pad = builder::make_cast(datatypes::s32,
                                           cur_iw + src_row_tile_size)
                              - (iw_padded - pw_e_);
                            copy_width = src_row_tile_size - real_r_pad;
                            process_tile_with_pad(d_unpad_begin, d_unpad_end,
                              h_unpad_begin_idx, h_unpad_end_idx, real_l_pad,
                              real_r_pad, copy_width, BOTH_PAD);
                          }
                        }
                        _else_ {
                          // right pad only
                          real_r_pad = builder::make_cast(datatypes::s32,
                                         cur_iw + src_row_tile_size)
                            - (iw_padded - pw_e_);
                          copy_width = src_row_tile_size - real_r_pad;
                          process_tile_with_pad(d_unpad_begin, d_unpad_end,
                            h_unpad_begin_idx, h_unpad_end_idx, 0, real_r_pad,
                            copy_width, RIGHT_PAD);
                        }
                      };

                      auto update_sub_tensor = [&](const int kd = 1) {
                        _tensor_(modified_indices, datatypes::index, {sh_});
                        _var_(m_idx, datatypes::index);
                        _var_(actual_idx, datatypes::index);
                        m_idx = 0;
                        _for_(idx, 0, kh_) {
                          prev_indices = g_cur_indices[{tid, idx}];
                          _if_(prev_indices < sh_) {
                            g_cur_indices[{tid, idx}]
                              = prev_indices + kh_ - sh_;
                            modified_indices[m_idx] = idx;
                            m_idx = m_idx + 1;
                          }
                          _else_ {
                            g_cur_indices[{tid, idx}] = prev_indices - sh_;
                          }
                        }

                        _for_(idx, 0, sh_) {
                          m_idx = modified_indices[idx];
                          actual_idx = g_cur_indices[{tid, m_idx}];
                          // update necessary row of sub-tensor according
                          // to actual_idx
                          _if_(cur_tile_begin < y_unpad_left) {
                            _if_(y_unpad_right >= 0
                              && cur_tile_end <= y_unpad_right) {
                              // left pad only
                              real_l_pad = pw_b_
                                - builder::make_cast(datatypes::s32, cur_iw);
                              process_tile_with_pad(0, kd, actual_idx,
                                actual_idx + 1, real_l_pad, 0,
                                src_row_tile_size, LEFT_PAD, m_idx, true);
                            }
                            _else_ {
                              // both left and right pad
                              real_l_pad = pw_b_
                                - builder::make_cast(datatypes::s32, cur_iw);
                              real_r_pad = builder::make_cast(datatypes::s32,
                                             cur_iw + src_row_tile_size)
                                - (iw_padded - pw_e_);
                              copy_width = src_row_tile_size - real_r_pad;
                              process_tile_with_pad(0, kd, actual_idx,
                                actual_idx + 1, real_l_pad, real_r_pad,
                                copy_width, BOTH_PAD, m_idx, true);
                            }
                          }
                          _else_ {
                            // right pad only
                            real_r_pad = builder::make_cast(datatypes::s32,
                                           cur_iw + src_row_tile_size)
                              - (iw_padded - pw_e_);
                            copy_width = src_row_tile_size - real_r_pad;
                            process_tile_with_pad(0, kd, actual_idx,
                              actual_idx + 1, 0, real_r_pad, copy_width,
                              RIGHT_PAD, m_idx, true);
                          }
                        }

                        // update A_list with reusable sub-tensor using
                        // cur_indices, no padding on depth or height axis.
                        _for_(di, 0, kd) {
                          _for_(hi, 0, kh_) {
                            _var_(sub_tsr_idx, datatypes::index);
                            sub_tsr_idx = builder::make_cast(
                              datatypes::index, g_cur_indices[{tid, hi}]);
                            _for_(wi, 0, num_kw) {
                              _var_(A_idx, datatypes::u32);

                              if (is_3d_) {
                                A_idx = builder::make_cast(datatypes::u32,
                                  di * kh_ * num_kw + sub_tsr_idx * num_kw
                                    + wi * kw_step);
                                A_list[A_idx] = tensor_ptr(g_sub_tensor,
                                  {tid, di, hi, wi * kw_step * dw_, 0});
                              } else {
                                A_idx = builder::make_cast(
                                  datatypes::u32, sub_tsr_idx * num_kw + wi);
                                A_list[A_idx] = tensor_ptr(g_sub_tensor,
                                  {tid, hi, wi * kw_step * dw_, 0});
                              }
                            }
                          }
                        }
                      };

                      auto call_brgemm = [&](int valid_kh, int valid_kd = 1) {
                        COMPILE_ASSERT(valid_kd > 0 && valid_kh > 0,
                          "Expect valid_kh and valid_kd are positive "
                          "integer, "
                          "but got valid_kh="
                            << valid_kh << ", valid_kd=" << valid_kd << ".");
                        auto valid_ker_size = valid_kd * valid_kh * num_kw;
                        int M = config.tile_q, K = config.C_block,
                            N = config.K_block;
                        if (use_rl == ops::rl_kind::KW_LOWERING) {
                          K = brgemm_k;
                        }
                        auto hint_A_size = M * K * valid_ker_size;
                        auto hint_B_size = K * N * valid_ker_size;
                        auto hint_C_size = M * N;
                        sc_brgemm_attrs_t brg_attrs {
                          {brgemm::attr_key::max_bs, valid_ker_size},
                          {brgemm::attr_key::hint_expected_A_size, hint_A_size},
                          {brgemm::attr_key::hint_expected_B_size, hint_B_size},
                          {brgemm::attr_key::hint_expected_C_size, hint_C_size},
                          {brgemm::attr_key::use_interleave_stores, true},
                          {brgemm::attr_key::use_uker, true},
                          {brgemm::attr_key::var_bs,
                            use_var_bs ? true : false}};

                        _if_(c_o == 0) {
                          builtin::brgemm_init_list_update(A_list, B_list,
                            tensor_ptr(output, output_pos), 1, M, N, K,
                            sw_ * LDA, config.K_block, LDC, 1, 1,
                            valid_ker_size, dtype_input, dtype_weight,
                            brg_attrs);
                        }
                        _else_ {
                          builtin::brgemm_list_update(A_list, B_list,
                            tensor_ptr(output, output_pos), 1, M, N, K,
                            sw_ * LDA, config.K_block, LDC, 1, 1,
                            valid_ker_size, dtype_input, dtype_weight,
                            brg_attrs);
                        }
                      };

                      auto generate_var_bs
                        = [](const std::function<void(int, int)> &func, int k,
                            int o, int s, int d, int p, int i, int valid_kd,
                            expr &cur_pos) {
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
                                else_stmt = make_if_else_unattached(
                                  cond, then_stmt, else_stmt);
                              }
                            }
                            current_builder->emit(else_stmt);
                          };

                      auto do_var_bs_for_2d = [&](const int kd, const int kh) {
                        generate_var_bs(call_brgemm, kh, oh_, sh_, dh_, ph_b_,
                          ih_, kd, cur_p);
                      };

                      if (is_3d_) {
                        auto cond = large_pad
                          ? (((cur_iw + src_row_tile_size <= pw_b_)
                               || (cur_iw > iw_ + pw_b_))
                            || (num_d_pad >= kd_ || num_h_pad >= kh_))
                          : (num_d_pad >= kd_ || num_h_pad >= kh_);
                        _if_(cond) { zero_out_sub_tensor(); }
                        _else_ {
                          // 1) fill A_list
                          if (!use_var_bs) {
                            _for_(di, 0, kd_) {
                              // all zero feature map
                              _if_(
                                di >= d_pad_begin_idx && di < d_pad_end_idx) {
                                _for_(hi, 0, kh_) {
                                  _for_(wi, 0, num_kw) {
                                    _var_(idx, datatypes::u32);
                                    idx = builder::make_cast(datatypes::u32,
                                      di * kh_ * num_kw + hi * num_kw + wi);
                                    A_list[idx] = tensor_ptr(pbuffer, {0, 0});
                                  }
                                }
                              }
                              _else_ {
                                _for_(hi, h_pad_begin_idx, h_pad_end_idx) {
                                  _for_(wi, 0, num_kw) {
                                    _var_(idx, datatypes::u32);
                                    idx = builder::make_cast(datatypes::u32,
                                      di * kh_ * num_kw + hi * num_kw + wi);
                                    A_list[idx] = tensor_ptr(pbuffer, {0, 0});
                                  }
                                }
                              }
                            }
                          }

                          // 1.1) The middle region which don't need to copy
                          // input rows but just refer to original input
                          // buffer.
                          _if_(cur_tile_begin >= y_unpad_left
                            && cur_tile_end <= y_unpad_right) {
                            _for_(di, d_unpad_begin_idx, d_unpad_end_idx) {
                              _for_(hi, h_unpad_begin_idx, h_unpad_end_idx) {
                                _for_(wi, 0, num_kw) {
                                  _var_(idx, datatypes::u32);
                                  auto valid_kh
                                    = h_unpad_end_idx - h_unpad_begin_idx;
                                  idx = builder::make_cast(datatypes::u32,
                                    use_var_bs ? ((di - d_unpad_begin_idx)
                                        * valid_kh * num_kw
                                      + (hi - h_unpad_begin_idx) * num_kw + wi)
                                               : (di * kh_ * num_kw
                                                 + hi * num_kw + wi));
                                  A_list[idx] = tensor_ptr(input,
                                    blocking_input_
                                      ? std::vector<expr> {n,
                                        g * C_num_block + c_o,
                                        cur_id + di * dd_ - pd_b_,
                                        cur_ih + hi * dh_ - ph_b_,
                                        cur_iw + wi * dw_ * kw_step - pw_b_, 0}
                                      : std::vector<expr> {n,
                                        cur_id + di * dd_ - pd_b_,
                                        cur_ih + hi * dh_ - ph_b_,
                                        cur_iw + wi * dw_ * kw_step - pw_b_,
                                        (g * C_num_block + c_o)
                                          * config.C_block});
                                }
                              }
                            }
                          }
                          _else_ {
                            // copy rows and do physical padding
                            if (!reuse_sub_tensor) {
                              fill_sub_tensor(
                                d_unpad_begin_idx, d_unpad_end_idx);
                            } else {
                              _if_(num_d_pad > 0 || num_h_pad > 0
                                || g_init_state[tid]) {
                                _if_(num_d_pad == 0 && num_h_pad == 0) {
                                  g_init_state[tid] = false;
                                }
                                fill_sub_tensor(
                                  d_unpad_begin_idx, d_unpad_end_idx);
                              }
                              _else_ {
                                // num_d_pad == 0 && num_h_pad == 0, reuse
                                // sub-tsr
                                update_sub_tensor(kd_);
                              }
                            }
                          }

                          // 2) fill B_list
                          if (use_var_bs) {
                            _for_(di, d_unpad_begin_idx, d_unpad_end_idx) {
                              _for_(hi, h_unpad_begin_idx, h_unpad_end_idx) {
                                _for_(wi, 0, num_kw) {
                                  _var_(idx, datatypes::u32);
                                  auto valid_kh
                                    = h_unpad_end_idx - h_unpad_begin_idx;
                                  idx = builder::make_cast(datatypes::u32,
                                    ((di - d_unpad_begin_idx) * valid_kh
                                        * num_kw
                                      + (hi - h_unpad_begin_idx) * num_kw
                                      + wi));
                                  if (inverse_filter_) {
                                    idx = builder::make_cast(datatypes::u32,
                                      (d_unpad_end_idx - d_unpad_begin_idx)
                                          * (h_unpad_end_idx
                                            - h_unpad_begin_idx)
                                          * num_kw
                                        - 1 - idx);
                                  }
                                  B_list[idx] = tensor_ptr(weight,
                                    kpack > 1
                                      ? std::vector<expr> {g * K_num_block
                                          + k_o,
                                        c_o, di, hi, wi * kw_step, 0, 0, 0}
                                      : std::vector<expr> {
                                        g * K_num_block + k_o, c_o, di, hi,
                                        wi * kw_step, 0, 0});
                                }
                              }
                            }
                          } else {
                            _for_(di, 0, kd_) {
                              _for_(hi, 0, kh_) {
                                _for_(wi, 0, num_kw) {
                                  _var_(idx, datatypes::u32);
                                  idx = builder::make_cast(datatypes::u32,
                                    di * kh_ * num_kw + hi * num_kw + wi);
                                  if (inverse_filter_) {
                                    idx = builder::make_cast(datatypes::u32,
                                      kd_ * kh_ * num_kw - 1 - idx);
                                  }
                                  B_list[idx] = tensor_ptr(weight,
                                    kpack > 1
                                      ? std::vector<expr> {g * K_num_block
                                          + k_o,
                                        c_o, di, hi, wi * kw_step, 0, 0, 0}
                                      : std::vector<expr> {
                                        g * K_num_block + k_o, c_o, di, hi,
                                        wi * kw_step, 0, 0});
                                }
                              }
                            }
                          }

                          if (use_var_bs) {
                            // determine the exact value of var_bs for
                            // brgemm call, Ai & Bi are already
                            // fulfilled at this stage.
                            generate_var_bs(do_var_bs_for_2d, kd_, od_, sd_,
                              dd_, pd_b_, id_, kh_, cur_d);
                          } else {
                            call_brgemm(kh_, kd_);
                          }
                        }
                      } else {
                        auto cond = large_pad
                          ? (((cur_iw + src_row_tile_size <= pw_b_)
                               || (cur_iw > iw_ + pw_b_))
                            || (num_h_pad >= kh_))
                          : (num_h_pad >= kh_);
                        _if_(cond) { zero_out_sub_tensor(); }
                        _else_ {
                          auto fill_A_and_B_list = [&]() {
                            if (!use_var_bs) {
                              // Add zero-padding tensorptr to A_list
                              _for_(hi, h_pad_begin_idx, h_pad_end_idx) {
                                _for_(wi, 0, num_kw) {
                                  _var_(idx, datatypes::u32);
                                  idx = builder::make_cast(
                                    datatypes::u32, hi * num_kw + wi);
                                  A_list[idx] = tensor_ptr(pbuffer, {0, 0});
                                }
                              }

                              _if_(
                                h_pad_begin_idx == 0 && h_unpad_end_idx < kh_) {
                                // Add zero-padding tensorptr to A_list
                                _for_(hi, h_unpad_end_idx, kh_) {
                                  _for_(wi, 0, num_kw) {
                                    _var_(idx, datatypes::u32);
                                    idx = builder::make_cast(
                                      datatypes::u32, hi * num_kw + wi);
                                    A_list[idx] = tensor_ptr(pbuffer, {0, 0});
                                  }
                                }
                              }
                            }
                            _if_(cur_tile_begin >= y_unpad_left
                              && cur_tile_end <= y_unpad_right) {
                              _for_(hi, h_unpad_begin_idx, h_unpad_end_idx) {
                                _for_(wi, 0, num_kw) {
                                  _var_(idx, datatypes::u32);
                                  idx = builder::make_cast(datatypes::u32,
                                    (use_var_bs ? (hi - h_unpad_begin_idx) : hi)
                                        * num_kw
                                      + wi);
                                  A_list[idx] = tensor_ptr(input,
                                    blocking_input_
                                      ? std::vector<expr> {n,
                                        g * C_num_block + c_o,
                                        cur_ih + hi * dh_ - ph_b_,
                                        cur_iw + wi * dw_ * kw_step - pw_b_, 0}
                                      : std::vector<expr> {n,
                                        cur_ih + hi * dh_ - ph_b_,
                                        cur_iw + wi * dw_ * kw_step - pw_b_,
                                        (g * C_num_block + c_o)
                                          * config.C_block});
                                }
                              }
                            }
                            _else_ {
                              // copy rows and do physical padding
                              if (!reuse_sub_tensor) {
                                fill_sub_tensor();
                              } else {
                                _if_(num_h_pad > 0 || g_init_state[tid]) {
                                  _if_(num_h_pad == 0) {
                                    g_init_state[tid] = false;
                                  }
                                  fill_sub_tensor();
                                }
                                _else_ { update_sub_tensor(); }
                              }
                            }

                            // 2) fill B_list
                            if (use_var_bs) {
                              _for_(hi, h_unpad_begin_idx, h_unpad_end_idx) {
                                _for_(wi, 0, num_kw) {
                                  _var_(idx, datatypes::u32);
                                  idx = builder::make_cast(datatypes::u32,
                                    (hi - h_unpad_begin_idx) * num_kw + wi);
                                  if (inverse_filter_) {
                                    idx = builder::make_cast(datatypes::u32,
                                      (h_unpad_end_idx - h_unpad_begin_idx)
                                          * num_kw
                                        - 1 - idx);
                                  }
                                  B_list[idx] = tensor_ptr(weight,
                                    kpack > 1
                                      ? std::vector<expr> {g * K_num_block
                                          + k_o,
                                        c_o, hi, wi * kw_step, 0, 0, 0}
                                      : std::vector<expr> {
                                        g * K_num_block + k_o, c_o, hi,
                                        wi * kw_step, 0, 0});
                                }
                              }
                            } else {
                              _for_(hi, 0, kh_) {
                                _for_(wi, 0, num_kw) {
                                  _var_(idx, datatypes::u32);
                                  idx = builder::make_cast(
                                    datatypes::u32, hi * num_kw + wi);
                                  if (inverse_filter_) {
                                    idx = builder::make_cast(
                                      datatypes::u32, kh_ * num_kw - 1 - idx);
                                  }
                                  B_list[idx] = tensor_ptr(weight,
                                    kpack > 1
                                      ? std::vector<expr> {g * K_num_block
                                          + k_o,
                                        c_o, hi, wi * kw_step, 0, 0, 0}
                                      : std::vector<expr> {
                                        g * K_num_block + k_o, c_o, hi,
                                        wi * kw_step, 0, 0});
                                }
                              }
                            }
                          };
                          if (use_rl == ops::rl_kind::KW_LOWERING
                            && extra_padding > 0) {
                            _if_((p_o == oh_ / config.tile_p - 1)
                              && (p_i == config.tile_p - 1)
                              && (q_o == ow_ / config.tile_q - 1)) {
                              COMPILE_ASSERT(extra_padding % ic_ == 0,
                                "Expect extra_padding is dividable by ic, "
                                "but got extra_padding="
                                  << extra_padding << ",ic=" << ic_ << ".");

                              int max_col_in_bytes = 32;
                              int ic_in_bytes
                                = ic_ * utils::get_sizeof_type(dtype_input);
                              COMPILE_ASSERT(ic_in_bytes <= max_col_in_bytes,
                                "Expect ic*dtype <=32 for kw lowering, but got "
                                  << ic_in_bytes << ".");
                              _tensor_(tmp_input, dtype_input,
                                {kh_, src_row_tile_size, ic_});

                              auto lanes = get_minimal_lanes(ic_in_bytes);
                              auto mask = convert_int_to_mask(ic_in_bytes);
                              auto mask_expr = mask != 0
                                ? builder::make_cast(get_dtype(lanes), mask)
                                : expr();
                              _for_(ih, 0, kh_) {
                                _for_(iw, 0, src_row_tile_size) {
                                  auto cur_h = (oh_ - 1) * sh_ + ih;
                                  auto cur_w = (ow_ - config.tile_q) * sw_ + iw;
                                  _if_(cur_h < ph_b_ + ih_ && cur_w >= pw_b_
                                    && cur_w < pw_b_ + iw_) {
                                    tmp_input[span_t(
                                      {ih, iw, 0}, lanes, mask_expr)]
                                      = input[blocking_input_
                                          ? span_t({n, g, cur_h - ph_b_,
                                                     cur_w - pw_b_, 0},
                                            lanes, mask_expr)
                                          : span_t(
                                            {n, cur_h - ph_b_, cur_w - pw_b_,
                                              g * config.C_block},
                                            lanes, mask_expr)];
                                  }
                                  _else_ {
                                    tmp_input[span_t(
                                      {ih, iw, 0}, lanes, mask_expr)]
                                      = pbuffer[span_t(
                                        {0, 0}, lanes, mask_expr)];
                                  }
                                }
                              }

                              _for_(r, 0, kh_) {
                                _for_(s, 0, num_kw) {
                                  auto idx = r * num_kw + s;
                                  A_list[idx] = tensor_ptr(
                                    tmp_input, {r, s * kw_step, 0});
                                  B_list[idx] = tensor_ptr(weight,
                                    kpack > 1
                                      ? std::vector<expr> {g * K_num_block
                                          + k_o,
                                        0, r, s * kw_step, 0, 0, 0}
                                      : std::vector<expr> {
                                        g * K_num_block + k_o, 0, r,
                                        s * kw_step, 0, 0});
                                }
                              }
                            }
                            _else_ { fill_A_and_B_list(); }
                          } else {
                            fill_A_and_B_list();
                          }
                          if (use_var_bs) {
                            do_var_bs_for_2d(kd_, kh_);
                          } else {
                            call_brgemm(kh_);
                          }
                        }
                      }
                    }

                    // tile_q * K_block
                    create_anchor(fusion, output, n, 1, g * K_num_block + k_o,
                      1, d_o * config.tile_d + d_i, 1,
                      p_o * config.tile_p + p_i, 1, q_o * config.tile_q,
                      config.tile_q, config.K_block, config.K_block,
                      blocking_output_, is_3d_);
                  }
                  // tile_p * tile_q *K_block
                  create_anchor(fusion, output, n, 1, g * K_num_block + k_o, 1,
                    d_o * config.tile_d + d_i, 1, p_o * config.tile_p,
                    config.tile_p, q_o * config.tile_q, config.tile_q,
                    config.K_block, config.K_block, blocking_output_, is_3d_);
                }
                if (reuse_sub_tensor) {
                  // oh_ * tile_q * K_block
                  create_anchor(fusion, output, n, 1, g * K_num_block + k_o, 1,
                    d_o * config.tile_d + d_i, 1, 0, oh_, q_o * config.tile_q,
                    config.tile_q, config.K_block, config.K_block,
                    blocking_output_, is_3d_);
                } else {
                  // tile_p * ow_ * K_block
                  create_anchor(fusion, output, n, 1, g * K_num_block + k_o, 1,
                    d_o * config.tile_d + d_i, 1, p_o * config.tile_p,
                    config.tile_p, 0, ow_, config.K_block, config.K_block,
                    blocking_output_, is_3d_);
                }
              }
              if (is_3d_) {
                if (reuse_sub_tensor) {
                  // tile_d * oh_ * tile_q * K_block
                  create_anchor(fusion, output, n, 1, g * K_num_block + k_o, 1,
                    d_o * config.tile_d, config.tile_d, 0, oh_,
                    q_o * config.tile_q, config.tile_q, config.K_block,
                    config.K_block, blocking_output_, true);
                } else {
                  // tile_d * tile_p * ow_ * K_block
                  create_anchor(fusion, output, n, 1, g * K_num_block + k_o, 1,
                    d_o * config.tile_d, config.tile_d, p_o * config.tile_p,
                    config.tile_p, 0, ow_, config.K_block, config.K_block,
                    blocking_output_, true);
                }
              }
            }
            if (is_3d_) {
              if (reuse_sub_tensor) {
                // od_ * oh_ * tile_q * K_block
                create_anchor(fusion, output, n, 1, g * K_num_block + k_o, 1, 0,
                  od_, 0, oh_, q_o * config.tile_q, config.tile_q,
                  config.K_block, config.K_block, blocking_output_, true);
              } else {
                // od_ * tile_p * ow_ * K_block
                create_anchor(fusion, output, n, 1, g * K_num_block + k_o, 1, 0,
                  od_, p_o * config.tile_p, config.tile_p, 0, ow_,
                  config.K_block, config.K_block, blocking_output_, true);
              }
            }
          }
          // od_ * oh_ * ow_ * K_block
          create_anchor(fusion, output, n, 1, g * K_num_block + k_o, 1, 0, od_,
            0, oh_, 0, ow_, config.K_block, config.K_block, blocking_output_,
            is_3d_);
        }
      }
      // od_ *oh_ *ow_ *oc
      if (groups_ == 1) {
        create_anchor(fusion, output, n, 1, outer_k * K_num_block / oc_split,
          K_num_block / oc_split, 0, od_, 0, oh_, 0, ow_, config.K_block,
          K_num_block / oc_split * config.K_block, blocking_output_, is_3d_);
      } else if (groups_ > 1 && oc_split == 1) {
        create_anchor(fusion, output, n, 1, groups_ * K_num_block, K_num_block,
          0, od_, 0, oh_, 0, ow_, config.K_block,
          K_num_block / oc_split * config.K_block, blocking_output_, is_3d_);
      } else { /*noops as discontiguous slice for groups >1 && outer_k > 1*/
      }
    }
  }
}

void gen_conv_fwd_t::schedule_loops(context_ptr ctx,
  const conv_fwd_config_t &config, stmt body,
  std::vector<for_loop> &fors) const {
  COMPILE_ASSERT(
    static_cast<int>(fors.size()) == 5 || static_cast<int>(fors.size()) == 6,
    "expected to have 5 for loops, but got " << fors.size() << " for loops.");
  for_loop ln = fors.at(0), lg = fors.at(1), lk = fors.at(2), ld = fors.at(3),
           lp = fors.at(4), lok;
  if (fors.size() == 6) {
    lok = fors[5];
    ln = lok->fuse(ln);
  }
  auto loop_sched = config.loop_sched;
  if (loop_sched == 0) {
    // default loop order ln->lk->lp->ld,
    // merge ln, lk, lp
    if (!is_1x1_conv_) {
      if (is_3d_) {
        ln->reorder(body, {ln, lg, lk, lp, ld});
      } else {
        ln->reorder(body, {ln, lg, lk, lp});
      }
    }
    auto outer = ln->fuse(lg);
    outer = outer->fuse(lk);
    outer = outer->fuse(lp);
    if (is_3d_ && ld.defined()) { outer->fuse(ld); }
    outer->kind_ = for_type::PARALLEL;
  } else if (loop_sched == 1) {
    // loop order lk->lp->ln
    // merge lk, lp, ln
    for_loop outer;
    if (is_3d_ && ld.defined()) {
      ln->reorder(body, {lg, lk, lp, ld, ln});
      outer = lg->fuse(lk);
      outer = outer->fuse(lp);
      outer = outer->fuse(ld);
    } else {
      ln->reorder(body, {lg, lk, lp, ln});
      outer = lg->fuse(lk);
      outer = outer->fuse(lp);
    }
    outer = outer->fuse(ln);
    outer->kind_ = for_type::PARALLEL;
  } else if (loop_sched == 2) {
    // loop order lp->lk->ln
    // merge lp, lk, ln
    for_loop outer;
    if (is_3d_ && ld.defined()) {
      ln->reorder(body, {lp, ld, lg, lk, ln});
      outer = lp->fuse(ld);
      outer = lp->fuse(lg);
      outer = outer->fuse(lk);
    } else {
      ln->reorder(body, {lp, lg, lk, ln});
      outer = lp->fuse(lg);
      outer = outer->fuse(lk);
    }
    outer = outer->fuse(ln);
    outer->kind_ = for_type::PARALLEL;
  } else if (loop_sched == 3) {
    // loop order lk->ln->lp
    // merge lk,ln,lp
    ln->reorder(body, {lg, lk, ln, lp});
    auto outer = lg->fuse(lk);
    outer = outer->fuse(ln);
    outer = outer->fuse(lp);
    if (is_3d_ && ld.defined()) { outer = outer->fuse(ld); }
    outer->kind_ = for_type::PARALLEL;
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
  int K_num_block = utils::rnd_up(oc_, K_block) / K_block;
  int C_num_block = utils::rnd_up(ic_, C_block) / C_block;

  // kpack is used to determine the vnni block format
  //  +----+--------------+
  //  | 1  | FP32         |
  //  +----+--------------+
  //  | 2  | VNNI_BF16    |
  //  +----+--------------+
  //  | 4  | VNNI_INT8    |
  //  +----+--------------+
  int kpack = 1;
  auto dtype_input = get_input_dtype();
  auto dtype_weight = get_weight_dtype();
  auto dtype_output = get_output_dtype();
  if (dtype_input == datatypes::bf16) {
    COMPILE_ASSERT((dtype_weight == datatypes::bf16),
      "Weights should be bf16 as "
      "data, the mixed datatypes is not supported yet!");
    COMPILE_ASSERT((dtype_output == datatypes::f32),
      "Output should be f32 when data and weights are in bf16.");
    kpack = 2;
  }
  if (dtype_input == datatypes::f16) {
    COMPILE_ASSERT((dtype_weight == datatypes::f16),
      "Weights should be f16 as "
      "data, the mixed datatypes is not supported yet!");
    COMPILE_ASSERT((dtype_output == datatypes::f32),
      "Output should be f32 when data and weights are in f16.");
    kpack = 2;
  }
  if (utils::is_one_of(dtype_input, datatypes::s8, datatypes::u8)) {
    COMPILE_ASSERT((dtype_weight == datatypes::s8),
      "Weights should be s8 when \
            data is s8/u8, the mixed datatypes is not supported yet!");
    COMPILE_ASSERT((dtype_output == datatypes::s32),
      "Output should be s32 when data and weights are in "
      "s8/u8.");
    kpack = 4;
  }

  const bool use_os_blocking
    = try_os_blocking_ && ops::is_amx_dtype(ctx, dtype_input);
  const bool pack_rows = use_os_blocking && (tile_os > 0 && ow_ % tile_os != 0);
  int os = actual_os_;
  COMPILE_ASSERT(tile_d && (od_ % tile_d == 0),
    "od should be dividable by tile_d, but got od=" << od_ << " tile_d="
                                                    << tile_d << ".");
  auto use_rl = attrs_.get_or_else("use_rl", ops::rl_kind::NO_LOWERING);
  COMPILE_ASSERT(use_rl != ops::rl_kind::FULL_LOWERING,
    "full reduce lowering should be dispatched to conv_rl!");
  if (use_rl == ops::rl_kind::KW_LOWERING) {
    COMPILE_ASSERT(C_block == ic_,
      "C_block should be same as ic for kw_lowering, but got C_block="
        << C_block << " ic=" << ic_ << ".");
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

  for_loop ln, lg, lk, ld, lp;
  loops = {ln, lg, lk, ld, lp};
  expr output = outputs[op_params_t::out];
  expr input = inputs[op_params_t::in_data];
  expr weight = inputs[op_params_t::in_weight];
  if (is_1x1_conv_) {
    COMPILE_ASSERT(pd_b_ == 0 && ph_b_ == 0 && pw_b_ == 0 && pd_e_ == 0
        && ph_e_ == 0 && pw_e_ == 0,
      "1x1 conv doesn't support padding!");
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
    if (pd_b_ == 0 && ph_b_ == 0 && pw_b_ == 0 && pd_e_ == 0 && ph_e_ == 0
      && pw_e_ == 0) {
      COMPILE_ASSERT(!inverse_filter_,
        "conv NxN (no padding) does not support inverse "
        "convolution.");
      if (is_3d_) {
        compute_conv3d_no_padding(ctx, config, fusion, output, input, weight,
          loops, K_num_block, C_num_block, os, kpack);
      } else {
        compute_conv_no_padding(ctx, config, fusion, output, input, weight,
          loops, K_num_block, C_num_block, os, kpack, use_os_blocking,
          pack_rows, os_acc_size, os_mask);
      }
    } else {
      if (ops::is_amx_dtype(ctx, dtype_input) || is_3d_
        || use_rl == ops::rl_kind::KW_LOWERING || inverse_filter_) {
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
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
