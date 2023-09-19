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

#include <algorithm>
#include <functional>
#include <numeric>
#include <utility>
#include "nested_conv_fwd.hpp"
#include "utils.hpp"
#include <compiler/ir/builder.hpp>
#include <compiler/ir/builtin.hpp>
#include <compiler/ir/easy_build.hpp>
#include <compiler/ir/graph/fusion_mgr.hpp>
#include <compiler/ir/graph/trait/configurable.hpp>
#include <compiler/ir/transform/auto_cast.hpp>
#include <compiler/ir/transform/constant_fold.hpp>
#include <compiler/ir/transform/tensor_shrink.hpp>
#include <ops/convolution.hpp>
#include <runtime/barrier.hpp>
#include <runtime/config.hpp>
#include <runtime/dynamic_dispatch/ops/config.hpp>
#include <runtime/dynamic_dispatch/utils.hpp>
#include <unordered_set>
#include <util/any_map.hpp>
#include <util/math_utils.hpp>
#include <util/reflection.hpp>
#include <util/utils.hpp>

#include <thread>

using namespace dnnl::impl::graph::gc::builder;
namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

using ops::nested_conv_fwd_config_t;
// clang-format off
SC_CLASS(nested_conv_fwd_config_t)
  SC_FIELD(K_block)
  SC_FIELD(C_block)
  SC_FIELD(pack_input)
  SC_FIELD(bs_threads)
  SC_FIELD(oc_threads)
  SC_FIELD(im_oc_block)
  SC_FIELD(im_ic_block)
  SC_FIELD(h_threads)
  SC_FIELD(w_threads)
  SC_FIELD(h_block)
  SC_FIELD(w_block)
  SC_FIELD(im_h_block)
  SC_FIELD(im_w_block)
SC_CLASS_END();
// clang-format on

namespace ops {

static inline int get_oc_split_factor(const int data_size,
  const int weight_size, const int L2_cache_size, const int K_num_block) {
  int oc_split = 1;
  // data_size == -1 for dynamic case
  if (weight_size >= L2_cache_size
    && (weight_size > data_size || data_size == -1)) {
    int expected_split_num = utils::divide_and_ceil(weight_size, L2_cache_size);
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

config_ptr_vec gen_nested_conv_fwd_t::get_dynamic_config_candidates(
  const context_ptr &ctx) const {
  config_ptr_vec ret;
  // align with static default config
  int num_threads = runtime_config_t::get().get_num_threads();
  auto h_threads_candidates = get_splits(num_threads);
  if (h_threads_candidates.size() > 4)
    h_threads_candidates = std::vector<int>(
      {h_threads_candidates.begin(), h_threads_candidates.begin() + 4});
  auto oc_threads_candidates = std::vector<int> {1, 4, 8};
  for (auto oc_c : oc_threads_candidates) {
    int candi = std::max(num_threads / oc_c, 1);
    if (std::count(
          h_threads_candidates.begin(), h_threads_candidates.end(), candi)
      == 0)
      h_threads_candidates.push_back(candi);
  }
  std::vector<int> im_h_block_candidates = std::vector<int> {1};
  // limit im_os_block smaller than 64.
  std::vector<int> im_w_block_candidates
    = ow_ < 0 ? std::vector<int> {64} : std::vector<int> {std::min(ow_, 64)};
  bool has_pad = (pd_b_ > 0) || (ph_b_ > 0) || (pw_b_ > 0) || (pd_e_ > 0)
    || (ph_e_ > 0) || (pw_e_ > 0);

  auto default_block = get_dyn_conv_default_block(is_1x1_conv_,
    utils::get_sizeof_type(get_input_dtype()), has_pad,
    get_input_dtype() == datatypes::f32);
  int k_blk_ = utils::get_blocks(oc_, 1, default_block).back();
  if (num_threads < 5) {
    h_threads_candidates = std::vector<int> {1, num_threads};
    oc_threads_candidates = std::vector<int> {1, num_threads};
    im_h_block_candidates = std::vector<int> {1, 2, 4};
  }
  for (auto &oc_thr : oc_threads_candidates) {
    for (auto &h_thr : h_threads_candidates) {
      if (num_threads % (h_thr * oc_thr) != 0 || oc_ % oc_thr != 0) continue;
      if (oc_ / oc_thr % k_blk_ != 0) continue;
      for (auto &im_h_block : im_h_block_candidates) {
        for (auto &im_w_blk : im_w_block_candidates) {
          auto gcfg
            = reflection::general_object_t::make<nested_conv_fwd_config_t>();
          nested_conv_fwd_config_t &cfg
            = *gcfg.unchecked_get_as<nested_conv_fwd_config_t>();
          cfg.h_threads = h_thr;
          cfg.oc_threads = oc_thr;
          cfg.im_h_block = im_h_block;
          cfg.im_w_block = im_w_blk;
          ret.emplace_back(std::move(gcfg));
        }
      }
    }
  }
  return ret;
}

std::vector<uint64_t> gen_nested_conv_fwd_t::convert_config_to_keys(
  const config_ptr &config) const {
  nested_conv_fwd_config_t &cfg
    = *config.unchecked_get_as<nested_conv_fwd_config_t>();
  std::vector<uint64_t> keys = {static_cast<uint64_t>(cfg.h_threads),
    static_cast<uint64_t>(cfg.oc_threads),
    static_cast<uint64_t>(cfg.im_h_block),
    static_cast<uint64_t>(cfg.im_w_block)};
  return keys;
}

config_ptr gen_nested_conv_fwd_t::get_default_config(context_ptr ctx) const {
  auto ret = reflection::general_object_t::make<nested_conv_fwd_config_t>();
  nested_conv_fwd_config_t &cfg
    = *ret.unchecked_get_as<nested_conv_fwd_config_t>();
  if (is_dynamic()) {
    cfg.h_threads = 1;
    cfg.w_threads = 1;
    cfg.bs_threads = runtime_config_t::get().get_num_threads();
    cfg.oc_threads = 1;
    cfg.K_block = oc_;
    cfg.C_block = ic_;
    cfg.h_block = oh_;
    cfg.w_block = ow_;
    cfg.im_h_block = 1;
    cfg.im_w_block = 64;
    bool has_pad = (pd_b_ > 0) || (ph_b_ > 0) || (pw_b_ > 0) || (pd_e_ > 0)
      || (ph_e_ > 0) || (pw_e_ > 0);
    auto default_block = get_dyn_conv_default_block(is_1x1_conv_,
      utils::get_sizeof_type(get_input_dtype()), has_pad,
      get_input_dtype() == datatypes::f32);
    cfg.im_oc_block = utils::get_blocks(oc_, 1, default_block).back();
    cfg.im_ic_block = utils::get_blocks(ic_, 1, default_block).back();
    return std::move(ret);
  }
  if (use_nested_2d_) {
    const int num_threads = runtime_config_t::get().get_num_threads();
    auto thread_split = get_splits(num_threads);
    cfg.bs_threads = mb_ > num_threads || (mb_ == num_threads && oc_ <= 128)
      ? num_threads
      : *(std::find_if(thread_split.rbegin(), thread_split.rend(),
        [&](int split) { return split == 1 || split < mb_; }));
    cfg.oc_threads = num_threads / cfg.bs_threads;
    cfg.h_threads = 1;
    cfg.w_threads = 1;
    auto ic_threads = 1;

    bool is_int8
      = utils::is_one_of(get_input_dtype(), datatypes::u8, datatypes::s8);
    bool is_vnni_low_fp = ops::is_vnni_low_fp(ctx, get_input_dtype());

    auto dtype_block = is_int8 ? 4 : (is_vnni_low_fp ? 2 : 1);
    auto default_block = dtype_block * 32;

    if (mb_ == 1 && num_threads == 4) { default_block = 64; };
    bool has_pad = (pd_b_ > 0) || (ph_b_ > 0) || (pw_b_ > 0) || (pd_e_ > 0)
      || (ph_e_ > 0) || (pw_e_ > 0);
    if (has_pad) { default_block = (is_int8 || is_vnni_low_fp) ? 128 : 64; }

    cfg.im_oc_block
      = get_blocks_if_not_satisfy(oc_, 1, default_block, [](int x) {
          return x % 32 != 0;
        }).back();
    cfg.im_ic_block = ic_ <= 512
      ? ic_
      : get_blocks_if_not_satisfy(ic_, 1, default_block, [](int x) {
          return x % 32 != 0;
        }).back();

    cfg.im_h_block = 1;
    cfg.im_w_block = ow_;

    if (oh_ <= 14 && ow_ <= 14) { cfg.im_h_block = oh_; }

    cfg.h_block = oh_;
    cfg.w_block = ow_;

    if (cfg.oc_threads != 1 && !has_pad) {
      int im_oc_num_block = oc_ / cfg.im_oc_block;
      if (im_oc_num_block % cfg.oc_threads != 0) {
        auto get_suitable_block
          = [](int total, int original_block, const std::vector<int> &splits,
              int threads) {
              int suitable_block = original_block;
              for (auto split : splits) {
                int num_block = total / split;
                if (num_block % threads == 0) {
                  if ((total / suitable_block) % threads != 0
                    || std::abs(original_block - split)
                      < std::abs(original_block - suitable_block))
                    suitable_block = split;
                }
              }
              return suitable_block;
            };
        // Get a suitable im_oc_block when im_oc_num_block can't be evenly
        // distributed
        cfg.im_oc_block = get_suitable_block(
          oc_, cfg.im_oc_block, get_splits(oc_), cfg.oc_threads);
      }
    }

    if (try_os_blocking_) {
      cfg.im_w_block = get_os_blocks(ow_, adj_os_).back();
      if (ow_ > 28 && ctx->use_amx()) {
        cfg.im_w_block = utils::get_blocks(ow_, 1, 256).back();
      } else {
        auto os_blocks = get_os_blocks(ow_, adj_os_);
        if (os_blocks.back() < 400) { cfg.im_w_block = os_blocks.back(); }
      }
      bool pack_rows = (cfg.im_w_block > 0 && ow_ % cfg.im_w_block != 0);
      cfg.w_block = pack_rows ? adj_os_ : actual_os_;
      if (mb_ == 1 && num_threads == 4) {
        if (oc_ >= 256) {
          cfg.bs_threads = 1;
          cfg.h_threads = 1;
          cfg.w_threads = 1;
          cfg.oc_threads = num_threads;
        } else {
          cfg.bs_threads = 1;
          cfg.oc_threads = 1;
          cfg.h_threads = num_threads;
          cfg.w_threads = 1;
        }
        cfg.im_oc_block
          = std::min(utils::get_blocks(oc_, 1, default_block).back(),
            oc_ / cfg.oc_threads);
        // if oc threads not enough, then try to split threads on h/w
        if (cfg.oc_threads == 1) {
          pack_rows = (cfg.im_w_block > 0 && ow_ % cfg.im_w_block != 0);
          if (pack_rows) {
            // use os blocking, try to refind a appropriate os block
            cfg.w_threads = num_threads;
            cfg.h_threads = 1;
            auto os_blocks = get_os_blocks(ow_, adj_os_);
            for (int i = os_blocks.size() - 1; i >= 0; i--) {
              if (os_blocks[i] <= 256
                && os_blocks[i] <= adj_os_ / cfg.w_threads) {
                cfg.im_w_block = os_blocks[i];
                break;
              }
            }
            if (cfg.im_w_block <= ow_) {
              cfg.w_threads = 1;
              cfg.h_threads = num_threads;
            }
          } else {
            // don't use os blocking, directly split threads on h
            cfg.h_threads = num_threads;
            cfg.w_threads = 1;
          }
          cfg.oc_threads = 1;
        }
        auto real_os = pack_rows ? adj_os_ : actual_os_;
        cfg.w_block = cfg.im_w_block;
      }
      pack_rows = (cfg.im_w_block > 0 && ow_ % cfg.im_w_block != 0);
      if (!pack_rows) {
        cfg.im_h_block = 1;
        cfg.h_block = cfg.h_threads == 1
          ? oh_
          : (utils::divide_and_ceil(
               utils::divide_and_ceil(oh_, cfg.im_h_block), cfg.h_threads)
            * cfg.im_h_block);
        cfg.w_block = cfg.w_threads == 1
          ? ow_
          : (utils::divide_and_ceil(
               utils::divide_and_ceil(ow_, cfg.im_w_block), cfg.w_threads)
            * cfg.im_w_block);
      }
    } else {
      if (!is_1x1_conv_ && has_pad) {
        if (mb_ == 1 && oc_ / cfg.im_oc_block < cfg.oc_threads) {
          cfg.im_w_block = utils::get_blocks(ow_, 1, 256).back();
          if (oc_ / cfg.oc_threads > 32) {
            cfg.bs_threads = 1;
            cfg.h_threads = 1;
            cfg.w_threads = 1;
            cfg.oc_threads = num_threads;
            cfg.im_oc_block
              = std::min(utils::get_blocks(oc_, 1, default_block).back(),
                oc_ / cfg.oc_threads);
          } else {
            cfg.bs_threads = 1;
            cfg.oc_threads = 1;
            cfg.h_threads = num_threads;
            cfg.w_threads = 1;
            cfg.im_h_block = 1;
            cfg.h_block = cfg.h_threads == 1
              ? oh_
              : (utils::divide_and_ceil(
                   utils::divide_and_ceil(oh_, cfg.im_h_block), cfg.h_threads)
                * cfg.im_h_block);
          }
        }
      }
    }

    if (is_1x1_conv_) {
      if (ic_ >= 256 && oc_ >= 256 && oh_ <= 14) {
        cfg.im_h_block = oh_;
      } else {
        cfg.im_h_block = 1;
        if (oh_ >= 28 && cfg.bs_threads % 2 == 0) {
          cfg.h_threads = 2;
          cfg.bs_threads /= 2;
        }
      }
      if (mb_ == 1 && num_threads == 4) {
        cfg.im_w_block = ow_;
        if (oc_ >= 512 && ic_ >= 512 && oh_ <= 32) {
          cfg.bs_threads = 1;
          cfg.h_threads = 1;
          cfg.w_threads = 1;
          cfg.oc_threads = num_threads;
        } else {
          cfg.bs_threads = 1;
          cfg.oc_threads = 1;
          cfg.h_threads = num_threads;
          cfg.w_threads = 1;
          cfg.im_h_block = 1;
        }
      }
      auto oc_default_block = default_block;
      cfg.im_oc_block
        = std::min(get_blocks_if_not_satisfy(oc_, 1, oc_default_block,
                     [](int x) { return x % 32 != 0; })
                     .back(),
          oc_ / cfg.oc_threads);

      if (cfg.im_h_block == 1 && cfg.im_oc_block == oc_default_block
        && cfg.im_ic_block == default_block) {
        if (ow_ >= 56 && ow_ % 2 == 0) {
          cfg.im_w_block = ow_ / 2;
        } else if (sw_ == 1 && ow_ >= 28 && oc_ >= ic_ && oc_ >= 512
          && ow_ % 2 == 0) {
          cfg.im_w_block = ow_ / 2;
        } else {
          cfg.im_w_block = ow_;
        }
      }

      auto L1_cache_size = ctx->machine_.cpu_flags_.getDCacheSize(1);
      if ((cfg.im_oc_block * ic_ + cfg.im_w_block * cfg.im_h_block * ic_)
            * (4 / dtype_block)
          > static_cast<int64_t>(L1_cache_size)
        && cfg.im_oc_block > 64) {
        // adjust brgemm when brgemm is larger than L1 cache
        oc_default_block = default_block / 2;
        cfg.im_oc_block
          = std::min(get_blocks_if_not_satisfy(oc_, 1, oc_default_block,
                       [](int x) { return x % 32 != 0; })
                       .back(),
            oc_ / cfg.oc_threads);
      }

      cfg.h_block = cfg.h_threads == 1
        ? oh_
        : (utils::divide_and_ceil(
             utils::divide_and_ceil(oh_, cfg.im_h_block), cfg.h_threads)
          * cfg.im_h_block);
    }

    if (!is_1x1_conv_ && oc_ > 128 && cfg.im_oc_block % 32 != 0) {
      cfg.im_oc_block = utils::rnd_up(cfg.im_oc_block, 32);
    }
    cfg.K_block = utils::divide_and_ceil(
                    utils::divide_and_ceil(
                      utils::rnd_up(oc_, cfg.im_oc_block), cfg.im_oc_block),
                    cfg.oc_threads)
      * cfg.im_oc_block;
    if (utils::rnd_up(oc_, cfg.im_oc_block) % cfg.K_block != 0) {
      cfg.K_block = cfg.im_oc_block;
    }

    if (!is_1x1_conv_ && ic_ > 32 && cfg.im_ic_block % 32 != 0) {
      // The performance is bad when ic_block % 32 != 0. The performance of ic =
      // 56 is worse than ic = 64(in total execution time).
      cfg.im_ic_block = utils::rnd_up(ic_ <= 512 ? ic_ : cfg.im_ic_block, 32);
    }

    cfg.C_block = utils::divide_and_ceil(
                    utils::divide_and_ceil(
                      utils::rnd_up(ic_, cfg.im_ic_block), cfg.im_ic_block),
                    ic_threads)
      * cfg.im_ic_block;
    if (utils::rnd_up(ic_, cfg.im_ic_block) % cfg.C_block != 0) {
      cfg.C_block = cfg.im_ic_block;
    }
  }
  if (use_conv1d) {
    const int num_threads = runtime_config_t::get().get_num_threads();
    auto thread_split = get_splits(num_threads);
    auto im_s_block = get_im_w_block(ctx);
    auto im_oc_block = get_im_oc_block(ctx);
    auto im_ic_block = get_im_ic_block(ctx);
    auto closest_split = [](int x, std::vector<int> splits) {
      int close_num = splits[0];
      for (auto split : splits) {
        if (x - split < x - close_num && x > split) { close_num = split; }
      }
      return close_num;
    };
    cfg.bs_threads
      = mb_ >= num_threads ? num_threads : closest_split(mb_, thread_split);
    cfg.w_threads = num_threads / cfg.bs_threads;
    cfg.oc_threads = 1;
    auto s_max_task_num = ow_ / im_s_block;
    if (mb_ == 1 && s_max_task_num < num_threads) {
      auto oc_max_task_num = oc_ / im_oc_block;
      if (oc_max_task_num == num_threads || oc_max_task_num % num_threads == 0
        || oc_max_task_num > num_threads * 8) {
        cfg.bs_threads = 1;
        cfg.oc_threads = num_threads;
        cfg.w_threads = 1;
      } else if (oc_ < 1024 && oh_ * ow_ <= 28 * 28 && num_threads % 2 == 0) {
        cfg.bs_threads = 1;
        cfg.oc_threads = num_threads / 2;
        cfg.w_threads = num_threads / 2;
      } else {
        cfg.bs_threads = 1;
        cfg.oc_threads = 1;
        cfg.w_threads = num_threads;
      }
    }
    auto ic_threads = 1;
    cfg.w_block
      = utils::divide_and_ceil(
          utils::divide_and_ceil(oh_ * ow_, im_s_block), cfg.w_threads)
      * im_s_block;
    cfg.K_block = utils::divide_and_ceil(
                    utils::divide_and_ceil(oc_, im_oc_block), cfg.oc_threads)
      * im_oc_block;
    cfg.C_block = utils::divide_and_ceil(
                    utils::divide_and_ceil(ic_, im_ic_block), ic_threads)
      * im_ic_block;
  }
  return std::move(ret);
}

int gen_nested_conv_fwd_t::get_im_w_block(const context_ptr &ctx) const {
  auto ret = default_im_block_;
  auto origin_ow = dim2unsigned(attrs_.get_or_else("origin_ow", sc_dim(ow_)));
  auto origin_oh = dim2unsigned(attrs_.get_or_else("origin_oh", sc_dim(1)));
  bool is_large_spatial = origin_ow > 64;
  if (is_large_spatial) { return utils::get_blocks(origin_ow, 1, 32).back(); }
  auto s_default_block = default_im_block_;
  if (origin_ow > 14) {
    auto L1_cache_size = ctx->machine_.cpu_flags_.getDCacheSize(1);
    // not use L1_cache too full
    s_default_block = L1_cache_size / 4 / get_im_oc_block(ctx);
  }
  auto s_block_list = utils::get_blocks(ow_, 1, s_default_block);
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

int gen_nested_conv_fwd_t::get_im_oc_block(const context_ptr &ctx) const {
  return im_oc_block_;
}

int gen_nested_conv_fwd_t::get_im_ic_block(const context_ptr &ctx) const {
  return im_ic_block_;
}

gen_nested_conv_fwd_t::gen_nested_conv_fwd_t(sc_op *owner,
  const sc_dims &stride, const sc_dims &dilation, const sc_dims &pads_begin,
  const sc_dims &pads_end, std::vector<logical_tensor_t> &&ins,
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
  ih_ = is_1d_ ? 1 : input_plain_dims[ndims_ - 2];
  iw_ = input_plain_dims[ndims_ - 1];
  oc_ = weight_plain_dims[0] / groups_;
  kd_ = is_3d_ ? weight_plain_dims[2] : 1;
  kh_ = is_1d_ ? 1 : weight_plain_dims[ndims_ - 2];
  kw_ = weight_plain_dims[ndims_ - 1];
  od_ = is_3d_ ? out_plain_dims[2] : 1;
  oh_ = is_1d_ ? 1 : out_plain_dims[ndims_ - 2];
  ow_ = out_plain_dims[ndims_ - 1];
  is_1x1_conv_ = (kd_ == 1 && kh_ == 1 && kw_ == 1);
  pd_b_ = is_3d_ ? pads_begin[0] : 0;
  ph_b_ = is_1d_ ? 0 : pads_begin[0], pw_b_ = pads_begin[0];
  pd_e_ = is_3d_ ? pads_end[0] : 0;
  ph_e_ = is_1d_ ? 0 : pads_end[0], pw_e_ = pads_end[0];
  if (owner) { attrs_ = owner->attrs_; }
  bool is_bf16 = get_input_dtype() == datatypes::bf16;
  bool is_int8
    = utils::is_one_of(get_input_dtype(), datatypes::u8, datatypes::s8);
  if (is_1d_) {
    auto dtype_block = is_int8 ? 4 : (is_bf16 ? 2 : 1);
    const int num_threads = runtime_config_t::get().get_num_threads();
    default_im_block_ = dtype_block * 64;
    if (ic_ * oc_ < 512 * 512) { default_im_block_ /= 2; }
    if (mb_ == 1 && num_threads == 4) { default_im_block_ = 64; }
    bool is_small_oc_with_enough_parallel
      = oc_ < 256 && is_parallel_space_enough(mb_ * ow_ / 32, num_threads);
    im_oc_block_ = utils::get_blocks(
      oc_, 1, is_small_oc_with_enough_parallel ? oc_ : default_im_block_)
                     .back();
    im_ic_block_ = utils::get_blocks(ic_, 1, default_im_block_).back();
    im_w_block_ = utils::get_blocks(ow_ * oh_, 1, default_im_block_).back();
  }

  if (pads_begin.size() > 1) {
    ph_b_ = pads_begin[ndims_ - 4];
    pw_b_ = pads_begin[ndims_ - 3];
  }
  if (pads_end.size() > 1) {
    ph_e_ = pads_end[ndims_ - 4];
    pw_e_ = pads_end[ndims_ - 3];
  }
  sd_ = is_3d_ ? stride[0] : 1;
  sh_ = is_1d_ ? 1 : stride[0], sw_ = stride[0];
  if (stride.size() > 1) {
    auto stride_size = stride.size();
    sh_ = stride[stride_size - 2];
    sw_ = stride[stride_size - 1];
  }

  dd_ = is_3d_ ? dilation[0] : 1;
  dh_ = is_1d_ ? 1 : dilation[0], dw_ = dilation[0];
  if (dilation.size() > 1) {
    auto dilation_size = dilation.size();
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
  try_os_blocking_ = (!is_1x1_conv_) && (!has_pad) && (!is_3d_)
    && (is_int8 || is_bf16) && !is_dynamic() && sh_ == 1;
  use_nested_2d_ = (!is_1d_ && !is_3d_);
  if (is_1d_) {
    use_conv1d = true;
    COMPILE_ASSERT((kw_ == 1 && pw_b_ == 0 && pw_e_ == 0),
      "Conv1d doesn't support padding and kernel size except 1x1.");
  }
  COMPILE_ASSERT(use_nested_2d_ || use_conv1d,
    "expect input is 2D in nested conv2d, but got " << ndims_ - 2 << "D input");
}

float gen_nested_conv_fwd_t::get_gflop() const {
  float result = (float)mb_ * groups_ * oc_ * 2.0 * ic_ * kd_ * kh_ * kw_ * od_
    * oh_ * ow_ / (float)1e9;
  return result;
}

void gen_nested_conv_fwd_t::generate_brgemm(const expr &im_s_block,
  int im_ic_block, int im_oc_block, int ic_block, const expr &o_ic,
  int ic_num_block_pt, const expr &A_list, const expr &B_list,
  const expr &out_tensor, const expr &LDA, const expr &LDC) const {
  brgemm::attrs_setting_t::attrs_map_t range_attr_0
    = {brgemm::attr_key::M_range_upper_bound, 64};
  sc_brgemm_attrs_t brg_attrs = sc_brgemm_attrs_t {
    {brgemm::attr_key::max_bs, kh_ * kw_ * ic_block / im_ic_block},
    {brgemm::attr_key::use_interleave_stores, true},
    {brgemm::attr_key::use_uker, true}, range_attr_0};

  if (ic_num_block_pt > 1) {
    _if_(o_ic == 0) {
      gc::builtin::brgemm_init_list_update(A_list, B_list, out_tensor, 1,
        im_s_block, im_oc_block, im_ic_block, LDA, im_oc_block, LDC,
        1 /*useless*/
        ,
        1 /*useless*/
        ,
        kh_ * kw_ * ic_block / im_ic_block, get_input_dtype(),
        get_weight_dtype(), brg_attrs);
    }
    _else_ {
      gc::builtin::brgemm_list_update(A_list, B_list, out_tensor, 1, im_s_block,
        im_oc_block, im_ic_block, LDA, im_oc_block, LDC, 1 /*useless*/
        ,
        1 /*useless*/
        ,
        kh_ * kw_ * ic_block / im_ic_block, get_input_dtype(),
        get_weight_dtype(), brg_attrs);
    }
  } else {
    gc::builtin::brgemm_init_list_update(A_list, B_list, out_tensor, 1,
      im_s_block, im_oc_block, im_ic_block, LDA, im_oc_block, LDC, 1 /*useless*/
      ,
      1 /*useless*/
      ,
      kh_ * kw_ * ic_block / im_ic_block, get_input_dtype(), get_weight_dtype(),
      brg_attrs);
  }
}

#define CONV_ARG_LIST \
  const context_ptr &ctx, const nested_conv_fwd_config_t &config, \
    fusion_manager *fusion, expr &output, const expr &input, \
    const expr &weight, std::vector<for_loop> &loops, const int os, \
    const int kpack, const bool use_os_blocking, const bool pack_rows, \
    const expr &os_acc_size, const std::vector<char> &os_mask

void gen_nested_conv_fwd_t::compute_conv1d(CONV_ARG_LIST) const {
  // TODO(zhicong):
  // 1. support blocking layout
  // 2. provide better support in scc bench
  // 3. add iterated anchor

  // std::max avoid grid tuning generate bad config
  int num_threads = runtime_config_t::get().get_num_threads();
  int bs_threads = std::max(1, config.bs_threads);
  int s_threads
    = std::max(1, std::min(num_threads / bs_threads, config.w_threads));
  int oc_threads = std::max(1, num_threads / bs_threads / s_threads);
  int ic_threads = 1;
  int oc_block = config.K_block;
  int s_block = config.w_block;
  int ic_block = config.C_block;
  int im_oc_block = get_im_oc_block(ctx);
  int im_ic_block = get_im_ic_block(ctx);
  int im_s_block = get_im_w_block(ctx);
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

  auto origin_ow = dim2unsigned(attrs_.get_or_else("origin_ow", sc_dim(ow_)));
  auto origin_oh = dim2unsigned(attrs_.get_or_else("origin_oh", sc_dim(oh_)));
  auto infer_input_idx = [&](std::vector<expr> output_idx) {
    std::vector<expr> input_idx = output_idx;
    auto origin_iw = dim2unsigned(attrs_.get_or_else("origin_iw", sc_dim(iw_)));
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
    _named_for_(lps, ps, 0, os_used_threads, 1) {
      _named_for_(lpoc, poc, 0, oc_used_threads, 1) {
        _named_for_(lpic, pic, 0, ic_used_threads, 1) {
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
          // single core
          expr ic_num_block = builder::make_select(
            pic < (ic_used_threads - 1), ic_num_block_pt, ic_tail_num_block_pt);
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
                        _tensor_(
                          A_list, datatypes::pointer, {ic_block / im_ic_block});
                        _tensor_(
                          B_list, datatypes::pointer, {ic_block / im_ic_block});

                        expr oc = poc * oc_num_block_pt * oc_block / im_oc_block
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
                            {brgemm::attr_key::max_bs, ic_block / im_ic_block},
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
                              builtin::brgemm_init_list_update(A_list, B_list,
                                tensor_ptr(output_tmp, output_pos), 1,
                                im_s_block, im_oc_block, im_ic_block, LDA, LDB,
                                LDC, stride_a, stride_b, stride_c,
                                get_input_dtype(), get_weight_dtype(),
                                brg_attrs);
                            }
                            _else_ {
                              builtin::brgemm_list_update(A_list, B_list,
                                tensor_ptr(output_tmp, output_pos), 1,
                                im_s_block, im_oc_block, im_ic_block, LDA, LDB,
                                LDC, stride_a, stride_b, stride_c,
                                get_input_dtype(), get_weight_dtype(),
                                brg_attrs);
                            }
                          } else {
                            builtin::brgemm_init_list_update(A_list, B_list,
                              tensor_ptr(output_tmp, output_pos), 1, im_s_block,
                              im_oc_block, im_ic_block, LDA, LDB, LDC, stride_a,
                              stride_b, stride_c, get_input_dtype(),
                              get_weight_dtype(), brg_attrs);
                          }
                          if (fusion && ic_used_threads == 1
                            && ic_num_block_pt == 1) {
                            _if_(o_ic == (ic_num_block - 1)) {
                              fusion->create_output_fusion_anchor(
                                {tensor_slice(output,
                                  std::vector<std::pair<expr, expr>> {{n, 1UL},
                                    {s, im_s_block},
                                    {oc * im_oc_block, im_oc_block}})});
                            }
                          }
                        }
                      }
                      if (fusion && ic_used_threads == 1 && ic_num_block_pt == 1
                        && oc_block * oc_used_threads == oc_) {
                        _if_(o_ic == (ic_num_block - 1)) {
                          fusion->create_output_fusion_anchor(
                            {tensor_slice(output,
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
                    && s_block * os_used_threads == os_
                    && s_block % (origin_oh * origin_ow) == 0) {
                    _if_(o_ic == (ic_num_block - 1)) {
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

void gen_nested_conv_fwd_t::compute_1x1_pack_input_nested(CONV_ARG_LIST) const {
  COMPILE_ASSERT(!is_3d_, "1x1 pack input doens't support 3D conv yet!");
  tensor input1;
  int lanes = get_lanes(ctx, config.im_ic_block, get_input_dtype());
  auto toutput = out_tensors_[0];
  auto out_fmt = toutput.get_format();
  auto oh_expr_ = oh_;
  if (!out_fmt.is_any()) {
    auto out_p2b_map = out_fmt.format_code_.collect_p2b_mapping();
    oh_expr_ = static_cast<int>(get_expr_as_int(
      output.checked_as<tensor>()->dims_[out_p2b_map[is_3d_ ? 3 : 2][0]]));
  }
  if (config.pack_input == 1 && (sd_ > 1 || sh_ > 1 || sw_ > 1)) {
    for_loop ln, lk, ld, lp;
    auto mb_expr = input.checked_as<tensor>()->dims_[0];
    if (blocking_input_) {
      // NCHWc
      auto im_c_num_block = ic_ / config.im_ic_block;
      _tensor_(input_tmp, get_input_dtype(),
        {mb_expr, im_c_num_block, oh_expr_, ow_, config.im_ic_block});
      _named_for_(ln, n, 0, mb_expr, 1, for_type::PARALLEL) {
        _named_for_(lk, c_o, 0, im_c_num_block) {
          _named_for_(lp, p, 0, oh_expr_) {
            _for_(q, 0, ow_) {
              _for_(c_i, 0, config.im_ic_block, (int)lanes) {
                input_tmp[span_t({n, c_o, p, q, c_i}, lanes)]
                  = input[span_t({n, c_o, p * sh_, q * sw_, c_i}, lanes)];
              }
            }
          }
        }
      }
      auto lnk = ln->fuse(lk);
      if (im_c_num_block * mb_
        < runtime_config_t::get().get_num_threads() * 2) {
        auto lnkp = lnk->fuse(lp);
      }
      input1 = input_tmp.static_as<tensor>();
    } else {
      _tensor_(input_tmp, get_input_dtype(), {mb_expr, oh_expr_, ow_, ic_});
      _named_for_(ln, n, 0, mb_expr, 1, for_type::PARALLEL) {
        _named_for_(lp, p, 0, oh_expr_) {
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

  int num_threads = runtime_config_t::get().get_num_threads();
  int bs_threads = config.bs_threads;
  int h_threads = config.h_threads;
  int w_threads = config.w_threads;
  int oc_threads = config.oc_threads;
  int ic_threads = 1;

  int oc_block = config.K_block;
  int h_block = config.h_block;
  int w_block = config.w_block;
  int ic_block = config.C_block;
  int im_oc_block = config.im_oc_block;
  int im_ic_block = config.im_ic_block;
  int im_h_block = config.im_h_block;
  int im_w_block = config.im_w_block;

  COMPILE_ASSERT(oc_block % im_oc_block == 0,
    "oc_block % im_oc_block != 0, config is invalid")

  COMPILE_ASSERT(ic_block % im_ic_block == 0,
    "ic_block % im_ic_block != 0, config is invalid")

  COMPILE_ASSERT(
    h_block % im_h_block == 0, "h_block % im_h_block != 0, config is invalid")

  COMPILE_ASSERT(
    w_block % im_w_block == 0, "w_block % im_w_block != 0, config is invalid")

  COMPILE_ASSERT(
    w_block % im_w_block == 0, "w_block % im_w_block != 0, config is invalid")

  COMPILE_ASSERT((im_w_block == ow_ || im_h_block == 1),
    "im_w_block or im_h_block config is invalid")

  // param
  expr output_tmp = output;
  auto tinput = in_tensors_[0];
  auto tweight = in_tensors_[1];
  const auto &input_blocking_dims = tinput.get_blocking_dims();
  const auto &weight_blocking_dims = tweight.get_blocking_dims();
  const auto &output_blocking_dims = toutput.get_blocking_dims();

  for_loop lpbs, lph, lpw, lpoc, lpic, loh, low, looc, loic, lioc, lih, liw;

  int oc_num_block_pt, oc_tail_num_block_pt, h_num_block_pt,
    h_tail_num_block_pt, w_num_block_pt, w_tail_num_block_pt, ic_num_block_pt,
    ic_tail_num_block_pt;

  int oc_used_threads = block_split(utils::divide_and_ceil(oc_, oc_block),
    oc_threads, oc_num_block_pt, oc_tail_num_block_pt);

  int oh_used_threads = block_split(utils::divide_and_ceil(oh_, h_block),
    h_threads, h_num_block_pt, h_tail_num_block_pt);

  int ow_used_threads = block_split(utils::divide_and_ceil(ow_, w_block),
    w_threads, w_num_block_pt, w_tail_num_block_pt);

  int ic_used_threads = block_split(utils::divide_and_ceil(ic_, ic_block),
    ic_threads, ic_num_block_pt, ic_tail_num_block_pt);

  if (ic_used_threads > 1) {
    // barrier
    // output temp buffer
    auto out_dims = output_blocking_dims;
    out_dims[0] *= ic_used_threads;
    _tensor_(out_tmp, toutput.dtype_, dims_to_expr(out_dims));
    output_tmp = out_tmp;
  }

  auto input_expr_dims = input1.checked_as<tensor>()->dims_;
  auto mb_expr_ = input_expr_dims[0];

  _named_for_(lpbs, pbs, 0, mb_expr_, 1, for_type::PARALLEL) {
    _named_for_(lph, ph, 0, oh_used_threads, 1) {
      _named_for_(lpw, pw, 0, ow_used_threads, 1) {
        _named_for_(lpoc, poc, 0, oc_used_threads, 1) {
          _named_for_(lpic, pic, 0, ic_used_threads, 1) {
            expr h_num_block = builder::make_select(ph < (oh_used_threads - 1),
                   h_num_block_pt, h_tail_num_block_pt),
                 w_num_block = builder::make_select(pw < (ow_used_threads - 1),
                   w_num_block_pt, w_tail_num_block_pt),
                 oc_num_block
              = builder::make_select(poc < (oc_used_threads - 1),
                oc_num_block_pt, oc_tail_num_block_pt);
            // single core
            expr ic_num_block
              = builder::make_select(pic < (ic_used_threads - 1),
                ic_num_block_pt, ic_tail_num_block_pt);

            expr n = pbs;
            _named_for_(loh, o_h, 0, h_num_block_pt) {
              _named_for_(low, o_w, 0, w_num_block_pt) {
                _named_for_(looc, o_oc, 0, oc_num_block_pt) {
                  _named_for_(loic, o_ic, 0, ic_num_block_pt) {
                    expr cond = o_h < h_num_block && o_w < w_num_block
                      && o_oc < oc_num_block && o_ic < ic_num_block;
                    _if_(cond) {
                      _named_for_(lih, i_h, 0, h_block / im_h_block) {
                        expr h = (ph * h_num_block_pt * h_block / im_h_block
                                   + o_h * h_block / im_h_block + i_h)
                          * im_h_block;
                        _if_(h < oh_expr_) {
                          _named_for_(liw, i_w, 0, w_block / im_w_block) {
                            expr w = (pw * w_num_block_pt * w_block / im_w_block
                                       + o_w * w_block / im_w_block + i_w)
                              * im_w_block;
                            _if_(w < ow_) {
                              _named_for_(
                                lioc, i_oc, 0, oc_block / im_oc_block) {
                                _tensor_(A_list, datatypes::pointer,
                                  {ic_block / im_ic_block});
                                _tensor_(B_list, datatypes::pointer,
                                  {ic_block / im_ic_block});
                                expr oc = poc * oc_num_block_pt * oc_block
                                    / im_oc_block
                                  + o_oc * oc_block / im_oc_block + i_oc;
                                _if_(oc * im_oc_block < oc_) {
                                  _for_(i_c, 0, ic_block / im_ic_block) {
                                    expr ic = pic * ic_num_block_pt * ic_block
                                        / im_ic_block
                                      + o_ic * ic_block / im_ic_block + i_c;
                                    _if_(ic * im_ic_block < ic_) {
                                      std::vector<expr> input_pos
                                        = blocking_input_
                                        ? std::vector<expr> {n, ic, h, w, 0}
                                        : std::vector<expr> {
                                          n, h, w, ic * im_ic_block};

                                      A_list[i_c]
                                        = tensor_ptr(input1, input_pos);
                                      B_list[i_c] = tensor_ptr(weight,
                                        kpack > 1 ? std::vector<expr> {oc, ic,
                                          0, 0, 0, 0, 0}
                                                  : std::vector<expr> {
                                                    oc, ic, 0, 0, 0, 0});
                                    }
                                  }
                                  const auto hint_A_size
                                    = im_h_block * im_w_block * ic_block;
                                  const auto hint_B_size
                                    = im_oc_block * ic_block;
                                  const auto hint_C_size
                                    = im_h_block * im_w_block * im_oc_block;
                                  sc_brgemm_attrs_t brg_attrs {
                                    {brgemm::attr_key::max_bs,
                                      ic_block / im_ic_block},
                                    {brgemm::attr_key::hint_expected_A_size,
                                      hint_A_size},
                                    {brgemm::attr_key::hint_expected_B_size,
                                      hint_B_size},
                                    {brgemm::attr_key::hint_expected_C_size,
                                      hint_C_size},
                                    {brgemm::attr_key::use_interleave_stores,
                                      true},
                                    {brgemm::attr_key::use_uker, true}};

                                  auto LDA
                                    = blocking_input_ ? im_ic_block : ic_;
                                  auto LDC
                                    = blocking_output_ ? im_oc_block : oc_;

                                  std::vector<expr> output_pos
                                    = blocking_output_
                                    ? std::vector<expr> {pic * mb_ + n, oc, h,
                                      w, 0}
                                    : std::vector<expr> {
                                      pic * mb_ + n, h, w, oc * im_oc_block};

                                  if (ic_num_block_pt > 1) {
                                    _if_(o_ic == 0) {
                                      builtin::brgemm_init_list_update(A_list,
                                        B_list,
                                        tensor_ptr(output_tmp, output_pos), 1,
                                        im_h_block * im_w_block, im_oc_block,
                                        im_ic_block, LDA, im_oc_block, LDC,
                                        1 /*useless*/
                                        ,
                                        1 /*useless*/
                                        ,
                                        ic_block / im_ic_block,
                                        get_input_dtype(), get_weight_dtype(),
                                        brg_attrs);
                                    }
                                    _else_ {
                                      builtin::brgemm_list_update(A_list,
                                        B_list,
                                        tensor_ptr(output_tmp, output_pos), 1,
                                        im_h_block * im_w_block, im_oc_block,
                                        im_ic_block, LDA, im_oc_block, LDC,
                                        1 /*useless*/
                                        ,
                                        1 /*useless*/
                                        ,
                                        ic_block / im_ic_block,
                                        get_input_dtype(), get_weight_dtype(),
                                        brg_attrs);
                                    }
                                  } else {
                                    builtin::brgemm_init_list_update(A_list,
                                      B_list,
                                      tensor_ptr(output_tmp, output_pos), 1,
                                      im_h_block * im_w_block, im_oc_block,
                                      im_ic_block, LDA, im_oc_block, LDC,
                                      1 /*useless*/
                                      ,
                                      1 /*useless*/
                                      ,
                                      ic_block / im_ic_block, get_input_dtype(),
                                      get_weight_dtype(), brg_attrs);
                                  }

                                  if (fusion && ic_used_threads == 1
                                    && ic_num_block_pt == 1) {
                                    _if_(o_ic == (ic_num_block - 1)) {
                                      fusion->create_output_fusion_anchor(
                                        {blocking_output_
                                            ? tensor_slice(output,
                                              {{n, 1UL}, {oc, 1},
                                                {h, im_h_block},
                                                {w, im_w_block},
                                                {0, im_oc_block}})
                                            : tensor_slice(output,
                                              {{n, 1UL}, {h, im_h_block},
                                                {w, im_w_block},
                                                {oc * im_oc_block,
                                                  im_oc_block}})});
                                    }
                                  }
                                }
                              }
                              if (fusion && ic_used_threads == 1
                                && ic_num_block_pt == 1
                                && oc_block * oc_used_threads == oc_) {
                                _if_(o_ic == (ic_num_block - 1)) {
                                  expr anch_c = poc * oc_num_block_pt * oc_block
                                      / im_oc_block
                                    + o_oc * oc_block / im_oc_block;
                                  fusion->create_output_fusion_anchor(
                                    {blocking_output_ ? tensor_slice(output,
                                       {{n, 1UL}, {anch_c, 1}, {h, im_h_block},
                                         {w, im_w_block}, {0, im_oc_block}})
                                                      : tensor_slice(output,
                                                        {{n, 1UL},
                                                          {h, im_h_block},
                                                          {w, im_w_block},
                                                          {anch_c * im_oc_block,
                                                            oc_block}})});
                                }
                              }
                            }
                          }

                          if (fusion && ic_used_threads == 1
                            && ic_num_block_pt == 1
                            && oc_block * oc_used_threads == oc_
                            && w_block * ow_used_threads == ow_) {
                            _if_(o_ic == (ic_num_block - 1)) {
                              expr anch_c
                                = poc * oc_num_block_pt * oc_block / im_oc_block
                                + o_oc * oc_block / im_oc_block;
                              expr anch_w
                                = (pw * w_num_block_pt * w_block / im_w_block
                                    + o_w * w_block / im_w_block)
                                * im_w_block;
                              fusion->create_output_fusion_anchor(
                                {blocking_output_
                                    ? tensor_slice(output,
                                      {{n, 1UL}, {anch_c, 1}, {h, im_h_block},
                                        {anch_w, w_block}, {0, im_oc_block}})
                                    : tensor_slice(output,
                                      {{n, 1UL}, {h, im_h_block},
                                        {anch_w, w_block},
                                        {anch_c * im_oc_block, oc_block}})});
                            }
                          }
                        }
                      }

                      if (fusion && ic_used_threads == 1 && ic_num_block_pt == 1
                        && oc_block * oc_used_threads == oc_
                        && w_block * ow_used_threads == ow_
                        && h_block * oh_used_threads == oh_) {
                        _if_(o_ic == (ic_num_block - 1)) {
                          expr anch_c
                            = poc * oc_num_block_pt * oc_block / im_oc_block
                            + o_oc * oc_block / im_oc_block;
                          expr anch_h
                            = (ph * h_num_block_pt * h_block / im_h_block
                                + o_h * h_block / im_h_block)
                            * im_h_block;
                          expr anch_w
                            = (pw * w_num_block_pt * w_block / im_w_block
                                + o_w * w_block / im_w_block)
                            * im_w_block;

                          fusion->create_output_fusion_anchor({blocking_output_
                              ? tensor_slice(output,
                                {{n, 1UL}, {anch_c, 1}, {anch_h, h_block},
                                  {anch_w, w_block}, {0, im_oc_block}})
                              : tensor_slice(output,
                                {{n, 1UL}, {anch_h, h_block}, {anch_w, w_block},
                                  {anch_c * im_oc_block, oc_block}})});
                        }
                      }
                    }
                  }
                  // TODO(xurui): need to add iterated anchor here to
                  // support more fusion opportunity
                }
              }
            }

            if (fusion && oc_threads == 1 && h_threads == 1 && w_threads == 1
              && ic_threads == 1) {
              fusion->create_output_fusion_anchor({blocking_output_
                  ? tensor_slice(output,
                    {{pbs, 1UL}, {0, oc_ / im_oc_block}, {0, oh_expr_},
                      {0, ow_}, {0, im_oc_block}})
                  : tensor_slice(
                    output, {{pbs, 1UL}, {0, oh_expr_}, {0, ow_}, {0, oc_}})});
            }
          }
          if (fusion && oc_threads == 1 && h_threads == 1 && w_threads == 1) {
            fusion->create_output_fusion_anchor({blocking_output_
                ? tensor_slice(output,
                  {{pbs, 1UL}, {0, oc_ / im_oc_block}, {0, oh_expr_}, {0, ow_},
                    {0, im_oc_block}})
                : tensor_slice(
                  output, {{pbs, 1UL}, {0, oh_expr_}, {0, ow_}, {0, oc_}})});
          }
        }
        if (fusion && h_threads == 1 && w_threads == 1) {
          fusion->create_output_fusion_anchor({blocking_output_
              ? tensor_slice(output,
                {{pbs, 1UL}, {0, oc_ / im_oc_block}, {0, oh_expr_}, {0, ow_},
                  {0, im_oc_block}})
              : tensor_slice(
                output, {{pbs, 1UL}, {0, oh_expr_}, {0, ow_}, {0, oc_}})});
        }
      }

      if (fusion && h_threads == 1) {
        fusion->create_output_fusion_anchor({blocking_output_
            ? tensor_slice(output,
              {{pbs, 1UL}, {0, oc_ / im_oc_block}, {0, oh_expr_}, {0, ow_},
                {0, im_oc_block}})
            : tensor_slice(
              output, {{pbs, 1UL}, {0, oh_expr_}, {0, ow_}, {0, oc_}})});
      }
    }
    if (fusion && mb_ > 1) {
      fusion->create_output_fusion_anchor(
        {blocking_output_ ? tensor_slice(output,
           {{pbs, 1UL}, {0, oc_ / im_oc_block}, {0, oh_expr_}, {0, ow_},
             {0, im_oc_block}})
                          : tensor_slice(output,
                            {{pbs, 1UL}, {0, oh_expr_}, {0, ow_}, {0, oc_}})});
    }
  }
  loops = {lpbs, lph, lpw, lpoc, lpic};
}

void gen_nested_conv_fwd_t::dynamic_compute_1x1_pack_input_nested(
  CONV_ARG_LIST) const {
  COMPILE_ASSERT(
    !is_3d_, "dynamic 1x1 pack input doens't support 3D conv yet!");
  COMPILE_ASSERT(!blocking_input_ && !blocking_output_,
    "dynamic 1x1 pack input doens't support blocking input / output yet!");
  tensor input1;
  auto input_expr_dims = input.checked_as<tensor>()->dims_;
  auto mb_expr_ = input_expr_dims[0];
  auto ih_expr_ = input_expr_dims[1];
  auto iw_expr_ = input_expr_dims[2];
  auto oh_expr_ = input_expr_dims.size() == 4
    ? (input_expr_dims[1] + ph_b_ + ph_e_ - kh_) / sh_ + 1
    : (input_expr_dims[2] + ph_b_ + ph_e_ - kh_) / sh_ + 1;
  auto ow_expr_ = input_expr_dims.size() == 4
    ? (input_expr_dims[2] + pw_b_ + pw_e_ - kw_) / sw_ + 1
    : (input_expr_dims[3] + pw_b_ + pw_e_ - kw_) / sw_ + 1;

  int lanes = get_lanes(ctx, config.im_ic_block, get_input_dtype());
  if (config.pack_input == 1 && (sd_ > 1 || sh_ > 1 || sw_ > 1)) {
    for_loop ln, lk, ld, lp;
    _tensor_(input_tmp, get_input_dtype(), {mb_expr_, oh_expr_, ow_expr_, ic_});
    _named_for_(ln, n, 0, mb_expr_, 1, for_type::PARALLEL) {
      _named_for_(lp, p, 0, oh_expr_) {
        _for_(q, 0, ow_expr_) {
          _for_(c_i, 0, ic_, (int)lanes) {
            input_tmp[span_t({n, p, q, c_i}, lanes)]
              = input[span_t({n, p * sh_, q * sw_, c_i}, lanes)];
          }
        }
      }
    }
    ln = ln->fuse(lp);
    input1 = input_tmp.static_as<tensor>();
  } else {
    input1 = input.static_as<tensor>();
  }

  int num_threads = runtime_config_t::get().get_num_threads();
  int h_threads = config.h_threads;
  int oc_threads = config.oc_threads;
  int bs_threads = num_threads / h_threads / oc_threads;
  int w_threads = config.w_threads;
  int ic_threads = 1;
  int oc_block = oc_ / oc_threads;
  // for small spatial
  expr h_block = builder::make_select(oh_expr_ / h_threads > 0
      && oh_expr_ / h_threads * ow_expr_ <= 64
      && oh_expr_ / h_threads * h_threads == oh_expr_,
    builder::make_cast(datatypes::s32, oh_expr_ / h_threads),
    config.im_h_block);
  expr w_block = ow_expr_ / w_threads;
  int ic_block = ic_ / ic_threads;
  int im_oc_block = config.im_oc_block;
  int im_ic_block = config.im_ic_block;
  expr im_h_block = h_block;
  expr im_w_block = builder::make_select(h_block * ow_expr_ <= 64,
    builder::make_cast(datatypes::s32, ow_expr_), config.im_w_block);

  COMPILE_ASSERT(oc_block % im_oc_block == 0,
    "oc_block % im_oc_block != 0, config is invalid")
  COMPILE_ASSERT(ic_block % im_ic_block == 0,
    "ic_block % im_ic_block != 0, config is invalid")

  // param
  auto tinput = in_tensors_[0];
  auto tweight = in_tensors_[1];
  auto toutput = out_tensors_[0];
  const auto &input_blocking_dims = tinput.get_blocking_dims();
  const auto &weight_blocking_dims = tweight.get_blocking_dims();
  const auto &output_blocking_dims = toutput.get_blocking_dims();

  for_loop lpbs, lph, lpw, lpoc, lpic, loh, low, looc, loic, lioc, lis, lih,
    liw;

  int oc_num_block_pt, oc_tail_num_block_pt, ic_num_block_pt,
    ic_tail_num_block_pt;

  int oc_used_threads = block_split(utils::divide_and_ceil(oc_, oc_block),
    oc_threads, oc_num_block_pt, oc_tail_num_block_pt);

  int ic_used_threads = block_split(utils::divide_and_ceil(ic_, ic_block),
    ic_threads, ic_num_block_pt, ic_tail_num_block_pt);

  expr h_num_block_pt
    = divide_and_ceil(divide_and_ceil(oh_expr_, h_block), h_threads);
  expr h_block_tail_pt = builder::make_select(
    divide_and_ceil(oh_expr_, h_block) % h_num_block_pt == 0, h_num_block_pt,
    divide_and_ceil(oh_expr_, h_block) % h_num_block_pt);
  expr oh_used_threads
    = divide_and_ceil(divide_and_ceil(oh_expr_, h_block), h_num_block_pt);
  expr w_num_block_pt = 1;

  _named_for_(lpbs, pbs, 0, mb_expr_, 1, for_type::PARALLEL, bs_threads) {
    _named_for_(lph, ph, 0, h_threads, 1, for_type::PARALLEL, h_threads) {
      _named_for_(lpw, pw, 0, w_threads, 1, for_type::PARALLEL, w_threads) {
        _named_for_(
          lpoc, poc, 0, oc_threads, 1, for_type::PARALLEL, oc_threads) {
          _named_for_(
            lpic, pic, 0, ic_threads, 1, for_type::PARALLEL, ic_threads) {
            _if_(ph < oh_used_threads && pw < w_threads && poc < oc_used_threads
              && pic < ic_used_threads) {
              // single core
              expr oc_num_block
                = builder::make_select(poc < (oc_used_threads - 1),
                  oc_num_block_pt, oc_tail_num_block_pt);
              expr ic_num_block
                = builder::make_select(pic < (ic_used_threads - 1),
                  ic_num_block_pt, ic_tail_num_block_pt);
              expr h_num_block = builder::make_select(
                ph < (oh_used_threads - 1), h_num_block_pt, h_block_tail_pt);
              expr n = pbs;
              _named_for_(loh, o_h, 0, h_num_block_pt) {
                _named_for_(low, o_w, 0, w_num_block_pt) {
                  _named_for_(looc, o_oc, 0, oc_num_block_pt) {
                    _named_for_(loic, o_ic, 0, ic_num_block_pt) {
                      expr cond = o_h < h_num_block && o_w < w_num_block_pt
                        && o_oc < oc_num_block && o_ic < ic_num_block;
                      // innermost loop
                      _if_(cond) {
                        _named_for_(lioc, i_oc, 0, oc_block / im_oc_block) {
                          _tensor_(A_list, datatypes::pointer,
                            {ic_block / im_ic_block});
                          _tensor_(B_list, datatypes::pointer,
                            {ic_block / im_ic_block});
                          expr oc
                            = poc * oc_num_block_pt * oc_block / im_oc_block
                            + o_oc * oc_block / im_oc_block + i_oc;
                          _if_(oc * im_oc_block < oc_) {
                            _named_for_(lih, i_h, 0,
                              divide_and_ceil(h_block, im_h_block)) {
                              expr h = ph * h_num_block_pt * h_block
                                + o_h * h_block + i_h * im_h_block;
                              _if_(h < oh_expr_) {
                                _named_for_(liw, i_w, 0,
                                  divide_and_ceil(w_block, im_w_block)) {
                                  expr w = pw * w_num_block_pt * w_block
                                    + o_w * w_block + i_w * im_w_block;
                                  _for_(i_c, 0, ic_block / im_ic_block) {
                                    expr ic = pic * ic_num_block_pt * ic_block
                                        / im_ic_block
                                      + o_ic * ic_block / im_ic_block + i_c;
                                    _if_(ic * im_ic_block < ic_) {
                                      std::vector<expr> input_pos
                                        = std::vector<expr> {
                                          n, h, w, ic * im_ic_block};
                                      A_list[i_c]
                                        = tensor_ptr(input1, input_pos);
                                      B_list[i_c] = tensor_ptr(weight,
                                        kpack > 1 ? std::vector<expr> {oc, ic,
                                          0, 0, 0, 0, 0}
                                                  : std::vector<expr> {
                                                    oc, ic, 0, 0, 0, 0});
                                    }
                                  }
                                  auto LDA = ic_;
                                  auto LDC = oc_;

                                  std::vector<expr> output_pos
                                    = blocking_output_
                                    ? std::vector<expr> {n, oc, h, w, 0}
                                    : std::vector<expr> {
                                      n, h, w, oc * im_oc_block};
                                  auto im_w_tail_block = builder::make_cast(
                                    datatypes::s32, ow_expr_ - w);
                                  im_w_block = builder::make_select(
                                    w + im_w_block <= ow_expr_, im_w_block,
                                    im_w_tail_block);
                                  auto im_s_block = builder::make_cast(
                                    datatypes::s32, im_w_block * im_h_block);
                                  generate_brgemm(im_s_block, im_ic_block,
                                    im_oc_block, ic_block, o_ic,
                                    ic_num_block_pt, A_list, B_list,
                                    tensor_ptr(output, output_pos), LDA, LDC);
                                  if (fusion) {
                                    fusion->create_output_fusion_anchor(
                                      {tensor_slice(output,
                                        {{n, 1UL}, {h, im_h_block},
                                          {w, im_w_block},
                                          {oc * im_oc_block, im_oc_block}})},
                                      0);
                                  }
                                } // i_w
                              } // check h_boundary
                              if (fusion && oc_block * oc_used_threads == oc_) {
                                _if_(o_ic == (ic_num_block - 1)) {
                                  fusion->create_output_fusion_anchor(
                                    {tensor_slice(output,
                                      {{n, 1UL}, {h, im_h_block}, {0, ow_expr_},
                                        {oc * im_oc_block, im_oc_block}})});
                                }
                              }
                            } // i_h
                            if (fusion && oc_block * oc_used_threads == oc_) {
                              _if_(o_ic == (ic_num_block - 1)) {
                                expr anchor_h
                                  = (ph * h_num_block_pt * h_block / im_h_block
                                      + o_h * h_block / im_h_block)
                                  * im_h_block;
                                fusion->create_output_fusion_anchor(
                                  {tensor_slice(output,
                                    {{n, 1UL}, {anchor_h, h_block},
                                      {0, ow_expr_},
                                      {oc * im_oc_block, im_oc_block}})});
                              }
                            }
                          } // check i_oc
                        } // i_oc
                        if (fusion && oc_block * oc_used_threads == oc_) {
                          _if_(h_block * oh_used_threads == oh_expr_) {
                            expr anchor_h
                              = (ph * h_num_block_pt * h_block / im_h_block
                                  + o_h * h_block / im_h_block)
                              * im_h_block;
                            expr anchor_c = poc * oc_num_block_pt * oc_block
                              + o_oc * oc_block;
                            fusion->create_output_fusion_anchor(
                              {tensor_slice(output,
                                {{n, 1UL}, {anchor_h, h_block}, {0, ow_expr_},
                                  {anchor_c, oc_block}})});
                          }
                        }
                      } // check innermost
                    } // o_ic
                  } // o_oc
                } // o_w
                if (fusion && oc_block * oc_used_threads == oc_) {
                  _if_(h_block * oh_used_threads == oh_expr_) {
                    expr anchor_h
                      = ph * h_num_block_pt * h_block + o_h * h_block;
                    expr anchor_c = poc * oc_num_block_pt * oc_block;
                    fusion->create_output_fusion_anchor({tensor_slice(output,
                      {{pbs, 1UL}, {anchor_h, h_block}, {0, ow_expr_},
                        {anchor_c, oc_num_block_pt * oc_block}})});
                  }
                }
              } // o_h
            } // check single core
            if (fusion && oc_threads == 1 && h_threads == 1) {
              fusion->create_output_fusion_anchor({tensor_slice(
                output, {{pbs, 1UL}, {0, oh_expr_}, {0, ow_expr_}, {0, oc_}})});
            }
          } // pic
        } // poc
        if (fusion && h_threads == 1) {
          expr anchor_h = ph * h_num_block_pt * h_block;
          fusion->create_output_fusion_anchor({tensor_slice(
            output, {{pbs, 1UL}, {0, oh_expr_}, {0, ow_expr_}, {0, oc_}})});
        }
      } // pw
    } // ph
    // TODO(xurui) disable the anchor fow now for dynamic bottleneck.
    // if (fusion && mb_ > 1) {
    //   fusion->create_output_fusion_anchor({tensor_slice(
    //     output, {{pbs, 1UL}, {0, oh_expr_}, {0, ow_expr_}, {0, oc_}})});
    // }
  } // pbs
  loops = {lpbs, lph, lpw, lpoc, lpic};
}

void gen_nested_conv_fwd_t::compute_1x1_no_pack_input_nested(
  CONV_ARG_LIST) const {
  int bs_threads = config.bs_threads;
  int h_threads = config.h_threads;
  int w_threads = config.w_threads;
  int oc_threads = config.oc_threads;
  int ic_threads = 1;

  int oc_block = config.K_block;
  int h_block = config.h_block;
  int w_block = config.w_block;
  int ic_block = config.C_block;
  int im_oc_block = config.im_oc_block;
  int im_ic_block = config.im_ic_block;
  int im_h_block = config.im_h_block;
  int im_w_block = config.im_w_block;

  COMPILE_ASSERT(oc_block % im_oc_block == 0,
    "oc_block % im_oc_block != 0, config is invalid")

  COMPILE_ASSERT(ic_block % im_ic_block == 0,
    "ic_block % im_ic_block != 0, config is invalid")

  COMPILE_ASSERT(
    h_block % im_h_block == 0, "h_block % im_h_block != 0, config is invalid")

  COMPILE_ASSERT(
    w_block % im_w_block == 0, "w_block % im_w_block != 0, config is invalid")

  COMPILE_ASSERT(
    w_block % im_w_block == 0, "w_block % im_w_block != 0, config is invalid")

  // param
  expr output_tmp = output;
  auto tinput = in_tensors_[0];
  auto tweight = in_tensors_[1];
  auto toutput = out_tensors_[0];
  const auto &input_blocking_dims = tinput.get_blocking_dims();
  const auto &weight_blocking_dims = tweight.get_blocking_dims();
  const auto &output_blocking_dims = toutput.get_blocking_dims();
  auto out_fmt = toutput.get_format();
  auto oh_expr_ = oh_;
  if (!out_fmt.is_any()) {
    auto out_p2b_map = out_fmt.format_code_.collect_p2b_mapping();
    oh_expr_ = static_cast<int>(get_expr_as_int(
      output.checked_as<tensor>()->dims_[out_p2b_map[is_3d_ ? 3 : 2][0]]));
  }

  for_loop lpbs, lph, lpw, lpoc, lpic, loh, low, looc, loic, lioc, lih, liw;

  int oc_num_block_pt, oc_tail_num_block_pt, h_num_block_pt,
    h_tail_num_block_pt, w_num_block_pt, w_tail_num_block_pt, ic_num_block_pt,
    ic_tail_num_block_pt;

  int oc_used_threads = block_split(utils::divide_and_ceil(oc_, oc_block),
    oc_threads, oc_num_block_pt, oc_tail_num_block_pt);

  int oh_used_threads = block_split(utils::divide_and_ceil(oh_, h_block),
    h_threads, h_num_block_pt, h_tail_num_block_pt);

  int ow_used_threads = block_split(utils::divide_and_ceil(ow_, w_block),
    w_threads, w_num_block_pt, w_tail_num_block_pt);

  int ic_used_threads = block_split(utils::divide_and_ceil(ic_, ic_block),
    ic_threads, ic_num_block_pt, ic_tail_num_block_pt);

  if (ic_used_threads > 1) {
    // barrier
    // output temp buffer
    auto out_dims = output_blocking_dims;
    out_dims[0] *= ic_used_threads;
    _tensor_(out_tmp, toutput.dtype_, dims_to_expr(out_dims));
    output_tmp = out_tmp;
  }

  auto input_expr_dims = input.checked_as<tensor>()->dims_;
  auto mb_expr_ = input_expr_dims[0];

  _named_for_(lpbs, pbs, 0, mb_expr_, 1, for_type::PARALLEL) {
    _named_for_(lph, ph, 0, oh_used_threads, 1) {
      _named_for_(lpw, pw, 0, ow_used_threads, 1) {
        _named_for_(lpoc, poc, 0, oc_used_threads, 1) {
          _named_for_(lpic, pic, 0, ic_used_threads, 1) {
            expr h_num_block = builder::make_select(ph < (oh_used_threads - 1),
                   h_num_block_pt, h_tail_num_block_pt),

                 w_num_block = builder::make_select(pw < (ow_used_threads - 1),
                   w_num_block_pt, w_tail_num_block_pt),

                 oc_num_block
              = builder::make_select(poc < (oc_used_threads - 1),
                oc_num_block_pt, oc_tail_num_block_pt);

            // single core
            expr ic_num_block
              = builder::make_select(pic < (ic_used_threads - 1),
                ic_num_block_pt, ic_tail_num_block_pt);

            expr n = pbs;
            _named_for_(loh, o_h, 0, h_num_block_pt) {
              _named_for_(low, o_w, 0, w_num_block_pt) {
                _named_for_(looc, o_oc, 0, oc_num_block_pt) {
                  _named_for_(loic, o_ic, 0, ic_num_block_pt) {
                    expr cond = o_h < h_num_block && o_w < w_num_block
                      && o_oc < oc_num_block && o_ic < ic_num_block;
                    _if_(cond) {
                      _named_for_(lih, i_h, 0, h_block / im_h_block) {
                        expr h = (ph * h_num_block_pt * h_block / im_h_block
                                   + o_h * h_block / im_h_block + i_h)
                          * im_h_block;
                        _named_for_(liw, i_w, 0, w_block / im_w_block) {
                          expr w = (pw * w_num_block_pt * w_block / im_w_block
                                     + o_w * w_block / im_w_block + i_w)
                            * im_w_block;
                          _if_(w < ow_) {
                            _named_for_(lioc, i_oc, 0, oc_block / im_oc_block) {
                              expr oc
                                = poc * oc_num_block_pt * oc_block / im_oc_block
                                + o_oc * oc_block / im_oc_block + i_oc;
                              _if_(oc * im_oc_block < oc_) {
                                _for_(im_h_i, 0, im_h_block) {
                                  _if_(h + im_h_i < oh_expr_) {
                                    _tensor_(A_list, datatypes::pointer,
                                      {ic_block / im_ic_block});
                                    _tensor_(B_list, datatypes::pointer,
                                      {ic_block / im_ic_block});

                                    _for_(i_c, 0, ic_block / im_ic_block) {
                                      expr ic = pic * ic_num_block_pt * ic_block
                                          / im_ic_block
                                        + o_ic * ic_block / im_ic_block + i_c;
                                      _if_(ic * im_ic_block < ic_) {
                                        std::vector<expr> input_pos
                                          = blocking_input_
                                          ? std::vector<expr> {n, ic,
                                            (h + im_h_i) * sh_, w * sw_, 0}
                                          : std::vector<expr> {n,
                                            (h + im_h_i) * sh_, w * sw_,
                                            ic * im_ic_block};

                                        A_list[i_c]
                                          = tensor_ptr(input, input_pos);
                                        B_list[i_c] = tensor_ptr(weight,
                                          kpack > 1 ? std::vector<expr> {oc, ic,
                                            0, 0, 0, 0, 0}
                                                    : std::vector<expr> {
                                                      oc, ic, 0, 0, 0, 0});
                                      }
                                    }
                                    const auto hint_A_size
                                      = im_w_block * ic_block;
                                    const auto hint_B_size
                                      = im_oc_block * ic_block;
                                    const auto hint_C_size
                                      = im_w_block * im_oc_block;

                                    sc_brgemm_attrs_t brg_attrs {
                                      {brgemm::attr_key::max_bs,
                                        ic_block / im_ic_block},
                                      {brgemm::attr_key::hint_expected_A_size,
                                        hint_A_size},
                                      {brgemm::attr_key::hint_expected_B_size,
                                        hint_B_size},
                                      {brgemm::attr_key::hint_expected_C_size,
                                        hint_C_size},
                                      {brgemm::attr_key::use_interleave_stores,
                                        true},
                                      {brgemm::attr_key::use_uker, true}};

                                    auto LDA = blocking_input_
                                      ? sw_ * im_ic_block
                                      : sw_ * ic_;
                                    auto LDC
                                      = blocking_output_ ? im_oc_block : oc_;

                                    std::vector<expr> output_pos
                                      = blocking_output_
                                      ? std::vector<expr> {pic * mb_ + n, oc,
                                        h + im_h_i, w, 0}
                                      : std::vector<expr> {pic * mb_ + n,
                                        h + im_h_i, w, oc * im_oc_block};

                                    if (ic_num_block_pt > 1) {
                                      _if_(o_ic == 0) {
                                        builtin::brgemm_init_list_update(A_list,
                                          B_list,
                                          tensor_ptr(output_tmp, output_pos), 1,
                                          im_w_block, im_oc_block, im_ic_block,
                                          LDA, im_oc_block, LDC, 1 /*useless*/
                                          ,
                                          1 /*useless*/
                                          ,
                                          ic_block / im_ic_block,
                                          get_input_dtype(), get_weight_dtype(),
                                          brg_attrs);
                                      }
                                      _else_ {
                                        builtin::brgemm_list_update(A_list,
                                          B_list,
                                          tensor_ptr(output_tmp, output_pos), 1,
                                          im_w_block, im_oc_block, im_ic_block,
                                          LDA, im_oc_block, LDC, 1 /*useless*/
                                          ,
                                          1 /*useless*/
                                          ,
                                          ic_block / im_ic_block,
                                          get_input_dtype(), get_weight_dtype(),
                                          brg_attrs);
                                      }
                                    } else {
                                      builtin::brgemm_init_list_update(A_list,
                                        B_list,
                                        tensor_ptr(output_tmp, output_pos), 1,
                                        im_w_block, im_oc_block, im_ic_block,
                                        LDA, im_oc_block, LDC, 1 /*useless*/
                                        ,
                                        1 /*useless*/
                                        ,
                                        ic_block / im_ic_block,
                                        get_input_dtype(), get_weight_dtype(),
                                        brg_attrs);
                                    }

                                    if (fusion && ic_used_threads == 1
                                      && ic_num_block_pt == 1) {
                                      _if_(o_ic == (ic_num_block - 1)) {
                                        fusion->create_output_fusion_anchor(
                                          {blocking_output_
                                              ? tensor_slice(output,
                                                {{n, 1UL}, {oc, 1},
                                                  {h + im_h_i, 1},
                                                  {w, im_w_block},
                                                  {0, im_oc_block}})
                                              : tensor_slice(output,
                                                {{n, 1UL}, {h + im_h_i, 1},
                                                  {w, im_w_block},
                                                  {oc * im_oc_block,
                                                    im_oc_block}})});
                                      }
                                    }
                                  }
                                }

                                if (fusion && ic_used_threads == 1) {
                                  _if_(o_ic == (ic_num_block - 1)) {
                                    fusion->create_output_fusion_anchor(
                                      {blocking_output_ ? tensor_slice(output,
                                         {{n, 1UL}, {oc, 1}, {h, im_h_block},
                                           {w, im_w_block}, {0, im_oc_block}})
                                                        : tensor_slice(output,
                                                          {{n, 1UL},
                                                            {h, im_h_block},
                                                            {w, im_w_block},
                                                            {oc * im_oc_block,
                                                              im_oc_block}})});
                                  }
                                }
                              }
                            }
                            if (fusion && ic_used_threads == 1
                              && ic_num_block_pt == 1
                              && oc_block * oc_used_threads == oc_) {
                              _if_(o_ic == (ic_num_block - 1)) {
                                expr anch_c = poc * oc_num_block_pt * oc_block
                                    / im_oc_block
                                  + o_oc * oc_block / im_oc_block;
                                fusion->create_output_fusion_anchor(
                                  {blocking_output_
                                      ? tensor_slice(output,
                                        {{n, 1UL}, {anch_c, 1}, {h, im_h_block},
                                          {w, im_w_block}, {0, im_oc_block}})
                                      : tensor_slice(output,
                                        {{n, 1UL}, {h, im_h_block},
                                          {w, im_w_block},
                                          {anch_c * im_oc_block, oc_block}})});
                              }
                            }
                          }
                        }

                        if (fusion && ic_used_threads == 1
                          && ic_num_block_pt == 1
                          && oc_block * oc_used_threads == oc_
                          && w_block * ow_used_threads == ow_) {
                          _if_(o_ic == (ic_num_block - 1)) {
                            expr anch_c
                              = poc * oc_num_block_pt * oc_block / im_oc_block
                              + o_oc * oc_block / im_oc_block;
                            expr anch_w
                              = (pw * w_num_block_pt * w_block / im_w_block
                                  + o_w * w_block / im_w_block)
                              * im_w_block;
                            fusion->create_output_fusion_anchor(
                              {blocking_output_
                                  ? tensor_slice(output,
                                    {{n, 1UL}, {anch_c, 1}, {h, im_h_block},
                                      {anch_w, w_block}, {0, im_oc_block}})
                                  : tensor_slice(output,
                                    {{n, 1UL}, {h, im_h_block},
                                      {anch_w, w_block},
                                      {anch_c * im_oc_block, oc_block}})});
                          }
                        }
                      }

                      if (fusion && ic_used_threads == 1 && ic_num_block_pt == 1
                        && oc_block * oc_used_threads == oc_
                        && w_block * ow_used_threads == ow_
                        && h_block * oh_used_threads == oh_) {
                        _if_(o_ic == (ic_num_block - 1)) {
                          expr anch_c
                            = (poc * oc_num_block_pt * oc_block / im_oc_block
                              + o_oc * oc_block / im_oc_block);
                          expr anch_h
                            = (ph * h_num_block_pt * h_block / im_h_block
                                + o_h * h_block / im_h_block)
                            * im_h_block;
                          expr anch_w
                            = (pw * w_num_block_pt * w_block / im_w_block
                                + o_w * w_block / im_w_block)
                            * im_w_block;
                          fusion->create_output_fusion_anchor({blocking_output_
                              ? tensor_slice(output,
                                {{n, 1UL}, {anch_c, 1}, {anch_h, h_block},
                                  {anch_w, w_block}, {0, im_oc_block}})
                              : tensor_slice(output,
                                {{n, 1UL}, {anch_h, h_block}, {anch_w, w_block},
                                  {anch_c * im_oc_block, oc_block}})});
                        }
                      }
                    }
                  }
                  // TODO(xurui): need to add iterated anchor here to
                  // support more fusion opportunity
                }
              }
            }

            if (fusion && oc_threads == 1 && ic_threads == 1 && h_threads == 1
              && w_threads == 1) {
              fusion->create_output_fusion_anchor({blocking_output_
                  ? tensor_slice(output,
                    {{pbs, 1UL}, {0, oc_ / im_oc_block}, {0, oh_expr_},
                      {0, ow_}, {0, im_oc_block}})
                  : tensor_slice(
                    output, {{pbs, 1UL}, {0, oh_expr_}, {0, ow_}, {0, oc_}})});
            }
          }
          if (fusion && oc_threads == 1 && h_threads == 1 && w_threads == 1) {
            fusion->create_output_fusion_anchor({blocking_output_
                ? tensor_slice(output,
                  {{pbs, 1UL}, {0, oc_ / im_oc_block}, {0, oh_expr_}, {0, ow_},
                    {0, im_oc_block}})
                : tensor_slice(
                  output, {{pbs, 1UL}, {0, oh_expr_}, {0, ow_}, {0, oc_}})});
          }
        }

        if (fusion && h_threads == 1 && w_threads == 1) {
          fusion->create_output_fusion_anchor({blocking_output_
              ? tensor_slice(output,
                {{pbs, 1UL}, {0, oc_ / im_oc_block}, {0, oh_expr_}, {0, ow_},
                  {0, im_oc_block}})
              : tensor_slice(
                output, {{pbs, 1UL}, {0, oh_expr_}, {0, ow_}, {0, oc_}})});
        }
      }

      if (fusion && h_threads == 1) {
        fusion->create_output_fusion_anchor({blocking_output_
            ? tensor_slice(output,
              {{pbs, 1UL}, {0, oc_ / im_oc_block}, {0, oh_expr_}, {0, ow_},
                {0, im_oc_block}})
            : tensor_slice(
              output, {{pbs, 1UL}, {0, oh_expr_}, {0, ow_}, {0, oc_}})});
      }
    }
    if (fusion && mb_ > 1) {
      fusion->create_output_fusion_anchor(
        {blocking_output_ ? tensor_slice(output,
           {{pbs, 1UL}, {0, oc_ / im_oc_block}, {0, oh_expr_}, {0, ow_},
             {0, im_oc_block}})
                          : tensor_slice(output,
                            {{pbs, 1UL}, {0, oh_expr_}, {0, ow_}, {0, oc_}})});
    }
  }
  loops = {lpbs, lph, lpw, lpoc, lpic};
}

void gen_nested_conv_fwd_t::compute_conv_no_padding_os_blocking_nested(
  CONV_ARG_LIST) const {
  COMPILE_ASSERT(
    pack_rows, "Use nested conv with os blocking only if pack_rows is true")
  int bs_threads = config.bs_threads;
  int s_threads = config.w_threads;
  int oc_threads = config.oc_threads;
  int ic_threads = 1;

  int oc_block = config.K_block;
  int s_block = config.w_block;
  int ic_block = config.C_block;

  int im_oc_block = config.im_oc_block;
  int im_ic_block = config.im_ic_block;
  int im_s_block = config.im_w_block;

  COMPILE_ASSERT(oc_block % im_oc_block == 0,
    "oc_block % im_oc_block != 0, config is invalid")
  COMPILE_ASSERT(ic_block % im_ic_block == 0,
    "ic_block % im_ic_block != 0, config is invalid");
  COMPILE_ASSERT(
    s_block % im_s_block == 0, "s_block % im_s_block != 0, config is invalid");

  // param
  expr output_tmp = output;
  auto tinput = in_tensors_[0];
  auto tweight = in_tensors_[1];
  auto toutput = out_tensors_[0];
  const auto &input_blocking_dims = tinput.get_blocking_dims();
  const auto &weight_blocking_dims = tweight.get_blocking_dims();
  const auto &output_blocking_dims = toutput.get_blocking_dims();

  for_loop lpbs, lps, lpoc, lpic, los, looc, loic, lioc, lis, lok;

  int bs_num_block_pt, bs_tail_num_block_pt, oc_num_block_pt,
    oc_tail_num_block_pt, s_num_block_pt, s_tail_num_block_pt, ic_num_block_pt,
    ic_tail_num_block_pt;
  int bs_used_threads
    = block_split(mb_, bs_threads, bs_num_block_pt, bs_tail_num_block_pt);
  int oc_used_threads = block_split(utils::divide_and_ceil(oc_, oc_block),
    oc_threads, oc_num_block_pt, oc_tail_num_block_pt);
  int os_used_threads = block_split(utils::divide_and_ceil(os, s_block),
    s_threads, s_num_block_pt, s_tail_num_block_pt);
  int ic_used_threads = block_split(utils::divide_and_ceil(ic_, ic_block),
    ic_threads, ic_num_block_pt, ic_tail_num_block_pt);

  auto input_expr_dims = input.checked_as<tensor>()->dims_;
  auto mb_expr_ = input_expr_dims[0];

  if (ic_used_threads > 1) {
    // barrier
    // output temp buffer
    auto out_dims = output_blocking_dims;
    out_dims[0] *= ic_used_threads;
    _tensor_(out_tmp, toutput.dtype_, dims_to_expr(out_dims));
    output_tmp = out_tmp;
  }
  auto LDA = blocking_input_ ? sw_ * im_ic_block : sw_ * ic_;
  auto LDC = blocking_output_ ? im_oc_block : oc_;

  int oc_split = 1;
  auto nthreads = runtime_config_t::get().get_num_threads();
  bool parallel_space_is_enough
    = (mb_ % nthreads == 0 || utils::divide_and_ceil(mb_, nthreads) > 8);
  auto weight_size
    = math_utils::get_dims_product(in_tensors_[1].get_blocking_dims())
    * utils::get_sizeof_type(get_weight_dtype());
  auto L2_cache_size = ctx->machine_.cpu_flags_.getDCacheSize(2);
  if (weight_size >= L2_cache_size && parallel_space_is_enough
    && oc_threads == 1 && oc_num_block_pt == 1) {
    int num_block = oc_block / im_oc_block;
    int expected_split_num = utils::divide_and_ceil(weight_size, L2_cache_size);
    for (auto &factor : utils::get_factors(num_block)) {
      if (factor >= expected_split_num) {
        expected_split_num = factor;
        break;
      }
    }
    oc_split = num_block < expected_split_num ? 1 : expected_split_num;
  }

  _named_for_(lok, outer_k, 0, oc_split, 1, for_type::PARALLEL) {
    _named_for_(lpbs, pbs, 0, mb_expr_, 1, for_type::PARALLEL) {
      _named_for_(lps, ps, 0, os_used_threads, 1) {
        _named_for_(lpoc, poc, 0, oc_used_threads, 1) {
          _named_for_(lpic, pic, 0, ic_used_threads, 1) {
            expr s_num_block = builder::make_select(ps < (os_used_threads - 1),
                   s_num_block_pt, s_tail_num_block_pt),
                 oc_num_block
              = builder::make_select(poc < (oc_used_threads - 1),
                oc_num_block_pt, oc_tail_num_block_pt);
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
                    _named_for_(
                      lioc, i_oc, 0, oc_block / im_oc_block / oc_split) {
                      expr oc = poc * oc_num_block_pt * oc_block / im_oc_block
                        + o_oc * oc_block / im_oc_block
                        + outer_k * oc_block / im_oc_block / oc_split + i_oc;

                      _if_(oc * im_oc_block < oc_) {
                        _named_for_(lis, i_s, 0, s_block / im_s_block) {
                          _tensor_(A_list, datatypes::pointer,
                            {kh_ * kw_ * ic_block / im_ic_block});
                          _tensor_(B_list, datatypes::pointer,
                            {kh_ * kw_ * ic_block / im_ic_block});
                          auto im_s_block_idx
                            = ps * s_num_block_pt * s_block / im_s_block
                            + o_s * s_block / im_s_block + i_s;

                          auto out_tsr = tensor_ptr(output,
                            blocking_output_
                              ? std::vector<expr> {n, oc,
                                (im_s_block_idx * im_s_block) / ow_,
                                im_s_block_idx * im_s_block % ow_, 0}
                              : std::vector<expr> {n,
                                (im_s_block_idx * im_s_block) / ow_,
                                (im_s_block_idx * im_s_block) % ow_,
                                oc * im_oc_block});

                          int adj_ow = ow_ + num_elems_skip_per_ow_;

                          if (os / im_s_block == 1) {
                            out_tsr = tensor_ptr(output,
                              blocking_output_
                                ? std::vector<expr> {n, oc, 0, 0, 0}
                                : std::vector<expr> {
                                  n, 0, 0, oc * config.im_oc_block});
                          } else {
                            auto acc_m = os_acc_size[{im_s_block_idx}];
                            out_tsr = tensor_ptr(output,
                              blocking_output_
                                ? std::vector<expr> {n, oc, acc_m / ow_,
                                  acc_m % ow_, 0}
                                : std::vector<expr> {n, acc_m / ow_,
                                  acc_m % ow_, oc * im_oc_block});
                          }

                          _for_(i_c, 0, ic_block / im_ic_block) {
                            expr ic
                              = pic * ic_num_block_pt * ic_block / im_ic_block
                              + o_ic * ic_block / im_ic_block + i_c;
                            _if_(ic * im_ic_block < ic_) {
                              _for_(r, 0, kh_) {
                                _for_(s, 0, kw_) {
                                  auto idx = i_c * kh_ * kw_ + r * kw_ + s;
                                  auto h
                                    = ((im_s_block_idx * im_s_block) / adj_ow);
                                  auto w
                                    = ((im_s_block_idx * im_s_block) % adj_ow);
                                  std::vector<expr> input_pos = blocking_input_
                                    ? std::vector<expr> {n, ic,
                                      h * sh_ + dh_ * r, w * sw_ + dw_ * s, 0}
                                    : std::vector<expr> {n, h * sh_ + dh_ * r,
                                      w * sw_ + dw_ * s, ic * im_ic_block};

                                  A_list[idx] = tensor_ptr(input, input_pos);
                                  B_list[idx] = tensor_ptr(weight,
                                    kpack > 1
                                      ? std::vector<expr> {oc, ic, r, s, 0, 0,
                                        0}
                                      : std::vector<expr> {oc, ic, r, s, 0, 0});
                                }
                              }
                            }
                          }
                          const auto hint_A_size = im_s_block * im_ic_block
                            * kh_ * kw_ * ic_block / im_ic_block;
                          const auto hint_B_size
                            = im_oc_block * ic_block * kh_ * kw_;
                          const auto hint_C_size = im_s_block * im_oc_block;

                          sc_brgemm_attrs_t brg_attrs {
                            {brgemm::attr_key::max_bs,
                              kh_ * kw_ * ic_block / im_ic_block},
                            {brgemm::attr_key::hint_expected_A_size,
                              hint_A_size},
                            {brgemm::attr_key::hint_expected_B_size,
                              hint_B_size},
                            {brgemm::attr_key::hint_expected_C_size,
                              hint_C_size},
                            {brgemm::attr_key::use_interleave_stores, true},
                            {brgemm::attr_key::use_uker, true},
                            {brgemm::attr_key::bd_mask_level, 2}};

                          builtin::brgemm_init_list_update(A_list, B_list,
                            out_tsr, 1, im_s_block, im_oc_block, im_ic_block,
                            LDA, im_oc_block, LDC, 1 /*useless*/, 1 /*useless*/,
                            kh_ * kw_ * ic_block / im_ic_block,
                            get_input_dtype(), get_weight_dtype(), brg_attrs,
                            os_mask, im_s_block_idx, os / im_s_block);

                          if (fusion && ic_used_threads == 1
                            && ic_num_block_pt == 1) {
                            _if_(o_ic == (ic_num_block - 1)) {
                              auto os_num_block = os / im_s_block;
                              if (oh_ % os_num_block == 0) {
                                fusion->create_output_fusion_anchor(
                                  {blocking_output_
                                      ? tensor_slice(output,
                                        {{n, 1UL}, {oc, 1},
                                          {im_s_block_idx
                                              * (oh_ / os_num_block),
                                            (oh_ / os_num_block)},
                                          {0, ow_}, {0, im_oc_block}})
                                      : tensor_slice(output,
                                        {{n, 1UL},
                                          {im_s_block_idx
                                              * (oh_ / os_num_block),
                                            (oh_ / os_num_block)},
                                          {0, ow_},
                                          {oc * im_oc_block, im_oc_block}})});
                              }
                            }
                          }
                        }
                      }
                    }
                  }
                }
              }
            }

            if (fusion && oc_threads == 1 && ic_threads == 1
              && s_threads == 1) {
              fusion->create_output_fusion_anchor({blocking_output_
                  ? tensor_slice(output,
                    {{pbs, 1UL},
                      {outer_k * oc_ / im_oc_block / oc_split,
                        oc_ / im_oc_block / oc_split},
                      {0, oh_}, {0, ow_}, {0, im_oc_block}})
                  : tensor_slice(output,
                    {{pbs, 1UL}, {0, oh_}, {0, ow_},
                      {outer_k * oc_ / oc_split, oc_ / oc_split}})});
            }
          }

          if (fusion && oc_threads == 1 && s_threads == 1) {
            fusion->create_output_fusion_anchor({blocking_output_
                ? tensor_slice(output,
                  {{pbs, 1UL},
                    {outer_k * oc_ / im_oc_block / oc_split,
                      oc_ / im_oc_block / oc_split},
                    {0, oh_}, {0, ow_}, {0, im_oc_block}})
                : tensor_slice(output,
                  {{pbs, 1UL}, {0, oh_}, {0, ow_},
                    {outer_k * oc_ / oc_split, oc_ / oc_split}})});
          }
        }
        if (fusion && s_threads == 1) {
          fusion->create_output_fusion_anchor({blocking_output_
              ? tensor_slice(output,
                {{pbs, 1UL},
                  {outer_k * oc_ / im_oc_block / oc_split,
                    oc_ / im_oc_block / oc_split},
                  {0, oh_}, {0, ow_}, {0, im_oc_block}})
              : tensor_slice(output,
                {{pbs, 1UL}, {0, oh_}, {0, ow_},
                  {outer_k * oc_ / oc_split, oc_ / oc_split}})});
        }
      }
      if (fusion && mb_ > 1) {
        fusion->create_output_fusion_anchor(
          {blocking_output_ ? tensor_slice(output,
             {{pbs, 1UL},
               {outer_k * oc_ / im_oc_block / oc_split,
                 oc_ / im_oc_block / oc_split},
               {0, oh_}, {0, ow_}, {0, im_oc_block}})
                            : tensor_slice(output,
                              {{pbs, 1UL}, {0, oh_}, {0, ow_},
                                {outer_k * oc_ / oc_split, oc_ / oc_split}})});
      }
    }
  }

  loops = {lpbs, lps, lpoc, lpic, lok};
}

void gen_nested_conv_fwd_t::dynamic_compute_conv_no_padding_nested(
  CONV_ARG_LIST) const {
  int num_threads = runtime_config_t::get().get_num_threads();
  int h_threads = config.h_threads;
  int oc_threads = config.oc_threads;
  int bs_threads = num_threads / h_threads / oc_threads;
  int w_threads = config.w_threads;
  int ic_threads = 1;
  auto input_expr_dims = input.checked_as<tensor>()->dims_;
  auto mb_expr_ = input_expr_dims[0];

  auto output_expr_dims = output.checked_as<tensor>()->dims_;
  auto oh_expr_ = input_expr_dims.size() == 4
    ? (input_expr_dims[1] + (ph_b_ + ph_e_) - kh_) / sh_ + 1
    : (input_expr_dims[2] + (ph_b_ + ph_e_) - kh_) / sh_ + 1;
  auto ow_expr_ = input_expr_dims.size() == 4
    ? (input_expr_dims[2] + (pw_b_ + pw_e_) - kw_) / sw_ + 1
    : (input_expr_dims[3] + (pw_b_ + pw_e_) - kw_) / sw_ + 1;

  int oc_block = oc_ / oc_threads;

  // by observation
  expr im_h_block = do_cast_and_fold(
    builder::make_select(oh_expr_ <= 14 && ow_expr_ <= 14 && h_threads == 1,
      builder::make_cast(datatypes::s32, oh_expr_), config.im_h_block));

  expr h_block
    = do_cast_and_fold(builder::make_select(oh_expr_ % h_threads == 0,
      builder::make_cast(datatypes::s32, oh_expr_ / h_threads),
      config.im_h_block));

  expr w_block = do_cast_and_fold((ow_expr_ + w_threads - 1) / w_threads);

  int ic_block = ic_ / ic_threads;
  int im_oc_block = config.im_oc_block;
  int im_ic_block = config.im_ic_block;

  int im_w_block = config.im_w_block;

  COMPILE_ASSERT(oc_block % im_oc_block == 0,
    "oc_block % im_oc_block != 0, config is invalid")
  COMPILE_ASSERT(ic_block % im_ic_block == 0,
    "ic_block % im_ic_block != 0, config is invalid")

  // param
  expr output_tmp = output;
  auto tinput = in_tensors_[0];
  auto tweight = in_tensors_[1];
  auto toutput = out_tensors_[0];
  const auto &input_blocking_dims = tinput.get_blocking_dims();
  const auto &weight_blocking_dims = tweight.get_blocking_dims();
  const auto &output_blocking_dims = toutput.get_blocking_dims();

  for_loop lpbs, lph, lpw, lpoc, lpic, loh, low, looc, loic, lioc, lih, liw,
    lok;

  int oc_num_block_pt, oc_tail_num_block_pt, ic_num_block_pt,
    ic_tail_num_block_pt;

  int oc_used_threads = block_split(utils::divide_and_ceil(oc_, oc_block),
    oc_threads, oc_num_block_pt, oc_tail_num_block_pt);
  expr h_num_block_pt
    = divide_and_ceil(divide_and_ceil(oh_expr_, h_block), h_threads);
  expr h_tail_num_block_pt = builder::make_select(
    divide_and_ceil(oh_expr_, h_block) % h_num_block_pt == 0, h_num_block_pt,
    divide_and_ceil(oh_expr_, h_block) % h_num_block_pt);
  expr oh_used_threads
    = divide_and_ceil(divide_and_ceil(oh_expr_, h_block), h_num_block_pt);

  expr ow_used_threads = do_cast_and_fold((ow_expr_ + w_block - 1) / w_block);
  expr w_num_block_pt = ow_used_threads / w_threads;
  expr w_tail_num_block_pt
    = builder::make_select(ow_used_threads % w_num_block_pt == 0,
      w_num_block_pt, ow_used_threads % w_num_block_pt);

  int ic_used_threads = block_split(utils::divide_and_ceil(ic_, ic_block),
    ic_threads, ic_num_block_pt, ic_tail_num_block_pt);

  if (ic_used_threads > 1) {
    // barrier
    // output temp buffer
    auto out_dims = output_blocking_dims;
    out_dims[0] *= ic_used_threads;
    _tensor_(out_tmp, toutput.dtype_, dims_to_expr(out_dims));
    output_tmp = out_tmp;
  }

  expr cond_tail_w = w_block % im_w_block != 0 || ow_expr_ % im_w_block != 0;
  expr cond_tail_h = h_block % im_h_block != 0 || oh_expr_ % im_h_block != 0;
  auto weight_size
    = math_utils::get_dims_product(in_tensors_[1].get_blocking_dims())
    * utils::get_sizeof_type(get_weight_dtype());
  auto L2_cache_size = ctx->machine_.cpu_flags_.getDCacheSize(2);
  int oc_split = (oc_threads == 1 && oc_num_block_pt == 1)
    ? get_oc_split_factor(
      -1, weight_size, L2_cache_size, oc_block / im_oc_block)
    : 1;

  auto LDA = blocking_input_ ? sw_ * im_ic_block : sw_ * ic_;
  auto LDC = blocking_output_ ? im_oc_block : oc_;
  // will update template for parallel merge in dynamic conv block
  _named_for_(lok, outer_k, 0, oc_split, 1, for_type::PARALLEL) {
    _named_for_(lpbs, pbs, 0, mb_expr_, 1, for_type::PARALLEL) {
      _named_for_(lph, ph, 0, h_threads, 1) {
        _named_for_(lpw, pw, 0, w_threads, 1) {
          _named_for_(lpoc, poc, 0, oc_threads, 1) {
            _named_for_(lpic, pic, 0, ic_threads, 1) {
              expr h_num_block
                = builder::make_select(ph < (oh_used_threads - 1),
                  h_num_block_pt, h_tail_num_block_pt),
                w_num_block = builder::make_select(pw < (ow_used_threads - 1),
                  w_num_block_pt, w_tail_num_block_pt),
                oc_num_block = builder::make_select(poc < (oc_used_threads - 1),
                  oc_num_block_pt, oc_tail_num_block_pt);
              _if_(ph < oh_used_threads && pw < ow_used_threads
                && poc < oc_used_threads && pic < ic_used_threads) {
                // single core
                expr ic_num_block
                  = builder::make_select(pic < (ic_used_threads - 1),
                    ic_num_block_pt, ic_tail_num_block_pt);

                expr n = pbs;
                _named_for_(loh, o_h, 0, h_num_block_pt) {
                  _named_for_(low, o_w, 0, w_num_block_pt) {
                    _named_for_(looc, o_oc, 0, oc_num_block_pt) {
                      _named_for_(loic, o_ic, 0, ic_num_block_pt) {
                        expr cond = o_h < h_num_block && o_w < w_num_block
                          && o_oc < oc_num_block && o_ic < ic_num_block;
                        _if_(cond) {
                          _named_for_(lih, i_h, 0,
                            (h_block + im_h_block - 1) / im_h_block) {
                            expr h = ph * h_num_block_pt * h_block
                              + o_h * h_block + i_h * im_h_block;
                            expr is_tail_h
                              = cond_tail_h && (h + im_h_block > oh_expr_);

                            expr real_im_h_block
                              = builder::make_select(is_tail_h,
                                builder::make_cast(
                                  datatypes::s32, oh_expr_ % im_h_block),
                                im_h_block);
                            _if_(h < oh_expr_) {
                              _named_for_(liw, i_w, 0,
                                (w_block + im_w_block - 1) / im_w_block) {
                                expr w = pw * w_num_block_pt * w_block
                                  + o_w * w_block + i_w * im_w_block;
                                _if_(w < ow_expr_) {
                                  expr is_tail_w = cond_tail_w
                                    && (w + im_w_block > ow_expr_);
                                  expr real_im_w_block
                                    = builder::make_select(is_tail_w,
                                      builder::make_cast(
                                        datatypes::s32, ow_expr_ % im_w_block),
                                      im_w_block);
                                  _named_for_(lioc, i_oc, 0,
                                    oc_block / im_oc_block / oc_split) {
                                    expr oc = poc * oc_num_block_pt * oc_block
                                        / im_oc_block
                                      + o_oc * oc_block / im_oc_block
                                      + outer_k * oc_block / im_oc_block
                                        / oc_split
                                      + i_oc;
                                    _if_(oc * im_oc_block < oc_) {
                                      _for_(im_h_i, 0, im_h_block) {
                                        _if_(h + im_h_i < oh_expr_) {
                                          _tensor_(A_list, datatypes::pointer,
                                            {kh_ * kw_ * ic_block
                                              / im_ic_block});
                                          _tensor_(B_list, datatypes::pointer,
                                            {kh_ * kw_ * ic_block
                                              / im_ic_block});

                                          _for_(
                                            i_c, 0, ic_block / im_ic_block) {
                                            expr ic = pic * ic_num_block_pt
                                                * ic_block / im_ic_block
                                              + o_ic * ic_block / im_ic_block
                                              + i_c;

                                            _if_(ic * im_ic_block < ic_) {
                                              _for_(r, 0, kh_) {
                                                _for_(s, 0, kw_) {
                                                  auto idx = i_c * kh_ * kw_
                                                    + r * kw_ + s;
                                                  std::vector<expr> input_pos
                                                    = blocking_input_
                                                    ? std::vector<expr> {n, ic,
                                                      (h + im_h_i) * sh_
                                                        + dh_ * r,
                                                      w * sw_ + dw_ * s, 0}
                                                    : std::vector<expr> {n,
                                                      (h + im_h_i) * sh_
                                                        + dh_ * r,
                                                      w * sw_ + dw_ * s,
                                                      ic * im_ic_block};

                                                  A_list[idx] = tensor_ptr(
                                                    input, input_pos);
                                                  B_list[idx]
                                                    = tensor_ptr(weight,
                                                      kpack > 1
                                                        ? std::vector<expr> {oc,
                                                          ic, r, s, 0, 0, 0}
                                                        : std::vector<expr> {
                                                          oc, ic, r, s, 0, 0});
                                                }
                                              }
                                            }
                                          }
                                          std::vector<expr> output_pos
                                            = blocking_output_
                                            ? std::vector<expr> {pic * mb_expr_
                                                + n,
                                              oc, h + im_h_i, w, 0}
                                            : std::vector<expr> {
                                              pic * mb_expr_ + n, h + im_h_i, w,
                                              oc * im_oc_block};

                                          generate_brgemm(real_im_w_block,
                                            im_ic_block, im_oc_block, ic_block,
                                            o_ic, ic_num_block_pt, A_list,
                                            B_list,
                                            tensor_ptr(output_tmp, output_pos),
                                            LDA, LDC);

                                          if (fusion && ic_used_threads == 1
                                            && ic_num_block_pt == 1) {
                                            _if_(o_ic == (ic_num_block - 1)) {
                                              fusion
                                                ->create_output_fusion_anchor(
                                                  {blocking_output_
                                                      ? tensor_slice(output,
                                                        {{n, 1UL}, {oc, 1},
                                                          {h + im_h_i, 1},
                                                          {w, real_im_w_block},
                                                          {0, im_oc_block}})
                                                      : tensor_slice(output,
                                                        {{n, 1UL},
                                                          {h + im_h_i, 1},
                                                          {w, real_im_w_block},
                                                          {oc * im_oc_block,
                                                            im_oc_block}})});
                                            }
                                          } // im_h_i
                                        }
                                      }
                                      if (fusion && ic_used_threads == 1
                                        && ic_num_block_pt == 1) {
                                        _if_(o_ic == (ic_num_block - 1)) {
                                          fusion->create_output_fusion_anchor(
                                            {blocking_output_
                                                ? tensor_slice(output,
                                                  {{n, 1UL}, {oc, 1},
                                                    {h, real_im_h_block},
                                                    {w, real_im_w_block},
                                                    {0, im_oc_block}})
                                                : tensor_slice(output,
                                                  {{n, 1UL},
                                                    {h, real_im_h_block},
                                                    {w, real_im_w_block},
                                                    {oc * im_oc_block,
                                                      im_oc_block}})});
                                        }
                                      }
                                    } // i_oc
                                  }
                                  if (fusion && ic_used_threads == 1
                                    && ic_num_block_pt == 1
                                    && oc_block * oc_used_threads == oc_) {
                                    _if_(o_ic == (ic_num_block - 1)) {
                                      expr anch_c = poc * oc_num_block_pt
                                          * oc_block / im_oc_block
                                        + o_oc * oc_block / im_oc_block
                                        + outer_k * oc_block / im_oc_block
                                          / oc_split;
                                      fusion->create_output_fusion_anchor(
                                        {blocking_output_
                                            ? tensor_slice(output,
                                              {{n, 1UL}, {anch_c, 1},
                                                {h, real_im_h_block},
                                                {w, real_im_w_block},
                                                {0, im_oc_block}})
                                            : tensor_slice(output,
                                              {{n, 1UL}, {h, real_im_h_block},
                                                {w, real_im_w_block},
                                                {anch_c * im_oc_block,
                                                  im_oc_block}})});
                                    }
                                  }
                                } // i_w
                              }
                              if (fusion && !is_dynamic_dim(ow_)
                                && get_expr_as_int(w_block)
                                    * get_expr_as_int(ow_used_threads)
                                  == ow_
                                && ic_used_threads == 1 && ic_num_block_pt == 1
                                && oc_block * oc_used_threads == oc_) {
                                _if_(o_ic == (ic_num_block - 1)) {
                                  expr anch_c = poc * oc_num_block_pt * oc_block
                                      / im_oc_block
                                    + o_oc * oc_block / im_oc_block
                                    + outer_k * oc_block / im_oc_block
                                      / oc_split;
                                  expr anch_w = pw * w_num_block_pt * w_block
                                    + o_w * w_block;

                                  fusion->create_output_fusion_anchor(
                                    {blocking_output_ ? tensor_slice(output,
                                       {{n, 1UL}, {anch_c, 1},
                                         {h, real_im_h_block},
                                         {anch_w, w_block}, {0, im_oc_block}})
                                                      : tensor_slice(output,
                                                        {{n, 1UL},
                                                          {h, real_im_h_block},
                                                          {anch_w, w_block},
                                                          {anch_c * im_oc_block,
                                                            oc_block}})});
                                }
                              }
                            } // i_h
                          }

                          if (fusion && ic_used_threads == 1
                            && ic_num_block_pt == 1 && !is_dynamic_dim(oh_)
                            && !is_dynamic_dim(ow_)
                            && oc_block * oc_used_threads == oc_
                            && get_expr_as_int(h_block)
                                * get_expr_as_int(oh_used_threads)
                              == oh_
                            && get_expr_as_int(w_block)
                                * get_expr_as_int(ow_used_threads)
                              == ow_) {
                            _if_(o_ic == (ic_num_block - 1)) {
                              expr anch_c
                                = poc * oc_num_block_pt * oc_block / im_oc_block
                                + o_oc * oc_block / im_oc_block
                                + outer_k * oc_block / im_oc_block / oc_split;
                              expr anch_h
                                = ph * h_num_block_pt * h_block + o_h * h_block;
                              expr anch_w = (pw * w_num_block_pt * w_block
                                + o_w * w_block);
                              fusion->create_output_fusion_anchor(
                                {blocking_output_
                                    ? tensor_slice(output,
                                      {{n, 1UL}, {anch_c, 1},
                                        {anch_h, oh_ / oh_used_threads},
                                        {anch_w, ow_ / ow_used_threads},
                                        {0, im_oc_block}})
                                    : tensor_slice(output,
                                      {{n, 1UL},
                                        {anch_h, oh_ / oh_used_threads},
                                        {anch_w, ow_ / ow_used_threads},
                                        {anch_c * im_oc_block, oc_block}})});
                            }
                          }
                        } // o_ic
                      }
                    }
                  }
                }
              }

              if (fusion && oc_threads == 1 && ic_threads == 1 && h_threads == 1
                && w_threads == 1 && !is_dynamic_dim(oh_)
                && !is_dynamic_dim(ow_)) {
                fusion->create_output_fusion_anchor({blocking_output_
                    ? tensor_slice(output,
                      {{pbs, 1UL},
                        {outer_k * oc_ / im_oc_block / oc_split,
                          oc_ / im_oc_block / oc_split},
                        {0, oh_}, {0, ow_}, {0, im_oc_block}})
                    : tensor_slice(output,
                      {{pbs, 1UL}, {0, oh_}, {0, ow_},
                        {outer_k * oc_ / oc_split, oc_ / oc_split}})});
              }
            }

            if (fusion && oc_threads == 1 && h_threads == 1 && w_threads == 1
              && !is_dynamic_dim(oh_) && !is_dynamic_dim(ow_)) {
              fusion->create_output_fusion_anchor({blocking_output_
                  ? tensor_slice(output,
                    {{pbs, 1UL},
                      {outer_k * oc_ / im_oc_block / oc_split,
                        oc_ / im_oc_block / oc_split},
                      {0, oh_}, {0, ow_}, {0, im_oc_block}})
                  : tensor_slice(output,
                    {{pbs, 1UL}, {0, oh_}, {0, ow_},
                      {outer_k * oc_ / oc_split, oc_ / oc_split}})});
            }
          }
          if (fusion && h_threads == 1 && w_threads == 1 && !is_dynamic_dim(oh_)
            && !is_dynamic_dim(ow_)) {
            fusion->create_output_fusion_anchor({blocking_output_
                ? tensor_slice(output,
                  {{pbs, 1UL},
                    {outer_k * oc_ / im_oc_block / oc_split,
                      oc_ / im_oc_block / oc_split},
                    {0, oh_}, {0, ow_}, {0, im_oc_block}})
                : tensor_slice(output,
                  {{pbs, 1UL}, {0, oh_}, {0, ow_},
                    {outer_k * oc_ / oc_split, oc_ / oc_split}})});
          }
        }

        if (fusion && h_threads == 1 && !is_dynamic_dim(oh_)
          && !is_dynamic_dim(ow_)) {
          fusion->create_output_fusion_anchor({blocking_output_
              ? tensor_slice(output,
                {{pbs, 1UL},
                  {outer_k * oc_ / im_oc_block / oc_split,
                    oc_ / im_oc_block / oc_split},
                  {0, oh_}, {0, ow_}, {0, im_oc_block}})
              : tensor_slice(output,
                {{pbs, 1UL}, {0, oh_}, {0, ow_},
                  {outer_k * oc_ / oc_split, oc_ / oc_split}})});
        }
      }
      if (fusion && !is_dynamic_dim(oh_) && !is_dynamic_dim(ow_)) {
        _if_(mb_expr_ > 1) {
          fusion->create_output_fusion_anchor({blocking_output_
              ? tensor_slice(output,
                {{pbs, 1UL},
                  {outer_k * oc_ / im_oc_block / oc_split,
                    oc_ / im_oc_block / oc_split},
                  {0, oh_}, {0, ow_}, {0, im_oc_block}})
              : tensor_slice(output,
                {{pbs, 1UL}, {0, oh_}, {0, ow_},
                  {outer_k * oc_ / oc_split, oc_ / oc_split}})});
        }
      }
    }
  }
  loops = {lpbs, lph, lpw, lpoc, lpic, lok};
}

void gen_nested_conv_fwd_t::compute_conv_no_padding_nested(
  CONV_ARG_LIST) const {
  int bs_threads = config.bs_threads;
  int h_threads = config.h_threads;
  int w_threads = config.w_threads;
  int oc_threads = config.oc_threads;
  int ic_threads = 1;

  int oc_block = config.K_block;
  int h_block = config.h_block;
  int w_block = config.w_block;
  int ic_block = config.C_block;
  int im_oc_block = config.im_oc_block;
  int im_ic_block = config.im_ic_block;
  int im_h_block = config.im_h_block;
  int im_w_block = config.im_w_block;

  COMPILE_ASSERT(oc_block % im_oc_block == 0,
    "oc_block % im_oc_block != 0, config is invalid")
  COMPILE_ASSERT(ic_block % im_ic_block == 0,
    "ic_block % im_ic_block != 0, config is invalid")
  COMPILE_ASSERT(
    h_block % im_h_block == 0, "h_block % im_h_block != 0, config is invalid")
  COMPILE_ASSERT(
    w_block % im_w_block == 0, "w_block % im_w_block != 0, config is invalid")

  // param
  expr output_tmp = output;
  auto tinput = in_tensors_[0];
  auto tweight = in_tensors_[1];
  auto toutput = out_tensors_[0];
  const auto &input_blocking_dims = tinput.get_blocking_dims();
  const auto &weight_blocking_dims = tweight.get_blocking_dims();
  const auto &output_blocking_dims = toutput.get_blocking_dims();

  for_loop lpbs, lph, lpw, lpoc, lpic, loh, low, looc, loic, lioc, lih, liw,
    lok;

  int oc_num_block_pt, oc_tail_num_block_pt, h_num_block_pt,
    h_tail_num_block_pt, w_num_block_pt, w_tail_num_block_pt, ic_num_block_pt,
    ic_tail_num_block_pt;

  int oc_used_threads = block_split(utils::divide_and_ceil(oc_, oc_block),
    oc_threads, oc_num_block_pt, oc_tail_num_block_pt);
  int oh_used_threads = block_split(utils::divide_and_ceil(oh_, h_block),
    h_threads, h_num_block_pt, h_tail_num_block_pt);

  int ow_used_threads = block_split(utils::divide_and_ceil(ow_, w_block),
    w_threads, w_num_block_pt, w_tail_num_block_pt);

  int ic_used_threads = block_split(utils::divide_and_ceil(ic_, ic_block),
    ic_threads, ic_num_block_pt, ic_tail_num_block_pt);

  if (ic_used_threads > 1) {
    // barrier
    // output temp buffer
    auto out_dims = output_blocking_dims;
    out_dims[0] *= ic_used_threads;
    _tensor_(out_tmp, toutput.dtype_, dims_to_expr(out_dims));
    output_tmp = out_tmp;
  }

  auto input_expr_dims = input.checked_as<tensor>()->dims_;
  auto mb_expr_ = input_expr_dims[0];

  auto LDA = blocking_input_ ? sw_ * im_ic_block : sw_ * ic_;
  auto LDC = blocking_output_ ? im_oc_block : oc_;

  int oc_split = 1;
  auto nthreads = runtime_config_t::get().get_num_threads();
  bool parallel_space_is_enough
    = (mb_ % nthreads == 0 || utils::divide_and_ceil(mb_, nthreads) > 8);
  auto weight_size
    = math_utils::get_dims_product(in_tensors_[1].get_blocking_dims())
    * utils::get_sizeof_type(get_weight_dtype());
  auto L2_cache_size = ctx->machine_.cpu_flags_.getDCacheSize(2);
  if (weight_size >= L2_cache_size && parallel_space_is_enough
    && oc_threads == 1 && oc_num_block_pt == 1) {
    int num_block = oc_block / im_oc_block;
    int expected_split_num = utils::divide_and_ceil(weight_size, L2_cache_size);
    for (auto &factor : utils::get_factors(num_block)) {
      if (factor >= expected_split_num) {
        expected_split_num = factor;
        break;
      }
    }
    oc_split = num_block < expected_split_num ? 1 : expected_split_num;
  }

  _named_for_(lok, outer_k, 0, oc_split, 1, for_type::PARALLEL) {
    _named_for_(lpbs, pbs, 0, mb_expr_, 1, for_type::PARALLEL) {
      _named_for_(lph, ph, 0, oh_used_threads, 1) {
        _named_for_(lpw, pw, 0, ow_used_threads, 1) {
          _named_for_(lpoc, poc, 0, oc_used_threads, 1) {
            _named_for_(lpic, pic, 0, ic_used_threads, 1) {
              expr h_num_block
                = builder::make_select(ph < (oh_used_threads - 1),
                  h_num_block_pt, h_tail_num_block_pt),
                w_num_block = builder::make_select(pw < (ow_used_threads - 1),
                  w_num_block_pt, w_tail_num_block_pt),
                oc_num_block = builder::make_select(poc < (oc_used_threads - 1),
                  oc_num_block_pt, oc_tail_num_block_pt);

              // single core
              expr ic_num_block
                = builder::make_select(pic < (ic_used_threads - 1),
                  ic_num_block_pt, ic_tail_num_block_pt);

              expr n = pbs;
              _named_for_(loh, o_h, 0, h_num_block_pt) {
                _named_for_(low, o_w, 0, w_num_block_pt) {
                  _named_for_(looc, o_oc, 0, oc_num_block_pt) {
                    _named_for_(loic, o_ic, 0, ic_num_block_pt) {
                      expr cond = o_h < h_num_block && o_w < w_num_block
                        && o_oc < oc_num_block && o_ic < ic_num_block;
                      _if_(cond) {
                        _named_for_(lih, i_h, 0, h_block / im_h_block) {
                          expr h = (ph * h_num_block_pt * h_block / im_h_block
                                     + o_h * h_block / im_h_block + i_h)
                            * im_h_block;
                          _named_for_(liw, i_w, 0, w_block / im_w_block) {
                            expr w = (pw * w_num_block_pt * w_block / im_w_block
                                       + o_w * w_block / im_w_block + i_w)
                              * im_w_block;
                            _if_(w < ow_) {
                              _named_for_(
                                lioc, i_oc, 0, oc_block / im_oc_block) {
                                expr oc = poc * oc_num_block_pt * oc_block
                                    / im_oc_block
                                  + o_oc * oc_block / im_oc_block
                                  + outer_k * oc_block / im_oc_block / oc_split
                                  + i_oc;
                                _if_(oc * im_oc_block < oc_) {
                                  _tensor_(A_list, datatypes::pointer,
                                    {kh_ * kw_ * ic_block / im_ic_block});
                                  _tensor_(B_list, datatypes::pointer,
                                    {kh_ * kw_ * ic_block / im_ic_block});

                                  _for_(im_h_i, 0, im_h_block) {
                                    _if_(h + im_h_i < oh_) {
                                      _for_(i_c, 0, ic_block / im_ic_block) {
                                        expr ic = pic * ic_num_block_pt
                                            * ic_block / im_ic_block
                                          + o_ic * ic_block / im_ic_block + i_c;
                                        _if_(ic * im_ic_block < ic_) {
                                          _for_(r, 0, kh_) {
                                            _for_(s, 0, kw_) {
                                              auto idx
                                                = i_c * kh_ * kw_ + r * kw_ + s;
                                              std::vector<expr> input_pos
                                                = blocking_input_
                                                ? std::vector<expr> {n, ic,
                                                  (h + im_h_i) * sh_ + r,
                                                  w * sw_ + s, 0}
                                                : std::vector<expr> {n,
                                                  (h + im_h_i) * sh_ + r,
                                                  w * sw_ + s,
                                                  ic * im_ic_block};

                                              A_list[idx]
                                                = tensor_ptr(input, input_pos);
                                              B_list[idx] = tensor_ptr(weight,
                                                kpack > 1
                                                  ? std::vector<expr> {oc, ic,
                                                    r, s, 0, 0, 0}
                                                  : std::vector<expr> {
                                                    oc, ic, r, s, 0, 0});
                                            }
                                          }
                                        }
                                      }
                                      const auto hint_A_size
                                        = im_w_block * ic_block * kh_ * kw_;
                                      const auto hint_B_size
                                        = im_oc_block * ic_block * kh_ * kw_;
                                      const auto hint_C_size
                                        = im_w_block * im_oc_block;

                                      sc_brgemm_attrs_t brg_attrs {
                                        {brgemm::attr_key::max_bs,
                                          kh_ * kw_ * ic_block / im_ic_block},
                                        {brgemm::attr_key::hint_expected_A_size,
                                          hint_A_size},
                                        {brgemm::attr_key::hint_expected_B_size,
                                          hint_B_size},
                                        {brgemm::attr_key::hint_expected_C_size,
                                          hint_C_size},
                                        {brgemm::attr_key::
                                            use_interleave_stores,
                                          true},
                                        {brgemm::attr_key::use_uker, true},
                                        {brgemm::attr_key::bd_mask_level, 0}};

                                      std::vector<expr> output_pos
                                        = blocking_output_
                                        ? std::vector<expr> {pic * mb_ + n, oc,
                                          h + im_h_i, w, 0}
                                        : std::vector<expr> {pic * mb_ + n,
                                          h + im_h_i, w, oc * im_oc_block};

                                      if (ic_num_block_pt > 1) {
                                        _if_(o_ic == 0) {
                                          builtin::brgemm_init_list_update(
                                            A_list, B_list,
                                            tensor_ptr(output_tmp, output_pos),
                                            1, im_w_block, im_oc_block,
                                            im_ic_block, LDA, im_oc_block, LDC,
                                            1 /*useless*/
                                            ,
                                            1 /*useless*/
                                            ,
                                            kh_ * kw_ * ic_block / im_ic_block,
                                            get_input_dtype(),
                                            get_weight_dtype(), brg_attrs);
                                        }
                                        _else_ {
                                          builtin::brgemm_list_update(A_list,
                                            B_list,
                                            tensor_ptr(output_tmp, output_pos),
                                            1, im_w_block, im_oc_block,
                                            im_ic_block, LDA, im_oc_block, LDC,
                                            1 /*useless*/
                                            ,
                                            1 /*useless*/
                                            ,
                                            kh_ * kw_ * ic_block / im_ic_block,
                                            get_input_dtype(),
                                            get_weight_dtype(), brg_attrs);
                                        }
                                      } else {
                                        builtin::brgemm_init_list_update(A_list,
                                          B_list,
                                          tensor_ptr(output_tmp, output_pos), 1,
                                          im_w_block, im_oc_block, im_ic_block,
                                          LDA, im_oc_block, LDC, 1 /*useless*/
                                          ,
                                          1 /*useless*/
                                          ,
                                          kh_ * kw_ * ic_block / im_ic_block,
                                          get_input_dtype(), get_weight_dtype(),
                                          brg_attrs);
                                      }

                                      if (fusion && ic_used_threads == 1
                                        && ic_num_block_pt == 1) {
                                        _if_(o_ic == (ic_num_block - 1)) {
                                          fusion->create_output_fusion_anchor(
                                            {blocking_output_
                                                ? tensor_slice(output,
                                                  {{n, 1UL}, {oc, 1},
                                                    {h + im_h_i, 1},
                                                    {w, im_w_block},
                                                    {0, im_oc_block}})
                                                : tensor_slice(output,
                                                  {{n, 1UL}, {h + im_h_i, 1},
                                                    {w, im_w_block},
                                                    {oc * im_oc_block,
                                                      im_oc_block}})});
                                        }
                                      }
                                    }
                                  }
                                  if (fusion && ic_used_threads == 1
                                    && ic_num_block_pt == 1) {
                                    _if_(o_ic == (ic_num_block - 1)) {
                                      fusion->create_output_fusion_anchor(
                                        {blocking_output_
                                            ? tensor_slice(output,
                                              {{n, 1UL}, {oc, 1},
                                                {h, im_h_block},
                                                {w, im_w_block},
                                                {0, im_oc_block}})
                                            : tensor_slice(output,
                                              {{n, 1UL}, {h, im_h_block},
                                                {w, im_w_block},
                                                {oc * im_oc_block,
                                                  im_oc_block}})});
                                    }
                                  }
                                }
                              }
                              if (fusion && ic_used_threads == 1
                                && ic_num_block_pt == 1
                                && oc_block * oc_used_threads == oc_) {
                                _if_(o_ic == (ic_num_block - 1)) {
                                  expr anch_c = poc * oc_num_block_pt * oc_block
                                      / im_oc_block
                                    + o_oc * oc_block / im_oc_block
                                    + outer_k * oc_block / im_oc_block
                                      / oc_split;
                                  fusion->create_output_fusion_anchor(
                                    {blocking_output_ ? tensor_slice(output,
                                       {{n, 1UL}, {anch_c, 1}, {h, im_h_block},
                                         {w, im_w_block}, {0, im_oc_block}})
                                                      : tensor_slice(output,
                                                        {{n, 1UL},
                                                          {h, im_h_block},
                                                          {w, im_w_block},
                                                          {anch_c * im_oc_block,
                                                            oc_block}})});
                                }
                              }
                            }
                          }

                          if (fusion && ic_used_threads == 1
                            && ic_num_block_pt == 1
                            && oc_block * oc_used_threads == oc_
                            && w_block * ow_used_threads == ow_) {
                            _if_(o_ic == (ic_num_block - 1)) {
                              expr anch_c
                                = poc * oc_num_block_pt * oc_block / im_oc_block
                                + o_oc * oc_block / im_oc_block
                                + outer_k * oc_block / im_oc_block / oc_split;
                              expr anch_w
                                = (pw * w_num_block_pt * w_block / im_w_block
                                    + o_w * w_block / im_w_block)
                                * im_w_block;
                              fusion->create_output_fusion_anchor(
                                {blocking_output_
                                    ? tensor_slice(output,
                                      {{n, 1UL}, {anch_c, 1}, {h, im_h_block},
                                        {anch_w, w_block}, {0, im_oc_block}})
                                    : tensor_slice(output,
                                      {{n, 1UL}, {h, im_h_block},
                                        {anch_w, w_block},
                                        {anch_c * im_oc_block, oc_block}})});
                            }
                          }
                        }

                        if (fusion && ic_used_threads == 1
                          && ic_num_block_pt == 1
                          && oc_block * oc_used_threads == oc_
                          && w_block * ow_used_threads == ow_
                          && h_block * oh_used_threads == oh_) {
                          _if_(o_ic == (ic_num_block - 1)) {
                            expr anch_c
                              = poc * oc_num_block_pt * oc_block / im_oc_block
                              + o_oc * oc_block / im_oc_block
                              + outer_k * oc_block / im_oc_block / oc_split;
                            expr anch_h
                              = (ph * h_num_block_pt * h_block / im_h_block
                                  + o_h * h_block / im_h_block)
                              * im_h_block;
                            expr anch_w
                              = (pw * w_num_block_pt * w_block / im_w_block
                                  + o_w * w_block / im_w_block)
                              * im_w_block;
                            fusion->create_output_fusion_anchor(
                              {blocking_output_
                                  ? tensor_slice(output,
                                    {{n, 1UL}, {anch_c, 1}, {anch_h, h_block},
                                      {anch_w, w_block}, {0, im_oc_block}})
                                  : tensor_slice(output,
                                    {{n, 1UL}, {anch_h, h_block},
                                      {anch_w, w_block},
                                      {anch_c * im_oc_block, oc_block}})});
                          }
                        }
                      }
                    }
                    // TODO(xurui): need to add iterated anchor here to
                    // support more fusion opportunity
                  }
                }
              }

              if (fusion && oc_threads == 1 && ic_threads == 1 && h_threads == 1
                && w_threads == 1) {
                fusion->create_output_fusion_anchor({blocking_output_
                    ? tensor_slice(output,
                      {{pbs, 1UL},
                        {outer_k * oc_ / im_oc_block / oc_split,
                          oc_ / im_oc_block / oc_split},
                        {0, oh_}, {0, ow_}, {0, im_oc_block}})
                    : tensor_slice(output,
                      {{pbs, 1UL}, {0, oh_}, {0, ow_},
                        {outer_k * oc_ / oc_split, oc_ / oc_split}})});
              }
            }

            if (fusion && oc_threads == 1 && h_threads == 1 && w_threads == 1) {
              fusion->create_output_fusion_anchor({blocking_output_
                  ? tensor_slice(output,
                    {{pbs, 1UL},
                      {outer_k * oc_ / im_oc_block / oc_split,
                        oc_ / im_oc_block / oc_split},
                      {0, oh_}, {0, ow_}, {0, im_oc_block}})
                  : tensor_slice(output,
                    {{pbs, 1UL}, {0, oh_}, {0, ow_},
                      {outer_k * oc_ / oc_split, oc_ / oc_split}})});
            }
          }
          if (fusion && h_threads == 1 && w_threads == 1) {
            fusion->create_output_fusion_anchor({blocking_output_
                ? tensor_slice(output,
                  {{pbs, 1UL},
                    {outer_k * oc_ / im_oc_block / oc_split,
                      oc_ / im_oc_block / oc_split},
                    {0, oh_}, {0, ow_}, {0, im_oc_block}})
                : tensor_slice(output,
                  {{pbs, 1UL}, {0, oh_}, {0, ow_},
                    {outer_k * oc_ / oc_split, oc_ / oc_split}})});
          }
        }

        if (fusion && h_threads == 1) {
          fusion->create_output_fusion_anchor({blocking_output_
              ? tensor_slice(output,
                {{pbs, 1UL},
                  {outer_k * oc_ / im_oc_block / oc_split,
                    oc_ / im_oc_block / oc_split},
                  {0, oh_}, {0, ow_}, {0, im_oc_block}})
              : tensor_slice(output,
                {{pbs, 1UL}, {0, oh_}, {0, ow_},
                  {outer_k * oc_ / oc_split, oc_ / oc_split}})});
        }
      }
      if (fusion && mb_ > 1) {
        fusion->create_output_fusion_anchor(
          {blocking_output_ ? tensor_slice(output,
             {{pbs, 1UL},
               {outer_k * oc_ / im_oc_block / oc_split,
                 oc_ / im_oc_block / oc_split},
               {0, oh_}, {0, ow_}, {0, im_oc_block}})
                            : tensor_slice(output,
                              {{pbs, 1UL}, {0, oh_}, {0, ow_},
                                {outer_k * oc_ / oc_split, oc_ / oc_split}})});
      }
    }
  }
  loops = {lpbs, lph, lpw, lpoc, lpic, lok};
}

void gen_nested_conv_fwd_t::single_thread_conv_padding_call(expr &output,
  const expr &input, const expr &weight, const expr &pbs, const expr &poc,
  const expr &ph, const expr &pw, const expr &pic, const expr &outer_k,
  const expr &h_num_block, const int h_num_block_pt, const expr &w_num_block,
  const int w_num_block_pt, const expr &oc_num_block, const int oc_num_block_pt,
  const expr &ic_num_block, const int ic_num_block_pt, const expr &pbuffer,
  for_loop &loh, for_loop &low, for_loop &looc, for_loop &loic, for_loop &lioc,
  for_loop &lih, for_loop &liw, const int oc_split, const int src_row_tile_size,
  const uint32_t lanes, const nested_conv_fwd_config_t &config,
  fusion_manager *fusion, const int ic_used_threads, const int oh_used_threads,
  const int ow_used_threads, const int y_unpad_top, const int y_unpad_bottom,
  const int y_unpad_left, const int y_unpad_right, const int iw_padded,
  const int kpack) const {
  auto h_block = config.h_block;
  auto w_block = config.w_block;
  auto im_h_block = config.im_h_block;
  auto im_w_block = config.im_w_block;

  auto ic_block = config.C_block;
  auto im_ic_block = config.im_ic_block;
  auto oc_block = config.K_block;
  auto im_oc_block = config.im_oc_block;

  auto dtypeInput = get_input_dtype();
  auto dtypeWeight = get_weight_dtype();
  auto dtypeOutput = get_output_dtype();

  auto LDA = blocking_input_ ? im_ic_block : ic_;
  auto LDC = blocking_output_ ? im_oc_block : oc_;

  expr n = pbs;
  _named_for_(loh, o_h, 0, h_num_block_pt) {
    _named_for_(low, o_w, 0, w_num_block_pt) {
      _named_for_(looc, o_oc, 0, oc_num_block_pt) {
        _named_for_(loic, o_ic, 0, ic_num_block_pt) {
          expr cond = o_h < h_num_block && o_w < w_num_block
            && o_oc < oc_num_block && o_ic < ic_num_block;
          _if_(cond) {
            _named_for_(lioc, i_oc, 0, oc_block / im_oc_block / oc_split) {
              expr oc = poc * oc_num_block_pt * oc_block / im_oc_block
                + o_oc * oc_block / im_oc_block
                + outer_k * oc_block / im_oc_block / oc_split + i_oc;
              _if_(oc * im_oc_block < oc_) {
                _named_for_(lih, i_h, 0, h_block / im_h_block) {
                  expr h = (ph * h_num_block_pt * h_block / im_h_block
                             + o_h * h_block / im_h_block + i_h)
                    * im_h_block;

                  _tensor_(A_list, datatypes::pointer, {kh_ * kw_});
                  _tensor_(B_list, datatypes::pointer, {kh_ * kw_});

                  // create a sub-tensor with maximum size which
                  // holds all the boundary
                  // that contains padding
                  _tensor_(
                    sub_tensor, dtypeInput, {kh_, src_row_tile_size, LDA});
                  _var_(pad_begin_index, datatypes::index);
                  _var_(pad_end_index, datatypes::index);
                  _var_(unpad_begin_index, datatypes::index);
                  _var_(unpad_end_index, datatypes::index);
                  _var_(real_pad_left, datatypes::u32);
                  _var_(real_pad_right, datatypes::u32);
                  _var_(num_pad_rows, datatypes::u32);
                  _var_(copy_width, datatypes::u32);

                  _named_for_(liw, i_w, 0, w_block / im_w_block) {
                    expr w = (pw * w_num_block_pt * w_block / im_w_block
                               + o_w * w_block / im_w_block + i_w)
                      * im_w_block;
                    _if_(w < ow_) {
                      _for_(im_h_i, 0, im_h_block) {
                        _if_(h + im_h_i < oh_) {
                          std::vector<expr> output_pos = blocking_output_
                            ? std::vector<expr> {pic * mb_ + n, oc, h + im_h_i,
                              w, 0}
                            : std::vector<expr> {
                              pic * mb_ + n, h + im_h_i, w, oc * im_oc_block};

                          if (ic_num_block_pt > 1) {
                            _if_(o_ic == 0) {
                              builtin::brgemm_init(
                                tensor_ptr(output, output_pos), im_w_block,
                                im_oc_block, LDC, dtypeOutput, 0);
                            }
                          } else {
                            builtin::brgemm_init(tensor_ptr(output, output_pos),
                              im_w_block, im_oc_block, LDC, dtypeOutput, 0);
                          }

                          _for_(i_c, 0, ic_block / im_ic_block) {
                            expr ic
                              = pic * ic_num_block_pt * ic_block / im_ic_block
                              + o_ic * ic_block / im_ic_block + i_c;
                            _if_(ic * im_ic_block < ic_) {
                              // 1) top or bottom region with
                              // padding inputs
                              // 1.1) calculate the number of
                              // padding rows
                              _if_(((h + im_h_i) >= y_unpad_top)
                                && ((h + im_h_i) <= y_unpad_bottom)) {
                                num_pad_rows = 0;
                                pad_begin_index = 0;
                                pad_end_index = 0;
                                unpad_begin_index = 0;
                                unpad_end_index = kh_;
                              }
                              _else_ {
                                _if_((h + im_h_i) < y_unpad_top) {
                                  num_pad_rows = builder::make_min(ph_b_
                                      - builder::make_cast(
                                          datatypes::u32, h + im_h_i)
                                        * sh_,
                                    kh_);
                                  pad_begin_index = 0;
                                  pad_end_index = num_pad_rows;
                                  unpad_begin_index = num_pad_rows;
                                  unpad_end_index = kh_;
                                }
                                _else_ {
                                  num_pad_rows = builder::make_min(
                                    builder::make_cast(
                                      datatypes::u32, h + im_h_i)
                                        * sh_
                                      + kh_ - (ih_ + ph_b_),
                                    kh_);
                                  pad_begin_index = kh_ - num_pad_rows;
                                  pad_end_index = kh_;
                                  unpad_begin_index = 0;
                                  unpad_end_index = kh_ - num_pad_rows;
                                }

                                // 1.2) Add zero-padding tensor to
                                // A_list
                                _for_(r, pad_begin_index, pad_end_index) {
                                  _for_(s, 0, kw_) {
                                    _var_(idx, datatypes::u32);
                                    idx = builder::make_cast(
                                      datatypes::u32, r * kw_ + s);
                                    A_list[idx] = tensor_ptr(pbuffer, {0, 0});
                                  }
                                }
                              }

                              // 1.3) copy sub-tensor and append
                              // to A_list
                              _if_(num_pad_rows < kh_) {
                                // 1.3.1) copy sub-tensor
                                _if_(w < y_unpad_left) {
                                  _if_((w + im_w_block - 1) <= y_unpad_right) {
                                    // 1.3.1.1) left pad only
                                    real_pad_left = pw_b_
                                      - builder::make_cast(
                                        datatypes::u32, w * sw_);

                                    // copy sub-tensor
                                    _for_(
                                      i, unpad_begin_index, unpad_end_index) {
                                      builtin::brgemm_init(
                                        tensor_ptr(sub_tensor,
                                          {i - unpad_begin_index, 0, 0}),
                                        builder::make_cast(
                                          datatypes::s32, real_pad_left),
                                        im_ic_block, LDA, dtypeInput, 0);

                                      // mapping dst to padding
                                      // src, then mapping padding
                                      // src to real src to get
                                      // the actual elements.
                                      _for_(
                                        j, real_pad_left, src_row_tile_size) {
                                        _for_(k, 0, im_ic_block, (int)lanes) {
                                          sub_tensor[span_t(
                                            {i - unpad_begin_index, j, k},
                                            lanes)]
                                            = input[blocking_input_
                                                ? span_t(
                                                  {n, ic,
                                                    (h + im_h_i) * sh_ + i
                                                      - ph_b_,
                                                    w * sw_ + j - pw_b_, k},
                                                  lanes)
                                                : span_t(
                                                  {n,
                                                    (h + im_h_i) * sh_ + i
                                                      - ph_b_,
                                                    w * sw_ + j - pw_b_,
                                                    ic * im_ic_block + k},
                                                  lanes)];
                                        }
                                      }
                                    }

                                    _for_(
                                      r, unpad_begin_index, unpad_end_index) {
                                      _for_(s, 0, kw_) {
                                        _var_(idx, datatypes::u32);
                                        idx = builder::make_cast(
                                          datatypes::u32, r * kw_ + s);
                                        A_list[idx] = tensor_ptr(sub_tensor,
                                          {r - unpad_begin_index, s, 0});
                                      }
                                    }
                                  }
                                  _else_ {
                                    // 1.3.1.2) both left and
                                    // right pad
                                    real_pad_left = pw_b_
                                      - builder::make_cast(
                                        datatypes::u32, w * sw_);
                                    real_pad_right
                                      = builder::make_cast(datatypes::u32,
                                          w * sw_ + src_row_tile_size)
                                      - (iw_padded - pw_e_);

                                    copy_width = src_row_tile_size
                                      - real_pad_left - real_pad_right;

                                    // copy sub-tensor
                                    _for_(
                                      i, unpad_begin_index, unpad_end_index) {
                                      // memzero left part
                                      builtin::brgemm_init(
                                        tensor_ptr(sub_tensor,
                                          {i - unpad_begin_index, 0, 0}),
                                        builder::make_cast(
                                          datatypes::s32, real_pad_left),
                                        im_ic_block, LDA, dtypeInput, 0);

                                      _for_(j, real_pad_left,
                                        copy_width + real_pad_left) {
                                        _for_(k, 0, im_ic_block, (int)lanes) {
                                          // N, C, H, W, c
                                          sub_tensor[span_t(
                                            {i - unpad_begin_index, j, k},
                                            lanes)]
                                            = input[blocking_input_
                                                ? span_t(
                                                  {n, ic,
                                                    (h + im_h_i) * sh_ + i
                                                      - ph_b_,
                                                    w * sw_ + j - pw_b_, k},
                                                  lanes)
                                                : span_t(
                                                  {n,
                                                    (h + im_h_i) * sh_ + i
                                                      - ph_b_,
                                                    w * sw_ + j - pw_b_,
                                                    ic * im_ic_block + k},
                                                  lanes)];
                                        }
                                      }

                                      builtin::brgemm_init(
                                        tensor_ptr(sub_tensor,
                                          {i - unpad_begin_index,
                                            copy_width + real_pad_left, 0}),
                                        builder::make_cast(
                                          datatypes::s32, real_pad_right),
                                        im_ic_block, LDA, dtypeInput, 0);
                                    }

                                    _for_(
                                      r, unpad_begin_index, unpad_end_index) {
                                      _for_(s, 0, kw_) {
                                        _var_(idx, datatypes::u32);
                                        idx = builder::make_cast(
                                          datatypes::u32, r * kw_ + s);
                                        A_list[idx] = tensor_ptr(sub_tensor,
                                          {r - unpad_begin_index, s, 0});
                                      }
                                    }
                                  }
                                }
                                _else_ {
                                  _if_((w + im_w_block - 1) <= y_unpad_right) {
                                    // 1.3.1.3) not using pad
                                    // buffer, use original buffer
                                    _for_(
                                      r, unpad_begin_index, unpad_end_index) {
                                      _for_(s, 0, kw_) {
                                        _var_(idx, datatypes::u32);
                                        idx = builder::make_cast(
                                          datatypes::u32, r * kw_ + s);
                                        A_list[idx] = tensor_ptr(input,
                                          blocking_input_
                                            ? std::vector<expr> {n, ic,
                                              (h + im_h_i) * sh_ + r - ph_b_,
                                              w * sw_ + s - pw_b_, 0}
                                            : std::vector<expr> {n,
                                              (h + im_h_i) * sh_ + r - ph_b_,
                                              w * sw_ + s - pw_b_,
                                              ic * im_ic_block});
                                      }
                                    }
                                  }
                                  _else_ {
                                    // 1.3.1.4) right pad only
                                    real_pad_right
                                      = builder::make_cast(datatypes::u32,
                                          w * sw_ + src_row_tile_size)
                                      - (iw_padded - pw_e_);
                                    copy_width
                                      = src_row_tile_size - real_pad_right;
                                    // copy sub-tensor

                                    _for_(
                                      i, unpad_begin_index, unpad_end_index) {
                                      _for_(j, 0, copy_width) {
                                        _for_(k, 0, im_ic_block, (int)lanes) {
                                          sub_tensor[span_t(
                                            {i - unpad_begin_index, j, k},
                                            lanes)]
                                            = input[blocking_input_
                                                ? span_t(
                                                  {n, ic,
                                                    (h + im_h_i) * sh_ + i
                                                      - ph_b_,
                                                    w * sw_ + j - pw_b_, k},
                                                  lanes)
                                                : span_t(
                                                  {n,
                                                    (h + im_h_i) * sh_ + i
                                                      - ph_b_,
                                                    w * sw_ + j - pw_b_,
                                                    ic * im_ic_block + k},
                                                  lanes)];
                                        }
                                      }
                                      builtin::brgemm_init(
                                        tensor_ptr(sub_tensor,
                                          {i - unpad_begin_index, copy_width,
                                            0}),
                                        builder::make_cast(
                                          datatypes::s32, real_pad_right),
                                        im_ic_block, LDA, dtypeInput, 0);
                                    }

                                    _for_(
                                      r, unpad_begin_index, unpad_end_index) {
                                      _for_(s, 0, kw_) {
                                        _var_(idx, datatypes::u32);
                                        idx = builder::make_cast(
                                          datatypes::u32, r * kw_ + s);
                                        A_list[idx] = tensor_ptr(sub_tensor,
                                          {r - unpad_begin_index, s, 0});
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
                                    idx = builder::make_cast(datatypes::u32,
                                      kh_ * kw_ - 1 - (r * kw_ + s));
                                  } else {
                                    idx = builder::make_cast(
                                      datatypes::u32, r * kw_ + s);
                                  }
                                  B_list[idx] = tensor_ptr(weight,
                                    kpack > 1
                                      ? std::vector<expr> {oc, ic, r, s, 0, 0,
                                        0}
                                      : std::vector<expr> {oc, ic, r, s, 0, 0});
                                }
                              }

                              const auto hint_A_size
                                = im_w_block * im_ic_block * kh_ * kw_;
                              const auto hint_B_size
                                = im_oc_block * im_ic_block;
                              const auto hint_C_size = im_w_block * im_oc_block;
                              sc_brgemm_attrs_t brg_attrs {
                                {brgemm::attr_key::max_bs, kh_ * kw_},
                                {brgemm::attr_key::hint_expected_A_size,
                                  hint_A_size},
                                {brgemm::attr_key::hint_expected_B_size,
                                  hint_B_size},
                                {brgemm::attr_key::hint_expected_C_size,
                                  hint_C_size},
                                {brgemm::attr_key::use_interleave_stores, true},
                                {brgemm::attr_key::use_uker, true}};

                              builtin::brgemm_list_update(A_list, B_list,
                                tensor_ptr(output, output_pos), 1, im_w_block,
                                im_oc_block, im_ic_block, sw_ * LDA,
                                im_oc_block, LDC, 1, 1, kh_ * kw_, dtypeInput,
                                dtypeWeight, brg_attrs);
                            }
                          }

                          if (fusion && ic_used_threads == 1
                            && ic_num_block_pt == 1) {
                            _if_(o_ic == (ic_num_block - 1)) {
                              fusion->create_output_fusion_anchor(
                                {blocking_output_
                                    ? tensor_slice(output,
                                      {{n, 1UL}, {oc, 1}, {h + im_h_i, 1},
                                        {w, im_w_block}, {0, im_oc_block}})
                                    : tensor_slice(output,
                                      {{n, 1UL}, {h + im_h_i, 1},
                                        {w, im_w_block},
                                        {oc * im_oc_block, im_oc_block}})});
                            }
                          }
                        } // im_h_i
                      }
                      if (fusion && ic_used_threads == 1
                        && ic_num_block_pt == 1) {
                        _if_(o_ic == (ic_num_block - 1)) {
                          fusion->create_output_fusion_anchor({blocking_output_
                              ? tensor_slice(output,
                                {{n, 1UL}, {oc, 1}, {h, im_h_block},
                                  {w, im_w_block}, {0, im_oc_block}})
                              : tensor_slice(output,
                                {{n, 1UL}, {h, im_h_block}, {w, im_w_block},
                                  {oc * im_oc_block, im_oc_block}})});
                        }
                      }
                    } // i_w
                  }

                  if (fusion && ic_used_threads == 1 && ic_num_block_pt == 1
                    && w_block * ow_used_threads == ow_) {
                    _if_(o_ic == (ic_num_block - 1)) {
                      expr anch_w = (pw * w_num_block_pt * w_block / im_w_block
                                      + o_w * w_block / im_w_block)
                        * im_w_block;
                      fusion->create_output_fusion_anchor({blocking_output_
                          ? tensor_slice(output,
                            {{n, 1UL}, {oc, 1}, {h, im_h_block},
                              {anch_w, w_block}, {0, im_oc_block}})
                          : tensor_slice(output,
                            {{n, 1UL}, {h, im_h_block}, {anch_w, w_block},
                              {oc * im_oc_block, im_oc_block}})});
                    }
                  }
                } // i_h

                if (fusion && ic_used_threads == 1 && ic_num_block_pt == 1
                  && h_block * oh_used_threads == oh_
                  && w_block * ow_used_threads == ow_) {
                  _if_(o_ic == (ic_num_block - 1)) {
                    expr anch_h = (ph * h_num_block_pt * h_block / im_h_block
                                    + o_h * h_block / im_h_block)
                      * im_h_block;
                    expr anch_w = (pw * w_num_block_pt * w_block / im_w_block
                                    + o_w * w_block / im_w_block)
                      * im_w_block;
                    fusion->create_output_fusion_anchor({blocking_output_
                        ? tensor_slice(output,
                          {{n, 1UL}, {oc, 1}, {anch_h, h_block},
                            {anch_w, w_block}, {0, im_oc_block}})
                        : tensor_slice(output,
                          {{n, 1UL}, {anch_h, h_block}, {anch_w, w_block},
                            {oc * im_oc_block, im_oc_block}})});
                  }
                }
              } // ioc
            }

            if (fusion && ic_used_threads == 1 && ic_num_block_pt == 1
              && h_block * oh_used_threads == oh_
              && w_block * ow_used_threads == ow_) {
              _if_(o_ic == (ic_num_block - 1)) {
                expr anch_h = (ph * h_num_block_pt * h_block / im_h_block
                                + o_h * h_block / im_h_block)
                  * im_h_block;
                expr anch_w = (pw * w_num_block_pt * w_block / im_w_block
                                + o_w * w_block / im_w_block)
                  * im_w_block;
                expr anch_oc = poc * oc_num_block_pt * oc_block / im_oc_block
                  + o_oc * oc_block / im_oc_block
                  + outer_k * oc_block / im_oc_block / oc_split;
                fusion->create_output_fusion_anchor({blocking_output_
                    ? tensor_slice(output,
                      {{n, 1UL}, {anch_oc, 1}, {anch_h, h_block},
                        {anch_w, w_block}, {0, im_oc_block}})
                    : tensor_slice(output,
                      {{n, 1UL}, {anch_h, h_block}, {anch_w, w_block},
                        {anch_oc * im_oc_block, im_oc_block}})});
              }
            } // o_ic
          }
        }
      }
    }
  }
}

void gen_nested_conv_fwd_t::single_thread_dynamic_conv_padding_call(
  expr &output, const expr &input, const expr &weight, const expr &pbs,
  const expr &poc, const expr &ph, const expr &pw, const expr &pic,
  const expr &outer_k, const expr &h_num_block, const expr &h_num_block_pt,
  const expr &w_num_block, const expr &w_num_block_pt, const expr &oc_num_block,
  const int oc_num_block_pt, const expr &ic_num_block,
  const int ic_num_block_pt, const expr &pbuffer, for_loop &loh, for_loop &low,
  for_loop &looc, for_loop &loic, for_loop &lioc, for_loop &lih, for_loop &liw,
  const int oc_split, const expr &src_row_tile_size, const uint32_t lanes,
  const nested_conv_fwd_config_t &config, fusion_manager *fusion,
  const int ic_used_threads, const int oc_used_threads,
  const expr &oh_used_threads, const expr &ow_used_threads,
  const expr &y_unpad_top, const expr &y_unpad_bottom, const expr &y_unpad_left,
  const expr &y_unpad_right, const expr &iw_padded, const int kpack,
  const expr &h_block, const expr &w_block, const expr &im_h_block,
  const expr &im_w_block, const expr &oh_expr_, const expr &ow_expr_,
  const expr &ih_expr_, const expr &iw_expr_, expr &cond_tail_h,
  expr &cond_tail_w, int oc_block, int ic_block) const {
  auto im_ic_block = config.im_ic_block;
  auto im_oc_block = config.im_oc_block;

  auto dtypeInput = get_input_dtype();
  auto dtypeWeight = get_weight_dtype();
  auto dtypeOutput = get_output_dtype();

  auto LDA = blocking_input_ ? im_ic_block : ic_;
  auto LDC = blocking_output_ ? im_oc_block : oc_;

  expr n = pbs;
  _named_for_(loh, o_h, 0, h_num_block_pt) {
    _named_for_(low, o_w, 0, w_num_block_pt) {
      _named_for_(looc, o_oc, 0, oc_num_block_pt) {
        _named_for_(loic, o_ic, 0, ic_num_block_pt) {
          expr cond = o_h < h_num_block && o_w < w_num_block
            && o_oc < oc_num_block && o_ic < ic_num_block;
          _if_(cond) {
            _named_for_(lioc, i_oc, 0, oc_block / im_oc_block / oc_split) {
              expr oc = poc * oc_num_block_pt * oc_block / im_oc_block
                + o_oc * oc_block / im_oc_block
                + outer_k * oc_block / im_oc_block / oc_split + i_oc;
              _if_(oc * im_oc_block < oc_) {
                _named_for_(
                  lih, i_h, 0, (h_block + im_h_block - 1) / im_h_block) {
                  expr h = ph * h_num_block_pt * h_block + o_h * h_block
                    + i_h * im_h_block;
                  expr is_tail_h = cond_tail_h && (h + im_h_block > oh_expr_);
                  expr real_im_h_block = builder::make_select(is_tail_h,
                    builder::make_cast(datatypes::s32, oh_expr_ % im_h_block),
                    im_h_block);
                  _tensor_(A_list, datatypes::pointer, {kh_ * kw_});
                  _tensor_(B_list, datatypes::pointer, {kh_ * kw_});

                  _named_for_(
                    liw, i_w, 0, (w_block + im_w_block - 1) / im_w_block) {
                    expr w = pw * w_num_block_pt * w_block + o_w * w_block
                      + i_w * im_w_block;
                    _if_(w < ow_expr_) {
                      expr is_tail_w
                        = cond_tail_w && (w + im_w_block > ow_expr_);
                      expr real_im_w_block = builder::make_select(is_tail_w,
                        builder::make_cast(
                          datatypes::s32, ow_expr_ % im_w_block),
                        im_w_block);
                      expr real_src_row_tile_size
                        = builder::make_select(is_tail_w,
                          builder::make_cast(
                            datatypes::s32, (ow_expr_ - w - 1) * sw_ + kw_),
                          src_row_tile_size);

                      _for_(im_h_i, 0, im_h_block) {
                        _if_(h + im_h_i < oh_expr_) {
                          // create a sub-tensor with maximum size
                          // which holds all the boundary that
                          // contains padding
                          _tensor_(sub_tensor, dtypeInput,
                            {kh_, real_src_row_tile_size, LDA});

                          _var_(pad_begin_index, datatypes::index);
                          _var_(pad_end_index, datatypes::index);
                          _var_(unpad_begin_index, datatypes::index);
                          _var_(unpad_end_index, datatypes::index);
                          _var_(real_pad_left, datatypes::index);
                          _var_(real_pad_right, datatypes::index);
                          _var_(num_pad_rows, datatypes::index);
                          _var_(copy_width, datatypes::index);
                          std::vector<expr> output_pos = blocking_output_
                            ? std::vector<expr> {pic * mb_ + n, oc, h + im_h_i,
                              w, 0}
                            : std::vector<expr> {
                              pic * mb_ + n, h + im_h_i, w, oc * im_oc_block};

                          if (ic_num_block_pt > 1) {
                            _if_(o_ic == 0) {
                              builtin::brgemm_init(
                                tensor_ptr(output, output_pos), real_im_w_block,
                                im_oc_block, LDC, dtypeOutput, 0);
                            }
                          } else {
                            builtin::brgemm_init(tensor_ptr(output, output_pos),
                              real_im_w_block, im_oc_block, LDC, dtypeOutput,
                              0);
                          }

                          _for_(i_c, 0, ic_block / im_ic_block) {
                            expr ic
                              = pic * ic_num_block_pt * ic_block / im_ic_block
                              + o_ic * ic_block / im_ic_block + i_c;
                            _if_(ic * im_ic_block < ic_) {
                              // 1) top or bottom region with
                              // padding inputs
                              // 1.1) calculate the number of
                              // padding rows
                              _if_(((h + im_h_i) >= y_unpad_top)
                                && ((h + im_h_i) <= y_unpad_bottom)) {
                                num_pad_rows = 0;
                                pad_begin_index = 0;
                                pad_end_index = 0;
                                unpad_begin_index = 0;
                                unpad_end_index = kh_;
                              }
                              _else_ {
                                _if_((h + im_h_i) < y_unpad_top) {
                                  num_pad_rows = builder::make_min(ph_b_
                                      - builder::make_cast(
                                          datatypes::u32, h + im_h_i)
                                        * sh_,
                                    kh_);
                                  pad_begin_index = 0;
                                  pad_end_index = num_pad_rows;
                                  unpad_begin_index = num_pad_rows;
                                  unpad_end_index = kh_;
                                }
                                _else_ {
                                  num_pad_rows = builder::make_min(
                                    builder::make_cast(
                                      datatypes::u32, h + im_h_i)
                                        * sh_
                                      + kh_ - (ih_expr_ + ph_b_),
                                    kh_);
                                  pad_begin_index = kh_ - num_pad_rows;
                                  pad_end_index = kh_;
                                  unpad_begin_index = 0;
                                  unpad_end_index = kh_ - num_pad_rows;
                                }

                                // 1.2) Add zero-padding tensor to
                                // A_list
                                _for_(r, pad_begin_index, pad_end_index) {
                                  _for_(s, 0, kw_) {
                                    _var_(idx, datatypes::u32);
                                    idx = builder::make_cast(
                                      datatypes::u32, r * kw_ + s);
                                    A_list[idx] = tensor_ptr(pbuffer, {0, 0});
                                  }
                                }
                              }

                              // 1.3) copy sub-tensor and append
                              // to A_list
                              _if_(num_pad_rows < kh_) {
                                // 1.3.1) copy sub-tensor
                                _if_(w < y_unpad_left) {
                                  _if_((w + real_im_w_block - 1)
                                    <= y_unpad_right) {
                                    // 1.3.1.1) left pad only
                                    real_pad_left = pw_b_
                                      - builder::make_cast(
                                        datatypes::u32, w * sw_);

                                    // copy sub-tensor
                                    _for_(
                                      i, unpad_begin_index, unpad_end_index) {
                                      builtin::brgemm_init(
                                        tensor_ptr(sub_tensor,
                                          {i - unpad_begin_index, 0, 0}),
                                        builder::make_cast(
                                          datatypes::s32, real_pad_left),
                                        im_ic_block, LDA, dtypeInput, 0);

                                      // mapping dst to padding
                                      // src, then mapping padding
                                      // src to real src to get
                                      // the actual elements.
                                      _for_(j, real_pad_left,
                                        real_src_row_tile_size) {
                                        _for_(k, 0, im_ic_block, (int)lanes) {
                                          sub_tensor[span_t(
                                            {i - unpad_begin_index, j, k},
                                            lanes)]
                                            = input[blocking_input_
                                                ? span_t(
                                                  {n, ic,
                                                    (h + im_h_i) * sh_ + i
                                                      - ph_b_,
                                                    w * sw_ + j - pw_b_, k},
                                                  lanes)
                                                : span_t(
                                                  {n,
                                                    (h + im_h_i) * sh_ + i
                                                      - ph_b_,
                                                    w * sw_ + j - pw_b_,
                                                    ic * im_ic_block + k},
                                                  lanes)];
                                        }
                                      }
                                    }

                                    _for_(
                                      r, unpad_begin_index, unpad_end_index) {
                                      _for_(s, 0, kw_) {
                                        _var_(idx, datatypes::u32);
                                        idx = builder::make_cast(
                                          datatypes::u32, r * kw_ + s);
                                        A_list[idx] = tensor_ptr(sub_tensor,
                                          {r - unpad_begin_index, s, 0});
                                      }
                                    }
                                  }
                                  _else_ {
                                    // 1.3.1.2) both left and
                                    // right pad
                                    real_pad_left = pw_b_
                                      - builder::make_cast(
                                        datatypes::u32, w * sw_);
                                    real_pad_right
                                      = builder::make_cast(datatypes::u32,
                                          w * sw_ + real_src_row_tile_size)
                                      - (iw_padded - pw_e_);

                                    copy_width = real_src_row_tile_size
                                      - real_pad_left - real_pad_right;

                                    // copy sub-tensor
                                    _for_(
                                      i, unpad_begin_index, unpad_end_index) {
                                      // memzero left part
                                      builtin::brgemm_init(
                                        tensor_ptr(sub_tensor,
                                          {i - unpad_begin_index, 0, 0}),
                                        builder::make_cast(
                                          datatypes::s32, real_pad_left),
                                        im_ic_block, LDA, dtypeInput, 0);

                                      _for_(j, real_pad_left,
                                        copy_width + real_pad_left) {
                                        _for_(k, 0, im_ic_block, (int)lanes) {
                                          // N, C, H, W, c
                                          sub_tensor[span_t(
                                            {i - unpad_begin_index, j, k},
                                            lanes)]
                                            = input[blocking_input_
                                                ? span_t(
                                                  {n, ic,
                                                    (h + im_h_i) * sh_ + i
                                                      - ph_b_,
                                                    w * sw_ + j - pw_b_, k},
                                                  lanes)
                                                : span_t(
                                                  {n,
                                                    (h + im_h_i) * sh_ + i
                                                      - ph_b_,
                                                    w * sw_ + j - pw_b_,
                                                    ic * im_ic_block + k},
                                                  lanes)];
                                        }
                                      }

                                      builtin::brgemm_init(
                                        tensor_ptr(sub_tensor,
                                          {i - unpad_begin_index,
                                            copy_width + real_pad_left, 0}),
                                        builder::make_cast(
                                          datatypes::s32, real_pad_right),
                                        im_ic_block, LDA, dtypeInput, 0);
                                    }

                                    _for_(
                                      r, unpad_begin_index, unpad_end_index) {
                                      _for_(s, 0, kw_) {
                                        _var_(idx, datatypes::u32);
                                        idx = builder::make_cast(
                                          datatypes::u32, r * kw_ + s);
                                        A_list[idx] = tensor_ptr(sub_tensor,
                                          {r - unpad_begin_index, s, 0});
                                      }
                                    }
                                  }
                                }
                                _else_ {
                                  _if_((w + real_im_w_block - 1)
                                    <= y_unpad_right) {
                                    // 1.3.1.3) not using pad
                                    // buffer, use original buffer
                                    _for_(
                                      r, unpad_begin_index, unpad_end_index) {
                                      _for_(s, 0, kw_) {
                                        _var_(idx, datatypes::u32);
                                        idx = builder::make_cast(
                                          datatypes::u32, r * kw_ + s);
                                        A_list[idx] = tensor_ptr(input,
                                          blocking_input_
                                            ? std::vector<expr> {n, ic,
                                              (h + im_h_i) * sh_ + r - ph_b_,
                                              w * sw_ + s - pw_b_, 0}
                                            : std::vector<expr> {n,
                                              (h + im_h_i) * sh_ + r - ph_b_,
                                              w * sw_ + s - pw_b_,
                                              ic * im_ic_block});
                                      }
                                    }
                                  }
                                  _else_ {
                                    // 1.3.1.4) right pad only
                                    real_pad_right = builder::make_min(
                                      builder::make_cast(datatypes::u32,
                                        w * sw_ + real_src_row_tile_size)
                                        - (iw_padded - pw_e_),
                                      real_src_row_tile_size);
                                    copy_width
                                      = real_src_row_tile_size - real_pad_right;
                                    // copy sub-tensor

                                    _for_(
                                      i, unpad_begin_index, unpad_end_index) {
                                      _for_(j, 0, copy_width) {
                                        _for_(k, 0, im_ic_block, (int)lanes) {
                                          sub_tensor[span_t(
                                            {i - unpad_begin_index, j, k},
                                            lanes)]
                                            = input[blocking_input_
                                                ? span_t(
                                                  {n, ic,
                                                    (h + im_h_i) * sh_ + i
                                                      - ph_b_,
                                                    w * sw_ + j - pw_b_, k},
                                                  lanes)
                                                : span_t(
                                                  {n,
                                                    (h + im_h_i) * sh_ + i
                                                      - ph_b_,
                                                    w * sw_ + j - pw_b_,
                                                    ic * im_ic_block + k},
                                                  lanes)];
                                        }
                                      }
                                      builtin::brgemm_init(
                                        tensor_ptr(sub_tensor,
                                          {i - unpad_begin_index, copy_width,
                                            0}),
                                        builder::make_cast(
                                          datatypes::s32, real_pad_right),
                                        im_ic_block, LDA, dtypeInput, 0);
                                    }

                                    _for_(
                                      r, unpad_begin_index, unpad_end_index) {
                                      _for_(s, 0, kw_) {
                                        _var_(idx, datatypes::u32);
                                        idx = builder::make_cast(
                                          datatypes::u32, r * kw_ + s);
                                        A_list[idx] = tensor_ptr(sub_tensor,
                                          {r - unpad_begin_index, s, 0});
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
                                    idx = builder::make_cast(datatypes::u32,
                                      kh_ * kw_ - 1 - (r * kw_ + s));
                                  } else {
                                    idx = builder::make_cast(
                                      datatypes::u32, r * kw_ + s);
                                  }
                                  B_list[idx] = tensor_ptr(weight,
                                    kpack > 1
                                      ? std::vector<expr> {oc, ic, r, s, 0, 0,
                                        0}
                                      : std::vector<expr> {oc, ic, r, s, 0, 0});
                                }
                              }

                              brgemm::attrs_setting_t::attrs_map_t range_attr_0
                                = {brgemm::attr_key::M_range_upper_bound, 64};
                              sc_brgemm_attrs_t brg_attrs = sc_brgemm_attrs_t {
                                {brgemm::attr_key::max_bs, kh_ * kw_},
                                {brgemm::attr_key::use_interleave_stores, true},
                                {brgemm::attr_key::use_uker, true},
                                range_attr_0};

                              builtin::brgemm_list_update(A_list, B_list,
                                tensor_ptr(output, output_pos), 1,
                                real_im_w_block, im_oc_block, im_ic_block,
                                sw_ * LDA, im_oc_block, LDC, 1, 1, kh_ * kw_,
                                dtypeInput, dtypeWeight, brg_attrs);
                            }
                          }
                          if (fusion && ic_used_threads == 1
                            && ic_num_block_pt == 1) {
                            _if_(o_ic == (ic_num_block - 1)) {
                              fusion->create_output_fusion_anchor(
                                {blocking_output_
                                    ? tensor_slice(output,
                                      {{n, 1}, {oc, 1}, {h + im_h_i, 1},
                                        {w, real_im_w_block}, {0, im_oc_block}})
                                    : tensor_slice(output,
                                      {{n, 1UL}, {h + im_h_i, 1},
                                        {w, real_im_w_block},
                                        {oc * im_oc_block, im_oc_block}})});
                            }
                          } // im_h_i
                        }
                      }
                      if (fusion && ic_used_threads == 1
                        && ic_num_block_pt == 1) {
                        _if_(o_ic == (ic_num_block - 1)) {
                          fusion->create_output_fusion_anchor({blocking_output_
                              ? tensor_slice(output,
                                {{n, 1UL}, {oc, 1}, {h, real_im_h_block},
                                  {w, real_im_w_block}, {0, im_oc_block}})
                              : tensor_slice(output,
                                {{n, 1UL}, {h, real_im_h_block},
                                  {w, real_im_w_block},
                                  {oc * im_oc_block, im_oc_block}})});
                        }
                      } // i_w
                    }
                  }

                  if (fusion && ic_used_threads == 1 && ic_num_block_pt == 1
                    && !is_dynamic_dim(ow_)
                    && get_expr_as_int(w_block)
                        * get_expr_as_int(ow_used_threads)
                      == ow_) {
                    _if_(o_ic == (ic_num_block - 1)) {
                      expr anch_w
                        = pw * w_num_block_pt * w_block + o_w * w_block;
                      fusion->create_output_fusion_anchor({blocking_output_
                          ? tensor_slice(output,
                            {{n, 1UL}, {oc, 1}, {h, real_im_h_block},
                              {anch_w, w_block}, {0, im_oc_block}})
                          : tensor_slice(output,
                            {{n, 1UL}, {h, real_im_h_block}, {anch_w, w_block},
                              {oc * im_oc_block, im_oc_block}})});
                    }
                  }
                } // i_h

                if (fusion && ic_used_threads == 1 && ic_num_block_pt == 1
                  && !is_dynamic_dim(ow_) && !is_dynamic_dim(oh_)
                  && get_expr_as_int(w_block) * get_expr_as_int(ow_used_threads)
                    == ow_
                  && get_expr_as_int(h_block) * get_expr_as_int(oh_used_threads)
                    == oh_) {
                  _if_(o_ic == (ic_num_block - 1)) {
                    expr anch_h = ph * h_num_block_pt * h_block + o_h * h_block;
                    expr anch_w = pw * w_num_block_pt * w_block + o_w * w_block;
                    fusion->create_output_fusion_anchor({blocking_output_
                        ? tensor_slice(output,
                          {{n, 1UL}, {oc, 1}, {anch_h, h_block},
                            {anch_w, w_block}, {0, im_oc_block}})
                        : tensor_slice(output,
                          {{n, 1UL}, {anch_h, h_block}, {anch_w, w_block},
                            {oc * im_oc_block, im_oc_block}})});
                  }
                } // i_oc
              }
            }

            if (fusion && ic_used_threads == 1 && ic_num_block_pt == 1
              && !is_dynamic_dim(ow_) && !is_dynamic_dim(oh_)
              && get_expr_as_int(w_block) * get_expr_as_int(ow_used_threads)
                == ow_
              && get_expr_as_int(h_block) * get_expr_as_int(oh_used_threads)
                == oh_
              && oc_block * oc_used_threads == oc_) {
              _if_(o_ic == (ic_num_block - 1)) {
                expr anch_h = ph * h_num_block_pt * h_block + o_h * h_block;
                expr anch_w = pw * w_num_block_pt * w_block + o_w * w_block;
                expr anch_oc = poc * oc_num_block_pt * oc_block / im_oc_block
                  + o_oc * oc_block / im_oc_block
                  + outer_k * oc_block / im_oc_block / oc_split;
                fusion->create_output_fusion_anchor({blocking_output_
                    ? tensor_slice(output,
                      {{n, 1UL}, {anch_oc, 1}, {anch_h, h_block},
                        {anch_w, w_block}, {0, im_oc_block}})
                    : tensor_slice(output,
                      {{n, 1UL}, {anch_h, h_block}, {anch_w, w_block},
                        {anch_oc * im_oc_block, im_oc_block}})});
              }
            }
          }
        }
      }
    }
  }
}
void gen_nested_conv_fwd_t::dynamic_compute_conv_padding_nested(
  CONV_ARG_LIST) const {
  int num_threads = runtime_config_t::get().get_num_threads();
  int h_threads = config.h_threads;
  int w_threads = config.w_threads;
  int oc_threads = config.oc_threads;
  int ic_threads = 1;

  int bs_threads = num_threads / h_threads / oc_threads;
  int oc_block = oc_ / oc_threads;
  int ic_block = ic_ / ic_threads;
  int im_oc_block = config.im_oc_block;
  int im_ic_block = config.im_ic_block;
  int im_w_block = config.im_w_block;

  COMPILE_ASSERT(oc_block % im_oc_block == 0,
    "oc_block % im_oc_block != 0, config is invalid")
  COMPILE_ASSERT(ic_block % im_ic_block == 0,
    "ic_block % im_ic_block != 0, config is invalid")
  // param
  expr output_tmp = output;
  auto tinput = in_tensors_[0];
  auto tweight = in_tensors_[1];
  auto toutput = out_tensors_[0];
  const auto &input_blocking_dims = tinput.get_blocking_dims();
  const auto &weight_blocking_dims = tweight.get_blocking_dims();
  const auto &output_blocking_dims = toutput.get_blocking_dims();

  for_loop lpbs, lph, lpw, lpoc, lpic, loh, low, looc, loic, lioc, lih, liw,
    lok;

  int oc_num_block_pt, oc_tail_num_block_pt, ic_num_block_pt,
    ic_tail_num_block_pt;

  int oc_used_threads = block_split(utils::divide_and_ceil(oc_, oc_block),
    oc_threads, oc_num_block_pt, oc_tail_num_block_pt);

  auto input_expr_dims = input.checked_as<tensor>()->dims_;
  auto mb_expr_ = input_expr_dims[0];
  auto ih_expr_
    = input_expr_dims.size() == 4 ? input_expr_dims[1] : input_expr_dims[2];
  auto iw_expr_
    = input_expr_dims.size() == 4 ? input_expr_dims[2] : input_expr_dims[3];

  auto oh_expr_ = input_expr_dims.size() == 4
    ? (input_expr_dims[1] + (ph_b_ + ph_e_) - kh_) / sh_ + 1
    : (input_expr_dims[2] + (ph_b_ + ph_e_) - kh_) / sh_ + 1;
  auto ow_expr_ = input_expr_dims.size() == 4
    ? (input_expr_dims[2] + (pw_b_ + pw_e_) - kw_) / sw_ + 1
    : (input_expr_dims[3] + (pw_b_ + pw_e_) - kw_) / sw_ + 1;

  // by observation
  expr im_h_block = do_cast_and_fold(
    builder::make_select(oh_expr_ <= 14 && ow_expr_ <= 14 && h_threads == 1,
      builder::make_cast(datatypes::s32, oh_expr_), config.im_h_block));

  expr h_block
    = do_cast_and_fold(builder::make_select(oh_expr_ % h_threads == 0,
      builder::make_cast(datatypes::s32, oh_expr_ / h_threads),
      config.im_h_block));
  expr w_block = do_cast_and_fold((ow_expr_ + w_threads - 1) / w_threads);

  expr h_num_block_pt
    = divide_and_ceil(divide_and_ceil(oh_expr_, h_block), h_threads);
  expr h_tail_num_block_pt = builder::make_select(
    divide_and_ceil(oh_expr_, h_block) % h_num_block_pt == 0, h_num_block_pt,
    divide_and_ceil(oh_expr_, h_block) % h_num_block_pt);
  expr oh_used_threads
    = divide_and_ceil(divide_and_ceil(oh_expr_, h_block), h_num_block_pt);

  expr ow_used_threads = do_cast_and_fold((ow_expr_ + w_block - 1) / w_block);
  expr w_num_block_pt = ow_used_threads / w_threads;
  expr w_tail_num_block_pt
    = builder::make_select(ow_used_threads % w_num_block_pt == 0,
      w_num_block_pt, ow_used_threads % w_num_block_pt);

  int ic_used_threads = block_split(utils::divide_and_ceil(ic_, ic_block),
    ic_threads, ic_num_block_pt, ic_tail_num_block_pt);

  if (ic_used_threads > 1) {
    // barrier
    // output temp buffer
    auto out_dims = output_blocking_dims;
    out_dims[0] *= ic_used_threads;
    _tensor_(out_tmp, toutput.dtype_, dims_to_expr(out_dims));
    output_tmp = out_tmp;
  }
  expr ih_padded = ih_expr_ + (ph_b_ + ph_e_),
       iw_padded = iw_expr_ + (pw_b_ + pw_e_);
  auto dtypeInput = get_input_dtype();
  uint32_t lanes = get_lanes(ctx, im_ic_block, dtypeInput);

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
  auto get_num_pad_end = [](const expr &ip, int k, int s, int p) {
    expr remaining = (ip - k) % s;
    if (remaining.isa<constant>() && get_expr_as_int(remaining)) {
      return expr(utils::divide_and_ceil(p, s));
    }
    return builder::make_select(p > remaining,
      builder::make_cast(datatypes::s32, (p - remaining + s - 1) / s), expr(0));
  };

  const int dst_num_pad_top = utils::divide_and_ceil(ph_b_, sh_);
  const int dst_num_pad_left = utils::divide_and_ceil(pw_b_, sw_);
  expr dst_num_pad_bottom = get_num_pad_end(ih_padded, kh_, sh_, ph_e_);
  expr dst_num_pad_right = get_num_pad_end(iw_padded, kw_, sw_, pw_e_);

  int y_unpad_top = dst_num_pad_top;
  expr y_unpad_bottom = oh_expr_ - dst_num_pad_bottom - 1;
  int y_unpad_left = dst_num_pad_left;
  expr y_unpad_right = ow_expr_ - dst_num_pad_right - 1;

  auto weight_size
    = math_utils::get_dims_product(in_tensors_[1].get_blocking_dims())
    * utils::get_sizeof_type(get_weight_dtype());
  auto L2_cache_size = ctx->machine_.cpu_flags_.getDCacheSize(2);
  int oc_split = (oc_threads == 1 && oc_num_block_pt == 1)
    ? get_oc_split_factor(
      -1, weight_size, L2_cache_size, oc_block / im_oc_block)
    : 1;

  // create a global shared zero-buffer referenced by padding
  expr src_row_tile_size
    = builder::make_select(im_w_block <= ow_expr_, (im_w_block - 1) * sw_ + kw_,
      builder::make_cast(datatypes::s32, (ow_expr_ - 1) * sw_ + kw_));

  auto LDA = blocking_input_ ? im_ic_block : ic_;
  _tensor_(pbuffer, dtypeInput, {src_row_tile_size, LDA});
  builtin::mem_zero(pbuffer, src_row_tile_size * LDA, dtypeInput);
  expr cond_tail_w = w_block % im_w_block != 0 || ow_expr_ % im_w_block != 0;
  expr cond_tail_h = h_block % im_h_block != 0;
  // will update template for parallel merge in dynamic conv block
  _named_for_(lok, outer_k, 0, oc_split, 1, for_type::PARALLEL) {
    _named_for_(lpbs, pbs, 0, mb_expr_, 1, for_type::PARALLEL) {
      _named_for_(lph, ph, 0, h_threads, 1) {
        _named_for_(lpw, pw, 0, w_threads, 1) {
          _named_for_(lpoc, poc, 0, oc_threads, 1) {
            _named_for_(lpic, pic, 0, ic_threads, 1) {
              expr h_num_block
                = builder::make_select(ph < (oh_used_threads - 1),
                  h_num_block_pt, h_tail_num_block_pt),
                w_num_block = builder::make_select(pw < (ow_used_threads - 1),
                  w_num_block_pt, w_tail_num_block_pt),
                oc_num_block = builder::make_select(poc < (oc_used_threads - 1),
                  oc_num_block_pt, oc_tail_num_block_pt);

              _if_(ph < oh_used_threads && pw < ow_used_threads
                && poc < oc_used_threads && pic < ic_used_threads) {
                // single core
                expr ic_num_block
                  = builder::make_select(pic < (ic_used_threads - 1),
                    ic_num_block_pt, ic_tail_num_block_pt);

                single_thread_dynamic_conv_padding_call(output, input, weight,
                  pbs, poc, ph, pw, pic, outer_k, h_num_block, h_num_block_pt,
                  w_num_block, w_num_block_pt, oc_num_block, oc_num_block_pt,
                  ic_num_block, ic_num_block_pt, pbuffer, loh, low, looc, loic,
                  lioc, lih, liw, oc_split, src_row_tile_size, lanes, config,
                  fusion, ic_used_threads, oc_used_threads, oh_used_threads,
                  ow_used_threads, y_unpad_top, y_unpad_bottom, y_unpad_left,
                  y_unpad_right, iw_padded, kpack, h_block, w_block, im_h_block,
                  im_w_block, oh_expr_, ow_expr_, ih_expr_, iw_expr_,
                  cond_tail_h, cond_tail_w, oc_block, ic_block);
              }

              if (fusion && oc_threads == 1 && ic_threads == 1 && h_threads == 1
                && w_threads == 1 && !is_dynamic_dim(ow_)
                && !is_dynamic_dim(oh_)) {
                fusion->create_output_fusion_anchor({blocking_output_
                    ? tensor_slice(output,
                      {{pbs, 1UL},
                        {outer_k * oc_ / im_oc_block / oc_split,
                          oc_ / im_oc_block / oc_split},
                        {0, oh_}, {0, ow_}, {0, im_oc_block}})
                    : tensor_slice(output,
                      {{pbs, 1UL}, {0, oh_}, {0, ow_},
                        {outer_k * oc_ / oc_split, oc_ / oc_split}})});
              }
            }

            if (fusion && oc_threads == 1 && h_threads == 1 && w_threads == 1
              && !is_dynamic_dim(ow_) && !is_dynamic_dim(oh_)) {
              fusion->create_output_fusion_anchor({blocking_output_
                  ? tensor_slice(output,
                    {{pbs, 1UL},
                      {outer_k * oc_ / im_oc_block / oc_split,
                        oc_ / im_oc_block / oc_split},
                      {0, oh_}, {0, ow_}, {0, im_oc_block}})
                  : tensor_slice(output,
                    {{pbs, 1UL}, {0, oh_}, {0, ow_},
                      {outer_k * oc_ / oc_split, oc_ / oc_split}})});
            }
          }
          if (fusion && h_threads == 1 && w_threads == 1 && !is_dynamic_dim(ow_)
            && !is_dynamic_dim(oh_)) {
            fusion->create_output_fusion_anchor({blocking_output_
                ? tensor_slice(output,
                  {{pbs, 1UL},
                    {outer_k * oc_ / im_oc_block / oc_split,
                      oc_ / im_oc_block / oc_split},
                    {0, oh_expr_}, {0, ow_expr_}, {0, im_oc_block}})
                : tensor_slice(output,
                  {{pbs, 1UL}, {0, oh_expr_}, {0, ow_expr_},
                    {outer_k * oc_ / oc_split, oc_ / oc_split}})});
          }
        }

        if (fusion && h_threads == 1 && !is_dynamic_dim(ow_)
          && !is_dynamic_dim(oh_)) {
          fusion->create_output_fusion_anchor({blocking_output_
              ? tensor_slice(output,
                {{pbs, 1UL},
                  {outer_k * oc_ / im_oc_block / oc_split,
                    oc_ / im_oc_block / oc_split},
                  {0, oh_}, {0, ow_}, {0, im_oc_block}})
              : tensor_slice(output,
                {{pbs, 1UL}, {0, oh_}, {0, ow_},
                  {outer_k * oc_ / oc_split, oc_ / oc_split}})});
        }
      }
      if (fusion && !is_dynamic_dim(ow_) && !is_dynamic_dim(oh_)) {
        fusion->create_output_fusion_anchor(
          {blocking_output_ ? tensor_slice(output,
             {{pbs, 1UL},
               {outer_k * oc_ / im_oc_block / oc_split,
                 oc_ / im_oc_block / oc_split},
               {0, oh_}, {0, ow_}, {0, im_oc_block}})
                            : tensor_slice(output,
                              {{pbs, 1UL}, {0, oh_}, {0, ow_},
                                {outer_k * oc_ / oc_split, oc_ / oc_split}})});
      }
    }
  }
  loops = {lpbs, lph, lpw, lpoc, lpic, lok};
}

void gen_nested_conv_fwd_t::schedule_loops(context_ptr ctx,
  const nested_conv_fwd_config_t &config, stmt body,
  std::vector<for_loop> &fors) const {
  if (use_conv1d) {
    auto lpbs = fors[0], lps = fors[1], lpoc = fors[2], lpic = fors[3];
    lpbs->fuse(lps)->fuse(lpoc)->fuse(lpic);
  } else if (use_nested_2d_) {
    if (!is_dynamic()) {
      const auto pack_rows
        = (config.im_w_block > 0 && ow_ % config.im_w_block != 0);
      if (try_os_blocking_ && pack_rows) {
        COMPILE_ASSERT(static_cast<int>(fors.size()) == 5,
          "expected to have 4 for loops, but got " << fors.size()
                                                   << " for loops.");
        auto lpbs = fors[0], lps = fors[1], lpoc = fors[2], lpic = fors[3],
             lok = fors[4];
        lok->fuse(lpbs)->fuse(lps)->fuse(lpoc)->fuse(lpic);
      } else {
        if (!is_1x1_conv_) {
          COMPILE_ASSERT(static_cast<int>(fors.size()) == 6,
            "expected to have 6 for loops, but got " << fors.size()
                                                     << " for loops.");
          auto lpbs = fors[0], lph = fors[1], lpw = fors[2], lpoc = fors[3],
               lpic = fors[4], lok = fors[5];
          lok->fuse(lpbs)->fuse(lph)->fuse(lpw)->fuse(lpoc)->fuse(lpic);
        } else {
          COMPILE_ASSERT(static_cast<int>(fors.size()) == 5,
            "expected to have 5 for loops, but got " << fors.size()
                                                     << " for loops.");
          auto lpbs = fors[0], lph = fors[1], lpw = fors[2], lpoc = fors[3],
               lpic = fors[4];
          lpbs->fuse(lph)->fuse(lpw)->fuse(lpoc)->fuse(lpic);
        }
      }
    } else if (!is_1x1_conv_) {
      COMPILE_ASSERT(static_cast<int>(fors.size()) == 6,
        "expected to have 5 for loops, but got " << fors.size()
                                                 << " for loops.");
      auto lpbs = fors[0], lph = fors[1], lpw = fors[2], lpoc = fors[3],
           lpic = fors[4], lok = fors[5];
      lok->fuse(lpbs)->fuse(lph)->fuse(lpw)->fuse(lpoc)->fuse(lpic);
    }
  }
}

bool gen_nested_conv_fwd_t::generate(context_ptr ctx,
  const nested_conv_fwd_config_t &config, fusion_manager *fusion,
  const std::vector<expr> &inputs, const std::vector<expr> &outputs,
  std::vector<for_loop> &loops) const {
  COMPILE_ASSERT(inputs.size() == 2,
    "Expecting 2 inputs for conv, but got " << inputs.size() << " inputs.");
  COMPILE_ASSERT(outputs.size() == 1,
    "Expecting 1 output for conv, but got " << outputs.size() << " output.");

  int K_block = is_dynamic() ? oc_ : config.K_block;
  int C_block = is_dynamic() ? ic_ : config.C_block;
  int im_s_block = config.im_w_block;

  int pack_input = config.pack_input;
  const bool use_os_blocking = try_os_blocking_ && ctx->use_amx();
  const bool pack_rows
    = use_os_blocking && (im_s_block > 0 && ow_ % im_s_block != 0);
  int os = actual_os_;
  if (use_conv1d) {
    COMPILE_ASSERT(im_oc_block_ && (oc_ % im_oc_block_ == 0),
      "oc should be dividable by K_block, but got oc=" << oc_ << " K_block="
                                                       << im_oc_block_ << ".");
    COMPILE_ASSERT(im_ic_block_ && (ic_ % im_ic_block_ == 0),
      "ic should be dividable by C_block, but got ic=" << ic_ << " C_block="
                                                       << im_ic_block_ << ".");
  } else {
    COMPILE_ASSERT(
      K_block && (utils::rnd_up(oc_, config.im_oc_block) % K_block == 0),
      "oc should be dividable by K_block, but got oc=" << oc_ << " K_block="
                                                       << K_block << ".");
    COMPILE_ASSERT(
      C_block && (utils::rnd_up(ic_, config.im_ic_block) % C_block == 0),
      "ic should be dividable by C_block, but got ic=" << ic_ << " C_block="
                                                       << C_block << ".");
  }
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
  if (dtypeInput == datatypes::f16) {
    COMPILE_ASSERT((dtypeWeight == datatypes::f16),
      "Weights should be f16 as "
      "data, the mixed datatypes is not supported yet!");
    COMPILE_ASSERT((dtypeOutput == datatypes::f32),
      "Output should be f32 when data and weights are in f16.");
    kpack = 1;
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
    int adj_ow = ow_ + num_elems_skip_per_ow_;
    os_mask.resize(os);
    for (int i = 0; i < os; ++i) {
      if (i % adj_ow < ow_) {
        os_mask[i] = 1;
      } else {
        os_mask[i] = 0;
      }
    }

    int im_os_num_block = os / im_s_block;
    _tensor_(conv_os_acc_size, datatypes::s32, {im_os_num_block});
    int acc_size = 0;
    int blk_size = 0;
    for (int i = 0; i < im_os_num_block; ++i) {
      blk_size = std::accumulate(os_mask.begin() + i * im_s_block,
        os_mask.begin() + (i + 1) * im_s_block, 0);
      conv_os_acc_size[i] = acc_size;
      acc_size += blk_size;
    }
    os_acc_size = conv_os_acc_size;
  }

  if (!is_dynamic()) {
    if (use_os_blocking) {
      COMPILE_ASSERT((im_s_block > 0) && (os % im_s_block == 0),
        "os should be dividable by im_w_block, but got os="
          << os << " im_w_block=" << config.im_w_block << ".");
    } else if (!use_conv1d) {
      COMPILE_ASSERT((config.im_h_block > 0) && (oh_ % config.im_h_block == 0),
        "oh should be dividable by im_h_block, but got oh="
          << oh_ << " im_h_block=" << config.im_h_block << ".");
      COMPILE_ASSERT((config.im_w_block > 0) && (ow_ % config.im_w_block == 0),
        "ow should be dividable by tile_q, but got ow="
          << ow_ << " im_w_block=" << config.im_w_block << ".");
    }
  }

  expr output = outputs[op_params_t::out];
  expr input = inputs[op_params_t::in_data];
  expr weight = inputs[op_params_t::in_weight];

  if (use_conv1d) {
    // no padding/stride 1x1 1d/2d
    compute_conv1d(
      ctx, config, fusion, output, input, weight, loops, os, kpack);
  } else if (is_1x1_conv_) {
    COMPILE_ASSERT(pd_b_ == 0 && ph_b_ == 0 && pw_b_ == 0 && pd_e_ == 0
        && ph_e_ == 0 && pw_e_ == 0,
      "1x1 conv doesn't support padding!");
    COMPILE_ASSERT(
      !inverse_filter_, "1x1 conv doesn't support inverse convolution.");
    if (pack_input == 0 && (sd_ > 1 || sh_ > 1 || sw_ > 1)) {
      compute_1x1_no_pack_input_nested(
        ctx, config, fusion, output, input, weight, loops, os, kpack);
    } else {
      if (is_dynamic()) {
        dynamic_compute_1x1_pack_input_nested(
          ctx, config, fusion, output, input, weight, loops, os, kpack);
      } else {
        compute_1x1_pack_input_nested(
          ctx, config, fusion, output, input, weight, loops, os, kpack);
      }
    }
  } else {
    if (pd_b_ == 0 && ph_b_ == 0 && pw_b_ == 0 && pd_e_ == 0 && ph_e_ == 0
      && pw_e_ == 0) {
      COMPILE_ASSERT(!inverse_filter_,
        "conv NxN (no padding) does not support inverse "
        "convolution.");
      if (is_3d_) {
        COMPILE_ASSERT(!is_3d_,
          "nested conv fwd does not support 3d convolution currently.");
      } else {
        if (use_os_blocking && pack_rows) {
          compute_conv_no_padding_os_blocking_nested(ctx, config, fusion,
            output, input, weight, loops, os, kpack, use_os_blocking, pack_rows,
            os_acc_size, os_mask);
        } else {
          if (is_dynamic()) {
            dynamic_compute_conv_no_padding_nested(ctx, config, fusion, output,
              input, weight, loops, os, kpack, use_os_blocking, pack_rows,
              os_acc_size, os_mask);
          } else {
            compute_conv_no_padding_nested(ctx, config, fusion, output, input,
              weight, loops, os, kpack, use_os_blocking, pack_rows, os_acc_size,
              os_mask);
          }
        }
      }
    } else {
      if (is_dynamic()) {
        dynamic_compute_conv_padding_nested(ctx, config, fusion, output, input,
          weight, loops, os, kpack, use_os_blocking, pack_rows, os_acc_size,
          os_mask);
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
