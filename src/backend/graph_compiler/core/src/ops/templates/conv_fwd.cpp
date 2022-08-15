/*******************************************************************************
 * Copyright 2020-2022 Intel Corporation
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
#include <numeric>
#include <utility>
#include "conv_fwd.hpp"
#include "utils.hpp"
#include <compiler/ir/builder.hpp>
#include <compiler/ir/easy_build.hpp>
#include <compiler/ir/graph/fusion_mgr.hpp>
#include <microkernel/builtin.hpp>
#include <runtime/config.hpp>
#include <unordered_set>
#include <util/any_map.hpp>
#include <util/reflection.hpp>
#include <util/utils.hpp>

using namespace sc::builder;
namespace sc {

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

static inline std::vector<int> get_os_blocks(const int ow, const int adj_os) {
  std::vector<int> factors = utils::get_factors(ow);
  std::vector<int> os_factors = utils::get_blocks(adj_os, 16);
  factors.insert(factors.end(), os_factors.begin(), os_factors.end());
  std::unordered_set<int> unique_factors(factors.begin(), factors.end());
  factors.assign(unique_factors.begin(), unique_factors.end());
  std::sort(factors.begin(), factors.end());

  return factors;
}

config_ptr gen_conv_fwd_t::get_default_config(context_ptr ctx) const {
  auto ret = reflection::general_object_t::make<conv_fwd_config_t>();
  conv_fwd_config_t &cfg = *ret.unchecked_get_as<conv_fwd_config_t>();
  if (oc_ % 32 == 0) {
    cfg.K_block = 32;
  } else {
    cfg.K_block = oc_;
  }
  if (ic_ % 32 == 0) {
    cfg.C_block = 32;
  } else {
    cfg.C_block = ic_;
  }
  cfg.tile_d = 1;
  cfg.tile_p = 1;
  cfg.tile_q = ow_;
  if (is_1x1_conv_) {
    cfg.tile_os = 1;
  } else {
    cfg.tile_os = ow_;
  }
  cfg.pack_input = 0;
  cfg.loop_sched = 3;
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
    utils::is_one_of(static_cast<int>(input_plain_dims.size()), 4, 5),
    "Wrong input dims, expected to be 4D or 5D input, but got "
      << input_plain_dims.size() << "D.");
  COMPILE_ASSERT(
    utils::is_one_of(static_cast<int>(weight_plain_dims.size()), 4, 5)
      && (weight_plain_dims.size() == input_plain_dims.size()),
    "Wrong weight dims, only support 4D or 5D weights, but got "
      << weight_plain_dims.size() << "D.");
  COMPILE_ASSERT(utils::is_one_of(static_cast<int>(out_plain_dims.size()), 4, 5)
      && (out_plain_dims.size() == input_plain_dims.size()),
    "Wrong output dims, only support 4D or 5D weights, but got "
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
  COMPILE_ASSERT(input_plain_dims[1] == weight_plain_dims[1],
    "expect input_plain_dims[1] == weight_plain_dims[1], but got "
      << input_plain_dims[1] << " vs " << weight_plain_dims[1] << ".");

  mb_ = input_plain_dims[0];
  ic_ = input_plain_dims[1];
  id_ = is_3d_ ? input_plain_dims[2] : 1;
  ih_ = input_plain_dims[ndims_ - 2];
  iw_ = input_plain_dims[ndims_ - 1];
  oc_ = weight_plain_dims[0];
  kd_ = is_3d_ ? weight_plain_dims[2] : 1;
  kh_ = weight_plain_dims[ndims_ - 2];
  kw_ = weight_plain_dims[ndims_ - 1];
  od_ = is_3d_ ? out_plain_dims[2] : 1;
  oh_ = out_plain_dims[ndims_ - 2];
  ow_ = out_plain_dims[ndims_ - 1];
  is_1x1_conv_ = (kd_ == 1 && kh_ == 1 && kw_ == 1);
  pd_ = is_3d_ ? pads_begin[0] : 0;
  ph_ = pads_begin[0], pw_ = pads_begin[0];
  if (pads_begin.size() > 1) {
    ph_ = pads_begin[ndims_ - 4];
    pw_ = pads_begin[ndims_ - 3];
  }
  sd_ = is_3d_ ? stride[0] : 1;
  sh_ = stride[0], sw_ = stride[0];
  if (stride.size() > 1) {
    sh_ = stride[ndims_ - 4];
    sw_ = stride[ndims_ - 3];
  }

  // For non 1x1 conv and AMX platform, spatial blocking instead of row blocking
  // is used, which needs to consider the border carefully, as the cross row
  // boundary (contains padding or not) will generate useless output which have
  // to be skipped before storing.
  actual_os_ = oh_ * ow_;
  num_elems_skip_per_ow_ = ((kw_ - 1) / sw_) * sh_ + (sh_ - 1) * ow_;
  adj_os_ = std::min(actual_os_ + num_elems_skip_per_ow_ * (oh_ - 1),
    (ih_ + 2 * ph_) * (iw_ + 2 * pw_));

  bool is_int8
    = utils::is_one_of(get_input_dtype(), datatypes::u8, datatypes::s8);
  // Note: os blocking is only valid for non_1x1, no pad and non 3D conv with
  // amx-int8 only so far.
  bool has_pad = (pd_ > 0) || (ph_ > 0) || (pw_ > 0);
  try_os_blocking_ = (!is_1x1_conv_) && (!has_pad) && (!is_3d_) && is_int8;
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
    const bool use_os_blocking, const bool pack_rows, const expr &os_blk_size, \
    const expr &os_acc_size, const std::vector<char> &os_mask

void gen_conv_fwd_t::compute_1x1_no_pack_input(CONV_ARG_LIST) const {
  assert(loops.size() == 4 && "expected to have 4 level loops!");
  for_loop &ln = loops.at(0), &lk = loops.at(1), &ld = loops.at(2),
           &lp = loops.at(3);
  _named_for_(ln, n, 0, mb_, 1, for_type::PARALLEL) {
    _named_for_(lk, k, 0, K_num_block) {
      _named_for_(lp, p_o, 0, oh_ / config.tile_p) {
        _named_for_(ld, d_o, 0, od_ / config.tile_d) {
          _for_(q_o, 0, ow_ / config.tile_q) {
            _for_(d_i, 0, config.tile_d) {
              _for_(p_i, 0, config.tile_p) {
                if (is_3d_) {
                  sc::builtin::brgemm_init_update(
                    tensor_ptr(input,
                      {n, 0, (d_o * config.tile_d + d_i) * sd_,
                        (p_o * config.tile_p + p_i) * sh_,
                        q_o * config.tile_q * sw_, 0}),

                    tensor_ptr(weight,
                      kpack > 1 ? std::vector<expr> {k, 0, 0, 0, 0, 0, 0, 0}
                                : std::vector<expr> {k, 0, 0, 0, 0, 0, 0}),
                    tensor_ptr(output,
                      {n, k, d_o * config.tile_d + d_i,
                        p_o * config.tile_p + p_i, q_o * config.tile_q, 0}),
                    C_num_block, config.tile_q, config.K_block, config.C_block,
                    sw_ * config.C_block, config.K_block, config.K_block,
                    id_ * ih_ * iw_ * config.C_block,
                    config.C_block * config.K_block, get_input_dtype(),
                    get_weight_dtype());

                } else {
                  sc::builtin::brgemm_init_update(
                    tensor_ptr(input,
                      {n, 0, (p_o * config.tile_p + p_i) * sh_,
                        q_o * config.tile_q * sw_, 0}),

                    tensor_ptr(weight,
                      kpack > 1 ? std::vector<expr> {k, 0, 0, 0, 0, 0, 0}
                                : std::vector<expr> {k, 0, 0, 0, 0, 0}),
                    tensor_ptr(output,
                      {n, k, p_o * config.tile_p + p_i, q_o * config.tile_q,
                        0}),
                    C_num_block, config.tile_q, config.K_block, config.C_block,
                    sw_ * config.C_block, config.K_block, config.K_block,
                    ih_ * iw_ * config.C_block, config.C_block * config.K_block,
                    get_input_dtype(), get_weight_dtype());
                }
              }
            }
          }

          if (fusion) {
            if (is_3d_) {
              fusion->create_output_fusion_anchor({tensor_slice(output,
                {{n, 1}, {k, 1}, {d_o * config.tile_d, config.tile_d},
                  {p_o * config.tile_p, config.tile_p}, {0, ow_},
                  {0, config.K_block}})});
            } else {
              fusion->create_output_fusion_anchor({tensor_slice(output,
                {{n, 1}, {k, 1}, {p_o * config.tile_p, config.tile_p}, {0, ow_},
                  {0, config.K_block}})});
            }
          }
        }
      }
    }
  }
}

void gen_conv_fwd_t::compute_1x1_pack_input(CONV_ARG_LIST) const {
  COMPILE_ASSERT(!is_3d_, "1x1 pack input doens't support 3D conv yet!");
  assert(loops.size() == 4 && "expected to have 4 level loops!");
  for_loop &ln = loops.at(0), &lk = loops.at(1), &ld = loops.at(2),
           &lp = loops.at(3);
  for_loop lc;
  tensor input1;
  if (config.pack_input == 1 && (sd_ > 1 || sh_ > 1 || sw_ > 1)) {
    _tensor_(input_tmp, get_input_dtype(),
      {mb_, C_num_block, oh_, ow_, config.C_block});
    _named_for_(ln, n, 0, mb_, 1, for_type::PARALLEL) {
      _named_for_(lk, c_o, 0, C_num_block) {
        _named_for_(lp, p, 0, oh_) {
          _for_(q, 0, ow_) {
            _for_(c_i, 0, config.C_block) {
              input_tmp[{n, c_o, p, q, c_i}]
                = input[{n, c_o, p * sh_, q * sw_, c_i}];
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
    input1 = input.static_as<tensor>();
  }
  _named_for_(ln, n, 0, mb_, 1, for_type::PARALLEL) {
    _named_for_(lk, k, 0, K_num_block) {
      if (config.loop_sched == 4 || config.loop_sched == 5) {
        // c_o shall not be fused; c_o == 0 is used for initialization
        // merging c_o will forbid c_o == 0 to be always executed at first
        // tagged as temp.loop_no_fuse
        _named_for_(lc, c_o, 0, C_num_block) {
          _for_(p_o, 0, oh_ / config.tile_p) {
            _if_(c_o == 0) {
              sc::builtin::brgemm_init_update(
                tensor_ptr(input1, {n, 0, p_o * config.tile_p, 0, 0}),
                tensor_ptr(weight,
                  kpack > 1 ? std::vector<expr> {k, 0, 0, 0, 0, 0, 0}
                            : std::vector<expr> {k, 0, 0, 0, 0, 0}),
                tensor_ptr(output, {n, k, p_o * config.tile_p, 0, 0}), 1,
                config.tile_p * ow_, config.K_block, config.C_block,
                config.C_block, config.K_block, config.K_block,
                oh_ * ow_ * config.C_block, config.C_block * config.K_block,
                get_input_dtype(), get_weight_dtype());
            }
            _else_ {
              sc::builtin::brgemm_update(
                tensor_ptr(input1, {n, c_o, p_o * config.tile_p, 0, 0}),
                tensor_ptr(weight,
                  kpack > 1 ? std::vector<expr> {k, c_o, 0, 0, 0, 0, 0}
                            : std::vector<expr> {k, c_o, 0, 0, 0, 0}),
                tensor_ptr(output, {n, k, p_o * config.tile_p, 0, 0}), 1,
                config.tile_p * ow_, config.K_block, config.C_block,
                config.C_block, config.K_block, config.K_block,
                oh_ * ow_ * config.C_block, config.C_block * config.K_block,
                get_input_dtype(), get_weight_dtype());
            }
          }
        }
        lc->attr().set("temp.loop_no_fuse", true);
        if (fusion) {
          fusion->create_output_fusion_anchor({tensor_slice(output,
            {{n, 1}, {k, 1}, {0, oh_}, {0, ow_}, {0, config.K_block}})});
        }
      } else {
        _named_for_(lp, p_o, 0, oh_ / config.tile_p) {
          sc::builtin::brgemm_init_update(
            tensor_ptr(input1, {n, 0, p_o * config.tile_p, 0, 0}),
            tensor_ptr(weight,
              kpack > 1 ? std::vector<expr> {k, 0, 0, 0, 0, 0, 0}
                        : std::vector<expr> {k, 0, 0, 0, 0, 0}),
            tensor_ptr(output, {n, k, p_o * config.tile_p, 0, 0}), C_num_block,
            config.tile_p * ow_, config.K_block, config.C_block, config.C_block,
            config.K_block, config.K_block, oh_ * ow_ * config.C_block,
            config.C_block * config.K_block, get_input_dtype(),
            get_weight_dtype());

          if (fusion) {
            fusion->create_output_fusion_anchor({tensor_slice(output,
              {{n, 1}, {k, 1}, {p_o * config.tile_p, config.tile_p}, {0, ow_},
                {0, config.K_block}})});
          }
        }
      }
    }
  }
}

void gen_conv_fwd_t::compute_conv3d_no_padding(CONV_ARG_LIST) const {
  COMPILE_ASSERT((pd_ == 0 && ph_ == 0 && pw_ == 0),
    "unexpected padding in no_padding kernels!");
  assert(loops.size() == 4 && "expected to have 4 level loops!");
  for_loop &ln = loops.at(0), &lk = loops.at(1), &ld = loops.at(2),
           &lp = loops.at(3);

  _named_for_(ln, n, 0, mb_, 1, for_type::PARALLEL) {
    _named_for_(lk, k_o, 0, K_num_block) {
      _named_for_(ld, d_o, 0, od_ / config.tile_d) {
        _named_for_(lp, p_o, 0, oh_ / config.tile_p) {
          _tensor_(A_list, datatypes::pointer, {kd_ * kh_ * kw_});
          _tensor_(B_list, datatypes::pointer, {kd_ * kh_ * kw_});
          _for_(q_o, 0, ow_ / config.tile_q) {
            _for_(d_i, 0, config.tile_d) {
              _for_(p_i, 0, config.tile_p) {
                sc::builtin::mem_zero(
                  tensor_ptr(output,
                    {n, k_o, d_o * config.tile_d + d_i,
                      p_o * config.tile_p + p_i, q_o * config.tile_q, 0}),
                  config.tile_q * config.K_block, get_output_dtype());
                _for_(c_o, 0, C_num_block) {
                  // assign pointers to A/B_list
                  _for_(d, 0, kd_) {
                    _for_(r, 0, kh_) {
                      _for_(s, 0, kw_) {
                        auto idx = d * kh_ * kw_ + r * kw_ + s;
                        A_list[idx] = tensor_ptr(input,
                          {n, c_o, (d_o * config.tile_d + d_i) * sd_ + d,
                            (p_o * config.tile_p + p_i) * sh_ + r,
                            q_o * config.tile_q * sw_ + s, 0});
                        B_list[idx] = tensor_ptr(weight,
                          kpack > 1
                            ? std::vector<expr> {k_o, c_o, d, r, s, 0, 0, 0}
                            : std::vector<expr> {k_o, c_o, d, r, s, 0, 0});
                      }
                    }
                  }

                  sc::builtin::brgemm_list_update(A_list, B_list,
                    tensor_ptr(output,
                      {n, k_o, d_o * config.tile_d + d_i,
                        p_o * config.tile_p + p_i, q_o * config.tile_q, 0}),
                    1, config.tile_q, config.K_block, config.C_block,
                    sw_ * config.C_block, config.K_block, config.K_block,
                    1 /*useless*/, 1 /*useless*/, kd_ * kh_ * kw_,
                    get_input_dtype(), get_weight_dtype());
                }

                if (fusion) {
                  fusion->create_output_fusion_anchor({tensor_slice(output,
                    {{n, 1}, {k_o, 1}, {d_o * config.tile_d + d_i, 1},
                      {p_o * config.tile_p + p_i, 1},
                      {q_o * config.tile_q, config.tile_q},
                      {0, config.K_block}})});
                }
              }
            }
          }
        }
      }
    }
  }
}

void gen_conv_fwd_t::compute_conv_no_padding(CONV_ARG_LIST) const {
  COMPILE_ASSERT((pd_ == 0 && ph_ == 0 && pw_ == 0),
    "unexpected padding in no_padding kernels!");
  assert(loops.size() == 4 && "expected to have 4 level loops!");
  for_loop &ln = loops.at(0), &lk = loops.at(1), &ld = loops.at(2),
           &lp = loops.at(3);

  _named_for_(ln, n, 0, mb_, 1, for_type::PARALLEL) {
    _named_for_(lk, k_o, 0, K_num_block) {
      if (use_os_blocking) {
        _named_for_(lp, o_o, 0, os / config.tile_os) {
          _tensor_(A_list, datatypes::pointer, {kh_});
          _tensor_(B_list, datatypes::pointer, {kh_});
          auto out_tsr = tensor_ptr(output,
            {n, k_o, o_o * config.tile_os / ow_, o_o * config.tile_os % ow_,
              0});
          int adj_ow = ow_ + (pack_rows ? num_elems_skip_per_ow_ : 0);

          if (pack_rows) {
            auto adj_m = os_blk_size[{o_o}];
            auto acc_m = os_acc_size[{o_o}];
            out_tsr = tensor_ptr(output, {n, k_o, acc_m / ow_, acc_m % ow_, 0});
            sc::builtin::mem_zero(
              out_tsr, adj_m * config.K_block, get_output_dtype());
          } else {
            sc::builtin::mem_zero(
              out_tsr, config.tile_os * config.K_block, get_output_dtype());
          }

          _for_(c_o, 0, C_num_block) {
            _for_(r, 0, kh_) {
              A_list[r] = tensor_ptr(input,
                {n, c_o, ((o_o * config.tile_os) / adj_ow) * sh_ + r,
                  ((o_o * config.tile_os) % adj_ow) * sw_, 0});
              B_list[r] = tensor_ptr(weight,
                kpack > 1 ? std::vector<expr> {k_o, c_o, r, 0, 0, 0, 0}
                          : std::vector<expr> {k_o, c_o, r, 0, 0, 0});
            }

            const auto hint_A_size
              = config.tile_os * config.C_block * kh_ * kw_;
            const auto hint_B_size
              = config.K_block * config.C_block * kh_ * kw_;
            // note, the actual C_size is <= tile_os if pack_rows=true
            const auto hint_C_size = config.tile_os * config.K_block;
            sc_brgemm_attrs_t brg_attrs {{brgemm::attr_key::max_bs, kh_ * kw_},
              {brgemm::attr_key::hint_expected_A_size, hint_A_size},
              {brgemm::attr_key::hint_expected_B_size, hint_B_size},
              {brgemm::attr_key::hint_expected_C_size, hint_C_size},
              {brgemm::attr_key::use_interleave_stores, true},
              {brgemm::attr_key::use_uker, true},
              {brgemm::attr_key::bd_mask_level, pack_rows ? 2 : 0}};

            sc::builtin::brgemm_list_update(A_list, B_list, out_tsr, kw_,
              config.tile_os, config.K_block, config.C_block,
              sw_ * config.C_block, config.K_block, config.K_block,
              config.C_block, config.C_block * config.K_block, kh_,
              get_input_dtype(), get_weight_dtype(), brg_attrs, os_mask, o_o,
              os / config.tile_os);
          }
        }
        if (fusion) {
          // Note: slice tensor might across multi-rows with non-rectangular
          // shapes. Currently, we just promote the fusion anchor to higher
          // level of loop, which will consume larger buffer and is non-optimal
          // This can be optimized in next version of fusion manager.
          fusion->create_output_fusion_anchor({tensor_slice(output,
            {{n, 1}, {k_o, 1}, {0, oh_}, {0, ow_}, {0, config.K_block}})});
        }
      } else {
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
                _for_(r, 0, kh_) {
                  A_list[r] = tensor_ptr(input,
                    {n, c_o, (p_o * config.tile_p + p_i) * sh_ + r,
                      q_o * config.tile_q * sw_, 0});
                  B_list[r] = tensor_ptr(weight,
                    kpack > 1 ? std::vector<expr> {k_o, c_o, r, 0, 0, 0, 0}
                              : std::vector<expr> {k_o, c_o, r, 0, 0, 0});
                }

                sc::builtin::brgemm_list_update(A_list, B_list,
                  tensor_ptr(output,
                    {n, k_o, p_o * config.tile_p + p_i, q_o * config.tile_q,
                      0}),
                  kw_, config.tile_q, config.K_block, config.C_block,
                  sw_ * config.C_block, config.K_block, config.K_block,
                  config.C_block, config.C_block * config.K_block, kh_,
                  get_input_dtype(), get_weight_dtype());
              }

              if (fusion) {
                fusion->create_output_fusion_anchor({tensor_slice(output,
                  {{n, 1}, {k_o, 1}, {p_o * config.tile_p + p_i, 1},
                    {q_o * config.tile_q, config.tile_q},
                    {0, config.K_block}})});
              }
            }
          }
        }
      }
    }
  }
}

void gen_conv_fwd_t::compute_conv_padding(CONV_ARG_LIST) const {
  COMPILE_ASSERT(!is_3d_, "3D conv with padding is not supported yet!");
  assert(loops.size() == 4 && "expected to have 4 level loops!");
  for_loop &ln = loops.at(0), &lk = loops.at(1), &ld = loops.at(2),
           &lp = loops.at(3);

  /* to do conv 3*3 with padding */
  std::unordered_set<int> Q1;
  std::unordered_set<int> Q2;
  std::unordered_set<int> Q3;

  int H_PADDED = ih_ + 2 * ph_, W_PADDED = iw_ + 2 * pw_;
  sc_dims padded_input_dims
    = sc_dims {mb_, C_num_block, H_PADDED, W_PADDED, config.C_block};

  // collect the possible values for Q_tmp
  for (int p_o = 0; p_o < oh_ / config.tile_p; p_o++) {
    for (int q_o = 0; q_o < ow_ / config.tile_q; q_o++) {
      for (int p_i = 0; p_i < config.tile_p; p_i++) {
        int x_start_offset = tensor_offset(padded_input_dims,
          std::vector<int> {0, 0, (p_o * config.tile_p + p_i) * sh_,
            q_o * config.tile_q * sw_, 0});
        int x_threshold_left = tensor_offset(padded_input_dims,
          std::vector<int> {0, 0, (p_o * config.tile_p + p_i) * sh_, pw_, 0});
        int x_threshold_right = tensor_offset(padded_input_dims,
          std::vector<int> {
            0, 0, (p_o * config.tile_p + p_i) * sh_, W_PADDED - pw_ - 1, 0});
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

  _named_for_(ln, n, 0, mb_, 1, for_type::PARALLEL) {
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

  int ih_padded = ih_ + 2 * ph_, iw_padded = iw_ + 2 * pw_;
  sc_dims padded_input_dims
    = sc_dims {mb_, C_num_block, ih_padded, iw_padded, config.C_block};
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
  _tensor_(pbuffer, dtypeInput, {src_row_tile_size, config.C_block});
  sc::builtin::mem_zero(
    pbuffer, (src_row_tile_size)*config.C_block, dtypeInput);

  uint32_t lanes = 1;
  if (utils::is_one_of(dtypeInput, datatypes::s8, datatypes::u8)) {
    if (config.C_block / 64 && config.C_block % 64 == 0) {
      lanes = std::min(64U, ctx->get_max_vector_lanes(dtypeInput.type_code_));
    } else if (config.C_block / 32 && config.C_block % 32 == 0) {
      lanes = std::min(32U, ctx->get_max_vector_lanes(dtypeInput.type_code_));
    } else if (config.C_block / 16 && config.C_block % 16 == 0) {
      lanes = std::min(16U, ctx->get_max_vector_lanes(dtypeInput.type_code_));
    }
  } else if (dtypeInput == datatypes::bf16) {
    if (config.C_block / 32 && config.C_block % 32 == 0) {
      lanes = std::min(32U, ctx->get_max_vector_lanes(dtypeInput.type_code_));
    } else if (config.C_block / 16 && config.C_block % 16 == 0) {
      lanes = std::min(16U, ctx->get_max_vector_lanes(dtypeInput.type_code_));
    }
  } else {
    if (config.C_block / 16 && config.C_block % 16 == 0) {
      lanes = std::min(16U, ctx->get_max_vector_lanes(dtypeInput.type_code_));
    }
  }

  _named_for_(ln, n, 0, mb_, 1, for_type::PARALLEL) {
    _named_for_(lk, k_o, 0, K_num_block) {
      _named_for_(lp, p_o, 0, oh_ / config.tile_p) {
        _tensor_(A_list, datatypes::pointer, {kh_ * kw_});
        _tensor_(B_list, datatypes::pointer, {kh_ * kw_});
        // create a sub-tensor with maximum size which holds all the boundary
        // that contains padding
        _tensor_(
          sub_tensor, dtypeInput, {kh_, src_row_tile_size, config.C_block});
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
            sc::builtin::mem_zero(
              tensor_ptr(output,
                {n, k_o, p_o * config.tile_p + p_i, q_o * config.tile_q, 0}),
              config.tile_q * config.K_block, dtypeOutput);

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
                      sc::builtin::mem_zero(
                        tensor_ptr(sub_tensor, {i - unpad_begin_index, 0, 0}),
                        real_pad_left * config.C_block, dtypeInput);

                      // mapping dst to padding src, then mapping
                      // padding src to real src to get the actual elements.
                      _for_(j, real_pad_left, src_row_tile_size) {
                        _for_(k, 0, config.C_block, (int)lanes) {
                          sub_tensor[span_t(
                            {i - unpad_begin_index, j, k}, lanes)]
                            = input[span_t(
                              {n, c_o,
                                (p_o * config.tile_p + p_i) * sh_ + i - ph_,
                                q_o * config.tile_q * sw_ + j - pw_, k},
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
                      sc::builtin::mem_zero(
                        tensor_ptr(sub_tensor, {i - unpad_begin_index, 0, 0}),
                        real_pad_left * config.C_block, dtypeInput);

                      _for_(j, real_pad_left, copy_width + real_pad_left) {
                        _for_(k, 0, config.C_block, (int)lanes) {
                          // N, C, H, W, c
                          sub_tensor[span_t(
                            {i - unpad_begin_index, j, k}, lanes)]
                            = input[span_t(
                              {n, c_o,
                                (p_o * config.tile_p + p_i) * sh_ + i - ph_,
                                q_o * config.tile_q * sw_ + j - pw_, k},
                              lanes)];
                        }
                      }

                      // memzero right part
                      sc::builtin::mem_zero(tensor_ptr(sub_tensor,
                                              {i - unpad_begin_index,
                                                copy_width + real_pad_left, 0}),
                        real_pad_right * config.C_block, dtypeInput);
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
                          {n, c_o, (p_o * config.tile_p + p_i) * sh_ + r - ph_,
                            q_o * config.tile_q * sw_ + s - pw_, 0});
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
                            = input[span_t(
                              {n, c_o,
                                (p_o * config.tile_p + p_i) * sh_ + i - ph_,
                                q_o * config.tile_q * sw_ + j - pw_, k},
                              lanes)];
                        }
                      }
                      sc::builtin::mem_zero(
                        tensor_ptr(
                          sub_tensor, {i - unpad_begin_index, copy_width, 0}),
                        real_pad_right * config.C_block, dtypeInput);
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
                  idx = builder::make_cast(datatypes::u32, r * kw_ + s);
                  B_list[idx] = tensor_ptr(weight,
                    kpack > 1 ? std::vector<expr> {k_o, c_o, r, s, 0, 0, 0}
                              : std::vector<expr> {k_o, c_o, r, s, 0, 0});
                }
              }
              sc::builtin::brgemm_list_update(A_list, B_list,
                tensor_ptr(output,
                  {n, k_o, p_o * config.tile_p + p_i, q_o * config.tile_q, 0}),
                1, config.tile_q, config.K_block, config.C_block,
                sw_ * config.C_block, config.K_block, config.K_block,
                config.C_block, config.C_block * config.K_block, kh_ * kw_,
                dtypeInput, dtypeWeight);
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

void gen_conv_fwd_t::schedule_loops(context_ptr ctx,
  const conv_fwd_config_t &config, stmt body,
  std::vector<for_loop> &fors) const {
  COMPILE_ASSERT(static_cast<int>(fors.size()) == 4,
    "expected to have 4 for loops, but got " << fors.size() << " for loops.");
  for_loop ln = fors.at(0), lk = fors.at(1), ld = fors.at(2), lp = fors.at(3);
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
  } else if (loop_sched == 4) {
    // 1x1 only: loop order ln->lk->lo->lp
    // merge ln, lk
    auto outer = ln->fuse(lk);
    outer->kind_ = for_type::PARALLEL;
  } else if (loop_sched == 5) {
    // 1x1 only: loop order lk->ln->lo->lp
    // merge lk, ln
    ln->reorder(body, {lk, ln});
    auto outer = lk->fuse(ln);
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
      "id/kd/od/tile_q should be 1 for non-3D conv, but got id="
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
  expr os_blk_size = expr();
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

    _tensor_(conv_os_blk_size, datatypes::s32, {os_num_block});
    _tensor_(conv_os_acc_size, datatypes::s32, {os_num_block});
    int acc_size = 0;
    int blk_size = 0;
    for (int i = 0; i < os_num_block; ++i) {
      blk_size = std::accumulate(
        os_mask.begin() + i * tile_os, os_mask.begin() + (i + 1) * tile_os, 0);
      conv_os_blk_size[i] = blk_size;
      conv_os_acc_size[i] = acc_size;
      acc_size += blk_size;
    }

    os_blk_size = conv_os_blk_size;
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
  if (is_1x1_conv_) {
    COMPILE_ASSERT(
      pd_ == 0 && ph_ == 0 && pw_ == 0, "1x1 conv doesn't support padding!");
    if (is_3d_ || (pack_input == 0 && (sd_ > 1 || sh_ > 1 || sw_ > 1))) {
      compute_1x1_no_pack_input(ctx, config, fusion, output, input, weight,
        loops, K_num_block, C_num_block, os, kpack);
    } else {
      compute_1x1_pack_input(ctx, config, fusion, output, input, weight, loops,
        K_num_block, C_num_block, os, kpack);
    }
  } else {
    if (pd_ == 0 && ph_ == 0 && pw_ == 0) {
      if (is_3d_) {
        compute_conv3d_no_padding(ctx, config, fusion, output, input, weight,
          loops, K_num_block, C_num_block, os, kpack);
      } else {
        compute_conv_no_padding(ctx, config, fusion, output, input, weight,
          loops, K_num_block, C_num_block, os, kpack, use_os_blocking,
          pack_rows, os_blk_size, os_acc_size, os_mask);
      }
    } else {
      if (is_use_amx(ctx) && (ph_ <= kh_ && pw_ <= kw_)) {
        compute_conv_padding_v2(ctx, config, fusion, output, input, weight,
          loops, K_num_block, C_num_block, os, kpack);
      } else {
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
