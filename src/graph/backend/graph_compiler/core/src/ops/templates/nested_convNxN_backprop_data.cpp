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

#include "nested_convNxN_backprop_data.hpp"
#include <algorithm>
#include <limits>
#include <numeric>
#include <string>
#include <utility>
#include "utils.hpp"
#include <compiler/ir/builder.hpp>
#include <compiler/ir/builtin.hpp>
#include <compiler/ir/easy_build.hpp>
#include <compiler/ir/graph/fusion_mgr.hpp>
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

static std::vector<int> get_iota(int range) {
  std::vector<int> result(range);
  std::iota(result.begin(), result.end(), 1);
  return result;
}

config_ptr gen_nested_convNxN_backprop_data_t::get_default_config(
  context_ptr ctx) const {
  auto ret
    = reflection::general_object_t::make<nested_conv_bwd_data_config_t>();
  nested_conv_bwd_data_config_t &cfg
    = *ret.unchecked_get_as<nested_conv_bwd_data_config_t>();
  const int num_threads = runtime_config_t::get().get_num_threads();
  int BS = get_input_grad_dims()[0], IC = get_input_grad_dims()[1];
  int IH = get_input_grad_dims()[ndims_ - 2],
      IW = get_input_grad_dims()[ndims_ - 1];
  int OC = get_output_grad_dims()[1];

  int ih_threads = 1;
  float cost = std::numeric_limits<float>::max();
  for (int i = 1; i <= num_threads; i++) {
    if (num_threads % i != 0) continue;
    int num_bs_threads = utils::divide_and_ceil(BS, num_threads / i);
    int num_ih_threads = utils::divide_and_ceil(IH, i);
    int num_brgemm = BS * IH * 2 * IC / im_ic_block_;
    // Cost = Shape_efficient_weight *
    // (workload_balance + divide_N_plenty) / core_utilitizaiton
    // single core gemm prefers square shape for A and B.
    // For small workload, the A and B shape is not a key problem, but the
    // num_core and num_brgemm is important to performance. Use 2048 to reduce
    // the shape weight on small shape.
    float new_cost = (1024 + BS * i / float(num_threads) + IH / float(i))
      * (num_brgemm + 8 * i) / float(num_threads);
    if (new_cost < cost && i <= IH && num_threads / i <= BS) {
      ih_threads = i;
      cost = new_cost;
    }
  }
  cfg.bs_threads = num_threads / ih_threads;
  cfg.spatial_threads = ih_threads; // ih_threads
  cfg.ic_threads = 1;
  // when IH is small and IC is large, prefer split on IC
  if (IH < 28 && IC > 256) {
    int ic_max_threads = IC / im_ic_block_;
    auto possible_factors = get_splits(ih_threads);
    for (int64_t i = possible_factors.size() - 1; i >= 0; --i) {
      if (ic_max_threads % possible_factors[i] == 0) {
        cfg.ic_threads = possible_factors[i];
        cfg.spatial_threads = ih_threads / possible_factors[i];
        break;
      }
    }
  }

  cfg.ic_num_blocks = 1;
  cfg.oc_num_blocks = 1;
  cfg.bs_num_blocks = BS / cfg.bs_threads;
  cfg.spatial_num_blocks
    = IH < 28 ? 1 : IH / cfg.spatial_threads; // ih_num_blocks
  return std::move(ret);
}

gen_nested_convNxN_backprop_data_t::gen_nested_convNxN_backprop_data_t(
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
  ndims_ = get_output_grad_dims().size();
  int OW = get_output_grad_dims()[3];
  int OC = get_output_grad_dims()[1];
  // TODO(yifei): enhance default values to deal with more flexible configs
  if (is_vnni_low_fp) {
    im_oc_block_ = OC;
    im_ic_block_ = 32;
    im_ow_block_ = OW;
  } else {
    im_oc_block_ = OC;
    im_ic_block_ = 16;
    im_ow_block_ = OW;
  }
}

float gen_nested_convNxN_backprop_data_t::get_gflop() const {
  const int OD = ndims_ == 5 ? get_output_grad_dims()[ndims_ - 3] : 1;
  const int P = get_output_grad_dims()[ndims_ - 2];
  const int Q = get_output_grad_dims()[ndims_ - 1];
  const int C = get_input_grad_dims()[1];
  const int K = get_output_grad_dims()[1];
  const int N = get_output_grad_dims()[0];
  const int KD = ndims_ == 5 ? get_weight_dims()[ndims_ - 3] : 1;
  const int KH = get_weight_dims()[ndims_ - 2];
  const int KW = get_weight_dims()[ndims_ - 1];
  float result = 2.0f * N * K * C * KD * KH * KW * OD * P * Q / (float)1e9;
  return result;
}

void gen_nested_convNxN_backprop_data_t::schedule_loops(context_ptr ctx,
  const nested_conv_bwd_data_config_t &config, stmt body,
  std::vector<for_loop> &fors) const {}

void gen_nested_convNxN_backprop_data_t::pad_delta_output(
  const context_ptr &ctx, const expr &delta_output,
  const expr &temp_delta_output_buffer, const expr &bs_block, int oc_block,
  int OH, int OW, const expr &oh_range, int ow_ext_range, const expr &bs_offset,
  const expr &oh_offset, const expr &ow_offset, const expr &oc_offset,
  const expr &temp_oh_offset) const {
  trace_guard_t trg(ctx, "pad_delta_output");
  // pad a piece of NPQK to NP(Q+pad)K
  int lanes = vectorize_step(ctx, get_A_dtype().type_code_, 32);
  // TODO(yifei): fallback to vectorized + tail if not divisible
  if (oc_block < lanes || oc_block % lanes != 0) { lanes = 1; }
  _for_(obs_reorder, 0, bs_block) {
    _for_(oh_reorder, 0, oh_range) {
      _for_(ow_reorder, 0, OW + 2 * ow_ext_range) {
        _for_(oc_reorder, 0, oc_block, lanes) {
          expr obs_idx = bs_offset + obs_reorder;
          expr oh_idx = oh_offset + oh_reorder;
          expr ow_idx = ow_offset + ow_reorder;
          expr oc_idx = oc_offset + oc_reorder;
          std::vector<expr> tmp_idx {
            obs_reorder, oh_reorder, ow_reorder, oc_reorder};
          std::vector<expr> delta_output_idx {
            obs_idx, oh_idx, ow_idx - ow_ext_range, oc_idx};
          _if_((oh_idx >= 0 && oh_idx < OH)
            && (ow_idx >= ow_ext_range && ow_idx < OW + ow_ext_range)) {
            temp_delta_output_buffer[span_t(tmp_idx, lanes)]
              = delta_output[span_t(delta_output_idx, lanes)];
          }
          _else_ {
            temp_delta_output_buffer[span_t(tmp_idx, lanes)]
              = builder::make_broadcast(
                builder::make_cast(get_A_dtype().type_code_, 0), lanes);
          }
        }
      }
    }
  }
}

void gen_nested_convNxN_backprop_data_t::inner_loop_call(const context_ptr &ctx,
  const expr &delta_input, const expr &delta_output, const expr &weight,
  const sc_data_type_t &dtype, int dtype_block, int ic_block, int oc_block,
  const expr &bs_block, int od_block, const expr &ih_block, int OW,
  int stride_h, int stride_w, int padding_h, int padding_w, int R, int S,
  int IC, int OC, int OH, int IW, const expr &obs_offset, const expr &oc_offset,
  const expr &ic_offset, const expr &ih_offset, fusion_manager *fusion) const {
  COMPILE_ASSERT(OW == im_ow_block_, "Use fixed config OW == im_ow_block_.");
  COMPILE_ASSERT(OC == im_oc_block_, "Use fixed config OC == im_oc_block_.");
  int num = oc_block / im_oc_block_;
  COMPILE_ASSERT(num == 1, "num is temporarily 1.");
  // TODO(yifei): Figure out why the hint size here lead to performance benefit
  const auto hint_A_size = 32 * im_oc_block_ * R;
  const auto hint_B_size = 32 * im_oc_block_ * R * S;
  const auto hint_C_size = 32 * 32;
  sc_brgemm_attrs_t brg_attrs {{brgemm::attr_key::max_bs, num * R * S},
    {brgemm::attr_key::hint_expected_A_size, hint_A_size},
    {brgemm::attr_key::hint_expected_B_size, hint_B_size},
    {brgemm::attr_key::hint_expected_C_size, hint_C_size},
    {brgemm::attr_key::use_interleave_stores, false},
    {brgemm::attr_key::use_uker, false},
    {brgemm::attr_key::hint_innermost_loop, true},
    {brgemm::attr_key::hint_prefetching, 2}};
  expr oh_ext_range = divide_and_ceil(ih_block + R, stride_h) + 1;
  // ow_ext_range is the potential overflow on either left or right
  // TODO(yifei): consider non-symmetric padding
  int ow_ext_range = utils::divide_and_ceil(padding_w, stride_w);
  _tensor_(temp_delta_output, get_A_dtype(),
    {bs_block, oh_ext_range, OW + 2 * ow_ext_range, oc_block});
  _var_(oh_offset, datatypes::index);
  _if_(ih_offset + padding_h < R) { oh_offset = 0; }
  _else_ { oh_offset = (ih_offset + padding_h - R) / stride_h; }
  // it will pad OW on both lhs and rhs by ow_ext_range
  pad_delta_output(ctx, delta_output, temp_delta_output, bs_block, oc_block, OH,
    OW, oh_ext_range, ow_ext_range, obs_offset, oh_offset, 0, oc_offset, 0);
  _for_(i_ic, 0, ic_block / im_ic_block_) {
    _for_(i_bs, 0, bs_block) {
      _for_(i_ih, 0, ih_block) {
        expr ih_idx = ih_offset + i_ih;
        // start the real inner most computation
        _for_(sw_idx, 0, stride_w, 1) {
          _tensor_(
            temp_delta_input, datatypes::f32, {im_ow_block_, im_ic_block_});
          _var_(len, datatypes::s32);
          len = 0;
          _tensor_(A_list, datatypes::pointer, {num * R * S});
          _tensor_(B_list, datatypes::pointer, {num * R * S});
          // the value of sh_idx ensuring oh_idx is an integer
          expr sh_idx = (ih_idx + padding_h) % stride_h;
          _for_(r, sh_idx, R, stride_h) {
            // avoid out of bound and calculation overflow
            _if_(ih_idx + padding_h >= r) {
              expr oh_idx = (ih_idx + padding_h - r) / stride_h;
              _if_(oh_idx < OH) {
                _for_(s, sw_idx, S, stride_w) {
                  // padding_w + iw_start = ow_start * stride_w + s;
                  // the starting point of ow for a valid (in bound) iw_start
                  _var_(ow_start, datatypes::index);
                  _if_(s < padding_w) {
                    ow_start
                      = ow_ext_range + divide_and_ceil(padding_w - s, stride_w);
                  }
                  _else_ {
                    _if_((OW - 1) * stride_w + s >= IW + padding_w) {
                      ow_start = ow_ext_range
                        - divide_and_ceil(
                          (OW - 1) * stride_w + s - (IW + padding_w - 1),
                          stride_w);
                    }
                    _else_ { ow_start = ow_ext_range; }
                  }
                  std::vector<expr> tmp_delta_output_index {
                    i_bs, oh_idx - oh_offset, ow_start, 0};
                  auto weight_index = dtype_block > 1
                    ? std::vector<expr> {ic_offset / im_ic_block_ + i_ic, 0, r,
                      s, 0, 0, 0}
                    : std::vector<expr> {
                      ic_offset / im_ic_block_ + i_ic, 0, r, s, 0, 0};
                  A_list[len]
                    = tensor_ptr(temp_delta_output, tmp_delta_output_index);
                  B_list[len] = tensor_ptr(weight, weight_index);
                  len = len + 1;
                }
              }
            }
          }
          trace_guard_t trg(ctx, "brgemm + final copy");
          builtin::brgemm_init_list_update(A_list, B_list,
            tensor_ptr(temp_delta_input, {0, 0}), 1, im_ow_block_, im_ic_block_,
            im_oc_block_, OC, im_ic_block_, im_ic_block_, 1, 1, len, dtype,
            dtype, brg_attrs);
          // copy temp_delta_input to delta_input
          int lanes = vectorize_step(ctx, get_C_dtype().type_code_, 32);
          if (im_ic_block_ < lanes || im_ic_block_ % lanes != 0) { lanes = 1; }
          _for_(ow_copy, 0, im_ow_block_) {
            _for_(ic_copy, 0, im_ic_block_, lanes) {
              _var_(ow_idx, datatypes::index);
              _var_(iw_idx, datatypes::index);
              _if_(sw_idx < padding_w) {
                ow_idx = divide_and_ceil(padding_w - sw_idx, stride_w);
              }
              _else_ { ow_idx = 0; }
              iw_idx = (ow_idx + ow_copy) * stride_w + sw_idx - padding_w;
              std::vector<expr> delta_input_index {obs_offset + i_bs, ih_idx,
                iw_idx, ic_offset + i_ic * im_ic_block_ + ic_copy};
              std::vector<expr> temp_output_index {ow_copy, ic_copy};
              delta_input[span_t(delta_input_index, lanes)]
                = temp_delta_input[span_t(temp_output_index, lanes)];
              if (fusion) {
                fusion->create_output_fusion_anchor({tensor_slice(delta_input,
                  {{obs_offset + i_bs, 1}, {ih_idx, 1}, {iw_idx, 1},
                    {ic_offset + i_ic * im_ic_block_ + ic_copy, lanes}})});
              }
            }
          }
        }
        if (fusion) {
          fusion->create_output_fusion_anchor({tensor_slice(delta_input,
            {{obs_offset + i_bs, 1}, {ih_idx, 1}, {0, IW},
              {ic_offset + i_ic * im_ic_block_, im_ic_block_}})});
        }
      }
    }
  }
}

bool gen_nested_convNxN_backprop_data_t::generate(context_ptr ctx,
  const nested_conv_bwd_data_config_t &config, fusion_manager *fusion,
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

  const int num_threads = runtime_config_t::get().get_num_threads();

  // setting dim values
  int BS = get_output_grad_dims()[0], IC = get_input_grad_dims()[1];
  int ID = ndims_ == 5 ? get_input_grad_dims()[ndims_ - 3] : 1;
  int IH = get_input_grad_dims()[ndims_ - 2],
      IW = get_input_grad_dims()[ndims_ - 1];
  int OC = get_output_grad_dims()[1];
  int OD = ndims_ == 5 ? get_output_grad_dims()[ndims_ - 3] : 1;
  int OH = get_output_grad_dims()[2], OW = get_output_grad_dims()[3];
  int KD = ndims_ == 5 ? get_weight_dims()[ndims_ - 3] : 1;
  int R = get_weight_dims()[ndims_ - 2], S = get_weight_dims()[ndims_ - 1];
  // setting configs
  int bs_threads = config.bs_threads, ic_threads = config.ic_threads,
      ih_threads = config.spatial_threads;
  int ic_num_blocks = config.ic_num_blocks,
      bs_num_blocks = config.bs_num_blocks,
      ih_num_blocks = config.spatial_num_blocks;
  COMPILE_ASSERT(config.oc_num_blocks == 1,
    "oc_num_blocks is not used in convNxN_bwd_data, so it shall be 1.");

  COMPILE_ASSERT(BS >= bs_threads, "BS shall be larger than bs_threads.");
  COMPILE_ASSERT(IH >= ih_threads, "IH shall be larger than ih_threads.");
  COMPILE_ASSERT(IC % ic_threads == 0, "imbalance on IC not supported.");
  COMPILE_ASSERT(num_threads == bs_threads * ih_threads * ic_threads,
    "All threads must be utilized.");

  // other template related pre-compute values
  auto dtype = get_A_dtype();
  bool is_vnni_low_fp = ops::is_vnni_low_fp(ctx, dtype);
  int dtype_block = is_vnni_low_fp ? 2 : 1;
  int ic_single_core = IC / ic_threads;
  int ih_single_core = IH / ih_threads;
  expr bs_single_core_size, ih_single_core_size;
  expr p_bs_offset, p_ih_offset;
  expr X_bigger_num;

  // define compute
  expr delta_input = outputs.at(op_params_t::out_input_grad),
       delta_output = inputs.at(op_params_t::in_output_grad),
       weight = inputs.at(op_params_t::in_weight);

  _for_(p_bs, 0, bs_threads, 1, for_type::PARALLEL, bs_threads) {
    _for_(p_ih, 0, ih_threads, 1, for_type::PARALLEL, ih_threads) {
      _for_(p_ic, 0, ic_threads, 1, for_type::PARALLEL, ic_threads) {
        bs_single_core_size = get_balance211_length(
          BS, bs_threads, p_bs, p_bs_offset, X_bigger_num);
        ih_single_core_size = get_balance211_length(
          IH, ih_threads, p_ih, p_ih_offset, X_bigger_num);
        // start single core computation
        _for_(o_bs, 0, bs_num_blocks) {
          expr bs_block_offset, bs_block_bigger_num;
          _var_init_(bs_block_size, datatypes::s32,
            get_balance211_length(bs_single_core_size, bs_num_blocks, o_bs,
              bs_block_offset, bs_block_bigger_num));
          _var_init_(
            obs_offset, datatypes::index, p_bs_offset + bs_block_offset);
          _for_(o_ih, 0, ih_num_blocks) {
            expr ih_block_offset, ih_block_bigger_num;
            _var_init_(ih_block_size, datatypes::s32,
              get_balance211_length(ih_single_core_size, ih_num_blocks, o_ih,
                ih_block_offset, ih_block_bigger_num));
            _var_init_(
              ih_offset, datatypes::index, p_ih_offset + ih_block_offset);
            _for_(o_ic, 0, ic_num_blocks) {
              int ic_block = ic_single_core / ic_num_blocks;
              expr oc_offset = 0;
              expr ic_offset = p_ic * ic_single_core + o_ic * ic_block;
              inner_loop_call(ctx, delta_input, delta_output, weight, dtype,
                dtype_block, ic_block, OC, bs_block_size, 0, ih_block_size, OW,
                stride_h, stride_w, padding_h, padding_w, R, S, IC, OC, OH, IW,
                obs_offset, oc_offset, ic_offset, ih_offset, fusion);
            }
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
