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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_OPS_TEMPLATES_NESTED_CONV1X1_BACKPROP_WEIGHT_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_OPS_TEMPLATES_NESTED_CONV1X1_BACKPROP_WEIGHT_HPP

#include <memory>
#include <tuple>
#include <vector>
#include "conv_bwd.hpp"
#include <ops/body_generator.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace ops {
class gen_nested_conv1x1_backprop_weight_t
  : public body_generator_t<nested_conv_bwd_weight_config_t> {
public:
  // inner most block
  int im_oc_block_;
  int im_ic_block_;
  int im_bs_block_;
  sc_dims stride_;
  sc_dims padding_;
  int ndims_;
  struct op_params_t {
    static constexpr int in_forward_input = 0;
    static constexpr int in_delta_output = 1;
    static constexpr int out_delta_weight = 0;
  };
  using parent = body_generator_t<nested_conv_bwd_weight_config_t>;
  using parent::generate;

  bool bwise_fusion_ = false;

  gen_nested_conv1x1_backprop_weight_t(sc_op *owner, const sc_dims &stride,
    const sc_dims &padding, std::vector<logical_tensor_t> &&ins,
    std::vector<logical_tensor_t> &&outs);

  float get_gflop() const override;

  const sc_dims &get_data_dims() const {
    return in_tensors_[op_params_t::in_forward_input].get_plain_dims();
  }
  const sc_dims &get_grad_dims() const {
    return in_tensors_[op_params_t::in_delta_output].get_plain_dims();
  }
  const sc_dims &get_output_dims() const {
    return out_tensors_[op_params_t::out_delta_weight].get_plain_dims();
  }

  sc_data_type_t get_A_dtype() const { return in_tensors_[0].dtype_; }
  sc_data_type_t get_B_dtype() const { return in_tensors_[1].dtype_; }
  sc_data_type_t get_C_dtype() const { return out_tensors_[0].dtype_; }

  void forward_input_reorder_call(context_ptr &ctx,
    const expr &temp_forward_input, const expr &forward_input,
    const logical_tensor_t &input_lt, const sc_data_type_t &dtype,
    int bs_single_core, int ic_single_core, int oh_single_core, int OW, int IH,
    int IW, const expr &bs_offset, const expr &ic_offset, const expr &oh_offset,
    const expr &ow_offset, int stride_h, int stride_w) const;

  void inner_loop_call(context_ptr &ctx, const expr &temp_forward_input,
    const std::vector<expr> &temp_forward_idx_non_block,
    const logical_tensor_t &delta_output_lt, const expr &delta_output,
    const expr &temp_delta_weight, const std::vector<expr> &temp_weight_idx,
    const sc_data_type_t &dtype, int dtype_block, int ic_block, int oc_block,
    int bs_block, int od_block, int oh_block, int ow_block, int stride_h,
    int stride_w, const expr &o_bs, const expr &o_od, const expr &o_oh,
    const expr &o_ow, const expr &obs_offset, const expr &oc_offset,
    const expr &oh_offset, const expr &ow_offset, fusion_manager *fusion,
    bool is_partial) const;

  bool generate(context_ptr ctx, const nested_conv_bwd_weight_config_t &config,
    fusion_manager *fusion, const std::vector<expr> &inputs,
    const std::vector<expr> &outputs,
    std::vector<for_loop> &loops) const override;

  config_ptr get_default_config(context_ptr ctx) const override;

  void schedule_loops(context_ptr ctx,
    const nested_conv_bwd_weight_config_t &config, stmt body,
    std::vector<for_loop> &fors) const override;
};
} // namespace ops
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
