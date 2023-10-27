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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_OPS_TEMPLATES_NESTED_CONVNXN_BACKPROP_DATA_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_OPS_TEMPLATES_NESTED_CONVNXN_BACKPROP_DATA_HPP

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
class gen_nested_convNxN_backprop_data_t
  : public body_generator_t<nested_conv_bwd_data_config_t> {
public:
  // inner most block
  int im_ow_block_;
  int im_ic_block_;
  int im_oc_block_;
  sc_dims stride_;
  sc_dims padding_;
  int ndims_;
  struct op_params_t {
    static constexpr int in_output_grad = 0;
    static constexpr int in_weight = 1;
    static constexpr int out_input_grad = 0;
  };
  using parent = body_generator_t<nested_conv_bwd_data_config_t>;
  using parent::generate;

  gen_nested_convNxN_backprop_data_t(sc_op *owner, const sc_dims &stride,
    const sc_dims &padding, std::vector<logical_tensor_t> &&ins,
    std::vector<logical_tensor_t> &&outs);

  float get_gflop() const override;

  const sc_dims &get_output_grad_dims() const {
    return in_tensors_[op_params_t::in_output_grad].get_plain_dims();
  }
  const sc_dims &get_weight_dims() const {
    return in_tensors_[op_params_t::in_weight].get_plain_dims();
  }
  const sc_dims &get_input_grad_dims() const {
    return out_tensors_[op_params_t::out_input_grad].get_plain_dims();
  }

  sc_data_type_t get_A_dtype() const { return in_tensors_[0].dtype_; }
  sc_data_type_t get_B_dtype() const { return in_tensors_[1].dtype_; }
  sc_data_type_t get_C_dtype() const { return out_tensors_[0].dtype_; }

  void pad_delta_output(const context_ptr &ctx, const expr &delta_output,
    const expr &temp_delta_output_buffer, const expr &bs_block, int oc_block,
    int OH, int OW, const expr &oh_range, int ow_range, const expr &bs_offset,
    const expr &oh_offset, const expr &ow_offset, const expr &oc_offset,
    const expr &temp_oh_offset) const;

  void inner_loop_call(const context_ptr &ctx, const expr &delta_input,
    const expr &delta_output, const expr &weight, const sc_data_type_t &dtype,
    int dtype_block, int ic_block, int oc_block, const expr &bs_block,
    int od_block, const expr &ih_block, int OW, int stride_h, int stride_w,
    int padding_h, int padding_w, int R, int S, int IC, int OC, int IH, int IW,
    const expr &obs_offset, const expr &oc_offset, const expr &ic_offset,
    const expr &ih_offset, fusion_anchor_mgr_t *fusion) const;

  bool generate(context_ptr ctx, const nested_conv_bwd_data_config_t &config,
    fusion_anchor_mgr_t *fusion, const std::vector<expr> &inputs,
    const std::vector<expr> &outputs,
    std::vector<for_loop> &loops) const override;

  config_ptr get_default_config(context_ptr ctx) const override;

  void schedule_loops(context_ptr ctx,
    const nested_conv_bwd_data_config_t &config, stmt body,
    std::vector<for_loop> &fors) const override;
};
} // namespace ops
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
