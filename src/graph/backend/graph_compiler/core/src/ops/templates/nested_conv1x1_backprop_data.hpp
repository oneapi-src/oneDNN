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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_OPS_TEMPLATES_NESTED_CONV1X1_BACKPROP_DATA_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_OPS_TEMPLATES_NESTED_CONV1X1_BACKPROP_DATA_HPP

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
class gen_nested_conv1x1_backprop_data_t
  : public body_generator_t<nested_conv_bwd_data_config_t> {
public:
  // inner most block
  int im_bs_block_;
  int im_ow_block_;
  int im_ic_block_;
  int im_oc_block_;
  sc_dims stride_;
  sc_dims padding_;
  struct op_params_t {
    static constexpr int in_fwd_output = 0;
    static constexpr int in_weight = 1;
    static constexpr int out_del_input = 0;
  };
  using parent = body_generator_t<nested_conv_bwd_data_config_t>;
  using parent::generate;

  gen_nested_conv1x1_backprop_data_t(sc_op *owner, const sc_dims &stride,
    const sc_dims &padding, std::vector<logical_tensor_t> &&ins,
    std::vector<logical_tensor_t> &&outs);

  float get_gflop() const override;

  const sc_dims &get_input_dims() const {
    return in_tensors_[0].get_plain_dims();
  }
  const sc_dims &get_weight_dims() const {
    return in_tensors_[1].get_plain_dims();
  }
  const sc_dims &get_output_dims() const {
    return out_tensors_[0].get_plain_dims();
  }
  sc_data_type_t get_dtype() const { return in_tensors_[0].dtype_; }
  sc_data_type_t get_weight_dtype() const { return in_tensors_[1].dtype_; }
  sc_data_type_t get_out_dtype() const { return out_tensors_[0].dtype_; }

  void weight_reorder(context_ptr ctx, const expr &temp_weight,
    const expr &weight, const sc_data_type_t &dtype, int oc_single_thr,
    int ic_single_thr, int OC, int IC, const expr &oc_s,
    const expr &ic_s) const;

  bool generate(context_ptr ctx, const nested_conv_bwd_data_config_t &config,
    fusion_manager *fusion, const std::vector<expr> &inputs,
    const std::vector<expr> &outputs,
    std::vector<for_loop> &loops) const override;

  void single_thread_conv1x1_backprop_data_call(const context_ptr &ctx,
    const logical_tensor_t &ta, const logical_tensor_t &tb,
    const logical_tensor_t &tc, const nested_conv_bwd_data_config_t &config,
    const expr &BS, const expr &S, const expr &IC, const expr &OC,
    const expr &bs_idx, const expr &s_idx, const expr &ic_idx,
    const expr &oc_idx, const int stride_d, const int stride_h,
    const int stride_w, const expr &A, const expr &B, const expr &C,
    int dtype_block, fusion_manager *fusion, const expr &bs_s, const expr &s_s,
    const expr &ic_s, std::vector<int> &BS_anchor_info,
    std::vector<int> &S_anchor_info, std::vector<int> &IC_anchor_info,
    const bool is_out_blocking, bool is_partial = false,
    const expr &oc_s = 0) const;

  config_ptr get_default_config(context_ptr ctx) const override;

  void schedule_loops(context_ptr ctx,
    const nested_conv_bwd_data_config_t &config, stmt body,
    std::vector<for_loop> &fors) const override;

private:
  int ndims_ = 0;
};
} // namespace ops

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
