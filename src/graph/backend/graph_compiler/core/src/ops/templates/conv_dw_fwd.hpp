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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_OPS_TEMPLATES_CONV_DW_FWD_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_OPS_TEMPLATES_CONV_DW_FWD_HPP

#include <memory>
#include <tuple>
#include <vector>
#include <ops/body_generator.hpp>
#include <util/any_map.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace ops {

struct conv_dw_fwd_config_t {
  int bs_threads = 1;
  int h_threads = 1;
  int w_threads = 1;
  int g_threads = 1;
  int h_block = 1;
  int w_block = 1;
  int g_block = 1;
  int im_h_block = 1;
  int im_w_block = 1;

  conv_dw_fwd_config_t() = default;

  conv_dw_fwd_config_t(int bs_threads, int h_threads, int w_threads,
    int g_threads, int h_block, int w_block, int g_block, int im_h_block,
    int im_w_block)
    : bs_threads(bs_threads)
    , h_threads(h_threads)
    , w_threads(w_threads)
    , g_threads(g_threads)
    , h_block(h_block)
    , w_block(w_block)
    , g_block(g_block)
    , im_h_block(im_h_block)
    , im_w_block(im_w_block) {}
};

class gen_conv_dw_fwd_t : public body_generator_t<conv_dw_fwd_config_t> {
public:
  struct op_params_t {
    static constexpr int data = 0;
    static constexpr int weight = 1;
    static constexpr int out = 0;
  };
  using parent = body_generator_t<conv_dw_fwd_config_t>;
  using parent::generate;

  std::tuple<int, int, int> get_output_shape() {
    return std::tuple<int, int, int>(od_, oh_, ow_);
  }

  gen_conv_dw_fwd_t(sc_op *owner, const sc_dims &stride,
    const sc_dims &dilation, const sc_dims &pads_begin, const sc_dims &pads_end,
    std::vector<logical_tensor_t> &&ins, std::vector<logical_tensor_t> &&outs);

  float get_gflop() const override;

  bool is_dynamic() const {
    return in_tensors_[0].is_dynamic() || in_tensors_[1].is_dynamic();
  }

  const sc_dims &get_input_plain_dims() const {
    return in_tensors_[0].get_plain_dims();
  }

  const sc_dims &get_input_blocking_dims() const {
    return in_tensors_[0].get_blocking_dims();
  }

  const sc_dims &get_weight_plain_dims() const {
    return in_tensors_[1].get_plain_dims();
  }
  const sc_dims &get_output_plain_dims() const {
    return out_tensors_[0].get_plain_dims();
  }

  const sc_dims &get_output_blocking_dims() const {
    return out_tensors_[0].get_blocking_dims();
  }

  sc_data_type_t get_input_dtype() const { return in_tensors_[0].dtype_; }
  sc_data_type_t get_weight_dtype() const { return in_tensors_[1].dtype_; }
  sc_data_type_t get_output_dtype() const { return out_tensors_[0].dtype_; }

  bool generate(context_ptr ctx, const conv_dw_fwd_config_t &config,
    fusion_anchor_mgr_t *fusion, const std::vector<expr> &inputs,
    const std::vector<expr> &outputs,
    std::vector<for_loop> &loops) const override;
  config_ptr_vec get_dynamic_config_candidates(
    const context_ptr &ctx) const override;
  // std::vector<uint64_t> convert_config_to_keys(
  //   const config_ptr &configs) const override;
  config_ptr get_default_config(context_ptr ctx) const override;

  void schedule_loops(context_ptr ctx, const conv_dw_fwd_config_t &config,
    stmt body, std::vector<for_loop> &fors) const override;

  int get_im_w_block(const context_ptr &ctx) const;

#define CONV_ARG_LIST \
  const context_ptr &ctx, const conv_dw_fwd_config_t &config, \
    fusion_anchor_mgr_t *fusion, expr &output, const expr &input, \
    const expr &weight, std::vector<for_loop> &loops
  void compute_conv_logical_padding(CONV_ARG_LIST) const;
  void compute_conv_physical_padding(CONV_ARG_LIST) const;
  void dynamic_conv_logical_padding(CONV_ARG_LIST) const;
  void dynamic_conv_physical_padding(CONV_ARG_LIST) const;
#undef CONV_ARG_LIST

  size_t ndims_ = 0;
  int groups_ = 1;
  int mb_ = 0, ic_ = 0, id_ = 0, ih_ = 0, iw_ = 0;
  int oc_ = 0, kd_ = 0, kh_ = 0, kw_ = 0;
  int od_ = 0, oh_ = 0, ow_ = 0;
  int sd_ = 0, sh_ = 0, sw_ = 0;
  int pd_b_ = 0, ph_b_ = 0, pw_b_ = 0;
  int pd_e_ = 0, ph_e_ = 0, pw_e_ = 0;
  int dd_ = 0, dh_ = 0, dw_ = 0;
  bool is_3d_ = false;
  any_map_t attrs_;
};

} // namespace ops
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
