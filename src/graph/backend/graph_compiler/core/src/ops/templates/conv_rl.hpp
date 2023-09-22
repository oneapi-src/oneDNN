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
#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_OPS_TEMPLATES_CONV_RL_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_OPS_TEMPLATES_CONV_RL_HPP

#include <tuple>
#include <vector>
#include <ops/body_generator.hpp>
#include <util/any_map.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace ops {

struct conv_fwd_rl_config_t {
  int brgemm_m = 1;
  int brgemm_n = 1;
  conv_fwd_rl_config_t() = default;
  conv_fwd_rl_config_t(int brgemm_m, int brgemm_n)
    : brgemm_m(brgemm_m), brgemm_n(brgemm_n) {}
};

enum class parallel_kind : int { BATCH = 0, WIDTH };
namespace rl_kind {
constexpr int NO_LOWERING = 0;
constexpr int FULL_LOWERING = 1;
constexpr int KW_LOWERING = 2;
} // namespace rl_kind
// enum class rl_kind : int { NO_LOWERING = 0, FULL_LOWERING, KW_LOWERING };

class gen_conv_fwd_rl_t : public body_generator_t<conv_fwd_rl_config_t> {
public:
  using parent = body_generator_t<conv_fwd_rl_config_t>;
  using parent::generate;

  std::tuple<int, int> get_output_shape() {
    return std::tuple<int, int>(oh_, ow_);
  }

  gen_conv_fwd_rl_t(sc_op *owner, const sc_dims &stride,
    const sc_dims &dilations, const sc_dims &pads_begin,
    const sc_dims &pads_end, std::vector<logical_tensor_t> &&ins,
    std::vector<logical_tensor_t> &&outs);

  float get_gflop() const override;

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

  bool generate(context_ptr ctx, const conv_fwd_rl_config_t &config,
    fusion_manager *fusion, const std::vector<expr> &inputs,
    const std::vector<expr> &outputs,
    std::vector<for_loop> &loops) const override;
  config_ptr get_default_config(context_ptr ctx) const override;
  void validate_default_config(
    const context_ptr &ctx, conv_fwd_rl_config_t &cfg) const;

  void schedule_loops(context_ptr ctx, const conv_fwd_rl_config_t &config,
    stmt body, std::vector<for_loop> &fors) const override;

  size_t ndims_ = 0;
  int groups_ = 1;
  int mb_ = 0, ic_ = 0, ih_ = 0, iw_ = 0;
  int oc_ = 0, oh_ = 0, ow_ = 0;
  int kh_ = 0, kw_ = 0;
  int sh_ = 0, sw_ = 0;
  int pt_ = 0, pb_ = 0, pl_ = 0, pr_ = 0;
  int actual_ih_ = 0, actual_iw_ = 0;
  int extra_padding_ = 0;
  int aux_buf_size_ = 0;
  int LDA_ = 0;
  int num_brgemm_k_ = 0, brgemm_k_ = 0;
  uint64_t init_mask_ = 0, update_mask_ = 0;
  int init_lanes_ = 0, update_lanes_ = 0;
  parallel_kind parallel_axis_;
  any_map_t attrs_;
};
} // namespace ops
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
