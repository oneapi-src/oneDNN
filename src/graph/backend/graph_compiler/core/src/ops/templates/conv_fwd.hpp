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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_OPS_TEMPLATES_CONV_FWD_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_OPS_TEMPLATES_CONV_FWD_HPP

#include <memory>
#include <tuple>
#include <utility>
#include <vector>
#include <ops/body_generator.hpp>
#include <util/any_map.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace ops {

struct conv_fwd_config_t {
  int K_block = 0;
  int C_block = 0;
  int tile_d = 1;
  int tile_p = 1;
  int tile_q = 1;
  int tile_os = -1;
  int pack_input = 0;
  int loop_sched = 0;

  conv_fwd_config_t() = default;

  conv_fwd_config_t(int K_block, int C_block, int tile_d, int tile_p,
    int tile_q, int tile_os, int pack_input, int loop_sched)
    : K_block(K_block)
    , C_block(C_block)
    , tile_d(tile_d)
    , tile_p(tile_p)
    , tile_q(tile_q)
    , tile_os(tile_os)
    , pack_input(pack_input)
    , loop_sched(loop_sched) {}
};

class gen_conv_fwd_t : public body_generator_t<conv_fwd_config_t> {
public:
  struct op_params_t {
    static constexpr int in_data = 0;
    static constexpr int in_weight = 1;
    static constexpr int out = 0;
  };
  using parent = body_generator_t<conv_fwd_config_t>;
  using parent::generate;

  std::tuple<int, int, int> get_output_shape() {
    return std::tuple<int, int, int>(od_, oh_, ow_);
  }

  gen_conv_fwd_t(sc_op *owner, const sc_dims &stride, const sc_dims &pads_begin,
    std::vector<logical_tensor_t> &&ins, std::vector<logical_tensor_t> &&outs)
    : gen_conv_fwd_t(owner, stride, sc_dims {1}, pads_begin, pads_begin,
      std::move(ins), std::move(outs)) {}

  gen_conv_fwd_t(sc_op *owner, const sc_dims &stride, const sc_dims &pads_begin,
    const sc_dims &pads_end, std::vector<logical_tensor_t> &&ins,
    std::vector<logical_tensor_t> &&outs)
    : gen_conv_fwd_t(owner, stride, sc_dims {1}, pads_begin, pads_end,
      std::move(ins), std::move(outs)) {}

  gen_conv_fwd_t(sc_op *owner, const sc_dims &stride, const sc_dims &dilation,
    const sc_dims &pads_begin, const sc_dims &pads_end,
    std::vector<logical_tensor_t> &&ins, std::vector<logical_tensor_t> &&outs);

  void adjust_config_for_parallelisem(
    const context_ptr &ctx, conv_fwd_config_t &cfg) const;
  void adjust_config_for_cache_efficiency(
    const context_ptr &ctx, conv_fwd_config_t &cfg) const;

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

  bool generate(context_ptr ctx, const conv_fwd_config_t &config,
    fusion_manager *fusion, const std::vector<expr> &inputs,
    const std::vector<expr> &outputs,
    std::vector<for_loop> &loops) const override;
  config_ptr get_default_config(context_ptr ctx) const override;

  void schedule_loops(context_ptr ctx, const conv_fwd_config_t &config,
    stmt body, std::vector<for_loop> &fors) const override;

  std::vector<int> get_os_factors();

  bool inverse_filter_ = false;

#define CONV_ARG_LIST \
  const context_ptr &ctx, const conv_fwd_config_t &config, \
    fusion_manager *fusion, expr &output, const expr &input, \
    const expr &weight, std::vector<for_loop> &loops, const int K_num_block, \
    const int C_num_block, const int os, \
    const int kpack = 1, const bool use_os_blocking = false, \
              const bool pack_rows = false, const expr &os_acc_size = expr(), \
              const std::vector<char> &os_mask = std::vector<char>()
  void compute_1x1_no_pack_input(CONV_ARG_LIST) const;
  void compute_1x1_pack_input(CONV_ARG_LIST) const;
  void compute_conv3d_no_padding(CONV_ARG_LIST) const;
  void compute_conv_no_padding(CONV_ARG_LIST) const;
  void compute_conv_padding(CONV_ARG_LIST) const;
  void compute_conv_padding_v2(CONV_ARG_LIST) const;
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
  int actual_os_ = 0, adj_os_ = 0;
  int num_elems_skip_per_ow_ = 0;
  bool try_os_blocking_ = false;
  bool is_1x1_conv_ = false;
  bool is_3d_ = false;
  bool blocking_input_ = false;
  bool blocking_output_ = false;
  any_map_t attrs_;
  void validate_conv_fwd_default_config(
    const context_ptr &ctx, conv_fwd_config_t &cfg) const;
};

} // namespace ops
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
