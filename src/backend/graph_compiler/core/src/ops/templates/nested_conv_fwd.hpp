/*******************************************************************************
 * Copyright 2022 Intel Corporation
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

#ifndef BACKEND_GRAPH_COMPILER_CORE_SRC_OPS_TEMPLATES_NESTED_CONV_FWD_HPP
#define BACKEND_GRAPH_COMPILER_CORE_SRC_OPS_TEMPLATES_NESTED_CONV_FWD_HPP

#include <memory>
#include <tuple>
#include <vector>
#include <ops/body_generator.hpp>
#include <util/any_map.hpp>

namespace sc {
namespace ops {

struct nested_conv_fwd_config_t {
  int K_block;
  int C_block;
  int bs_threads = 1;
  int h_threads = 1;
  int w_threads = 1;
  int oc_threads = 1;
  int h_block = -1;
  int w_block = -1;
  int pack_input = 1;

  // keep this for tuning
  int im_oc_block = -1;
  int im_ic_block = -1;
  int im_h_block = -1;
  int im_w_block = -1;

  nested_conv_fwd_config_t() = default;

  nested_conv_fwd_config_t(int bs_threads, int h_threads, int w_threads,
    int oc_threads, int oc_block, int ic_block, int h_block, int w_block,
    int pack_input, int loop_sched)
    : K_block(oc_block)
    , C_block(ic_block)
    , bs_threads(bs_threads)
    , h_threads(h_threads)
    , w_threads(w_threads)
    , oc_threads(oc_threads)
    , h_block(h_block)
    , w_block(w_block)
    , pack_input(pack_input) {}
};

class gen_nested_conv_fwd_t
  : public body_generator_t<nested_conv_fwd_config_t> {
public:
  struct op_params_t {
    static constexpr int in_data = 0;
    static constexpr int in_weight = 1;
    static constexpr int out = 0;
  };
  using parent = body_generator_t<nested_conv_fwd_config_t>;
  using parent::generate;

  std::tuple<int, int, int> get_output_shape() {
    return std::tuple<int, int, int>(od_, oh_, ow_);
  }

  gen_nested_conv_fwd_t(sc_op *owner, const sc_dims &stride,
    const sc_dims &pads_begin, std::vector<logical_tensor_t> &&ins,
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

  bool generate(context_ptr ctx, const nested_conv_fwd_config_t &config,
    fusion_manager *fusion, const std::vector<expr> &inputs,
    const std::vector<expr> &outputs,
    std::vector<for_loop> &loops) const override;
  config_ptr get_default_config(context_ptr ctx) const override;

  void schedule_loops(context_ptr ctx, const nested_conv_fwd_config_t &config,
    stmt body, std::vector<for_loop> &fors) const override;

  bool inverse_filter_ = false;

#define CONV_ARG_LIST \
  const context_ptr &ctx, const nested_conv_fwd_config_t &config, \
    fusion_manager *fusion, expr &output, const expr &input, \
    const expr &weight, std::vector<for_loop> &loops, const int os, \
    const int kpack = 1, const bool use_os_blocking = false, \
              const bool pack_rows = false, const expr &os_acc_size = expr(), \
              const std::vector<char> &os_mask = std::vector<char>()

  void compute_1x1_pack_input_nested(CONV_ARG_LIST) const;
  void compute_1x1_no_pack_input_nested(CONV_ARG_LIST) const;
  void compute_conv_no_padding_nested(CONV_ARG_LIST) const;
  void compute_conv_no_padding_os_blocking_nested(CONV_ARG_LIST) const;
#undef CONV_ARG_LIST

  size_t ndims_ = 0;
  int mb_ = 0, ic_ = 0, id_ = 0, ih_ = 0, iw_ = 0;
  int oc_ = 0, kd_ = 0, kh_ = 0, kw_ = 0;
  int od_ = 0, oh_ = 0, ow_ = 0;
  int sd_ = 0, sh_ = 0, sw_ = 0;
  int pd_ = 0, ph_ = 0, pw_ = 0;
  int actual_os_ = 0, adj_os_ = 0;
  int num_elems_skip_per_ow_ = 0;
  bool try_os_blocking_ = false;
  bool is_1x1_conv_ = false;
  bool is_3d_ = false;
  bool is_1d_ = false;
  bool use_nested_2d_ = false;
  bool blocking_input_ = false;
  bool blocking_output_ = false;
  sc::any_map_t attrs_;
};

} // namespace ops
} // namespace sc

#endif
