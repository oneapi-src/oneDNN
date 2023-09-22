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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_OPS_TEMPLATES_NESTED_CONV_FWD_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_OPS_TEMPLATES_NESTED_CONV_FWD_HPP

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

  void generate_brgemm(const expr &im_s_block, int im_ic_block, int im_oc_block,
    int ic_block, const expr &o_ic, int ic_num_block_pt, const expr &A_list,
    const expr &B_list, const expr &out_tensor, const expr &LDA,
    const expr &LDC) const;

  bool generate(context_ptr ctx, const nested_conv_fwd_config_t &config,
    fusion_manager *fusion, const std::vector<expr> &inputs,
    const std::vector<expr> &outputs,
    std::vector<for_loop> &loops) const override;
  config_ptr_vec get_dynamic_config_candidates(
    const context_ptr &ctx) const override;
  std::vector<uint64_t> convert_config_to_keys(
    const config_ptr &configs) const override;
  config_ptr get_default_config(context_ptr ctx) const override;

  void schedule_loops(context_ptr ctx, const nested_conv_fwd_config_t &config,
    stmt body, std::vector<for_loop> &fors) const override;

  bool inverse_filter_ = false;

  int get_im_w_block(const context_ptr &ctx) const;
  int get_im_oc_block(const context_ptr &ctx) const;
  int get_im_ic_block(const context_ptr &ctx) const;

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
  void compute_conv1d(CONV_ARG_LIST) const;
  void dynamic_compute_conv_no_padding_nested(CONV_ARG_LIST) const;
  void dynamic_compute_conv_padding_nested(CONV_ARG_LIST) const;
  void dynamic_compute_1x1_pack_input_nested(CONV_ARG_LIST) const;
#undef CONV_ARG_LIST

  void single_thread_conv_padding_call(expr &output, const expr &input,
    const expr &weight, const expr &pbs, const expr &poc, const expr &ph,
    const expr &pw, const expr &pic, const expr &outer_k,
    const expr &h_num_block, const int h_num_block_pt, const expr &w_num_block,
    const int w_num_block_pt, const expr &oc_num_block,
    const int oc_num_block_pt, const expr &ic_num_block,
    const int ic_num_block_pt, const expr &pbuffer, for_loop &loh,
    for_loop &low, for_loop &looc, for_loop &loic, for_loop &lioc,
    for_loop &lih, for_loop &liw, const int oc_split,
    const int src_row_tile_size, const uint32_t lanes,
    const nested_conv_fwd_config_t &config, fusion_manager *fusion,
    const int ic_used_threads, const int oh_used_threads,
    const int ow_used_threads, const int y_unpad_top, const int y_unpad_bottom,
    const int y_unpad_left, const int y_unpad_right, const int iw_padded,
    const int kpack) const;

  void single_thread_dynamic_conv_padding_call(expr &output, const expr &input,
    const expr &weight, const expr &pbs, const expr &poc, const expr &ph,
    const expr &pw, const expr &pic, const expr &outer_k,
    const expr &h_num_block, const expr &h_num_block_pt,
    const expr &w_num_block, const expr &w_num_block_pt,
    const expr &oc_num_block, const int oc_num_block_pt,
    const expr &ic_num_block, const int ic_num_block_pt, const expr &pbuffer,
    for_loop &loh, for_loop &low, for_loop &looc, for_loop &loic,
    for_loop &lioc, for_loop &lih, for_loop &liw, const int oc_split,
    const expr &src_row_tile_size, const uint32_t lanes,
    const nested_conv_fwd_config_t &config, fusion_manager *fusion,
    const int ic_used_threads, const int oc_used_threads,
    const expr &oh_used_threads, const expr &ow_used_threads,
    const expr &y_unpad_top, const expr &y_unpad_bottom,
    const expr &y_unpad_left, const expr &y_unpad_right, const expr &iw_padded,
    const int kpack, const expr &h_block, const expr &w_block,
    const expr &im_h_block, const expr &im_w_block, const expr &oh_expr_,
    const expr &ow_expr_, const expr &ih_expr_, const expr &iw_expr_,
    expr &cond_tail_h, expr &cond_tail_w, int oc_block, int ic_block) const;

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
  int default_im_block_ = 64;
  int im_oc_block_, im_ic_block_, im_w_block_, im_h_block_;
  int num_elems_skip_per_ow_ = 0;
  bool try_os_blocking_ = false;
  bool is_1x1_conv_ = false;
  bool is_3d_ = false;
  bool is_1d_ = false;
  bool use_conv1d = false;
  bool use_nested_2d_ = false;
  bool blocking_input_ = false;
  bool blocking_output_ = false;
  any_map_t attrs_;
};

} // namespace ops
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
