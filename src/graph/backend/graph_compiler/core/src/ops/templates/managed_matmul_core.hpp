/*******************************************************************************
 * Copyright 2022-2024 Intel Corporation
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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_OPS_TEMPLATES_MANAGED_MATMUL_CORE_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_OPS_TEMPLATES_MANAGED_MATMUL_CORE_HPP

#include <memory>
#include <utility>
#include <vector>
#include <ops/body_generator.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace ops {
struct managed_matmul_core_config_t {
  int M_split_num;
  int N_split_num;
  int M_sub_block;
  int N_sub_block;
  int K_sub_block;
  // inner most loop order
  int im_loop_order;
};
class gen_managed_matmul_core_t
  : public body_generator_t<managed_matmul_core_config_t> {
public:
  // for specific shapes, dtypes and num_threads (refer to
  // ops::managed_matmul_core_op_t for exact value), we dispatch to avx instead
  // of amx.
  int64_t dispatch_avx_ = 0;

  int split_iim_ = -1;

  // inner most block
  int iim_block_;
  int iin_block_;
  int iik_block_;
  bool is_partial_; // for dynamic combined impl kind. Unused in static.
  struct op_params_t {
    static constexpr int in_A = 0;
    static constexpr int in_B = 1;
    static constexpr int out_C = 0;
  };
  using parent = body_generator_t<managed_matmul_core_config_t>;
  using parent::generate;

  gen_managed_matmul_core_t(sc_op *owner, std::vector<logical_tensor_t> &&ins,
    std::vector<logical_tensor_t> &&outs);

  float get_gflop() const override;

  const sc_dims get_a_batch_dims() const {
    return {in_tensors_[0].get_plain_dims().begin(),
      in_tensors_[0].get_plain_dims().end() - 2};
  }

  const sc_dims get_b_batch_dims() const {
    return {in_tensors_[1].get_plain_dims().begin(),
      in_tensors_[1].get_plain_dims().end() - 2};
  }

  const sc_dims get_mma_plain_dims() const {
    return {in_tensors_[0].get_plain_dims().begin() + get_a_batch_dims().size(),
      in_tensors_[0].get_plain_dims().end()};
  };

  const sc_dims get_mmb_plain_dims() const {
    return {in_tensors_[1].get_plain_dims().begin() + get_b_batch_dims().size(),
      in_tensors_[1].get_plain_dims().end()};
  };
  bool is_dynamic() const {
    return in_tensors_[0].is_dynamic() || in_tensors_[1].is_dynamic();
  }

  sc_data_type_t get_A_dtype() const { return in_tensors_[0].dtype_; }
  sc_data_type_t get_B_dtype() const { return in_tensors_[1].dtype_; }
  sc_data_type_t get_C_dtype() const { return out_tensors_[0].dtype_; }

  bool generate(context_ptr ctx, const managed_matmul_core_config_t &config,
    fusion_anchor_mgr_t *fusion, const std::vector<expr> &inputs,
    const std::vector<expr> &outputs,
    std::vector<for_loop> &loops) const override;

  func_t get_single_core_func(context_ptr ctx,
    const managed_matmul_core_config_t &config, fusion_anchor_mgr_t *fusion,
    const std::vector<expr> &inputs, const std::vector<expr> &outputs,
    std::vector<for_loop> &loops) const override;
  std::vector<expr> get_extra_args_from_func(const func_t &f) const override;

  void single_thread_matmul_call(const context_ptr &ctx, sc_graph_t &graph,
    const logical_tensor_t &ta, const logical_tensor_t &tb,
    const logical_tensor_t &tc, const managed_matmul_core_config_t &config,
    const expr &M, const expr &N, const expr &K, const expr &m_idx,
    const expr &n_idx, const expr &k_idx, const expr &A, const expr &B,
    const expr &C, int dtype_block, fusion_anchor_mgr_t *fusion,
    const expr &m_s, const expr &n_s, std::vector<int> &M_anchor_info,
    std::vector<int> &N_anchor_info, bool is_partial = false,
    const expr &k_s = expr(), bool is_dynamic = false,
    const expr &N_block_size_expr = expr()) const;

  void dynamic_single_thread_matmul_call(
    const managed_matmul_core_config_t &config,
    const std::vector<expr> &buffer_args, const expr &m_s, const expr &n_s,
    const expr &k_s, int K_split_num, expr &iim_block, expr &iin_block,
    expr &iik_block) const;

  void generate_prefetcher_body_for_tensor(const context_ptr &ctx,
    const managed_matmul_core_config_t &config,
    const std::vector<expr> &func_args, const std::vector<expr> &ins,
    const std::vector<int> &indices);

  bool is_okay_to_prefetch(const context_ptr &ctx,
    const managed_matmul_core_config_t &config, bool is_global);

  bool is_valid_config(const context_ptr &ctx,
    const managed_matmul_core_config_t &config) const override;
  config_ptr_vec get_dynamic_config_candidates(
    const context_ptr &ctx) const override;
  std::vector<uint64_t> convert_config_to_keys(
    const config_ptr &configs) const override;
  config_ptr get_default_config(context_ptr ctx) const override;
  config_ptr get_default_transposed_a_config(const context_ptr &ctx) const;
  config_ptr get_default_post_rd_config(const context_ptr &ctx) const;

  void schedule_loops(context_ptr ctx,
    const managed_matmul_core_config_t &config, stmt body,
    std::vector<for_loop> &fors) const override;

  int suggest_aligned_iim_block(
    const size_t plain_M, const size_t default_block, bool is_f32);
};
} // namespace ops
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
