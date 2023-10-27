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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_OPS_TEMPLATES_MATMUL_CORE_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_OPS_TEMPLATES_MATMUL_CORE_HPP

#include <memory>
#include <utility>
#include <vector>
#include <ops/body_generator.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace ops {
struct matmul_core_config_t {
  int M_block;
  int N_block;
  int K_block;
};

struct blocking_axis_t {
  std::vector<int> A_bs;
  std::vector<int> A_m;
  std::vector<int> A_k;
  std::vector<int> B_bs;
  std::vector<int> B_k;
  std::vector<int> B_n;
  std::vector<int> C_bs;
  std::vector<int> C_m;
  std::vector<int> C_n;
};

class gen_matmul_core_t : public body_generator_t<matmul_core_config_t> {
public:
  struct op_params_t {
    static constexpr int in_A = 0;
    static constexpr int in_B = 1;
    static constexpr int out_C = 0;
  };
  using parent = body_generator_t<matmul_core_config_t>;
  using parent::generate;

  gen_matmul_core_t(sc_op *owner, std::vector<logical_tensor_t> &&ins,
    std::vector<logical_tensor_t> &&outs);

  float get_gflop() const override;
  bool is_dynamic() const {
    return in_tensors_[0].is_dynamic() || in_tensors_[1].is_dynamic();
  }
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

  void get_and_check_blocks(sc_graph_t &graph, const std::vector<expr> &inputs,
    const matmul_core_config_t &config, expr &M_num_blocks, expr &K_num_blocks,
    int &M_block, int &K_block, int &N_block, expr &B_K_num_blocks,
    expr &N_num_blocks) const;

  void get_brgemm_and_fusion_params(sc_graph_t &graph,
    const std::vector<expr> &inputs, const std::vector<expr> &outputs,
    const int &M_block, const int &K_block, const int &N_block,
    std::vector<expr> &aidx, std::vector<expr> &bidx, std::vector<expr> &cidx,
    expr &LDA, expr &LDB, expr &LDC, expr &stride_a, expr &stride_b,
    std::vector<std::pair<expr, expr>> &fidx1,
    std::vector<std::pair<expr, expr>> &fidx2,
    std::vector<std::pair<expr, expr>> &fidx3) const;

  sc_data_type_t get_A_dtype() const { return in_tensors_[0].dtype_; }
  sc_data_type_t get_B_dtype() const { return in_tensors_[1].dtype_; }
  sc_data_type_t get_C_dtype() const { return out_tensors_[0].dtype_; }

  bool generate(context_ptr ctx, const matmul_core_config_t &config,
    fusion_anchor_mgr_t *fusion, const std::vector<expr> &inputs,
    const std::vector<expr> &outputs,
    std::vector<for_loop> &loops) const override;
  config_ptr get_default_config(context_ptr ctx) const override;

  void schedule_loops(context_ptr ctx, const matmul_core_config_t &config,
    stmt body, std::vector<for_loop> &fors) const override;
  void init_axis();

private:
  blocking_axis_t blocking_axis_;
};
} // namespace ops
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
