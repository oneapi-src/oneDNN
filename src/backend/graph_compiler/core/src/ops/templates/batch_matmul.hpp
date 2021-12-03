/*******************************************************************************
 * Copyright 2020-2021 Intel Corporation
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

#ifndef BACKEND_GRAPH_COMPILER_CORE_SRC_OPS_TEMPLATES_BATCH_MATMUL_HPP
#define BACKEND_GRAPH_COMPILER_CORE_SRC_OPS_TEMPLATES_BATCH_MATMUL_HPP

#include <memory>
#include <utility>
#include <vector>
#include "matmul2d.hpp"
#include <ops/body_generator.hpp>
namespace sc {
namespace ops {
using batch_matmul_config = struct matmul2d_config_t;

class gen_batch_matmul_t : public body_generator_t<batch_matmul_config> {
public:
  struct op_params_t {
    static constexpr int in_A = 0;
    static constexpr int in_B = 1;
    static constexpr int out_C = 0;
  };
  using parent = body_generator_t<batch_matmul_config>;
  using parent::generate;

  gen_batch_matmul_t(
    std::vector<logical_tensor_t> &&ins, std::vector<logical_tensor_t> &&outs);

  float get_gflop() const override;

  const sc_dims get_batch_dims() const { // NOLINT
    return {in_tensors_[0].get_plain_dims().begin(),
      in_tensors_[0].get_plain_dims().end() - 2};
  }

  const sc_dims get_mma_plain_dims() const { // NOLINT
    return {in_tensors_[0].get_plain_dims().begin() + get_batch_dims().size(),
      in_tensors_[0].get_plain_dims().end()};
  };

  const sc_dims get_mmb_plain_dims() const { // NOLINT
    return {in_tensors_[1].get_plain_dims().begin() + get_batch_dims().size(),
      in_tensors_[1].get_plain_dims().end()};
  };

  sc_data_type_t get_A_dtype() const { return in_tensors_[0].dtype_; }
  sc_data_type_t get_B_dtype() const { return in_tensors_[1].dtype_; }
  sc_data_type_t get_C_dtype() const { return out_tensors_[0].dtype_; }

  static void get_and_check_blocks(const logical_tensor_t &ta,
    const logical_tensor_t &tb, const batch_matmul_config &config,
    int &M_num_blocks, int &K_num_blocks, int &M_block, int &K_block,
    int &N_block, int &B_K_num_blocks, int &N_num_blocks);

  static void get_brgemm_and_fusion_params(const logical_tensor_t &ta,
    const logical_tensor_t &tb, const logical_tensor_t &tc,
    std::vector<expr> &aidx, std::vector<expr> &bidx, std::vector<expr> &cidx,
    int &stride_a, int &stride_b, std::vector<std::pair<expr, expr>> &fidx1,
    std::vector<std::pair<expr, expr>> &fidx2);

  bool is_valid_config(
    const context_ptr &ctx, const batch_matmul_config &config) const override;
  bool generate(context_ptr ctx, const batch_matmul_config &config,
    fusion_manager *fusion, const std::vector<expr> &inputs,
    const std::vector<expr> &outputs,
    std::vector<for_loop> &loops) const override;
  std::shared_ptr<void> get_default_config(context_ptr ctx) const override;

  void schedule_loops(context_ptr ctx, const batch_matmul_config &config,
    stmt body, std::vector<for_loop> &fors) const override;
  std::shared_ptr<gen_matmul2d_t> matmul2d_;
};
} // namespace ops
} // namespace sc

#endif
