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

#ifndef BACKEND_GRAPH_COMPILER_CORE_SRC_OPS_TEMPLATES_MATMUL2D_HPP
#define BACKEND_GRAPH_COMPILER_CORE_SRC_OPS_TEMPLATES_MATMUL2D_HPP

#include <memory>
#include <vector>
#include <ops/body_generator.hpp>
namespace sc {
namespace ops {
struct matmul2d_config_t {
  int M_block;
  int N_block;
  int K_block;
  int num_tile_k;
};
class gen_matmul2d_t : public body_generator_t<matmul2d_config_t> {
public:
  struct op_params_t {
    static constexpr int in_A = 0;
    static constexpr int in_B = 1;
    static constexpr int out_C = 0;
  };
  using parent = body_generator_t<matmul2d_config_t>;
  using parent::generate;

  gen_matmul2d_t(
    std::vector<logical_tensor_t> &&ins, std::vector<logical_tensor_t> &&outs);

  float get_gflop() const override;

  const sc_dims &get_A_blocking_dims() const {
    return in_tensors_[0].get_blocking_dims();
  }
  const sc_dims &get_B_blocking_dims() const {
    return in_tensors_[1].get_blocking_dims();
  }
  const sc_dims &get_C_blocking_dims() const {
    return out_tensors_[0].get_blocking_dims();
  }
  const sc_dims &get_A_plain_dims() const {
    return in_tensors_[0].get_plain_dims();
  }
  const sc_dims &get_B_plain_dims() const {
    return in_tensors_[1].get_plain_dims();
  }
  const sc_dims &get_C_plain_dims() const {
    return out_tensors_[0].get_plain_dims();
  }
  sc_data_type_t get_A_dtype() const { return in_tensors_[0].dtype_; }
  sc_data_type_t get_B_dtype() const { return in_tensors_[1].dtype_; }
  sc_data_type_t get_C_dtype() const { return out_tensors_[0].dtype_; }

  bool is_valid_config(
    const context_ptr &ctx, const matmul2d_config_t &config) const override;
  bool generate(context_ptr ctx, const matmul2d_config_t &config,
    fusion_manager *fusion, const std::vector<expr> &inputs,
    const std::vector<expr> &outputs,
    std::vector<for_loop> &loops) const override;
  std::shared_ptr<void> get_default_config(context_ptr ctx) const override;

  void schedule_loops(context_ptr ctx, const matmul2d_config_t &config,
    stmt body, std::vector<for_loop> &fors) const override;
};
} // namespace ops
} // namespace sc

#endif
