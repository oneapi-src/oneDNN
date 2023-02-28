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
#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_OPS_BATCHNORM_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_OPS_BATCHNORM_HPP

#include <memory>
#include <vector>
#include <compiler/ir/graph/graph_op.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace ops {

class batchnorm_inference_op : public graph_op_t,
                               public op_traits::auto_copyable_t {
public:
    batchnorm_inference_op(const std::vector<graph_tensor_ptr> &ins,
            const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs);
    void get_graph_impl(std::shared_ptr<sc_graph_t> &graph) override;
    void query_format(context_ptr ctx,
            std::vector<std::vector<format_stride_pair>> &supported_ins,
            std::vector<std::vector<format_stride_pair>> &supported_outs)
            override;
};

/**
 * The batchnorm_forward_training_op
 * Inputs:
 *  - ins[0] - the data input
 *  - ins[1] - mean value
 *  - ins[2] - variance value
 *  - ins[3] - gamma
 *  - ins[4] - beta
 * Outputs:
 *  - outs[0] - bn result
 *  - outs[1] - running mean value
 *  - outs[2] - running variance value
 *  - outs[3] - batch mean value
 *  - outs[4] - batch variance value
 * Attrs:
 *  - epsilon: float - Default = 1e-5
 *  - momentum(Optional): float - Default = 1, which is used to compute
 * running_mean and running_variance
 *  - data_format(Optional): string - Default = "NXC", indicating whether data
 * is channel last.
 * */
class batchnorm_forward_training_op : public graph_op_t,
                                      public op_traits::auto_copyable_t {
public:
    batchnorm_forward_training_op(const std::vector<graph_tensor_ptr> &ins,
            const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs);
    void get_graph_impl(std::shared_ptr<sc_graph_t> &graph) override;
    void query_format(context_ptr ctx,
            std::vector<std::vector<format_stride_pair>> &supported_ins,
            std::vector<std::vector<format_stride_pair>> &supported_outs)
            override;
};

/**
 * The batchnorm_training_backprop_op
 * Inputs:
 *  - ins[0] - input (src of forward)
 *  - ins[1] - output_delta
 *  - ins[2] - mean
 *  - ins[3] - variance
 *  - ins[4] - gamma
 * Outputs:
 *  - outs[0] - input_delta
 *  - outs[1] - gamma_delta
 *  - outs[2] - beta_delta
 * Attrs:
 *  - epsilon: float - The value add to variance to increase numeric stability
 *  - data_format: string - Default: "NXC". Data format of input
 * */
class batchnorm_training_backprop_op_t : public graph_op_t,
                                         public op_traits::auto_copyable_t {
public:
    batchnorm_training_backprop_op_t(const std::vector<graph_tensor_ptr> &ins,
            const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs);
    void get_graph_impl(std::shared_ptr<sc_graph_t> &graph) override;
    void query_format(context_ptr ctx,
            std::vector<std::vector<format_stride_pair>> &supported_ins,
            std::vector<std::vector<format_stride_pair>> &supported_outs)
            override;
};
} // namespace ops
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
