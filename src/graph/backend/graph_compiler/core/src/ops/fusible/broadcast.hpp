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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_OPS_FUSIBLE_BROADCAST_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_OPS_FUSIBLE_BROADCAST_HPP

#include <vector>

#include <compiler/ir/graph/fusible_op.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

/**
 * The broadcast op
 * Inputs:
 *  - The input to be broadcasted
 * Outputs:
 *  - The resulting tensor
 * Attrs:
 *  - output_shape: sc_dims - Specifies the shape of the output
 *  - bc_axis: std::vector<int> (optional). If it is not set, the calculation
 *    will strictly follow auto-broadcast semantics. If set, the broadcast axis
 *    will follow the specified "bc_axis".
 * */
class broadcast_op_t : public movement_op_t, public op_traits::auto_copyable_t {
public:
    DECLARE_QUERY_AND_COMPUTE();

    broadcast_op_t(const std::vector<graph_tensor_ptr> &ins,
            const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs);
    broadcast_op_t(graph_tensor_ptr v, std::vector<int> &output_shape,
            std::vector<int> &bc_axis);

    void query_format(context_ptr ctx,
            std::vector<std::vector<format_stride_pair>> &supported_ins,
            std::vector<std::vector<format_stride_pair>> &supported_outs)
            override;
    size_t compute_workload(const std::vector<shape_dtype_pair> &,
            const std::vector<shape_dtype_pair> &) override;

    shape_rl_vec get_dynamic_shape_relations() const override;

    // get broadcast axis on blocking dims (inferred from the bc_axis_ on plain
    // dims)
    std::vector<int> get_bc_axis() const;

    std::vector<int> get_plain_bc_axis() const { return plain_bc_axis_; }

private:
    sc_dims output_shape_;
    std::vector<int> plain_bc_axis_;
    vectorized_info_t vx_info_;
};

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
#endif
