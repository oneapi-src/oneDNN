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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_OPS_FUSIBLE_TERNARY_ELEMWISE_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_OPS_FUSIBLE_TERNARY_ELEMWISE_HPP

#include <utility>
#include <vector>
#include <compiler/ir/graph/fusible_op.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

class select_op_t : public fusible_op_t,
                    public op_traits::auto_copyable_t,
                    public op_traits::may_inplace_t,
                    public op_traits::may_broadcast_t {
public:
    DECLARE_QUERY_AND_COMPUTE();

    std::vector<std::pair<int, std::vector<tensor_inplace_info_t>>>
    get_inplace_map() override;

    select_op_t(
            graph_tensor_ptr cond, graph_tensor_ptr then, graph_tensor_ptr els);
    select_op_t(const std::vector<graph_tensor_ptr> &ins,
            const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs);

    uint32_t get_lanes() const { return vx_info_.lanes; }

    void query_format(context_ptr ctx,
            std::vector<std::vector<format_stride_pair>> &supported_ins,
            std::vector<std::vector<format_stride_pair>> &supported_outs)
            override;

    std::vector<int> get_bc_axis(const int axis1, const int axis2) const;

    std::vector<int> get_non_broadcast_input_index(
            bool assert_non_empty) const override;

    int get_ref_input_index(bool assert_determined) const override;

    vectorized_info_t &get_vx_info() { return vx_info_; }

    shape_rl_vec get_dynamic_shape_relations() const override;

    void infer_binding_axis(bound_axis_map &bdax_map) override;
    void pre_binding_axis(bound_axis_map &bdax_map) override;

private:
    vectorized_info_t vx_info_;
};

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
#endif
