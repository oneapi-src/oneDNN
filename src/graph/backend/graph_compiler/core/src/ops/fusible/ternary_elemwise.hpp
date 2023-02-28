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

class select_op_t : public fusible_op_t, public op_traits::auto_copyable_t {
public:
    DECLARE_QUERY_AND_COMPUTE();

    select_op_t(graph_tensor_ptr cond, graph_tensor_ptr then,
            graph_tensor_ptr els, int inplace = 1);
    select_op_t(const std::vector<graph_tensor_ptr> &ins,
            const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs);

    uint32_t get_lanes() const { return vx_info_.lanes; }

    void query_format(context_ptr ctx,
            std::vector<std::vector<format_stride_pair>> &supported_ins,
            std::vector<std::vector<format_stride_pair>> &supported_outs)
            override;
    // get real broadcast axis, generaly, you should set bc_axis on plain format
    // semantics if necessary.
    std::vector<int> get_bc_axis(const int l, const int r) const;
    std::vector<int> infer_broadcast_axis(const int l, const int r) const;
    int get_broadcast_input(const int l, const int r) const;

    vectorized_info_t &get_vx_info() { return vx_info_; }

    int get_max_input() const;

    const std::vector<int> &get_plain_bc_axis() const { return plain_bc_axis_; }
    shape_rl_vec get_dynamic_shape_relations() const override;

    void infer_binding_axis(bound_axis_map &bdax_map) override;
    void pre_binding_axis(bound_axis_map &bdax_map) override;

protected:
    std::vector<int> plain_bc_axis_;

private:
    int inplace_;
    vectorized_info_t vx_info_;
};

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
#endif
