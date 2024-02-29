/*******************************************************************************
 * Copyright 2023-2024 Intel Corporation
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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_OPS_FUSIBLE_BINARY_BACKWARD_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_OPS_FUSIBLE_BINARY_BACKWARD_HPP

#include <utility>
#include <vector>
#include <compiler/ir/graph/fusible_op.hpp>
#include <runtime/microkernel/cpu/brgemm_alg_kind.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

enum class binary_backward_operator { PRELU_BWD };

class binary_backward_op_impl_t : public binary_backward_op_t {
public:
    DECLARE_QUERY_AND_COMPUTE();
    std::vector<std::pair<int, std::vector<tensor_inplace_info_t>>>
    get_inplace_map() override;
    binary_backward_op_impl_t(const std::vector<graph_tensor_ptr> &ins,
            const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs,
            const binary_backward_operator &backward_opt);

    void set_backward_operator(binary_backward_operator backward_op) {
        backward_op_type = backward_op;
    }
    uint32_t get_lanes() const { return vx_info_.lanes; }

    void query_format(context_ptr ctx,
            std::vector<std::vector<format_stride_pair>> &supported_ins,
            std::vector<std::vector<format_stride_pair>> &supported_outs)
            override;
    shape_rl_vec get_dynamic_shape_relations() const override;
    vectorized_info_t &get_vx_info() { return vx_info_; }

    void infer_binding_axis(binding_axis_map &bdax_map) override;
    void pre_infer_binding_axis(binding_axis_map &bdax_map) override;

private:
    vectorized_info_t vx_info_;
    binary_backward_operator backward_op_type;
};

class prelu_bwd_op_t : public binary_backward_op_impl_t {
public:
    // ins: ins[0] is src, ins[1] is diff_dst
    prelu_bwd_op_t(const std::vector<graph_tensor_ptr> &ins,
            const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs)
        : binary_backward_op_impl_t(
                ins, outs, attrs, binary_backward_operator::PRELU_BWD) {
        attrs_ = attrs;
        op_name_ = "prelu_bwd";
    }
};

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
#endif
