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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_OPS_FUSIBLE_PADDING_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_OPS_FUSIBLE_PADDING_HPP

#include <memory>
#include <vector>
#include <compiler/ir/graph/fusible_op.hpp>
#include <compiler/ir/graph/graph_op.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

class padding_op_t : public movement_op_t, public op_traits::auto_copyable_t {
public:
    DECLARE_QUERY_AND_COMPUTE();

    padding_op_t(const std::vector<graph_tensor_ptr> &ins,
            const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs);

    padding_op_t(graph_tensor_ptr v, sc_dims &pads_begin, sc_dims &pads_end);

    sc_dims infer_out_dims(const sc_dims &input_dims, const sc_dims &pads_begin,
            const sc_dims &pads_end);

    void query_format(context_ptr ctx,
            std::vector<std::vector<format_stride_pair>> &supported_ins,
            std::vector<std::vector<format_stride_pair>> &supported_outs)
            override;

    size_t compute_workload(const std::vector<shape_dtype_pair> &,
            const std::vector<shape_dtype_pair> &) override;

    std::vector<int> get_real_padding_axis();

    std::vector<expr> get_padding_offsets_exprs();

    shape_rl_vec get_dynamic_shape_relations() const override;

    reflection::shared_general_object_t get_dynamic_runtime_info() override;

    void calculate_dynamic_shape_expression() override;

    stmt get_zero_out_stmt(
            const tensor &out, const slice_range_list &range_list);
};

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
