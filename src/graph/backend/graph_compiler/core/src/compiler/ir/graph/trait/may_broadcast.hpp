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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_GRAPH_TRAIT_MAY_BROADCAST_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_GRAPH_TRAIT_MAY_BROADCAST_HPP

#include <algorithm>
#include <vector>
#include <compiler/ir/graph/traits.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace op_traits {
struct may_broadcast_t : public virtual op_base_trait_t {
    static constexpr int NOT_DETERMINED = -2;
    /**
     * Returns the index of all the input logical tensors that do
     * not need to be broadcasted (whose shape same as the output)
     * */
    virtual std::vector<int> get_non_broadcast_input_index(
            bool assert_non_empty) const = 0;

    /**
     * Returns the input index serving as the layout reference, which tends to
     * be the index of the largest input
     * */
    virtual int get_ref_input_index(bool assert_determined) const = 0;

    const std::vector<std::vector<int>> &get_plain_bc_axis() const {
        return plain_bc_axis_;
    }

    /**
     * Returns a vector of input_shape's axis whose shape matches with the
     * corresponding dimension of the output (so no need to be broadcasted)
     * */
    static std::vector<int> get_auto_broadcast_bc_axis(
            const sc_dims &input_shape, const sc_dims &output_shape);

    /**
     * Returns whether shape1 and shape2 are equal (means shape1 to shape 2 does
     * not involve broadcast).
     * Assuming that shape1 is broadcastable to shape2.
     * */
    static bool broadcastable_shape_equal(
            const sc_dims &shape1, const sc_dims &shape2);

    static sc_dims infer_auto_broadcast_output_shape(
            const sc_dims &lhs, const sc_dims &rhs);

    static sc_data_format_t infer_broadcast_format(
            const logical_tensor_t &target_lt, const logical_tensor_t &bc_lt);

protected:
    std::vector<std::vector<int>> plain_bc_axis_;
};

} // namespace op_traits
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
