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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_GRAPH_TRAIT_MAY_INPLACE_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_GRAPH_TRAIT_MAY_INPLACE_HPP

#include <memory>
#include <utility>
#include <vector>
#include <compiler/ir/graph/traits.hpp>
#include <compiler/ir/transform/tensor_inplace_info.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace op_traits {
struct may_inplace_t : public virtual op_base_trait_t {
    /**
     * @brief Get the inplace mapping
     *
     * @return the list of <output_tensor_index, list<input_tensor_index>>.
     * output_tensor_index is the index of an output tensor of this op staring
     * from 0. input_tensor_index is the index of an input tensor of this op
     * staring from 0. returning empty vector means that this op cannot inplace
     * reuse any input
     */
    virtual std::vector<std::pair<int, std::vector<tensor_inplace_info_t>>>
    get_inplace_map() = 0;
};

} // namespace op_traits
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
