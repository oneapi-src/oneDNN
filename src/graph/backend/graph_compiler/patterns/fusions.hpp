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
#ifndef BACKEND_GRAPH_COMPILER_PATTERNS_FUSIONS_HPP
#define BACKEND_GRAPH_COMPILER_PATTERNS_FUSIONS_HPP

#include <vector>

#include "graph/backend/graph_compiler/patterns/transformation_pattern.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace compiler_impl {
namespace pass {

template <graph::data_type_t DTYPE>
bool check_input_dtype(op_t *op) {
    for (size_t i = 0; i < op->num_inputs(); ++i) {
        const logical_tensor_t &iport
                = op->get_input_value(i)->get_logical_tensor();
        if (iport.data_type != DTYPE) return false;
    }

    return true;
}

bool check_reduce_attrs(op_t *op) {
    auto attrs = op->get_attributes();
    if (attrs.find(op_attr::axes) != attrs.end()
            && !attrs[op_attr::axes].get<std::vector<int64_t>>().empty()) {
        return true;
    }
    return false;
}

bool check_conv_attrs(op_t *op) {
    auto attrs = op->get_attributes();
    // dilations must be {1, 1, ...}
    if (attrs.find(graph::op_attr::dilations) != attrs.end()) {
        auto dilations
                = attrs[graph::op_attr::dilations].get<std::vector<int64_t>>();
        if (!std::all_of(dilations.begin(), dilations.end(),
                    [&](const int64_t &d) { return d == 1; })) {
            return false;
        }
    }
    // groups must be 1
    if (attrs.find(graph::op_attr::groups) != attrs.end()
            && attrs[graph::op_attr::groups].get<int64_t>() != 1) {
        return false;
    }
    // preferred to be a 2D conv
    auto strides = attrs[graph::op_attr::strides].get<std::vector<int64_t>>();
    if (strides.size() != 2) { return false; }
    // preferred to be symmetric padding
    // if no auto_pad set, needs to check pads_begin == pads_end
    if (attrs.find(op_attr::auto_pad) == attrs.end()) {
        auto pads_begin
                = attrs[graph::op_attr::pads_begin].get<std::vector<int64_t>>();
        auto pads_end
                = attrs[graph::op_attr::pads_end].get<std::vector<int64_t>>();
        if (pads_begin != pads_end) { return false; }
    }
    return true;
}

} // namespace pass
} // namespace compiler_impl
} // namespace graph
} // namespace impl
} // namespace dnnl
#endif
