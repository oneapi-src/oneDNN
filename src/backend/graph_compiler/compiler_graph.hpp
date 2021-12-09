/*******************************************************************************
 * Copyright 2021 Intel Corporation
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
#ifndef BACKEND_GRAPH_COMPILER_COMPILER_GRAPH_HPP
#define BACKEND_GRAPH_COMPILER_COMPILER_GRAPH_HPP

#include "interface/graph.hpp"
#include "interface/op.hpp"
#include "utils.hpp"

#include "compiler/ir/graph/fusible_op.hpp"
#include "compiler/ir/graph/graph.hpp"

namespace dnnl {
namespace graph {
namespace impl {
namespace compiler_impl {

class compiler_graph_impl_t : public sc::sc_graph_t {
public:
    compiler_graph_impl_t() = default;

    // return whether an op is supported by compiler backend or not
    static bool is_supported_op(op_kind_t name);
};

} // namespace compiler_impl
} // namespace impl
} // namespace graph
} // namespace dnnl

#endif
