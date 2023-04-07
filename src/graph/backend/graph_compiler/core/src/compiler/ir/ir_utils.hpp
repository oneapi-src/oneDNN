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
#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_IR_UTILS_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_IR_UTILS_HPP

#include <functional>
#include <vector>
#include "sc_expr.hpp"
#include <util/array_ref.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

// finds the direct dependency of an SSA expr. Will return the exprs to the
// callback.
void get_direct_dependency_of_expr(
        const expr &v, const std::function<void(array_ref<expr>)> &callback);
/**
 * calculate dense stride from tensor dims.
 * @param v the input dims of tensor.
 */
std::vector<expr> dims_to_dense_stride(const std::vector<expr> &v);

/**
 * Gets the base tensor of the argument. It will look into 1) tensorptr 2) cast
 * nodes. If the expr has not base tensor, returns null.
 */
tensor_c get_base_tensor_of(const expr &p);
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
