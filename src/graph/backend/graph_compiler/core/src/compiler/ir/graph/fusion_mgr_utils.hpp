/*******************************************************************************
 * Copyright 2020-2023 Intel Corporation
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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_GRAPH_FUSION_MGR_UTILS_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_GRAPH_FUSION_MGR_UTILS_HPP
#include <functional>
#include <vector>
#include "fusion_mgr.hpp"
namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
// generates element-wise op in loops
void compute_vectorized_op(const std::vector<const tensor_slice *> &src,
        const tensor_slice &dst, sc_op_info_t &info,
        const vectorized_info_t &vx_info,
        std::function<stmt(
                const std::vector<expr> &, std::vector<expr::lvalue_proxy_t> &)>
                compute_lanes,
        std::function<stmt(
                const std::vector<expr> &, std::vector<expr::lvalue_proxy_t> &)>
                compute_scalar);

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
