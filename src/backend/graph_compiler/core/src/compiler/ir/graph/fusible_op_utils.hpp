/*******************************************************************************
 * Copyright 2020-2021 Intel Corporation
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

#ifndef BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_GRAPH_FUSIBLE_OP_UTILS_HPP
#define BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_GRAPH_FUSIBLE_OP_UTILS_HPP

#include <string>
#include <vector>
#include "fusible_op.hpp"

namespace sc {

using fusion_compute_func_t = std::function<stmt(
        const std::vector<expr> &, std::vector<expr::lvalue_proxy_t> &)>;

void compute_vectorized_op(const std::vector<const tensor_slice *> &src,
        const tensor_slice &dst, sc_op_info_t &info,
        const vectorized_info_t &vx_info, fusion_compute_func_t compute_lanes,
        fusion_compute_func_t compute_scalar);

void compute_block_elemwise(const std::vector<const tensor_slice *> &src,
        const tensor_slice &dst, sc_op_info_t &info,
        fusion_compute_func_t compute);

std::string fusion_create_var_idx();
} // namespace sc

#endif
