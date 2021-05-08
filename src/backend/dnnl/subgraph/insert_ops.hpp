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
#ifndef BACKEND_DNNL_SUBGRAPH_INSERT_OPS_HPP
#define BACKEND_DNNL_SUBGRAPH_INSERT_OPS_HPP

#include <memory>
#include <vector>

#include "interface/c_types_map.hpp"

#include "utils.hpp"

namespace dnnl {
namespace graph {
namespace impl {
namespace dnnl_impl {

void insert_reorder(std::vector<std::shared_ptr<op_t>> &subgraph);

void insert_permute_for_conv(std::vector<std::shared_ptr<op_t>> &subgraph);

void insert_to_group_for_conv(std::vector<std::shared_ptr<op_t>> &subgraph);

/// Insert a transpose op for matmul's input tensors
///
/// Only valid for below scenarios:
/// (1) src or weight's ndims is greater than 1
/// (2) either `transpose_a` or `transpose_b` is true
void insert_transpose_for_matmul(std::vector<std::shared_ptr<op_t>> &subgraph);

/// Insert a broadcast op for matmul's input tensors
///
/// There maybe three scenarios as below:
/// (1) one of inputs (src or weight) has only one dimension, DNNL require at
///     two dimensions, so need to insert dim 1 before/after the current dim
/// (2) The batch dimensions of src and weight are not matched, need to
///     broadcast
/// (3) bias dimensions are not matched with dst, need to broadcast
void insert_broadcast_for_matmul(std::vector<std::shared_ptr<op_t>> &subgraph);

} // namespace dnnl_impl
} // namespace impl
} // namespace graph
} // namespace dnnl

#endif
