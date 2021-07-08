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
#ifndef BACKEND_DNNL_SUBGRAPH_UTILS_HPP
#define BACKEND_DNNL_SUBGRAPH_UTILS_HPP

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "interface/value.hpp"

#include "backend/dnnl/transformation_pass.hpp"
#include "backend/dnnl/utils.hpp"

#include "dnnl.hpp"

namespace dnnl {
namespace graph {
namespace impl {
namespace dnnl_impl {

void insert_op_before(std::shared_ptr<impl::op_t> &inserted_op,
        std::shared_ptr<impl::op_t> &base_op, size_t offset);

void insert_op_after(std::shared_ptr<impl::op_t> &inserted_op,
        std::shared_ptr<impl::op_t> &base_op, size_t offset);

void fuse_op_to_successor(
        op_t *op, std::vector<std::shared_ptr<op_t>> &subgraph);

void fuse_op_to_predecessor(op_t *op,
        std::vector<std::shared_ptr<op_t>> &subgraph, size_t in_offset = 0);

void set_given_inputs_outputs(std::vector<std::shared_ptr<op_t>> &subgraph,
        const std::vector<impl::logical_tensor_t> &inputs,
        const std::vector<impl::logical_tensor_t> &outputs);

void set_all_layout_to_any(std::vector<std::shared_ptr<op_t>> &subgraph);

} // namespace dnnl_impl
} // namespace impl
} // namespace graph
} // namespace dnnl

#endif
