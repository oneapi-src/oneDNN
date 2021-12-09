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
#include <string>
#include <unordered_map>

#include "compiler_graph.hpp"

namespace dnnl {
namespace graph {
namespace impl {
namespace compiler_impl {

static const std::unordered_map<op_kind_t, std::string, utils::enum_hash_t>
        compiler_backend_op {{op_kind::Add, "add"}, {op_kind::MatMul, "matmul"},
                {op_kind::Quantize, "quantize"},
                {op_kind::Dequantize, "dequantize"},
                {op_kind::StaticReshape, "tensor_view"},
                {op_kind::StaticTranspose, "transpose"},
                {op_kind::SoftMax, "softmax"}, {op_kind::Divide, "div"},
                {op_kind::Multiply, "mul"}};

bool compiler_graph_impl_t::is_supported_op(op_kind_t name) {
    return compiler_backend_op.find(name) != compiler_backend_op.end();
}

} // namespace compiler_impl
} // namespace impl
} // namespace graph
} // namespace dnnl
