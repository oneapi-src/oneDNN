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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_TRANSFORM_TENSOR_INPLACE_INFO_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_TRANSFORM_TENSOR_INPLACE_INFO_HPP
#include <memory>
namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace alias_info {
struct tensor_alias_identity_t;
}
enum class inplace_kind {
    ZERO_OFFSET, // this requires that the tensor share the same base
    // pointer of the replaced tensor
    FREE, // the tensor can freely choose any offset on this tensor
};

struct tensor_inplace_info_t {
    int used_arg_idx_;
    inplace_kind kind_;
};

struct temp_tensor_inplace_info_t {
    std::shared_ptr<alias_info::tensor_alias_identity_t> to_reuse_;
    inplace_kind kind_;
};

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
