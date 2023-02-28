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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_DYNAMIC_DISPATCH_DYNAMIC_TENSOR_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_DYNAMIC_DISPATCH_DYNAMIC_TENSOR_HPP
#include <stdint.h>
#include "../data_type.hpp"
#include <compiler/dimensions.hpp>
namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace runtime {
/** This tensor is prepared for dynamic shape. If tensor node has uncertain
 * shapes during compile-time, then tensor node will be transformed to `void *`
 * which is exactly `dynamic_tensor_t *` instead of raw data pointer by
 * dyn_tsr_transform pass.
 * @param data, raw data pointer.
 * @param dims, pointer to olain shape dimensions of tensor
 * @param ndims, number of shape dimensions
 * @param dtype, datatype of the tensor
 * @param dyn_mask, pointer to a boolean list who match with pdims and indicate
 * which dimension is dynamic.
 * */
struct dynamic_tensor_t {
    dynamic_tensor_t() = default;
    dynamic_tensor_t(void *data, sc_dim *dims, int ndims, uint32_t dtype,
            uint8_t dyn_mask)
        : data_(data)
        , dims_(dims)
        , ndims_(ndims)
        , dtype_(dtype)
        , dyn_mask_(dyn_mask) {}
    // the raw opaque data pointer.
    void *data_;
    sc_dim *dims_;
    // number of dimensions;
    int ndims_;
    uint32_t dtype_;
    uint8_t dyn_mask_;
};
} // namespace runtime
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
#endif
