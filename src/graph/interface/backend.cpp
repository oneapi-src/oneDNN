/*******************************************************************************
* Copyright 2021-2024 Intel Corporation
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

#include "oneapi/dnnl/dnnl_graph.h"

#include "graph/interface/backend.hpp"

#include "graph/utils/utils.hpp"

namespace dnnl {
namespace impl {
namespace graph {

std::pair<size_t, size_t> backend_registry_t::decode_layout_id(
        size_t layout_id) {
    size_t backend_id = layout_id & (size_t)((1 << BACKEND_ID_LENGTH) - 1);
    size_t layout_idx = layout_id >> BACKEND_ID_LENGTH;
    return {layout_idx, backend_id};
}

size_t backend_registry_t::encode_layout_id(
        size_t layout_idx, size_t backend_id) {
    size_t layout_id = (layout_idx << BACKEND_ID_LENGTH)
            | (backend_id & (size_t)((1 << BACKEND_ID_LENGTH) - 1));
    return layout_id;
}

size_t backend_registry_t::extract_layout_id(size_t layout_id) {
    return layout_id >> BACKEND_ID_LENGTH;
}

size_t backend_registry_t::extract_backend_id(size_t layout_id) {
    return layout_id & (size_t)((1 << BACKEND_ID_LENGTH) - 1);
}

void backend_registry_t::invoke_backend_registration() {
    // Note: `std::call_once` should be kept in a single translation unit since
    // GCC 11.
    std::call_once(register_flag_, []() {
        register_dnnl_backend();
        register_fake_backend();
#ifdef DNNL_ENABLE_COMPILER_BACKEND
        register_compiler_backend();
#endif
    });
}

} // namespace graph
} // namespace impl
} // namespace dnnl
