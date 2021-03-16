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

#include "backend.hpp"

namespace dnnl {
namespace graph {
namespace impl {
backend *backend::register_backend(const backend *abackend) {
    return backend_registry::get_singleton().register_backend(abackend);
}

std::pair<size_t, size_t> backend_registry::decode_layout_id(size_t layout_id) {
    size_t backend_id = layout_id & (size_t)((1 << BACKEND_ID_LENGTH) - 1);
    size_t layout_idx = layout_id >> BACKEND_ID_LENGTH;
    return {layout_idx, backend_id};
}

size_t backend_registry::encode_layout_id(
        size_t layout_idx, size_t backend_id) {
    size_t layout_id = (layout_idx << BACKEND_ID_LENGTH)
            | (backend_id & (size_t)((1 << BACKEND_ID_LENGTH) - 1));
    return layout_id;
}

size_t backend_registry::extract_layout_id(size_t layout_id) {
    return layout_id >> BACKEND_ID_LENGTH;
}

size_t backend_registry::extract_backend_id(size_t layout_id) {
    return layout_id & (size_t)((1 << BACKEND_ID_LENGTH) - 1);
}

} // namespace impl
} // namespace graph
} // namespace dnnl
