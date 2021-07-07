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

#include "backend/dnnl/resource.hpp"

namespace dnnl {
namespace graph {
namespace impl {
namespace dnnl_impl {

thread_local std::unordered_map<resource_cache_t::key_t,
        resource_cache_t::cached_t>
        resource_cache_t::resource_map_;

bool resource_cache_t::has_resource(const key_t &key) const {
    return resource_map_.count(key);
}

size_t resource_cache_t::size() const {
    return resource_map_.size();
}

void resource_cache_t::clear() {
    resource_map_.clear();
}

} // namespace dnnl_impl
} // namespace impl
} // namespace graph
} // namespace dnnl
