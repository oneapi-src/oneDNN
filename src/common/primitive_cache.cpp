/*******************************************************************************
* Copyright 2019 Intel Corporation
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

#include "primitive_cache.hpp"

namespace dnnl {
namespace impl {

lru_primitive_cache_t &primitive_cache() {
    constexpr int capacity = 200;
    static lru_primitive_cache_t cache(capacity);
    return cache;
}

} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
