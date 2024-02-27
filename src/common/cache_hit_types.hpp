/*******************************************************************************
* Copyright 2024 Intel Corporation
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

#ifndef COMMON_CACHE_HIT_TYPES_HPP
#define COMMON_CACHE_HIT_TYPES_HPP

#include <cassert>
#include <string>

namespace dnnl {
namespace impl {

enum class cache_state_t {
    miss, //< complete cache miss, for all types of caches
    primitive_hit, //< primitive cache hit, complete primitive was available
    kernel_hit, //< kernel cache hit, primitive cache miss, but kernel was in cache
    persistent_hit, //< cache_blob() persistent cache hit from disk/long term storage
    nested_primitive_hit, //< signifies nested kernel required creation but hit cache
    compiled_partition_hit //< graph partition cache hit, already compiled
};

inline const char *cache_state2str(const cache_state_t cache_hit) {
    switch (cache_hit) {
        case cache_state_t::miss: return ":cache_miss";
        case cache_state_t::primitive_hit: return ":cache_hit";
        case cache_state_t::kernel_hit: return ":kernel_cache_hit";
        case cache_state_t::persistent_hit: return ":persistent_cache_hit";
        case cache_state_t::nested_primitive_hit:
            return ":nested_primitive_cache_hit";
        case cache_state_t::compiled_partition_hit:
            return ":compiled_partition_cache_hit";
    }
    assert(!"no matching string representation for cache_state_t");
    return "";
}

} // namespace impl
} // namespace dnnl

#endif
