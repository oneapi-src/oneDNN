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

#ifndef COMMON_CACHE_STATS_TYPES_HPP
#define COMMON_CACHE_STATS_TYPES_HPP

#include <cassert>
#include <string>

namespace dnnl {
namespace impl {

enum class cache_hit_t {
    cache_miss, //< complete cache miss, for all types of caches
    primitive_cache_hit, //< primitive cache hit, complete primitive was available
    kernel_cache_hit, //< kernel cache hit, primitive cache miss, but kernel was in cache
    persistent_cache_hit, //< cache_blob() persistent cache hit from disk/long term storage
    compiled_partition_cache_hit //< graph partition cache hit, already compiled
};

inline const char *cache_hit_string(const cache_hit_t cache_hit) {
    switch (cache_hit) {
        case cache_hit_t::cache_miss: return ":cache_miss";
        case cache_hit_t::primitive_cache_hit: return ":cache_hit";
        case cache_hit_t::kernel_cache_hit: return ":kernel_cache_hit";
        case cache_hit_t::persistent_cache_hit: return ":persistent_cache_hit";
        case cache_hit_t::compiled_partition_cache_hit:
            return ":compiled_partition_cache_hit";
        default:
            assert(!"no matching string representation for cache_hit_t");
            return nullptr;
    }
}

} // namespace impl
} // namespace dnnl

#endif
