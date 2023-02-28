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
#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_UTIL_OPTIONAL_FIND_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_UTIL_OPTIONAL_FIND_HPP
#include "optional.hpp"
#include <type_traits>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace utils {

namespace impl {
template <typename MapT>
struct extract_map_types {};

template <typename MapT>
struct extract_map_types<const MapT &> {
    using ptr_value_t = const typename MapT::mapped_type *;
    using ptr_pair_t = const typename MapT::value_type *;
};

template <typename MapT>
struct extract_map_types<MapT &> {
    using ptr_value_t = typename MapT::mapped_type *;
    using ptr_pair_t = typename MapT::value_type *;
};

} // namespace impl

template <typename MapT>
optional<typename impl::extract_map_types<MapT>::ptr_value_t> find_map_value(
        MapT &&m, const typename std::decay<MapT>::type::key_type &key) {
    auto itr = m.find(key);
    if (itr != m.end()) { return &(itr->second); }
    return none_opt {};
}

template <typename MapT>
optional<typename impl::extract_map_types<MapT>::ptr_pair_t> find_map_pair(
        MapT &&m, const typename std::decay<MapT>::type::key_type &key) {
    auto itr = m.find(key);
    if (itr != m.end()) { return &(*itr); }
    return none_opt {};
}

} // namespace utils
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
