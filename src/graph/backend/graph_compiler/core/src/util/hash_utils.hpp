/*******************************************************************************
 * Copyright 2020-2023 Intel Corporation
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
#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_UTIL_HASH_UTILS_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_UTIL_HASH_UTILS_HPP

#include <functional>
#include <vector>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
// The following code is derived from Boost C++ library
// Copyright 2005-2014 Daniel James.
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
// https://github.com/boostorg/container_hash/blob/master/include/boost/container_hash/hash.hpp
template <typename T>
inline void hash_combine(std::size_t &seed, T const &v) {
    seed ^= std::hash<T>()(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

// a stable hash that does not depend on stl implementation
inline void hash_combine_stable(std::size_t &seed, std::size_t v) {
    seed ^= v + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

namespace std {
template <typename T>
struct hash<std::vector<T>> {
    std::size_t operator()(const std::vector<T> &v) const {
        size_t seed = 0;
        for (size_t i = 0; i < v.size(); i++) {
            dnnl::impl::graph::gc::hash_combine(seed, v[i]);
        }
        return seed;
    }
};
} // namespace std

#endif
