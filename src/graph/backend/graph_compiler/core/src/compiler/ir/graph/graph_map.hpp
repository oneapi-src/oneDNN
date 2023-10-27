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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_GRAPH_GRAPH_MAP_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_GRAPH_GRAPH_MAP_HPP

#include <vector>
#include "graph.hpp"
#include <unordered_map>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

template <typename T>
struct is_vector {
    static constexpr bool value = false;
};

template <typename T, typename Alloc>
struct is_vector<std::vector<T, Alloc>> {
    static constexpr bool value = true;
};

// the map based on graph_tensor key
template <typename valT>
struct gt_map_t {
    std::unordered_map<graph_tensor *, valT> datamap_;

    gt_map_t() = default;
    template <typename valT2>
    gt_map_t(const std::unordered_map<graph_tensor *, valT2> &datamap)
        : datamap_(datamap) {}
    // get the reference of value for the corresponding key, can be either read
    // or write
    valT &get(graph_tensor *);
    valT &get(const graph_tensor_ptr &);
    // return true if has key
    bool haskey(const graph_tensor_ptr &) const;
    bool haskey(graph_tensor *) const;

    // specialization for vector tpyes of valT, which owns `empty` method
    template <typename T = void,
            typename dummy = typename std::enable_if<
                    std::is_same<T, void>::value && is_vector<valT>::value,
                    bool>::type>
    bool hasvalue(graph_tensor *v) {
        return haskey(v) && !get(v).empty();
    }

    template <typename T = void,
            typename dummy = typename std::enable_if<
                    std::is_same<T, void>::value && is_vector<valT>::value,
                    bool>::type>
    bool hasvalue(const graph_tensor_ptr &v) {
        return hasvalue(v.get());
    }

    // clear datamap
    void clear() { datamap_.clear(); }
    // return true if empty
    bool empty() const { return datamap_.empty(); }
    // erase by key
    void erase(const graph_tensor_ptr k) { datamap_.erase(k.get()); }
    gt_map_t &operator=(const gt_map_t &other) = delete;
};
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
#endif
