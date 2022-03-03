/*******************************************************************************
 * Copyright 2022 Intel Corporation
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

#ifndef BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_GRAPH_GRAPH_MAP_HPP
#define BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_GRAPH_GRAPH_MAP_HPP

#include <unordered_map>

namespace sc {
// the map based on graph_tensor key
template <typename valT>
struct gt_map_t {
    std::unordered_map<graph_tensor *, valT> datamap_;
    valT &get(graph_tensor *);
    valT &get(const graph_tensor_ptr &);
    bool haskey(const graph_tensor_ptr &) const;
    void clear() { datamap_.clear(); }
    bool empty() const { return datamap_.empty(); }
    gt_map_t &operator=(const gt_map_t &other) = delete;
};
} // namespace sc
#endif
