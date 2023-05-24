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
#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_GRAPH_DYNAMIC_LOWER_INFO_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_GRAPH_DYNAMIC_LOWER_INFO_HPP
#include <memory>
#include <compiler/dimensions.hpp>
#include <compiler/ir/sc_expr.hpp>
#include <unordered_map>
namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
struct dynamic_lower_info_t {
    static constexpr sc_dim init_placeholder = dimensions::dynamic_any - 1;
    // decreasing placeholder to represent dynamic dim in graph.
    // range is [-2, min of int64_t]
    sc_dim cur_dynamic_placeholder_ = init_placeholder;
    // dynamic sc_dim => expr(uint64_t) map, we maitain one map for one graph.
    std::unordered_map<sc_dim, expr> dim2expr_map_;
    // Is specific sub graph for parallel compilation.
    bool is_specific_ = false;

    const sc_dim &get_cur_dynamic_placeholder() const {
        return cur_dynamic_placeholder_;
    }
    const std::unordered_map<sc_dim, expr> &get_dim2expr_map() const {
        return dim2expr_map_;
    }
    sc_dim &get_cur_dynamic_placeholder() { return cur_dynamic_placeholder_; }
    std::unordered_map<sc_dim, expr> &get_dim2expr_map() {
        return dim2expr_map_;
    }
    void set_cur_dynamic_placeholder(const sc_dim &cur_dynamic_placeholder) {
        cur_dynamic_placeholder_ = cur_dynamic_placeholder;
    }
    void set_dim2expr_map(
            const std::unordered_map<sc_dim, expr> &dim2expr_map) {
        dim2expr_map_ = dim2expr_map;
    }
};
using dyn_lower_info_ptr = std::shared_ptr<dynamic_lower_info_t>;
class sc_graph_t;
bool is_dyn_specific_graph(sc_graph_t &);
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
