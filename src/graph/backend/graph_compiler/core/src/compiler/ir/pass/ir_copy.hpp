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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_PASS_IR_COPY_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_PASS_IR_COPY_HPP

#include "../function_pass.hpp"
#include "../sc_function.hpp"
#include "../viewer.hpp"
#include <unordered_map>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

/**
 * Deeply copies the IR. Will NOT copy the function referenced by call_nodes
 *
 * @param create_var_tensor_ If true, creates new var and tensor. Otherwise, use
 * the old var and tensor
 * @param replace_map_ In and out parameter. Holds the mapping from old IR to
 * new IR. Before copying an IR node (var/tensor nodes only), the copier will
 * check if the old var/tensor node is in the map. If so, it will use the new IR
 * node in the map as the new copied IR node. Also, if a var/tensor node is
 * copied, the copier will put the old-new mapping in the replace_map_.
 * @param stmt_replace_map_, optional in and out parameter. Similar to
 * replace_map_ but holds the mapping from old stmt to new stmt.
 * */
class ir_copier_t : public function_pass_t {
protected:
    std::unordered_map<expr_c, expr> &replace_map_;
    std::unordered_map<stmt_c, stmt> *stmt_replace_map_;
    bool create_var_tensor_;

public:
    ir_copier_t(std::unordered_map<expr_c, expr> &replace_map,
            bool create_var_tensor = true)
        : replace_map_(replace_map)
        , stmt_replace_map_(nullptr)
        , create_var_tensor_(create_var_tensor) {}
    ir_copier_t(std::unordered_map<expr_c, expr> &replace_map,
            std::unordered_map<stmt_c, stmt> *stmt_replace_map,
            bool create_var_tensor = true)
        : replace_map_(replace_map)
        , stmt_replace_map_(stmt_replace_map)
        , create_var_tensor_(create_var_tensor) {}
    virtual expr_c operator()(const expr_c &v);
    virtual stmt_c operator()(const stmt_c &v);
    func_c operator()(func_c v) override;
};

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
