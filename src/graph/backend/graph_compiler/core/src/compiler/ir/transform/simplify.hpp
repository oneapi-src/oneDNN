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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_TRANSFORM_SIMPLIFY_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_TRANSFORM_SIMPLIFY_HPP

#include "../function_pass.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

/**
 * Remove empty stmts nodes in parent stmts nodes. Simplify for nodes if
 * boundaris are constants/loop body is empty. Simplify if-else nodes if
 * condition is constant.
 * @param skip_rename skip renaming the variables if the it has conflicts with
 * parent scopes. Enabling this feature will slow down this pass a lot.
 * @param skip_if_loop skip simplifying if-else and for-loop node
 * */
class ir_simplifier_t : public function_pass_t {
public:
    bool skip_rename_;
    bool skip_if_loop_;
    ir_simplifier_t(bool skip_rename, bool skip_if_loop = false)
        : skip_rename_(skip_rename), skip_if_loop_(skip_if_loop) {}
    func_c operator()(func_c f) override;
    stmt_c operator()(stmt_c f) const;
    SC_DECL_PASS_INFO_FUNC();
};

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
