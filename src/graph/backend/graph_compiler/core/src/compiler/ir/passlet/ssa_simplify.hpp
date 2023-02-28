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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_PASSLET_SSA_SIMPLIFY_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_PASSLET_SSA_SIMPLIFY_HPP

#include <compiler/ir/sc_expr.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace passlet {

// the passlet for SSA copy propagation and constant propagation
// for each var X, will recursively apply the following rules:
// 0) if the parent expr is PHI, return X (don't optimize in PHI). Otherwise:
// 1) if X's definition is X=constant, return the constant
// 2) if X's definition is X=another_var and another_var is not a global var,
// 3) if X's definition is X=phi(y), return y
// return another_var
// To use, call enter_phi()/leave_phi() in visit(ssa_phi v) before and after
// dispatching down ssa_phi. And call visit() for var node to get simplified
// result
struct ssa_simplify_t {
    bool is_in_phi_ = false;
    void enter_phi() { is_in_phi_ = true; }
    void leave_phi() { is_in_phi_ = false; }

    expr_c visit(const ssa_phi_c &v);
    expr_c visit(const var_c &v);
};

} // namespace passlet
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
#endif
