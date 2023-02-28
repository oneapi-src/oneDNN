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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_TRANSFORM_CLOSURIZE_IMPL_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_TRANSFORM_CLOSURIZE_IMPL_HPP

#include <string>
#include <utility>
#include <vector>
#include <compiler/ir/ir_module.hpp>
#include <compiler/ir/visitor.hpp>
#include <unordered_map>
#include <unordered_set>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
/**
 * Base class for closurizer. Different targets should extend the
 * class to make target-specific parallel functions and parallel calls. Replaces
 * parallel for nodes with call_parallel nodes. Also moves the bodies of the
 * parallel for nodes to new closure functions. The closure functions have the
 * interface like `void closure_name(uint64_t i, T1 capture1, T2 capture2,
 * ...);`, where `i` is the loop variable, and captureX is the captured
 * variables that may be used in the body of the original for-loop.
 *
 * The target specific subclass should at least override `make_closure_func` and
 * `make_parallel_call`. See comments below.
 * */
class closurize_impl_t : public ir_visitor_t {
protected:
    using ir_visitor_t::dispatch;
    using ir_visitor_t::visit;
    std::unordered_map<expr_c, expr_c> captures_set_;
    // the variables defined within the current captures
    std::unordered_set<expr_c> defined_set_;
    // the variables defined in the module as globals
    std::unordered_set<expr_c> globals_set_;
    bool in_parallel_for = false;
    std::vector<expr> captures_;
    func_c cur_func_;
    ir_module_ptr modu_;
    int kernel_cnt = 0;

    closurize_impl_t(const std::vector<define> &globals, ir_module_ptr modu);

    func_c dispatch(func_c f) override;
    expr_c create_or_get_tensor_or_var(expr_c v);

    stmt_c visit(define_c v) override;
    expr_c visit(tensor_c v) override;
    expr_c visit(var_c v) override;
    stmt_c visit(assign_c v) override;

    // makes the closure function and its generic wrapper, returns the callee
    // function for the parallel_call
    virtual func_t make_closure_func(const std::string &name,
            std::vector<expr_c> &&params, stmt_c body,
            const std::vector<call_node::parallel_attr_t> &para_attr)
            = 0;
    // make the parallel_call node and the arguments for it. Can be wrapped in a
    // stmts node
    virtual stmt make_parallel_call(func_t target, std::vector<expr> &captures,
            std::vector<call_node::parallel_attr_t> &&para_attr)
            = 0;
    stmt_c visit(for_loop_c v) override;
};

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
