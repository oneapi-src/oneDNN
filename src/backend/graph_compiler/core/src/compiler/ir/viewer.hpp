/*******************************************************************************
 * Copyright 2020-2022 Intel Corporation
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

#ifndef BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_VIEWER_HPP
#define BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_VIEWER_HPP

#include <vector>
#include "visitor.hpp"

namespace sc {
/**
 * The base class of read-only passes.
 * Override the overloaded view() function that you are
 * interested. In view(), call dispatch() on each of the sub-node
 * of the IR node, if you need to "view" the sub-nodes.
 * @note Dispatch rule for super-classes and sub-classes: If you
 * override the view() function of a sub-class (e.g. add), and you
 * call dispatch() on a sub-class object (e.g. a+b), the view()
 * function of the sub-class will be called. And the super-class
 * view() function (e.g. binary) will not be called. If you did not
 * override view() of the sub-class, the super-class view() function
 * will be used by default.
 * */
class ir_viewer_t : public ir_visitor_t {
public:
    /**
     * Visit an array of expr.
     */
    void dispatch_expr_arr(const std::vector<expr> &);

    /**
     * Override the view() functions below to visit the
     * IR that you are interested
     * */
    virtual void view(constant_c v);
    virtual void view(var_c v);
    virtual void view(cast_c v);

    virtual void view(binary_c v);
    virtual void view(add_c v);
    virtual void view(sub_c v);
    virtual void view(mul_c v);
    virtual void view(div_c v);
    virtual void view(mod_c v);

    virtual void view(cmp_c v);
    virtual void view(cmp_eq_c v);
    virtual void view(cmp_lt_c v);
    virtual void view(cmp_le_c v);
    virtual void view(cmp_gt_c v);
    virtual void view(cmp_ge_c v);
    virtual void view(cmp_ne_c v);

    virtual void view(logic_c v);
    virtual void view(logic_and_c v);
    virtual void view(logic_or_c v);

    virtual void view(logic_not_c v);
    virtual void view(select_c v);
    virtual void view(indexing_c v);
    virtual void view(call_c v);
    virtual void view(tensor_c v);
    virtual void view(tensorptr_c v);
    virtual void view(intrin_call_c v);
    virtual void view(func_addr_c v);
    virtual void view(ssa_phi_c v);

    virtual void view(assign_c v);
    virtual void view(stmts_c v);
    virtual void view(if_else_c v);
    virtual void view(evaluate_c v);
    virtual void view(returns_c v);
    virtual void view(define_c v);
    virtual void view(for_loop_c v);

private:
    expr_c visit(constant_c v) final;
    expr_c visit(var_c v) final;
    expr_c visit(cast_c v) final;

    expr_c visit(binary_c v) final;
    expr_c visit(add_c v) final;
    expr_c visit(sub_c v) final;
    expr_c visit(mul_c v) final;
    expr_c visit(div_c v) final;
    expr_c visit(mod_c v) final;

    expr_c visit(cmp_c v) final;
    expr_c visit(cmp_eq_c v) final;
    expr_c visit(cmp_lt_c v) final;
    expr_c visit(cmp_le_c v) final;
    expr_c visit(cmp_gt_c v) final;
    expr_c visit(cmp_ge_c v) final;
    expr_c visit(cmp_ne_c v) final;

    expr_c visit(logic_c v) final;
    expr_c visit(logic_and_c v) final;
    expr_c visit(logic_or_c v) final;

    expr_c visit(logic_not_c v) final;
    expr_c visit(select_c v) final;
    expr_c visit(indexing_c v) final;
    expr_c visit(call_c v) final;
    expr_c visit(tensor_c v) final;
    expr_c visit(tensorptr_c v) final;
    expr_c visit(intrin_call_c v) final;
    expr_c visit(func_addr_c v) final;
    expr_c visit(ssa_phi_c v) final;

    stmt_c visit(assign_c v) final;
    stmt_c visit(stmts_c v) final;
    stmt_c visit(if_else_c v) final;
    stmt_c visit(evaluate_c v) final;
    stmt_c visit(returns_c v) final;
    stmt_c visit(define_c v) final;
    stmt_c visit(for_loop_c v) final;
};

} // namespace sc

#endif
