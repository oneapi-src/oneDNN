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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_PASS_IR_COPY_INTERNAL_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_PASS_IR_COPY_INTERNAL_HPP

#include <algorithm>
#include "../sc_function.hpp"
#include "../viewer.hpp"
#include <unordered_map>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

/**
 * This header exposes the implementation of ir_copier_t for code reuse.
 * Use ir_copier_t for simple IR copying
 * */

/**
 * Deeply copies the IR. Will NOT copy the function referenced by call_nodes
 *
 * @param create_var_tensor If true, creates new var and tensor. Otherwise, use
 * the old var and tensor
 * @param replace_map_ In and out parameter. Holds the mapping from old IR to
 * new IR. Before copying an IR node (var/tensor nodes only), the copier will
 * check if the old var/tensor node is in the map. If so, it will use the new IR
 * node in the map as the new copied IR node. Also, if a var/tensor node is
 * copied, the copier will put the old-new mapping in the replace_map_.
 * @param stmt_replace_map_, optional in and out parameter. Similar to
 * replace_map_ but holds the mapping from old stmt to new stmt.
 * */
class ir_copier_impl_t : public ir_viewer_t {
protected:
    std::unordered_map<expr_c, expr> &replace_map_;
    std::unordered_map<stmt_c, stmt> *stmt_replace_map_;

    // the returned IR pointers after calling dispatch(...).
    // Should not read them directly
    // We are using ir_viewer_t, so we cannot return the pointer via the return
    // value. We store the return values here. These values are valid after
    // calling dispatch(...). They should be consumed immediately after
    // dispatch(...) returns.
    // In this ir_viewer_t, you should use copy(..) instead of directly calling
    // dispatch(...). copy(..) will call "dispatch" and return-move the
    // returned_*_ to its return value
    expr returned_expr_;
    stmt returned_stmt_;
    func_t returned_func_;

    bool create_var_tensor_;

    /**
     * Finds a var/tensor in the replace map. If there is a match, sets
     * returned_expr_ to the mapped expression.
     * @param v the `old` expression
     * @return true if `v` is in the replace map
     * */
    bool find_and_return(const expr_c &v);
    /**
     * Finds a stmt in the replace map, similar to above method
     * */
    bool find_and_return(const stmt_c &s);
    void update_shrink_info(const expr_c &v, const expr &ret);

public:
    ir_copier_impl_t(std::unordered_map<expr_c, expr> &replace_map,
            bool create_var_tensor = true);
    ir_copier_impl_t(std::unordered_map<expr_c, expr> &replace_map,
            std::unordered_map<stmt_c, stmt> *stmt_replace_map,
            bool create_var_tensor = true);

    using ir_viewer_t::dispatch;
    using ir_viewer_t::view;

    /**
     * Copies the IR by calling dispatch(...) on it. Move-returns
     * `returned_expr_`
     * @return the copied IR
     * */
    virtual expr copy(const expr_c &v);

    /**
     * Copies the IR by calling dispatch(...) on it. Move-returns
     * `returned_stmt_`
     * @return the copied IR
     * */
    stmt copy(const stmt_c &v);

    /**
     * Copies the IR by calling dispatch(...) on it. Move-returns
     * `returned_stmt_`
     * @return the copied IR
     * */
    func_t copy(const func_c &v);

    func_c dispatch(func_c e) override;
    stmt_c dispatch(stmt_c s) override;
    void view(constant_c v) override;
    void view(var_c v) override;
    void view(cast_c v) override;
    void view(add_c v) override;
    void view(sub_c v) override;
    void view(mul_c v) override;
    void view(div_c v) override;
    void view(mod_c v) override;

    void view(cmp_eq_c v) override;
    void view(cmp_lt_c v) override;
    void view(cmp_le_c v) override;
    void view(cmp_gt_c v) override;
    void view(cmp_ge_c v) override;
    void view(cmp_ne_c v) override;

    void view(logic_and_c v) override;
    void view(logic_or_c v) override;

    void view(logic_not_c v) override;
    void view(select_c v) override;
    void view(indexing_c v) override;
    void view(call_c v) override;
    void view(tensor_c v) override;
    void view(tensorptr_c v) override;
    void view(intrin_call_c v) override;
    void view(low_level_intrin_c v) override;
    void view(func_addr_c v) override;
    void view(ssa_phi_c v) override;

    void view(assign_c v) override;
    void view(stmts_c v) override;
    void view(if_else_c v) override;
    void view(evaluate_c v) override;
    void view(returns_c v) override;
    void view(define_c v) override;
    void view(for_loop_c v) override;
};

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
