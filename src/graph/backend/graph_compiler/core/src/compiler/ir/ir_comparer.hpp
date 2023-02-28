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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_IR_COMPARER_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_IR_COMPARER_HPP

#include <memory>
#include <ostream>
#include <utility>
#include <vector>
#include "sc_stmt.hpp"
#include <unordered_map>
#include <util/utils.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

struct ir_comparer_diff_t {
    std::pair<func_c, func_c> first_diff_func_;
    std::pair<expr_c, expr_c> first_diff_expr_;
    std::pair<stmt_c, stmt_c> first_diff_stmt_;
};

class ir_comparer {
    std::unordered_map<const expr_base *, const expr_base *> var_mapping_;

public:
    std::unique_ptr<ir_comparer_diff_t> diff;
    bool cmp_names_;
    bool cmp_callee_;
    bool cmp_var_ref_;
    bool cmp_commutative_;
    bool same_;

    void reset();
    /**
     * The IR comparison context.
     * @param needs_diff true if need to return the first met different IR in
     *  "diff" field
     * @param cmp_names if false, will ignore the names of func/var/tensor.
     *  i.e. nodes with different "name_" may be treated as the same
     * @param cmp_var_ref if true, the comparison of var/tensor will be
     *  comparing vars/tensors by their pointers instead of comparing the names
     *  (decided by cmp_names), shapes and dtypes. Note: you can use
     *  set_expr_mapping() to map one var/tensor to another even if this flag is
     *  true. If the LHS var/tensor is in the mapping, IR comparison will check
     *  whether the pointer of the LHS expr is mapped in the mapping instead
     * @param cmp_callee similar to cmp_var_ref. if false, will compare callee
     * functions in call_nodes by their pointer instead of comparing using
     * func_base::equals()
     * @param cmp_commutative if the expr is commutative, try to match both
     * orders. DO NOT enable this when the expr is very complex
     * */
    ir_comparer(bool needs_diff = false, bool cmp_names = false,
            bool cmp_var_ref = false, bool cmp_callee = false,
            bool cmp_commutative = false);
    bool set_result(func_c l, func_c r, bool cond);
    bool set_result(expr_c l, expr_c r, bool cond);
    bool set_result(stmt_c l, stmt_c r, bool cond);
    bool expr_arr_equals(
            const std::vector<expr> &a, const std::vector<expr> &b);

    bool compare(const func_c &l, const func_c &r, bool auto_reset = true);
    bool compare(const expr_c &l, expr_c r, bool auto_reset = true);
    bool compare(const stmt_c &l, stmt_c r, bool auto_reset = true);

    /**
     * Checks if the var/tensor definition mapping is correct.
     * Will set a mapping l=>r if the mapping of l is not found
     * It will find the difference of:
     * var a:int
     * var b:int
     * a=b
     * ----------------------- and:
     * var a:int
     * var b:int
     * b=a
     * */
    bool check_or_set_expr_mapping(const expr_c &l, const expr_c &r);

    /**
     * Sets the expr mapping from l to r.
     * l and r should be var/tensor
     * The result of l->equals(r, *this) will be true
     * */
    void set_expr_mapping(const expr_c &l, const expr_c &r);

    /**
     * Gets the expr mapping from l to r.
     * l and r should be var/tensor
     * If l is mapped to r, returns true
     * */
    bool get_expr_mapping(const expr_c &l, const expr_c &r);

    template <typename T>
    bool check_equals_may_null(T a, T b) {
        if (!a.defined()) { return set_result(a, b, !b.defined()); }
        if (!b.defined()) { return set_result(a, b, false); }
        return a->equals(b, *this);
    }
};

std::ostream &operator<<(std::ostream &, ir_comparer &);

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
