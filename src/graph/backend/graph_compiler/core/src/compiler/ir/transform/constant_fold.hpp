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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_TRANSFORM_CONSTANT_FOLD_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_TRANSFORM_CONSTANT_FOLD_HPP

#include <utility>
#include "../module_pass.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

namespace constant_folding {
std::pair<expr_c, expr_c> get_operand_from_binary(const expr_c &a);
bool is_op_commutative_and_associative(const expr_c &v);
} // namespace constant_folding

/**
 * Fold the constants.
 * Supported nodes:
 *  binary, cmp, logic, logic_not, cast, if_else
 *
 * It will do the following (c as constant, "+" as an example):
 * c1 + c2 => c3
 * c + x => x + c
 * (x + c1) + c2 => x + (c1 + c2)
 * (x + c) + y => (x + y) + c
 * x + (y + c) => (x + y) + c
 * (x + c1) + (y + c2) => (x + y) + (c1 + c2)
 *
 * Also fold special expr:
 * a (+ - * && ||) 0/false
 * a (* / % && ||) 1/true
 * a (- / % && || max min > >= < <= == !=) a
 *
 * @param fast whether check if varaibles are actually constants. If true, will
 * skip this check and keep the variables
 * */
class constant_folder_t : public module_pass_t {
public:
    bool fast_;
    constant_folder_t(bool fast = true) : fast_(fast) {}
    func_c operator()(func_c f) const;
    stmt_c operator()(stmt_c f) const;
    expr_c operator()(expr_c f) const;
    const_ir_module_ptr operator()(const_ir_module_ptr f) override;
    expr_c expand_polynomial(expr_c f, int max_iter = 1, bool skip_mod = false);
    SC_DECL_PASS_INFO_FUNC();
};
// do auto cast and constant fold for input expr.
expr do_cast_and_fold(const expr &in);
expr_c do_cast_and_fold(const expr_c &in);
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
