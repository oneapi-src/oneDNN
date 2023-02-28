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
#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_CODEGEN_CODEGEN_C_INTERNAL_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_CODEGEN_CODEGEN_C_INTERNAL_HPP

#include <vector>
#include "../ir/util_module_passes.hpp"
#include "../ir/viewer.hpp"

/**
 * This header exposes some utility functions for C++-like language codegen.
 * To use standard C++ codegen, please include codegen_c.hpp instead
 * */

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
class codegen_c_vis : public ir_viewer_t {
protected:
    ostream *os;
    int indents = 0;
    bool prototype_only;
    bool is_static;
    ostream &print_param(const expr &v);
    ostream &print_func_params(const func_c &v, bool with_type = true);
    void trinary_func_codegen_c(
            const std::vector<expr> &args, const char *funcname);
    void binary_func_codegen_c(
            const std::vector<expr> &args, const char *funcname);
    void unary_func_codegen_c(const expr &arg, const char *funcname);

public:
    bool is_offline_ = false;
    virtual ostream &print_cpp_var_def(const var &v);
    virtual ostream &print_tensor_def(const tensor &v);
    using ir_viewer_t::dispatch;
    using ir_viewer_t::view;
    codegen_c_vis(ostream *os, bool prototype_only, bool is_static = false);
    func_c dispatch(func_c v) override;
    stmt_c dispatch(stmt_c v) override;
    void view(constant_c v) override;
    void view(var_c v) override;
    void view(cast_c v) override;

    void print_binary(const binary_c &v, const char *op);
    void print_binary(const logic_c &v, const char *op);
    void print_binary(const cmp_c &v, const char *op);

    virtual void print_type(sc_data_type_t dtype);

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
    void view(tensorptr_c v) override;
    void view(intrin_call_c v) override;
    void view(func_addr_c v) override;
    void view(call_c v) override;
    void view(tensor_c v) override;
    void view(assign_c v) override;
    void view(stmts_c v) override;
    void view(if_else_c v) override;
    void view(evaluate_c v) override;
    void view(returns_c v) override;
    void view(define_c v) override;
    void view(for_loop_c v) override;
};

struct c_generator_optional_out_t;
extern const_ir_module_ptr preprocess_module_and_make_decl(
        const const_ir_module_ptr &mod, module_pass_t &pre_passes,
        std::ostream &source,
        c_generator_optional_out_t *optionalout = nullptr);
extern ostream &print_cpp_type(ostream &os, sc_data_type_t dtype);
extern void write_cpp_prototype(
        std::ostream *source_, const func_c &f, bool is_offline = false);
extern void write_cpp_generic_wrapper(
        std::ostream *source_, const func_c &f, bool is_parallel);
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
