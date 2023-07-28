/*******************************************************************************
 * Copyright 2023 Intel Corporation
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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_CODEGEN_LLVM_LLVM_VISITOR_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_CODEGEN_LLVM_LLVM_VISITOR_HPP

/*
Developer notes:
We break the implementation of codegen_llvm_vis_t into multiple files, because
we need to workaround a hidden g++-7 bug linking with LLVM16. We found that when
the size of a single cpp source/binary for LLVM-IR generation increases to a
certain size, the generated in-memory LLVM-IR will be broken. It results in
segfault in LLVM's internal passes. g++-9 does not break LLVM16.
*/

#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <compiler/config/context.hpp>
#include <compiler/ir/pass/printer.hpp>
#include <compiler/ir/transform/pointer_alias_info.hpp>
#include <compiler/ir/viewer.hpp>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DIBuilder.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/MDBuilder.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>
#include <unordered_map>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
class codegen_llvm_vis_t : public ir_viewer_t {
public:
    context_ptr ctx_;
    llvm::LLVMContext &context_;
    llvm::IRBuilder<> builder_;
    std::unique_ptr<llvm::Module> module_;
    std::unique_ptr<llvm::DIBuilder> dbuilder_;
    llvm::DICompileUnit *dbg_cu_ = nullptr;
    std::vector<llvm::DIScope *> dbg_scopes_;
    llvm::Function *current_func_ = nullptr;
    llvm::Value *current_val_ = nullptr;
    // the **pointer** of local var in a function
    std::unordered_map<expr_c, llvm::Value *> var_ptr_in_func_;
    using vec_metadata = llvm::SmallVector<llvm::Metadata *, 4>;
    // tensor to <alias scope, noalias>
    std::unordered_map<expr_c, std::pair<llvm::MDNode *, llvm::MDNode *> *>
            tsr_to_alias_scope_;
    std::unordered_map<alias_info::tensor_alias_identity_t *,
            std::pair<llvm::MDNode *, llvm::MDNode *>>
            alias_set_to_alias_scope_;
    std::unordered_map<std::string, llvm::Function *> name_to_func_;
    bool is_lvalue_mode_ = false;

    codegen_llvm_vis_t(const context_ptr &ctx, llvm::LLVMContext &context,
            const std::string &source_dir, const std::string &source_file_name);

    using ir_viewer_t::dispatch;
    using ir_viewer_t::view;

    expr_c dispatch(expr_c v) override;

    stmt_c dispatch(stmt_c v) override;

    const std::string &get_node_name(const expr_c &c);

    llvm::FunctionType *create_func_type(const func_c &v);

    llvm::DISubroutineType *create_func_dtype(const func_c &v);

    llvm::Function *get_or_create_func(const func_c &v);

    llvm::Value *generate_expr(const expr_c &e);

    std::unordered_map<uint64_t, std::pair<llvm::Type *, llvm::DIType *>>
            type_cache_;
    std::pair<llvm::Type *, llvm::DIType *> do_get_type(sc_data_type_t dtype);

    std::pair<llvm::Type *, llvm::DIType *> get_type_both(sc_data_type_t dtype);

    llvm::Type *get_type(sc_data_type_t dtype);

    llvm::Value *get_defined_var_ptr(const expr_c &e);

    llvm::Value *define_var(const expr_c &e, llvm::Value *initv);

    void set_dbg_info_for_func_arg(llvm::Value *v, llvm::DISubprogram *SP,
            llvm::DIFile *dunit, sc_data_type_t type, const std::string &name,
            int argidx, int lineno, bool need_ref);
    void emit_location(const node_base *p);
    void prepare_alias_metadata(
            const std::vector<alias_info::tensor_alias_identity_t *> &tensors,
            llvm::MDNode *alias_domain, llvm::MDBuilder &MDB);

    func_c dispatch(func_c v) override;
    void view(constant_c v) override;
    void view(var_c v) override;

    llvm::Type *get_llvm_bf16_native_type(unsigned lanes);

    void view(cast_c v) override;
    void generate_bin_op(const expr_c &v, const expr_c &l, const expr_c &r);
    void view(binary_c v) override;
    void view(cmp_c v) override;

    void view(logic_c v) override;
    void view(logic_not_c v) override;
    void view(select_c v) override;

    llvm::Instruction *set_alias(llvm::Instruction *inst, const expr_c &tsr);

    llvm::Value *set_alias(llvm::Value *inst, const expr_c &tsr);

    llvm::Value *convert_mask(const expr &in, const bool is_int4 = false);

    void view(indexing_c v) override;
    void view(tensorptr_c v) override;

    llvm::Value *gen_vec_const(uint64_t elements, float f);

    llvm::Value *call_unary_llvm_intrin(const intrin_call_c &v,
            type_category cate, llvm::Intrinsic::ID id, bool must_fp);

    llvm::Value *call_binary_llvm_intrin(const intrin_call_c &v,
            type_category cate, llvm::Intrinsic::ID id, bool must_fp);

    typedef llvm::Value *(llvm::IRBuilder<>::*llvm_binary_func)(
            llvm::Value *LHS, llvm::Value *RHS, const llvm::Twine &Name);
    llvm::Value *call_binary_llvm_normal(
            const intrin_call_c &v, llvm_binary_func op);

    llvm::Value *make_int_min_max(
            const intrin_call_c &v, bool ismin, type_category cate);

    llvm::Value *make_int_min_max(
            llvm::Value *v1, llvm::Value *v2, bool ismin, type_category cate);

    llvm::Value *do_lower_saturated_cast(const intrin_call_c &v);

    void view(intrin_call_c v) override;
    void view(func_addr_c v) override;
    void view(call_c v) override;
    void view(tensor_c v) override;
    // void view(stmts_c v) override;
    // void view(evaluate_c v) override;

    void generate_codeblock(
            const stmt_c &v, llvm::BasicBlock *current, llvm::BasicBlock *cont);

    void view(assign_c v) override;

    void view(if_else_c v) override;

    void view(returns_c v) override;

    void set_dbg_info_for_local_var(const source_pos *pos, sc_data_type_t type,
            const std::string &name, llvm::Value *llvm_value, bool need_ref);

    void set_dbg_info_for_local_var(const define_node_t *v,
            const std::string &name, llvm::Value *llvm_value, bool need_ref);

    void view(define_c v) override;

    void view(for_loop_c v) override;
    ~codegen_llvm_vis_t();
};
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
