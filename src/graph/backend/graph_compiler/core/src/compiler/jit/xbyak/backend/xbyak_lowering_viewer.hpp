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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_JIT_XBYAK_BACKEND_XBYAK_LOWERING_VIEWER_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_JIT_XBYAK_BACKEND_XBYAK_LOWERING_VIEWER_HPP

#include <map>
#include <memory>
#include <ostream>
#include <set>
#include <string>
#include <vector>

#include <compiler/jit/xbyak/configured_xbyak.hpp>

#include <compiler/ir/viewer.hpp>
#include <compiler/jit/xbyak/backend/location_manager.hpp>
#include <compiler/jit/xbyak/backend/stack_frame_model.hpp>
#include <compiler/jit/xbyak/backend/xbyak_jit_generator.hpp>
#include <compiler/jit/xbyak/ir/xbyak_expr.hpp>
#include <compiler/jit/xbyak/x86_64/abi_function_interface.hpp>
#include <compiler/jit/xbyak/x86_64/abi_value_location.hpp>
#include <compiler/jit/xbyak/x86_64/native_types.hpp>
#include <compiler/jit/xbyak/xbyak_jit.hpp>
#include <util/array_ref.hpp>
#include <util/string_utils.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace xbyak {

/**
 * Provides most of the logic for translating Graphcompiler IR modules
 * into in-memory code and data.
 */
class xbyak_lowering_viewer : protected ir_viewer_t {
public:
    /// This performs the JIT translation. If no problem occurs, the
    /// JIT output can be obtained via the 'get_jit_output()' method.
    ///
    /// \param xjm The \c xbyak_jit_module that will present the lowered
    /// code to the rest of the Graphcompiler.
    ///
    /// \param ir_mod The Graphcompiler IR module to be JIT'ed. The
    /// reference only needs to be valid during this c'tor's execution.
    xbyak_lowering_viewer(const xbyak_jit &xje, const ir_module_t &ir_mod,
            const x86_64::target_profile_t &profile);

    virtual ~xbyak_lowering_viewer();

    /// Obtain the object used to access the JIT'ed code and data.
    ///
    /// For a given instance of @c xbyak_lowering_generator, this will
    /// always return the same object.
    /// The returned object remains valid even after this
    /// @c xbyak_lowering_viewer is destroyed.
    std::shared_ptr<xbyak_jit_generator> get_jit_output() const;

private:
    //--------------------------------------------------------------------------
    // Support for SC_XBYAK_JIT_ASM_LISTING ... FIXME: Redesign this
    //--------------------------------------------------------------------------
    struct code_comment {
        code_comment(const Xbyak::Label &label, std::string comment);
        Xbyak::Label label_;
        std::string comment_;
        const node_base *ir_node_;
    };

    struct label_line {
        Xbyak::Label label_;
        const node_base *ir_node_;
    };

    std::vector<code_comment> code_comments_;
    std::vector<label_line> debug_lines_;

    void add_code_comment(std::string text);

    /// Only call this after labels have been resolved to actual memory
    /// addresses.
    std::vector<std::unique_ptr<debug_info_mgr>> dump_code_comments(
            std::ostream &os);

    //--------------------------------------------------------------------------
    // Member variables
    //--------------------------------------------------------------------------
    enum class simd_level {
        sse = 0,
        avx,
        avx2,
        avx512,
    } simd_level_;

    const xbyak_jit &xje_;

    /// A pointer to the module that's being lowered.
    /// Only non-null during the execution of this object's c'tor.
    const ir_module_t *p_ir_mod_;

    const x86_64::target_profile_t &profile_;
    const runtime::cpu_flags_t &cpu_flags_;

    std::shared_ptr<xbyak_jit_generator> gen_;
    std::unique_ptr<location_manager> location_manager_;

    // the map only covers IR-defined functions, not external ones.
    std::map<std::string, Xbyak::Label> func_name_to_entry_label_;
    std::map<std::string, Xbyak::Label> func_name_to_exit_label_;

    stack_frame_model sf_model_;

    // During the JIT-translation of a function, this indicates the beginning
    // of the function's epilogue code.
    Xbyak::Label l_func_epilogue_;

    // During the JIT-translation of a function, this describes the ABI
    // details for calling that function.
    x86_64::abi_function_interface::ptr func_iface_;

    utils::indentation_t logging_ind_;

    utils::indentation_t asm_listing_ind_;
    //--------------------------------------------------------------------------
    // Member functions
    //--------------------------------------------------------------------------
    const std::vector<expr_c> &cached_func_global_spilled(const func_t &v);
    const std::set<virt_reg_index_t> &cached_func_register_usage(
            const func_t &v);

    // Obtaining the callee's address:
    //
    //  - If the callee is an IR function, then our lookup will provide us
    //    with an Xbyak::Label. Xbyak will resolve that label to an actual
    //    memory address during final codegen.
    //
    //  - If the callee is a registered external function, then our lookup
    //    will provide us with an actual memory address.
    //
    using execute_func_label = std::function<void(const Xbyak::Label &label)>;
    using execute_func_addr = std::function<void(const uint64_t &addr)>;

    void handle_func_resolve(const std::string &name,
            const execute_func_label &label_f, const execute_func_addr &addr_f);
    void handle_local_definition(const expr_c &v, const expr_c &v_init);

    //--------------------------------------------------------------------------
    // Oprations handlers
    //--------------------------------------------------------------------------
    // dispatch operations
    void handle_operations(const expr_c &dst, const expr_c &src);
    void handle_xbyak_intrin(const expr_c &lhs, const xbyak_intrin_c &rhs);
    void handle_x86_intrisic(const expr_c &dst, array_ref<expr> args,
            const xbyak_intrin_type &intrin,
            const xbyak_intrin_modifier &modifier = xbyak_intrin_modifier());
    void handle_avx_intrisic(const expr_c &dst, array_ref<expr> args,
            const xbyak_intrin_type &intrin,
            const xbyak_intrin_modifier &modifier = xbyak_intrin_modifier());

    // general operations
    void handle_assign(const expr_c &lhs, const expr_c &rhs);
    void handle_func_addr(const expr_c &lhs, const func_addr_c &rhs);
    void handle_tensorptr(const expr_c &lhs, const tensorptr_c &rhs);

    // call operations
    void handle_call(const expr_c &lhs, const call_c &v);
    void handle_pre_call(const stmts_c &v);
    void handle_post_call();

    // cast operations
    void handle_cast(const expr_c &lhs, const cast_c &v);
    void handle_saturated_cast(const expr_c &dst, const expr_c &src);
    void handle_round_and_cast(const expr_c &dst, const expr_c &src);
    void handle_reinterpret(const expr_c &dst, const expr_c &src);

    // x86 operations
    void handle_x86_mov(const operand &op_dst, const operand &op_src);

    void handle_x86_test(const operand &op_cond);
    void handle_x86_sign_ext(
            const operand &op_rdx, const x86_64::cpu_data_type &cpu_dtype);
    void handle_x86_div(
            const operand &op_div, const x86_64::cpu_data_type &cpu_dtype);
    void handle_x86_cmp(const operand &op_lhs, const operand &op_rhs);
    void handle_x86_set(const operand &op_dst, const xbyak_condition &cond,
            const x86_64::cpu_data_type &cpu_dtype);
    void handle_x86_cmov(const operand &op_dst, const operand &op_src,
            const xbyak_condition &code,
            const x86_64::cpu_data_type &cpu_dtype);

    // avx operations
    void handle_avx_movq(const operand &op_dst, const operand &op_src);
    void handle_avx_movss(const operand &op_dst, const operand &op_src);
    void handle_avx_movsh(const operand &op_dst, const operand &op_src);
    void handle_avx_movps(const operand &op_dst, const operand &op_src);
    void handle_avx_movph(const operand &op_dst, const operand &op_src);
    void handle_avx512_kmov(const operand &op_dst, const operand &op_src,
            const x86_64::cpu_data_type &cpu_dtype);

    void handle_avx_add(const operand &op_dst, const operand &op_lhs,
            const operand &op_rhs, const x86_64::cpu_data_type &cpu_dtype);
    void handle_avx_sub(const operand &op_dst, const operand &op_lhs,
            const operand &op_rhs, const x86_64::cpu_data_type &cpu_dtype);
    void handle_avx_mul(const operand &op_dst, const operand &op_lhs,
            const operand &op_rhs, const x86_64::cpu_data_type &cpu_dtype);
    void handle_avx_mulhl(const operand &op_dst, const operand &op_lhs,
            const operand &op_rhs, const x86_64::cpu_data_type &cpu_dtype);
    void handle_avx_div(const operand &op_dst, const operand &op_lhs,
            const operand &op_rhs, const x86_64::cpu_data_type &cpu_dtype);
    void handle_avx_shl(const operand &op_dst, const operand &op_lhs,
            const operand &op_sft, const x86_64::cpu_data_type &cpu_dtype,
            bool variable);
    void handle_avx_shr(const operand &op_dst, const operand &op_lhs,
            const operand &op_sft, const x86_64::cpu_data_type &cpu_dtype,
            bool variable);
    void handle_avx_sar(const operand &op_dst, const operand &op_lhs,
            const operand &op_sft, const x86_64::cpu_data_type &cpu_dtype,
            bool variable);
    void handle_avx_max(const operand &op_dst, const operand &op_lhs,
            const operand &op_rhs, const x86_64::cpu_data_type &cpu_dtype);
    void handle_avx_abs(const operand &op_lhs, const operand &op_rhs,
            const x86_64::cpu_data_type &cpu_dtype);
    void handle_avx_min(const operand &op_dst, const operand &op_lhs,
            const operand &op_rhs, const x86_64::cpu_data_type &cpu_dtype);
    void handle_avx_bit_or(const operand &op_dst, const operand &op_lhs,
            const operand &op_rhs, const x86_64::cpu_data_type &cpu_dtype);
    void handle_avx_bit_and(const operand &op_dst, const operand &op_lhs,
            const operand &op_rhs, const x86_64::cpu_data_type &cpu_dtype);
    void handle_avx_bit_xor(const operand &op_dst, const operand &op_lhs,
            const operand &op_rhs, const x86_64::cpu_data_type &cpu_dtype);
    void handle_avx_round(const operand &op_lhs, const operand &op_rhs,
            const x86_64::cpu_data_type &cpu_dtype, const int64_t &imm);
    void handle_avx_sqrt(const operand &op_dst, const operand &op_src,
            const x86_64::cpu_data_type &cpu_dtype);
    void handle_avx_rsqrt(const operand &op_dst, const operand &op_src,
            const x86_64::cpu_data_type &cpu_dtype);
    void handle_avx_fmadd(const operand &op_dst, const operand &op_mul,
            const operand &op_add, const x86_64::cpu_data_type &cpu_dtype);
    void handle_avx_pshuffle(const operand &op_dst, const operand &op_lhs,
            const operand &op_rhs, const x86_64::cpu_data_type &cpu_dtype);
    void handle_avx_shuffle(const operand &op_dst, const operand &op_lhs,
            const operand &op_rhs, const operand &op_imm,
            const operand &op_bits);
    void handle_avx_permute(const operand &op_dst, const operand &op_lhs,
            const operand &op_rhs, const operand &op_imm,
            const x86_64::cpu_data_type &cpu_dtype);
    void handle_avx_gather(const operand &op_dst, const operand &op_ptr,
            const operand &op_idx, const operand &op_msk,
            const x86_64::cpu_data_type &cpu_dtype);
    void handle_avx_insert(const operand &op_dst, const operand &op_b,
            const operand &op_imm, const operand &op_elem_bits);
    void handle_avx_extract(const operand &op_dst, const operand &op_b,
            const operand &op_imm, const operand &op_elem_bits);
    void handle_avx_permutex2var(const operand &op_dst, const operand &op_idx,
            const operand &op_src, const x86_64::cpu_data_type &cpu_dtype);
    void handle_avx_permutexvar(const operand &op_dst, const operand &op_idx,
            const operand &op_src, const x86_64::cpu_data_type &cpu_dtype,
            const operand &bits);
    void handle_avx_unpack_low(const operand &op_dst, const operand &op_lhs,
            const operand &op_rhs, const operand &op_imm);
    void handle_avx_unpack_high(const operand &op_dst, const operand &op_lhs,
            const operand &op_rhs, const operand &op_imm);
    void handle_avx_extract_low(const operand &op_dst, const operand &op_src,
            const x86_64::cpu_data_type &cpu_dtype);
    void handle_avx_extract_high(const operand &op_dst, const operand &op_src,
            const x86_64::cpu_data_type &cpu_dtype);
    void handle_avx_broadcast(const operand &op_dst, const operand &op_src,
            const x86_64::cpu_data_type &cpu_dtype,
            const x86_64::cpu_data_type &src_dtype);
    void handle_avx_blend(const operand &op_dst, const operand &op_lhs,
            const operand &op_rhs, const operand &op_cond,
            const x86_64::cpu_data_type &cpu_dtype);
    void handle_avx_mask_mov(const operand &op_dst, const operand &op_src,
            const operand &op_cond, const x86_64::cpu_data_type &cpu_dtype,
            bool zero);
    void handle_avx_mov_mask(const operand &op_dst, const operand &op_src,
            const x86_64::cpu_data_type &cpu_dtype);
    void handle_avx_cmov(const operand &op_dst, const operand &op_src,
            const xbyak_condition &code,
            const x86_64::cpu_data_type &cpu_dtype);
    void handle_avx_cmp_set(const operand &op_dst, const operand &op_lhs,
            const operand &op_rhs, const xbyak_condition &code,
            const x86_64::cpu_data_type &cpu_dtype);

protected:
    using ir_viewer_t::dispatch;
    using ir_viewer_t::view;

    //--------------------------------------------------------------------------
    // Overrides of ir_viewer_t methods...
    //--------------------------------------------------------------------------
    stmt_c dispatch(stmt_c v) override;
    func_c dispatch(func_c v) override;

    void view(stmts_c v) override;
    void view(evaluate_c v) override;
    void view(assign_c v) override;
    void view(define_c v) override;
    void view(returns_c v) override;
    void view(if_else_c v) override;
    void view(for_loop_c v) override;
};

} // namespace xbyak
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
