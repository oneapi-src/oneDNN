/*******************************************************************************
 * Copyright 2021-2023 Intel Corporation
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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_JIT_XBYAK_BACKEND_LOCATION_MANAGER_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_JIT_XBYAK_BACKEND_LOCATION_MANAGER_HPP

#include <memory>
#include <ostream>
#include <set>
#include <string>
#include <vector>
#include <unordered_map>

#include <compiler/ir/content_hash.hpp>
#include <compiler/ir/ir_module.hpp>
#include <util/string_utils.hpp>

#include <compiler/jit/xbyak/backend/expr_location.hpp>
#include <compiler/jit/xbyak/backend/operand.hpp>
#include <compiler/jit/xbyak/backend/stack_frame_model.hpp>
#include <compiler/jit/xbyak/configured_xbyak.hpp>
#include <compiler/jit/xbyak/ir/reg_allocation/virtual_slot.hpp>
#include <compiler/jit/xbyak/ir/xbyak_expr.hpp>
#include <compiler/jit/xbyak/x86_64/abi_function_interface.hpp>
#include <compiler/jit/xbyak/x86_64/native_types.hpp>
#include <compiler/jit/xbyak/x86_64/target_profile.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace xbyak {

class location_manager {
public:
    location_manager(stack_frame_model &sf_model, Xbyak::CodeGenerator &gen,
            const x86_64::target_profile_t &profile);

    virtual ~location_manager();

    //--------------------------------------------------------------------------
    // Stack operation interface
    //--------------------------------------------------------------------------
    int64_t stack_push(const expr_c &v);
    int64_t stack_push(const expr_location &location);
    int64_t stack_push(const uint64_t &imm, x86_64::cpu_data_type dtype);
    int64_t stack_push(const Xbyak::Reg &reg, x86_64::cpu_data_type dtype);
    int64_t stack_push(const Xbyak::Address &addr, x86_64::cpu_data_type dtype);

    int64_t stack_pop(const expr_c &v);
    int64_t stack_pop(const expr_location &location);
    int64_t stack_pop(const Xbyak::Reg &reg, x86_64::cpu_data_type dtype);
    int64_t stack_pop(const Xbyak::Address &addr, x86_64::cpu_data_type dtype);

    void stack_padding(const size_t &padding_bytes_needed,
            const std::string &comment = "");
    void stack_restore(const size_t &stack_diff_to_restore);

    size_t stack_var_define(x86_64::cpu_data_type cpu_dtype,
            const std::string &name = "", const std::string &comment = "");
    size_t stack_tensor_define(x86_64::cpu_data_type cpu_dtype, size_t num_elem,
            const std::string &name = "", const std::string &comment = "");

    void stack_allocate(size_t slot_size);

    size_t get_stack_current_size();
    int64_t get_stack_top_rbp_offset();

    void conserve_stack_size();
    void restore_stack_size();

    //--------------------------------------------------------------------------
    // Function call interface
    //--------------------------------------------------------------------------
    /// Prepare value of each callee argument to facilitate abi function call
    void handle_call_arg(const expr_c &arg, const expr_c &v);
    /// Push caller saved expr
    void push_caller_saved(const std::vector<expr_c> &caller_saved);
    /// Prepare value of each function argument to track value passed from the
    /// caller to the callee. Update expr_location_map_ accordingly.
    void handle_func_params(const std::vector<expr> &func_params,
            const x86_64::abi_function_interface &func_iface);
    /// Prepare stack to meet ABI alignment requirement
    void align_call_stack(const x86_64::abi_function_interface &callee_iface);
    /// Pop caller saved expr
    void pop_caller_saved();

    //--------------------------------------------------------------------------
    // Codegen operation interface
    //--------------------------------------------------------------------------
    void handle_definition(const expr_c &v);
    void handle_spilled_definition(const std::vector<expr_c> &defined_spill);

    void prepare_local_scope(const std::vector<expr_c> &local_spill);
    void conclude_local_scope();

    void emit_callee_prologue(const std::set<virt_reg_index_t> &register_usage);
    void emit_callee_epilogue();

    void expire(stmt_index_t current_index);

    void clear();

    //--------------------------------------------------------------------------
    // Codegen Operand interface
    //--------------------------------------------------------------------------
    // get operand of expr location
    operand get_operand(const expr_location &v);
    // get operand of expr if expr exist
    operand get_operand(const expr_c &v);

    // get operand addr of indexing expr ptr[idx]
    operand get_operand_indexing(const indexing_c &v);
    // get operand addr of SIB structured for amx load/store
    operand get_operand_sib(
            const expr_c &base, const expr_c &indx, const expr_c &disp);

    //--------------------------------------------------------------------------
    // MISC. interface
    //--------------------------------------------------------------------------
    bool is_stack_tensor(const expr_c &v);

    size_t get_data_type_size(x86_64::cpu_data_type data_type);
    size_t get_data_slot_size(x86_64::cpu_data_type data_type);
    size_t get_tensor_slot_size(
            x86_64::cpu_data_type data_type, const size_t &num_elem);

    size_t get_tensor_static_num_elements(const tensor_c &v);
    size_t get_conserved_stack_size() const;

    /// Assuming our usual mapping of sc_data_type_t to CPU-native data types, /
    /// return a data frame of the appropriate width. E.g., \c gen_->dword or
    /// \c gen_->qword.
    const Xbyak::AddressFrame *get_address_frame(
            const x86_64::cpu_data_type cpu_dtype);

    template <typename T>
    void encode_simd_to_buffer(T *buffer, uint32_t lanes,
            const std::vector<union_val> &value,
            std::function<T(union_val)> select_val);
    const content_hash_map<expr_c, Xbyak::Label> &encode_simd_constant();

private:
    //--------------------------------------------------------------------------
    // Location management
    //--------------------------------------------------------------------------
    expr_location get_location(const expr_c &v);
    expr_location get_location(const constant_c &v);

    void load_location_to_reg(const Xbyak::Reg &reg, //
            const expr_location &location);
    void load_imm_value_to_reg(const Xbyak::Reg &reg, //
            const uint64_t &imm, x86_64::cpu_data_type data_type);
    void load_reg_value_to_reg(const Xbyak::Reg &reg, //
            const Xbyak::Reg &src, x86_64::cpu_data_type data_type);
    void load_mem_value_to_reg(const Xbyak::Reg &reg, //
            const Xbyak::Address &addr, x86_64::cpu_data_type data_type);
    void load_mem_addr_to_reg(const Xbyak::Reg &reg, //
            const Xbyak::Address &addr, x86_64::cpu_data_type data_type);

    Xbyak::Address get_address(
            const Xbyak::RegExp &exp, x86_64::cpu_data_type cpu_dtype);
    Xbyak::Address get_address(
            const Xbyak::RegRip &rxp, x86_64::cpu_data_type cpu_dtype);
    // %rbp + offset
    Xbyak::RegExp get_rbp_offset(const int64_t &offset);
    // %rip + label
    Xbyak::RegRip get_rip_offset(const Xbyak::Label &label);
    // AddressFrame[%rbp + offset]
    Xbyak::Address get_offset_address(
            const int64_t &offset, x86_64::cpu_data_type cpu_dtype);
    // AddressFrame[%rip + label]
    Xbyak::Address get_offset_address(
            const Xbyak::Label &label, x86_64::cpu_data_type cpu_dtype);

    //--------------------------------------------------------------------------
    // Register management
    //--------------------------------------------------------------------------
    expr_location allocate_free_reg(const expr_c &v);
    expr_location convert_virtual_reg(const expr_c &v);

    //--------------------------------------------------------------------------
    // MISC.
    //--------------------------------------------------------------------------
    stack_frame_model &sf_model_;
    Xbyak::CodeGenerator &gen_;
    const x86_64::target_profile_t &profile_;
    const runtime::cpu_flags_t &cpu_flags_;

    std::shared_ptr<virtual_slots_map_t> virtual_slots_map_;

    std::vector<expr_c> caller_saved_;
    std::vector<expr_location> callee_saved_;

    std::vector<size_t> conserved_stack_;

    std::unordered_map<expr_c, expr_location> local_location_map_;
    std::unordered_map<expr_c, expr_location> expr_location_map_;
    content_hash_map<expr_c, Xbyak::Label> simd_constant_map_;
    std::vector<expr_c> simd_constant_vec_;
};

} // namespace xbyak
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
