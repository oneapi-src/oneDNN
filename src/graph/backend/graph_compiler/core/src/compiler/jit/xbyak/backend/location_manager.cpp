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
#include <cfloat>
#include <cmath>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <compiler/ir/transform/auto_cast.hpp>
#include <compiler/jit/xbyak/ir/transform/register_allocation.hpp>
#include <compiler/jit/xbyak/x86_64/type_mapping.hpp>
#include <util/bf16.hpp>
#include <util/utils.hpp>

#include "location_manager.hpp"
#include "util/fp16.hpp"

SC_MODULE(xbyakjit.location_manager)

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace xbyak {

using namespace xbyak::x86_64;

//==============================================================================
// LOCATION MANAGER
//==============================================================================

location_manager::location_manager(stack_frame_model &sf_model,
        Xbyak::CodeGenerator &gen, const x86_64::target_profile_t &profile)
    : sf_model_(sf_model)
    , gen_(gen)
    , profile_(profile)
    , cpu_flags_(profile.target_machine_.cpu_flags_) {
    // Current target virtual reg mapping
    virtual_slots_map_ = std::make_shared<virtual_slots_map_t>(profile_);
}

location_manager::~location_manager() = default;

//==============================================================================
// Stack operation interface
//==============================================================================

// stack_push expr
int64_t location_manager::stack_push(const expr_c &v) {
    return stack_push(get_location(v));
}

// stack_push location
int64_t location_manager::stack_push(const expr_location &location) {
    auto cpu_type = location.get_data_type();
    switch (location.get_type()) {
        case expr_location::type::reg: {
            return stack_push(location.get_reg(), cpu_type);
        } break;
        case expr_location::type::imm: {
            return stack_push(location.get_imm(), cpu_type);
        } break;
        case expr_location::type::stack_var:
        case expr_location::type::simd_constant: {
            auto op = get_operand(location);
            return stack_push(op.get_addr(), cpu_type);
        } break;
        default: {
            COMPILE_ASSERT(false, "Invalid stack push: " << location);
        }
    }

    return get_stack_top_rbp_offset();
}

// stack_push imm
int64_t location_manager::stack_push(
        const uint64_t &imm, x86_64::cpu_data_type dtype) {
    const size_t slot_size = get_data_slot_size(dtype);

    switch (dtype) {
        case x86_64::cpu_data_type::uint_8:
        case x86_64::cpu_data_type::sint_8:
        case x86_64::cpu_data_type::sint_32:
        case x86_64::cpu_data_type::uint_64: {
            gen_.sub(gen_.rsp, slot_size);
            gen_.mov(gen_.qword[gen_.rsp], imm);
        } break;
        default: COMPILE_ASSERT(false, "Invalid stack push imm: " << dtype);
    }

    sf_model_.push_anonymous_object(
            dtype, slot_size, "stack push imm: " + std::to_string(imm));
    return get_stack_top_rbp_offset();
}

// stack_push reg
int64_t location_manager::stack_push(
        const Xbyak::Reg &reg, x86_64::cpu_data_type dtype) {
    const size_t slot_size = get_data_slot_size(dtype);

    switch (dtype) {
        // integer 8-bit/ 1-byte
        case cpu_data_type::uint_8:
        case cpu_data_type::sint_8:
        // integer 16-bit/ 2-byte
        case cpu_data_type::uint_16:
        // integer 32-bit/ 4-byte
        case cpu_data_type::uint_32:
        case cpu_data_type::sint_32:
        // integer 64-bit/ 8-byte
        case cpu_data_type::uint_64: {
            gen_.push(to_reg64(reg));
        } break;
        // simd 32-bit/ 4-byte
        case cpu_data_type::float_16: {
            gen_.sub(gen_.rsp, slot_size);
            gen_.vmovsh(gen_.word[gen_.rsp], to_xmm(reg));
        } break;
        case cpu_data_type::float_32: {
            gen_.sub(gen_.rsp, slot_size);
            gen_.vmovss(gen_.dword[gen_.rsp], to_xmm(reg));
        } break;
        // simd 64-bit/ 8-byte
        case cpu_data_type::uint_8_x8:
        case cpu_data_type::sint_8_x8:
        case cpu_data_type::uint_16_x4:
        case cpu_data_type::uint_32_x2:
        case cpu_data_type::sint_32_x2:
        case cpu_data_type::float_16_x4:
        case cpu_data_type::float_32_x2: {
            gen_.sub(gen_.rsp, slot_size);
            gen_.vmovq(gen_.qword[gen_.rsp], to_xmm(reg));
        } break;
        // simd 128-bit/ 16-byte
        case cpu_data_type::uint_8_x16:
        case cpu_data_type::sint_8_x16:
        case cpu_data_type::uint_16_x8:
        case cpu_data_type::uint_32_x4:
        case cpu_data_type::sint_32_x4:
        case cpu_data_type::uint_64_x2:
        case cpu_data_type::float_16_x8:
        case cpu_data_type::float_32_x4: {
            gen_.sub(gen_.rsp, slot_size);
            gen_.vmovups(gen_.xword[gen_.rsp], to_xmm(reg));
        } break;
        // simd 256-bit/ 32-byte
        case cpu_data_type::uint_8_x32:
        case cpu_data_type::sint_8_x32:
        case cpu_data_type::uint_16_x16:
        case cpu_data_type::uint_32_x8:
        case cpu_data_type::sint_32_x8:
        case cpu_data_type::uint_64_x4:
        case cpu_data_type::float_16_x16:
        case cpu_data_type::float_32_x8: {
            gen_.sub(gen_.rsp, slot_size);
            gen_.vmovups(gen_.yword[gen_.rsp], to_ymm(reg));
        } break;
        // simd 512-bit/ 64-byte
        case cpu_data_type::uint_8_x64:
        case cpu_data_type::sint_8_x64:
        case cpu_data_type::uint_16_x32:
        case cpu_data_type::uint_32_x16:
        case cpu_data_type::sint_32_x16:
        case cpu_data_type::uint_64_x8:
        case cpu_data_type::float_16_x32:
        case cpu_data_type::float_32_x16: {
            gen_.sub(gen_.rsp, slot_size);
            gen_.vmovups(gen_.zword[gen_.rsp], to_zmm(reg));
        } break;
        // not supported
        case cpu_data_type::mask_x4:
        case cpu_data_type::mask_x8:
        case cpu_data_type::mask_x16:
        case cpu_data_type::mask_x32:
        case cpu_data_type::mask_x64:
        case cpu_data_type::void_t: {
            COMPILE_ASSERT(false, "Invalid stack push reg: " << dtype);
        } break;
    }

    sf_model_.push_anonymous_object(
            dtype, slot_size, "stack push reg: " + std::string(reg.toString()));
    return get_stack_top_rbp_offset();
}

// stack_push addr
int64_t location_manager::stack_push(
        const Xbyak::Address &addr, x86_64::cpu_data_type dtype) {
    const size_t slot_size = get_data_slot_size(dtype);

    switch (dtype) {
        case x86_64::cpu_data_type::uint_8:
        case x86_64::cpu_data_type::sint_8:
        case x86_64::cpu_data_type::sint_32:
        case x86_64::cpu_data_type::uint_64:
        case x86_64::cpu_data_type::float_32: {
            auto addr64 = addr;
            addr64.setBit(64);
            gen_.push(addr64);
        } break;
        default: COMPILE_ASSERT(false, "Invalid stack push addr: " << dtype);
    }

    sf_model_.push_anonymous_object(dtype, slot_size, "stack push addr");
    return get_stack_top_rbp_offset();
}

// stack_pop expr
int64_t location_manager::stack_pop(const expr_c &v) {
    return stack_pop(get_location(v));
}

// stack_pop location
int64_t location_manager::stack_pop(const expr_location &location) {
    auto cpu_type = location.get_data_type();
    switch (location.get_type()) {
        case expr_location::type::reg: {
            return stack_pop(location.get_reg(), cpu_type);
        } break;
        case expr_location::type::stack_var: {
            auto op = get_operand(location);
            return stack_pop(op.get_addr(), cpu_type);
        } break;
        default: {
            COMPILE_ASSERT(false, "Invalid stack pop: " << location);
        }
    }

    return get_stack_top_rbp_offset();
}

// stack_pop reg
int64_t location_manager::stack_pop(
        const Xbyak::Reg &reg, x86_64::cpu_data_type dtype) {
    const size_t slot_size = get_data_slot_size(dtype);

    switch (dtype) {
        // integer 8-bit/ 1-byte
        case cpu_data_type::uint_8:
        case cpu_data_type::sint_8:
        // integer 16-bit/ 2-byte
        case cpu_data_type::uint_16:
        // integer 32-bit/ 4-byte
        case cpu_data_type::uint_32:
        case cpu_data_type::sint_32:
        // integer 64-bit/ 8-byte
        case cpu_data_type::uint_64: {
            auto reg_gp = to_reg64(reg);
            gen_.pop(reg_gp);
        } break;
        // simd 16-bit/ 2-byte
        case cpu_data_type::float_16: {
            auto reg_xmm = to_xmm(reg);
            gen_.vmovsh(reg_xmm, gen_.word[gen_.rsp]);
            gen_.add(gen_.rsp, slot_size);
        } break;
        // simd 32-bit/ 4-byte
        case cpu_data_type::float_32: {
            auto reg_xmm = to_xmm(reg);
            gen_.vmovss(reg_xmm, gen_.dword[gen_.rsp]);
            gen_.add(gen_.rsp, slot_size);
        } break;
        // simd 64-bit/ 8-byte
        case cpu_data_type::uint_8_x8:
        case cpu_data_type::sint_8_x8:
        case cpu_data_type::uint_16_x4:
        case cpu_data_type::uint_32_x2:
        case cpu_data_type::sint_32_x2:
        case cpu_data_type::float_16_x4:
        case cpu_data_type::float_32_x2: {
            auto reg_xmm = to_xmm(reg);
            gen_.vmovq(reg_xmm, gen_.qword[gen_.rsp]);
            gen_.add(gen_.rsp, slot_size);
        } break;
        // simd 128-bit/ 16-byte
        case cpu_data_type::uint_8_x16:
        case cpu_data_type::sint_8_x16:
        case cpu_data_type::uint_16_x8:
        case cpu_data_type::uint_32_x4:
        case cpu_data_type::sint_32_x4:
        case cpu_data_type::uint_64_x2:
        case cpu_data_type::float_16_x8:
        case cpu_data_type::float_32_x4: {
            auto reg_xmm = to_xmm(reg);
            gen_.vmovups(reg_xmm, gen_.xword[gen_.rsp]);
            gen_.add(gen_.rsp, slot_size);
        } break;
        // simd 256-bit/ 32-byte
        case cpu_data_type::uint_8_x32:
        case cpu_data_type::sint_8_x32:
        case cpu_data_type::uint_16_x16:
        case cpu_data_type::uint_32_x8:
        case cpu_data_type::sint_32_x8:
        case cpu_data_type::uint_64_x4:
        case cpu_data_type::float_16_x16:
        case cpu_data_type::float_32_x8: {
            auto reg_ymm = to_ymm(reg);
            gen_.vmovups(reg_ymm, gen_.yword[gen_.rsp]);
            gen_.add(gen_.rsp, slot_size);
        } break;
        // simd 512-bit/ 64-byte
        case cpu_data_type::uint_8_x64:
        case cpu_data_type::sint_8_x64:
        case cpu_data_type::uint_16_x32:
        case cpu_data_type::uint_32_x16:
        case cpu_data_type::sint_32_x16:
        case cpu_data_type::uint_64_x8:
        case cpu_data_type::float_16_x32:
        case cpu_data_type::float_32_x16: {
            auto reg_zmm = to_zmm(reg);
            gen_.vmovups(reg_zmm, gen_.zword[gen_.rsp]);
            gen_.add(gen_.rsp, slot_size);
        } break;
        // not supported
        case cpu_data_type::mask_x4:
        case cpu_data_type::mask_x8:
        case cpu_data_type::mask_x16:
        case cpu_data_type::mask_x32:
        case cpu_data_type::mask_x64:
        case cpu_data_type::void_t: {
            COMPILE_ASSERT(false, "Invalid stack pop reg: " << dtype);
        } break;
    }

    sf_model_.shrink(slot_size);
    return get_stack_top_rbp_offset();
}

// stack_pop addr
int64_t location_manager::stack_pop(
        const Xbyak::Address &addr, x86_64::cpu_data_type dtype) {
    const size_t slot_size = get_data_slot_size(dtype);

    switch (dtype) {
        case x86_64::cpu_data_type::uint_8:
        case x86_64::cpu_data_type::sint_8:
        case x86_64::cpu_data_type::sint_32:
        case x86_64::cpu_data_type::uint_64:
        case x86_64::cpu_data_type::float_32: {
            auto addr64 = addr;
            addr64.setBit(64);
            gen_.pop(addr64);
        } break;
        default: COMPILE_ASSERT(false, "Invalid stack pop addr: " << dtype);
    }

    sf_model_.shrink(slot_size);
    return get_stack_top_rbp_offset();
}

void location_manager::stack_padding(
        const size_t &padding_bytes_needed, const std::string &comment) {
    assert(padding_bytes_needed > 0);
    stack_allocate(padding_bytes_needed);
    for (size_t i = 0; i < padding_bytes_needed; ++i) {
        // The choice of uint_8 is irrelevant. It's just because
        // stack_frame_model currently requires a cpu_data_type for every
        // modeled slot.
        sf_model_.push_anonymous_object(cpu_data_type::uint_8, 1, comment);
    }
}

void location_manager::stack_restore(const size_t &stack_diff_to_restore) {
    gen_.add(gen_.rsp, stack_diff_to_restore);
    sf_model_.shrink(stack_diff_to_restore);
}

size_t location_manager::stack_var_define(x86_64::cpu_data_type cpu_dtype,
        const std::string &name, const std::string &comment) {
    size_t slot_size = get_data_slot_size(cpu_dtype);
    sf_model_.push_named_object(name, cpu_dtype, slot_size, comment);
    return slot_size;
}

size_t location_manager::stack_tensor_define(x86_64::cpu_data_type cpu_dtype,
        size_t num_elem, const std::string &name, const std::string &comment) {
    size_t slot_size = get_tensor_slot_size(cpu_dtype, num_elem);
    sf_model_.push_named_tensor_buffer_object(
            name, cpu_dtype, num_elem, slot_size, comment);
    return slot_size;
}

void location_manager::stack_allocate(size_t slot_size) {
    if (slot_size > 0) { gen_.sub(gen_.rsp, slot_size); }
}

size_t location_manager::get_stack_current_size() {
    return sf_model_.get_size();
}

int64_t location_manager::get_stack_top_rbp_offset() {
    return (sf_model_.get_size() == 0)
            ? 0
            : sf_model_.get_top_slot()->get_rbp_offset();
}

void location_manager::conserve_stack_size() {
    conserved_stack_.push_back(get_stack_current_size());
}

void location_manager::restore_stack_size() {
    assert(!conserved_stack_.empty());
    size_t pre_stack_size = conserved_stack_.back();
    if (const size_t stack_growth_after_conservation
            = get_stack_current_size() - pre_stack_size) {
        stack_restore(stack_growth_after_conservation);
    }
    conserved_stack_.pop_back();
}

//==============================================================================
// Function argument interface
//==============================================================================

void location_manager::handle_func_params(const std::vector<expr> &func_params,
        const x86_64::abi_function_interface &func_iface) {
    assert(func_params.size() == func_iface.param_locs_.size());

    // When this method gets called, the callee has already created its own
    // stack frame by pushing %rbp. So the call stack currently looks like this:
    //
    // |          ......          |
    // |--------------------------| -|
    // | (right-most stack param) |  |
    // |--------------------------|  |
    // | (next stack param)       |  |
    // |--------------------------|  |- stack parameter area
    // |          ......          |  |  (EXISTS ONLY IF ONE OR MORE PARAMETERS
    // |--------------------------|  |  IS ACTUALLY PASSED ON THE STACK.)
    // | (left-most stack param)  |  |
    // |--------------------------| -|
    // | (return address)         |
    // |--------------------------| <-- %rsp when control first reached callee
    // | saved value of %rbp      |
    // |--------------------------| <-- %rsp, %rbp after callee created stack
    // frame

    // For parameters passed via that stack parameter area, we're not going to
    // emit any asm code. We'll just update our book-keeping information so we
    // remember where the callee can find them at runtime if needed.
    //
    // For parameters passed via CPU registers, we'll emi asm that pushes their
    // values onto the stack when spilled by allocator, otherwise we just keep
    // the value in original registers. Assuming that there are n spilled
    // register-passed parameters, the emitted asm will make the stack look like
    // this:
    //
    // |          ......          |
    // |--------------------------| -|
    // | (right-most stack param) |  |
    // |--------------------------|  |
    // | (next stack param)       |  |
    // |--------------------------|  |- stack parameter area
    // |          ......          |  |  (EXISTS ONLY IF ONE OR MORE PARAMETERS
    // |--------------------------|  |  IS ACTUALLY PASSED ON THE STACK.)
    // | (left-most stack param)  |  |
    // |--------------------------| -|
    // | (return address)         |
    // |--------------------------| <-- %rsp when control first reached callee
    // | saved value of %rbp      |
    // |--------------------------| <-- %rbp after callee created stack frame
    // | 1st spilled param value  | -|
    // |--------------------------|  |
    // |          ......          |  |- spilled register parameter area
    // |--------------------------|  | (EXISTS ONLY IF ONE OR MORE REGISTER
    // | nth spilled param value  | -|  PARAMETERS IS ACTUALLY SPILLED.)
    // |--------------------------| <-- %rsp (current)

    /// NOTE! The \c get_rsp_offset() method gives the offset
    /// relative to what %rbp was when control first entered
    /// the callee.
    /// But the code we're generating here is executed after
    /// the callee has created pushed %rbp onto the stack to
    /// create the new stack frame.

    for (size_t i = 0; i < func_params.size(); ++i) {
        const abi_value_location &initial_loc = func_iface.param_locs_[i];
        const expr &ir_expr = func_params[i];
        const std::string &name = get_node_name(ir_expr);

        const cpu_data_type_table::row &r
                = get_cpu_data_type_row(ir_expr->dtype_);
        const cpu_data_type cpu_dtype = r.type_;

        switch (initial_loc.get_type()) {
            case abi_value_location::tag_type::REGISTER: {
                // Get param on reg
                Xbyak::Reg src_reg = initial_loc.get_register();
                COMPILE_ASSERT(src_reg.isREG(64) || src_reg.isXMM(),
                        "Unhandled register kind: " << src_reg.toString());
                // Push reg param to stack if spilled
                if (GET_VIRTUAL_REG(ir_expr).spilled()) {
                    assert(r.local_value_stack_slot_size_ == 8);
                    int64_t rbp_offset = stack_push(src_reg, cpu_dtype);
                    expr_location_map_[ir_expr] = expr_location::make_stack_var(
                            rbp_offset, cpu_dtype);
                } else if (GET_VIRTUAL_REG(ir_expr).allocated()) {
                    allocate_free_reg(ir_expr);
                }
            } break;
            case abi_value_location::tag_type::STACK: {
                assert(r.abi_stack_slot_size_ == 8);
                // Get param on stack
                const size_t saved_rbp_slot_size = 8;
                int64_t rbp_offset
                        = initial_loc.get_rsp_offset() + saved_rbp_slot_size;
                auto param_slot = stack_frame_model::caller_param_slot(name,
                        r.abi_stack_slot_size_, cpu_dtype, rbp_offset,
                        "caller-created parameter slot");
                sf_model_.add_caller_param_slot(param_slot);
                // Get location on stack
                auto location
                        = expr_location::make_stack_var(rbp_offset, cpu_dtype);
                // Load stack param to reg if allocated reg
                // TODO(XXX): optimize when to load
                if (GET_VIRTUAL_REG(ir_expr).spilled()) {
                    expr_location_map_[ir_expr] = location;
                } else if (GET_VIRTUAL_REG(ir_expr).allocated()) {
                    const auto &reg = allocate_free_reg(ir_expr).get_reg();
                    load_location_to_reg(reg, location);
                }
            } break;
            default: assert(!"Unreachable");
        }
    }
}

void location_manager::align_call_stack(
        const x86_64::abi_function_interface &callee_iface) {
    //--------------------------------------------------------------------------
    // The Microsoft/SystemV ABI requires that just prior to the 'call'
    // instruction, ((%rsp mod 16) == 0).
    //
    // We could just use a snippet of asm code to do this at JIT-execution time,
    // but that would be wasteful since we have enough information during
    // codegen.
    //--------------------------------------------------------------------------

    // TODO(xxx): When we add support for data types with stricter alignment
    // requirements (i.e., _m256 or _m512), we'll need to stop treating 16 as
    // a fixed requirement.
    assert(callee_iface.initial_rsp_alignment_ == 16);

    // The Microsoft/SystemV ABI also guarantees that upon entry to any
    // function, ((%rsp mod 16) == 8).
    constexpr size_t initial_rsp_mod16 = 8;

    // When control first entered the *caller* function, we pushed the
    // value of $rbp onto the stack. That's not considered part of the new
    // stack frame, and therefore get_stack_current_size()
    // doesn't include them. But we need to consider them here.
    constexpr size_t saved_rbp_bytes = 8;

    const size_t current_rsp_mod16
            = (initial_rsp_mod16 + saved_rbp_bytes + get_stack_current_size())
            % 16;

    const size_t rsp_mod16_after_params_pushed
            = (current_rsp_mod16 + callee_iface.get_param_area_size()) % 16;

    // pre-call stack alignment needed
    if (rsp_mod16_after_params_pushed != 0) {
        const size_t padding_bytes_needed = 16 - rsp_mod16_after_params_pushed;
        // We're going to pad with 8-byte dummy slots.
        stack_padding(padding_bytes_needed);
    }
}

void location_manager::handle_call_arg(const expr_c &arg, const expr_c &v) {
    if (GET_VIRTUAL_REG(arg).spilled()) {
        stack_push(v);
    } else {
        const auto &call_reg = allocate_free_reg(arg).get_reg();
        load_location_to_reg(call_reg, get_location(v));
    }
}

void location_manager::push_caller_saved(
        const std::vector<expr_c> &caller_saved) {
    for (auto &v : caller_saved) {
        int64_t rbp_offset = stack_push(v);
        local_location_map_[v] = expr_location::make_stack_var(
                rbp_offset, get_cpu_data_type(v->dtype_));
        caller_saved_.push_back(v);
    }
}

void location_manager::pop_caller_saved() {
    while (!caller_saved_.empty()) {
        auto v = caller_saved_.back();
        local_location_map_.erase(v);
        stack_pop(v);
        caller_saved_.pop_back();
    }
}

//==============================================================================
// Codegen operation interface
//==============================================================================

void location_manager::handle_definition(const expr_c &v) {
    if (GET_VIRTUAL_REG(v).allocated()) { allocate_free_reg(v); }
}

void location_manager::handle_spilled_definition(
        const std::vector<expr_c> &defined_spill) {
    size_t slot_size = 0;
    for (auto &v : defined_spill) {
        if (expr_location_map_.find(v) != expr_location_map_.end()) {
            continue;
        } else if (GET_VIRTUAL_REG(v).spilled()) {
            const auto name = get_node_name(v.remove_const());
            const auto data_type = get_cpu_data_type(v->dtype_);
            // v is a var
            slot_size += stack_var_define(
                    data_type, name, "allocate on stack: " + name);
            auto offset = get_stack_top_rbp_offset();
            expr_location_map_[v]
                    = expr_location::make_stack_var(offset, data_type);
        } else if (GET_VIRTUAL_REG(v).buffered()) {
            auto vv = v.static_as<tensor_c>();
            const auto name = get_node_name(vv.remove_const());
            const auto num_elem = get_tensor_static_num_elements(vv);
            const auto elem_dtype = get_cpu_data_type(vv->elem_dtype_);
            assert(num_elem <= 128);
            // v is a tensor
            slot_size += stack_tensor_define(
                    elem_dtype, num_elem, name, "allocate on stack: " + name);
            auto offset = get_stack_top_rbp_offset();
            expr_location_map_[v] = expr_location::make_stack_tensor(offset);
        } else {
            COMPILE_ASSERT(false, "Invalid spilled define: " << v);
        }
    }
    stack_allocate(slot_size);
}

void location_manager::prepare_local_scope(
        const std::vector<expr_c> &local_spill) {
    std::vector<expr_c> local_defined;
    for (auto &v : local_spill) {
        if (GET_VIRTUAL_REG(v).spilled()) {
            local_defined.push_back(v);
        } else {
            int64_t rbp_offset = stack_push(v);
            local_location_map_[v] = expr_location::make_stack_var(
                    rbp_offset, get_cpu_data_type(v->dtype_));
        }
    }
    handle_spilled_definition(local_defined);
}

void location_manager::conclude_local_scope() {
    local_location_map_.clear();
}

void location_manager::emit_callee_prologue(
        const std::set<virt_reg_index_t> &register_usage) {
    // CALL STACK BEFORE:
    //
    // [ zero or more caller-pushed args    ]
    // [ caller-pushed return address       ] <-- %rsp

    gen_.push(regs::rbp);
    gen_.mov(regs::rbp, regs::rsp);
    // NOTE: We intentionally avoid updating the stack-frame model
    // regarding our pushing of %rbp, because we don't consider it
    // to be part of the new stack frame.

    // Get callee saved reg type
    auto callee_save_cpu_type = [&](const virt_reg_type &reg_type) {
        if (reg_type == virt_reg_type::gp_reg) {
            return x86_64::cpu_data_type::uint_64;
        } else if (reg_type == virt_reg_type::fp_reg) {
            // XMM6-XMM15 nonvolatile
            // Upper portions of YMM0-YMM15 and ZMM0-ZMM15 volatile
            // Thus only save 128-bit
            return x86_64::cpu_data_type::float_32_x4;
        } else {
            COMPILE_ASSERT(false, "Invalid callee save type.");
        }
    };
    // Push callee saved reg by type
    auto gen_callee_save = [&](const virt_reg_type &reg_type) {
        const auto &target_callee_save
                = virtual_slots_map_->get_callee_save(reg_type);
        for (auto i : target_callee_save) {
            if (register_usage.find(i) != register_usage.end()) {
                auto reg = virtual_slots_map_->get_reg_physical(i);
                auto loc = expr_location::make_reg(
                        reg, callee_save_cpu_type(reg_type));
                callee_saved_.push_back(loc);
                stack_push(loc);
            }
        }
    };
    // Push all callee saved regs
    gen_callee_save(virt_reg_type::gp_reg);
    gen_callee_save(virt_reg_type::fp_reg);
    // Save current stack
    conserve_stack_size();

    // CALL STACK AFTER:
    //
    // [ zero or more caller-pushed args    ]
    // [ caller-pushed return address       ] <-- %rsp as of function entry
    // [ callee-pushed save of %rbp         ] <-- %rbp
    // [ callee-pushed save reg [0]         ]
    // [ callee-pushed save reg [1]         ]
    // ....
    // [ callee-pushed save reg [n]         ] <-- %rsp
}

void location_manager::emit_callee_epilogue() {
    // CALL STACK BEFORE:
    //
    // [ zero or more caller-pushed args    ]
    // [ caller-pushed return address       ] <-- %rsp as of function entry
    // [ callee-pushed save of %rbp         ] <-- %rbp
    // [ callee-pushed save reg [0]         ]
    // [ callee-pushed save reg [1]         ]
    // ....
    // [ callee-pushed save reg [n]         ]
    // [ (any additional stack allocations) ] <-- %rsp

    // Shrink the stack to remove all allocations that occurred after saved
    // registers.
    restore_stack_size();

    // Pop all callee saved regs
    while (!callee_saved_.empty()) {
        const auto &loc = callee_saved_.back();
        stack_pop(loc);
        callee_saved_.pop_back();
    }

    // NOTE: We don't consider the saved %rbp value to be part of the stack
    // frame itself, so don't update the stack-frame model when we pop %rbp.
    gen_.pop(regs::rbp);

    // CALL STACK AFTER:
    //
    // [ zero or more caller-pushed args    ]
    // [ caller-pushed return address       ] <-- %rsp = %rsp as of function
}

void location_manager::expire(stmt_index_t current_index) {
    std::vector<expr_c> to_be_expired;

    // Get linvness ended expr
    for (auto &loc_kv : expr_location_map_) {
        auto &v = loc_kv.first;
        auto &loc = loc_kv.second;
        auto &last_ll = GET_LIVE_RANGE(v);
        if (last_ll.end_ <= current_index) { to_be_expired.push_back(v); }
    }
    // release location
    for (auto &v : to_be_expired) {
        // SC_MODULE_FATAL << "to_be_expired: " << v;
        expr_location_map_.erase(v);
    }
}

void location_manager::clear() {
    caller_saved_.clear();
    callee_saved_.clear();
    conserved_stack_.clear();
    local_location_map_.clear();
    expr_location_map_.clear();
    simd_constant_map_.clear();
    simd_constant_vec_.clear();
}

//==============================================================================
// Codegen Operand interface
//==============================================================================
operand location_manager::get_operand(const expr_location &location) {
    switch (location.get_type()) {
        case expr_location::type::imm: {
            return operand(operand::type::imm, location.get_op_ptr());
        } break;
        case expr_location::type::reg: {
            return operand(operand::type::reg, location.get_op_ptr());
        } break;
        case expr_location::type::stack_var: {
            auto addr = get_offset_address(
                    location.get_stack_var(), location.get_data_type());
            return operand(operand::type::addr, wrap_op_ptr(addr));
        } break;
        case expr_location::type::stack_tensor: {
            auto addr = get_offset_address(
                    location.get_stack_tensor(), location.get_data_type());
            return operand(operand::type::addr, wrap_op_ptr(addr));
        } break;
        case expr_location::type::simd_constant: {
            return operand(operand::type::addr, location.get_op_ptr());
        } break;
        default: {
            COMPILE_ASSERT(false, "Invalid: get_operand: " << location);
        }
    }
    return operand();
}

operand location_manager::get_operand(const expr_c &v) {
    if (v.isa<indexing_c>()) {
        auto vv = v.static_as<indexing_c>();
        return get_operand_indexing(vv);
    } else {
        return get_operand(get_location(v));
    }
}

operand location_manager::get_operand_indexing(const indexing_c &v) {
    // +----------------+-------------------+----------------------------+
    // | ptr[idx] addr  | ptr var: V        | ptr stack tensor: %rbp + o |
    // +----------------+-------------------+----------------------------+
    // | idx var: I     | REG(V) + s*REG(I) | %rbp + o + s*REG(I)        |
    // +----------------+-------------------+----------------------------+
    // | idx const: i   | REG(V) + s*i      | %rbp + o + s*i             |
    // +----------------+-------------------+----------------------------+
    assert(v->idx_.size() == 1);
    // Get the index value
    expr_c idx = v->idx_.back();
    // Get the tensor buffer's base address
    expr_c ptr = v->ptr_;
    assert(ptr.defined());
    // get ptr and idx locations
    auto ptr_location = get_location(ptr);
    auto idx_location = get_location(idx);
    auto ptr_loc_type = ptr_location.get_type();
    auto idx_loc_type = idx_location.get_type();
    COMPILE_ASSERT(ptr_loc_type == expr_location::type::reg
                    || ptr_loc_type == expr_location::type::stack_tensor,
            "Invalid base address location: " << v);
    COMPILE_ASSERT(idx_loc_type == expr_location::type::imm
                    || idx_loc_type == expr_location::type::reg,
            "Invalid index value location: " << v);
    // Get RegExp for ptr
    Xbyak::RegExp addr_exp;
    if (ptr_loc_type == expr_location::type::stack_tensor) {
        // addr_exp = %rbp + o
        addr_exp = get_rbp_offset(ptr_location.get_stack_tensor());
    } else {
        // addr_exp = REG(V)
        addr_exp = Xbyak::RegExp(ptr_location.get_reg());
    }
    // Get scale for indexing
    auto elem_type = ptr->dtype_.get_pointer_element();
    auto scale = get_data_type_size(get_cpu_data_type(elem_type));
    // Get RegExp for idx
    if (idx_loc_type == expr_location::type::imm) {
        // addr_exp += s*i
        addr_exp = addr_exp + idx_location.get_imm() * scale;
    } else {
        // addr_exp += s*REG(I)
        addr_exp = addr_exp + to_reg64(idx_location.get_reg()) * scale;
    }
    // Get address frame
    return operand(get_address(addr_exp, get_cpu_data_type(v->dtype_)));
}

operand location_manager::get_operand_sib(
        const expr_c &base, const expr_c &indx, const expr_c &disp) {
    auto loc_base = get_location(base);
    auto loc_indx = get_location(indx);
    auto loc_disp = get_location(disp);

    COMPILE_ASSERT(loc_base.get_type() == expr_location::type::reg
                    && loc_indx.get_type() == expr_location::type::reg
                    && loc_disp.get_type() == expr_location::type::imm,
            "Invalid sib operand type: " << loc_base << ", " << loc_indx << ", "
                                         << loc_disp);

    return operand(gen_.ptr[loc_base.get_reg() + loc_indx.get_reg()
            + loc_disp.get_imm()]);
}

//==============================================================================
// MISC. interface
//==============================================================================

bool location_manager::is_stack_tensor(const expr_c &v) {
    return get_location(v).get_type() == expr_location::type::stack_tensor;
}

size_t location_manager::get_data_type_size(x86_64::cpu_data_type data_type) {
    const size_t data_size
            = get_cpu_data_types().lookup(data_type).size_in_bytes_;
    return data_size;
}

size_t location_manager::get_data_slot_size(x86_64::cpu_data_type data_type) {
    const size_t slot_size = get_cpu_data_types()
                                     .lookup(data_type)
                                     .local_value_stack_slot_size_;
    return slot_size;
}

size_t location_manager::get_tensor_slot_size(
        x86_64::cpu_data_type data_type, const size_t &num_elem) {
    COMPILE_ASSERT(num_elem > 0, "cannot allocate zero-element tensors");

    // TODO(xxx): We're supposed to ensure that *every* element of the tensor
    // buffer will meet the "natural" alignment standard. We might want an
    // assert to verify that, instead of just assuming that no intra-element
    // padding is needed to get that outcome.

    size_t buffer_size = get_data_type_size(data_type) * num_elem;

    if (const size_t excess = buffer_size % 8) { buffer_size += (8 - excess); }

    return buffer_size;
}

size_t location_manager::get_tensor_static_num_elements(const tensor_c &v) {
    COMPILE_ASSERT(v->dims_.size() == 1, "Tensors must be one-dimensional");

    const constant_c dim0_node = v->dims_[0].dyn_as<constant_c>();
    COMPILE_ASSERT(dim0_node.defined(), "Unexpected dims_[0] node type: " << v);

    return get_const_as_int(dim0_node);
}

size_t location_manager::get_conserved_stack_size() const {
    return conserved_stack_.size();
}

const Xbyak::AddressFrame *location_manager::get_address_frame(
        const cpu_data_type cpu_dtype) {
    switch (cpu_dtype) {
        // integer 8-bit/ 1-byte
        case cpu_data_type::uint_8: return &(gen_.byte);
        case cpu_data_type::sint_8: return &(gen_.byte);
        // integer 16-bit/ 2-byte
        case cpu_data_type::uint_16: return &(gen_.word);
        // fp16 16bit / 2-byte
        case cpu_data_type::float_16: return &(gen_.word);
        // integer 32-bit/ 4-byte
        case cpu_data_type::uint_32: return &(gen_.dword);
        case cpu_data_type::sint_32: return &(gen_.dword);
        // integer 64-bit/ 8-byte
        case cpu_data_type::uint_64: return &(gen_.qword);
        // simd 32-bit/ 4-byte
        case cpu_data_type::float_32: return &(gen_.dword);
        // simd 64-bit/ 8-byte
        case cpu_data_type::uint_8_x8: return &(gen_.qword);
        case cpu_data_type::sint_8_x8: return &(gen_.qword);
        case cpu_data_type::uint_16_x4: return &(gen_.qword);
        case cpu_data_type::uint_32_x2: return &(gen_.qword);
        case cpu_data_type::sint_32_x2: return &(gen_.qword);
        case cpu_data_type::float_16_x4: return &(gen_.qword);
        case cpu_data_type::float_32_x2: return &(gen_.qword);
        // simd 128-bit/ 16-byte
        case cpu_data_type::uint_8_x16: return &(gen_.xword);
        case cpu_data_type::sint_8_x16: return &(gen_.xword);
        case cpu_data_type::uint_16_x8: return &(gen_.xword);
        case cpu_data_type::uint_32_x4: return &(gen_.xword);
        case cpu_data_type::sint_32_x4: return &(gen_.xword);
        case cpu_data_type::uint_64_x2: return &(gen_.xword);
        case cpu_data_type::float_16_x8: return &(gen_.xword);
        case cpu_data_type::float_32_x4: return &(gen_.xword);
        // simd 256-bit/ 32-byte
        case cpu_data_type::uint_8_x32: return &(gen_.yword);
        case cpu_data_type::sint_8_x32: return &(gen_.yword);
        case cpu_data_type::uint_16_x16: return &(gen_.yword);
        case cpu_data_type::uint_32_x8: return &(gen_.yword);
        case cpu_data_type::sint_32_x8: return &(gen_.yword);
        case cpu_data_type::uint_64_x4: return &(gen_.yword);
        case cpu_data_type::float_16_x16: return &(gen_.yword);
        case cpu_data_type::float_32_x8: return &(gen_.yword);
        // simd 512-bit/ 64-byte
        case cpu_data_type::uint_8_x64: return &(gen_.zword);
        case cpu_data_type::sint_8_x64: return &(gen_.zword);
        case cpu_data_type::uint_16_x32: return &(gen_.zword);
        case cpu_data_type::uint_32_x16: return &(gen_.zword);
        case cpu_data_type::sint_32_x16: return &(gen_.zword);
        case cpu_data_type::uint_64_x8: return &(gen_.zword);
        case cpu_data_type::float_16_x32: return &(gen_.zword);
        case cpu_data_type::float_32_x16: return &(gen_.zword);
        // avx512 mask
        case cpu_data_type::mask_x4: return &(gen_.byte);
        case cpu_data_type::mask_x8: return &(gen_.byte);
        case cpu_data_type::mask_x16: return &(gen_.word);
        case cpu_data_type::mask_x32: return &(gen_.dword);
        case cpu_data_type::mask_x64: return &(gen_.qword);
        // not supported
        case cpu_data_type::void_t: {
            COMPILE_ASSERT(false, "Invalid address_frame: " << cpu_dtype);
        } break;
    }
    return nullptr;
}

const content_hash_map<expr_c, Xbyak::Label> &
location_manager::encode_simd_constant() {
    std::function<int8_t(union_val)> select_s8
            = [](union_val u) -> int8_t { return (int8_t)u.s64; };
    std::function<uint8_t(union_val)> select_u8
            = [](union_val u) -> uint8_t { return (uint8_t)u.u64; };
    std::function<uint16_t(union_val)> select_u16
            = [](union_val u) -> uint16_t { return (uint16_t)u.u64; };
    std::function<uint32_t(union_val)> select_u32
            = [](union_val u) -> uint32_t { return (uint32_t)u.u64; };
    std::function<int32_t(union_val)> select_s32
            = [](union_val u) -> int32_t { return (int32_t)u.s64; };
    std::function<float(union_val)> select_f32
            = [](union_val u) -> float { return (float)u.f32; };
    std::function<uint16_t(union_val)> select_bf16
            = [](union_val u) -> uint16_t { return bf16_t(u.f32).storage_; };
    std::function<uint16_t(union_val)> select_f16
            = [](union_val u) -> uint16_t { return fp16_t(u.f32).storage_; };
    uint8_t buffer[64];
    for (auto &c : simd_constant_vec_) {
        auto simd_it = simd_constant_map_.find(c);
        assert(simd_it != simd_constant_map_.end());
        auto v = simd_it->first.static_as<constant_c>();
        auto lanes = v->dtype_.lanes_;
        auto type_code = v->dtype_.type_code_;
        auto size = get_data_type_size(get_cpu_data_type(v->dtype_));
        assert(size <= 64);
        switch (type_code) {
            case sc_data_etype::BF16: {
                encode_simd_to_buffer(
                        (uint16_t *)buffer, lanes, v->value_, select_bf16);
            } break;
            case sc_data_etype::U8: {
                encode_simd_to_buffer(
                        (uint8_t *)buffer, lanes, v->value_, select_u8);
            } break;
            case sc_data_etype::S8: {
                encode_simd_to_buffer(
                        (int8_t *)buffer, lanes, v->value_, select_s8);
            } break;
            case sc_data_etype::U16: {
                encode_simd_to_buffer(
                        (uint16_t *)buffer, lanes, v->value_, select_u16);
            } break;
            case sc_data_etype::U32: {
                encode_simd_to_buffer(
                        (uint32_t *)buffer, lanes, v->value_, select_u32);
            } break;
            case sc_data_etype::S32: {
                encode_simd_to_buffer(
                        (int32_t *)buffer, lanes, v->value_, select_s32);
            } break;
            case sc_data_etype::F32: {
                encode_simd_to_buffer(
                        (float *)buffer, lanes, v->value_, select_f32);
            } break;
            case sc_data_etype::F16: {
                encode_simd_to_buffer(
                        (uint16_t *)buffer, lanes, v->value_, select_f16);
            } break;
            default:
                COMPILE_ASSERT(false, "Can't encode constant: " << v->dtype_);
        }
        gen_.align(16);
        gen_.L(simd_it->second);
        gen_.db(buffer, size);
    }
    return simd_constant_map_;
}

template <typename T>
void location_manager::encode_simd_to_buffer(T *buffer, uint32_t lanes,
        const std::vector<union_val> &value,
        std::function<T(union_val)> select_val) {
    if (value.size() == lanes) {
        for (size_t i = 0; i < lanes; i++) {
            T val = select_val(value[i]);
            buffer[i] = val;
        }
    } else if (value.size() == 1) {
        T val = select_val(value[0]);
        for (size_t i = 0; i < lanes; i++) {
            buffer[i] = val;
        }
    } else {
        COMPILE_ASSERT(false, "Encode constant error");
    }
}

template void location_manager::encode_simd_to_buffer<int8_t>( //
        int8_t *, uint32_t, const std::vector<union_val> &,
        std::function<int8_t(union_val)>);
template void location_manager::encode_simd_to_buffer<uint8_t>( //
        uint8_t *, uint32_t, const std::vector<union_val> &,
        std::function<uint8_t(union_val)>);
template void location_manager::encode_simd_to_buffer<uint16_t>( //
        uint16_t *, uint32_t, const std::vector<union_val> &,
        std::function<uint16_t(union_val)>);
template void location_manager::encode_simd_to_buffer<uint32_t>( //
        uint32_t *, uint32_t, const std::vector<union_val> &,
        std::function<uint32_t(union_val)>);
template void location_manager::encode_simd_to_buffer<int32_t>( //
        int32_t *, uint32_t, const std::vector<union_val> &,
        std::function<int32_t(union_val)>);
template void location_manager::encode_simd_to_buffer<float>( //
        float *, uint32_t, const std::vector<union_val> &,
        std::function<float(union_val)>);

//==============================================================================
// Location management
//==============================================================================

expr_location location_manager::get_location(const expr_c &v) {
    if (local_location_map_.find(v) != local_location_map_.end()) {
        return local_location_map_[v];
    } else if (expr_location_map_.find(v) != expr_location_map_.end()) {
        return expr_location_map_[v];
    } else if (GET_VIRTUAL_REG(v).allocated()) {
        return allocate_free_reg(v);
    } else if (v.isa<constant>()) {
        return get_location(v.static_as<constant_c>());
    } else {
        COMPILE_ASSERT(false,
                "expr not found, v=" << v << "={" << GET_VIRTUAL_REG(v) << "}");
    }
    return expr_location();
}

expr_location location_manager::get_location(const constant_c &c) {
    auto constant_clear_nan = [](const constant_c &c) -> constant_c {
        // if the constant to encode is f32 NaN, represent it as u32
        // so content_hash can regard same NaN binary as equal
        // e.g. used in fabs and log
        if (c->dtype_.is_etype(sc_data_etype::F32) && c->value_.size() == 1
                && std::isnan(c->value_[0].f32)) {
            auto new_c = c->remake();
            new_c->dtype_.type_code_ = sc_data_etype::U32;
            new_c->temp_data() = GET_EXPR_DATA(c);
            return new_c.static_as<constant_c>();
        }
        return c;
    };
    auto v = constant_clear_nan(c);
    auto data_type = get_cpu_data_type(v->dtype_);
    if (GET_VIRTUAL_REG(v).spilled()) {
        // Save to encode value inside code later
        auto simd_iter = simd_constant_map_.find(v);
        if (simd_iter == simd_constant_map_.end()) {
            simd_iter = simd_constant_map_.insert(simd_iter,
                    std::make_pair<expr_c, Xbyak::Label>(v, Xbyak::Label()));
            simd_constant_vec_.emplace_back(v);
        }
        // Add to location map
        auto addr = get_offset_address(simd_iter->second, data_type);
        auto loc_iter = expr_location_map_.find(v);
        assert(loc_iter == expr_location_map_.end());
        loc_iter = expr_location_map_.insert(loc_iter,
                std::make_pair<expr_c, expr_location>(
                        v, expr_location::make_simd_constant(addr, data_type)));
        return loc_iter->second;
    } else {
        // Get immediate value
        if (v->dtype_ == datatypes::bf16) {
            // bf16x1 treated as uint_16, thus imm needs conversion
            return expr_location::make_imm(
                    bf16_t(v->value_[0].f32).storage_, data_type);
        } else {
            return expr_location::make_imm(v->value_[0].s64, data_type);
        }
    }
}

void location_manager::load_location_to_reg(
        const Xbyak::Reg &reg, const expr_location &location) {
    auto data_type = location.get_data_type();
    auto op = get_operand(location);
    switch (location.get_type()) {
        case expr_location::type::imm: {
            const auto imm = op.get_imm();
            load_imm_value_to_reg(reg, imm, data_type);
        } break;
        case expr_location::type::reg: {
            const auto &src = op.get_reg();
            load_reg_value_to_reg(reg, src, data_type);
        } break;
        case expr_location::type::simd_constant: {
            const auto &addr = op.get_addr();
            load_mem_value_to_reg(reg, addr, data_type);
        } break;
        case expr_location::type::stack_var: {
            const auto &addr = op.get_addr();
            load_mem_value_to_reg(reg, addr, data_type);
        } break;
        case expr_location::type::stack_tensor: {
            const auto &addr = op.get_addr();
            load_mem_addr_to_reg(reg, addr, data_type);
        } break;
        default: {
            COMPILE_ASSERT(false, "Invalid: load_location_to_reg");
        }
    }
}

void location_manager::load_imm_value_to_reg(const Xbyak::Reg &reg,
        const uint64_t &imm, x86_64::cpu_data_type data_type) {
    switch (data_type) {
        case cpu_data_type::uint_8:
        case cpu_data_type::sint_8: {
            gen_.mov(to_reg8(reg), imm);
        } break;
        case cpu_data_type::sint_32: {
            gen_.mov(to_reg32(reg), imm);
        } break;
        case cpu_data_type::uint_64: {
            gen_.mov(to_reg64(reg), imm);
        } break;
        default: {
            COMPILE_ASSERT(false, "Invalid: load_imm_value_to_reg");
        }
    }
}

void location_manager::load_reg_value_to_reg(const Xbyak::Reg &reg,
        const Xbyak::Reg &src, x86_64::cpu_data_type data_type) {
    if (operand(reg) == operand(src)) { return; }
    switch (data_type) {
        case cpu_data_type::uint_8:
        case cpu_data_type::sint_8:
        case cpu_data_type::sint_32:
        case cpu_data_type::uint_16:
        case cpu_data_type::uint_32:
        case cpu_data_type::uint_64: {
            gen_.mov(to_reg64(reg), to_reg64(src));
        } break;
        case cpu_data_type::float_32: {
            gen_.vmovss(to_xmm(reg), to_xmm(src));
        } break;
        default: {
            COMPILE_ASSERT(
                    false, "Invalid: load_reg_value_to_reg " << data_type);
        }
    }
}

void location_manager::load_mem_value_to_reg(const Xbyak::Reg &reg,
        const Xbyak::Address &addr, x86_64::cpu_data_type data_type) {
    switch (data_type) {
        case cpu_data_type::uint_8:
        case cpu_data_type::sint_8: {
            gen_.mov(to_reg8(reg), addr);
        } break;
        case cpu_data_type::sint_32: {
            gen_.mov(to_reg32(reg), addr);
        } break;
        case cpu_data_type::uint_64: {
            gen_.mov(to_reg64(reg), addr);
        } break;
        case cpu_data_type::float_32: {
            gen_.vmovss(to_xmm(reg), addr);
        } break;
        case cpu_data_type::uint_8_x16:
        case cpu_data_type::sint_8_x16: {
            gen_.vmovups(to_xmm(reg), addr);
        } break;
        case cpu_data_type::sint_32_x16:
        case cpu_data_type::float_32_x16: {
            gen_.vmovups(to_zmm(reg), addr);
        } break;
        default: {
            COMPILE_ASSERT(false, "Invalid: load_mem_value_to_reg");
        }
    }
}

void location_manager::load_mem_addr_to_reg(const Xbyak::Reg &reg,
        const Xbyak::Address &addr, x86_64::cpu_data_type data_type) {
    switch (data_type) {
        case cpu_data_type::uint_64: {
            gen_.lea(to_reg64(reg), addr);
        } break;
        default: {
            COMPILE_ASSERT(false, "Invalid: load_mem_addr_to_reg");
        }
    }
}

Xbyak::RegExp location_manager::get_rbp_offset(const int64_t &offset) {
    return gen_.rbp + offset;
}

Xbyak::RegRip location_manager::get_rip_offset(const Xbyak::Label &label) {
    return gen_.rip + label;
}

Xbyak::Address location_manager::get_address(
        const Xbyak::RegExp &exp, x86_64::cpu_data_type cpu_dtype) {
    const auto xaf = get_address_frame(cpu_dtype);
    return (*xaf)[exp];
}

Xbyak::Address location_manager::get_address(
        const Xbyak::RegRip &rxp, x86_64::cpu_data_type cpu_dtype) {
    const auto xaf = get_address_frame(cpu_dtype);
    return (*xaf)[rxp];
}

Xbyak::Address location_manager::get_offset_address(
        const int64_t &offset, x86_64::cpu_data_type cpu_dtype) {
    return get_address(get_rbp_offset(offset), cpu_dtype);
}

Xbyak::Address location_manager::get_offset_address(
        const Xbyak::Label &label, x86_64::cpu_data_type cpu_dtype) {
    return get_address(get_rip_offset(label), cpu_dtype);
}

//==============================================================================
// Register management
//==============================================================================

expr_location location_manager::allocate_free_reg(const expr_c &v) {
    auto reg_loc = convert_virtual_reg(v);
    expr_location_map_[v] = reg_loc;
    return reg_loc;
}

expr_location location_manager::convert_virtual_reg(const expr_c &v) {
    const auto cpu_dtype = get_cpu_data_type(v->dtype_);
    const auto index = GET_VIRTUAL_REG(v).index_;
    COMPILE_ASSERT(
            index != virt_reg_const::invalid, "convert_virtual_reg failed");
    auto reg = virtual_slots_map_->get_reg_physical(index);

    if (v->dtype_.is_tile()) {
        // skip get_cpu_data_type for tmm
        return expr_location::make_reg(to_tmm(reg), cpu_dtype);
    }
    switch (cpu_dtype) {
        // integer 8-bit/ 1-byte
        case cpu_data_type::uint_8:
        case cpu_data_type::sint_8: {
            return expr_location::make_reg(to_reg8(reg), cpu_dtype);
        }
        // integer 16-bit/ 2-byte
        case cpu_data_type::uint_16: {
            return expr_location::make_reg(to_reg16(reg), cpu_dtype);
        }
        // integer 32-bit/ 4-byte
        case cpu_data_type::uint_32:
        case cpu_data_type::sint_32: {
            return expr_location::make_reg(to_reg32(reg), cpu_dtype);
        }
        // integer 64-bit/ 8-byte
        case cpu_data_type::uint_64: {
            return expr_location::make_reg(to_reg64(reg), cpu_dtype);
        }
        // simd 32-bit/ 4-byte (fp16 16bit/2-byte)
        case cpu_data_type::float_16:
        case cpu_data_type::float_32: {
            return expr_location::make_reg(to_xmm(reg), cpu_dtype);
        }
        // simd 64-bit/ 8-byte
        case cpu_data_type::uint_8_x8:
        case cpu_data_type::sint_8_x8:
        case cpu_data_type::uint_16_x4:
        case cpu_data_type::uint_32_x2:
        case cpu_data_type::sint_32_x2:
        case cpu_data_type::float_16_x4:
        case cpu_data_type::float_32_x2: {
            return expr_location::make_reg(to_xmm(reg), cpu_dtype);
        }
        // simd 128-bit/ 16-byte
        case cpu_data_type::uint_8_x16:
        case cpu_data_type::sint_8_x16:
        case cpu_data_type::uint_16_x8:
        case cpu_data_type::uint_32_x4:
        case cpu_data_type::sint_32_x4:
        case cpu_data_type::uint_64_x2:
        case cpu_data_type::float_16_x8:
        case cpu_data_type::float_32_x4: {
            return expr_location::make_reg(to_xmm(reg), cpu_dtype);
        }
        // simd 256-bit/ 32-byte
        case cpu_data_type::uint_8_x32:
        case cpu_data_type::sint_8_x32:
        case cpu_data_type::uint_16_x16:
        case cpu_data_type::uint_32_x8:
        case cpu_data_type::sint_32_x8:
        case cpu_data_type::uint_64_x4:
        case cpu_data_type::float_16_x16:
        case cpu_data_type::float_32_x8: {
            return expr_location::make_reg(to_ymm(reg), cpu_dtype);
        }
        // simd 512-bit/ 64-byte
        case cpu_data_type::uint_8_x64:
        case cpu_data_type::sint_8_x64:
        case cpu_data_type::uint_16_x32:
        case cpu_data_type::uint_32_x16:
        case cpu_data_type::sint_32_x16:
        case cpu_data_type::uint_64_x8:
        case cpu_data_type::float_16_x32:
        case cpu_data_type::float_32_x16: {
            return expr_location::make_reg(to_zmm(reg), cpu_dtype);
        }
        // simd mask
        case cpu_data_type::mask_x4:
        case cpu_data_type::mask_x8:
        case cpu_data_type::mask_x16:
        case cpu_data_type::mask_x32:
        case cpu_data_type::mask_x64: {
            if (cpu_flags_.fAVX512F) {
                return expr_location::make_reg(to_mask(reg), cpu_dtype);
            } else {
                return expr_location::make_reg(to_reg64(reg), cpu_dtype);
            }
        }
        // not supported
        case cpu_data_type::void_t: {
            COMPILE_ASSERT(false, "Invalid virtual_reg cpu dtype.");
        } break;
    }
    return expr_location();
}

} // namespace xbyak
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
