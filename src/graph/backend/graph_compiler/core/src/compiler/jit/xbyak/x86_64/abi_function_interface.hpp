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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_JIT_XBYAK_X86_64_ABI_FUNCTION_INTERFACE_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_JIT_XBYAK_X86_64_ABI_FUNCTION_INTERFACE_HPP

#include <compiler/jit/xbyak/configured_xbyak.hpp>

#include <memory>
#include <vector>

#include <compiler/ir/sc_expr.hpp>
#include <compiler/jit/xbyak/x86_64/abi_value_location.hpp>
#include <compiler/jit/xbyak/x86_64/native_types.hpp>
#include <compiler/jit/xbyak/x86_64/target_profile.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace xbyak {
namespace x86_64 {
/// For a particular combination of (ABI, function signature), this indicates
/// where various values related to a function call are to be placed.
/// Contains information useful for both the caller and callee.
///
/// Limitations:
///    - No way to indicate that Boolean-typed values must obey certain
///      bit-pattern rules that are tighter than what C/C++ allow.
struct abi_function_interface {
    /// Ptr for abi_function_interface
    typedef std::shared_ptr<abi_function_interface> ptr;

    /// Indicates where each parameter value can be found upon entry to the
    /// callee. None of the vector elements will have tag_type::NONE.
    std::vector<abi_value_location> param_locs_;

    /// Where the callee is supposed to store the return value.
    /// If the callee return type is \c void, this has tag_type::NONE.
    abi_value_location return_val_loc_;

    /// Indicates the %rsp's alignment requirement that must be satisfied
    /// before any stack-based parameters are pushed (or if there are no
    /// stack-based parameters for this call, then before the call instruction
    /// is issued).
    ///
    /// I.e., the caller is responsible for ensuring that
    /// (%rsp modulo initial_rsp_alignment_ == 0) at the stated program point.
    size_t initial_rsp_alignment_;

    /// A convenience method. Returns the indices into \c param_locs_
    /// for the stack-based parameters. The indices are presented in
    /// descending order (i.e., right-to-left order from the function
    /// parameter list).
    std::vector<size_t> get_stack_params_descending_idx() const;

    std::vector<size_t> get_register_params_ascending_idx() const;

    /// Returns the number of stack bytes that the stack-located parameters
    /// will require.
    ///
    /// Does *not* include / and initial padding need to meet the psABI's
    /// requirement for / %rsp alignment when control enters the callee.
    size_t get_param_area_size() const;

    static abi_function_interface::ptr make_interface(
            const target_profile_t &profile,
            const std::vector<sc_data_type_t> &param_types,
            sc_data_type_t ret_type);
    static abi_function_interface::ptr make_interface(
            const target_profile_t &profile, const std::vector<expr> &params,
            sc_data_type_t ret_type);
};

} // namespace x86_64
} // namespace xbyak
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
