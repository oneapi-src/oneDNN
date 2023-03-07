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

#include <algorithm>
#include <memory>

#include <compiler/jit/xbyak/x86_64/abi_common.hpp>
#include <compiler/jit/xbyak/x86_64/abi_function_interface.hpp>
#include <compiler/jit/xbyak/x86_64/type_mapping.hpp>
#include <util/utils.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace xbyak {
namespace x86_64 {

/// See psABI 1.0 section 3.2.3
static abi_function_interface compute_iface_system_v(
        const target_profile_t &profile,
        const std::vector<cpu_data_type> &param_types,
        const cpu_data_type ret_cpu_dtype) {
    // Our algorithm can have a simpler structure than the psABI's,
    // because we support only a small subset of the psABI's data types...

    abi_function_interface iface;
    iface.param_locs_.resize(param_types.size());

    // The minimum stack alignment permitted by the psABI.
    iface.initial_rsp_alignment_ = 16;

    size_t next_arg_gp_idx = 0;
    size_t next_arg_xmm_idx = 0;

    // The elements of 'param_types' that will need to be stored on the
    // stack, in ascending-index order.
    std::vector<size_t> stack_param_indices;

    // Idenify the parameters that will be transfered via registers vs. the
    // stack. For register parameters, assign the register immediately.
    for (size_t param_idx = 0; param_idx < param_types.size(); ++param_idx) {
        // The "Classification" step from psABI section 3.2.3...
        const cpu_data_type_table::row &r
                = get_cpu_data_types().lookup(param_types[param_idx]);

        abi_value_location &loc = iface.param_locs_[param_idx];

        // The "Passing" step from psABI section 3.2.3...
        switch (r.abi_initial_val_kind_) {
            case abi_value_kind::INTEGER:
                if (next_arg_gp_idx < profile.func_arg_gp_regs_.size()) {
                    const Xbyak::Reg64 reg = to_reg64(
                            profile.func_arg_gp_regs_[next_arg_gp_idx++]);
                    loc.set_to_register(reg);
                } else {
                    stack_param_indices.push_back(param_idx);
                }
                break;

            case abi_value_kind::SSE:
                if (next_arg_xmm_idx < profile.func_arg_xmm_regs_.size()) {
                    const Xbyak::Xmm reg = to_xmm(
                            profile.func_arg_xmm_regs_[next_arg_xmm_idx++]);
                    loc.set_to_register(reg);
                } else {
                    stack_param_indices.push_back(param_idx);
                }
                break;

            case abi_value_kind::SSEUPx15_SSE:
                COMPILE_ASSERT(false,
                        "Not yet supported: callers/callees with parameter "
                        "type f32x16");
                break;
        }
    }

    // The psABI specifies that stack-based parameters are pushed to the call
    // stack in right-to-left parameter-list order.
    //
    // When control is first transfered to the callee, the call stack looks
    // like this:
    //
    // |          ......          |
    // |--------------------------| -|
    // | (right-most stack param) |  |
    // |--------------------------|  |
    // | (next stack param)       |  |
    // |--------------------------|  |- stack parameter area
    // |          ......          |  |
    // |--------------------------|  |
    // | (left-most stack param)  |  |
    // |--------------------------| -|
    // | (return address)         |
    // |--------------------------| <-- %rsp
    //
    // When we specify a parameter's location on the stack, it's relative to
    // the value of %rsp as shown in this diagram. So the address of the
    // last-pushed stack parameter is always %rsp + 8.
    size_t next_param_rsp_offset = 8;

    // Note the iteration order. We're dealing with the stack-based indices
    // in left-to-right order, which means we're starting with LAST-PUSHED
    // parameter.
    for (const size_t param_idx : stack_param_indices) {
        const cpu_data_type_table::row &r
                = get_cpu_data_types().lookup(param_types[param_idx]);

        abi_value_location &loc = iface.param_locs_[param_idx];

        // Pre-call stack alignment is determined by whichever argument has the
        // strictest alignment requrirements.
        iface.initial_rsp_alignment_ = std::max(
                iface.initial_rsp_alignment_, r.abi_precall_stack_alignment_);

        loc.set_to_rsp_offset(next_param_rsp_offset);
        next_param_rsp_offset += r.abi_stack_slot_size_;
    }

    if (ret_cpu_dtype != cpu_data_type::void_t) {
        const auto &r = get_cpu_data_types().lookup(ret_cpu_dtype);

        switch (r.abi_initial_val_kind_) {
            case abi_value_kind::INTEGER:
                iface.return_val_loc_.set_to_register(
                        profile.func_return_gp_reg_);
                break;

            case abi_value_kind::SSE:
                iface.return_val_loc_.set_to_register(
                        profile.func_return_xmm_reg_);
                break;

            case abi_value_kind::SSEUPx15_SSE:
                COMPILE_ASSERT(false,
                        "Not yet supported: callers/callees with return type "
                        "f32x16");
                break;
        }
    }

    return iface;
}

static abi_function_interface compute_iface_microsoft(
        const target_profile_t &profile,
        const std::vector<cpu_data_type> &param_types,
        const cpu_data_type ret_cpu_dtype) {
    // Our algorithm can have a simpler structure than the msABI's,
    // because we support only a small subset of the msABI's data types...

    abi_function_interface iface;
    iface.param_locs_.resize(param_types.size());

    // The minimum stack alignment permitted by the msABI.
    iface.initial_rsp_alignment_ = 16;

    // The elements of 'param_types' that will need to be stored on the
    // stack, in ascending-index order.
    std::vector<size_t> stack_param_indices;

    // Idenify the parameters that will be transfered via registers vs. the
    // stack. For register parameters, assign the register immediately.
    for (size_t param_idx = 0; param_idx < param_types.size(); ++param_idx) {
        const cpu_data_type_table::row &r
                = get_cpu_data_types().lookup(param_types[param_idx]);

        abi_value_location &loc = iface.param_locs_[param_idx];

        switch (r.abi_initial_val_kind_) {
            case abi_value_kind::INTEGER:
                if (param_idx < profile.func_arg_gp_regs_.size()) {
                    const Xbyak::Reg64 reg
                            = to_reg64(profile.func_arg_gp_regs_[param_idx]);
                    loc.set_to_register(reg);
                } else {
                    stack_param_indices.push_back(param_idx);
                }
                break;

            case abi_value_kind::SSE:
                if (param_idx < profile.func_arg_xmm_regs_.size()) {
                    const Xbyak::Xmm reg
                            = to_xmm(profile.func_arg_xmm_regs_[param_idx]);
                    loc.set_to_register(reg);
                } else {
                    stack_param_indices.push_back(param_idx);
                }
                break;

            case abi_value_kind::SSEUPx15_SSE:
                COMPILE_ASSERT(false,
                        "Not yet supported: callers/callees with parameter "
                        "type f32x16");
                break;
        }
    }

    // The msABI specifies that stack-based parameters are pushed to the call
    // stack in right-to-left parameter-list order.
    //
    // When control is first transfered to the callee, the call stack looks
    // like this:
    //
    // |          ......          |
    // |--------------------------| -|
    // | (right-most stack param) |  |
    // |--------------------------|  |
    // | (next stack param)       |  |
    // |--------------------------|  |- stack parameter area
    // |          ......          |  |
    // |--------------------------|  |
    // | (left-most stack param)  |  |
    // |--------------------------| -|
    // |                          |
    // | (32 bytes shadow space)  |
    // |                          |
    // |--------------------------|
    // | (return address)         |
    // |--------------------------| <-- %rsp
    //
    // When we specify a parameter's location on the stack, it's relative to
    // the value of %rsp as shown in this diagram. So the address of the
    // last-pushed stack parameter is always %rsp + 8 + shadow_space_bytes.
    size_t next_param_rsp_offset = 8 + profile.shadow_space_bytes_;

    // Note the iteration order. We're dealing with the stack-based indices
    // in left-to-right order, which means we're starting with LAST-PUSHED
    // parameter.
    for (const size_t param_idx : stack_param_indices) {
        const cpu_data_type_table::row &r
                = get_cpu_data_types().lookup(param_types[param_idx]);

        abi_value_location &loc = iface.param_locs_[param_idx];

        // Pre-call stack alignment is determined by whichever argument has the
        // strictest alignment requrirements.
        iface.initial_rsp_alignment_ = std::max(
                iface.initial_rsp_alignment_, r.abi_precall_stack_alignment_);

        loc.set_to_rsp_offset(next_param_rsp_offset);
        next_param_rsp_offset += r.abi_stack_slot_size_;
    }

    if (ret_cpu_dtype != cpu_data_type::void_t) {
        const auto &r = get_cpu_data_types().lookup(ret_cpu_dtype);

        switch (r.abi_initial_val_kind_) {
            case abi_value_kind::INTEGER:
                iface.return_val_loc_.set_to_register(
                        profile.func_return_gp_reg_);
                break;

            case abi_value_kind::SSE:
                iface.return_val_loc_.set_to_register(
                        profile.func_return_xmm_reg_);
                break;

            case abi_value_kind::SSEUPx15_SSE:
                COMPILE_ASSERT(false,
                        "Not yet supported: callers/callees with return type "
                        "f32x16");
                break;
        }
    }

    return iface;
}

static abi_function_interface compute_iface(const target_profile_t &profile,
        const std::vector<cpu_data_type> &param_types,
        const cpu_data_type ret_cpu_dtype) {
    if (profile.call_convention_ == call_convention::system_v) {
        return compute_iface_system_v(profile, param_types, ret_cpu_dtype);
    } else if (profile.call_convention_ == call_convention::microsoft) {
        return compute_iface_microsoft(profile, param_types, ret_cpu_dtype);
    } else {
        assert(false && "Unsupported Calling Convention!");
        return abi_function_interface();
    }
}

std::vector<size_t>
abi_function_interface::get_stack_params_descending_idx() const {
    std::vector<size_t> v;

    for (int idx = int(param_locs_.size() - 1); idx >= 0; --idx) {
        if (param_locs_[idx].get_type()
                == abi_value_location::tag_type::STACK) {
            v.push_back(idx);
        }
    }

    return v;
}

std::vector<size_t>
abi_function_interface::get_register_params_ascending_idx() const {
    std::vector<size_t> v;

    for (size_t idx = 0; idx < param_locs_.size(); ++idx) {
        if (param_locs_[idx].get_type()
                == abi_value_location::tag_type::REGISTER) {
            v.push_back(idx);
        }
    }

    return v;
}

size_t abi_function_interface::get_param_area_size() const {
    size_t n = 0;

    for (const auto &loc : param_locs_) {
        if (loc.get_type() == abi_value_location::tag_type::STACK) {
            // TODO(xxx): For now we're assuming that each stack slot is exactly
            // 8 bytes large, but that assumption will be wrong once we start
            // handling larger objects (e.g., simd vectors).
            n += 8;
        }
    }

    return n;
}

abi_function_interface::ptr abi_function_interface::make_interface(
        const target_profile_t &profile, const std::vector<expr> &params,
        sc_data_type_t ret_type) {
    std::vector<cpu_data_type> param_cpu_dtypes;
    for (auto &p : params) {
        const auto t = get_cpu_data_type(p->dtype_);
        param_cpu_dtypes.push_back(t);
    }

    const auto ret_cpu_dtype = get_cpu_data_type(ret_type);
    return std::make_shared<abi_function_interface>(
            compute_iface(profile, param_cpu_dtypes, ret_cpu_dtype));
}

} // namespace x86_64
} // namespace xbyak
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
