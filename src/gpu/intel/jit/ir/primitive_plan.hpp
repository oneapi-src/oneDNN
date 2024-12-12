/*******************************************************************************
* Copyright 2024-2025 Intel Corporation
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

#ifndef GPU_INTEL_JIT_IR_PRIMITIVE_PLAN_HPP
#define GPU_INTEL_JIT_IR_PRIMITIVE_PLAN_HPP

#include "gpu/intel/jit/ir/kernel_info.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {

class primitive_exec_plan_t {
public:
    status_t execute(
            const gpu_primitive_t *primitive, const exec_ctx_t &ctx) const {
        for (auto &e : kernel_entries_) {
            std::vector<memory_storage_wrapper_t> storage_list;
            e.kernel_info.init_memory_storage_list(
                    storage_list, ctx, primitive);
            compute::kernel_arg_list_t arg_list;
            e.kernel_info.set_args(arg_list, storage_list);
            CHECK(primitive->parallel_for(
                    ctx, e.kernel_info.nd_range(), e.kernel, arg_list));
        }
        return status::success;
    }

    void add_kernel(
            const compute::kernel_t &kernel, const kernel_info_t &kernel_info) {
        kernel_entry_t e;
        e.kernel = kernel;
        e.kernel_info = kernel_info;
        kernel_entries_.push_back(e);
    }

private:
    struct kernel_entry_t {
        compute::kernel_t kernel;
        kernel_info_t kernel_info;
    };

    std::vector<kernel_entry_t> kernel_entries_;
};

class primitive_init_plan_t {
public:
    void set_regs(int regs) { regs_ = regs; }
    void set_simd(int simd) { simd_ = simd; }
    void set_dpas(bool dpas) { dpas_ = dpas; }

    void add_kernel(const std::shared_ptr<kernel_desc_base_t> &desc,
            const std::shared_ptr<kernel_params_base_t> &params) {
        kernel_entry_t e;
        e.desc = desc;
        e.params = params;
        kernel_entries_.push_back(e);
    }

    void add_user_buffer(const std::string &name, const layout_t &layout,
            bool is_input, bool is_output, int arg_key, bool zero_out) {
        buf_entries_.emplace_back();
        auto &e = buf_entries_.back();
        e.name = name;
        e.layout = layout;
        e.is_user_input = is_input;
        e.is_user_output = is_output;
        e.arg_key = arg_key;
        e.zero_out = zero_out;
    }

    void add_internal_buffer(const std::string &name, const layout_t &layout,
            const std::string &user_name, int scratchpad_key, bool zero_out) {
        buf_entries_.emplace_back();
        auto &e = buf_entries_.back();
        e.name = name;
        e.layout = layout;
        e.user_name = user_name;
        e.arg_key = scratchpad_key;
        e.zero_out = zero_out;
    }

    status_t create_exec_plan(primitive_exec_plan_t &exec_plan,
            gpu_primitive_t *primitive, impl::engine_t *engine) const;

private:
    struct buffer_entry_t {
        std::string name;
        jit::layout_t layout;
        int arg_key = 0;
        bool is_user_input = false;
        bool is_user_output = false;
        std::string user_name;
        bool zero_out = false;

        operator bool() const { return !name.empty(); }
        bool is_user() const { return is_user_input || is_user_output; }
    };

    struct kernel_entry_t {
        std::shared_ptr<kernel_desc_base_t> desc;
        std::shared_ptr<kernel_params_base_t> params;

        kernel_entry_t() = default;
        kernel_entry_t(const std::shared_ptr<kernel_desc_base_t> &desc,
                const std::shared_ptr<kernel_params_base_t> &params)
            : desc(desc), params(params) {}
    };

    buffer_entry_t find_buf(const std::string &name) const;
    kernel_info_t create_kernel_info(const kernel_desc_base_t &desc,
            const std::unordered_map<std::string, std::string> &buf_map) const;
    status_t add_kernel(primitive_exec_plan_t &exec_plan,
            const kernel_desc_base_t &desc, const kernel_params_base_t &params,
            gpu_primitive_t *primitive, impl::engine_t *engine,
            const std::unordered_map<std::string, std::string> &buf_map
            = {}) const;
    status_t add_zero_out_kernel(primitive_exec_plan_t &exec_plan,
            const buffer_entry_t &buf, gpu_primitive_t *primitive,
            impl::engine_t *engine) const;
    status_t add_reorder_kernel(primitive_exec_plan_t &exec_plan,
            const buffer_entry_t &src, const buffer_entry_t &dst,
            gpu_primitive_t *primitive, impl::engine_t *engine) const;

    std::vector<kernel_entry_t> kernel_entries_;
    std::vector<buffer_entry_t> buf_entries_;

    // Hints.
    int regs_ = 0;
    int simd_ = 0;
    bool dpas_ = false;
};

} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
