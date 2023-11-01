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

#ifndef GPU_JIT_IR_KERNEL_INFO_HPP
#define GPU_JIT_IR_KERNEL_INFO_HPP

#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "common/c_types_map.hpp"
#include "common/primitive_exec_types.hpp"
#include "gpu/compute/compute.hpp"
#include "gpu/gpu_primitive.hpp"
#include "gpu/jit/ir/ir.hpp"
#include "gpu/jit/ir/kernel_desc.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

class memory_storage_ptr_t {
public:
    memory_storage_ptr_t(std::unique_ptr<memory_storage_t> &&ptr)
        : unique_ptr_(std::move(ptr)) {}
    memory_storage_ptr_t(const memory_storage_t *ptr) : raw_ptr_(ptr) {}
    memory_storage_ptr_t(const memory_storage_ptr_t &) = delete;

    const memory_storage_t *get() const {
        if (unique_ptr_) return unique_ptr_.get();
        return raw_ptr_;
    }

private:
    std::unique_ptr<memory_storage_t> unique_ptr_; // Owning pointer.
    const memory_storage_t *raw_ptr_ = nullptr; // Non-owning pointer.
};

class memory_storage_wrapper_t {
public:
    memory_storage_wrapper_t() = default;
    memory_storage_wrapper_t(std::unique_ptr<memory_storage_t> &&ptr)
        : ptr_(new memory_storage_ptr_t(std::move(ptr))) {}
    memory_storage_wrapper_t(const memory_storage_t *ptr)
        : ptr_(new memory_storage_ptr_t(ptr)) {}
    memory_storage_wrapper_t(const memory_storage_t &ref)
        : memory_storage_wrapper_t(&ref) {}

    const memory_storage_t *get() const {
        if (!ptr_) return nullptr;
        return ptr_.get()->get();
    }

private:
    std::shared_ptr<memory_storage_ptr_t> ptr_;
};

class grid_context_t {
public:
    grid_context_t(bool create_empty = false) {
        if (!create_empty) {
            for (int i = 0; i < ndims(); i++) {
                local_ids_[i] = var_t::make(
                        type_t::u16(), "local_id" + std::to_string(i));
                tg_idxs_[i] = var_t::make(
                        type_t::s32(), "tg_idx" + std::to_string(i));
            }
        }
    }

    int ndims() const { return grid_ndims_; }
    void set_tg_idx(int idx, const expr_t &e) {
        ir_assert(idx >= 0 && idx < ndims());
        tg_idxs_[idx] = e;
    }
    void set_local_id(int idx, const expr_t &e) {
        ir_assert(idx >= 0 && idx < ndims());
        local_ids_[idx] = e;
    }

    const expr_t &tg_idx(int idx) const {
        ir_assert(idx >= 0 && idx < ndims());
        return tg_idxs_[idx];
    }
    const expr_t &local_id(int idx) const {
        ir_assert(idx >= 0 && idx < ndims());
        return local_ids_[idx];
    }

private:
    static const int grid_ndims_ = 3;
    std::array<expr_t, grid_ndims_> tg_idxs_;
    std::array<expr_t, grid_ndims_> local_ids_;
};

enum class kernel_id_t {
    undef,
    convolution,
    pre_reorder,
    post_reorder,
    zero_out,
};

// Kernel information, includes:
// - Kernel identifier
// - Kernel arguments
// - ND-range for submission (optional)
// Kernel arguments can be of the following kinds:
// - Internal arguments: only scalar
//   - Examples: common output scales (contain a single value)
// - Resource arguments: stored to a resource storage during primitive creation
//   - Examples: output scales or zero points
// - User arguments: passed by the user at run time
//   - Examples: source, weights, destination
class kernel_info_t {
public:
    void set_id(kernel_id_t id) { id_ = id; }

    kernel_id_t id() const { return id_; }

    // Returns stage ID, kernels with smaller stage IDs are executed first.
    int stage_id() const {
        switch (id()) {
            case kernel_id_t::convolution: return 1;
            case kernel_id_t::pre_reorder: return 0;
            case kernel_id_t::post_reorder: return 2;
            case kernel_id_t::zero_out: return 0;
            default: ir_error_not_expected();
        }
        return -1;
    }

    void set_nd_range(const compute::nd_range_t &nd_range) {
        nd_range_ = nd_range;
    }

    const compute::nd_range_t &nd_range() const { return nd_range_; }

    void register_internal_arg(
            const expr_t &var, const expr_t &value = expr_t()) {
        register_arg(var, arg_kind_t::internal, -1, /*is_input=*/true);
        set_internal_arg(var.as<var_t>().name, value);
    }

    void set_internal_arg(const std::string &name, const expr_t &value) {
        auto *arg = find_arg_impl(name);
        ir_assert(arg);
        arg->value = value;
    }

    void register_resource_arg(const expr_t &var) {
        // TODO: Check key uniqueness.
        register_arg(var, arg_kind_t::resource, nargs(), /*is_input=*/true);
    }

    void register_user_arg(const expr_t &var, int dnnl_arg, bool is_input) {
        register_arg(var, arg_kind_t::user, dnnl_arg, is_input);
    }

    void register_scratchpad_arg(
            const expr_t &var, int key, bool is_input, size_t size) {
        register_arg(var, arg_kind_t::scratchpad, key, is_input, size);
    }

    const std::string &arg_name(int idx) const {
        ir_assert(idx >= 0 && idx < nargs());
        return args_[idx].name();
    }

    const expr_t &arg_var(int idx) const {
        ir_assert(idx >= 0 && idx < nargs());
        return args_[idx].var;
    }

    const type_t &arg_type(int idx) const { return arg_var(idx).type(); }

    expr_t find_arg(const std::string &name, bool allow_empty = false) const {
        auto *arg = find_arg_impl(name);
        if (arg) return arg->var;
        if (!allow_empty)
            ir_error_not_expected() << "Argument not found: " << name;
        return expr_t();
    }

    int key(int idx) const {
        ir_assert(idx >= 0 && idx < nargs());
        return args_[idx].key;
    }

    int key(const std::string &name) const {
        for (int i = 0; i < nargs(); i++) {
            if (arg_name(i) == name) return key(i);
        }
        ir_error_not_expected() << "Argument not found: " << name;
        return -1;
    }

    int nargs() const { return int(args_.size()); }

    bool is_resource(int idx) const {
        ir_assert(idx >= 0 && idx < nargs());
        return args_[idx].kind == arg_kind_t::resource;
    }

    bool is_scratchpad(int idx) const {
        ir_assert(idx >= 0 && idx < nargs());
        return args_[idx].kind == arg_kind_t::scratchpad;
    }

    bool is_user(int idx) const {
        ir_assert(idx >= 0 && idx < nargs());
        return args_[idx].kind == arg_kind_t::user;
    }

    bool is_input(int idx) const {
        ir_assert(idx >= 0 && idx < nargs());
        return args_[idx].is_input;
    }

    bool is_output(int idx) const { return !is_input(idx); }

    memory_storage_wrapper_t arg_storage(int idx, const exec_ctx_t &ctx,
            const gpu_primitive_t *primitive) const {
        ir_assert(idx >= 0 && idx < nargs());
        bool is_input = args_[idx].is_input;
        int key = args_[idx].key;
        switch (args_[idx].kind) {
            case arg_kind_t::resource:
                return *(ctx.get_resource_mapper()
                                 ->template get<gpu_resource_t>(primitive)
                                 ->get_memory_storage(key));
            case arg_kind_t::scratchpad:
                return ctx.get_scratchpad_grantor().get_memory_storage(key);
            case arg_kind_t::user: {
                if (is_input) return CTX_IN_STORAGE(args_[idx].key);
                return CTX_OUT_STORAGE(args_[idx].key);
            }
            // No storage for internal arguments.
            case arg_kind_t::internal: return memory_storage_wrapper_t();
            default: ir_error_not_expected();
        }
        return memory_storage_wrapper_t();
    }

    size_t arg_size(int idx, const gpu_primitive_t *primitive) const {
        switch (args_[idx].kind) {
            case arg_kind_t::user: {
                auto *md = primitive->pd()->arg_md(key(idx));
                return memory_desc_wrapper(md).size();
            }
            case arg_kind_t::scratchpad: return args_[idx].scratchpad_size;
            default: ir_error_not_expected();
        }
        return std::numeric_limits<size_t>::max();
    }

    void init_memory_storage_list(std::vector<memory_storage_wrapper_t> &list,
            const exec_ctx_t &ctx, const gpu_primitive_t *primitive) const {
        list = std::vector<memory_storage_wrapper_t>(nargs());
        for (int i = 0; i < nargs(); i++) {
            list[i] = arg_storage(i, ctx, primitive);
        }
    }

    void set_args(compute::kernel_arg_list_t &arg_list,
            const std::vector<memory_storage_wrapper_t> &storage_list) const {
        for (int i = 0; i < nargs(); i++) {
            switch (args_[i].kind) {
                case arg_kind_t::internal: {
                    auto &value = args_[i].value;
                    auto &type = args_[i].var.type();

                    do {
#define CASE(ir_type, cpp_type) \
    if (type == type_t::ir_type()) { \
        arg_list.set(i, to_cpp<cpp_type>(value)); \
        break; \
    }

                        CASE(f32, float)
                        CASE(s16, int16_t)
                        CASE(s32, int32_t)
                        CASE(s64, int64_t)
                        CASE(u16, uint16_t)
                        CASE(u32, uint32_t)
                        CASE(u64, uint64_t)
#undef CASE

                        ir_error_not_expected() << type;
                    } while (false);
                    break;
                }
                case arg_kind_t::resource:
                case arg_kind_t::scratchpad:
                case arg_kind_t::user: {
                    arg_list.set(i, *storage_list[i].get());
                    break;
                }
                default: ir_error_not_expected();
            }
        }
    }

private:
    enum class arg_kind_t { internal, resource, scratchpad, user };

    struct arg_t {
        arg_t(const expr_t &var, arg_kind_t kind, int key, bool is_input,
                size_t scratchpad_size)
            : var(var)
            , kind(kind)
            , key(key)
            , is_input(is_input)
            , scratchpad_size(scratchpad_size) {}

        const std::string &name() const { return var.as<var_t>().name; }

        expr_t var;
        arg_kind_t kind;
        int key; // Unique key across arguments with the same kind.
        bool is_input;
        expr_t value; // For internal arguments, must be a constant.
        size_t scratchpad_size; // For scratchpad arguments only.
    };

    void register_arg(const expr_t &var, arg_kind_t kind, int key,
            bool is_input, size_t scratchpad_size = 0) {
        ir_assert(is_var(var)) << "Expected var, got: " << var;
        args_.emplace_back(var, kind, key, is_input, scratchpad_size);
    }

    const arg_t *find_arg_impl(const std::string &name) const {
        for (int i = 0; i < nargs(); i++) {
            if (args_[i].name() == name) return &args_[i];
        }
        return nullptr;
    }

    arg_t *find_arg_impl(const std::string &name) {
        auto *arg
                = const_cast<const kernel_info_t *>(this)->find_arg_impl(name);
        return const_cast<arg_t *>(arg);
    }

    kernel_id_t id_ = kernel_id_t::undef;
    compute::nd_range_t nd_range_;

    std::vector<arg_t> args_;
};

class exec_plan_t {
public:
    int kernel_count() const { return (int)entries_.size(); }

    status_t create_kernels(std::vector<compute::kernel_t> &kernels,
            gpu_primitive_t *primitive, engine_t *engine) const {
        for (auto &e : entries_) {
            compute::kernel_t kernel;
            CHECK(e.desc->create_kernel(kernel, primitive, engine));
            kernels.push_back(kernel);
        }
        return status::success;
    }

    template <typename T>
    status_t execute(const T *primitive, const exec_ctx_t &ctx,
            const std::vector<compute::kernel_t> &kernels) const {
        for (int i = 0; i < kernel_count(); i++) {
            auto &e = entries_[i];
            kernel_info_t info;
            CHECK(e.params->init_dispatch_kernel_info(info, *e.desc));
            std::vector<memory_storage_wrapper_t> storage_list;
            info.init_memory_storage_list(storage_list, ctx, primitive);
            compute::kernel_arg_list_t arg_list;
            info.set_args(arg_list, storage_list);
            CHECK(primitive->parallel_for(
                    ctx, info.nd_range(), kernels[i], arg_list));
        }
        return status::success;
    }

    void add_kernel(const std::shared_ptr<kernel_desc_base_t> &desc,
            const std::shared_ptr<kernel_params_base_t> &params) {
        entry_t e;
        e.desc = desc;
        e.params = params;
        entries_.push_back(e);
    }

private:
    struct entry_t {
        std::shared_ptr<kernel_desc_base_t> desc;
        std::shared_ptr<kernel_params_base_t> params;
    };

    std::vector<entry_t> entries_;
};

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
