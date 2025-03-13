/*******************************************************************************
* Copyright 2019-2025 Intel Corporation
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

#ifndef GPU_INTEL_COMPUTE_KERNEL_HPP
#define GPU_INTEL_COMPUTE_KERNEL_HPP

#if defined(__linux__) && (defined(DNNL_DEV_MODE) || !defined(NDEBUG))
#include <unistd.h>
#endif

#include <functional>
#include <memory>
#include <utility>

#include "common/utils.hpp"
#include "common/verbose.hpp"
#include "gpu/intel/compute/kernel_arg_list.hpp"
#include "gpu/intel/compute/utils.hpp"
#include "gpu/intel/utils.hpp"
#include "xpu/context.hpp"
#include "xpu/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace compute {

#if defined(__linux__) && (defined(DNNL_DEV_MODE) || !defined(NDEBUG))
struct program_src_t {
    program_src_t() = default;
    program_src_t(const std::string &src_str) {
        // Only enable if gdb-oneapi debugging is active
        if (getenv_int("ZET_ENABLE_PROGRAM_DEBUGGING", 0) == 0) return;

        const int name_size = 29;
        char name[name_size] = "/tmp/dnnl_ocl_jit_src.XXXXXX";

        // Ensure /tmp is a valid target for writing a temporary file
        bool is_symlink = false;
        status_t status = check_for_symlinks("/tmp", &is_symlink);
        if (status != status::success || is_symlink) return;

        // Guaranteed to have permissions 600 per the mkstemp specification,
        // which is the minimum required for writing and then subsequently
        // reading when debugging.
        int fd = mkstemp(name);
        if (fd == -1) return;

        auto delete_fd = [&](int fd, char *name) {
            // Unlink is called before close to ensure the file always exists
            // and cannot be replaced with another file
            unlink(name);
            close(fd);
        };

        if (write(fd, src_str.c_str(), src_str.length()) == -1) {
            delete_fd(fd, name);
            return;
        }
        if (fsync(fd) == -1) {
            delete_fd(fd, name);
            return;
        }

        auto deleter = [&](char *name) {
            delete_fd(fd, name);
            delete[] name;
        };

        name_ = std::shared_ptr<char>(new char[name_size], deleter);
        std::memcpy(name_.get(), name, name_size);
    }
    operator bool() const { return name_ != nullptr; };
    const char *name() const { return name_.get(); }

private:
    std::shared_ptr<char> name_;
};
#else
struct program_src_t {
    program_src_t() = default;
    program_src_t(const std::string &src_str) {}
    operator bool() const { return false; }
    const char *name() const { return nullptr; }
};
#endif

class kernel_impl_t {
public:
    kernel_impl_t() = default;

    kernel_impl_t(const kernel_impl_t &) = delete;
    kernel_impl_t &operator=(const kernel_impl_t &) = delete;
    virtual ~kernel_impl_t() = default;

    virtual status_t parallel_for(impl::stream_t &stream,
            const nd_range_t &range, const kernel_arg_list_t &arg_list,
            const xpu::event_t &deps, xpu::event_t &out_dep) {
        gpu_assert(false) << "unimplemented function parallel_for() called";
        return status::runtime_error;
    }

    virtual status_t parallel_for(
            impl::stream_t &stream, const std::function<void(void *)> &cgf) {
        gpu_assert(false) << "unimplemented function parallel_for() called";
        return status::runtime_error;
    }

    virtual status_t get_binary_size(
            const impl::engine_t *engine, size_t *binary_size) const {
        gpu_assert(false) << "unimplemented function get_binary_size() called";
        return status::runtime_error;
    }
    virtual status_t get_binary(
            const impl::engine_t *engine, xpu::binary_t &binary) const {
        gpu_assert(false) << "unimplemented function get_binary() called";
        return status::runtime_error;
    }

    virtual const std::vector<scalar_type_t> &arg_types() const {
        static const std::vector<scalar_type_t> dummy;
        return dummy;
    }

    virtual void save_output_events() {}

    virtual status_t dump() const {
        gpu_assert(false) << "unimplemented function dump() called";
        return status::runtime_error;
    }

    virtual std::string name() const {
        gpu_assert(false) << "unimplemented function name() called";
        return "unknown";
    }

    status_t check_scalar_arguments(const kernel_arg_list_t &arg_list) const {
        // Some kernels may not support argument validation.
        if (arg_types().empty()) return status::success;

        for (int i = 0; i < arg_list.nargs(); i++) {
            auto &arg = arg_list.get(i);
            auto req_arg_type = arg_types()[i];
            if (!arg.is_global() && !arg.is_local() && !arg.is_svm_pointer()) {
                if (req_arg_type == gpu::intel::compute::scalar_type_t::undef) {
                    // Types of kernel arguments may not be available when zebin
                    // is used.
                    continue;
                }

                if (req_arg_type != arg.scalar_type()) {
                    VERROR(primitive, gpu,
                            "%s: scalar kernel argument #%d (%s) is "
                            "different from the type of the given scalar (%s)",
                            name().c_str(), i, to_string(req_arg_type).c_str(),
                            to_string(arg.scalar_type()).c_str());
                    return status::invalid_arguments;
                }
            }
        }
        return status::success;
    }
};

class kernel_t {
public:
    kernel_t(std::nullptr_t) : impl_(nullptr) {}
    kernel_t(std::shared_ptr<kernel_impl_t> &impl) : impl_(impl) {}
    kernel_t(std::shared_ptr<kernel_impl_t> &&impl) : impl_(std::move(impl)) {}

    kernel_t() = default;
    kernel_t(kernel_t &&other) = default;
    kernel_t(const kernel_t &other) = default;
    kernel_t &operator=(const kernel_t &other) = default;
    kernel_t &operator=(kernel_t &&other) = default;

    virtual ~kernel_t() = default;

    operator bool() const { return bool(impl_); }

    kernel_impl_t *impl() const { return impl_.get(); }

    status_t parallel_for(impl::stream_t &stream, const nd_range_t &range,
            const kernel_arg_list_t &arg_list, const xpu::event_t &deps,
            xpu::event_t &out_dep) const {
        return impl_->parallel_for(stream, range, arg_list, deps, out_dep);
    }

    status_t parallel_for(impl::stream_t &stream,
            const std::function<void(void *)> &cgf) const {
        return impl_->parallel_for(stream, cgf);
    }

    status_t get_binary_size(
            const impl::engine_t *engine, size_t *binary_size) const {
        return impl_->get_binary_size(engine, binary_size);
    }

    status_t get_binary(
            const impl::engine_t *engine, xpu::binary_t &binary) const {
        return impl_->get_binary(engine, binary);
    }

    const std::vector<scalar_type_t> &arg_types() const {
        return impl_->arg_types();
    }

    void save_output_events() { return impl_->save_output_events(); }

    status_t dump() const {
        if (!gpu_utils::is_jit_dump_enabled()) return status::success;
        return impl_->dump();
    }

    std::string name() const { return impl_->name(); }

private:
    std::shared_ptr<kernel_impl_t> impl_;
};

class kernel_bundle_t {
public:
    kernel_bundle_t() = default;
    kernel_bundle_t(std::vector<kernel_t> &&kernels,
            const std::vector<const char *> &kernel_names) {
        for (size_t i = 0; i < kernels.size(); i++) {
            bundle[kernel_names[i]] = std::move(kernels[i]);
        }
    }
    // Copies may be expensive, require explicit clone
    kernel_bundle_t(const kernel_bundle_t &other) = delete;
    kernel_bundle_t &operator=(const kernel_bundle_t &other) = delete;
    kernel_bundle_t(kernel_bundle_t &&other) = default;
    kernel_bundle_t &operator=(kernel_bundle_t &&other) = default;
    ~kernel_bundle_t() = default;

    status_t get_kernels(std::vector<kernel_t> &kernels,
            const std::vector<const char *> &kernel_names) const {
        kernels = std::vector<kernel_t>(kernel_names.size());
        for (size_t i = 0; i < kernel_names.size(); i++) {
            if (!kernel_names[i]) continue;
            auto kernel_entry = bundle.find(kernel_names[i]);
            if (kernel_entry == bundle.end()) return status::runtime_error;
            kernels[i] = kernel_entry->second;
        }
        return status::success;
    }

    std::unordered_map<std::string, kernel_t> bundle;
};

} // namespace compute
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif // GPU_INTEL_COMPUTE_KERNEL_HPP
