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

#ifndef GPU_COMPUTE_KERNEL_GENERATOR_HPP
#define GPU_COMPUTE_KERNEL_GENERATOR_HPP

#include "common/engine.hpp"
#include "common/kernel_cache.hpp"
#include "gpu/compute/kernel.hpp"
#include "gpu/compute/kernel_ctx.hpp"
#include "gpu/jit/jit_generator_base.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace compute {

//  Abstract interface for kernel generation.
struct kernel_generator_t {
    virtual ~kernel_generator_t() = default;

    status_t get_kernel(const engine_t *engine, kernel_t &kernel,
            const char *kernel_name) const {
        std::vector<kernel_t> kernels {};
        std::vector<const char *> names {kernel_name};
        CHECK(get_kernels(engine, kernels, names));
        kernel = kernels[0];
        return status::success;
    }
    virtual status_t get_kernels(const engine_t *engine,
            std::vector<kernel_t> &kernels,
            const std::vector<const char *> &kernel_names) const = 0;
};

// Kernel generator which contains a set of compiled kernels
struct compiled_bundle_t final : public kernel_generator_t {
    compiled_bundle_t() = default;
    explicit compiled_bundle_t(const binary_t &binary) : binary_(binary) {}

    status_t get_kernels(const engine_t *engine, std::vector<kernel_t> &kernels,
            const std::vector<const char *> &kernel_names) const override;

    static status_t create(compiled_bundle_t &jit_binary, engine_t *engine,
            const std::vector<const char *> &kernel_names,
            const kernel_ctx_t &kernel_ctx);

    const binary_t &binary() const { return binary_; }

private:
    binary_t binary_;
};

// Kernel generator which contains a single compiled kernel
struct compiled_kernel_t final : public kernel_generator_t {
    compiled_kernel_t() = default;
    explicit compiled_kernel_t(const binary_t &binary, const char *name)
        : binary_(binary), name_(name) {}

    status_t get_kernels(const engine_t *engine, std::vector<kernel_t> &kernels,
            const std::vector<const char *> &kernel_names) const override;

    static status_t create(compiled_kernel_t &jit_binary, engine_t *engine,
            jit::jit_generator_base &jitter);

    const binary_t &binary() const { return binary_; }

private:
    binary_t binary_;
    std::string name_;
};

} // namespace compute
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
