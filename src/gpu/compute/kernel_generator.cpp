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

#include "gpu/compute/kernel_generator.hpp"

#include "common/utils.hpp"
#include "gpu/compute/compute_engine.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace compute {

status_t compiled_bundle_t::get_kernels(const engine_t *engine,
        std::vector<kernel_t> &kernels,
        const std::vector<const char *> &kernel_names) const {
    auto *compute_engine = utils::downcast<const compute_engine_t *>(engine);
    auto status = compute_engine->create_kernels_from_bundle(
            kernels, kernel_names, *this);
    return status;
}

status_t compiled_bundle_t::create(compiled_bundle_t &kernel_generator,
        engine_t *engine, const std::vector<const char *> &kernel_names,
        const kernel_ctx_t &kernel_ctx) {
    auto *compute_engine = utils::downcast<const compute_engine_t *>(engine);

    CHECK(compute_engine->create_compiled_bundle(
            kernel_generator, kernel_names, kernel_ctx));

    return status::success;
}

status_t compiled_kernel_t::get_kernels(const engine_t *engine,
        std::vector<kernel_t> &kernels,
        const std::vector<const char *> &kernel_names) const {
    if (kernel_names.size() != 1) return status::runtime_error;

    // As there is only one kernel, allow the kernel_name to be unspecified
    if (kernel_names[0] && std::string(kernel_names[0]) != name_)
        return status::runtime_error;

    auto *compute_engine = utils::downcast<const compute_engine_t *>(engine);
    kernels = {kernel_t()};
    auto status = compute_engine->create_kernel_from_binary(
            kernels[0], binary_, name_.c_str());
    return status;
}

status_t compiled_kernel_t::create(compiled_kernel_t &kernel_generator,
        engine_t *engine, jit::jit_generator_base &jitter) {
    auto *compute_engine = utils::downcast<const compute_engine_t *>(engine);
    CHECK(compute_engine->create_compiled_kernel(kernel_generator, jitter));
    return status::success;
}

} // namespace compute
} // namespace gpu
} // namespace impl
} // namespace dnnl
