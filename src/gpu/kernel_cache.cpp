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

#include "gpu/kernel_cache.hpp"

namespace dnnl {
namespace impl {
namespace gpu {

status_t get_or_create(const kernel_cache::key_t &key,
        gpu_kernel_value_t &jit_generator, engine_t *engine) {
    struct create_context_t {
        const gpu_kernel_key_impl_t &params;
        engine_t *engine;
    };

    kernel_cache::iface_t::create_func_ptr_t create = [](void *context) {
        auto &c = *static_cast<create_context_t *>(context);
        gpu_kernel_value_t generator;
        auto status = c.params.create_generator(c.engine, generator);
        return kernel_cache::iface_t::result_t {generator.release(), status};
    };
    create_context_t context {
            *utils::downcast<gpu_kernel_key_impl_t *>(key.impl()), engine};
    auto result = kernel_cache::get().get_or_create(key, *create, &context);
    jit_generator = std::static_pointer_cast<gpu_kernel_value_impl_t>(
            result.value.release());
    return result.status;
}

status_t get_cached_kernels(std::shared_ptr<gpu_kernel_key_impl_t> &&key_impl,
        engine_t *engine, std::vector<compute::kernel_t> &kernels,
        const std::vector<const char *> &kernel_names) {
    kernel_cache::key_t key {std::move(key_impl)};

    gpu_kernel_value_t value;
    CHECK(get_or_create(key, value, engine));
    CHECK(value.get_kernels(engine, kernels, kernel_names));

    return status::success;
}

} // namespace gpu
} // namespace impl
} // namespace dnnl
