/*******************************************************************************
* Copyright 2023-2024 Intel Corporation
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

#include "gpu/intel/kernel_cache.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {

using namespace compute;

status_t get_or_create(const kernel_cache::key_t &key,
        gpu_kernel_value_t &jit_generator, impl::engine_t *engine,
        cache_state_t &kernel_cache_hit) {
    struct create_context_t {
        const gpu_kernel_key_impl_t &params;
        impl::engine_t *engine;
        cache_state_t cache_status;
    };

    kernel_cache::iface_t::create_func_ptr_t create = [](void *context) {
        auto &c = *static_cast<create_context_t *>(context);
        gpu_kernel_value_t generator;
        auto status = c.params.create_generator(c.engine, generator);
        c.cache_status = cache_state_t::miss;
        return kernel_cache::iface_t::result_t {generator.release(), status};
    };
    create_context_t context {
            *utils::downcast<gpu_kernel_key_impl_t *>(key.impl()), engine,
            cache_state_t::kernel_hit};
    auto result = kernel_cache::get().get_or_create(key, *create, &context);
    kernel_cache_hit = context.cache_status;
    jit_generator = std::static_pointer_cast<kernel_cache::value_impl_t>(
            result.value.release());
    return result.status;
}

template <typename value_type>
status_t get_cached_kernels(std::shared_ptr<gpu_kernel_key_impl_t> &&key_impl,
        impl::engine_t *engine, std::vector<kernel_t> &kernels,
        const std::vector<const char *> &kernel_names,
        cache_state_t &kernel_cache_hit) {
    kernel_cache::key_t key {std::move(key_impl)};

    gpu_kernel_value_t value;
    CHECK(get_or_create(key, value, engine, kernel_cache_hit));

    static_assert(std::is_same<value_type, kernel_t>()
                    || std::is_same<value_type, kernel_bundle_t>(),
            "Only support caching kernel_t or kernel_bundle_t");

    if (std::is_same<value_type, kernel_t>()) {
        if (kernel_names.size() != 1) return status::runtime_error;
        const kernel_t &kernel = utils::downcast<
                const gpu_kernel_value_container_t<kernel_t> *>(value.impl())
                                         ->value;
        // As there is only one kernel, allow the kernel_name to be unspecified
        if (kernel_names[0] && std::string(kernel_names[0]) != kernel.name())
            return status::runtime_error;

        kernels[0] = kernel;
        return status::success;
    } else if (std::is_same<value_type, kernel_bundle_t>()) {
        const kernel_bundle_t &bundle = utils::downcast<
                const gpu_kernel_value_container_t<kernel_bundle_t> *>(
                value.impl())
                                                ->value;
        return bundle.get_kernels(kernels, kernel_names);
    }
    return status::runtime_error;
}

template status_t get_cached_kernels<kernel_t>(
        std::shared_ptr<gpu_kernel_key_impl_t> &&key_impl,
        impl::engine_t *engine, std::vector<kernel_t> &kernels,
        const std::vector<const char *> &kernel_names,
        cache_state_t &kernel_cache_hit);
template status_t get_cached_kernels<kernel_bundle_t>(
        std::shared_ptr<gpu_kernel_key_impl_t> &&key_impl,
        impl::engine_t *engine, std::vector<kernel_t> &kernels,
        const std::vector<const char *> &kernel_names,
        cache_state_t &kernel_cache_hit);

} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
