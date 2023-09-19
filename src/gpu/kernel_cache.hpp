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

#ifndef GPU_KERNEL_CACHE_HPP
#define GPU_KERNEL_CACHE_HPP

#include <functional>
#include <memory>
#include <type_traits>

#include "common/engine.hpp"
#include "common/kernel_cache.hpp"
#include "common/utils.hpp"
#include "gpu/compute/compute.hpp"
#include "gpu/serialization.hpp"

namespace dnnl {
namespace impl {
namespace gpu {

template <typename T>
struct trivial_key_validator_t {
    static bool is_valid(const T &t) {
        // Runtime validation only occurs in C++20 as default comparisons
        // significantly improves the reliability of this check.
        static_assert(
                std::is_same<T, decltype(T::deserialize(t.serialize()))>::value,
                "serialization and deserialization must be supported for "
                "validation in C++20 builds");
#if __cplusplus >= 202002L
        return t == T::deserialize(t.serialize());
#else
        return true;
#endif
    }
};

// Helper structure to generate a hashing interface for a gpu_key_impl_t from a
// trivial structure.
template <typename T>
struct trivial_key_t : public T {
    // helper for extracting the value type associated with the key
    template <typename S>
    struct create_signature {};

    template <typename R, typename C, typename A1, typename A2>
    struct create_signature<R (C::*)(A1, A2) const> {
        using result_type = R;
        using class_type = C;
        using arg1_type = A1;
        using arg2_type = A2;
    };

    using value_type =
            typename std::remove_reference<typename create_signature<decltype(
                    &T::create_generator)>::arg2_type>::type;

    trivial_key_t() = delete;
    trivial_key_t(const T &t, const engine_id_t &id)
        : T(t)
        , id_(id)
        , serialization(t.serialize())
        , hash_(hash_combine(serialization.hash(), id_.hash())) {}
    bool operator==(const trivial_key_t &other) const {
        return serialization == other.serialization && id_ == other.id_;
    }
    size_t hash() const { return hash_; }

    bool is_valid() const {
        const T *base = this;
        return trivial_key_validator_t<T>::is_valid(*base);
    }

private:
    engine_id_t id_;
    serialized_t serialization;
    size_t hash_;
};

// Container of kernel cache values.
template <typename T>
struct gpu_kernel_value_container_t : public kernel_cache::value_impl_t {
    gpu_kernel_value_container_t(T &&t) : value(std::move(t)) {}
    T value;
};

// GPU specific interface for kernel_cache::value_t
struct gpu_kernel_value_t {
    gpu_kernel_value_t() = default;
    gpu_kernel_value_t(const std::shared_ptr<kernel_cache::value_impl_t> &impl)
        : impl_(impl) {}

    const kernel_cache::value_impl_t *impl() const { return impl_.get(); };

    std::shared_ptr<kernel_cache::value_impl_t> release() {
        std::shared_ptr<kernel_cache::value_impl_t> ret = nullptr;
        std::swap(ret, impl_);
        return ret;
    }

private:
    std::shared_ptr<kernel_cache::value_impl_t> impl_;
};

// GPU specific abstract interface for kernel_cache::key_impl_t
struct gpu_kernel_key_impl_t : public kernel_cache::key_impl_t {
    virtual status_t create_generator(
            engine_t *engine, gpu_kernel_value_t &generator) const = 0;
};

// Templated key container which implements the necessary virtual interfaces
// into the gpu_kernel_key_impl_t. This allows key implementations to be kept as
// simple data containing structures with no dependencies on the kernel cache.
template <typename K>
struct gpu_kernel_key_container_t : public gpu_kernel_key_impl_t {
    using value_type = typename K::value_type;

    template <typename... Args>
    gpu_kernel_key_container_t(Args &&...args)
        : key(std::forward<Args>(args)...) {}

    status_t create_generator(
            engine_t *engine, gpu_kernel_value_t &generator) const override {

        auto g = std::make_shared<gpu_kernel_value_container_t<value_type>>(
                value_type());

        auto *compute_engine
                = utils::downcast<compute::compute_engine_t *>(engine);
        auto status = key.create_generator(*compute_engine, g->value);
        generator = std::static_pointer_cast<kernel_cache::value_impl_t>(g);
        return status;
    }

    bool compare(const key_impl_t *key_impl) const override {
        auto *o = dynamic_cast<const gpu_kernel_key_container_t<K> *>(key_impl);
        if (o == nullptr) return false;
        return key == o->key;
    }

    size_t hash() const override { return key.hash(); }

    K key;
};

template <typename K>
using trivial_key_container_t = gpu_kernel_key_container_t<trivial_key_t<K>>;

// Interface to the kernel cache to consolidate cache related logic in one
// location
template <typename value_type>
status_t get_cached_kernels(std::shared_ptr<gpu_kernel_key_impl_t> &&key_impl,
        engine_t *engine, std::vector<compute::kernel_t> &kernels,
        const std::vector<const char *> &kernel_names);

extern template status_t get_cached_kernels<compute::kernel_t>(
        std::shared_ptr<gpu_kernel_key_impl_t> &&key_impl, engine_t *engine,
        std::vector<compute::kernel_t> &kernels,
        const std::vector<const char *> &kernel_names);

extern template status_t get_cached_kernels<compute::kernel_bundle_t>(
        std::shared_ptr<gpu_kernel_key_impl_t> &&key_impl, engine_t *engine,
        std::vector<compute::kernel_t> &kernels,
        const std::vector<const char *> &kernel_names);

} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif // GPU_JIT_DESC_HPP
