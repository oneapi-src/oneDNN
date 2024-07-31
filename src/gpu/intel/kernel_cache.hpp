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

#ifndef GPU_INTEL_KERNEL_CACHE_HPP
#define GPU_INTEL_KERNEL_CACHE_HPP

#include <memory>
#include <type_traits>

#include "common/cache_hit_types.hpp"
#include "common/engine_id.hpp"
#include "common/kernel_cache.hpp"
#include "common/utils.hpp"
#include "gpu/intel/compute/compute_engine.hpp"
#include "gpu/intel/serialization.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {

template <typename T>
struct trivial_key_validator_t {

    template <typename V>
    struct is_trivially_validatable {
        using yes_t = uint8_t;
        using no_t = uint16_t;

        template <typename U>
        static yes_t test(bool value = V::is_trivially_validatable);
        template <typename U>
        static no_t test(...);

        static const bool value = sizeof(test<V>(false)) == sizeof(yes_t);
    };

    template <typename U,
            gpu_utils::enable_if_t<is_trivially_validatable<U>::value,
                    bool> = true>
    static bool is_valid(const U &t) {
        static_assert(std::is_same<T, U>::value,
                "key validation is not intended for comparing different types");
        return true;
    }

    template <typename U,
            gpu_utils::enable_if_t<!is_trivially_validatable<U>::value,
                    bool> = true>
    static bool is_valid(const U &t) {
        // Runtime validation only occurs in C++20 as default comparisons
        // significantly improves the reliability of this check.
        static_assert(
                std::is_same<T, decltype(T::deserialize(t.serialize()))>::value,
                "serialization and deserialization must be supported for "
                "validation in C++20 builds");
        static_assert(std::is_same<T, U>::value,
                "key validation is not intended for comparing different types");

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
            impl::engine_t *engine, gpu_kernel_value_t &generator) const = 0;
};

// Templated key container which implements the necessary virtual interfaces
// into the gpu_kernel_key_impl_t. This allows key implementations to be kept as
// simple data containing structures with no dependencies on the kernel cache.
template <typename K>
struct gpu_kernel_key_container_t final : public gpu_kernel_key_impl_t {
    using value_type = typename K::value_type;

    ~gpu_kernel_key_container_t() final = default;
    gpu_kernel_key_container_t(const gpu_kernel_key_container_t &) = default;
    gpu_kernel_key_container_t(gpu_kernel_key_container_t &&) = default;
    gpu_kernel_key_container_t &operator=(const gpu_kernel_key_container_t &)
            = default;
    gpu_kernel_key_container_t &operator=(gpu_kernel_key_container_t &&)
            = default;

    template <typename... Args>
    gpu_kernel_key_container_t(Args &&...args)
        : key(std::forward<Args>(args)...) {}

    status_t create_generator(impl::engine_t *engine,
            gpu_kernel_value_t &generator) const override {

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
        impl::engine_t *engine, std::vector<compute::kernel_t> &kernels,
        const std::vector<const char *> &kernel_names,
        cache_state_t &kernel_cache_hit);

extern template status_t get_cached_kernels<compute::kernel_t>(
        std::shared_ptr<gpu_kernel_key_impl_t> &&key_impl,
        impl::engine_t *engine, std::vector<compute::kernel_t> &kernels,
        const std::vector<const char *> &kernel_names,
        cache_state_t &kernel_cache_hit);

extern template status_t get_cached_kernels<compute::kernel_bundle_t>(
        std::shared_ptr<gpu_kernel_key_impl_t> &&key_impl,
        impl::engine_t *engine, std::vector<compute::kernel_t> &kernels,
        const std::vector<const char *> &kernel_names,
        cache_state_t &kernel_cache_hit);

} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif // GPU_INTEL_KERNEL_CACHE_HPP
