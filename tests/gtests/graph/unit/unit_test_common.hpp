/*******************************************************************************
* Copyright 2020-2024 Intel Corporation
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

#ifndef GRAPH_UNIT_UNIT_TEST_COMMON_HPP
#define GRAPH_UNIT_UNIT_TEST_COMMON_HPP

#include <memory>
#include <vector>
#include <type_traits>

#include "common/engine.hpp"
#include "common/stream.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "graph/interface/c_types_map.hpp"
#include "graph/interface/partition_cache.hpp"
#include "graph/interface/tensor.hpp"

#include "tests/gtests/dnnl_test_common.hpp"

#ifdef DNNL_WITH_SYCL
#include "gpu/intel/sycl/compat.hpp"
#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#elif __has_include(<CL/sycl.hpp>)
#include <CL/sycl.hpp>
#else
#error "Unsupported compiler"
#endif
#endif

#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_THREADPOOL
#include "test_thread.hpp"
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
#include "gpu/intel/ocl/ocl_engine.hpp"
#endif

#include "tests/gtests/dnnl_test_macros.hpp"

#ifdef DNNL_WITH_SYCL
::sycl::device &get_device();
::sycl::context &get_context();
#endif // DNNL_WITH_SYCL

dnnl::impl::graph::engine_t *get_engine();

dnnl::impl::graph::stream_t *get_stream();

dnnl::impl::graph::engine_kind_t get_test_engine_kind();

void set_test_engine_kind(dnnl::impl::graph::engine_kind_t kind);

inline int get_compiled_partition_cache_size() {
    int result = 0;
#ifndef DNNL_GRAPH_DISABLE_COMPILED_PARTITION_CACHE
    result = dnnl::impl::graph::compiled_partition_cache().get_size();
#endif
    return result;
}

inline int set_compiled_partition_cache_capacity(int capacity) {
    if (capacity < 0) return -1;
#ifndef DNNL_GRAPH_DISABLE_COMPILED_PARTITION_CACHE
    return dnnl::impl::graph::compiled_partition_cache().set_capacity(capacity);
#endif
    return 0;
}

class test_tensor {
private:
    using ltw = dnnl::impl::graph::logical_tensor_wrapper_t;

    struct deletor_wrapper {
        deletor_wrapper(const dnnl::impl::graph::engine_t *eng) : eng_(eng) {}
        void operator()(void *p) {
            if (p) {
                const auto k = eng_->kind();
                auto alc = static_cast<dnnl::impl::graph::allocator_t *>(
                        eng_->get_allocator());
                if (k == dnnl::impl::graph::engine_kind::cpu) {
#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_SYCL
                    alc->deallocate(p, get_device(), get_context(), {});
#else
                    alc->deallocate(p);
#endif
                } else if (k == dnnl::impl::graph::engine_kind::gpu) {
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL
                    alc->deallocate(p, get_device(), get_context(), {});
#elif DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
                    const auto *ocl_engine = dnnl::impl::utils::downcast<
                            const dnnl::impl::gpu::intel::ocl::ocl_gpu_engine_t
                                    *>(eng_);
                    auto dev = ocl_engine->device();
                    auto ctx = ocl_engine->context();
                    alc->deallocate(p, dev, ctx, {});
#else
                    assert(!"only sycl and ocl runtime is supported on gpu");
#endif
                } else {
                    assert(!"unknown engine kind");
                }
            }
        }
        const dnnl::impl::graph::engine_t *eng_;
    };

    /// @brief Alloc memory by engine
    /// @return Alloced memory
    static std::shared_ptr<char> allocate(
            const dnnl::impl::graph::engine_t *e, size_t size) {
        std::shared_ptr<char> data;
        auto alc = static_cast<dnnl::impl::graph::allocator_t *>(
                e->get_allocator());
        if (e->kind() == dnnl::impl::graph::engine_kind::cpu) { // cpu kind
#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_SYCL
            data.reset(static_cast<char *>(alc->allocate(
                               size, get_device(), get_context())),
                    deletor_wrapper {e});
#else
            data.reset(static_cast<char *>(alc->allocate(size)),
                    deletor_wrapper {e});
#endif
        } else { // gpu kind
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL
            data.reset(static_cast<char *>(alc->allocate(
                               size, get_device(), get_context())),
                    deletor_wrapper {e});
#elif DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
            const auto *ocl_engine = dnnl::impl::utils::downcast<
                    const dnnl::impl::gpu::intel::ocl::ocl_gpu_engine_t *>(e);
            auto dev = ocl_engine->device();
            auto ctx = ocl_engine->context();

            data.reset(static_cast<char *>(alc->allocate(size, dev, ctx)),
                    deletor_wrapper {e});
#else
            assert(!"only sycl and ocl runtime is supported on gpu");
#endif
        }
        return data;
    }

public:
    test_tensor() = default;

    test_tensor(const dnnl::impl::graph::logical_tensor_t &lt,
            const dnnl::impl::graph::engine_t *e)
        : num_bytes_(ltw(lt).size()) {
        data_ = allocate(e, num_bytes_);
        ts_ = dnnl::impl::graph::tensor_t(lt, e, data_.get());
    }

    template <typename T>
    test_tensor(const dnnl::impl::graph::logical_tensor_t &lt,
            const dnnl::impl::graph::engine_t *e, const std::vector<T> &data)
        : test_tensor(lt, e) {
        this->fill(data);
    }

    /// @brief Returns memory size (in bytes) of this tensor
    /// @return The size
    size_t get_size() const { return num_bytes_; };

    /// @brief Returns the current graph tensor
    /// @return The graph tensor
    const dnnl::impl::graph::tensor_t &get() const { return ts_; }

    /// @brief return @c true if tensor handle is not nullptr, otherwise
    /// return @c false
    operator bool() const { return ts_.get_data_handle() != nullptr; }

    /// @brief Return a vector copying elements in the tensor
    /// @tparam T Data type of tensor element
    /// @return The vector
    template <typename T>
    std::vector<T> as_vec_type() const {
        const auto dptr = static_cast<typename std::add_pointer<T>::type>(
                ts_.get_data_handle());
        if (!dptr) return {};
        size_t volume = (num_bytes_ + sizeof(T) - 1) / sizeof(T);
        return {dptr, dptr + volume};
    }

    template <typename T>
    void fill(T mean, T deviation, double sparsity = 1.) {
        if (num_bytes_ == 0 || !data_) return;
        T *format_ptr = static_cast<typename std::add_pointer<T>::type>(
                ts_.get_data_handle());
        size_t volume = (num_bytes_ + sizeof(T) - 1) / sizeof(T);
        fill_data<T>(volume, format_ptr, mean, deviation, sparsity);
    }

    template <typename T>
    void fill() {
        this->fill<T>(T(1), T(0.2f));
    }

    template <typename T>
    void fill(T val) {
        if (num_bytes_ == 0 || !data_) return;
        T *format_ptr = static_cast<typename std::add_pointer<T>::type>(
                ts_.get_data_handle());
        size_t volume = (num_bytes_ + sizeof(T) - 1) / sizeof(T);
        for (size_t i = 0; i < volume; ++i) {
            format_ptr[i] = val;
        }
    }

    template <typename T>
    void fill(const std::vector<T> &vec) {
        size_t volume = (num_bytes_ + sizeof(T) - 1) / sizeof(T);
        if (vec.size() != volume) {
            assert(!"number of elements in vector does not match test "
                    "tensor.");
            return;
        }
        if (num_bytes_ == 0 || !data_) return;
        T *format_ptr = static_cast<typename std::add_pointer<T>::type>(
                ts_.get_data_handle());
        for (size_t i = 0; i < vec.size(); ++i) {
            format_ptr[i] = vec[i];
        }
    }

    static std::vector<dnnl::impl::graph::tensor_t> to_graph_tensor(
            const std::vector<test_tensor> &vecs) {
        std::vector<dnnl::impl::graph::tensor_t> res;
        for (const auto &e : vecs) {
            res.emplace_back(e.get());
        }
        return res;
    }

private:
    dnnl::impl::graph::tensor_t ts_;
    std::shared_ptr<char> data_ {nullptr};
    size_t num_bytes_ {0};
};

#endif
