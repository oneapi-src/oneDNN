/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#ifndef LLGA_TESTS_CPP_UNIT_UNIT_TEST_COMMON_HPP
#define LLGA_TESTS_CPP_UNIT_UNIT_TEST_COMMON_HPP

#include <memory>
#include <vector>

#include "interface/engine.hpp"
#include "interface/stream.hpp"

#if DNNL_GRAPH_WITH_SYCL
#include <CL/sycl.hpp>
#endif

namespace impl = dnnl::graph::impl;

#if DNNL_GRAPH_WITH_SYCL
cl::sycl::device &get_device();
cl::sycl::context &get_context();
void *sycl_alloc(size_t n, const void *dev, const void *ctx);
void sycl_free(void *ptr, const void *ctx);
#endif // DNNL_GRAPH_WITH_SYCL

impl::engine_t &get_engine(dnnl::graph::impl::engine_kind_t engine_kind
        = dnnl::graph::impl::engine_kind::any_engine);

impl::stream &get_stream();

namespace test {

#if DNNL_GRAPH_WITH_SYCL
constexpr size_t usm_alignment = 16;

template <typename T>
using AllocatorBase = cl::sycl::usm_allocator<T, cl::sycl::usm::alloc::shared,
        usm_alignment>;
#else
template <typename T>
using AllocatorBase = std::allocator<T>;
#endif // DNNL_GRAPH_WITH_SYCL

template <typename T>
class TestAllocator : public AllocatorBase<T> {
public:
#if DNNL_GRAPH_WITH_SYCL
    TestAllocator() : AllocatorBase<T>(get_context(), get_device()) {}
#else
    TestAllocator() : AllocatorBase<T>() {}
#endif

    template <typename U>
    struct rebind {
        using other = TestAllocator<U>;
    };
};

template <typename T>
using vector = std::vector<T, TestAllocator<T>>;
} // namespace test

#endif
