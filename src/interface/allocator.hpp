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

#ifndef LLGA_INTERFACE_ALLOCATOR_HPP
#define LLGA_INTERFACE_ALLOCATOR_HPP

#include <cstdlib>

#include "oneapi/dnnl/dnnl_graph.h"

#include "c_types_map.hpp"
#include "utils.hpp"

#if DNNL_GRAPH_WITH_SYCL
#include <CL/sycl.hpp>
#endif

struct dnnl_graph_allocator final {
public:
    /// Allocator attributes
    struct attribute {
        friend struct dnnl_graph_allocator;

        llga::impl::allocator_attr_t data;

        /// Default constructor for an uninitialized attribute
        attribute() {
            data.type = llga::impl::allocator_lifetime::persistent;
            data.alignment = 0;
        }

        attribute(llga::impl::allocator_lifetime_t type, size_t alignment) {
            data.type = type;
            data.alignment = alignment;
        }

        /// Copy constructor
        attribute(const attribute &other) = default;

        /// Assign operator
        attribute &operator=(const attribute &other) = default;
    };

    dnnl_graph_allocator() = default;

    dnnl_graph_allocator(dnnl_graph_cpu_allocate_f cpu_malloc,
            dnnl_graph_cpu_deallocate_f cpu_free)
        : cpu_malloc_ {cpu_malloc}, cpu_free_ {cpu_free} {}

#if DNNL_GRAPH_WITH_SYCL
    dnnl_graph_allocator(dnnl_graph_sycl_allocate_f sycl_malloc,
            dnnl_graph_sycl_deallocate_f sycl_free)
        : sycl_malloc_(sycl_malloc), sycl_free_(sycl_free) {}
#endif

    void *allocate(size_t n, attribute attr = {}) {
        return cpu_malloc_(n, attr.data);
    }

#if DNNL_GRAPH_WITH_SYCL
    void *allocate(size_t n, const cl::sycl::device &dev,
            const cl::sycl::context &ctx, attribute attr) {
        return sycl_malloc_(n, static_cast<const void *>(&dev),
                static_cast<const void *>(&ctx), attr.data);
    }
#endif

    template <typename T>
    T *allocate(size_t num_elem, attribute attr = {}) {
        return static_cast<T *>(cpu_malloc_(num_elem * sizeof(T), attr.data));
    }

#if DNNL_GRAPH_WITH_SYCL
    template <typename T>
    T *allocate(size_t num_elem, const cl::sycl::device &dev,
            const cl::sycl::context &ctx, attribute attr = {}) {
        return static_cast<T *>(sycl_malloc_(num_elem * sizeof(T),
                static_cast<const void *>(&dev),
                static_cast<const void *>(&ctx), attr.data));
    }
#endif

    void deallocate(void *buffer) {
        if (buffer) cpu_free_(buffer);
    }

#if DNNL_GRAPH_WITH_SYCL
    void deallocate(void *buffer, const cl::sycl::context &ctx) {
        if (buffer) sycl_free_(buffer, static_cast<const void *>(&ctx));
    }
#endif

private:
    dnnl_graph_cpu_allocate_f cpu_malloc_ {
            llga::impl::utils::cpu_allocator::malloc};
    dnnl_graph_cpu_deallocate_f cpu_free_ {
            llga::impl::utils::cpu_allocator::free};

#if DNNL_GRAPH_WITH_SYCL
    dnnl_graph_sycl_allocate_f sycl_malloc_ {
            llga::impl::utils::sycl_allocator::malloc};
    dnnl_graph_sycl_deallocate_f sycl_free_ {
            llga::impl::utils::sycl_allocator::free};
#endif
};

namespace llga {
namespace impl {
using allocator = ::dnnl_graph_allocator;
} // namespace impl
} // namespace llga

#endif
