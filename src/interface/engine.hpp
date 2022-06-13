/*******************************************************************************
* Copyright 2020-2022 Intel Corporation
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

#ifndef INTERFACE_ENGINE_HPP
#define INTERFACE_ENGINE_HPP

#include <memory>

#include "allocator.hpp"
#include "c_types_map.hpp"

#ifdef DNNL_GRAPH_WITH_SYCL
#include <CL/sycl.hpp>
#endif

struct dnnl_graph_engine {
public:
    explicit dnnl_graph_engine(dnnl::graph::impl::engine_kind_t kind,
            size_t index, const dnnl::graph::impl::allocator_t *alloc = nullptr)
        : kind_(kind), index_(index) {
        if (alloc) {
            allocator_.reset(dnnl::graph::impl::allocator_t::create(alloc),
                    default_destroy_allocator);
        } else {
            allocator_.reset(dnnl::graph::impl::allocator_t::create(),
                    default_destroy_allocator);
        }
    }

    // default engine
    dnnl_graph_engine()
        : dnnl_graph_engine(dnnl::graph::impl::engine_kind::cpu, 0, nullptr) {}

#ifdef DNNL_GRAPH_WITH_SYCL
    dnnl_graph_engine(dnnl::graph::impl::engine_kind_t kind,
            const ::sycl::device &dev, const ::sycl::context &ctx,
            const dnnl::graph::impl::allocator_t *alloc = nullptr)
        : kind_(kind), dev_(dev), ctx_(ctx) {
        if (alloc) {
            allocator_.reset(dnnl::graph::impl::allocator_t::create(alloc),
                    default_destroy_allocator);
        } else {
            allocator_.reset(dnnl::graph::impl::allocator_t::create(),
                    default_destroy_allocator);
        }
    }
#endif

    ~dnnl_graph_engine() = default;

    size_t index() const noexcept { return index_; }

    dnnl::graph::impl::engine_kind_t kind() const noexcept { return kind_; }

    dnnl::graph::impl::allocator_t *get_allocator() const {
        return allocator_.get();
    };

#ifdef DNNL_GRAPH_WITH_SYCL
    const ::sycl::device &sycl_device() const { return dev_; }

    const ::sycl::context &sycl_context() const { return ctx_; }
#endif

    bool match(const dnnl::graph::impl::engine_t &eng) const {
        bool ok = true && (kind() == eng.kind());
#ifdef DNNL_GRAPH_WITH_SYCL
        ok = ok && (eng.sycl_device() == dev_) && (eng.sycl_context() == ctx_);
#endif
        return ok;
    }

private:
    static void default_destroy_allocator(
            dnnl::graph::impl::allocator_t *alloc) {
        alloc->release();
    }

    static void dummy_destroy_allocator(dnnl::graph::impl::allocator_t *) {}

    dnnl::graph::impl::engine_kind_t kind_ {};
    size_t index_ {};
    std::shared_ptr<dnnl::graph::impl::allocator_t> allocator_ {nullptr};
#ifdef DNNL_GRAPH_WITH_SYCL
    ::sycl::device dev_;
    ::sycl::context ctx_;
#endif
};

#endif
