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

#ifndef INTERFACE_STREAM_HPP
#define INTERFACE_STREAM_HPP

#include "c_types_map.hpp"

#ifdef DNNL_GRAPH_WITH_SYCL
#include <CL/sycl.hpp>
#endif

#if DNNL_GRAPH_CPU_RUNTIME == DNNL_GRAPH_RUNTIME_THREADPOOL
#include "oneapi/dnnl/dnnl_graph_threadpool_iface.hpp"
#endif
struct dnnl_graph_thread_pool {
    int num_threads_ {0};
    dnnl_graph_thread_pool(int num_threads) : num_threads_(num_threads) {}
};

struct dnnl_graph_stream_attr {
    dnnl_graph_stream_attr(dnnl_graph_thread_pool *thread_pool)
        : thread_pool_ {thread_pool} {}
    dnnl_graph_thread_pool *thread_pool_ {};
};

struct dnnl_graph_stream {
public:
    dnnl_graph_stream() = delete;
    dnnl_graph_stream(const dnnl::graph::impl::engine_t *engine,
            const dnnl::graph::impl::stream_attr_t *attr = nullptr)
        : engine_ {engine} {
        UNUSED(attr);
    }

#ifdef DNNL_GRAPH_WITH_SYCL
    // Create an stream from SYCL queue.
    dnnl_graph_stream(const dnnl::graph::impl::engine_t *engine,
            const cl::sycl::queue &queue,
            const dnnl::graph::impl::stream_attr_t *attr = nullptr)
        : engine_ {engine}, queue_ {queue} {
        UNUSED(attr);
    }
#endif // DNNL_GRAPH_WITH_SYCL

#if DNNL_GRAPH_CPU_RUNTIME == DNNL_GRAPH_RUNTIME_THREADPOOL
    dnnl_graph_stream(const dnnl::graph::impl::engine_t *engine,
            dnnl::graph::threadpool_interop::threadpool_iface *threadpool,
            const dnnl::graph::impl::stream_attr_t *attr = nullptr)
        : dnnl_graph_stream(engine) {
        UNUSED(attr);
        assert(engine->kind() == dnnl::graph::impl::engine_kind::cpu);
        threadpool_ = threadpool;
    }

    dnnl::graph::impl::status_t get_threadpool(
            dnnl::graph::threadpool_interop::threadpool_iface **threadpool)
            const {
        if (engine_->kind() != dnnl::graph::impl::engine_kind::cpu)
            return dnnl::graph::impl::status::invalid_argument;
        *threadpool = threadpool_;
        return dnnl::graph::impl::status::success;
    }
#endif
    ~dnnl_graph_stream() = default;

    const dnnl::graph::impl::engine_t *get_engine() const noexcept {
        return engine_;
    }

#ifdef DNNL_GRAPH_WITH_SYCL
    const cl::sycl::queue &get_queue() const noexcept { return queue_; }
#endif

    dnnl::graph::impl::status_t wait() {
#ifdef DNNL_GRAPH_WITH_SYCL
        queue_.wait();
#endif
        return dnnl::graph::impl::status::success;
    }

private:
    const dnnl::graph::impl::engine_t *engine_;
#ifdef DNNL_GRAPH_WITH_SYCL
    cl::sycl::queue queue_;
#endif
#if DNNL_GRAPH_CPU_RUNTIME == DNNL_GRAPH_RUNTIME_THREADPOOL
    dnnl::graph::threadpool_interop::threadpool_iface *threadpool_ = nullptr;
#endif
};

#endif
