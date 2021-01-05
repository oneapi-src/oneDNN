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

#ifndef LLGA_INTERFACE_STREAM_HPP
#define LLGA_INTERFACE_STREAM_HPP

#include "c_types_map.hpp"

#if DNNL_GRAPH_WITH_SYCL
#include <CL/sycl.hpp>
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
    dnnl_graph_stream(const dnnl_graph_engine_t *engine,
            const dnnl_graph_stream_attr_t *attr = nullptr)
        : engine_ {engine} {
        UNUSED(attr);
        // TODO(Patryk): dnnl_graph_stream_attr_t operations?
    }
#if DNNL_GRAPH_WITH_SYCL
    // Create an stream from SYCL queue.
    dnnl_graph_stream(const dnnl_graph_engine_t *engine,
            const cl::sycl::queue &queue,
            const dnnl_graph_stream_attr_t *attr = nullptr)
        : engine_ {engine}, queue_ {queue} {
        UNUSED(attr);
    }
#endif // DNNL_GRAPH_WITH_SYCL
    ~dnnl_graph_stream() = default;

    const dnnl_graph_engine_t *get_engine() const noexcept { return engine_; }

#if DNNL_GRAPH_WITH_SYCL
    const cl::sycl::queue &get_queue() const noexcept { return queue_; }
#endif

private:
    const dnnl_graph_engine_t *engine_;
#if DNNL_GRAPH_WITH_SYCL
    cl::sycl::queue queue_;
#endif
};

namespace dnnl {
namespace graph {
namespace impl {
using thread_pool = ::dnnl_graph_thread_pool;
using stream_attr = ::dnnl_graph_stream_attr;
using stream = ::dnnl_graph_stream;
} // namespace impl
} // namespace graph
} // namespace dnnl

#endif
