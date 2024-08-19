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

#ifndef GRAPH_BACKEND_DNNL_KERNELS_BATCH_NORM_HPP
#define GRAPH_BACKEND_DNNL_KERNELS_BATCH_NORM_HPP

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "graph/backend/dnnl/kernels/kernel_base.hpp"

#include "graph/backend/dnnl/dnnl_constant_tensor_cache.hpp"
#include "graph/backend/dnnl/dnnl_partition_impl.hpp"
#include "graph/backend/dnnl/scratchpad.hpp"
#include "graph/backend/dnnl/subgraph.hpp"
#include "graph/backend/dnnl/thread_local_cache.hpp"

#include "graph/backend/dnnl/passes/memory_planning.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {

struct batch_norm_fwd_t : public kernel_base_t {
private:
    allocator_t *g_alloc_ = nullptr;

    std::shared_ptr<subgraph_t> subgraph_;
    memory_planner_t memory_planner_;

    std::function<std::shared_ptr<execution_args_set_t>()> resource_ctor_;
    constant_cache_t::key_t constant_key_ = 0;

public:
    batch_norm_fwd_t() {
        thread_local_cache_t<execution_args_set_t> res_cache;
        res_cache.retain();
    }

    ~batch_norm_fwd_t() override {
        thread_local_cache_t<execution_args_set_t> res_cache;
        res_cache.remove_if_exist(reinterpret_cast<size_t>(this));
        res_cache.release();
    }

    void prepare_args_set(const execution_args_set_t *res,
            const std::vector<tensor_t> &inputs,
            const std::vector<tensor_t> &outputs,
            const scratchpad_t &scratchpad);

    status_t compile_impl(const dnnl_partition_impl_t *part,
            const engine_t *g_engine,
            const std::vector<logical_tensor_t> &inputs,
            const std::vector<logical_tensor_t> &outputs) override;

    status_t execute_impl(const stream_t *g_stream,
            const std::vector<tensor_t> &inputs,
            const std::vector<tensor_t> &outputs) override;

#ifdef DNNL_WITH_SYCL
    status_t sycl_execute_impl(const stream_t *g_stream,
            const std::vector<tensor_t> &inputs,
            const std::vector<tensor_t> &outputs,
            const std::vector<::sycl::event> &sycl_deps,
            ::sycl::event *sycl_event) override;
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    status_t ocl_execute_impl(const stream_t *g_stream,
            const std::vector<tensor_t> &inputs,
            const std::vector<tensor_t> &outputs,
            const std::vector<cl_event> &cl_deps, cl_event *ret_event) override;
#endif

    DEF_KERNEL_METHOD_STR(batch_norm_fwd_t)
};

#if BUILD_TRAINING
struct batch_norm_bwd_t : public kernel_base_t {
private:
    allocator_t *g_alloc_ = nullptr;

    std::shared_ptr<subgraph_t> subgraph_;
    memory_planner_t memory_planner_;

    std::function<std::shared_ptr<execution_args_set_t>()> resource_ctor_;

public:
    batch_norm_bwd_t() {
        thread_local_cache_t<execution_args_set_t> res_cache;
        res_cache.retain();
    }

    ~batch_norm_bwd_t() override {
        thread_local_cache_t<execution_args_set_t> res_cache;
        res_cache.remove_if_exist(reinterpret_cast<size_t>(this));
        res_cache.release();
    }

    status_t compile_impl(const dnnl_partition_impl_t *part,
            const engine_t *g_engine,
            const std::vector<logical_tensor_t> &inputs,
            const std::vector<logical_tensor_t> &outputs) override;

    void prepare_args_set(const execution_args_set_t *res,
            const std::vector<tensor_t> &inputs,
            const std::vector<tensor_t> &outputs,
            const scratchpad_t &scratchpad);

    status_t execute_impl(const stream_t *g_stream,
            const std::vector<tensor_t> &inputs,
            const std::vector<tensor_t> &outputs) override;

#ifdef DNNL_WITH_SYCL
    status_t sycl_execute_impl(const stream_t *g_stream,
            const std::vector<tensor_t> &inputs,
            const std::vector<tensor_t> &outputs,
            const std::vector<::sycl::event> &sycl_deps,
            ::sycl::event *sycl_event) override;
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    status_t ocl_execute_impl(const stream_t *g_stream,
            const std::vector<tensor_t> &inputs,
            const std::vector<tensor_t> &outputs,
            const std::vector<cl_event> &cl_deps, cl_event *ret_event) override;
#endif

    DEF_KERNEL_METHOD_STR(batch_norm_bwd_t)
};
#endif // BUILD_TRAINING

} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
