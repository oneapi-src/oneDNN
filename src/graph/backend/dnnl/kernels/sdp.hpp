/*******************************************************************************
* Copyright 2024 Intel Corporation
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

#ifndef GRAPH_BACKEND_DNNL_KERNELS_SDP_HPP
#define GRAPH_BACKEND_DNNL_KERNELS_SDP_HPP

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "graph/backend/dnnl/kernels/kernel_base.hpp"
#include "graph/backend/dnnl/kernels/large_partition.hpp"
#include "graph/backend/dnnl/kernels/sdp_decomp.hpp"
#include "graph/backend/dnnl/kernels/sdp_primitive.hpp"

#include "graph/backend/dnnl/dnnl_partition_impl.hpp"

#define VDISPATCH_GRAPH_SDP(msg, ...) \
    VINFO(graph, create, dispatch, compile, msg, ##__VA_ARGS__)

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {

// The second template param dt is used to indicate the internal data type of
// int8 sdp pattern. It doesn't take any effect if quantized param is false.
template <bool quantized = false, memory::data_type dt = memory::data_type::f32>
struct sdp_base_t : public kernel_base_t {
private:
    std::shared_ptr<kernel_base_t> kernel;

public:
    status_t compile_impl(const dnnl_partition_impl_t *part,
            const engine_t *g_engine,
            const std::vector<logical_tensor_t> &inputs,
            const std::vector<logical_tensor_t> &outputs) override {
        const engine_kind_t ekind = g_engine->kind();
        bool enable_decomp = false;
        bool enable_ukernel = false;

        if (ekind == engine_kind::cpu) {
            enable_decomp = enable_decomp_kernel();
        } else if (ekind == engine_kind::gpu) {
            enable_ukernel = !force_primitive();
        } else {
            assert(!"unknown engine kind");
            return status::invalid_arguments;
        }

        status_t ret = status::unimplemented;

        if (enable_ukernel) {
            kernel = std::make_shared<sdp_primitive_kernel_t<quantized>>();
            ret = kernel->compile_impl(part, g_engine, inputs, outputs);
        }

        if (ret != status::success && enable_decomp) {
            kernel = std::make_shared<sdp_decomp_kernel_t<quantized, dt>>();
            ret = kernel->compile_impl(part, g_engine, inputs, outputs);
        }

        if (ret != status::success) {
            kernel = std::make_shared<larger_partition_kernel_t>();
            ret = kernel->compile_impl(part, g_engine, inputs, outputs);
        }
        if (ret == status::success)
            VDISPATCH_GRAPH_SDP(
                    "sdpa is dispatched to (%s)", kernel->str().c_str());
        else
            VDISPATCH_GRAPH_SDP("sdpa is failed to dispatch");
        return ret;
    }

    // It is used to check if enable the decomposition kernel based on user's
    // env and params. Decomposition kernel is enabled when:
    // - CPU runtime is OMP or THREADPOOl.
    // - Primitive based implementation is not forced by the internal env var.
    bool enable_decomp_kernel() const {
#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_OMP \
        || DNNL_CPU_RUNTIME == DNNL_RUNTIME_THREADPOOL
        const bool force_prim = force_primitive();
        return !force_prim;
#else
        return false;
#endif
    }

    // An internal env var is provided to force using primitive based SDPA
    // implementation and skipping ukernel based optimization on GPU or
    // decomposition based optimization on CPU. Currently it's for oneDNN debug
    // and testing only.
    bool force_primitive() const {
        const int force = graph::utils::getenv_int_internal(
                "GRAPH_SDPA_FORCE_PRIMITIVE", 0);
        return force > 0;
    }

    status_t execute_impl(const stream_t *g_stream,
            const std::vector<tensor_t> &inputs,
            const std::vector<tensor_t> &outputs) override {
        return kernel->execute_impl(g_stream, inputs, outputs);
    }

#ifdef DNNL_WITH_SYCL
    status_t sycl_execute_impl(const stream_t *g_stream,
            const std::vector<tensor_t> &inputs,
            const std::vector<tensor_t> &outputs,
            const std::vector<::sycl::event> &sycl_deps,
            ::sycl::event *sycl_event) override {
        return kernel->sycl_execute_impl(
                g_stream, inputs, outputs, sycl_deps, sycl_event);
    }
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    status_t ocl_execute_impl(const stream_t *g_stream,
            const std::vector<tensor_t> &inputs,
            const std::vector<tensor_t> &outputs,
            const std::vector<cl_event> &deps, cl_event *event) override {
        return kernel->ocl_execute_impl(g_stream, inputs, outputs, deps, event);
    }
#endif

    std::string str() const override { return kernel->str(); }
};
} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
