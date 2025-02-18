/*******************************************************************************
 * Copyright 2024-2025 Intel Corporation
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

#ifndef GRAPH_BACKEND_DNNL_KERNELS_KERNEL_BASE_HPP
#define GRAPH_BACKEND_DNNL_KERNELS_KERNEL_BASE_HPP

#include <algorithm>
#include <memory>
#include <vector>

#include "graph/interface/c_types_map.hpp"
#include "graph/interface/logical_tensor.hpp"

// required for dnnl::engine
#include "oneapi/dnnl/dnnl.hpp"

#ifdef DNNL_WITH_SYCL
#include "oneapi/dnnl/dnnl_sycl.hpp"
#endif

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {

class dnnl_partition_impl_t;

struct kernel_base_t {
    virtual ~kernel_base_t() = default;

    status_t compile(const dnnl_partition_impl_t *part, const engine_t *aengine,
            const std::vector<logical_tensor_t> &inputs,
            const std::vector<logical_tensor_t> &outputs);

    // each subclass should implement compile_impl()
    virtual status_t compile_impl(const dnnl_partition_impl_t *part,
            const engine_t *aengine,
            const std::vector<logical_tensor_t> &inputs,
            const std::vector<logical_tensor_t> &outputs)
            = 0;

    status_t execute(const stream_t *astream,
            const std::vector<tensor_t> &inputs,
            const std::vector<tensor_t> &outputs);

    // each subclass should implement execute_impl()
    virtual status_t execute_impl(const stream_t *astream,
            const std::vector<tensor_t> &inputs,
            const std::vector<tensor_t> &outputs)
            = 0;

#ifdef DNNL_WITH_SYCL
    status_t execute_sycl(const stream_t *astream,
            const std::vector<tensor_t> &inputs,
            const std::vector<tensor_t> &outputs,
            const std::vector<::sycl::event> &sycl_deps,
            ::sycl::event *sycl_event) {
        return sycl_execute_impl(
                astream, inputs, outputs, sycl_deps, sycl_event);
    }

    virtual status_t sycl_execute_impl(const stream_t *astream,
            const std::vector<tensor_t> &inputs,
            const std::vector<tensor_t> &outputs,
            const std::vector<::sycl::event> &sycl_deps,
            ::sycl::event *sycl_event)
            = 0;
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    status_t execute_ocl(const stream_t *astream,
            const std::vector<tensor_t> &inputs,
            const std::vector<tensor_t> &outputs,
            const std::vector<cl_event> &ocl_deps, cl_event *ocl_event) {
        return ocl_execute_impl(astream, inputs, outputs, ocl_deps, ocl_event);
    }

    virtual status_t ocl_execute_impl(const stream_t *astream,
            const std::vector<tensor_t> &inputs,
            const std::vector<tensor_t> &outputs,
            const std::vector<cl_event> &ocl_deps, cl_event *ocl_event)
            = 0;
#endif

    virtual status_t prepare_inplace_pairs_impl() { return status::success; };

    // A string identity used in verbose indicating which kernels is dispatched
    // for a compiled partition.
    virtual std::string str() const = 0;

    bool enabled_constant_cache() const;

    size_t encode_constant_cache_key(
            const std::vector<tensor_t> &inputs, size_t cache_key) const;

    const std::vector<inplace_pair_t> &get_inplace_pairs() const;

protected:
    std::vector<inplace_pair_t> inplace_pairs_;
    dnnl::engine p_engine_;
};

using kernel_ptr = std::shared_ptr<kernel_base_t>;
using FCreateKernel = std::function<kernel_ptr(void)>;

#define DEF_KERNEL_METHOD_STR(name) \
    std::string str() const override { return #name; }

} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
