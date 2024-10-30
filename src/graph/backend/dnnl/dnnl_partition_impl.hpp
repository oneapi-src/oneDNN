/*******************************************************************************
* Copyright 2021-2024 Intel Corporation
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

#ifndef GRAPH_BACKEND_DNNL_DNNL_PARTITION_IMPL_HPP
#define GRAPH_BACKEND_DNNL_DNNL_PARTITION_IMPL_HPP

#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <unordered_set>

#include "graph/interface/backend.hpp"
#include "graph/interface/partition.hpp"

#include "graph/backend/dnnl/dnnl_backend.hpp"
#include "graph/backend/dnnl/internal_ops.hpp"

#include "graph/backend/dnnl/kernels/kernel_base.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {

class dnnl_compiled_partition_impl_t : public compiled_partition_impl_t {
    friend class dnnl_backend_t;
    friend class dnnl_partition_impl_t;

public:
    dnnl_compiled_partition_impl_t(const engine_t &engine,
            const std::vector<logical_tensor_t> &inputs,
            const std::vector<logical_tensor_t> &outputs, kernel_ptr &kernel)
        : compiled_partition_impl_t(
                engine, inputs, outputs, kernel->get_inplace_pairs())
        , kernel_(kernel) {}

    status_t execute(const stream_t *g_stream,
            const std::vector<tensor_t> &inputs,
            const std::vector<tensor_t> &outputs) override {
        // We don't need to resort the inputs and outputs
        return kernel_->execute(g_stream, inputs, outputs);
    }

#ifdef DNNL_WITH_SYCL
    status_t execute_sycl(const stream_t *g_stream,
            const std::vector<tensor_t> &inputs,
            const std::vector<tensor_t> &outputs,
            const std::vector<::sycl::event> &sycl_deps,
            ::sycl::event *sycl_event) override {
        // We don't need to resort the inputs and outputs
        return kernel_->execute_sycl(
                g_stream, inputs, outputs, sycl_deps, sycl_event);
    }
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    // It looks very similar to execute_sycl. Consider to merge them in the
    // future.
    status_t execute_ocl(const stream_t *g_stream,
            const std::vector<tensor_t> &inputs,
            const std::vector<tensor_t> &outputs,
            const std::vector<cl_event> &ocl_deps,
            cl_event *ocl_event) override {
        return kernel_->execute_ocl(
                g_stream, inputs, outputs, ocl_deps, ocl_event);
    }
#endif

    std::string str() const override { return kernel_->str(); }

private:
    kernel_ptr kernel_;
};

class dnnl_partition_impl_t : public partition_impl_t {
    friend class dnnl_backend_t;

public:
    dnnl_partition_impl_t(engine_kind_t engine_kind,
            const fpmath_t &fpmath_mode, partition_kind_t pkind)
        : partition_impl_t(engine_kind, fpmath_mode, pkind) {}

    ~dnnl_partition_impl_t() override = default;

    ///// The following are used only in backend for constructing object
    void init(FCreateKernel kernel_creator);
    void add_op(const std::shared_ptr<op_t> &op);

    // init backend partition's input/output logical tensors
    // based on ops in the partition
    void init_inputs_outputs();

    FCreateKernel get_kernel_creator() const;

    /////////////// the followings are the implementation of interface

    bool is_initialized() const override { return kernel_creator_ != nullptr; }

    std::shared_ptr<partition_impl_t> clone() const override;

    const backend_t *get_assigned_backend() const override;

    status_t compile(compiled_partition_t *compiled_partition,
            const std::vector<logical_tensor_t> &inputs,
            const std::vector<logical_tensor_t> &outputs,
            const engine_t *g_engine) const override;

    status_t infer_shape(std::vector<const logical_tensor_t *> &inputs,
            std::vector<logical_tensor_t *> &outputs) const override;

private:
    FCreateKernel kernel_creator_;
};

} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
