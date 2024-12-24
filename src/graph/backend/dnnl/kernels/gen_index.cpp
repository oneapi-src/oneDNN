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
#include "graph/backend/dnnl/kernels/gen_index.hpp"

#include "graph/backend/dnnl/passes/compile_ops.hpp"
#include "graph/backend/dnnl/passes/constant_propagation.hpp"
#include "graph/backend/dnnl/passes/insert_ops.hpp"
#include "graph/backend/dnnl/passes/layout_propagation.hpp"
#include "graph/backend/dnnl/passes/lower.hpp"
#include "graph/backend/dnnl/passes/memory_planning.hpp"
#include "graph/backend/dnnl/passes/transform.hpp"
#include "graph/backend/dnnl/passes/utils.hpp"
namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {

status_t genindex_t::compile_impl(const dnnl_partition_impl_t *part,
        const engine_t *g_engine, const std::vector<logical_tensor_t> &inputs,
        const std::vector<logical_tensor_t> &outputs) {
    p_engine_ = make_dnnl_engine(*g_engine);
    const auto &ops = part->get_ops();
    genindex_cfg_.nelems = ltw(inputs[0]).nelems();
    genindex_cfg_.axis
            = ops[0]->get_attr<int64_t>(dnnl::impl::graph::op_attr::axis);
    return status::success;
}
status_t genindex_t::execute_impl(const stream_t *g_stream,
        const std::vector<tensor_t> &inputs,
        const std::vector<tensor_t> &outputs) {
    parallel_nd(genindex_cfg_.nelems, [&](dim_t i) {
        dims_t dims_src; // decomposition for physical offsets
        auto dst = outputs[0].get_logical_tensor();
        dnnl::impl::utils::l_dims_by_l_offset(
                dims_src, i, ltw(dst).dims(), ltw(dst).ndims());
        auto strides_dst = ltw(dst).vstrides();
        auto offset = offset_compute(strides_dst, dims_src);
        auto output = static_cast<int32_t *>(outputs[0].get_data_handle());
        output[offset] = dims_src[genindex_cfg_.axis];
    });
    return status::success;
}
#ifdef DNNL_WITH_SYCL
status_t genindex_t::sycl_execute_impl(const stream_t *g_stream,
        const std::vector<tensor_t> &inputs,
        const std::vector<tensor_t> &outputs,
        const std::vector<::sycl::event> &sycl_deps,
        ::sycl::event *sycl_event) {
    if (p_engine_.get()->kind() == engine_kind::cpu) {
        parallel_nd(genindex_cfg_.nelems, [&](dim_t i) {
            dims_t dims_src; // decomposition for physical offsets
            auto dst = outputs[0].get_logical_tensor();
            dnnl::impl::utils::l_dims_by_l_offset(
                    dims_src, i, ltw(dst).dims(), ltw(dst).ndims());
            auto strides_dst = ltw(dst).vstrides();
            auto offset = offset_compute(strides_dst, dims_src);
            auto output = static_cast<int32_t *>(outputs[0].get_data_handle());
            output[offset] = dims_src[genindex_cfg_.axis];
        });
    } else if (p_engine_.get()->kind() == engine_kind::cpu) {
    } else
        return status::unimplemented;
    return status::success;
}
#endif
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
status_t genindex_t::ocl_execute_impl(const stream_t *g_stream,
        const std::vector<tensor_t> &inputs,
        const std::vector<tensor_t> &outputs,
        const std::vector<cl_event> &ocl_deps, cl_event *ocl_event) {
    return status::success;
}
#endif
} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl
