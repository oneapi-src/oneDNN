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

#include "graph/backend/dnnl/kernels/pure/gen_index.hpp"

#include "common/stream.hpp"
#include "graph/backend/dnnl/passes/compile_ops.hpp"
#include "graph/backend/dnnl/passes/constant_propagation.hpp"
#include "graph/backend/dnnl/passes/insert_ops.hpp"
#include "graph/backend/dnnl/passes/layout_propagation.hpp"
#include "graph/backend/dnnl/passes/lower.hpp"
#include "graph/backend/dnnl/passes/memory_planning.hpp"
#include "graph/backend/dnnl/passes/transform.hpp"
#include "graph/backend/dnnl/passes/utils.hpp"
#include "xpu/sycl/engine_impl.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL
struct genindex_kernel_vec_t {
    genindex_kernel_vec_t(void *dst, dim_t axis, const logical_tensor_t &src_lt,
            const logical_tensor_t &dst_lt)
        : dst(dst), axis(axis), src_lt(src_lt), dst_lt(dst_lt) {}
    void operator()(::sycl::nd_item<1> item) const {
        int id = item.get_global_id(0);
        int ndims = src_lt.ndims;
        const dnnl_dims_t &dst_dims = dst_lt.dims;
        const dnnl_dims_t &dst_strides = dst_lt.layout.strides;
        dim_t offdst = 0, result = 0;
        for (int i = 0; i < ndims; i++) {
            // calculate the idx on each dimension
            int idx = id % dst_dims[i];
            if ((const dim_t)i == axis) result = idx;
            id /= dst_dims[i];
            offdst += dst_strides[i] * idx;
        }
        static_cast<int32_t *>(dst)[offdst] = result;
    }

private:
    void *dst;
    const dim_t axis;
    const logical_tensor_t src_lt, dst_lt;
};
#endif

status_t genindex_t::compile_impl(const dnnl_partition_impl_t *part,
        const engine_t *g_engine, const std::vector<logical_tensor_t> &inputs,
        const std::vector<logical_tensor_t> &outputs) {
    p_engine_ = make_dnnl_engine(*g_engine);
    const auto &ops = part->get_ops();
    genindex_cfg_.nelems = ltw(inputs[0]).nelems();
    genindex_cfg_.axis
            = ops[0]->get_attr<int64_t>(dnnl::impl::graph::op_attr::axis);
    if (g_engine->kind() == engine_kind::gpu) {
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL
        using kernel_bundle_e_t
                = ::sycl::kernel_bundle<::sycl::bundle_state::executable>;
        const auto kid = ::sycl::get_kernel_id<genindex_kernel_vec_t>();

        auto ctx = dnnl::impl::utils::downcast<
                const dnnl::impl::xpu::sycl::engine_impl_t *>(g_engine->impl())
                           ->context();
        auto input_bundle
                = ::sycl::get_kernel_bundle<::sycl::bundle_state::input>(
                        ctx, {kid});
        auto exe_bundle = ::sycl::build(input_bundle);
        kernel_bundle_
                = dnnl::impl::utils::make_unique<kernel_bundle_e_t>(exe_bundle);
#endif
    }

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
    } else if (p_engine_.get()->kind() == engine_kind::gpu) {
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL
        dnnl::stream p_stream;
        p_stream.reset(const_cast<stream_t *>(g_stream), true); // not own
        auto *sycl_stream_impl = dnnl::impl::utils::downcast<
                dnnl::impl::xpu::sycl::stream_impl_t *>(p_stream.get()->impl());
        auto &queue = *sycl_stream_impl->queue();
        auto &deps = sycl_stream_impl->sycl_ctx().get_sycl_deps().events;

        auto event = queue.submit([&](::sycl::handler &cgh) {
            cgh.depends_on(deps);
            cgh.use_kernel_bundle(*kernel_bundle_);

            genindex_kernel_vec_t genindex_kernel(outputs[0].get_data_handle(),
                    genindex_cfg_.axis, inputs[0].get_logical_tensor(),
                    outputs[0].get_logical_tensor());
            cgh.parallel_for(::sycl::nd_range<1>(genindex_cfg_.nelems, 32),
                    genindex_kernel);
        });
        if (sycl_event) *sycl_event = event;
#endif
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
