/*******************************************************************************
* Copyright 2025 Intel Corporation
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

#include "genindex.hpp"

using namespace dnnl::impl::gpu::intel;
#define MAX_NDIMS 6

void create_genindex_kernel(const compute::compute_engine_t *compute_engine,
        compute::kernel_t &kernel, const int ndims,
        const dnnl::impl::graph::dims_t output_dims,
        const dnnl::impl::graph::dims_t output_strides) {
    compute::kernel_ctx_t kernel_ctx;
    kernel_ctx.define_int("NDIMS", ndims);
    for (int d = 0; d < MAX_NDIMS; ++d) {
        dnnl::impl::graph::dim_t dim = (d < ndims) ? output_dims[d] : 1;
        dnnl::impl::graph::dim_t stride = (d < ndims) ? output_strides[d] : 0;
        kernel_ctx.define_int(dnnl::impl::utils::format("D%d", d), dim);
        kernel_ctx.define_int(dnnl::impl::utils::format("S%d", d), stride);
    }
    std::vector<compute::kernel_t> kernels(1);
    compute_engine->create_kernels(&kernels, {"gen_index"}, kernel_ctx);
    kernel = kernels[0];
}

void execute_genindex_kernel(compute::compute_stream_t *compute_stream,
        const compute::kernel_t &kernel,
        const dnnl::impl::memory_storage_t &dst, const int nelems,
        const int axis) {
    compute::range_t gws = {static_cast<size_t>(nelems)};
    auto nd_range = compute::nd_range_t(gws);
    compute::kernel_arg_list_t arg_list;

    arg_list.set(0, dst);
    arg_list.set(1, axis);

    kernel.parallel_for(*compute_stream, nd_range, arg_list,
            compute_stream->ctx().get_deps(), compute_stream->ctx().get_deps());
}
