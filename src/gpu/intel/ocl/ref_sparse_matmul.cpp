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

#include "gpu/intel/ocl/ref_sparse_matmul.hpp"

#include "common/c_types_map.hpp"
#include "common/type_helpers.hpp"
#include "gpu/intel/compute/utils.hpp"
namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace ocl {

status_t ref_sparse_matmul_t::execute_ref(const exec_ctx_t &ctx) const {
    const auto &a_values = CTX_IN_STORAGE(DNNL_ARG_SRC, 0);
    const auto &a_rows = CTX_IN_STORAGE(DNNL_ARG_SRC, 1);
    const auto &a_cols = CTX_IN_STORAGE(DNNL_ARG_SRC, 2);

    const auto a_d = ctx.memory_mdw(DNNL_ARG_SRC, pd()->src_md());
    const auto c_d = ctx.memory_mdw(DNNL_ARG_DST, pd()->dst_md());
    const dim_t nnz = a_d.nnz();

    const auto &b = CTX_IN_STORAGE(DNNL_ARG_WEIGHTS);
    auto &c = CTX_OUT_STORAGE(DNNL_ARG_DST);

    const dim_t M = c_d.dims()[0];
    const dim_t N = c_d.dims()[1];

    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, a_values);
    arg_list.set(1, a_rows);
    arg_list.set(2, a_cols);
    arg_list.set(3, b);
    arg_list.set(4, c);
    arg_list.set(5, nnz);
    compute::range_t gws = {(size_t)M, (size_t)N};

    auto nd_range = compute::nd_range_t(gws);

    status_t status = parallel_for(ctx, nd_range, kernel_, arg_list);
    return status;
}

} // namespace ocl
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
