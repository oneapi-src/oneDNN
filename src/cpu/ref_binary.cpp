/*******************************************************************************
* Copyright 2019 Intel Corporation
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

#include <assert.h>
#include <float.h>
#include <math.h>

#include "c_types_map.hpp"
#include "dnnl_thread.hpp"
#include "type_helpers.hpp"

#include "ref_binary.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

template <impl::data_type_t data_type>
void ref_binary_t<data_type>::execute_ref(const exec_ctx_t &ctx) const {
    const auto src0 = CTX_IN_MEM(const data_t *, DNNL_ARG_SRC_0);
    const auto src1 = CTX_IN_MEM(const data_t *, DNNL_ARG_SRC_1);
    auto dst = CTX_OUT_MEM(data_t *, DNNL_ARG_DST);

    const memory_desc_wrapper src0_d(pd()->src_md(0));
    const memory_desc_wrapper src1_d(pd()->src_md(1));

    const auto alg = pd()->desc()->alg_kind;
    auto perform_op = [&](data_t *d, data_t x, data_t y) {
        if (alg == alg_kind::binary_add) {
            *d = x + y;
        } else if (alg == alg_kind::binary_mul) {
            *d = x * y;
        } else {
            assert(!"not supported operation!");
        }
    };

    const dims_t &dims_bcast = pd()->broadcast_dims();
    const dims_t &dims_A = src0_d.dims();
    const int ndims = pd()->ndims();
    const auto nelems_A = src0_d.nelems();

    auto map_idx_B = [&](dim_t off) {
        dims_t dims;
        for (int d = ndims - 1; d >= 0; --d) {
            dims[d] = off % dims_A[d];
            off /= dims_A[d];
        }
        assert(off == 0);

        for (int d = 0; d < ndims; ++d) {
            dims[d] *= (!dims_bcast[d]);
        }

        return src1_d.off_v(dims);
    };

    parallel_nd(nelems_A, [&](dim_t i) {
        auto off_A = src0_d.off_l(i);
        auto off_B = pd()->is_tensor_op() ? src1_d.off_l(i) : map_idx_B(i);
        perform_op(&dst[off_A], src0[off_A], src1[off_B]);
    });
}

template struct ref_binary_t<data_type::f32>;
template struct ref_binary_t<data_type::bf16>;

} // namespace cpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
