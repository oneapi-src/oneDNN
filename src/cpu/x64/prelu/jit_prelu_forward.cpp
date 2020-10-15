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
#include <cmath>
#include "common/memory_desc_wrapper.hpp"
#include "common/type_helpers.hpp"
#include "cpu/x64/cpu_isa_traits.hpp"
#include "cpu/x64/prelu/jit_prelu_forward.hpp"
#include "cpu/x64/prelu/jit_prelu_utils.hpp"
#include "cpu/x64/prelu/jit_uni_prelu_forward_kernel.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

status_t jit_prelu_forward_t::pd_t::init(engine_t *engine) {
    const memory_desc_wrapper src_d {src_md()};
    const memory_desc_wrapper weights_d {weights_md()};

    const bool ok = is_fwd() && dt_supported(src_d, weights_d)
            && set_default_formats() && bcast_supported(src_d, weights_d)
            && !has_zero_dim_memory() && src_d.is_dense(true)
            && weights_d.is_dense(true) && attr()->has_default_values()
            && utils::one_of(prelu::get_supported_isa(), avx512_core_bf16,
                    avx512_core, avx512_common, avx2, avx, sse41);

    return ok ? status::success : status::unimplemented;
}

bool jit_prelu_forward_t::pd_t::dt_supported(const memory_desc_wrapper &src_d,
        const memory_desc_wrapper &weights_d) const noexcept {
    const auto &src_dt = src_d.data_type();
    const auto &weights_dt = weights_d.data_type();

    return utils::everyone_is(src_dt, weights_dt)
            && utils::one_of(src_dt, data_type::bf16, data_type::f32)
            && IMPLICATION(src_dt == data_type::bf16, mayiuse(avx512_core));
}

bool jit_prelu_forward_t::pd_t::bcast_supported(
        const memory_desc_wrapper &src_d,
        const memory_desc_wrapper &weights_d) const {

    const auto bcast = prelu::get_bcast_type(src_d, weights_d);
    if (bcast == prelu::bcast::full)
        return true;
    else if (bcast == prelu::bcast::unsupported)
        return false;
    else if (bcast == prelu::bcast::per_oc_blocked) {
        const int simd_w = mayiuse(avx512_common) ? 16 : (mayiuse(avx) ? 8 : 4);
        const auto check_block_consistency
                = [&](const memory_desc_wrapper &mdw) {
                      const auto &bd = mdw.blocking_desc();

                      return bd.inner_nblks == 1 && bd.inner_blks[0] == simd_w
                              && bd.inner_idxs[0] == 1;
                  };

        return check_block_consistency(src_d)
                && check_block_consistency(weights_d);
    } else {
        const auto &src_strides = src_d.blocking_desc().strides;
        const auto &weights_strides = weights_d.blocking_desc().strides;
        // C should be on second position in tag (example nchw or ncw) or on
        // last postion (nhwc)
        return src_strides[0] >= src_strides[1]
                && IMPLICATION(
                        src_strides[1] > 1, src_strides[1] >= src_strides[2])
                && weights_strides[0] >= weights_strides[1];
    }

    return true;
}

const jit_prelu_forward_t::pd_t *jit_prelu_forward_t::pd() const {
    return static_cast<const pd_t *>(primitive_t::pd().get());
}

jit_prelu_forward_t::jit_prelu_forward_t(const pd_t *apd) : primitive_t(apd) {}
jit_prelu_forward_t::~jit_prelu_forward_t() = default;

status_t jit_prelu_forward_t::init(engine_t *engine) {
    CHECK(safe_ptr_assign(kernel_, jit_prelu_forward_kernel_t::create(pd())));
    return kernel_->create_kernel();
}

status_t jit_prelu_forward_t::execute(const exec_ctx_t &ctx) const {
    using byte = unsigned char;
    const byte *const src = CTX_IN_MEM(const byte *, DNNL_ARG_SRC);
    const byte *const weights = CTX_IN_MEM(const byte *, DNNL_ARG_WEIGHTS);
    byte *const dst = CTX_OUT_MEM(byte *, DNNL_ARG_DST);
    const memory_desc_wrapper src_d {pd()->src_md()};
    const auto dt_size = types::data_type_size(src_d.data_type());
    const auto kernel = kernel_.get();
    const auto bcast = kernel->get_bcast();
    const auto ndims = src_d.ndims();
    const auto &dims = src_d.dims();
    const dim_t MB = dims[0];
    const dim_t C = ndims >= 2 ? dims[1] : 1;
    const dim_t D = ndims >= 5 ? dims[ndims - 3] : 1;
    const dim_t H = ndims >= 4 ? dims[ndims - 2] : 1;
    const dim_t W = ndims >= 3 ? dims[ndims - 1] : 1;
    const dim_t SP = D * H * W;

    if (bcast == prelu::bcast::full) {
        const auto nelems = src_d.nelems(true);
        const auto simd_w = kernel->simd_w();
        const auto res = std::div(nelems, simd_w);
        const auto &nelems_simd = res.quot;
        const auto &nelems_tail = res.rem;
        const auto nelems_parallel = nelems_simd + (nelems_tail ? 1 : 0);

        parallel(0, [&](const int ithr, const int nthr) {
            dim_t start = 0, end = 0;
            balance211(nelems_parallel, nthr, ithr, start, end);
            if (start >= end) return;

            const bool ithr_process_tail
                    = nelems_tail && end == nelems_parallel;
            const auto n_simd_size = (end - start - ithr_process_tail) * simd_w;
            const auto offset = start * simd_w * dt_size;

            jit_prelu_forward_kernel_t::call_params_t params;

            params.compute_data_size
                    = (n_simd_size + (nelems_tail ? nelems_tail : 0)) * dt_size;
            params.src = src + offset;
            params.weights = weights + offset;
            params.dst = dst + offset;

            (*kernel)(&params);
        });
    } else {

        const dim_t nelems_single_mb
                = utils::array_product(src_d.padded_dims() + 1, ndims - 1);

        if (bcast == prelu::bcast::per_oc_n_spatial_c) {
            parallel_nd(MB, SP, [&](dim_t mb, dim_t sp) {
                const auto offset = (mb * nelems_single_mb + sp * C) * dt_size;
                jit_prelu_forward_kernel_t::call_params_t params;
                params.compute_data_size = C * dt_size;
                params.src = src + offset;
                params.weights = weights;
                params.dst = dst + offset;
                (*kernel)(&params);
            });
        } else if (bcast == prelu::bcast::per_oc_n_c_spatial) {
            parallel_nd(MB, C, [&](dim_t mb, dim_t c) {
                jit_prelu_forward_kernel_t::call_params_t params;
                const auto offset = (mb * nelems_single_mb + c * SP) * dt_size;
                params.compute_data_size = SP * dt_size;
                params.src = src + offset;
                params.weights = weights + c * dt_size;
                params.dst = dst + offset;
                (*kernel)(&params);
            });
        } else if (bcast == prelu::bcast::per_oc_blocked) {
            const auto simd_w = kernel->simd_w();
            const dim_t C_blocks = std::ceil(static_cast<float>(C) / simd_w);

            parallel_nd(MB, C_blocks, [&](dim_t mb, dim_t c_blk) {
                jit_prelu_forward_kernel_t::call_params_t params;
                params.compute_data_size = SP * simd_w * dt_size;
                const dim_t offset
                        = (mb * nelems_single_mb + c_blk * SP * simd_w)
                        * dt_size;
                params.src = src + offset;
                params.weights = weights + c_blk * simd_w * dt_size;
                params.dst = dst + offset;
                (*kernel)(&params);
            });
        }
    }
    return status::success;
}

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl