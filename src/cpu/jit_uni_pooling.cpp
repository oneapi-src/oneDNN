/*******************************************************************************
* Copyright 2017-2020 Intel Corporation
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

#include "dnnl_types.h"

#include "c_types_map.hpp"
#include "nstl.hpp"
#include "type_helpers.hpp"

#include "jit_uni_pooling.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

template <cpu_isa_t isa, data_type_t d_type>
void jit_uni_pooling_fwd_t<isa, d_type>::execute_forward(
        const data_t *src, data_t *dst, char *indices) const {
    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper dst_d(pd()->dst_md());
    const memory_desc_wrapper indices_d(pd()->workspace_md());
    const size_t ind_dt_size
            = indices ? types::data_type_size(indices_d.data_type()) : 0;

    const auto &jpp = pd()->jpp_;

    auto ker = [&](int n, int b_c, int oh) {
        auto arg = jit_pool_call_s();

        const int ij = oh * jpp.stride_h;
        const int i_t_overflow = nstl::max(0, jpp.t_pad - ij);
        const int i_b_overflow
                = nstl::max(jpp.ih, ij + jpp.kh - jpp.t_pad) - jpp.ih;
        const int ih = nstl::max(ij - jpp.t_pad, 0);
        assert(IMPLICATION(pd()->ndims() == 3, utils::everyone_is(0, ih, oh)));

        arg.src = (const void *)&src[src_d.blk_off(n, b_c, ih)];
        arg.dst = (const void *)&dst[dst_d.blk_off(n, b_c, oh)];
        if (indices) {
            const size_t ind_off = indices_d.blk_off(n, b_c, oh);
            arg.indices = (const void *)&indices[ind_off * ind_dt_size];
        }
        arg.oh = oh == 0;
        arg.kh_padding = jpp.kh - i_t_overflow - i_b_overflow;
        arg.kh_padding_shift = i_t_overflow * jpp.kw;
        arg.ker_area_h = (float)(jpp.kh
                - nstl::max(0, oh * jpp.stride_h - jpp.t_pad + jpp.kh - jpp.ih)
                - nstl::max(0, jpp.t_pad - oh * jpp.stride_h));
        (*kernel_)(&arg);
    };

    parallel_nd(jpp.mb, jpp.nb_c, jpp.oh,
            [&](int n, int b_c, int oh) { ker(n, b_c, oh); });
}

template <cpu_isa_t isa, data_type_t d_type>
void jit_uni_pooling_fwd_t<isa, d_type>::execute_forward_3d(
        const data_t *src, data_t *dst, char *indices) const {
    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper dst_d(pd()->dst_md());
    const memory_desc_wrapper indices_d(pd()->workspace_md());
    const size_t ind_dt_size
            = indices ? types::data_type_size(indices_d.data_type()) : 0;

    const auto &jpp = pd()->jpp_;

    auto ker = [&](int n, int b_c, int od, int oh, int id, int d_t_overflow,
                       int d_b_overflow) {
        auto arg = jit_pool_call_s();

        const int ij = oh * jpp.stride_h;
        const int i_t_overflow = nstl::max(0, jpp.t_pad - ij);
        const int i_b_overflow
                = nstl::max(jpp.ih, ij + jpp.kh - jpp.t_pad) - jpp.ih;
        const int ih = nstl::max(ij - jpp.t_pad, 0);

        arg.src = &src[src_d.blk_off(n, b_c, id, ih)];
        arg.dst = &dst[dst_d.blk_off(n, b_c, od, oh)];
        if (indices) {
            const size_t ind_off = indices_d.blk_off(n, b_c, od, oh);
            arg.indices = &indices[ind_off * ind_dt_size];
        }
        arg.oh = (oh + od == 0);
        arg.kd_padding = jpp.kd - d_t_overflow - d_b_overflow;
        arg.kh_padding = jpp.kh - i_t_overflow - i_b_overflow;
        arg.kh_padding_shift
                = i_t_overflow * jpp.kw + d_t_overflow * jpp.kw * jpp.kh;
        arg.kd_padding_shift = (i_t_overflow + i_b_overflow) * jpp.kw;
        arg.ker_area_h = (float)(jpp.kh
                                 - nstl::max(0,
                                         oh * jpp.stride_h - jpp.t_pad + jpp.kh
                                                 - jpp.ih)
                                 - nstl::max(0, jpp.t_pad - oh * jpp.stride_h))
                * (jpp.kd
                        - nstl::max(0,
                                od * jpp.stride_d - jpp.f_pad + jpp.kd - jpp.id)
                        - nstl::max(0, jpp.f_pad - od * jpp.stride_d));

        (*kernel_)(&arg);
    };

    parallel_nd(jpp.mb, jpp.nb_c, jpp.od, [&](int n, int b_c, int od) {
        const int ik = od * jpp.stride_d;
        const int d_t_overflow = nstl::max(0, jpp.f_pad - ik);
        const int d_b_overflow
                = nstl::max(jpp.id, ik + jpp.kd - jpp.f_pad) - jpp.id;
        const int id = nstl::max(ik - jpp.f_pad, 0);
        for (int oh = 0; oh < jpp.oh; ++oh) {
            ker(n, b_c, od, oh, id, d_t_overflow, d_b_overflow);
        }
    });
}

template <cpu_isa_t isa, data_type_t d_type>
void jit_uni_pooling_bwd_t<isa, d_type>::execute_backward(
        const data_t *diff_dst, const char *indices, data_t *diff_src) const {
    const memory_desc_wrapper diff_src_d(pd()->diff_src_md());
    const memory_desc_wrapper diff_dst_d(pd()->diff_dst_md());
    const memory_desc_wrapper indices_d(pd()->workspace_md());
    const size_t ind_dt_size
            = indices ? types::data_type_size(indices_d.data_type()) : 0;

    const auto &jpp = pd()->jpp_;

    auto ker = [&](int n, int b_c, int oh) {
        auto arg = jit_pool_call_s();

        const int ij = oh * jpp.stride_h;
        const int i_t_overflow = nstl::max(0, jpp.t_pad - ij);
        const int i_b_overflow
                = nstl::max(jpp.ih, ij + jpp.kh - jpp.t_pad) - jpp.ih;
        const int ih = nstl::max(ij - jpp.t_pad, 0);
        assert(IMPLICATION(pd()->ndims() == 3, utils::everyone_is(0, ih, oh)));
        assert(pd()->ndims() != 3 || utils::everyone_is(0, ih, oh));

        arg.src = &diff_src[diff_src_d.blk_off(n, b_c, ih)];
        arg.dst = &diff_dst[diff_dst_d.blk_off(n, b_c, oh)];
        if (indices) {
            const size_t ind_off = indices_d.blk_off(n, b_c, oh);
            arg.indices = &indices[ind_off * ind_dt_size];
        }
        arg.oh = (oh == 0);
        arg.kh_padding = jpp.kh - i_t_overflow - i_b_overflow;
        arg.kh_padding_shift = i_t_overflow * jpp.kw;
        arg.ker_area_h = (float)(jpp.kh
                - nstl::max(0, oh * jpp.stride_h - jpp.t_pad + jpp.kh - jpp.ih)
                - nstl::max(0, jpp.t_pad - oh * jpp.stride_h));

        (*kernel_)(&arg);
    };

    const data_t zero_val = 0;
    parallel_nd(jpp.mb, jpp.nb_c, [&](int n, int b_c) {
        auto src_diff_base_ptr = &diff_src[diff_src_d.blk_off(n, b_c, 0)];
        auto block_size = (ptrdiff_t)jpp.ih * (ptrdiff_t)jpp.iw
                * (ptrdiff_t)jpp.c_block;

        for (ptrdiff_t idx = 0; idx != block_size; ++idx) {
            src_diff_base_ptr[idx] = zero_val;
        }

        for (int oh = 0; oh < jpp.oh; ++oh) {
            ker(n, b_c, oh);
        }
    });
}

template <cpu_isa_t isa, data_type_t d_type>
void jit_uni_pooling_bwd_t<isa, d_type>::execute_backward_3d(
        const data_t *diff_dst, const char *indices, data_t *diff_src) const {
    const memory_desc_wrapper diff_src_d(pd()->diff_src_md());
    const memory_desc_wrapper diff_dst_d(pd()->diff_dst_md());
    const memory_desc_wrapper indices_d(pd()->workspace_md());
    const size_t ind_dt_size
            = indices ? types::data_type_size(indices_d.data_type()) : 0;

    const auto &jpp = pd()->jpp_;

    auto ker = [&](int n, int b_c, int od, int oh, int id, int d_t_overflow,
                       int d_b_overflow, int zero_size, int kd) {
        auto arg = jit_pool_call_s();

        const int ij = oh * jpp.stride_h;
        const int i_t_overflow = nstl::max(0, jpp.t_pad - ij);
        const int i_b_overflow
                = nstl::max(jpp.ih, ij + jpp.kh - jpp.t_pad) - jpp.ih;
        const int ih = nstl::max(ij - jpp.t_pad, 0);

        arg.src = (const void
                        *)&diff_src[diff_src_d.blk_off(n, b_c, id + kd, ih)];
        arg.dst = (const void *)&diff_dst[diff_dst_d.blk_off(n, b_c, od, oh)];
        if (indices) {
            const size_t ind_off = indices_d.blk_off(n, b_c, od, oh);
            arg.indices = (const void *)&indices[ind_off * ind_dt_size];
        }
        arg.oh = zero_size;
        arg.kd_padding = jpp.kd - d_t_overflow - d_b_overflow;
        arg.kh_padding = jpp.kh - i_t_overflow - i_b_overflow;
        arg.kh_padding_shift = i_t_overflow * jpp.kw
                + d_t_overflow * jpp.kw * jpp.kh + kd * jpp.kw * jpp.kh;
        arg.kd_padding_shift = (i_t_overflow + i_b_overflow) * jpp.kw;
        arg.ker_area_h = (float)(jpp.kh
                                 - nstl::max(0,
                                         oh * jpp.stride_h - jpp.t_pad + jpp.kh
                                                 - jpp.ih)
                                 - nstl::max(0, jpp.t_pad - oh * jpp.stride_h))
                * (jpp.kd
                        - nstl::max(0,
                                od * jpp.stride_d - jpp.f_pad + jpp.kd - jpp.id)
                        - nstl::max(0, jpp.f_pad - od * jpp.stride_d));

        (*kernel_)(&arg);
    };

    const data_t zero_val = 0;
    if (jpp.simple_alg) {

        const int neg_back_pad
                = -(jpp.od - 1) * jpp.stride_d - jpp.kd + jpp.f_pad + jpp.id;

        parallel_nd(jpp.mb, jpp.nb_c, jpp.od, [&](int n, int b_c, int od) {
            const int ik = od * jpp.stride_d;
            const int d_t_overflow = nstl::max(0, jpp.f_pad - ik);
            const int d_b_overflow
                    = nstl::max(jpp.id, ik + jpp.kd - jpp.f_pad) - jpp.id;
            const int id = nstl::max(ik - jpp.f_pad, 0);
            int zero_s = jpp.stride_d - d_t_overflow
                    - (nstl::max(jpp.id, ik + jpp.stride_d - jpp.f_pad)
                            - jpp.id);
            for (int oh = 0; oh < jpp.oh; ++oh) {
                ker(n, b_c, od, oh, id, d_t_overflow, d_b_overflow,
                        (oh == 0) ? zero_s : 0, 0);
            }
            // zero-out untouched portion of diff_src when back_pad is negative
            if (neg_back_pad > 0 && od == jpp.od - 1) {

                auto blk_start_ptr = &diff_src[diff_src_d.blk_off(
                        n, b_c, jpp.id - neg_back_pad, 0, 0)];
                auto blk_size = neg_back_pad * jpp.ih * jpp.iw;

                for (auto blk_idx = 0; blk_idx < blk_size; ++blk_idx) {
                    auto blk_ptr = blk_start_ptr + blk_idx * jpp.c_block;

                    PRAGMA_OMP_SIMD()
                    for (auto ch_idx = 0; ch_idx < jpp.c_block; ++ch_idx)
                        blk_ptr[ch_idx] = zero_val;
                }
            }
        });
    } else {
        const size_t chunk_size
                = (size_t)jpp.id * jpp.ih * jpp.iw * jpp.c_block;
        parallel_nd(jpp.mb, jpp.nb_c, [&](int n, int b_c) {
            const size_t offset = ((size_t)n * jpp.nb_c + b_c) * chunk_size;
            PRAGMA_OMP_SIMD()
            for (size_t idx = 0; idx < chunk_size; ++idx)
                diff_src[offset + idx] = zero_val;
        });

        for (int kd = 0; kd < jpp.kd; ++kd) {
            parallel_nd(jpp.mb, jpp.nb_c, [&](int n, int b_c) {
                for (int od = 0; od < jpp.od; ++od) {
                    const int ik = od * jpp.stride_d;
                    const int d_t_overflow = nstl::max(0, jpp.f_pad - ik);
                    const int d_b_overflow
                            = nstl::max(jpp.id, ik + jpp.kd - jpp.f_pad)
                            - jpp.id;
                    if (kd >= jpp.kd - d_t_overflow - d_b_overflow) continue;
                    const int id = nstl::max(ik - jpp.f_pad, 0);
                    for (int oh = 0; oh < jpp.oh; ++oh) {
                        ker(n, b_c, od, oh, id, d_t_overflow, d_b_overflow, 0,
                                kd);
                    }
                }
            });
        }
    }
}

template struct jit_uni_pooling_fwd_t<sse41, data_type::f32>;
template struct jit_uni_pooling_bwd_t<sse41, data_type::f32>;
template struct jit_uni_pooling_fwd_t<avx, data_type::f32>;
template struct jit_uni_pooling_bwd_t<avx, data_type::f32>;
template struct jit_uni_pooling_fwd_t<avx512_common, data_type::f32>;
template struct jit_uni_pooling_bwd_t<avx512_common, data_type::f32>;
template struct jit_uni_pooling_fwd_t<avx512_core, data_type::bf16>;
template struct jit_uni_pooling_bwd_t<avx512_core, data_type::bf16>;

} // namespace cpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
