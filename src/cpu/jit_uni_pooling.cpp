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
jit_uni_pooling_bwd_t<isa, d_type>::jit_uni_pooling_bwd_t(const pd_t *apd)
    : primitive_impl_t(apd)
    , trans_inp_kernel_(nullptr)
    , trans_out_kernel_(nullptr)
    , trans_inp_tail_kernel_(nullptr)
    , trans_out_tail_kernel_(nullptr)
    , trans_ind_kernel_(nullptr)
    , trans_ind_tail_kernel_(nullptr) {
    kernel_ = new jit_uni_pool_kernel<isa>(pd()->jpp_);

    const memory_desc_wrapper diff_src_d(pd()->diff_src_md());
    const memory_desc_wrapper diff_dst_d(pd()->diff_dst_md());
    if (diff_src_d.is_plain() && diff_dst_d.is_plain()) {
        using namespace cpu::tr;
        const auto &jpp = pd()->jpp_;
        auto diff_src_sp_size = (dim_t)jpp.id * jpp.ih * jpp.iw;
        auto diff_dst_sp_size = (dim_t)jpp.od * jpp.oh * jpp.ow;
        dim_t c_tail = jpp.c_without_padding % jpp.c_block;

        cpu::tr::prb_t inp_prb, out_prb;
        inp_prb.ndims = out_prb.ndims = 2;
        inp_prb.ioff = out_prb.ioff = 0;
        inp_prb.ooff = out_prb.ooff = 0;
        inp_prb.scale_type = out_prb.scale_type = scale_type_t::NONE;
        inp_prb.beta = out_prb.beta = 0;

        inp_prb.itype = d_type;
        inp_prb.otype = data_type::f32;
        // channels
        inp_prb.nodes[0].n = jpp.c_block;
        inp_prb.nodes[0].is = diff_dst_sp_size;
        inp_prb.nodes[0].os = 1;
        inp_prb.nodes[0].ss = 1;
        // spatial
        inp_prb.nodes[1].n = diff_dst_sp_size;
        inp_prb.nodes[1].is = 1;
        inp_prb.nodes[1].os = jpp.c_block;
        inp_prb.nodes[1].ss = 1;

        out_prb.itype = data_type::f32;
        out_prb.otype = d_type;
        // spatial
        out_prb.nodes[0].n = diff_src_sp_size;
        out_prb.nodes[0].is = jpp.c_block;
        out_prb.nodes[0].os = 1;
        out_prb.nodes[0].ss = 1;
        // channels
        out_prb.nodes[1].n = jpp.c_block;
        out_prb.nodes[1].is = 1;
        out_prb.nodes[1].os = diff_src_sp_size;
        out_prb.nodes[1].ss = 1;

        kernel_t::desc_t inp_desc, out_desc;

        kernel_t::desc_init(inp_desc, inp_prb, 2);
        kernel_t::desc_init(out_desc, out_prb, 2);

        trans_inp_kernel_ = kernel_t::create(inp_desc);
        trans_out_kernel_ = kernel_t::create(out_desc);

        if (c_tail != 0) {
            // Tails
            cpu::tr::prb_t inp_tail_prb, out_tail_prb;

            inp_tail_prb = inp_prb;
            out_tail_prb = out_prb;

            // channels
            inp_tail_prb.nodes[0].n = c_tail;

            // channels
            out_tail_prb.nodes[1].n = c_tail;

            kernel_t::desc_t inp_tail_desc, out_tail_desc;

            kernel_t::desc_init(inp_tail_desc, inp_tail_prb, 2);
            kernel_t::desc_init(out_tail_desc, out_tail_prb, 2);

            trans_inp_tail_kernel_ = kernel_t::create(inp_tail_desc);
            trans_out_tail_kernel_ = kernel_t::create(out_tail_desc);
        }
        const memory_desc_wrapper indices_d(pd()->workspace_md());
        const size_t ind_size = indices_d.size();
        if (ind_size) {
            cpu::tr::prb_t ind_prb = inp_prb;
            ind_prb.itype = indices_d.data_type();
            ind_prb.otype = indices_d.data_type();

            kernel_t::desc_t ind_desc;

            kernel_t::desc_init(ind_desc, ind_prb, 2);

            trans_ind_kernel_ = kernel_t::create(ind_desc);
            if (c_tail != 0) {
                cpu::tr::prb_t ind_tail_prb = ind_prb;

                // channels
                ind_tail_prb.nodes[0].n = c_tail;

                kernel_t::desc_t ind_tail_desc;

                kernel_t::desc_init(ind_tail_desc, ind_tail_prb, 2);

                trans_ind_tail_kernel_ = kernel_t::create(ind_tail_desc);
            }
        }
    }
}

template <cpu_isa_t isa, data_type_t d_type>
void jit_uni_pooling_bwd_t<isa, d_type>::execute_backward(
        const data_t *diff_dst, const char *indices, data_t *diff_src,
        const exec_ctx_t &ctx) const {
    const memory_desc_wrapper diff_src_d(pd()->diff_src_md());
    const memory_desc_wrapper diff_dst_d(pd()->diff_dst_md());
    const memory_desc_wrapper indices_d(pd()->workspace_md());
    const size_t ind_dt_size
            = indices ? types::data_type_size(indices_d.data_type()) : 0;

    const auto &jpp = pd()->jpp_;

    auto diff_src_sp_size = (dim_t)jpp.id * jpp.ih * jpp.iw;
    auto diff_src_slice_size = diff_src_sp_size * jpp.c_block;
    auto diff_dst_sp_size = (dim_t)jpp.od * jpp.oh * jpp.ow;
    auto diff_dst_slice_size = diff_dst_sp_size * jpp.c_block;
    dim_t c_tail = jpp.c_without_padding % jpp.c_block;

    // scratchpad for c_block slice of src and/or dst
    const data_type_t wsp_dt = data_type::f32; //  d_type;
    typedef typename prec_traits<wsp_dt>::type wsp_data_t;

    wsp_data_t *__restrict cvt_slice_src_wsp {nullptr};
    wsp_data_t *__restrict cvt_slice_dst_wsp {nullptr};
    char *__restrict cvt_slice_ind_wsp {nullptr};
    auto scratchpad = ctx.get_scratchpad_grantor();

    // if spatial size == 1 && jpp.c_without_padding != jpp.c
    // then we don't need to transpose data
    bool transpose_diff_src = diff_src_d.is_plain()
            && (diff_src_sp_size > 1 || jpp.c_without_padding != jpp.c
                    || d_type != wsp_dt);
    bool transpose_diff_dst = diff_dst_d.is_plain()
            && (indices || diff_dst_sp_size > 1
                    || jpp.c_without_padding != jpp.c || d_type != wsp_dt);

    if (transpose_diff_src)
        cvt_slice_src_wsp = scratchpad.template get<wsp_data_t>(
                memory_tracking::names::key_pool_src_plain2blocked_cvt);
    if (transpose_diff_dst) {
        cvt_slice_dst_wsp = scratchpad.template get<wsp_data_t>(
                memory_tracking::names::key_pool_dst_plain2blocked_cvt);
        cvt_slice_ind_wsp = scratchpad.template get<char>(
                memory_tracking::names::key_pool_ind_plain2blocked_cvt);
    }

    auto ker = [&](int ithr, int n, int b_c, int oh) {
        auto arg = jit_pool_call_s();

        const int ij = oh * jpp.stride_h;
        const int i_t_overflow = nstl::max(0, jpp.t_pad - ij);
        const int i_b_overflow
                = nstl::max(jpp.ih, ij + jpp.kh - jpp.t_pad) - jpp.ih;
        const int ih = nstl::max(ij - jpp.t_pad, 0);
        assert(IMPLICATION(pd()->ndims() == 3, utils::everyone_is(0, ih, oh)));
        assert(pd()->ndims() != 3 || utils::everyone_is(0, ih, oh));

        if (diff_src_d.is_plain()) {
            if (transpose_diff_src) {
                wsp_data_t *wsp_
                        = cvt_slice_src_wsp + ithr * diff_src_slice_size;
                arg.src = (const void *)&wsp_[ih * jpp.iw * jpp.c_block];
            } else
                arg.src = diff_src
                        + diff_src_d.blk_off(n, b_c * jpp.c_block, ih);
        } else
            arg.src = &diff_src[diff_src_d.blk_off(n, b_c, ih)];

        if (diff_dst_d.is_plain()) {
            if (transpose_diff_dst) {
                wsp_data_t *wsp_
                        = cvt_slice_dst_wsp + ithr * diff_dst_slice_size;
                arg.dst = (const void *)&wsp_[oh * jpp.ow * jpp.c_block];
            } else
                arg.dst = diff_dst
                        + diff_dst_d.blk_off(n, b_c * jpp.c_block, oh);
        } else
            arg.dst = &diff_dst[diff_dst_d.blk_off(n, b_c, oh)];

        if (indices) {
            if (transpose_diff_dst) {
                char *wsp_ = cvt_slice_ind_wsp
                        + ithr * diff_dst_slice_size * ind_dt_size;
                arg.indices = (const void *)&wsp_[oh * jpp.ow * jpp.c_block
                        * ind_dt_size];
            } else {
                const size_t ind_off = indices_d.blk_off(n, b_c, oh);
                arg.indices = &indices[ind_off * ind_dt_size];
            }
        }
        arg.oh = (oh == 0);
        arg.kh_padding = jpp.kh - i_t_overflow - i_b_overflow;
        arg.kh_padding_shift = i_t_overflow * jpp.kw;
        arg.ker_area_h = (float)(jpp.kh
                - nstl::max(0, oh * jpp.stride_h - jpp.t_pad + jpp.kh - jpp.ih)
                - nstl::max(0, jpp.t_pad - oh * jpp.stride_h));

        (*kernel_)(&arg);
    };

    auto trans_exec = [&](tr::kernel_t *trans_ker, const void *inp, void *out) {
        tr::call_param_t cp;
        cp.in = inp;
        cp.out = out;
        cp.scale = 0;
        trans_ker->operator()(&cp);
    };

    auto process_block = [&](int ithr, int n, int b_c) {
        if (diff_src_d.is_plain()) {
            if (transpose_diff_src) {
                const wsp_data_t zero_val = 0;
                wsp_data_t *src_diff_base_ptr
                        = cvt_slice_src_wsp + ithr * diff_src_slice_size;
                for (dim_t idx = 0; idx < diff_src_slice_size; ++idx)
                    src_diff_base_ptr[idx] = zero_val;
            } else {
                const data_t zero_val = 0;
                data_t *src_diff_base_ptr = &diff_src[diff_src_d.blk_off(
                        n, jpp.c_block * b_c, 0)];
                for (dim_t idx = 0; idx < diff_src_slice_size; ++idx)
                    src_diff_base_ptr[idx] = zero_val;
            }
        } else {
            const data_t zero_val = 0;
            data_t *src_diff_base_ptr
                    = &diff_src[diff_src_d.blk_off(n, b_c, 0)];
            for (dim_t idx = 0; idx < diff_src_slice_size; ++idx)
                src_diff_base_ptr[idx] = zero_val;
        }

        if (transpose_diff_dst) {
            wsp_data_t *__restrict wsp_
                    = cvt_slice_dst_wsp + ithr * diff_dst_slice_size;
            const data_t *__restrict diff_dst_
                    = diff_dst + diff_dst_d.blk_off(n, b_c * jpp.c_block, 0);

            const char *__restrict indices_ = indices ? indices
                            + ind_dt_size
                                    * indices_d.blk_off(n, b_c * jpp.c_block, 0)
                                                      : nullptr;
            char *__restrict indices_wsp_ = indices ? cvt_slice_ind_wsp
                            + ithr * diff_dst_slice_size * ind_dt_size
                                                    : nullptr;
            if (b_c < jpp.nb_c - 1 || c_tail == 0) {
                trans_exec(trans_inp_kernel_, diff_dst_, wsp_);
                if (indices)
                    trans_exec(trans_ind_kernel_, indices_, indices_wsp_);
            } else {
                trans_exec(trans_inp_tail_kernel_, diff_dst_, wsp_);
                if (indices)
                    trans_exec(trans_ind_tail_kernel_, indices_, indices_wsp_);
            }
        }
        for (int oh = 0; oh < jpp.oh; ++oh)
            ker(ithr, n, b_c, oh);
        if (transpose_diff_src) {
            wsp_data_t *__restrict wsp_
                    = cvt_slice_src_wsp + ithr * diff_src_slice_size;
            data_t *__restrict diff_src_
                    = diff_src + diff_src_d.blk_off(n, b_c * jpp.c_block, 0);
            if (b_c < jpp.nb_c - 1 || c_tail == 0)
                trans_exec(trans_out_kernel_, wsp_, diff_src_);
            else
                trans_exec(trans_out_tail_kernel_, wsp_, diff_src_);
        }
    };

    parallel(0, [&](int ithr, int nthr) {
        const size_t work_amount = (size_t)jpp.mb * jpp.nb_c;
        if (ithr >= work_amount) return;

        if (diff_dst_d.is_plain() && c_tail != 0) {
            wsp_data_t *__restrict wsp_ptr
                    = cvt_slice_dst_wsp + ithr * diff_dst_slice_size;
            for_(dim_t s = 0; s < diff_dst_sp_size; s++)
            for (dim_t c = c_tail; c < jpp.c_block; c++)
                wsp_ptr[s * jpp.c_block + c] = 0.f;

            char *__restrict ind_ptr = cvt_slice_ind_wsp
                    + ithr * diff_dst_slice_size * ind_dt_size;
            for_(dim_t s = 0; s < diff_dst_sp_size; s++)
            for_(dim_t c = c_tail; c < jpp.c_block; c++)
            for (size_t i = 0; i < ind_dt_size; i++)
                ind_ptr[(s * jpp.c_block + c) * ind_dt_size + i] = 0;
        }

        size_t start {0}, end {0};
        balance211(work_amount, nthr, ithr, start, end);
        int n {0}, b_c {0};
        utils::nd_iterator_init(start, n, jpp.mb, b_c, jpp.nb_c);
        for (size_t iwork = start; iwork < end; ++iwork) {
            process_block(ithr, n, b_c);
            utils::nd_iterator_step(n, jpp.mb, b_c, jpp.nb_c);
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
