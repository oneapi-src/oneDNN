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

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/nstl.hpp"
#include "common/type_helpers.hpp"

#include "cpu/x64/jit_uni_pooling.hpp"

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
        const int c_off = ((jpp.tag_kind == jptg_nspc) ? jpp.c_block : 1) * b_c;

        arg.src = (const void *)&src[src_d.blk_off(n, c_off, ih)];
        arg.dst = (const void *)&dst[dst_d.blk_off(n, c_off, oh)];
        if (indices) {
            const size_t ind_off = indices_d.blk_off(n, c_off, oh);
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
        const int c_off = ((jpp.tag_kind == jptg_nspc) ? jpp.c_block : 1) * b_c;

        arg.src = &src[src_d.blk_off(n, c_off, id, ih)];
        arg.dst = &dst[dst_d.blk_off(n, c_off, od, oh)];
        if (indices) {
            const size_t ind_off = indices_d.blk_off(n, c_off, od, oh);
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

namespace jit_uni_pooling_utils {
struct trans_wrapper_t {
    trans_wrapper_t(data_type_t inp_dt, dim_t inp_str, data_type_t out_dt,
            dim_t out_str, dim_t ysize, dim_t xsize)
        : inp_dt_size_(types::data_type_size(inp_dt))
        , out_dt_size_(types::data_type_size(out_dt))
        , inp_str_(inp_str)
        , out_str_(out_str)
        , nb_x_(xsize / 8)
        , nb_y_(ysize / 8)
        , x_tail_(xsize % 8)
        , y_tail_(ysize % 8) {
        using namespace cpu::tr;

        auto create_ker = [=](dim_t ys, dim_t y_inp_str, dim_t y_out_str,
                                  dim_t xs, dim_t x_inp_str, dim_t x_out_str) {
            tr::prb_t prb;
            kernel_t::desc_t desc;

            prb.ndims = 2;
            prb.ioff = 0;
            prb.ooff = 0;
            prb.scale_type = scale_type_t::NONE;
            prb.beta = 0;
            prb.nodes[0].ss = prb.nodes[1].ss = 1;

            prb.itype = inp_dt;
            prb.otype = out_dt;

            prb.nodes[0].n = ys;
            prb.nodes[0].is = y_inp_str;
            prb.nodes[0].os = y_out_str;

            prb.nodes[1].n = xs;
            prb.nodes[1].is = x_inp_str;
            prb.nodes[1].os = x_out_str;

            kernel_t::desc_init(desc, prb, 2);
            return kernel_t::create(desc);
        };

        if (nb_x_ * nb_y_ > 0)
            ker_.reset(create_ker(8, inp_str_, 1, 8, 1, out_str_));

        if (x_tail_)
            ker_x_tail_.reset(create_ker(8, inp_str_, 1, x_tail_, 1, out_str_));

        if (y_tail_)
            ker_y_tail_.reset(
                    create_ker(y_tail_, inp_str_, 1, xsize, 1, out_str_));
    }

    void exec(const void *inp, void *out) {
        dim_t x_blocked = nb_x_ * 8;
        dim_t y_blocked = nb_y_ * 8;

        auto call_ker = [&](tr::kernel_t &ker, dim_t inp_y, dim_t inp_x,
                                dim_t out_y, dim_t out_x) {
            tr::call_param_t cp;
            cp.scale = 0;

            dim_t inp_off = (inp_y * inp_str_ + inp_x) * inp_dt_size_;
            dim_t out_off = (out_y * out_str_ + out_x) * out_dt_size_;
            cp.in = (uint8_t *)inp + inp_off;
            cp.out = (uint8_t *)out + out_off;
            (ker)(&cp);
        };

        for (dim_t by = 0; by < nb_y_; by++) {
            for (dim_t bx = 0; bx < nb_x_; bx++)
                call_ker(*ker_, 8 * by, 8 * bx, 8 * bx, 8 * by);

            if (x_tail_)
                call_ker(*ker_x_tail_, 8 * by, x_blocked, x_blocked, 8 * by);
        }
        if (y_tail_) call_ker(*ker_y_tail_, y_blocked, 0, 0, y_blocked);
    }

    ~trans_wrapper_t() {}

private:
    std::unique_ptr<tr::kernel_t> ker_;
    std::unique_ptr<tr::kernel_t> ker_x_tail_;
    std::unique_ptr<tr::kernel_t> ker_y_tail_;

    const size_t inp_dt_size_;
    const size_t out_dt_size_;

    const dim_t inp_str_;
    const dim_t out_str_;
    const dim_t nb_x_;
    const dim_t nb_y_;
    const dim_t x_tail_;
    const dim_t y_tail_;
};
} // namespace jit_uni_pooling_utils
template <cpu_isa_t isa, data_type_t d_type>
jit_uni_pooling_bwd_t<isa, d_type>::jit_uni_pooling_bwd_t(const pd_t *apd)
    : primitive_t(apd)
    , diff_dst_trans_(nullptr)
    , diff_dst_tail_trans_(nullptr)
    , ind_trans_(nullptr)
    , ind_tail_trans_(nullptr)
    , diff_src_trans_(nullptr)
    , diff_src_tail_trans_(nullptr) {
    kernel_ = new jit_uni_pool_kernel<isa>(pd()->jpp_);

    const auto &jpp = pd()->jpp_;
    if (jpp.tag_kind == jptg_ncsp) {
        using namespace jit_uni_pooling_utils;
        auto diff_src_sp = (dim_t)jpp.id * jpp.ih * jpp.iw;
        auto diff_dst_sp = (dim_t)jpp.od * jpp.oh * jpp.ow;
        dim_t nb_c = jpp.c_without_padding / jpp.c_block;
        dim_t c_tail = jpp.c_without_padding % jpp.c_block;
        const memory_desc_wrapper indices_d(pd()->workspace_md());
        bool have_indices = indices_d.data_type() != data_type::undef;

        if (nb_c) {
            diff_dst_trans_ = new trans_wrapper_t(d_type, diff_dst_sp, wsp_dt_,
                    jpp.c_block, jpp.c_block, diff_dst_sp);
            diff_src_trans_ = new trans_wrapper_t(wsp_dt_, jpp.c_block, d_type,
                    diff_src_sp, diff_src_sp, jpp.c_block);
            if (have_indices)
                ind_trans_ = new trans_wrapper_t(indices_d.data_type(),
                        diff_dst_sp, indices_d.data_type(), jpp.c_block,
                        jpp.c_block, diff_dst_sp);
        }
        if (c_tail) {
            diff_dst_tail_trans_ = new trans_wrapper_t(d_type, diff_dst_sp,
                    wsp_dt_, jpp.c_block, c_tail, diff_dst_sp);
            diff_src_tail_trans_ = new trans_wrapper_t(wsp_dt_, jpp.c_block,
                    d_type, diff_src_sp, diff_src_sp, c_tail);
            if (have_indices)
                ind_tail_trans_ = new trans_wrapper_t(indices_d.data_type(),
                        diff_dst_sp, indices_d.data_type(), jpp.c_block, c_tail,
                        diff_dst_sp);
        }
    }
}

template <cpu_isa_t isa, data_type_t d_type>
jit_uni_pooling_bwd_t<isa, d_type>::~jit_uni_pooling_bwd_t() {
    delete kernel_;

    delete diff_dst_trans_;
    delete diff_dst_tail_trans_;
    delete ind_trans_;
    delete ind_tail_trans_;
    delete diff_src_trans_;
    delete diff_src_tail_trans_;
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

    auto diff_src_sp = (dim_t)jpp.id * jpp.ih * jpp.iw;
    auto diff_src_slice = diff_src_sp * jpp.c_block;
    auto diff_dst_sp = (dim_t)jpp.od * jpp.oh * jpp.ow;
    auto diff_dst_slice = diff_dst_sp * jpp.c_block;
    dim_t c_tail = jpp.c_without_padding % jpp.c_block;

    // scratchpad for c_block slice of src and/or dst
    typedef typename prec_traits<wsp_dt_>::type wsp_data_t;

    wsp_data_t *__restrict cvt_slice_src_wsp {nullptr};
    wsp_data_t *__restrict cvt_slice_dst_wsp {nullptr};
    char *__restrict cvt_slice_ind_wsp {nullptr};
    auto scratchpad = ctx.get_scratchpad_grantor();

    // if spatial size == 1 && jpp.c_without_padding != jpp.c
    // then we don't need to transpose data
    bool transpose_diff_src = jpp.tag_kind == jptg_ncsp
            && (diff_src_sp > 1 || jpp.c_without_padding != jpp.c
                    || d_type != wsp_dt_);
    bool transpose_diff_dst = jpp.tag_kind == jptg_ncsp
            && (indices || diff_dst_sp > 1 || jpp.c_without_padding != jpp.c
                    || d_type != wsp_dt_);

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

        auto c_off = jpp.is_plain() ? b_c * jpp.c_block : b_c;
        if (transpose_diff_src) {
            wsp_data_t *wsp_ = cvt_slice_src_wsp + ithr * diff_src_slice;
            arg.src = (const void *)&wsp_[ih * jpp.iw * jpp.c_block];
        } else
            arg.src = &diff_src[diff_src_d.blk_off(n, c_off, ih)];

        if (transpose_diff_dst) {
            wsp_data_t *wsp_ = cvt_slice_dst_wsp + ithr * diff_dst_slice;
            arg.dst = (const void *)&wsp_[oh * jpp.ow * jpp.c_block];
        } else
            arg.dst = &diff_dst[diff_dst_d.blk_off(n, c_off, oh)];

        if (indices) {
            if (transpose_diff_dst) {
                char *wsp_ = cvt_slice_ind_wsp
                        + ithr * diff_dst_slice * ind_dt_size;
                arg.indices = (const void *)&wsp_[oh * jpp.ow * jpp.c_block
                        * ind_dt_size];
            } else {
                const size_t ind_off = indices_d.blk_off(n, c_off, oh);
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

    using namespace jit_uni_pooling_utils;
    auto trans_exec = [&](trans_wrapper_t *trans, trans_wrapper_t *trans_tail,
                              dim_t cs, const void *inp, void *out) {
        if (cs == jpp.c_block)
            trans->exec(inp, out);
        else
            trans_tail->exec(inp, out);
    };

    auto process_block = [&](int ithr, int n, int b_c) {
        if (transpose_diff_src) {
            const wsp_data_t zero_val = 0;
            wsp_data_t *src_diff_base_ptr
                    = cvt_slice_src_wsp + ithr * diff_src_slice;
            for (dim_t idx = 0; idx < diff_src_slice; ++idx)
                src_diff_base_ptr[idx] = zero_val;
        } else {
            const data_t zero_val = 0;
            auto ch_off = jpp.is_plain() ? b_c * jpp.c_block : b_c;
            data_t *src_diff_base_ptr
                    = &diff_src[diff_src_d.blk_off(n, ch_off, 0)];
            if (jpp.tag_kind == jptg_nspc) {
                auto sp = (ptrdiff_t)jpp.ih * jpp.iw;
                for (ptrdiff_t x = 0; x < sp; x++) {
                    PRAGMA_OMP_SIMD()
                    for (ptrdiff_t c = 0; c < jpp.c_block; c++)
                        src_diff_base_ptr[x * jpp.c + c] = zero_val;
                }
            } else {
                for (dim_t idx = 0; idx < diff_src_slice; ++idx)
                    src_diff_base_ptr[idx] = zero_val;
            }
        }

        dim_t cs = nstl::min(
                jpp.c_without_padding - b_c * jpp.c_block, jpp.c_block);

        if (transpose_diff_dst) {
            wsp_data_t *wsp_ = cvt_slice_dst_wsp + ithr * diff_dst_slice;
            const data_t *diff_dst_
                    = diff_dst + diff_dst_d.blk_off(n, b_c * jpp.c_block, 0);
            const char *indices_ = indices ? indices
                            + ind_dt_size
                                    * indices_d.blk_off(n, b_c * jpp.c_block, 0)
                                           : nullptr;
            char *indices_wsp_ = indices
                    ? cvt_slice_ind_wsp + ithr * diff_dst_slice * ind_dt_size
                    : nullptr;
            trans_exec(
                    diff_dst_trans_, diff_dst_tail_trans_, cs, diff_dst_, wsp_);
            if (indices)
                trans_exec(ind_trans_, ind_tail_trans_, cs, indices_,
                        indices_wsp_);
        }
        for (int oh = 0; oh < jpp.oh; ++oh)
            ker(ithr, n, b_c, oh);
        if (transpose_diff_src) {
            wsp_data_t *wsp_ = cvt_slice_src_wsp + ithr * diff_src_slice;
            data_t *diff_src_
                    = diff_src + diff_src_d.blk_off(n, b_c * jpp.c_block, 0);
            trans_exec(
                    diff_src_trans_, diff_src_tail_trans_, cs, wsp_, diff_src_);
        }
    };

    parallel(0, [&](int ithr, int nthr) {
        const size_t work_amount = (size_t)jpp.mb * jpp.nb_c;
        if ((size_t)ithr >= work_amount) return;

        if (transpose_diff_dst && c_tail != 0) {
            wsp_data_t *__restrict wsp_ptr
                    = cvt_slice_dst_wsp + ithr * diff_dst_slice;
            for_(dim_t s = 0; s < diff_dst_sp; s++)
            for (dim_t c = c_tail; c < jpp.c_block; c++)
                wsp_ptr[s * jpp.c_block + c] = 0.f;

            char *__restrict ind_ptr
                    = cvt_slice_ind_wsp + ithr * diff_dst_slice * ind_dt_size;
            for_(dim_t s = 0; s < diff_dst_sp; s++)
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
        const int c_off = ((jpp.tag_kind == jptg_nspc) ? jpp.c_block : 1) * b_c;

        arg.src = (const void
                        *)&diff_src[diff_src_d.blk_off(n, c_off, id + kd, ih)];
        arg.dst = (const void *)&diff_dst[diff_dst_d.blk_off(n, c_off, od, oh)];
        if (indices) {
            const size_t ind_off = indices_d.blk_off(n, c_off, od, oh);
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
