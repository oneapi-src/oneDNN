/*******************************************************************************
* Copyright 2017 Intel Corporation
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

#include "mkldnn_types.h"

#include "c_types_map.hpp"
#include "mkldnn_thread.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

#include "jit_generator.hpp"

#include "jit_avx512_core_u8s8u8_convolution.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace Xbyak;

using namespace mkldnn::impl::status;
using namespace mkldnn::impl::memory_format;
using namespace mkldnn::impl::utils;

struct jit_avx512_core_u8s8u8_conv_fwd_ker_t: public jit_generator {
    typedef jit_avx512_core_u8s8u8_convolution_fwd_t::src_data_t src_data_t;
    typedef jit_avx512_core_u8s8u8_convolution_fwd_t::wei_data_t wei_data_t;
    typedef jit_avx512_core_u8s8u8_convolution_fwd_t::acc_data_t acc_data_t;
    typedef jit_avx512_core_u8s8u8_convolution_fwd_t::dst_data_t dst_data_t;

    enum { STATE_LOAD_DST_U8 = 0x1U };

    struct call_params_t {
        const src_data_t *src_u8;
        const wei_data_t *wei_s8;
        const char *bia_c8;
        const dst_data_t *dst_u8;
        const int *dst_s32;
        size_t kh_range;
    };

    void (*ker_)(const call_params_t *);
    jit_conv_conf_t c_;

    Reg64 reg_kh = rax;
    Reg64 reg_ic_b2 = rbx;

    Reg32 reg_state = esi;

    Reg64 reg_off_src_u8 = r8;
    Reg64 reg_off_dst_u8 = r9;
    Reg64 reg_off_dst_s32 = r10;

    Reg64 reg_ptr_src_u8 = r11;
    Reg64 reg_ptr_wei_s8 = r12;
    Reg64 reg_ptr_bia_c8 = r13;
    Reg64 reg_ptr_dst_u8 = r14;
    Reg64 reg_ptr_dst_s32 = r15;

    Zmm vreg_src_u8 = zmm29;
    Zmm vreg_zero = zmm30;
    Zmm vreg_one_s16 = zmm31;

    int id_vreg_dst(int o) {
        assert(o < c_.ur_ow_max);
        return c_.ic_nb1 * c_.kw + o;
    }

    Zmm vreg_wei_s8(int ic_b1, int k) {
        const int id_reg_wei = ic_b1 * c_.kw + k;
        assert(id_reg_wei < c_.ic_nb1 * c_.kw);
        return Zmm(id_reg_wei);
    }

    Zmm vreg_dst_s32(int o) {
        return Zmm(id_vreg_dst(o));
    }

    void load_wei_s8();
    void load_dst_s32(int ur_ow);
    void store_dst(int ur_ow);

    void compute(int o, int iw_off, int k);
    void compute_part_ur_ow_oc_block(int ur_ow, int iw_start);
    void compute_part_ow_oc_block();
    void compute_ow_oc_block();
    void generate();

    jit_avx512_core_u8s8u8_conv_fwd_ker_t(const jit_conv_conf_t &c): c_(c) {
        generate();
        ker_ = reinterpret_cast<decltype(ker_)>(const_cast<uint8_t*>(
                        getCode()));
    }

    static status_t init_conf(jit_conv_conf_t &c, const convolution_desc_t &cd,
            const memory_desc_wrapper &src_d, const memory_desc_wrapper &wei_d,
            const memory_desc_wrapper &dst_d, bool with_relu,
            double negative_slope);
};

void jit_avx512_core_u8s8u8_conv_fwd_ker_t::load_wei_s8() {
    assert(c_.oc_block * c_.ic_block * sizeof(wei_data_t)
            == cpu_isa_traits<avx512_mic>::vlen);
    for (int ic_b1 = 0; ic_b1 < c_.ic_nb1; ++ic_b1) {
        for (int kw = 0; kw < c_.kw; ++kw) {
            const int off = (ic_b1 * c_.kw + kw) * c_.oc_block * c_.ic_block;
            vmovups(vreg_wei_s8(ic_b1, kw), ptr[reg_ptr_wei_s8
                    + off * sizeof(wei_data_t)]);
        }
    }
}

void jit_avx512_core_u8s8u8_conv_fwd_ker_t::load_dst_s32(int ur_ow) {
    using namespace data_type;

    Label l_load_u8, l_ret;
    test(reg_state, STATE_LOAD_DST_U8);
    jne(l_load_u8, T_NEAR);

    for (int o = 0; o < ur_ow; ++o)
        vmovups(vreg_dst_s32(o), ptr[reg_ptr_dst_s32 + reg_off_dst_s32
                + o * c_.oc_block * sizeof(acc_data_t)]);
    jmp(l_ret, T_NEAR);

    L(l_load_u8);
    if (c_.with_bias) {
        switch (c_.bia_dt) {
            case s8: vpmovsxbd(vreg_dst_s32(0), ptr[reg_ptr_bia_c8]); break;
            case u8: vpmovzxbd(vreg_dst_s32(0), ptr[reg_ptr_bia_c8]); break;
            case s32: vmovups(vreg_dst_s32(0), zword [reg_ptr_bia_c8]); break;
            default: assert(!"unsupported bias data type");
        }
        for (int o = 1; o < ur_ow; ++o)
            vmovaps(vreg_dst_s32(o), vreg_dst_s32(0));
    } else {
        for (int o = 0; o < ur_ow; ++o)
            vpxord(vreg_dst_s32(o), vreg_dst_s32(o), vreg_dst_s32(o));
    }

    L(l_ret);
}

void jit_avx512_core_u8s8u8_conv_fwd_ker_t::store_dst(int ur_ow) {
    Label l_store_u8, l_ret;

    add(reg_ic_b2, reg_kh); /* non-destructive check on 0 */
    je(l_store_u8, T_NEAR); /* jump if ic_b2 == 0 && kh == 0 */

    sub(reg_ic_b2, reg_kh); /* recover reg_ic_b2 */

    for (int o = 0; o < ur_ow; ++o)
        vmovups(ptr[reg_ptr_dst_s32 + reg_off_dst_s32
                + o * c_.oc_block * sizeof(acc_data_t)], vreg_dst_s32(o));

    jmp(l_ret, T_NEAR);

    L(l_store_u8);
    for (int o = 0; o < ur_ow; ++o) {
        const int r = id_vreg_dst(o);
        vpmaxsd(Zmm(r), vreg_zero, Zmm(r));
        vpmovusdb(Xmm(r), Zmm(r));
        vmovups(ptr[reg_ptr_dst_u8 + reg_off_dst_u8
                + o * c_.ngroups * c_.oc * sizeof(dst_data_t)], Xmm(r));
    }
    add(reg_off_dst_u8, ur_ow * c_.ngroups * c_.oc * sizeof(dst_data_t));

    L(l_ret);
}

/** computes:
 *      i_u8 [ic_nb1]         [4i]
 * (*)  w_s8 [ic_nb1][kw][16o][4i]
 * (+=) --------------------------
 *      o_s32            [16o]
 *
 * parameters:
 *   i_u8 = input[reg_ptr_src_u8 + reg_off_src_u8 + width:iw_off]
 *   w_s8 = weights[width:k]
 *
 * assumptions:
 *   ic_block == 4i
 *   oc_block == 16o
 */
void jit_avx512_core_u8s8u8_conv_fwd_ker_t::compute(int o, int iw_off, int k) {
    assert(0 <= k && k < c_.kw);
    assert(0 <= o && o < c_.ur_ow_max);

    Zmm vreg_t_s16 = vreg_src_u8;
    Zmm vreg_t_s32 = vreg_src_u8;

    for (int ic_b1 = 0; ic_b1 < c_.ic_nb1; ++ic_b1) {
        const int off = iw_off * c_.ngroups * c_.ic + ic_b1 * c_.ic_block;

        // [4i, 4i, ..., 4i] (16)
        vpbroadcastd(vreg_src_u8, ptr[reg_ptr_src_u8 + reg_off_src_u8
                + off * sizeof(src_data_t)]);
        // [2t, 2t, ..., 2t] (16) <-- i0 * w0 + i1 * w1
        vpmaddubsw(vreg_t_s16, vreg_src_u8, vreg_wei_s8(ic_b1, k));
        // [1u, 1u, ..., 1u] (16) <-- t0 * 1 + t1 * 1
        vpmaddwd(vreg_t_s32, vreg_t_s16, vreg_one_s16);
        // [1o, 1o, ..., 1o] (16) <-- o + u
        vpaddd(vreg_dst_s32(o), vreg_dst_s32(o), vreg_t_s32);
    }
}

/** computes:
 *  i_u8 [~ur_ow~][-ic_nb2][ic_nb1]         [4i] (*)
 *  w_s8                   [ic_nb1][kw][16o][4i]
 * o_s32 [ ur_ow ]                     [16o]
 *
 * with no reduction over ic_nb2
 */
void jit_avx512_core_u8s8u8_conv_fwd_ker_t::compute_part_ur_ow_oc_block(
        int ur_ow, int iw_start) {
    Label l_iw_0;

    if (c_.l_pad && iw_start == 0) {
        /* [r1]: left padding handling happens only at the first iteration */
        test(reg_off_src_u8, reg_off_src_u8);
        je(l_iw_0, T_NEAR);
    }

    const int i_start = - c_.l_pad;
    const int i_end = ur_ow * c_.stride_w + c_.kw - c_.l_pad;
    for (int i = i_start; i < i_end; ++i) {
        if (i == 0)
            L(l_iw_0);

        /* handle right padding */
        if (iw_start + i >= c_.iw)
            continue;

        for (int k = 0; k < c_.kw; ++k) {
            if ((i + k - c_.l_pad) % c_.stride_w != 0)
                continue;
            const int o = (i - k + c_.l_pad) / c_.stride_w;

            if (o < 0 || o >= ur_ow)
                continue;

            compute(o, i, k);
        }
    }
}

/** computes:
 *  i_u8 [~ow~][-ic_nb2][ic_nb1]         [4i] (*)
 *  w_s8       [-ic_nb2][ic_nb1][kw][16o][4i]
 * --------------------------------------------
 * o_s32 [ ow ]                     [16o]     (cast)
 * --------------------------------------------
 *  o_u8 [ ow ]                     [16o]
 *
 * with no reduction over ic_nb2
 */
void jit_avx512_core_u8s8u8_conv_fwd_ker_t::compute_part_ow_oc_block() {
    const int ow_tail_start = c_.ur_ow_nsteps * c_.ur_ow;
    const int iw_tail_start = ow_tail_start * c_.stride_w;

    load_wei_s8();

    xor_(reg_off_src_u8, reg_off_src_u8);
    xor_(reg_off_dst_u8, reg_off_dst_u8);
    xor_(reg_off_dst_s32, reg_off_dst_s32);

    Label l_ur_ow_step;
    L(l_ur_ow_step); {
        load_dst_s32(c_.ur_ow);
        compute_part_ur_ow_oc_block(c_.ur_ow, 0); /* see [r1] */
        store_dst(c_.ur_ow); /* also increases reg_off_dst_u8 */

        const int step_src_u8 = c_.ur_ow * c_.stride_w * c_.ngroups * c_.ic;
        const int step_dst_s32 = c_.ur_ow * c_.oc_block;

        add(reg_off_src_u8, step_src_u8 * sizeof(src_data_t));
        add(reg_off_dst_s32, step_dst_s32 * sizeof(acc_data_t));
        /* increasing reg_off_dst_u8 happens inside store_dst() */

        cmp(reg_off_dst_s32, ow_tail_start * c_.oc_block * sizeof(acc_data_t));
        jne(l_ur_ow_step, T_NEAR);
    }

    if (c_.ur_ow_tail == 0)
        return;

    /* tail ur_ow_tail processing and/or handling right padding
     * [r2]: only this part of the kernel handles right padding */

    load_dst_s32(c_.ur_ow_tail);
    compute_part_ur_ow_oc_block(c_.ur_ow_tail, iw_tail_start);
    store_dst(c_.ur_ow_tail);
}

/** computes:
 *  i_u8 [~ow~][ic_nb2][ic_nb1]         [4i] (*)
 *  w_s8       [ic_nb2][ic_nb1][kw][16o][4i]
 * --------------------------------------------
 * o_s32 [ ow ]                    [16o]     (cast)
 * --------------------------------------------
 *  o_u8 [ ow ]                    [16o]
 *
 * with reduction over ic_nb2
 */
void jit_avx512_core_u8s8u8_conv_fwd_ker_t::compute_ow_oc_block() {
    Label l_kh, l_ic_b2;

    Reg16 reg_tmp = reg_ic_b2.cvt16();
    mov(reg_tmp, 0x1);
    vpbroadcastw(vreg_one_s16, reg_tmp);
    vpxord(vreg_zero, vreg_zero, vreg_zero);

    xor_(reg_state, reg_state);
    or_(reg_state, STATE_LOAD_DST_U8);

    L(l_kh); {
        mov(reg_ic_b2, c_.ic_nb2);
        dec(reg_kh);

        L(l_ic_b2); {
            dec(reg_ic_b2);

            compute_part_ow_oc_block();

            const int step_src = c_.ic_nb1 * c_.ic_block;
            const int step_wei = c_.ic_nb1 * c_.kw * c_.oc_block * c_.ic_block;
            add(reg_ptr_src_u8, step_src * sizeof(src_data_t));
            add(reg_ptr_wei_s8, step_wei * sizeof(wei_data_t));

            and_(reg_state, ~STATE_LOAD_DST_U8);

            test(reg_ic_b2, reg_ic_b2);
            jne(l_ic_b2, T_NEAR);
        }

        const int step_src = - c_.ic + c_.iw * c_.ngroups * c_.ic; // [ih:+1]
        add(reg_ptr_src_u8, step_src * sizeof(src_data_t));

        test(reg_kh, reg_kh);
        jne(l_kh, T_NEAR);
    }
}

void jit_avx512_core_u8s8u8_conv_fwd_ker_t::generate() {
    preamble();

#   define READ_PARAM(reg, field) \
        mov(reg, ptr[abi_param1 + offsetof(call_params_t, field)])
    READ_PARAM(reg_ptr_src_u8, src_u8);
    READ_PARAM(reg_ptr_wei_s8, wei_s8);
    READ_PARAM(reg_ptr_bia_c8, bia_c8);
    READ_PARAM(reg_ptr_dst_u8, dst_u8);
    READ_PARAM(reg_ptr_dst_s32, dst_s32);
    READ_PARAM(reg_kh, kh_range);
#   undef READ_PARAM

    compute_ow_oc_block();

    postamble();
}

status_t jit_avx512_core_u8s8u8_conv_fwd_ker_t::init_conf(jit_conv_conf_t &c,
        const convolution_desc_t &cd, const memory_desc_wrapper &src_d,
        const memory_desc_wrapper &wei_d, const memory_desc_wrapper &dst_d,
        bool with_relu, double negative_slope) {
    if (!mayiuse(avx512_core))
        return status::unimplemented;

    const bool with_groups = wei_d.ndims() == src_d.ndims() + 1;

    c.ngroups = with_groups ? wei_d.dims()[0] : 1;
    c.mb = src_d.dims()[0];
    c.oc = dst_d.dims()[1] / c.ngroups;
    c.ic = src_d.dims()[1] / c.ngroups;
    c.ih = src_d.dims()[2];
    c.iw = src_d.dims()[3];
    c.oh = dst_d.dims()[2];
    c.ow = dst_d.dims()[3];
    c.kh = wei_d.dims()[with_groups + 2];
    c.kw = wei_d.dims()[with_groups + 3];
    c.t_pad = cd.padding[0][0];
    c.b_pad = cd.padding[1][0];
    c.l_pad = cd.padding[0][1];
    c.r_pad = cd.padding[1][1];
    c.stride_h = cd.strides[0];
    c.stride_w = cd.strides[1];
    c.src_fmt = src_d.format();
    c.with_bias = cd.bias_desc.format != memory_format::undef;
    c.with_relu = with_relu;
    c.bia_dt = c.with_bias ? cd.bias_desc.data_type : data_type::undef;

    c.ic_block = 4;
    c.oc_block = 16;

    const bool args_ok = true
        && c.ic % c.ic_block == 0
        && c.oc % c.oc_block == 0
        && everyone_is(nhwc, src_d.format(), dst_d.format())
        && (wei_d.format() == with_groups ? gOhIw16o4i : OhIw16o4i)
        && one_of(cd.bias_desc.format, memory_format::undef, any, x)
        && implication(with_relu, negative_slope == 0.);

    if (!args_ok)
        return status::unimplemented;

    c.oc_nb1 = c.oc / c.oc_block;

    const int ic_nb = c.ic / c.ic_block;
    c.ic_nb1 = c.kw < 7 && ic_nb % 4 == 0 ? 4 : (ic_nb % 2 == 0 ? 2 : 1);
    c.ic_nb2 = ic_nb / c.ic_nb1;

    const int nregs = cpu_isa_traits<avx512_core>::n_vregs;
    const int nregs_aux = 3; // src_u8, 0, 1_s16
    const int nregs_wei = c.ic_nb1 * c.kw;

    assert(nregs_wei + nregs_aux < nregs);

    c.ur_ow_max = nregs - nregs_wei - nregs_aux;

    /* ideally it would be great to have:
     *
     * c.ur_ow = nstl::min(c.ow, c.ur_ow_max);
     * c.ur_ow_nsteps = c.ow / c.ur_ow;
     * c.ur_ow_tail = c.ow - c.ur_ow_nsteps * c.ur_ow;
     *
     * but due to the restriction [r2] we need to call `separate` kernel to
     * handle right padding.
     */
    for (c.ur_ow = nstl::min(c.ow, c.ur_ow_max); c.ur_ow > 0; --c.ur_ow) {
        c.ur_ow_nsteps = c.ow / c.ur_ow;

        if (c.r_pad == 0)
            /* nothing special if there is no right padding */
            break;

        if (c.ur_ow_nsteps * c.ur_ow == c.ow) {
            if (c.ur_ow != c.ow) {
                /* to handle right padding special part of the kernel is used
                 * see [r2] */
                assert(c.r_pad != 0);
                assert(c.ur_ow_nsteps > 1);
                --c.ur_ow_nsteps;
            }
            break;
        }

        /* check whether last non-tail processing kernel has to deal with right
         * padding. */
        const int input_right_edge
            = (c.ur_ow * c.ur_ow_nsteps - 1) * c.stride_w - c.l_pad + c.kw - 1;
        /* if it has not -- we are fine.
         * if it has to -- try next ur_ow (again, see [r2]) */
        if (input_right_edge < c.iw)
            break;
    }
    c.ur_ow_tail = c.ow - c.ur_ow_nsteps * c.ur_ow;

    assert(c.ur_ow * c.stride_w >= c.l_pad); /* see [r1] */

    assert(c.ur_ow <= c.ur_ow_max && c.ur_ow_tail <= c.ur_ow_max);
    assert(c.ur_ow * c.ur_ow_nsteps + c.ur_ow_tail == c.ow);

    return success;
}

/*****************************************************************************/

template <bool with_relu>
status_t _jit_avx512_core_u8s8u8_convolution_fwd_t<with_relu>::pd_t::jit_conf()
{
    return jit_avx512_core_u8s8u8_conv_fwd_ker_t::init_conf(jcp_,
            this->cdesc_(), *this->src_pd_.desc(), *this->weights_pd_.desc(),
            *this->dst_pd_.desc(), with_relu, this->negative_slope());
}

template <bool with_relu>
_jit_avx512_core_u8s8u8_convolution_fwd_t<with_relu>::
_jit_avx512_core_u8s8u8_convolution_fwd_t(const pd_t *pd,
        const input_vector &inputs, const output_vector &outputs)
    : cpu_primitive_t(&conf_, inputs, outputs), conf_(*pd), ker_(nullptr)
    , ws_(nullptr) {
    ker_ = new jit_avx512_core_u8s8u8_conv_fwd_ker_t(conf_.jcp_);

    const int nthreads = omp_get_max_threads();
    ws_per_thread_ = conf_.jcp_.ow * conf_.jcp_.oc_block;
    ws_ = (acc_data_t *)malloc(nthreads * ws_per_thread_ * sizeof(acc_data_t),
            64);
}

template <bool with_relu>
_jit_avx512_core_u8s8u8_convolution_fwd_t<with_relu>::
~_jit_avx512_core_u8s8u8_convolution_fwd_t() { delete ker_; free(ws_); }

template <bool with_relu>
void _jit_avx512_core_u8s8u8_convolution_fwd_t<with_relu>::execute_forward() {
    auto src_u8 = reinterpret_cast<const src_data_t *>(input_memory(0));
    auto wei_s8 = reinterpret_cast<const wei_data_t *>(input_memory(1));
    auto bia_c8 = reinterpret_cast<const char *>(input_memory(2));
    auto dst_u8 = reinterpret_cast<dst_data_t *>(memory());

    const memory_desc_wrapper src_d(conf_.src_pd());
    const memory_desc_wrapper wei_d(conf_.weights_pd(0));
    const memory_desc_wrapper dst_d(conf_.dst_pd());

    const size_t bia_dt_size = conf_.with_bias()
        ? types::data_type_size(conf_.cdesc()->bias_desc.data_type) : 0;

    const auto &c = ker_->c_;

    /*
     * s [mb]              [ih]              [iw][g]       [ic/16*4i]     [4i]
     * w     [g][oc/16]    [kh][ic/16*4i]    [kw]                    [16o][4i]
     * d [mb]          [oh]              [ow]    [g][oc/16]          [16o]
     *
     *   \______drv_______/\_______________________ker_______________________/
     */

    auto ker = [&](int ithr, int nthr) {
        const int work_amount = c.mb * c.ngroups * c.oh * c.oc_nb1;

        int start{0}, end{0};
        balance211(work_amount, nthr, ithr, start, end);

        int n{0}, g{0}, oh{0}, oc_b1{0};
        nd_iterator_init(start, n, c.mb, g, c.ngroups, oh, c.oh, oc_b1,
                c.oc_nb1);

        jit_avx512_core_u8s8u8_conv_fwd_ker_t::call_params_t p = {};
        p.dst_s32 = ws_ + ithr * ws_per_thread_;

        for (int iwork = start; iwork < end; ++iwork) {
            const int kh_start = nstl::max(0, c.t_pad - oh * c.stride_h);
            const int kh_end = nstl::min(c.kh,
                    c.ih + c.t_pad - oh * c.stride_h);

            assert(oh * c.stride_h + kh_start - c.t_pad >= 0);
            assert(oh * c.stride_h + kh_end - c.t_pad <= c.ih);

            const int ih_start = oh * c.stride_h + kh_start - c.t_pad;
            const int oc_start = (g * c.oc_nb1 + oc_b1) * c.oc_block;

            p.src_u8 = &src_u8[src_d.blk_off(n, g * c.ic, ih_start)];
            p.wei_s8 = &wei_s8[conf_.with_groups()
                ? wei_d.blk_off(g, oc_b1, 0, kh_start)
                : wei_d.blk_off(oc_b1, 0, kh_start)];
            p.bia_c8 = &bia_c8[oc_start * bia_dt_size];
            p.dst_u8 = &dst_u8[dst_d.blk_off(n, oc_start, oh)];

            p.kh_range = (size_t)(kh_end - kh_start);

            ker_->ker_(&p);

            nd_iterator_step(n, c.mb, g, c.ngroups, oh, c.oh, oc_b1, c.oc_nb1);
        }
    };

#   pragma omp parallel
    {
        ker(omp_get_thread_num(), omp_get_num_threads());
    }
}

template struct _jit_avx512_core_u8s8u8_convolution_fwd_t<true>;
template struct _jit_avx512_core_u8s8u8_convolution_fwd_t<false>;

}
}
}
