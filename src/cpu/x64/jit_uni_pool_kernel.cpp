/*******************************************************************************
* Copyright 2017-2020 Intel Corporation
* Copyright 2018 YANDEX LLC
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

#include "common/c_types_map.hpp"
#include "common/nstl.hpp"
#include "common/utils.hpp"
#include "cpu/cpu_pooling_pd.hpp"

#include "cpu/x64/jit_uni_pool_kernel.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

using namespace Xbyak;
using namespace alg_kind;

#define GET_OFF(field) offsetof(jit_pool_call_s, field)

template <cpu_isa_t isa>
status_t jit_uni_pool_kernel<isa>::init_conf(jit_pool_conf_t &jpp,
        memory_tracking::registrar_t &scratchpad, const pooling_pd_t *ppd,
        int nthreads) {

    const auto &pd = *ppd->desc();
    const memory_desc_wrapper src_d(
            ppd->is_fwd() ? ppd->src_md() : ppd->diff_src_md());
    const memory_desc_wrapper dst_d(
            ppd->is_fwd() ? ppd->dst_md() : ppd->diff_dst_md());

    const int ndims = src_d.ndims();

    jpp.is_training = pd.prop_kind == prop_kind::forward_training;
    jpp.is_backward = pd.prop_kind == prop_kind::backward_data;

    jpp.oh = (ndims == 3) ? 1 : dst_d.dims()[ndims - 2];

    using namespace format_tag;
    const auto blocked_fmt_tag = utils::one_of(isa, avx512_common, avx512_core)
            ? utils::pick(ndims - 3, nCw16c, nChw16c, nCdhw16c)
            : utils::pick(ndims - 3, nCw8c, nChw8c, nCdhw8c);

    // src_d.data_type() is equal to dst_d.data_type(). This is checked in init
    const bool forward_ncsp_allowed = !jpp.is_backward && jpp.oh == 1;
    const auto ncsp_fmt_tag
            = ((forward_ncsp_allowed || jpp.is_backward) && isa == avx512_core
                      && ndims < 5 && src_d.data_type() == data_type::bf16)
            ? utils::pick(ndims - 3, ncw, nchw)
            : format_tag::undef;

    const auto nspc_fmt_tag = (ndims <= 5)
            ? utils::pick(ndims - 3, nwc, nhwc, ndhwc)
            : format_tag::undef;

    const auto fmt_tag = src_d.matches_one_of_tag(
            blocked_fmt_tag, ncsp_fmt_tag, nspc_fmt_tag);

    if (!dst_d.matches_tag(fmt_tag)) return status::unimplemented;

    if (fmt_tag == ncsp_fmt_tag) {
        // plain layout allowed for BWD_D only now:
        // transform input to blocked f32, call f32 jit, transform result to
        // plain output
        jpp.is_bf16 = false;
        jpp.dt_size = types::data_type_size(data_type::f32);
        jpp.tag_kind = jptg_ncsp;
    } else {
        jpp.is_bf16 = (src_d.data_type() == data_type::bf16
                && dst_d.data_type() == data_type::bf16);
        jpp.dt_size = types::data_type_size(src_d.data_type());
        jpp.tag_kind = (fmt_tag == nspc_fmt_tag) ? jptg_nspc : jptg_blocked;
    }

    jpp.isa = (jpp.is_bf16 && mayiuse(avx512_core_bf16)) ? avx512_core_bf16
                                                         : isa;

    const bool args_ok = true && mayiuse(isa) && (fmt_tag != format_tag::undef)
            && IMPLICATION(jpp.is_bf16, mayiuse(avx512_core))
            && utils::one_of(pd.alg_kind, pooling_max,
                    pooling_avg_include_padding, pooling_avg_exclude_padding);
    if (!args_ok) return status::unimplemented;

    const bool is_avx512 = utils::one_of(isa, avx512_common, avx512_core);
    const int simd_w = is_avx512 ? 16 : 8;

    jpp.ndims = ndims;
    jpp.mb = src_d.dims()[0];
    jpp.c_without_padding = src_d.dims()[1];
    jpp.c = utils::rnd_up(jpp.c_without_padding, simd_w);
    if (fmt_tag != ncsp_fmt_tag && jpp.c > src_d.padded_dims()[1])
        return status::unimplemented;

    jpp.id = (ndims == 5) ? src_d.dims()[2] : 1;
    jpp.ih = (ndims == 3) ? 1 : src_d.dims()[ndims - 2];
    jpp.iw = src_d.dims()[ndims - 1];
    jpp.od = (ndims == 5) ? dst_d.dims()[2] : 1;
    jpp.ow = dst_d.dims()[ndims - 1];

    jpp.stride_d = (ndims == 5) ? pd.strides[0] : 1;
    jpp.stride_h = (ndims == 3) ? 1 : pd.strides[ndims - 4];
    jpp.stride_w = pd.strides[ndims - 3];
    jpp.kd = (ndims == 5) ? pd.kernel[0] : 1;
    jpp.kh = (ndims == 3) ? 1 : pd.kernel[ndims - 4];
    jpp.kw = pd.kernel[ndims - 3];

    jpp.f_pad = (ndims == 5) ? pd.padding[0][0] : 0;
    jpp.t_pad = (ndims == 3) ? 0 : pd.padding[0][ndims - 4];
    jpp.l_pad = pd.padding[0][ndims - 3];

    const int back_pad = calculate_end_padding(
            jpp.f_pad, jpp.od, jpp.id, jpp.stride_d, jpp.kd);
    const int bottom_pad = calculate_end_padding(
            jpp.t_pad, jpp.oh, jpp.ih, jpp.stride_h, jpp.kh);
    const int right_pad = calculate_end_padding(
            jpp.l_pad, jpp.ow, jpp.iw, jpp.stride_w, jpp.kw);

    if (jpp.f_pad >= jpp.kd || jpp.t_pad >= jpp.kh || jpp.l_pad >= jpp.kw
            || back_pad >= jpp.kd || bottom_pad >= jpp.kh
            || right_pad >= jpp.kw)
        return status::unimplemented;

    jpp.alg = pd.alg_kind;

    jpp.ind_dt = ppd->workspace_md() ? ppd->workspace_md()->data_type
                                     : data_type::undef;

    jpp.simple_alg = jpp.is_training
            || IMPLICATION(jpp.is_backward, jpp.kd <= jpp.stride_d);

    jpp.c_block = simd_w;

    jpp.nb_c = jpp.c / jpp.c_block;
    jpp.ur = 0;
    if (jpp.alg == pooling_max) {
        jpp.ur = is_avx512 ? 16 : 4;
        if (jpp.is_training)
            jpp.ur = is_avx512 ? 9 : 3;
        else if (jpp.is_backward)
            jpp.ur = is_avx512 ? 6 : 3;
    } else {
        if (jpp.is_backward)
            jpp.ur = is_avx512 ? 12 : 6;
        else
            jpp.ur = is_avx512 ? 24 : 12;
    }
    if (jpp.is_bf16) {
        jpp.ur = (!isa_has_bf16(jpp.isa))
                ? jpp.ur - 4 // Free registers for AVX512 emulation
                : jpp.ur - 1; // Free register for cvt from bf16 to f32
    }

    // select jpp.ur_bc and ur_w
    if (jpp.tag_kind == jptg_nspc) {
        auto min_ur_w = nstl::max(1, utils::div_up(jpp.l_pad, jpp.stride_w));
        int min_ur_w1 = utils::div_up(right_pad, jpp.stride_w);
        if (min_ur_w < min_ur_w1) { min_ur_w = min_ur_w1; }
        jpp.ur_bc = nstl::min(jpp.nb_c, nstl::max(1, jpp.ur / min_ur_w));
        //take into account threading - to have enough work for parallelization
        float best_eff = 0;
        for (int ur_bc = jpp.ur_bc; ur_bc > 0; ur_bc--) {

            const auto nb2_c = utils::div_up(jpp.nb_c, ur_bc);
            auto work = jpp.is_backward
                    ? (ndims == 5 && jpp.simple_alg ? jpp.od : 1)
                    : (ndims == 5 ? jpp.od : jpp.oh);
            work *= jpp.mb * nb2_c;
            auto eff = (float)work / utils::rnd_up(work, nthreads);
            if (eff > best_eff) {

                best_eff = eff;
                jpp.ur_bc = ur_bc;
            }
            if (eff > 0.9) break; // Heuristic threshold
        }

        //take into account cache re-usage after zeroing on backward
        if (jpp.is_backward && ndims < 5) {
            const int L2 = platform::get_per_core_cache_size(2)
                    / sizeof(jpp.dt_size);
            int ur_bc = nstl::max(1, L2 / (jpp.kh * jpp.iw * jpp.c_block));
            jpp.ur_bc = nstl::min(jpp.ur_bc, ur_bc);
        }

        jpp.ur_bc_tail = jpp.nb_c % jpp.ur_bc;
    } else {
        jpp.ur_bc = 1;
        jpp.ur_bc_tail = 0;
    }
    auto ur_w = nstl::min(jpp.ow, jpp.ur / jpp.ur_bc);
    if (utils::div_up(jpp.l_pad, jpp.stride_w) > ur_w)
        return status::unimplemented;
    if (utils::div_up(right_pad, jpp.stride_w) > ur_w)
        return status::unimplemented;

    // scratchpad for c_block slice of input and/or output
    using namespace memory_tracking::names;
    const int nscr = nstl::min(dnnl_get_max_threads(), jpp.mb * jpp.nb_c);
    if (jpp.tag_kind == jptg_ncsp) {
        scratchpad.book(key_pool_src_plain2blocked_cvt,
                jpp.c_block * jpp.id * jpp.ih * jpp.iw * nscr, jpp.dt_size);
        scratchpad.book(key_pool_dst_plain2blocked_cvt,
                jpp.c_block * jpp.od * jpp.oh * jpp.ow * nscr, jpp.dt_size);
        scratchpad.book<uint32_t>(key_pool_ind_plain2blocked_cvt,
                jpp.c_block * jpp.od * jpp.oh * jpp.ow * nscr);
    }

    return status::success;
}

template <cpu_isa_t isa>
inline void jit_uni_pool_kernel<isa>::maybe_recalculate_divisor(
        int jj, int ur_w, int pad_l, int pad_r) {
    if (jpp.alg == pooling_avg_exclude_padding) {
        int kw = jpp.kw;
        int stride_w = jpp.stride_w;

        int non_zero_kw = kw;
        non_zero_kw -= nstl::max(0, pad_l - jj * stride_w);
        non_zero_kw -= nstl::max(0, pad_r - (ur_w - 1 - jj) * stride_w);

        if (non_zero_kw != prev_kw) {
            mov(tmp_gpr, float2int((float)non_zero_kw));
            movq(xmm_tmp, tmp_gpr);
            uni_vbroadcastss(vmm_tmp, xmm_tmp);
            uni_vmulps(vmm_tmp, vmm_tmp, vmm_ker_area_h);
            prev_kw = non_zero_kw;
        }
    }
}

template <cpu_isa_t isa>
inline void jit_uni_pool_kernel<isa>::avg_step(
        int ur_w, int ur_bc, int pad_l, int pad_r) {

    auto iw = jpp.iw;
    auto kw = jpp.kw;
    auto stride_w = jpp.stride_w;
    auto c_block = jpp.c_block;
    auto dt_size = jpp.dt_size;
    const int c_off = (jpp.tag_kind == jptg_nspc) ? jpp.c : c_block;
    Label kd_label, kh_label;

    auto reg_ind = [&](int shift, int bc, int j) {
        return shift * ur_bc * ur_w + bc * ur_w + j;
    };

    for (int jj = 0; jj < ur_w; jj++) {
        if (jpp.is_backward) maybe_recalculate_divisor(jj, ur_w, pad_l, pad_r);
        for (int bci = 0; bci < ur_bc; bci++) {
            auto accr_i = reg_ind(0, bci, jj);
            auto accvr = vreg(accr_i);
            if (jpp.is_backward) {
                auto output_offset = dt_size * (jj * c_off + bci * c_block);
                load(accr_i, reg_output, output_offset);
                uni_vdivps(accvr, accvr, vmm_tmp);
            } else {
                uni_vpxor(accvr, accvr, accvr);
            }
        }
    }

    if (jpp.simple_alg && jpp.ndims == 5) {
        push(reg_input);
        push(reg_output);
        mov(aux_reg_input_d, reg_input);
        mov(ki, ptr[reg_param + GET_OFF(kd_padding)]);
        L(kd_label);
        mov(aux_reg_input, aux_reg_input_d);
    } else {
        mov(aux_reg_input, reg_input);
    }

    xor_(kj, kj);
    L(kh_label);
    {
        for (int ki = 0; ki < kw; ki++) {
            int jj_start = nstl::max(0, utils::div_up(pad_l - ki, stride_w));
            int jj_end = ur_w
                    - utils::div_up(
                            nstl::max(0, ki + pad_r - (kw - 1)), stride_w);

            for_(int jj = jj_start; jj < jj_end; jj++)
            for (int bci = 0; bci < ur_bc; bci++) {
                auto accvr = vreg(reg_ind(0, bci, jj));
                auto inpr_i = reg_ind(1, bci, jj);
                auto inpvr = vreg(inpr_i);
                int aux_input_offset
                        = (ki + jj * stride_w - pad_l) * c_off + bci * c_block;
                if (aux_input_offset >= iw * c_off) continue;
                int input_offset = dt_size * aux_input_offset;
                if (jpp.is_backward) {
                    auto inpyr = yreg(inpr_i);
                    load(inpr_i, aux_reg_input, input_offset);
                    uni_vaddps(inpvr, inpvr, accvr);
                    if (jpp.is_bf16) {
                        if (!isa_has_bf16(jpp.isa))
                            bf16_emu_->vcvtneps2bf16(inpyr, zreg(inpr_i));
                        else
                            vcvtneps2bf16(inpyr, inpvr);
                        vmovdqu16(ptr[aux_reg_input + input_offset], inpyr);
                    } else {
                        uni_vmovups(
                                vmmword[aux_reg_input + input_offset], inpvr);
                    }
                } else {
                    if (jpp.is_bf16) {
                        vmovups(ymm_tmp_1, ptr[aux_reg_input + input_offset]);
                        vpermw(vmm_tmp_1 | k_mask_cvt | T_z, vmm_idx(),
                                vmm_tmp_1);

                        uni_vaddps(accvr, accvr, vmm_tmp_1);
                    } else {
                        uni_vaddps(accvr, accvr,
                                ptr[aux_reg_input + input_offset]);
                    }
                }
            }
        }
        add(aux_reg_input, jpp.dt_size * iw * c_off);
        inc(kj);
        cmp(kj, reg_kh);
        jl(kh_label, T_NEAR);
    }

    if (jpp.simple_alg && jpp.ndims == 5) {
        add(aux_reg_input_d, jpp.dt_size * jpp.ih * iw * c_off);
        dec(ki);
        cmp(ki, 0);
        jg(kd_label, T_NEAR);
        pop(reg_output);
        pop(reg_input);
    }

    if (!jpp.is_backward) {
        for (int jj = 0; jj < ur_w; jj++) {
            maybe_recalculate_divisor(jj, ur_w, pad_l, pad_r);
            for (int bci = 0; bci < ur_bc; bci++) {
                auto accr_i = reg_ind(0, bci, jj);
                auto accvr = vreg(accr_i);
                auto output_offset = dt_size * (jj * c_off + bci * c_block);
                uni_vdivps(accvr, accvr, vmm_tmp);
                if (jpp.is_bf16) {
                    auto acczr = zreg(accr_i);
                    auto accyr = yreg(accr_i);
                    if (!isa_has_bf16(jpp.isa))
                        bf16_emu_->vcvtneps2bf16(accyr, acczr);
                    else
                        vcvtneps2bf16(accyr, accvr);
                    vmovdqu16(ptr[reg_output + output_offset], accyr);
                } else {
                    uni_vmovups(vmmword[reg_output + output_offset], accvr);
                }
            }
        }
    }
}

template <cpu_isa_t isa>
inline void jit_uni_pool_kernel<isa>::max_step_fwd(
        int ur_w, int ur_bc, int pad_l, int pad_r) {
    int iw = jpp.iw;
    int kw = jpp.kw;
    int stride_w = jpp.stride_w;
    int c_block = jpp.c_block;
    const int c_off = (jpp.tag_kind == jptg_nspc) ? jpp.c : c_block;
    Label kd_label, kh_label;

    auto reg_ind = [&](int shift, int bc, int j) {
        return shift * ur_bc * ur_w + bc * ur_w + j;
    };

    mov(tmp_gpr, float2int(nstl::numeric_limits<float>::lowest()));
    movq(xmm_tmp, tmp_gpr);
    uni_vbroadcastss(vmm_tmp, xmm_tmp);

    for_(int jj = 0; jj < ur_w; jj++)
    for (int bci = 0; bci < ur_bc; bci++) {
        auto accvr = vreg(reg_ind(0, bci, jj));
        uni_vmovups(accvr, vmm_tmp);
        if (jpp.is_training) {
            auto indvr = vreg(reg_ind(2, bci, jj));
            uni_vpxor(indvr, indvr, indvr);
        }
    }
    if (jpp.is_training) {
        movq(xmm_tmp, reg_k_shift);
        uni_vpbroadcastd(vmm_k_offset, xmm_tmp);
    }
    if (jpp.ndims == 5) {
        push(reg_input);
        push(reg_output);
        mov(aux_reg_input_d, reg_input);
        mov(ki, ptr[reg_param + GET_OFF(kd_padding)]);
        L(kd_label);
        mov(aux_reg_input, aux_reg_input_d);
    } else {
        mov(aux_reg_input, reg_input);
    }
    xor_(kj, kj);
    L(kh_label);
    {
        for (int ki = 0; ki < kw; ki++) {
            int jj_start = nstl::max(0, utils::div_up(pad_l - ki, stride_w));
            int jj_end = ur_w
                    - utils::div_up(
                            nstl::max(0, ki + pad_r - (kw - 1)), stride_w);
            for_(int jj = jj_start; jj < jj_end; jj++)
            for (int bci = 0; bci < ur_bc; bci++) {
                auto accvr = vreg(reg_ind(0, bci, jj));
                auto inpr_i = reg_ind(1, bci, jj);
                auto inpvr = vreg(inpr_i);
                auto indvr = vreg(reg_ind(2, bci, jj));
                auto cvtvr = vreg(reg_ind(3, bci, jj));
                int aux_input_offset
                        = (ki + jj * stride_w - pad_l) * c_off + bci * c_block;
                if (aux_input_offset >= iw * c_off) continue;
                int input_offset = jpp.dt_size * aux_input_offset;
                load(inpr_i, aux_reg_input, input_offset);
                if (isa == sse41) {
                    movups(vmm_mask, accvr);
                    cmpps(vmm_mask, inpvr, _cmp_lt_os);
                    blendvps(accvr, inpvr);
                    if (jpp.is_training) blendvps(indvr, vmm_k_offset);
                } else if (isa == avx) {
                    vcmpps(cvtvr, accvr, inpvr, _cmp_lt_os);
                    vblendvps(accvr, accvr, inpvr, cvtvr);
                    if (jpp.is_training)
                        vblendvps(indvr, indvr, vmm_k_offset, cvtvr);
                } else {
                    vcmpps(k_store_mask, accvr, inpvr, _cmp_lt_os);
                    vblendmps(accvr | k_store_mask, accvr, inpvr);
                    if (jpp.is_training)
                        vblendmps(indvr | k_store_mask, indvr, vmm_k_offset);
                }
            }
            if (jpp.is_training) {
                if (isa == avx && !mayiuse(avx2))
                    avx_vpadd1(vmm_k_offset, vmm_one, xmm_tmp);
                else
                    uni_vpaddd(vmm_k_offset, vmm_k_offset, vmm_one);
            }
        }
        add(aux_reg_input, jpp.dt_size * iw * c_off);
        inc(kj);
        cmp(kj, reg_kh);
        jl(kh_label, T_NEAR);
    }

    if (jpp.ndims == 5) {
        add(aux_reg_input_d, jpp.dt_size * jpp.ih * iw * c_off);
        if (jpp.is_training) {
            mov(tmp_gpr, ptr[reg_param + GET_OFF(kd_padding_shift)]);
            movq(xmm_tmp, tmp_gpr);
            uni_vpbroadcastd(vmm_tmp, xmm_tmp);
            if (isa == avx && !mayiuse(avx2)) {
                Xmm t(vmm_mask.getIdx());
                avx_vpadd1(vmm_k_offset, xmm_tmp, t);
            } else {
                uni_vpaddd(vmm_k_offset, vmm_k_offset, vmm_tmp);
            }
        }

        dec(ki);
        cmp(ki, 0);
        jg(kd_label, T_NEAR);
        pop(reg_output);
        pop(reg_input);
    }

    for_(int jj = 0; jj < ur_w; jj++)
    for (int bci = 0; bci < ur_bc; bci++) {
        auto accr_i = reg_ind(0, bci, jj);
        auto accvr = vreg(accr_i);
        auto output_offset = jpp.dt_size * (jj * c_off + bci * c_block);
        if (jpp.is_bf16) {
            auto acczr = zreg(accr_i);
            auto accyr = yreg(accr_i);
            if (!isa_has_bf16(jpp.isa))
                bf16_emu_->vcvtneps2bf16(accyr, acczr);
            else
                vcvtneps2bf16(accyr, accvr);
            vmovups(ptr[reg_output + output_offset], accyr);
        } else {
            uni_vmovups(vmmword[reg_output + output_offset], accvr);
        }
        if (jpp.is_training) {
            const size_t step_index = (jj * c_off + bci * c_block)
                    * types::data_type_size(jpp.ind_dt);

            auto indr_i = reg_ind(2, bci, jj);
            auto vr = vreg(indr_i);
            if (jpp.ind_dt == data_type::u8) {
                auto xr = xreg(indr_i);
                if (isa == sse41) {
                    for (int i = 0; i < 4; ++i)
                        pextrb(ptr[reg_index + step_index + i], xr, 4 * i);
                } else if (isa == avx) {
                    auto yr = yreg(indr_i);
                    if (jj == 0) {
                        vmovd(xmm_tmp, reg_shuf_mask);
                        uni_vpbroadcastd(vmm_tmp, xmm_tmp);
                    }
                    if (mayiuse(avx2)) {
                        vpshufb(yr, yr, vmm_tmp);
                        vmovd(ptr[reg_index + step_index], xr);
                        vperm2i128(yr, yr, yr, 0x1u);
                        vmovd(ptr[reg_index + step_index + 4], xr);
                    } else {
                        Xmm t(vmm_mask.getIdx());
                        vextractf128(t, yr, 0);
                        vpshufb(t, t, xmm_tmp);
                        vmovd(ptr[reg_index + step_index], t);
                        vextractf128(t, yr, 1);
                        vpshufb(t, t,
                                xmm_tmp); // ymm_tmp[:128]==ymm_tmp[127:0]
                        vmovd(ptr[reg_index + step_index + 4], t);
                    }
                } else {
                    vpmovusdb(xr, vr);
                    vmovups(ptr[reg_index + step_index], vr | k_index_mask);
                }
            } else {
                uni_vmovups(ptr[reg_index + step_index], vr);
            }
        }
    }
}

template <cpu_isa_t isa>
inline void jit_uni_pool_kernel<isa>::max_step_bwd(
        int ur_w, int ur_bc, int pad_l, int pad_r) {

    int iw = jpp.iw;
    int kw = jpp.kw;
    int stride_w = jpp.stride_w;
    int c_block = jpp.c_block;
    const int c_off = (jpp.tag_kind == jptg_nspc) ? jpp.c : c_block;
    Label kd_label, kh_label;

    auto reg_ind = [&](int shift, int bc, int j) {
        return shift * ur_bc * ur_w + bc * ur_w + j;
    };

    for_(int jj = 0; jj < ur_w; jj++)
    for (int bci = 0; bci < ur_bc; bci++) {
        auto outr_i = reg_ind(0, bci, jj);
        auto out_offset = jpp.dt_size * (jj * c_off + bci * c_block);
        load(outr_i, reg_output, out_offset);
        const size_t step_index = (jj * c_off + bci * c_block)
                * types::data_type_size(jpp.ind_dt);

        auto indr_i = reg_ind(1, bci, jj);
        auto indvr = vreg(indr_i);
        if (jpp.ind_dt == data_type::u8) {
            auto indxr = xreg(indr_i);
            if (isa == sse41) {
                movd(indxr, ptr[reg_index + step_index]);
                pmovzxbd(indvr, indxr);
            } else if (isa == avx) {
                vmovq(indxr, ptr[reg_index + step_index]);
                if (!mayiuse(avx2)) {
                    avx_pmovzxbd(indvr, indxr, xmm_tmp);
                } else {
                    vpmovzxbd(indvr, indxr);
                }
            } else {
                vmovups(indvr | k_index_mask, ptr[reg_index + step_index]);
                vpmovzxbd(indvr, indxr);
            }
        } else {
            uni_vmovups(indvr, ptr[reg_index + step_index]);
        }
    }
    movq(xmm_tmp, reg_k_shift);
    uni_vpbroadcastd(vmm_k_offset, xmm_tmp);

    if (jpp.simple_alg && jpp.ndims == 5) {
        push(reg_input);
        push(reg_output);
        if (isa == sse41) {
            // Save rdi since it is used in maskmovdqu
            assert(dst_ptr == rdi);
            push(dst_ptr);
        }
        mov(aux_reg_input_d, reg_input);
        mov(ki, ptr[reg_param + GET_OFF(kd_padding)]);
        mov(reg_kd_pad_shift, ptr[reg_param + GET_OFF(kd_padding_shift)]);
        L(kd_label);
        mov(aux_reg_input, aux_reg_input_d);
    } else {
        mov(aux_reg_input, reg_input);
    }

    xor_(kj, kj);
    L(kh_label);
    {
        for (int ki = 0; ki < kw; ki++) {
            int jj_start = nstl::max(0, utils::div_up(pad_l - ki, stride_w));
            int jj_end = ur_w
                    - utils::div_up(
                            nstl::max(0, ki + pad_r - (kw - 1)), stride_w);
            for_(int jj = jj_start; jj < jj_end; jj++)
            for (int bci = 0; bci < ur_bc; bci++) {
                auto outvr = vreg(reg_ind(0, bci, jj));
                auto indvr = vreg(reg_ind(1, bci, jj));
                auto inpr_i = reg_ind(2, bci, jj);
                auto inpvr = vreg(inpr_i);
                auto cvtvr = vreg(reg_ind(3, bci, jj));
                int aux_inp_offset
                        = (ki + jj * stride_w - pad_l) * c_off + bci * c_block;
                if (aux_inp_offset >= iw * c_off) continue;
                int inp_offset = jpp.dt_size * aux_inp_offset;
                load(inpr_i, aux_reg_input, inp_offset);
                if (isa == sse41) {
                    mov(dst_ptr, aux_reg_input);
                    add(dst_ptr, inp_offset);

                    movups(cvtvr, indvr);
                    pcmpeqd(cvtvr, vmm_k_offset);
                    addps(inpvr, outvr);
                    maskmovdqu(inpvr, cvtvr);
                } else if (isa == avx) {
                    if (mayiuse(avx2)) {
                        vpcmpeqd(cvtvr, indvr, vmm_k_offset);
                    } else {
                        avx_pcmpeqd(cvtvr, indvr, vmm_k_offset, xmm_tmp);
                    }
                    vaddps(inpvr, inpvr, outvr);
                    vmaskmovps(
                            vmmword[aux_reg_input + inp_offset], cvtvr, inpvr);
                } else {
                    auto indzr = zreg(inpr_i);
                    auto indyr = yreg(inpr_i);
                    vpcmpeqd(k_store_mask, indvr, vmm_k_offset);
                    vblendmps(vmm_tmp | k_store_mask | T_z, outvr, outvr);
                    vaddps(inpvr, inpvr, vmm_tmp);
                    if (jpp.is_bf16) {
                        if (!isa_has_bf16(jpp.isa))
                            bf16_emu_->vcvtneps2bf16(indyr, indzr);
                        else
                            vcvtneps2bf16(indyr, inpvr);
                        vmovdqu16(ptr[aux_reg_input + inp_offset], indyr);
                    } else {
                        vmovups(vmmword[aux_reg_input + inp_offset], inpvr);
                    }
                }
            }
            if (isa == avx && !mayiuse(avx2)) {
                avx_vpadd1(vmm_k_offset, vmm_one, xmm_tmp);
            } else {
                uni_vpaddd(vmm_k_offset, vmm_k_offset, vmm_one);
            }
        }
        add(aux_reg_input, jpp.dt_size * iw * c_off);
        inc(kj);
        cmp(kj, reg_kh);
        jl(kh_label, T_NEAR);
    }
    if (jpp.simple_alg && jpp.ndims == 5) {
        add(aux_reg_input_d, jpp.dt_size * jpp.ih * iw * c_off);

        mov(tmp_gpr, reg_kd_pad_shift);
        movq(xmm_tmp, tmp_gpr);
        uni_vpbroadcastd(vmm_tmp, xmm_tmp);
        if (isa == avx && !mayiuse(avx2)) {
            Xmm t(vmm_mask.getIdx());
            avx_vpadd1(vmm_k_offset, vmm_tmp, t);
        } else {
            uni_vpaddd(vmm_k_offset, vmm_k_offset, vmm_tmp);
        }

        dec(ki);
        cmp(ki, 0);
        jg(kd_label, T_NEAR);
        if (isa == sse41) {
            // Save rdi since it is used in maskmovdqu
            assert(dst_ptr == rdi);
            pop(dst_ptr);
        }
        pop(reg_output);
        pop(reg_input);
    }
}

template <cpu_isa_t isa>
void jit_uni_pool_kernel<isa>::maybe_zero_diff_src(int ur_bc) {
    const int c_off = (jpp.tag_kind == jptg_nspc) ? jpp.c : jpp.c_block;
    assert(c_off * sizeof(float) % cpu_isa_traits<isa>::vlen == 0);

    Label l_skip, l_zero;

    auto reg_oh = tmp_gpr;
    mov(reg_oh, ptr[reg_param + GET_OFF(oh)]);
    cmp(reg_oh, 0);
    jz(l_skip, T_NEAR);

    if (jpp.ndims == 5)
        mov(zero_size, ptr[reg_param + GET_OFF(oh)]);
    else
        mov(zero_size, 1);

    const int width_size = jpp.iw * c_off * jpp.dt_size;
    mov(tmp_gpr, jpp.ih * width_size);
    imul(zero_size, tmp_gpr);

    auto vzero = vmm_tmp;
    auto yzero = ymm_tmp;
    uni_vpxor(vzero, vzero, vzero);

    auto reg_off = tmp_gpr;
    xor_(reg_off, reg_off);

    L(l_zero);
    {
        const auto vlen = cpu_isa_traits<isa>::vlen;
        const int step = (jpp.tag_kind == jptg_nspc)
                ? jpp.dt_size * jpp.c
                : (jpp.is_bf16) ? vlen / 2 : vlen;
        // TODO: maybe a big code generated here
        for_(int i = 0; i < width_size; i += step)
        for (int bci = 0; bci < ur_bc; bci++) {
            const int offs = i + bci * jpp.c_block * jpp.dt_size;
            if (jpp.is_bf16)
                vmovdqu16(ptr[reg_input + reg_off + offs], yzero);
            else
                uni_vmovups(ptr[reg_input + reg_off + offs], vzero);
            if (isa == sse41)
                uni_vmovups(ptr[reg_input + reg_off + offs + vlen], vzero);
        }
        add(reg_off, width_size);
        cmp(reg_off, zero_size);
        jl(l_zero, T_NEAR);
    }

    L(l_skip);
}

template <cpu_isa_t isa>
void jit_uni_pool_kernel<isa>::generate() {

    this->preamble();

    Label idx_table;

    int ow = jpp.ow;
    int iw = jpp.iw;
    int kw = jpp.kw;
    int kh = jpp.kh;
    int c_block = jpp.c_block;
    int stride_w = jpp.stride_w;
    int l_pad = jpp.l_pad;
    const int c_off = (jpp.tag_kind == jptg_nspc) ? jpp.c : c_block;

    int vlen = cpu_isa_traits<isa>::vlen;

#if defined(_WIN32)
    // Always mimic the Unix ABI (see the note about maskmovdqu in the header
    // file).
    xor_(rdi, rcx);
    xor_(rcx, rdi);
    xor_(rdi, rcx);
#endif
    if (!isa_has_bf16(jpp.isa) && jpp.is_bf16) bf16_emu_->init_vcvtneps2bf16();

    mov(reg_input, ptr[reg_param + GET_OFF(src)]);
    mov(reg_output, ptr[reg_param + GET_OFF(dst)]);
    if (jpp.alg == pooling_max && (jpp.is_training || jpp.is_backward))
        mov(reg_index, ptr[reg_param + GET_OFF(indices)]);
    mov(reg_kh, ptr[reg_param + GET_OFF(kh_padding)]);
    mov(reg_k_shift, ptr[reg_param + GET_OFF(kh_padding_shift)]);
    mov(reg_ker_area_h, ptr[reg_param + GET_OFF(ker_area_h)]);
    mov(reg_nbc, ptr[reg_param + GET_OFF(ur_bc)]);

    if (jpp.is_bf16) {
        mov(tmp_gpr.cvt32(), 0xAAAAAAAA);
        kmovd(k_mask_cvt, tmp_gpr.cvt32());

        mov(tmp_gpr, idx_table);
        vmovups(vmm_idx(), ptr[tmp_gpr]);
    }

    int r_pad
            = nstl::max(0, calculate_end_padding(l_pad, ow, iw, stride_w, kw));

    auto process_oi = [&](int ur_w, int ur_bc, int lpad, int rpad,
                              bool inc_reg = true) {
        step(ur_w, ur_bc, lpad, rpad);

        if (isa == sse41) step_high_half(ur_w, ur_bc, lpad, rpad);

        if (!inc_reg) return;

        auto dt_size = jpp.dt_size;
        auto shift = (isa == sse41) ? vlen : 0;
        add(reg_input, dt_size * (ur_w * stride_w - lpad) * c_off - shift);
        add(reg_output, dt_size * ur_w * c_off - shift);
        if (jpp.alg == pooling_max && (jpp.is_training || jpp.is_backward)) {
            auto ishift = (isa == sse41) ? jpp.c_block / 2 : 0;
            auto ind_dt_size = types::data_type_size(jpp.ind_dt);
            add(reg_index, (ur_w * c_off - ishift) * ind_dt_size);
        }
    };

    auto perform_ker = [&](int ur_bc) {
        prev_kw = 0; // re-initialize this value for avg steps

        if (jpp.is_backward && jpp.simple_alg) maybe_zero_diff_src(ur_bc);

        if (jpp.alg == pooling_avg_exclude_padding) {
            movq(xmm_ker_area_h, reg_ker_area_h);
            uni_vpbroadcastd(vmm_ker_area_h, xmm_ker_area_h);
        }

        if (jpp.alg == pooling_avg_include_padding) {
            mov(tmp_gpr, float2int((float)(kw * kh * jpp.kd)));
            movq(xmm_tmp, tmp_gpr);
            uni_vpbroadcastd(vmm_tmp, xmm_tmp);
        }

        if (jpp.alg == pooling_max && (jpp.is_training || jpp.is_backward)) {
            mov(tmp_gpr, 1);
            movq(xmm_one, tmp_gpr);
            uni_vpbroadcastd(vmm_one, xmm_one);

            if (isa == avx) {
                mov(reg_shuf_mask, 0x0c080400);
            } else if (isa >= avx512_common) {
                mov(tmp_gpr.cvt32(), 0x000f);
                kmovw(k_index_mask, tmp_gpr.cvt32());
            }
        }

        auto ur_w = nstl::min(jpp.ow, jpp.ur / jpp.ur_bc);
        auto ur_w_tail = jpp.ow % ur_w;

        int n_oi = ow / ur_w;

        int r_pad1
                = calculate_end_padding(l_pad, ur_w * n_oi, iw, stride_w, kw);
        if (r_pad1 > 0) n_oi--;

        if (l_pad > 0) {
            n_oi--;
            if (n_oi < 0 && r_pad1 > 0)
                process_oi(ur_w, ur_bc, l_pad, r_pad1);
            else
                process_oi(ur_w, ur_bc, l_pad, 0);
        }

        xor_(oi_iter, oi_iter);
        if (n_oi > 0) {
            Label ow_loop;
            L(ow_loop);
            {
                process_oi(ur_w, ur_bc, 0, 0);

                inc(oi_iter);
                cmp(oi_iter, n_oi);
                jl(ow_loop, T_NEAR);
            }
        }

        if (r_pad1 > 0 && n_oi >= 0) process_oi(ur_w, ur_bc, 0, r_pad1);

        if (ur_w_tail != 0) process_oi(ur_w_tail, ur_bc, 0, r_pad, false);
    };
    Label ur_bc_tail_label, finish_label;

    if (jpp.ur_bc_tail > 0) {
        cmp(reg_nbc, jpp.ur_bc);
        jne(ur_bc_tail_label, T_NEAR);
    }

    perform_ker(jpp.ur_bc);

    if (jpp.ur_bc_tail > 0) {
        jmp(finish_label, T_NEAR);

        L(ur_bc_tail_label);
        perform_ker(jpp.ur_bc_tail);

        L(finish_label);
    }

    this->postamble();

    if (jpp.is_bf16) {
        align(64);
        L(idx_table);
        const uint16_t _idx[] = {0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7,
                8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15};
        for (size_t i = 0; i < sizeof(_idx) / sizeof(_idx[0]); ++i)
            dw(_idx[i]);
    }
}

template struct jit_uni_pool_kernel<sse41>;
template struct jit_uni_pool_kernel<avx>; // implements both <avx> and <avx2>
template struct jit_uni_pool_kernel<avx512_common>;
template struct jit_uni_pool_kernel<avx512_core>;

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
