/*******************************************************************************
* Copyright 2016-2018 Intel Corporation
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

#include "c_types_map.hpp"
#include "nstl.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"
#include "cpu_memory.hpp"

#include "jit_avx512_common_conv_kernel.hpp"

#include <iostream>
#include <cmath>

#define GET_OFF(field) offsetof(jit_conv_call_s, field)
#define KNx_L2_EFFECTIVE_CAPACITY ((512-64)*1024)

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace mkldnn::impl::memory_format;
using namespace mkldnn::impl::utils;
using namespace Xbyak;

using std::cout;
using std::endl;

namespace {

constexpr auto small_spatial = 14;
unsigned int L1_cache_size = get_cache_size(1, true);

inline void pick_loop_order(jit_conv_conf_t &jcp) {
    using namespace prop_kind;
    assert(one_of(jcp.prop_kind,
                forward_training, forward_inference, backward_data));
    auto w = (jcp.prop_kind == backward_data) ? jcp.iw : jcp.ow;
    auto h = (jcp.prop_kind == backward_data) ? jcp.ih : jcp.oh;
    switch (jcp.ver) {
    case ver_fma:
        jcp.loop_order = loop_cgn;
    case ver_4vnni:
    case ver_vnni:
        // TBD: Tune on HW
    case ver_4fma:
        jcp.loop_order
            = (w <= small_spatial && h <= small_spatial) ? loop_cgn : loop_gnc;
        break;
    default:
        assert(!"unsupported convolution version");
    }
}

inline bool is_1stconv(const jit_conv_conf_t &jcp) {
    if (mayiuse(avx512_core) && !mayiuse(avx512_core_vnni))
        return jcp.ic < 16;
    else
        return one_of(jcp.ic, 1, 3);
}

}


void jit_avx512_common_conv_fwd_kernel::compute_loop_fma_sparse() {

    /**********************************************

    reg64_t param = abi_param1;
    reg64_t reg_inp = r8;
    reg64_t reg_ker = r9;
    reg64_t reg_out = r10;

    reg64_t reg_inp_prf = r11;
    reg64_t reg_ker_prf = r12;
    reg64_t reg_out_prf = r13;

    reg64_t aux_reg_inp = r14;
    reg64_t aux_reg_ker = r15;

    reg64_t aux_reg_inp_prf = rsi;
    reg64_t aux_reg_ker_prf = rdx;

    reg64_t reg_channel = rsi;
    reg64_t reg_bias = rdx;

    reg64_t aux_reg_ker_d = r9;
    reg64_t aux_reg_inp_d = rbx;
    reg64_t aux_reg_inp_d_prf = r13;
    reg64_t aux_reg_ker_d_prf = abi_not_param1;
    reg64_t reg_ki = r10;

    reg64_t reg_kj = rax;
    reg64_t reg_relu_ns = rax;
    reg64_t reg_oi = rbx;
    reg64_t reg_kh = abi_not_param1;

    reg64_t reg_tmp = rbp;

    reg64_t reg_ic_loop = rdx;
    reg64_t reg_inp_loop = rsi;

    reg64_t reg_init_flag = r13;
    reg64_t reg_bias_ptr = param;

    reg64_t aux_reg_ic = r12;
    reg64_t reg_binp = rax;
    reg64_t reg_bout = r11;
    reg64_t aux1_reg_inp = rbx;
    reg64_t aux_reg_out = abi_not_param1;

    reg64_t reg_long_offt = r11;
    reg64_t reg_out_long_offt = r14;

    ***********************************************/

    Label kh_label, kd_label, skip_kd_loop;
    Label end_label, clear_label;

    int kw = jcp.kw;
    int kh = jcp.kh;

    int ow = jcp.ow;
    int oh = jcp.oh;

    int nb_ic = jcp.nb_ic;
    int nb_oc = jcp.nb_oc;

    int iw = jcp.iw;
    int ih = jcp.ih;

    int mb = jcp.mb;
    int mb_block = jcp.mb_block;

    int ic_block = jcp.ic_block;
    int oc_block = jcp.oc_block;
    int l_pad = jcp.l_pad;
    int dilate_w = jcp.dilate_w + 1;
    int stride_w = jcp.stride_w;
    int stride_h = jcp.stride_h;

    int ker_pipeline_depth = 4;
    assert(ker_reg_base_idx + ker_pipeline_depth <= 32);
    assert(ic_block >= ker_pipeline_depth);

    int nr = jcp.ur_sparse;
    int oc_buffs = jcp.oc_buffs;

    assert(nr >= kw);
    assert(oc_block == 16); // needed for specific optimization
    assert(typesize == 4);

    int oc_iters = nb_oc / oc_buffs;

    auto zmm_o = Xbyak::Zmm(31);

    auto ymm_zero = Xbyak::Ymm(30);
    auto zmm_zero = Xbyak::Zmm(30);

    vpxord(zmm_zero, zmm_zero, zmm_zero);

    vcmpps(k7, zmm_zero, ptr[reg_inp], 4);
    prefetcht1(ptr[reg_inp_prf]);

    assert(nr % kw == 0);

    Label no_init_label;

    mov(reg_channel, ptr[param + GET_OFF(channel)]);
    cmp(reg_channel, 0);
    jne(no_init_label, T_NEAR);

    if (oc_buffs * ow > 128 || jcp.with_bias) { // threshold may be tweaked later

        Reg64 aux_reg_out = aux_reg_inp;
        Reg64 aux_reg_out_prf = aux_reg_ker;

        if (jcp.with_bias) {
            mov(reg_bias, ptr[param1 + GET_OFF(bias)]);
        }

        mov(aux_reg_out, reg_out);
        mov(aux_reg_out_prf, reg_out_prf);

        mov(reg_channel, oc_buffs);

        Label oc_loop_label;
        L(oc_loop_label);

        if (jcp.with_bias) {
            vmovups(zmm_zero, ptr[reg_bias]);
            add(reg_bias, typesize * oc_block);
        }

        for (int oi = 0; oi < ow; oi++) {
            vmovups(EVEX_compress_addr_safe(aux_reg_out, oi * oc_block * typesize,
                        reg_long_offt), zmm_zero);
            prefetcht1(EVEX_compress_addr_safe(aux_reg_out_prf, oi * oc_block * typesize,
                        reg_long_offt));
        }

        add(aux_reg_out, typesize * oc_block * mb_block * ow);
        add(aux_reg_out_prf, typesize * oc_block * mb_block * ow);

        dec(reg_channel);
        cmp(reg_channel, 0);
        jne(oc_loop_label);

        if (jcp.with_bias) {
            vpxord(zmm_zero, zmm_zero, zmm_zero);
        }

    } else {

        for (int oc = 0; oc < oc_buffs; oc++) {
            for (int oi = 0; oi < ow; oi++) {
                vmovups(EVEX_compress_addr_safe(reg_out, (oc * oc_block * ow 
                            + oi * oc_block) * typesize,
                            reg_long_offt), zmm_zero);
                prefetcht1(EVEX_compress_addr_safe(reg_out_prf, (oc * oc_block * ow 
                            + oi * oc_block) * typesize,
                            reg_long_offt));
            }
        }
    }

    L(no_init_label);

    cmp(reg_kh, 0);
    je(end_label, T_NEAR);

    Reg64 reg_long_offt = reg_kj;

    auto get_reg_idx = [=](int oi, int oc_buff) {
        
        if (oc_buffs * (kw + 1) <= 30) {
            return oc_buff * (kw + 1) + oi % (kw + 1);
        } else {
            return oc_buff * kw + oi % kw;
        }
    };

    auto comp_unrolled = [&](int ii, int step, int cur_oc_buffs) {

        Reg64 mask_reg = reg_oi;
        Reg32 ic_itr_reg = reg_kh.cvt32();
        Reg64 lzcnt_reg = reg_channel;

        kmovw(mask_reg.cvt32(), k7);
        popcnt(ic_itr_reg, mask_reg.cvt32());

        if (ii < iw - step) { // pipelined
            size_t aux_src_offset = typesize * (ii + step) * ic_block;
            vcmpps(k7, zmm_zero, EVEX_compress_addr_safe(reg_inp, aux_src_offset,
                                    reg_long_offt), 4);

            prefetcht1(EVEX_compress_addr_safe(reg_inp_prf, aux_src_offset,
                                    reg_long_offt));
        }

        cout << ii << ":";

        if (ii >= step) {

            cout << " st:";

            for (int ki = kw - 1; ki > -1; ki--) {
                for (int oc_buff = 0; oc_buff < cur_oc_buffs; oc_buff++) {

                    int n = (ii - step) - ki * dilate_w + l_pad;

                    if (n >= 0 && n / stride_w < ow && n % stride_w == 0) {

                        int oi = n / stride_w;

                        int reg_idx = get_reg_idx(oi, oc_buff);

                        Zmm zmm = Xbyak::Zmm(reg_idx);

                        if (stride_w / dilate_w >= kw || dilate_w > stride_w || stride_w % dilate_w != 0
                            || ki == kw - 1) {

                            if (oc_buff == 0) {
                                cout << " " << oi << "-r" << reg_idx;
                            }
                        
                            size_t aux_dst_offset = (size_t)typesize
                                * (oc_buff * oc_block * ow * mb_block
                                + oi * oc_block);
                            vmovups(EVEX_compress_addr_safe(reg_out, aux_dst_offset,
                                        reg_long_offt), zmm);
                        }
                    }
                }
            }
        }

        cout << " ld:";

        if (cur_oc_buffs * (kw + 1) <= 30) {

            if (ii == 0) {
                for (int ki = kw - 1; ki > -1; ki--) {
                    for (int oc_buff = 0; oc_buff < cur_oc_buffs; oc_buff++) {

                        int n = - ki * dilate_w + l_pad;

                        if (n >= 0 && n / stride_w < ow && n % stride_w == 0) {

                            int oi = n / stride_w;

                            int reg_idx = get_reg_idx(oi, oc_buff);

                            Zmm zmm = Xbyak::Zmm(reg_idx);

                            if (oc_buff == 0) {
                                cout << " " << oi << "-r" << reg_idx;
                            }

                            size_t aux_dst_offset = (size_t)typesize
                                * (oc_buff * oc_block * ow * mb_block
                                + oi * oc_block);

                            vmovups(zmm, EVEX_compress_addr_safe(reg_out, aux_dst_offset,
                                        reg_long_offt));
                            //prefetcht1(EVEX_compress_addr_safe(reg_out_prf, aux_dst_offset,
                                        //reg_long_offt));

                        }
                    }
                }

            }
            if (ii < iw - step) {
            
                for (int ki = kw - 1; ki > -1; ki--) {
                    for (int oc_buff = 0; oc_buff < cur_oc_buffs; oc_buff++) {

                        int n = (ii + step) - ki * dilate_w + l_pad;

                        if (n >= 0 && n / stride_w < ow && n % stride_w == 0) {

                            int oi = n / stride_w;

                            int reg_idx = get_reg_idx(oi, oc_buff);

                            Zmm zmm = Xbyak::Zmm(reg_idx);

                            if (stride_w / dilate_w >= kw || dilate_w > stride_w || stride_w % dilate_w != 0
                                || ki == 0) {

                                if (oc_buff == 0) {
                                    cout << " " << oi << "-r" << reg_idx;
                                }

                                size_t aux_dst_offset = (size_t)typesize
                                    * (oc_buff * oc_block * ow * mb_block
                                    + oi * oc_block);

                                vmovups(zmm, EVEX_compress_addr_safe(reg_out, aux_dst_offset,
                                            reg_long_offt));
                                //prefetcht1(EVEX_compress_addr_safe(reg_out_prf, aux_dst_offset,
                                            //reg_long_offt));

                            }
                        }
                    }
                }
            }/* else {
                for (int ki = kw - 1; ki > -1; ki--) {
                    for (int oc_buff = 0; oc_buff < cur_oc_buffs; oc_buff++) {

                        int n = - ki * dilate_w + l_pad;

                        if (n >= 0 && n / stride_w < ow && n % stride_w == 0) {

                            int oi = n / stride_w;

                            size_t aux_dst_offset = (size_t)typesize
                                * (oc_buff * oc_block * ow * mb_block
                                + oi * oc_block);

                            //prefetcht1(EVEX_compress_addr_safe(reg_out_prf, aux_dst_offset,
                            //            reg_long_offt));

                        }
                    }
                }

            }*/

        } else {
            for (int ki = kw - 1; ki > -1; ki--) {
                for (int oc_buff = 0; oc_buff < cur_oc_buffs; oc_buff++) {

                    int n = ii - ki * dilate_w + l_pad;

                    if (n >= 0 && n / stride_w < ow && n % stride_w == 0) {

                        int oi = n / stride_w;

                        int reg_idx = get_reg_idx(oi, oc_buff);

                        Zmm zmm = Xbyak::Zmm(reg_idx);

                        if (stride_w / dilate_w >= kw || dilate_w > stride_w || stride_w % dilate_w != 0
                            || ii == 0 || ki == 0) {

                            if (oc_buff == 0) {
                                cout << " " << oi << "-r" << reg_idx;
                            }

                            size_t aux_dst_offset = (size_t)typesize
                                * (oc_buff * oc_block * ow * mb_block
                                + oi * oc_block);

                            vmovups(zmm, EVEX_compress_addr_safe(reg_out, aux_dst_offset,
                                        reg_long_offt));
                            //prefetcht1(EVEX_compress_addr_safe(reg_out_prf, aux_dst_offset,
                                        //reg_long_offt));

                        }
                    }
                }
            }
        }


        Label ic_loop_end_label;
        jz(ic_loop_end_label, T_NEAR);

        tzcnt(lzcnt_reg.cvt32(), mask_reg.cvt32());
        inc(lzcnt_reg.cvt32());

        shrx(mask_reg.cvt32(), mask_reg.cvt32(), lzcnt_reg.cvt32());

        mov(aux_reg_ker, reg_ker);
        mov(aux_reg_inp, reg_inp);

        Label ic_loop_label;
        L(ic_loop_label); {

            lea(aux_reg_inp, ptr[aux_reg_inp + lzcnt_reg * typesize]);

            int aux_src_offset = typesize * (ii * ic_block - 1);
            vbroadcastss(zmm_o, ptr[aux_reg_inp + aux_src_offset]);

            shl(lzcnt_reg.cvt32(), 6);
            add(aux_reg_ker, lzcnt_reg);

            tzcnt(lzcnt_reg.cvt32(), mask_reg.cvt32()); // pipelined
            inc(lzcnt_reg.cvt32());

            dec(ic_itr_reg);

            shrx(mask_reg.cvt32(), mask_reg.cvt32(), lzcnt_reg.cvt32()); // does not change flags

            cout << " op:";

            for (int oc_buff = 0; oc_buff < cur_oc_buffs; oc_buff++) {                
                for (int ki = 0; ki < kw; ki++) {

                    int n = ii - ki * dilate_w + l_pad;

                    if (n >= 0 && n / stride_w < ow && n % stride_w == 0) {

                        int oi = n / stride_w;

                        int reg_idx = get_reg_idx(oi, oc_buff);

                        Zmm zmm = Xbyak::Zmm(reg_idx);

                        if (oc_buff == 0) {
                            cout << " " << oi << "-r" << reg_idx;
                        }
                    
                        size_t aux_kernel_offset = typesize * (oc_buff
                                * kw * oc_block * ic_block
                                + ki * oc_block * ic_block);
                        vfmadd231ps(zmm, zmm_o,
                                EVEX_compress_addr_safe(aux_reg_ker, aux_kernel_offset,
                                    reg_long_offt)); // probably don't need safe for weight tensor
                    }

                }
            }

            cout << endl;

            jnz(ic_loop_label, T_NEAR);
        }

        L(ic_loop_end_label);


        if (ii >= iw - step) {

            for (int ki = kw - 1; ki > -1; ki--) {
                for (int oc_buff = 0; oc_buff < cur_oc_buffs; oc_buff++) {

                    int n = ii - ki * dilate_w + l_pad;

                    if (n >= 0 && n / stride_w < ow && n % stride_w == 0) {

                        int reg_idx = get_reg_idx(n / stride_w, oc_buff);

                        Zmm zmm = Xbyak::Zmm(reg_idx);
                        
                        size_t aux_dst_offset = (size_t)typesize
                            * (oc_buff * oc_block * ow * mb_block
                            + n / stride_w * oc_block);
                        vmovups(EVEX_compress_addr_safe(reg_out, aux_dst_offset,
                                    reg_long_offt), zmm);
                    }
                }
            }

        }


    };

    auto outer_loop = [&](int cur_oc_buffs) {

        sub(reg_ker, oc_block * typesize);

        int rotation_unroll_factor = cur_oc_buffs * (kw + 1) <= 30 ? (kw + 1) * stride_w : kw * stride_w;
        int l_iw = kw > l_pad ? kw - l_pad : 1;
        l_iw++; // unroll one more due to pipelined vector write

        int r_iw = iw - 1 - (iw - 1 - l_iw) % rotation_unroll_factor;

        if (l_iw <= r_iw - rotation_unroll_factor * 5) { // threshold needs to be dynamically calculated based on the instruction count per section


            int niter = (r_iw - l_iw) / rotation_unroll_factor;

            cout << "nr:" << nr << " l_iw:" << l_iw << " r_iw:" << r_iw << " oc_iters:" << oc_iters
                << " oc_buffs:" << oc_buffs << endl;

            cout << "leading :" << l_iw << " trailing:" << ow - r_iw
                << " factor:" << rotation_unroll_factor
                << " niter:" << niter << endl;

            int istart = 0;
            int step = 1; // for now

            for (int ii = istart; ii < l_iw; ii++) {
                comp_unrolled(ii, step, cur_oc_buffs);
            }

            Reg64 iw_itr_reg = reg_out_prf;

            mov(iw_itr_reg, niter);

            Label iw_loop_label;
            L(iw_loop_label); {

                for (int i = 0; i < rotation_unroll_factor; i++) {
                    comp_unrolled(l_iw + i, step, cur_oc_buffs);
                }

                add(reg_inp, ic_block * typesize * rotation_unroll_factor);
                add(reg_out, oc_block * typesize * rotation_unroll_factor / stride_w);

                add(reg_inp_prf, ic_block * typesize * rotation_unroll_factor);

                dec(iw_itr_reg);
                jnz(iw_loop_label, T_NEAR);

            }

            sub(reg_inp, ic_block * typesize * rotation_unroll_factor * niter);
            sub(reg_out, oc_block * typesize * rotation_unroll_factor / stride_w * niter);
            sub(reg_inp_prf, ic_block * typesize * rotation_unroll_factor * niter);

            for (int ii = r_iw; ii < iw; ii++) {
                comp_unrolled(ii, step, cur_oc_buffs);
            }

        } else {

            cout << "fully unrolled oc_buffs:" << oc_buffs << endl;

            int istart = 0;
            int step = 1;

            if (kw == 1 && stride_w > 1) {
                istart = l_pad % 2;
                step = stride_w;
            }

            for (int ii = istart; ii < iw; ii += step) {
                comp_unrolled(ii, step, cur_oc_buffs);
            }
        }
    };

    outer_loop(oc_buffs);

    L(end_label);
}

void jit_avx512_common_conv_fwd_kernel::generate()
{
    preamble();

    mov(reg_inp, ptr[param1 + GET_OFF(src)]);
    mov(reg_out, ptr[param1 + GET_OFF(dst)]);
    mov(reg_ker, ptr[param1 + GET_OFF(filt)]);
    mov(reg_ker_prf, ptr[param1 + GET_OFF(filt_prf)]);
    mov(reg_kh, ptr[param1 + GET_OFF(kh_padding)]);

    mov(reg_inp_prf, ptr[param1 + GET_OFF(src_prf)]);
    mov(reg_out_prf, ptr[param1 + GET_OFF(dst_prf)]);
    compute_loop_fma_sparse();

    postamble();
}

bool jit_avx512_common_conv_fwd_kernel::post_ops_ok(
        jit_conv_conf_t &jcp, const primitive_attr_t &attr) {
    const auto &p = attr.post_ops_;

    auto is_relu = [&](int idx) { return p.entry_[idx].is_relu(); };
    auto is_sum = [&](int idx) { return p.entry_[idx].is_sum(); };

    switch (p.len_) {
    case 0: return true; // no post_ops
    case 1:
        return true // sum OR relu
                && !jcp.with_relu && (is_relu(0) || is_sum(0));
    case 2:
        return true // sum->relu
                && !jcp.with_relu && (is_sum(0) && is_relu(1));
    default: return false;
    }

    return false;
}

status_t jit_avx512_common_conv_fwd_kernel::init_conf(
            jit_conv_conf_t &jcp,
            const convolution_desc_t &cd, cpu_memory_t::pd_t &src_pd,
            cpu_memory_t::pd_t &weights_pd, cpu_memory_t::pd_t &dst_pd,
            cpu_memory_t::pd_t &bias_pd, const primitive_attr_t &attr,
            int nthreads, bool with_relu, float relu_negative_slope)
{
    using namespace prop_kind;

    if (!mayiuse(avx512_common))
        return status::unimplemented;

    const int simd_w = cpu_isa_traits<avx512_common>::vlen / sizeof(float);

    const memory_desc_wrapper src_d(&src_pd);
    const memory_desc_wrapper weights_d(&weights_pd);
    const memory_desc_wrapper dst_d(&dst_pd);
    const memory_desc_wrapper bias_d(&bias_pd);

    const int regs = 28;
    const bool with_groups = weights_d.ndims() == src_d.ndims() + 1;
    int ndims = src_d.ndims();

    jcp = zero<decltype(jcp)>();
    jcp.ndims = ndims;
    jcp.prop_kind = cd.prop_kind;
    jcp.ngroups = with_groups ? weights_d.dims()[0] : 1;
    jcp.mb = src_d.dims()[0];
    jcp.oc = dst_d.dims()[1] / jcp.ngroups;
    jcp.oc_without_padding = jcp.oc;
    jcp.ic = src_d.dims()[1] / jcp.ngroups;
    jcp.id = (ndims == 5) ? src_d.dims()[2] : 1;
    jcp.ih = src_d.dims()[ndims-2];
    jcp.iw = src_d.dims()[ndims-1];
    jcp.od = (ndims == 5) ? dst_d.dims()[2] : 1;
    jcp.oh = dst_d.dims()[ndims-2];
    jcp.ow = dst_d.dims()[ndims-1];
    jcp.kd = (ndims == 5) ? weights_d.dims()[with_groups + 2] : 1;
    jcp.kh = weights_d.dims()[with_groups + ndims-2];
    jcp.kw = weights_d.dims()[with_groups + ndims-1];
    jcp.f_pad = (ndims == 5) ? cd.padding[0][0] : 0;
    jcp.t_pad = cd.padding[0][ndims-4];
    jcp.l_pad = cd.padding[0][ndims-3];
    jcp.stride_d = (ndims == 5) ? cd.strides[0] : 1;
    jcp.stride_h = cd.strides[ndims-4];
    jcp.stride_w = cd.strides[ndims-3];
    jcp.src_fmt = src_d.format();
    jcp.with_relu = with_relu;
    jcp.relu_negative_slope = relu_negative_slope;

    jcp.dilate_d = (ndims == 5) ? cd.dilates[0] : 0;
    jcp.dilate_h = cd.dilates[ndims-4];
    jcp.dilate_w = cd.dilates[ndims-3];

    jcp.b_pad = (jcp.oh - 1) * jcp.stride_h + (jcp.kh - 1) * (jcp.dilate_h + 1)
            - (jcp.ih + jcp.t_pad - 1);
    jcp.back_pad = (jcp.od - 1) * jcp.stride_d
            + (jcp.kd - 1) * (jcp.dilate_d + 1) - (jcp.id + jcp.f_pad - 1);

    jcp.is_1stconv = is_1stconv(jcp);

    jcp.oc_block = simd_w;
    jcp.ic_block = jcp.is_1stconv ? jcp.ic : simd_w;
    jcp.aligned_threads = 0;

    bool ok_to_pad_channels = true
        && jcp.ngroups == 1
        && src_d.data_type() == data_type::f32;

    if (ok_to_pad_channels) {
        jcp.oc = rnd_up(jcp.oc, jcp.oc_block);
        jcp.ic = rnd_up(jcp.ic, jcp.ic_block);
    }
    bool args_ok = true
        && jcp.oc % jcp.oc_block == 0
        && jcp.ic % jcp.ic_block == 0;
    if (!args_ok)
        return status::unimplemented;

    if (!post_ops_ok(jcp, attr))
        return status::unimplemented;

    const auto &p = attr.post_ops_;
    jcp.with_sum = p.find(primitive_kind::sum) != -1;
    if (!jcp.with_relu) {
        jcp.with_relu = p.find(primitive_kind::eltwise) != -1;
        jcp.relu_negative_slope = 0;
    }

    auto src_format = (ndims == 5)
        ? (jcp.is_1stconv) ? ncdhw : NhC16nw16c
        : (jcp.is_1stconv) ? nchw : NhC16nw16c;
    auto dst_format = (ndims == 5) ? nCdhw16c : NhC16nw16c;
    auto wei_format = (ndims == 5)
        ? (with_groups) ? hIOw16i16o : hIOw16i16o
        : (with_groups) ? hIOw16i16o : hIOw16i16o;

    if (src_d.format() == any)
        CHECK(src_pd.set_format(src_format));
    if (dst_d.format() == any)
        CHECK(dst_pd.set_format(dst_format));

    switch (src_d.format()) {
        case NhC8nw16c: jcp.mb_block = 8; if (dst_d.format() != NhC8nw16c) return status::unimplemented; break;
        case NhC16nw16c: jcp.mb_block = 16; if (dst_d.format() != NhC16nw16c) return status::unimplemented; break;
        case NhC32nw16c: jcp.mb_block = 32; if (dst_d.format() != NhC32nw16c) return status::unimplemented; break;
        case NhC64nw16c: jcp.mb_block = 64; if (dst_d.format() != NhC64nw16c) return status::unimplemented; break;
        default: return status::unimplemented; break;
    }

    jcp.with_bias = cd.bias_desc.format != memory_format::undef;
    if (jcp.with_bias) {
        if (bias_d.format() == any)
            CHECK(bias_pd.set_format(x));
        if (bias_d.format() != x)
            return status::unimplemented;
    }

    if ((mayiuse(avx512_mic_4ops) || mayiuse(avx512_core_vnni))
         && src_d.data_type() == data_type::s16
         && weights_d.data_type() == data_type::s16
         && dst_d.data_type() == data_type::s32)
    {
        if (jcp.is_1stconv)
            return status::unimplemented;

        if (mayiuse(avx512_mic_4ops)) {
            jcp.ver = ver_4vnni;
        } else {
            jcp.ver = ver_vnni;
        }
        jcp.typesize_in = sizeof(int16_t);
        jcp.typesize_out = sizeof(int32_t);

        const auto w_format = (ndims == 5)
            ? with_groups ? gOIdhw8i16o2i : OIdhw8i16o2i
            : with_groups ? gOIhw8i16o2i : OIhw8i16o2i;
        if (weights_d.format() == any)
            CHECK(weights_pd.set_format(w_format));
        if (weights_d.format() != w_format)
            return status::unimplemented;
    } else if (mayiuse(avx512_common) &&
            src_d.data_type() == data_type::f32
         && weights_d.data_type() == data_type::f32
         && dst_d.data_type() == data_type::f32) {
        jcp.ver = ver_fma;
        jcp.typesize_in = sizeof(float);
        jcp.typesize_out = sizeof(float);
        if (mayiuse(avx512_mic_4ops))
           jcp.ver = ver_4fma;

        if (jcp.is_1stconv) {
            // TODO: fix & remove constraints below
            if (jcp.l_pad != 0 || jcp.r_pad != 0
                || jcp.b_pad != 0 || jcp.t_pad != 0
                || (jcp.kw < 7 && jcp.kh < 7))
                jcp.ver = ver_fma;
            if (jcp.ver == ver_4fma) {
                const auto w_format = (ndims == 5)
                    ? (with_groups) ? gOidhw16o : Oidhw16o
                    : (with_groups) ? gOihw16o : Oihw16o;
                if (weights_d.format() == any)
                    CHECK(weights_pd.set_format(w_format));
                if (weights_d.format() != w_format)
                    return status::unimplemented;
            } else {
                const auto w_format = (ndims == 5)
                    ? (with_groups) ? gOdhwi16o : Odhwi16o
                    : (with_groups) ? gOhwi16o : Ohwi16o;
                if (weights_d.format() == any)
                    CHECK(weights_pd.set_format(w_format));
                if (weights_d.format() != w_format)
                    return status::unimplemented;
            }
        } else {
            if (weights_d.format() == any)
                CHECK(weights_pd.set_format(wei_format));
            switch (weights_d.format()) {
                case hIOw16i16o:
                case IhOw16i16o: break;
                default: return status::unimplemented; break;
            }
        }
    } else {
        return status::unimplemented;
    }

    if (jcp.is_1stconv) {
        jcp.ur_w = nstl::min(jcp.ow, regs);
    } else {
        // avx512_core guard - just to avoid possible regression for other archs
        if (jcp.ver == ver_fma && mayiuse(avx512_core)) {
            jcp.ur_w = nstl::min(jcp.ow, regs);
        } else {
            for (int ur_w = regs; ur_w > 0; --ur_w) {
                if (jcp.ow % ur_w == 0) {
                    jcp.ur_w = ur_w;
                    break;
                }
            }
        }
        if ((ndims == 5 && jcp.ur_w <= 8) || (jcp.ur_w <= 1)) {
            jcp.ur_w = nstl::min(jcp.ow, regs);
        }
    }
    // TODO (Tanya): currently applied to Segnet convolutions only.
    // Need to try for other topologies
    if (jcp.ow > 150 && jcp.ur_w < regs/2)
        jcp.ur_w = regs;

    int n_oi = (jcp.ow / jcp.ur_w);
    int r_pad = (jcp.ur_w * n_oi - 1) * jcp.stride_w
            + (jcp.kw - 1) * (jcp.dilate_w + 1) - (jcp.iw + jcp.l_pad - 1);
    if (jcp.l_pad > 0 && r_pad > 0)
        n_oi--;

    bool large_code_size = jcp.ur_w != jcp.ow && jcp.l_pad > 0 && r_pad > 0
            && ((jcp.l_pad <= 0 && n_oi > 0) || (jcp.l_pad > 0 && n_oi > 1));
    if (large_code_size) {
        const int max_code_size = 24 * 1024;
        const int num_ops_per_reg = 6 + jcp.ic_block * jcp.kw;
        int mult = 1;
        if (jcp.l_pad > 0) mult += 1;
        if (r_pad > 0) mult += 1;
        for (int ur_w = jcp.ur_w; ur_w > regs/2; --ur_w) {
            if (ur_w * mult * num_ops_per_reg * 9.0 < max_code_size) {
                jcp.ur_w = ur_w;
                break;
            }
        }
    }

    jcp.nb_ic = jcp.ic / jcp.ic_block;
    jcp.nb_oc = jcp.oc / jcp.oc_block;
    jcp.nb_ic_blocking = jcp.nb_oc_blocking = 1;

    if (jcp.ver == ver_4vnni) {
        jcp.kernel_kind = embd_bcast;
    }
    if (jcp.ver == ver_vnni) {
        // TODO: kernel_kind and nb_oc_blocking selection
        //       should be tuned on real HW
        if (jcp.ow <= 8 && jcp.oh <= 8 && jcp.od <= 8) {
            jcp.kernel_kind = expl_bcast;
            jcp.nb_oc_blocking = 2;
        } else {
            jcp.kernel_kind = embd_bcast;
            jcp.nb_oc_blocking = 2;
        }
        if (jcp.nb_oc_blocking > 1) {
            if (jcp.nb_oc < jcp.nb_oc_blocking) jcp.nb_oc_blocking = jcp.nb_oc;
            if (jcp.nb_oc % jcp.nb_oc_blocking != 0)
                for (int i = jcp.nb_oc_blocking; i > 0; i--)
                    if (jcp.nb_oc % i == 0) {
                        jcp.nb_oc_blocking = i;
                        break;
                    }
            jcp.ur_w = 31 / (jcp.nb_oc_blocking + 1);
            if (jcp.ow < jcp.ur_w)
                jcp.ur_w = jcp.ow;
        }
    }

    if (one_of(jcp.ver, ver_4vnni, ver_4fma) && !jcp.is_1stconv) {
        if (jcp.kw == 3 && jcp.kh == 3 && jcp.ow == 7 && jcp.oh == 7) {
            if (jcp.nb_oc % 2 == 0)
                jcp.nb_oc_blocking = 2;
        } else {
            for (int i = jcp.nb_oc; i > 0; i--)
                if (i * jcp.ur_w <= regs && jcp.nb_oc % i == 0) {
                    jcp.nb_oc_blocking = i;
                    break;
                }
        }
    }

    if (jcp.ver == ver_fma && mayiuse(avx512_core)) {
        int try_nb_oc_blocking = 2;
        unsigned int ker_inp_size = typesize * (jcp.iw / jcp.stride_w)
            * jcp.ic_block * jcp.kh * jcp.kd;
        unsigned int ker_out_size = typesize * jcp.ow * jcp.oc_block
            * try_nb_oc_blocking;
        unsigned int ker_wei_size = typesize * jcp.kh * jcp.kw * jcp.ic_block
            * jcp.oc_block * try_nb_oc_blocking * jcp.kd;
        unsigned int ker_total_size = ker_inp_size + ker_out_size
            + ker_wei_size;

        bool embd_bcast_condition = true
            && (jcp.kw == 3 && jcp.ow <= 28 && ker_total_size < L1_cache_size)
            && !(jcp.kw == 3 && jcp.ow == 13 && jcp.ic >= 192)
            && !(jcp.kw == 3 && jcp.ow == 28 && jcp.ic >= 512);

        if (jcp.mb == 1) {
            jcp.kernel_kind = embd_bcast;
            unsigned int inp_size = jcp.mb * (jcp.ih / jcp.stride_h)
                    * (jcp.iw / jcp.stride_w) * jcp.ic;
            unsigned int wei_size = jcp.ic * jcp.oc * jcp.kh * jcp.kw;

            // Estimate whether we need to limit the number of threads
            // and calculate this number. Includes some heuristic.
            int oc_chunks = jcp.nb_oc / jcp.nb_oc_blocking;
            int work_amount = jcp.mb * jcp.ngroups * oc_chunks * jcp.oh;
            int job_size_min = work_amount / nthreads;
            int job_size_max = div_up(work_amount, nthreads);
            int ch_max = rnd_up(jcp.oh, job_size_max);
            int ch_min = (job_size_min == 0)
                ? jcp.oh
                : rnd_up(jcp.oh, job_size_min);
            bool not_aligned_max = ch_max % jcp.oh != 0 && ch_max / jcp.oh < 2
                    && (jcp.oh != 8 || ch_max / jcp.oh > 1);
            bool not_aligned_min = ch_min % jcp.oh != 0 && ch_min / jcp.oh < 2
                    && (jcp.oh != 8 || ch_min / jcp.oh > 1);
            bool eligible_case = (jcp.stride_h == 1 && jcp.stride_w == 1)
                    || nthreads > oc_chunks;
            if (jcp.loop_order == loop_cgn && oc_chunks > 1 && nthreads > 1
                && wei_size / inp_size > 24
                && (not_aligned_max || not_aligned_min)
                && eligible_case) {
                jcp.aligned_threads = nthreads;
                for (int i = nthreads; i > 0; i--) {
                    if (oc_chunks % i == 0 || i % oc_chunks == 0) {
                        jcp.aligned_threads = i;
                        break;
                    }
                }
            }
        } else if (jcp.kw > 3
            || (jcp.stride_w == 1 && jcp.stride_h == 1
                && embd_bcast_condition)
            || ((jcp.stride_w != 1 || jcp.stride_h != 1)
                && ((jcp.mb <= 16 && (jcp.oc <= 192 || jcp.oh <= 10)
                     && embd_bcast_condition)))
            ) {
            jcp.kernel_kind = embd_bcast;
            jcp.ur_w = nstl::min(jcp.ow, regs);
            jcp.nb_ic_blocking = jcp.nb_oc_blocking = 1;
            if (ker_total_size < L1_cache_size && jcp.ow <= 8 && jcp.kh <= 3
                && jcp.kw <= 3) {
                if (jcp.nb_oc % try_nb_oc_blocking == 0 && !jcp.is_1stconv) {
                    jcp.nb_oc_blocking = try_nb_oc_blocking;
                    jcp.ur_w = 31 / (jcp.nb_oc_blocking + 1);
                    if (jcp.ow < jcp.ur_w)
                        jcp.ur_w = jcp.ow;
                }
            }
        } else {
            jcp.kernel_kind = expl_bcast;
            jcp.nb_ic_blocking = 1;
            jcp.nb_oc_blocking = 4;
            if (jcp.nb_oc < jcp.nb_oc_blocking) jcp.nb_oc_blocking = jcp.nb_oc;
            if (jcp.nb_oc % jcp.nb_oc_blocking != 0)
                for (int i = jcp.nb_oc_blocking; i > 0; i--)
                    if (jcp.nb_oc % i == 0) {
                        jcp.nb_oc_blocking = i;
                        break;
                    }
            jcp.ur_w = 31 / (jcp.nb_oc_blocking + 1);
            if (jcp.ow < jcp.ur_w)
                jcp.ur_w = jcp.ow;
        }
    }


    jcp.kernel_kind = embd_bcast;
    jcp.ur_w = nstl::min(jcp.ow, regs);
    jcp.nb_ic_blocking = jcp.nb_oc_blocking = 1;

    jcp.ur_w_tail = jcp.ow % jcp.ur_w;

    args_ok = true
        && jcp.l_pad <= jcp.ur_w
        && jcp.ic <= src_d.blocking_desc().padding_dims[1]
        && jcp.oc <= dst_d.blocking_desc().padding_dims[1]
        && jcp.ic <= weights_d.blocking_desc().padding_dims[with_groups + 1]
        && jcp.oc <= weights_d.blocking_desc().padding_dims[with_groups + 0];
    if (!args_ok)
        return status::unimplemented;

    int r_pad_no_tail = nstl::max(0, (jcp.ow - jcp.ur_w_tail - 1) * jcp.stride_w
                    + (jcp.kw - 1) * (jcp.dilate_w + 1)
                    - (jcp.iw + jcp.l_pad - 1));
    if (r_pad_no_tail > jcp.ur_w)
        return status::unimplemented;

    pick_loop_order(jcp);

    jcp.nb_ic_L2 = jcp.nb_ic;

    // TODO check for 4vnni
    if (jcp.ver == ver_4fma) {
        for (int divf = 2, temp_nb = jcp.nb_ic_L2; divf <= jcp.nb_ic;
              divf++) {
            size_t l2_src
                = (size_t)jcp.iw * jcp.ic_block * jcp.ih * temp_nb * jcp.id;
            size_t l2_dst = (size_t)jcp.ow * jcp.oc_block * jcp.nb_oc_blocking
                * jcp.oh * jcp.od;
            size_t l2_filt = (size_t)jcp.kw * jcp.oc_block * jcp.ic_block
                * jcp.kh * jcp.nb_oc_blocking * temp_nb * jcp.kd;
            if (4 * (l2_src + l2_dst + l2_filt) > KNx_L2_EFFECTIVE_CAPACITY) {
                if (jcp.kh == 3 && jcp.oh == 7) {
                    jcp.nb_ic_L2 = 1;
                    break;
                }
                temp_nb = (jcp.nb_ic_L2 % divf == 0 ? jcp.nb_ic_L2 / divf
                                : jcp.nb_ic_L2);
            } else {
                jcp.nb_ic_L2 = temp_nb;
                break;
            }
        }
    }

    int nregs = 30;
    int ur_sparse;
    if (jcp.kw * jcp.nb_oc <= nregs)
        ur_sparse = jcp.kw * jcp.nb_oc;
    else {
        for (int tmp_ur_w = nregs; tmp_ur_w > 0; tmp_ur_w--)
            if (tmp_ur_w % jcp.kw == 0) {
                ur_sparse = tmp_ur_w;
                break;
            }
    }

    jcp.oc_buffs = ur_sparse / jcp.kw;
    for (int i = jcp.oc_buffs; i > 0; i--) {
        if (jcp.nb_oc % i == 0) {
            jcp.oc_buffs = i;
            break;
        }
    }

    // higher than 8 will cause subpar memory performance
    if (jcp.oc_buffs > 8) jcp.oc_buffs = 8;

    jcp.ur_sparse = jcp.oc_buffs * jcp.kw;
    jcp.nb_mb = jcp.mb / jcp.mb_block;

    cout << "stride_w:" << jcp.stride_w << " stride_h:" << jcp.stride_h
        << " l_pad:" << jcp.l_pad << " r_pad:" << jcp.r_pad
        << " t_pad:" << jcp.t_pad << " b_pad:" << jcp.b_pad
        << " iw:" << jcp.iw << " ih:" << jcp.ih << " ic:" << jcp.ic
        << " ow:" << jcp.ow << " oh:" << jcp.oh << " oc:"<< jcp.oc
        << " kw:" << jcp.kw << " kh:"<< jcp.kh << " mb:" << jcp.mb
        << " nb_ic_blocking:" << jcp.nb_ic_blocking
        << " nb_oc_blocking:" << jcp.nb_oc_blocking
        << " ngroups:" << jcp.ngroups << " dilate_w:" << jcp.dilate_w
        << " typesize_in:" << jcp.typesize_in
        << " typesize_out:" << jcp.typesize_out << " with_bias:" << jcp.with_bias
        << " with_sum:" << jcp.with_sum << " with_relu:" << jcp.with_relu << endl;

    return status::success;
}

void jit_avx512_common_conv_bwd_data_kernel_f32::compute_loop_fma_sparse() {

    /**********************************************

    reg64_t param = abi_param1;
    reg64_t reg_dst = r8;
    reg64_t reg_ker = r9;
    reg64_t reg_src = r10;

    reg64_t reg_dst_prf = r11;
    reg64_t reg_ker_prf = r12;
    reg64_t reg_src_prf = r13;

    reg64_t aux_reg_dst = r14;
    reg64_t aux_reg_ker = r15;

    reg64_t aux_reg_dst_prf = rsi;
    reg64_t aux_reg_ker_prf = rdx;

    reg64_t aux_reg_dst_d_prf = r13;
    reg64_t aux_reg_dst_d = rbx;
    reg64_t aux_reg_ker_d_prf = abi_not_param1;
    reg64_t aux_reg_ker_d = r9;
    reg64_t reg_ki = r10;

    reg64_t reg_kj = rax;
    reg64_t reg_oi = rbx;
    reg64_t reg_kh = abi_not_param1;

    reg64_t reg_channel = rsi;

    reg64_t reg_tmp = rbp;
    reg64_t reg_long_offt = r14;

    ***********************************************/


    Label kh_label, kd_label, skip_kd_loop;
    Label end_label, clear_label;

    int kw = jcp.kw;
    int kh = jcp.kh;

    int ow = jcp.ow;
    int oh = jcp.oh;

    int nb_ic = jcp.nb_ic;

    int iw = jcp.iw;
    int ih = jcp.ih;

    int mb_block = jcp.mb_block;

    int ic_block = jcp.ic_block;
    int oc_block = jcp.oc_block;
    int l_pad = jcp.l_pad;
    int dilate_w = jcp.dilate_w + 1;
    int stride_w = jcp.stride_w;
    int stride_h = jcp.stride_h;

    int ker_pipeline_depth = 4;
    assert(ker_reg_base_idx + ker_pipeline_depth <= 32);
    assert(oc_block >= ker_pipeline_depth);

    int nr = jcp.ur_sparse;
    int ic_buffs = jcp.ic_buffs;

    assert(nr >= kw);
    assert(ic_block == 16); // needed for specific optimization
    assert(typesize == 4);

    int ic_iters = nb_ic / ic_buffs;

    int l_ow = 1;
    while (l_ow * stride_w - l_pad < 0) { // left unroll factor, first iteration always peeled
        l_ow++;
    }
    l_ow++; // unroll one more due to pipelined vector write

    int r_ow = ow - 2;
    while (r_ow * stride_w - l_pad + (kw - 1) * dilate_w >= iw) { // right unroll factor, last iteration always peeled
        r_ow--;
    }

    int rotation_unroll_factor;
    int rw = ic_buffs * (kw + 1) <= 30 && stride_w == 1 && kw > 1 ? kw + 1 : kw;

    if (stride_w / dilate_w < kw && dilate_w <= stride_w && stride_w % dilate_w == 0) {
        rotation_unroll_factor = rw % (stride_w / dilate_w) == 0 ? rw / (stride_w / dilate_w) : rw;
    } else {
        rotation_unroll_factor = 1;
    }

    cout << "nr:" << nr << " l_ow:" << l_ow << " r_ow:" << r_ow << " ic_iters:" << ic_iters
        << " ic_buffs:" << ic_buffs << endl;

    auto zmm_o = Xbyak::Zmm(31);

    auto ymm_zero = Xbyak::Ymm(30);
    auto zmm_zero = Xbyak::Zmm(30);

    vpxord(zmm_zero, zmm_zero, zmm_zero);

    vcmpps(k7, zmm_zero, ptr[reg_dst], 4);
    prefetcht1(ptr[reg_dst_prf]);

    assert(nr % kw == 0);

    Label no_init_label;

    mov(reg_channel, ptr[param + GET_OFF(channel)]);
    cmp(reg_channel, 0);
    jne(no_init_label, T_NEAR);

    if (ic_buffs * iw > 128) { // threshold may be tweaked later

        Reg64 aux_reg_src = aux_reg_dst;
        Reg64 aux_reg_src_prf = aux_reg_ker;

        mov(aux_reg_src, reg_src);
        mov(aux_reg_src_prf, reg_src_prf);

        mov(reg_channel, ic_buffs);

        Label ic_loop_label;
        L(ic_loop_label);

        for (int ii = 0; ii < iw; ii++) {
            vmovups(EVEX_compress_addr_safe(aux_reg_src, ii * ic_block * typesize,
                        reg_long_offt), zmm_zero);
            prefetcht1(EVEX_compress_addr_safe(aux_reg_src_prf, ii * ic_block * typesize,
                        reg_long_offt));
        }

        add(aux_reg_src, typesize * ic_block * mb_block * iw);
        add(aux_reg_src_prf, typesize * ic_block * mb_block * iw);

        dec(reg_channel);
        cmp(reg_channel, 0);
        jne(ic_loop_label);

    } else {

        for (int ic = 0; ic < ic_buffs; ic++) {
            for (int ii = 0; ii < iw; ii++) {
                vmovups(EVEX_compress_addr_safe(reg_src, (ic * ic_block * mb_block * iw 
                            + ii * ic_block) * typesize,
                            reg_long_offt), zmm_zero);
                prefetcht1(EVEX_compress_addr_safe(reg_src_prf, (ic * ic_block * mb_block * iw 
                            + ii * ic_block) * typesize,
                            reg_long_offt));
            }
        }
    }

    L(no_init_label);

    cmp(reg_kh, 0);
    je(end_label, T_NEAR);

    Reg64 reg_long_offt = reg_kj;

    auto get_reg_idx = [=](int oi, int ic_buff, int ki) {

        int rotation_idx = oi % rotation_unroll_factor;
        
        if (stride_w / dilate_w < kw && dilate_w <= stride_w && stride_w % dilate_w == 0) {
            return ic_buff * rw + ((stride_w / dilate_w) * rotation_idx + ki) % rw;
        } else {
            return ic_buff * rw + ki;
        }
    };

    auto comp_unrolled = [&](int oi, int cur_ic_buffs) {

        Reg64 mask_reg = reg_oi;
        Reg32 oc_itr_reg = reg_kh.cvt32();
        Reg64 tzcnt_reg = reg_channel;

        kmovw(mask_reg.cvt32(), k7);
        popcnt(oc_itr_reg, mask_reg.cvt32());

        if (oi < ow - 1) { // pipelined
            size_t aux_dst_offset = typesize * (oi + 1) * oc_block;
            vcmpps(k7, zmm_zero, EVEX_compress_addr_safe(reg_dst, aux_dst_offset,
                                    reg_long_offt), 4);

            prefetcht1(EVEX_compress_addr_safe(reg_dst_prf, aux_dst_offset,
                                    reg_long_offt));
        }


        if (oi > 0) {
                
            for (int ki = 0; ki < kw; ki++) {
                for (int ic_buff = 0; ic_buff < cur_ic_buffs; ic_buff++) {

                    if ((oi - 1) * stride_w - l_pad + ki * dilate_w >= 0
                        && (oi - 1) * stride_w - l_pad + ki * dilate_w < iw) {

                        int reg_idx = get_reg_idx(oi - 1, ic_buff, ki);

                        Zmm zmm = Xbyak::Zmm(reg_idx);

                        Label no_update_label;

                        if (stride_w / dilate_w >= kw || dilate_w > stride_w || stride_w % dilate_w != 0
                            || ki < stride_w) {
                        
                            size_t aux_src_offset = (size_t)typesize
                                * (ic_buff * ic_block* mb_block * iw 
                                + ((oi - 1) * stride_w - l_pad + ki * dilate_w) * ic_block);
                            vmovups(EVEX_compress_addr_safe(reg_src, aux_src_offset,
                                        reg_long_offt), zmm);
                        }
                    }
                }
            }
        }

        if (cur_ic_buffs * (kw + 1) <= 30 && stride_w == 1 && kw > 1) {

            if (oi == 0) {
                for (int ki = 0; ki < kw; ki++) {
                    for (int ic_buff = 0; ic_buff < cur_ic_buffs; ic_buff++) {

                        if (oi * stride_w - l_pad + ki * dilate_w >= 0
                            && oi * stride_w - l_pad + ki * dilate_w < iw) {

                            int reg_idx = get_reg_idx(oi, ic_buff, ki);

                            Zmm zmm = Xbyak::Zmm(reg_idx);


                            size_t aux_src_offset = (size_t)typesize
                                * (ic_buff * ic_block * mb_block * iw 
                                + (oi * stride_w - l_pad + ki * dilate_w) * ic_block);

                            vmovups(zmm, EVEX_compress_addr_safe(reg_src, aux_src_offset,
                                        reg_long_offt));
                            //prefetcht1(EVEX_compress_addr_safe(reg_src_prf, aux_src_offset,
                            //            reg_long_offt));

                        }
                    }
                }
            }
            if (oi < ow - 1) {
                for (int ki = 0; ki < kw; ki++) {
                    for (int ic_buff = 0; ic_buff < cur_ic_buffs; ic_buff++) {

                        if ((oi + 1) * stride_w - l_pad + ki * dilate_w >= 0
                            && (oi + 1) * stride_w - l_pad + ki * dilate_w < iw) {

                            int reg_idx = get_reg_idx(oi + 1, ic_buff, ki);

                            Zmm zmm = Xbyak::Zmm(reg_idx);

                            if (stride_w / dilate_w >= kw || dilate_w > stride_w || stride_w % dilate_w != 0
                                || ki >= kw - stride_w) {

                                size_t aux_src_offset = (size_t)typesize
                                    * (ic_buff * ic_block * mb_block * iw 
                                    + ((oi + 1) * stride_w - l_pad + ki * dilate_w) * ic_block);

                                vmovups(zmm, EVEX_compress_addr_safe(reg_src, aux_src_offset,
                                            reg_long_offt));
                                //prefetcht1(EVEX_compress_addr_safe(reg_src_prf, aux_src_offset,
                                //            reg_long_offt));

                            }
                        }
                    }
                }
            }

        } else {
        
            for (int ki = 0; ki < kw; ki++) {
                for (int ic_buff = 0; ic_buff < cur_ic_buffs; ic_buff++) {

                    if (oi * stride_w - l_pad + ki * dilate_w >= 0
                        && oi * stride_w - l_pad + ki * dilate_w < iw) {

                        int reg_idx = get_reg_idx(oi, ic_buff, ki);

                        Zmm zmm = Xbyak::Zmm(reg_idx);

                        if (stride_w / dilate_w >= kw || dilate_w > stride_w || stride_w % dilate_w != 0
                            || oi == 0 || ki >= kw - stride_w) {

                            size_t aux_src_offset = (size_t)typesize
                                * (ic_buff * ic_block * mb_block * iw 
                                + (oi * stride_w - l_pad + ki * dilate_w) * ic_block);

                            vmovups(zmm, EVEX_compress_addr_safe(reg_src, aux_src_offset,
                                        reg_long_offt));
                            //prefetcht1(EVEX_compress_addr_safe(reg_src_prf, aux_src_offset,
                            //            reg_long_offt));

                        }
                    }
                }
            }
        }


        Label oc_loop_end_label;
        jz(oc_loop_end_label, T_NEAR);

        tzcnt(tzcnt_reg.cvt32(), mask_reg.cvt32());
        inc(tzcnt_reg.cvt32());

        shrx(mask_reg.cvt32(), mask_reg.cvt32(), tzcnt_reg.cvt32());

        mov(aux_reg_ker, reg_ker);
        mov(aux_reg_dst, reg_dst);

        Label oc_loop_label;
        L(oc_loop_label); {

            lea(aux_reg_dst, ptr[aux_reg_dst + tzcnt_reg * typesize]);

            int aux_dst_offset = typesize * (oi * oc_block - 1);
            vbroadcastss(zmm_o, ptr[aux_reg_dst + aux_dst_offset]);

            shl(tzcnt_reg.cvt32(), 6);
            add(aux_reg_ker, tzcnt_reg);

            tzcnt(tzcnt_reg.cvt32(), mask_reg.cvt32()); // pipelined
            inc(tzcnt_reg.cvt32());

            dec(oc_itr_reg);

            shrx(mask_reg.cvt32(), mask_reg.cvt32(), tzcnt_reg.cvt32()); // does not change flags

            for (int ic_buff = 0; ic_buff < cur_ic_buffs; ic_buff++) {
                for (int ki = 0; ki < kw; ki++) {

                    if (oi * stride_w - l_pad + ki * dilate_w < 0
                        || oi * stride_w - l_pad + ki * dilate_w >= iw) {
                        continue;
                    }

                    int reg_idx = get_reg_idx(oi, ic_buff, ki);

                    Zmm zmm = Xbyak::Zmm(reg_idx);
                    
                    size_t aux_kernel_offset = typesize * (ic_buff
                            * kw * ic_block * oc_block
                            + ki * ic_block * oc_block);
                    vfmadd231ps(zmm, zmm_o,
                            EVEX_compress_addr_safe(aux_reg_ker, aux_kernel_offset,
                                reg_long_offt)); // probably don't need safe for weight tensor

                }
            }

            jnz(oc_loop_label, T_NEAR);
        }

        L(oc_loop_end_label);

        if (oi == ow - 1) {
            for (int ki = 0; ki < kw; ki++) {
                for (int ic_buff = 0; ic_buff < cur_ic_buffs; ic_buff++) {

                    if (oi * stride_w - l_pad + ki * dilate_w >= 0
                        && oi * stride_w - l_pad + ki * dilate_w < iw) {

                        int reg_idx = get_reg_idx(oi, ic_buff, ki);

                        Zmm zmm = Xbyak::Zmm(reg_idx);

                        Label no_update_label;
                        
                        size_t aux_src_offset = (size_t)typesize
                            * (ic_buff * ic_block * mb_block * iw 
                            + (oi * stride_w - l_pad + ki * dilate_w) * ic_block);
                        vmovups(EVEX_compress_addr_safe(reg_src, aux_src_offset,
                                    reg_long_offt), zmm);
                    }
                }
            }
        }

    };

    auto outer_loop = [&](int cur_ic_buffs) {

        sub(reg_ker, ic_block * typesize);

        int rr_ow = r_ow - (r_ow - l_ow) % rotation_unroll_factor;

        if (l_ow <= rr_ow - rotation_unroll_factor * 5) { // threshold needs to be dynamically calculated based on the instruction count per section

            int niter = (rr_ow - l_ow) / rotation_unroll_factor;

            cout << "leading :" << l_ow << " trailing:" << ow - rr_ow
                << " factor:" << rotation_unroll_factor
                << " niter:" << niter << endl;

            for (int oi = 0; oi < l_ow; oi++) {
                comp_unrolled(oi, cur_ic_buffs);
            }

            Reg64 ow_itr_reg = reg_src_prf;

            mov(ow_itr_reg, niter);

            Label ow_loop_label;
            L(ow_loop_label); {

                for (int i = 0; i < rotation_unroll_factor; i++) {
                    comp_unrolled(l_ow + i, cur_ic_buffs);
                }

                add(reg_dst, oc_block * typesize * rotation_unroll_factor);
                add(reg_src, ic_block * typesize * rotation_unroll_factor * stride_w);

                add(reg_dst_prf, oc_block * typesize * rotation_unroll_factor);

                dec(ow_itr_reg);
                jnz(ow_loop_label, T_NEAR);

            }

            sub(reg_dst, oc_block * typesize * rotation_unroll_factor * niter);
            sub(reg_src, ic_block * typesize * rotation_unroll_factor * stride_w * niter);
            sub(reg_dst_prf, oc_block * typesize * rotation_unroll_factor * niter);

            for (int oi = rr_ow; oi < ow; oi++) {
                comp_unrolled(oi, cur_ic_buffs);
            }

        } else {

            cout << "fully unrolled" << endl;

            for (int oi = 0; oi < ow; oi++) {
                comp_unrolled(oi, cur_ic_buffs);
            }
        }

    };

    outer_loop(ic_buffs);

    L(end_label);
}

void jit_avx512_common_conv_bwd_data_kernel_f32::generate()
{
    int iw = jcp.iw;
    int kw = jcp.kw;
    int ur_w = jcp.ur_w;
    int ic_block = jcp.ic_block;
    int oc_block = jcp.oc_block;
    int ur_w_tail = jcp.ur_w_tail;
    int dilate_w = jcp.dilate_w + 1;
    int stride_w = jcp.stride_w;

    int dst_shift = jcp.typesize_in * (ur_w / stride_w) * ic_block;
    int src_shift = jcp.typesize_out * ur_w * oc_block;

    preamble();

    mov(reg_src, ptr[param + GET_OFF(src)]);
    mov(reg_dst, ptr[param + GET_OFF(dst)]);
    mov(reg_ker, ptr[param + GET_OFF(filt)]);

    mov(reg_kh, ptr[param + GET_OFF(kh_padding)]);
    mov(reg_src_prf, ptr[param + GET_OFF(src_prf)]);
    mov(reg_dst_prf, ptr[param + GET_OFF(dst_prf)]);
    mov(reg_ker_prf, ptr[param + GET_OFF(filt_prf)]);

    cout << "stride_w:" << stride_w << " stride_h:" << jcp.stride_h
            << " l_pad:" << jcp.l_pad << " r_pad:" << jcp.r_pad
            << " iw:" << iw << " ih:" << jcp.ih << " ic:" << jcp.ic
            << " ow:" << jcp.ow << " oh:" << jcp.oh << " oc:"<< jcp.oc
            << " kw:" << jcp.kw << " kh:"<< jcp.kh << " mb:" << jcp.mb
            << " nb_ic_blocking:" << jcp.nb_ic_blocking
            << " ngroups:" << jcp.ngroups << " dilate_w:" << dilate_w
            << " typesize_in:" << jcp.typesize_in
            << " typesize_out:" << jcp.typesize_out << endl;

    compute_loop_fma_sparse();
    
    postamble();
}

status_t jit_avx512_common_conv_bwd_data_kernel_f32::init_conf(
        jit_conv_conf_t &jcp,
        const convolution_desc_t &cd,
        const memory_desc_wrapper &diff_src_d,
        const memory_desc_wrapper &weights_d,
        const memory_desc_wrapper &diff_dst_d)
{
    if (!mayiuse(avx512_common)) return status::unimplemented;

    const int simd_w = cpu_isa_traits<avx512_common>::vlen / sizeof(float);
    const bool with_groups = weights_d.ndims() == diff_src_d.ndims() + 1;
    int ndims = diff_src_d.ndims();

    jcp.ndims = ndims;
    jcp.prop_kind = cd.prop_kind;

    jcp.ngroups = with_groups ? weights_d.dims()[0] : 1;
    jcp.mb = diff_src_d.dims()[0];

    jcp.oc = diff_dst_d.dims()[1] / jcp.ngroups;
    jcp.oc_without_padding = jcp.oc;
    jcp.ic = diff_src_d.dims()[1] / jcp.ngroups;

    jcp.id = (ndims == 5) ? diff_src_d.dims()[2] : 1;
    jcp.ih = diff_src_d.dims()[ndims-2];
    jcp.iw = diff_src_d.dims()[ndims-1];
    jcp.od = (ndims == 5) ? diff_dst_d.dims()[2] : 1;
    jcp.oh = diff_dst_d.dims()[ndims-2];
    jcp.ow = diff_dst_d.dims()[ndims-1];

    jcp.kd = (ndims == 5) ? weights_d.dims()[with_groups + 2] : 1;
    jcp.kh = weights_d.dims()[with_groups + ndims - 2];
    jcp.kw = weights_d.dims()[with_groups + ndims - 1];

    jcp.f_pad = (ndims == 5) ? cd.padding[0][0] : 0;
    jcp.t_pad = cd.padding[0][ndims-4];
    jcp.l_pad = cd.padding[0][ndims-3];

    jcp.stride_d = (ndims == 5) ? cd.strides[0] : 1;
    jcp.stride_h = cd.strides[ndims-4];
    jcp.stride_w = cd.strides[ndims-3];

    jcp.dilate_d = (ndims == 5) ? cd.dilates[0] : 0;
    jcp.dilate_h = cd.dilates[ndims-4];
    jcp.dilate_w = cd.dilates[ndims-3];
    if ((jcp.dilate_w != 0 && jcp.stride_w != 1)
            || (jcp.dilate_d != 0 && jcp.stride_d != 1)
            || (jcp.dilate_h != 0 && jcp.stride_h != 1))
        return status::unimplemented;

    jcp.r_pad = (jcp.ow - 1) * jcp.stride_w + (jcp.kw - 1) * (jcp.dilate_w + 1)
            - (jcp.iw + jcp.l_pad - 1);
    jcp.b_pad = (jcp.oh - 1) * jcp.stride_h + (jcp.kh - 1) * (jcp.dilate_h + 1)
            - (jcp.ih + jcp.t_pad - 1);
    jcp.back_pad = (jcp.od - 1) * jcp.stride_d
            + (jcp.kd - 1) * (jcp.dilate_d + 1) - (jcp.id + jcp.f_pad - 1);

    jcp.aligned_threads = 0;

    jcp.is_1stconv = false;

    jcp.oc_block = simd_w;
    jcp.ic_block = jcp.is_1stconv ? jcp.ic : simd_w;

    bool ok_to_pad_channels = true
        && jcp.ngroups == 1
        && diff_src_d.data_type() == data_type::f32;

    if (ok_to_pad_channels) {
        jcp.oc = rnd_up(jcp.oc, jcp.oc_block);
        jcp.ic = rnd_up(jcp.ic, jcp.ic_block);
    }

    auto src_format = (ndims == 5) ? nCdhw16c : nChw16c;
    auto wei_format = (ndims == 5)
        ? (with_groups) ? OhIw16o16i : OhIw16o16i
        : (with_groups) ? OhIw16o16i : OhIw16o16i;

    bool args_ok = true
        && jcp.oc % jcp.oc_block == 0
        && jcp.ic % jcp.ic_block == 0;
    if (!args_ok)
        return status::unimplemented;

    switch (diff_src_d.format()) {
        case NhC8nw16c: jcp.mb_block = 8; if (diff_dst_d.format() != NhC8nw16c) return status::unimplemented; break;
        case NhC16nw16c: jcp.mb_block = 16; if (diff_dst_d.format() != NhC16nw16c) return status::unimplemented; break;
        case NhC32nw16c: jcp.mb_block = 32; if (diff_dst_d.format() != NhC32nw16c) return status::unimplemented; break;
        case NhC64nw16c: jcp.mb_block = 64; if (diff_dst_d.format() != NhC64nw16c) return status::unimplemented; break;
        default: return status::unimplemented; break;
    }

    jcp.nb_ic = jcp.ic / jcp.ic_block;
    jcp.nb_oc = jcp.oc / jcp.oc_block;

    jcp.ur_w = jcp.stride_w;

    int regs = 28;
    if (jcp.iw <= regs)
        jcp.ur_w = jcp.iw;
    else {
        for (int ur_w = regs; ur_w > 0; --ur_w)
            if (ur_w % jcp.stride_w == 0) {
                jcp.ur_w = ur_w;
                break;
            }
    }
    int l_overflow = nstl::max(0, ((jcp.kw - 1) * (jcp.dilate_w + 1)
                    - jcp.l_pad) / jcp.stride_w);
    int r_overflow1 = nstl::max(0, ((jcp.kw - 1) * (jcp.dilate_w + 1)
                    - nstl::max(0, jcp.r_pad) - jcp.iw % jcp.ur_w) / jcp.stride_w);
    int n_oi = jcp.iw / jcp.ur_w;
    if (r_overflow1 > 0) n_oi--;

    if ((mayiuse(avx512_mic_4ops) || mayiuse(avx512_core_vnni))
           && jcp.stride_w == 1 && jcp.stride_h == 1
           && diff_dst_d.data_type() == data_type::s16
           && weights_d.data_type() == data_type::s16
           && diff_src_d.data_type() == data_type::s32) {
        if (weights_d.format() != (with_groups ? gOIhw8o16i2o : OIhw8o16i2o))
            return status::unimplemented;
        if (mayiuse(avx512_mic_4ops)) {
            jcp.ver = ver_4vnni;
        } else {
            jcp.ver = ver_vnni;
        }
        jcp.typesize_in = sizeof(int16_t);
        jcp.typesize_out = sizeof(int32_t);
    } else if (mayiuse(avx512_common)
         && diff_dst_d.data_type() == data_type::f32
         && weights_d.data_type() == data_type::f32
         && diff_src_d.data_type() == data_type::f32) {
        switch (weights_d.format()) {
            case hOIw16o16i:
            case OhIw16o16i: break;
            default: return status::unimplemented; break;
        }
        jcp.ver = ver_fma;
        jcp.typesize_in = sizeof(float);
        jcp.typesize_out = sizeof(float);
        if (mayiuse(avx512_mic_4ops)
            && jcp.stride_w == 1 && jcp.stride_h == 1 && jcp.stride_d == 1) {
                jcp.ver = ver_4fma;
            }
    } else {
        return status::unimplemented;
    }
    if (!utils::everyone_is(0, jcp.dilate_d, jcp.dilate_h, jcp.dilate_w)
            && jcp.ver != ver_fma)
        return status::unimplemented;

    jcp.nb_ic_blocking = jcp.nb_oc_blocking = 1;
    if (jcp.ver == ver_4vnni) {
        jcp.kernel_kind = embd_bcast;
    }
    if (jcp.ver == ver_vnni) {
        // TODO: kernel_kind and nb_oc_blocking selection
        //       should be tuned on real HW
        if ((jcp.iw <= 56 && jcp.ih <= 56 && jcp.kh < 5)
            || (jcp.iw <= 17 && jcp.ih <= 17 && jcp.kh >= 5) ) {
            jcp.kernel_kind = expl_bcast;
            jcp.nb_ic_blocking = 4;
        } else {
            jcp.kernel_kind = embd_bcast;
            jcp.nb_ic_blocking = 2;
        }
        if (jcp.nb_ic_blocking > 1) {
            if (jcp.nb_ic < jcp.nb_ic_blocking) jcp.nb_ic_blocking = jcp.nb_ic;
            if (jcp.nb_ic % jcp.nb_ic_blocking != 0)
                for (int i = jcp.nb_ic_blocking; i > 0; i--)
                    if (jcp.nb_ic % i == 0) {
                        jcp.nb_ic_blocking = i;
                        break;
                    }
            jcp.ur_w = 31 / (jcp.nb_ic_blocking + 1);
            if (jcp.iw < jcp.ur_w) jcp.ur_w = jcp.iw;
        }
    }
    if (jcp.ver == ver_4fma) {
        if (jcp.kw == 3 && jcp.kh == 3 && jcp.iw == 7 && jcp.ih == 7) {
            jcp.nb_ic_blocking = 2;
        } else {
            for (int i = jcp.nb_ic; i > 0; i--)
                if (i * jcp.ur_w <= regs && jcp.nb_ic % i == 0) {
                    jcp.nb_ic_blocking = i;
                    break;
                }
        }
    }

    jcp.loop_order = loop_gnc;

    bool large_code_size = (jcp.ur_w != jcp.ow)
         && ((l_overflow <= 0 && n_oi > 0) ||(l_overflow > 0 && n_oi > 1))
         && (r_overflow1 > 0) && (l_overflow > 0);
    if (large_code_size) {
        const int max_code_size = 24 * 1024;
        const int num_ops_per_reg = 6 + jcp.oc_block * jcp.kw;
        int mult = 1;
        if (l_overflow > 0) mult += 1;
        if (r_overflow1 > 0) mult += 1;
        for (int ur_w = jcp.ur_w; ur_w > regs/2; --ur_w) {
            if ((ur_w / jcp.stride_w) * mult * num_ops_per_reg * 9.2
                    < max_code_size) {
                if (ur_w % jcp.stride_w == 0) {
                    jcp.ur_w = ur_w;
                    break;
                }
            }
        }
    }

    if (jcp.ver == ver_fma && mayiuse(avx512_core)) {
        int try_nb_ic_blocking = 2;
        unsigned int ker_inp_size = typesize * jcp.iw * jcp.ic_block
            * try_nb_ic_blocking * jcp.kh;
        unsigned int ker_out_size = typesize * jcp.ow * jcp.oc_block;
        unsigned int ker_wei_size = typesize * jcp.kh * jcp.kw * jcp.ic_block
            * jcp.oc_block * try_nb_ic_blocking;
        unsigned int ker_total_size = ker_inp_size + ker_out_size
            + ker_wei_size;
        if (!(jcp.kw == 1 || (jcp.kw == 5 && jcp.iw < 8)
            || (jcp.kw < 5 && ((jcp.iw <= 5 || (jcp.iw > 8 && jcp.iw <= 13))
            || ker_total_size > L1_cache_size )))
                || jcp.stride_h > 1 || jcp.stride_d > 1) {
            jcp.kernel_kind = embd_bcast;
            jcp.ur_w = nstl::min(jcp.iw, regs);
            jcp.nb_ic_blocking = jcp.nb_oc_blocking = 1;
            if (!(jcp.kw > 3 || (jcp.kw == 3 && ker_total_size < L1_cache_size
                && jcp.ow > 8)) && jcp.stride_h == 1)
                if (jcp.nb_ic % try_nb_ic_blocking == 0) {
                    jcp.nb_ic_blocking = try_nb_ic_blocking;
                    jcp.ur_w = 31 / (jcp.nb_ic_blocking + 1);
                    if (jcp.iw < jcp.ur_w) jcp.ur_w = jcp.iw;
                }
         } else {
            jcp.kernel_kind = expl_bcast;
            jcp.nb_oc_blocking = 1;
            jcp.nb_ic_blocking = 4;
            if (jcp.nb_ic < jcp.nb_ic_blocking) jcp.nb_ic_blocking = jcp.nb_ic;
            if (jcp.nb_ic % jcp.nb_ic_blocking != 0)
                for (int i = jcp.nb_ic_blocking; i > 0; i--)
                    if (jcp.nb_ic % i == 0) {
                        jcp.nb_ic_blocking = i;
                        break;
                    }
            jcp.ur_w = 31 / (jcp.nb_ic_blocking + 1);
            if (jcp.iw < jcp.ur_w) jcp.ur_w = jcp.iw;
        }
    }
    jcp.ur_w_tail = jcp.iw % jcp.ur_w;

    if (l_overflow * jcp.stride_w > jcp.ur_w)
        return status::unimplemented;
    int r_overflow_no_tail = nstl::max(0, ((jcp.kw - 1) * (jcp.dilate_w + 1)
                    - nstl::max(0, jcp.r_pad) - jcp.ur_w_tail) / jcp.stride_w);
    if (r_overflow_no_tail * jcp.stride_w > jcp.ur_w)
        return status::unimplemented;
    if ((jcp.iw > jcp.ur_w) && (jcp.ur_w % jcp.stride_w != 0))
        return status::unimplemented;

    pick_loop_order(jcp);

    jcp.nb_oc_L2 = jcp.nb_oc;
    // TODO check for 4vnni
    if (jcp.ver == ver_4fma && (jcp.kh < 5 && jcp.kw < 5)) {
        for (int divf = 2, temp_nb = jcp.nb_oc_L2; divf <= jcp.nb_oc;
              divf++) {
            size_t l2_src = jcp.iw * jcp.ic_block * jcp.nb_ic_blocking * jcp.ih
                * jcp.id;
            size_t l2_dst = jcp.ow * jcp.oc_block * temp_nb * jcp.oh * jcp.od;
            size_t l2_filt = jcp.kw * jcp.oc_block * jcp.ic_block * jcp.kh
                * jcp.kd * jcp.nb_ic_blocking * temp_nb;
            if (4 * (l2_src + l2_dst + l2_filt) > KNx_L2_EFFECTIVE_CAPACITY) {
                if (jcp.kh == 3 && jcp.ih == 7) {
                    jcp.nb_oc_L2 = 1;
                    break;
                }
                temp_nb = (jcp.nb_oc_L2 % divf == 0 ? jcp.nb_oc_L2 / divf
                                : jcp.nb_oc_L2);
            } else {
                jcp.nb_oc_L2 = temp_nb;
                break;
            }
        }
    }

    regs = 30;
    int ur_sparse;
    if (jcp.kw * jcp.nb_ic <= regs)
        ur_sparse = jcp.kw * jcp.nb_ic;
    else {
        for (int tmp_ur_w = regs; tmp_ur_w > 0; tmp_ur_w--)
            if (tmp_ur_w % jcp.kw == 0) {
                ur_sparse = tmp_ur_w;
                break;
            }
    }

    jcp.ic_buffs = ur_sparse / jcp.kw;
    for (int i = jcp.ic_buffs; i > 0; i--) {
        if (jcp.nb_ic % i == 0) {
            jcp.ic_buffs = i;
            break;
        }
    }


    // higher than 8 will cause subpar memory performance
    if (jcp.ic_buffs > 8) jcp.ic_buffs = 8;

    jcp.ur_sparse = jcp.ic_buffs * jcp.kw;
    jcp.nb_mb = jcp.mb / jcp.mb_block;

    args_ok = true
        && jcp.ic <= diff_src_d.blocking_desc().padding_dims[1]
        && jcp.oc <= diff_dst_d.blocking_desc().padding_dims[1]
        && jcp.ic <= weights_d.blocking_desc().padding_dims[with_groups + 1]
        && jcp.oc <= weights_d.blocking_desc().padding_dims[with_groups + 0];

    return args_ok ? status::success : status::unimplemented;
}

const int jit_avx512_common_conv_bwd_weights_kernel_f32::max_ur_w = 28;


void jit_avx512_common_conv_bwd_weights_kernel_f32::bias_kernel()
{
    /*Label skip_bias, bias_loop, skip_load_bias;

    mov(reg_tmp, ptr[param + GET_OFF(flags)]);
    test(reg_tmp,reg_tmp);
    jne(skip_bias, T_NEAR);

    mov(reg_bias, ptr[param + GET_OFF(bias)]);
    mov(reg_output, ptr[param + GET_OFF(dst)]);
    vpxord(Zmm(1), Zmm(1), Zmm(1));

    mov(reg_tmp, ptr[param + GET_OFF(channel)]);
    cmp(reg_tmp, 0);
    jne(skip_load_bias, T_NEAR);
    vmovups(Zmm(1), ptr[reg_bias]);

    L(skip_load_bias);

    mov(reg_oi, ptr[param + GET_OFF(kh_padding)]);
    mov(reg_tmp, jcp.oc_block * jcp.ow * jcp.oh * jcp.typesize_out);
    imul(reg_oi, reg_tmp);

    xor_(reg_tmp, reg_tmp);
    L(bias_loop); {
        vmovups(Zmm(0), ptr[reg_output + reg_tmp]);
        vaddps(Zmm(1), Zmm(1), Zmm(0));
        add(reg_tmp, jcp.oc_block * jcp.typesize_out);
        cmp(reg_tmp, reg_oi);
        jl(bias_loop);
    }
    vmovups(EVEX_compress_addr(reg_bias,0), Zmm(1));

    L(skip_bias);*/
}

void jit_avx512_common_conv_bwd_weights_kernel_f32::compute_loop_fma_sparse() {

    Label kh_label, kd_label, skip_kd_loop;
    Label end_label, clear_label;

    /**********************************************

    reg64_t param = abi_param1;
    reg64_t reg_inp = r8;
    reg64_t reg_ker = r9;
    reg64_t reg_out = r10;

    reg64_t reg_inp_prf = r11;
    reg64_t reg_ker_prf = r12;
    reg64_t reg_out_prf = r13;

    reg64_t aux_reg_inp = r14;
    reg64_t aux_reg_ker = r15;

    reg64_t aux_reg_inp_prf = rsi;
    reg64_t aux_reg_ker_prf = rdx;

    reg64_t reg_channel = rsi;
    reg64_t reg_bias = rdx;

    reg64_t aux_reg_ker_d = r9;
    reg64_t aux_reg_inp_d = rbx;
    reg64_t aux_reg_inp_d_prf = r13;
    reg64_t aux_reg_ker_d_prf = abi_not_param1;
    reg64_t reg_ki = r10;

    reg64_t reg_kj = rax;
    reg64_t reg_relu_ns = rax;
    reg64_t reg_oi = rbx;
    reg64_t reg_kh = abi_not_param1;

    reg64_t reg_tmp = rbp;

    reg64_t reg_ic_loop = rdx;
    reg64_t reg_inp_loop = rsi;

    reg64_t reg_init_flag = r13;
    reg64_t reg_bias_ptr = param;

    reg64_t aux_reg_ic = r12;
    reg64_t reg_binp = rax;
    reg64_t reg_bout = r11;
    reg64_t aux1_reg_inp = rbx;
    reg64_t aux_reg_out = abi_not_param1;

    reg64_t reg_long_offt = r11;
    reg64_t reg_out_long_offt = r14;

    ***********************************************/

    int kw = jcp.kw;
    int kh = jcp.kh;

    int ow = jcp.ow;
    int oh = jcp.oh;

    int nb_ic = jcp.nb_ic;
    int nb_oc = jcp.nb_oc;

    int iw = jcp.iw;
    int ih = jcp.ih;

    int mb = jcp.mb;
    int mb_block = jcp.mb_block;

    int ic_block = jcp.ic_block;
    int oc_block = jcp.oc_block;
    int l_pad = jcp.l_pad;
    int dilate_w = jcp.dilate_w + 1;
    int stride_w = jcp.stride_w;
    int stride_h = jcp.stride_h;

    int nr = jcp.ur_sparse;
    int oc_buffs = jcp.oc_buffs;

    assert(nr >= kw);
    assert(oc_block == 16); // needed for specific optimization
    assert(typesize == 4);

    int oc_iters = nb_oc / oc_buffs;

    auto zmm_o = Xbyak::Zmm(31);

    auto zmm_zero = Xbyak::Zmm(30);

    auto zmm_gather = Xbyak::Zmm(0);

    auto zmm_cmp = Xbyak::Zmm(28);

    assert(nr % kw == 0);
    
    const int vsize = 16;

    auto get_reg_idx = [=](int ki, int oc_buff) {
        
        return oc_buff * kw + ki + 1;

    };

    Reg64 reg_long_offt = reg_kj;
    Reg64 reg_mul_src = rdx;

    int disp = oc_block * oc_buffs * typesize;
    int disp_shift = log2(disp);
    bool use_shift = ceil(log2(disp)) == floor(log2(disp));

    cout << "use_shift:" << use_shift << " disp_shift:" << disp_shift << endl;

    auto comp_unrolled = [&](int ii, int cur_oc_buffs) {

        Reg64 mask_reg = reg_oi;
        Reg32 ic_itr_reg = reg_out_prf.cvt32();
        Reg64 lzcnt_reg = aux_reg_ker;

        kmovw(mask_reg.cvt32(), k7);
        popcnt(ic_itr_reg, mask_reg.cvt32());

        if (ii < iw - 1) { // pipelined
            size_t aux_src_offset = typesize * (ii + 1) * mb_block;
            vcmpps(k7, zmm_zero, EVEX_compress_addr_safe(reg_inp, aux_src_offset,
                                    reg_long_offt), 4);

            prefetcht1(EVEX_compress_addr_safe(reg_inp_prf, aux_src_offset,
                                    reg_long_offt));

            /*for (int oc_buff = 0; oc_buff < cur_oc_buffs; oc_buff++) {
                for (int ki = kw - 1; ki > -1; ki--) {

                    int n = (ii + 1) - ki * dilate_w + l_pad;

                    if (n >= 0 && n / stride_w < ow && n % stride_w == 0) {

                        int oi = n / stride_w;

                        size_t aux_dst_offset = (size_t)typesize
                                * (oc_buff * oc_block * ow * mb_block
                                + oi * oc_block + ic_block * ow);

                        prefetcht0(EVEX_compress_addr_safe(reg_out, aux_dst_offset,
                                reg_long_offt));
                    }

                }
            }*/
        }

        Label ic_loop_end_label;
        jz(ic_loop_end_label, T_NEAR);

        tzcnt(lzcnt_reg.cvt32(), mask_reg.cvt32());
        inc(lzcnt_reg.cvt32());

        if (!use_shift) {
            mulx(reg_kj, reg_ker_prf, lzcnt_reg);
        }
        shrx(mask_reg.cvt32(), mask_reg.cvt32(), lzcnt_reg.cvt32());

        mov(aux_reg_out, reg_out);
        mov(aux_reg_inp, reg_inp);

        Label ic_loop_label;
        L(ic_loop_label); {

            lea(aux_reg_inp, ptr[aux_reg_inp + lzcnt_reg * typesize]);

            int aux_src_offset = typesize * (ii * mb_block - 1);
            vbroadcastss(zmm_o, ptr[aux_reg_inp + aux_src_offset]);

            if (use_shift) {
                shl(lzcnt_reg.cvt32(), disp_shift);
                add(aux_reg_out, lzcnt_reg);
            } else {
                add(aux_reg_out, reg_ker_prf);
            }

            tzcnt(lzcnt_reg.cvt32(), mask_reg.cvt32()); // pipelined
            inc(lzcnt_reg.cvt32());

            dec(ic_itr_reg);

            if (!use_shift) {
                mulx(reg_kj, reg_ker_prf, lzcnt_reg);
            }
            shrx(mask_reg.cvt32(), mask_reg.cvt32(), lzcnt_reg.cvt32()); // does not change flags

            cout << "op:";

            for (int ki = kw - 1; ki > -1; ki--) {
                for (int oc_buff = 0; oc_buff < cur_oc_buffs; oc_buff++) {

                    int n = ii - ki * dilate_w + l_pad;

                    if (n >= 0 && n / stride_w < ow && n % stride_w == 0) {

                        int oi = n / stride_w;

                        int reg_idx = get_reg_idx(ki, oc_buff);

                        Zmm zmm = Xbyak::Zmm(reg_idx);

                        if (oc_buff == 0) {
                            cout << " " << oi << "-r" << reg_idx;
                        }

                        size_t aux_dst_offset = (size_t)typesize
                                * (oc_buff * oc_block
                                + oi * oc_block * cur_oc_buffs * mb_block);
                    
                        vfmadd231ps(zmm, zmm_o,
                                EVEX_compress_addr_safe(aux_reg_out, aux_dst_offset,
                                    reg_long_offt));
                    }

                }
            }

            cout << endl;

            jnz(ic_loop_label, T_NEAR);
        }

        L(ic_loop_end_label);

    };

    auto outer_loop = [&](int cur_oc_buffs) {

        vpxord(zmm_zero, zmm_zero, zmm_zero);

        vcmpps(k7, zmm_zero, ptr[reg_inp], 4);
        prefetcht1(ptr[reg_inp_prf]);

        if (!use_shift) {
            mov(reg_mul_src, disp);
        }
        sub(reg_out, disp);

        for (int oc_buff = 0; oc_buff < oc_buffs; oc_buff++) {
            for (int ki = 0; ki < kw; ki++) {
                int idx = get_reg_idx(ki, oc_buff);
                auto zmm = Xbyak::Zmm(idx);
                vpxord(zmm, zmm, zmm);

                size_t kernel_offset = typesize * (oc_buff
                        * kw * oc_block * ic_block
                        + ki * oc_block * ic_block);
                prefetcht1(EVEX_compress_addr_safe(reg_ker, kernel_offset,
                                        reg_long_offt));

            }
        }

        int rotation_unroll_factor = kw * stride_w;
        int l_iw = kw > l_pad ? kw - l_pad : 1;
        l_iw++; // unroll one more due to pipelined vector write

        int r_iw = iw - 1 - (iw - 1 - l_iw) % rotation_unroll_factor;

        if (l_iw <= r_iw - rotation_unroll_factor * 5) { // threshold needs to be dynamically calculated based on the instruction count per section


            int niter = (r_iw - l_iw) / rotation_unroll_factor;

            cout << "nr:" << nr << " l_iw:" << l_iw << " r_iw:" << r_iw << " oc_iters:" << oc_iters
                << " oc_buffs:" << oc_buffs << endl;

            cout << "leading :" << l_iw << " trailing:" << ow - r_iw
                << " factor:" << rotation_unroll_factor
                << " niter:" << niter << endl;

            for (int ii = 0; ii < l_iw; ii++) {
                comp_unrolled(ii, cur_oc_buffs);
            }

            Reg64 iw_itr_reg = reg_channel;

            mov(iw_itr_reg, niter);

            Label iw_loop_label;
            L(iw_loop_label); {

                for (int i = 0; i < rotation_unroll_factor; i++) {
                    comp_unrolled(l_iw + i, cur_oc_buffs);
                }

                add(reg_inp, mb_block * typesize * rotation_unroll_factor);
                add(reg_out, oc_block * cur_oc_buffs * mb_block * typesize
                    * rotation_unroll_factor / stride_w);

                add(reg_inp_prf, mb_block * typesize * rotation_unroll_factor);

                dec(iw_itr_reg);
                jnz(iw_loop_label, T_NEAR);

            }

            sub(reg_inp, mb_block * typesize * rotation_unroll_factor * niter);
            sub(reg_out, oc_block * cur_oc_buffs * mb_block * typesize
                * rotation_unroll_factor / stride_w * niter);
            sub(reg_inp_prf, mb_block * typesize * rotation_unroll_factor * niter);

            mov(reg_channel, ptr[param + GET_OFF(channel)]);

            for (int ii = r_iw; ii < iw; ii++) {
                comp_unrolled(ii, cur_oc_buffs);
            }


        } else {

            mov(reg_channel, ptr[param + GET_OFF(channel)]);

            for (int ii = 0; ii < iw; ii++) {
                comp_unrolled(ii, cur_oc_buffs);
            }
        }
    };

    outer_loop(oc_buffs);

    Label no_load_label;

    cmp(reg_channel, 0);
    je(no_load_label, T_NEAR);

    for (int oc_buff = 0; oc_buff < oc_buffs; oc_buff++) {
        for (int ki = 0; ki < kw; ki++) {
            int idx = get_reg_idx(ki, oc_buff);
            auto zmm = Xbyak::Zmm(idx);

            size_t kernel_offset = typesize * (oc_buff
                    * kw * oc_block * ic_block
                    + ki * oc_block * ic_block);
            vaddps(zmm, zmm, EVEX_compress_addr_safe(reg_ker, kernel_offset,
                                    reg_long_offt));

        }
    }

    L(no_load_label);

    for (int oc_buff = 0; oc_buff < oc_buffs; oc_buff++) {
        for (int ki = 0; ki < kw; ki++) {
            int idx = get_reg_idx(ki, oc_buff);
            auto zmm = Xbyak::Zmm(idx);

            size_t kernel_offset = typesize * (oc_buff
                    * kw * oc_block * ic_block
                    + ki * oc_block * ic_block);
            vmovups(EVEX_compress_addr_safe(reg_ker, kernel_offset,
                                    reg_long_offt), zmm);
            //prefetcht1(EVEX_compress_addr_safe(reg_ker_prf, kernel_offset,
                                    //reg_long_offt));

        }
    }

}

void jit_avx512_common_conv_bwd_weights_kernel_f32::generate()
{
    preamble();

    mov(reg_inp, ptr[param1 + GET_OFF(src)]);
    mov(reg_out, ptr[param1 + GET_OFF(dst)]);
    mov(reg_ker, ptr[param1 + GET_OFF(filt)]);

    mov(reg_inp_prf, ptr[param1 + GET_OFF(src_prf)]);
    //mov(reg_out_prf, ptr[param1 + GET_OFF(dst_prf)]);
    //mov(reg_ker_prf, ptr[param + GET_OFF(filt_prf)]);

    compute_loop_fma_sparse();

    postamble();
}

status_t jit_avx512_common_conv_bwd_weights_kernel_f32::init_conf(
    jit_conv_conf_t &jcp, const convolution_desc_t &cd,
    cpu_memory_t::pd_t &src_pd, cpu_memory_t::pd_t &diff_weights_pd,
    cpu_memory_t::pd_t &diff_bias_pd, cpu_memory_t::pd_t &diff_dst_pd)
{
    if (!mayiuse(avx512_common))
        return status::unimplemented;

    const int simd_w = cpu_isa_traits<avx512_common>::vlen / sizeof(float);

    const memory_desc_wrapper src_d(&src_pd);
    const memory_desc_wrapper diff_weights_d(&diff_weights_pd);
    const memory_desc_wrapper diff_bias_d(&diff_bias_pd);
    const memory_desc_wrapper diff_dst_d(&diff_dst_pd);

    const bool with_groups = diff_weights_d.ndims() == src_d.ndims() + 1;
    int ndims = src_d.ndims();

    jcp = zero<decltype(jcp)>();
    jcp.ndims = ndims;
    jcp.prop_kind = cd.prop_kind;

    jcp.ngroups = with_groups ? diff_weights_d.dims()[0] : 1;
    jcp.mb = src_d.dims()[0];

    jcp.oc = diff_dst_d.dims()[1] / jcp.ngroups;
    jcp.oc_without_padding = jcp.oc;
    jcp.ic = src_d.dims()[1] / jcp.ngroups;

    jcp.id = (ndims == 5) ? src_d.dims()[2] : 1;
    jcp.ih = src_d.dims()[ndims-2];
    jcp.iw = src_d.dims()[ndims-1];
    jcp.od = (ndims == 5) ? diff_dst_d.dims()[2] : 1;
    jcp.oh = diff_dst_d.dims()[ndims-2];
    jcp.ow = diff_dst_d.dims()[ndims-1];

    jcp.kd = (ndims == 5) ? diff_weights_d.dims()[with_groups + 2] : 1;
    jcp.kh = diff_weights_d.dims()[with_groups + ndims-2];
    jcp.kw = diff_weights_d.dims()[with_groups + ndims-1];

    jcp.f_pad = (ndims == 5) ? cd.padding[0][0] : 0;
    jcp.t_pad = cd.padding[0][ndims-4];
    jcp.l_pad = cd.padding[0][ndims-3];

    jcp.stride_d = (ndims == 5) ? cd.strides[0] : 1;
    jcp.stride_h = cd.strides[ndims-4];
    jcp.stride_w = cd.strides[ndims-3];

    jcp.dilate_d = (ndims == 5) ? cd.dilates[0] : 0;
    jcp.dilate_h = cd.dilates[ndims-4];
    jcp.dilate_w = cd.dilates[ndims-3];

    const int kh_range = 1 + (jcp.kh - 1) * (jcp.dilate_h + 1);
    bool ok = true
        // general condition to simplify dilations
        && implication(jcp.dilate_d != 0, jcp.stride_d == 1)
        && implication(jcp.dilate_h != 0, jcp.stride_h == 1)
        && implication(jcp.dilate_w != 0, jcp.stride_w == 1)
        // special condition to simplify dilations in compute_oh_loop_common
        && implication(jcp.dilate_h != 0, kh_range <= jcp.ih);
    if (!ok)
        return status::unimplemented;

    jcp.r_pad = nstl::max(0, (jcp.ow - 1) * jcp.stride_w
            + (jcp.kw - 1) * (jcp.dilate_w + 1) - (jcp.iw + jcp.l_pad - 1));
    jcp.b_pad = nstl::max(0, (jcp.oh - 1) * jcp.stride_h
            + (jcp.kh - 1) * (jcp.dilate_h + 1) - (jcp.ih + jcp.t_pad - 1));
    jcp.back_pad = nstl::max(0, (jcp.od - 1) * jcp.stride_d
            + (jcp.kd - 1) * (jcp.dilate_d + 1) - (jcp.id + jcp.f_pad - 1));

    if ( ndims == 5 )
        if (jcp.f_pad != 0 || jcp.back_pad != 0)
            return status::unimplemented;

    jcp.ihp = jcp.ih + jcp.t_pad + jcp.b_pad;
    jcp.iwp = jcp.iw + jcp.l_pad + jcp.r_pad;
    jcp.ohp = jcp.oh;
    jcp.owp = jcp.ow;
    jcp.aligned_threads = 0;

    /* check for the 1st convolution */
    jcp.is_1stconv = is_1stconv(jcp);

    jcp.oc_block = simd_w;

    bool ok_to_pad_channels = true
        && jcp.ngroups == 1
        && src_d.data_type() == data_type::f32;

    if (ok_to_pad_channels)
        jcp.oc = rnd_up(jcp.oc, simd_w);

    if (jcp.oc % jcp.oc_block)
        return status::unimplemented;

    auto src_format = (ndims == 5) ? nCdhw16c : Nhcw16n;
    auto dst_format = (ndims == 5) ? nCdhw16c : NhCw16n128c;
    auto wei_format = (ndims == 5)
        ? (with_groups) ? hIOw16i16o : hIOw16i16o
        : (with_groups) ? hIOw16i16o : hIOw16i16o;

    /* conditions on bias memory */
    jcp.with_bias = cd.diff_bias_desc.format != memory_format::undef;
    if (jcp.with_bias) {
        if (diff_bias_d.format() == any)
            CHECK(diff_bias_pd.set_format(x));
        if (diff_bias_d.format() != x)
            return status::unimplemented;
    }

    jcp.nb_oc = jcp.oc / jcp.oc_block;

    if (src_d.format() == any)
        CHECK(src_pd.set_format(src_format));
    if (diff_dst_d.format() == any)
        CHECK(diff_dst_pd.set_format(dst_format));

    switch (src_d.format()) {
        case Nhcw16n:
            jcp.mb_block = 16;
            if (diff_dst_d.format() != NhCw16n128c && diff_dst_d.format() != NhCw16n64c) {
                return status::unimplemented;
            }
            break;
        default: return status::unimplemented; break;
    }

    /* kernel applicability check wrt boundaries
     * the conditions are quite general across the kernels we have,
     * but ideally the check should belong to a specific kernel... */
    const int max_pad = ((jcp.kh - 1) * (jcp.dilate_h + 1) + 1) / 2;
    const bool boundaries_ok = true
        && jcp.t_pad <= max_pad
        && jcp.b_pad <= max_pad;
    if (!boundaries_ok)
        return status::unimplemented;

    /* yet another common check */
    if (jcp.kw > 14)
        return status::unimplemented;

    /* setting register strategy */
    for (int ur_w = nstl::min(max_ur_w, jcp.ow); ur_w > 0; --ur_w) {
        if (jcp.ow % ur_w == 0) { jcp.ur_w = ur_w; break; }
    }

    if (jcp.is_1stconv) {
        const auto want_src_format = (ndims == 5) ? ncdhw : nchw;
        if (src_d.format() == any)
            CHECK(src_pd.set_format(want_src_format));

        const bool src_ok = true
            && utils::everyone_is(data_type::f32,
                src_d.data_type(), diff_weights_d.data_type(),
                diff_dst_d.data_type())
            && one_of(jcp.ic, 1, 3)
            && implication(jcp.ic == 1, one_of(src_d.format(), want_src_format,
                (ndims == 5) ? ndhwc : nhwc))
            && implication(jcp.ic != 1, src_d.format() == want_src_format)
            && jcp.ngroups == 1;
        if (!src_ok)
            return status::unimplemented;

        const int tr_ld = rnd_up(div_up(jcp.iw + jcp.l_pad + jcp.r_pad,
                    jcp.stride_w), 16);
        const int kh_step = nstl::max((28 - jcp.with_bias) / jcp.kw, 1);
        const int kh_step_rem = jcp.kh % kh_step;
        const auto want_4fma_wfmt = (ndims == 5)
            ? with_groups ? gOidhw16o : Oidhw16o
            : with_groups ? gOihw16o : Oihw16o;
        const bool use_4fma = true
            && ndims == 4
            && mayiuse(avx512_mic_4ops)
            && mkldnn_thr_syncable()
            && everyone_is(0, jcp.dilate_d, jcp.dilate_h, jcp.dilate_w)
            && everyone_is(0, jcp.l_pad, jcp.r_pad, jcp.t_pad, jcp.b_pad)
            && jcp.kw <= 28 - jcp.with_bias
            && jcp.stride_w == 4
            && tr_ld / simd_w <= 4 /* [bwd_w:tr_src:r1] */
            && implication(jcp.with_bias, kh_step_rem == 1) /* [bwd_w:b:r1] */
            && implication(diff_weights_d.format() != any,
                    diff_weights_d.format() == want_4fma_wfmt);

        if (use_4fma) {
            jcp.ver = ver_4fma;
            jcp.kh_step = kh_step;
            jcp.tr_ld = tr_ld;
            jcp.ic_block = 1;
            if (diff_weights_d.format() == any)
                CHECK(diff_weights_pd.set_format(want_4fma_wfmt));
        } else {
            jcp.ver = ver_fma;
            jcp.ic_block = jcp.ic;

            const auto want_wfmt = (ndims == 5)
                ? with_groups ? gOdhwi16o : Odhwi16o
                : with_groups ? gOhwi16o : Ohwi16o;
            if (diff_weights_d.format() == any)
                CHECK(diff_weights_pd.set_format(want_wfmt));
            if (diff_weights_d.format() != want_wfmt)
                return status::unimplemented;
        }

        jcp.nb_ic = jcp.ic / jcp.ic_block;
        jcp.src_fmt = src_d.format();
    } else {

        if (diff_weights_d.format() == any)
            CHECK(diff_weights_pd.set_format(wei_format));
        switch (diff_weights_d.format()) {
            case hIOw16i16o:
            case IhOw16i16o: break;
            default: return status::unimplemented; break;
        }

        jcp.ic_block = simd_w;
        if (ok_to_pad_channels)
            jcp.ic = rnd_up(jcp.ic, jcp.ic_block);
        jcp.nb_ic = jcp.ic / jcp.ic_block;
        jcp.src_fmt = src_d.format();
        if ((mayiuse(avx512_mic_4ops) || mayiuse(avx512_core_vnni))
            && mkldnn_thr_syncable()
            && ndims == 4
            && jcp.stride_w == 1
            && everyone_is(0, jcp.dilate_d, jcp.dilate_h, jcp.dilate_w)
            && ((src_d.data_type() == data_type::s16
            && diff_weights_d.data_type() == data_type::s32
            && diff_dst_d.data_type() == data_type::s16))) {
            if (mayiuse(avx512_core_vnni)) jcp.ver = ver_vnni;
            else jcp.ver = ver_4vnni;
        } else if ((mayiuse(avx512_mic) || mayiuse(avx512_core))
                && utils::everyone_is(data_type::f32,
                    src_d.data_type(), diff_weights_d.data_type(),
                    diff_dst_d.data_type())) {
            jcp.ver = ver_fma;
            if (ndims == 4 && mayiuse(avx512_mic_4ops) && jcp.stride_w == 1 &&
                    everyone_is(0, jcp.dilate_d, jcp.dilate_h, jcp.dilate_w) &&
                    mkldnn_thr_syncable()) {
                jcp.ver = ver_4fma;
            }
        } else {
            return status::unimplemented;
        }
        if (utils::one_of(jcp.ver, ver_4fma, ver_4vnni, ver_vnni)) {
            jcp.ur_w = jcp.ow;
            // XXX, BUGBUGBUG, but not a FIXME: this assumes that it's OK to
            // cross the right boundary. The only requirement is not to have
            // NaNs there because another multiplicand is always guaranteed to
            // be zero. This also may require the top-level driver to allocate
            // four extra guarding elements at the very end of the buffer.
            // I'm not proud of this hack, but it improves performance by
            // about 5-10% depending on the dimensions (Roma)

            // for vnni, that's results of performance tuning
            const int tr_round = (utils::one_of(jcp.ver, ver_4fma, ver_vnni))
                ? 4 : 8;

            jcp.tr_iw = rnd_up(jcp.iw + jcp.kw - 1, tr_round);
            jcp.tr_src_num_guard_elems = tr_round; // upper bound

            if (utils::one_of(jcp.ver, ver_4vnni, ver_vnni)) {
                jcp.tr_ow = rnd_up(jcp.ow, 2);
                jcp.ur_w = jcp.tr_ow;
            }
        }
    }

    if (utils::one_of(jcp.ver, ver_4vnni, ver_vnni)) {
        jcp.typesize_in = sizeof(int16_t);
        jcp.typesize_out = sizeof(int32_t);
    } else if (utils::one_of(jcp.ver, ver_4fma, ver_fma)) {
        jcp.typesize_in = sizeof(float);
        jcp.typesize_out = sizeof(float);
    } else
        return status::unimplemented;


    int nregs = 30;
    int ur_sparse;
    if (jcp.kw * jcp.nb_oc <= nregs)
        ur_sparse = jcp.kw * jcp.nb_oc;
    else {
        for (int tmp_ur_w = nregs; tmp_ur_w > 0; tmp_ur_w--)
            if (tmp_ur_w % jcp.kw == 0) {
                ur_sparse = tmp_ur_w;
                break;
            }
    }

    jcp.oc_buffs = ur_sparse / jcp.kw;
    for (int i = jcp.oc_buffs; i > 0; i--) {
        if (jcp.nb_oc % i == 0) {
            jcp.oc_buffs = i;
            break;
        }
    }

    //jcp.oc_buffs = 1;

    jcp.ur_sparse = jcp.oc_buffs * jcp.kw;
    jcp.nb_mb = jcp.mb / jcp.mb_block;


    bool args_ok = true
        && jcp.ic % jcp.ic_block == 0
        && jcp.oc % jcp.oc_block == 0
        && jcp.ic <= src_d.blocking_desc().padding_dims[1]
        && jcp.oc <= diff_dst_d.blocking_desc().padding_dims[1]
        && jcp.ic <= diff_weights_d.blocking_desc().padding_dims[with_groups + 1]
        && jcp.oc <= diff_weights_d.blocking_desc().padding_dims[with_groups + 0];


    return args_ok ? status::success : status::unimplemented;
}

}
}
}

