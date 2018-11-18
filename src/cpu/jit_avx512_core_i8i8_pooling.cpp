/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
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

#include <math.h>

#include "mkldnn_types.h"

#include "mkldnn_thread.hpp"
#include "utils.hpp"

#include "jit_generator.hpp"

#include "jit_avx512_core_i8i8_pooling.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace Xbyak;

using namespace mkldnn::impl::utils;
using namespace mkldnn::impl::memory_format;
using namespace mkldnn::impl::utils;
using namespace mkldnn::impl::types;
using namespace alg_kind;

template <cpu_isa_t isa>
struct jit_avx512_core_i8i8_pool_fwd_ker_t: public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx512_core_i8i8_pool_fwd_ker_t)

    struct call_params_t {
        const char *src_i8;
        const char *dst_i8;
        size_t kw_range;
        size_t kh_range;
        float idivider;
    };

    using Vmm = typename cpu_isa_traits<isa>::Vmm;
    constexpr Xmm xreg(int idx) {
        int vidx = cpu_isa_traits<isa>::n_vregs - 1 - idx; // reverse ordering
        assert(vidx >= 0);
        return Xmm(vidx);
    }
    constexpr Ymm yreg(int idx) { return Ymm(xreg(idx).getIdx()); }
    constexpr Vmm vreg(int idx) { return Vmm(xreg(idx).getIdx()); }

    // Rounding modes for axv2
    enum:uint8_t { rnd_op_nearest = 0x0 };

    // In case of avx2 with data type i8 we need to use
    // maskmovdqu instruction which has its destination hardcoded in rdi.
    // Windows ABI: abi_param1 is rcx - nothing to do else
    // Unix ABI: abi_param1 is rdi - copy to rcx and use it as abi_param1
    Reg64 reg_param      = rcx; // Our "unified abi_param1"
    Reg64 reg_ptr_src_i8 = r8;
    Reg64 reg_ptr_dst_i8 = r9;
    Reg64 reg_ptr_maskmovdqu_dst = rdi; // store destination - must be rdi

    Reg64 ki = r10;
    Reg64 kj = r11;
    Reg64 reg_kw = r12;
    Reg64 reg_kh = r13;
    Reg64 c_iter = r14;

    Reg64 aux_reg_src_h = rax;
    Reg64 aux_reg_src_w = rbx;

    Reg64 reg_tmp = rdx;

    Reg64 reg_mask = r15;

    Opmask k_cmp_mask = Opmask(7);

    Opmask mask(int idx) {
        return Opmask(6 - idx);
    }

    // ref to any of XYZ-regs via xreg/yreg/vreg functions
    Xmm xmm_tmp = xreg(0);     // temp to init vreg_tmp
    Vmm vreg_tmp = vreg(0);    // max pooling : holds minimum values for data_type
    Vmm vreg_zeros = vreg(1);

    // only in case of <isa> == avx2
    Vmm vreg_mask    = vreg(2); // full byte-mask
    Xmm xreg_mask_lo = xreg(2); // low 128-bits part of byte-mask (alias for xmm part of vreg_mask)
    Xmm xreg_mask_hi = xreg(3); // "max" - high 128-bits part of byte-mask (stored separately)
    Xmm xreg_mask_q  = xreg(3); // "avg" - 1/4 part of the mask for s8/u8 operations
    Vmm vreg_mask_q  = vreg(3); // "avg" - 1/4 part for non-zero tails

    enum:int {vidx_base = isa == avx2 ? 4 : 2};

    size_t sizeof_src_dt() const { return data_type_size(jpp.src_dt); }
    size_t sizeof_dst_dt() const { return data_type_size(jpp.dst_dt); }

    /* max pooling */
    constexpr Vmm max_pool_base_vreg(int idx) { return vreg(vidx_base + idx); }
    Vmm vreg_src(int idx) { return max_pool_base_vreg(idx); }            // [0    .. ur_c-1]
    Vmm vreg_dst(int idx) { return max_pool_base_vreg(jpp.ur_c + idx); } // [ur_c .. 2*ur_c-1]

    /* avg pooling */
    enum:int {max_num_ll = 4};
    Vmm vreg_src_s32(int jj, int ll) { return Vmm(12*jj + ll); }      // ll: 0..4 [0..3]
    Vmm vreg_dst_s32(int jj, int ll) { return Vmm(12*jj + ll + 4); }  // ll: 0..4 [4..7]
    Vmm vreg_dst_f32(int jj, int ll) { return Vmm(12*jj + ll + 8); }  // ll: 0..4 [8..11]

    void (*ker_)(const call_params_t *);
    jit_pool_conf_t jpp;

    void init_tmp_reg();
    void init_mask();

    void load_src(int jj, int ll, int c_tail);
    void store_dst(int jj, int ll, int c_tail);

    void compute_avg_step(int ur_c, int c_tail);
    void compute_max_step(int ur_c, int c_tail);
    void compute_step(int ur_c, int c_tail);

    void compute_c_block();
    void generate();

    static status_t init_conf(jit_pool_conf_t &jpp,
        const pooling_desc_t &pd, const memory_desc_wrapper &src_d,
        const memory_desc_wrapper &dst_d);

    jit_avx512_core_i8i8_pool_fwd_ker_t(const jit_pool_conf_t &jpp_)
           : jpp(jpp_) {
        generate();
        ker_ = reinterpret_cast<decltype(ker_)>(const_cast<uint8_t*>(
                       getCode()));
    }
};

template <cpu_isa_t isa>
void jit_avx512_core_i8i8_pool_fwd_ker_t<isa>::load_src(int jj, int ll, int c_tail) {
    using namespace data_type;

    int c_block = jpp.c_block;
    int ur_c = jpp.ur_c;

    switch (jpp.alg) {
        case pooling_max: {
            auto offset = jj*c_block*sizeof_src_dt();
            if (jj == ur_c - 1 && c_tail) {
                if (jpp.src_dt == data_type::s32) {
                    if (isa == avx2)
                        vpblendd(vreg_src(jj), vreg_tmp, ptr[aux_reg_src_w + offset], static_cast<uint8_t>(jpp.tail[0]));
                    else
                        vmovups(vreg_src(jj) | mask(0), ptr[aux_reg_src_w + offset]);
                } else {
                    if (isa == avx2)
                        vpblendvb(vreg_src(jj), vreg_tmp, ptr[aux_reg_src_w + offset], vreg_mask);
                    else
                        vmovdqu8(vreg_src(jj) | mask(0), ptr[aux_reg_src_w + offset]);
                }
            } else {
                vmovups(vreg_src(jj), ptr[aux_reg_src_w + offset]);
            }
            break;
        }
        case pooling_avg_include_padding:
        case pooling_avg_exclude_padding: {
            auto offset = (ll*(c_block/max_num_ll) + jj*c_block)*sizeof_src_dt();
            if (jj == jpp.ur_c - 1 && c_tail) {
                if (jpp.tail[ll]) {
                    switch (jpp.src_dt) {
                        case s32:
                            if (isa == avx2)
                                vpblendd(vreg_src_s32(jj, ll), vreg_zeros, ptr[aux_reg_src_w + offset],
                                    static_cast<uint8_t>(jpp.tail[ll]));
                            else
                                vmovups(vreg_src_s32(jj, ll) | mask(ll), ptr[aux_reg_src_w + offset]);
                            break;
                        case s8:
                            if (isa == avx2) {
                                // extract ll-th part of mask (ll-th QWORD)
                                vpblendd(vreg_mask_q, vreg_zeros, vreg_mask, 0x3 << ll); // 0x3 - mask for 2 x DWORD

                                // Move mask from ll-th pos to 0-th pos
                                if (ll>0)
                                    vpermq(vreg_mask_q, vreg_mask_q, ll);

                                // Load by mask
                                vpblendvb(vreg_src_s32(jj, ll), vreg_zeros, ptr[aux_reg_src_w + offset], vreg_mask_q);
                                vpmovsxbd(vreg_src_s32(jj, ll), vreg_src_s32(jj, ll));
                            } else
                                vpmovsxbd(vreg_src_s32(jj, ll) | mask(ll), ptr[aux_reg_src_w + offset]);
                            break;
                        case u8:
                            if (isa == avx2) {
                                // extract ll-th part of mask (ll-th QWORD)
                                vpblendd(vreg_mask_q, vreg_zeros, vreg_mask, 0x3 << ll); // 0x3 - mask for 2 x DWORD

                                // Move mask from ll-th pos to 0-th pos
                                if (ll>0)
                                    vpermq(vreg_mask_q, vreg_mask_q, ll);

                                // Load by mask
                                vpblendvb(vreg_src_s32(jj, ll), vreg_zeros, ptr[aux_reg_src_w + offset], vreg_mask_q);
                                vpmovzxbd(vreg_src_s32(jj, ll), vreg_src_s32(jj, ll));
                            } else
                                vpmovzxbd(vreg_src_s32(jj, ll) | mask(ll), ptr[aux_reg_src_w + offset]);
                            break;
                        default: assert(!"unsupported src data type");
                    }
                }
            } else {
                switch (jpp.src_dt) {
                    case s32:
                        vmovups(vreg_src_s32(jj, ll), ptr[aux_reg_src_w + offset]);
                        break;
                    case s8:
                        vpmovsxbd(vreg_src_s32(jj, ll), ptr[aux_reg_src_w + offset]);
                        break;
                    case u8:
                        vpmovzxbd(vreg_src_s32(jj, ll), ptr[aux_reg_src_w + offset]);
                        break;
                    default: assert(!"unsupported src data type");
                }
            }
            break;
        }
        default: assert(!"unsupported algorithm");
    }
}

template <cpu_isa_t isa>
void jit_avx512_core_i8i8_pool_fwd_ker_t<isa>::store_dst(int jj, int ll,
        int c_tail) {
    using namespace data_type;

    int c_block = jpp.c_block;
    int ur_c = jpp.ur_c;

    switch(jpp.alg) {
        case pooling_max: {
            auto offset = jj*c_block*sizeof_dst_dt();
            if (jj == ur_c - 1 && c_tail) {
                if (jpp.src_dt == data_type::s32) {
                    if (isa == avx2)
                        vpmaskmovd(ptr[reg_ptr_dst_i8 + offset], vreg_mask, vreg_dst(jj));
                    else
                        vmovups(ptr[reg_ptr_dst_i8 + offset], vreg_dst(jj) | mask(0));
                } else {
                    if (isa == avx2) {
                        // Store low half by mask (bytes 0...15)
                        lea(reg_ptr_maskmovdqu_dst, ptr[reg_ptr_dst_i8 + offset]);
                        maskmovdqu(vreg_dst(jj), xreg_mask_lo);

                        // Do we need to store high half (bytes 16...31) ?
                        if (c_tail > c_block / 2) {
                            vextracti128(Xmm(vreg_dst(jj).getIdx()), vreg_dst(jj), 1);
                            add(reg_ptr_maskmovdqu_dst, c_block / 2);
                            maskmovdqu(vreg_dst(jj), xreg_mask_hi);
                        }
                    } else
                        vmovdqu8(ptr[reg_ptr_dst_i8 + offset], vreg_dst(jj) | mask(0));
                }
            } else {
                vmovups(ptr[reg_ptr_dst_i8 + offset], vreg_dst(jj));
            }
            break;
        }
        case pooling_avg_include_padding:
        case pooling_avg_exclude_padding: {
            auto offset = (ll*(c_block/max_num_ll) + jj*c_block)*sizeof_dst_dt();
            if (jj == ur_c - 1 && c_tail) {
                if (jpp.tail[ll]) {
                    switch (jpp.dst_dt) {
                        case s32:
                            if (isa == avx2) {
                                vpmaskmovd(ptr[reg_ptr_dst_i8 + offset], vreg_mask, vreg_dst_s32(jj, ll));
                            } else
                                vmovups(ptr[reg_ptr_dst_i8 + offset], vreg_dst_s32(jj, ll) | mask(ll));
                            break;
                        case s8:
                            if (isa == avx2) {
                                // conversion: s32 -> s16 : {8 x s32}{8 x 0} -> {16 x s16}
                                // Result QWORDs (qw0, qw1) permuted: {qw0, 0, qw1, 0}
                                vpackssdw(vreg_dst_s32(jj, ll), vreg_dst_s32(jj, ll), vreg_zeros);

                                // Permute qwords to restore original order
                                // {qw0, 0, qw1, 0} -> {qw0, qw1, 0, 0}
                                vpermq(vreg_dst_s32(jj, ll), vreg_dst_s32(jj, ll), 0x58);

                                // conversion: s16 -> s8 : {16 x s16}{16 x 0} -> {32 x s8}
                                // Target QWORD qw = {8 x s8} has right position: {qw, xx, xx, xx}
                                vpacksswb(vreg_dst_s32(jj, ll), vreg_dst_s32(jj, ll), vreg_zeros);


                                // extract ll-th mask part
                                vpblendd(vreg_mask_q, vreg_zeros, vreg_mask, 0x3 << ll); // 0x3 - mask for 2 x DWORD

                                // Move mask from ll-th pos to 0-th pos
                                if (ll>0)
                                    vpermq(vreg_mask_q, vreg_mask_q, ll);

                                // store only 8 bytes
                                lea(reg_ptr_maskmovdqu_dst, ptr[reg_ptr_dst_i8 + offset]);
                                maskmovdqu(vreg_dst_s32(jj, ll), xreg_mask_q);
                            } else
                                vpmovdb(ptr[reg_ptr_dst_i8 + offset], vreg_dst_s32(jj, ll) | mask(ll));
                            break;
                        case u8:
                            if (isa == avx2) {
                                // conversion: s32 -> u16 : {8 x s32}{8 x 0} -> {16 x u16}
                                // Result QWORDs (qw0, qw1) permuted: {qw0, 0, qw1, 0}
                                vpackusdw(vreg_dst_s32(jj, ll), vreg_dst_s32(jj, ll), vreg_zeros);

                                // Permute qwords to restore original order
                                // {qw0, 0, qw1, 0} -> {qw0, qw1, 0, 0} ->
                                vpermq(vreg_dst_s32(jj, ll), vreg_dst_s32(jj, ll), 0x58);

                                // conversion: u16 -> u8 : {16 x u16}{16 x 0} -> {32 x u8}
                                // Target QWORD qw = {8 x u8} has right position: {qw, xx, xx, xx}
                                vpackuswb(vreg_dst_s32(jj, ll), vreg_dst_s32(jj, ll), vreg_zeros);


                                // extract ll-th mask part
                                vpblendd(vreg_mask_q, vreg_zeros, vreg_mask, 0x3 << ll); // 0x3 - mask for 2 x DWORD

                                // Move mask from ll-th pos to 0-th pos
                                if (ll>0)
                                    vpermq(vreg_mask_q, vreg_mask_q, ll);

                                // store only 8 bytes
                                lea(reg_ptr_maskmovdqu_dst, ptr[reg_ptr_dst_i8 + offset]);
                                maskmovdqu(vreg_dst_s32(jj, ll), xreg_mask_q);
                            } else
                                vpmovusdb(ptr[reg_ptr_dst_i8 + offset],
                                        vreg_dst_s32(jj, ll) | mask(ll));
                            break;
                        default: assert(!"unsupported dst data_type");
                    }
                }
            } else {
                switch (jpp.dst_dt) {
                    case s32:
                        vmovups(ptr[reg_ptr_dst_i8 + offset],
                            vreg_dst_s32(jj, ll));
                        break;
                    case s8:
                        if (isa == avx2) {

                            // conversion: s32 -> s16 : {8 x s32}{8 x 0} -> {16 x s16}
                            // Result QWORDs (qw0, qw1) permuted: {qw0, 0, qw1, 0}
                            vpackssdw(vreg_dst_s32(jj, ll), vreg_dst_s32(jj, ll), vreg_zeros);

                            // Permute qwords to restore original order
                            // {qw0, 0, qw1, 0} -> {qw0, qw1, 0, 0}
                            vpermq(vreg_dst_s32(jj, ll), vreg_dst_s32(jj, ll), 0x58);

                            // conversion: s16 -> s8 : {16 x s16}{16 x 0} -> {32 x s8}
                            // Target QWORD qw = {8 x s8} has right position: {qw, xx, xx, xx}
                            vpacksswb(vreg_dst_s32(jj, ll), vreg_dst_s32(jj, ll), vreg_zeros);

                            // store only 8 bytes
                            lea(reg_ptr_maskmovdqu_dst, ptr[reg_ptr_dst_i8 + offset]);
                            maskmovdqu(vreg_dst_s32(jj, ll), xreg_mask_q);

                        } else
                            vpmovdb(ptr[reg_ptr_dst_i8 + offset], vreg_dst_s32(jj, ll));
                        break;
                    case u8:
                        if (isa == avx2) {

                            // conversion: s32 -> u16 : {8 x s32}{8 x 0} -> {16 x u16}
                            // Result QWORDs (qw0, qw1) permuted: {qw0, 0, qw1, 0}
                            vpackusdw(vreg_dst_s32(jj, ll), vreg_dst_s32(jj, ll), vreg_zeros);

                            // Permute qwords to restore original order
                            // {qw0, 0, qw1, 0} -> {qw0, qw1, 0, 0} ->
                            vpermq(vreg_dst_s32(jj, ll), vreg_dst_s32(jj, ll), 0x58);

                            // conversion: u16 -> u8 : {16 x u16}{16 x 0} -> {32 x u8}
                            // Target QWORD qw = {8 x u8} has right position: {qw, xx, xx, xx}
                            vpackuswb(vreg_dst_s32(jj, ll), vreg_dst_s32(jj, ll), vreg_zeros);

                            // store only 8 bytes
                            lea(reg_ptr_maskmovdqu_dst, ptr[reg_ptr_dst_i8 + offset]);
                            maskmovdqu(vreg_dst_s32(jj, ll), xreg_mask_q);

                        } else
                            vpmovusdb(ptr[reg_ptr_dst_i8 + offset],vreg_dst_s32(jj, ll));
                        break;
                    default: assert(!"unsuppotred dst data_type");
                }
            }
            break;
        }
        default: assert(!"unsupported pooling algorithm");
    }
}

template <cpu_isa_t isa>
void jit_avx512_core_i8i8_pool_fwd_ker_t<isa>::compute_max_step(int ur_c, int c_tail)
{
    Label l_kw, l_kh;

    int iw = jpp.iw;
    int c = jpp.c;

    for (int jj = 0; jj < ur_c; jj++)
        vmovups(vreg_dst(jj), vreg_tmp);

    mov(aux_reg_src_h, reg_ptr_src_i8);

    xor_(kj, kj);
    L(l_kh);
    {
        mov(aux_reg_src_w, aux_reg_src_h);
        xor_(ki, ki);
        L(l_kw);
        {
            for (int jj = 0; jj < ur_c; jj++) {
                load_src(jj, 0, c_tail);
                if (jpp.src_dt == data_type::s32) {
                    if (isa == avx2) {
                        vpmaxsd(vreg_dst(jj), vreg_dst(jj), vreg_src(jj));
                    } else if (isa == avx512_core) {
                        vpcmpd(k_cmp_mask, vreg_dst(jj), vreg_src(jj), _cmp_lt_os);
                        vpblendmd(vreg_dst(jj) | k_cmp_mask, vreg_dst(jj), vreg_src(jj));
                    }
                } else {
                    if (isa == avx2) {
                        if (jpp.src_dt == data_type::s8) {
                            vpmaxsb(vreg_dst(jj), vreg_dst(jj), vreg_src(jj));
                        } else
                            vpmaxub(vreg_dst(jj), vreg_dst(jj), vreg_src(jj));
                    } else if (isa == avx512_core) {
                        if (jpp.src_dt == data_type::s8)
                            vpcmpb(k_cmp_mask, vreg_dst(jj), vreg_src(jj),
                                    _cmp_lt_os);
                        else
                            vpcmpub(k_cmp_mask, vreg_dst(jj), vreg_src(jj),
                                    _cmp_lt_os);
                        vpblendmb(vreg_dst(jj) | k_cmp_mask, vreg_dst(jj),
                                vreg_src(jj));
                    }
                }
            }
            add(aux_reg_src_w, c * sizeof_src_dt());
            inc(ki);
            cmp(ki, reg_kw);
            jl(l_kw, T_NEAR);
        }
        add(aux_reg_src_h, iw * c * sizeof_src_dt());
        inc(kj);
        cmp(kj, reg_kh);
        jl(l_kh, T_NEAR);
    }

    for (int jj = 0; jj < ur_c; jj++)
        store_dst(jj, 0, c_tail);
}

template <cpu_isa_t isa>
void jit_avx512_core_i8i8_pool_fwd_ker_t<isa>::compute_avg_step(int ur_c, int c_tail)
{
    using namespace data_type;

    Label l_kw, l_kh;

    int iw = jpp.iw;
    int c = jpp.c;

    int num_ll = jpp.src_dt == data_type::s32 ? 1 : 4;

    for (int jj = 0; jj < ur_c; jj++) {
        for (int ll = 0; ll < max_num_ll; ll++) {
            uni_vpxor(vreg_src_s32(jj, ll), vreg_src_s32(jj, ll), vreg_src_s32(jj, ll));
            uni_vpxor(vreg_dst_s32(jj, ll), vreg_dst_s32(jj, ll), vreg_dst_s32(jj, ll));
        }
    }

    mov(aux_reg_src_h, reg_ptr_src_i8);

    xor_(kj, kj);
    L(l_kh);
    {
        mov(aux_reg_src_w, aux_reg_src_h);
        xor_(ki, ki);
        L(l_kw);
        {
            for (int jj = 0; jj < ur_c; jj++) {
                for (int ll = 0; ll < num_ll; ll++) {
                    load_src(jj, ll, c_tail);
                    vpaddd(vreg_dst_s32(jj, ll), vreg_dst_s32(jj, ll),
                            vreg_src_s32(jj, ll));
                }
            }
            add(aux_reg_src_w, c * sizeof_src_dt());
            inc(ki);
            cmp(ki, reg_kw);
            jl(l_kw, T_NEAR);
        }
        add(aux_reg_src_h, iw * c * sizeof_src_dt());
        inc(kj);
        cmp(kj, reg_kh);
        jl(l_kh, T_NEAR);
    }

    for (int jj = 0; jj < ur_c; jj++) {
        for (int ll = 0; ll < num_ll; ll++) {
            vcvtdq2ps(vreg_dst_f32(jj, ll), vreg_dst_s32(jj, ll));
            vfmadd132ps(vreg_dst_f32(jj, ll), vreg_zeros, vreg_tmp);

            if (isa == avx2) {
                uni_vroundps(vreg_dst_f32(jj, ll), vreg_dst_f32(jj, ll), rnd_op_nearest);
                vcvtps2dq(vreg_dst_s32(jj, ll), vreg_dst_f32(jj, ll));
            } else if (isa >= avx512_common) {
                // AVX512: use of EVEX-embedded static rounding override
                vcvtps2dq(vreg_dst_s32(jj, ll) | T_rn_sae, vreg_dst_f32(jj, ll));
            }

            store_dst(jj, ll, c_tail);
        }
    }
}

template <cpu_isa_t isa>
void jit_avx512_core_i8i8_pool_fwd_ker_t<isa>::compute_step(int ur_c, int c_tail) {
    switch (jpp.alg) {
        case pooling_max:
            compute_max_step(ur_c, c_tail); break;
        case pooling_avg_include_padding:
        case pooling_avg_exclude_padding:
            compute_avg_step(ur_c, c_tail); break;
        default: assert(!"unsupported pooling algorithm");
    }
}

template <cpu_isa_t isa>
void jit_avx512_core_i8i8_pool_fwd_ker_t<isa>::compute_c_block(){
    Label l_main_loop;

    int nb_c = jpp.nb_c;
    int c_block = jpp.c_block;
    int ur_c = jpp.ur_c;
    int ur_c_tail = jpp.ur_c_tail;
    int c_steps = nb_c / ur_c;
    int c_tail = jpp.c_tail;

    xor_(c_iter, c_iter);
    if (c_steps > 0) {
        L(l_main_loop); {
            compute_step(ur_c, 0);
            add(reg_ptr_src_i8, ur_c*c_block*sizeof_src_dt());
            add(reg_ptr_dst_i8, ur_c*c_block*sizeof_dst_dt());
            inc(c_iter);
            cmp(c_iter, c_steps);
            jl(l_main_loop, T_NEAR);
        }
    }

    if (ur_c_tail != 0) {
        compute_step(ur_c_tail, c_tail);
    }
}

template <cpu_isa_t isa>
void jit_avx512_core_i8i8_pool_fwd_ker_t<isa>::init_mask() {
    using namespace data_type;

    // AVX2 mask initialization: mask stored in Ymm-regs
    auto axv2_init_mask = [&](uint64_t bit_mask, bool init_mask_q) {
        const size_t QW_PER_VREG = cpu_isa_traits<isa>::vlen / sizeof(uint64_t);

        uint64_t vmask[QW_PER_VREG];
        for (size_t i = 0; i < QW_PER_VREG; i++){

            uint64_t qw_vmask=0ULL;
            const size_t DBITS = 8*sizeof_src_dt();
            const uint64_t VMSK = 1ULL << (DBITS-1);
            const size_t D_PER_QW = (8*sizeof(qw_vmask))/DBITS;
            for (size_t j = 0; j < D_PER_QW; j++) {
                if (bit_mask & 1)
                    qw_vmask |= VMSK << DBITS * j;
                bit_mask >>= 1;
            }
            vmask[i] = qw_vmask;
        }

        // Put QWORDS with target mask into xmm regs
        const int xdst_i[QW_PER_VREG] = {
                xreg_mask_lo.getIdx(),
                xreg_mask_lo.getIdx(),
                xreg_mask_hi.getIdx(),
                xreg_mask_hi.getIdx()
        };
        const int xsrc_i[QW_PER_VREG] = {
                vreg_zeros.getIdx(),   // 0-th qword insert in zeros -> {qw0,  0}
                xreg_mask_lo.getIdx(), // 1-st and 0-th merge        -> {qw0,qw1}
                vreg_zeros.getIdx(),
                xreg_mask_hi.getIdx()
        };
        const uint8 qw_dst_idx[QW_PER_VREG] = {0, 1, 0, 1}; // qword index in 128-bit xreg

        for (size_t i = 0; i < QW_PER_VREG; i++) {
            mov(reg_mask, vmask[i]);
            vpinsrq(Xmm(xdst_i[i]), Xmm(xsrc_i[i]), reg_mask, qw_dst_idx[i]);
        }

        // Merge Low (xreg_mask_lo alias for vreg_mask.xreg)
        // and High (xreg_mask_hi) into full vreg_mask
        // vreg_mask -> {xreg_mask_hi, vreg_mask.xreg}
        vinserti128(vreg_mask, vreg_mask, xreg_mask_hi, 1);

        // Keep only low qword of mask in xreg_mask_q
        if (init_mask_q) {
            mov(reg_mask, vmask[0]);
            vpinsrq(xreg_mask_q, Xmm(vreg_zeros.getIdx()), reg_mask, 0);
        }
    };

    switch (isa) {
        case avx2: {
            uint64_t tail_mask = (1ULL << jpp.c_tail) - 1;
            switch (jpp.alg) {
                case pooling_max:
                    // For "max" we need mask only in case of non-zero tail
                    if (tail_mask)
                        axv2_init_mask(tail_mask, false);
                    break;
                case pooling_avg_include_padding:
                case pooling_avg_exclude_padding:
                    // For "avg" we need mask:
                    // - s32   - in case of the non-zero tail
                    // - s8/u8 - irrespective of the tail
                    if (tail_mask || one_of(jpp.src_dt, s8, u8))
                        axv2_init_mask(tail_mask ? tail_mask : -1ULL, tail_mask == 0);
                    switch (jpp.src_dt) {
                        case s32:
                            if (tail_mask)
                                axv2_init_mask(tail_mask, false);
                            break;
                        case s8:
                        case u8:
                            axv2_init_mask(tail_mask ? tail_mask : -1ULL, tail_mask == 0);
                            break;
                        default: assert(!"unsupported src data type");
                    }
                    break;
                default: assert(!"unsupported pooling algorithm");
            }
        } break;

        case avx512_core: {
            for (int ll = 0; ll < max_num_ll; ll++) {
                mov(reg_mask, jpp.tail[ll]);
                kmovq(mask(ll), reg_mask);
            }
        } break;

        default: assert(!"unsupported isa");
    }
}

template <cpu_isa_t isa>
void jit_avx512_core_i8i8_pool_fwd_ker_t<isa>::init_tmp_reg() {
    using namespace data_type;

    switch (jpp.alg) {
        case pooling_avg_include_padding:
        case pooling_avg_exclude_padding:
            mov(reg_tmp, ptr[reg_param + offsetof(call_params_t, idivider)]);
            movq(xmm_tmp, reg_tmp);
            vpbroadcastd(vreg_tmp, xmm_tmp);
            break;
        case pooling_max:
            switch (jpp.src_dt) {
                case s32:
                    mov(reg_tmp, nstl::numeric_limits<int32_t>::lowest());
                    break;
                case s8:
                    mov(reg_tmp, nstl::numeric_limits<int8_t>::lowest());
                    break;
                case u8:
                    mov(reg_tmp, nstl::numeric_limits<uint8_t>::lowest());
                    break;
                default: assert(!"unsupported src data_type");
            }

            movq(xmm_tmp, reg_tmp);
            if (jpp.src_dt == s32)
                vpbroadcastd(vreg_tmp, xmm_tmp);
            else
                vpbroadcastb(vreg_tmp, xmm_tmp);
            break;
        default: assert(!"unsupported pooling algorithm");
    }

}

template <cpu_isa_t isa>
void jit_avx512_core_i8i8_pool_fwd_ker_t<isa>::generate() {
    preamble();

#if !defined(_WIN32)
    // Always use rcx as abi_param1 -
    // see the note about maskmovdqu near reg_param.
    mov(rcx, rdi);
#endif

#   define READ_PARAM(reg, field) \
        mov(reg, ptr[reg_param + offsetof(call_params_t, field)])
    READ_PARAM(reg_ptr_src_i8, src_i8);
    READ_PARAM(reg_ptr_dst_i8, dst_i8);
    READ_PARAM(reg_kw, kw_range);
    READ_PARAM(reg_kh, kh_range);

#   undef READ_PARAM

    uni_vpxor(vreg_zeros, vreg_zeros, vreg_zeros);

    init_mask();

    init_tmp_reg();

    compute_c_block();

    postamble();
}

template <cpu_isa_t isa>
status_t jit_avx512_core_i8i8_pool_fwd_ker_t<isa>::init_conf(jit_pool_conf_t &jpp,
        const pooling_desc_t &pd, const memory_desc_wrapper &src_d,
        const memory_desc_wrapper &dst_d) {
    if (!mayiuse(isa))
        return status::unimplemented;

    jpp.mb = src_d.dims()[0];
    jpp.c = src_d.dims()[1];
    jpp.ih = src_d.dims()[2];
    jpp.iw = src_d.dims()[3];
    jpp.oh = dst_d.dims()[2];
    jpp.ow = dst_d.dims()[3];

    jpp.stride_h = pd.strides[0];
    jpp.stride_w = pd.strides[1];
    jpp.kh = pd.kernel[0];
    jpp.kw = pd.kernel[1];

    jpp.t_pad = pd.padding[0][0];
    jpp.l_pad = pd.padding[0][1];

    jpp.alg = pd.alg_kind;

    jpp.src_dt = pd.src_desc.data_type;
    jpp.dst_dt = pd.dst_desc.data_type;

    // data_type items per one vreg on the <isa>
    //     isa == avx2    : 32 bytes -> 32 for s8/u8, 8 for s32
    //     isa == avx512* : 64 bytes -> 64 for s8/u8, 16 for s32
    int simd_w = cpu_isa_traits<isa>::vlen / data_type_size(jpp.src_dt);

    jpp.c_block = simd_w;
    jpp.c_tail = jpp.c % jpp.c_block;
    jpp.nb_c = jpp.c / jpp.c_block;
    jpp.ur_c = 1;
    jpp.ur_c_tail = jpp.nb_c - (jpp.nb_c / jpp.ur_c)*jpp.ur_c +
            (jpp.c_tail != 0);

    size_t tail_mask = (1ULL << jpp.c_tail) - 1;

    switch (jpp.alg) {
        case pooling_max:

//            return status::unimplemented; // XXX: !!! for test AVG-only

            // XXX: log
            printf("\t\t\t\t\t\t\t\t\t\t\t\t%s : pooling_max : %s tail_mask = 0x%lx\n",
                    __FUNCTION__,
                    jpp.src_dt == data_type::s32 ? "s32" : "i8",
                    tail_mask );

            jpp.tail[0] = tail_mask;
            jpp.tail[1] = 0;
            jpp.tail[2] = 0;
            jpp.tail[3] = 0;
            break;
        case pooling_avg_include_padding:
        case pooling_avg_exclude_padding: {

//            if (!tail_mask || jpp.src_dt != data_type::u8)
//                return status::unimplemented; // XXX: until implemented
//            if (jpp.src_dt != data_type::s8)
//                return status::unimplemented; // XXX: until implemented


            // XXX: log
            printf("\t\t\t\t\t\t\t\t\t\t\t\t%s : pooling_avg : %s tail_mask = 0x%lx\n",
                    __FUNCTION__,
                    jpp.src_dt == data_type::s32 ? "s32" : "i8",
                    tail_mask );

            // s32 defines granularity (u8/s8 processed as s32)
            // avx2 : 8, avx512 : 16
            const size_t msk_gran = cpu_isa_traits<isa>::vlen / data_type_size(data_type::s32);
            const size_t msk_msk = (1ULL << msk_gran) - 1;
            size_t m = tail_mask;
            for (size_t ll = 0; ll < max_num_ll; ll++) {
                jpp.tail[ll] = m & msk_msk;
                m = m >> msk_gran;
            }

            } break;
        default: return status::unimplemented;
    }

    return status::success;
}

template <cpu_isa_t isa>
status_t jit_avx512_core_i8i8_pooling_fwd_t<isa>::pd_t::jit_conf() {
    return jit_avx512_core_i8i8_pool_fwd_ker_t<isa>::init_conf(jpp_,
       desc_, src_pd_.desc(), dst_pd_.desc());
}

template <cpu_isa_t isa>
jit_avx512_core_i8i8_pooling_fwd_t<isa>::
jit_avx512_core_i8i8_pooling_fwd_t(const pd_t *pd,
          const input_vector &inputs, const output_vector &outputs)
    : cpu_primitive_t(&conf_, inputs, outputs), conf_(*pd), ker_(nullptr)
{ ker_ = new jit_avx512_core_i8i8_pool_fwd_ker_t<isa>(conf_.jpp_); }

template <cpu_isa_t isa>
jit_avx512_core_i8i8_pooling_fwd_t<isa>::
~jit_avx512_core_i8i8_pooling_fwd_t() { delete ker_; }

template <cpu_isa_t isa>
void jit_avx512_core_i8i8_pooling_fwd_t<isa>::execute_forward() {
    auto src_i8 = reinterpret_cast<const char *>(input_memory(0));
    auto dst_i8 = reinterpret_cast<char *>(memory());

    const memory_desc_wrapper src_d(conf_.src_pd());
    const memory_desc_wrapper dst_d(conf_.dst_pd());

    const auto &jpp = conf_.jpp_;

    parallel_nd(jpp.mb, jpp.oh, jpp.ow,
            [&](int n, int oh, int ow) {
        const int ih = nstl::max(oh*jpp.stride_h - jpp.t_pad, 0);
        const int iw = nstl::max(ow*jpp.stride_w - jpp.l_pad, 0);

        const int kh_start = nstl::max(0, jpp.t_pad - oh * jpp.stride_h);
        const int kh_end = nstl::min(jpp.kh,
                jpp.ih + jpp.t_pad - oh * jpp.stride_h);
        const int kw_start = nstl::max(0, jpp.l_pad - ow * jpp.stride_w);
        const int kw_end = nstl::min(jpp.kw,
                jpp.iw + jpp.l_pad - ow * jpp.stride_w);

        auto p = typename jit_avx512_core_i8i8_pool_fwd_ker_t<isa>::call_params_t();
        p.src_i8 = &src_i8[
            src_d.blk_off(n, 0, ih, iw) * src_d.data_type_size()];
        p.dst_i8 = &dst_i8[
            dst_d.blk_off(n, 0, oh, ow) * dst_d.data_type_size()];
        p.kw_range = (size_t)(kw_end - kw_start);
        p.kh_range = (size_t)(kh_end - kh_start);
        p.idivider = 1.0f / ((jpp.alg == pooling_avg_exclude_padding) ?
            p.kh_range*p.kw_range : jpp.kw*jpp.kh);

        ker_->ker_(&p);
    });
}

// Explicit instantiation only for supported <isa> values.
//
template struct jit_avx512_core_i8i8_pool_fwd_ker_t<avx512_core>;
template struct jit_avx512_core_i8i8_pooling_fwd_t<avx512_core>;

template struct jit_avx512_core_i8i8_pool_fwd_ker_t<avx2>;
template struct jit_avx512_core_i8i8_pooling_fwd_t<avx2>;

}
}
}
