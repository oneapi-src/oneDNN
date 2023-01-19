/*******************************************************************************
* Copyright 2019-2023 Intel Corporation
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
#include <float.h>

#include "common/dnnl_thread.hpp"
#include "common/utils.hpp"

#include "cpu/x64/jit_uni_xf16_sum.hpp"

#define GET_OFF(field) offsetof(jit_sum_call_t, field)

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

using namespace dnnl::impl::prop_kind;
using namespace dnnl::impl::utils;

using namespace Xbyak;

void jit_avx512_core_bf16_sum_kernel_t::pre_compute_init() {
    mov(reg_idx_table, idx_table);
    vmovups(zmm_idx, ptr[reg_idx_table]);
    if (!isa_has_bf16(jsp.isa)) bf16_emu_->init_vcvtneps2bf16();
}

void jit_avx512_core_bf16_sum_kernel_t::broadcast_scale(int scale_iter) {
    Zmm vscale = Zmm(scale_vreg_idx(scale_iter));
    vpbroadcastd(vscale, ptr[reg_scales + 2 * scale_iter * jsp.typesize_in]);
}

void jit_avx512_core_bf16_sum_kernel_t::read_iter(
        int acc_iter, int u_idx, int src_shift) {
    Zmm vsrc0 = Zmm(src_vreg_idx(u_idx, 2 * acc_iter));
    Zmm vsrc1_vtmp = Zmm(tmp_vreg_idx(u_idx, acc_iter));
    vmovups(vsrc0, ptr[reg_src[2 * acc_iter] + u_idx * src_shift]);
    if (num_acc_iters * 2 > jsp.num_srcs && acc_iter == num_acc_iters - 1)
        uni_vpxor(vsrc1_vtmp, vsrc1_vtmp,
                vsrc1_vtmp); /* imitate additional zero input
                                        if number of srcs is odd */
    else
        vmovups(vsrc1_vtmp, ptr[reg_src[2 * acc_iter + 1] + u_idx * src_shift]);
}

void jit_avx512_core_bf16_sum_kernel_t::add_iter(int acc_iter, int u_idx) {
    Zmm vacc0 = Zmm(acc_vreg_idx(u_idx, 0));
    Zmm vacc1 = Zmm(acc_vreg_idx(u_idx, 1));
    Zmm vscale = Zmm(scale_vreg_idx(acc_iter));
    Zmm vsrc0 = Zmm(src_vreg_idx(u_idx, 2 * acc_iter));
    Zmm vsrc1 = Zmm(src_vreg_idx(u_idx, 2 * acc_iter + 1));
    Zmm vsrc1_vtmp = Zmm(tmp_vreg_idx(u_idx, acc_iter));

    vshuff64x2(vsrc1, vsrc0, vsrc1_vtmp, 0xEE);
    vpermw(vsrc1, zmm_idx, vsrc1);
    vshuff64x2(vsrc0, vsrc0, vsrc1_vtmp, 0x44);
    vpermw(vsrc0, zmm_idx, vsrc0);

    if (!isa_has_bf16(jsp.isa)) {
        bf16_emu_->vdpbf16ps(vacc0, vsrc0, vscale);
        bf16_emu_->vdpbf16ps(vacc1, vsrc1, vscale);
    } else {
        vdpbf16ps(vacc0, vsrc0, vscale);
        vdpbf16ps(vacc1, vsrc1, vscale);
    }
}

void jit_avx512_core_bf16_sum_kernel_t::write_iter(int u_idx, int dst_shift) {
    Zmm vacc0 = Zmm(acc_vreg_idx(u_idx, 0));
    Zmm vacc1 = Zmm(acc_vreg_idx(u_idx, 1));
    if (!jsp.is_bf16_dst) {
        vmovups(zword[reg_dst + 2 * u_idx * dst_shift], vacc0);
        vmovups(zword[reg_dst + (2 * u_idx + 1) * dst_shift], vacc1);
    } else {
        if (isa_has_bf16(jsp.isa)) {
            Zmm zmm_str = Zmm(tmp_vreg_idx(u_idx, 0));
            vcvtne2ps2bf16(zmm_str, vacc1, vacc0);
            vmovups(zword[reg_dst + 2 * u_idx * dst_shift], zmm_str);
        } else {
            auto ymm_str = Ymm(tmp_vreg_idx(u_idx, 0));
            bf16_emu_->vcvtneps2bf16(ymm_str, vacc0);
            bf16_emu_->vcvtneps2bf16(ymm_str, vacc0);
            vmovups(yword[reg_dst + 2 * u_idx * dst_shift], ymm_str);
            bf16_emu_->vcvtneps2bf16(ymm_str, vacc1);
            bf16_emu_->vcvtneps2bf16(ymm_str, vacc1);
            vmovups(yword[reg_dst + (2 * u_idx + 1) * dst_shift], ymm_str);
        }
    }
}

void jit_avx512_core_bf16_sum_kernel_t::tail_iteration() {
    Label tail_label, mask_label;
    L(tail_label);
    cmp(reg_sz, 0);
    jle(exit_label, T_NEAR);

    const int bf16_half_reg = vreg_traits<Zmm>::vlen / 4;
    mov(reg32_mask, 0xffff);
    cmp(reg_sz, bf16_half_reg);
    jge(mask_label, T_NEAR);

    mov(reg32_mask, 1);
    mov(rcx, reg_sz);
    shl(reg32_mask, cl);
    sub(reg32_mask, 1);

    L(mask_label);
    kmovd(k_mask, reg32_mask);
    Zmm vacc = Zmm(acc_vreg_idx(0, 0));
    uni_vpxor(vacc, vacc, vacc);

    for (int acc_iter = 0; acc_iter < num_acc_iters; acc_iter++) {
        const int isrc0 = 2 * acc_iter;
        const int isrc1 = 2 * acc_iter + 1;
        Zmm vscale = Zmm(scale_vreg_idx(acc_iter));
        Zmm vsrc = Zmm(src_vreg_idx(0, isrc0));
        Ymm vysrc0 = Ymm(src_vreg_idx(0, isrc0));
        Ymm vysrc1 = Ymm(src_vreg_idx(0, isrc1));
        uni_vpxor(vysrc0, vysrc0, vysrc0);
        uni_vpxor(vysrc1, vysrc1, vysrc1);

        vmovdqu16(vysrc0 | k_mask | T_z, ptr[reg_src[isrc0]]);
        if (!(num_acc_iters * 2 > jsp.num_srcs
                    && acc_iter == num_acc_iters - 1))
            vmovdqu16(vysrc1 | k_mask | T_z, ptr[reg_src[isrc1]]);
        vinserti64x4(vsrc, vsrc, vysrc1, 0x1);
        vpermw(vsrc, zmm_idx, vsrc);

        if (!isa_has_bf16(jsp.isa)) {
            bf16_emu_->vdpbf16ps(vacc, vsrc, vscale);
        } else {
            vdpbf16ps(vacc, vsrc, vscale);
        }
    }
    if (!jsp.is_bf16_dst) {
        vmovups(ptr[reg_dst] | k_mask, vacc);
    } else {
        if (isa_has_bf16(jsp.isa)) {
            auto ymm_str = Ymm(tmp_vreg_idx(0, 0));
            vcvtneps2bf16(ymm_str, vacc);
            vmovdqu16(ptr[reg_dst] | k_mask, ymm_str);
        } else {
            auto ymm_str = Ymm(tmp_vreg_idx(0, 0));
            bf16_emu_->vcvtneps2bf16(ymm_str, vacc);
            vmovdqu16(ptr[reg_dst] | k_mask, ymm_str);
        }
    }

    sub(reg_sz, bf16_half_reg);
    cmp(reg_sz, 0);
    jle(exit_label, T_NEAR);

    for (int s = 0; s < jsp.num_srcs; s++)
        add(reg_src[s], bf16_half_reg * jsp.typesize_in);
    add(reg_dst, (vreg_traits<Zmm>::vlen / 4) * jsp.typesize_out);

    jmp(tail_label, T_NEAR);
}

void jit_avx512_core_bf16_sum_kernel_t::index_tables() {
    align(64);
    L(idx_table);
    const uint16_t _idx[] = {0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7,
            23, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31};
    const dim_t _idx_size = sizeof(_idx) / sizeof(_idx[0]);
    for (dim_t i = 0; i < _idx_size; ++i)
        dw(_idx[i]);
}

status_t jit_avx512_core_bf16_sum_kernel_t::init_conf(
        jit_sum_conf_t &jsp, const int num_srcs, const memory_desc_t &dst_d) {
    jsp.num_srcs = num_srcs;
    jsp.loop_unroll = 0;
    jsp.isa = mayiuse(avx512_core_bf16) ? avx512_core_bf16
                                        : bf16_emulation_t::get_isa();

    const int max_unroll = 6; // maximum possible value of unroll is 6
    for (/*continue*/; jsp.loop_unroll < max_unroll; jsp.loop_unroll++) {
        const int num_regs
                = num_vregs_required(jsp.loop_unroll + 1, jsp.num_srcs);
        if (num_regs > (cpu_isa_traits<avx512_core>::n_vregs
                    - (isa_has_bf16(jsp.isa) ? 1 : 6)))
            break;
    }
    if (jsp.loop_unroll == 0) return status::unimplemented;
    jsp.size_blocking = (vreg_traits<Zmm>::vlen / 2) * jsp.loop_unroll;

    const memory_desc_wrapper o_d(&dst_d);
    jsp.is_bf16_dst = data_type::bf16 == o_d.data_type();

    jsp.typesize_in = sizeof(bfloat16_t);
    jsp.typesize_out = types::data_type_size(o_d.data_type());

    return status::success;
}

void jit_avx2_vnni_2_xf16_sum_kernel_t::broadcast_scale(int scale_iter) {
    Ymm vscale = Ymm(scale_vreg_idx(scale_iter));
    if (jsp.src_dt == data_type::bf16)
        vbcstnebf162ps(vscale, ptr[reg_scales + scale_iter * jsp.typesize_in]);
    else
        vbcstnesh2ps(vscale, ptr[reg_scales + scale_iter * jsp.typesize_in]);
}

void jit_avx2_vnni_2_xf16_sum_kernel_t::read_iter(
        int acc_iter, int u_idx, int src_shift) {
    Ymm vsrc0 = Ymm(src_vreg_idx(u_idx, 2 * acc_iter));
    Ymm vsrc1 = Ymm(src_vreg_idx(u_idx, 2 * acc_iter + 1));
    if (jsp.src_dt == data_type::bf16) {
        vcvtneebf162ps(vsrc0, ptr[reg_src[acc_iter] + u_idx * src_shift]);
        vcvtneobf162ps(vsrc1, ptr[reg_src[acc_iter] + u_idx * src_shift]);
    } else {
        vcvtneeph2ps(vsrc0, ptr[reg_src[acc_iter] + u_idx * src_shift]);
        vcvtneoph2ps(vsrc1, ptr[reg_src[acc_iter] + u_idx * src_shift]);
    }
}

void jit_avx2_vnni_2_xf16_sum_kernel_t::add_iter(int acc_iter, int u_idx) {
    Ymm vscale = Ymm(scale_vreg_idx(acc_iter));
    Ymm vsrc0 = Ymm(src_vreg_idx(u_idx, 2 * acc_iter));
    Ymm vsrc1 = Ymm(src_vreg_idx(u_idx, 2 * acc_iter + 1));
    Ymm vacc0 = Ymm(acc_vreg_idx(u_idx, 0));
    Ymm vacc1 = Ymm(acc_vreg_idx(u_idx, 1));
    vfmadd231ps(vacc0, vsrc0, vscale);
    vfmadd231ps(vacc1, vsrc1, vscale);
}

void jit_avx2_vnni_2_xf16_sum_kernel_t::write_iter(int u_idx, int dst_shift) {
    Ymm vacc0 = Ymm(acc_vreg_idx(u_idx, 0));
    Ymm vacc1 = Ymm(acc_vreg_idx(u_idx, 1));
    Ymm vtmp0 = Ymm(tmp_vreg_idx(u_idx, 0));
    Ymm vtmp1 = Ymm(tmp_vreg_idx(u_idx, 1));
    vunpcklps(vtmp0, vacc0, vacc1);
    vunpckhps(vtmp1, vacc0, vacc1);
    vperm2f128(vacc0, vtmp0, vtmp1, 0x20);
    vperm2f128(vacc1, vtmp0, vtmp1, 0x31);
    store_data<Ymm>(jsp.dst_dt, vacc0, reg_dst, 2 * u_idx * dst_shift, 8);
    store_data<Ymm>(jsp.dst_dt, vacc1, reg_dst, (2 * u_idx + 1) * dst_shift, 8);
}

void jit_avx2_vnni_2_xf16_sum_kernel_t::tail_iteration() {
    Label tail_label, tail_unroll_labels[4], shift_ptrs_label;

    L(tail_label);

    cmp(reg_sz, 0);
    jle(exit_label, T_NEAR);
    for (int unroll = 3; unroll >= 0; unroll--) {
        const unsigned char process_elems = 1 << unroll;
        mov(rsi, process_elems);
        cmp(reg_sz, rsi);
        jge(tail_unroll_labels[unroll], T_NEAR);
    }

    for (int unroll = 3; unroll >= 0; unroll--) {
        const unsigned char process_elems = 1 << unroll;
        L(tail_unroll_labels[unroll]);
        if (process_elems == 8) {
            Xmm vacc0l = Xmm(acc_vreg_idx(unroll, 0));
            Xmm vacc1l = Xmm(acc_vreg_idx(unroll, 1));
            uni_vpxor(vacc0l, vacc0l, vacc0l);
            uni_vpxor(vacc1l, vacc1l, vacc1l);
            for (int acc_iter = 0; acc_iter < num_acc_iters; acc_iter++) {
                Xmm vscale = Xmm(scale_vreg_idx(acc_iter));
                Xmm vsrc0 = Xmm(src_vreg_idx(unroll, 2 * acc_iter));
                Xmm vsrc1 = Xmm(src_vreg_idx(unroll, 2 * acc_iter + 1));
                if (jsp.src_dt == data_type::bf16) {
                    vcvtneebf162ps(vsrc0, ptr[reg_src[acc_iter]]);
                    vcvtneobf162ps(vsrc1, ptr[reg_src[acc_iter]]);
                } else {
                    vcvtneeph2ps(vsrc0, ptr[reg_src[acc_iter]]);
                    vcvtneoph2ps(vsrc1, ptr[reg_src[acc_iter]]);
                }
                vfmadd231ps(vacc0l, vsrc0, vscale);
                vfmadd231ps(vacc1l, vsrc1, vscale);
            }
            Xmm vtmp0 = Xmm(tmp_vreg_idx(unroll, 0));
            Xmm vtmp1 = Xmm(tmp_vreg_idx(unroll, 1));
            vunpcklps(vtmp0, vacc0l, vacc1l);
            vunpckhps(vtmp1, vacc0l, vacc1l);
            store_data<Xmm>(jsp.dst_dt, vtmp0, reg_dst, 0, 4);
            store_data<Xmm>(
                    jsp.dst_dt, vtmp1, reg_dst, 4 * jsp.typesize_out, 4);
        } else {
            Xmm vacc0l = Xmm(acc_vreg_idx(unroll, 0));
            uni_vpxor(vacc0l, vacc0l, vacc0l);
            for (int acc_iter = 0; acc_iter < num_acc_iters; acc_iter++) {
                Xmm vscale = Xmm(scale_vreg_idx(acc_iter));
                Xmm vsrc0 = Xmm(src_vreg_idx(unroll, acc_iter));
                Ymm vsrc0h = Ymm(src_vreg_idx(unroll, acc_iter));
                load_data<Xmm>(jsp.src_dt, vsrc0h, ptr[reg_src[acc_iter]],
                        process_elems);
                vfmadd231ps(vacc0l, vsrc0, vscale);
            }
            store_data<Xmm>(jsp.dst_dt, vacc0l, reg_dst, 0, process_elems);
        }
        jmp(shift_ptrs_label, T_NEAR);
    }

    L(shift_ptrs_label);
    sub(reg_sz, rsi);
    mov(rdi, rsi);
    shl(rsi, jsp.typesize_in / 2); // src shift
    shl(rdi, jsp.typesize_out / 2); // dst shift
    for (int s = 0; s < jsp.num_srcs; s++)
        add(reg_src[s], rsi);
    add(reg_dst, rdi);
    jmp(tail_label, T_NEAR);
}

status_t jit_avx2_vnni_2_xf16_sum_kernel_t::init_conf(jit_sum_conf_t &jsp,
        const int num_srcs, const std::vector<memory_desc_t> &src_d,
        const memory_desc_t &dst_d) {
    jsp.num_srcs = num_srcs;
    jsp.loop_unroll = 0;
    jsp.isa = avx2_vnni_2;
    jsp.loop_unroll = 6;
    jsp.unroll_reg_count = 2 * num_srcs + 4;
    jsp.size_blocking = (vreg_traits<Ymm>::vlen / 2) * jsp.loop_unroll;

    const memory_desc_wrapper i_d(&(src_d.front()));
    const memory_desc_wrapper o_d(&dst_d);
    jsp.is_bf16_dst = data_type::bf16 == o_d.data_type();

    jsp.src_dt = i_d.data_type();
    jsp.dst_dt = o_d.data_type();

    jsp.typesize_in = types::data_type_size(i_d.data_type());
    jsp.typesize_out = types::data_type_size(o_d.data_type());

    return status::success;
}

template <typename Vmm>
void jit_uni_xf16_sum_kernel_t<Vmm>::loop_iteration(int current_unroll) {
    Label loop_label, loop_exit_label;
    const int num_compute_elements
            = (vreg_traits<Vmm>::vlen / 2) * current_unroll;
    dim_t src_shift = (vreg_traits<Vmm>::vlen / 2) * jsp.typesize_in;
    dim_t dst_shift = (vreg_traits<Vmm>::vlen / 4) * jsp.typesize_out;
    L(loop_label);
    cmp(reg_sz, num_compute_elements);
    jl(loop_exit_label, T_NEAR);
    for (int u_idx = 0; u_idx < current_unroll; u_idx++) {
        Vmm vacc0 = Vmm(acc_vreg_idx(u_idx, 0));
        Vmm vacc1 = Vmm(acc_vreg_idx(u_idx, 1));
        uni_vpxor(vacc0, vacc0, vacc0);
        uni_vpxor(vacc1, vacc1, vacc1);
        for (int acc_iter = 0; acc_iter < num_acc_iters; acc_iter++) {
            read_iter(acc_iter, u_idx, src_shift);
            add_iter(acc_iter, u_idx);
        }
        write_iter(u_idx, dst_shift);
    }
    sub(reg_sz, num_compute_elements);
    for (int s = 0; s < jsp.num_srcs; s++)
        add(reg_src[s], current_unroll * src_shift);
    add(reg_dst, 2 * current_unroll * dst_shift);
    jge(loop_label, T_NEAR);
    L(loop_exit_label);
}

template <typename Vmm>
void jit_uni_xf16_sum_kernel_t<Vmm>::generate() {
    preamble();

    mov(reg_dst, ptr[param + GET_OFF(dst)]);
    mov(reg_srcs, ptr[param + GET_OFF(srcs)]);
    for (int s = 0; s < jsp.num_srcs; s++)
        mov(reg_src[s], ptr[reg_srcs + sizeof(void *) * s]);
    mov(reg_scales, ptr[param + GET_OFF(scales)]);
    mov(reg_sz, ptr[param + GET_OFF(size)]);

    pre_compute_init();

    for (int scale_iter = 0; scale_iter < num_acc_iters; scale_iter++)
        broadcast_scale(scale_iter);

    if (jsp.loop_unroll > 1) loop_iteration(jsp.loop_unroll);
    loop_iteration(1);

    tail_iteration();

    L(exit_label);
    postamble();

    index_tables();
}

template <data_type_t src_data_type, data_type_t dst_data_type, cpu_isa_t isa>
status_t jit_xf16_sum_t<src_data_type, dst_data_type, isa>::execute(
        const exec_ctx_t &ctx) const {
    auto output = CTX_OUT_MEM(dst_data_t *, DNNL_ARG_DST);
    const memory_desc_wrapper o_d(pd()->dst_md());
    output += o_d.blk_off(0);
    const int num_arrs = pd()->n_inputs();
    const dim_t nelems = o_d.nelems(true);
    const src_data_t
            *input_ptrs[jit_avx512_core_bf16_sum_kernel_t::max_num_arrs];
    /* Number of scales needs to be multiple of 2 in order
    to use VNNI instructions */
    src_data_t scales[jit_avx512_core_bf16_sum_kernel_t::max_num_arrs];
    for (int a = 0; a < num_arrs; ++a) {
        const memory_desc_wrapper i_d(pd()->src_md(a));

        input_ptrs[a]
                = CTX_IN_MEM(const src_data_t *, DNNL_ARG_MULTIPLE_SRC + a)
                + i_d.blk_off(0);
    }
    if (src_data_type == data_type::bf16)
        cvt_float_to_bfloat16(
                (bfloat16_t *)scales, &pd()->scales()[0], num_arrs);
    else
        cvt_float_to_float16((float16_t *)scales, &pd()->scales()[0], num_arrs);

    if (isa == avx512_core && num_arrs % 2 != 0) scales[num_arrs] = 0.0f;

    const dim_t half_L1 = 16 * 1024; // bytes
    const dim_t num_elems_in_block = utils::rnd_up(
            utils::div_up(half_L1,
                    num_arrs * sizeof(src_data_t) + sizeof(dst_data_t)),
            pd()->jsp_.size_blocking);
    const dim_t num_blocks = nelems / num_elems_in_block;
    const dim_t tail = nelems % num_elems_in_block;

#if defined(__GNUC__) && __GNUC__ == 4 && __GNUC_MINOR__ == 8 \
        && __GNUC_PATCHLEVEL__ == 3
// GCC issues a false positive warning 'array subscript is above array bounds'
// with gcc 4.8.3 + -march=native option, so disable it for now
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Warray-bounds"
#endif
    parallel(0, [&](const int ithr, const int nthr) {
        dim_t start {0}, end {0};
        balance211(num_blocks, nthr, ithr, start, end);
        auto arg = jit_sum_call_t();
        const src_data_t *local_input_ptrs
                [jit_avx512_core_bf16_sum_kernel_t::max_num_arrs];
        dst_data_t *local_output;

        for (dim_t nb = start; nb < end; ++nb) {
            dim_t start_e = nb * num_elems_in_block;
            for (int a = 0; a < num_arrs; ++a) {
                local_input_ptrs[a] = &input_ptrs[a][start_e];
            }
            local_output = &output[start_e];
            arg.srcs = (const void **)local_input_ptrs;
            arg.dst = (const void *)local_output;
            arg.scales = (const void *)scales;
            arg.size = num_elems_in_block;
            (*kernel_)(&arg);
        }

        if (tail != 0 && ithr == nthr - 1) {
            dim_t start_e = nelems - tail;
            for (int a = 0; a < num_arrs; ++a) {
                local_input_ptrs[a] = &input_ptrs[a][start_e];
            }
            local_output = &output[start_e];
            arg.srcs = (const void **)local_input_ptrs;
            arg.dst = (const void *)local_output;
            arg.scales = (const void *)scales;
            arg.size = tail;
            (*kernel_)(&arg);
        }
    });
#if defined(__GNUC__) && __GNUC__ == 4 && __GNUC_MINOR__ == 8 \
        && __GNUC_PATCHLEVEL__ == 3
#pragma GCC diagnostic pop
#endif
    return status::success;
}

template struct jit_xf16_sum_t<data_type::bf16, data_type::f32, avx512_core>;
template struct jit_xf16_sum_t<data_type::bf16, data_type::bf16, avx512_core>;
template struct jit_xf16_sum_t<data_type::bf16, data_type::f32, avx2_vnni_2>;
template struct jit_xf16_sum_t<data_type::bf16, data_type::bf16, avx2_vnni_2>;
template struct jit_xf16_sum_t<data_type::f16, data_type::f32, avx2_vnni_2>;
template struct jit_xf16_sum_t<data_type::f16, data_type::f16, avx2_vnni_2>;

template struct jit_uni_xf16_sum_kernel_t<Xbyak::Zmm>;
template struct jit_uni_xf16_sum_kernel_t<Xbyak::Ymm>;

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
