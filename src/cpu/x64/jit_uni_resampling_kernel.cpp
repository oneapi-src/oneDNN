/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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

#include "cpu/x64/jit_avx512_core_bf16cvt.hpp"
#include "cpu/x64/jit_generator.hpp"
#include "cpu/x64/jit_uni_resampling_kernel.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

using namespace Xbyak;
using namespace format_tag;

#define GET_OFF(field) offsetof(jit_resampling_call_s, field)

template <cpu_isa_t isa>
jit_uni_resampling_kernel<isa>::jit_uni_resampling_kernel(
        const jit_resampling_conf_t conf)
    : jit_generator(nullptr, MAX_CODE_SIZE, true, isa), conf_(conf) {
    const bool use_bf16_emulation = conf_.data_type == data_type::bf16
            && conf_.isa != avx512_core_bf16;
    bf16_emulation_ = use_bf16_emulation
            ? utils::make_unique<bf16_emulation_t>(this, bf16_emu_reserv_1,
                    bf16_emu_reserv_2, bf16_emu_reserv_3, bf16_emu_scratch,
                    bf16_emu_reserv_4)
            : nullptr;
}

template <cpu_isa_t isa>
void jit_uni_resampling_kernel<isa>::prepare_mask() {}

template <>
void jit_uni_resampling_kernel<avx512_common>::prepare_mask() {
    const size_t tail_mask = (1ULL << conf_.tail) - 1ULL;
    const Reg64 &reg_tail = reg_tmp_;
    mov(reg_tail.cvt32(), tail_mask);
    kmovw(k_tail_mask_, reg_tail.cvt32());
}

template <>
void jit_uni_resampling_kernel<avx>::prepare_mask() {
    static constexpr uint32_t mask[16]
            = {0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff,
                    0xffffffff, 0xffffffff, 0xffffffff, 0, 0, 0, 0, 0, 0, 0, 0};
    mov(reg_tmp_, reinterpret_cast<size_t>(&mask[8 - conf_.tail]));
    vmovups(vmm_tail_mask_, ptr[reg_tmp_]);
}

template <>
void jit_uni_resampling_kernel<avx512_common>::emu_gather_data(
        const Reg64 &reg_src_addr, const int indices_idx, const int data_idx,
        const bool is_tail) {
    assert(conf_.data_type == data_type::bf16);

    const Xmm xmm_tmp = Xmm(vmm_full_mask_.getIdx());
    const Xmm xmm_dst = Xmm(vmm_tmp_.getIdx());

    mov(reg_tmp_, 0);
    mov(reg_tmp1_, reg_src_addr);

    constexpr unsigned xmm_size_elem = 4;

    const unsigned number_of_xmms = is_tail
            ? utils::div_up(conf_.tail, xmm_size_elem)
            : utils::div_up(conf_.simd_w, xmm_size_elem);
    for (unsigned i = 0; i < number_of_xmms; i++) {
        vextractf32x4(xmm_tmp, Zmm(indices_idx), i);

        const unsigned number_of_values_to_load = i == number_of_xmms - 1
                        && is_tail && conf_.tail % xmm_size_elem != 0
                ? conf_.tail % xmm_size_elem
                : xmm_size_elem;
        for (unsigned j = 0; j < number_of_values_to_load; j++) {
            vpextrd(reg_tmp_.cvt32(), xmm_tmp, j);
            add(reg_src_addr, reg_tmp_);
            vpinsrw(xmm_dst, xmm_dst, ptr[reg_src_addr], j * 2);
            mov(reg_src_addr, reg_tmp1_);
        }

        vinsertf32x4(Zmm(data_idx), Zmm(data_idx), xmm_dst, i);
    }

    uni_vpslld(Zmm(data_idx), Zmm(data_idx), 16);
}

template <>
void jit_uni_resampling_kernel<avx>::emu_gather_data(const Reg64 &reg_src_addr,
        const int indices_idx, const int data_idx, const bool is_tail) {
    const Xmm xmm_tmp = Xmm(vmm_full_mask_.getIdx());
    const Xmm xmm_dst = Xmm(vmm_tmp_.getIdx());

    mov(reg_tmp_, 0);
    mov(reg_tmp1_, reg_src_addr);

    constexpr unsigned xmm_size_elem = 4;

    const unsigned number_of_xmms = is_tail
            ? utils::div_up(conf_.tail, xmm_size_elem)
            : utils::div_up(conf_.simd_w, xmm_size_elem);
    for (unsigned i = 0; i < number_of_xmms; i++) {
        vextractf128(xmm_tmp, Ymm(indices_idx), i);

        const unsigned number_of_values_to_load = i == number_of_xmms - 1
                        && is_tail && conf_.tail % xmm_size_elem != 0
                ? conf_.tail % xmm_size_elem
                : xmm_size_elem;
        for (unsigned j = 0; j < number_of_values_to_load; j++) {
            vpextrd(reg_tmp_.cvt32(), xmm_tmp, j);
            add(reg_src_addr, reg_tmp_);
            vpinsrd(xmm_dst, xmm_dst, ptr[reg_src_addr], j);
            mov(reg_src_addr, reg_tmp1_);
        }

        vinsertf128(Ymm(data_idx), Ymm(data_idx), xmm_dst, i);
    }
}

template <>
void jit_uni_resampling_kernel<sse41>::emu_gather_data(
        const Reg64 &reg_src_addr, const int indices_idx, const int data_idx,
        const bool is_tail) {
    mov(reg_tmp_, 0);
    mov(reg_tmp1_, reg_src_addr);

    constexpr unsigned xmm_size_elem = 4;

    const unsigned number_of_values_to_load
            = is_tail ? conf_.tail : xmm_size_elem;
    for (unsigned j = 0; j < number_of_values_to_load; j++) {
        pextrd(reg_tmp_.cvt32(), Xmm(indices_idx), j);
        add(reg_src_addr, reg_tmp_);
        pinsrd(Xmm(data_idx), ptr[reg_src_addr], j);
        mov(reg_src_addr, reg_tmp1_);
    }
}

template <>
void jit_uni_resampling_kernel<avx512_common>::gather_data(
        const Reg64 &reg_src_addr, const int indices_idx, const int data_idx,
        const bool is_tail) {
    if (conf_.data_type == data_type::f32) {
        const Opmask &mask = is_tail ? k_tail_mask_ : k_full_mask_;
        if (!is_tail) {
            // Have to set the all bits to 1 gather full
            // register. It is needed after each gather, because
            // vgatherdps zeros the mask if successful
            mov(reg_tmp_.cvt32(), (1ULL << conf_.simd_w) - 1ULL);
            kmovw(k_full_mask_, reg_tmp_.cvt32());
        }
        vgatherdps(Vmm(data_idx) | mask, ptr[reg_src_addr + Vmm(indices_idx)]);
        // Have to restore tail processing mask after gather because mask
        // was zeroed after vgatherdps.
        if (is_tail) prepare_mask();
    } else {
        emu_gather_data(reg_src_addr, indices_idx, data_idx, is_tail);
    }
}

template <>
void jit_uni_resampling_kernel<avx>::gather_data(const Reg64 &reg_src_addr,
        const int indices_idx, const int data_idx, const bool is_tail) {
    if (conf_.isa == avx2) {
        const Vmm &mask = is_tail ? vmm_tail_mask_ : vmm_full_mask_;
        if (!is_tail) {
            // Have to set the all bits to 1 gather full
            // register. It is needed after each gather, because
            // vgatherdps zeros the mask if successful
            vcmpps(vmm_full_mask_, vmm_full_mask_, vmm_full_mask_, _cmp_eq_oq);
        }
        vgatherdps(Vmm(data_idx), ptr[reg_src_addr + Vmm(indices_idx)], mask);
        // Have to restore tail processing mask after gather because mask
        // was zeroed after vgatherdps.
        if (is_tail) prepare_mask();
    } else {
        emu_gather_data(reg_src_addr, indices_idx, data_idx, is_tail);
    }
}

template <>
void jit_uni_resampling_kernel<sse41>::gather_data(const Reg64 &reg_src_addr,
        const int indices_idx, const int data_idx, const bool is_tail) {
    emu_gather_data(reg_src_addr, indices_idx, data_idx, is_tail);
}

template <>
void jit_uni_resampling_kernel<avx512_common>::store_data(const int data_idx,
        const Reg64 &reg_dst_addr, const int offset, const bool is_tail) {
    if (conf_.data_type == data_type::bf16) {
        const Ymm to_store_data = Ymm(data_idx);

        if (bf16_emulation_)
            bf16_emulation_->vcvtneps2bf16(to_store_data, Zmm(data_idx));
        else
            vcvtneps2bf16(to_store_data, Zmm(data_idx));

        if (is_tail) {
            vmovdqu16(ptr[reg_dst_addr + offset] | k_tail_mask_, to_store_data);
        } else {
            if (conf_.is_data_size_bigger_than_L3 && conf_.tail == 0)
                vmovntps(ptr[reg_dst_addr + offset], to_store_data);
            else
                vmovups(ptr[reg_dst_addr + offset], to_store_data);
        }
    } else {
        if (is_tail) {
            vmovups(ptr[reg_dst_addr + offset] | k_tail_mask_, Vmm(data_idx));
        } else {
            if (conf_.is_data_size_bigger_than_L3 && conf_.tail == 0)
                vmovntps(ptr[reg_dst_addr + offset], Vmm(data_idx));
            else
                vmovups(ptr[reg_dst_addr + offset], Vmm(data_idx));
        }
    }
}

template <>
void jit_uni_resampling_kernel<avx>::store_data(const int data_idx,
        const Reg64 &reg_dst_addr, const int offset, const bool is_tail) {
    if (is_tail) {
        vmaskmovps(ptr[reg_dst_addr + offset], vmm_tail_mask_, Vmm(data_idx));
    } else {
        if (conf_.is_data_size_bigger_than_L3 && conf_.tail == 0
                && conf_.isa == avx2)
            vmovntps(ptr[reg_dst_addr + offset], Vmm(data_idx));
        else
            vmovups(ptr[reg_dst_addr + offset], Vmm(data_idx));
    }
}

template <>
void jit_uni_resampling_kernel<sse41>::store_data(const int data_idx,
        const Reg64 &reg_dst_addr, const int offset, const bool is_tail) {
    if (is_tail) {
        for (unsigned i = 0; i < conf_.tail; i++) {
            pextrd(ptr[reg_dst_addr + offset + i * conf_.dt_size],
                    Xmm(data_idx), i);
        }
    } else {
        movups(ptr[reg_dst_addr + offset], Vmm(data_idx));
    }
}

template <>
void jit_uni_resampling_kernel<avx512_common>::load_data(
        const Reg64 &reg_src_addr, const int offset, const int data_idx,
        const bool is_tail) {
    if (conf_.data_type == data_type::bf16) {
        const Zmm loaded_data = is_tail
                ? Zmm(data_idx) | k_tail_mask_ | Xbyak::util::T_z
                : Zmm(data_idx);
        vpmovzxwd(loaded_data, ptr[reg_src_addr + offset]);
        vpslld(loaded_data, loaded_data, 16);
    } else {
        if (is_tail) {
            vmovups(Vmm(data_idx) | k_tail_mask_ | T_z,
                    ptr[reg_src_addr + offset]);
        } else {
            vmovups(Vmm(data_idx), ptr[reg_src_addr + offset]);
        }
    }
}

template <>
void jit_uni_resampling_kernel<avx>::load_data(const Reg64 &reg_src_addr,
        const int offset, const int data_idx, const bool is_tail) {
    if (is_tail) {
        vmaskmovps(Vmm(data_idx), vmm_tail_mask_, ptr[reg_src_addr + offset]);
    } else {
        vmovups(Vmm(data_idx), ptr[reg_src_addr + offset]);
    }
}

template <>
void jit_uni_resampling_kernel<sse41>::load_data(const Reg64 &reg_src_addr,
        const int offset, const int data_idx, const bool is_tail) {
    if (is_tail) {
        for (unsigned i = 0; i < conf_.tail; i++) {
            pinsrd(Xmm(data_idx),
                    ptr[reg_src_addr + offset + i * conf_.dt_size], i);
        }
    } else {
        movups(Vmm(data_idx), ptr[reg_src_addr + offset]);
    }
}

template <cpu_isa_t isa>
void jit_uni_resampling_kernel<isa>::nearest_ncsp_format() {
    const Reg64 &reg_indices_h = reg_aux_src_0_;
    const Reg64 &reg_indices_w = reg_aux_src_1_;
    const Reg64 &reg_src_shifted = reg_aux_src_2_;
    const Reg64 &reg_oh = reg_tmp1_;

    auto nearest_interpolation = ([&](bool is_tail) {
        uni_vmovdqu(vmm_indices_, ptr[reg_indices_w]);
        gather_data(reg_src_shifted, vmm_indices_.getIdx(), vmm_src_.getIdx(),
                is_tail);
        store_data(vmm_src_.getIdx(), reg_dst_, 0, is_tail);
    });

    mov(reg_indices_h, reg_indices_);
    mov(reg_indices_w, reg_indices_);
    add(reg_indices_w, conf_.oh * conf_.el_size_of_indices);

    Label oh_loop_begin, oh_loop_end;
    Label ow_loop_begin, ow_loop_end;
    mov(reg_oh, 0);

    L(oh_loop_begin);
    {
        cmp(reg_oh, conf_.oh);
        jge(oh_loop_end, T_NEAR);
        push(reg_oh);

        mov(reg_work_, conf_.ow);
        mov(reg_src_shifted, reg_src_);
        mov(reg_tmp_, 0);
        mov(reg_tmp_.cvt32(), dword[reg_indices_h]);
        add(reg_src_shifted, reg_tmp_);

        push(reg_indices_w);

        L(ow_loop_begin);
        {
            cmp(reg_work_, conf_.simd_w);
            jl(ow_loop_end, T_NEAR);

            nearest_interpolation(false);

            add(reg_dst_, conf_.simd_w * conf_.dt_size);
            add(reg_indices_w, conf_.simd_w * conf_.el_size_of_indices);
            sub(reg_work_, conf_.simd_w);

            jmp(ow_loop_begin, T_NEAR);
        }
        L(ow_loop_end);

        if (conf_.tail > 0) {
            nearest_interpolation(true);
            add(reg_dst_, conf_.tail * conf_.dt_size);
        }

        add(reg_indices_h, conf_.el_size_of_indices);
        pop(reg_indices_w);
        pop(reg_oh);
        add(reg_oh, 1);
        jmp(oh_loop_begin);
    }
    L(oh_loop_end);
}

template <cpu_isa_t isa>
void jit_uni_resampling_kernel<isa>::nearest_c_oriented_format() {
    const Reg64 &reg_c = reg_tmp_;
    const Reg64 &reg_src_shifted = reg_aux_src_0_;

    Label loop_begin, loop_end;

    L(loop_begin);
    {
        cmp(reg_work_, 1);
        jl(loop_end, T_NEAR);

        mov(reg_src_shifted, reg_src_);
        mov(reg_tmp1_.cvt32(), dword[reg_indices_]);
        add(reg_src_shifted, reg_tmp1_);

        Label c_loop_begin, c_loop_end;
        mov(reg_c, conf_.inner_stride);
        L(c_loop_begin);
        {
            cmp(reg_c, conf_.simd_w);
            jl(c_loop_end, T_NEAR);

            load_data(reg_src_shifted, 0, vmm_src_.getIdx());
            store_data(vmm_src_.getIdx(), reg_dst_);
            add(reg_src_shifted, conf_.simd_w * conf_.dt_size);
            add(reg_dst_, conf_.simd_w * conf_.dt_size);

            sub(reg_c, conf_.simd_w);
            jmp(c_loop_begin, T_NEAR);
        }
        L(c_loop_end);

        if (conf_.tail > 0) {
            load_data(reg_src_shifted, 0, vmm_src_.getIdx(), true);
            store_data(vmm_src_.getIdx(), reg_dst_, 0, true);
            add(reg_dst_, conf_.tail * conf_.dt_size);
        }

        add(reg_indices_, conf_.el_size_of_indices);

        dec(reg_work_);
        jmp(loop_begin, T_NEAR);
    }
    L(loop_end);
}

template <cpu_isa_t isa>
void jit_uni_resampling_kernel<isa>::linear_ncsp_format() {
    const unsigned indices_stride
            = conf_.ow * conf_.oh * conf_.od * conf_.el_size_of_indices;
    const unsigned weights_stride
            = conf_.ow * conf_.oh * conf_.od * sizeof(float);

    auto linear_interpolation = ([&](const bool is_tail) {
        for (unsigned i = 0; i < conf_.number_of_corners; i++) {
            uni_vmovdqu(vmm_indices_, ptr[reg_indices_ + i * indices_stride]);
            gather_data(reg_src_, vmm_indices_.getIdx(), vmm_idx(i), is_tail);
        }

        uni_vmovups(vmm_weights_, ptr[reg_weights]);
        uni_vmulps(Vmm(vmm_idx(0)), Vmm(vmm_idx(0)), vmm_weights_);
        for (unsigned i = 1; i < conf_.number_of_corners; i++) {
            uni_vmovups(vmm_weights_, ptr[reg_weights + i * weights_stride]);
            uni_vfmadd231ps(Vmm(vmm_idx(0)), Vmm(vmm_idx(i)), vmm_weights_);
        }

        store_data(vmm_idx(0), reg_dst_, 0, is_tail);
    });

    Label loop_begin, loop_end;

    L(loop_begin);
    {
        cmp(reg_work_, conf_.simd_w);
        jl(loop_end, T_NEAR);

        linear_interpolation(false);

        add(reg_dst_, conf_.simd_w * conf_.dt_size);
        add(reg_weights, conf_.simd_w * sizeof(float));
        add(reg_indices_, conf_.simd_w * conf_.el_size_of_indices);
        sub(reg_work_, conf_.simd_w);

        jmp(loop_begin, T_NEAR);
    }
    L(loop_end);

    if (conf_.tail > 0) linear_interpolation(true);
}

template <cpu_isa_t isa>
void jit_uni_resampling_kernel<isa>::linear_c_oriented_format() {
    const Reg64 &reg_c = reg_tmp_;
    const Reg64 &reg_index_left = reg_tmp_;
    const Reg64 &reg_index_right = reg_tmp_;

    std::vector<std::reference_wrapper<const Reg64>> src_regs
            = {reg_src_ftl_, reg_src_ftr_, reg_src_fbl_, reg_src_fbr_,
                    reg_src_btl_, reg_src_btr_, reg_src_bbl_, reg_src_bbr_};
    std::vector<std::reference_wrapper<const Vmm>> src_vmms
            = {src_ftl_, src_ftr_, src_fbl_, src_fbr_, src_btl_, src_btr_,
                    src_bbl_, src_bbr_};

    assert(src_regs.size() >= conf_.number_of_corners
            && src_vmms.size() >= conf_.number_of_corners);

    auto linear_interpolation = ([&](const unsigned offset,
                                         const bool is_tail) {
        for (unsigned i = 0; i < conf_.number_of_corners; i++) {
            load_data(src_regs[i], offset, src_vmms[i].get().getIdx(), is_tail);
        }

        // w_d[0]*(w_h[0]*(src[0][0][0]*w_w[0] + src[0][0][1]*w_w[1]) +
        //         w_h[1]*(src[0][1][0]*w_w[0] + src[0][1][1]*w_w[1]))
        // +
        // w_d[1]*(w_h[0]*(src[1][0][0]*w_w[0] + src[1][0][1]*w_w[1]) +
        //         w_h[1]*(src[1][1][0]*w_w[0] + src[1][1][1]*w_w[1]))
        uni_vmulps(src_ftl_, src_ftl_, weight_left_);
        uni_vfmadd231ps(src_ftl_, src_ftr_, weight_right_);
        if (conf_.ndims == 4 || conf_.ndims == 5) {
            uni_vmulps(src_fbl_, src_fbl_, weight_left_);
            uni_vfmadd231ps(src_fbl_, src_fbr_, weight_right_);
            uni_vmulps(src_ftl_, src_ftl_, weight_top_);
            uni_vfmadd231ps(src_ftl_, src_fbl_, weight_bottom_);
        }
        if (conf_.ndims == 5) {
            uni_vmulps(src_btl_, src_btl_, weight_left_);
            uni_vfmadd231ps(src_btl_, src_btr_, weight_right_);
            uni_vmulps(src_bbl_, src_bbl_, weight_left_);
            uni_vfmadd231ps(src_bbl_, src_bbr_, weight_right_);
            uni_vmulps(src_btl_, src_btl_, weight_top_);
            uni_vfmadd231ps(src_btl_, src_bbl_, weight_bottom_);
            uni_vmulps(src_ftl_, src_ftl_, weight_front_);
            uni_vfmadd231ps(src_ftl_, src_btl_, weight_back_);
        }

        store_data(src_ftl_.getIdx(), reg_dst_, offset, is_tail);
    });

    mov(reg_index_left, 0);

    Label loop_begin, loop_end;
    L(loop_begin);
    {
        cmp(reg_work_, 1);
        jl(loop_end, T_NEAR);

        for (unsigned i = 0; i < conf_.number_of_corners; i++) {
            push(src_regs[i]);
        }

        mov(reg_index_left.cvt32(), dword[reg_indices_]);
        for (unsigned i = 0; i < conf_.number_of_corners / 2; i++) {
            add(src_regs[2 * i], reg_index_left);
        }
        mov(reg_index_right.cvt32(),
                dword[reg_indices_ + conf_.el_size_of_indices]);
        for (unsigned i = 0; i < conf_.number_of_corners / 2; i++) {
            add(src_regs[2 * i + 1], reg_index_right);
        }

        uni_vbroadcastss(weight_left_, ptr[reg_weights]);
        uni_vbroadcastss(weight_right_, ptr[reg_weights + sizeof(float)]);

        Label c_loop_begin, c_loop_end;
        mov(reg_c, conf_.inner_stride);
        L(c_loop_begin);
        {
            cmp(reg_c, conf_.simd_w);
            jl(c_loop_end, T_NEAR);

            linear_interpolation(0, false);
            add(reg_dst_, conf_.simd_w * conf_.dt_size);

            for (unsigned i = 0; i < conf_.number_of_corners; i++)
                add(src_regs[i], conf_.simd_w * conf_.dt_size);

            sub(reg_c, conf_.simd_w);
            jmp(c_loop_begin, T_NEAR);
        }
        L(c_loop_end);

        if (conf_.tail > 0) {
            linear_interpolation(0, true);
            add(reg_dst_, conf_.tail * conf_.dt_size);
        }

        // During one loop cycle are read two values for left and
        // right corners from both the weights and indices tables.
        // These two values occurs one after the other in memory,
        // so the address should be shifted by two elements.
        add(reg_indices_, 2 * conf_.el_size_of_indices);
        add(reg_weights, 2 * sizeof(float));

        for (unsigned i = 0; i < conf_.number_of_corners; i++) {
            pop(src_regs[(conf_.number_of_corners - 1) - i]);
        }

        dec(reg_work_);
        jmp(loop_begin, T_NEAR);
    }
    L(loop_end);
}

template <cpu_isa_t isa>
void jit_uni_resampling_kernel<isa>::generate() {
    preamble();

#if defined(_WIN32)
    // Always mimic the Unix ABI
    xor_(rdi, rcx);
    xor_(rcx, rdi);
    xor_(rdi, rcx);
#endif
    if (bf16_emulation_) bf16_emulation_->init_vcvtneps2bf16();

    if (conf_.isa == avx2 && conf_.tag_kind == jit_memory_tag_kind_t::ncsp) {
        // Sometimes the values in the register can be nan at the
        // beginning of the kernel, then using vcmpps(vmm, vmm, vmm)
        // will not set all bits to 1 for that value, instead
        // this instruction will zeroed this value. At the beginning,
        // it is worth to zeroing this register to be sure, that vcmpps
        // will work properly.
        uni_vxorps(vmm_full_mask_, vmm_full_mask_, vmm_full_mask_);
    }

    if (conf_.tail > 0) prepare_mask();

    mov(reg_dst_, ptr[reg_param + GET_OFF(dst)]);
    mov(reg_work_, ptr[reg_param + GET_OFF(batch_of_sp_points_to_process)]);
    mov(reg_indices_, ptr[reg_param + GET_OFF(indices)]);

    if (conf_.alg == alg_kind::resampling_nearest) {
        mov(reg_src_, ptr[reg_param + GET_OFF(src)]);
        if (conf_.tag_kind == jit_memory_tag_kind_t::ncsp)
            nearest_ncsp_format();
        else if (conf_.tag_kind == jit_memory_tag_kind_t::nspc
                || conf_.tag_kind == jit_memory_tag_kind_t::blocked)
            nearest_c_oriented_format();
    } else if (utils::one_of(conf_.alg, alg_kind::resampling_linear,
                       alg_kind::resampling_linear_no_shift)) {
        mov(reg_weights, ptr[reg_param + GET_OFF(weights)]);
        if (conf_.tag_kind == jit_memory_tag_kind_t::ncsp) {
            mov(reg_src_, ptr[reg_param + GET_OFF(src)]);
            linear_ncsp_format();
        } else if (conf_.tag_kind == jit_memory_tag_kind_t::nspc
                || conf_.tag_kind == jit_memory_tag_kind_t::blocked) {
            mov(reg_src_ftl_, ptr[reg_param + GET_OFF(src)]);
            add(reg_src_ftl_, ptr[reg_param + GET_OFF(src_offset_front)]);
            add(reg_src_ftl_, ptr[reg_param + GET_OFF(src_offset_top)]);
            mov(reg_src_ftr_, reg_src_ftl_);

            if (conf_.ndims == 4 || conf_.ndims == 5) {
                uni_vbroadcastss(
                        weight_top_, ptr[reg_param + GET_OFF(weight_top)]);
                uni_vbroadcastss(weight_bottom_,
                        ptr[reg_param + GET_OFF(weight_bottom)]);
                mov(reg_src_fbl_, ptr[reg_param + GET_OFF(src)]);
                add(reg_src_fbl_, ptr[reg_param + GET_OFF(src_offset_front)]);
                add(reg_src_fbl_, ptr[reg_param + GET_OFF(src_offset_bottom)]);
                mov(reg_src_fbr_, reg_src_fbl_);
            }
            if (conf_.ndims == 5) {
                uni_vbroadcastss(
                        weight_front_, ptr[reg_param + GET_OFF(weight_front)]);
                uni_vbroadcastss(
                        weight_back_, ptr[reg_param + GET_OFF(weight_back)]);
                mov(reg_src_btl_, ptr[reg_param + GET_OFF(src)]);
                add(reg_src_btl_, ptr[reg_param + GET_OFF(src_offset_back)]);
                add(reg_src_btl_, ptr[reg_param + GET_OFF(src_offset_top)]);
                mov(reg_src_btr_, reg_src_btl_);

                mov(reg_src_bbl_, ptr[reg_param + GET_OFF(src)]);
                add(reg_src_bbl_, ptr[reg_param + GET_OFF(src_offset_back)]);
                add(reg_src_bbl_, ptr[reg_param + GET_OFF(src_offset_bottom)]);
                mov(reg_src_bbr_, reg_src_bbl_);
            }
            linear_c_oriented_format();
        }
    }

    postamble();
}

template struct jit_uni_resampling_kernel<avx512_common>;
template struct jit_uni_resampling_kernel<avx>;
template struct jit_uni_resampling_kernel<sse41>;

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
