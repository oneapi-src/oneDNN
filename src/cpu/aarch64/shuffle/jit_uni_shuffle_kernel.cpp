/*******************************************************************************
* Copyright 2021-2022 Intel Corporation
* Copyright 2022 FUJITSU LIMITED
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

#include <cassert>

#include "common/bfloat16.hpp"
#include "common/c_types_map.hpp"

#include "cpu/aarch64/jit_generator.hpp"
#include "cpu/aarch64/shuffle/jit_uni_shuffle_kernel.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

using namespace Xbyak_aarch64;

#define GET_OFF(field) offsetof(jit_shuffle_call_s, field)

static size_t get_padding_size(const jit_shuffle_conf_t &conf) {
    const auto padding_tail_size = conf.c % conf.blk_size;
    return (padding_tail_size) ? conf.blk_size - padding_tail_size : 0;
}

template <cpu_isa_t isa>
jit_uni_shuffle_kernel_t<isa>::jit_uni_shuffle_kernel_t(
        const jit_shuffle_conf_t conf)
    : jit_generator(nullptr, MAX_CODE_SIZE, true)
    , conf_(conf)
    , padding_size_(get_padding_size(conf)) {}

template <cpu_isa_t isa>
void jit_uni_shuffle_kernel_t<isa>::prepare_mask() {
    using namespace data_type;
    if (conf_.simd_tail > 0) {
        assert(utils::one_of(conf_.data_type, f32, s32));
        assert(conf_.simd_tail < isa_sveLen / sizeof(float));
        index(vmm_tmp_.s, 0, 1);
        cmplt(k_tail_mask_.s, P_ALL_ONE / T_z, vmm_tmp_.s, conf_.simd_tail);
    }

    if (isa_sveLen == util::SVE_512)
        ptrue(k_full_mask_.s, VL16);
    else if (isa_sveLen == util::SVE_256)
        ptrue(k_full_mask_.s, VL8);
    else if (isa_sveLen == util::SVE_128)
        ptrue(k_full_mask_.s, VL4);
}

template <>
void jit_uni_shuffle_kernel_t<asimd>::prepare_mask() {}

template <cpu_isa_t isa>
void jit_uni_shuffle_kernel_t<isa>::gather_data(const XReg &reg_src_addr,
        const int indices_idx, const int data_idx, const bool is_tail) {
    if (conf_.dt_size == sizeof(float)) {
        const PReg &mask = is_tail ? k_tail_mask_ : k_full_mask_;
        lsr(TRegS(indices_idx), TRegS(indices_idx), 2);
        ld1w(TRegS(data_idx), mask / T_z,
                ptr(reg_src_addr, TRegS(indices_idx), UXTW, 2));
    } else {
        assert(!"unsupported emu_gather_data");
    }
}

template <>
void jit_uni_shuffle_kernel_t<asimd>::gather_data(const XReg &addr,
        const int indices_idx, const int data_idx, const bool is_tail) {
    constexpr unsigned xmm_size_elem = 4;
    const VReg4S v(data_idx);

    const unsigned number_of_values_to_load
            = is_tail ? conf_.simd_tail : xmm_size_elem;

    for (unsigned j = 0; j < number_of_values_to_load; j++) {
        mov(W_TMP_0, VReg4S(indices_idx)[j]);
        add(X_DEFAULT_ADDR, addr, X_TMP_0);
        ld1(v[j], ptr(X_DEFAULT_ADDR));
    }
}

template <cpu_isa_t isa>
void jit_uni_shuffle_kernel_t<isa>::store_data(const int data_idx,
        const XReg &reg_dst_addr, const int offset, const bool is_tail) {
    const auto extend_for_padding
            = is_tail && padding_size_ + conf_.simd_tail >= conf_.simd_w;
    if (extend_for_padding) {
        sel(vmm_tmp_.s, k_tail_mask_, TRegS(data_idx), vmm_zero_.s);
        add_imm(X_DEFAULT_ADDR, reg_dst_addr, offset, X_TMP_0);
        st1w(vmm_tmp_.s, P_ALL_ONE, ptr(X_DEFAULT_ADDR));
    } else {
        if (is_tail) {
            add_imm(X_DEFAULT_ADDR, reg_dst_addr, offset, X_TMP_0);
            st1w(TRegS(data_idx), k_tail_mask_, ptr(X_DEFAULT_ADDR));
        } else {
            add_imm(X_DEFAULT_ADDR, reg_dst_addr, offset, X_TMP_0);
            st1w(TRegS(data_idx), P_ALL_ONE, ptr(X_DEFAULT_ADDR));
        }
    }
    append_zero_padding(
            reg_dst_, isa_sveLen > 128 ? extend_for_padding : false);
}

template <>
void jit_uni_shuffle_kernel_t<asimd>::store_data(const int data_idx,
        const XReg &reg_dst_addr, const int offset, const bool is_tail) {
    if (is_tail) {
        for (unsigned i = 0; i < conf_.simd_tail; i++) {
            add_imm(X_DEFAULT_ADDR, reg_dst_addr, offset + i * conf_.dt_size,
                    X_TMP_0);
            st1(VReg4S(data_idx)[i], ptr(X_DEFAULT_ADDR));
        }
    } else {
        add_imm(X_DEFAULT_ADDR, reg_dst_addr, offset, X_TMP_0);
        str(QReg(data_idx), ptr(X_DEFAULT_ADDR));
    }

    append_zero_padding(reg_dst_, false);
}

template <cpu_isa_t isa>
void jit_uni_shuffle_kernel_t<isa>::shuffle_blocked_format() {
    const XReg &reg_sp = reg_tmp2_;
    const XReg &reg_cb = reg_tmp3_;
    const XReg &reg_cb_loop_size = reg_tmp4_;
    const XReg &reg_blk_tail = reg_tmp5_;
    const XReg &reg_src_save = reg_tmp6_;
    const int simd_in_blk = conf_.blk_size / conf_.simd_w;
    const int simd_in_tail_blk
            = utils::div_up(conf_.c % conf_.blk_size, conf_.simd_w);
    const TReg vmm_tmp[4] = {TReg(5), TReg(6), TReg(7), TReg(8)};

    auto load_indices = ([&](bool is_blk_tail) {
        const int simd_to_process
                = is_blk_tail ? simd_in_tail_blk : simd_in_blk;
        for (int i = 0; i < simd_to_process; ++i) {
            if (is_superset(isa, sve_128))
                ld1w(ZRegS(vmm_tmp[i].getIdx()), P_ALL_ONE,
                        ptr(addr_off(reg_indices_,
                                i * conf_.simd_w * conf_.el_size_of_indices,
                                X_DEFAULT_ADDR, X_TMP_0)));
            else
                uni_ldr(vmm_tmp[i], reg_indices_,
                        i * conf_.simd_w * conf_.el_size_of_indices);
        }
    });

    auto shuffle = ([&](bool is_blk_tail) {
        const int simd_to_process
                = is_blk_tail ? simd_in_tail_blk : simd_in_blk;
        for (int i = 0; i < simd_to_process; ++i) {
            const bool simd_tail_condition = is_blk_tail && conf_.simd_tail > 0
                    && i == simd_to_process - 1;
            uni_orr(TReg(vmm_indices_.getIdx()), vmm_tmp[i], vmm_tmp[i]);
            gather_data(reg_src_, vmm_indices_.getIdx(), vmm_src_.getIdx(),
                    simd_tail_condition);

            store_data(vmm_src_.getIdx(), reg_dst_,
                    i * conf_.simd_w * conf_.dt_size, simd_tail_condition);
        }
    });

    add_imm(X_DEFAULT_ADDR, reg_param, GET_OFF(cb_loop_size), X_TMP_0);
    ldr(reg_cb_loop_size, ptr(X_DEFAULT_ADDR));

    Label sp_loop_begin, sp_loop_end;
    Label sp_tail_loop_begin, sp_tail_loop_end;
    Label cb_loop_begin, cb_loop_end;
    Label simd_loop_begin, simd_loop_end;
    Label blk_tail_loop_begin, blk_tail_loop_end;
    Label blk_tail_check_end;
    Label no_tail;

    eor(reg_blk_tail, reg_blk_tail, reg_blk_tail);

    mov_imm(X_TMP_0, conf_.blk_size);
    cmp(reg_cb_loop_size, X_TMP_0);
    b(EQ, no_tail);

    mov(reg_blk_tail, reg_cb_loop_size);
    eor(reg_cb_loop_size, reg_cb_loop_size, reg_cb_loop_size);

    L(no_tail);

    eor(reg_cb, reg_cb, reg_cb);
    L(cb_loop_begin);
    {
        cmp(reg_cb, reg_cb_loop_size);
        b(EQ, cb_loop_end);

        load_indices(false);

        mov(reg_src_save, reg_src_);

        eor(reg_sp, reg_sp, reg_sp);
        L(sp_loop_begin);
        {
            mov_imm(X_TMP_0, conf_.sp_split_size);
            cmp(reg_sp, X_TMP_0);
            b(EQ, sp_loop_end);

            shuffle(false);

            add_imm(reg_sp, reg_sp, 1, X_TMP_0);
            add_imm(reg_src_, reg_src_, conf_.blk_size * conf_.dt_size,
                    X_TMP_0);
            add_imm(reg_dst_, reg_dst_, conf_.blk_size * conf_.dt_size,
                    X_TMP_0);

            b(sp_loop_begin);
        }
        L(sp_loop_end);

        mov(reg_src_, reg_src_save);

        add_imm(reg_cb, reg_cb, conf_.blk_size, X_TMP_0);
        add_imm(reg_dst_, reg_dst_,
                conf_.blk_size * (conf_.sp - conf_.sp_split_size)
                        * conf_.dt_size,
                X_TMP_0);
        add_imm(reg_indices_, reg_indices_,
                conf_.blk_size * conf_.el_size_of_indices, X_TMP_0);

        b(cb_loop_begin);
    }
    L(cb_loop_end);

    cmp(reg_blk_tail, 0);
    b(EQ, blk_tail_check_end);

    load_indices(true);

    eor(reg_sp, reg_sp, reg_sp);
    L(sp_tail_loop_begin);
    {
        mov_imm(X_TMP_0, conf_.sp_split_size);
        cmp(reg_sp, X_TMP_0);
        b(EQ, sp_tail_loop_end);

        shuffle(true);

        add_imm(reg_sp, reg_sp, 1, X_TMP_0);
        add_imm(reg_src_, reg_src_, conf_.blk_size * conf_.dt_size, X_TMP_0);
        add_imm(reg_dst_, reg_dst_, conf_.blk_size * conf_.dt_size, X_TMP_0);

        b(sp_tail_loop_begin);
    }
    L(sp_tail_loop_end);

    L(blk_tail_check_end);
}

template <cpu_isa_t isa>
void jit_uni_shuffle_kernel_t<isa>::append_zero_padding(
        const XReg &reg_dst_addr, const bool extend_for_padding) {

    static constexpr size_t reg64_size = 8;
    const size_t simd_w_byte = conf_.simd_w * sizeof(float);

    if (!padding_size_) return;

    const auto padding_start
            = (extend_for_padding) ? conf_.simd_w : conf_.c % conf_.blk_size;
    const auto padding_end = (extend_for_padding)
            ? padding_size_ - (conf_.simd_w - conf_.simd_tail)
            : padding_size_;
    const auto off_start = padding_start * conf_.dt_size;
    const auto padding_to_add = padding_end * conf_.dt_size;

    if (!padding_to_add) return;

    Label end;
    unsigned int off = 0;

    cmp(WReg(reg_padded_block.getIdx()), 0);
    b(EQ, end);

    if (simd_w_byte <= padding_to_add) {
        uni_eor(TReg(vmm_zero_.getIdx()), TReg(vmm_zero_.getIdx()),
                TReg(vmm_zero_.getIdx()));

        for (; off + simd_w_byte < padding_to_add; off += simd_w_byte) {
            add_imm(X_DEFAULT_ADDR, reg_dst_addr, off_start + off, X_TMP_0);

            if (is_superset(isa, sve_128))
                st1w(ZRegS(vmm_zero_.getIdx()), P_ALL_ONE, ptr(X_DEFAULT_ADDR));
            else
                uni_str(vmm_zero_, X_DEFAULT_ADDR);
        }
    }

    if (off != padding_to_add) {
        eor(reg_tmp_, reg_tmp_, reg_tmp_);
        for (; off + reg64_size < padding_to_add; off += reg64_size) {
            add_imm(X_DEFAULT_ADDR, reg_dst_addr, off_start + off, X_TMP_0);
            str(reg_tmp_, ptr(X_DEFAULT_ADDR));
        }
        for (; off < padding_to_add; off++) {
            add_imm(X_DEFAULT_ADDR, reg_dst_addr, off_start + off, X_TMP_0);
            strb(WReg(reg_tmp_.getIdx()), ptr(X_DEFAULT_ADDR));
        }
    }

    L(end);
}

template <cpu_isa_t isa>
void jit_uni_shuffle_kernel_t<isa>::generate() {
    preamble();

    /* Overwrite P_ALL_ONE, if required sve size < 512. */
    if (isa_sveLen == util::SVE_128)
        ptrue(P_ALL_ONE.b, VL16);
    else if (isa_sveLen == util::SVE_256)
        ptrue(P_ALL_ONE.b, VL32);

    uni_clear(vmm_zero_);
    prepare_mask();

    add_imm(X_DEFAULT_ADDR, reg_param, GET_OFF(input_off_ptr), X_TMP_0);
    ldr(reg_indices_, ptr(X_DEFAULT_ADDR));

    add_imm(X_DEFAULT_ADDR, reg_param, GET_OFF(src), X_TMP_0);
    ldr(reg_src_, ptr(X_DEFAULT_ADDR));
    add_imm(X_DEFAULT_ADDR, reg_param, GET_OFF(dst), X_TMP_0);
    ldr(reg_dst_, ptr(X_DEFAULT_ADDR));
    add_imm(X_DEFAULT_ADDR, reg_param, GET_OFF(is_padded_block), X_TMP_0);
    ldrb(WReg(reg_padded_block.getIdx()), ptr(X_DEFAULT_ADDR));

    shuffle_blocked_format();

    postamble();
}

template struct jit_uni_shuffle_kernel_t<sve_512>;
template struct jit_uni_shuffle_kernel_t<sve_256>;
template struct jit_uni_shuffle_kernel_t<sve_128>;
template struct jit_uni_shuffle_kernel_t<asimd>;

#undef GET_OFF

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
