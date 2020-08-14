/*******************************************************************************
* Copyright 2020 Intel Corporation
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
#include "common/type_helpers.hpp"
#include "common/utils.hpp"
#include "cpu/x64/jit_generator.hpp"

#include "cpu/x64/jit_brgemm_transpose_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

using namespace dnnl::impl::format_tag;
using namespace dnnl::impl::utils;
using namespace Xbyak;

#define GET_OFF(x) offsetof(ctx_t, x)

struct jit_brgemm_trans_wei_f32_t : public jit_brgemm_trans_wei_t,
                                    public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_brgemm_trans_wei_f32_t)

    jit_brgemm_trans_wei_f32_t(const jit_brgemm_primitive_conf_t *conf)
        : jit_brgemm_trans_wei_t(conf) {}

    void operator()(ctx_t *ctx) override { jit_generator::operator()(ctx); }
    status_t create_kernel() override { return jit_generator::create_kernel(); }

private:
    using reg64_t = const Xbyak::Reg64;
    using reg32_t = const Xbyak::Reg32;
    using opmask_t = const Xbyak::Opmask;

    enum { typesize = sizeof(float), transpose_size = 16 };
    dim_t src_stride, tr_src_stride;

    opmask_t k3333 = k1;
    opmask_t k5555 = k2;
    opmask_t kAAAA = k3;
    opmask_t kCCCC = k4;
    opmask_t k0F0F = k5;
    opmask_t kF0F0 = k6;
    opmask_t kTail = k7;

    reg64_t reg_src = r8;
    reg64_t reg_tr_src = r9;
    reg64_t reg_loop_N = r10;
    reg64_t reg_loop_K = r11;
    reg64_t reg_loop_batch = r12;
    reg64_t reg_tr_src_tmp = r13;
    reg32_t regw_tmp = r14d;

    void transpose_16x16(int nrows, int ncolumns = transpose_size);
    void generate() override;
};

void jit_brgemm_trans_wei_f32_t::transpose_16x16(int nrows, int ncolumns) {
    assert(nrows >= 0 && nrows <= transpose_size);
    static_assert(transpose_size == 16, "Unsupported transpose size");
    if (!nrows) return;

    auto src_zmm = [=](int i) {
        assert(i >= 0 && i < 16);
        return Zmm(i);
    };

    auto tmp_zmm = [=](int i) {
        assert(i >= 0 && i < 16);
        return Zmm(16 + i);
    };

    auto kmovw = [=](Opmask k, unsigned w) {
        mov(regw_tmp, w);
        jit_generator::kmovw(k, regw_tmp);
    };

    auto load = [=](int i) {
        auto src_load = src_zmm(i);
        if (ncolumns < transpose_size) {
            kmovw(kTail, (1 << ncolumns) - 1);
            src_load = src_zmm(i) | kTail | T_z;
        }
        vmovups(src_load, EVEX_compress_addr(reg_src, i * src_stride));
    };

    auto store = [=](Zmm r, int i) {
        mov(reg_tr_src_tmp, reg_tr_src);
        if (nrows < transpose_size) kmovw(kTail, (1 << nrows) - 1);

        // Xbyak does not allow k0 to be specified explicitly via the '|'
        // operator, so we have to do this via a method call (implicitly
        // EVEX encoding uses k0 to mean 'no mask')
        bool partial_store = nrows < transpose_size;
        auto k = partial_store ? kTail : k0;
        auto base = reg_tr_src_tmp;
        base.setOpmaskIdx(k.getIdx(), true);

        auto addr = EVEX_compress_addr(base, i * tr_src_stride);
        vmovups(addr, r);
    };

    auto transpose16x8 = [=](int base_idx) {
        assert(base_idx == 0 || base_idx == 8);

        // swap 1
        for (int i = 0; i < 4; i++) {
            int src_idx0 = base_idx + i * 2;
            int src_idx1 = src_idx0 + 1;

            int next_src_idx0 = src_idx0 + 2;
            int next_src_idx1 = src_idx1 + 2;
            bool load_next = base_idx == 0 || i < 3;

            if (base_idx == 0 && i == 0) {
                load(src_idx0);
                if (src_idx1 < nrows)
                    load(src_idx1);
                else
                    vpxord(src_zmm(src_idx1), src_zmm(src_idx1),
                            src_zmm(src_idx1));
            }

            auto tmp0 = tmp_zmm(src_idx0);
            auto tmp1 = tmp_zmm(src_idx1);
            auto src0 = src_zmm(src_idx0);
            auto src1 = src_zmm(src_idx1);

            if (next_src_idx0 < nrows && load_next) load(next_src_idx0);
            valignd(tmp0, src0, src0, 0x1);

            if (next_src_idx1 < nrows && load_next) load(next_src_idx1);
            valignd(tmp1, src1, src1, 0xf);

            vmovaps(src0 | kAAAA, tmp1);
            vmovaps(src1 | k5555, tmp0);
        }
        // swap 2
        for (int i = 0; i < 4; i++) {
            int select_half = (i < 2) ? 0 : 2;
            int src_idx0 = base_idx + i + select_half + 0;
            int src_idx2 = src_idx0 + 2;

            auto tmp0 = tmp_zmm(src_idx0);
            auto tmp1 = tmp_zmm(src_idx2);
            auto src0 = src_zmm(src_idx0);
            auto src2 = src_zmm(src_idx2);

            valignd(tmp0, src0, src0, 0x2);
            valignd(tmp1, src2, src2, 0xe);
            vmovaps(src2 | k3333, tmp0);
            vmovaps(src0 | kCCCC, tmp1);
        }

        // swap 4
        for (int i = 0; i < 4; i++) {
            int src_idx0 = base_idx + i;
            int src_idx4 = src_idx0 + 4;

            auto tmp0 = tmp_zmm(src_idx0);
            auto src0 = src_zmm(src_idx0);
            auto src4 = src_zmm(src_idx4);

            vmovaps(tmp0, src0);
            vshuff32x4(src0 | kF0F0, src4, src4, 0xb1);
            vshuff32x4(src4 | k0F0F, tmp0, tmp0, 0xb1);
        }
    };

    auto fixup16x16 = [=]() {
        // swap 8
        for (int i = 0; i < 8; i++) {
            auto tmp = tmp_zmm(i);
            auto src0 = src_zmm(i);
            auto src8 = src_zmm(8 + i);
            vshuff64x2(tmp, src0, src8, 0x44);
            store(tmp, i);
        }

        for (int i = 0; i < 8; i++) {
            auto tmp = tmp_zmm(8 + i);
            auto src0 = src_zmm(i);
            auto src8 = src_zmm(8 + i);
            vshuff64x2(tmp, src0, src8, 0xee);
            store(tmp, 8 + i);
        }
    };

    transpose16x8(0);
    transpose16x8(8);
    fixup16x16();
}

void jit_brgemm_trans_wei_f32_t::generate() {
    preamble();
    assert(transpose_size == conf_->oc_block);
    int fwd_ic_block = conf_->simd_w;
    int fwd_oc_block = 0;
    switch (conf_->wei_tag) {
        case OI16i64o:
        case OI8i64o2i: fwd_oc_block = 4 * conf_->simd_w; break;
        case OI16i32o:
        case OI8i32o2i: fwd_oc_block = 2 * conf_->simd_w; break;
        default: fwd_oc_block = conf_->simd_w;
    };

    int oc_tail = conf_->K_tail % transpose_size;
    int ic_block = conf_->ic_block;
    int ic_tail = conf_->N_tail % transpose_size;
    src_stride = fwd_oc_block * typesize;
    tr_src_stride = ic_block * typesize;
    dim_t N_src_shift = conf_->kd * conf_->kh * conf_->kw * fwd_ic_block
            * fwd_oc_block * typesize;
    dim_t N_tr_src_shift = conf_->simd_w * typesize;

    mov(reg_src, ptr[param1 + GET_OFF(src)]);
    mov(reg_tr_src, ptr[param1 + GET_OFF(tr_src)]);
    mov(reg_loop_batch, ptr[param1 + GET_OFF(current_gemm_batch)]);
    mov(reg_loop_K, ptr[param1 + GET_OFF(current_K)]);
    mov(reg_loop_N, ptr[param1 + GET_OFF(current_N)]);

    auto kmovw = [=](Opmask k, unsigned w) {
        mov(regw_tmp, w);
        jit_generator::kmovw(k, regw_tmp);
    };

    kmovw(k3333, 0x3333); // 0011001100110011
    kmovw(k5555, 0x5555); // 0101010101010101
    kmovw(kAAAA, 0xaaaa); // 1010101010101010
    kmovw(kCCCC, 0xcccc); // 1100110011001100
    kmovw(k0F0F, 0x0f0f); // 0000111100001111
    kmovw(kF0F0, 0xf0f0); // 1111000011110000

    auto compute_N = [=](bool is_oc_tail) {
        Label N_loop, N_loop_tail;

        cmp(reg_loop_N, transpose_size);
        jl(N_loop_tail, T_NEAR);

        L(N_loop);

        transpose_16x16(transpose_size, is_oc_tail ? oc_tail : transpose_size);
        add(reg_src, N_src_shift);
        add(reg_tr_src, N_tr_src_shift);

        sub(reg_loop_N, transpose_size);
        cmp(reg_loop_N, transpose_size);
        jge(N_loop, T_NEAR);

        L(N_loop_tail);
        if (ic_tail > 0) {
            Label N_loop_done;
            cmp(reg_loop_N, 0);
            jle(N_loop_done, T_NEAR);
            transpose_16x16(ic_tail, is_oc_tail ? oc_tail : transpose_size);
            L(N_loop_done);
        }
    };

    Label K_tail;
    if (oc_tail > 0) {
        cmp(reg_loop_K, transpose_size);
        jl(K_tail, T_NEAR);
    }

    compute_N(false);

    if (oc_tail > 0) {
        Label done;
        jmp(done, T_NEAR);

        L(K_tail);
        compute_N(true);
        L(done);
    }

    postamble();
}

struct jit_brgemm_trans_wei_bf16_t : public jit_brgemm_trans_wei_t,
                                     public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_brgemm_trans_wei_bf16_t)

    jit_brgemm_trans_wei_bf16_t(const jit_brgemm_primitive_conf_t *conf)
        : jit_brgemm_trans_wei_t(conf) {}

    void operator()(ctx_t *ctx) override { jit_generator::operator()(ctx); }
    status_t create_kernel() override { return jit_generator::create_kernel(); }

private:
    using reg64_t = const Xbyak::Reg64;
    using reg32_t = const Xbyak::Reg32;
    using opmask_t = const Xbyak::Opmask;
    using zmm = const Xbyak::Zmm;

    enum { typesize = sizeof(int16_t), transpose_size = 16 };
    dim_t src_stride, tr_src_stride;

    opmask_t kTail = k7;

    reg64_t reg_src_base = rax;
    reg64_t reg_tr_src_base = rbx;

    reg64_t reg_src = r8;
    reg64_t reg_tr_src = r9;
    reg64_t reg_loop_N = r10;
    reg64_t reg_loop_K = r11;
    reg64_t reg_loop_batch = r12;
    reg64_t reg_tr_src_tmp = r13;
    reg32_t regw_tmp = r14d;
    reg64_t imm_addr64 = r15;

    zmm v_abcdefgh_to_abefcdgh = zmm31;

    void transpose_16x16_vnni(int nrows, int ncolumns = transpose_size);
    void generate() override;
};

void jit_brgemm_trans_wei_bf16_t::transpose_16x16_vnni(
        int nrows, int ncolumns) {
    assert(nrows >= 0 && nrows <= transpose_size);
    static_assert(transpose_size == 16, "Unsupported transpose size");
    if (!nrows) return;

    auto src_zmm = [=](int i) {
        assert(i >= 0 && i < 8);
        return Zmm(i);
    };

    auto tmp_zmm = [=](int i) {
        assert(i >= 0 && i < 8);
        return Zmm(8 + i);
    };

    auto kmovw = [=](Opmask k, unsigned w) {
        mov(regw_tmp, w);
        jit_generator::kmovw(k, regw_tmp);
    };

    auto load = [=](int i) {
        auto src_load = src_zmm(i);
        if (ncolumns < transpose_size) {
            kmovw(kTail, (1 << ncolumns) - 1);
            src_load = src_zmm(i) | kTail | T_z;
        }
        vmovups(src_load, EVEX_compress_addr(reg_src, i * src_stride));
    };

    auto store = [=](Zmm r, int i) {
        mov(reg_tr_src_tmp, reg_tr_src);
        if (nrows < transpose_size) kmovw(kTail, (1 << nrows) - 1);

        // Xbyak does not allow k0 to be specified explicitly via the '|'
        // operator, so we have to do this via a method call (implicitly
        // EVEX encoding uses k0 to mean 'no mask')
        bool partial_store = nrows < transpose_size;
        auto k = partial_store ? kTail : k0;
        auto base = reg_tr_src_tmp;
        base.setOpmaskIdx(k.getIdx(), true);

        auto addr = EVEX_compress_addr(base, i * tr_src_stride);
        vmovups(addr, r);
    };

    for (int i = 0; i < 8; i++)
        load(i);

    for (int i = 0; i < 8; i++)
        vpshufb(src_zmm(i), src_zmm(i), v_abcdefgh_to_abefcdgh);

    for (int i = 0; i < 2; i++) {
        vpunpcklqdq(tmp_zmm(2 * i + 0), src_zmm(2 * i), src_zmm(2 * i + 1));
        vpunpckhqdq(tmp_zmm(2 * i + 1), src_zmm(2 * i), src_zmm(2 * i + 1));
    }

    for (int i = 0; i < 2; i++) {
        vpunpcklqdq(
                src_zmm(2 * i + 0), src_zmm(4 + 2 * i), src_zmm(4 + 2 * i + 1));
        vpunpckhqdq(
                src_zmm(2 * i + 1), src_zmm(4 + 2 * i), src_zmm(4 + 2 * i + 1));
    }

    for (int i = 0; i < 2; i++) {
        vshufi32x4(src_zmm(4 + 0 + i), tmp_zmm(i), tmp_zmm(2 + i), 0x88);
        vshufi32x4(src_zmm(4 + 2 + i), tmp_zmm(i), tmp_zmm(2 + i), 0xdd);
    }

    for (int i = 0; i < 2; i++) {
        vshufi32x4(tmp_zmm(0 + i), src_zmm(i), src_zmm(2 + i), 0x88);
        vshufi32x4(tmp_zmm(2 + i), src_zmm(i), src_zmm(2 + i), 0xdd);
    }

    for (int i = 0; i < 4; i++)
        vshufi32x4(src_zmm(i), src_zmm(4 + i), tmp_zmm(i), 0x88);

    for (int i = 0; i < 4; i++)
        vshufi32x4(src_zmm(4 + i), src_zmm(4 + i), tmp_zmm(i), 0xdd);

    for (int i = 0; i < 8; i++)
        store(src_zmm(i), i);
}

void jit_brgemm_trans_wei_bf16_t::generate() {
    preamble();
    assert(transpose_size == conf_->oc_block);
    int fwd_oc_block = 0;
    switch (conf_->wei_tag) {
        case OI16i64o:
        case OI8i64o2i: fwd_oc_block = 4 * conf_->simd_w; break;
        case OI16i32o:
        case OI8i32o2i: fwd_oc_block = 2 * conf_->simd_w; break;
        default: fwd_oc_block = conf_->simd_w;
    };

    int oc_tail = conf_->K_tail % transpose_size;
    int ic_block = conf_->ic_block;
    int ic_tail = conf_->N_tail % transpose_size;
    src_stride = 2 * fwd_oc_block * typesize;
    tr_src_stride = 2 * ic_block * typesize;
    dim_t N_src_shift = conf_->simd_w * fwd_oc_block * typesize;
    dim_t N_tr_src_shift = 2 * conf_->simd_w * typesize;
    dim_t K_src_shift = 2 * conf_->simd_w * typesize;
    dim_t K_tr_src_shift = conf_->ic_block * conf_->simd_w * typesize;

    mov(reg_src_base, ptr[param1 + GET_OFF(src)]);
    mov(reg_tr_src_base, ptr[param1 + GET_OFF(tr_src)]);
    mov(reg_loop_batch, ptr[param1 + GET_OFF(current_gemm_batch)]);
    mov(reg_loop_K, ptr[param1 + GET_OFF(current_K)]);

    alignas(64) static constexpr const int32_t abcdefgh_to_abefcdgh[16]
            = {0x05040100, 0x07060302, 0x0d0c0908, 0x0f0e0b0a, 0x05040100,
                    0x07060302, 0x0d0c0908, 0x0f0e0b0a, 0x05040100, 0x07060302,
                    0x0d0c0908, 0x0f0e0b0a, 0x05040100, 0x07060302, 0x0d0c0908,
                    0x0f0e0b0a};

    auto vmovdqa64 = [=](Zmm z, const int64_t *addr) {
        mov(imm_addr64, reinterpret_cast<size_t>(addr));
        jit_generator::vmovdqa64(z, ptr[imm_addr64]);
    };

    vmovdqa64(v_abcdefgh_to_abefcdgh, (const int64_t *)abcdefgh_to_abefcdgh);
    auto compute_N = [=](bool is_oc_tail) {
        mov(reg_src, reg_src_base);
        mov(reg_tr_src, reg_tr_src_base);
        mov(reg_loop_N, ptr[param1 + GET_OFF(current_N)]);

        Label N_loop, N_loop_tail;
        cmp(reg_loop_N, transpose_size);
        jl(N_loop_tail, T_NEAR);

        L(N_loop);

        transpose_16x16_vnni(
                transpose_size, is_oc_tail ? oc_tail : transpose_size);
        add(reg_src, N_src_shift);
        add(reg_tr_src, N_tr_src_shift);

        sub(reg_loop_N, transpose_size);
        cmp(reg_loop_N, transpose_size);
        jge(N_loop, T_NEAR);

        L(N_loop_tail);
        if (ic_tail > 0) {
            Label N_loop_done;
            cmp(reg_loop_N, 0);
            jle(N_loop_done, T_NEAR);
            transpose_16x16_vnni(
                    ic_tail, is_oc_tail ? oc_tail : transpose_size);
            L(N_loop_done);
        }
    };

    Label K_loop, K_tail;
    if (oc_tail > 0) {
        cmp(reg_loop_K, transpose_size);
        jl(K_tail, T_NEAR);
    }

    L(K_loop);
    compute_N(false);
    add(reg_src_base, K_src_shift);
    add(reg_tr_src_base, K_tr_src_shift);

    sub(reg_loop_K, transpose_size);
    cmp(reg_loop_K, transpose_size);
    jge(K_loop, T_NEAR);

    if (oc_tail > 0) {
        Label done;
        jmp(done, T_NEAR);

        L(K_tail);
        compute_N(true);
        L(done);
    }

    postamble();
}

status_t create_brgemm_trans_wei(
        std::unique_ptr<jit_brgemm_trans_wei_t> &trans_ker,
        const jit_brgemm_primitive_conf_t *conf) {
    if (conf->prop_kind == dnnl_backward_data && conf->wei_dt == data_type::f32)
        CHECK(safe_ptr_assign(trans_ker, new jit_brgemm_trans_wei_f32_t(conf)));
    else if (conf->prop_kind == dnnl_backward_data
            && conf->wei_dt == data_type::bf16)
        CHECK(safe_ptr_assign(
                trans_ker, new jit_brgemm_trans_wei_bf16_t(conf)));
    else
        return status::invalid_arguments;

    return trans_ker->create_kernel();
}

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
