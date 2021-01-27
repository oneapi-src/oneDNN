/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#include "cpu/x64/matmul/brgemm_matmul_copy_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace matmul {

using namespace dnnl::impl::format_tag;
using namespace dnnl::impl::utils;
using namespace Xbyak;

#define GET_OFF(x) offsetof(ctx_t, x)

struct jit_brgemm_matmul_copy_A_int8_t : public jit_brgemm_matmul_copy_A_t,
                                         public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_brgemm_matmul_copy_A_int8_t)

    jit_brgemm_matmul_copy_A_int8_t(const brgemm_matmul_conf_t *conf)
        : jit_brgemm_matmul_copy_A_t(conf) {}

    void operator()(ctx_t *ctx) override { jit_generator::operator()(ctx); }
    status_t create_kernel() override { return jit_generator::create_kernel(); }

private:
    using reg64_t = const Xbyak::Reg64;
    using reg32_t = const Xbyak::Reg32;
    using opmask_t = const Xbyak::Opmask;
    using zmm = const Xbyak::Zmm;

    enum { typesize = sizeof(int8_t), k_step = 64 };
    dim_t src_stride = 0, tr_src_stride = 0;

    opmask_t kTail_load = k7;
    opmask_t kTail_store = k6;

    reg64_t reg_src = rax;
    reg64_t reg_tr_src = rbx;

    reg64_t reg_M_blk = r9;
    reg64_t reg_K_blk = r10;
    reg64_t reg_batch = r11;
    reg64_t reg_aux_src = r12;
    reg64_t reg_aux_tr_src = r13;
    reg64_t regq_tmp = r14;
    reg64_t imm_addr64 = r15;

    zmm zmm_comp_add = zmm30;
    zmm zmm_zero = zmm31;

    // Allows to shift A data by 128 for s8s8 problem for AVX512 in copy
    // routine, not in compute kernel. It's disabled for now, as it
    // requires setting some hint to brgemm kerenel to avoid double shifting
    const bool allow_input_shift_for_s8s8 = false;

    void copy_row(int ncolumns);
    void generate() override;
};

void jit_brgemm_matmul_copy_A_int8_t::generate() {
    preamble();
    vpxord(zmm_zero, zmm_zero, zmm_zero);
    src_stride = conf_->K * typesize;
    const dim_t LDA = conf_->use_buffer_a_tail_only ? (dim_t)conf_->wei_k_blk
                                                    : conf_->LDA;
    tr_src_stride = LDA * typesize;

    mov(reg_src, ptr[param1 + GET_OFF(src)]);
    mov(reg_tr_src, ptr[param1 + GET_OFF(tr_src)]);
    mov(reg_K_blk, ptr[param1 + GET_OFF(current_K_blk)]);
    mov(reg_M_blk, ptr[param1 + GET_OFF(current_M_blk)]);

    if (allow_input_shift_for_s8s8 && conf_->signed_input) {
        mov(imm_addr64, 128);
        vpbroadcastb(zmm_comp_add, imm_addr64.cvt8());
    }

    auto copy_K_loop = [=](bool is_K_tail) {
        const int k_unroll = 16;
        const int K_blk = is_K_tail ? conf_->K % conf_->K_blk
                                    : nstl::min(conf_->K, conf_->K_blk);
        const int k_tail = K_blk % k_step;
        const int num_k_iters = K_blk / k_step;
        for (int kb = 0; kb < div_up(num_k_iters, k_unroll); kb++) {
            int k_start = kb * k_unroll;
            int k_end = nstl::min(k_start + k_unroll, num_k_iters);
            for (int k = k_start; k < k_end; k++) {
                vmovdqu8(zmm(k), EVEX_compress_addr(reg_src, k * k_step));
            }
            if (allow_input_shift_for_s8s8 && conf_->signed_input) {
                for (int k = k_start; k < k_end; k++)
                    vpaddb(zmm(k), zmm(k), zmm_comp_add);
            }

            for (int k = k_start; k < k_end; k++) {
                vmovdqu8(EVEX_compress_addr(reg_tr_src, k * k_step), zmm(k));
            }
        }
        if (k_tail > 0) {
            auto kmovq = [=](Opmask k, size_t q) {
                mov(regq_tmp, q);
                jit_generator::kmovq(k, regq_tmp);
            };
            const size_t k_gran
                    = conf_->isa == avx512_core_bf16_amx_int8 ? 4 : 1;
            const size_t tail_mask_load = size_t(((size_t)1 << k_tail) - 1);
            kmovq(kTail_load, tail_mask_load);
            size_t k_tail_st = rnd_up(k_tail, k_gran);
            const size_t tail_mask_store = k_tail_st == k_step
                    ? 0xffffffffffffffff
                    : size_t(((size_t)1 << k_tail_st) - 1);
            kmovq(kTail_store, tail_mask_store);
        }

        if (k_tail > 0) {
            auto zmm_tail = zmm(0) | kTail_load | T_z;
            vmovdqu8(zmm_tail,
                    EVEX_compress_addr(reg_src, num_k_iters * k_step));
            if (allow_input_shift_for_s8s8 && conf_->signed_input)
                vpaddb(zmm(0), zmm(0), zmm_comp_add);

            vmovdqu8(EVEX_compress_addr(reg_tr_src, num_k_iters * k_step),
                    zmm(0) | kTail_store);
        }
    };
    auto copy_M_loop = [=](bool is_K_tail) {
        Label loop_M;
        L(loop_M);

        copy_K_loop(is_K_tail);

        add(reg_src, src_stride);
        add(reg_tr_src, tr_src_stride);
        dec(reg_M_blk);
        jnz(loop_M, T_NEAR);
    };

    Label done;
    // might be different from conf_->K_tail
    const dim_t K_blk_tail = conf_->K_tail > 0 ? conf_->K % conf_->K_blk : 0;
    if (K_blk_tail > 0) {
        Label not_K_tail;
        cmp(reg_K_blk, K_blk_tail);
        jne(not_K_tail, T_NEAR);
        copy_M_loop(true);
        jmp(done, T_NEAR);

        L(not_K_tail);
    }

    copy_M_loop(false);
    L(done);

    postamble();
}

struct jit_brgemm_matmul_copy_B_int8_t : public jit_brgemm_matmul_copy_B_t,
                                         public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_brgemm_matmul_copy_B_int8_t)

    jit_brgemm_matmul_copy_B_int8_t(const brgemm_matmul_conf_t *conf)
        : jit_brgemm_matmul_copy_B_t(conf) {}

    void operator()(ctx_t *ctx) override { jit_generator::operator()(ctx); }
    status_t create_kernel() override { return jit_generator::create_kernel(); }

private:
    using reg64_t = const Xbyak::Reg64;
    using reg32_t = const Xbyak::Reg32;
    using opmask_t = const Xbyak::Opmask;
    using zmm = const Xbyak::Zmm;
    using ymm = const Xbyak::Ymm;

    enum { typesize = sizeof(int8_t), k_blk_step = 4, n_blk_step = 64 };
    dim_t src_stride = 0, tr_src_stride = 0;

    opmask_t kTail = k7;

    reg64_t reg_src = rax;
    reg64_t reg_tr_src = rbx;
    reg64_t reg_comp_ptr = rdx;

    reg64_t reg_K_iters = r8;
    reg64_t reg_N_blk = r9;
    reg64_t reg_K_start = r10;
    reg64_t regq_tmp = r14;
    reg64_t imm_addr64 = r15;

    zmm vreg_idx_lo_256 = zmm26;
    zmm vreg_idx_hi_256 = zmm27;
    zmm vreg_idx_lo_128 = zmm28;
    zmm vreg_idx_hi_128 = zmm29;
    zmm zmm_comp_mul = zmm30;
    zmm zmm_zero = zmm31;

    Xbyak::Zmm get_comp_acc(int i) { return Xbyak::Zmm(25 - i); }

    void copy_4x64_vnni(int nrows, int ncolumns);
    void generate() override;
};

void jit_brgemm_matmul_copy_B_int8_t::copy_4x64_vnni(int nrows, int ncolumns) {
    auto kmovq = [=](Opmask k, size_t q) {
        mov(regq_tmp, q);
        jit_generator::kmovq(k, regq_tmp);
    };

    const auto tail_mask = size_t(((size_t)1 << ncolumns) - 1);
    if (ncolumns < n_blk_step) kmovq(kTail, tail_mask);

    const int blk_sz = 6;
    const int max_unroll = (conf_->signed_input ? 21 : 25) / blk_sz;
    auto get_zmm = [=](int blk, int idx) {
        assert(idx >= 0 && idx < blk_sz && blk >= 0);
        auto reg_idx = blk_sz * blk + idx;
        assert(reg_idx >= 0 && reg_idx < 32);
        return zmm(reg_idx);
    };
    auto load = [=](int blk, int i) {
        auto src_reg = get_zmm(blk, i % k_blk_step);
        auto src_load = ncolumns < n_blk_step ? src_reg | kTail | T_z : src_reg;
        vmovdqu8(src_load, EVEX_compress_addr(reg_src, i * src_stride));
    };

    for_(int kb = 0; kb < div_up(nrows, max_unroll * k_blk_step); kb++)
    for (int k = 0;
            k < nstl::min(max_unroll,
                    div_up(nrows - kb * max_unroll * k_blk_step, k_blk_step));
            k++) {
        const int row_start = (kb * max_unroll + k) * k_blk_step;
        const int row_end = nstl::min(row_start + k_blk_step, nrows);

        for (int i = row_start; i < row_end; i++)
            load(k, i);
        if (row_end == nrows && nrows % k_blk_step > 0) {
            for (int i = nrows; i < rnd_up(nrows, k_blk_step); i++) {
                auto src_reg = get_zmm(k, i % k_blk_step);
                vpxord(src_reg, src_reg, src_reg);
            }
        }

        vpunpcklbw(get_zmm(k, 4), get_zmm(k, 0), get_zmm(k, 1));
        vpunpckhbw(get_zmm(k, 5), get_zmm(k, 0), get_zmm(k, 1));
        vpunpcklbw(get_zmm(k, 0), get_zmm(k, 2), get_zmm(k, 3));
        vpunpckhbw(get_zmm(k, 1), get_zmm(k, 2), get_zmm(k, 3));

        vpunpcklwd(get_zmm(k, 2), get_zmm(k, 4), get_zmm(k, 0));
        vpunpckhwd(get_zmm(k, 3), get_zmm(k, 4), get_zmm(k, 0));
        vpunpcklwd(get_zmm(k, 4), get_zmm(k, 5), get_zmm(k, 1));
        vpunpckhwd(get_zmm(k, 5), get_zmm(k, 5), get_zmm(k, 1));

        vmovups(get_zmm(k, 0), vreg_idx_lo_256);
        vpermi2q(get_zmm(k, 0), get_zmm(k, 2), get_zmm(k, 4));
        vmovups(get_zmm(k, 1), vreg_idx_hi_256);
        vpermi2q(get_zmm(k, 1), get_zmm(k, 2), get_zmm(k, 4));
        vmovups(get_zmm(k, 2), vreg_idx_lo_256);
        vpermi2q(get_zmm(k, 2), get_zmm(k, 3), get_zmm(k, 5));
        vmovups(get_zmm(k, 4), vreg_idx_hi_256);
        vpermi2q(get_zmm(k, 4), get_zmm(k, 3), get_zmm(k, 5));

        vmovups(get_zmm(k, 3), vreg_idx_lo_128);
        vpermi2q(get_zmm(k, 3), get_zmm(k, 0), get_zmm(k, 2));
        dim_t tr_src_off_base = (kb * max_unroll + k) * tr_src_stride;
        vmovups(EVEX_compress_addr(reg_tr_src, tr_src_off_base), get_zmm(k, 3));
        if (conf_->signed_input)
            vpdpbusd(get_comp_acc(0), zmm_comp_mul, get_zmm(k, 3));

        if (ncolumns > 16) {
            vmovups(get_zmm(k, 5), vreg_idx_hi_128);
            vpermi2q(get_zmm(k, 5), get_zmm(k, 0), get_zmm(k, 2));
            vmovups(EVEX_compress_addr(reg_tr_src, tr_src_off_base + 64),
                    get_zmm(k, 5));
            if (conf_->signed_input)
                vpdpbusd(get_comp_acc(1), zmm_comp_mul, get_zmm(k, 5));
        } else if (conf_->wei_n_blk > 16) {
            vmovups(EVEX_compress_addr(reg_tr_src, tr_src_off_base + 64),
                    zmm_zero);
        }

        if (ncolumns > 32) {
            vmovups(get_zmm(k, 0), vreg_idx_lo_128);
            vpermi2q(get_zmm(k, 0), get_zmm(k, 1), get_zmm(k, 4));
            vmovups(EVEX_compress_addr(reg_tr_src, tr_src_off_base + 128),
                    get_zmm(k, 0));
            if (conf_->signed_input)
                vpdpbusd(get_comp_acc(2), zmm_comp_mul, get_zmm(k, 0));
        } else if (conf_->wei_n_blk > 32) {
            vmovups(EVEX_compress_addr(reg_tr_src, tr_src_off_base + 128),
                    zmm_zero);
        }

        if (ncolumns > 48) {
            vmovups(get_zmm(k, 2), vreg_idx_hi_128);
            vpermi2q(get_zmm(k, 2), get_zmm(k, 1), get_zmm(k, 4));
            vmovups(EVEX_compress_addr(reg_tr_src, tr_src_off_base + 192),
                    get_zmm(k, 2));
            if (conf_->signed_input)
                vpdpbusd(get_comp_acc(3), zmm_comp_mul, get_zmm(k, 2));
        } else if (conf_->wei_n_blk > 48) {
            vmovups(EVEX_compress_addr(reg_tr_src, tr_src_off_base + 192),
                    zmm_zero);
        }
    }
}

void jit_brgemm_matmul_copy_B_int8_t::generate() {
    preamble();
    vpxord(zmm_zero, zmm_zero, zmm_zero);
    src_stride = conf_->N * typesize;
    tr_src_stride = conf_->LDB * k_blk_step * typesize;

    mov(reg_src, ptr[param1 + GET_OFF(src)]);
    mov(reg_tr_src, ptr[param1 + GET_OFF(tr_src)]);
    mov(reg_K_iters, ptr[param1 + GET_OFF(current_K_iters)]);
    mov(reg_N_blk, ptr[param1 + GET_OFF(current_N_blk)]);

    auto vmovdqa64 = [=](Zmm z, const int64_t *addr) {
        mov(imm_addr64, reinterpret_cast<size_t>(addr));
        jit_generator::vmovdqa64(z, ptr[imm_addr64]);
    };
    alignas(64) static constexpr const int64_t idx_lo_256[8]
            = {0, 1, 2, 3, 8, 9, 10, 11};
    alignas(64) static constexpr const int64_t idx_hi_256[8]
            = {4, 5, 6, 7, 12, 13, 14, 15};

    alignas(64) static constexpr const int64_t idx_lo_128[8]
            = {0, 1, 8, 9, 4, 5, 12, 13};
    alignas(64) static constexpr const int64_t idx_hi_128[8]
            = {2, 3, 10, 11, 6, 7, 14, 15};

    vmovdqa64(vreg_idx_lo_256, idx_lo_256);
    vmovdqa64(vreg_idx_hi_256, idx_hi_256);
    vmovdqa64(vreg_idx_lo_128, idx_lo_128);
    vmovdqa64(vreg_idx_hi_128, idx_hi_128);

    if (conf_->signed_input) {
        mov(reg_comp_ptr, ptr[param1 + GET_OFF(compensation_ptr)]);
        mov(reg_K_start, ptr[param1 + GET_OFF(current_K_start)]);
        int n_iters = div_up(conf_->wei_n_blk, 16);
        for (int i = 0; i < n_iters; i++)
            vpxord(get_comp_acc(i), get_comp_acc(i), get_comp_acc(i));
        mov(imm_addr64, 1);
        vpbroadcastb(zmm_comp_mul, imm_addr64.cvt8());
    }

    auto compute_K_loop = [=](bool is_N_tail) {
        const int k_unroll = 4;
        int ncolumns = is_N_tail ? conf_->N_tail : conf_->N_blk;

        Label K_loop_unrolled, K_loop_single, K_loop_tail_or_done;
        cmp(reg_K_iters, k_unroll * k_blk_step);
        jl(K_loop_single, T_NEAR);

        L(K_loop_unrolled);
        copy_4x64_vnni(k_unroll * k_blk_step, ncolumns);
        add(reg_src, k_unroll * k_blk_step * src_stride);
        add(reg_tr_src, k_unroll * tr_src_stride);

        sub(reg_K_iters, k_unroll * k_blk_step);
        cmp(reg_K_iters, k_unroll * k_blk_step);
        jge(K_loop_unrolled, T_NEAR);

        L(K_loop_single);
        cmp(reg_K_iters, k_blk_step);
        jl(K_loop_tail_or_done, T_NEAR);

        copy_4x64_vnni(k_blk_step, ncolumns);
        add(reg_src, k_blk_step * src_stride);
        add(reg_tr_src, tr_src_stride);

        sub(reg_K_iters, k_blk_step);
        jmp(K_loop_single, T_NEAR);

        L(K_loop_tail_or_done);

        int k_blk_tail = conf_->K % k_blk_step;
        if (k_blk_tail > 0) {
            Label K_loop_done;
            cmp(reg_K_iters, 0);
            jle(K_loop_done, T_NEAR);

            copy_4x64_vnni(k_blk_tail, ncolumns);
            sub(reg_K_iters, k_blk_tail);
            L(K_loop_done);
        }
    };

    Label done;
    if (conf_->N_tail > 0) {
        Label not_N_tail;
        cmp(reg_N_blk, conf_->N_tail);
        jne(not_N_tail, T_NEAR);
        compute_K_loop(true);
        jmp(done, T_NEAR);

        L(not_N_tail);
    }

    compute_K_loop(false);
    L(done);

    if (conf_->signed_input) {
        Label skip_acc, store;
        mov(reg_comp_ptr, ptr[param1 + GET_OFF(compensation_ptr)]);
        mov(reg_K_start, ptr[param1 + GET_OFF(current_K_start)]);
        cmp(reg_K_start, 0);
        je(skip_acc, T_NEAR);
        int n_iters = div_up(conf_->wei_n_blk, 16);
        for (int i = 0; i < n_iters; i++)
            vpaddd(get_comp_acc(i), get_comp_acc(i),
                    EVEX_compress_addr(reg_comp_ptr, i * 64));
        L(skip_acc);
        cmp(reg_K_start, rnd_up(conf_->K, conf_->K_blk) - conf_->K_blk);
        jl(store, T_NEAR);

        mov(imm_addr64, 0xffffffff);
        vpbroadcastd(zmm_comp_mul, imm_addr64.cvt32());
        mov(imm_addr64, 0x1);
        auto zmm_one = zmm_zero;
        vpbroadcastd(zmm_one, imm_addr64.cvt32());

        for (int i = 0; i < n_iters; i++) {
            // multiply by 128
            vpslld(get_comp_acc(i), get_comp_acc(i), 7);
            // change sign
            vpandnq(get_comp_acc(i), get_comp_acc(i), zmm_comp_mul);
            vpaddd(get_comp_acc(i), get_comp_acc(i), zmm_one);
        }

        L(store);
        for (int i = 0; i < n_iters; i++)
            vmovups(EVEX_compress_addr(reg_comp_ptr, i * 64), get_comp_acc(i));
    }

    postamble();
}

status_t create_brgemm_matmul_copy_B(
        std::unique_ptr<jit_brgemm_matmul_copy_B_t> &copy_ker,
        const brgemm_matmul_conf_t *conf) {
    CHECK(safe_ptr_assign(copy_ker, new jit_brgemm_matmul_copy_B_int8_t(conf)));
    return copy_ker->create_kernel();
}

status_t create_brgemm_matmul_copy_A(
        std::unique_ptr<jit_brgemm_matmul_copy_A_t> &copy_ker,
        const brgemm_matmul_conf_t *conf) {
    CHECK(safe_ptr_assign(copy_ker, new jit_brgemm_matmul_copy_A_int8_t(conf)));
    return copy_ker->create_kernel();
}

} // namespace matmul
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
