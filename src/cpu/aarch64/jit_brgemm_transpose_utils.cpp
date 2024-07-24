/*******************************************************************************
* Copyright 2020-2023 Intel Corporation
* Copyright 2024 FUJITSU LIMITED
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
#include "common/math_utils.hpp"
#include "common/nstl.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"
#include "cpu/aarch64/jit_generator.hpp"

#include "cpu/aarch64/jit_brgemm_transpose_utils.hpp"

#define LD_MUL_VL(mn, op, mask, addr, off, size) \
    { \
        const int mul_vl_len = (cpu_sveLen / 4) * size; \
        const int off_mod = off % mul_vl_len; \
        const int off_mul_vl = off / mul_vl_len; \
        if (off_mod == 0 && -8 <= off_mul_vl && off_mul_vl <= 7) \
            mn(op, mask / T_z, ptr(addr, off_mul_vl, MUL_VL)); \
        else \
            mn(op, mask / T_z, \
                    ptr(addr_off(addr, off, X_DEFAULT_ADDR, X_TMP_0))); \
    }

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

using namespace dnnl::impl::format_tag;
using namespace dnnl::impl::utils;
using namespace Xbyak_aarch64;

#define GET_OFF(x) offsetof(ctx_t, x)

struct jit_brgemm_trans_m_k_f32_t : public jit_brgemm_trans_src_t,
                                    public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_brgemm_trans_m_k_f32_t)

    jit_brgemm_trans_m_k_f32_t(const jit_brgemm_primitive_conf_t *conf)
        : jit_brgemm_trans_src_t(conf) {}

    void operator()(ctx_t *ctx) override { jit_generator::operator()(ctx); }
    status_t create_kernel() override { return jit_generator::create_kernel(); }

private:
    enum { typesize = sizeof(float), transpose_size = 16 };
    dim_t src_stride = 0, tr_src_stride = 0;

    const PReg k3333 = p1;
    const PReg k5555 = p2;
    const PReg kAAAA = p3;
    const PReg kCCCC = p4;
    const PReg k0F0F = p5;
    const PReg kF0F0 = p6;
    const PReg kTail = p7;

    const XReg reg_src_base = x0;
    const XReg reg_tr_src_base = x3;

    const XReg reg_src = x8;
    const XReg reg_tr_src = x9;
    const XReg reg_loop_K = x10;
    const XReg reg_loop_M = x11;
    const XReg reg_loop_batch = x12;
    const XReg reg_tr_src_tmp = x13;
    const WReg regw_tmp = w14;

    void transpose_16x16(int nrows, int ncolumns = transpose_size);
    void generate() override;
};

void jit_brgemm_trans_m_k_f32_t::transpose_16x16(int nrows, int ncolumns) {
    assert(!"unsupported\n");
}

void jit_brgemm_trans_m_k_f32_t::generate() {
    preamble();
    assert(conf_->ic_block % transpose_size == 0);
    const int os_block = conf_->os_block;
    const int last_os_block_tail = conf_->K_tail % transpose_size;
    const int ic_tail = conf_->M_tail % transpose_size;
    src_stride = conf_->ic * typesize;
    tr_src_stride = conf_->LDA * typesize;
    const dim_t m_src_shift = transpose_size * typesize;
    const dim_t m_tr_src_shift = tr_src_stride * transpose_size;

    const dim_t batch_src_shift = src_stride * os_block;
    const dim_t batch_tr_src_shift = tr_src_stride * conf_->M;

    add_imm(X_DEFAULT_ADDR, param1, GET_OFF(src), X_TMP_0);
    ldr(reg_src_base, ptr(X_DEFAULT_ADDR));
    add_imm(X_DEFAULT_ADDR, param1, GET_OFF(tr_src), X_TMP_0);
    ldr(reg_tr_src_base, ptr(X_DEFAULT_ADDR));
    add_imm(X_DEFAULT_ADDR, param1, GET_OFF(current_gemm_batch), X_TMP_0);
    ldr(reg_loop_batch, ptr(X_DEFAULT_ADDR));
    add_imm(X_DEFAULT_ADDR, param1, GET_OFF(current_K), X_TMP_0);
    ldr(reg_loop_K, ptr(X_DEFAULT_ADDR));

    auto kmovw = [=](PReg k, unsigned w) {
        mov_imm(regw_tmp, w);
        set_preg(k.h, w, X_TMP_0, X_TMP_1);
    };

    kmovw(k3333, 0x3333); // 0011001100110011
    kmovw(k5555, 0x5555); // 0101010101010101
    kmovw(kAAAA, 0xaaaa); // 1010101010101010
    kmovw(kCCCC, 0xcccc); // 1100110011001100
    kmovw(k0F0F, 0x0f0f); // 0000111100001111
    kmovw(kF0F0, 0xf0f0); // 1111000011110000

    auto compute_M = [=](bool is_os_tail) {
        const auto nrows = is_os_tail ? last_os_block_tail : transpose_size;
        add_imm(X_DEFAULT_ADDR, param1, GET_OFF(current_M), X_TMP_0);
        ldr(reg_loop_M, ptr(X_DEFAULT_ADDR));
        mov(reg_src, reg_src_base);
        mov(reg_tr_src, reg_tr_src_base);
        Label M_loop, M_tail_or_done, M_done;
        if (ic_tail > 0) {
            cmp_imm(reg_loop_M, transpose_size, X_TMP_0);
            b(LT, M_tail_or_done);
        }

        L(M_loop);
        transpose_16x16(nrows, transpose_size);
        if (conf_->ic_block > transpose_size) {
            add_imm(reg_src, reg_src, m_src_shift, X_TMP_0);
            add_imm(reg_tr_src, reg_tr_src, m_tr_src_shift, X_TMP_1);
            sub_imm(reg_loop_M, reg_loop_M, (int)transpose_size, X_TMP_0);
            cmp_imm(reg_loop_M, transpose_size, X_TMP_1);
            b(GE, M_loop);
        } else {
            b(M_done);
        }

        L(M_tail_or_done);
        if (ic_tail > 0) {
            cmp_imm(reg_loop_M, 0, X_TMP_0);
            b(LE, M_done);
            transpose_16x16(nrows, ic_tail);
        }
        L(M_done);
    };

    auto compute_batch = [=](bool is_os_tail) {
        Label batch_loop;
        L(batch_loop);

        compute_M(is_os_tail);
        add_imm(reg_src_base, reg_src_base, batch_src_shift, X_TMP_0);
        add_imm(reg_tr_src_base, reg_tr_src_base, batch_tr_src_shift, X_TMP_1);

        sub_imm(reg_loop_batch, reg_loop_batch, 1, X_TMP_0);
        b(NE, batch_loop);
    };

    Label K_tail;
    if (last_os_block_tail > 0) {
        cmp_imm(reg_loop_K, transpose_size, X_TMP_0);
        b(LT, K_tail);
    }

    compute_batch(false);

    if (last_os_block_tail > 0) {
        Label K_done;
        b(K_done);

        L(K_tail);
        compute_batch(true);
        L(K_done);
    }

    postamble();
}

struct jit_brgemm_trans_m_k_bf16_t : public jit_brgemm_trans_src_t,
                                     public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_brgemm_trans_m_k_bf16_t)
    jit_brgemm_trans_m_k_bf16_t(const jit_brgemm_primitive_conf_t *conf)
        : jit_brgemm_trans_src_t(conf) {}

    void operator()(ctx_t *ctx) override { jit_generator::operator()(ctx); }
    status_t create_kernel() override { return jit_generator::create_kernel(); }

private:
    enum {
        typesize = sizeof(int16_t),
        transpose_size = 16,
    };

    const PReg kFFFF = p1;
    const PReg k5555 = p2;
    const PReg kAAAA = p3;
    const PReg kAA = p4;
    const PReg k55 = p5;
    const PReg kCC = p6;
    const PReg k33 = p7;
    const PReg kTail = p1;

    const WReg regw_tmp = w15;

    const XReg reg_k_src = x14;
    const XReg reg_k_tr_src = x13;

    const XReg reg_m_src = x12;
    const XReg reg_m_tr_src = x11;

    const XReg reg_batch_src = x10;
    const XReg reg_batch_tr_src = x9;

    const XReg reg_loop_batch = x8;
    const XReg reg_loop_K = x0;
    const XReg reg_loop_M = x3;

    const XReg reg_tr_src_tmp = x1; // lnx -> rcx
    const XReg imm_addr64 = x2;

    ZReg vidx1 = z31;
    ZReg vidx2 = z30;
    ZReg vidx3 = z29;
    ZReg vidx4 = z28;
    ZReg vidx5 = z27;
    ZReg zmm_tmp = z26;

    void transpose(const XReg dst, const XReg src, int nrows,
            int ncolumns = transpose_size);
    void generate() override;
};

void jit_brgemm_trans_m_k_bf16_t::transpose(
        XReg dst, XReg src, int nrows, int ncolumns) {
    assert(!"unsupported\n");
}

void jit_brgemm_trans_m_k_bf16_t::generate() {
    assert(!"unsupported\n");
}

void jit_brgemm_copy_to_coarse_t::copy_row_blks(int num_row_blks) {
    assert(!"unsupported\n");
}

void jit_brgemm_copy_to_coarse_t::copy_row_tail(
        bool is_last_iteration, int row_offset) {
    assert(!"unsupported\n");
}

void jit_brgemm_copy_to_coarse_t::zero_out_rows() {
    assert(!"unsupported\n");
}

void jit_brgemm_copy_to_coarse_t::copy_row_loop() {
    Label label_row_tail, label_row_exit;

    // Note: copying is done in chunks of size row_step_
    const auto copy_row = [&](bool is_last_iteration) {
        const int row_blk
                = is_last_iteration ? (row_size_ % tr_row_size_) : tr_row_size_;
        const int row_iters = row_blk / row_step_;
        const int row_iters_tail = row_blk % row_step_;

        copy_row_blks(row_iters);
        if (row_iters_tail != 0)
            copy_row_tail(is_last_iteration, /* row_offset = */ row_iters);

        // For the last iteration, zero-out rows if needed
        if (is_last_iteration) zero_out_rows();
    };

    const bool only_row_tail = row_size_ < tr_row_size_;

    if (!only_row_tail) {
        cmp_imm(reg_last_row_blk, 0, X_TMP_0);
        b(NE, label_row_tail);

        copy_row(/* is_last_iteration = */ false);
        b(label_row_exit);
    }

    L(label_row_tail);
    copy_row(/* is_last_iteration = */ true);

    L(label_row_exit);
}

void jit_brgemm_copy_to_coarse_t::copy_os_loop() {

    Label loop_os;
    L(loop_os);

    copy_row_loop();
    add_imm(reg_data, reg_data, data_stride_, X_TMP_0);
    add_imm(reg_tr_data, reg_tr_data, tr_data_stride_, X_TMP_1);

    subs(reg_os_work, reg_os_work, 1);
    cmp_imm(reg_os_work, 0, X_TMP_0);
    b(NE, loop_os);
}

void jit_brgemm_copy_to_coarse_t::set_last_row_tail_masks() {
    const int row_tail = (row_size_ % tr_row_size_) % row_step_;
    assert(row_tail > 0 && "kernel is meant to be used with tail processing");

    // Set load mask
    set_preg(
            reg_m_last_row_tail_load.d, typesize_ * row_tail, X_TMP_0, X_TMP_1);

    // Caution: Since size of ZMM equals 64 bytes therefore we need
    // different masks to store tails with smaller row_block_size_
    constexpr auto full_mask = size_t {0xffffffffffffffff};
    constexpr auto half_mask = size_t {0x00000000ffffffff};
    constexpr auto quad_mask = size_t {0x000000000000ffff};

    const auto num_bytes = [](size_t mask) -> int {
        // Given by 1 + position of leftmost 1 bit
        return 1 + math::ilog2q(mask);
    };

    const int row_tail_store_size
            = utils::rnd_up(row_tail, row_block_size_) * typesize_;
    if (row_tail_store_size >= num_bytes(full_mask)) {
        ptrue(reg_m_last_row_tail_load.b);
    } else if (row_tail_store_size >= num_bytes(half_mask))
        set_preg(reg_m_last_row_tail_load.b, half_mask, X_TMP_0, X_TMP_1);
    else {
        assert(row_tail_store_size == num_bytes(quad_mask));
        set_preg(reg_m_last_row_tail_load.b, quad_mask, X_TMP_0, X_TMP_1);
    }
}

void jit_brgemm_copy_to_coarse_t::set_full_row_tail_masks() {
    const auto full_row_tail = tr_row_size_ % row_step_;
    assert(row_step_ == 2 * full_row_tail || row_step_ == 4 * full_row_tail);

    const auto tail_mask = row_step_ == 2 * full_row_tail
            ? size_t {0x00000000ffffffff}
            : size_t {0x000000000000ffff};

    set_preg(reg_m_full_row_tail_store.b, tail_mask, X_TMP_0, X_TMP_1);
    set_preg(reg_m_full_row_tail_load.b, tail_mask, X_TMP_0, X_TMP_1);
}

void jit_brgemm_copy_to_coarse_t::generate() {
    preamble();

    // set up masks for tail processing
    set_last_row_tail_masks();
    const bool has_full_row_tail_ = tr_row_size_ % row_step_ != 0;
    if (has_full_row_tail_) set_full_row_tail_masks();

    // init zero vreg (zmm_zero) if it is needed
    const int last_row_size
            = utils::rnd_up(row_size_ % tr_row_size_, row_step_);
    const bool zero_iters_needed
            = last_row_size > 0 && last_row_size < tr_row_size_;
    if (zero_iters_needed) eor(zmm_zero.d, zmm_zero.d, zmm_zero.d);

    // load arguments to the jit kernel
    add_imm(X_DEFAULT_ADDR, param1, GET_OFF(data), X_TMP_0);
    ldr(reg_data, ptr(X_DEFAULT_ADDR));
    add_imm(X_DEFAULT_ADDR, param1, GET_OFF(tr_data), X_TMP_0);
    ldr(reg_tr_data, ptr(X_DEFAULT_ADDR));
    add_imm(X_DEFAULT_ADDR, param1, GET_OFF(os_work), X_TMP_0);
    ldr(reg_os_work, ptr(X_DEFAULT_ADDR));
    add_imm(X_DEFAULT_ADDR, param1, GET_OFF(last_row_blk), X_TMP_0);
    ldr(reg_last_row_blk, ptr(X_DEFAULT_ADDR));

    // enter the `main` loop
    copy_os_loop();

    postamble();
}

struct jit_trans_to_vnni_t : public jit_brgemm_trans_to_vnni_t,
                             public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_trans_to_vnni_t)
    jit_trans_to_vnni_t(const jit_brgemm_primitive_conf_t *conf,
            jit_brgemm_trans_to_vnni_t::matrix_to_transform_t
                    matrix_to_transform)
        : jit_brgemm_trans_to_vnni_t(conf, matrix_to_transform) {}

    void operator()(ctx_t *ctx) override { jit_generator::operator()(ctx); }
    status_t create_kernel() override { return jit_generator::create_kernel(); }

private:
    enum {
        typesize_data = sizeof(int16_t),
        typesize_acc = sizeof(float),
        transpose_size = 16,
    };

    PReg kFFFF = p1;
    PReg mask_tail = p2;

    ZReg vidx1 = z31;

    WReg regw_tmp = w15;

    XReg reg_batch_src = x14;
    XReg reg_batch_tr_src = x13;

    XReg reg_row_src = x12;
    XReg reg_row_tr_src = x11;

    XReg reg_col_src = x10;
    XReg reg_col_tr_src = x9;

    XReg reg_loop_batch = x8;
    XReg reg_loop_row = x0;
    XReg reg_loop_col = x3;

    XReg imm_addr64 = x1; // lnx -> rcx

    void maybe_zero_pad_col(XReg dst);
    void transpose(XReg dst, XReg src, int nrows, int ncolumns = transpose_size,
            bool pad_by_zeroes = false);
    void generate() override;
};

void jit_trans_to_vnni_t::maybe_zero_pad_col(XReg dst) {
    assert(!"unsupported\n");
}

void jit_trans_to_vnni_t::transpose(
        XReg dst, XReg src, int nrows, int ncolumns, bool pad_by_zeroes) {
    assert(!"unsupported\n");
}

void jit_trans_to_vnni_t::generate() {
    assert(!"unsupported\n");
}

struct jit_copy_f32_t : public jit_brgemm_trans_to_vnni_t,
                        public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_copy_f32_t)
    jit_copy_f32_t(const jit_brgemm_primitive_conf_t *conf,
            jit_brgemm_trans_to_vnni_t::matrix_to_transform_t
                    matrix_to_transform)
        : jit_brgemm_trans_to_vnni_t(conf, matrix_to_transform) {}

    void operator()(ctx_t *ctx) override { jit_generator::operator()(ctx); }
    status_t create_kernel() override { return jit_generator::create_kernel(); }

private:
    enum {
        typesize_data = sizeof(float),
        column_step = 16,
        num_regs = 32,
    };

    dim_t src_stride = 0, tr_src_stride = 0;
    dim_t src_batch_shift = 0, tr_src_batch_shift = 0;

    PReg mask_tail = p2;

    XReg reg_src = x8;
    XReg reg_tr_src = x9;
    XReg reg_loop_batch = x10;
    XReg reg_loop_row = x11;
    XReg reg_loop_col = x12;
    WReg regw_tmp = w14;
    XReg reg_long_offt = x15;

    void copy_block(int nrows, int ncolumns);
    void generate() override;
};

void jit_copy_f32_t::copy_block(int nrows, int ncolumns) {
    assert(!"unsupported\n");
}

void jit_copy_f32_t::generate() {
    preamble();

    const int row_block = conf_->os_block;
    const int row_tail = conf_->os % row_block;
    const int col_block = conf_->oc_block * conf_->nb_oc_blocking;
    const int col_tail = conf_->oc % col_block;
    src_stride = conf_->oc * typesize_data;
    tr_src_stride = conf_->LDB * typesize_data;
    src_batch_shift = src_stride * row_block;
    tr_src_batch_shift = tr_src_stride * row_block;
    add_imm(X_DEFAULT_ADDR, param1, GET_OFF(src), X_TMP_0);
    ldr(reg_src, ptr(X_DEFAULT_ADDR));
    add_imm(X_DEFAULT_ADDR, param1, GET_OFF(tr_src), X_TMP_0);
    ldr(reg_tr_src, ptr(X_DEFAULT_ADDR));
    add_imm(X_DEFAULT_ADDR, param1, GET_OFF(current_gemm_batch), X_TMP_0);
    ldr(reg_loop_batch, ptr(X_DEFAULT_ADDR));
    add_imm(X_DEFAULT_ADDR, param1, GET_OFF(current_row_size), X_TMP_0);
    ldr(reg_loop_row, ptr(X_DEFAULT_ADDR));
    add_imm(X_DEFAULT_ADDR, param1, GET_OFF(current_col_size), X_TMP_0);
    ldr(reg_loop_col, ptr(X_DEFAULT_ADDR));

    auto compute_batch = [=](int nrows, int ncolumns) {
        Label batch_loop;
        L(batch_loop);

        copy_block(nrows, ncolumns);
        add_imm(reg_src, reg_src, src_batch_shift, X_TMP_0);
        add_imm(reg_tr_src, reg_tr_src, tr_src_batch_shift, X_TMP_1);

        sub_imm(reg_loop_batch, reg_loop_batch, 1, X_TMP_0);
        cmp_imm(reg_loop_batch, 0, X_TMP_0);
        b(NE, batch_loop);
    };

    auto compute_rows = [=](int ncolumns) {
        Label row_done;
        if (row_tail > 0) {
            Label row_common;
            cmp_imm(reg_loop_row, row_block, X_TMP_0);
            b(EQ, row_common);

            compute_batch(row_tail, ncolumns);
            b(row_done);

            L(row_common);
        }

        compute_batch(row_block, ncolumns);
        L(row_done);
    };

    Label col_done;
    if (col_tail > 0) {
        Label col_common;
        cmp_imm(reg_loop_col, col_block, X_TMP_0);
        b(EQ, col_common);

        compute_rows(col_tail);
        b(col_done);

        L(col_common);
    }

    compute_rows(col_block);
    L(col_done);

    postamble();
}

struct jit_brgemm_trans_wei_f32_t : public jit_brgemm_trans_wei_t,
                                    public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_brgemm_trans_wei_f32_t)

    jit_brgemm_trans_wei_f32_t(const jit_brgemm_primitive_conf_t *conf)
        : jit_brgemm_trans_wei_t(conf) {}

    void operator()(ctx_t *ctx) override { jit_generator::operator()(ctx); }
    status_t create_kernel() override { return jit_generator::create_kernel(); }

private:
    enum { typesize = sizeof(float), transpose_size = 16 };
    dim_t src_stride = 0, tr_src_stride = 0;

    PReg k3333 = p1;
    PReg k5555 = p2;
    PReg kAAAA = p3;
    PReg kCCCC = p4;
    PReg k0F0F = p5;
    PReg kF0F0 = p6;
    PReg kTail = p7;

    XReg reg_src_base = x0;
    XReg reg_tr_src_base = x3;

    XReg reg_src = x8;
    XReg reg_tr_src = x9;
    XReg reg_loop_N = x10;
    XReg reg_loop_K = x11;
    XReg reg_loop_batch = x12;
    XReg reg_tr_src_tmp = x13;
    WReg regw_tmp = w14;

    void transpose_16x16(int nrows, int ncolumns = transpose_size);
    void generate() override;
};

void jit_brgemm_trans_wei_f32_t::transpose_16x16(int nrows, int ncolumns) {
    assert(!"unsupported\n");
}

void jit_brgemm_trans_wei_f32_t::generate() {
    preamble();
    assert(conf_->oc_block % transpose_size == 0);
    int fwd_ic_block = conf_->simd_w;
    int fwd_oc_block = 0;
    switch (conf_->wei_tag) {
        case OI16i64o:
        case OIw16i64o:
        case OIhw16i64o:
        case OIdhw16i64o:
        case OI8i64o2i:
        case OIw8i64o2i:
        case OIhw8i64o2i:
        case OIdhw8i64o2i:
        case OI16i64o2i:
        case OIw16i64o2i:
        case OIhw16i64o2i:
        case OIdhw16i64o2i: fwd_oc_block = 4 * conf_->simd_w; break;
        case OI16i32o:
        case OIw16i32o:
        case OIhw16i32o:
        case OIdhw16i32o:
        case OI8i32o2i:
        case OIw8i32o2i:
        case OIhw8i32o2i:
        case OIdhw8i32o2i:
        case OI16i32o2i:
        case OIw16i32o2i:
        case OIhw16i32o2i:
        case OIdhw16i32o2i: fwd_oc_block = 2 * conf_->simd_w; break;
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
    dim_t K_src_shift = conf_->simd_w * typesize;
    dim_t K_tr_src_shift = conf_->ic_block * conf_->simd_w * typesize;

    add_imm(X_DEFAULT_ADDR, param1, GET_OFF(src), X_TMP_0);
    ldr(reg_src_base, ptr(X_DEFAULT_ADDR));
    add_imm(X_DEFAULT_ADDR, param1, GET_OFF(tr_src), X_TMP_0);
    ldr(reg_tr_src_base, ptr(X_DEFAULT_ADDR));
    add_imm(X_DEFAULT_ADDR, param1, GET_OFF(current_gemm_batch), X_TMP_0);
    ldr(reg_loop_batch, ptr(X_DEFAULT_ADDR));
    add_imm(X_DEFAULT_ADDR, param1, GET_OFF(current_K), X_TMP_0);
    ldr(reg_loop_K, ptr(X_DEFAULT_ADDR));

    auto kmovw
            = [=](PReg k, unsigned w) { set_preg(k.h, w, X_TMP_0, X_TMP_1); };

    kmovw(k3333, 0x3333); // 0011001100110011
    kmovw(k5555, 0x5555); // 0101010101010101
    kmovw(kAAAA, 0xaaaa); // 1010101010101010
    kmovw(kCCCC, 0xcccc); // 1100110011001100
    kmovw(k0F0F, 0x0f0f); // 0000111100001111
    kmovw(kF0F0, 0xf0f0); // 1111000011110000

    auto compute_N = [=](bool is_oc_tail) {
        add_imm(X_DEFAULT_ADDR, param1, GET_OFF(current_N), X_TMP_0);
        ldr(reg_loop_N, ptr(X_DEFAULT_ADDR));
        mov(reg_src, reg_src_base);
        mov(reg_tr_src, reg_tr_src_base);
        Label N_loop, N_loop_tail;

        cmp_imm(reg_loop_N, transpose_size, X_TMP_0);
        b(LT, N_loop_tail);

        L(N_loop);

        transpose_16x16(transpose_size, is_oc_tail ? oc_tail : transpose_size);
        add_imm(reg_src, reg_src, N_src_shift, X_TMP_0);
        add_imm(reg_tr_src, reg_tr_src, N_tr_src_shift, X_TMP_1);

        sub_imm(reg_loop_N, reg_loop_N, (int)transpose_size, X_TMP_0);
        cmp_imm(reg_loop_N, transpose_size, X_TMP_1);
        b(GE, N_loop);

        L(N_loop_tail);
        if (ic_tail > 0) {
            Label N_loop_done;
            cmp_imm(reg_loop_N, 0, X_TMP_0);
            b(LE, N_loop_done);
            transpose_16x16(ic_tail, is_oc_tail ? oc_tail : transpose_size);
            L(N_loop_done);
        }
    };

    Label K_loop, K_tail;
    if (oc_tail > 0) {
        cmp_imm(reg_loop_K, transpose_size, X_TMP_0);
        b(LT, K_tail);
    }

    L(K_loop);
    compute_N(false);
    add_imm(reg_src_base, reg_src_base, K_src_shift, X_TMP_0);
    add_imm(reg_tr_src_base, reg_tr_src_base, K_tr_src_shift, X_TMP_1);

    sub_imm(reg_loop_K, reg_loop_K, (int)transpose_size, X_TMP_0);
    cmp_imm(reg_loop_K, transpose_size, X_TMP_1);
    b(GT, K_loop);

    L(K_tail);
    if (oc_tail > 0) {
        Label K_loop_done;
        cmp_imm(reg_loop_K, 0, X_TMP_0);
        b(LE, K_loop_done);

        compute_N(true);
        L(K_loop_done);
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
    enum { typesize = sizeof(int16_t), transpose_size = 16 };

    PReg kTail = p7;

    XReg reg_src_base = x0;
    XReg reg_tr_src_base = x3;

    XReg reg_src = x8;
    XReg reg_tr_src = x9;
    XReg reg_loop_N = x10;
    XReg reg_loop_K = x11;
    XReg reg_loop_batch = x12;
    XReg reg_tr_src_tmp = x13;
    WReg regw_tmp = w14;
    XReg imm_addr64 = x15;

    ZReg v_abcdefgh_to_abefcdgh = z31;

    void transpose_16x16_vnni(int nrows, int ncolumns = transpose_size);
    void generate() override;
};

void jit_brgemm_trans_wei_bf16_t::transpose_16x16_vnni(
        int nrows, int ncolumns) {
    assert(!"unsupported\n");
}

void jit_brgemm_trans_wei_bf16_t::generate() {
    assert(!"unsupported\n");
}

#undef GET_OFF

status_t create_brgemm_trans_src(
        std::unique_ptr<jit_brgemm_trans_src_t> &trans_ker,
        const jit_brgemm_primitive_conf_t *conf) {
    if (conf->prop_kind == dnnl_backward_weights
            && conf->src_dt == data_type::f32)
        CHECK(safe_ptr_assign(trans_ker, new jit_brgemm_trans_m_k_f32_t(conf)));
    else if (conf->prop_kind == dnnl_backward_weights
            && conf->src_dt == data_type::bf16)
        CHECK(safe_ptr_assign(
                trans_ker, new jit_brgemm_trans_m_k_bf16_t(conf)));
    else
        return status::invalid_arguments;

    return trans_ker->create_kernel();
}

status_t create_brgemm_copy_to_coarse(
        std::unique_ptr<jit_brgemm_copy_to_coarse_t> &copy_ker,
        const jit_brgemm_primitive_conf_t *conf) {
    if (false)
        CHECK(safe_ptr_assign(copy_ker, new jit_brgemm_copy_to_coarse_t(conf)));
    else
        return status::invalid_arguments;

    return copy_ker->create_kernel();
}

status_t create_brgemm_trans_to_vnni(
        std::unique_ptr<jit_brgemm_trans_to_vnni_t> &trans_ker,
        const jit_brgemm_primitive_conf_t *conf,
        jit_brgemm_trans_to_vnni_t::matrix_to_transform_t matrix_to_transform) {
    if (conf->prop_kind == dnnl_backward_weights
            && conf->dst_dt == data_type::bf16)
        CHECK(safe_ptr_assign(
                trans_ker, new jit_trans_to_vnni_t(conf, matrix_to_transform)));
    else if (conf->prop_kind == dnnl_backward_weights
            && conf->dst_dt == data_type::f32)
        CHECK(safe_ptr_assign(
                trans_ker, new jit_copy_f32_t(conf, matrix_to_transform)));
    else
        return status::invalid_arguments;

    return trans_ker->create_kernel();
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

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
