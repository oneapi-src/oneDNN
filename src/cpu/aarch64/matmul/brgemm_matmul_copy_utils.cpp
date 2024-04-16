/*******************************************************************************
* Copyright 2021-2023 Intel Corporation
* Copyright 2024 FUJITSU LIMITED
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
#include "cpu/aarch64/jit_generator.hpp"

#include "cpu/aarch64/matmul/brgemm_matmul_copy_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {
namespace matmul {

using namespace dnnl::impl::format_tag;
using namespace dnnl::impl::utils;
using namespace Xbyak_aarch64;

#define GET_OFF(x) offsetof(ctx_t, x)

template <cpu_isa_t isa>
struct jit_brgemm_matmul_copy_a_impl_t : public jit_brgemm_matmul_copy_a_t,
                                         public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_brgemm_matmul_copy_a_impl_t)

    jit_brgemm_matmul_copy_a_impl_t(const brgemm_matmul_conf_t *conf)
        : jit_brgemm_matmul_copy_a_t(conf)
        , jit_generator()
        , typesize_(conf_->a_dt_sz)
        , tr_typesize_(conf_->tr_a_dt_sz)
        , vnni_granularity_(data_type_vnni_granularity(conf_->src_dt))
        , k_step_(vlen_ / nstl::max(typesize_, tr_typesize_))
        , src_stride_(conf_->copy_A_src_stride)
        , tr_src_stride_((conf_->use_buffer_a_tail_only
                                         ? static_cast<dim_t>(conf_->wei_k_blk)
                                         : conf_->LDA)
                  * tr_typesize_)
        , do_compute_compensation_(conf_->has_zero_point_b)
        , k_loop_unroll_(is_ymm_ ? 7 : 16)
        , vmm_copy_idx_(29) {}

    void operator()(ctx_t *ctx) override { jit_generator::operator()(ctx); }
    status_t create_kernel() override { return jit_generator::create_kernel(); }

private:
    using reg64_t = const Xbyak_aarch64::XReg;
    using reg32_t = const Xbyak_aarch64::WReg;
    using opmask_t = const Xbyak_aarch64::PReg;

    static constexpr int vlen_ = cpu_isa_traits<isa>::vlen;
    static constexpr bool is_ymm_ = isa == sve_256;
    static constexpr int num_comp_acc_ = is_ymm_ ? 7 : 8;

    const int typesize_;
    const int tr_typesize_;
    const int vnni_granularity_;
    const int k_step_;
    const dim_t src_stride_;
    const dim_t tr_src_stride_;
    const bool do_compute_compensation_;

    const int k_loop_unroll_;
    const int vmm_copy_idx_;

    opmask_t kTail_load = p7;
    opmask_t kTail_store = p6;
    opmask_t kTail_comp = p5;

    reg64_t reg_src = x1;
    reg64_t reg_tr_src = x2;
    reg64_t reg_K_start = abi_not_param1;

    reg64_t reg_zp_comp_buf_ptr = x3;
    reg64_t reg_zp_comp_res_ptr = x4;

    reg64_t reg_M_blk = x9;
    reg64_t reg_K_blk = x10;
    reg64_t reg_batch = x11;
    reg64_t reg_aux_src = x12;
    reg64_t reg_aux_tr_src = x13;
    reg64_t regq_tmp = x14;
    reg64_t imm_addr64 = x15;
    reg64_t reg_zp_ab_comp_ptr = imm_addr64;
    reg64_t reg_zp_b_neg_val_ptr = reg_K_blk;

    ZReg vmm_ones_words = ZReg(28);
    ZReg vmm_dot_product_temp = ZReg(29);

    ZReg vmm_comp_mul = ZReg(is_ymm_ ? 14 : 30); // 1s
    ZReg vmm_comp_add = ZReg(is_ymm_ ? 15 : 31); // 128

    // Allows to shift A data by 128 for s8s8 problem for SVE512 in copy
    // routine, not in compute kernel. It's disabled for now, as it
    // requires setting some hint to brgemm kernel to avoid double shifting
    const bool allow_input_shift_for_s8s8 = false;

    ZReg get_vmm_comp_acc(int i) {
        assert(i >= 0 && i < num_comp_acc_);
        return ZReg(i);
    }

    ZReg get_vmm_copy(int i) {
        assert(i >= 0 && i < k_loop_unroll_);
        return ZReg(vmm_copy_idx_ - i);
    }

    void load_vmm(int idx, int offset) {}
    void store_vmm(int idx, int offset) {}
    void load_tail(int k_tail, size_t offset) {}
    void store_tail(int k_tail, size_t offset) {}
    void reduce_compensation_across_accumulators(int num_accumulators);
    void copy_K_loop(bool is_K_tail, bool is_first_K_iter, bool is_last_K_iter);
    void copy_M_loop(bool is_K_tail, bool is_first_K_iter, bool is_last_K_iter);
    inline void dot_product(ZReg v1, ZReg v2, ZReg v3) {
        assert(!"under construction");
    }
    void generate() override;
};

template <>
void jit_brgemm_matmul_copy_a_impl_t<sve_512>::load_vmm(int idx, int offset) {
    assert(!"under construction");
}

template <>
void jit_brgemm_matmul_copy_a_impl_t<sve_256>::load_vmm(int idx, int offset) {
    assert(!"under construction");
}

template <>
void jit_brgemm_matmul_copy_a_impl_t<sve_512>::store_vmm(int idx, int offset) {
    assert(!"under construction");
}

template <>
void jit_brgemm_matmul_copy_a_impl_t<sve_256>::store_vmm(int idx, int offset) {
    assert(!"under construction");
}

template <>
void jit_brgemm_matmul_copy_a_impl_t<sve_512>::load_tail(
        int k_tail, size_t offset) {
    assert(!"under construction");
}

template <>
void jit_brgemm_matmul_copy_a_impl_t<sve_256>::load_tail(
        int k_tail, size_t offset) {
    assert(!"under construction");
}

template <>
void jit_brgemm_matmul_copy_a_impl_t<sve_512>::store_tail(
        int k_tail, size_t offset) {
    assert(!"under construction");
}

template <>
void jit_brgemm_matmul_copy_a_impl_t<sve_256>::store_tail(
        int k_tail, size_t offset) {
    assert(!"under construction");
}

template <cpu_isa_t isa>
void jit_brgemm_matmul_copy_a_impl_t<
        isa>::reduce_compensation_across_accumulators(int num_accumulators) {
    assert(!"under construction");
}

template <cpu_isa_t isa>
void jit_brgemm_matmul_copy_a_impl_t<isa>::copy_K_loop(
        bool is_K_tail, bool is_first_K_iter, bool is_last_K_iter) {
    assert(!"under construction");
}

template <cpu_isa_t isa>
void jit_brgemm_matmul_copy_a_impl_t<isa>::copy_M_loop(
        bool is_K_tail, bool is_first_K_iter, bool is_last_K_iter) {
    assert(!"under construction");
}

template <cpu_isa_t isa>
void jit_brgemm_matmul_copy_a_impl_t<isa>::generate() {
    preamble();
    assert(!"under construction");

    postamble();
}

template struct jit_brgemm_matmul_copy_a_impl_t<sve_512>;
template struct jit_brgemm_matmul_copy_a_impl_t<sve_256>;

struct jit_brgemm_matmul_copy_a_transposed_impl_t
    : public jit_brgemm_matmul_copy_a_t,
      public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_brgemm_matmul_copy_a_transposed_impl_t)

    jit_brgemm_matmul_copy_a_transposed_impl_t(const brgemm_matmul_conf_t *conf)
        : jit_brgemm_matmul_copy_a_t(conf)
        , jit_generator()
        , typesize(conf_->a_dt_sz)
        , tr_typesize(conf_->tr_a_dt_sz)
        , src_stride(conf_->copy_A_src_stride)
        , dst_stride(conf_->LDA * tr_typesize)
        , m_loop_src_shift(columns_step * typesize)
        , m_loop_dst_shift(columns_step * dst_stride)
        , k_loop_src_shift(rows_step * src_stride)
        , k_loop_dst_shift(rows_step * tr_typesize)
        , is_f32(everyone_is(data_type::f32, conf_->src_dt, conf_->wei_dt))
        , is_bf32(conf_->is_bf32)
        , is_dynamic_src_ld(conf_->is_runtime_M) {
            MAYBE_UNUSED(m_loop_src_shift);
            MAYBE_UNUSED(k_loop_src_shift);
            MAYBE_UNUSED(m_loop_dst_shift);
            MAYBE_UNUSED(is_bf32);
            MAYBE_UNUSED(is_dynamic_src_ld);
            MAYBE_UNUSED(k_loop_dst_shift);
        }

    void operator()(ctx_t *ctx) override { jit_generator::operator()(ctx); }
    status_t create_kernel() override { return jit_generator::create_kernel(); }

private:
    using reg64_t = const Xbyak_aarch64::XReg;
    using reg32_t = const Xbyak_aarch64::WReg;
    using opmask_t = const Xbyak_aarch64::PReg;

    const size_t typesize;
    const size_t tr_typesize;
    static constexpr int rows_step = 16;
    static constexpr int columns_step = rows_step;
    const dim_t src_stride, dst_stride;
    const dim_t m_loop_src_shift;
    const dim_t m_loop_dst_shift;
    const dim_t k_loop_src_shift;
    const dim_t k_loop_dst_shift;
    const bool is_f32;
    const bool is_bf32;
    const bool is_dynamic_src_ld;
    

    opmask_t kFFFF = p1;
    opmask_t k3333 = p1;
    opmask_t k5555 = p2;
    opmask_t kAAAA = p3;
    opmask_t kAA = p4;
    opmask_t kCCCC = p4;
    opmask_t k55 = p5;
    opmask_t k0F0F = p5;
    opmask_t kCC = p6;
    opmask_t kF0F0 = p6;
    opmask_t k33 = p7;
    opmask_t kTail = is_f32 ? p7 : p1;

    reg64_t regq_tmp = X_TMP_4;
    reg32_t regw_tmp = W_TMP_4;
    reg64_t reg_k_src = x14;
    reg64_t reg_k_dst = x13;
    reg64_t reg_m_src = x12;
    reg64_t reg_m_dst = x11;
    reg64_t reg_aux_src0 = x10;
    reg64_t reg_aux_src1 = x9;
    reg64_t reg_loop_k = x1;
    reg64_t reg_loop_m = x2;
    reg64_t imm_addr64 = x3;

    reg64_t reg_opmask_shift_compute = x4;

    Xbyak_aarch64::ZReg vidx1 = z31;
    Xbyak_aarch64::ZReg vidx2 = z30;
    Xbyak_aarch64::ZReg vidx3 = z29;
    Xbyak_aarch64::ZReg vidx4 = z28;
    Xbyak_aarch64::ZReg vidx5 = z27;
    Xbyak_aarch64::ZReg zmm_tmp = z26;

    constexpr static int current_M_blk_offt_ = 0;
    constexpr static int src_offt_ = 8;
    constexpr static int tr_src_offt_ = 16;
    constexpr static int current_K_blk_offt_ = 24;
    constexpr static int dynamic_src_ld_offt_ = 32;
    constexpr static int dynamic_src_ld_x_2_offt_ = 40;
    constexpr static int dynamic_src_ld_x_kstep_offt_ = 48;
    constexpr static int stack_space_needed_ = 56;

    void transpose_f32(reg64_t dst, reg64_t src, int nrows, int ncolumns);
    void transpose_bf16(reg64_t dst, reg64_t src, int nrows, int ncolumns);
    void deploy_transpose(reg64_t dst, reg64_t src, int nrows, int ncolumns);
    void generate() override;
};

void jit_brgemm_matmul_copy_a_transposed_impl_t::transpose_bf16(
        reg64_t dst, reg64_t src, int nrows, int ncolumns) {
    assert(!"under construction");
}

void jit_brgemm_matmul_copy_a_transposed_impl_t::transpose_f32(
        reg64_t dst, reg64_t src, int nrows, int ncolumns) {
    assert(!"under construction");
}

void jit_brgemm_matmul_copy_a_transposed_impl_t::deploy_transpose(
        reg64_t dst, reg64_t src, int nrows, int ncolumns) {
    if (is_f32)
        transpose_f32(dst, src, nrows, ncolumns);
    else
        assert(!"unreachable");
}

void jit_brgemm_matmul_copy_a_transposed_impl_t::generate() {

    // only bf16, f16 and f32 supported for now
    if (!one_of(conf_->src_dt, data_type::bf16, data_type::f32, data_type::f16))
        return;
    preamble();

    assert(!"under construction");
    postamble();
}

template <cpu_isa_t isa>
struct jit_brgemm_matmul_copy_b_int8_t : public jit_brgemm_matmul_copy_b_t,
                                         public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_brgemm_matmul_copy_b_int8_t)

    jit_brgemm_matmul_copy_b_int8_t(const brgemm_matmul_conf_t *conf)
        : jit_brgemm_matmul_copy_b_t(conf)
        , jit_generator()
        , src_stride_(conf->wei_tag == format_tag::acbd
                          ? conf->copy_B_wei_stride
                          : conf->N * sizeof(int8_t))
        , tr_src_stride_(conf->LDB * k_blk_step_ * sizeof(int8_t))
        , do_compute_compensation_(
                  conf->s8s8_compensation_required || conf->has_zero_point_a)
        , comp_acc_idx_(25) {}

    void operator()(ctx_t *ctx) override { jit_generator::operator()(ctx); }
    status_t create_kernel() override { return jit_generator::create_kernel(); }

protected:
    using reg64_t = const Xbyak_aarch64::XReg;
    using reg32_t = const Xbyak_aarch64::WReg;

    static constexpr bool is_ymm_ = cpu_isa_traits<isa>::vlen == 32;
    static constexpr int k_blk_step_ = 4;
    static constexpr int n_blk_step_ = 64;
    static constexpr int blk_sz_ = 6;
    static constexpr int simd_w_ = cpu_isa_traits<isa>::vlen;

    const dim_t src_stride_;
    const dim_t tr_src_stride_;
    const bool do_compute_compensation_;

    const int comp_acc_idx_;

    const Xbyak_aarch64::PReg kTail = p7;

    reg64_t reg_src = x1;
    reg64_t reg_tr_src = x2;
    reg64_t reg_comp_ptr = x3;
    reg64_t reg_zp_comp_ptr = x11;
    reg64_t reg_zp_a_neg_val_ptr = x12;

    reg64_t reg_K_iters = x8;
    reg64_t reg_N_blk = x9;
    reg64_t reg_K_start = x10;
    reg64_t regq_tmp = x14;
    reg64_t imm_addr64 = x15;

    ZReg vmm_ones_words = ZReg(24);
    ZReg vmm_dot_product_temp = ZReg(25);

    ZReg vreg_idx_lo_256 = z26;
    ZReg vreg_idx_hi_256 = z27;
    ZReg vreg_idx_lo_128 = z28;
    ZReg vreg_idx_hi_128 = z29;

    ZReg vmm_comp_mul = z30;
    ZReg vmm_zero = z31;

    ZReg get_comp_acc(int i) { return ZReg(comp_acc_idx_ - i); }
    ZReg get_vmm_zp_comp_res(int i) { return get_comp_acc(i); }
    ZReg get_vmm_oscale_comp_res(int i) { return ZReg(i); }

    inline void vmovdqa64(ZReg vmm, const void *addr) {
        assert(!"under construction");
    }

    inline ZReg get_vmm(int blk, int idx) {
        if (idx < 0 || idx >= 32) assert(!"idx > vregs");
        assert(IMPLICATION(!is_ymm_, idx < blk_sz_ && blk >= 0));
        auto reg_idx = blk_sz_ * blk + idx;
        return ZReg(reg_idx);
    }
    inline void load(int blk, int i, bool is_tail) {}
    inline void kmovq(Xbyak_aarch64::PReg k, size_t q) {
        assert(!"under construction");
    }
    virtual void init_permute() {}
    virtual void copy_4x64(int nrows, int ncolumns) {}
    inline void dot_product(ZReg v1, ZReg v2, ZReg v3) {
        assert(!"under construction");
    }
    void generate() override;
};

template <>
inline void jit_brgemm_matmul_copy_b_int8_t<sve_512>::load(
        int blk, int i, bool is_tail) {
    assert(!"under construction");
}

template struct jit_brgemm_matmul_copy_b_int8_t<sve_512>;
template struct jit_brgemm_matmul_copy_b_int8_t<sve_256>;

struct jit_sve_512_core_brgemm_matmul_copy_b_int8_t
    : public jit_brgemm_matmul_copy_b_int8_t<sve_512> {

    jit_sve_512_core_brgemm_matmul_copy_b_int8_t(
            const brgemm_matmul_conf_t *conf)
        : jit_brgemm_matmul_copy_b_int8_t<sve_512>(conf) {}

private:
    void init_permute() override { assert(!"under construction"); }

    void copy_4x64(int nrows, int ncolumns) override {
        assert(!"under construction");
    }
};

struct jit_sve_256_core_brgemm_matmul_copy_b_int8_t
    : public jit_brgemm_matmul_copy_b_int8_t<sve_256> {

    jit_sve_256_core_brgemm_matmul_copy_b_int8_t(
            const brgemm_matmul_conf_t *conf)
        : jit_brgemm_matmul_copy_b_int8_t<sve_256>(conf) {}

private:
    static constexpr int perm2i128_l
            = 0x20; // dst[127:0]=src1_low_128; dst[128:255]=src2_low_128
    static constexpr int perm2i128_h
            = 0x31; // dst[127:0]=src1_hi_128; dst[128:255]=src2_hi_128

    Xbyak_aarch64::ZReg get_ymm(int idx) { return get_vmm(0, idx); }

    void load_ymm(int ymm_idx, size_t offset, bool is_tail, size_t tail_sz) {
        assert(!"under construction");
    }

    void copy_4x64(int nrows, int ncolumns) override {
        assert(!"under construction");
    }
};

template <cpu_isa_t isa>
void jit_brgemm_matmul_copy_b_int8_t<isa>::generate() {
    preamble();
    assert(!"under construction");
    postamble();
}

struct jit_brgemm_matmul_copy_b_f32_t : public jit_brgemm_matmul_copy_b_t,
                                        public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_brgemm_matmul_copy_b_f32_t)

    jit_brgemm_matmul_copy_b_f32_t(const brgemm_matmul_conf_t *conf)
        : jit_brgemm_matmul_copy_b_t(conf)
        , jit_generator()
        , dt_in_(data_type::f32)
        , typesize_in_(types::data_type_size(dt_in_))
        , src_stride_(conf_->wei_tag == acbd ? conf_->copy_B_wei_stride
                                             : conf_->N * typesize_in_)
        , tr_src_stride_(conf_->LDB * typesize_out_) {
            MAYBE_UNUSED(src_stride_);
            MAYBE_UNUSED(tr_src_stride_);
        }

    void operator()(ctx_t *ctx) override { jit_generator::operator()(ctx); }
    status_t create_kernel() override { return jit_generator::create_kernel(); }

private:
    using reg64_t = const Xbyak_aarch64::XReg;
    using reg32_t = const Xbyak_aarch64::WReg;
    using opmask_t = const Xbyak_aarch64::PReg;
    using zmm = const Xbyak_aarch64::ZReg;

    enum { n_blk_step = 16, max_regs_available = 30 };
    const data_type_t dt_in_;
    const size_t typesize_in_;
    const size_t typesize_out_ = sizeof(float);
    dim_t src_stride_, tr_src_stride_;

    opmask_t kTail = p7;
    opmask_t kFFFF = p6;

    reg64_t reg_src = x1;
    reg64_t reg_tr_src = x2;

    reg64_t reg_K_iters = x8;
    reg64_t reg_N_blk = x9;
    reg64_t reg_K_start = x10;
    reg32_t regw_tmp = w14;
    reg64_t imm_addr64 = x15;

    zmm zmm_permw = z30;
    zmm zmm_zero = z31;

    inline void kmovw(Xbyak_aarch64::PReg k, unsigned w) {
        assert(!"under construction");
    }
    void copy_16_x_n_block(int nrows, int ncolumns);
    void compute_k_loop(int ncolumns);
    void generate() override;
};

void jit_brgemm_matmul_copy_b_f32_t::copy_16_x_n_block(
        int nrows, int ncolumns) {
    assert(!"under construction");
}

void jit_brgemm_matmul_copy_b_f32_t::compute_k_loop(int ncolumns) {
    assert(!"under construction");
}

void jit_brgemm_matmul_copy_b_f32_t::generate() {
    preamble();
    assert(!"under construction");

    postamble();
}

template <cpu_isa_t isa>
struct jit_brgemm_matmul_copy_b_transposed_t
    : public jit_brgemm_matmul_copy_b_t,
      public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_brgemm_matmul_copy_b_transposed_t)

    jit_brgemm_matmul_copy_b_transposed_t(const brgemm_matmul_conf_t *conf)
        : jit_brgemm_matmul_copy_b_t(conf)
        , jit_generator()
        , typesize_(conf_->b_dt_sz)
        , tr_typesize_(conf_->tr_b_dt_sz)
        , vnni_granularity_(data_type_vnni_granularity(conf_->wei_dt))
        , k_blk_step_(vlen_ / tr_typesize_)
        , do_compute_compensation_(
                  conf_->has_zero_point_a || conf_->s8s8_compensation_required)
        , is_bf32_(conf->is_bf32)
        , req_zp_comp_(conf_->has_zero_point_a)
        , req_s8s8_comp_(conf_->s8s8_compensation_required)
        , max_tmp_idx(16 - (do_compute_compensation_ ? 6 : 0))
        , src_stride_(conf_->wei_tag == format_tag::adbc
                          ? conf_->copy_B_wei_stride
                          : conf_->K * typesize_)
        , tr_src_stride_(conf_->LDB * vnni_granularity_ * tr_typesize_) {}

    void operator()(ctx_t *ctx) override { jit_generator::operator()(ctx); }
    status_t create_kernel() override { return jit_generator::create_kernel(); }

private:
    using reg64_t = const Xbyak_aarch64::XReg;
    using reg32_t = const Xbyak_aarch64::WReg;
    using opmask_t = const Xbyak_aarch64::PReg;
    using ZReg = const Xbyak_aarch64::ZReg;

    static constexpr bool is_ymm_ = isa == sve_256;
    static constexpr cpu_isa_t isa_ = isa;
    static constexpr int max_vmm_regs_ = cpu_isa_traits<isa_>::n_vregs;
    static constexpr int vlen_ = cpu_isa_traits<isa>::vlen;
    static constexpr int n_blk_step_ = 16;
    static constexpr int bf32_k_blk_step_ = 16;
    static constexpr size_t comp_shift_ = vlen_;

    const int typesize_;
    const int tr_typesize_;
    const int vnni_granularity_;
    const int k_blk_step_;
    const bool do_compute_compensation_;
    const bool is_bf32_;
    const bool req_zp_comp_;
    const bool req_s8s8_comp_;
    const int max_tmp_idx;

    const dim_t src_stride_, tr_src_stride_;

    opmask_t k3333 = p1;
    opmask_t k5555 = p2;
    opmask_t kAAAA = p3;
    opmask_t kCCCC = p4;
    opmask_t k0F0F = p5;
    opmask_t kF0F0 = p6;
    opmask_t kTail = p7;

    reg64_t reg_src_base = x1;
    reg64_t reg_tr_src_base = x2;
    reg64_t reg_comp_ptr = x3;

    reg64_t reg_K_iters = x8;
    reg64_t reg_N_iters = x9;
    reg64_t reg_src = x10;
    reg64_t reg_tr_src = x11;
    reg64_t reg_zp_comp_ptr = x12;
    reg64_t reg_zp_a_neg_val_ptr = x13;
    reg64_t reg_K_start = x14;

    reg64_t regq_tmp = x15;
    reg32_t regw_tmp = w15;
    reg64_t imm_addr64 = abi_not_param1;

    // Note: for the SVE256 implementation, reserve ZReg(8) and ZReg(9) as
    // temporary compute registers.
    ZReg vmm_comp_mul = Xbyak_aarch64::ZReg(max_vmm_regs_ - 1);
    ZReg vmm_comp_acc = Xbyak_aarch64::ZReg(max_vmm_regs_ - 2);
    ZReg vmm_zp_a_neg_val = Xbyak_aarch64::ZReg(max_vmm_regs_ - 3);
    ZReg vmm_s8s8_comp_acc = Xbyak_aarch64::ZReg(max_vmm_regs_ - 4);
    ZReg vmm_all_bits_1 = Xbyak_aarch64::ZReg(max_vmm_regs_ - 5);
    ZReg vmm_one_s32 = Xbyak_aarch64::ZReg(max_vmm_regs_ - 6);

    // Required in every dot product for INT8 non-VNNI computation.
    ZReg vmm_ones_words = ZReg(max_vmm_regs_ - 7);
    ZReg vmm_dot_product_temp = ZReg(max_vmm_regs_ - 8);

    void kmovw(Xbyak_aarch64::PReg k, unsigned w) {
        assert(!"under construction");
    };

    void kmovq(Xbyak_aarch64::PReg k, size_t q) {
        assert(!"under construction");
    };

    ZReg src_vmm(int i) {
        assert(i >= 0 && i < n_blk_step_);
        return ZReg(i);
    }

    ZReg tmp_vmm(int i) {
        // If compensation compute is required - last 6 zregs are reserved for it
        assert(i >= 0 && IMPLICATION(!is_ymm_, i < max_tmp_idx)
                && IMPLICATION(is_ymm_, i < 2));
        return ZReg(n_blk_step_ + i);
    }

    void copy_row_x_col(int nrows, int ncolumns);
    void compute_K_loop(bool is_N_tail, int curr_K_tail, bool is_first_K_iter,
            bool is_last_K_iter);
    void compute_N_loop(
            int curr_K_tail, bool is_first_K_iter, bool is_last_K_iter);

    inline void dot_product(ZReg v1, ZReg v2, ZReg v3) {
        assert(!"under construction");
    }
    void generate() override;
};

template <cpu_isa_t isa>
void jit_brgemm_matmul_copy_b_transposed_t<isa>::copy_row_x_col(
        int nrows, int ncolumns) {
    assert(!"under construction");
}

template <>
void jit_brgemm_matmul_copy_b_transposed_t<sve_256>::copy_row_x_col(
        int nrows, int ncolumns) {
    assert(!"under construction");
}

template <cpu_isa_t isa>
void jit_brgemm_matmul_copy_b_transposed_t<isa>::compute_K_loop(bool is_N_tail,
        int curr_K_tail, bool is_first_K_iter, bool is_last_K_iter) {
    assert(!"under construction");
}

template <cpu_isa_t isa>
void jit_brgemm_matmul_copy_b_transposed_t<isa>::compute_N_loop(
        int curr_K_tail, bool is_first_K_iter, bool is_last_K_iter) {
    assert(!"under construction");
}

template <cpu_isa_t isa>
void jit_brgemm_matmul_copy_b_transposed_t<isa>::generate() {

    preamble();
    assert(!"under construction");

    postamble();
}

template struct jit_brgemm_matmul_copy_b_transposed_t<sve_512>;
template struct jit_brgemm_matmul_copy_b_transposed_t<sve_256>;

status_t create_brgemm_matmul_copy_b(
        std::unique_ptr<jit_brgemm_matmul_copy_b_t> &copy_ker,
        const brgemm_matmul_conf_t *conf) {
    const bool is_B_transposed
            = one_of(conf->wei_tag, ba, acb, abdc, adbc, abced, abcdfe, abcdegf,
                    abcdefhg, abcdefgih, abcdefghji, abcdefghikj, abcdefghijlk);
    const bool is_bf16
            = everyone_is(data_type::bf16, conf->src_dt, conf->wei_dt);
    const bool is_f32 = everyone_is(data_type::f32, conf->src_dt, conf->wei_dt);

    const bool is_f16 = everyone_is(data_type::f16, conf->src_dt, conf->wei_dt);
    assert(is_f32);
    assert(!(is_bf16 || is_f16));

    if (is_B_transposed) {
        if (is_superset(conf->isa, sve_512))
            CHECK(safe_ptr_assign(copy_ker,
                    new jit_brgemm_matmul_copy_b_transposed_t<sve_512>(conf)));
        else {
            assert(is_superset(conf->isa, sve_256));
            CHECK(safe_ptr_assign(copy_ker,
                    new jit_brgemm_matmul_copy_b_transposed_t<sve_256>(conf)));
        }
    } else {
        if (is_bf16 || is_f16 || conf->is_bf32) {
            assert(!"unreacable");
        } else if (is_f32) {
            CHECK(safe_ptr_assign(
                    copy_ker, new jit_brgemm_matmul_copy_b_f32_t(conf)));
        } else {
            if (is_superset(conf->isa, sve_512))
                CHECK(safe_ptr_assign(copy_ker,
                        new jit_sve_512_core_brgemm_matmul_copy_b_int8_t(
                                conf)));
            else {
                assert(is_superset(conf->isa, sve_256));
                CHECK(safe_ptr_assign(copy_ker,
                        new jit_sve_256_core_brgemm_matmul_copy_b_int8_t(
                                conf)));
            }
        }
    }

    return copy_ker->create_kernel();
}

status_t create_brgemm_matmul_copy_a(
        std::unique_ptr<jit_brgemm_matmul_copy_a_t> &copy_ker,
        const brgemm_matmul_conf_t *conf) {
    if (conf->transposed_A) {
        CHECK(safe_ptr_assign(copy_ker,
                new jit_brgemm_matmul_copy_a_transposed_impl_t(conf)));
    } else {
        if (is_superset(conf->isa, sve_512))
            CHECK(safe_ptr_assign(copy_ker,
                    new jit_brgemm_matmul_copy_a_impl_t<sve_512>(conf)));
        else {
            assert(one_of(conf->isa, sve_256));
            CHECK(safe_ptr_assign(copy_ker,
                    new jit_brgemm_matmul_copy_a_impl_t<sve_256>(conf)));
        }
    }

    return copy_ker->create_kernel();
}

} // namespace matmul
} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
