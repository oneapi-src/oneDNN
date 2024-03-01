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

    // Required in every dot product for INT8 non-VNNI computation.
    ZReg vmm_ones_words = ZReg(28);
    ZReg vmm_dot_product_temp = ZReg(29);

    ZReg vmm_comp_mul = ZReg(is_ymm_ ? 14 : 30); // 1s
    ZReg vmm_comp_add = ZReg(is_ymm_ ? 15 : 31); // 128

    // Allows to shift A data by 128 for s8s8 problem for AVX512 in copy
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
#if 0
            vpdpbusd(v1, v2, v3,
                    mayiuse(avx512_core) ? EvexEncoding : VexEncoding);
#endif
    }
    void generate() override;
};

template <>
void jit_brgemm_matmul_copy_a_impl_t<sve_512>::load_vmm(int idx, int offset) {
    assert(!"under construction");
#if 0
    const auto addr = EVEX_compress_addr(reg_src, offset);
    if (conf_->isa == avx512_core_fp16) {
        vcvtph2psx(get_vmm_copy(idx), addr);
    } else
        vmovdqu8(get_vmm_copy(idx), addr);
#endif
}

template <>
void jit_brgemm_matmul_copy_a_impl_t<sve_256>::load_vmm(int idx, int offset) {
    assert(!"under construction");
#if 0
    uni_vmovups(get_vmm_copy(idx), ptr[reg_src + offset]);
#endif
}

template <>
void jit_brgemm_matmul_copy_a_impl_t<sve_512>::store_vmm(int idx, int offset) {
    assert(!"under construction");
#if 0
    auto tr_src_addr = EVEX_compress_addr(reg_tr_src, offset);
    vmovdqu8(tr_src_addr, get_vmm_copy(idx));
#endif
}

template <>
void jit_brgemm_matmul_copy_a_impl_t<sve_256>::store_vmm(int idx, int offset) {
    assert(!"under construction");
#if 0
    uni_vmovups(ptr[reg_tr_src + offset], get_vmm_copy(idx));
#endif
}

template <>
void jit_brgemm_matmul_copy_a_impl_t<sve_512>::load_tail(
        int k_tail, size_t offset) {
    assert(!"under construction");
#if 0
    const auto kmovx = [this](Opmask k, size_t q) {
        if (conf_->is_bf32) {
            mov(regq_tmp.cvt32(), q);
            jit_generator::kmovw(k, regq_tmp.cvt32());
        } else {
            mov(regq_tmp, q);
            jit_generator::kmovq(k, regq_tmp);
        }
    };

    const size_t dt_step
            = conf_->is_bf32 || conf_->isa == avx512_core_fp16 ? 1 : typesize_;
    const size_t tail_mask_load = size_t(((size_t)1 << (dt_step * k_tail)) - 1);
    kmovx(kTail_load, tail_mask_load);
    const int k_tail_st = rnd_up(k_tail, vnni_granularity_);
    const size_t full_mask
            = conf_->is_bf32 ? ((size_t)1 << 16) - 1 : 0xffffffffffffffff;
    const size_t tail_mask_store = k_tail_st == k_step_
            ? full_mask
            : size_t(((size_t)1 << (dt_step * k_tail_st)) - 1);
    kmovx(kTail_store, tail_mask_store);

    auto zmm_tail = get_vmm_copy(0) | kTail_load | T_z;
    auto load_addr = EVEX_compress_addr(reg_src, offset * typesize_);
    if (conf_->is_bf32)
        vmovups(zmm_tail, load_addr);
    else if (conf_->isa == avx512_core_fp16)
        vcvtph2psx(zmm_tail, load_addr);
    else
        vmovdqu8(zmm_tail, load_addr);
#endif
}

template <>
void jit_brgemm_matmul_copy_a_impl_t<sve_256>::load_tail(
        int k_tail, size_t offset) {
    assert(!"under construction");
#if 0
    const auto vmm_tail = get_vmm_copy(0);
    load_bytes(vmm_tail, reg_src, offset * typesize_, k_tail);
#endif
}

template <>
void jit_brgemm_matmul_copy_a_impl_t<sve_512>::store_tail(
        int k_tail, size_t offset) {
    assert(!"under construction");
#if 0
    auto tr_src_addr = EVEX_compress_addr(reg_tr_src, offset * tr_typesize_);
    if (conf_->is_bf32) {
        Ymm ymm_downcvt_bf16 = Ymm(get_vmm_copy(0).getIdx());
        vcvtneps2bf16(ymm_downcvt_bf16, get_vmm_copy(0));
        vmovdqu16(tr_src_addr, ymm_downcvt_bf16 | kTail_store);
    } else if (conf_->isa == avx512_core_fp16) {
        vmovups(tr_src_addr, get_vmm_copy(0) | kTail_store);
    } else
        vmovdqu8(tr_src_addr, get_vmm_copy(0) | kTail_store);
#endif
}

template <>
void jit_brgemm_matmul_copy_a_impl_t<sve_256>::store_tail(
        int k_tail, size_t offset) {
    assert(!"under construction");
#if 0
    const int k_tail_st = rnd_up(k_tail, vnni_granularity_);
    const auto vmm_tail = get_vmm_copy(0);
    store_bytes(vmm_tail, reg_tr_src, offset * tr_typesize_, k_tail_st);
#endif
}

template <cpu_isa_t isa>
void jit_brgemm_matmul_copy_a_impl_t<
        isa>::reduce_compensation_across_accumulators(int num_accumulators) {
    assert(!"under construction");
#if 0
    int num = num_accumulators;
    while (num > 1) {
        for (int i = 0; i < num / 2; i++) {
            const auto vmm_acc0 = get_vmm_comp_acc(i);
            const auto vmm_acc1 = get_vmm_comp_acc(div_up(num, 2) + i);
            uni_vpaddd(vmm_acc0, vmm_acc0, vmm_acc1);
        }
        num = div_up(num, 2);
    }
#endif
}

template <cpu_isa_t isa>
void jit_brgemm_matmul_copy_a_impl_t<isa>::copy_K_loop(
        bool is_K_tail, bool is_first_K_iter, bool is_last_K_iter) {
    assert(!"under construction");
#if 0
    const int K_blk = is_K_tail ? conf_->K % conf_->K_blk
                                : nstl::min(conf_->K, conf_->K_blk);
    const int k_tail = K_blk % k_step_;
    const int num_k_iters = K_blk / k_step_;
    const int num_acc = utils::saturate(1, (int)num_comp_acc_, num_k_iters);

    if (do_compute_compensation_) {
        for (int i = 0; i < num_acc; i++) {
            const auto vmm_acc = get_vmm_comp_acc(i);
            uni_vpxor(vmm_acc, vmm_acc, vmm_acc);
        }
    }

    auto maybe_compute_compensation = [this, num_acc](int k_idx, ZReg vmm_copy) {
        if (do_compute_compensation_) {
            const auto vmm_comp_acc = get_vmm_comp_acc(k_idx % num_acc);
            if (conf_->src_dt == data_type::s8)
                dot_product(vmm_comp_acc, vmm_comp_mul, vmm_copy);
            else
                dot_product(vmm_comp_acc, vmm_copy, vmm_comp_mul);
        }
    };

    for (int kb = 0; kb < div_up(num_k_iters, k_loop_unroll_); kb++) {
        const int k_end
                = nstl::min(k_loop_unroll_, num_k_iters - kb * k_loop_unroll_);
        for (int k = 0; k < k_end; k++) {
            const int k_idx = kb * k_loop_unroll_ + k;
            const size_t offset
                    = static_cast<size_t>(k_idx) * k_step_ * typesize_;
            load_vmm(k, offset);
            maybe_compute_compensation(k_idx, get_vmm_copy(k));
        }
        if (allow_input_shift_for_s8s8 && conf_->s8s8_compensation_required) {
            for (int k = 0; k < k_end; k++)
                vpaddb(get_vmm_copy(k), get_vmm_copy(k), vmm_comp_add);
        }
        if (conf_->is_bf32) {
            assert(typesize_ != tr_typesize_);
            int k = 0;
            const int k_end_2 = rnd_dn(k_end, 2);
            for (; k < k_end_2; k += 2) {
                const size_t offset = ((size_t)kb * k_loop_unroll_ + k)
                        * k_step_ * tr_typesize_;
                auto tr_src_addr = EVEX_compress_addr(reg_tr_src, offset);

                auto zmm_src = get_vmm_copy(k);
                auto zmm_src_next = get_vmm_copy(k + 1);

                vcvtne2ps2bf16(zmm_src, zmm_src_next, zmm_src);
                vmovups(tr_src_addr, zmm_src);
            }
            if (k < k_end) {
                const size_t offset = ((size_t)kb * k_loop_unroll_ + k)
                        * k_step_ * tr_typesize_;
                auto tr_src_addr = EVEX_compress_addr(reg_tr_src, offset);
                Ymm ymm_downcvt_bf16 = Ymm(get_vmm_copy(k).getIdx());
                vcvtneps2bf16(ymm_downcvt_bf16, get_vmm_copy(k));
                vmovdqu16(tr_src_addr, ymm_downcvt_bf16);
            }
        } else {
            for (int k = 0; k < k_end; k++) {
                const size_t offset
                        = (static_cast<size_t>(kb) * k_loop_unroll_ + k)
                        * k_step_ * tr_typesize_;
                store_vmm(k, offset);
            }
        }
    }

    if (k_tail > 0) {
        load_tail(k_tail, num_k_iters * k_step_);
        maybe_compute_compensation(0, get_vmm_copy(0));

        if (allow_input_shift_for_s8s8 && conf_->s8s8_compensation_required)
            vpaddb(get_vmm_copy(0), get_vmm_copy(0), vmm_comp_add);

        store_tail(k_tail, num_k_iters * k_step_);
    }

    if (do_compute_compensation_) {
        reduce_compensation_across_accumulators(num_acc);

        const auto addr_buf = ptr[reg_zp_comp_buf_ptr];
        if (!is_first_K_iter)
            uni_vpaddd(get_vmm_comp_acc(0), get_vmm_comp_acc(0), addr_buf);
        if (!is_last_K_iter) {
            uni_vmovups(addr_buf, get_vmm_comp_acc(0));
            return;
        }

        // is_last_K_iter == true: we need to reduce values within acc
        // register, add mixed ab_compensation component if any, multiply
        // it by negative zp_b_value and finally store the result

        // step 1: reduce values within acc register
        const auto ymm_red0 = Ymm(get_vmm_comp_acc(0).getIdx());
        const auto ymm_red1 = Ymm(get_vmm_comp_acc(1).getIdx());
        if (!is_ymm_) {
            vextracti64x4(ymm_red1, Zmm(get_vmm_comp_acc(0).getIdx()), 1);
            vphaddd(ymm_red0, ymm_red0, ymm_red1);
        }
        uni_vpxor(ymm_red1, ymm_red1, ymm_red1);
        uni_vphaddd(ymm_red0, ymm_red0, ymm_red1);
        uni_vphaddd(ymm_red0, ymm_red0, ymm_red1);
        const auto xmm_red1 = Xmm(ymm_red1.getIdx());
        vextractf128(xmm_red1, ymm_red0, 1);
        uni_vpaddd(ymm_red0, ymm_red0, ymm_red1);

        const auto vmm_in_mask = get_vmm_comp_acc(1);
        if (is_ymm_) {
            static const uint32_t mask_in[8]
                    = {0xffffffff, 0, 0, 0, 0, 0, 0, 0};
            mov(regq_tmp, reinterpret_cast<size_t>(mask_in));
            vmovups(vmm_in_mask, ptr[regq_tmp]);
        }

        // step 2: add -K * zp_a_val as mixed ab_compensation component
        if (conf_->src_zp_type != brgemm_broadcast_t::none) {
            assert(conf_->src_zp_type == brgemm_broadcast_t::per_tensor);
            reg64_t reg_zp_ab_comp_ptr = imm_addr64;
            mov(reg_zp_ab_comp_ptr, ptr[param1 + GET_OFF(zp_ab_comp_ptr)]);
            if (is_ymm_) {
                const auto vmm_zp = get_vmm_comp_acc(2);
                vmaskmovps(vmm_zp, vmm_in_mask, ptr[reg_zp_ab_comp_ptr]);
                uni_vpaddd(ymm_red0, ymm_red0, vmm_zp);
            } else {
                const auto addr_ab_comp = zword_b[reg_zp_ab_comp_ptr];
                const auto zmm_res = get_vmm_comp_acc(0) | kTail_comp;
                vpaddd(zmm_res, get_vmm_comp_acc(0), addr_ab_comp);
            }
        }

        // step 3: multiply by zp_b_val
        mov(reg_zp_b_neg_val_ptr, ptr[param1 + GET_OFF(zp_b_neg_value_ptr)]);
        const auto vmm_zp_b_neg_val = get_vmm_comp_acc(is_ymm_ ? 2 : 1);
        uni_vbroadcastss(vmm_zp_b_neg_val, ptr[reg_zp_b_neg_val_ptr]);
        uni_vpmulld(get_vmm_comp_acc(0), get_vmm_comp_acc(0), vmm_zp_b_neg_val);

        // step 4: store the final result value
        if (is_ymm_)
            vmaskmovps(ptr[reg_zp_comp_res_ptr], vmm_in_mask, ymm_red0);
        else
            vmovups(ptr[reg_zp_comp_res_ptr], get_vmm_comp_acc(0) | kTail_comp);
    }
#endif
}

template <cpu_isa_t isa>
void jit_brgemm_matmul_copy_a_impl_t<isa>::copy_M_loop(
        bool is_K_tail, bool is_first_K_iter, bool is_last_K_iter) {
    assert(!"under construction");
#if 0

    if (do_compute_compensation_) {
        mov(imm_addr64, 1);
        uni_vpbroadcastb(vmm_comp_mul, imm_addr64.cvt8());
        if (!(is_first_K_iter && is_last_K_iter))
            mov(reg_zp_comp_buf_ptr,
                    ptr[param1 + GET_OFF(zp_b_compensation_buffer_ptr)]);

        if (is_last_K_iter) {
            mov(reg_zp_comp_res_ptr,
                    ptr[param1 + GET_OFF(zp_a_compensation_result_ptr)]);
            if (!is_ymm_) {
                mov(regq_tmp, 1);
                jit_generator::kmovw(kTail_comp, imm_addr64.cvt32());
            }
        }
    }

    Label loop_M;
    L(loop_M);

    copy_K_loop(is_K_tail, is_first_K_iter, is_last_K_iter);

    add(reg_src, src_stride_);
    add(reg_tr_src, tr_src_stride_);
    if (do_compute_compensation_) {
        // shift comp pointers
        if (!(is_first_K_iter && is_last_K_iter))
            add(reg_zp_comp_buf_ptr, sizeof(int32_t) * 16);
        if (is_last_K_iter) add(reg_zp_comp_res_ptr, sizeof(int32_t));
    }

    dec(reg_M_blk);
    jnz(loop_M, T_NEAR);
#endif
}

template <cpu_isa_t isa>
void jit_brgemm_matmul_copy_a_impl_t<isa>::generate() {
    preamble();
    assert(!"under construction");
#if 0

    mov(reg_src, ptr[param1 + GET_OFF(src)]);
    mov(reg_tr_src, ptr[param1 + GET_OFF(tr_src)]);
    mov(reg_K_blk, ptr[param1 + GET_OFF(current_K_blk)]);
    mov(reg_M_blk, ptr[param1 + GET_OFF(current_M_blk)]);

    if (allow_input_shift_for_s8s8 && conf_->s8s8_compensation_required) {
        mov(imm_addr64, 128);
        uni_vpbroadcastb(vmm_comp_add, imm_addr64.cvt8());
    }

    auto copy_body = [this](bool is_first_K_iter, bool is_last_K_iter) {
        Label copy_body_done;
        // might be different from conf_->K_tail
        const dim_t K_blk_tail
                = conf_->K_tail > 0 ? conf_->K % conf_->K_blk : 0;
        if (K_blk_tail > 0) {
            Label not_K_tail;
            cmp(reg_K_blk, K_blk_tail);
            jne(not_K_tail, T_NEAR);
            copy_M_loop(true, is_first_K_iter, is_last_K_iter);
            jmp(copy_body_done, T_NEAR);

            L(not_K_tail);
        }

        copy_M_loop(false, is_first_K_iter, is_last_K_iter);
        L(copy_body_done);
    };

    Label done;
    if (do_compute_compensation_) {
        assert(conf_->wei_zp_type == brgemm_broadcast_t::per_tensor);

        mov(reg_K_start, ptr[param1 + GET_OFF(current_K_start)]);
        const auto last_K_threshold
                = rnd_up(conf_->K, conf_->K_blk) - conf_->K_blk;
        Label not_first, not_first_not_last;
        cmp(reg_K_start, 0);
        jne(not_first, T_NEAR);
        {
            // first K iteration
            Label first_not_last;
            cmp(reg_K_start, last_K_threshold);
            jl(first_not_last, T_NEAR);
            copy_body(true, true);
            jmp(done, T_NEAR);

            L(first_not_last);
            copy_body(true, false);
            jmp(done, T_NEAR);
        }

        L(not_first);
        cmp(reg_K_start, last_K_threshold);
        jl(not_first_not_last, T_NEAR);

        copy_body(false, true);
        jmp(done, T_NEAR);
        L(not_first_not_last);
    }
    copy_body(false, false);
    L(done);
#endif

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
        , is_dynamic_src_ld(conf_->is_runtime_M) {}

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
    // Note: this must be assigned to rcx as it's used in shl instruction,
    // clashes with abi_param1 on Windows OS
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
#if 0
    assert(nrows >= 0 && nrows <= rows_step && ncolumns >= 0
            && ncolumns <= columns_step);
    if (!nrows) return;

    auto src_zmm = [](int i) { return ZReg(i); };

    auto src_ymm = [](int i) {
        assert(i >= 0 && i < 16);
        return ZReg(i);
    };

    Label transpose_bf16_done;
    const bool dynamic_column_size = ncolumns == 0 && is_dynamic_src_ld;
    auto kmovx
            = [this, dynamic_column_size](Opmask k, unsigned w,
                      bool load_mask_stage = false, bool use_word_sz = false) {
                  if (dynamic_column_size && load_mask_stage) {
                      // reg_opmask_shift_compute is rcx, and we need cl for the shift
                      mov(reg_opmask_shift_compute, reg_loop_m);
                      mov(regq_tmp, 1);
                      shl(regq_tmp, cl);
                      sub(regq_tmp, 1);
                  } else
                      mov(regw_tmp, w);
                  if (use_word_sz)
                      jit_generator::kmovw(k, regw_tmp);
                  else
                      jit_generator::kmovd(k, regw_tmp);
              };

    auto store = [this, dst](Zmm r, int i) {
        auto addr = EVEX_compress_addr(dst, i * dst_stride);
        vmovdqu16(addr, r | kTail);
    };

    const int load_mask
            = ncolumns < columns_step ? (1 << ncolumns) - 1 : 0xffff;
    kmovx(kFFFF, load_mask, true, is_bf32);

    for (int i = 0; i < nrows / 2; i++) {
        auto idx0 = 2 * i;
        auto idx1 = 2 * i + 1;
        auto zmm_src0 = src_zmm(idx0);
        auto zmm_src1 = src_zmm(idx1);
        if (is_dynamic_src_ld) {
            if (i == 0) {
                mov(reg_aux_src0, src);
                mov(reg_aux_src1, src);
                add(reg_aux_src1, ptr[rsp + dynamic_src_ld_offt_]);
            } else {
                add(reg_aux_src0, ptr[rsp + dynamic_src_ld_x_2_offt_]);
                add(reg_aux_src1, ptr[rsp + dynamic_src_ld_x_2_offt_]);
            }
        }
        auto src_addr_0 = is_dynamic_src_ld
                ? ptr[reg_aux_src0]
                : EVEX_compress_addr(src, idx0 * src_stride);
        auto src_addr_1 = is_dynamic_src_ld
                ? ptr[reg_aux_src1]
                : EVEX_compress_addr(src, idx1 * src_stride);
        if (is_bf32) {
            vmovups(zmm_src0 | kFFFF | T_z, src_addr_0);
            vmovups(zmm_src1 | kFFFF | T_z, src_addr_1);
            vcvtne2ps2bf16(zmm_src0, zmm_src1, zmm_src0);
        } else {
            auto src1 = src_ymm(idx1);
            vmovdqu16(zmm_src0 | kFFFF | T_z, src_addr_0);
            vmovdqu16(zmm_src1 | kFFFF | T_z, src_addr_1);
            vinsertf64x4(zmm_src0, zmm_src0, src1, 1);
        }
        vpermw(zmm_src0, vidx5, zmm_src0);
    }

    // for odd numbers we need to mix row with zeroes
    if (nrows % 2) {
        int i = nrows / 2;
        auto zmm_src0 = src_zmm(2 * i);
        if (is_dynamic_src_ld) {
            if (i == 0) {
                mov(reg_aux_src0, src);
            } else {
                add(reg_aux_src0, ptr[rsp + dynamic_src_ld_x_2_offt_]);
            }
        }
        auto src_addr = is_dynamic_src_ld
                ? ptr[reg_aux_src0]
                : EVEX_compress_addr(src, 2 * i * src_stride);
        if (is_bf32) {
            vmovups(zmm_src0 | kFFFF | T_z, src_addr);
            vcvtneps2bf16(Ymm(zmm_src0.getIdx()), zmm_src0);
        } else
            vmovdqu16(zmm_src0 | kFFFF | T_z, src_addr);
        vpermw(zmm_src0, vidx5, zmm_src0);
    }

    for (int i = rnd_up(nrows, 2); i < rows_step; i += 2) {
        vpxord(src_zmm(i), src_zmm(i), src_zmm(i));
    }

    // swap 1
    for (int i = 0; i < 4; i++) {
        auto zmm0 = src_zmm(4 * i);
        auto zmm1 = src_zmm(4 * i + 2);
        auto tmp0 = src_zmm(4 * i + 1);
        auto tmp1 = src_zmm(4 * i + 3);

        vmovups(tmp0, zmm0);
        vmovups(tmp1, zmm1);

        vpermps(tmp0 | kAAAA, vidx3, zmm1);
        vpermps(tmp1 | k5555, vidx3, zmm0);
    }
    // swap 2
    int base_idx;
    base_idx = 0;
    for (int i = 0; i < 2; i++) {
        auto zmm0 = src_zmm(base_idx + 2 * i + 1);
        auto zmm1 = src_zmm(base_idx + 2 * i + 5);

        auto tmp0 = src_zmm(base_idx + 2 * i);
        auto tmp1 = src_zmm(base_idx + 2 * i + 4);

        vmovupd(tmp0, zmm0);
        vmovupd(tmp1, zmm1);

        vpermpd(tmp0 | kAA, vidx2, zmm1);
        vpermpd(tmp1 | k55, vidx2, zmm0);
    }
    base_idx = 8;
    for (int i = 0; i < 2; i++) {
        auto zmm0 = src_zmm(base_idx + 2 * i + 1);
        auto zmm1 = src_zmm(base_idx + 2 * i + 5);

        auto tmp0 = src_zmm(base_idx + 2 * i);
        auto tmp1 = src_zmm(base_idx + 2 * i + 4);

        vmovupd(tmp0, zmm0);
        vmovupd(tmp1, zmm1);

        vpermpd(tmp0 | kAA, vidx2, zmm1);
        vpermpd(tmp1 | k55, vidx2, zmm0);
    }

    // swap 3
    for (int i = 0; i < 4; i++) {
        auto zmm0 = src_zmm(2 * i);
        auto zmm1 = src_zmm(2 * i + 8);

        auto tmp0 = src_zmm(2 * i + 1);
        auto tmp1 = src_zmm(2 * i + 9);

        vmovupd(tmp0, zmm0);
        vmovupd(tmp1, zmm1);

        vpermpd(tmp0 | kCC, vidx1, zmm1);
        vpermpd(tmp1 | k33, vidx1, zmm0);
    }

    // all stores
    for (int i = 0; i < 8; i++)
        vextracti64x4(src_ymm(2 * i), src_zmm(2 * i + 1), 1);

    auto get_vec_idx = [](int col_idx) {
        assert(col_idx < columns_step && col_idx >= 0);
        const int blk_sz = 4;
        const int blk_idx = col_idx / blk_sz;
        const int idx_within_blk = col_idx % blk_sz;

        // 0 1 2 3 -> 0 2 1 3
        const int mapped_blk_idx = 2 * blk_idx - (blk_idx / 2) * 3;
        // 0 1 2 3 -> 1 0 3 2
        const int mapped_idx_within_blk
                = idx_within_blk + 1 - 2 * (idx_within_blk % 2);
        return blk_sz * mapped_blk_idx + mapped_idx_within_blk;
    };
    const int rows_to_store = rnd_up(nrows, 2);
    const int store_mask
            = rows_to_store < rows_step ? (1 << rows_to_store) - 1 : 0xffff;
    kmovx(kTail, store_mask);

    const int columns_to_store = dynamic_column_size ? columns_step : ncolumns;
    for (int col_idx = 0; col_idx < columns_to_store; col_idx++) {
        store(src_zmm(get_vec_idx(col_idx)), col_idx);
        if (dynamic_column_size) {
            dec(reg_opmask_shift_compute);
            jz(transpose_bf16_done, T_NEAR);
        }
    }

    L(transpose_bf16_done);
#endif
}

void jit_brgemm_matmul_copy_a_transposed_impl_t::transpose_f32(
        reg64_t dst, reg64_t src, int nrows, int ncolumns) {
    assert(!"under construction");
#if 0
    assert(nrows >= 0 && nrows <= rows_step && ncolumns >= 0
            && ncolumns <= columns_step);
    if (!nrows) return;
    Label transpose_f32_done;
    const bool dynamic_column_size = ncolumns == 0 && is_dynamic_src_ld;
    auto kmovw = [this, dynamic_column_size](
                         Opmask k, size_t q, bool load_mask = false) {
        if (dynamic_column_size && load_mask) {
            // reg_opmask_shift_compute is rcx, and we need cl for the shift
            mov(reg_opmask_shift_compute, reg_loop_m);
            mov(regq_tmp, 1);
            shl(regq_tmp, cl);
            sub(regq_tmp, 1);
        } else
            mov(regw_tmp, q);
        jit_generator::kmovw(k, regw_tmp);
    };

    const int load_mask
            = ncolumns < columns_step ? (1 << ncolumns) - 1 : 0xffff;
    kmovw(kTail, load_mask, true);

    auto src_zmm = [](int i) {
        assert(i >= 0 && i < 16);
        return Zmm(i);
    };

    auto tmp_zmm = [](int i) {
        assert(i >= 0 && i < 16);
        return Zmm(16 + i);
    };

    auto load = [this, src, nrows, src_zmm](int i) {
        const auto addr = is_dynamic_src_ld
                ? ptr[i % 2 == 0 ? reg_aux_src0 : reg_aux_src1]
                : EVEX_compress_addr(src, i * src_stride);
        if (i < nrows)
            if (conf_->isa == avx512_core_fp16)
                vcvtph2psx(src_zmm(i) | kTail | T_z, addr);
            else
                vmovups(src_zmm(i) | kTail | T_z, addr);
        else
            vpxord(src_zmm(i), src_zmm(i), src_zmm(i));
    };

    auto store = [this, dst](Zmm r, int i) {
        auto addr = EVEX_compress_addr(dst, i * dst_stride);
        vmovups(addr, r | kTail);
    };

    auto transpose16x8 = [&](int base_idx) {
        assert(base_idx == 0 || base_idx == 8);

        // swap 1
        for (int i = 0; i < 4; i++) {
            int src_idx0 = base_idx + i * 2;
            int src_idx1 = src_idx0 + 1;

            int next_src_idx0 = src_idx0 + 2;
            int next_src_idx1 = src_idx1 + 2;
            bool load_next = base_idx == 0 || i < 3;

            if (base_idx == 0 && i == 0) {
                if (is_dynamic_src_ld) {
                    mov(reg_aux_src0, src);
                    mov(reg_aux_src1, src);
                    add(reg_aux_src1, ptr[rsp + dynamic_src_ld_offt_]);
                }
                load(src_idx0);
                load(src_idx1);
            }

            auto tmp0 = tmp_zmm(src_idx0);
            auto tmp1 = tmp_zmm(src_idx1);
            auto src0 = src_zmm(src_idx0);
            auto src1 = src_zmm(src_idx1);

            if (next_src_idx0 < nrows && load_next) {
                if (is_dynamic_src_ld)
                    add(reg_aux_src0, ptr[rsp + dynamic_src_ld_x_2_offt_]);
                load(next_src_idx0);
            }
            valignd(tmp0, src0, src0, 0x1);

            if (next_src_idx1 < nrows && load_next) {
                if (is_dynamic_src_ld)
                    add(reg_aux_src1, ptr[rsp + dynamic_src_ld_x_2_offt_]);
                load(next_src_idx1);
            }
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

    auto fixup16x16 = [&]() {
        const int store_mask = nrows < rows_step ? (1 << nrows) - 1 : 0xffff;
        kmovw(kTail, store_mask);

        // swap 8
        int columns_to_store = dynamic_column_size ? 8 : nstl::min(8, ncolumns);
        for (int i = 0; i < columns_to_store; i++) {
            auto tmp = tmp_zmm(i);
            auto src0 = src_zmm(i);
            auto src8 = src_zmm(8 + i);
            vshuff64x2(tmp, src0, src8, 0x44);
            store(tmp, i);
            if (dynamic_column_size) {
                dec(reg_opmask_shift_compute);
                jz(transpose_f32_done, T_NEAR);
            }
        }

        columns_to_store = dynamic_column_size ? 8 : nstl::max(0, ncolumns - 8);
        for (int i = 0; i < columns_to_store; i++) {
            auto tmp = tmp_zmm(8 + i);
            auto src0 = src_zmm(i);
            auto src8 = src_zmm(8 + i);
            vshuff64x2(tmp, src0, src8, 0xee);
            store(tmp, 8 + i);
            if (dynamic_column_size) {
                dec(reg_opmask_shift_compute);
                jz(transpose_f32_done, T_NEAR);
            }
        }
    };

    transpose16x8(0);
    transpose16x8(8);
    fixup16x16();
    L(transpose_f32_done);
#endif
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
#if 0
    sub(rsp, stack_space_needed_);
    mov(regq_tmp, ptr[param1 + GET_OFF(current_M_blk)]);
    mov(ptr[rsp + current_M_blk_offt_], regq_tmp);
    mov(regq_tmp, ptr[param1 + GET_OFF(src)]);
    mov(ptr[rsp + src_offt_], regq_tmp);
    mov(regq_tmp, ptr[param1 + GET_OFF(tr_src)]);
    mov(ptr[rsp + tr_src_offt_], regq_tmp);
    mov(regq_tmp, ptr[param1 + GET_OFF(current_K_blk)]);
    mov(ptr[rsp + current_K_blk_offt_], regq_tmp);
    if (is_dynamic_src_ld) {
        // dynamic src_stride
        mov(regq_tmp, ptr[param1 + GET_OFF(dynamic_src_ld)]);
        mov(ptr[rsp + dynamic_src_ld_offt_], regq_tmp);

        // src_stride * 2
        shl(regq_tmp, 1);
        mov(ptr[rsp + dynamic_src_ld_x_2_offt_], regq_tmp);

        // src_stride * rows_step
        assert(rows_step == 16);
        shl(regq_tmp, 3);
        mov(ptr[rsp + dynamic_src_ld_x_kstep_offt_], regq_tmp);
    }

    alignas(64) static constexpr const int64_t idx1[8]
            = {2, 3, 0, 1, 6, 7, 4, 5};
    alignas(64) static constexpr const int64_t idx2[8]
            = {1, 0, 3, 2, 5, 4, 7, 6};
    alignas(64) static constexpr const int32_t idx3[16]
            = {1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14};
    alignas(64) static constexpr const int32_t idx4[16]
            = {8, 10, 12, 14, 0, 2, 4, 6, 9, 11, 13, 15, 1, 3, 5, 7};
    alignas(64) static constexpr const uint16_t idx5[32]
            = {0, 16, 2, 18, 8, 24, 10, 26, 4, 20, 6, 22, 12, 28, 14, 30, 1, 17,
                    3, 19, 9, 25, 11, 27, 5, 21, 7, 23, 13, 29, 15, 31};

    const int k_block_tail = conf_->K_blk % rows_step;
    const int last_k_block_tail = (conf_->K % conf_->K_blk) % rows_step;
    const int m_block_tail = conf_->M_blk % columns_step;
    const int last_m_block_tail = conf_->M_tail % columns_step;

    auto kmovw = [this](Opmask k, unsigned w) {
        mov(regw_tmp, w);
        jit_generator::kmovw(k, regw_tmp);
    };

    if (is_f32) {
        kmovw(k3333, 0x3333); // 0011001100110011
        kmovw(k5555, 0x5555); // 0101010101010101
        kmovw(kAAAA, 0xaaaa); // 1010101010101010
        kmovw(kCCCC, 0xcccc); // 1100110011001100
        kmovw(k0F0F, 0x0f0f); // 0000111100001111
        kmovw(kF0F0, 0xf0f0); // 1111000011110000
    } else {
        kmovw(kFFFF, 0xffff);
        kmovw(k5555, 0x5555);
        kmovw(kAAAA, 0xaaaa);
        kmovw(kAA, 0xaa);
        kmovw(k55, 0x55);
        kmovw(kCC, 0xcc);
        kmovw(k33, 0x33);
    }

    auto vmovdqa64 = [this](Zmm z, const int64_t *addr) {
        mov(imm_addr64, reinterpret_cast<size_t>(addr));
        jit_generator::vmovdqa64(z, ptr[imm_addr64]);
    };

    auto vmovdqa32 = [this](Zmm z, const int32_t *addr) {
        mov(imm_addr64, reinterpret_cast<size_t>(addr));
        jit_generator::vmovdqa32(z, ptr[imm_addr64]);
    };

    if (!is_f32) {
        vmovdqa64(vidx1, idx1);
        vmovdqa64(vidx2, idx2);
        vmovdqa32(vidx3, idx3);
        vmovdqa32(vidx4, idx4);
        vmovdqa32(vidx5, (const int32_t *)idx5);
    }

    auto compute_m_loop = [&](reg64_t &reg_base, reg64_t &reg_tr_base,
                                  int nrows) {
        mov(reg_loop_m, ptr[rsp + current_M_blk_offt_]);
        mov(reg_m_src, reg_base);
        mov(reg_m_dst, reg_tr_base);

        Label m_loop_tail_or_done, m_loop, compute_m_loop_done;
        cmp(reg_loop_m, columns_step);
        jl(m_loop_tail_or_done, T_NEAR);

        L(m_loop);
        {
            deploy_transpose(reg_m_dst, reg_m_src, nrows, columns_step);
            add(reg_m_src, m_loop_src_shift);
            add(reg_m_dst, m_loop_dst_shift);
        }
        sub(reg_loop_m, columns_step);
        cmp(reg_loop_m, columns_step);
        jge(m_loop, T_NEAR);

        if (m_block_tail > 0 || last_m_block_tail > 0 || is_dynamic_src_ld)
            jz(compute_m_loop_done, T_NEAR);

        L(m_loop_tail_or_done);

        if (m_block_tail > 0) {
            Label m_block_tail_done;
            cmp(reg_loop_m, m_block_tail);
            jne(m_block_tail_done, T_NEAR);

            deploy_transpose(reg_m_dst, reg_m_src, nrows, m_block_tail);
            jmp(compute_m_loop_done, T_NEAR);

            L(m_block_tail_done);
        }
        if (IMPLICATION(
                    last_m_block_tail <= 0 || last_m_block_tail == m_block_tail,
                    is_dynamic_src_ld)) {
            Label last_m_block_tail_done;
            cmp(reg_loop_m, 0);
            jle(last_m_block_tail_done, T_NEAR);

            deploy_transpose(reg_m_dst, reg_m_src, nrows,
                    is_dynamic_src_ld ? 0 : last_m_block_tail);

            L(last_m_block_tail_done);
        }

        L(compute_m_loop_done);
    };

    auto compute_k_loop = [&]() {
        mov(reg_k_src, ptr[rsp + src_offt_]);
        mov(reg_k_dst, ptr[rsp + tr_src_offt_]);
        mov(reg_loop_k, ptr[rsp + current_K_blk_offt_]);

        Label k_tail_or_done, k_loop, compute_k_loop_done;
        cmp(reg_loop_k, rows_step);
        jl(k_tail_or_done, T_NEAR);

        L(k_loop);
        {
            compute_m_loop(reg_k_src, reg_k_dst, rows_step);
            if (is_dynamic_src_ld)
                add(reg_k_src, ptr[rsp + dynamic_src_ld_x_kstep_offt_]);
            else
                add(reg_k_src, k_loop_src_shift);
            add(reg_k_dst, k_loop_dst_shift);
        }
        sub(reg_loop_k, rows_step);
        cmp(reg_loop_k, rows_step);
        jge(k_loop, T_NEAR);

        if (k_block_tail > 0 || last_k_block_tail > 0)
            jz(compute_k_loop_done, T_NEAR);

        L(k_tail_or_done);

        if (k_block_tail > 0) {
            Label k_block_tail_done;
            cmp(reg_loop_k, k_block_tail);
            jne(k_block_tail_done, T_NEAR);

            compute_m_loop(reg_k_src, reg_k_dst, k_block_tail);
            jmp(compute_k_loop_done, T_NEAR);

            L(k_block_tail_done);
        }
        if (last_k_block_tail > 0 && last_k_block_tail != k_block_tail) {
            Label last_k_block_tail_done;
            cmp(reg_loop_k, last_k_block_tail);
            jne(last_k_block_tail_done, T_NEAR);

            compute_m_loop(reg_k_src, reg_k_dst, last_k_block_tail);
            jmp(compute_k_loop_done, T_NEAR);

            L(last_k_block_tail_done);
        }

        L(compute_k_loop_done);
    };

    compute_k_loop();

    add(rsp, stack_space_needed_);
#endif
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

    // Required in every dot product for INT8 non-VNNI computation.
    ZReg vmm_ones_words = ZReg(24);
    ZReg vmm_dot_product_temp = ZReg(25);

    // ZMM stuff
    ZReg vreg_idx_lo_256 = z26;
    ZReg vreg_idx_hi_256 = z27;
    ZReg vreg_idx_lo_128 = z28;
    ZReg vreg_idx_hi_128 = z29;

    // Shared
    ZReg vmm_comp_mul = z30;
    ZReg vmm_zero = z31;

    ZReg get_comp_acc(int i) { return ZReg(comp_acc_idx_ - i); }
    ZReg get_vmm_zp_comp_res(int i) { return get_comp_acc(i); }
    ZReg get_vmm_oscale_comp_res(int i) { return ZReg(i); }

    inline void vmovdqa64(ZReg vmm, const void *addr) {
        assert(!"under construction");
#if 0
        mov(imm_addr64, reinterpret_cast<size_t>(addr));
        jit_generator::vmovdqa64(vmm, ptr[imm_addr64]);
#endif
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
#if 0
    mov(regq_tmp, q);
    jit_generator::kmovq(k, regq_tmp);
#endif
    }
    virtual void init_permute() {}
    virtual void copy_4x64(int nrows, int ncolumns) {}
    inline void dot_product(ZReg v1, ZReg v2, ZReg v3) {
        assert(!"under construction");
#if 0
            vpdpbusd(v1, v2, v3,
                    mayiuse(avx512_core) ? EvexEncoding : VexEncoding);
#endif
    }
    void generate() override;
};

template <>
inline void jit_brgemm_matmul_copy_b_int8_t<sve_512>::load(
        int blk, int i, bool is_tail) {
    assert(!"under construction");
#if 0
    auto vmm_src = get_vmm(blk, i % k_blk_step_);
    auto src_load = is_tail ? vmm_src | kTail | T_z : vmm_src;
    vmovdqu8(src_load, EVEX_compress_addr(reg_src, i * src_stride_));
#endif
}

template struct jit_brgemm_matmul_copy_b_int8_t<sve_512>;
template struct jit_brgemm_matmul_copy_b_int8_t<sve_256>;

struct jit_sve_512_core_brgemm_matmul_copy_b_int8_t
    : public jit_brgemm_matmul_copy_b_int8_t<sve_512> {

    jit_sve_512_core_brgemm_matmul_copy_b_int8_t(
            const brgemm_matmul_conf_t *conf)
        : jit_brgemm_matmul_copy_b_int8_t<sve_512>(conf) {}

private:
    void init_permute() override {
        assert(!"under construction");
#if 0
        alignas(64) static constexpr const int64_t idx_lo_256[8]
                = {0, 1, 2, 3, 8, 9, 10, 11};
        alignas(64) static constexpr const int64_t idx_hi_256[8]
                = {4, 5, 6, 7, 12, 13, 14, 15};

        alignas(64) static constexpr const int64_t idx_lo_128[8]
                = {0, 1, 8, 9, 4, 5, 12, 13};
        alignas(64) static constexpr const int64_t idx_hi_128[8]
                = {2, 3, 10, 11, 6, 7, 14, 15};

        vmovdqa64(vreg_idx_lo_256, (const void *)idx_lo_256);
        vmovdqa64(vreg_idx_hi_256, (const void *)idx_hi_256);
        vmovdqa64(vreg_idx_lo_128, (const void *)idx_lo_128);
        vmovdqa64(vreg_idx_hi_128, (const void *)idx_hi_128);
#endif
    }

    void copy_4x64(int nrows, int ncolumns) override {
        assert(!"under construction");
#if 0
        const bool is_tail = ncolumns < n_blk_step_;
        if (is_tail) {
            const auto tail_mask = size_t(((size_t)1 << ncolumns) - 1);
            kmovq(kTail, tail_mask);
        }

        const int max_unroll = (do_compute_compensation_ ? 21 : 25) / blk_sz_;

        for_(int kb = 0; kb < div_up(nrows, max_unroll * k_blk_step_); kb++)
        for (int k = 0; k < nstl::min(max_unroll,
                                div_up(nrows - kb * max_unroll * k_blk_step_,
                                        k_blk_step_));
                k++) {
            const int row_start = (kb * max_unroll + k) * k_blk_step_;
            const int row_end = nstl::min(row_start + k_blk_step_, nrows);

            for (int i = row_start; i < row_end; i++)
                load(k, i, is_tail);
            if (row_end == nrows && nrows % k_blk_step_ > 0) {
                for (int i = nrows; i < rnd_up(nrows, k_blk_step_); i++) {
                    auto src_reg = get_vmm(k, i % k_blk_step_);
                    vpxord(src_reg, src_reg, src_reg);
                }
            }

            vpunpcklbw(get_vmm(k, 4), get_vmm(k, 0), get_vmm(k, 1));
            vpunpckhbw(get_vmm(k, 5), get_vmm(k, 0), get_vmm(k, 1));
            vpunpcklbw(get_vmm(k, 0), get_vmm(k, 2), get_vmm(k, 3));
            vpunpckhbw(get_vmm(k, 1), get_vmm(k, 2), get_vmm(k, 3));

            vpunpcklwd(get_vmm(k, 2), get_vmm(k, 4), get_vmm(k, 0));
            vpunpckhwd(get_vmm(k, 3), get_vmm(k, 4), get_vmm(k, 0));
            vpunpcklwd(get_vmm(k, 4), get_vmm(k, 5), get_vmm(k, 1));
            vpunpckhwd(get_vmm(k, 5), get_vmm(k, 5), get_vmm(k, 1));

            vmovups(get_vmm(k, 0), vreg_idx_lo_256);
            vpermi2q(get_vmm(k, 0), get_vmm(k, 2), get_vmm(k, 4));
            vmovups(get_vmm(k, 1), vreg_idx_hi_256);
            vpermi2q(get_vmm(k, 1), get_vmm(k, 2), get_vmm(k, 4));
            vmovups(get_vmm(k, 2), vreg_idx_lo_256);
            vpermi2q(get_vmm(k, 2), get_vmm(k, 3), get_vmm(k, 5));
            vmovups(get_vmm(k, 4), vreg_idx_hi_256);
            vpermi2q(get_vmm(k, 4), get_vmm(k, 3), get_vmm(k, 5));

            vmovups(get_vmm(k, 3), vreg_idx_lo_128);
            vpermi2q(get_vmm(k, 3), get_vmm(k, 0), get_vmm(k, 2));
            dim_t tr_src_off_base = (kb * max_unroll + k) * tr_src_stride_;
            vmovups(EVEX_compress_addr(reg_tr_src, tr_src_off_base),
                    get_vmm(k, 3));
            if (do_compute_compensation_)
                dot_product(get_comp_acc(0), vmm_comp_mul, get_vmm(k, 3));

            if (ncolumns > 16) {
                vmovups(get_vmm(k, 5), vreg_idx_hi_128);
                vpermi2q(get_vmm(k, 5), get_vmm(k, 0), get_vmm(k, 2));
                vmovups(EVEX_compress_addr(reg_tr_src, tr_src_off_base + 64),
                        get_vmm(k, 5));
                if (do_compute_compensation_)
                    dot_product(get_comp_acc(1), vmm_comp_mul, get_vmm(k, 5));
            } else if (conf_->wei_n_blk > 16) {
                vmovups(EVEX_compress_addr(reg_tr_src, tr_src_off_base + 64),
                        vmm_zero);
            }

            if (ncolumns > 32) {
                vmovups(get_vmm(k, 0), vreg_idx_lo_128);
                vpermi2q(get_vmm(k, 0), get_vmm(k, 1), get_vmm(k, 4));
                vmovups(EVEX_compress_addr(reg_tr_src, tr_src_off_base + 128),
                        get_vmm(k, 0));
                if (do_compute_compensation_)
                    dot_product(get_comp_acc(2), vmm_comp_mul, get_vmm(k, 0));
            } else if (conf_->wei_n_blk > 32) {
                vmovups(EVEX_compress_addr(reg_tr_src, tr_src_off_base + 128),
                        vmm_zero);
            }

            if (ncolumns > 48) {
                vmovups(get_vmm(k, 2), vreg_idx_hi_128);
                vpermi2q(get_vmm(k, 2), get_vmm(k, 1), get_vmm(k, 4));
                vmovups(EVEX_compress_addr(reg_tr_src, tr_src_off_base + 192),
                        get_vmm(k, 2));
                if (do_compute_compensation_)
                    dot_product(get_comp_acc(3), vmm_comp_mul, get_vmm(k, 2));
            } else if (conf_->wei_n_blk > 48) {
                vmovups(EVEX_compress_addr(reg_tr_src, tr_src_off_base + 192),
                        vmm_zero);
            }
        }
#endif
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
#if 0
        Xbyak_aarch64::Ymm vmm_src = Xbyak_aarch64::Ymm(ymm_idx);
        if (is_tail) {
            load_bytes(vmm_src, reg_src, offset, tail_sz);
        } else
            uni_vmovups(vmm_src, ptr[reg_src + offset]);
#endif
    }

    void copy_4x64(int nrows, int ncolumns) override {
        assert(!"under construction");
#if 0
        const bool is_tail = ncolumns < n_blk_step_;
        const int k_end = div_up(nrows, k_blk_step_);
        for_(int k = 0; k < k_end; k++)
        for (int pass = 0; pass < 2; ++pass) {
            assert(one_of(pass, 0, 1));
            const dim_t tr_src_off_base = k * tr_src_stride_;
            const int set_1_tr_src_offset
                    = tr_src_off_base + pass * 2 * n_blk_step_;
            const int row_start = k * k_blk_step_;
            const int row_end = nstl::min(row_start + k_blk_step_, nrows);
            for (int i = row_start; i < rnd_up(row_end, k_blk_step_); i++) {
                const bool do_load = i < row_end
                        && IMPLICATION(pass == 1, ncolumns >= simd_w_);
                if (do_load) {
                    const bool do_tail = is_tail
                            && IMPLICATION(pass == 0, ncolumns < simd_w_);
                    load_ymm(i % 4, i * src_stride_ + pass * simd_w_, do_tail,
                            ncolumns - pass * simd_w_);
                } else {
                    const auto src_ymm_1 = get_ymm(i % 4);
                    uni_vpxor(src_ymm_1, src_ymm_1, src_ymm_1);
                }
            }

            vpunpcklbw(get_ymm(4), get_ymm(0), get_ymm(1));
            vpunpckhbw(get_ymm(5), get_ymm(0), get_ymm(1));
            vpunpcklbw(get_ymm(0), get_ymm(2), get_ymm(3));
            vpunpckhbw(get_ymm(1), get_ymm(2), get_ymm(3));

            vpunpcklwd(get_ymm(2), get_ymm(4), get_ymm(0));
            vpunpckhwd(get_ymm(3), get_ymm(4), get_ymm(0));
            vpunpcklwd(get_ymm(4), get_ymm(5), get_ymm(1));
            vpunpckhwd(get_ymm(5), get_ymm(5), get_ymm(1));

            auto get_accum
                    = [&](int idx) { return get_comp_acc(idx + pass * 4); };

            if (IMPLICATION(
                        pass == 1, ncolumns > 32)) { // check against {0, 32}
                vperm2i128(get_ymm(0), get_ymm(2), get_ymm(3), perm2i128_l);
                vperm2i128(get_ymm(1), get_ymm(4), get_ymm(5), perm2i128_l);
                uni_vmovups(ptr[reg_tr_src + set_1_tr_src_offset], get_ymm(0));
                uni_vmovups(ptr[reg_tr_src + set_1_tr_src_offset + simd_w_],
                        get_ymm(1));
                if (do_compute_compensation_) {
                    vpdpbusd(get_accum(0), vmm_comp_mul, get_ymm(0),
                            VexEncoding);
                    vpdpbusd(get_accum(1), vmm_comp_mul, get_ymm(1),
                            VexEncoding);
                }
            } else if (conf_->wei_n_blk > 32) {
                uni_vmovups(ptr[reg_tr_src + set_1_tr_src_offset], vmm_zero);
                uni_vmovups(ptr[reg_tr_src + set_1_tr_src_offset + simd_w_],
                        vmm_zero);
            }

            const int set_2_tr_src_offset = set_1_tr_src_offset + n_blk_step_;
            const int upper_check = 16 + pass * 32; // check against {16, 48}
            if (ncolumns > upper_check) {
                vperm2i128(get_ymm(2), get_ymm(2), get_ymm(3), perm2i128_h);
                vperm2i128(get_ymm(3), get_ymm(4), get_ymm(5), perm2i128_h);
                uni_vmovups(ptr[reg_tr_src + set_2_tr_src_offset], get_ymm(2));
                uni_vmovups(ptr[reg_tr_src + set_2_tr_src_offset + simd_w_],
                        get_ymm(3));
                if (do_compute_compensation_) {
                    vpdpbusd(get_accum(2), vmm_comp_mul, get_ymm(2),
                            VexEncoding);
                    vpdpbusd(get_accum(3), vmm_comp_mul, get_ymm(3),
                            VexEncoding);
                }
            } else if (conf_->wei_n_blk > upper_check) {
                uni_vmovups(ptr[reg_tr_src + set_2_tr_src_offset], vmm_zero);
                uni_vmovups(ptr[reg_tr_src + set_2_tr_src_offset + simd_w_],
                        vmm_zero);
            }
        }
#endif
    }
};

template <cpu_isa_t isa>
void jit_brgemm_matmul_copy_b_int8_t<isa>::generate() {
    preamble();
    assert(!"under construction");
#if 0

    uni_vpxor(vmm_zero, vmm_zero, vmm_zero);
    mov(reg_src, ptr[param1 + GET_OFF(src)]);
    mov(reg_tr_src, ptr[param1 + GET_OFF(tr_src)]);
    mov(reg_K_iters, ptr[param1 + GET_OFF(current_K_iters)]);
    mov(reg_N_blk, ptr[param1 + GET_OFF(current_N_blk)]);

    init_permute();

    if (do_compute_compensation_) {
        int n_iters = div_up(conf_->wei_n_blk, 16) * (is_ymm_ ? 2 : 1);
        for (int i = 0; i < n_iters; i++)
            uni_vpxor(get_comp_acc(i), get_comp_acc(i), get_comp_acc(i));
        mov(imm_addr64, 1);
        uni_vpbroadcastb(vmm_comp_mul, imm_addr64.cvt8());
    }

    auto compute_K_loop = [&](bool is_N_tail) {
        const int k_unroll = 4;
        int ncolumns = is_N_tail ? conf_->N_tail : conf_->N_blk;

        Label K_loop_unrolled, K_loop_single, K_loop_tail_or_done;
        cmp(reg_K_iters, k_unroll * k_blk_step_);
        jl(K_loop_single, T_NEAR);

        L(K_loop_unrolled);
        copy_4x64(k_unroll * k_blk_step_, ncolumns);
        add(reg_src, k_unroll * k_blk_step_ * src_stride_);
        add(reg_tr_src, k_unroll * tr_src_stride_);

        sub(reg_K_iters, k_unroll * k_blk_step_);
        cmp(reg_K_iters, k_unroll * k_blk_step_);
        jge(K_loop_unrolled, T_NEAR);

        L(K_loop_single);
        cmp(reg_K_iters, k_blk_step_);
        jl(K_loop_tail_or_done, T_NEAR);

        copy_4x64(k_blk_step_, ncolumns);
        add(reg_src, k_blk_step_ * src_stride_);
        add(reg_tr_src, tr_src_stride_);

        sub(reg_K_iters, k_blk_step_);
        jmp(K_loop_single, T_NEAR);

        L(K_loop_tail_or_done);

        int k_blk_tail = conf_->K % k_blk_step_;
        if (k_blk_tail > 0) {
            Label K_loop_done;
            cmp(reg_K_iters, 0);
            jle(K_loop_done, T_NEAR);

            copy_4x64(k_blk_tail, ncolumns);
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

    if (do_compute_compensation_) {
        const bool req_s8s8_comp = conf_->s8s8_compensation_required;
        const bool req_zp_comp = conf_->has_zero_point_a;
        int n_iters = div_up(conf_->wei_n_blk, 16);
        assert(IMPLICATION(req_zp_comp,
                conf_->src_zp_type == brgemm_broadcast_t::per_tensor));

        if (req_s8s8_comp)
            mov(reg_comp_ptr, ptr[param1 + GET_OFF(compensation_ptr)]);
        if (req_zp_comp)
            mov(reg_zp_comp_ptr, ptr[param1 + GET_OFF(zp_a_compensation_ptr)]);
        mov(reg_K_start, ptr[param1 + GET_OFF(current_K_start)]);

        // YMM Note: 16 vmm registers would be needed, so only compute by halves
        const bool do_outer_unroll = req_s8s8_comp;
        const int outer_unroll = is_ymm_ && do_outer_unroll ? 2 : 1;
        const int inner_unroll = is_ymm_ && (!do_outer_unroll) ? 2 : 1;
        for (int out_ur = 0; out_ur < outer_unroll; ++out_ur) {

            // copy 'comp_acc' into s8s8_comp accumulator
            if (req_s8s8_comp) {
                for (int i = 0; i < n_iters; i++) {
                    const int accum_idx = i + out_ur * n_iters;
                    uni_vmovups(get_vmm_oscale_comp_res(i),
                            get_comp_acc(accum_idx));
                }
            }

            Label skip_acc, store;
            cmp(reg_K_start, 0);
            je(skip_acc, T_NEAR);
            if (req_s8s8_comp) {
                for (int i = 0; i < n_iters; i++) {
                    const int idx = i + out_ur * n_iters;
                    const auto vmm_acc = get_comp_acc(idx);
                    const auto vmm_res = get_vmm_oscale_comp_res(i);
                    const auto addr = !is_ymm_
                            ? EVEX_compress_addr(reg_comp_ptr, idx * simd_w_)
                            : ptr[reg_comp_ptr + idx * simd_w_];
                    uni_vpaddd(vmm_res, vmm_acc, addr);
                }
            }

            if (req_zp_comp) {
                for_(int i = 0; i < n_iters; i++)
                for (int in_ur = 0; in_ur < inner_unroll; ++in_ur) {
                    const int idx = i * inner_unroll + in_ur + out_ur * n_iters;
                    const auto vmm_acc = get_comp_acc(idx);
                    const auto vmm_res = get_vmm_zp_comp_res(idx);
                    const auto addr = !is_ymm_
                            ? EVEX_compress_addr(reg_zp_comp_ptr, idx * simd_w_)
                            : ptr[reg_zp_comp_ptr + idx * simd_w_];
                    uni_vpaddd(vmm_res, vmm_acc, addr);
                }
            }

            L(skip_acc);
            cmp(reg_K_start, rnd_up(conf_->K, conf_->K_blk) - conf_->K_blk);
            jl(store, T_NEAR);

            if (req_s8s8_comp) {
                mov(imm_addr64, 0xffffffff);
                const auto vmm_all_bits_1 = vmm_comp_mul;
                uni_vpbroadcastd(vmm_all_bits_1, imm_addr64.cvt32());
                mov(imm_addr64, 0x1);
                const auto vmm_one_s32 = vmm_zero;
                uni_vpbroadcastd(vmm_one_s32, imm_addr64.cvt32());

                for (int i = 0; i < n_iters; i++) {
                    const auto vmm_res = get_vmm_oscale_comp_res(i);
                    // multiply by 128
                    uni_vpslld(vmm_res, vmm_res, 7);
                    // change sign
                    uni_vpandnd(vmm_res, vmm_res, vmm_all_bits_1);
                    uni_vpaddd(vmm_res, vmm_res, vmm_one_s32);
                }
            }

            if (req_zp_comp) {
                mov(reg_zp_a_neg_val_ptr,
                        ptr[param1 + GET_OFF(zp_a_neg_value_ptr)]);
                const auto vmm_zp_a_neg_val = vmm_zero;
                uni_vbroadcastss(vmm_zp_a_neg_val, ptr[reg_zp_a_neg_val_ptr]);

                for_(int i = 0; i < n_iters; i++)
                for (int in_ur = 0; in_ur < inner_unroll; ++in_ur) {
                    const int idx = i * inner_unroll + in_ur + out_ur * n_iters;
                    const auto vmm_res = get_vmm_zp_comp_res(idx);
                    uni_vpmulld(vmm_res, vmm_res, vmm_zp_a_neg_val);
                }
            }

            L(store);
            if (req_s8s8_comp) {
                for (int i = 0; i < n_iters; i++) {
                    const auto vmm_res = get_vmm_oscale_comp_res(i);
                    const int idx_offset = i + out_ur * n_iters;
                    const auto addr = !is_ymm_
                            ? EVEX_compress_addr(
                                    reg_comp_ptr, idx_offset * simd_w_)
                            : ptr[reg_comp_ptr + idx_offset * simd_w_];
                    uni_vmovups(addr, vmm_res);
                }
            }
            if (req_zp_comp) {
                for_(int i = 0; i < n_iters; i++)
                for (int in_ur = 0; in_ur < inner_unroll; ++in_ur) {
                    const int idx = i * inner_unroll + in_ur + out_ur * n_iters;
                    const auto vmm_res = get_vmm_zp_comp_res(idx);
                    const auto addr = !is_ymm_
                            ? EVEX_compress_addr(reg_zp_comp_ptr, idx * simd_w_)
                            : ptr[reg_zp_comp_ptr + idx * simd_w_];
                    uni_vmovups(addr, vmm_res);
                }
            }
        }
    }
#endif
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
        , tr_src_stride_(conf_->LDB * typesize_out_) {}

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
#if 0
        mov(regw_tmp, w);
        jit_generator::kmovd(k, regw_tmp);
#endif
    }
    void copy_16_x_n_block(int nrows, int ncolumns);
    void compute_k_loop(int ncolumns);
    void generate() override;
};

void jit_brgemm_matmul_copy_b_f32_t::copy_16_x_n_block(
        int nrows, int ncolumns) {
    assert(!"under construction");
#if 0

    auto get_zmm = [](int reg_idx) {
        assert(reg_idx >= 0 && reg_idx < max_regs_available);
        return zmm(reg_idx);
    };

    auto load = [this, get_zmm](int blk, int k, int n, opmask_t current_mask) {
        auto src_zmm = get_zmm(blk);
        auto src_zmm_m = src_zmm | current_mask | T_z;
        auto addr = EVEX_compress_addr(
                reg_src, k * src_stride_ + n * typesize_in_);
        if (dt_in_ == data_type::f16)
            vcvtph2psx(src_zmm_m, addr);
        else
            vmovups(src_zmm_m, addr);
    };

    const int columns_tail = ncolumns % n_blk_step;
    const auto tail_mask = (1 << columns_tail) - 1;
    if (columns_tail < n_blk_step) kmovw(kTail, tail_mask);

    int iter = 0;
    for_(int k = 0; k < nrows; k++)
    for (int n = 0; n < conf_->wei_n_blk; n += n_blk_step) {
        const dim_t tr_src_off = k * tr_src_stride_ + n * typesize_out_;
        const auto store_addr = EVEX_compress_addr(reg_tr_src, tr_src_off);

        const int zero_padding = ncolumns - n;
        if (zero_padding <= 0) {
            vmovups(store_addr, zmm_zero);
            continue;
        }

        const opmask_t curr_msk = zero_padding < n_blk_step ? kTail : kFFFF;
        const int blk_idx = iter % max_regs_available;
        load(blk_idx, k, n, curr_msk);

        const auto src_zmm0 = get_zmm(blk_idx);
        vmovups(store_addr, src_zmm0);
        iter++;
    }
#endif
}

void jit_brgemm_matmul_copy_b_f32_t::compute_k_loop(int ncolumns) {
    assert(!"under construction");
#if 0

    auto compute_uni_k_loop = [&](int unroll) {
        Label K_start_label, K_end_label;

        L(K_start_label);
        cmp(reg_K_iters, unroll);
        jl(K_end_label, T_NEAR);

        copy_16_x_n_block(unroll, ncolumns);
        add(reg_src, unroll * src_stride_);
        add(reg_tr_src, unroll * tr_src_stride_);

        sub(reg_K_iters, unroll);
        jmp(K_start_label, T_NEAR);

        L(K_end_label);
    };

    constexpr int k_unroll = 16;
    compute_uni_k_loop(k_unroll);
    compute_uni_k_loop(1);
#endif
}

void jit_brgemm_matmul_copy_b_f32_t::generate() {
    preamble();
    assert(!"under construction");
#if 0
    vpxord(zmm_zero, zmm_zero, zmm_zero);

    mov(reg_src, ptr[param1 + GET_OFF(src)]);
    mov(reg_tr_src, ptr[param1 + GET_OFF(tr_src)]);
    mov(reg_K_iters, ptr[param1 + GET_OFF(current_K_iters)]);
    mov(reg_N_blk, ptr[param1 + GET_OFF(current_N_blk)]);
    kmovw(kFFFF, 0xffff); // 1111111111111111

    Label done;
    if (conf_->N_tail > 0) {
        Label not_N_tail;
        cmp(reg_N_blk, conf_->N_tail);
        jne(not_N_tail, T_NEAR);
        compute_k_loop(conf_->N_tail);
        jmp(done, T_NEAR);

        L(not_N_tail);
    }

    compute_k_loop(conf_->N_blk);
    L(done);
#endif

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

    // Note: for the AVX2 implementation, reserve Ymm(8) and Ymm(9) as
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
#if 0
        mov(regw_tmp, w);
        jit_generator::kmovw(k, regw_tmp);
#endif
    };

    void kmovq(Xbyak_aarch64::PReg k, size_t q) {
        assert(!"under construction");
#if 0
        mov(regq_tmp, q);
        jit_generator::kmovq(k, regq_tmp);
#endif
    };

    ZReg src_vmm(int i) {
        assert(i >= 0 && i < n_blk_step_);
        return ZReg(i);
    }

    ZReg tmp_vmm(int i) {
        // If compensation compute is required - last 6 zmms are reserved for it
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
#if 0
	vpdpbusd(v1, v2, v3,
		 mayiuse(avx512_core) ? EvexEncoding : VexEncoding);
#endif
    }
    void generate() override;
};

template <cpu_isa_t isa>
void jit_brgemm_matmul_copy_b_transposed_t<isa>::copy_row_x_col(
        int nrows, int ncolumns) {
    assert(!"under construction");
#if 0
    assert(nrows >= 0 && nrows <= n_blk_step_ && ncolumns >= 0
            && ncolumns <= k_blk_step_);
    if (!nrows) return;

    const int columns_tail
            = ncolumns % (is_bf32_ ? bf32_k_blk_step_ : k_blk_step_);
    if (columns_tail > 0) {
        const int dt_step
                = (is_bf32_ || conf_->isa == avx512_core_fp16) ? 1 : typesize_;
        const auto tail_mask
                = size_t(((size_t)1 << dt_step * columns_tail) - 1);
        if (is_bf32_)
            kmovw(kTail, tail_mask);
        else
            kmovq(kTail, tail_mask);
    }

    auto load_bf32 = [this, nrows, columns_tail, ncolumns](int i) {
        auto src_reg = src_vmm(i);
        auto src_reg_next = tmp_vmm(i);

        if (i >= nrows) {
            vpxord(src_reg, src_reg, src_reg);
            return;
        }

        // check if k_tail exists and it's in the first zmm
        auto zmm_src = columns_tail > 0 && ncolumns < bf32_k_blk_step_
                ? src_reg | kTail | T_z
                : src_reg;
        vmovups(zmm_src, EVEX_compress_addr(reg_src, i * src_stride_));

        if (ncolumns <= bf32_k_blk_step_) {
            vpxord(src_reg_next, src_reg_next, src_reg_next);
        } else {
            auto zmm_src_next = columns_tail > 0 ? src_reg_next | kTail | T_z
                                                 : src_reg_next;
            vmovups(zmm_src_next,
                    EVEX_compress_addr(reg_src,
                            i * src_stride_ + bf32_k_blk_step_ * typesize_));
        }

        vcvtne2ps2bf16(src_reg, src_reg_next, src_reg);
    };

    auto load = [this, nrows, columns_tail](int i) {
        auto src_reg = src_vmm(i);
        if (i >= nrows) {
            vpxord(src_reg, src_reg, src_reg);
            return;
        }

        auto src_load = columns_tail > 0 ? src_reg | kTail | T_z : src_reg;
        const auto addr = EVEX_compress_addr(reg_src, i * src_stride_);
        if (conf_->isa == avx512_core_fp16)
            vcvtph2psx(src_load, addr);
        else
            vmovdqu8(src_load, addr);
    };

    auto store = [this](Zmm r, int i) {
        auto addr = EVEX_compress_addr(reg_tr_src, i * tr_src_stride_);
        vmovups(addr, r);
    };

    auto transpose16x8 = [&](int base_idx) {
        assert(base_idx == 0 || base_idx == 8);
        // If compensation compute is required - use tmp(0) ... tmp(7)
        // to not spoil reserved registers' values
        const int tmp_corr_idx = do_compute_compensation_ * base_idx;

        // swap 1
        if (is_bf32_) {
            for (int i = 0; i < 4; i++) {
                const int src_idx0 = base_idx + i * 2;
                const int src_idx1 = src_idx0 + 1;

                if (base_idx == 0 && i == 0) {
                    load_bf32(src_idx0);
                    load_bf32(src_idx1);
                }

                const int next_src_idx0 = src_idx0 + 2;
                const int next_src_idx1 = src_idx1 + 2;

                const bool load_next = base_idx == 0 || i < 3;

                const auto tmp0 = tmp_vmm(src_idx0 - tmp_corr_idx);
                const auto tmp1 = tmp_vmm(src_idx1 - tmp_corr_idx);
                const auto src0 = src_vmm(src_idx0);
                const auto src1 = src_vmm(src_idx1);

                if (next_src_idx0 < nrows && load_next)
                    load_bf32(next_src_idx0);
                valignd(tmp0, src0, src0, 0x1);

                if (next_src_idx1 < nrows && load_next)
                    load_bf32(next_src_idx1);
                valignd(tmp1, src1, src1, 0xf);

                vmovaps(src0 | kAAAA, tmp1);
                vmovaps(src1 | k5555, tmp0);
            }
        } else {
            for (int i = 0; i < 4; i++) {
                const int src_idx0 = base_idx + i * 2;
                const int src_idx1 = src_idx0 + 1;

                const int next_src_idx0 = src_idx0 + 2;
                const int next_src_idx1 = src_idx1 + 2;
                const bool load_next = base_idx == 0 || i < 3;

                if (base_idx == 0 && i == 0) {
                    load(src_idx0);
                    load(src_idx1);
                }

                const auto tmp0 = tmp_vmm(src_idx0 - tmp_corr_idx);
                const auto tmp1 = tmp_vmm(src_idx1 - tmp_corr_idx);
                const auto src0 = src_vmm(src_idx0);
                const auto src1 = src_vmm(src_idx1);

                if (next_src_idx0 < nrows && load_next) load(next_src_idx0);
                valignd(tmp0, src0, src0, 0x1);

                if (next_src_idx1 < nrows && load_next) load(next_src_idx1);
                valignd(tmp1, src1, src1, 0xf);

                vmovaps(src0 | kAAAA, tmp1);
                vmovaps(src1 | k5555, tmp0);
            }
        }
        // swap 2
        for (int i = 0; i < 4; i++) {
            const int select_half = (i < 2) ? 0 : 2;
            const int src_idx0 = base_idx + i + select_half + 0;
            const int src_idx2 = src_idx0 + 2;

            const auto tmp0 = tmp_vmm(src_idx0 - tmp_corr_idx);
            const auto tmp1 = tmp_vmm(src_idx2 - tmp_corr_idx);
            const auto src0 = src_vmm(src_idx0);
            const auto src2 = src_vmm(src_idx2);

            valignd(tmp0, src0, src0, 0x2);
            valignd(tmp1, src2, src2, 0xe);
            vmovaps(src2 | k3333, tmp0);
            vmovaps(src0 | kCCCC, tmp1);
        }
        // swap 4
        for (int i = 0; i < 4; i++) {
            const int src_idx0 = base_idx + i;
            const int src_idx4 = src_idx0 + 4;

            const auto tmp0 = tmp_vmm(src_idx0 - tmp_corr_idx);
            const auto src0 = src_vmm(src_idx0);
            const auto src4 = src_vmm(src_idx4);

            vmovaps(tmp0, src0);
            vshuff32x4(src0 | kF0F0, src4, src4, 0xb1);
            vshuff32x4(src4 | k0F0F, tmp0, tmp0, 0xb1);
        }
    };

    auto fixup16x16 = [&]() {
        for (int i = 0; i < 8; i++) {
            const auto tmp = tmp_vmm(i);
            const auto src0 = src_vmm(i);
            const auto src8 = src_vmm(8 + i);
            vshuff64x2(tmp, src0, src8, 0x44);
            if (do_compute_compensation_)
                dot_product(vmm_comp_acc, vmm_comp_mul, tmp);
            store(tmp, i);
        }

        for (int i = 0; i < 8; i++) {
            const auto tmp = tmp_vmm(i);
            const auto src0 = src_vmm(i);
            const auto src8 = src_vmm(8 + i);
            vshuff64x2(tmp, src0, src8, 0xee);
            if (do_compute_compensation_)
                dot_product(vmm_comp_acc, vmm_comp_mul, tmp);
            store(tmp, 8 + i);
        }
    };

    transpose16x8(0);
    transpose16x8(8);
    fixup16x16();
#endif
}

template <>
void jit_brgemm_matmul_copy_b_transposed_t<sve_256>::copy_row_x_col(
        int nrows, int ncolumns) {
    assert(!"under construction");
#if 0
    assert(nrows >= 0 && nrows <= n_blk_step_ && ncolumns >= 0
            && ncolumns <= k_blk_step_);
    if (!nrows) return;

    const int columns_tail = ncolumns % k_blk_step_;
    auto load = [this, nrows, columns_tail](int i) {
        auto vmm_src = src_vmm(i);
        if (i >= nrows) {
            uni_vpxor(vmm_src, vmm_src, vmm_src);
            return;
        }
        if (columns_tail > 0) {
            load_bytes(vmm_src, reg_src, i * src_stride_,
                    columns_tail * typesize_);
        } else
            uni_vmovups(vmm_src, ptr[reg_src + i * src_stride_]);
    };

    // swap 1
    for (int i = 0; i < 4; ++i) {
        const int src_idx0 = i * 2;
        const int src_idx1 = src_idx0 + 1;

        const int next_src_idx0 = src_idx0 + 2;
        const int next_src_idx1 = src_idx1 + 2;
        const bool load_next = i < 3;

        if (i == 0) {
            load(src_idx0);
            load(src_idx1);
        }

        const auto tmp0 = tmp_vmm(0);
        const auto tmp1 = tmp_vmm(1);
        const auto src0 = src_vmm(src_idx0);
        const auto src1 = src_vmm(src_idx1);

        if (next_src_idx0 < nrows && load_next) { load(next_src_idx0); }
        vperm2i128(tmp0, src0, src0, 0x1);
        vpalignr(tmp0, tmp0, src0, 0x4);

        if (next_src_idx1 < nrows && load_next) { load(next_src_idx1); }
        vperm2i128(tmp1, src1, src1, 0x1);
        vpalignr(tmp1, src1, tmp1, 0xC);

        vpblendd(src0, src0, tmp1, 0xAA);
        vpblendd(src1, src1, tmp0, 0x55);
    }
    // swap 2
    for (int i = 0; i < 4; ++i) {
        const int select_half = (i < 2) ? 0 : 2;
        const int src_idx0 = i + select_half;
        const int src_idx2 = src_idx0 + 2;

        const auto tmp0 = tmp_vmm(0);
        const auto tmp1 = tmp_vmm(1);
        const auto src0 = src_vmm(src_idx0);
        const auto src2 = src_vmm(src_idx2);

        vperm2i128(tmp0, src0, src0, 0x1);
        vpalignr(tmp0, tmp0, src0, 0x8);

        vperm2i128(tmp1, src2, src2, 0x1);
        vpalignr(tmp1, src2, tmp1, 0x8);

        vpblendd(src2, src2, tmp0, 0x33);
        vpblendd(src0, src0, tmp1, 0xCC);
    }
    // swap 4
    for (int i = 0; i < 4; ++i) {
        const int src_idx0 = i;
        const int src_idx4 = src_idx0 + 4;

        const auto tmp0 = tmp_vmm(0);
        const auto tmp4 = tmp_vmm(1);
        const auto src0 = src_vmm(src_idx0);
        const auto src4 = src_vmm(src_idx4);

        vperm2i128(tmp4, src4, src4, 0x01);
        vperm2i128(tmp0, src0, src0, 0x01);
        vpblendd(src0, src0, tmp4, 0xF0);
        vpblendd(src4, src4, tmp0, 0x0F);
    }
    // swap 8
    for (int i = 0; i < 8; i++) {
        const auto src0 = src_vmm(i);
        if (do_compute_compensation_)
            dot_product(vmm_comp_acc, vmm_comp_mul, src0);
        uni_vmovups(ptr[reg_tr_src + i * tr_src_stride_], src0);
    }
#endif
}

template <cpu_isa_t isa>
void jit_brgemm_matmul_copy_b_transposed_t<isa>::compute_K_loop(bool is_N_tail,
        int curr_K_tail, bool is_first_K_iter, bool is_last_K_iter) {
    assert(!"under construction");
#if 0
    MAYBE_UNUSED(is_first_K_iter);
    MAYBE_UNUSED(is_last_K_iter);
    const int N_chunk_tail = conf_->N % n_blk_step_;
    const int nrows = is_N_tail ? N_chunk_tail : n_blk_step_;
    if (do_compute_compensation_)
        uni_vpxor(vmm_comp_acc, vmm_comp_acc, vmm_comp_acc);

    Label K_loop, K_loop_tail_or_done;
    mov(reg_K_iters, ptr[param1 + GET_OFF(current_K_iters)]);

    mov(reg_src, reg_src_base);
    mov(reg_tr_src, reg_tr_src_base);
    if (curr_K_tail > 0) {
        cmp(reg_K_iters, k_blk_step_);
        jl(K_loop_tail_or_done, T_NEAR);
    }

    L(K_loop);
    copy_row_x_col(nrows, k_blk_step_);
    add(reg_src, k_blk_step_ * typesize_);
    add(reg_tr_src, k_blk_step_ / vnni_granularity_ * tr_src_stride_);

    sub(reg_K_iters, k_blk_step_);
    cmp(reg_K_iters, k_blk_step_);
    jge(K_loop, T_NEAR);

    L(K_loop_tail_or_done);

    if (curr_K_tail > 0) copy_row_x_col(nrows, curr_K_tail);

    if (req_s8s8_comp_) {
        const auto addr = ptr[reg_comp_ptr];
        if (!is_first_K_iter)
            uni_vpaddd(vmm_s8s8_comp_acc, vmm_comp_acc, addr);
        else
            uni_vmovups(vmm_s8s8_comp_acc, vmm_comp_acc);

        if (is_last_K_iter) {
            // multiply by 128
            uni_vpslld(vmm_s8s8_comp_acc, vmm_s8s8_comp_acc, 7);
            // change sign
            uni_vpandnd(vmm_s8s8_comp_acc, vmm_s8s8_comp_acc, vmm_all_bits_1);
            uni_vpaddd(vmm_s8s8_comp_acc, vmm_s8s8_comp_acc, vmm_one_s32);
        }
        uni_vmovups(addr, vmm_s8s8_comp_acc);
    }
    if (req_zp_comp_) {
        const auto addr = ptr[reg_zp_comp_ptr];
        if (!is_first_K_iter) vpaddd(vmm_comp_acc, vmm_comp_acc, addr);
        if (is_last_K_iter)
            uni_vpmulld(vmm_comp_acc, vmm_comp_acc, vmm_zp_a_neg_val);
        uni_vmovups(addr, vmm_comp_acc);
    }
#endif
}

template <cpu_isa_t isa>
void jit_brgemm_matmul_copy_b_transposed_t<isa>::compute_N_loop(
        int curr_K_tail, bool is_first_K_iter, bool is_last_K_iter) {
    assert(!"under construction");
#if 0
    const int N_chunk_tail = conf_->N % n_blk_step_;

    Label N_loop, N_loop_tail_or_done;
    if (N_chunk_tail > 0) {
        cmp(reg_N_iters, n_blk_step_);
        jl(N_loop_tail_or_done, T_NEAR);
    }

    L(N_loop);
    compute_K_loop(false, curr_K_tail, is_first_K_iter, is_last_K_iter);
    add(reg_src_base, n_blk_step_ * src_stride_);
    add(reg_tr_src_base, n_blk_step_ * vnni_granularity_ * tr_typesize_);

    if (req_zp_comp_) add(reg_zp_comp_ptr, comp_shift_);
    if (req_s8s8_comp_) add(reg_comp_ptr, comp_shift_);

    sub(reg_N_iters, n_blk_step_);
    cmp(reg_N_iters, n_blk_step_);
    jge(N_loop, T_NEAR);

    L(N_loop_tail_or_done);
    if (N_chunk_tail > 0) {
        Label N_loop_done;
        cmp(reg_N_iters, 0);
        jle(N_loop_done, T_NEAR);

        compute_K_loop(true, curr_K_tail, is_first_K_iter, is_last_K_iter);
        L(N_loop_done);
    }
#endif
}

template <cpu_isa_t isa>
void jit_brgemm_matmul_copy_b_transposed_t<isa>::generate() {

    preamble();
    assert(!"under construction");
#if 0

    if (avx512_core_dot_product_) {
        mov(regq_tmp.cvt16(), 1);
        vpbroadcastw(vmm_ones_words, regq_tmp.cvt16());
    }

    mov(reg_src_base, ptr[param1 + GET_OFF(src)]);
    mov(reg_tr_src_base, ptr[param1 + GET_OFF(tr_src)]);
    mov(reg_K_iters, ptr[param1 + GET_OFF(current_K_iters)]);
    mov(reg_N_iters, ptr[param1 + GET_OFF(current_N_blk)]);

    if (!is_ymm_) {
        kmovw(k5555, 0x5555);
        kmovw(kAAAA, 0xaaaa);
        kmovw(k3333, 0x3333);
        kmovw(kCCCC, 0xcccc);
        kmovw(k0F0F, 0x0f0f);
        kmovw(kF0F0, 0xf0f0);
    }

    const dim_t N_chunk_elems = conf_->N_chunk_elems;
    assert(N_chunk_elems % n_blk_step_ == 0 || N_chunk_elems == conf_->N);
    UNUSED(N_chunk_elems);

    const auto K_blk_tail = nstl::min(conf_->K, conf_->K_blk) % k_blk_step_;
    const auto K_tail_tail = (conf_->K % conf_->K_blk) % k_blk_step_;

    auto compute_body = [&](bool is_first_K_iter, bool is_last_K_iter) {
        if (is_last_K_iter) {
            if (req_s8s8_comp_) {
                mov(imm_addr64, 0xffffffff);
                uni_vpbroadcastd(vmm_all_bits_1, imm_addr64.cvt32());
                mov(imm_addr64, 0x1);
                uni_vpbroadcastd(vmm_one_s32, imm_addr64.cvt32());
            }
            if (req_zp_comp_) {
                mov(reg_zp_a_neg_val_ptr,
                        ptr[param1 + GET_OFF(zp_a_neg_value_ptr)]);
                uni_vbroadcastss(vmm_zp_a_neg_val, ptr[reg_zp_a_neg_val_ptr]);
            }
        }

        Label compute_body_done;
        if (conf_->K_tail > 0 && K_blk_tail != K_tail_tail) {
            Label not_K_tail;
            cmp(reg_K_iters, conf_->K_blk);
            je(not_K_tail, T_NEAR);
            compute_N_loop(K_tail_tail, is_first_K_iter, is_last_K_iter);
            jmp(compute_body_done, T_NEAR);

            L(not_K_tail);
        }

        compute_N_loop(K_blk_tail, is_first_K_iter, is_last_K_iter);
        L(compute_body_done);
    };

    Label done;
    if (do_compute_compensation_) {
        assert(IMPLICATION(req_zp_comp_,
                conf_->src_zp_type == brgemm_broadcast_t::per_tensor));

        mov(reg_K_start, ptr[param1 + GET_OFF(current_K_start)]);
        if (req_s8s8_comp_)
            mov(reg_comp_ptr, ptr[param1 + GET_OFF(compensation_ptr)]);
        if (req_zp_comp_)
            mov(reg_zp_comp_ptr, ptr[param1 + GET_OFF(zp_a_compensation_ptr)]);

        mov(regq_tmp, 1);
        uni_vpbroadcastb(vmm_comp_mul, regq_tmp.cvt8());

        const auto last_K_threshold
                = rnd_up(conf_->K, conf_->K_blk) - conf_->K_blk;
        Label not_first, not_first_not_last;
        cmp(reg_K_start, 0);
        jne(not_first, T_NEAR);
        {
            // first K iteration
            Label first_not_last;
            cmp(reg_K_start, last_K_threshold);
            jl(first_not_last, T_NEAR);
            compute_body(true, true);
            jmp(done, T_NEAR);

            L(first_not_last);
            compute_body(true, false);
            jmp(done, T_NEAR);
        }

        L(not_first);
        cmp(reg_K_start, last_K_threshold);
        jl(not_first_not_last, T_NEAR);

        compute_body(false, true);
        jmp(done, T_NEAR);
        L(not_first_not_last);
    }

    compute_body(false, false);
    L(done);
#endif

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
    // Note: f16 support through avx512_core_fp16 sets src_dt and wei_dt as f32
    // to imply upconverting. So, the assumption is `is_f1`6 below evaluates to
    // `false` on avx512_core_fp16.
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
