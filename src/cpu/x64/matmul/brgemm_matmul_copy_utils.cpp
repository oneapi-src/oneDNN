/*******************************************************************************
* Copyright 2021-2024 Intel Corporation
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

template <typename Vmm>
struct jit_brgemm_matmul_copy_a_impl_t : public jit_brgemm_matmul_copy_a_t,
                                         public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_brgemm_matmul_copy_a_impl_t)

    jit_brgemm_matmul_copy_a_impl_t(const brgemm_matmul_conf_t *conf)
        : jit_brgemm_matmul_copy_a_t(conf)
        , jit_generator(jit_name())
        , typesize_(conf_->a_dt_sz)
        , tr_typesize_(conf_->tr_a_dt_sz)
        , vnni_granularity_(data_type_vnni_granularity(conf_->src_dt))
        , k_step_(vlen_ / nstl::max(typesize_, tr_typesize_))
        , src_stride_(conf_->copy_A_src_stride)
        , tr_src_stride_((conf_->use_buffer_a_tail_only
                                         ? static_cast<dim_t>(conf_->wei_k_blk)
                                         : conf_->LDA)
                  * tr_typesize_)
        , do_compute_compensation_(
                  conf_->has_zero_point_b && !conf_->with_wei_decompression)
        , avx512_core_dot_product_(
                  do_compute_compensation_ && !isa_has_int8_vnni(conf->isa))
        // See the note in `create_brgemm_matmul_copy_b` why `orig_src_dt` used.
        , use_fp16_instructions_(conf_->isa == avx512_core_fp16
                  && conf_->orig_src_dt == data_type::f16
                  && conf_->src_dt == data_type::f32)
        , k_loop_unroll_(is_ymm_ ? 7 : 16)
        , vmm_copy_idx_(is_ymm_                      ? 13
                          : avx512_core_dot_product_ ? 27
                                                     : 29) {}

    void operator()(ctx_t *ctx) override { jit_generator::operator()(ctx); }
    status_t create_kernel() override { return jit_generator::create_kernel(); }

private:
    using reg64_t = const Xbyak::Reg64;
    using reg32_t = const Xbyak::Reg32;
    using opmask_t = const Xbyak::Opmask;

    static constexpr int vlen_ = vreg_traits<Vmm>::vlen;
    static constexpr bool is_ymm_ = std::is_same<Vmm, Xbyak::Ymm>::value;
    static constexpr int num_comp_acc_ = is_ymm_ ? 7 : 8;

    const int typesize_;
    const int tr_typesize_;
    const int vnni_granularity_;
    const int k_step_;
    const dim_t src_stride_;
    const dim_t tr_src_stride_;
    const bool do_compute_compensation_;
    const bool avx512_core_dot_product_;
    const bool use_fp16_instructions_;

    const int k_loop_unroll_;
    const int vmm_copy_idx_;

    opmask_t kTail_load = k7;
    opmask_t kTail_store = k6;
    opmask_t kTail_comp = k5;

    reg64_t reg_src = rax;
    reg64_t reg_tr_src = rbx;
    reg64_t reg_K_start = abi_not_param1;

    reg64_t reg_zp_comp_buf_ptr = rdx;
    reg64_t reg_zp_comp_res_ptr = rsi;

    reg64_t reg_M_blk = r9;
    reg64_t reg_K_blk = r10;
    reg64_t reg_batch = r11;
    reg64_t reg_aux_src = r12;
    reg64_t reg_aux_tr_src = r13;
    reg64_t regq_tmp = r14;
    reg64_t imm_addr64 = r15;
    reg64_t reg_zp_ab_comp_ptr = imm_addr64;
    reg64_t reg_zp_b_neg_val_ptr = reg_K_blk;

    // Required in every dot product for INT8 non-VNNI computation.
    Vmm vmm_ones_words = Vmm(28);
    Vmm vmm_dot_product_temp = Vmm(29);

    Vmm vmm_comp_mul = Vmm(is_ymm_ ? 14 : 30); // 1s
    Vmm vmm_comp_add = Vmm(is_ymm_ ? 15 : 31); // 128

    // Allows to shift A data by 128 for s8s8 problem for AVX512 in copy
    // routine, not in compute kernel. It's disabled for now, as it
    // requires setting some hint to brgemm kernel to avoid double shifting
    const bool allow_input_shift_for_s8s8 = false;

    Vmm get_vmm_comp_acc(int i) {
        assert(i >= 0 && i < num_comp_acc_);
        return Vmm(i);
    }

    Vmm get_vmm_copy(int i) {
        assert(i >= 0 && i < k_loop_unroll_);
        return Vmm(vmm_copy_idx_ - i);
    }

    void load_vmm(int idx, int offset) {}
    void store_vmm(int idx, int offset) {}
    void load_tail(int k_tail, size_t offset) {}
    void store_tail(int k_tail, size_t offset) {}
    void reduce_compensation_across_accumulators(int num_accumulators);
    void copy_K_loop(bool is_K_tail, bool is_first_K_iter, bool is_last_K_iter);
    void copy_M_loop(bool is_K_tail, bool is_first_K_iter, bool is_last_K_iter);
    inline void dot_product(Vmm v1, Vmm v2, Vmm v3) {
        if (!avx512_core_dot_product_)
            vpdpbusd(v1, v2, v3,
                    mayiuse(avx512_core) ? EvexEncoding : VexEncoding);
        else {
            vpmaddubsw(vmm_dot_product_temp, v2, v3);
            vpmaddwd(
                    vmm_dot_product_temp, vmm_dot_product_temp, vmm_ones_words);
            vpaddd(v1, v1, vmm_dot_product_temp);
        }
    }
    void generate() override;
};

template <>
void jit_brgemm_matmul_copy_a_impl_t<Zmm>::load_vmm(int idx, int offset) {
    const auto addr = EVEX_compress_addr(reg_src, offset);
    if (use_fp16_instructions_) {
        vcvtph2psx(get_vmm_copy(idx), addr);
    } else {
        vmovdqu8(get_vmm_copy(idx), addr);
    }
}

template <>
void jit_brgemm_matmul_copy_a_impl_t<Ymm>::load_vmm(int idx, int offset) {
    uni_vmovups(get_vmm_copy(idx), ptr[reg_src + offset]);
}

template <>
void jit_brgemm_matmul_copy_a_impl_t<Zmm>::store_vmm(int idx, int offset) {
    auto tr_src_addr = EVEX_compress_addr(reg_tr_src, offset);
    vmovdqu8(tr_src_addr, get_vmm_copy(idx));
}

template <>
void jit_brgemm_matmul_copy_a_impl_t<Ymm>::store_vmm(int idx, int offset) {
    uni_vmovups(ptr[reg_tr_src + offset], get_vmm_copy(idx));
}

template <>
void jit_brgemm_matmul_copy_a_impl_t<Zmm>::load_tail(
        int k_tail, size_t offset) {
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
            = conf_->is_bf32 || use_fp16_instructions_ ? 1 : typesize_;
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
    else if (use_fp16_instructions_)
        vcvtph2psx(zmm_tail, load_addr);
    else
        vmovdqu8(zmm_tail, load_addr);
}

template <>
void jit_brgemm_matmul_copy_a_impl_t<Ymm>::load_tail(
        int k_tail, size_t offset) {
    const auto vmm_tail = get_vmm_copy(0);
    load_bytes(vmm_tail, reg_src, offset * typesize_, k_tail * typesize_);
}

template <>
void jit_brgemm_matmul_copy_a_impl_t<Zmm>::store_tail(
        int k_tail, size_t offset) {
    auto tr_src_addr = EVEX_compress_addr(reg_tr_src, offset * tr_typesize_);
    if (conf_->is_bf32) {
        Ymm ymm_downcvt_bf16 = Ymm(get_vmm_copy(0).getIdx());
        vcvtneps2bf16(ymm_downcvt_bf16, get_vmm_copy(0));
        vmovdqu16(tr_src_addr, ymm_downcvt_bf16 | kTail_store);
    } else if (use_fp16_instructions_) {
        vmovups(tr_src_addr, get_vmm_copy(0) | kTail_store);
    } else
        vmovdqu8(tr_src_addr, get_vmm_copy(0) | kTail_store);
}

template <>
void jit_brgemm_matmul_copy_a_impl_t<Ymm>::store_tail(
        int k_tail, size_t offset) {
    const int k_tail_st = rnd_up(k_tail, vnni_granularity_);
    const auto vmm_tail = get_vmm_copy(0);
    store_bytes(
            vmm_tail, reg_tr_src, offset * tr_typesize_, k_tail_st * typesize_);
}

template <typename Vmm>
void jit_brgemm_matmul_copy_a_impl_t<
        Vmm>::reduce_compensation_across_accumulators(int num_accumulators) {
    int num = num_accumulators;
    while (num > 1) {
        for (int i = 0; i < num / 2; i++) {
            const auto vmm_acc0 = get_vmm_comp_acc(i);
            const auto vmm_acc1 = get_vmm_comp_acc(div_up(num, 2) + i);
            uni_vpaddd(vmm_acc0, vmm_acc0, vmm_acc1);
        }
        num = div_up(num, 2);
    }
}

template <typename Vmm>
void jit_brgemm_matmul_copy_a_impl_t<Vmm>::copy_K_loop(
        bool is_K_tail, bool is_first_K_iter, bool is_last_K_iter) {
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

    auto maybe_compute_compensation = [this, num_acc](int k_idx, Vmm vmm_copy) {
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
}

template <typename Vmm>
void jit_brgemm_matmul_copy_a_impl_t<Vmm>::copy_M_loop(
        bool is_K_tail, bool is_first_K_iter, bool is_last_K_iter) {

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
}

template <typename Vmm>
void jit_brgemm_matmul_copy_a_impl_t<Vmm>::generate() {
    preamble();

    if (avx512_core_dot_product_) {
        mov(regq_tmp.cvt16(), 1);
        vpbroadcastw(vmm_ones_words, regq_tmp.cvt16());
    }

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

    postamble();
}

template struct jit_brgemm_matmul_copy_a_impl_t<Zmm>;
template struct jit_brgemm_matmul_copy_a_impl_t<Ymm>;

template <typename Vmm>
struct jit_brgemm_matmul_copy_a_transposed_impl_t
    : public jit_brgemm_matmul_copy_a_t,
      public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_brgemm_matmul_copy_a_transposed_impl_t)

    jit_brgemm_matmul_copy_a_transposed_impl_t(const brgemm_matmul_conf_t *conf)
        : jit_brgemm_matmul_copy_a_t(conf)
        , jit_generator(jit_name())
        , typesize(conf_->a_dt_sz)
        , tr_typesize(conf_->tr_a_dt_sz)
        , rows_step(16)
        , columns_step(rows_step)
        , src_stride(conf_->copy_A_src_stride)
        , dst_stride(conf_->LDA * tr_typesize)
        , m_loop_src_shift(columns_step * typesize)
        , m_loop_dst_shift(columns_step * dst_stride)
        , k_loop_src_shift(rows_step * src_stride)
        , k_loop_dst_shift(rows_step * tr_typesize)
        , is_f32(everyone_is(data_type::f32, conf_->src_dt, conf_->wei_dt))
        , is_bf32(conf_->is_bf32)
        , is_dynamic_src_ld(conf_->is_runtime_M)
        // See the note in `create_brgemm_matmul_copy_b` why `orig_src_dt` used.
        , use_fp16_instructions_(conf_->isa == avx512_core_fp16
                  && conf_->orig_src_dt == data_type::f16
                  && conf_->src_dt == data_type::f32) {}

    void operator()(ctx_t *ctx) override { jit_generator::operator()(ctx); }
    status_t create_kernel() override { return jit_generator::create_kernel(); }

private:
    using reg64_t = const Xbyak::Reg64;
    using reg32_t = const Xbyak::Reg32;
    using opmask_t = const Xbyak::Opmask;

    const size_t typesize;
    const size_t tr_typesize;
    const int rows_step;
    const int columns_step;
    const dim_t src_stride, dst_stride;
    const dim_t m_loop_src_shift;
    const dim_t m_loop_dst_shift;
    const dim_t k_loop_src_shift;
    const dim_t k_loop_dst_shift;
    const bool is_f32;
    const bool is_bf32;
    const bool is_dynamic_src_ld;
    const bool use_fp16_instructions_;

    opmask_t kFFFF = k1;
    opmask_t k3333 = k1;
    opmask_t k5555 = k2;
    opmask_t kAAAA = k3;
    opmask_t kAA = k4;
    opmask_t kCCCC = k4;
    opmask_t k55 = k5;
    opmask_t k0F0F = k5;
    opmask_t kCC = k6;
    opmask_t kF0F0 = k6;
    opmask_t k33 = k7;
    opmask_t kTail = is_f32 ? k7 : k1;

    reg64_t regq_tmp = r15;
    reg32_t regw_tmp = regq_tmp.cvt32();
    reg64_t reg_k_src = r14;
    reg64_t reg_k_dst = r13;
    reg64_t reg_m_src = r12;
    reg64_t reg_m_dst = r11;
    reg64_t reg_aux_src0 = r10;
    reg64_t reg_aux_src1 = r9;
    reg64_t reg_loop_k = rax;
    reg64_t reg_loop_m = rbx;
    reg64_t imm_addr64 = rdx;
    // Note: this must be assigned to rcx as it's used in shl instruction,
    // clashes with abi_param1 on Windows OS
    reg64_t reg_opmask_shift_compute = rcx;

    Xbyak::Zmm vidx1 = zmm31;
    Xbyak::Zmm vidx2 = zmm30;
    Xbyak::Zmm vidx3 = zmm29;
    Xbyak::Zmm vidx4 = zmm28;
    Xbyak::Zmm vidx5 = zmm27;
    Xbyak::Zmm zmm_tmp = zmm26;

    constexpr static int current_M_blk_offt_ = 0;
    constexpr static int src_offt_ = 8;
    constexpr static int tr_src_offt_ = 16;
    constexpr static int current_K_blk_offt_ = 24;
    constexpr static int dynamic_src_ld_offt_ = 32;
    constexpr static int dynamic_src_ld_x_2_offt_ = 40;
    constexpr static int dynamic_src_ld_x_kstep_offt_ = 48;
    constexpr static int stack_space_needed_ = 56;

    void vmovdqa64(Vmm v, const int64_t *addr) {
        mov(imm_addr64, reinterpret_cast<size_t>(addr));
        jit_generator::vmovdqa64(v, ptr[imm_addr64]);
    }

    void vmovdqa32(Vmm v, const int32_t *addr) {
        mov(imm_addr64, reinterpret_cast<size_t>(addr));
        jit_generator::vmovdqa32(v, ptr[imm_addr64]);
    }

    void kmovw(Opmask mask_reg, size_t mask) {
        mov(regw_tmp, mask);
        jit_generator::kmovw(mask_reg, regw_tmp);
    }

    void transpose_f32(reg64_t dst, reg64_t src, int nrows, int ncolumns);
    void transpose_bf16(reg64_t dst, reg64_t src, int nrows, int ncolumns);
    void deploy_transpose(reg64_t dst, reg64_t src, int nrows, int ncolumns);
    void init_masks();
    void generate() override;
};

template <>
void jit_brgemm_matmul_copy_a_transposed_impl_t<Xbyak::Zmm>::transpose_bf16(
        reg64_t dst, reg64_t src, int nrows, int ncolumns) {
    assert(nrows >= 0 && nrows <= rows_step && ncolumns >= 0
            && ncolumns <= columns_step);
    if (!nrows) return;

    auto src_zmm = [](int i) { return Zmm(i); };

    auto src_ymm = [](int i) {
        assert(i >= 0 && i < 16);
        return Ymm(i);
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

    auto get_vec_idx = [this](int col_idx) {
        MAYBE_UNUSED(this);
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
}

template <typename Vmm>
void jit_brgemm_matmul_copy_a_transposed_impl_t<Vmm>::transpose_bf16(
        reg64_t dst, reg64_t src, int nrows, int ncolumns) {
    assert(!"unsupported transpose_bf16 copy_a_transposed_impl");
}

template <typename Vmm>
void jit_brgemm_matmul_copy_a_transposed_impl_t<Vmm>::transpose_f32(
        reg64_t reg_dst, reg64_t reg_src, int nrows, int ncolumns) {
    assert(!"unsupported transpose_f32 copy_a_transposed_impl");
}

template <>
void jit_brgemm_matmul_copy_a_transposed_impl_t<Xbyak::Ymm>::transpose_f32(
        reg64_t reg_dst, reg64_t reg_src, int nrows, int ncolumns) {
    Ymm ymm_tail_mask = ymm15;
    Ymm ymm_upper_tail_mask = ymm14;
    Xmm xmm_upper_tail_mask = xmm14;
    Ymm ymm_tmp = ymm13;

    // avx2 transpose is 8x8, but we need 16x16 transpose. We use four 8x8
    // transposes as below.
    // _    _T       _      _
    // |A, B|   =>   |At, Ct|
    // |C, D|        |Bt, Dt|

    constexpr int avx2_transpose_size = 8;
    const int tail_size = ncolumns % avx2_transpose_size;
    if (tail_size > 0) {
        Xbyak::Reg64 reg_tmp = regq_tmp;
        init_f32_avx2_mask_ymm(ymm_tail_mask, reg_tmp, tail_size);
        const int upper_xmm_tail_size = tail_size - 4;
        if (upper_xmm_tail_size > 0)
            init_f32_avx2_mask_ymm(
                    ymm_upper_tail_mask, reg_tmp, upper_xmm_tail_size);
    }

    const int A_rows = nstl::min(avx2_transpose_size, nrows);
    const int A_columns = nstl::min(avx2_transpose_size, ncolumns);
    jit_generator::transpose(reg_src, reg_dst, src_stride, dst_stride, A_rows,
            A_columns, data_type::f32, ymm_tmp, ymm_tail_mask,
            xmm_upper_tail_mask);
    if (rows_step <= 8) return;

    const dim_t src_B_offset = sizeof(float) * avx2_transpose_size;
    const dim_t dst_B_offset = dst_stride * avx2_transpose_size;
    const int B_rows = nstl::min(avx2_transpose_size, nrows);
    const int B_columns = nstl::max(ncolumns - avx2_transpose_size, 0);
    add(reg_src, src_B_offset);
    add(reg_dst, dst_B_offset);
    jit_generator::transpose(reg_src, reg_dst, src_stride, dst_stride, B_rows,
            B_columns, data_type::f32, ymm_tmp, ymm_tail_mask,
            xmm_upper_tail_mask);

    const dim_t src_C_offset = src_stride * avx2_transpose_size;
    const dim_t dst_C_offset = sizeof(float) * avx2_transpose_size;
    const int C_rows = nstl::max(nrows - avx2_transpose_size, 0);
    const int C_columns = nstl::min(avx2_transpose_size, ncolumns);
    add(reg_src, -src_B_offset + src_C_offset);
    add(reg_dst, -dst_B_offset + dst_C_offset);
    jit_generator::transpose(reg_src, reg_dst, src_stride, dst_stride, C_rows,
            C_columns, data_type::f32, ymm_tmp, ymm_tail_mask,
            xmm_upper_tail_mask);

    const dim_t src_D_offset = src_stride * avx2_transpose_size
            + sizeof(float) * avx2_transpose_size;
    const dim_t dst_D_offset = dst_stride * avx2_transpose_size
            + sizeof(float) * avx2_transpose_size;
    const int D_rows = nstl::max(nrows - avx2_transpose_size, 0);
    const int D_columns = nstl::max(ncolumns - avx2_transpose_size, 0);
    add(reg_src, -src_C_offset + src_D_offset);
    add(reg_dst, -dst_C_offset + dst_D_offset);
    jit_generator::transpose(reg_src, reg_dst, src_stride, dst_stride, D_rows,
            D_columns, data_type::f32, ymm_tmp, ymm_tail_mask,
            xmm_upper_tail_mask);
    sub(reg_src, src_D_offset);
    sub(reg_dst, dst_D_offset);
}

template <>
void jit_brgemm_matmul_copy_a_transposed_impl_t<Xbyak::Zmm>::transpose_f32(
        reg64_t dst, reg64_t src, int nrows, int ncolumns) {
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
        if (i < nrows) {
            if (use_fp16_instructions_)
                vcvtph2psx(src_zmm(i) | kTail | T_z, addr);
            else
                vmovups(src_zmm(i) | kTail | T_z, addr);
        } else {
            vpxord(src_zmm(i), src_zmm(i), src_zmm(i));
        }
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
}

template <typename Vmm>
void jit_brgemm_matmul_copy_a_transposed_impl_t<Vmm>::deploy_transpose(
        reg64_t dst, reg64_t src, int nrows, int ncolumns) {
    if (is_f32 || use_fp16_instructions_)
        transpose_f32(dst, src, nrows, ncolumns);
    else
        transpose_bf16(dst, src, nrows, ncolumns);
}

template <typename Vmm>
void jit_brgemm_matmul_copy_a_transposed_impl_t<Vmm>::init_masks() {
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
    if (is_superset(conf_->isa, avx512_core)) {
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
        if (!is_f32) {
            vmovdqa64(vidx1, idx1);
            vmovdqa64(vidx2, idx2);
            vmovdqa32(vidx3, idx3);
            vmovdqa32(vidx4, idx4);
            vmovdqa32(vidx5, (const int32_t *)idx5);
        }
    }
}

template <typename Vmm>
void jit_brgemm_matmul_copy_a_transposed_impl_t<Vmm>::generate() {

    // only bf16, f16 and f32 supported for now
    if (!one_of(conf_->src_dt, data_type::bf16, data_type::f32, data_type::f16))
        return;
    preamble();
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

    init_masks();

    const int k_block_tail = conf_->K_blk % rows_step;
    const int last_k_block_tail = (conf_->K % conf_->K_blk) % rows_step;
    const int m_block_tail = conf_->M_blk % columns_step;
    const int last_m_block_tail = conf_->M_tail % columns_step;

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
    postamble();
}

struct jit_brgemm_matmul_copy_a_transposed_int8_impl_t
    : public jit_brgemm_matmul_copy_a_t,
      public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(
            jit_brgemm_matmul_copy_a_transposed_int8_impl_t)

    jit_brgemm_matmul_copy_a_transposed_int8_impl_t(
            const brgemm_matmul_conf_t *conf)
        : jit_brgemm_matmul_copy_a_t(conf)
        , jit_generator(jit_name())
        , src_stride_(conf_->copy_A_src_stride)
        , dst_stride_(conf_->LDA * conf_->tr_a_dt_sz)
        , m_loop_src_shift_(columns_step_ * conf_->a_dt_sz)
        , m_loop_dst_shift_(columns_step_ * dst_stride_)
        , k_loop_src_shift_(rows_step_ * src_stride_)
        , k_loop_dst_shift_(rows_step_ * conf_->tr_a_dt_sz)
        , has_vpermb_(cpu().has(Xbyak::util::Cpu::tAVX512_VBMI))
        , is_dynamic_src_ld_(conf_->is_runtime_M)
        , k_block_tail_(conf_->K_blk % rows_step_)
        , last_k_block_tail_((conf_->K % conf_->K_blk) % rows_step_)
        , m_block_tail_(conf_->M_blk % columns_step_)
        , last_m_block_tail_(conf_->M_tail % columns_step_)
        , do_compute_compensation_(conf_->has_zero_point_b) {}

    void operator()(ctx_t *ctx) override { jit_generator::operator()(ctx); }
    status_t create_kernel() override { return jit_generator::create_kernel(); }

private:
    constexpr static int rows_step_ = 16;
    constexpr static int columns_step_ = 16;
    constexpr static int current_M_blk_offt_ = 0;
    constexpr static int current_K_blk_offt_ = 8;
    constexpr static int src_offt_ = 16;
    constexpr static int tr_src_offt_ = 24;
    constexpr static int dynamic_src_ld_offt_ = 32;
    constexpr static int dynamic_src_ld_x_2_offt_ = 40;
    constexpr static int dynamic_src_ld_x_kstep_offt_ = 48;
    constexpr static int stack_space_needed_ = 56;

    const dim_t src_stride_, dst_stride_;
    const dim_t m_loop_src_shift_, m_loop_dst_shift_;
    const dim_t k_loop_src_shift_, k_loop_dst_shift_;
    const bool has_vpermb_;
    const bool is_dynamic_src_ld_;

    const int k_block_tail_, last_k_block_tail_;
    const int m_block_tail_, last_m_block_tail_;

    const bool do_compute_compensation_;

    Opmask kFFFF_ = k1;
    Opmask k5555_ = k2;
    Opmask kAAAA_ = k3;
    Opmask kAA_ = k4;
    Opmask k55_ = k5;
    Opmask kCC_ = k6;
    Opmask k33_ = k7;
    Opmask kTail_ = k1;

    Reg64 reg_tmp_ = r15;
    Reg64 reg_k_src_ = r14;
    Reg64 reg_k_dst_ = r13;
    Reg64 reg_m_src_ = r12;
    Reg64 reg_m_dst_ = r11;
    Reg64 reg_aux_src0_ = r10;
    Reg64 reg_aux_src1_ = r9;
    Reg64 reg_zp_comp_res_ptr_ = r8;
    Reg64 reg_zp_comp_prev_data_ = rdx;
    Reg64 reg_loop_m_ = rbx;
    Reg64 reg_loop_k_ = rax;
    // Note: this must be assigned to rcx as it's used in shl instruction,
    // clashes with abi_param1 on Windows OS
    Reg64 reg_opmask_shift_compute_ = rcx;

    // Indices used in permutations
    Zmm zmm_idx_1_ = zmm31;
    Zmm zmm_idx_2_ = zmm30;
    Zmm zmm_idx_3_ = zmm29;
    Zmm zmm_idx_4_ = zmm28;

    // zmm_idx_5_ is used in vpermb implementation only
    // zmm_conversion_tmp_ is used in vpermw implementation only
    Zmm zmm_idx_5_ = zmm27;
    Zmm zmm_conversion_tmp_ = zmm27;

    // Required for zero-point
    Zmm zmm_comp_temp_ = zmm26;
    Zmm zmm_comp_acc_ = zmm25;

    Zmm get_zmm_src(int i) {
        assert(i >= 0 && i < columns_step_);
        return Zmm(i);
    }
    void kmovd(const bool dynamic_column_size, Opmask k, unsigned w,
            bool load_mask_stage = false) {
        if (dynamic_column_size && load_mask_stage) {
            // reg_opmask_shift_compute_ is rcx, and we need cl for the shift
            mov(reg_opmask_shift_compute_, reg_loop_m_);
            mov(reg_tmp_, 1);
            shl(reg_tmp_, cl);
            sub(reg_tmp_, 1);
        } else
            mov(reg_tmp_, w);
        jit_generator::kmovd(k, reg_tmp_.cvt32());
    }

    void transpose_int8_vpermb(Reg64 dst, Reg64 src, int nrows, int ncolumns);
    void transpose_int8_vpermw(Reg64 dst, Reg64 src, int nrows, int ncolumns);

    inline void deploy_transpose(
            Reg64 dst, Reg64 src, int nrows, int ncolumns) {
        if (has_vpermb_)
            transpose_int8_vpermb(dst, src, nrows, ncolumns);
        else
            transpose_int8_vpermw(dst, src, nrows, ncolumns);
    }

    void reset_compensation_accumulator() {
        if (do_compute_compensation_)
            uni_vpxor(zmm_comp_acc_, zmm_comp_acc_, zmm_comp_acc_);
    }
    void accumulate_compensation(Zmm zmm_copy) {
        if (do_compute_compensation_) {
            if (conf_->src_dt == data_type::s8)
                vpmovsxbd(zmm_comp_temp_, zmm_copy);
            else
                vpmovzxbd(zmm_comp_temp_, zmm_copy);
            vpaddd(zmm_comp_acc_, zmm_comp_acc_, zmm_comp_temp_);
        }
    }
    void save_partial_compensation() {
        if (do_compute_compensation_) {
            Label no_previous_data;
            test(reg_zp_comp_prev_data_, reg_zp_comp_prev_data_);
            je(no_previous_data);
            vpaddd(zmm_comp_acc_, zmm_comp_acc_, ptr[reg_zp_comp_res_ptr_]);
            L(no_previous_data);

            vmovups(ptr[reg_zp_comp_res_ptr_], zmm_comp_acc_);
            add(reg_zp_comp_res_ptr_, sizeof(int32_t) * 16);
        }
    }

    void compute_m_loop(int nrows);
    void compute_k_loop(bool is_first_K_iter, bool is_last_K_iter);
    void generate() override;
};

void jit_brgemm_matmul_copy_a_transposed_int8_impl_t::transpose_int8_vpermb(
        Reg64 dst, Reg64 src, int nrows, int ncolumns) {
    assert(nrows >= 0 && nrows <= rows_step_ && ncolumns >= 0
            && ncolumns <= columns_step_);
    if (!nrows) return;

    auto load = [this, src](Zmm r, int i, Reg64 reg_aux) {
        auto addr = is_dynamic_src_ld_
                ? ptr[reg_aux]
                : EVEX_compress_addr(src, i * src_stride_);
        vmovdqu8(r | kFFFF_ | T_z, addr);
        accumulate_compensation(r);
    };

    auto store = [this, dst](Zmm r, int i) {
        auto addr = EVEX_compress_addr(dst, i * dst_stride_);
        vmovdqu8(addr, r | kTail_);
    };

    Label transpose_int8_done;

    reset_compensation_accumulator();

    const bool dynamic_column_size = ncolumns == 0 && is_dynamic_src_ld_;
    const int load_mask
            = ncolumns < columns_step_ ? (1 << ncolumns) - 1 : 0xffff;
    kmovd(dynamic_column_size, kFFFF_, load_mask, true);

    // load rows and swap bytes
    for (int i = 0; i < nrows; i += 4) {
        auto idx0 = i;
        auto zmm_src0 = get_zmm_src(idx0);
        if (is_dynamic_src_ld_) {
            if (i == 0)
                mov(reg_aux_src0_, src);
            else
                add(reg_aux_src0_, ptr[rsp + dynamic_src_ld_offt_]);
        }
        load(zmm_src0, idx0, reg_aux_src0_);

        auto idx1 = i + 1;
        auto zmm_src1 = get_zmm_src(idx1);
        if (is_dynamic_src_ld_)
            add(reg_aux_src0_, ptr[rsp + dynamic_src_ld_offt_]);
        if (idx1 < nrows)
            load(zmm_src1, idx1, reg_aux_src0_);
        else
            vpxord(zmm_src1, zmm_src1, zmm_src1);

        auto idx2 = i + 2;
        auto zmm_src2 = get_zmm_src(idx2);
        if (is_dynamic_src_ld_)
            add(reg_aux_src0_, ptr[rsp + dynamic_src_ld_offt_]);
        if (idx2 < nrows)
            load(zmm_src2, idx2, reg_aux_src0_);
        else
            vpxord(zmm_src2, zmm_src2, zmm_src2);

        auto idx3 = i + 3;
        auto zmm_src3 = get_zmm_src(idx3);
        if (is_dynamic_src_ld_)
            add(reg_aux_src0_, ptr[rsp + dynamic_src_ld_offt_]);
        if (idx3 < nrows)
            load(zmm_src3, idx3, reg_aux_src0_);
        else
            vpxord(zmm_src3, zmm_src3, zmm_src3);

        // concatenate 4 rows
        vinserti64x2(Ymm(zmm_src0.getIdx()), Ymm(zmm_src0.getIdx()),
                Xmm(zmm_src1.getIdx()), 1);
        vinserti64x2(Ymm(zmm_src2.getIdx()), Ymm(zmm_src2.getIdx()),
                Xmm(zmm_src3.getIdx()), 1);
        vinserti64x4(zmm_src0, zmm_src0, Ymm(zmm_src2.getIdx()), 1);

        // swap bytes
        vpermb(zmm_src0, zmm_idx_1_, zmm_src0);
    }

    // swap doubles
    for (int i = 0; i < 2; i++) {
        auto idx0 = 8 * i;
        auto idx1 = idx0 + 4;

        auto zmm_src0 = get_zmm_src(idx0);
        auto zmm_src1 = get_zmm_src(idx1);

        auto zmm_tmp0 = get_zmm_src(idx0 + 1);
        auto zmm_tmp1 = get_zmm_src(idx1 + 1);

        vmovups(zmm_tmp0, zmm_idx_2_);
        vmovups(zmm_tmp1, zmm_idx_3_);

        vpermi2d(zmm_tmp0, zmm_src0, zmm_src1);
        vpermi2d(zmm_tmp1, zmm_src0, zmm_src1);
    }

    // swap quads
    for (int i = 0; i < 2; i++) {
        auto idx0 = 4 * i;
        auto idx1 = idx0 + 8;

        auto zmm_src0 = get_zmm_src(idx0 + 1);
        auto zmm_src1 = get_zmm_src(idx1 + 1);

        auto zmm_tmp0 = get_zmm_src(idx0);
        auto zmm_tmp1 = get_zmm_src(idx1);

        vmovups(zmm_tmp0, zmm_idx_4_);
        vmovups(zmm_tmp1, zmm_idx_5_);

        vpermi2q(zmm_tmp0, zmm_src0, zmm_src1);
        vpermi2q(zmm_tmp1, zmm_src0, zmm_src1);
    }

    // extract columns
    for (int i = 0; i < 16; i += 4) {
        vextracti64x4(
                Ymm(get_zmm_src(i + 2).getIdx()) | T_z, get_zmm_src(i), 1);
        vextracti32x4(
                Xmm(get_zmm_src(i + 1).getIdx()) | T_z, get_zmm_src(i), 1);
        vextracti32x4(Xmm(get_zmm_src(i + 3).getIdx()) | T_z,
                Ymm(get_zmm_src(i + 2).getIdx()), 1);
    }

    // store columns
    const int rows_to_store = rnd_up(nrows, 2);
    const int store_mask
            = rows_to_store < rows_step_ ? (1 << rows_to_store) - 1 : 0xffff;
    kmovd(dynamic_column_size, kTail_, store_mask);

    auto get_vec_idx = [](int col_idx) {
        assert(col_idx < columns_step_ && col_idx >= 0);

        const auto div = col_idx / 4;
        const auto mod = col_idx % 4;

        return mod * 4 + div;
    };

    const int columns_to_store = dynamic_column_size ? columns_step_ : ncolumns;
    for (int col_idx = 0; col_idx < columns_to_store; col_idx++) {
        store(get_zmm_src(get_vec_idx(col_idx)), col_idx);
        if (dynamic_column_size) {
            dec(reg_opmask_shift_compute_);
            jz(transpose_int8_done, T_NEAR);
        }
    }

    L(transpose_int8_done);

    save_partial_compensation();
}

void jit_brgemm_matmul_copy_a_transposed_int8_impl_t::transpose_int8_vpermw(
        Reg64 dst, Reg64 src, int nrows, int ncolumns) {
    assert(nrows >= 0 && nrows <= rows_step_ && ncolumns >= 0
            && ncolumns <= columns_step_);
    if (!nrows) return;

    auto load = [this, src](Zmm r, int i, Reg64 reg_aux) {
        auto addr = is_dynamic_src_ld_
                ? ptr[reg_aux]
                : EVEX_compress_addr(src, i * src_stride_);
        vmovdqu8(zmm_conversion_tmp_ | kFFFF_ | T_z, addr);
        accumulate_compensation(zmm_conversion_tmp_);
        if (conf_->src_dt == data_type::s8)
            vpmovsxbw(r, zmm_conversion_tmp_);
        else
            vpmovzxbw(r, zmm_conversion_tmp_);
    };

    auto store = [this, dst](Zmm r, int i) {
        if (conf_->src_dt == data_type::s8)
            vpmovswb(Ymm(zmm_conversion_tmp_.getIdx()), r);
        else
            vpmovuswb(Ymm(zmm_conversion_tmp_.getIdx()), r);
        auto addr = EVEX_compress_addr(dst, i * dst_stride_);
        vmovdqu8(addr, zmm_conversion_tmp_ | kTail_);
    };

    Label transpose_int8_done;

    reset_compensation_accumulator();

    const bool dynamic_column_size = ncolumns == 0 && is_dynamic_src_ld_;
    const int load_mask
            = ncolumns < columns_step_ ? (1 << ncolumns) - 1 : 0xffff;
    kmovd(dynamic_column_size, kFFFF_, load_mask, true);

    for (int i = 0; i < nrows / 2; i++) {
        auto idx0 = 2 * i;
        auto idx1 = 2 * i + 1;
        auto zmm_src0 = get_zmm_src(idx0);
        auto zmm_src1 = get_zmm_src(idx1);
        if (is_dynamic_src_ld_) {
            if (i == 0) {
                mov(reg_aux_src0_, src);
                mov(reg_aux_src1_, src);
                add(reg_aux_src1_, ptr[rsp + dynamic_src_ld_offt_]);
            } else {
                add(reg_aux_src0_, ptr[rsp + dynamic_src_ld_x_2_offt_]);
                add(reg_aux_src1_, ptr[rsp + dynamic_src_ld_x_2_offt_]);
            }
        }

        load(zmm_src0, idx0, reg_aux_src0_);
        load(zmm_src1, idx1, reg_aux_src1_);

        vinserti64x4(zmm_src0, zmm_src0, Ymm(zmm_src1.getIdx()), 1);
        vpermw(zmm_src0, zmm_idx_1_, zmm_src0);
    }

    // for odd numbers we need to mix row with zeroes
    if (nrows % 2) {
        int i = nrows / 2;
        auto zmm_src0 = get_zmm_src(2 * i);
        if (is_dynamic_src_ld_) {
            if (i == 0)
                mov(reg_aux_src0_, src);
            else
                add(reg_aux_src0_, ptr[rsp + dynamic_src_ld_x_2_offt_]);
        }

        load(zmm_src0, 2 * i, reg_aux_src0_);

        vpermw(zmm_src0, zmm_idx_1_, zmm_src0);
    }

    for (int i = rnd_up(nrows, 2); i < rows_step_; i += 2)
        vpxord(get_zmm_src(i), get_zmm_src(i), get_zmm_src(i));

    // swap 1
    for (int i = 0; i < 4; i++) {
        auto zmm0 = get_zmm_src(4 * i);
        auto zmm1 = get_zmm_src(4 * i + 2);
        auto tmp0 = get_zmm_src(4 * i + 1);
        auto tmp1 = get_zmm_src(4 * i + 3);

        vmovups(tmp0, zmm0);
        vmovups(tmp1, zmm1);

        vpermps(tmp0 | kAAAA_, zmm_idx_2_, zmm1);
        vpermps(tmp1 | k5555_, zmm_idx_2_, zmm0);
    }

    // swap 2
    int base_idx;
    base_idx = 0;
    for (int i = 0; i < 2; i++) {
        auto zmm0 = get_zmm_src(base_idx + 2 * i + 1);
        auto zmm1 = get_zmm_src(base_idx + 2 * i + 5);

        auto tmp0 = get_zmm_src(base_idx + 2 * i);
        auto tmp1 = get_zmm_src(base_idx + 2 * i + 4);

        vmovupd(tmp0, zmm0);
        vmovupd(tmp1, zmm1);

        vpermpd(tmp0 | kAA_, zmm_idx_3_, zmm1);
        vpermpd(tmp1 | k55_, zmm_idx_3_, zmm0);
    }
    base_idx = 8;
    for (int i = 0; i < 2; i++) {
        auto zmm0 = get_zmm_src(base_idx + 2 * i + 1);
        auto zmm1 = get_zmm_src(base_idx + 2 * i + 5);

        auto tmp0 = get_zmm_src(base_idx + 2 * i);
        auto tmp1 = get_zmm_src(base_idx + 2 * i + 4);

        vmovupd(tmp0, zmm0);
        vmovupd(tmp1, zmm1);

        vpermpd(tmp0 | kAA_, zmm_idx_3_, zmm1);
        vpermpd(tmp1 | k55_, zmm_idx_3_, zmm0);
    }

    // swap 3
    for (int i = 0; i < 4; i++) {
        auto zmm0 = get_zmm_src(2 * i);
        auto zmm1 = get_zmm_src(2 * i + 8);

        auto tmp0 = get_zmm_src(2 * i + 1);
        auto tmp1 = get_zmm_src(2 * i + 9);

        vmovupd(tmp0, zmm0);
        vmovupd(tmp1, zmm1);

        vpermpd(tmp0 | kCC_, zmm_idx_4_, zmm1);
        vpermpd(tmp1 | k33_, zmm_idx_4_, zmm0);
    }

    // all stores
    for (int i = 0; i < 8; i++)
        vextracti64x4(Ymm(get_zmm_src(2 * i).getIdx()) | T_z,
                get_zmm_src(2 * i + 1), 1);

    const int rows_to_store = rnd_up(nrows, 2);
    const int store_mask
            = rows_to_store < rows_step_ ? (1 << rows_to_store) - 1 : 0xffff;
    kmovd(dynamic_column_size, kTail_, store_mask);

    auto get_vec_idx = [](int col_idx) {
        assert(col_idx < columns_step_ && col_idx >= 0);
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

    const int columns_to_store = dynamic_column_size ? columns_step_ : ncolumns;
    for (int col_idx = 0; col_idx < columns_to_store; col_idx++) {
        store(get_zmm_src(get_vec_idx(col_idx)), col_idx);
        if (dynamic_column_size) {
            dec(reg_opmask_shift_compute_);
            jz(transpose_int8_done, T_NEAR);
        }
    }

    L(transpose_int8_done);

    save_partial_compensation();
}

void jit_brgemm_matmul_copy_a_transposed_int8_impl_t::compute_m_loop(
        int nrows) {
    mov(reg_loop_m_, ptr[rsp + current_M_blk_offt_]);
    mov(reg_m_src_, reg_k_src_);
    mov(reg_m_dst_, reg_k_dst_);

    if (do_compute_compensation_)
        mov(reg_zp_comp_res_ptr_,
                ptr[param1 + GET_OFF(zp_a_compensation_result_ptr)]);

    Label m_loop_tail_or_done, m_loop, compute_m_loop_done;
    cmp(reg_loop_m_, columns_step_);
    jl(m_loop_tail_or_done, T_NEAR);

    L(m_loop);
    {
        deploy_transpose(reg_m_dst_, reg_m_src_, nrows, columns_step_);
        add(reg_m_src_, m_loop_src_shift_);
        add(reg_m_dst_, m_loop_dst_shift_);
    }
    sub(reg_loop_m_, columns_step_);
    cmp(reg_loop_m_, columns_step_);
    jge(m_loop, T_NEAR);

    if (m_block_tail_ > 0 || last_m_block_tail_ > 0 || is_dynamic_src_ld_)
        jz(compute_m_loop_done, T_NEAR);

    L(m_loop_tail_or_done);

    if (m_block_tail_ > 0) {
        Label m_block_tail_done;
        cmp(reg_loop_m_, m_block_tail_);
        jne(m_block_tail_done, T_NEAR);

        deploy_transpose(reg_m_dst_, reg_m_src_, nrows, m_block_tail_);
        jmp(compute_m_loop_done, T_NEAR);

        L(m_block_tail_done);
    }
    if (IMPLICATION(
                last_m_block_tail_ <= 0 || last_m_block_tail_ == m_block_tail_,
                is_dynamic_src_ld_)) {
        Label last_m_block_tail_done;
        cmp(reg_loop_m_, 0);
        jle(last_m_block_tail_done, T_NEAR);

        deploy_transpose(reg_m_dst_, reg_m_src_, nrows,
                is_dynamic_src_ld_ ? 0 : last_m_block_tail_);
        L(last_m_block_tail_done);
    }

    L(compute_m_loop_done);

    if (do_compute_compensation_) mov(reg_zp_comp_prev_data_, 1);
}

void jit_brgemm_matmul_copy_a_transposed_int8_impl_t::compute_k_loop(
        bool is_first_K_iter, bool is_last_K_iter) {
    mov(reg_k_src_, ptr[rsp + src_offt_]);
    mov(reg_k_dst_, ptr[rsp + tr_src_offt_]);
    mov(reg_loop_k_, ptr[rsp + current_K_blk_offt_]);

    if (do_compute_compensation_) {
        if (is_first_K_iter)
            mov(reg_zp_comp_prev_data_, 0);
        else
            mov(reg_zp_comp_prev_data_, 1);
    }

    Label k_tail_or_done, k_loop, compute_k_loop_done;
    cmp(reg_loop_k_, rows_step_);
    jl(k_tail_or_done, T_NEAR);

    L(k_loop);
    {
        compute_m_loop(rows_step_);
        if (is_dynamic_src_ld_)
            add(reg_k_src_, ptr[rsp + dynamic_src_ld_x_kstep_offt_]);
        else
            add(reg_k_src_, k_loop_src_shift_);
        add(reg_k_dst_, k_loop_dst_shift_);
    }
    sub(reg_loop_k_, rows_step_);
    cmp(reg_loop_k_, rows_step_);

    jge(k_loop, T_NEAR);

    if (k_block_tail_ > 0 || last_k_block_tail_ > 0)
        jz(compute_k_loop_done, T_NEAR);

    L(k_tail_or_done);

    if (k_block_tail_ > 0) {
        Label k_block_tail_done;
        cmp(reg_loop_k_, k_block_tail_);
        jne(k_block_tail_done, T_NEAR);

        compute_m_loop(k_block_tail_);
        jmp(compute_k_loop_done, T_NEAR);

        L(k_block_tail_done);
    }
    if (last_k_block_tail_ > 0 && last_k_block_tail_ != k_block_tail_) {
        Label last_k_block_tail_done;
        cmp(reg_loop_k_, last_k_block_tail_);
        jne(last_k_block_tail_done, T_NEAR);

        compute_m_loop(last_k_block_tail_);
        jmp(compute_k_loop_done, T_NEAR);

        L(last_k_block_tail_done);
    }

    L(compute_k_loop_done);

    if (do_compute_compensation_ && is_last_K_iter) {
        mov(reg_zp_comp_res_ptr_,
                ptr[param1 + GET_OFF(zp_a_compensation_result_ptr)]);

        auto calculate_final_compensation = [this]() {
            // load accumulated compensation
            vmovups(zmm_comp_acc_, ptr[reg_zp_comp_res_ptr_]);

            // add -K * zp_a_val as mixed ab_compensation component
            if (conf_->src_zp_type != brgemm_broadcast_t::none) {
                mov(reg_tmp_, ptr[param1 + GET_OFF(zp_ab_comp_ptr)]);
                vbroadcastss(get_zmm_src(0), ptr[reg_tmp_]);
                vpaddd(zmm_comp_acc_, zmm_comp_acc_, get_zmm_src(0));
            }

            // multiply by zp_b_val
            mov(reg_tmp_, ptr[param1 + GET_OFF(zp_b_neg_value_ptr)]);
            vbroadcastss(get_zmm_src(0), ptr[reg_tmp_]);
            vpmulld(zmm_comp_acc_, zmm_comp_acc_, get_zmm_src(0));

            // store the final result value
            vmovups(ptr[reg_zp_comp_res_ptr_], zmm_comp_acc_);
            add(reg_zp_comp_res_ptr_, sizeof(int32_t) * 16);
        };

        Label m_loop, m_loop_tail_or_done, compute_m_loop_done;

        mov(reg_loop_m_, ptr[rsp + current_M_blk_offt_]);
        cmp(reg_loop_m_, columns_step_);
        jl(m_loop_tail_or_done, T_NEAR);

        L(m_loop);
        calculate_final_compensation();
        sub(reg_loop_m_, columns_step_);
        cmp(reg_loop_m_, columns_step_);
        jge(m_loop, T_NEAR);

        if (m_block_tail_ > 0 || last_m_block_tail_ > 0 || is_dynamic_src_ld_)
            jz(compute_m_loop_done, T_NEAR);

        L(m_loop_tail_or_done);

        if (m_block_tail_ > 0) {
            Label m_block_tail_done;
            cmp(reg_loop_m_, m_block_tail_);
            jne(m_block_tail_done, T_NEAR);

            calculate_final_compensation();
            jmp(compute_m_loop_done, T_NEAR);

            L(m_block_tail_done);
        }
        if (IMPLICATION(last_m_block_tail_ <= 0
                            || last_m_block_tail_ == m_block_tail_,
                    is_dynamic_src_ld_)) {
            Label last_m_block_tail_done;
            cmp(reg_loop_m_, 0);
            jle(last_m_block_tail_done, T_NEAR);

            calculate_final_compensation();
            L(last_m_block_tail_done);
        }

        L(compute_m_loop_done);
    }
}

void jit_brgemm_matmul_copy_a_transposed_int8_impl_t::generate() {
    preamble();
    sub(rsp, stack_space_needed_);
    mov(reg_tmp_, ptr[param1 + GET_OFF(current_M_blk)]);
    mov(ptr[rsp + current_M_blk_offt_], reg_tmp_);
    mov(reg_tmp_, ptr[param1 + GET_OFF(src)]);
    mov(ptr[rsp + src_offt_], reg_tmp_);
    mov(reg_tmp_, ptr[param1 + GET_OFF(tr_src)]);
    mov(ptr[rsp + tr_src_offt_], reg_tmp_);
    mov(reg_tmp_, ptr[param1 + GET_OFF(current_K_blk)]);
    mov(ptr[rsp + current_K_blk_offt_], reg_tmp_);
    if (is_dynamic_src_ld_) {
        // dynamic src_stride
        mov(reg_tmp_, ptr[param1 + GET_OFF(dynamic_src_ld)]);
        mov(ptr[rsp + dynamic_src_ld_offt_], reg_tmp_);

        // src_stride * 2
        shl(reg_tmp_, 1);
        mov(ptr[rsp + dynamic_src_ld_x_2_offt_], reg_tmp_);

        // src_stride * rows_step_
        assert(rows_step_ == 16);
        shl(reg_tmp_, 3);
        mov(ptr[rsp + dynamic_src_ld_x_kstep_offt_], reg_tmp_);
    }

    auto kmovw = [this](Opmask k, unsigned w) {
        mov(reg_tmp_, w);
        jit_generator::kmovw(k, reg_tmp_.cvt32());
    };

    kmovw(kFFFF_, 0xffff);
    kmovw(k5555_, 0x5555);
    kmovw(kAAAA_, 0xaaaa);
    kmovw(kAA_, 0xaa);
    kmovw(k55_, 0x55);
    kmovw(kCC_, 0xcc);
    kmovw(k33_, 0x33);

    auto vmovdqa64 = [this](Zmm z, const int64_t *addr) {
        mov(reg_tmp_, reinterpret_cast<size_t>(addr));
        jit_generator::vmovdqa64(z, ptr[reg_tmp_]);
    };

    if (has_vpermb_) {
        alignas(64) static constexpr const uint8_t idx1[64] = {0, 16, 32, 48, 1,
                17, 33, 49, 2, 18, 34, 50, 3, 19, 35, 51, 4, 20, 36, 52, 5, 21,
                37, 53, 6, 22, 38, 54, 7, 23, 39, 55, 8, 24, 40, 56, 9, 25, 41,
                57, 10, 26, 42, 58, 11, 27, 43, 59, 12, 28, 44, 60, 13, 29, 45,
                61, 14, 30, 46, 62, 15, 31, 47, 63};
        alignas(64) static constexpr const uint32_t idx2[16]
                = {0, 16, 2, 18, 4, 20, 6, 22, 8, 24, 10, 26, 12, 28, 14, 30};
        alignas(64) static constexpr const uint32_t idx3[16]
                = {1, 17, 3, 19, 5, 21, 7, 23, 9, 25, 11, 27, 13, 29, 15, 31};
        alignas(64) static constexpr const uint64_t idx4[8]
                = {0, 8, 2, 10, 4, 12, 6, 14};
        alignas(64) static constexpr const uint64_t idx5[8]
                = {1, 9, 3, 11, 5, 13, 7, 15};

        vmovdqa64(zmm_idx_1_, (const int64_t *)idx1);
        vmovdqa64(zmm_idx_2_, (const int64_t *)idx2);
        vmovdqa64(zmm_idx_3_, (const int64_t *)idx3);
        vmovdqa64(zmm_idx_4_, (const int64_t *)idx4);
        vmovdqa64(zmm_idx_5_, (const int64_t *)idx5);
    } else {
        alignas(64) static constexpr const uint16_t idx1[32]
                = {0, 16, 2, 18, 8, 24, 10, 26, 4, 20, 6, 22, 12, 28, 14, 30, 1,
                        17, 3, 19, 9, 25, 11, 27, 5, 21, 7, 23, 13, 29, 15, 31};
        alignas(64) static constexpr const uint32_t idx2[16]
                = {1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14};
        alignas(64) static constexpr const uint64_t idx3[8]
                = {1, 0, 3, 2, 5, 4, 7, 6};
        alignas(64) static constexpr const uint64_t idx4[8]
                = {2, 3, 0, 1, 6, 7, 4, 5};

        vmovdqa64(zmm_idx_1_, (const int64_t *)idx1);
        vmovdqa64(zmm_idx_2_, (const int64_t *)idx2);
        vmovdqa64(zmm_idx_3_, (const int64_t *)idx3);
        vmovdqa64(zmm_idx_4_, (const int64_t *)idx4);
    }

    Label done;
    if (do_compute_compensation_) {
        assert(conf_->wei_zp_type == brgemm_broadcast_t::per_tensor);

        mov(reg_tmp_, ptr[param1 + GET_OFF(current_K_start)]);
        const auto last_K_threshold
                = rnd_up(conf_->K, conf_->K_blk) - conf_->K_blk;
        Label not_first, not_first_not_last;
        cmp(reg_tmp_, 0);
        jne(not_first, T_NEAR);
        {
            Label first_not_last;
            cmp(reg_tmp_, last_K_threshold);
            jl(first_not_last, T_NEAR);
            compute_k_loop(true, true);
            jmp(done, T_NEAR);

            L(first_not_last);
            compute_k_loop(true, false);
            jmp(done, T_NEAR);
        }

        L(not_first);
        cmp(reg_tmp_, last_K_threshold);
        jl(not_first_not_last, T_NEAR);

        compute_k_loop(false, true);
        jmp(done, T_NEAR);
        L(not_first_not_last);
    }
    compute_k_loop(false, false);
    L(done);

    add(rsp, stack_space_needed_);
    postamble();
}
template struct jit_brgemm_matmul_copy_a_transposed_impl_t<Zmm>;
template struct jit_brgemm_matmul_copy_a_transposed_impl_t<Ymm>;

template <typename Vmm>
struct jit_brgemm_matmul_copy_b_int8_t : public jit_brgemm_matmul_copy_b_t,
                                         public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_brgemm_matmul_copy_b_int8_t)

    jit_brgemm_matmul_copy_b_int8_t(const brgemm_matmul_conf_t *conf)
        : jit_brgemm_matmul_copy_b_t(conf)
        , jit_generator(jit_name())
        , src_stride_(conf->copy_B_wei_stride)
        , tr_src_stride_(conf->LDB * k_blk_step_ * sizeof(int8_t))
        , is_amx_(mayiuse(avx512_core_amx))
        , do_compute_compensation_(
                  conf->s8s8_compensation_required || conf->has_zero_point_a)
        , avx512_core_dot_product_(
                  do_compute_compensation_ && !isa_has_int8_vnni(conf->isa))
        , is_dynamic_stride_(is_runtime_value(src_stride_))
        , is_dynamic_N_(conf->is_runtime_N)
        , comp_acc_idx_(is_ymm_                      ? 13
                          : avx512_core_dot_product_ ? 23
                                                     : 25) {}

    void operator()(ctx_t *ctx) override { jit_generator::operator()(ctx); }
    status_t create_kernel() override { return jit_generator::create_kernel(); }

protected:
    using reg64_t = const Xbyak::Reg64;
    using reg32_t = const Xbyak::Reg32;

    static constexpr bool is_ymm_ = std::is_same<Vmm, Xbyak::Ymm>::value;
    static constexpr int k_blk_step_ = 4;
    static constexpr int n_blk_step_ = 64;
    static constexpr int blk_sz_ = 6;
    static constexpr int simd_w_ = vreg_traits<Vmm>::vlen;

    const dim_t src_stride_;
    const dim_t tr_src_stride_;
    const bool is_amx_;
    const bool do_compute_compensation_;
    const bool avx512_core_dot_product_;
    const bool is_dynamic_stride_;
    const bool is_dynamic_N_;

    constexpr static int reg_src_offs_ = 0;
    constexpr static int reg_tr_src_offs_ = 8;
    constexpr static int stack_space_needed_ = 16;

    const int comp_acc_idx_;

    const Xbyak::Opmask kTail = k7;

    reg64_t reg_src = rax;
    reg64_t reg_tr_src = rbx;
    reg64_t reg_comp_ptr = rdx;
    reg64_t reg_zp_comp_ptr = r11;
    reg64_t reg_zp_a_neg_val_ptr = r12;

    reg64_t reg_K_iters = r8;
    reg64_t reg_N_blk = r9;
    reg64_t reg_K_start = r10;
    reg64_t reg_src_stride = r13;
    reg64_t reg_src_backup = r14;
    reg64_t reg_tmp = r15;

    reg64_t reg_copy_block_n_shift = rsi;

    reg64_t reg_dynamic_tail = rcx;
    Xbyak::Reg8 reg8_mask_shift = reg_dynamic_tail.cvt8();

    // Required in every dot product for INT8 non-VNNI computation.
    Vmm vmm_ones_words = Vmm(24);
    Vmm vmm_dot_product_temp = Vmm(25);

    // ZMM stuff
    Vmm vreg_idx_lo_256 = Vmm(26);
    Vmm vreg_idx_hi_256 = Vmm(27);
    Vmm vreg_idx_lo_128 = Vmm(28);
    Vmm vreg_idx_hi_128 = Vmm(29);

    // Shared
    Vmm vmm_comp_mul = Vmm(is_ymm_ ? 14 : 30);
    Vmm vmm_zero = Vmm(is_ymm_ ? 15 : 31);

    Vmm get_comp_acc(int i) { return Vmm(comp_acc_idx_ - i); }
    Vmm get_vmm_zp_comp_res(int i) { return get_comp_acc(i); }
    Vmm get_vmm_oscale_comp_res(int i) { return Vmm(i); }

    inline void vmovdqa64(Vmm vmm, const void *addr) {
        mov(reg_tmp, reinterpret_cast<size_t>(addr));
        jit_generator::vmovdqa64(vmm, ptr[reg_tmp]);
    }

    inline Vmm get_vmm(int blk, int idx) {
        if (idx < 0 || idx >= isa_num_vregs(is_ymm_ ? avx2 : avx512_core))
            assert(!"idx > vregs");
        assert(IMPLICATION(!is_ymm_, idx < blk_sz_ && blk >= 0));
        auto reg_idx = blk_sz_ * blk + idx;
        return Vmm(reg_idx);
    }
    inline void load(int blk, int i, bool is_tail) {}
    inline void kmovq(Opmask k, size_t q) {}
    virtual void init_permute() {}
    virtual void copy_block(int nrows, int ncolumns, bool n_tail) {
        UNUSED(n_tail);
        copy_4x64(nrows, ncolumns);
    }
    virtual void copy_4x64(int nrows, int ncolumns) {}
    inline void dot_product(Vmm v1, Vmm v2, Vmm v3) {
        if (!avx512_core_dot_product_)
            vpdpbusd(v1, v2, v3,
                    mayiuse(avx512_core) ? EvexEncoding : VexEncoding);
        else {
            vpmaddubsw(vmm_dot_product_temp, v2, v3);
            vpmaddwd(
                    vmm_dot_product_temp, vmm_dot_product_temp, vmm_ones_words);
            vpaddd(v1, v1, vmm_dot_product_temp);
        }
    }
    void generate() override;
};

template <>
inline void jit_brgemm_matmul_copy_b_int8_t<Zmm>::load(
        int blk, int i, bool is_tail) {
    auto vmm_src = get_vmm(blk, i % k_blk_step_);
    auto src_load = is_tail ? vmm_src | kTail | T_z : vmm_src;
    const auto offset = is_dynamic_stride_ ? 0 : i * src_stride_;
    vmovdqu8(src_load, EVEX_compress_addr(reg_src, offset));
    if (is_dynamic_stride_) add(reg_src, reg_src_stride);
}

template <>
inline void jit_brgemm_matmul_copy_b_int8_t<Zmm>::kmovq(Opmask k, size_t q) {
    if (is_dynamic_N_) {
        mov(reg_tmp, 1);
        shl(reg_tmp, reg8_mask_shift /* reg_dynamic_tail.cvt8() == cl */);
        sub(reg_tmp, 1);
    } else
        mov(reg_tmp, q);
    jit_generator::kmovq(k, reg_tmp);
}

template struct jit_brgemm_matmul_copy_b_int8_t<Zmm>;
template struct jit_brgemm_matmul_copy_b_int8_t<Ymm>;

struct jit_amx_brgemm_matmul_copy_b_int8_t
    : public jit_brgemm_matmul_copy_b_int8_t<Xbyak::Zmm> {

    jit_amx_brgemm_matmul_copy_b_int8_t(const brgemm_matmul_conf_t *conf)
        : jit_brgemm_matmul_copy_b_int8_t<Xbyak::Zmm>(conf) {}

private:
    void init_permute() override {
        alignas(64) static constexpr const uint8_t idx_lo_16[64] = {0, 1, 64,
                65, 4, 5, 68, 69, 2, 3, 66, 67, 6, 7, 70, 71, 8, 9, 72, 73, 12,
                13, 76, 77, 10, 11, 74, 75, 14, 15, 78, 79, 16, 17, 80, 81, 20,
                21, 84, 85, 18, 19, 82, 83, 22, 23, 86, 87, 24, 25, 88, 89, 28,
                29, 92, 93, 26, 27, 90, 91, 30, 31, 94, 95};

        alignas(64) static constexpr const uint8_t idx_hi_16[64] = {32, 33, 96,
                97, 36, 37, 100, 101, 34, 35, 98, 99, 38, 39, 102, 103, 40, 41,
                104, 105, 44, 45, 108, 109, 42, 43, 106, 107, 46, 47, 110, 111,
                48, 49, 112, 113, 52, 53, 116, 117, 50, 51, 114, 115, 54, 55,
                118, 119, 56, 57, 120, 121, 60, 61, 124, 125, 58, 59, 122, 123,
                62, 63, 126, 127};

        alignas(64) static constexpr const uint8_t idx_lo_8[64] = {0, 64, 2, 66,
                1, 65, 3, 67, 8, 72, 10, 74, 9, 73, 11, 75, 4, 68, 6, 70, 5, 69,
                7, 71, 12, 76, 14, 78, 13, 77, 15, 79, 16, 80, 18, 82, 17, 81,
                19, 83, 24, 88, 26, 90, 25, 89, 27, 91, 20, 84, 22, 86, 21, 85,
                23, 87, 28, 92, 30, 94, 29, 93, 31, 95};

        alignas(64) static constexpr const uint8_t idx_hi_8[64] = {32, 96, 34,
                98, 33, 97, 35, 99, 40, 104, 42, 106, 41, 105, 43, 107, 36, 100,
                38, 102, 37, 101, 39, 103, 44, 108, 46, 110, 45, 109, 47, 111,
                48, 112, 50, 114, 49, 113, 51, 115, 56, 120, 58, 122, 57, 121,
                59, 123, 52, 116, 54, 118, 53, 117, 55, 119, 60, 124, 62, 126,
                61, 125, 63, 127};

        vmovdqa64(vreg_idx_lo_256, (const void *)idx_lo_16);
        vmovdqa64(vreg_idx_hi_256, (const void *)idx_hi_16);
        vmovdqa64(vreg_idx_lo_128, (const void *)idx_lo_8);
        vmovdqa64(vreg_idx_hi_128, (const void *)idx_hi_8);
    }

    void copy_block(int nrows, int ncolumns, bool n_tail) override {
        if (!is_dynamic_N_ || !n_tail) {
            copy_4x64(nrows, ncolumns);
            return;
        }

        mov(reg_dynamic_tail, reg_N_blk);
        // dynamic tail processing: main loop with ncolumns = n_blk_step and
        // finally process tail < n_blk_step with dynamically computed mask
        // NOTE: for dynamic_stride case copy_4x64() shifts reg_src pointer
        // so we need to backup/restore its value for every iteration wrt n
        // except the last one

        mov(ptr[rsp + reg_tr_src_offs_], reg_tr_src);
        xor_(reg_copy_block_n_shift, reg_copy_block_n_shift);
        const auto typesize = sizeof(int8_t);

        Label loop_row_start, loop_row_tail, loop_row_done;
        cmp(reg_dynamic_tail, n_blk_step_);
        jl(loop_row_tail, T_NEAR);
        L(loop_row_start);
        {
            mov(ptr[rsp + reg_src_offs_], reg_src);
            add(reg_src, reg_copy_block_n_shift);
            copy_4x64(nrows, n_blk_step_);
            add(reg_copy_block_n_shift, n_blk_step_ * typesize);
            add(reg_src, n_blk_step_ * typesize);
            add(reg_tr_src, n_blk_step_ * k_blk_step_ * typesize);
            sub(reg_dynamic_tail, n_blk_step_);

            cmp(reg_dynamic_tail, 0);
            jle(loop_row_done, T_NEAR);

            mov(reg_src, ptr[rsp + reg_src_offs_]);

            cmp(reg_dynamic_tail, n_blk_step_);
            jl(loop_row_tail, T_NEAR);

            jmp(loop_row_start, T_NEAR);
        }

        L(loop_row_tail);
        {
            cmp(reg_dynamic_tail, 0);
            jle(loop_row_done, T_NEAR);

            add(reg_src, reg_copy_block_n_shift);
            copy_4x64(nrows, 1 /* to force tail case */);
        }
        L(loop_row_done);

        // restore pointers
        sub(reg_src, reg_copy_block_n_shift);
        mov(reg_tr_src, ptr[rsp + reg_tr_src_offs_]);
    }

    void copy_4x64(int nrows, int ncolumns) override {
        const bool is_tail = ncolumns < n_blk_step_;
        const auto tail_mask = size_t(((size_t)1 << ncolumns) - 1);

        if (is_tail) kmovq(kTail, tail_mask);

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

            vmovups(get_vmm(k, 4), vreg_idx_lo_256);
            vpermi2b(get_vmm(k, 4), get_vmm(k, 0), get_vmm(k, 2));
            vmovups(get_vmm(k, 5), vreg_idx_hi_256);
            vpermi2b(get_vmm(k, 5), get_vmm(k, 0), get_vmm(k, 2));
            vmovups(get_vmm(k, 0), vreg_idx_lo_256);
            vpermi2b(get_vmm(k, 0), get_vmm(k, 1), get_vmm(k, 3));
            vmovups(get_vmm(k, 2), vreg_idx_hi_256);
            vpermi2b(get_vmm(k, 2), get_vmm(k, 1), get_vmm(k, 3));

            vmovups(get_vmm(k, 1), vreg_idx_lo_128);
            vpermi2b(get_vmm(k, 1), get_vmm(k, 4), get_vmm(k, 0));
            dim_t tr_src_off_base = (kb * max_unroll + k) * tr_src_stride_;
            vmovups(EVEX_compress_addr(reg_tr_src, tr_src_off_base),
                    get_vmm(k, 1));
            if (do_compute_compensation_)
                vpdpbusd(get_comp_acc(0), vmm_comp_mul, get_vmm(k, 1));
            const bool dynamic_tail = is_dynamic_N_ && ncolumns < n_blk_step_;

            Label k_loop_done;
            if (dynamic_tail) {
                cmp(reg_dynamic_tail, 16);
                jle(k_loop_done, T_NEAR);
            }
            if (ncolumns > 16 || dynamic_tail) {
                vmovups(get_vmm(k, 3), vreg_idx_hi_128);
                vpermi2b(get_vmm(k, 3), get_vmm(k, 4), get_vmm(k, 0));
                vmovups(EVEX_compress_addr(reg_tr_src, tr_src_off_base + 64),
                        get_vmm(k, 3));
                if (do_compute_compensation_)
                    vpdpbusd(get_comp_acc(1), vmm_comp_mul, get_vmm(k, 3));
            } else if (conf_->wei_n_blk > 16) {
                vmovups(EVEX_compress_addr(reg_tr_src, tr_src_off_base + 64),
                        vmm_zero);
            }

            if (dynamic_tail) {
                cmp(reg_dynamic_tail, 32);
                jle(k_loop_done, T_NEAR);
            }
            if (ncolumns > 32 || dynamic_tail) {
                vmovups(get_vmm(k, 4), vreg_idx_lo_128);
                vpermi2b(get_vmm(k, 4), get_vmm(k, 5), get_vmm(k, 2));
                vmovups(EVEX_compress_addr(reg_tr_src, tr_src_off_base + 128),
                        get_vmm(k, 4));
                if (do_compute_compensation_)
                    vpdpbusd(get_comp_acc(2), vmm_comp_mul, get_vmm(k, 4));
            } else if (conf_->wei_n_blk > 32) {
                vmovups(EVEX_compress_addr(reg_tr_src, tr_src_off_base + 128),
                        vmm_zero);
            }

            if (dynamic_tail) {
                cmp(reg_dynamic_tail, 48);
                jle(k_loop_done, T_NEAR);
            }
            if (ncolumns > 48 || dynamic_tail) {
                vmovups(get_vmm(k, 0), vreg_idx_hi_128);
                vpermi2b(get_vmm(k, 0), get_vmm(k, 5), get_vmm(k, 2));
                vmovups(EVEX_compress_addr(reg_tr_src, tr_src_off_base + 192),
                        get_vmm(k, 0));
                if (do_compute_compensation_)
                    vpdpbusd(get_comp_acc(3), vmm_comp_mul, get_vmm(k, 0));
            } else if (conf_->wei_n_blk > 48) {
                vmovups(EVEX_compress_addr(reg_tr_src, tr_src_off_base + 192),
                        vmm_zero);
            }
            L(k_loop_done);
        }
    }
};

struct jit_avx512_core_brgemm_matmul_copy_b_int8_t
    : public jit_brgemm_matmul_copy_b_int8_t<Xbyak::Zmm> {

    jit_avx512_core_brgemm_matmul_copy_b_int8_t(
            const brgemm_matmul_conf_t *conf)
        : jit_brgemm_matmul_copy_b_int8_t<Xbyak::Zmm>(conf) {}

private:
    void init_permute() override {
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
    }

    void copy_4x64(int nrows, int ncolumns) override {
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
    }
};

struct jit_avx2_vnni_brgemm_matmul_copy_b_int8_t
    : public jit_brgemm_matmul_copy_b_int8_t<Xbyak::Ymm> {

    jit_avx2_vnni_brgemm_matmul_copy_b_int8_t(const brgemm_matmul_conf_t *conf)
        : jit_brgemm_matmul_copy_b_int8_t<Xbyak::Ymm>(conf) {}

private:
    static constexpr int perm2i128_l
            = 0x20; // dst[127:0]=src1_low_128; dst[128:255]=src2_low_128
    static constexpr int perm2i128_h
            = 0x31; // dst[127:0]=src1_hi_128; dst[128:255]=src2_hi_128

    Xbyak::Ymm get_ymm(int idx) { return get_vmm(0, idx); }

    void load_ymm(int ymm_idx, size_t offset, bool is_tail, size_t tail_sz) {
        Xbyak::Ymm vmm_src = Xbyak::Ymm(ymm_idx);
        if (is_tail) {
            load_bytes(vmm_src, reg_src, offset, tail_sz);
        } else
            uni_vmovups(vmm_src, ptr[reg_src + offset]);
    }

    void copy_4x64(int nrows, int ncolumns) override {
        const bool is_tail = ncolumns < n_blk_step_;
        const int k_end = div_up(nrows, k_blk_step_);
        for_(int k = 0; k < k_end; k++)
        for (int pass = 0; pass < 2; ++pass) {
            if (pass == 0 && ncolumns >= simd_w_) mov(reg_src_backup, reg_src);
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
                    const auto offset
                            = (is_dynamic_stride_ ? 0 : i * src_stride_)
                            + pass * simd_w_;
                    load_ymm(i % 4, offset, do_tail, ncolumns - pass * simd_w_);
                    if (is_dynamic_stride_) add(reg_src, reg_src_stride);
                } else {
                    const auto src_ymm_1 = get_ymm(i % 4);
                    uni_vpxor(src_ymm_1, src_ymm_1, src_ymm_1);
                }
            }
            if (pass == 0 && ncolumns >= simd_w_) mov(reg_src, reg_src_backup);

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
    }
};

template <typename Vmm>
void jit_brgemm_matmul_copy_b_int8_t<Vmm>::generate() {
    preamble();
    sub(rsp, stack_space_needed_);

    if (avx512_core_dot_product_) {
        mov(reg_tmp.cvt16(), 1);
        vpbroadcastw(vmm_ones_words, reg_tmp.cvt16());
    }

    uni_vpxor(vmm_zero, vmm_zero, vmm_zero);
    mov(reg_src, ptr[param1 + GET_OFF(src)]);
    mov(reg_tr_src, ptr[param1 + GET_OFF(tr_src)]);
    mov(reg_K_iters, ptr[param1 + GET_OFF(current_K_iters)]);
    mov(reg_N_blk, ptr[param1 + GET_OFF(current_N_blk)]);
    if (is_dynamic_stride_) {
        mov(reg_src_stride, ptr[param1 + GET_OFF(dynamic_src_stride)]);
    }

    init_permute();

    if (do_compute_compensation_) {
        int n_iters = div_up(conf_->wei_n_blk, 16) * (is_ymm_ ? 2 : 1);
        for (int i = 0; i < n_iters; i++)
            uni_vpxor(get_comp_acc(i), get_comp_acc(i), get_comp_acc(i));
        mov(reg_tmp, 1);
        uni_vpbroadcastb(vmm_comp_mul, reg_tmp.cvt8());
    }

    auto compute_K_loop = [&](bool is_N_tail) {
        const int k_unroll = 4;
        int ncolumns = is_N_tail ? conf_->N_tail : conf_->N_blk;

        Label K_loop_unrolled, K_loop_single, K_loop_tail_or_done;
        cmp(reg_K_iters, k_unroll * k_blk_step_);
        jl(K_loop_single, T_NEAR);

        L(K_loop_unrolled);
        copy_block(k_unroll * k_blk_step_, ncolumns, is_N_tail);
        if (!is_dynamic_stride_)
            add(reg_src, k_unroll * k_blk_step_ * src_stride_);
        add(reg_tr_src, k_unroll * tr_src_stride_);

        sub(reg_K_iters, k_unroll * k_blk_step_);
        cmp(reg_K_iters, k_unroll * k_blk_step_);
        jge(K_loop_unrolled, T_NEAR);

        L(K_loop_single);
        cmp(reg_K_iters, k_blk_step_);
        jl(K_loop_tail_or_done, T_NEAR);

        copy_block(k_blk_step_, ncolumns, is_N_tail);
        if (!is_dynamic_stride_) add(reg_src, k_blk_step_ * src_stride_);
        add(reg_tr_src, tr_src_stride_);

        sub(reg_K_iters, k_blk_step_);
        jmp(K_loop_single, T_NEAR);

        L(K_loop_tail_or_done);

        int k_blk_tail = conf_->K % k_blk_step_;
        if (k_blk_tail > 0) {
            Label K_loop_done;
            cmp(reg_K_iters, 0);
            jle(K_loop_done, T_NEAR);

            copy_block(k_blk_tail, ncolumns, is_N_tail);
            sub(reg_K_iters, k_blk_tail);
            L(K_loop_done);
        }
    };

    Label done;
    cmp(reg_N_blk, 0);
    jle(done, T_NEAR);

    if (conf_->N_tail > 0 || is_dynamic_N_) {
        Label main_N_blk;
        cmp(reg_N_blk, conf_->N_blk);
        je(main_N_blk, T_NEAR);
        compute_K_loop(true);
        jmp(done, T_NEAR);

        L(main_N_blk);
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
                mov(reg_tmp, 0xffffffff);
                const auto vmm_all_bits_1 = vmm_comp_mul;
                uni_vpbroadcastd(vmm_all_bits_1, reg_tmp.cvt32());
                mov(reg_tmp, 0x1);
                const auto vmm_one_s32 = vmm_zero;
                uni_vpbroadcastd(vmm_one_s32, reg_tmp.cvt32());

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

    add(rsp, stack_space_needed_);
    postamble();
}

template <typename Vmm>
struct jit_brgemm_matmul_copy_b_bf16_t : public jit_brgemm_matmul_copy_b_t,
                                         public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_brgemm_matmul_copy_b_bf16_t)

    jit_brgemm_matmul_copy_b_bf16_t(const brgemm_matmul_conf_t *conf)
        : jit_brgemm_matmul_copy_b_t(conf)
        , jit_generator(jit_name())
        , typesize(conf->b_dt_sz)
        , tr_typesize(conf->tr_b_dt_sz)
        , scales_typesize(sizeof(float))
        , src_stride(conf->copy_B_wei_stride)
        , tr_src_stride(conf_->LDB * k_blk_step * tr_typesize)
        , scales_N_stride(conf_->N * scales_typesize)
        , is_src_int4(one_of(conf->orig_wei_dt, data_type::s4, data_type::u4))
        , is_dynamic_stride(is_runtime_value(src_stride))
        , is_dynamic_N(conf->is_runtime_N)
        , req_cvtps2bf16(conf->is_bf32 || conf->is_bf16_with_int_wei)
        , req_zp_b_shift(conf->has_zero_point_b && conf->with_wei_decompression)
        , req_apply_scales(conf->apply_scales_in_buffer_b)
        , typesize_scale(is_src_int4 ? 2 : 1) {}

    void operator()(ctx_t *ctx) override { jit_generator::operator()(ctx); }
    status_t create_kernel() override { return jit_generator::create_kernel(); }

private:
    using reg64_t = const Xbyak::Reg64;
    using reg32_t = const Xbyak::Reg32;
    using opmask_t = const Xbyak::Opmask;
    using zmm = const Xbyak::Zmm;
    using ymm = const Xbyak::Ymm;
    using Vmm_lower_t = typename vreg_traits<Vmm>::Vmm_lower_t;

    enum { k_blk_step = 2, n_blk_step = 16 };
    const int typesize, tr_typesize, scales_typesize;
    const dim_t src_stride, tr_src_stride, scales_N_stride;
    const bool is_src_int4;
    const bool is_dynamic_stride;
    const bool is_dynamic_N;
    const bool req_cvtps2bf16;
    const bool req_zp_b_shift;
    const bool req_apply_scales;
    const dim_t typesize_scale;

    constexpr static int reg_src_offs = 0;

    constexpr static int reg_tr_src_offs = 8;
    constexpr static int stack_space_needed = 16;

    opmask_t kTail = k7;
    opmask_t kFFFF = k6;
    opmask_t kTail_int4 = k5;
    opmask_t kAAAA = k4;
    opmask_t k5555 = k3;

    reg64_t reg_src = rax;
    reg64_t reg_tr_src = rbx;

    reg64_t reg_K_iters = r8;
    reg64_t reg_N_blk = r9;
    reg64_t reg_K_start = r10;
    reg64_t reg_src_stride = r11;
    reg64_t reg_src_stride_x2 = r12;
    reg64_t reg_src_load_0 = r13;
    reg64_t reg_src_load_1 = r14;
    reg64_t reg_tmp = r15;

    reg64_t reg_copy_block_n_shift = rsi;
    reg64_t reg_scales = rdx;

    reg64_t reg_dynamic_tail = rcx;
    Xbyak::Reg8 reg8_mask_shift = reg_dynamic_tail.cvt8();

    Vmm vmm_zero = Vmm(0);
    Vmm vmm_permw = Vmm(1);
    Vmm vmm_tmp = Vmm(1); // used only for avx2_vnni_2
    Vmm vmm_zp_b_shift = Vmm(2);
    Vmm vmm_permd = Vmm(3);

    void kmovx(Opmask k, unsigned w) {
        if (!isa_has_masks(conf_->isa)) return;
        const auto regw_tmp = reg_tmp.cvt32();
        if (is_dynamic_N) {
            mov(reg_tmp, 1);
            shl(reg_tmp, reg8_mask_shift /* reg_dynamic_tail.cvt8() == cl */);
            sub(reg_tmp, 1);
        } else
            mov(regw_tmp, w);
        if (req_cvtps2bf16)
            jit_generator::kmovw(k, regw_tmp);
        else
            jit_generator::kmovd(k, regw_tmp);
    }
    void copy_half_int4(const Zmm &zmm, const Ymm &ymm_half) {
        vinserti64x4(zmm, zmm, ymm_half, 1);
    }
    void copy_half_int4(const Ymm &ymm, const Xmm &xmm_half) {
        vinserti128(ymm, ymm, xmm_half, 1);
    }
    Vmm_lower_t maybe_mask(Vmm_lower_t vmm_lower, bool is_tail) {
        assert(is_src_int4);
        if (isa_has_masks(conf_->isa)) {
            return is_tail ? vmm_lower | kTail_int4 | T_z
                           : vmm_lower | kFFFF | T_z;
        } else {
            return vmm_lower;
        }
    }
    Vmm maybe_mask(Vmm vmm, bool is_tail) {
        if (isa_has_masks(conf_->isa)) {
            return is_tail ? vmm | kTail | T_z : vmm | kFFFF | T_z;
        } else {
            return vmm;
        }
    }
    void load_data(const Vmm vmm_in, const Xbyak::Operand &op, bool is_tail);
    void copy_block(int nrows, int ncolumns, bool n_tail);
    void copy_2x32(int nrows, int ncolumns);
    void init_masks();
    void generate() override;
};

template <typename Vmm>
void jit_brgemm_matmul_copy_b_bf16_t<Vmm>::load_data(
        const Vmm vmm_in, const Xbyak::Operand &op, bool is_tail) {
    const auto vmm = maybe_mask(vmm_in, is_tail);
    const auto vmm_lower = Vmm_lower_t(vmm.getIdx());
    MAYBE_UNUSED(vmm_lower);

    switch (conf_->orig_wei_dt) {
        case data_type::f32: uni_vmovups(vmm, op); break;
        case data_type::f16:
        case data_type::bf16: vmovdqu16(vmm, op); break;
        case data_type::s8: uni_vpmovsxbd(vmm, op); break;
        case data_type::u8: uni_vpmovzxbd(vmm, op); break;
        // For int4, we see two int4 as one int8 and extend them int32
        // low half stores in lower bytes of vmm and high half in higher
        // bytes of vmm, then permute them into correct order
        // Finally, we process the extend bytes for s4/u4 accordingly
        case data_type::s4:
            uni_vpmovsxbd(maybe_mask(vmm_lower, is_tail), op);
            copy_half_int4(vmm_in, vmm_lower);
            vpermd(vmm_in, vmm_permd, vmm_in);
            uni_vpslld(vmm_in | k5555, vmm_in, 28);
            vpsrad(vmm_in | k5555, vmm_in, 28);
            vpsrad(vmm_in | kAAAA, vmm_in, 4);
            break;
        case data_type::u4:
            uni_vpmovzxbd(maybe_mask(vmm_lower, is_tail), op);
            copy_half_int4(vmm_in, vmm_lower);
            vpermd(vmm_in, vmm_permd, vmm_in);
            uni_vpslld(vmm_in | k5555, vmm_in, 28);
            vpsrld(vmm_in | k5555, vmm_in, 28);
            vpsrld(vmm_in | kAAAA, vmm_in, 4);
            break;
        default: assert(!"unsupported data type");
    }
}

template <typename Vmm>
void jit_brgemm_matmul_copy_b_bf16_t<Vmm>::copy_2x32(int nrows, int ncolumns) {

    const int columns_tail = ncolumns % n_blk_step;
    if (columns_tail > 0 && columns_tail < n_blk_step) {
        const auto tail_mask = (1 << columns_tail) - 1;
        kmovx(kTail, tail_mask);
        if (is_src_int4) {
            const auto int4_tail_mask = (1 << (columns_tail / 2)) - 1;
            kmovx(kTail_int4, int4_tail_mask);
        }
    }

    static constexpr int blk_sz = k_blk_step;
    const int reserved_regs = is_src_int4 ? 4 : req_zp_b_shift ? 3 : 2;
    const int max_isa_regs = isa_num_vregs(conf_->isa);
    const int max_regs_available = max_isa_regs - reserved_regs;
    const int max_unroll = max_regs_available / blk_sz;

    auto get_vmm = [max_unroll, max_isa_regs, reserved_regs](int blk, int idx) {
        assert(idx >= 0 && idx < blk_sz && blk >= 0);
        auto reg_idx = reserved_regs + max_unroll * ((idx + 1) % blk_sz) + blk;
        UNUSED(max_isa_regs);
        assert(reg_idx >= reserved_regs && reg_idx < max_isa_regs);
        return Vmm(reg_idx);
    };

    auto load = [this, get_vmm, ncolumns, columns_tail](int blk, int k, int n) {
        auto src_reg = get_vmm(blk, k % k_blk_step);
        const bool is_tail = ncolumns - n < n_blk_step;
        auto src_load = maybe_mask(src_reg, is_tail);
        const auto offset
                = ((is_dynamic_stride ? 0 : k * src_stride) + (n * typesize))
                / typesize_scale;
        const auto reg_src_load
                = is_dynamic_stride && k % 2 != 0 ? reg_src_load_1 : reg_src;
        auto load_addr = maybe_EVEX_compress_addr(reg_src_load, offset);
        if (!isa_has_masks(conf_->isa)) {
            if (is_tail)
                load_bytes(src_load, load_addr, columns_tail * tr_typesize);
            else
                uni_vmovups(src_load, load_addr);
        } else {
            load_data(src_reg, load_addr, is_tail);
        }

        if (utils::one_of(conf_->orig_wei_dt, data_type::s8, data_type::u8,
                    data_type::s4, data_type::u4)) {
            if (req_zp_b_shift) uni_vpsubd(src_load, src_load, vmm_zp_b_shift);
            uni_vcvtdq2ps(src_load, src_load);
            if (req_apply_scales) {
                const auto scales_offset
                        = (is_dynamic_stride ? 0 : k * scales_N_stride)
                        + n * scales_typesize;
                const auto scales_addr
                        = maybe_EVEX_compress_addr(reg_scales, scales_offset);
                uni_vmulps(src_load, src_load, scales_addr);
            }

            if (conf_->wei_dt == data_type::f16)
                vcvtps2phx(Vmm_lower_t(src_reg.getIdx()), src_reg);
        }
    };

    int iter = 0;
    for_(int k = 0; k < nrows; k += k_blk_step)
    for (int n = 0; n < (is_dynamic_N ? ncolumns : conf_->wei_n_blk);
            n += n_blk_step) {
        const int k_blk = k / k_blk_step;
        const dim_t tr_src_off
                = k_blk * tr_src_stride + n * k_blk_step * tr_typesize;
        const auto store_addr
                = maybe_EVEX_compress_addr(reg_tr_src, tr_src_off);
        const auto store_addr_ymm1
                = ptr[reg_tr_src + tr_src_off + vreg_traits<Vmm>::vlen];
        const int blk_idx = iter % max_unroll;
        const auto src_vmm0 = get_vmm(blk_idx, 0);
        const auto src_zmm0 = zmm(src_vmm0.getIdx());
        const auto src_vmm1 = get_vmm(blk_idx, 1);
        if (is_dynamic_stride && n == 0) {
            if (k == 0) {
                mov(reg_src_load_1, reg_src);
                add(reg_src_load_1, reg_src_stride);
            } else {
                add(reg_src, reg_src_stride_x2);
                add(reg_src_load_1, reg_src_stride_x2);
            }
        }

        if (ncolumns - n <= 0) {
            uni_vmovups(store_addr, vmm_zero);
            if (!is_superset(conf_->isa, avx512_core))
                uni_vmovups(store_addr_ymm1, vmm_zero);
            continue;
        }

        load(blk_idx, k, n);

        if (nrows - k >= k_blk_step) {
            load(blk_idx, k + 1, n);
            if (req_cvtps2bf16) {
                vcvtne2ps2bf16(src_vmm0, src_vmm1, src_vmm0);
            } else if (is_superset(conf_->isa, avx512_core)) {
                const auto src_ymm1 = ymm(src_vmm1.getIdx());
                vinsertf64x4(src_zmm0, src_zmm0, src_ymm1, 1);
            }
        } else if (req_cvtps2bf16) {
            vcvtneps2bf16(ymm(src_vmm0.getIdx()), src_vmm0);
        } else if (!is_superset(conf_->isa, avx512_core)) {
            uni_vxorps(src_vmm1, src_vmm1, src_vmm1);
        }

        if (is_superset(conf_->isa, avx512_core)) {
            vpermw(src_zmm0, vmm_permw, src_zmm0);
            uni_vmovups(store_addr, src_zmm0);
        } else {
            assert(is_superset(conf_->isa, avx2));
            vpunpcklwd(vmm_tmp, src_vmm0, src_vmm1);
            vpunpckhwd(src_vmm1, src_vmm0, src_vmm1);
            vperm2i128(src_vmm0, vmm_tmp, src_vmm1, 0x20);
            vperm2i128(src_vmm1, vmm_tmp, src_vmm1, 0x31);
            uni_vmovups(store_addr, src_vmm0);
            uni_vmovups(store_addr_ymm1, src_vmm1);
        }

        iter++;
    }
    if (is_dynamic_stride && nrows > 0) {
        add(reg_src, nrows % 2 == 0 ? reg_src_stride_x2 : reg_src_stride);
    }
}

template <typename Vmm>
void jit_brgemm_matmul_copy_b_bf16_t<Vmm>::init_masks() {
    alignas(64) static constexpr const int16_t bf16_vnni_permute[32]
            = {0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23, 8, 24, 9,
                    25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31};

    if (is_superset(conf_->isa, avx512_core)) {
        kxnorw(kFFFF, kFFFF, kFFFF); // 1111 1111 1111 1111

        mov(reg_tmp, reinterpret_cast<size_t>(bf16_vnni_permute));
        vmovdqa64(vmm_permw, ptr[reg_tmp]);

        if (is_src_int4) {
            alignas(64) static constexpr const uint32_t int4_permute[16]
                    = {0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15};
            mov(reg_tmp, reinterpret_cast<size_t>(int4_permute));
            vmovdqa32(vmm_permd, ptr[reg_tmp]);

            kmovx(kAAAA, 0xaaaa);
            kmovx(k5555, 0x5555);
        }
    }
}

template <typename Vmm>
void jit_brgemm_matmul_copy_b_bf16_t<Vmm>::copy_block(
        int nrows, int ncolumns, bool n_tail) {
    if (!is_dynamic_N || !n_tail) {
        copy_2x32(nrows, ncolumns);
        return;
    }

    mov(reg_dynamic_tail, reg_N_blk);
    // dynamic tail processing: main loop with ncolumns = n_blk_step and
    // finally process tail < n_blk_step with dynamically computed mask
    // NOTE: for dynamic_stride case copy_2x32() shifts reg_src pointer
    // so we need to backup/restore its value for every iteration wrt n
    // except the last one

    mov(ptr[rsp + reg_tr_src_offs], reg_tr_src);
    xor_(reg_copy_block_n_shift, reg_copy_block_n_shift);

    Label loop_row_start, loop_row_tail, loop_row_done;
    cmp(reg_dynamic_tail, n_blk_step);
    jl(loop_row_tail, T_NEAR);
    L(loop_row_start);
    {
        mov(ptr[rsp + reg_src_offs], reg_src);
        add(reg_src, reg_copy_block_n_shift);
        copy_2x32(nrows, n_blk_step);
        add(reg_copy_block_n_shift, n_blk_step * typesize);
        add(reg_src, n_blk_step * typesize);
        add(reg_tr_src, n_blk_step * k_blk_step * tr_typesize);
        sub(reg_dynamic_tail, n_blk_step);

        cmp(reg_dynamic_tail, 0);
        jle(loop_row_done, T_NEAR);

        mov(reg_src, ptr[rsp + reg_src_offs]);

        cmp(reg_dynamic_tail, n_blk_step);
        jl(loop_row_tail, T_NEAR);

        jmp(loop_row_start, T_NEAR);
    }

    L(loop_row_tail);
    {
        cmp(reg_dynamic_tail, 0);
        jle(loop_row_done, T_NEAR);

        add(reg_src, reg_copy_block_n_shift);
        copy_2x32(nrows, 1 /* to force tail case */);
    }
    L(loop_row_done);

    // restore pointers
    sub(reg_src, reg_copy_block_n_shift);
    mov(reg_tr_src, ptr[rsp + reg_tr_src_offs]);
}

template <typename Vmm>
void jit_brgemm_matmul_copy_b_bf16_t<Vmm>::generate() {
    assert(tr_typesize == sizeof(bfloat16_t));
    preamble();
    sub(rsp, stack_space_needed);
    uni_vxorps(vmm_zero, vmm_zero, vmm_zero);

    mov(reg_src, ptr[param1 + GET_OFF(src)]);
    mov(reg_tr_src, ptr[param1 + GET_OFF(tr_src)]);
    mov(reg_K_iters, ptr[param1 + GET_OFF(current_K_iters)]);
    mov(reg_N_blk, ptr[param1 + GET_OFF(current_N_blk)]);
    mov(reg_scales, ptr[param1 + GET_OFF(scales_ptr)]);
    if (is_dynamic_stride) {
        mov(reg_src_stride, ptr[param1 + GET_OFF(dynamic_src_stride)]);
        mov(reg_src_stride_x2, ptr[param1 + GET_OFF(dynamic_src_stride)]);
        shl(reg_src_stride_x2, 1);
    }
    if (req_zp_b_shift) {
        mov(reg_tmp, ptr[param1 + GET_OFF(zp_b_value_ptr)]);
        uni_vpbroadcastd(vmm_zp_b_shift, ptr[reg_tmp]);
    }

    init_masks();
    auto compute_K_loop = [&](bool is_N_tail) {
        const int k_unroll = 8;
        int ncolumns = is_N_tail ? conf_->N_tail : conf_->N_blk;

        Label K_loop_unrolled, K_loop_single, K_loop_tail_or_done;
        cmp(reg_K_iters, k_unroll * k_blk_step);
        jl(K_loop_single, T_NEAR);

        L(K_loop_unrolled);
        copy_block(k_unroll * k_blk_step, ncolumns, is_N_tail);

        if (!is_dynamic_stride)
            add(reg_src, (k_unroll * k_blk_step * src_stride) / typesize_scale);
        if (req_apply_scales)
            add(reg_scales, k_unroll * k_blk_step * scales_N_stride);
        add(reg_tr_src, k_unroll * tr_src_stride);

        sub(reg_K_iters, k_unroll * k_blk_step);
        cmp(reg_K_iters, k_unroll * k_blk_step);
        jge(K_loop_unrolled, T_NEAR);

        L(K_loop_single);
        cmp(reg_K_iters, k_blk_step);
        jl(K_loop_tail_or_done, T_NEAR);

        copy_block(k_blk_step, ncolumns, is_N_tail);
        if (!is_dynamic_stride)
            add(reg_src, (k_blk_step * src_stride) / typesize_scale);
        if (req_apply_scales) add(reg_scales, k_blk_step * scales_N_stride);
        add(reg_tr_src, tr_src_stride);

        sub(reg_K_iters, k_blk_step);
        jmp(K_loop_single, T_NEAR);

        L(K_loop_tail_or_done);

        int k_blk_tail = conf_->K % k_blk_step;
        if (k_blk_tail > 0) {
            Label K_loop_done;
            cmp(reg_K_iters, 0);
            jle(K_loop_done, T_NEAR);

            copy_block(k_blk_tail, ncolumns, is_N_tail);
            sub(reg_K_iters, k_blk_tail);
            L(K_loop_done);
        }
    };

    Label done;
    cmp(reg_N_blk, 0);
    jle(done, T_NEAR);

    if (conf_->N_tail > 0 || is_dynamic_N) {
        Label main_N_blk;
        cmp(reg_N_blk, conf_->N_blk);
        je(main_N_blk, T_NEAR);
        compute_K_loop(true);
        jmp(done, T_NEAR);

        L(main_N_blk);
    }

    compute_K_loop(false);
    L(done);

    add(rsp, stack_space_needed);
    postamble();
}

template struct jit_brgemm_matmul_copy_b_bf16_t<Zmm>;
template struct jit_brgemm_matmul_copy_b_bf16_t<Ymm>;

template <typename Vmm>
struct jit_brgemm_matmul_copy_b_f32_t : public jit_brgemm_matmul_copy_b_t,
                                        public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_brgemm_matmul_copy_b_f32_t)

    jit_brgemm_matmul_copy_b_f32_t(const brgemm_matmul_conf_t *conf)
        : jit_brgemm_matmul_copy_b_t(conf)
        , jit_generator(jit_name())
        , dt_in_(conf->orig_wei_dt)
        , simd_w_(vreg_traits<Vmm>::vlen / sizeof(float))
        , is_src_int4_(one_of(conf->orig_wei_dt, data_type::s4, data_type::u4))
        , req_zp_b_shift_(
                  conf->has_zero_point_b && conf->with_wei_decompression)
        , req_apply_scales_(conf->apply_scales_in_buffer_b)
        , typesize_in_(types::data_type_size(dt_in_))
        , typesize_scale_(is_src_int4_ ? 2 : 1)
        , scales_typesize_(sizeof(float))
        , src_stride_(conf_->copy_B_wei_stride)
        , tr_src_stride_(conf_->LDB * typesize_out_)
        , scales_N_stride_(conf_->N * scales_typesize_) {}

    void operator()(ctx_t *ctx) override { jit_generator::operator()(ctx); }
    status_t create_kernel() override { return jit_generator::create_kernel(); }

private:
    using reg64_t = const Xbyak::Reg64;
    using reg32_t = const Xbyak::Reg32;
    using opmask_t = const Xbyak::Opmask;
    using Vmm_lower_t = typename vreg_traits<Vmm>::Vmm_lower_t;

    const data_type_t dt_in_;
    const int simd_w_;
    const bool is_src_int4_, req_zp_b_shift_, req_apply_scales_;
    const size_t typesize_in_, typesize_scale_, scales_typesize_;
    const size_t typesize_out_ = sizeof(float);
    dim_t src_stride_, tr_src_stride_, scales_N_stride_;

    opmask_t kTail = k7;
    opmask_t kFFFF = k6;
    opmask_t k5555 = k5;
    opmask_t kAAAA = k4;
    opmask_t kTail_int4 = k3;

    reg64_t reg_src = rax;
    reg64_t reg_tr_src = rbx;

    reg64_t reg_K_iters = r8;
    reg64_t reg_N_blk = r9;
    reg64_t reg_K_start = r10;
    reg64_t reg_tmp = r15;
    reg32_t regw_tmp = r15d;
    reg64_t reg_scales = rdx;

    Vmm vmm_zero = Vmm(0);
    Vmm vmm_permw = Vmm(1);
    Vmm vmm_permd = Vmm(2);
    Vmm vmm_zp_b_shift = Vmm(3);
    Ymm ymm_tail_mask = ymm1;

    inline void kmovw(Opmask k, unsigned w) {
        if (!isa_has_masks(conf_->isa)) return;
        mov(regw_tmp, w);
        jit_generator::kmovd(k, regw_tmp);
    }
    void copy_half_int4(const Zmm &zmm, const Ymm &ymm_half) {
        vinserti64x4(zmm, zmm, ymm_half, 1);
    }
    void copy_half_int4(const Ymm &ymm, const Xmm &xmm_half) {
        vinserti128(ymm, ymm, xmm_half, 1);
    }
    Vmm_lower_t maybe_mask(Vmm_lower_t vmm_lower, bool is_tail) {
        assert(is_src_int4_);
        return is_tail && isa_has_masks(conf_->isa)
                ? vmm_lower | kTail_int4 | T_z
                : vmm_lower;
    }
    Vmm maybe_mask(Vmm vmm, bool is_tail) {
        return is_tail && isa_has_masks(conf_->isa) ? vmm | kTail | T_z : vmm;
    }
    void load_data(const Vmm vmm_in, const Xbyak::Operand &op, bool is_tail);
    void copy_16_x_n_block(int nrows, int ncolumns);
    void compute_k_loop(int ncolumns);
    void generate() override;
};

template <typename Vmm>
void jit_brgemm_matmul_copy_b_f32_t<Vmm>::load_data(
        const Vmm vmm_in, const Xbyak::Operand &op, bool is_tail) {
    const auto vmm = maybe_mask(vmm_in, is_tail);
    const auto vmm_lower = Vmm_lower_t(vmm.getIdx());
    MAYBE_UNUSED(vmm_lower);

    switch (dt_in_) {
        case data_type::f32: uni_vmovups(vmm, op); break;
        case data_type::f16: vcvtph2psx(vmm, op); break;
        case data_type::s8: uni_vpmovsxbd(vmm, op); break;
        case data_type::u8: uni_vpmovzxbd(vmm, op); break;
        // For int4, we see two int4 as one int8 and extend them int32
        // low half stores in lower bytes of vmm and high half in higher
        // bytes of vmm, then permute them into correct order
        // Finally, we process the extend bytes for s4/u4 accordingly
        case data_type::s4:
            uni_vpmovsxbd(maybe_mask(vmm_lower, is_tail), op);
            copy_half_int4(vmm_in, vmm_lower);
            vpermd(vmm_in, vmm_permd, vmm_in);
            uni_vpslld(vmm_in | k5555, vmm_in, 28);
            vpsrad(vmm_in | k5555, vmm_in, 28);
            vpsrad(vmm_in | kAAAA, vmm_in, 4);
            break;
        case data_type::u4:
            uni_vpmovzxbd(maybe_mask(vmm_lower, is_tail), op);
            copy_half_int4(vmm_in, vmm_lower);
            vpermd(vmm_in, vmm_permd, vmm_in);
            uni_vpslld(vmm_in | k5555, vmm_in, 28);
            vpsrld(vmm_in | k5555, vmm_in, 28);
            vpsrld(vmm_in | kAAAA, vmm_in, 4);
            break;
        default: assert(!"unsupported data type");
    }

    if (one_of(dt_in_, data_type::s8, data_type::u8, data_type::s4,
                data_type::u4))
        uni_vcvtdq2ps(vmm_in, vmm_in);
}

template <typename Vmm>
void jit_brgemm_matmul_copy_b_f32_t<Vmm>::copy_16_x_n_block(
        int nrows, int ncolumns) {
    const int max_isa_regs = isa_num_vregs(conf_->isa);
    const int reserved_regs = req_zp_b_shift_ ? 4 : is_src_int4_ ? 3 : 2;
    const int max_regs_available = max_isa_regs - reserved_regs;

    auto get_vmm = [max_regs_available, reserved_regs](int reg_idx) {
        MAYBE_UNUSED(max_regs_available);
        MAYBE_UNUSED(reserved_regs); // some compilers detect it as unused
        assert(reg_idx >= 0 && reg_idx < max_regs_available);
        return Vmm(reg_idx + reserved_regs);
    };

    auto load = [this, get_vmm, ncolumns](int blk, int k, int n) {
        auto src_vmm = get_vmm(blk);
        const bool is_tail = ncolumns - n < simd_w_;
        auto addr = maybe_EVEX_compress_addr(reg_src,
                (k * src_stride_ + n * typesize_in_) / typesize_scale_);
        if (is_tail && !isa_has_masks(conf_->isa))
            vmaskmovps(src_vmm, ymm_tail_mask, addr);
        else
            load_data(src_vmm, addr, is_tail);

        if (req_zp_b_shift_)
            uni_vsubps(maybe_mask(src_vmm, is_tail), src_vmm, vmm_zp_b_shift);
        if (req_apply_scales_) {
            const auto scales_addr = maybe_EVEX_compress_addr(
                    reg_scales, k * scales_N_stride_ + n * scales_typesize_);
            vmulps(maybe_mask(src_vmm, is_tail), src_vmm, scales_addr);
        }
    };

    const int columns_tail = ncolumns % simd_w_;
    if (columns_tail < simd_w_) {
        if (isa_has_masks(conf_->isa)) {
            const auto tail_mask = (1 << columns_tail) - 1;
            kmovw(kTail, tail_mask);
            if (is_src_int4_) {
                const auto int4_tail_mask
                        = (1 << (columns_tail / typesize_scale_)) - 1;
                kmovw(kTail_int4, int4_tail_mask);
            }
        } else {
            init_f32_avx2_mask_ymm(ymm_tail_mask, reg_tmp, columns_tail);
        }
    }

    int iter = 0;
    for_(int k = 0; k < nrows; k++)
    for (int n = 0; n < conf_->wei_n_blk; n += simd_w_) {
        const dim_t tr_src_off = k * tr_src_stride_ + n * typesize_out_;
        const auto store_addr
                = maybe_EVEX_compress_addr(reg_tr_src, tr_src_off);

        const int zero_padding = ncolumns - n;
        if (zero_padding <= 0) {
            uni_vmovups(store_addr, vmm_zero);
            continue;
        }

        const int blk_idx = iter % max_regs_available;
        load(blk_idx, k, n);

        const auto src_vmm0 = get_vmm(blk_idx);
        uni_vmovups(store_addr, src_vmm0);
        iter++;
    }
}

template <typename Vmm>
void jit_brgemm_matmul_copy_b_f32_t<Vmm>::compute_k_loop(int ncolumns) {

    auto compute_uni_k_loop = [&](int unroll) {
        Label K_start_label, K_end_label;

        L(K_start_label);
        cmp(reg_K_iters, unroll);
        jl(K_end_label, T_NEAR);

        copy_16_x_n_block(unroll, ncolumns);
        add(reg_src, (unroll * src_stride_) / typesize_scale_);
        add(reg_tr_src, unroll * tr_src_stride_);
        if (req_apply_scales_) add(reg_scales, unroll * scales_N_stride_);

        sub(reg_K_iters, unroll);
        jmp(K_start_label, T_NEAR);

        L(K_end_label);
    };

    constexpr int k_unroll = 16;
    compute_uni_k_loop(k_unroll);
    compute_uni_k_loop(1);
}

template <typename Vmm>
void jit_brgemm_matmul_copy_b_f32_t<Vmm>::generate() {
    preamble();
    uni_vxorps(vmm_zero, vmm_zero, vmm_zero);

    mov(reg_src, ptr[param1 + GET_OFF(src)]);
    mov(reg_tr_src, ptr[param1 + GET_OFF(tr_src)]);
    mov(reg_K_iters, ptr[param1 + GET_OFF(current_K_iters)]);
    mov(reg_N_blk, ptr[param1 + GET_OFF(current_N_blk)]);
    mov(reg_scales, ptr[param1 + GET_OFF(scales_ptr)]);
    kmovw(kFFFF, 0xffff); // 1111111111111111
    if (is_src_int4_) {
        alignas(64) static constexpr const uint32_t int4_permute[16]
                = {0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15};
        mov(reg_tmp, reinterpret_cast<size_t>(int4_permute));
        vmovdqa32(vmm_permd, ptr[reg_tmp]);

        kmovw(kAAAA, 0xaaaa);
        kmovw(k5555, 0x5555);
    }
    if (req_zp_b_shift_) {
        mov(reg_tmp, ptr[param1 + GET_OFF(zp_b_value_ptr)]);
        uni_vpbroadcastd(vmm_zp_b_shift, ptr[reg_tmp]);
        uni_vcvtdq2ps(vmm_zp_b_shift, vmm_zp_b_shift);
    }

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

    postamble();
}

template struct jit_brgemm_matmul_copy_b_f32_t<Zmm>;
template struct jit_brgemm_matmul_copy_b_f32_t<Ymm>;

template <typename Vmm>
struct jit_brgemm_matmul_copy_b_transposed_t
    : public jit_brgemm_matmul_copy_b_t,
      public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_brgemm_matmul_copy_b_transposed_t)

    jit_brgemm_matmul_copy_b_transposed_t(const brgemm_matmul_conf_t *conf)
        : jit_brgemm_matmul_copy_b_t(conf)
        , jit_generator(jit_name())
        , typesize_(conf_->b_dt_sz)
        , tr_typesize_(conf_->tr_b_dt_sz)
        , scales_typesize_(sizeof(float))
        , vnni_granularity_(data_type_vnni_granularity(conf_->wei_dt))
        , k_blk_step_(vlen_ / tr_typesize_)
        , do_compute_compensation_(
                  conf_->has_zero_point_a || conf_->s8s8_compensation_required)
        , is_bf32_(conf->is_bf32)
        , is_bf16_with_int_wei_(conf->is_bf16_with_int_wei)
        , is_src_int4_(one_of(conf->orig_wei_dt, data_type::s4, data_type::u4))
        , req_cvtps2xf16_(conf->is_bf32 || conf->is_bf16_with_int_wei
                  || (conf->is_f16_with_int_wei
                          && conf->wei_dt == data_type::f16))
        , req_zp_comp_(conf_->has_zero_point_a)
        , req_s8s8_comp_(conf_->s8s8_compensation_required)
        , req_zp_b_shift_(
                  conf_->has_zero_point_b && conf_->with_wei_decompression)
        , req_apply_scales_(conf_->apply_scales_in_buffer_b)
        , avx512_core_dot_product_(
                  do_compute_compensation_ && !isa_has_int8_vnni(conf->isa))
        // See the note in `create_brgemm_matmul_copy_b` why `orig_wei_dt` used.
        , use_fp16_instructions_(conf_->isa == avx512_core_fp16
                  && conf_->orig_wei_dt == data_type::f16
                  && conf_->wei_dt == data_type::f32)
        , max_tmp_idx(16
                  - (avx512_core_dot_product_
                                  ? 8
                                  : (do_compute_compensation_       ? 6
                                                  : is_src_int4_    ? 2
                                                  : req_zp_b_shift_ ? 1
                                                                    : 0)))
        , src_stride_(conf_->copy_B_wei_stride)
        , tr_src_stride_(conf_->LDB * vnni_granularity_ * tr_typesize_)
        , scales_K_stride_(conf_->K * scales_typesize_)
        , typesize_scale_(is_src_int4_ ? 2 : 1)
        , is_dynamic_N_(conf->is_runtime_N) {}

    void operator()(ctx_t *ctx) override { jit_generator::operator()(ctx); }
    status_t create_kernel() override { return jit_generator::create_kernel(); }

private:
    using reg64_t = const Xbyak::Reg64;
    using reg32_t = const Xbyak::Reg32;
    using opmask_t = const Xbyak::Opmask;
    using Vmm_lower_t = typename vreg_traits<Vmm>::Vmm_lower_t;

    static constexpr bool is_ymm_ = std::is_same<Vmm, Xbyak::Ymm>::value;
    static constexpr cpu_isa_t isa_ = is_ymm_ ? avx2 : avx512_core;
    static constexpr int max_vmm_regs_ = cpu_isa_traits<isa_>::n_vregs;
    static constexpr int vlen_ = vreg_traits<Vmm>::vlen;
    static constexpr int n_blk_step_ = is_ymm_ ? 8 : 16;
    static constexpr int req_cvt_bf16_k_blk_step_ = 16;
    static constexpr size_t comp_shift_ = vlen_;

    const int typesize_;
    const int tr_typesize_;
    const int scales_typesize_;
    const int vnni_granularity_;
    const int k_blk_step_;
    const bool do_compute_compensation_;
    const bool is_bf32_;
    const bool is_bf16_with_int_wei_;
    const bool is_src_int4_;
    const bool req_cvtps2xf16_;
    const bool req_zp_comp_;
    const bool req_s8s8_comp_;
    const bool req_zp_b_shift_;
    const bool req_apply_scales_;
    const bool avx512_core_dot_product_;
    const bool use_fp16_instructions_;
    const int max_tmp_idx;

    const dim_t src_stride_, tr_src_stride_, scales_K_stride_, typesize_scale_;
    const bool is_dynamic_N_;

    opmask_t k3333 = k1;
    opmask_t k5555 = k2;
    opmask_t kAAAA = k3;
    opmask_t kCCCC = k4;
    opmask_t k0F0F = k5;
    opmask_t kF0F0 = k6;
    opmask_t kTail = k7;
    // reuse k7 for int4 and restore the value after use
    opmask_t kTail_int4 = k7;

    reg64_t reg_src_base = rax;
    reg64_t reg_tr_src_base = rbx;
    reg64_t reg_comp_ptr = rdx;
    reg64_t reg_scales_base = rsi;

    reg64_t reg_K_iters = r8;
    reg64_t reg_N_iters = r9;
    reg64_t reg_src = r10;
    reg64_t reg_tr_src = r11;
    reg64_t reg_zp_comp_ptr = r12;
    reg64_t reg_zp_a_neg_val_ptr = r13;
    reg64_t reg_K_start = r14;
    reg64_t reg_scales = rdx;

    reg64_t regq_tmp = r15;
    reg32_t regw_tmp = r15d;
    reg64_t imm_addr64 = abi_not_param1;

    // Note: for the AVX2 implementation, reserve Ymm(8) and Ymm(9) as
    // temporary compute registers.
    Vmm vmm_comp_mul = Vmm(max_vmm_regs_ - 1);
    Vmm vmm_comp_acc = Vmm(max_vmm_regs_ - 2);
    Vmm vmm_zp_a_neg_val = Vmm(max_vmm_regs_ - 3);
    Vmm vmm_s8s8_comp_acc = Vmm(max_vmm_regs_ - 4);
    Vmm vmm_all_bits_1 = Vmm(max_vmm_regs_ - 5);
    Vmm vmm_one_s32 = Vmm(max_vmm_regs_ - 6);

    // Required in every dot product for INT8 non-VNNI computation.
    Vmm vmm_ones_words = Vmm(max_vmm_regs_ - 7);
    Vmm vmm_dot_product_temp = Vmm(max_vmm_regs_ - 8);

    Vmm vmm_zp_b_val = Vmm(max_vmm_regs_ - 1);
    Vmm vmm_permd = Vmm(max_vmm_regs_ - 2);

    void kmovw(Opmask k, unsigned w) {
        mov(regw_tmp, w);
        jit_generator::kmovw(k, regw_tmp);
    };

    void kmovq(Opmask k, size_t q) {
        mov(regq_tmp, q);
        jit_generator::kmovq(k, regq_tmp);
    };

    Vmm src_vmm(int i) {
        assert(i >= 0 && i < n_blk_step_);
        return Vmm(i);
    }

    Vmm tmp_vmm(int i) {
        // If compensation compute is required - last 6 zmms are reserved for it
        assert(i >= 0 && IMPLICATION(!is_ymm_, i < max_tmp_idx)
                && IMPLICATION(is_ymm_, i < 2));
        return Vmm(n_blk_step_ + i);
    }

    void copy_half_int4(const Zmm &zmm, const Ymm &ymm_half) {
        vinserti64x4(zmm, zmm, ymm_half, 1);
    }

    void copy_half_int4(const Ymm &ymm, const Xmm &xmm_half) {
        vinserti128(ymm, ymm, xmm_half, 1);
    }

    Vmm_lower_t maybe_mask(Vmm_lower_t vmm_lower, bool is_tail) {
        assert(is_src_int4_);
        return isa_has_masks(conf_->isa) && is_tail
                ? vmm_lower | kTail_int4 | T_z
                : vmm_lower;
    }

    Vmm maybe_mask(Vmm vmm, bool is_tail) {
        return isa_has_masks(conf_->isa) && is_tail ? vmm | kTail | T_z : vmm;
    }

    void init_tail_mask(const int columns_tail, const bool use_int4_mask);
    void maybe_apply_scales(
            const Vmm vmm_in, const size_t offset, const bool is_tail);
    void maybe_apply_zp_b_shift(const Vmm vmm_in, const bool is_tail);
    void load_int(const Vmm vmm_in, const dim_t offset, const int i,
            const int columns_tail, bool is_tail);
    void copy_row_x_col(int nrows, int ncolumns);
    void compute_K_loop(bool is_N_tail, int curr_K_tail, bool is_first_K_iter,
            bool is_last_K_iter);
    void compute_N_loop(
            int curr_K_tail, bool is_first_K_iter, bool is_last_K_iter);

    inline void dot_product(Vmm v1, Vmm v2, Vmm v3) {
        if (!avx512_core_dot_product_)
            vpdpbusd(v1, v2, v3,
                    mayiuse(avx512_core) ? EvexEncoding : VexEncoding);
        else {
            vpmaddubsw(vmm_dot_product_temp, v2, v3);
            vpmaddwd(
                    vmm_dot_product_temp, vmm_dot_product_temp, vmm_ones_words);
            vpaddd(v1, v1, vmm_dot_product_temp);
        }
    }
    inline bool valid_to_load_next(int next_row_idx, int num_rows) {
        const bool dynamic_tail = is_dynamic_N_ && num_rows < n_blk_step_;
        return next_row_idx < num_rows || dynamic_tail;
    }

    void generate() override;
};

template <typename Vmm>
void jit_brgemm_matmul_copy_b_transposed_t<Vmm>::maybe_apply_scales(
        const Vmm vmm_in, const size_t offset, const bool is_tail) {
    if (!req_apply_scales_) return;

    const auto vmm = maybe_mask(vmm_in, is_tail);
    const auto scales_addr = EVEX_compress_addr(reg_scales, offset);
    vmulps(vmm, vmm, scales_addr);
}

template <typename Vmm>
void jit_brgemm_matmul_copy_b_transposed_t<Vmm>::maybe_apply_zp_b_shift(
        const Vmm vmm_in, const bool is_tail) {
    if (!req_zp_b_shift_) return;

    const auto vmm = maybe_mask(vmm_in, is_tail);
    vpsubd(vmm, vmm, vmm_zp_b_val);
}

template <typename Vmm>
void jit_brgemm_matmul_copy_b_transposed_t<Vmm>::init_tail_mask(
        const int columns_tail, const bool use_int4_mask) {
    assert(IMPLICATION(use_int4_mask, is_src_int4_));
    if (columns_tail > 0) {
        const int dt_step
                = req_cvtps2xf16_ || use_fp16_instructions_ ? 1 : typesize_;
        const auto tail_mask = use_int4_mask
                ? size_t(((size_t)1 << div_up(dt_step * columns_tail, 2)) - 1)
                : size_t(((size_t)1 << dt_step * columns_tail) - 1);
        if (req_cvtps2xf16_)
            kmovw(kTail, tail_mask);
        else
            kmovq(kTail, tail_mask);
    }
}

template <typename Vmm>
void jit_brgemm_matmul_copy_b_transposed_t<Vmm>::load_int(const Vmm vmm_in,
        const dim_t offset, const int i, int columns_tail, bool is_tail) {
    const auto vmm = maybe_mask(vmm_in, is_tail);
    const auto vmm_lower = Vmm_lower_t(vmm.getIdx());
    const auto xmm_in = Xmm(vmm_in.getIdx());
    const auto addr = EVEX_compress_addr(reg_src, offset);
    MAYBE_UNUSED(xmm_in);
    MAYBE_UNUSED(vmm_lower);
    if (is_src_int4_) init_tail_mask(columns_tail, true);

    // Two additional operations are needed for int4 when i * src_stride_ % 2 != 0.
    // The maximum data size for a bitwise shift is 8 bytes (quadwords).
    // If the loaded data size is smaller than 8, we can directly perform a right
    // shift to eliminate the unnecessary half-byte at the front.
    // If the loaded data size is 8, we need two registers to handle the
    // unnecessary half-byte at the front and back, respectively.
    const bool need_preload_int4 = is_src_int4_ && (i * src_stride_) % 2 != 0;
    const auto max_shift_sz = 8;
    if (need_preload_int4) {
        const auto load_sz = is_tail ? columns_tail
                : req_cvtps2xf16_    ? req_cvt_bf16_k_blk_step_ / 2
                                     : k_blk_step_ / 2;
        assert(load_sz <= max_shift_sz);
        if (load_sz < max_shift_sz) {
            load_bytes(xmm_in, addr, div_up(columns_tail, 2));
            vpsrlq(xmm_in, xmm_in, 4);
        } else {
            const auto xmm_tmp = Xmm(tmp_vmm(3).getIdx());
            load_bytes(xmm_in, addr, load_sz);
            load_bytes(
                    xmm_tmp, EVEX_compress_addr(reg_src, offset + 1), load_sz);
            vpsrlq(xmm_in, xmm_in, 4);
            vpsllq(xmm_tmp, xmm_tmp, 4);
            vpord(xmm_in, xmm_in, xmm_tmp);
        }
    }

    switch (conf_->orig_wei_dt) {
        case data_type::s8: uni_vpmovsxbd(vmm, addr); break;
        case data_type::u8: uni_vpmovzxbd(vmm, addr); break;
        // For int4, we see two int4 as one int8 and extend them int32
        // low half stores in lower bytes of vmm and high half in higher
        // bytes of vmm, then permute them into correct order
        // Finally, we process the extend bytes for s4/u4 accordingly
        case data_type::s4:
            if (need_preload_int4)
                uni_vpmovsxbd(maybe_mask(vmm_lower, is_tail), xmm_in);
            else
                uni_vpmovsxbd(maybe_mask(vmm_lower, is_tail), addr);
            copy_half_int4(vmm_in, vmm_lower);
            vpermd(vmm_in, vmm_permd, vmm_in);
            uni_vpslld(vmm_in | k5555, vmm_in, 28);
            vpsrad(vmm_in | k5555, vmm_in, 28);
            vpsrad(vmm_in | kAAAA, vmm_in, 4);
            break;
        case data_type::u4:
            if (need_preload_int4)
                uni_vpmovzxbd(maybe_mask(vmm_lower, is_tail), xmm_in);
            else
                uni_vpmovzxbd(maybe_mask(vmm_lower, is_tail), addr);
            copy_half_int4(vmm_in, vmm_lower);
            vpermd(vmm_in, vmm_permd, vmm_in);
            uni_vpslld(vmm_in | k5555, vmm_in, 28);
            vpsrld(vmm_in | k5555, vmm_in, 28);
            vpsrld(vmm_in | kAAAA, vmm_in, 4);
            break;
        default: assert(!"unsupported data type");
    }
    // restore the tail_mask
    if (is_src_int4_) init_tail_mask(columns_tail, false);
}

template <typename Vmm>
void jit_brgemm_matmul_copy_b_transposed_t<Vmm>::copy_row_x_col(
        int nrows, int ncolumns) {
    assert(nrows >= 0 && nrows <= n_blk_step_ && ncolumns >= 0
            && ncolumns <= k_blk_step_);
    if (!nrows) return;

    const int columns_tail = ncolumns
            % (req_cvtps2xf16_ ? req_cvt_bf16_k_blk_step_ : k_blk_step_);
    init_tail_mask(columns_tail, false);

    auto load2bf16 = [this, nrows, columns_tail, ncolumns](int i) {
        auto src_reg = src_vmm(i);
        auto src_reg_next = tmp_vmm(2);

        Label load_done;
        if (is_dynamic_N_ && nrows < n_blk_step_) {
            Label general_load;
            cmp(reg_N_iters, i);
            jg(general_load); // i < dynamic nrows -> general load

            // i >= dynamic nrows -> zero out values in src_reg
            vpxord(src_reg, src_reg, src_reg);
            jmp(load_done);

            L(general_load);
        } else if (i >= nrows) {
            vpxord(src_reg, src_reg, src_reg);
            return;
        }

        // check if k_tail exists and it's in the first zmm
        auto zmm_src = columns_tail > 0 && ncolumns < req_cvt_bf16_k_blk_step_
                ? src_reg | kTail | T_z
                : src_reg;
        const auto src_offset = (i * src_stride_) / typesize_scale_;
        const auto addr = EVEX_compress_addr(reg_src, src_offset);
        if (is_bf32_)
            vmovups(zmm_src, addr);
        else if (is_bf16_with_int_wei_ || conf_->is_f16_with_int_wei) {
            const bool is_tail
                    = columns_tail > 0 && ncolumns < req_cvt_bf16_k_blk_step_;
            load_int(src_reg, src_offset, i, columns_tail, is_tail);
            maybe_apply_zp_b_shift(src_reg, is_tail);
            vcvtdq2ps(zmm_src, zmm_src);
            maybe_apply_scales(src_reg, i * scales_K_stride_, is_tail);
        } else
            assert(!"Unsupported data type in loading");

        if (ncolumns <= req_cvt_bf16_k_blk_step_) {
            vpxord(src_reg_next, src_reg_next, src_reg_next);
        } else {
            auto zmm_src_next = columns_tail > 0 ? src_reg_next | kTail | T_z
                                                 : src_reg_next;
            const auto next_src_offset
                    = (i * src_stride_ + req_cvt_bf16_k_blk_step_ * typesize_)
                    / typesize_scale_;
            const auto next_addr = EVEX_compress_addr(reg_src, next_src_offset);
            if (is_bf32_)
                vmovups(zmm_src_next, next_addr);
            else if (is_bf16_with_int_wei_ || conf_->is_f16_with_int_wei) {
                const auto is_tail = columns_tail > 0;
                load_int(src_reg_next, next_src_offset, i, columns_tail,
                        columns_tail > 0);
                maybe_apply_zp_b_shift(src_reg_next, is_tail);
                vcvtdq2ps(zmm_src_next, zmm_src_next);
                maybe_apply_scales(src_reg_next,
                        i * scales_K_stride_
                                + req_cvt_bf16_k_blk_step_ * scales_typesize_,
                        is_tail);
            } else
                assert(!"Unsupported data type in loading");
        }

        if (conf_->wei_dt == data_type::bf16) {
            vcvtne2ps2bf16(src_reg, src_reg_next, src_reg);
        } else {
            const auto src_vmm_lower0 = Vmm_lower_t(src_reg.getIdx());
            const auto src_vmm_lower1 = Vmm_lower_t(src_reg_next.getIdx());
            vcvtps2phx(src_vmm_lower0, src_reg);
            vcvtps2phx(src_vmm_lower1, src_reg_next);
            vinsertf64x4(src_reg, src_reg, src_vmm_lower1, 1);
        }
        L(load_done);
    };

    auto load = [this, nrows, columns_tail](int i, int base_idx) {
        Label load_done;

        auto src_reg = src_vmm(i);
        if (is_dynamic_N_ && nrows < n_blk_step_) {
            Label general_load;
            cmp(reg_N_iters, i);
            jg(general_load); // i < dynamic nrows -> general load

            // i >= dynamic nrows -> zero out values in src_reg
            vpxord(src_reg, src_reg, src_reg);
            jmp(load_done);

            L(general_load);
        } else if (i >= nrows) {
            vpxord(src_reg, src_reg, src_reg);
            return;
        }

        const auto is_tail = columns_tail > 0;
        auto src_load = is_tail ? src_reg | kTail | T_z : src_reg;
        const auto src_offset = (i * src_stride_) / typesize_scale_;
        const auto addr = EVEX_compress_addr(reg_src, src_offset);
        if (conf_->is_f16_with_int_wei && conf_->wei_dt == data_type::f32) {
            load_int(src_reg, src_offset, i, columns_tail, is_tail);
            maybe_apply_zp_b_shift(src_reg, is_tail);
            vcvtdq2ps(src_load, src_load);
            maybe_apply_scales(src_reg, i * scales_K_stride_, is_tail);
        } else if (use_fp16_instructions_) {
            vcvtph2psx(src_load, addr);
        } else {
            vmovdqu8(src_load, addr);
        }
        L(load_done);
    };

    auto store = [this](Zmm r, int i) {
        auto addr = EVEX_compress_addr(reg_tr_src, i * tr_src_stride_);
        vmovups(addr, r);
    };

    auto transpose16x8 = [&](int base_idx) {
        assert(base_idx == 0 || base_idx == 8);

        // swap 1
        if (req_cvtps2xf16_) {
            for (int i = 0; i < 4; i++) {
                const int src_idx0 = base_idx + i * 2;
                const int src_idx1 = src_idx0 + 1;

                if (base_idx == 0 && i == 0) {
                    load2bf16(src_idx0);
                    load2bf16(src_idx1);
                }

                const int next_src_idx0 = src_idx0 + 2;
                const int next_src_idx1 = src_idx1 + 2;

                const bool load_next = base_idx == 0 || i < 3;

                const auto tmp0 = tmp_vmm(0);
                const auto tmp1 = tmp_vmm(1);
                const auto src0 = src_vmm(src_idx0);
                const auto src1 = src_vmm(src_idx1);

                if (valid_to_load_next(next_src_idx0, nrows) && load_next)
                    load2bf16(next_src_idx0);
                valignd(tmp0, src0, src0, 0x1);

                if (valid_to_load_next(next_src_idx1, nrows) && load_next)
                    load2bf16(next_src_idx1);
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
                    load(src_idx0, base_idx);
                    load(src_idx1, base_idx);
                }

                const auto tmp0 = tmp_vmm(0);
                const auto tmp1 = tmp_vmm(1);
                const auto src0 = src_vmm(src_idx0);
                const auto src1 = src_vmm(src_idx1);

                if (valid_to_load_next(next_src_idx0, nrows) && load_next)
                    load(next_src_idx0, base_idx);
                valignd(tmp0, src0, src0, 0x1);

                if (valid_to_load_next(next_src_idx1, nrows) && load_next)
                    load(next_src_idx1, base_idx);
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

            const auto tmp0 = tmp_vmm(0);
            const auto tmp1 = tmp_vmm(1);
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

            const auto tmp0 = tmp_vmm(0);
            const auto src0 = src_vmm(src_idx0);
            const auto src4 = src_vmm(src_idx4);

            vmovaps(tmp0, src0);
            vshuff32x4(src0 | kF0F0, src4, src4, 0xb1);
            vshuff32x4(src4 | k0F0F, tmp0, tmp0, 0xb1);
        }
    };

    auto fixup16x16 = [&]() {
        for (int i = 0; i < 8; i++) {
            const auto tmp = tmp_vmm(0);
            const auto src0 = src_vmm(i);
            const auto src8 = src_vmm(8 + i);
            vshuff64x2(tmp, src0, src8, 0x44);
            if (do_compute_compensation_)
                dot_product(vmm_comp_acc, vmm_comp_mul, tmp);
            store(tmp, i);
        }

        for (int i = 0; i < 8; i++) {
            const auto tmp = tmp_vmm(0);
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
}

template <>
void jit_brgemm_matmul_copy_b_transposed_t<Ymm>::copy_row_x_col(
        int nrows, int ncolumns) {
    assert(nrows >= 0 && nrows <= n_blk_step_ && ncolumns >= 0
            && ncolumns <= k_blk_step_);
    if (!nrows) return;

    const int columns_tail = ncolumns % k_blk_step_;
    auto load = [this, nrows, columns_tail](int i) {
        auto vmm_src = src_vmm(i);

        Label load_done;
        if (is_dynamic_N_ && nrows < n_blk_step_) {
            Label general_load;
            cmp(reg_N_iters, i);
            jg(general_load); // i < dynamic nrows -> general load

            // i >= dynamic nrows -> zero out values in src_reg
            vpxord(vmm_src, vmm_src, vmm_src);
            jmp(load_done);

            L(general_load);
        } else if (i >= nrows) {
            uni_vpxor(vmm_src, vmm_src, vmm_src);
            return;
        }
        if (columns_tail > 0) {
            load_bytes(vmm_src, reg_src, i * src_stride_,
                    columns_tail * typesize_);
        } else
            uni_vmovups(vmm_src, ptr[reg_src + i * src_stride_]);

        L(load_done);
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

        if (valid_to_load_next(next_src_idx0, nrows) && load_next) {
            load(next_src_idx0);
        }
        vperm2i128(tmp0, src0, src0, 0x1);
        vpalignr(tmp0, tmp0, src0, 0x4);

        if (valid_to_load_next(next_src_idx1, nrows) && load_next) {
            load(next_src_idx1);
        }
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
}

template <typename Vmm>
void jit_brgemm_matmul_copy_b_transposed_t<Vmm>::compute_K_loop(bool is_N_tail,
        int curr_K_tail, bool is_first_K_iter, bool is_last_K_iter) {
    MAYBE_UNUSED(is_first_K_iter);
    MAYBE_UNUSED(is_last_K_iter);
    const int N_chunk_tail = is_dynamic_N_
            ? 1 /* just to force tail processing */
            : conf_->N % n_blk_step_;
    const int nrows = is_N_tail ? N_chunk_tail : n_blk_step_;
    if (do_compute_compensation_)
        uni_vpxor(vmm_comp_acc, vmm_comp_acc, vmm_comp_acc);

    Label K_loop, K_loop_tail_or_done;
    mov(reg_K_iters, ptr[param1 + GET_OFF(current_K_iters)]);

    mov(reg_src, reg_src_base);
    mov(reg_tr_src, reg_tr_src_base);
    if (req_apply_scales_) mov(reg_scales, reg_scales_base);
    if (curr_K_tail > 0) {
        cmp(reg_K_iters, k_blk_step_);
        jl(K_loop_tail_or_done, T_NEAR);
    }

    L(K_loop);
    copy_row_x_col(nrows, k_blk_step_);
    add(reg_src, (k_blk_step_ * typesize_) / typesize_scale_);
    add(reg_tr_src, k_blk_step_ / vnni_granularity_ * tr_src_stride_);
    if (req_apply_scales_) add(reg_scales, k_blk_step_ * scales_typesize_);

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
}

template <typename Vmm>
void jit_brgemm_matmul_copy_b_transposed_t<Vmm>::compute_N_loop(
        int curr_K_tail, bool is_first_K_iter, bool is_last_K_iter) {
    const bool generate_N_tail = is_dynamic_N_ || (conf_->N % n_blk_step_ > 0);

    Label N_loop, N_loop_tail_or_done;
    if (generate_N_tail) {
        cmp(reg_N_iters, n_blk_step_);
        jl(N_loop_tail_or_done, T_NEAR);
    }

    L(N_loop);
    compute_K_loop(false, curr_K_tail, is_first_K_iter, is_last_K_iter);
    add(reg_src_base, (n_blk_step_ * src_stride_) / typesize_scale_);
    add(reg_tr_src_base, n_blk_step_ * vnni_granularity_ * tr_typesize_);
    if (req_apply_scales_) add(reg_scales_base, n_blk_step_ * scales_K_stride_);

    if (req_zp_comp_) add(reg_zp_comp_ptr, comp_shift_);
    if (req_s8s8_comp_) add(reg_comp_ptr, comp_shift_);

    sub(reg_N_iters, n_blk_step_);
    cmp(reg_N_iters, n_blk_step_);
    jge(N_loop, T_NEAR);

    L(N_loop_tail_or_done);
    if (generate_N_tail) {
        Label N_loop_done;
        cmp(reg_N_iters, 0);
        jle(N_loop_done, T_NEAR);

        compute_K_loop(true, curr_K_tail, is_first_K_iter, is_last_K_iter);
        L(N_loop_done);
    }
}

template <typename Vmm>
void jit_brgemm_matmul_copy_b_transposed_t<Vmm>::generate() {
    preamble();

    if (avx512_core_dot_product_) {
        mov(regq_tmp.cvt16(), 1);
        vpbroadcastw(vmm_ones_words, regq_tmp.cvt16());
    }
    if (req_zp_b_shift_) {
        mov(regq_tmp, ptr[param1 + GET_OFF(zp_b_value_ptr)]);
        uni_vpbroadcastd(vmm_zp_b_val, ptr[regq_tmp]);
    }

    mov(reg_src_base, ptr[param1 + GET_OFF(src)]);
    mov(reg_tr_src_base, ptr[param1 + GET_OFF(tr_src)]);
    mov(reg_K_iters, ptr[param1 + GET_OFF(current_K_iters)]);
    mov(reg_N_iters, ptr[param1 + GET_OFF(current_N_blk)]);
    mov(reg_scales_base, ptr[param1 + GET_OFF(scales_ptr)]);

    if (!is_ymm_) {
        kmovw(k5555, 0x5555);
        kmovw(kAAAA, 0xaaaa);
        kmovw(k3333, 0x3333);
        kmovw(kCCCC, 0xcccc);
        kmovw(k0F0F, 0x0f0f);
        kmovw(kF0F0, 0xf0f0);
    }
    if (is_src_int4_ && is_superset(conf_->isa, avx512_core)) {
        alignas(64) static constexpr const uint32_t int4_permute[16]
                = {0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15};
        mov(regq_tmp, reinterpret_cast<size_t>(int4_permute));
        vmovdqa32(vmm_permd, ptr[regq_tmp]);
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

    postamble();
}

template struct jit_brgemm_matmul_copy_b_transposed_t<Zmm>;
template struct jit_brgemm_matmul_copy_b_transposed_t<Ymm>;

template <typename Vmm>
struct jit_brgemm_matmul_copy_b_cvt_bf16_t : public jit_brgemm_matmul_copy_b_t,
                                             public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_brgemm_matmul_copy_b_cvt_bf16_t)

    jit_brgemm_matmul_copy_b_cvt_bf16_t(const brgemm_matmul_conf_t *conf)
        : jit_brgemm_matmul_copy_b_t(conf)
        , jit_generator(jit_name())
        , typesize_(conf->b_dt_sz)
        , tr_typesize_(conf->tr_b_dt_sz)
        , scales_typesize_(sizeof(float))
        , is_src_int4_(one_of(conf->orig_wei_dt, data_type::s4, data_type::u4))
        , typesize_scale_(is_src_int4_ ? 2 : 1)
        , src_stride_((conf->LDB * k_blk_step * typesize_) / typesize_scale_)
        , tr_src_stride_(conf_->LDB * k_blk_step * tr_typesize_)
        , scales_N_stride_(conf->N * scales_typesize_)
        , req_zp_b_shift_(
                  conf_->has_zero_point_b && conf_->with_wei_decompression)
        , req_apply_scales_(conf_->apply_scales_in_buffer_b)
        , reserved_regs_(req_apply_scales_  ? 5
                          : is_src_int4_    ? 2
                          : req_zp_b_shift_ ? 1
                                            : 0) {}

    void operator()(ctx_t *ctx) override { jit_generator::operator()(ctx); }
    status_t create_kernel() override { return jit_generator::create_kernel(); }

private:
    using reg64_t = const Xbyak::Reg64;
    using reg32_t = const Xbyak::Reg32;
    using opmask_t = const Xbyak::Opmask;
    using Vmm_lower_t = typename vreg_traits<Vmm>::Vmm_lower_t;
    using zmm = const Xbyak::Zmm;
    using ymm = const Xbyak::Ymm;

    enum { k_blk_step = 2, n_blk_step = 16 };
    const int typesize_, tr_typesize_, scales_typesize_;
    const bool is_src_int4_;
    const dim_t typesize_scale_, src_stride_, tr_src_stride_, scales_N_stride_;
    const bool req_zp_b_shift_;
    const bool req_apply_scales_;
    const int reserved_regs_;

    opmask_t kTail = k7;
    opmask_t kFFFF = k6;
    opmask_t kAAAA = k5;
    opmask_t k5555 = k4;

    reg64_t reg_src = rax;
    reg64_t reg_tr_src = rbx;

    reg64_t reg_K_iters = r8;
    reg64_t reg_N_blk = r9;
    reg64_t reg_scales = r10;
    reg64_t reg_tmp = r11;
    reg32_t regw_tmp = r11d;

    Vmm vmm_zp_b_val = Vmm(0);
    Vmm vmm_permd = Vmm(1);
    Vmm vmm_scales0 = Vmm(2);
    Vmm vmm_scales1 = Vmm(3);
    Vmm vmm_tmp = Vmm(4);

    void copy_half_int4(const Zmm &zmm, const Ymm &ymm_half) {
        vinserti64x4(zmm, zmm, ymm_half, 1);
    }
    Vmm maybe_mask(Vmm vmm, bool is_tail) {
        if (isa_has_masks(conf_->isa)) {
            return is_tail ? vmm | kTail | T_z : vmm | kFFFF | T_z;
        } else {
            return vmm;
        }
    }

    Vmm get_vmm(const int blk, const int idx) {
        const int max_isa_regs = isa_num_vregs(conf_->isa);
        const int max_unroll = (max_isa_regs - reserved_regs_) / k_blk_step;
        assert(idx >= 0 && idx < k_blk_step && blk >= 0);
        const auto reg_idx
                = max_unroll * ((idx + 1) % k_blk_step) + blk + reserved_regs_;
        assert(reg_idx >= reserved_regs_ && reg_idx < max_isa_regs);
        return Vmm(reg_idx);
    }

    void init_masks();
    void load_int(const Vmm vmm_in, const Xbyak::Operand &op);
    void get_scales(const int blk, const int k, const int n,
            const bool is_n_tail, const bool is_k_tail);
    void copy_block(const int nrows, const int ncolumns);
    void generate() override;
};

template <typename Vmm>
void jit_brgemm_matmul_copy_b_cvt_bf16_t<Vmm>::init_masks() {
    alignas(64) static constexpr const uint32_t bf16_vnni_permute[16]
            = {0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15};

    if (is_superset(conf_->isa, avx512_core)) {
        kxnorw(kFFFF, kFFFF, kFFFF); // 1111 1111 1111 1111

        mov(reg_tmp, reinterpret_cast<size_t>(bf16_vnni_permute));
        vmovdqa32(vmm_permd, ptr[reg_tmp]);

        mov(regw_tmp, 0x5555);
        kmovw(k5555, regw_tmp);
        mov(regw_tmp, 0xaaaa);
        kmovw(kAAAA, regw_tmp);
    }
}

template <typename Vmm>
void jit_brgemm_matmul_copy_b_cvt_bf16_t<Vmm>::load_int(
        const Vmm vmm_in, const Xbyak::Operand &op) {
    const auto vmm_lower = Vmm_lower_t(vmm_in.getIdx());
    MAYBE_UNUSED(vmm_lower);

    switch (conf_->orig_wei_dt) {
        case data_type::s8: uni_vpmovsxbd(vmm_in, op); break;
        case data_type::u8: uni_vpmovzxbd(vmm_in, op); break;
        // For int4, we see two int4 as one int8 and extend them int32
        // low half stores in lower bytes of vmm and high half in higher
        // bytes of vmm, then permute them into correct order
        // Finally, we process the extend bytes for s4/u4 accordingly
        case data_type::s4:
            uni_vpmovsxbd(vmm_lower, op);
            copy_half_int4(vmm_in, vmm_lower);
            vpermd(vmm_in, vmm_permd, vmm_in);
            uni_vpslld(vmm_in | k5555, vmm_in, 28);
            vpsrad(vmm_in | k5555, vmm_in, 28);
            vpsrad(vmm_in | kAAAA, vmm_in, 4);
            break;
        case data_type::u4:
            uni_vpmovzxbd(vmm_lower, op);
            copy_half_int4(vmm_in, vmm_lower);
            vpermd(vmm_in, vmm_permd, vmm_in);
            uni_vpslld(vmm_in | k5555, vmm_in, 28);
            vpsrld(vmm_in | k5555, vmm_in, 28);
            vpsrld(vmm_in | kAAAA, vmm_in, 4);
            break;
        default: assert(!"unsupported data type");
    }
}

template <typename Vmm>
void jit_brgemm_matmul_copy_b_cvt_bf16_t<Vmm>::get_scales(const int blk,
        const int k, const int n, const bool is_n_tail, const bool is_k_tail) {
    const auto zmm_scales1 = maybe_mask(vmm_scales1, is_n_tail);
    const auto zmm_tmp = maybe_mask(vmm_tmp, is_n_tail);
    const auto base_offset = k * scales_N_stride_ + n * scales_typesize_;
    auto scales_addr0 = maybe_EVEX_compress_addr(reg_scales, base_offset);
    auto scales_addr1 = maybe_EVEX_compress_addr(
            reg_scales, (k + 1) * scales_N_stride_ + n * scales_typesize_);
    vmovups(zmm_tmp, scales_addr0);
    if (is_k_tail)
        vpxord(vmm_scales1, vmm_scales1, vmm_scales1);
    else
        vmovups(zmm_scales1, scales_addr1);

    vinsertf64x4(vmm_scales0, vmm_tmp, Ymm(vmm_scales1.getIdx()), 1);
    vextractf64x4(Ymm(vmm_tmp.getIdx()), vmm_tmp, 1);
    vinsertf64x4(vmm_scales1, zmm_scales1, Ymm(vmm_tmp.getIdx()), 0);
    vpermd(vmm_scales0, vmm_permd, vmm_scales0);
    vpermd(vmm_scales1, vmm_permd, vmm_scales1);
}

template <typename Vmm>
void jit_brgemm_matmul_copy_b_cvt_bf16_t<Vmm>::copy_block(
        const int nrows, int ncolumns) {
    const int columns_tail = ncolumns % n_blk_step;
    if (columns_tail > 0 && columns_tail < n_blk_step) {
        const auto regw_tmp = reg_tmp.cvt32();
        const auto tail_mask = (1 << columns_tail) - 1;
        mov(regw_tmp, tail_mask);
        kmovw(kTail, regw_tmp);
    }

    static constexpr int blk_sz = k_blk_step;
    const int max_regs_available = isa_num_vregs(conf_->isa) - reserved_regs_;
    const int max_unroll = max_regs_available / blk_sz;

    // Every load converts unroll * k_blk_step * n_blk_step
    auto load = [this, nrows, ncolumns](int blk, int k, int n) {
        const int k_blk = k / k_blk_step;
        const auto src_vmm0 = get_vmm(blk, 0);
        const auto src_vmm1 = get_vmm(blk, 1);
        const dim_t offset = k_blk * src_stride_
                + (n * k_blk_step * typesize_) / typesize_scale_;
        const auto stride = (n_blk_step * typesize_) / typesize_scale_;
        auto load_addr0 = maybe_EVEX_compress_addr(reg_src, offset);
        auto load_addr1 = maybe_EVEX_compress_addr(reg_src, offset + stride);
        load_int(src_vmm0, load_addr0);
        load_int(src_vmm1, load_addr1);
        if (req_zp_b_shift_) {
            vpsubd(src_vmm0, src_vmm0, vmm_zp_b_val);
            vpsubd(src_vmm1, src_vmm1, vmm_zp_b_val);
        }
        vcvtdq2ps(src_vmm0, src_vmm0);
        vcvtdq2ps(src_vmm1, src_vmm1);
        if (req_apply_scales_) {
            const bool is_n_tail = ncolumns - n < n_blk_step;
            const bool is_k_tail = nrows - k < k_blk_step;
            get_scales(blk, k, n, is_n_tail, is_k_tail);
            vmulps(src_vmm0, src_vmm0, vmm_scales0);
            vmulps(src_vmm1, src_vmm1, vmm_scales1);
        }

        if (conf_->wei_dt == data_type::bf16) {
            vcvtne2ps2bf16(src_vmm0, src_vmm1, src_vmm0);
        } else {
            const auto src_vmm_lower0 = Vmm_lower_t(src_vmm0.getIdx());
            const auto src_vmm_lower1 = Vmm_lower_t(src_vmm1.getIdx());
            vcvtps2phx(src_vmm_lower0, src_vmm0);
            vcvtps2phx(src_vmm_lower1, src_vmm1);
            vinsertf64x4(src_vmm0, src_vmm0, src_vmm_lower1, 1);
        }
    };

    int iter = 0;
    for_(int k = 0; k < nrows; k += k_blk_step)
    for (int n = 0; n < ncolumns; n += n_blk_step) {
        const int k_blk = k / k_blk_step;
        const dim_t tr_src_off
                = k_blk * tr_src_stride_ + n * k_blk_step * tr_typesize_;
        const auto store_addr
                = maybe_EVEX_compress_addr(reg_tr_src, tr_src_off);
        const int blk_idx = iter % max_unroll;

        load(blk_idx, k, n);
        uni_vmovups(store_addr, get_vmm(blk_idx, 0));

        iter++;
    }
}

template <typename Vmm>
void jit_brgemm_matmul_copy_b_cvt_bf16_t<Vmm>::generate() {
    assert(tr_typesize_ == sizeof(bfloat16_t));
    preamble();

    init_masks();

    mov(reg_src, ptr[param1 + GET_OFF(src)]);
    mov(reg_tr_src, ptr[param1 + GET_OFF(tr_src)]);
    mov(reg_K_iters, ptr[param1 + GET_OFF(current_K_iters)]);
    mov(reg_N_blk, ptr[param1 + GET_OFF(current_N_blk)]);
    mov(reg_scales, ptr[param1 + GET_OFF(scales_ptr)]);

    if (req_zp_b_shift_) {
        mov(reg_tmp, ptr[param1 + GET_OFF(zp_b_value_ptr)]);
        uni_vpbroadcastd(vmm_zp_b_val, ptr[reg_tmp]);
    }

    auto compute_K_loop = [&](const int ncolumns) {
        const int k_unroll = 8;

        Label K_loop_unrolled, K_loop_single, K_loop_tail_or_done;
        cmp(reg_K_iters, k_unroll * k_blk_step);
        jl(K_loop_single, T_NEAR);

        L(K_loop_unrolled);
        copy_block(k_unroll * k_blk_step, ncolumns);
        add(reg_src, k_unroll * src_stride_);
        add(reg_tr_src, k_unroll * tr_src_stride_);
        if (req_apply_scales_)
            add(reg_scales, k_unroll * k_blk_step * scales_N_stride_);

        sub(reg_K_iters, k_unroll * k_blk_step);
        cmp(reg_K_iters, k_unroll * k_blk_step);
        jge(K_loop_unrolled, T_NEAR);

        L(K_loop_single);
        cmp(reg_K_iters, k_blk_step);
        jl(K_loop_tail_or_done, T_NEAR);

        copy_block(k_blk_step, ncolumns);
        add(reg_src, src_stride_);
        add(reg_tr_src, tr_src_stride_);
        if (req_apply_scales_) add(reg_scales, k_blk_step * scales_N_stride_);

        sub(reg_K_iters, k_blk_step);
        jmp(K_loop_single, T_NEAR);

        L(K_loop_tail_or_done);

        const int k_blk_tail = conf_->K % k_blk_step;
        if (k_blk_tail > 0) {
            Label K_loop_done;
            cmp(reg_K_iters, 0);
            jle(K_loop_done, T_NEAR);

            copy_block(k_blk_tail, ncolumns);
            sub(reg_K_iters, k_blk_tail);
            L(K_loop_done);
        }
    };

    Label done;
    cmp(reg_N_blk, 0);
    jle(done, T_NEAR);

    if (conf_->N_tail > 0) {
        Label main_N_blk;
        cmp(reg_N_blk, conf_->N_blk);
        je(main_N_blk, T_NEAR);
        compute_K_loop(conf_->N_tail);
        jmp(done, T_NEAR);

        L(main_N_blk);
    }

    compute_K_loop(conf_->N_blk);
    L(done);

    postamble();
}

template struct jit_brgemm_matmul_copy_b_cvt_bf16_t<Zmm>;
status_t create_brgemm_matmul_copy_b(
        std::unique_ptr<jit_brgemm_matmul_copy_b_t> &copy_ker,
        const brgemm_matmul_conf_t *conf) {
    const bool is_bf16
            = everyone_is(data_type::bf16, conf->src_dt, conf->wei_dt);
    const bool is_f32 = everyone_is(data_type::f32, conf->src_dt, conf->wei_dt);
    // Note: f16 support through avx512_core_fp16 sets src_dt and wei_dt as f32
    // to imply upconverting. So, the assumption is `is_f16` below evaluates to
    // `false` on avx512_core_fp16.
    const bool is_f16 = everyone_is(data_type::f16, conf->src_dt, conf->wei_dt);
    if (conf->transposed_B) {
        if (is_superset(conf->isa, avx512_core))
            CHECK(safe_ptr_assign(copy_ker,
                    new jit_brgemm_matmul_copy_b_transposed_t<Zmm>(conf)));
        else {
            assert(is_superset(conf->isa, avx2));
            CHECK(safe_ptr_assign(copy_ker,
                    new jit_brgemm_matmul_copy_b_transposed_t<Ymm>(conf)));
        }
    } else {
        if ((conf->is_bf16_with_int_wei
                    || (conf->is_f16_with_int_wei
                            && conf->isa != avx512_core_fp16))
                && conf->blocked_B) {
            if (is_superset(conf->isa, avx512_core))
                CHECK(safe_ptr_assign(copy_ker,
                        new jit_brgemm_matmul_copy_b_cvt_bf16_t<Zmm>(conf)));
            else {
                assert(!"Unsupported isa for bf16_with_int_wei");
                return status::unimplemented;
            }
        } else if (is_bf16 || is_f16 || conf->is_bf32
                || (conf->is_f16_with_int_wei
                        && conf->isa != avx512_core_fp16)) {
            if (is_superset(conf->isa, avx512_core))
                CHECK(safe_ptr_assign(copy_ker,
                        new jit_brgemm_matmul_copy_b_bf16_t<Zmm>(conf)));
            else
                CHECK(safe_ptr_assign(copy_ker,
                        new jit_brgemm_matmul_copy_b_bf16_t<Ymm>(conf)));
        } else if (is_f32
                || (conf->isa == avx512_core_fp16
                        && conf->orig_wei_dt == data_type::f16)) {
            // See the note above why `orig_wei_dt` is used.
            if (is_superset(conf->isa, avx512_core))
                CHECK(safe_ptr_assign(copy_ker,
                        new jit_brgemm_matmul_copy_b_f32_t<Zmm>(conf)));
            else
                CHECK(safe_ptr_assign(copy_ker,
                        new jit_brgemm_matmul_copy_b_f32_t<Ymm>(conf)));
        } else {
            if (mayiuse(avx512_core_amx))
                CHECK(safe_ptr_assign(copy_ker,
                        new jit_amx_brgemm_matmul_copy_b_int8_t(conf)));
            else if (is_superset(conf->isa, avx512_core))
                CHECK(safe_ptr_assign(copy_ker,
                        new jit_avx512_core_brgemm_matmul_copy_b_int8_t(conf)));
            else {
                assert(one_of(conf->isa, avx2_vnni, avx2_vnni_2));
                CHECK(safe_ptr_assign(copy_ker,
                        new jit_avx2_vnni_brgemm_matmul_copy_b_int8_t(conf)));
            }
        }
    }

    return copy_ker->create_kernel();
}

status_t create_brgemm_matmul_copy_a(
        std::unique_ptr<jit_brgemm_matmul_copy_a_t> &copy_ker,
        const brgemm_matmul_conf_t *conf) {
    if (conf->transposed_A) {
        if (utils::one_of(conf->src_dt, data_type::s8, data_type::u8))
            CHECK(safe_ptr_assign(copy_ker,
                    new jit_brgemm_matmul_copy_a_transposed_int8_impl_t(conf)));
        else if (is_superset(conf->isa, avx512_core))
            CHECK(safe_ptr_assign(copy_ker,
                    new jit_brgemm_matmul_copy_a_transposed_impl_t<Zmm>(conf)));
        else
            CHECK(safe_ptr_assign(copy_ker,
                    new jit_brgemm_matmul_copy_a_transposed_impl_t<Ymm>(conf)));
    } else {
        if (is_superset(conf->isa, avx512_core))
            CHECK(safe_ptr_assign(
                    copy_ker, new jit_brgemm_matmul_copy_a_impl_t<Zmm>(conf)));
        else {
            if (is_superset(conf->isa, avx2)) {
                CHECK(safe_ptr_assign(copy_ker,
                        new jit_brgemm_matmul_copy_a_impl_t<Ymm>(conf)));
            } else {
                assert(!"Unsupported isa for jit_brgemm_matmul_copy_a_impl_t");
                return status::unimplemented;
            }
        }
    }

    return copy_ker->create_kernel();
}

} // namespace matmul
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
