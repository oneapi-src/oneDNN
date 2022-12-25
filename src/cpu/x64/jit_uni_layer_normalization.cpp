/*******************************************************************************
* Copyright 2019-2022 Intel Corporation
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

#include <assert.h>
#include <math.h>

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/reorder.hpp"
#include "common/type_helpers.hpp"

#include "cpu/cpu_primitive.hpp"

#include "cpu/x64/jit_avx512_core_bf16cvt.hpp"
#include "cpu/x64/jit_generator.hpp"
#include "cpu/x64/jit_uni_layer_normalization.hpp"
#include "cpu/x64/utils/jit_io_helper.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

using namespace memory_tracking::names;
using namespace data_type;
using namespace Xbyak;

cpu_isa_t get_io_isa(cpu_isa_t isa, bool has_f16, bool has_bf16) {
    // re-using avx512_core instantiation for xf16
    // re-using avx2 instantiation for xf16
    if (has_f16 || has_bf16)
        return is_superset(isa, avx512_core) ? (has_f16    ? avx512_core_fp16
                               : mayiuse(avx512_core_bf16) ? avx512_core_bf16
                                                           : avx512_core)
                                             : avx2_vnni_2;
    else
        return isa;
}

template <cpu_isa_t isa>
struct jit_stat_and_data_base_kernel_t : stat_and_data_kernel_t,
                                         public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_lnorm_stat_and_data_kernel_t);

    void operator()(const void *src, void *dst, const float *scale,
            const float *shift, float *mean, float *var,
            const float *src_scales, const float *dst_scales,
            const size_t block_size) const override {
        ker_args_t args;
        args.src = src;
        args.dst = dst;
        args.scale = scale;
        args.shift = shift;
        args.mean = mean;
        args.var = var;
        args.src_scales = src_scales;
        args.dst_scales = dst_scales;
        args.block_size
                = block_size * C_ * types::data_type_size(src_d_.data_type());
        args.eps = eps_;
        jit_generator::operator()(&args);
    }

    status_t create_kernel() override { return jit_generator::create_kernel(); }

    jit_stat_and_data_base_kernel_t(const layer_normalization_pd_t *pd)
        : stat_and_data_kernel_t(pd)
        , jit_generator(jit_name())
        , src_d_(pd_->src_md())
        , dst_d_(pd_->dst_md())
        , simd_w_(vlen / sizeof(float))
        , C_(pd_->norm_axis())
        , axis_simd_full_(C_ / simd_w_)
        , axis_simd_tail_(C_ % simd_w_)
        , use_scale_(pd_->use_scale())
        , use_shift_(pd_->use_shift())
        , save_stats_(pd_->is_training())
        , calculate_stats_(!pd_->stats_are_src())
        , eps_(pd_->desc()->layer_norm_epsilon)
        , has_ne_convert_src_xf16_(isa == avx2 && mayiuse(avx2_vnni_2)
                  && utils::one_of(src_d_.data_type(), data_type::f16,
                          data_type::bf16)) {

        io::io_conf_t io_conf;
        io::io_tail_conf_t io_tail_conf(simd_w_, axis_simd_tail_,
                tail_opmask_idx, vmm_tail_mask.getIdx(), reg_tmp);
        io::io_emu_bf16_conf_t io_bf16_conf(bf16_emu_zmm_1_idx,
                bf16_emu_zmm_2_idx, bf16_emu_zmm_3_idx, reg_tmp,
                bf16_emu_zmm_4_idx);
        io::io_saturation_conf_t io_saturation_conf(
                vmm_zero.getIdx(), vmm_saturation_ubound.getIdx(), reg_tmp);
        const auto io_isa = get_io_isa(isa,
                utils::one_of(f16, src_d_.data_type(), dst_d_.data_type()),
                utils::one_of(bf16, src_d_.data_type(), dst_d_.data_type()));
        io_ = io::jit_io_multi_dt_helper_t<Vmm>(this, io_isa,
                {src_d_.data_type(), dst_d_.data_type(), f32 /* stats */},
                io_conf, io_tail_conf, io_bf16_conf,
                {{dst_d_.data_type(), io_saturation_conf}});
    }

protected:
    static constexpr int unroll_factor_ = 4;
    using Vmm = typename cpu_isa_traits<isa>::Vmm;
    const AddressFrame &vmmword = (isa == sse41) ? xword
            : (isa == avx2)                      ? yword
                                                 : zword;
    const int vlen = cpu_isa_traits<isa>::vlen;

    struct ker_args_t {
        const void *src;
        void *dst;
        const float *scale;
        const float *shift;
        const float *mean;
        const float *var;
        const float *src_scales;
        const float *dst_scales;
        size_t block_size;
        float eps;
    };

    io::jit_io_multi_dt_helper_t<Vmm> io_;
    const memory_desc_wrapper src_d_, dst_d_;
    const size_t simd_w_;
    const dim_t C_;
    const dim_t axis_simd_full_;
    const dim_t axis_simd_tail_;
    const bool use_scale_;
    const bool use_shift_;
    const bool save_stats_;
    const bool calculate_stats_;
    const float eps_;
    const bool has_ne_convert_src_xf16_;

    const Reg64 reg_param = abi_param1;
    const Reg64 reg_src = rdx;
    const Reg64 reg_dst = rax;
    const Reg64 reg_mean = rbx;
    const Reg64 reg_scale = r8;
    const Reg64 reg_block_end = r9;
    const Reg64 reg_eps = r10;
    const Reg64 reg_tmp = r11;
    const Reg64 reg_shift = r12;
    const Reg64 reg_var = r13;
    const Reg64 reg_src_scales = r14;
    const Reg64 reg_dst_scales = r15;

    const Vmm vmm_tail_mask = Vmm(0);
    const Vmm vmm_zero = Vmm(4); // In unroll range, safe for dst compute.
    const Vmm vmm_saturation_ubound
            = Vmm(5); // In unroll range, safe for dst compute.
    const Vmm vmm_combined_scales
            = Vmm(6); // In unroll range, safe for dst compute.
    const Vmm vmm_scale = Vmm(7); // In unroll range, safe for dst compute.
    const Vmm vmm_shift = Vmm(8); // In unroll range, safe for dst compute.
    const Vmm vmm_ones = Vmm(9);
    const Vmm vmm_eps = Vmm(10);
    const Vmm vmm_c = Vmm(11);
    const Vmm vmm_mean = Vmm(12);
    const Vmm vmm_inv_sqrtvar = Vmm(13);
    const Vmm vmm_dst = Vmm(14);
    const Vmm vmm_tmp = Vmm(15);
    const Xmm xmm_tmp = Xmm(15);
    const Vmm vmm_dst_even = vmm_dst;
    const Vmm vmm_dst_odd = Vmm(3); // In unroll range, safe for dst compute

    const int bf16_emu_zmm_1_idx = 28;
    const int bf16_emu_zmm_2_idx = 29;
    const int bf16_emu_zmm_3_idx = 30;
    const int bf16_emu_zmm_4_idx = 31;
    const int tail_opmask_idx = 1;

    Address src_ptr(size_t offt = 0) {
        return vmmword[reg_src + offt * src_d_.data_type_size()];
    }

    Address dst_ptr(size_t offt = 0) {
        return vmmword[reg_dst + offt * dst_d_.data_type_size()];
    }

    Address mean_ptr(size_t offt = 0) {
        return vmmword[reg_mean + offt * sizeof(float)];
    }

    Address var_ptr(size_t offt = 0) {
        return vmmword[reg_var + offt * sizeof(float)];
    }

    Address scale_ptr(size_t offt = 0) {
        return vmmword[reg_scale + offt * sizeof(float)];
    }

    Address shift_ptr(size_t offt = 0) {
        return vmmword[reg_shift + offt * sizeof(float)];
    }

    virtual void reduce(Vmm vmm_src, Vmm vmm_tmp) = 0;

    void uni_vsubps_maybe_tail(const Vmm &x1, const Vmm &x2, const bool tail) {
        // Need to preserve zeros after subtract for correct answer.
        if (!tail)
            uni_vsubps(x1, x1, x2);
        else {
            if (is_superset(isa, avx512_core))
                uni_vsubps(x1 | Opmask(tail_opmask_idx) | T_z, x1, x2);
            else if (is_superset(isa, sse41)) {
                // We need to call tail version once, it's fine to use vmm_tmp
                uni_vpxor(vmm_tmp, vmm_tmp, vmm_tmp);
                uni_vblendvps(vmm_tmp, vmm_tmp, x2, vmm_tail_mask);
                uni_vsubps(x1, x1, vmm_tmp);
            }
        }
    }

    template <typename F>
    void compute_ne_convert_xf16(Vmm vmm_stat, F op) {
        bool need_tail = false;
        int base_idx = 1; // Preserve `0` for tail on AVX2.

        uni_vpxor(Vmm(base_idx), Vmm(base_idx), Vmm(base_idx));
        if (axis_simd_full_ > 0) {
            const int unroll
                    = axis_simd_full_ >= unroll_factor_ ? unroll_factor_ : 1;
            assert(math::is_pow2(unroll));

            for (int i = base_idx + 1; i < base_idx + unroll; i++)
                uni_vpxor(Vmm(i), Vmm(i), Vmm(i));

            // unrolled loop
            for (int i = 0; i < axis_simd_full_ / unroll; i++)
                for (int j = base_idx; j < base_idx + unroll; j += 2) {
                    const bool can_load_two_simdw = base_idx + unroll - j >= 2;
                    if (!can_load_two_simdw)
                        io_[src_d_.data_type()]->load(
                                src_ptr((i * unroll + j - base_idx) * simd_w_),
                                Vmm(j + unroll), need_tail);
                    else
                        io_[src_d_.data_type()]->load_two_simdw_xf16(
                                src_ptr((i * unroll + j - base_idx) * simd_w_),
                                Vmm(j + unroll), Vmm(j + 1 + unroll));
                    op(Vmm(j), Vmm(j + unroll), need_tail);
                    if (can_load_two_simdw)
                        op(Vmm(j + 1), Vmm(j + 1 + unroll), need_tail);
                }

            int n = unroll;
            while (n > 1) {
                for (int j = base_idx; j < base_idx + n / 2; j++)
                    uni_vaddps(Vmm(j), Vmm(j), Vmm(j + n / 2));
                n = n / 2;
            }

            // unrolled loop remainder
            for (int i = utils::rnd_dn(axis_simd_full_, unroll);
                    i < axis_simd_full_; i += 2) {
                const bool can_load_two_simdw = axis_simd_full_ - i >= 2;
                if (!can_load_two_simdw)
                    io_[src_d_.data_type()]->load(
                            src_ptr(i * simd_w_), Vmm(base_idx + 1), need_tail);
                else
                    io_[src_d_.data_type()]->load_two_simdw_xf16(
                            src_ptr(i * simd_w_), Vmm(base_idx + 1),
                            Vmm(base_idx + 2));
                op(Vmm(base_idx), Vmm(base_idx + 1), need_tail);
                if (can_load_two_simdw)
                    op(Vmm(base_idx), Vmm(base_idx + 2), need_tail);
            }
        }

        if (axis_simd_tail_ > 0) {
            need_tail = true;
            // vector remainder
            io_[src_d_.data_type()]->load(src_ptr(axis_simd_full_ * simd_w_),
                    Vmm(base_idx + 1), need_tail);
            op(Vmm(base_idx), Vmm(base_idx + 1), need_tail);
        }

        reduce(Vmm(base_idx), Vmm(base_idx + 1));
        uni_vdivps(Vmm(base_idx), Vmm(base_idx), vmm_c, vmm_tmp);
        uni_vmovups(vmm_stat, Vmm(base_idx));
    }

    template <typename F>
    void compute(Vmm vmm_stat, F op) {
        bool need_tail = false;
        int base_idx = 1; // Preserve `0` for tail on AVX2.

        uni_vpxor(Vmm(base_idx), Vmm(base_idx), Vmm(base_idx));
        if (axis_simd_full_ > 0) {
            const int unroll
                    = axis_simd_full_ >= unroll_factor_ ? unroll_factor_ : 1;
            assert(math::is_pow2(unroll));

            for (int i = base_idx + 1; i < base_idx + unroll; i++)
                uni_vpxor(Vmm(i), Vmm(i), Vmm(i));

            // unrolled loop
            for (int i = 0; i < axis_simd_full_ / unroll; i++)
                for (int j = base_idx; j < base_idx + unroll; j++) {
                    io_[src_d_.data_type()]->load(
                            src_ptr((i * unroll + j - base_idx) * simd_w_),
                            Vmm(j + unroll), need_tail);
                    op(Vmm(j), Vmm(j + unroll), need_tail);
                }

            // unrolled loop reduction
            int n = unroll;
            while (n > 1) {
                for (int j = base_idx; j < base_idx + n / 2; j++)
                    uni_vaddps(Vmm(j), Vmm(j), Vmm(j + n / 2));
                n = n / 2;
            }

            // unrolled loop remainder
            for (int i = utils::rnd_dn(axis_simd_full_, unroll);
                    i < axis_simd_full_; i++) {
                io_[src_d_.data_type()]->load(
                        src_ptr(i * simd_w_), Vmm(base_idx + 1), need_tail);
                op(Vmm(base_idx), Vmm(base_idx + 1), need_tail);
            }
        }

        if (axis_simd_tail_ > 0) {
            need_tail = true;
            // vector remainder
            io_[src_d_.data_type()]->load(src_ptr(axis_simd_full_ * simd_w_),
                    Vmm(base_idx + 1), need_tail);
            op(Vmm(base_idx), Vmm(base_idx + 1), need_tail);
        }

        reduce(Vmm(base_idx), Vmm(base_idx + 1));
        uni_vdivps(Vmm(base_idx), Vmm(base_idx), vmm_c, vmm_tmp);
        uni_vmovups(vmm_stat, Vmm(base_idx));
    }

    void compute_mean() {
        if (has_ne_convert_src_xf16_)
            compute_ne_convert_xf16(
                    vmm_mean, [&](Vmm vmm_dst, Vmm vmm_src, bool need_tail) {
                        uni_vaddps(vmm_dst, vmm_dst, vmm_src);
                    });
        else
            compute(vmm_mean, [&](Vmm vmm_dst, Vmm vmm_src, bool need_tail) {
                uni_vaddps(vmm_dst, vmm_dst, vmm_src);
            });
        if (save_stats_) uni_vmovss(ptr[reg_mean], Xmm(vmm_mean.getIdx()));
    }

    void compute_var() {
        if (has_ne_convert_src_xf16_)
            compute_ne_convert_xf16(vmm_inv_sqrtvar,
                    [&](Vmm vmm_dst, Vmm vmm_src, bool need_tail) {
                        uni_vsubps_maybe_tail(vmm_src, vmm_mean, need_tail);
                        uni_vfmadd231ps(vmm_dst, vmm_src, vmm_src);
                    });
        else
            compute(vmm_inv_sqrtvar,
                    [&](Vmm vmm_dst, Vmm vmm_src, bool need_tail) {
                        uni_vsubps_maybe_tail(vmm_src, vmm_mean, need_tail);
                        uni_vfmadd231ps(vmm_dst, vmm_src, vmm_src);
                    });
        if (save_stats_)
            uni_vmovss(ptr[reg_var], Xmm(vmm_inv_sqrtvar.getIdx()));
    }

    void calculate_ne_convert_xf16_dst_body(
            size_t offt_elems, bool tail = false) {
        io_[src_d_.data_type()]->load_two_simdw_xf16(
                src_ptr(offt_elems), vmm_dst_even, vmm_dst_odd);
        io_[src_d_.data_type()]->merge_interleaved_to_plain(
                vmm_dst_even, vmm_dst_odd, vmm_tmp);
        for (int j = 0; j < 2; j++) {
            const auto vmm_dst = j == 0 ? vmm_dst_even : vmm_dst_odd;
            if (use_scale_)
                io_[f32]->load(
                        scale_ptr(offt_elems + j * simd_w_), vmm_scale, tail);
            if (use_shift_)
                io_[f32]->load(
                        shift_ptr(offt_elems + j * simd_w_), vmm_shift, tail);
            uni_vsubps(vmm_dst, vmm_dst, vmm_mean);
            uni_vmulps(vmm_dst, vmm_dst, vmm_inv_sqrtvar);
            if (use_scale_ && use_shift_)
                uni_vfmadd213ps(vmm_dst, vmm_scale, vmm_shift);
            else {
                if (use_scale_) uni_vmulps(vmm_dst, vmm_dst, vmm_scale);
                if (use_shift_) uni_vaddps(vmm_dst, vmm_dst, vmm_shift);
            }
            uni_vmulps(vmm_dst, vmm_dst, vmm_combined_scales);
            io_[dst_d_.data_type()]->store(
                    vmm_dst, dst_ptr(offt_elems + j * simd_w_), tail);
        }
    }

    void calculate_dst_body(size_t offt_elems, bool tail = false) {
        if (use_scale_) {
            io_[f32]->load(scale_ptr(offt_elems), vmm_scale, tail);
        }
        if (use_shift_) {
            io_[f32]->load(shift_ptr(offt_elems), vmm_shift, tail);
        }
        io_[src_d_.data_type()]->load(src_ptr(offt_elems), vmm_dst, tail);
        uni_vsubps(vmm_dst, vmm_dst, vmm_mean);
        uni_vmulps(vmm_dst, vmm_dst, vmm_inv_sqrtvar);
        if (use_scale_ && use_shift_)
            uni_vfmadd213ps(vmm_dst, vmm_scale, vmm_shift);
        else {
            if (use_scale_) uni_vmulps(vmm_dst, vmm_dst, vmm_scale);
            if (use_shift_) uni_vaddps(vmm_dst, vmm_dst, vmm_shift);
        }
        uni_vmulps(vmm_dst, vmm_dst, vmm_combined_scales);
        io_[dst_d_.data_type()]->store(vmm_dst, dst_ptr(offt_elems), tail);
    }

    void calculate_dst() {
        if (has_ne_convert_src_xf16_) {
            for (int i = 0; i < axis_simd_full_; i += 2) {
                const bool can_load_two_simdw = axis_simd_full_ - i >= 2;
                if (can_load_two_simdw)
                    calculate_ne_convert_xf16_dst_body(i * simd_w_);
                else
                    calculate_dst_body(i * simd_w_);
            }
        } else {
            for (int i = 0; i < axis_simd_full_; i++)
                calculate_dst_body(i * simd_w_);
        }
        if (axis_simd_tail_)
            calculate_dst_body(axis_simd_full_ * simd_w_, true);
    }

    void generate() override {
        const size_t c_src_size
                = C_ * types::data_type_size(src_d_.data_type());
        const size_t c_dst_size
                = C_ * types::data_type_size(dst_d_.data_type());
        static const size_t float_size = types::data_type_size(f32);

        preamble();

        io_.init_bf16();
        if (axis_simd_tail_) io_.prepare_tail_mask();

#define PARAM_OFF(x) offsetof(ker_args_t, x)
        mov(reg_src, ptr[reg_param + PARAM_OFF(src)]);
        mov(reg_dst, ptr[reg_param + PARAM_OFF(dst)]);
        mov(reg_scale, ptr[reg_param + PARAM_OFF(scale)]);
        mov(reg_shift, ptr[reg_param + PARAM_OFF(shift)]);
        mov(reg_mean, ptr[reg_param + PARAM_OFF(mean)]);
        mov(reg_var, ptr[reg_param + PARAM_OFF(var)]);
        mov(reg_src_scales, ptr[reg_param + PARAM_OFF(src_scales)]);
        mov(reg_dst_scales, ptr[reg_param + PARAM_OFF(dst_scales)]);
        mov(reg_block_end, ptr[reg_param + PARAM_OFF(block_size)]);
        mov(reg_eps, ptr[reg_param + PARAM_OFF(eps)]);
#undef PARAM_OFF

        uni_vmovq(xmm_tmp, reg_eps);
        uni_vbroadcastss(vmm_eps, xmm_tmp);
        mov(reg_tmp, float2int(1.f));
        uni_vmovq(xmm_tmp, reg_tmp);
        uni_vbroadcastss(vmm_ones, xmm_tmp);
        mov(reg_tmp, float2int(C_));
        uni_vmovq(xmm_tmp, reg_tmp);
        uni_vbroadcastss(vmm_c, xmm_tmp);

        // add block_start to block_size to define block_end
        add(reg_block_end, reg_src);

        Label unroll_loop, end;
        L(unroll_loop);
        {
            cmp(reg_block_end, reg_src);
            jle(end, T_NEAR);

            if (calculate_stats_) {
                // compute stats
                compute_mean();
                compute_var();
            } else {
                // read mean and var from input
                uni_vmovss(xmm_tmp, dword[reg_mean]);
                uni_vbroadcastss(vmm_mean, xmm_tmp);
                uni_vmovss(xmm_tmp, dword[reg_var]);
                uni_vbroadcastss(vmm_inv_sqrtvar, xmm_tmp);
            }

            // calculate inv_sqrtvar
            uni_vaddps(vmm_inv_sqrtvar, vmm_inv_sqrtvar, vmm_eps);
            uni_vsqrtps(vmm_inv_sqrtvar, vmm_inv_sqrtvar);
            uni_vdivps(vmm_inv_sqrtvar, vmm_ones, vmm_inv_sqrtvar, vmm_tmp);

            // precompute and broadcast scales (in case of runtime)
            uni_vmovss(xmm_tmp, dword[reg_src_scales]);
            uni_vbroadcastss(vmm_combined_scales, xmm_tmp);
            uni_vmovss(xmm_tmp, dword[reg_dst_scales]);
            uni_vbroadcastss(vmm_tmp, xmm_tmp);
            uni_vmulps(vmm_combined_scales, vmm_combined_scales, vmm_tmp);
            io_.init_saturate_f32({dst_d_.data_type()});

            // calculate dst
            calculate_dst();

            add(reg_src, c_src_size);
            add(reg_dst, c_dst_size);
            add(reg_mean, float_size);
            add(reg_var, float_size);
            jmp(unroll_loop);
        }
        L(end);

        postamble();
    }
};

template <cpu_isa_t isa>
struct jit_stat_and_data_kernel_t;

template <>
struct jit_stat_and_data_kernel_t<avx512_core>
    : public jit_stat_and_data_base_kernel_t<avx512_core> {

    using jit_stat_and_data_base_kernel_t::jit_stat_and_data_base_kernel_t;

    void reduce(Vmm vmm_src, Vmm vmm_tmp) override {
        vshuff32x4(vmm_tmp, vmm_src, vmm_src, 0x4E); // 256-bit shuffle
        vaddps(vmm_src, vmm_src, vmm_tmp);
        vshuff32x4(vmm_tmp, vmm_src, vmm_src, 0xB1); // 128/256-bit shuffle
        vaddps(vmm_src, vmm_src, vmm_tmp);
        vshufps(vmm_tmp, vmm_src, vmm_src, 0x4E); // 64/128-bit shuffle
        vaddps(vmm_src, vmm_src, vmm_tmp);
        vshufps(vmm_tmp, vmm_src, vmm_src, 0xB1); // 32/64-bit shuffle
        vaddps(vmm_src, vmm_src, vmm_tmp);
    }
};

template <>
struct jit_stat_and_data_kernel_t<avx2>
    : jit_stat_and_data_base_kernel_t<avx2> {

    using jit_stat_and_data_base_kernel_t::jit_stat_and_data_base_kernel_t;

    void reduce(Vmm vmm_src, Vmm vmm_tmp) override {
        vperm2f128(vmm_tmp, vmm_src, vmm_src, 0x1); // 128/256-bit shuffle
        vaddps(vmm_src, vmm_src, vmm_tmp);
        vshufps(vmm_tmp, vmm_src, vmm_src, 0x4E); // 64/128-bit shuffle
        vaddps(vmm_src, vmm_src, vmm_tmp);
        vshufps(vmm_tmp, vmm_src, vmm_src, 0xB1); // 32/64-bit shuffle
        vaddps(vmm_src, vmm_src, vmm_tmp);
    }
};

template <>
struct jit_stat_and_data_kernel_t<sse41>
    : jit_stat_and_data_base_kernel_t<sse41> {

    using jit_stat_and_data_base_kernel_t::jit_stat_and_data_base_kernel_t;

    void reduce(Vmm vmm_src, Vmm vmm_tmp) override {
        uni_vmovups(vmm_tmp, vmm_src);
        shufps(vmm_tmp, vmm_tmp, 0x4E); // 64/128-bit shuffle
        uni_vaddps(vmm_src, vmm_src, vmm_tmp);
        uni_vmovups(vmm_tmp, vmm_src);
        shufps(vmm_tmp, vmm_tmp, 0xB1); // 32/64-bit shuffle
        uni_vaddps(vmm_src, vmm_src, vmm_tmp);
    }
};

stat_and_data_kernel_t *stat_and_data_kernel_t::create(
        const layer_normalization_pd_t *pd) {
    if (mayiuse(avx512_core)) {
        return new jit_stat_and_data_kernel_t<avx512_core>(pd);
    } else if (mayiuse(avx2)) {
        return new jit_stat_and_data_kernel_t<avx2>(pd);
    } else if (mayiuse(sse41)) {
        return new jit_stat_and_data_kernel_t<sse41>(pd);
    } else {
        assert(!"kernel is empty.");
        return nullptr;
    }
}

template <cpu_isa_t isa>
struct jit_diff_ss_kernel_t : diff_ss_kernel_t, public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_lnorm_diff_ss_kernel_t);

    void operator()(const void *src, const void *diff_dst, float *diff_scale,
            float *diff_shift, const float *mean, const float *var,
            float *const inv_sqrtvar, const size_t block_size) const override {
        ker_args_t args;
        args.src = src;
        args.diff_dst = diff_dst;
        args.diff_scale = diff_scale;
        args.diff_shift = diff_shift;
        args.mean = mean;
        for (size_t i = 0; i < block_size; i++) {
#ifdef __INTEL_COMPILER
            //Without volatile ICC with -O2 & -O3 optimizes out denominator from
            //inv_sqrtvar and computes 1/denom with lower precision
            const volatile float denom = sqrtf(var[i] + eps_);
#else
            const float denom = sqrtf(var[i] + eps_);
#endif
            inv_sqrtvar[i] = 1.f / denom;
        }
        args.inv_sqrtvar = inv_sqrtvar;
        args.block_size
                = block_size * C_ * types::data_type_size(src_d_.data_type());
        jit_generator::operator()(&args);
    }

    status_t create_kernel() override { return jit_generator::create_kernel(); }

    jit_diff_ss_kernel_t(const layer_normalization_pd_t *pd)
        : diff_ss_kernel_t(pd)
        , jit_generator(jit_name())
        , src_d_(pd_->src_md())
        , d_dst_d_(pd_->diff_dst_md())
        , simd_w_(vlen / sizeof(float))
        , C_(pd_->norm_axis())
        , axis_simd_full_(C_ / simd_w_)
        , axis_simd_tail_(C_ % simd_w_)
        , eps_(pd_->desc()->layer_norm_epsilon) {

        io::io_conf_t io_conf;
        io::io_tail_conf_t io_tail_conf(simd_w_, axis_simd_tail_,
                tail_opmask_idx, vmm_tail_mask.getIdx(), reg_tmp);
        io::io_emu_bf16_conf_t io_bf16_conf(bf16_emu_zmm_1_idx,
                bf16_emu_zmm_2_idx, bf16_emu_zmm_3_idx, reg_tmp,
                bf16_emu_zmm_4_idx);
        const auto io_isa = get_io_isa(isa,
                utils::one_of(f16, src_d_.data_type(), d_dst_d_.data_type()),
                utils::one_of(bf16, src_d_.data_type(), d_dst_d_.data_type()));
        io_ = io::jit_io_multi_dt_helper_t<Vmm>(this, io_isa,
                {src_d_.data_type(), d_dst_d_.data_type(), f32 /* stats */},
                io_conf, io_tail_conf, io_bf16_conf);
    }

protected:
    using Vmm = typename cpu_isa_traits<isa>::Vmm;
    const AddressFrame &vmmword = (isa == sse41) ? xword
            : (isa == avx2)                      ? yword
                                                 : zword;
    const int vlen = cpu_isa_traits<isa>::vlen;

    struct ker_args_t {
        const void *src;
        const void *diff_dst;
        float *diff_scale;
        float *diff_shift;
        const float *mean;
        const float *inv_sqrtvar;
        size_t block_size;
    };

    io::jit_io_multi_dt_helper_t<Vmm> io_;
    const memory_desc_wrapper src_d_, d_dst_d_;
    const size_t simd_w_;
    const dim_t C_;
    const dim_t axis_simd_full_;
    const dim_t axis_simd_tail_;
    const float eps_;

    const Reg64 reg_param = abi_param1;
    const Reg64 reg_src = rdx;
    const Reg64 reg_diff_dst = rax;
    const Reg64 reg_mean = rbx;
    const Reg64 reg_diff_scale = r8;
    const Reg64 reg_block_end = r9;
    const Reg64 reg_tmp = r11;
    const Reg64 reg_diff_shift = r12;
    const Reg64 reg_inv_sqrtvar = r13;

    const Vmm vmm_tail_mask = Vmm(0);
    const Xmm xmm_tmp = Xmm(9);
    const Vmm vmm_inv_sqrtvar = Vmm(10);
    const Vmm vmm_ddst = Vmm(11);
    const Vmm vmm_dscale = Vmm(12);
    const Vmm vmm_dshift = Vmm(13);
    const Vmm vmm_src = Vmm(14);
    const Vmm vmm_mean = Vmm(15);

    const int bf16_emu_zmm_1_idx = 28;
    const int bf16_emu_zmm_2_idx = 29;
    const int bf16_emu_zmm_3_idx = 30;
    const int bf16_emu_zmm_4_idx = 31;
    const int tail_opmask_idx = 1;

    Address src_ptr(size_t offt = 0) {
        return vmmword[reg_src + offt * src_d_.data_type_size()];
    }

    Address d_dst_ptr(size_t offt = 0) {
        return vmmword[reg_diff_dst + offt * d_dst_d_.data_type_size()];
    }

    Address d_scale_ptr(size_t offt = 0) {
        return vmmword[reg_diff_scale + offt * sizeof(float)];
    }

    Address d_shift_ptr(size_t offt = 0) {
        return vmmword[reg_diff_shift + offt * sizeof(float)];
    }

    void calculate_diff_scale_shift(size_t offt_elems, bool tail = false) {
        io_[d_dst_d_.data_type()]->load(d_dst_ptr(offt_elems), vmm_ddst, tail);
        io_[f32]->load(d_scale_ptr(offt_elems), vmm_dscale, tail);
        io_[f32]->load(d_shift_ptr(offt_elems), vmm_dshift, tail);
        io_[src_d_.data_type()]->load(src_ptr(offt_elems), vmm_src, tail);

        uni_vaddps(vmm_dshift, vmm_dshift, vmm_ddst);
        uni_vsubps(vmm_src, vmm_src, vmm_mean);
        uni_vmulps(vmm_src, vmm_src, vmm_inv_sqrtvar);
        uni_vfmadd231ps(vmm_dscale, vmm_src, vmm_ddst);

        io_[f32]->store(vmm_dscale, d_scale_ptr(offt_elems), tail);
        io_[f32]->store(vmm_dshift, d_shift_ptr(offt_elems), tail);
    };

    void generate() override {
        const size_t c_src_size
                = C_ * types::data_type_size(src_d_.data_type());
        const size_t c_ddst_size
                = C_ * types::data_type_size(d_dst_d_.data_type());
        static const size_t float_size = types::data_type_size(f32);

        preamble();

        io_.init_bf16();
        if (axis_simd_tail_) io_.prepare_tail_mask();

#define PARAM_OFF(x) offsetof(ker_args_t, x)
        mov(reg_src, ptr[reg_param + PARAM_OFF(src)]);
        mov(reg_diff_dst, ptr[reg_param + PARAM_OFF(diff_dst)]);
        mov(reg_diff_scale, ptr[reg_param + PARAM_OFF(diff_scale)]);
        mov(reg_diff_shift, ptr[reg_param + PARAM_OFF(diff_shift)]);
        mov(reg_mean, ptr[reg_param + PARAM_OFF(mean)]);
        mov(reg_inv_sqrtvar, ptr[reg_param + PARAM_OFF(inv_sqrtvar)]);
        mov(reg_block_end, ptr[reg_param + PARAM_OFF(block_size)]);
#undef PARAM_OFF

        // add block_start to block_size to define block_end
        add(reg_block_end, reg_src);

        Label unroll_loop, end;
        L(unroll_loop);
        {
            cmp(reg_block_end, reg_src);
            jle(end, T_NEAR);

            uni_vmovss(xmm_tmp, dword[reg_mean]);
            uni_vbroadcastss(vmm_mean, xmm_tmp);
            uni_vmovss(xmm_tmp, dword[reg_inv_sqrtvar]);
            uni_vbroadcastss(vmm_inv_sqrtvar, xmm_tmp);

            for (int i = 0; i < axis_simd_full_; i++)
                calculate_diff_scale_shift(i * simd_w_);
            if (axis_simd_tail_)
                calculate_diff_scale_shift(axis_simd_full_ * simd_w_, true);

            add(reg_src, c_src_size);
            add(reg_diff_dst, c_ddst_size);
            add(reg_mean, float_size);
            add(reg_inv_sqrtvar, float_size);
            jmp(unroll_loop);
        }
        L(end);

        postamble();
    }
};

diff_ss_kernel_t *diff_ss_kernel_t::create(const layer_normalization_pd_t *pd) {
    if (mayiuse(avx512_core)) {
        return new jit_diff_ss_kernel_t<avx512_core>(pd);
    } else if (mayiuse(avx2)) {
        return new jit_diff_ss_kernel_t<avx2>(pd);
    } else {
        assert(!"kernel is empty.");
        return nullptr;
    }
}

template <cpu_isa_t isa>
struct jit_diff_data_base_kernel_t : diff_data_kernel_t, public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_lnorm_diff_data_kernel_t);

    void operator()(const void *src, const void *diff_dst, void *diff_src,
            const float *ss, const float *mean, float *const inv_sqrtvar,
            const size_t block_size) const override {
        ker_args_t args;
        args.src = src;
        args.diff_dst = diff_dst;
        args.diff_src = diff_src;
        args.ss = ss;
        args.mean = mean;
        args.inv_sqrtvar = inv_sqrtvar;
        args.block_size
                = block_size * C_ * types::data_type_size(src_d_.data_type());
        jit_generator::operator()(&args);
    }

    status_t create_kernel() override { return jit_generator::create_kernel(); }

    jit_diff_data_base_kernel_t(const layer_normalization_pd_t *pd)
        : diff_data_kernel_t(pd)
        , jit_generator(jit_name())
        , src_d_(pd_->src_md())
        , d_dst_d_(pd_->diff_dst_md())
        , d_src_d_(pd_->diff_src_md())
        , simd_w_(vlen / sizeof(float))
        , C_(pd_->norm_axis())
        , axis_simd_full_(C_ / simd_w_)
        , axis_simd_tail_(C_ % simd_w_)
        , use_scale_(pd_->use_scale())
        , use_shift_(pd_->use_shift())
        , calculate_diff_stats_(!pd_->stats_are_src()) {

        io::io_conf_t io_conf;
        io::io_tail_conf_t io_tail_conf(simd_w_, axis_simd_tail_,
                tail_opmask_idx, vmm_tail_mask.getIdx(), reg_tmp);
        io::io_emu_bf16_conf_t io_bf16_conf(bf16_emu_zmm_1_idx,
                bf16_emu_zmm_2_idx, bf16_emu_zmm_3_idx, reg_tmp,
                bf16_emu_zmm_4_idx);
        const auto io_isa = get_io_isa(isa,
                utils::one_of(f16, src_d_.data_type(), d_dst_d_.data_type(),
                        d_src_d_.data_type()),
                utils::one_of(bf16, src_d_.data_type(), d_dst_d_.data_type(),
                        d_src_d_.data_type()));
        io_ = io::jit_io_multi_dt_helper_t<Vmm>(this, io_isa,
                {src_d_.data_type(), d_dst_d_.data_type(), d_src_d_.data_type(),
                        f32 /* stats */},
                io_conf, io_tail_conf, io_bf16_conf);
    }

protected:
    static constexpr int unroll_factor_ = 4;
    using Vmm = typename cpu_isa_traits<isa>::Vmm;
    const AddressFrame &vmmword = (isa == sse41) ? xword
            : (isa == avx2)                      ? yword
                                                 : zword;
    const int vlen = cpu_isa_traits<isa>::vlen;

    struct ker_args_t {
        const void *src;
        const void *diff_dst;
        void *diff_src;
        const float *ss;
        const float *mean;
        const float *inv_sqrtvar;
        size_t block_size;
    };

    io::jit_io_multi_dt_helper_t<Vmm> io_;
    const memory_desc_wrapper src_d_, d_dst_d_, d_src_d_;
    const size_t simd_w_;
    const dim_t C_;
    const dim_t axis_simd_full_;
    const dim_t axis_simd_tail_;
    const bool use_scale_;
    const bool use_shift_;
    const bool calculate_diff_stats_;

    const Reg64 reg_param = abi_param1;
    const Reg64 reg_src = rdx;
    const Reg64 reg_diff_dst = rax;
    const Reg64 reg_diff_src = r14;
    const Reg64 reg_mean = rbx;
    const Reg64 reg_inv_sqrtvar = r13;
    const Reg64 reg_scale = r8;
    const Reg64 reg_tmp = r11;
    const Reg64 reg_dd_scale = r10;
    const Reg64 reg_dd_scale_x = r12;
    const Reg64 reg_block_end = r9;

    const Vmm vmm_tail_mask = Vmm(0);
    const Vmm vmm_C = Vmm(7);
    const Vmm vmm_scale = Vmm(8);
    const Xmm xmm_tmp = Xmm(9);
    const Vmm vmm_tmp = Vmm(9);
    const Vmm vmm_inv_sqrtvar = Vmm(10);
    const Vmm vmm_dsrc = Vmm(11);
    const Vmm vmm_dd_scale_x = Vmm(12);
    const Vmm vmm_dd_scale = Vmm(13);
    const Vmm vmm_src = Vmm(14);
    const Vmm vmm_mean = Vmm(15);

    const int bf16_emu_zmm_1_idx = 28;
    const int bf16_emu_zmm_2_idx = 29;
    const int bf16_emu_zmm_3_idx = 30;
    const int bf16_emu_zmm_4_idx = 31;
    const int tail_opmask_idx = 1;

    Address src_ptr(size_t offt = 0) {
        return vmmword[reg_src + offt * src_d_.data_type_size()];
    }

    Address d_dst_ptr(size_t offt = 0) {
        return vmmword[reg_diff_dst + offt * d_dst_d_.data_type_size()];
    }

    Address d_src_ptr(size_t offt = 0) {
        return vmmword[reg_diff_src + offt * d_src_d_.data_type_size()];
    }

    Address scale_ptr(size_t offt = 0) {
        return vmmword[reg_scale + offt * sizeof(float)];
    }

    virtual void reduce(Vmm vmm_src, Vmm vmm_tmp) = 0;

    void compute_dd_scales(size_t offt_elems, bool tail = false) {
        Vmm vmm_ddst = vmm_dsrc;
        io_[d_dst_d_.data_type()]->load(d_dst_ptr(offt_elems), vmm_ddst, tail);
        if (use_scale_) {
            io_[f32]->load(scale_ptr(offt_elems), vmm_scale, tail);
            uni_vmulps(vmm_ddst, vmm_ddst, vmm_scale);
        }
        io_[src_d_.data_type()]->load(src_ptr(offt_elems), vmm_src, tail);

        uni_vaddps(vmm_dd_scale, vmm_dd_scale, vmm_ddst);
        uni_vsubps(vmm_src, vmm_src, vmm_mean);
        uni_vfmadd231ps(vmm_dd_scale_x, vmm_ddst, vmm_src);
    };

    void compute_diff_src(size_t offt_elems, bool tail = false) {
        Vmm vmm_ddst = vmm_dsrc;
        io_[d_dst_d_.data_type()]->load(d_dst_ptr(offt_elems), vmm_ddst, tail);
        if (use_scale_) {
            io_[f32]->load(scale_ptr(offt_elems), vmm_scale, tail);
            uni_vmulps(vmm_dsrc, vmm_dsrc, vmm_scale);
        }
        if (calculate_diff_stats_) {
            io_[src_d_.data_type()]->load(src_ptr(offt_elems), vmm_src, tail);
            uni_vsubps(vmm_src, vmm_src, vmm_mean);
            uni_vmulps(vmm_src, vmm_src, vmm_inv_sqrtvar);
            uni_vfmadd213ps(vmm_src, vmm_dd_scale_x, vmm_dd_scale);
            uni_vdivps(vmm_src, vmm_src, vmm_C);
            uni_vsubps(vmm_dsrc, vmm_dsrc, vmm_src);
        }
        uni_vmulps(vmm_dsrc, vmm_dsrc, vmm_inv_sqrtvar);
        io_[d_src_d_.data_type()]->store(vmm_dsrc, d_src_ptr(offt_elems), tail);
    };

    void generate() override {
        const size_t c_src_size
                = C_ * types::data_type_size(src_d_.data_type());
        const size_t c_ddst_size
                = C_ * types::data_type_size(d_dst_d_.data_type());
        const size_t c_dsrc_size
                = C_ * types::data_type_size(d_src_d_.data_type());
        static const size_t float_size = types::data_type_size(f32);

        preamble();

        io_.init_bf16();
        if (axis_simd_tail_) io_.prepare_tail_mask();

#define PARAM_OFF(x) offsetof(ker_args_t, x)
        mov(reg_src, ptr[reg_param + PARAM_OFF(src)]);
        mov(reg_diff_dst, ptr[reg_param + PARAM_OFF(diff_dst)]);
        mov(reg_diff_src, ptr[reg_param + PARAM_OFF(diff_src)]);
        mov(reg_scale, ptr[reg_param + PARAM_OFF(ss)]);

        if (calculate_diff_stats_)
            mov(reg_mean, ptr[reg_param + PARAM_OFF(mean)]);
        mov(reg_inv_sqrtvar, ptr[reg_param + PARAM_OFF(inv_sqrtvar)]);
        mov(reg_block_end, ptr[reg_param + PARAM_OFF(block_size)]);
#undef PARAM_OFF

        mov(reg_tmp, float2int(C_));
        uni_vmovq(xmm_tmp, reg_tmp);
        uni_vbroadcastss(vmm_C, xmm_tmp);

        // add block_start to block_size to define block_end
        add(reg_block_end, reg_src);

        Label unroll_loop, end;
        L(unroll_loop);
        {
            cmp(reg_block_end, reg_src);
            jle(end, T_NEAR);

            uni_vmovss(xmm_tmp, dword[reg_inv_sqrtvar]);
            uni_vbroadcastss(vmm_inv_sqrtvar, xmm_tmp);

            if (calculate_diff_stats_) {
                uni_vmovss(xmm_tmp, dword[reg_mean]);
                uni_vbroadcastss(vmm_mean, xmm_tmp);

                uni_vpxor(vmm_dd_scale, vmm_dd_scale, vmm_dd_scale);
                uni_vpxor(vmm_dd_scale_x, vmm_dd_scale_x, vmm_dd_scale_x);

                for (int i = 0; i < axis_simd_full_; i++)
                    compute_dd_scales(i * simd_w_);
                if (axis_simd_tail_)
                    compute_dd_scales(axis_simd_full_ * simd_w_, true);

                reduce(vmm_dd_scale, vmm_tmp);
                reduce(vmm_dd_scale_x, vmm_tmp);
                uni_vmulps(vmm_dd_scale_x, vmm_dd_scale_x, vmm_inv_sqrtvar);
            }

            for (int i = 0; i < axis_simd_full_; i++)
                compute_diff_src(i * simd_w_);
            if (axis_simd_tail_)
                compute_diff_src(axis_simd_full_ * simd_w_, true);

            add(reg_src, c_src_size);
            add(reg_diff_dst, c_ddst_size);
            add(reg_diff_src, c_dsrc_size);
            if (calculate_diff_stats_) add(reg_mean, float_size);
            add(reg_inv_sqrtvar, float_size);
            jmp(unroll_loop);
        }
        L(end);

        postamble();
    }
};

template <cpu_isa_t isa>
struct jit_diff_data_kernel_t;

template <>
struct jit_diff_data_kernel_t<avx512_core>
    : public jit_diff_data_base_kernel_t<avx512_core> {

    using jit_diff_data_base_kernel_t::jit_diff_data_base_kernel_t;

    void reduce(Vmm vmm_src, Vmm vmm_tmp) override {
        vshuff32x4(vmm_tmp, vmm_src, vmm_src, 0x4E); // 256-bit shuffle
        vaddps(vmm_src, vmm_src, vmm_tmp);
        vshuff32x4(vmm_tmp, vmm_src, vmm_src, 0xB1); // 128/256-bit shuffle
        vaddps(vmm_src, vmm_src, vmm_tmp);
        vshufps(vmm_tmp, vmm_src, vmm_src, 0x4E); // 64/128-bit shuffle
        vaddps(vmm_src, vmm_src, vmm_tmp);
        vshufps(vmm_tmp, vmm_src, vmm_src, 0xB1); // 32/64-bit shuffle
        vaddps(vmm_src, vmm_src, vmm_tmp);
    }
};

template <>
struct jit_diff_data_kernel_t<avx2> : public jit_diff_data_base_kernel_t<avx2> {

    using jit_diff_data_base_kernel_t::jit_diff_data_base_kernel_t;

    void reduce(Vmm vmm_src, Vmm vmm_tmp) override {
        vperm2f128(vmm_tmp, vmm_src, vmm_src, 0x1); // 128/256-bit shuffle
        vaddps(vmm_src, vmm_src, vmm_tmp);
        vshufps(vmm_tmp, vmm_src, vmm_src, 0x4E); // 64/128-bit shuffle
        vaddps(vmm_src, vmm_src, vmm_tmp);
        vshufps(vmm_tmp, vmm_src, vmm_src, 0xB1); // 32/64-bit shuffle
        vaddps(vmm_src, vmm_src, vmm_tmp);
    }
};

diff_data_kernel_t *diff_data_kernel_t::create(
        const layer_normalization_pd_t *pd) {
    if (mayiuse(avx512_core)) {
        return new jit_diff_data_kernel_t<avx512_core>(pd);
    } else if (mayiuse(avx2)) {
        return new jit_diff_data_kernel_t<avx2>(pd);
    } else {
        assert(!"kernel is empty.");
        return nullptr;
    }
}

status_t jit_uni_layer_normalization_fwd_t::execute_forward(
        const exec_ctx_t &ctx) const {
    auto scratchpad = ctx.get_scratchpad_grantor();
    const auto src = CTX_IN_MEM(const void *, DNNL_ARG_SRC);
    auto dst = CTX_OUT_MEM(void *, DNNL_ARG_DST);

    auto scale = CTX_IN_MEM(const float *, DNNL_ARG_SCALE);
    auto shift = CTX_IN_MEM(const float *, DNNL_ARG_SHIFT);

    float *mean, *variance;
    if (pd()->use_tmp_stats()) {
        mean = scratchpad.template get<float>(key_lnorm_tmp_mean);
        variance = scratchpad.template get<float>(key_lnorm_tmp_var);
    } else {
        mean = pd()->stats_are_src()
                ? const_cast<float *>(CTX_IN_MEM(const float *, DNNL_ARG_MEAN))
                : CTX_OUT_MEM(float *, DNNL_ARG_MEAN);
        variance = pd()->stats_are_src()
                ? const_cast<float *>(
                        CTX_IN_MEM(const float *, DNNL_ARG_VARIANCE))
                : CTX_OUT_MEM(float *, DNNL_ARG_VARIANCE);
    }

    DEFINE_ARG_SCALES_BUFFER(src_scales, DNNL_ARG_SRC);
    DEFINE_ARG_SCALES_BUFFER(dst_scales, DNNL_ARG_DST);

    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper dst_d(pd()->dst_md());

    const dim_t N = pd()->across_axis();
    const dim_t C_padded = src_d.padded_dims()[pd()->ndims() - 1];

    parallel(0, [&](const int ithr, const int nthr) {
        dim_t N_start = 0, N_end = 0;
        balance211(N, nthr, ithr, N_start, N_end);
        const char *const __restrict src_ptr
                = reinterpret_cast<const char *>(src)
                + N_start * C_padded * src_d.data_type_size();
        char *const __restrict dst_ptr = reinterpret_cast<char *>(dst)
                + N_start * C_padded * dst_d.data_type_size();
        const int block_size = N_end - N_start;
        (*stat_and_data_kernel_)(src_ptr, dst_ptr, scale, shift, &mean[N_start],
                &variance[N_start], src_scales, dst_scales, block_size);
    });
    return status::success;
}

status_t jit_uni_layer_normalization_bwd_t::execute_backward(
        const exec_ctx_t &ctx) const {
    status_t status = status::success;

    auto scratchpad = ctx.get_scratchpad_grantor();
    auto src = CTX_IN_MEM(const void *, DNNL_ARG_SRC);
    auto diff_dst = CTX_IN_MEM(const void *, DNNL_ARG_DIFF_DST);
    auto scale = CTX_IN_MEM(float *, DNNL_ARG_SCALE);
    auto diff_src = CTX_OUT_CLEAN_MEM(void *, DNNL_ARG_DIFF_SRC, status);

    auto diff_scale = CTX_OUT_CLEAN_MEM(float *, DNNL_ARG_DIFF_SCALE, status);
    CHECK(status);
    auto diff_shift = CTX_OUT_CLEAN_MEM(float *, DNNL_ARG_DIFF_SHIFT, status);
    CHECK(status);

    const float *mean, *variance;
    if (pd()->use_tmp_stats()) {
        mean = scratchpad.template get<float>(key_lnorm_tmp_mean);
        variance = scratchpad.template get<float>(key_lnorm_tmp_var);
    } else {
        mean = CTX_IN_MEM(const float *, DNNL_ARG_MEAN);
        variance = CTX_IN_MEM(const float *, DNNL_ARG_VARIANCE);
    }

    float *const inv_sqrtvar
            = scratchpad.template get<float>(key_lnorm_inv_sqrtvar);

    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper diff_dst_d(pd()->diff_dst_md());
    const memory_desc_wrapper diff_src_d(pd()->diff_src_md());

    const dim_t N = pd()->across_axis();
    const dim_t C = pd()->norm_axis();
    const dim_t C_padded = src_d.padded_dims()[pd()->ndims() - 1];

    float *reduce = scratchpad.template get<float>(key_lnorm_reduction);
    if (diff_scale == nullptr)
        diff_scale = scratchpad.template get<float>(key_lnorm_tmp_diff_ss);
    if (diff_shift == nullptr) {
        diff_shift = scratchpad.template get<float>(key_lnorm_tmp_diff_ss);
    }

    const int max_nthr = pd()->nthr_;

    parallel(max_nthr, [&](int ithr, int nthr) {
        dim_t N_start = 0, N_end = 0;
        balance211(N, nthr, ithr, N_start, N_end);
        const int block_size = N_end - N_start;
        const char *const __restrict src_ptr
                = reinterpret_cast<const char *>(src)
                + N_start * C_padded * src_d.data_type_size();
        const char *const __restrict diff_dst_ptr
                = reinterpret_cast<const char *>(diff_dst)
                + N_start * C_padded * diff_dst_d.data_type_size();

        float *my_diff_gamma = reduce + C * ithr;
        float *my_diff_beta = reduce + C * nthr + C * ithr;
        for (dim_t c = 0; c < C; c++) {
            my_diff_gamma[c] = 0.;
            my_diff_beta[c] = 0.;
        }
        (*diff_ss_kernel_)(src_ptr, diff_dst_ptr, my_diff_gamma, my_diff_beta,
                &mean[N_start], &variance[N_start], &inv_sqrtvar[N_start],
                block_size);
    });

    parallel_nd(C, [&](dim_t c) {
        float diff_gamma = 0, diff_beta = 0;
        for (dim_t n = 0; n < max_nthr; n++) {
            diff_gamma += reduce[C * n + c];
            diff_beta += reduce[C * max_nthr + C * n + c];
        }
        diff_scale[c] = diff_gamma;
        diff_shift[c] = diff_beta;
    });

    parallel(max_nthr, [&](int ithr, int nthr) {
        dim_t N_start = 0, N_end = 0;
        balance211(N, nthr, ithr, N_start, N_end);
        const int block_size = N_end - N_start;
        const char *const __restrict src_ptr
                = reinterpret_cast<const char *>(src)
                + N_start * C_padded * src_d.data_type_size();
        const char *const __restrict diff_dst_ptr
                = reinterpret_cast<const char *>(diff_dst)
                + N_start * C_padded * diff_dst_d.data_type_size();
        char *const __restrict diff_src_ptr = reinterpret_cast<char *>(diff_src)
                + N_start * C_padded * diff_src_d.data_type_size();

        (*diff_data_kernel_)(src_ptr, diff_dst_ptr, diff_src_ptr, scale,
                &mean[N_start], &inv_sqrtvar[N_start], block_size);
    });
    return status::success;
}

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
