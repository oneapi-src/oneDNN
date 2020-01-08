/*******************************************************************************
* Copyright 2019 Intel Corporation
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

#include "c_types_map.hpp"
#include "dnnl_thread.hpp"
#include "math_utils.hpp"
#include "nstl.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

#include "jit_generator.hpp"

#include "jit_uni_i8i8_binary.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

using namespace Xbyak;

struct i8i8_binary_kernel_t {
    struct call_params_t {
        const float *scales_src0, *scales_src1;
        const char *src0;
        const char *src1;
        const char *dst;
        size_t spat_offt_count;
    };

    i8i8_binary_kernel_t(int vlen) : vlen_(vlen) {}
    virtual ~i8i8_binary_kernel_t() = default;

    void operator()(const call_params_t *p) {
        assert(ker_);
        ker_(p);
    }
    int vlen() const { return vlen_; }

protected:
    int vlen_ = 0;
    void (*ker_)(const call_params_t *) = nullptr;
};

template <cpu_isa_t isa>
struct jit_uni_i8i8_binary_kernel_t : public i8i8_binary_kernel_t,
                                      public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_i8i8_binary_kernel_t)

    using Vmm = typename cpu_isa_traits<isa>::Vmm;
    const AddressFrame &vmmword = (isa == avx2) ? yword : zword;

    const binary_pd_t *pd_;

    Reg64 reg_param = abi_param1;

    Reg64 reg_scales_src0 = rbx;
    Reg64 reg_scales_src1 = rbp;

    Reg64 reg_src0 = r8;
    Reg64 reg_src1 = r9;
    Reg64 reg_dst = r10;
    Reg64 reg_spat_offt = r11;
    Reg64 reg_spat_offt_count = r12;
    Reg64 reg_reverse_spat_offt = r13;
    Reg64 reg_tmp = r14;

    size_t unroll_regs_ = isa == avx512_common ? 8 : 4;
    size_t simd_w_ = vlen() / sizeof(float);
    size_t tail_size_ = 0;
    bool do_scale_src0_ = false;
    bool do_scale_src1_ = false;

    Vmm vreg_scales_src0 = Vmm(isa == avx512_common ? 17 : 9);
    Vmm vreg_scales_src1 = Vmm(isa == avx512_common ? 18 : 10);
    Vmm vreg_one = Vmm(isa == avx512_common ? 19 : 11);
    Vmm vreg_zero = Vmm(isa == avx512_common ? 20 : 12);

    Xmm xreg_tmp = Xmm(0);
    Xmm xreg_one = Xmm(1);

    enum { nargs = 2 };
    // 0:src0 1:src1
    scales_t scales[nargs];

    void init() {
        const memory_desc_wrapper src0_d(pd_->src_md(0));
        size_t nelems = src0_d.nelems(true);
        tail_size_ = nelems % simd_w_;

        scales[0] = pd_->attr()->scales_.get(DNNL_ARG_SRC_0);
        scales[1] = pd_->attr()->scales_.get(DNNL_ARG_SRC_1);

        do_scale_src0_ = !scales[0].has_default_values();
        do_scale_src1_ = !scales[1].has_default_values();
    }

    void load_kernel_params() {
#define PARAM_OFF(x) offsetof(call_params_t, x)
        mov(reg_spat_offt_count, ptr[reg_param + PARAM_OFF(spat_offt_count)]);
        mov(reg_src0, ptr[reg_param + PARAM_OFF(src0)]);
        mov(reg_src1, ptr[reg_param + PARAM_OFF(src1)]);
        mov(reg_dst, ptr[reg_param + PARAM_OFF(dst)]);
        if (do_scale_src0_)
            mov(reg_scales_src0, ptr[reg_param + PARAM_OFF(scales_src0)]);

        if (do_scale_src1_)
            mov(reg_scales_src1, ptr[reg_param + PARAM_OFF(scales_src1)]);
#undef PARAM_OFF
    }

    Address src0_ptr(size_t offt = 0) {
        return vmmword[reg_src0 + reg_spat_offt + offt];
    }

    Address src1_ptr(size_t offt = 0) {
        return vmmword[reg_src1 + reg_spat_offt + offt];
    }

    Address dst_ptr(size_t offt = 0) {
        return vmmword[reg_dst + reg_spat_offt + offt];
    }

    void perform_op(const Vmm &v0, const Vmm &v1, const Vmm &s_src0,
            const Vmm &s_src1) {
        using namespace alg_kind;
        const auto alg = pd_->desc()->alg_kind;
        if (do_scale_src0_) uni_vmulps(v0, v0, s_src0);
        if (do_scale_src1_) uni_vmulps(v1, v1, s_src1);

        if (alg == alg_kind::binary_add) {
            uni_vaddps(v0, v0, v1);
        } else if (alg == alg_kind::binary_mul) {
            uni_vmulps(v0, v0, v1);
        }
    }

    void load_and_convert(const Vmm &v, const Address &m, data_type_t idt) {
        switch (idt) {
            case data_type::u8: vpmovzxbd(v, m); break;
            case data_type::s8: vpmovsxbd(v, m); break;
            default: assert(!"unreachable");
        }
    }

    void load_and_convert_tail(
            const Vmm &v, data_type_t idt, int src_num, size_t nelems) {
        for (size_t i = 0; i < nelems; i++) {
            const auto src_ptr = (src_num == 0) ? src0_ptr(i) : src1_ptr(i);
            vpinsrb(xreg_tmp, xreg_tmp, src_ptr, i);
        }

        switch (idt) {
            case data_type::u8: vpmovzxbd(v, xreg_tmp); break;
            case data_type::s8: vpmovsxbd(v, xreg_tmp); break;
            default: assert(!"unreachable");
        }
    }

    void load_and_prepare_scales(const Vmm &vs_src0, const Vmm &vs_src1) {
        // Only mask 0 is supported at this point
        if (do_scale_src0_) vbroadcastss(vs_src0, dword[reg_scales_src0]);

        if (do_scale_src1_) vbroadcastss(vs_src1, dword[reg_scales_src1]);
    }

    void store_tail(const Xmm &x, size_t nelems) {
        for (size_t i = 0; i < nelems; i++)
            vpextrb(dst_ptr(i), x, i);
    }

    virtual void compute_dst(int unroll, bool tail = false) = 0;

    void forward() {
        uni_vpxor(vreg_zero, vreg_zero, vreg_zero);
        load_and_prepare_scales(vreg_scales_src0, vreg_scales_src1);

        Label unroll_loop, unroll_loop_tail, nelems_tail, end;

        // reverse spat_offt to dispatch between labels
        mov(reg_reverse_spat_offt, reg_spat_offt_count);
        xor_(reg_spat_offt, reg_spat_offt); // spat_offt to get addr of src/dst
        L(unroll_loop);
        {
            cmp(reg_reverse_spat_offt, unroll_regs_ * simd_w_);
            jl(unroll_loop_tail, T_NEAR);
            compute_dst(unroll_regs_);
            sub(reg_reverse_spat_offt, unroll_regs_ * simd_w_);
            add(reg_spat_offt, unroll_regs_ * simd_w_);
            jmp(unroll_loop);
        }

        L(unroll_loop_tail);
        {
            cmp(reg_reverse_spat_offt, simd_w_);
            jl(nelems_tail, T_NEAR);

            compute_dst(1);
            sub(reg_reverse_spat_offt, simd_w_);
            add(reg_spat_offt, simd_w_);
            jmp(unroll_loop_tail);
        }

        L(nelems_tail);
        {
            cmp(reg_reverse_spat_offt, 1);
            jl(end, T_NEAR);

            compute_dst(1, true);
        }

        L(end);
    }

    void generate() {
        preamble();
        load_kernel_params();
        forward();
        postamble();

        ker_ = getCode<decltype(ker_)>();
    }

    jit_uni_i8i8_binary_kernel_t(const binary_pd_t *pd)
        : i8i8_binary_kernel_t(cpu_isa_traits<isa>::vlen), pd_(pd) {
        init();
    }
    virtual ~jit_uni_i8i8_binary_kernel_t() = default;
};

template <cpu_isa_t isa, data_type_t src0_type, data_type_t src1_type>
struct jit_i8i8_binary_subkernel_t;

template <data_type_t src0_type, data_type_t src1_type>
struct jit_i8i8_binary_subkernel_t<avx512_common, src0_type, src1_type>
    : public jit_uni_i8i8_binary_kernel_t<avx512_common> {

    void cvt2odt(const Operand &dst, const Vmm &src, data_type_t odt) {
        switch (odt) {
            case data_type::s8: vpmovsdb(dst, src); break;
            case data_type::u8:
                vpmaxsd(src, src, vreg_zero);
                vpmovusdb(dst, src);
                break;
            default: assert(!"unreachable");
        }
    }

    void compute_dst(int unroll, bool tail = false) override {
        for (int i = 0; i < unroll; i++) {
            Vmm vreg_tmp_src0 = Vmm(2 * i + 1);
            Vmm vreg_tmp_src1 = Vmm(2 * i + 2);

            if (!tail) {
                load_and_convert(
                        vreg_tmp_src0, src0_ptr(simd_w_ * i), src0_type);
                load_and_convert(
                        vreg_tmp_src1, src1_ptr(simd_w_ * i), src1_type);
                // s32 -> f32
                vcvtdq2ps(vreg_tmp_src0, vreg_tmp_src0);
                vcvtdq2ps(vreg_tmp_src1, vreg_tmp_src1);
                perform_op(vreg_tmp_src0, vreg_tmp_src1, vreg_scales_src0,
                        vreg_scales_src1);
                // f32 -> s32
                vcvtps2dq(vreg_tmp_src0, vreg_tmp_src0);
                // s32 -> s8 and store
                cvt2odt(dst_ptr(simd_w_ * i), vreg_tmp_src0, src0_type);
            } else {
                load_and_convert_tail(vreg_tmp_src0, src0_type, 0, tail_size_);
                load_and_convert_tail(vreg_tmp_src1, src1_type, 1, tail_size_);
                // s32 -> f32
                vcvtdq2ps(vreg_tmp_src0, vreg_tmp_src0);
                vcvtdq2ps(vreg_tmp_src1, vreg_tmp_src1);
                perform_op(vreg_tmp_src0, vreg_tmp_src1, vreg_scales_src0,
                        vreg_scales_src1);
                // f32 -> s32
                vcvtps2dq(vreg_tmp_src0, vreg_tmp_src0);
                // s32 -> i8
                cvt2odt(xreg_tmp, vreg_tmp_src0, src0_type);
                store_tail(xreg_tmp, tail_size_);
            }
        }
    }

    jit_i8i8_binary_subkernel_t(const binary_pd_t *pd)
        : jit_uni_i8i8_binary_kernel_t(pd) {
        generate();
    }
};

template <data_type_t src0_type, data_type_t src1_type>
struct jit_i8i8_binary_subkernel_t<avx2, src0_type, src1_type>
    : public jit_uni_i8i8_binary_kernel_t<avx2> {

    void cvt2odt(const Vmm &v, data_type_t odt) {
        // v = { 8x32 }
        vpackssdw(v, v, vreg_zero);
        // v = { 4x16, 0, 4x16, 0 }
        vpermq(v, v, 0x58);
        // v =  { 8x16, 0 }

        switch (odt) {
            case data_type::u8: vpackuswb(v, v, vreg_zero); break;
            case data_type::s8: vpacksswb(v, v, vreg_zero); break;
            default: assert(!"unreachable");
        }
        // v = { 8x8, 0 }
    }

    void compute_dst(int unroll, bool tail = false) override {
        for (int i = 0; i < unroll; i++) {
            Vmm vreg_tmp_src0 = Vmm(2 * i + 1);
            Vmm vreg_tmp_src1 = Vmm(2 * i + 2);

            if (!tail) {
                load_and_convert(
                        vreg_tmp_src0, src0_ptr(simd_w_ * i), src0_type);
                load_and_convert(
                        vreg_tmp_src1, src1_ptr(simd_w_ * i), src1_type);
                // s32 -> f32
                vcvtdq2ps(vreg_tmp_src0, vreg_tmp_src0);
                vcvtdq2ps(vreg_tmp_src1, vreg_tmp_src1);
                perform_op(vreg_tmp_src0, vreg_tmp_src1, vreg_scales_src0,
                        vreg_scales_src1);
                // f32 -> s32
                vcvtps2dq(vreg_tmp_src0, vreg_tmp_src0);
                // s32 -> i8
                cvt2odt(vreg_tmp_src0, src0_type);
                // store 64 bits
                vmovq(dst_ptr(simd_w_ * i), Xmm(vreg_tmp_src0.getIdx()));
            } else {
                load_and_convert_tail(vreg_tmp_src0, src0_type, 0, tail_size_);
                load_and_convert_tail(vreg_tmp_src1, src1_type, 1, tail_size_);
                // s32 -> f32
                vcvtdq2ps(vreg_tmp_src0, vreg_tmp_src0);
                vcvtdq2ps(vreg_tmp_src1, vreg_tmp_src1);
                perform_op(vreg_tmp_src0, vreg_tmp_src1, vreg_scales_src0,
                        vreg_scales_src1);
                // f32 -> s32
                vcvtps2dq(vreg_tmp_src0, vreg_tmp_src0);
                // s32 -> i8
                cvt2odt(vreg_tmp_src0, src0_type);
                store_tail(Xmm(vreg_tmp_src0.getIdx()), tail_size_);
            }
        }
    }

    jit_i8i8_binary_subkernel_t(const binary_pd_t *pd)
        : jit_uni_i8i8_binary_kernel_t(pd) {
        generate();
    }
};

template <data_type_t src0_type, data_type_t src1_type>
std::unique_ptr<i8i8_binary_kernel_t> create_i8i8_binary_kernel(
        const binary_pd_t *pd) {
    if (mayiuse(avx512_common)) {
        using subkernel_t = jit_i8i8_binary_subkernel_t<avx512_common,
                src0_type, src1_type>;
        return std::unique_ptr<subkernel_t> {new subkernel_t(pd)};
    } else if (mayiuse(avx2)) {
        using subkernel_t
                = jit_i8i8_binary_subkernel_t<avx2, src0_type, src1_type>;
        return std::unique_ptr<subkernel_t> {new subkernel_t(pd)};
    }
    return nullptr;
}

template <data_type_t src0_type, data_type_t src1_type>
jit_uni_i8i8_binary_t<src0_type, src1_type>::jit_uni_i8i8_binary_t(
        const pd_t *apd)
    : primitive_impl_t(apd) {
    kernel_ = create_i8i8_binary_kernel<src0_type, src1_type>(pd());
}

template <data_type_t src0_type, data_type_t src1_type>
jit_uni_i8i8_binary_t<src0_type, src1_type>::~jit_uni_i8i8_binary_t() = default;

template <data_type_t src0_type, data_type_t src1_type>
status_t jit_uni_i8i8_binary_t<src0_type, src1_type>::execute(
        const exec_ctx_t &ctx) const {

    const auto src0 = CTX_IN_MEM(const char *, DNNL_ARG_SRC_0);
    const auto src1 = CTX_IN_MEM(const char *, DNNL_ARG_SRC_1);
    auto dst = CTX_OUT_MEM(char *, DNNL_ARG_DST);

    const memory_desc_wrapper src0_d(pd()->src_md(0));
    const dim_t nelems0 = src0_d.nelems(true);

    const int simd_w = (*kernel_).vlen(); // 1-byte elements
    const dim_t nelems0_simd = nelems0 / simd_w;
    const dim_t nelems0_tail = nelems0 % simd_w;
    bool has_tail = nelems0_tail > 0;

    static constexpr int nargs = 2;
    scales_t scales[nargs];
    scales[0] = pd()->attr()->scales_.get(DNNL_ARG_SRC_0);
    scales[1] = pd()->attr()->scales_.get(DNNL_ARG_SRC_1);

    parallel(0, [&](const int ithr, const int nthr) {
        dim_t start = 0, end = 0;
        balance211(nelems0_simd + has_tail, nthr, ithr, start, end);
        if (start >= end) return;

        dim_t spat_offt_count = ((end - start) * simd_w);

        if (has_tail) {
            if (nelems0_simd == 0) {
                // there is only tail
                spat_offt_count = nelems0_tail;
            } else if (end == nelems0_simd + has_tail) {
                // last thread takes care of tail
                spat_offt_count = ((end - start - 1) * simd_w) + nelems0_tail;
            }
        }

        i8i8_binary_kernel_t::call_params_t p;
        p.spat_offt_count = spat_offt_count;
        p.src0 = src0 + start * simd_w;
        p.src1 = src1 + start * simd_w;
        p.dst = dst + start * simd_w;

        if (!scales[0].has_default_values()) p.scales_src0 = scales[0].scales_;
        if (!scales[1].has_default_values()) p.scales_src1 = scales[1].scales_;

        (*kernel_)(&p);
    });

    return status::success;
}

using namespace data_type;

template struct jit_uni_i8i8_binary_t<u8, u8>;
template struct jit_uni_i8i8_binary_t<u8, s8>;
template struct jit_uni_i8i8_binary_t<s8, s8>;
template struct jit_uni_i8i8_binary_t<s8, u8>;

} // namespace cpu
} // namespace impl
} // namespace dnnl
