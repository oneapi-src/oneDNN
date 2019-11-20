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

#include "jit_uni_binary.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

namespace {

typedef float data_t;

using namespace Xbyak;

template <cpu_isa_t isa>
struct jit_binary_base_t : public jit_generator {
    struct call_params_t {
        // keep all sizes at 8 bytes -- jit code expects this
        const data_t *src0, *src1, *dst;
        size_t spat_offt_count;
    };
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_binary_t)

    // cpu specific part
    using Vmm = typename cpu_isa_traits<isa>::Vmm;
    const AddressFrame &vmmword = (isa == avx2) ? yword : zword;
    const int vlen = cpu_isa_traits<isa>::vlen;

    const binary_pd_t *pd_;

    void (*ker)(const call_params_t *);
    void operator()(const call_params_t *p) { (*ker)(p); }

    Reg64 reg_param = abi_param1;

    Reg64 reg_src0 = r8;
    Reg64 reg_src1 = r9;
    Reg64 reg_dst = r10;
    Reg64 reg_spat_offt = r11;
    Reg64 reg_spat_offt_count = r12;
    Reg64 reg_reverse_spat_offt = r13;
    Reg64 reg_tmp = r14;

    size_t unroll_regs_ = isa == avx512_common ? 8 : 4;
    size_t simd_w_ = vlen / sizeof(data_t);
    size_t nelems_simd_tail_;

    void compute_predefined_variables() {
        const memory_desc_wrapper src0_d(pd_->src_md(0));
        size_t nelems = src0_d.nelems(true);
        nelems_simd_tail_ = nelems % simd_w_;
    }

    void load_common_params() {
#define PARAM_OFF(x) offsetof(call_params_t, x)
        mov(reg_spat_offt_count, ptr[reg_param + PARAM_OFF(spat_offt_count)]);
        mov(reg_src0, ptr[reg_param + PARAM_OFF(src0)]);
        mov(reg_src1, ptr[reg_param + PARAM_OFF(src1)]);
        mov(reg_dst, ptr[reg_param + PARAM_OFF(dst)]);
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

    void perform_op(const Vmm &v0, const Vmm &v1) {
        using namespace alg_kind;
        const auto alg = pd_->desc()->alg_kind;
        if (alg == alg_kind::binary_add)
            uni_vaddps(v0, v0, v1);
        else if (alg == alg_kind::binary_mul)
            uni_vmulps(v0, v0, v1);
    }

    virtual void prepare_tail_mask() {}
    virtual void compute_dst(int unroll, bool tail = false) {}

    void forward() {
        Label unroll_loop, unroll_loop_tail, nelems_tail, end;

        // reverse spat_offt to dispatch between labels
        mov(reg_reverse_spat_offt, reg_spat_offt_count);
        xor_(reg_spat_offt, reg_spat_offt); // spat_offt to get addr of src/dst
        L(unroll_loop);
        {
            cmp(reg_reverse_spat_offt, unroll_regs_ * vlen);
            jl(unroll_loop_tail, T_NEAR);

            compute_dst(unroll_regs_);
            sub(reg_reverse_spat_offt, unroll_regs_ * vlen);
            add(reg_spat_offt, unroll_regs_ * vlen);
            jmp(unroll_loop);
        }

        L(unroll_loop_tail);
        {
            cmp(reg_reverse_spat_offt, vlen);
            jl(nelems_tail, T_NEAR);

            compute_dst(1);
            sub(reg_reverse_spat_offt, vlen);
            add(reg_spat_offt, vlen);
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

    // either this stub or duplication at each jit_binary_t ctor due to methods
    // that are participated are not defined at the moment of base ctor
    // initialization.
    void get_code() {
        preamble();
        compute_predefined_variables();
        load_common_params();
        prepare_tail_mask();
        forward();
        postamble();

        ker = reinterpret_cast<decltype(ker)>(
                const_cast<uint8_t *>(this->getCode()));
    }

    jit_binary_base_t(const binary_pd_t *pd) : pd_(pd) {}
};

template <cpu_isa_t isa>
struct jit_binary_t;

template <>
struct jit_binary_t<avx512_common> : public jit_binary_base_t<avx512_common> {
    Opmask tail_opmask = Opmask(1);

    void prepare_tail_mask() override {
        if (!nelems_simd_tail_) return;

        const int mask_f32 = (1 << nelems_simd_tail_) - 1;

        Reg32 regw_tmp = reg_tmp.cvt32();
        mov(regw_tmp, mask_f32);
        kmovw(tail_opmask, regw_tmp);
    }

    void compute_dst(int unroll, bool tail = false) override {
        for (int i = 0; i < unroll; i++) {
            Vmm vreg_tmp_src0 = Vmm(2 * i + 1);
            Vmm vreg_tmp_src1 = Vmm(2 * i + 2);
            if (!tail) {
                uni_vmovups(vreg_tmp_src0, src0_ptr(vlen * i));
                uni_vmovups(vreg_tmp_src1, src1_ptr(vlen * i));
                perform_op(vreg_tmp_src0, vreg_tmp_src1);
                uni_vmovups(dst_ptr(vlen * i), vreg_tmp_src0);
            } else {
                uni_vmovups_tail(
                        vreg_tmp_src0, tail_opmask, src0_ptr(vlen * i));
                uni_vmovups_tail(
                        vreg_tmp_src1, tail_opmask, src1_ptr(vlen * i));
                perform_op(vreg_tmp_src0, vreg_tmp_src1);
                uni_vmovups_tail(dst_ptr(vlen * i), tail_opmask, vreg_tmp_src0);
            }
        }
    }

    jit_binary_t(const binary_pd_t *pd) : jit_binary_base_t(pd) { get_code(); }
};

template <>
struct jit_binary_t<avx2> : public jit_binary_base_t<avx2> {
    Vmm tail_vmask = Vmm(0);

    void prepare_tail_mask() override {
        if (!nelems_simd_tail_) return;

        static const uint32_t mask_f32[14]
                = {0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff,
                        0xffffffff, 0xffffffff, 0, 0, 0, 0, 0, 0, 0};

        mov(reg_tmp,
                reinterpret_cast<size_t>(&mask_f32[7 - nelems_simd_tail_]));
        vmovups(tail_vmask, ptr[reg_tmp]);
    }

    void compute_dst(int unroll, bool tail = false) override {
        for (int i = 0; i < unroll; i++) {
            Vmm vreg_tmp_src0 = Vmm(2 * i + 1);
            Vmm vreg_tmp_src1 = Vmm(2 * i + 2);
            if (!tail) {
                uni_vmovups(vreg_tmp_src0, src0_ptr(vlen * i));
                uni_vmovups(vreg_tmp_src1, src1_ptr(vlen * i));
                perform_op(vreg_tmp_src0, vreg_tmp_src1);
                uni_vmovups(dst_ptr(vlen * i), vreg_tmp_src0);
            } else {
                uni_vmovups_tail(vreg_tmp_src0, tail_vmask, src0_ptr(vlen * i));
                uni_vmovups_tail(vreg_tmp_src1, tail_vmask, src1_ptr(vlen * i));
                perform_op(vreg_tmp_src0, vreg_tmp_src1);
                uni_vmovups_tail(dst_ptr(vlen * i), tail_vmask, vreg_tmp_src0);
            }
        }
    }

    jit_binary_t(const binary_pd_t *pd) : jit_binary_base_t(pd) { get_code(); }
};

} // namespace

template <cpu_isa_t isa>
jit_uni_binary_t<isa>::jit_uni_binary_t(const pd_t *apd)
    : primitive_impl_t(apd) {
    binary_driver_ = new binary_impl::driver_t<isa>(pd());
}

template <cpu_isa_t isa>
jit_uni_binary_t<isa>::~jit_uni_binary_t() {
    delete binary_driver_;
}

template <cpu_isa_t isa>
status_t jit_uni_binary_t<isa>::execute(const exec_ctx_t &ctx) const {
    const auto src0 = CTX_IN_MEM(const data_t *, DNNL_ARG_SRC_0);
    const auto src1 = CTX_IN_MEM(const data_t *, DNNL_ARG_SRC_1);
    auto dst = CTX_OUT_MEM(data_t *, DNNL_ARG_DST);

    // Consider moving to parallel_nd when additional support will be added.
    // It's not used now due to need smaller granularity than nelems.
    parallel(0, [&](const int ithr, const int nthr) {
        binary_driver_->exec(ithr, nthr, src0, src1, dst);
    });

    return status::success;
}

namespace binary_impl {

template <cpu_isa_t isa>
struct driver_t : public c_compatible {

    driver_t(const binary_pd_t *pd) : pd_(pd), ker_(pd_) {}
    ~driver_t() {}

    // Compute strategy:
    // Compute number of full vectors, divide it equally between all threads.
    // Last one will also handle a tail if present.
    void exec(int ithr, int nthr, const data_t *src0, const data_t *src1,
            data_t *dst) {
        typename jit_binary_t<isa>::call_params_t p;

        const memory_desc_wrapper src0_d(pd_->src_md(0));
        const dim_t nelems0 = src0_d.nelems(true);

        const int vlen = cpu_isa_traits<isa>::vlen;
        const int simd_w = vlen / sizeof(data_t);
        const dim_t nelems0_simd = nelems0 / simd_w;
        const dim_t nelems0_tail = nelems0 % simd_w;
        bool has_tail = nelems0_tail > 0;

        dim_t start = 0, end = 0;
        balance211(nelems0_simd + has_tail, nthr, ithr, start, end);
        if (start >= end) return;

        dim_t spat_offt_count = ((end - start) * simd_w) * sizeof(data_t);

        if (has_tail) {
            if (nelems0_simd == 0) {
                // there is only tail
                spat_offt_count = nelems0_tail;
            } else if (end == nelems0_simd + has_tail) {
                // last thread takes care of tail
                spat_offt_count = ((end - start - 1) * simd_w) * sizeof(data_t)
                        + nelems0_tail;
            }
        }

        p.spat_offt_count = spat_offt_count;
        p.src0 = src0 + start * simd_w;
        p.src1 = src1 + start * simd_w;
        p.dst = dst + start * simd_w;

        if (p.spat_offt_count != 0) ker_(&p);
    }

private:
    const binary_pd_t *pd_;

    jit_binary_t<isa> ker_;
};

} // namespace binary_impl

/* struct instantiation */
template struct jit_uni_binary_t<avx2>;
template struct jit_uni_binary_t<avx512_common>;

} // namespace cpu
} // namespace impl
} // namespace dnnl
