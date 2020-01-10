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

#include "jit_avx512_core_bf16cvt.hpp"
#include "jit_generator.hpp"
#include "jit_uni_eltwise_injector.hpp"

#include "jit_uni_binary.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

using namespace Xbyak;

struct binary_kernel_t {
    struct call_params_t {
        // keep all sizes at 8 bytes -- jit code expects this
        const void *src0, *src1, *dst;
        size_t spat_offt_count;
    };

    binary_kernel_t(int vlen) : vlen_(vlen) {}
    virtual ~binary_kernel_t() = default;

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
struct jit_uni_binary_kernel_t : public binary_kernel_t, public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_binary_kernel_t)

    using Vmm = typename cpu_isa_traits<isa>::Vmm;
    const AddressFrame &vmmword = (isa == avx2) ? yword : zword;

    const binary_pd_t *pd_;
    bool is_bf16_;

    Reg64 reg_param = abi_param1;

    Reg64 reg_src0 = r8;
    Reg64 reg_src1 = r9;
    Reg64 reg_dst = r10;
    Reg64 reg_spat_offt = r11;
    Reg64 reg_spat_offt_count = r12;
    Reg64 reg_reverse_spat_offt = r13;
    Reg64 reg_tmp = r14;
    Reg64 reg_elt_inj_table = r15;

    Xmm xsum_scale = Xmm(15);
    Vmm vsum_scale = Vmm(isa == avx2 ? 15 : 27); // 28-31 are for bf16_emu

    size_t unroll_regs_ = isa == avx2 ? 4 : 8;
    size_t simd_w_ = 0;
    size_t tail_size_ = 0;
    size_t data_type_size_ = 0;
    bool do_sum_ = false;
    float sum_scale_ = 0.f;

    static constexpr cpu_isa_t inject_isa
            = isa == avx512_core_bf16 ? avx512_core : isa;
    std::unique_ptr<jit_uni_eltwise_injector_f32<inject_isa>> eltwise_injector_;
    Opmask elt_inj_opmask = Opmask(2);

    void init() {
        const memory_desc_wrapper src0_d(pd_->src_md(0));
        size_t nelems = src0_d.nelems(true);
        is_bf16_ = src0_d.data_type() == data_type::bf16;
        // it's float due to for bfloat16 we still load 16 elements, not 32.
        simd_w_ = vlen_ / sizeof(float);
        tail_size_ = nelems % simd_w_;
        data_type_size_ = is_bf16_ ? sizeof(bfloat16_t) : sizeof(float);

        const auto &po = pd_->attr()->post_ops_;
        do_sum_ = po.contain(primitive_kind::sum, 0)
                && po.entry_[0].sum.scale != 0.f;
        sum_scale_ = do_sum_ ? po.entry_[0].sum.scale : 0.f;

        int elt_idx = po.find(primitive_kind::eltwise);
        if (elt_idx != -1) {
            const auto &e = po.entry_[elt_idx].eltwise;
            eltwise_injector_.reset(
                    new jit_uni_eltwise_injector_f32<inject_isa>(this, e.alg,
                            e.alpha, e.beta, 1.f, true, reg_elt_inj_table,
                            elt_inj_opmask));
        }
    }

    void load_kernel_params() {
        mov(reg_tmp, float2int(sum_scale_));
        uni_vmovq(xsum_scale, reg_tmp);
        uni_vbroadcastss(vsum_scale, xsum_scale);
#define PARAM_OFF(x) offsetof(call_params_t, x)
        mov(reg_spat_offt_count, ptr[reg_param + PARAM_OFF(spat_offt_count)]);
        mov(reg_src0, ptr[reg_param + PARAM_OFF(src0)]);
        mov(reg_src1, ptr[reg_param + PARAM_OFF(src1)]);
        mov(reg_dst, ptr[reg_param + PARAM_OFF(dst)]);
#undef PARAM_OFF
        if (eltwise_injector_) eltwise_injector_->load_table_addr();
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

    virtual void prepare_isa_subkernel() = 0;
    virtual void compute_dst(int unroll, bool tail = false) = 0;

    void forward() {
        Label unroll_loop, unroll_loop_tail, nelems_tail, end;

        // reverse spat_offt to dispatch between labels
        mov(reg_reverse_spat_offt, reg_spat_offt_count);
        xor_(reg_spat_offt, reg_spat_offt); // spat_offt to get addr of src/dst
        size_t offt = simd_w_ * data_type_size_;
        L(unroll_loop);
        {
            cmp(reg_reverse_spat_offt, unroll_regs_ * offt);
            jl(unroll_loop_tail, T_NEAR);

            compute_dst(unroll_regs_);
            sub(reg_reverse_spat_offt, unroll_regs_ * offt);
            add(reg_spat_offt, unroll_regs_ * offt);
            jmp(unroll_loop);
        }

        L(unroll_loop_tail);
        {
            cmp(reg_reverse_spat_offt, offt);
            jl(nelems_tail, T_NEAR);

            compute_dst(1);
            sub(reg_reverse_spat_offt, offt);
            add(reg_spat_offt, offt);
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

    void get_code() {
        preamble();
        load_kernel_params();
        prepare_isa_subkernel();
        forward();
        postamble();

        if (eltwise_injector_) eltwise_injector_->prepare_table();

        ker_ = getCode<decltype(ker_)>();
    }

    jit_uni_binary_kernel_t(const binary_pd_t *pd)
        : binary_kernel_t(cpu_isa_traits<isa>::vlen), pd_(pd) {
        init();
    }
    virtual ~jit_uni_binary_kernel_t() = default;
};

template <cpu_isa_t isa, data_type_t src_type>
struct jit_uni_binary_subkernel_t;

template <data_type_t src_type>
struct jit_uni_binary_subkernel_t<avx512_core_bf16, src_type>
    : public jit_uni_binary_kernel_t<avx512_core_bf16> {
    Opmask tail_opmask = Opmask(1);

    void prepare_tail_mask() {
        if (!tail_size_) return;

        const int mask_f32 = (1 << tail_size_) - 1;
        Reg32 regw_tmp = reg_tmp.cvt32();
        mov(regw_tmp, mask_f32);
        kmovd(tail_opmask, regw_tmp);
    }

    void prepare_isa_subkernel() override { prepare_tail_mask(); }

    void load_no_tail(const Vmm &dst, const Address &src, data_type_t dt) {
        switch (dt) {
            case data_type::f32: uni_vmovups(dst, src); break;
            case data_type::bf16:
                vpmovzxwd(dst, src);
                vpslld(dst, dst, 0x10);
                break;
            default: assert(!"unreachable");
        }
    }

    void load_tail(const Vmm &dst, const Opmask &opmask, const Address &src,
            data_type_t dt) {
        switch (dt) {
            case data_type::f32: uni_vmovups_tail(dst, opmask, src); break;
            case data_type::bf16:
                vpmovzxwd(dst | opmask, src);
                vpslld(dst, dst, 0x10);
                break;
            default: assert(!"unreachable");
        }
    }

    void store_no_tail(const Address &dst, const Vmm &src, data_type_t dt) {
        Ymm ymm_src = Ymm(src.getIdx());
        switch (dt) {
            case data_type::f32: uni_vmovups(dst, src); break;
            case data_type::bf16:
                vcvtneps2bf16(ymm_src, src);
                vmovdqu16(dst, ymm_src);
                break;
            default: assert(!"unreachable");
        }
    }

    void store_tail(const Address &dst, const Opmask &opmask, const Vmm &src,
            data_type_t dt) {
        Ymm ymm_src = Ymm(src.getIdx());
        switch (dt) {
            case data_type::f32: uni_vmovups_tail(dst, opmask, src); break;
            case data_type::bf16:
                vcvtneps2bf16(ymm_src, src);
                vmovdqu16(dst | opmask, ymm_src);
                break;
            default: assert(!"unreachable");
        }
    }

    void load(const Vmm &dst, const Address &src, data_type_t dt, bool tail) {
        if (!tail)
            load_no_tail(dst, src, dt);
        else
            load_tail(dst, tail_opmask, src, dt);
    }

    void store(const Address &dst, const Vmm &src, data_type_t dt, bool tail) {
        if (!tail)
            store_no_tail(dst, src, dt);
        else
            store_tail(dst, tail_opmask, src, dt);
    }

    void compute_dst(int unroll, bool tail = false) override {
        for (int i = 0; i < unroll; i++) {
            Vmm vreg_tmp_src0 = Vmm(2 * i + 1);
            Vmm vreg_tmp_src1 = Vmm(2 * i + 2);
            int offt = i * (vlen_ / (is_bf16_ ? 2 : 1));
            load(vreg_tmp_src0, src0_ptr(offt), src_type, tail);
            load(vreg_tmp_src1, src1_ptr(offt), src_type, tail);
            perform_op(vreg_tmp_src0, vreg_tmp_src1);
            if (do_sum_) {
                load(vreg_tmp_src1, dst_ptr(offt), src_type, tail);
                uni_vfmadd231ps(vreg_tmp_src0, vreg_tmp_src1, vsum_scale);
            }
            if (eltwise_injector_)
                eltwise_injector_->compute_vector(vreg_tmp_src0.getIdx());
            store(dst_ptr(offt), vreg_tmp_src0, src_type, tail);
        }
    }

    jit_uni_binary_subkernel_t(const binary_pd_t *pd)
        : jit_uni_binary_kernel_t(pd) {
        get_code();
    }
};

template <data_type_t src_type>
struct jit_uni_binary_subkernel_t<avx512_core, src_type>
    : public jit_uni_binary_kernel_t<avx512_core> {
    Opmask tail_opmask = Opmask(1);

    // FP32->BF16 emulation
    bf16_emulation_t *bf16_emu_ {nullptr};
    Reg64 reg_bf16_tmp = reg_tmp;
    Zmm bf16_emu_reserved_1 = Zmm(28);
    Zmm bf16_emu_reserved_2 = Zmm(29);
    Zmm bf16_emu_reserved_3 = Zmm(30);
    Zmm bf16_emu_reserved_4 = Zmm(31);

    void prepare_tail_mask() {
        if (!tail_size_) return;

        const int mask_f32 = (1 << tail_size_) - 1;

        Reg32 regw_tmp = reg_tmp.cvt32();
        mov(regw_tmp, mask_f32);
        kmovd(tail_opmask, regw_tmp);
    }

    void prepare_bf16_emulator() {
        if (is_bf16_) { // init emulation of bfloat16 operations
            bf16_emu_ = new bf16_emulation_t(this, bf16_emu_reserved_1,
                    bf16_emu_reserved_2, bf16_emu_reserved_3, reg_bf16_tmp,
                    bf16_emu_reserved_4, bf16_emu_reserved_4);
            bf16_emu_->init_vcvtneps2bf16();
        }
    }

    void prepare_isa_subkernel() override {
        prepare_tail_mask();
        prepare_bf16_emulator();
    }

    void load_no_tail(const Vmm &dst, const Address &src, data_type_t dt) {
        switch (dt) {
            case data_type::f32: uni_vmovups(dst, src); break;
            case data_type::bf16:
                vpmovzxwd(dst, src);
                vpslld(dst, dst, 0x10);
                break;
            default: assert(!"unreachable");
        }
    }

    void load_tail(const Vmm &dst, const Opmask &opmask, const Address &src,
            data_type_t dt) {
        switch (dt) {
            case data_type::f32: uni_vmovups_tail(dst, opmask, src); break;
            case data_type::bf16:
                vpmovzxwd(dst | opmask, src);
                vpslld(dst, dst, 0x10);
                break;
            default: assert(!"unreachable");
        }
    }

    void store_no_tail(const Address &dst, const Vmm &src, data_type_t dt) {
        Ymm ymm_src = Ymm(src.getIdx());
        switch (dt) {
            case data_type::f32: uni_vmovups(dst, src); break;
            case data_type::bf16:
                bf16_emu_->vcvtneps2bf16(ymm_src, src);
                vmovdqu16(dst, ymm_src);
                break;
            default: assert(!"unreachable");
        }
    }

    void store_tail(const Address &dst, const Opmask &opmask, const Vmm &src,
            data_type_t dt) {
        Ymm ymm_src = Ymm(src.getIdx());
        switch (dt) {
            case data_type::f32: uni_vmovups_tail(dst, opmask, src); break;
            case data_type::bf16:
                bf16_emu_->vcvtneps2bf16(ymm_src, src);
                vmovdqu16(dst | opmask, ymm_src);
                break;
            default: assert(!"unreachable");
        }
    }

    void load(const Vmm &dst, const Address &src, data_type_t dt, bool tail) {
        if (!tail)
            load_no_tail(dst, src, dt);
        else
            load_tail(dst, tail_opmask, src, dt);
    }

    void store(const Address &dst, const Vmm &src, data_type_t dt, bool tail) {
        if (!tail)
            store_no_tail(dst, src, dt);
        else
            store_tail(dst, tail_opmask, src, dt);
    }

    void compute_dst(int unroll, bool tail = false) override {
        for (int i = 0; i < unroll; i++) {
            Vmm vreg_tmp_src0 = Vmm(2 * i + 1);
            Vmm vreg_tmp_src1 = Vmm(2 * i + 2);
            int offt = i * (vlen_ / (is_bf16_ ? 2 : 1));
            load(vreg_tmp_src0, src0_ptr(offt), src_type, tail);
            load(vreg_tmp_src1, src1_ptr(offt), src_type, tail);
            perform_op(vreg_tmp_src0, vreg_tmp_src1);
            if (do_sum_) {
                load(vreg_tmp_src1, dst_ptr(offt), src_type, tail);
                uni_vfmadd231ps(vreg_tmp_src0, vreg_tmp_src1, vsum_scale);
            }
            if (eltwise_injector_)
                eltwise_injector_->compute_vector(vreg_tmp_src0.getIdx());
            store(dst_ptr(offt), vreg_tmp_src0, src_type, tail);
        }
    }

    jit_uni_binary_subkernel_t(const binary_pd_t *pd)
        : jit_uni_binary_kernel_t(pd) {
        get_code();
    }
    virtual ~jit_uni_binary_subkernel_t() { delete bf16_emu_; }
};

template <data_type_t src_type>
struct jit_uni_binary_subkernel_t<avx2, src_type>
    : public jit_uni_binary_kernel_t<avx2> {
    Vmm tail_vmask = Vmm(0);

    void prepare_tail_mask() {
        if (!tail_size_) return;

        static const uint32_t mask_f32[14]
                = {0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff,
                        0xffffffff, 0xffffffff, 0, 0, 0, 0, 0, 0, 0};

        mov(reg_tmp, reinterpret_cast<size_t>(&mask_f32[7 - tail_size_]));
        vmovups(tail_vmask, ptr[reg_tmp]);
    }

    void prepare_isa_subkernel() override { prepare_tail_mask(); }

    void load(const Vmm &dst, const Address &src, bool tail) {
        if (!tail)
            uni_vmovups(dst, src);
        else
            uni_vmovups_tail(dst, tail_vmask, src);
    }

    void store(const Address &dst, const Vmm &src, bool tail) {
        if (!tail)
            uni_vmovups(dst, src);
        else
            uni_vmovups_tail(dst, tail_vmask, src);
    }

    void compute_dst(int unroll, bool tail = false) override {
        for (int i = 0; i < unroll; i++) {
            Vmm vreg_tmp_src0 = Vmm(2 * i + 1);
            Vmm vreg_tmp_src1 = Vmm(2 * i + 2);
            int offt = vlen_ * i;
            load(vreg_tmp_src0, src0_ptr(offt), tail);
            load(vreg_tmp_src1, src1_ptr(offt), tail);
            perform_op(vreg_tmp_src0, vreg_tmp_src1);
            if (do_sum_) {
                load(vreg_tmp_src1, dst_ptr(offt), tail);
                uni_vfmadd231ps(vreg_tmp_src0, vreg_tmp_src1, vsum_scale);
            }
            if (eltwise_injector_)
                eltwise_injector_->compute_vector(vreg_tmp_src0.getIdx());
            store(dst_ptr(offt), vreg_tmp_src0, tail);
        }
    }

    jit_uni_binary_subkernel_t(const binary_pd_t *pd)
        : jit_uni_binary_kernel_t(pd) {
        get_code();
    }
};

template <data_type_t src_type>
std::unique_ptr<binary_kernel_t> create_binary_kernel(const binary_pd_t *pd) {
    if (mayiuse(avx512_core_bf16)) {
        using subkernel_t
                = jit_uni_binary_subkernel_t<avx512_core_bf16, src_type>;
        return std::unique_ptr<subkernel_t> {new subkernel_t(pd)};
    } else if (mayiuse(avx512_core)) {
        using subkernel_t = jit_uni_binary_subkernel_t<avx512_core, src_type>;
        return std::unique_ptr<subkernel_t> {new subkernel_t(pd)};
    } else if (mayiuse(avx2)) {
        using subkernel_t = jit_uni_binary_subkernel_t<avx2, src_type>;
        return std::unique_ptr<subkernel_t> {new subkernel_t(pd)};
    }
    return nullptr;
}

template <data_type_t src_type>
jit_uni_binary_t<src_type>::jit_uni_binary_t(const pd_t *apd)
    : primitive_impl_t(apd) {
    kernel_ = create_binary_kernel<src_type>(pd());
}

template <data_type_t src_type>
jit_uni_binary_t<src_type>::~jit_uni_binary_t() = default;

template <data_type_t src_type>
status_t jit_uni_binary_t<src_type>::execute(const exec_ctx_t &ctx) const {
    using data_t = typename prec_traits<src_type>::type;

    const auto src0 = CTX_IN_MEM(const data_t *, DNNL_ARG_SRC_0);
    const auto src1 = CTX_IN_MEM(const data_t *, DNNL_ARG_SRC_1);
    auto dst = CTX_OUT_MEM(data_t *, DNNL_ARG_DST);

    const memory_desc_wrapper src0_d(pd()->src_md(0));
    const dim_t nelems0 = src0_d.nelems(true);

    const int simd_w = (*kernel_).vlen() / sizeof(float);
    const dim_t nelems0_simd = nelems0 / simd_w;
    const dim_t nelems0_tail = nelems0 % simd_w;
    bool has_tail = nelems0_tail > 0;

    // Compute strategy:
    // Compute number of full vectors, divide it equally between all threads.
    // Last one will also handle a tail if present.
    parallel(0, [&](const int ithr, const int nthr) {
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
                spat_offt_count = (((end - start - 1) * simd_w) + nelems0_tail)
                        * sizeof(data_t);
            }
        }

        binary_kernel_t::call_params_t p;
        p.spat_offt_count = spat_offt_count;
        p.src0 = src0 + start * simd_w;
        p.src1 = src1 + start * simd_w;
        p.dst = dst + start * simd_w;

        (*kernel_)(&p);
    });

    return status::success;
}

using namespace data_type;

template struct jit_uni_binary_t<f32>;
template struct jit_uni_binary_t<bf16>;

} // namespace cpu
} // namespace impl
} // namespace dnnl
