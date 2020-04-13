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
#include "common/dnnl_thread.hpp"
#include "common/nstl.hpp"
#include "common/utils.hpp"

#include "cpu/x64/jit_generator.hpp"

#include "cpu/x64/jit_uni_eltwise_int.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

using namespace Xbyak;

struct jit_args {
    const void *from;
    const void *for_comparison;
    const void *to;
    size_t work_amount;
};

struct jit_uni_eltwise_int_kernel : public c_compatible {
    jit_uni_eltwise_int_kernel(const eltwise_desc_t &desc) : desc_(desc) {}
    virtual ~jit_uni_eltwise_int_kernel() {}

    void operator()(const jit_args *args) {
        assert(ker_);
        ker_(args);
    }

protected:
    void (*ker_)(const jit_args *) = nullptr;

    data_type_t data_type() const { return desc_.data_desc.data_type; }
    int dtype_size() const { return types::data_type_size(data_type()); }

private:
    const eltwise_desc_t &desc_;
};

/* jit kernels */
namespace {
using namespace Xbyak;

template <cpu_isa_t isa>
struct jit_uni_relu_kernel_int : public jit_uni_eltwise_int_kernel,
                                 public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_relu_kernel_int)

    jit_uni_relu_kernel_int(const eltwise_desc_t &desc)
        : jit_uni_eltwise_int_kernel(desc), jit_generator() {
        using namespace data_type;

        // Relu for int types: s32, s8; Only forward direction
        assert(desc.alg_kind == alg_kind::eltwise_relu);
        assert(utils::one_of(data_type(), s32, s8));
        assert(utils::one_of(isa, sse41, avx2, avx512_common));

        Reg64 param = abi_param1;

        const size_t vlen = cpu_isa_traits<isa>::vlen;
        const size_t simd_w = vlen / sizeof(float);
        const size_t loop_dec[] = {simd_w, 1};
        const size_t uf[] = {1, 1};
        const size_t shift[] = {dtype_size() * simd_w, (size_t)dtype_size()};
        const bool loop_vectorize[] = {true, false};

        preamble();

#define GET_OFF(field) offsetof(jit_args, field)
        mov(reg_from, ptr[param + GET_OFF(from)]);
        mov(reg_to, ptr[param + GET_OFF(to)]);
        mov(reg_work_amount, ptr[param + GET_OFF(work_amount)]);
#undef GET_OFF

        mov(imm_addr64, float2int(desc.alpha));
        uni_vmovq(xmm_ns, imm_addr64);
        uni_vbroadcastss(vmm_ns, xmm_ns);

        uni_vpxor(vmm_zero, vmm_zero, vmm_zero);
        xor_(reg_s8, reg_s8);
        if (isa == avx512_common) {
            mov(reg_s8.cvt8(), 0x01);
            kmovw(k_mask_s8, reg_s8.cvt32());
        }

        Label loop_label[3];

        for (int id = 0; id < 2; id++) {
            L(loop_label[id]);
            cmp(reg_work_amount, uf[id] * loop_dec[id] - 1);
            jle(loop_label[id + 1], T_NEAR);

            compute_step(loop_vectorize[id], uf[id], shift[id]);

            add(reg_from, uf[id] * shift[id]);
            add(reg_to, uf[id] * shift[id]);

            sub(reg_work_amount, uf[id] * loop_dec[id]);
            jmp(loop_label[id]);
        }

        L(loop_label[2]);
        postamble();

        ker_ = (decltype(ker_))this->getCode();
    }

private:
    enum {
        _rnd_trunc = 3u // Round toward zero
    };
    using Vmm = typename cpu_isa_traits<isa>::Vmm;
    using opmask_t = const Xbyak::Opmask;

    Reg64 reg_from = rax;
    Reg64 reg_to = r8;
    Reg64 reg_work_amount = rsi;
    Reg64 imm_addr64 = rbx;
    Reg64 reg_s8 = r9;

    Xmm xmm_ns = Xmm(14);

    Vmm vmm_ns = Vmm(isa == avx512_common ? 28 : 14);
    Vmm vmm_zero = Vmm(isa == avx512_common ? 29 : 15);
    Vmm vmm_mask = Vmm(isa == avx512_common ? 30 : 12);

    opmask_t k_mask = k1;
    opmask_t k_mask_s8 = k2; // Mask for store 1 byte in case of AVX512

    bool is32bit() const { return data_type() == data_type::s32; }

    // Load 32bit data type (s32)
    void load_32bit(
            const bool vectorize, const Vmm &vr_from, const Address &mem_from) {

        if (vectorize) {
            // load full Vmm size
            uni_vmovups(vr_from, mem_from);
        } else {
            // load exactly one data item
            movss(Xmm(vr_from.getIdx()), mem_from);
        }
    }

    // Load 8bit data type (s8)
    void load_8bit(
            const bool vectorize, const Vmm &vr_from, const Address &mem_from) {

        // data type s8 load as s32
        if (vectorize) {
            // load full Vmm size
            if (isa == sse41)
                pmovsxbd(vr_from, mem_from);
            else
                vpmovsxbd(vr_from, mem_from);
        } else {
            // load exactly one data item
            mov(reg_s8.cvt8(), mem_from);
            movsx(reg_s8.cvt32(), reg_s8.cvt8());
            uni_vmovq(Xmm(vr_from.getIdx()), reg_s8);
        }
    }

    // Load vregs with data from mem
    void load(
            const bool vectorize, const Vmm &vr_from, const Address &mem_from) {

        // Branching on data size
        if (is32bit())
            load_32bit(vectorize, vr_from, mem_from);
        else
            load_8bit(vectorize, vr_from, mem_from);
    }

    // Processing
    void process(const Vmm &vr_to, const Vmm &vr_from);

    // Store s32 for any isa
    void store_32bit(
            const bool vectorize, const Address &mem_to, const Vmm &vr_to) {
        if (vectorize) {
            // store full Vmm size
            uni_vmovups(mem_to, vr_to);
        } else {
            // store exactly one data item
            movss(mem_to, Xmm(vr_to.getIdx()));
        }
    }

    // Store s8 - isa-dependent
    void store_8bit(
            const bool vectorize, const Address &mem_to, const Vmm &vr_to);

    // Store results from vregs to mem
    void store(const bool vectorize, const Address &mem_to, const Vmm &vr_to) {
        // Branching on data size
        if (is32bit())
            store_32bit(vectorize, mem_to, vr_to);
        else
            store_8bit(vectorize, mem_to, vr_to);
    }

    void compute_step(bool vectorize, const size_t uf, const size_t shift) {

        auto vreg_from = [&](const size_t i) -> Vmm { return Vmm(i + 1); };
        auto vreg_to = [&](const size_t i) -> Vmm { return Vmm(uf + i + 1); };

        // 1. Load (vregs <- mem)
        for (size_t i = 0; i < uf; i++)
            load(vectorize, vreg_from(i), ptr[reg_from + i * shift]);

        // 2. Process (vregs <- vergs)
        for (size_t i = 0; i < uf; i++)
            process(vreg_to(i), vreg_from(i));

        // 3. Store (mem <- vregs)
        for (size_t i = 0; i < uf; i++)
            store(vectorize, ptr[reg_to + i * shift], vreg_to(i));
    }
};

template <cpu_isa_t isa>
void jit_uni_relu_kernel_int<isa>::process(
        const Vmm &vr_to, const Vmm &vr_from) {
    assert(!"unsupported isa");
}

template <>
void jit_uni_relu_kernel_int<sse41>::process(
        const Vmm &vr_to, const Vmm &vr_from) {

    cvtdq2ps(vr_from, vr_from);
    movups(vr_to, vr_from);
    mulps(vr_to, vmm_ns);

    Vmm mask = Vmm(0);
    movups(mask, vr_from);
    cmpps(mask, vmm_zero, _cmp_nle_us);
    blendvps(vr_to, vr_from);
    uni_vroundps(vr_to, vr_to, _rnd_trunc);
    cvtps2dq(vr_to, vr_to);
}

template <>
void jit_uni_relu_kernel_int<avx2>::process(
        const Vmm &vr_to, const Vmm &vr_from) {

    vcvtdq2ps(vr_from, vr_from);
    vmulps(vr_to, vr_from, vmm_ns);
    vcmpgtps(vmm_mask, vr_from, vmm_zero);
    vblendvps(vr_to, vr_to, vr_from, vmm_mask);
    uni_vroundps(vr_to, vr_to, _rnd_trunc);
    vcvtps2dq(vr_to, vr_to);
}

template <>
void jit_uni_relu_kernel_int<avx512_common>::process(
        const Vmm &vr_to, const Vmm &vr_from) {

    vcvtdq2ps(vr_from, vr_from);
    vmulps(vr_to, vr_from, vmm_ns);
    vcmpps(k_mask, vr_from, vmm_zero, _cmp_nle_us);
    vblendmps(vr_to | k_mask, vr_to, vr_from);
    vcvtps2dq(vr_to | T_rz_sae, vr_to);
}

template <cpu_isa_t isa>
void jit_uni_relu_kernel_int<isa>::store_8bit(
        const bool vectorize, const Address &mem_to, const Vmm &vr_to) {
    assert(!"unsupported isa");
}

template <>
void jit_uni_relu_kernel_int<sse41>::store_8bit(
        const bool vectorize, const Address &mem_to, const Vmm &vr_to) {
    if (vectorize) {
        // store full Vmm size
        // s32 -> s16
        packssdw(vr_to, vmm_zero);

        // s16 -> s8
        packsswb(vr_to, vmm_zero);
        movd(mem_to, Xmm(vr_to.getIdx()));
    } else {
        // store exactly one data item
        // s32 save as s8
        packssdw(vr_to, vmm_zero);
        packsswb(vr_to, vmm_zero);
        movd(reg_s8.cvt32(), Xmm(vr_to.getIdx()));
        mov(mem_to, reg_s8.cvt8());
    }
}

template <>
void jit_uni_relu_kernel_int<avx2>::store_8bit(
        const bool vectorize, const Address &mem_to, const Vmm &vr_to) {
    if (vectorize) {
        // store full Vmm size
        // s32 -> s16 = {qw0, 0, qw1, 0}
        vpackssdw(vr_to, vr_to, vmm_zero);

        // permute to restore order{qw0, 0, qw1, 0} -> {qw0, qw1, 0, 0}
        vpermq(Ymm(vr_to.getIdx()), Ymm(vr_to.getIdx()), 0x58);

        // s16 -> s8 : {16 x s16}{16 x 0} -> {32 x s8}
        vpacksswb(vr_to, vr_to, vmm_zero);
        uni_vmovq(mem_to, Xmm(vr_to.getIdx()));
    } else {
        // store exactly one data item
        // s32 save as s8
        vpackssdw(vr_to, vr_to, vmm_zero);
        vpacksswb(vr_to, vr_to, vmm_zero);
        vmovd(reg_s8.cvt32(), Xmm(vr_to.getIdx()));
        mov(mem_to, reg_s8.cvt8());
    }
}

template <>
void jit_uni_relu_kernel_int<avx512_common>::store_8bit(
        const bool vectorize, const Address &mem_to, const Vmm &vr_to) {
    if (vectorize) {
        // store full Vmm size
        vpmovsdb(mem_to, vr_to);
    } else {
        // store exactly one data item
        // s32 save as s8
        vpmovsdb(mem_to, vr_to | k_mask_s8);
    }
}

} /* namespace */

template <cpu_isa_t isa, data_type_t d_type>
status_t jit_uni_eltwise_int_fwd_t<isa, d_type>::pd_t::init(engine_t *engine) {
    bool ok = mayiuse(isa) && desc()->data_desc.data_type == d_type
            && desc()->alg_kind == alg_kind::eltwise_relu // only relu so far
            && !has_zero_dim_memory()
            && memory_desc_wrapper(src_md()).is_dense(true)
            && attr()->has_default_values();

    return ok ? status::success : status::unimplemented;
}

template <cpu_isa_t isa, data_type_t d_type>
jit_uni_eltwise_int_fwd_t<isa, d_type>::jit_uni_eltwise_int_fwd_t(
        const pd_t *apd)
    : primitive_t(apd) {
    const auto &desc = *pd()->desc();
    kernel_ = new jit_uni_relu_kernel_int<isa>(desc);
}

template <cpu_isa_t isa, data_type_t d_type>
jit_uni_eltwise_int_fwd_t<isa, d_type>::~jit_uni_eltwise_int_fwd_t() {
    delete kernel_;
}

template <cpu_isa_t isa, impl::data_type_t d_type>
void jit_uni_eltwise_int_fwd_t<isa, d_type>::execute_forward(
        const exec_ctx_t &ctx) const {
    auto src = CTX_IN_MEM(const data_t *, DNNL_ARG_SRC);
    auto dst = CTX_OUT_MEM(data_t *, DNNL_ARG_DST);

    const memory_desc_wrapper data_d(pd()->src_md());

    const size_t nelems = data_d.nelems(true);

    src += data_d.offset0();
    dst += data_d.offset0();

    const int cache_line = 64 / data_d.data_type_size();
    parallel(0, [&](const int ithr, const int nthr) {
        size_t start {0}, end {0};

        balance211(utils::div_up(nelems, cache_line), nthr, ithr, start, end);
        start = nstl::min(nelems, start * cache_line);
        end = nstl::min(nelems, end * cache_line);

        auto arg = jit_args();
        arg.from = (const void *)&src[start];
        arg.for_comparison = (const void *)&src[start];
        arg.to = (const void *)&dst[start];
        arg.work_amount = end - start;
        if (arg.work_amount) (*kernel_)(&arg);
    });
}

using namespace data_type;

template struct jit_uni_eltwise_int_fwd_t<sse41, s32>;
template struct jit_uni_eltwise_int_fwd_t<avx2, s32>;
template struct jit_uni_eltwise_int_fwd_t<avx512_common, s32>;

template struct jit_uni_eltwise_int_fwd_t<sse41, s8>;
template struct jit_uni_eltwise_int_fwd_t<avx2, s8>;
template struct jit_uni_eltwise_int_fwd_t<avx512_common, s8>;

} // namespace cpu
} // namespace impl
} // namespace dnnl
