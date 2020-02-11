/*******************************************************************************
* Copyright 2017-2019 Intel Corporation
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

#include "c_types_map.hpp"
#include "dnnl_thread.hpp"
#include "nstl.hpp"
#include "utils.hpp"

#include "bfloat16.hpp"
#include "jit_avx512_core_bf16cvt.hpp"
#include "jit_uni_eltwise.hpp"
#include "jit_uni_eltwise_injector.hpp"

#define GET_OFF(field) offsetof(jit_args, field)

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

struct jit_uni_eltwise_kernel : public c_compatible {
    const eltwise_desc_t &desc_;

    void (*ker_)(const jit_args *);
    void operator()(const jit_args *args) {
        assert(ker_);
        ker_(args);
    }

    jit_uni_eltwise_kernel(const eltwise_desc_t &desc)
        : desc_(desc), ker_(nullptr) {}
    virtual ~jit_uni_eltwise_kernel() {}

protected:
    bool is_bwd() const { return desc_.prop_kind == prop_kind::backward_data; }
    data_type_t data_type() const { return desc_.data_desc.data_type; }
    bool is_bf16() const { return data_type() == data_type::bf16; }
    int dtype_size() const { return types::data_type_size(data_type()); }
};

/* jit kernels */
namespace {
using namespace Xbyak;
struct jit_bf16_eltwise_injector {
    jit_bf16_eltwise_injector(jit_generator *host, Zmm zmm_idx,
            Opmask k_mask_cvt, Opmask k_tail_mask, Opmask k_full_mask,
            bf16_emulation_t *emu)
        : h(host)
        , emu_(emu)
        , zmm_idx_(zmm_idx)
        , k_mask_cvt_(k_mask_cvt)
        , k_tail_mask_(k_tail_mask)
        , k_full_mask_(k_full_mask) {}

    void write_idx_table() {
        h->align(64);
        h->L(idx_table_);
        const uint16_t _idx[] = {0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7,
                8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15};
        for (size_t i = 0; i < sizeof(_idx) / sizeof(_idx[0]); ++i)
            h->dw(_idx[i]);
    }

    void load_idx_table() {
        Reg64 p_idx_table = Xbyak::util::r13;
        h->push(p_idx_table);
        h->mov(p_idx_table, idx_table_);
        h->vmovups(zmm_idx_, h->ptr[p_idx_table]);
        h->pop(p_idx_table);
    }
    void prepare_cvt_mask() {
        Reg64 mask_reg = Xbyak::util::r14;
        h->push(mask_reg);
        h->mov(mask_reg.cvt32(), 0xAAAAAAAA);
        h->kmovd(k_mask_cvt_, mask_reg.cvt32());

        h->mov(mask_reg.cvt32(), 0x1);
        h->kmovd(k_tail_mask_, mask_reg.cvt32());

        h->mov(mask_reg.cvt32(), 0xffff);
        h->kmovd(k_full_mask_, mask_reg.cvt32());
        h->pop(mask_reg);
    }

    void load_bf16_cvt_to_f32(size_t idx, Reg64 reg_from, bool is_tail = false,
            size_t offset = 0) {
        Zmm zmm_f32 = Zmm(idx);
        zmm_f32 = is_tail ? zmm_f32 | k_tail_mask_ | Xbyak::util::T_z : zmm_f32;
        h->vpmovzxwd(zmm_f32, h->ptr[reg_from + offset]);
        h->vpslld(zmm_f32, zmm_f32, 16);
    }

    void cvt_f32_to_bf16_store(int step, size_t idx, Reg64 reg_to,
            bool is_tail = false, size_t offset = 0) {
        assert(step >= 1 && step <= 2
                && IMPLICATION(step == 2, is_tail == false));
        if (step == 2 && !is_tail) {
            Ymm ymm_bf16_0 = Ymm(idx);
            Ymm ymm_bf16_1 = Ymm(idx + 1);
            Zmm zmm_f32_0 = Zmm(idx);
            Zmm zmm_f32_1 = Zmm(idx + 1);
            if (emu_) {
                emu_->vcvtneps2bf16(ymm_bf16_0, zmm_f32_0);
                emu_->vcvtneps2bf16(ymm_bf16_1, zmm_f32_1);
                h->vinserti64x4(zmm_f32_0, zmm_f32_0, ymm_bf16_1, 1);
                h->vmovups(h->ptr[reg_to + offset], zmm_f32_0);
            } else {
                h->vcvtne2ps2bf16(zmm_f32_1, zmm_f32_1, zmm_f32_0);
                h->vmovups(h->ptr[reg_to + offset], zmm_f32_1);
            }
        } else {
            Ymm ymm_bf16 = Ymm(idx);
            Zmm zmm_f32 = Zmm(idx);
            if (emu_)
                emu_->vcvtneps2bf16(ymm_bf16, zmm_f32);
            else
                h->vcvtneps2bf16(ymm_bf16, zmm_f32);
            if (!is_tail)
                h->vmovdqu16(h->ptr[reg_to + offset] | k_full_mask_, ymm_bf16);
            else
                h->vmovdqu16(h->ptr[reg_to + offset] | k_tail_mask_, ymm_bf16);
        }
    }

private:
    jit_generator *const h;
    bf16_emulation_t *const emu_;
    Xbyak::Label idx_table_;
    Xbyak::Zmm zmm_idx_;
    Xbyak::Opmask k_mask_cvt_, k_tail_mask_, k_full_mask_;
};

template <cpu_isa_t isa>
struct jit_uni_relu_kernel_float : public jit_uni_eltwise_kernel,
                                   public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_relu_kernel_float)

    void compute_step(bool vectorize, const int uf, const int shift) {
        auto load_vec = [=](int idx, Reg64 reg_addr, int offset) {
            if (is_bf16()) {
                bf16_injector_->load_bf16_cvt_to_f32(
                        idx, reg_addr, false, offset);
            } else {
                uni_vmovups(Vmm(idx), ptr[reg_addr + offset]);
            }
        };

        auto load_elem = [=](int idx, Reg64 reg_addr, int offset) {
            if (is_bf16()) {
                bf16_injector_->load_bf16_cvt_to_f32(
                        idx, reg_addr, true, offset);
            } else {
                movss(Xmm(idx), ptr[reg_addr + offset]);
            }
        };

        for (int i = 0; i < uf; i++) {
            int offset = i * shift;
            if (vectorize) {
                load_vec(i + 1, reg_from, offset);
                if (is_bwd()) {
                    load_vec(uf + i + 1, reg_for_comparison, offset);
                }
            } else {
                load_elem(i + 1, reg_from, offset);
                if (is_bwd()) {
                    load_elem(uf + i + 1, reg_for_comparison, offset);
                }
            }
        }

        if (isa == sse41) {
            for (int i = 0; i < uf; i++) {
                movups(Vmm(2 * uf + i + 1), Vmm(i + 1));
                mulps(Vmm(2 * uf + i + 1), vmm_ns);

                Vmm mask = Vmm(0);
                if (is_bwd()) {
                    movups(mask, Vmm(uf + i + 1));
                    cmpps(mask, vmm_zero, _cmp_nle_us);
                } else {
                    movups(mask, Vmm(i + 1));
                    cmpps(mask, vmm_zero, _cmp_nle_us);
                }
                blendvps(Vmm(2 * uf + i + 1), Vmm(i + 1));
            }
        } else {
            for (int i = 0; i < uf; i++) {
                vmulps(Vmm(2 * uf + i + 1), Vmm(i + 1), vmm_ns);
                if (isa == avx2) {
                    if (is_bwd())
                        vcmpgtps(vmm_mask, Vmm(uf + i + 1), vmm_zero);
                    else
                        vcmpgtps(vmm_mask, Vmm(i + 1), vmm_zero);

                    vblendvps(Vmm(2 * uf + i + 1), Vmm(2 * uf + i + 1),
                            Vmm(i + 1), vmm_mask);

                } else {
                    if (is_bwd())
                        vcmpps(k_mask, Vmm(uf + i + 1), vmm_zero, _cmp_nle_us);
                    else
                        vcmpps(k_mask, Vmm(i + 1), vmm_zero, _cmp_nle_us);
                    vblendmps(Vmm(2 * uf + i + 1) | k_mask, Vmm(2 * uf + i + 1),
                            Vmm(i + 1));
                }
            }
        }

        const int i_step = (mayiuse(avx512_core_bf16) && is_bf16() && vectorize
                                   && uf % 2 == 0)
                ? 2
                : 1;
        for (int i = 0; i < uf; i += i_step) {
            size_t idx = 2 * uf + i + 1;
            size_t offset = i * shift;
            if (vectorize)
                if (is_bf16())
                    bf16_injector_->cvt_f32_to_bf16_store(
                            i_step, idx, reg_to, false, i * shift);
                else
                    uni_vmovups(ptr[reg_to + offset], Vmm(idx));
            else if (is_bf16())
                bf16_injector_->cvt_f32_to_bf16_store(
                        i_step, idx, reg_to, true, offset);
            else
                movss(ptr[reg_to + offset], Xmm(idx));
        }
    }

    ~jit_uni_relu_kernel_float() {
        delete bf16_injector_;
        delete bf16_emu_;
    }

    jit_uni_relu_kernel_float(const eltwise_desc_t &desc)
        : jit_uni_eltwise_kernel(desc)
        , jit_generator()
        , bf16_injector_(nullptr)
        , bf16_emu_(nullptr) {
        assert(desc.alg_kind == alg_kind::eltwise_relu);
        assert(utils::one_of(isa, sse41, avx2, avx512_common, avx512_core));

        Reg64 param = abi_param1;

        if (is_bf16()) {
            if (!mayiuse(avx512_core_bf16))
                bf16_emu_ = new bf16_emulation_t(this, bf16_emu_reserv_1,
                        bf16_emu_reserv_2, bf16_emu_reserv_3, bf16_emu_scratch,
                        bf16_emu_reserv_4);
            bf16_injector_ = new jit_bf16_eltwise_injector(this, zmm_idx,
                    k_mask_cvt, k_tail_mask, k_full_mask, bf16_emu_);
        }

        preamble();

        if (is_bf16()) {
            bf16_injector_->load_idx_table();
            bf16_injector_->prepare_cvt_mask();
            if (!mayiuse(avx512_core_bf16)) bf16_emu_->init_vcvtneps2bf16();
        }

        mov(reg_from, ptr[param + GET_OFF(from)]);
        if (is_bwd())
            mov(reg_for_comparison, ptr[param + GET_OFF(for_comparison)]);
        mov(reg_to, ptr[param + GET_OFF(to)]);
        mov(reg_work_amount, ptr[param + GET_OFF(work_amount)]);

        mov(imm_addr64, float2int(desc.alpha));
        uni_vmovq(xmm_ns, imm_addr64);
        uni_vbroadcastss(vmm_ns, xmm_ns);

        uni_vpxor(vmm_zero, vmm_zero, vmm_zero);

        const int loop_dec_unr[] = {simd_w(), simd_w(), simd_w(), 1};
        const int uf_unr[] = {4, 2, 1, 1};
        const int shift_unr[] = {vlen(), vlen(), vlen(), dtype_size()};
        const bool loop_vectorize_unr[] = {true, true, true, false};

        const int loop_dec_def[] = {simd_w(), 1};
        const int uf_def[] = {1, 1};
        const int shift_def[] = {vlen(), dtype_size()};
        const bool loop_vectorize_def[] = {true, false};

        const int mid = is_bf16() ? 4 : 2;
        const int *loop_dec = is_bf16() ? loop_dec_unr : loop_dec_def;
        const int *uf = is_bf16() ? uf_unr : uf_def;
        const int *shift = is_bf16() ? shift_unr : shift_def;
        const bool *loop_vectorize
                = is_bf16() ? loop_vectorize_unr : loop_vectorize_def;

        Label loop_label[5];

        for (int id = 0; id < mid; id++) {
            L(loop_label[id]);
            cmp(reg_work_amount, uf[id] * loop_dec[id] - 1);
            jle(loop_label[id + 1], T_NEAR);

            compute_step(loop_vectorize[id], uf[id], shift[id]);

            add(reg_from, uf[id] * shift[id]);
            add(reg_to, uf[id] * shift[id]);
            if (is_bwd()) add(reg_for_comparison, uf[id] * shift[id]);

            sub(reg_work_amount, uf[id] * loop_dec[id]);
            jmp(loop_label[id]);
        }

        L(loop_label[mid]);
        postamble();

        if (is_bf16()) bf16_injector_->write_idx_table();

        ker_ = (decltype(ker_))this->getCode();
    }

private:
    using Vmm = typename utils::conditional3<isa == sse41, Xmm, isa == avx2,
            Ymm, Zmm>::type;
    using opmask_t = const Xbyak::Opmask;

    int vlen() {
        int vlen = cpu_isa_traits<isa>::vlen;
        return is_bf16() ? vlen / 2 : vlen;
    }
    int simd_w() { return vlen() / dtype_size(); }

    Reg64 reg_from = rax;
    Reg64 reg_for_comparison = is_bwd() ? rdx : reg_from;
    Reg64 reg_to = r8;
    Reg64 reg_work_amount = rsi;
    Reg64 imm_addr64 = rbx;

    Xmm xmm_ns = Xmm(14);

    Vmm vmm_ns = Vmm(utils::one_of(isa, avx512_common, avx512_core) ? 28 : 14);
    Vmm vmm_zero
            = Vmm(utils::one_of(isa, avx512_common, avx512_core) ? 29 : 15);
    Vmm vmm_mask
            = Vmm(utils::one_of(isa, avx512_common, avx512_core) ? 30 : 12);

    opmask_t k_mask = k1;

    /* bf16 support */
    Zmm bf16_emu_reserv_1 = Zmm(24);
    Zmm bf16_emu_reserv_2 = Zmm(25);
    Zmm bf16_emu_reserv_3 = Zmm(26);
    Reg64 bf16_emu_scratch = r14;
    Zmm bf16_emu_reserv_4 = Zmm(27);

    Zmm zmm_idx = Zmm(31);
    opmask_t k_mask_cvt = k7;
    opmask_t k_tail_mask = k6;
    opmask_t k_full_mask = k5;

    jit_bf16_eltwise_injector *bf16_injector_;
    bf16_emulation_t *bf16_emu_;
};

template <cpu_isa_t isa>
struct jit_uni_relu_kernel_int : public jit_uni_eltwise_kernel,
                                 public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_relu_kernel_int)

    jit_uni_relu_kernel_int(const eltwise_desc_t &desc)
        : jit_uni_eltwise_kernel(desc), jit_generator() {
        using namespace data_type;

        // Relu for int types: s32, s8; Only forward direction
        assert(desc.alg_kind == alg_kind::eltwise_relu);
        assert(utils::one_of(data_type(), s32, s8));
        assert(!is_bwd());
        assert(utils::one_of(isa, sse41, avx2, avx512_common));

        Reg64 param = abi_param1;

        // f32 used for processing of any data type
        // thus we need to take into account size of f32
        const size_t proc_dt_size = sizeof(typename prec_traits<f32>::type);
        const size_t vlen = cpu_isa_traits<isa>::vlen;
        const size_t simd_w = vlen / proc_dt_size;
        const size_t loop_dec[] = {simd_w, 1};
        const size_t uf[] = {1, 1};
        const size_t shift[]
                = {dtype_size() * (vlen / proc_dt_size), (size_t)dtype_size()};
        const bool loop_vectorize[] = {true, false};

        preamble();

        mov(reg_from, ptr[param + GET_OFF(from)]);
        mov(reg_to, ptr[param + GET_OFF(to)]);
        mov(reg_work_amount, ptr[param + GET_OFF(work_amount)]);

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

    bool is32bit() const {
        return utils::one_of(data_type(), data_type::s32, data_type::f32);
    }

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
    };

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

template <cpu_isa_t isa>
struct jit_uni_kernel_fwd : public jit_uni_eltwise_kernel,
                            public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_kernel_fwd)

    jit_uni_kernel_fwd(const eltwise_desc_t &desc)
        : jit_uni_eltwise_kernel(desc)
        , jit_generator()
        , bf16_injector_(nullptr)
        , bf16_emu_(nullptr) {
        if (is_bf16()) {
            if (!mayiuse(avx512_core_bf16))
                bf16_emu_ = new bf16_emulation_t(this, bf16_emu_reserv_1,
                        bf16_emu_reserv_2, bf16_emu_reserv_3, bf16_emu_scratch,
                        bf16_emu_reserv_5);
            bf16_injector_ = new jit_bf16_eltwise_injector(this, zmm_idx,
                    k_mask_cvt, k_tail_mask, k_full_mask, bf16_emu_);
        }

        eltwise_injector_
                = new jit_uni_eltwise_injector_f32<isa>(this, desc.alg_kind,
                        desc.alpha, desc.beta, 1.f, false, r9, Opmask(1));

        preamble();

        if (is_bf16()) {
            bf16_injector_->load_idx_table();
            bf16_injector_->prepare_cvt_mask();
            if (!mayiuse(avx512_core_bf16)) bf16_emu_->init_vcvtneps2bf16();
        }

        Reg64 param = abi_param1;
        mov(reg_from, ptr[param + GET_OFF(from)]);
        mov(reg_to, ptr[param + GET_OFF(to)]);
        mov(reg_work_amount, ptr[param + GET_OFF(work_amount)]);
        eltwise_injector_->load_table_addr();

        Label reminder_loop_start, reminder_loop_end;
        Label vectorized_loop_start, vectorized_loop_end;

        cmp(reg_work_amount, simd_w());
        jl(reminder_loop_start, T_NEAR);

        L(vectorized_loop_start);

        // TODO: consider improving.
        // This piece of code is responsible for the preserve_zero function
        // being a natural restriction of this implementation. It works with any
        // dense and blocked layout, but the problem raises when blocking
        // dimension is not divisible by block size. For such case, the code
        // below should save the mask, where zero padding should be preserved
        // and apply it on register before storing into dst memory. Until
        // there's a restriction on certain blocked layouts, when this behavior
        // can be relevantly easy controlled, this will cost much from code
        // perspective and will complicate the compute logic significantly.
        if (is_bf16()) {
            bf16_injector_->load_bf16_cvt_to_f32(vmm_src.getIdx(), reg_from);
            eltwise_injector_->compute_vector(vmm_src.getIdx());
            bf16_injector_->cvt_f32_to_bf16_store(1, vmm_src.getIdx(), reg_to);
        } else {
            uni_vmovups(vmm_src, ptr[reg_from]);
            eltwise_injector_->compute_vector(vmm_src.getIdx());
            uni_vmovups(ptr[reg_to], vmm_src);
        }
        auto shift = vlen();
        add(reg_from, shift);
        add(reg_to, shift);

        sub(reg_work_amount, simd_w());
        cmp(reg_work_amount, simd_w());
        jge(vectorized_loop_start, T_NEAR);

        L(vectorized_loop_end);

        L(reminder_loop_start);

        cmp(reg_work_amount, 0);
        jle(reminder_loop_end, T_NEAR);
        if (is_bf16()) {
            bf16_injector_->load_bf16_cvt_to_f32(
                    vmm_src.getIdx(), reg_from, true);
            eltwise_injector_->compute_vector(vmm_src.getIdx());
            bf16_injector_->cvt_f32_to_bf16_store(
                    1, vmm_src.getIdx(), reg_to, true);
        } else {
            movss(xmm_src, ptr[reg_from]);
            eltwise_injector_->compute_vector(xmm_src.getIdx());
            movss(ptr[reg_to], xmm_src);
        }
        add(reg_from, dtype_size());
        add(reg_to, dtype_size());

        dec(reg_work_amount);
        jmp(reminder_loop_start, T_NEAR);

        L(reminder_loop_end);

        postamble();

        if (is_bf16()) bf16_injector_->write_idx_table();

        eltwise_injector_->prepare_table();

        ker_ = (decltype(ker_))this->getCode();
    }

    ~jit_uni_kernel_fwd() {
        delete eltwise_injector_;
        delete bf16_injector_;
        delete bf16_emu_;
    }

private:
    using Vmm = typename utils::conditional3<isa == sse41, Xmm, isa == avx2,
            Ymm, Zmm>::type;
    using opmask_t = const Xbyak::Opmask;

    int vlen() {
        int vlen = cpu_isa_traits<isa>::vlen;
        return is_bf16() ? vlen / 2 : vlen;
    }
    int simd_w() { return vlen() / dtype_size(); }

    Reg64 reg_from = rax;
    Reg64 reg_to = r8;
    Reg64 reg_work_amount = rsi;
    Reg64 imm_addr64 = rbx;

    Xmm xmm_src = Xmm(1);
    Vmm vmm_src = Vmm(1);
    jit_uni_eltwise_injector_f32<isa> *eltwise_injector_;

    /* bf16 support */
    Zmm bf16_emu_reserv_1 = Zmm(26);
    Zmm bf16_emu_reserv_2 = Zmm(27);
    Zmm bf16_emu_reserv_3 = Zmm(28);
    Reg64 bf16_emu_scratch = r14;
    Zmm bf16_emu_reserv_5 = Zmm(29);

    Zmm zmm_idx = Zmm(31);
    opmask_t k_mask_cvt = k7;
    opmask_t k_tail_mask = k6;
    opmask_t k_full_mask = k5;

    jit_bf16_eltwise_injector *bf16_injector_;
    bf16_emulation_t *bf16_emu_;
};

} /* namespace */

template <cpu_isa_t isa, data_type_t d_type>
status_t jit_uni_eltwise_fwd_t<isa, d_type>::pd_t::init() {
    using namespace alg_kind;
    using namespace data_type;

    const auto alg = desc()->alg_kind;
    // relu supports bf16, f32, s32 and s8
    bool relu_ok
            = alg == eltwise_relu && utils::one_of(d_type, bf16, f32, s32, s8);

    // others supports bf16 and f32
    bool non_relu_ok
            = utils::one_of(alg, eltwise_tanh, eltwise_elu, eltwise_square,
                      eltwise_abs, eltwise_sqrt, eltwise_linear,
                      eltwise_bounded_relu, eltwise_soft_relu, eltwise_logistic,
                      eltwise_exp, eltwise_gelu_tanh, eltwise_swish,
                      eltwise_log, eltwise_clip, eltwise_pow, eltwise_gelu_erf,
                      eltwise_relu_use_dst_for_bwd,
                      eltwise_tanh_use_dst_for_bwd, eltwise_elu_use_dst_for_bwd,
                      eltwise_sqrt_use_dst_for_bwd,
                      eltwise_logistic_use_dst_for_bwd,
                      eltwise_exp_use_dst_for_bwd)
            && utils::one_of(d_type, bf16, f32);

    bool ok = true && mayiuse(isa) && is_fwd()
            && desc()->data_desc.data_type == d_type
            && IMPLICATION(
                    desc()->data_desc.data_type == bf16, mayiuse(avx512_core))
            && utils::one_of(true, relu_ok, non_relu_ok)
            && !has_zero_dim_memory()
            && memory_desc_wrapper(src_md()).is_dense(true)
            // refer to a comment in jit_uni_kernel_fwd why this is needed
            && IMPLICATION(!memory_desc_wrapper(src_md()).is_dense(false),
                    is_zero_preserved())
            && attr()->has_default_values();

    return ok ? status::success : status::unimplemented;
}

template <cpu_isa_t isa, data_type_t d_type>
jit_uni_eltwise_fwd_t<isa, d_type>::jit_uni_eltwise_fwd_t(const pd_t *apd)
    : primitive_impl_t(apd), kernel_(nullptr) {
    const auto &desc = *pd()->desc();
    switch (desc.alg_kind) {
        case alg_kind::eltwise_relu:
            if (utils::one_of(d_type, data_type::s32, data_type::s8))
                kernel_ = new jit_uni_relu_kernel_int<isa>(desc);
            else
                kernel_ = new jit_uni_relu_kernel_float<isa>(desc);
            break;
        default: kernel_ = new jit_uni_kernel_fwd<isa>(desc);
    }
}

template <cpu_isa_t isa, data_type_t d_type>
jit_uni_eltwise_fwd_t<isa, d_type>::~jit_uni_eltwise_fwd_t() {
    delete kernel_;
}

template <cpu_isa_t isa, impl::data_type_t d_type>
void jit_uni_eltwise_fwd_t<isa, d_type>::execute_forward(
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

template <cpu_isa_t isa, data_type_t d_type>
status_t jit_uni_eltwise_bwd_t<isa, d_type>::pd_t::init() {
    bool ok = true && !is_fwd()
            && utils::one_of(desc()->alg_kind, alg_kind::eltwise_relu)
            && src_md()->data_type == d_type
            && IMPLICATION(desc()->data_desc.data_type == data_type::bf16,
                    mayiuse(avx512_core))
            && !has_zero_dim_memory() && mayiuse(isa)
            && set_default_formats_common()
            && memory_desc_wrapper(src_md()).is_dense()
            && memory_desc_wrapper(diff_dst_md())
                    == memory_desc_wrapper(src_md())
            && attr()->has_default_values();

    return ok ? status::success : status::unimplemented;
}

template <cpu_isa_t isa, data_type_t d_type>
jit_uni_eltwise_bwd_t<isa, d_type>::jit_uni_eltwise_bwd_t(const pd_t *apd)
    : primitive_impl_t(apd), kernel_(nullptr) {
    const auto &desc = *pd()->desc();
    switch (desc.alg_kind) {
        case alg_kind::eltwise_relu:
            kernel_ = new jit_uni_relu_kernel_float<isa>(desc);
            break;
        default: assert(!"unknown eltwise alg_kind");
    }
}

template <cpu_isa_t isa, data_type_t d_type>
jit_uni_eltwise_bwd_t<isa, d_type>::~jit_uni_eltwise_bwd_t() {
    delete kernel_;
}

template <cpu_isa_t isa, data_type_t d_type>
void jit_uni_eltwise_bwd_t<isa, d_type>::execute_backward(
        const exec_ctx_t &ctx) const {
    auto src = CTX_IN_MEM(const data_t *, DNNL_ARG_SRC);
    auto diff_dst = CTX_IN_MEM(const data_t *, DNNL_ARG_DIFF_DST);
    auto diff_src = CTX_OUT_MEM(data_t *, DNNL_ARG_DIFF_SRC);

    const memory_desc_wrapper data_d(pd()->src_md());
    const memory_desc_wrapper diff_data_d(pd()->diff_src_md());

    const size_t nelems = data_d.nelems();

    src += data_d.offset0();
    diff_dst += diff_data_d.offset0();
    diff_src += diff_data_d.offset0();
    const int cache_line = 64 / data_d.data_type_size();

    parallel(0, [&](const int ithr, const int nthr) {
        size_t start {0}, end {0};

        balance211(utils::div_up(nelems, cache_line), nthr, ithr, start, end);
        start = nstl::min(nelems, start * cache_line);
        end = nstl::min(nelems, end * cache_line);

        auto arg = jit_args();
        arg.from = (const void *)&diff_dst[start];
        arg.to = (const void *)&diff_src[start];
        arg.for_comparison = (const void *)&src[start];
        arg.work_amount = end - start;
        if (arg.work_amount) (*kernel_)(&arg);
    });
}

template struct jit_uni_eltwise_fwd_t<sse41, data_type::f32>;
template struct jit_uni_eltwise_fwd_t<sse41, data_type::s32>;
template struct jit_uni_eltwise_fwd_t<sse41, data_type::s8>;
template struct jit_uni_eltwise_bwd_t<sse41, data_type::f32>;
template struct jit_uni_eltwise_fwd_t<avx2, data_type::f32>;
template struct jit_uni_eltwise_fwd_t<avx2, data_type::s32>;
template struct jit_uni_eltwise_fwd_t<avx2, data_type::s8>;
template struct jit_uni_eltwise_bwd_t<avx2, data_type::f32>;
template struct jit_uni_eltwise_fwd_t<avx512_common, data_type::f32>;
template struct jit_uni_eltwise_fwd_t<avx512_core, data_type::bf16>;
template struct jit_uni_eltwise_fwd_t<avx512_common, data_type::s32>;
template struct jit_uni_eltwise_fwd_t<avx512_common, data_type::s8>;
template struct jit_uni_eltwise_bwd_t<avx512_common, data_type::f32>;
template struct jit_uni_eltwise_bwd_t<avx512_core, data_type::bf16>;

} // namespace cpu
} // namespace impl
} // namespace dnnl
