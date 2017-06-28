/*******************************************************************************
* Copyright 2017 Intel Corporation
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

#include "mkldnn_types.h"
#include "mkldnn_thread.hpp"
#include "nstl.hpp"
#include "utils.hpp"
#include "jit_generator.hpp"

#include "jit_uni_eltwise.hpp"

#define GET_OFF(field) offsetof(jit_args, field)

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace Xbyak;

struct jit_args {
    const float *from;
    const float *for_comparison;
    const float *to;
    size_t work_amount;
};

struct jit_uni_eltwise_kernel_f32 : public c_compatible {
    const eltwise_desc_t &desc_;

    void (*ker_)(const jit_args *);
    void operator()(const jit_args *args) { assert(ker_); ker_(args); }

    jit_uni_eltwise_kernel_f32(const eltwise_desc_t &desc)
        : desc_(desc), ker_(nullptr) {}
    virtual ~jit_uni_eltwise_kernel_f32() {}

protected:
    bool is_bwd() const { return desc_.prop_kind == prop_kind::backward_data; }
};

/* jit kernels */
namespace {
template <cpu_isa_t isa>
struct jit_uni_relu_kernel_f32 : public jit_uni_eltwise_kernel_f32,
    public jit_generator
{
    void compute_step(bool vectorize, const int uf, const int shift) {
        unsigned char _cmp_gt_os = isa == avx512_common ? 14 : 6;

        for (int i = 0; i < uf; i++) {
            if (vectorize) {
                uni_vmovups(Vmm(i + 1), ptr[reg_from + i * shift]);
                if (is_bwd())
                    uni_vmovups(Vmm(uf + i + 1),
                                ptr[reg_for_comparison + i * shift]);
            } else {
                movss(Xmm(i + 1), ptr[reg_from + i * shift]);
                if (is_bwd())
                    movss(Xmm(uf + i + 1),
                          ptr[reg_for_comparison + i * shift]);
            }
        }

        if (isa == sse42) {
            for (int i = 0; i < uf; i++) {
                movups(Vmm(2 * uf + i + 1), Vmm(i + 1));
                mulps(Vmm(2 * uf + i + 1), vmm_ns);

                Vmm mask = Vmm(0);
                if (is_bwd()) {
                    movups(mask, Vmm(uf + i + 1));
                    cmpps(mask, vmm_zero, _cmp_gt_os);
                } else {
                    movups(mask, Vmm(i + 1));
                    cmpps(mask, vmm_zero, _cmp_gt_os);
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
                        vcmpps(k_mask, Vmm(uf + i + 1), vmm_zero, _cmp_gt_os);
                    else
                        vcmpps(k_mask, Vmm(i + 1), vmm_zero, _cmp_gt_os);
                    vblendmps(Vmm(2 * uf + i + 1) | k_mask, Vmm(2 * uf + i + 1),
                              Vmm(i + 1));
                }
            }
        }

        for (int i = 0; i < uf; i++) {
            if (vectorize) {
                uni_vmovups(ptr[reg_to + i * shift], Vmm(2 * uf + i + 1));
            } else {
                movss(ptr[reg_to + i * shift], Xmm(2 * uf + i + 1));
            }
        }
    }

    jit_uni_relu_kernel_f32(const eltwise_desc_t &desc)
        : jit_uni_eltwise_kernel_f32(desc), jit_generator()
    {
        assert(desc.alg_kind == alg_kind::eltwise_relu);
        assert(isa == sse42 || isa == avx2 || isa == avx512_common);

        Reg64 param = abi_param1;

        const int simd_w = cpu_isa_traits<isa>::vlen / sizeof(float);
        const int loop_dec[] = {simd_w, 1};
        const int uf[] = {1, 1};
        const int shift[] = {cpu_isa_traits<isa>::vlen, sizeof(float)};
        const bool loop_vectorize[] = {true, false};

        this->preamble();

        mov(reg_from, ptr[param + GET_OFF(from)]);
        if (is_bwd())
            mov(reg_for_comparison, ptr[param + GET_OFF(for_comparison)]);
        mov(reg_to, ptr[param + GET_OFF(to)]);
        mov(reg_work_amount, ptr[param + GET_OFF(work_amount)]);

        mov(imm_addr64, float2int(desc.alpha));
        movq(xmm_ns, imm_addr64);
        uni_vbroadcastss(vmm_ns, xmm_ns);

        uni_vpxor(vmm_zero, vmm_zero, vmm_zero);

        Label loop_label[3];

        for (int id = 0; id < 2; id++) {
            L(loop_label[id]);
            cmp(reg_work_amount, uf[id] * loop_dec[id] - 1);
            jle(loop_label[id + 1], T_NEAR);

            compute_step(loop_vectorize[id], uf[id], shift[id]);

            add(reg_from, uf[id] * shift[id]);
            add(reg_to, uf[id] * shift[id]);
            if (is_bwd())
                add(reg_for_comparison, uf[id] * shift[id]);

            sub(reg_work_amount, uf[id] * loop_dec[id]);
            jmp(loop_label[id]);
        }

        L(loop_label[2]);
        this->postamble();

        ker_ = (decltype(ker_))this->getCode();
    }

private:
    using Vmm = typename utils::conditional3<isa == sse42, Xmm,
                                             isa == avx2, Ymm, Zmm>::type;

    Reg64 reg_from = rax;
    Reg64 reg_for_comparison = is_bwd() ? rdx : reg_from;
    Reg64 reg_to = r8;
    Reg64 reg_work_amount = rsi;
    Reg64 imm_addr64 = rbx;

    Xmm xmm_ns = Xmm(14);

    Vmm vmm_ns = Vmm(isa == avx512_common ? 30 : 14);
    Vmm vmm_zero = Vmm(isa == avx512_common ? 31 : 15);

    Vmm vmm_mask = Vmm(isa == avx512_common ? 28 : 12);
    Opmask k_mask = Opmask(1);
};

template <cpu_isa_t isa>
struct jit_uni_elu_kernel_f32 : public jit_uni_eltwise_kernel_f32,
    public jit_generator
{
    void prepare_table() {
        const unsigned int cvals[] = {
            0x3f800000, // 0 1.0f
            0x3f000000, // 1 0.5f
            0x42b0c0a5, // 2 max logf = 88.3762589f
            0xc2b0c0a5, // 3 min logf = -88.3762589f
            0x3fb8aa3b, // 4 log2ef = 1.44269502f
            0x3f318000, // 5 c1 = 0.693359375f
            0xb95e8083, // 6 c2 = -0.000212194442f
            0x39506967, // 7 p0 = 0.000198756912f
            0x3ab743ce, // 8 p1 = 0.00139819994f
            0x3c088908, // 9 p2 = 0.008333405205f
            0x3d2aa9c1, // 10 p3 = 0.0416657962f
            0x3e2aaaaa, // 11 p4 = 0.166666657f
            0x0000007f, // 12 0x7f
        };

        align(64);
        L(l_table);
        for (size_t i = 0; i < sizeof(cvals) / sizeof(cvals[0]); ++i) {
            for (size_t d = 0; d < vlen / sizeof(float); ++d) {
                dd(cvals[i]);
            }
        }
    }

    void vectorized_expf() {
        uni_vmovups(Vmm(8), Vmm(1));
        uni_vminps(Vmm(8), Vmm(8), ptr[imm_addr64 + 2 * vlen]);
        uni_vmaxps(Vmm(8), Vmm(8), ptr[imm_addr64 + 3 * vlen]);
        uni_vmovups(Vmm(4), Vmm(8));

        // fx = x * log2ef + 0.5
        uni_vmulps(Vmm(4), Vmm(4), ptr[imm_addr64 + 4 * vlen]);

        if (isa < avx512_common) {
            uni_vaddps(Vmm(4), Vmm(4), ptr[imm_addr64 + 1 * vlen]);
            // tmp = floorf(fx)
            uni_vroundps(Vmm(5), Vmm(4), _op_floor);

            uni_vmovups(vmm_tmp2, Vmm(5));
            uni_vcmpgtps(vmm_tmp2, vmm_tmp2, Vmm(4));
            uni_vandps(vmm_tmp2, vmm_tmp2, ptr[imm_addr64 + 0 * vlen]);
        } else {
            uni_vaddps(Vmm(4), Vmm(4), ptr[imm_addr64 + 1 * vlen]);
            vcvtps2dq(Vmm(5) | T_rd_sae, Vmm(4));
            vcvtdq2ps(Vmm(5), Vmm(5));

            vcmpps(k_mask_tmp, Vmm(5), Vmm(4), _cmp_gt_os);
            vmovups(vmm_tmp2 | k_mask_tmp | T_z, zword[imm_addr64 + 0 * vlen]);
        }

        // fx = fx - 1 (if there are fraction bits)
        uni_vsubps(Vmm(5), Vmm(5), vmm_tmp2);
        // keep fx for further computations
        uni_vmovups(Vmm(4), Vmm(5));
        uni_vmovups(Vmm(6), Vmm(5));

        uni_vmulps(Vmm(4), Vmm(4), ptr[imm_addr64 + 5 * vlen]);
        uni_vmulps(Vmm(6), Vmm(6), ptr[imm_addr64 + 6 * vlen]);
        // x = x - fx * c1
        uni_vsubps(Vmm(8), Vmm(8), Vmm(4));
        // x = x - fx * c2
        uni_vsubps(Vmm(8), Vmm(8), Vmm(6));

        // keep x for further computations
        uni_vmovups(Vmm(4), Vmm(8));
        // x2 = x * x
        uni_vmulps(Vmm(4), Vmm(4), Vmm(4));

        // y = p0
        uni_vmovups(Vmm(2 + 1), ptr[imm_addr64 + 7 * vlen]);
        // y = y * x + p1
        uni_vfmadd213ps(Vmm(2 + 1), Vmm(8), ptr[imm_addr64 + 8 * vlen]);
        // y = y * x + p2
        uni_vfmadd213ps(Vmm(2 + 1), Vmm(8), ptr[imm_addr64 + 9 * vlen]);
        // y = y * x + p3
        uni_vfmadd213ps(Vmm(2 + 1), Vmm(8), ptr[imm_addr64 + 10 * vlen]);
        // y = y * x + p4
        uni_vfmadd213ps(Vmm(2 + 1), Vmm(8), ptr[imm_addr64 + 11 * vlen]);
        // y = y * x + p5
        uni_vfmadd213ps(Vmm(2 + 1), Vmm(8), ptr[imm_addr64 + 1 * vlen]);
        // y = y * x2 + x
        uni_vfmadd213ps(Vmm(2 + 1), Vmm(4), Vmm(8));
        // y = y + 1
        uni_vaddps(Vmm(2 + 1), Vmm(2 + 1), ptr[imm_addr64 + 0 * vlen]);
        // compute 2^n
        uni_vcvtps2dq(Vmm(6), Vmm(5));
        uni_vpaddd(Vmm(6), Vmm(6), ptr[imm_addr64 + 12 * vlen]);
        uni_vpslld(Vmm(6), Vmm(6), 23);
        // y = y * 2^n
        uni_vmulps(Vmm(2 + 1), Vmm(2 + 1), Vmm(6));
    }

    void vectorized_loop_body() {
        uni_vmovups(Vmm(1), ptr[reg_from]);
        // compute mask
        if (isa < avx512_common) {
            uni_vmovups(vmm_mask, Vmm(1));
            uni_vcmpgtps(vmm_mask, vmm_mask, vmm_zero);
            // early exit if all elems positive
            uni_vmovmskps(reg_mask, vmm_mask);
        } else {
            vcmpps(k_mask, Vmm(1), vmm_zero, _cmp_gt_os);
            kmovw(reg_mask.cvt32(), k_mask);
        }
        cmp(reg_mask, isa == sse42 ? 0x0f : (isa == avx2 ? 0xff : 0xffff));
        je("early_exit", T_NEAR);

        // compute exponent
        vectorized_expf();

        // alpha * (exp(x) - 1)
        uni_vsubps(Vmm(2 + 1), Vmm(2 + 1), ptr[imm_addr64 + 0 * vlen]);
        uni_vmulps(Vmm(2 + 1), Vmm(2 + 1), vmm_alpha);
        // combine with mask
        if (isa < avx512_common)
            uni_vblendvps(Vmm(2 + 1), Vmm(2 + 1), Vmm(1), vmm_mask);
        else
            vblendmps(Vmm(2 + 1) | k_mask, Vmm(2 + 1), Vmm(1));
        // store result
        uni_vmovups(ptr[reg_to], Vmm(2 + 1));

        jmp("exit", T_NEAR);

        L("early_exit");
        uni_vmovups(ptr[reg_to], Vmm(1));

        L("exit");
    }

    void reminder_loop_body() {
        const unsigned int _cmp_gt_os_sse = 6;

        movss(Xmm(1), ptr[reg_from]);
        // compute mask
        movss(xmm_mask, Xmm(1));
        cmpss(xmm_mask, xmm_zero, _cmp_gt_os_sse);

        // early exit if all elems positive
        movmskps(reg_mask, xmm_mask);
        cmp(reg_mask, 0x01);
        je("reminder_early_exit", T_NEAR);

        // compute exponent
        movaps(Xmm(8), Xmm(1));
        minss(Xmm(8), ptr[imm_addr64 + 2 * vlen]);
        maxss(Xmm(8), ptr[imm_addr64 + 3 * vlen]);
        movaps(Xmm(4), Xmm(8));

        // fx = x * log2ef + 0.5
        mulss(Xmm(4), ptr[imm_addr64 + 4 * vlen]);
        addss(Xmm(4), ptr[imm_addr64 + 1 * vlen]);

        // tmp = floorf(fx)
        roundss(Xmm(5), Xmm(4), _op_floor);
        movaps(xmm_tmp2, Xmm(5));

        cmpss(xmm_tmp2, Xmm(4), _cmp_gt_os_sse);
        andps(xmm_tmp2, ptr[imm_addr64 + 0 * vlen]);
        // fx = fx - 1 (if there are fraction bits)
        subss(Xmm(5), xmm_tmp2);
        // keep fx for further computations
        movaps(Xmm(4), Xmm(5));
        movaps(Xmm(6), Xmm(5));

        mulss(Xmm(4), ptr[imm_addr64 + 5 * vlen]);
        mulss(Xmm(6), ptr[imm_addr64 + 6 * vlen]);
        // x = x - fx * c1
        subss(Xmm(8), Xmm(4));
        // x = x - fx * c2
        subss(Xmm(8), Xmm(6));
        // keep x for further computations
        movaps(Xmm(4), Xmm(8));
        // x2 = x * x
        mulss(Xmm(4), Xmm(4));

        // y = p0
        movups(xmm_tmp1, ptr[imm_addr64 + 7 * vlen]);
        // y = y * x
        mulss(xmm_tmp1, Xmm(8));
        // y = y + p1
        addss(xmm_tmp1, ptr[imm_addr64 + 8 * vlen]);
        // y = y * x
        mulss(xmm_tmp1, Xmm(8));
        // y = y + p2
        addss(xmm_tmp1, ptr[imm_addr64 + 9 * vlen]);
        // y = y * x
        mulss(xmm_tmp1, Xmm(8));
        // y = y + p3
        addss(xmm_tmp1, ptr[imm_addr64 + 10 * vlen]);
        // y = y * x
        mulss(xmm_tmp1, Xmm(8));
        // y = y + p4
        addss(xmm_tmp1, ptr[imm_addr64 + 11 * vlen]);
        // y = y * x
        mulss(xmm_tmp1, Xmm(8));
        // y = y + p5
        addss(xmm_tmp1, ptr[imm_addr64 + 1 * vlen]);
        // y = y * x2
        mulss(xmm_tmp1, Xmm(4));
        // y = y + x
        addss(xmm_tmp1, Xmm(8));
        // y = y + 1
        addss(xmm_tmp1, ptr[imm_addr64 + 0 * vlen]);
        // compute 2^n
        cvtps2dq(Xmm(6), Xmm(5));
        paddd(Xmm(6), ptr[imm_addr64 + 12 * vlen]);
        pslld(Xmm(6), 23);
        // y = y * 2^n
        mulss(xmm_tmp1, Xmm(6));
        // put result to output register
        movaps(Xmm(2 + 1), xmm_tmp1);
        // alpha * (exp(x) - 1)
        subss(Xmm(2 + 1), ptr[imm_addr64 + 0 * vlen]);
        mulss(Xmm(2 + 1), xmm_alpha);
        // combine with mask (in xmm0)
        blendvps(Xmm(2 + 1), Xmm(1));
        // store result
        movss(ptr[reg_to], Xmm(2 + 1));
        jmp("reminder_exit", T_NEAR);

        L("reminder_early_exit");
        movss(ptr[reg_to], Xmm(1));

        L("reminder_exit");
    }

    jit_uni_elu_kernel_f32(const eltwise_desc_t &desc)
        : jit_uni_eltwise_kernel_f32(desc), jit_generator()
    {
        assert(desc.alg_kind == alg_kind::eltwise_elu);
        assert(is_bwd() == false);

        this->preamble();

        mov(reg_from, ptr[abi_param1 + GET_OFF(from)]);
        mov(reg_to, ptr[abi_param1 + GET_OFF(to)]);
        mov(reg_work_amount, ptr[abi_param1 + GET_OFF(work_amount)]);

        mov(imm_addr64, float2int(desc.alpha));
        movq(xmm_alpha, imm_addr64);
        uni_vbroadcastss(vmm_alpha, xmm_alpha);

        uni_vpxor(vmm_zero, vmm_zero, vmm_zero);
        mov(imm_addr64, l_table);

        cmp(reg_work_amount, simd_w);
        jl("reminder_loop_start", T_NEAR);

        L("vectorized_loop_start");

        vectorized_loop_body();

        add(reg_from, vlen);
        add(reg_to, vlen);

        sub(reg_work_amount, simd_w);
        cmp(reg_work_amount, simd_w);
        jge("vectorized_loop_start", T_NEAR);

        L("vectorized_loop_end");

        L("reminder_loop_start");

        cmp(reg_work_amount, 0);
        jle("reminder_loop_end", T_NEAR);

        reminder_loop_body();

        add(reg_from, sizeof(float));
        add(reg_to, sizeof(float));

        dec(reg_work_amount);
        jmp("reminder_loop_start", T_NEAR);

        L("reminder_loop_end");

        this->postamble();

        prepare_table();

        ker_ = (decltype(ker_))this->getCode();
    }

private:
    using Vmm = typename utils::conditional3<isa == sse42, Xmm,
          isa == avx2, Ymm, Zmm>::type;

    const int simd_w = cpu_isa_traits<isa>::vlen / sizeof(float);
    const int vlen = cpu_isa_traits<isa>::vlen;

    unsigned char _cmp_gt_os = isa == avx512_common ? 14 : 6;
    unsigned char _op_floor = 1;

    Reg64 reg_from = rax;
    Reg64 reg_to = r8;
    Reg64 reg_work_amount = rsi;
    Reg64 imm_addr64 = rbx;
    Reg64 reg_mask = r9;

    Xmm xmm_mask = Xmm(0);
    Vmm vmm_mask = Vmm(0);

    Opmask k_mask = Opmask(1);
    Opmask k_mask_tmp = Opmask(2);

    Xmm xmm_tmp1 = Xmm(12);
    Xmm xmm_tmp2 = Xmm(13);
    Vmm vmm_tmp1 = Vmm(12);
    Vmm vmm_tmp2 = Vmm(13);

    Xmm xmm_alpha = Xmm(14);
    Xmm xmm_zero = Xmm(15);
    Vmm vmm_alpha = Vmm(14);
    Vmm vmm_zero = Vmm(15);

    Label l_table;
};
}

template <cpu_isa_t isa>
status_t jit_uni_eltwise_fwd_t<isa>::pd_t::init()
{
    using namespace prop_kind;

    assert(engine()->kind() == engine_kind::cpu);
    bool ok = true && mayiuse(isa)
        && utils::one_of(desc()->prop_kind, forward_training,
                forward_inference)
        && utils::one_of(desc()->alg_kind, alg_kind::eltwise_relu,
                alg_kind::eltwise_elu)
        && utils::everyone_is(data_type::f32, desc()->data_desc.data_type)
        && memory_desc_wrapper(src_pd()).is_dense();

    return ok ? status::success : status::unimplemented;
}

template <cpu_isa_t isa>
jit_uni_eltwise_fwd_t<isa>::jit_uni_eltwise_fwd_t(const pd_t *pd,
        const input_vector &inputs, const output_vector &outputs)
    : cpu_primitive_t(&conf_, inputs, outputs), conf_(*pd), kernel_(nullptr)
{
    const auto &desc = *conf_.desc();
    switch (desc.alg_kind) {
    case alg_kind::eltwise_relu:
        kernel_ = new jit_uni_relu_kernel_f32<isa>(desc); break;
    case alg_kind::eltwise_elu:
        kernel_ = new jit_uni_elu_kernel_f32<isa>(desc); break;
    default: assert(!"unknown eltwise alg_kind");
    }
}

template <cpu_isa_t isa>
jit_uni_eltwise_fwd_t<isa>::~jit_uni_eltwise_fwd_t()
{ delete kernel_; }

template <cpu_isa_t isa>
void jit_uni_eltwise_fwd_t<isa>::execute_forward()
{
    auto src = reinterpret_cast<const data_t *>(this->input_memory(0));
    auto dst = reinterpret_cast<data_t *>(this->memory(0));

    const memory_desc_wrapper data_d(conf_.src_pd());

    const size_t nelems = data_d.nelems();

    src += data_d.blocking_desc().offset_padding;
    dst += data_d.blocking_desc().offset_padding;

    auto ker = [&](const int ithr, const int nthr) {
        size_t start{0}, end{0};

        const int cache_line = 16;

        balance211(utils::div_up(nelems, cache_line), nthr, ithr, start, end);
        start = nstl::min(nelems, start * cache_line);
        end = nstl::min(nelems, end * cache_line);

        jit_args arg = {};
        arg.from = &src[start];
        arg.for_comparison = &src[start];
        arg.to = &dst[start];
        arg.work_amount = end - start;
        if (arg.work_amount)
            (*kernel_)(&arg);
    };

#pragma omp parallel
    {
        ker(omp_get_thread_num(), omp_get_num_threads());
    }
}

template <cpu_isa_t isa>
status_t jit_uni_eltwise_bwd_t<isa>::pd_t::init()
{
    assert(engine()->kind() == engine_kind::cpu);

    bool ok = true
        && mayiuse(isa)
        && desc()->prop_kind == prop_kind::backward_data
        && utils::one_of(desc()->alg_kind, alg_kind::eltwise_relu)
        && src_pd()->desc()->data_type == data_type::f32
        && memory_desc_wrapper(src_pd()).is_dense()
        && memory_desc_wrapper(diff_dst_pd()) == memory_desc_wrapper(src_pd());

    return ok ? status::success : status::unimplemented;
}

template <cpu_isa_t isa>
jit_uni_eltwise_bwd_t<isa>::jit_uni_eltwise_bwd_t(const pd_t *pd,
        const input_vector &inputs, const output_vector &outputs)
    : cpu_primitive_t(&conf_, inputs, outputs), conf_(*pd), kernel_(nullptr)
{
    const auto &desc = *conf_.desc();
    switch (desc.alg_kind) {
    case alg_kind::eltwise_relu:
        kernel_ = new jit_uni_relu_kernel_f32<isa>(desc); break;
    case alg_kind::eltwise_elu:
        kernel_ = new jit_uni_elu_kernel_f32<isa>(desc); break;
    default: assert(!"unknown eltwise alg_kind");
    }
}

template <cpu_isa_t isa>
jit_uni_eltwise_bwd_t<isa>::~jit_uni_eltwise_bwd_t()
{ delete kernel_; }

template <cpu_isa_t isa>
void jit_uni_eltwise_bwd_t<isa>::execute_backward()
{
    auto src = reinterpret_cast<const data_t *>(this->input_memory(0));
    auto diff_dst = reinterpret_cast<const data_t *>(this->input_memory(1));
    auto diff_src = reinterpret_cast<data_t *>(this->memory(0));

    const memory_desc_wrapper data_d(conf_.src_pd());
    const memory_desc_wrapper diff_data_d(conf_.diff_src_pd());

    const size_t nelems = data_d.nelems();

    src += data_d.blocking_desc().offset_padding;
    diff_dst += diff_data_d.blocking_desc().offset_padding;
    diff_src += diff_data_d.blocking_desc().offset_padding;

    auto ker = [&](const int ithr, const int nthr) {
        size_t start{0}, end{0};

        const int cache_line = 16;

        balance211(utils::div_up(nelems, cache_line), nthr, ithr, start, end);
        start = nstl::min(nelems, start * cache_line);
        end = nstl::min(nelems, end * cache_line);

        jit_args arg = {};
        arg.from = &diff_dst[start];
        arg.to = &diff_src[start];
        arg.for_comparison = &src[start];
        arg.work_amount = end - start;
        if (arg.work_amount)
            (*kernel_)(&arg);
    };

#pragma omp parallel
    {
        ker(omp_get_thread_num(), omp_get_num_threads());
    }
}

template struct jit_uni_eltwise_fwd_t<sse42>;
template struct jit_uni_eltwise_bwd_t<sse42>;
template struct jit_uni_eltwise_fwd_t<avx2>;
template struct jit_uni_eltwise_bwd_t<avx2>;
template struct jit_uni_eltwise_fwd_t<avx512_common>;
template struct jit_uni_eltwise_bwd_t<avx512_common>;

}
}
}
