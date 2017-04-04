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

#include "jit_uni_relu.hpp"

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

template <cpu_isa_t isa>
struct jit_uni_relu_kernel_f32 : public jit_generator {
    bool isBackward;
    float negative_slope;

    void (*jit_ker)(jit_args *);
    void operator()(jit_args *arg){jit_ker(arg);}

    void compute_step(bool vectorize, const int uf, const int shift)
    {
        unsigned char _cmp_gt_os = 14;

        for (int i = 0; i < uf; i++) {
            if (vectorize) {
                vmovups(Vmm(i), ptr[reg_from + i * shift]);
                if (this->isBackward)
                    vmovups(Vmm(uf + i), ptr[reg_for_comparison + i * shift]);
            } else {
                movss(Xmm(i), ptr[reg_from + i * shift]);
                if (this->isBackward)
                    movss(Xmm(uf + i), ptr[reg_for_comparison + i * shift]);
            }
        }

        for (int i = 0; i < uf; i++) {
            vmulps(Vmm(2 * uf + i), Vmm(i), vmm_ns);
            if (isa == avx2) {
                if (this->isBackward)
                    vcmpgtps(vmm_mask, Vmm(uf + i), vmm_zero);
                else
                    vcmpgtps(vmm_mask, Vmm(i), vmm_zero);

                vblendvps(Vmm(2 * uf + i), Vmm(2 * uf + i), Vmm(i), vmm_mask);

            } else {
                if (this->isBackward)
                    vcmpps(k_mask, Vmm(uf + i), vmm_zero, _cmp_gt_os);
                else
                    vcmpps(k_mask, Vmm(i), vmm_zero, _cmp_gt_os);
                vblendmps(Vmm(2 * uf + i) | k_mask, Vmm(2 * uf + i), Vmm(i));
            }
        }

        for (int i = 0; i < uf; i++) {
            if (vectorize) {
                vmovups(ptr[reg_to + i * shift], Vmm(2 * uf + i));
            } else {
                movss(ptr[reg_to + i * shift], Xmm(2 * uf + i));
            }
        }
    }

    jit_uni_relu_kernel_f32(bool isbackward, float nslope)
        : jit_generator(), isBackward(isbackward), negative_slope(nslope)
    {
        assert(isa == avx2 || isa == avx512_mic);

        Reg64 param = abi_param1;

        const int simd_w = cpu_isa_trait<isa>::vlen / sizeof(float);
        const int loop_dec[] = {simd_w, 1};
        const int uf[] = {1, 1};
        const int shift[] = {cpu_isa_trait<isa>::vlen, sizeof(float)};
        const bool loop_vectorize[] = {true, false};

        this->preamble();

        mov(reg_from, ptr[param + GET_OFF(from)]);
        if (this->isBackward)
            mov(reg_for_comparison, ptr[param + GET_OFF(for_comparison)]);
        mov(reg_to, ptr[param + GET_OFF(to)]);
        mov(reg_work_amount, ptr[param + GET_OFF(work_amount)]);

        mov(imm_addr64, float2int(this->negative_slope));
        movq(xmm_ns, imm_addr64);
        vbroadcastss(vmm_ns, xmm_ns);

        uni_vpxor(vmm_zero, vmm_zero, vmm_zero);

        Label loop_label[3];

        for (int id = 0; id < 2; id++) {
            L(loop_label[id]);
            cmp(reg_work_amount, uf[id] * loop_dec[id] - 1);
            jle(loop_label[id + 1], T_NEAR);

            compute_step(loop_vectorize[id], uf[id], shift[id]);

            add(reg_from, uf[id] * shift[id]);
            add(reg_to, uf[id] * shift[id]);
            if (this->isBackward)
                add(reg_for_comparison, uf[id] * shift[id]);

            sub(reg_work_amount, uf[id] * loop_dec[id]);
            jmp(loop_label[id]);
        }

        L(loop_label[2]);
        this->postamble();

        jit_ker = (void (*)(jit_args *)) this->getCode();
    }

private:
    using Vmm = typename utils::conditional<isa == avx2, Ymm, Zmm>::type;
    void uni_vpxor(const Xmm &x1, const Xmm &x2, const Operand &op)
    {
        if (isa == avx2)
            vpxor(x1, x2, op);
        else
            vpxord(x1, x2, op);
    }

    Reg64 reg_from = rax;
    Reg64 reg_for_comparison = this->isBackward ? rdx : reg_from;
    Reg64 reg_to = r8;
    Reg64 reg_work_amount = rsi;
    Reg64 imm_addr64 = rbx;

    Xmm xmm_ns = Xmm(14);

    Vmm vmm_ns = Vmm(isa == avx2 ? 14 : 30);
    Vmm vmm_zero = Vmm(isa == avx2 ? 15 : 31);

    Vmm vmm_mask = Vmm(isa == avx2 ? 12 : 28);
    Opmask k_mask = Opmask(1);
};

template <cpu_isa_t isa>
status_t jit_uni_relu_fwd_t<isa>::pd_t::init()
{
    using namespace prop_kind;

    assert(engine()->kind() == engine_kind::cpu);
    bool ok = true && mayiuse(isa)
        && utils::one_of(desc()->prop_kind, forward_training,
                forward_inference)
        && utils::everyone_is(data_type::f32, desc()->data_desc.data_type)
        && memory_desc_wrapper(src_pd()).is_dense();
    return ok ? status::success : status::unimplemented;
}

template <cpu_isa_t isa>
jit_uni_relu_fwd_t
        <isa>::jit_uni_relu_fwd_t(const pd_t *pd, const input_vector &inputs,
                                  const output_vector &outputs)
    : cpu_primitive_t(&conf_, inputs, outputs), conf_(*pd)
{ 
    kernel_ = new jit_uni_relu_kernel_f32<isa>(false,
        conf_.desc()->negative_slope);
}

template <cpu_isa_t isa>
jit_uni_relu_fwd_t<isa>::~jit_uni_relu_fwd_t()
{ delete kernel_; }

template <cpu_isa_t isa>
void jit_uni_relu_fwd_t<isa>::execute_forward()
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
status_t jit_uni_relu_bwd_t<isa>::pd_t::init()
{
    using namespace prop_kind;

    assert(engine()->kind() == engine_kind::cpu);

    bool ok = true
        && mayiuse(isa)
        && utils::one_of(desc()->prop_kind, backward_data, backward)
        && utils::everyone_is(data_type::f32, desc()->data_desc.data_type,
                desc()->diff_data_desc.data_type)
        && utils::everyone_is(desc()->data_desc.format,
                desc()->diff_data_desc.format)
        && memory_desc_wrapper(src_pd()).is_dense()
        && memory_desc_wrapper(diff_dst_pd()).is_dense()
        && memory_desc_wrapper(diff_src_pd()).is_dense();

    return ok ? status::success : status::unimplemented;
}

template <cpu_isa_t isa>
jit_uni_relu_bwd_t<isa>::jit_uni_relu_bwd_t(const pd_t *pd,
        const input_vector &inputs, const output_vector &outputs)
    : cpu_primitive_t(&conf_, inputs, outputs), conf_(*pd)
{
    kernel_ = new jit_uni_relu_kernel_f32<isa>(true,
        conf_.desc()->negative_slope);
}

template <cpu_isa_t isa>
jit_uni_relu_bwd_t<isa>::~jit_uni_relu_bwd_t()
{ delete kernel_; }

template <cpu_isa_t isa>
void jit_uni_relu_bwd_t<isa>::execute_backward()
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

template struct jit_uni_relu_fwd_t<avx2>;
template struct jit_uni_relu_bwd_t<avx2>;
template struct jit_uni_relu_fwd_t<avx512_mic>;
template struct jit_uni_relu_bwd_t<avx512_mic>;

}
}
}
