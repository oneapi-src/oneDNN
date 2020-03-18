/*******************************************************************************
* Copyright 2017-2020 Intel Corporation
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

#include "bfloat16.hpp"
#include "c_types_map.hpp"
#include "dnnl_thread.hpp"
#include "nstl.hpp"
#include "utils.hpp"

#include "jit_avx512_core_bf16cvt.hpp"
#include "jit_generator.hpp"

#include "eltwise/jit_uni_eltwise.hpp"
#include "eltwise/jit_uni_eltwise_injector.hpp"

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
    jit_uni_eltwise_kernel(const eltwise_desc_t &desc) : desc_(desc) {}
    virtual ~jit_uni_eltwise_kernel() {}

    void operator()(const jit_args *args) {
        assert(ker_);
        ker_(args);
    }

protected:
    void (*ker_)(const jit_args *) = nullptr;

    bool is_bwd() const { return desc_.prop_kind == prop_kind::backward_data; }
    data_type_t data_type() const { return desc_.data_desc.data_type; }
    bool is_bf16() const { return data_type() == data_type::bf16; }
    int dtype_size() const { return types::data_type_size(data_type()); }

private:
    const eltwise_desc_t &desc_;
};

/* jit kernels */
namespace {
using namespace Xbyak;
struct jit_bf16_eltwise_injector {
    jit_bf16_eltwise_injector(
            jit_generator *host, Opmask k_tail_mask, bf16_emulation_t *emu)
        : h(host), emu_(emu), k_tail_mask_(k_tail_mask) {}

    void prepare_masks() {
        Reg64 mask_reg = Xbyak::util::r14;
        h->push(mask_reg);

        h->mov(mask_reg.cvt32(), 0x1);
        h->kmovd(k_tail_mask_, mask_reg.cvt32());

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
                h->vmovdqu16(h->ptr[reg_to + offset], ymm_bf16);
            else
                h->vmovdqu16(h->ptr[reg_to + offset] | k_tail_mask_, ymm_bf16);
        }
    }

private:
    jit_generator *const h;
    bf16_emulation_t *const emu_;
    Xbyak::Opmask k_tail_mask_;
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
            bf16_injector_ = new jit_bf16_eltwise_injector(
                    this, k_tail_mask, bf16_emu_);
        }

        preamble();

        if (is_bf16()) {
            bf16_injector_->prepare_masks();
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

    opmask_t k_tail_mask = k6;

    jit_bf16_eltwise_injector *bf16_injector_;
    bf16_emulation_t *bf16_emu_;
};

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
            bf16_injector_ = new jit_bf16_eltwise_injector(
                    this, k_tail_mask, bf16_emu_);
        }

        eltwise_injector_
                = new jit_uni_eltwise_injector_f32<isa>(this, desc.alg_kind,
                        desc.alpha, desc.beta, 1.f, false, r9, Opmask(1));

        preamble();

        if (is_bf16()) {
            bf16_injector_->prepare_masks();
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

    opmask_t k_tail_mask = k6;

    jit_bf16_eltwise_injector *bf16_injector_;
    bf16_emulation_t *bf16_emu_;
};

} /* namespace */

template <cpu_isa_t isa, data_type_t d_type>
status_t jit_uni_eltwise_fwd_t<isa, d_type>::pd_t::init() {
    using namespace alg_kind;
    using namespace data_type;

    bool alg_ok = utils::one_of(desc()->alg_kind, eltwise_relu, eltwise_tanh,
            eltwise_elu, eltwise_square, eltwise_abs, eltwise_sqrt,
            eltwise_linear, eltwise_bounded_relu, eltwise_soft_relu,
            eltwise_logistic, eltwise_exp, eltwise_gelu_tanh, eltwise_swish,
            eltwise_log, eltwise_clip, eltwise_pow, eltwise_gelu_erf,
            eltwise_relu_use_dst_for_bwd, eltwise_tanh_use_dst_for_bwd,
            eltwise_elu_use_dst_for_bwd, eltwise_sqrt_use_dst_for_bwd,
            eltwise_logistic_use_dst_for_bwd, eltwise_exp_use_dst_for_bwd);

    bool ok = mayiuse(isa) && is_fwd() && desc()->data_desc.data_type == d_type
            && IMPLICATION(
                    desc()->data_desc.data_type == bf16, mayiuse(avx512_core))
            && alg_ok && !has_zero_dim_memory()
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
template struct jit_uni_eltwise_fwd_t<avx2, data_type::f32>;
template struct jit_uni_eltwise_fwd_t<avx512_common, data_type::f32>;
template struct jit_uni_eltwise_fwd_t<avx512_core, data_type::bf16>;

template struct jit_uni_eltwise_bwd_t<sse41, data_type::f32>;
template struct jit_uni_eltwise_bwd_t<avx2, data_type::f32>;
template struct jit_uni_eltwise_bwd_t<avx512_common, data_type::f32>;
template struct jit_uni_eltwise_bwd_t<avx512_core, data_type::bf16>;

} // namespace cpu
} // namespace impl
} // namespace dnnl
