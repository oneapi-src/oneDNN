/*******************************************************************************
* Copyright 2017-2025 Intel Corporation
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

#include "common/bfloat16.hpp"
#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/nstl.hpp"
#include "common/utils.hpp"

#include "cpu/x64/jit_avx512_core_bf16cvt.hpp"
#include "cpu/x64/jit_generator.hpp"

#include "cpu/x64/injectors/jit_uni_eltwise_injector.hpp"
#include "cpu/x64/jit_uni_eltwise.hpp"
#include "cpu/x64/utils/jit_io_helper.hpp"

#define GET_OFF(field) offsetof(jit_args_t, field)

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

using namespace Xbyak;

struct jit_args_t {
    const void *src; // fwd: src;  bwd: src/dst based on alg;
    const void *dst; // fwd: dst;  bwd: diff_src;
    const void *diff_dst; // fwd: nullptr;  bwd: diff_dst;
    size_t work_amount;
};

struct jit_uni_eltwise_kernel : public jit_generator {
    jit_uni_eltwise_kernel(
            const eltwise_pd_t *pd, const char *name, cpu_isa_t isa)
        : jit_generator(name, isa), pd_(pd) {}

    void operator()(jit_args_t *p) { jit_generator::operator()(p); }

protected:
    const eltwise_pd_t *pd_;

    data_type_t data_type() const {
        return pd_->use_dst() ? pd_->dst_md()->data_type
                              : pd_->src_md()->data_type;
    }
    bool is_bf16() const { return data_type() == data_type::bf16; }
    bool is_f16() const { return data_type() == data_type::f16; }
    bool is_f8() const {
        return utils::one_of(
                data_type(), data_type::f8_e5m2, data_type::f8_e4m3);
    }
    int dtype_size() const { return types::data_type_size(data_type()); }
    cpu_isa_t get_io_isa(cpu_isa_t isa) const {
        // reusing avx512_core instantiation for bf16
        return is_bf16() && is_superset(isa, avx512_core)
                        && mayiuse(avx512_core_bf16)
                ? avx512_core_bf16
                : isa;
    }
};

// jit kernels
namespace {
template <cpu_isa_t isa>
struct jit_uni_kernel_t : public jit_uni_eltwise_kernel {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_kernel)

    jit_uni_kernel_t(const eltwise_pd_t *pd)
        : jit_uni_eltwise_kernel(pd, jit_name(), isa)
        , vlen_(is_bf16() || is_f16() ? cpu_isa_traits<isa>::vlen / 2
                          : is_f8()   ? cpu_isa_traits<isa>::vlen / 4
                                      : cpu_isa_traits<isa>::vlen)
        , simd_w_(vlen_ / dtype_size())
        , is_fwd_(pd_->is_fwd()) {

        const auto &desc = *pd_->desc();
        // we can consider that there's no auxiliary vregs on fwd path
        // using the first 7 vregs can be considered volatile during the call
        // to eltwise injector
        const bool save_state = is_fwd_ ? false : true;
        eltwise_injector_.reset(new jit_uni_eltwise_injector<injector_isa>(this,
                desc.alg_kind, desc.alpha, desc.beta, 1.f, data_type::f32,
                save_state, reg_injector_table, injector_mask, is_fwd_,
                pd_->use_dst()));
        io::io_conf_t io_conf;
        io::io_tail_conf_t io_tail_conf(simd_w_, tail_size_, tail_opmask_idx_,
                vmm_tail_mask.getIdx(), reg_tmp);
        io::io_emu_bf16_conf_t io_bf16_conf(emu_zmm_1_idx_, emu_zmm_2_idx_,
                emu_zmm_3_idx_, reg_tmp, emu_zmm_4_idx_);
        io::io_emu_fp8_conf_t io_fp8_conf(emu_zmm_1_idx_, emu_zmm_2_idx_,
                emu_zmm_3_idx_, emu_zmm_4_idx_, emu_zmm_5_idx_,
                emu_kmask_aux_idx_, reg_tmp);
        io_ = io::jit_io_multi_dt_helper_t<Vmm>(this, get_io_isa(isa),
                {data_type()}, io_conf, io_tail_conf, io_bf16_conf, {},
                utils::nullopt, io_fp8_conf);
    }

    void compute_dst(const bool tail) {
        io_[data_type()]->load(ptr[reg_src], vmm_src, tail);
        eltwise_injector_->compute_vector(vmm_src.getIdx());
        if (!is_fwd_) {
            io_[data_type()]->load(ptr[reg_diff_dst], vmm_diff_dst, tail);
            uni_vmulps(vmm_src, vmm_src, vmm_diff_dst);
        }
        io_[data_type()]->store(vmm_src, ptr[reg_dst], tail);
    }

    void compute_two_simdw_xf16_dst(const bool tail) {
        io_[data_type()]->load_two_simdw_xf16(
                ptr[reg_src], vmm_src_even, vmm_src_odd);
        io_[data_type()]->merge_interleaved_to_plain(
                vmm_src_even, vmm_src_odd, vmm_tmp);
        if (!is_fwd_) {
            io_[data_type()]->load_two_simdw_xf16(
                    ptr[reg_diff_dst], vmm_diff_dst_even, vmm_diff_dst_odd);
            io_[data_type()]->merge_interleaved_to_plain(
                    vmm_diff_dst_even, vmm_diff_dst_odd, vmm_tmp);
        }
        for (int i = 0; i < 2; ++i) {
            const auto vsrc = i == 0 ? vmm_src_even : vmm_src_odd;
            const auto vdiff_dst
                    = i == 0 ? vmm_diff_dst_even : vmm_diff_dst_odd;
            eltwise_injector_->compute_vector(vsrc.getIdx());
            if (!is_fwd_) uni_vmulps(vsrc, vsrc, vdiff_dst);
            io_[data_type()]->store(vsrc, ptr[reg_dst + i * vlen_], tail);
        }
    }

    void compute_two_simdw_xf16() {
        Label loop_start, loop_end;

        cmp(reg_work_amount, 2 * simd_w_);
        jl(loop_end, T_NEAR);

        L(loop_start);
        {
            compute_two_simdw_xf16_dst(false);
            add(reg_src, 2 * vlen_);
            add(reg_dst, 2 * vlen_);
            if (!is_fwd_) add(reg_diff_dst, 2 * vlen_);

            sub(reg_work_amount, 2 * simd_w_);
            cmp(reg_work_amount, 2 * simd_w_);
            jge(loop_start, T_NEAR);
        }
        L(loop_end);
    }

    void compute() {
        // Compute two simdw at once in vectorized loop first
        // when ne_convert instructions is available for xf16
        if (isa == avx2_vnni_2 && (is_bf16() || is_f16()))
            compute_two_simdw_xf16();

        Label vectorized_loop_start, reminder_loop_start, loop_end;

        cmp(reg_work_amount, simd_w_);
        jl(reminder_loop_start, T_NEAR);

        L(vectorized_loop_start);
        {
            compute_dst(false);
            add(reg_src, vlen_);
            add(reg_dst, vlen_);
            if (!is_fwd_) add(reg_diff_dst, vlen_);

            sub(reg_work_amount, simd_w_);
            cmp(reg_work_amount, simd_w_);
            jge(vectorized_loop_start, T_NEAR);
        }

        L(reminder_loop_start);
        {
            cmp(reg_work_amount, 0);
            jle(loop_end, T_NEAR);

            compute_dst(true);
            add(reg_src, dtype_size());
            add(reg_dst, dtype_size());
            if (!is_fwd_) add(reg_diff_dst, dtype_size());

            dec(reg_work_amount);
            jmp(reminder_loop_start, T_NEAR);
        }
        L(loop_end);
    }

    void generate() override {
        preamble();

        io_.prepare_tail_mask();
        if (is_bf16()) io_.init_bf16();

        Reg64 param = abi_param1;
        mov(reg_src, ptr[param + GET_OFF(src)]);
        mov(reg_dst, ptr[param + GET_OFF(dst)]);
        if (!is_fwd_) mov(reg_diff_dst, ptr[param + GET_OFF(diff_dst)]);
        mov(reg_work_amount, ptr[param + GET_OFF(work_amount)]);
        eltwise_injector_->load_table_addr();

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
        compute();

        postamble();

        eltwise_injector_->prepare_table();
        if (is_f8()) io_.prepare_table_fp8();
    }

private:
    using Vmm = typename cpu_isa_traits<isa>::Vmm;
    static constexpr cpu_isa_t injector_isa
            = isa == avx512_core_amx ? avx512_core : isa;

    const int vlen_;
    const int simd_w_;
    const bool is_fwd_;
    const int tail_size_ = 1;

    Reg64 reg_src = rax;
    Reg64 reg_dst = r8;
    Reg64 reg_injector_table = r9;
    Reg64 reg_diff_dst = r10;
    Reg64 reg_work_amount = rsi;
    Reg64 imm_addr64 = rbx;
    Reg64 reg_tmp = r14;

    Opmask injector_mask = Opmask(1);

    Vmm vmm_src = Vmm(1);
    Vmm vmm_diff_dst = Vmm(2);
    Vmm vmm_tmp = Vmm(3);
    // vmm_tail_mask for load/store data with tail
    // vmm_src_odd/vmm_src_even for load/store xf16 data with NE_CONVERT
    // instructions
    Vmm vmm_tail_mask = Vmm(7);
    Vmm vmm_src_even = vmm_src;
    Vmm vmm_src_odd = Vmm(8);
    Vmm vmm_diff_dst_even = vmm_diff_dst;
    Vmm vmm_diff_dst_odd = Vmm(9);
    std::unique_ptr<jit_uni_eltwise_injector<injector_isa>> eltwise_injector_;
    io::jit_io_multi_dt_helper_t<Vmm> io_;

    /* bf16 and fp8 support */
    const int emu_zmm_1_idx_ = 25;
    const int emu_zmm_2_idx_ = 26;
    const int emu_zmm_3_idx_ = 27;
    const int emu_zmm_4_idx_ = 28;
    const int emu_zmm_5_idx_ = 29;
    const int tail_opmask_idx_ = 6;
    const int emu_kmask_aux_idx_ = 2;
};

} // namespace

template <cpu_isa_t isa, data_type_t d_type>
status_t jit_uni_eltwise_fwd_t<isa, d_type>::pd_t::init(engine_t *engine) {
    using namespace alg_kind;

    const memory_desc_wrapper src_d(src_md());

    // disabling verbose dispatch messages for unsupported isa for better readability
    if (!mayiuse(isa)) return status::unimplemented;

    static constexpr cpu_isa_t injector_isa
            = isa == avx512_core_amx ? avx512_core : isa;

    VDISPATCH_ELTWISE(is_fwd(), VERBOSE_BAD_PROPKIND);
    VDISPATCH_ELTWISE(utils::everyone_is(
                              d_type, src_md()->data_type, dst_md()->data_type),
            VERBOSE_UNSUPPORTED_DT);
    VDISPATCH_ELTWISE(IMPLICATION(src_md()->data_type == data_type::bf16,
                              mayiuse(avx512_core) || mayiuse(avx2_vnni_2)),
            VERBOSE_ISA_DT_MISMATCH);
    VDISPATCH_ELTWISE(
            IMPLICATION(src_md()->data_type == data_type::f16,
                    mayiuse(avx512_core_fp16) || mayiuse(avx2_vnni_2)),
            VERBOSE_ISA_DT_MISMATCH);
    VDISPATCH_ELTWISE(!has_zero_dim_memory(), VERBOSE_EMPTY_TENSOR, "");
    VDISPATCH_ELTWISE(src_d.is_dense(true), VERBOSE_UNSUPPORTED_SPARSE_CFG);
    VDISPATCH_ELTWISE(eltwise_injector::is_supported(
                              injector_isa, desc_.alg_kind, data_type::f32),
            VERBOSE_BAD_ALGORITHM);
    // refer to a comment in jit_uni_kernel why this is needed
    VDISPATCH_ELTWISE(IMPLICATION(!src_d.is_dense(), is_zero_preserved()),
            VERBOSE_UNSUPPORTED_SPARSE_CFG);
    VDISPATCH_ELTWISE(attr()->has_default_values(), VERBOSE_UNSUPPORTED_ATTR);
    VDISPATCH_ELTWISE(set_default_formats_common(), VERBOSE_UNSUPPORTED_TAG);
    VDISPATCH_ELTWISE(src_d == memory_desc_wrapper(dst_md()),
            VERBOSE_INCONSISTENT_MDS, "src", "dst");

    return status::success;
}

template <cpu_isa_t isa, data_type_t d_type>
jit_uni_eltwise_fwd_t<isa, d_type>::jit_uni_eltwise_fwd_t(const pd_t *apd)
    : primitive_t(apd) {}

template <cpu_isa_t isa, data_type_t d_type>
jit_uni_eltwise_fwd_t<isa, d_type>::~jit_uni_eltwise_fwd_t() = default;

template <cpu_isa_t isa, data_type_t d_type>
status_t jit_uni_eltwise_fwd_t<isa, d_type>::init(engine_t *engine) {
    CHECK(safe_ptr_assign(kernel_, new jit_uni_kernel_t<isa>(pd())));
    return kernel_->create_kernel();
}

template <cpu_isa_t isa, data_type_t d_type>
status_t jit_uni_eltwise_fwd_t<isa, d_type>::execute(exec_ctx_t &ctx) const {
    auto src = CTX_IN_MEM(const data_t *, DNNL_ARG_SRC);
    auto dst = CTX_OUT_MEM(data_t *, DNNL_ARG_DST);

    const memory_desc_wrapper data_d(pd()->src_md());
    const auto nelems = data_d.nelems(true);
    const int simd_w = 64 / data_d.data_type_size();

    src += data_d.offset0();
    dst += data_d.offset0();

    parallel(0, [&](const int ithr, const int nthr) {
        dim_t start {0}, end {0};

        balance211(utils::div_up(nelems, simd_w), nthr, ithr, start, end);
        start = nstl::min(nelems, start * simd_w);
        end = nstl::min(nelems, end * simd_w);
        if (start == end) return;

        jit_args_t args;
        args.src = src + start;
        args.dst = dst + start;
        args.diff_dst = nullptr;
        args.work_amount = end - start;
        (*kernel_)(&args);
    });

    return status::success;
}

template <cpu_isa_t isa, data_type_t d_type>
status_t jit_uni_eltwise_bwd_t<isa, d_type>::pd_t::init(engine_t *engine) {
    using namespace alg_kind;

    const memory_desc_wrapper data_d(data_md());

    // disabling verbose dispatch messages for unsupported isa for better readability
    if (!mayiuse(isa)) return status::unimplemented;

    static constexpr cpu_isa_t injector_isa
            = isa == avx512_core_amx ? avx512_core : isa;

    VDISPATCH_ELTWISE(!is_fwd(), VERBOSE_BAD_PROPKIND);
    VDISPATCH_ELTWISE(
            utils::everyone_is(d_type, data_md()->data_type,
                    diff_src_md()->data_type, diff_dst_md()->data_type),
            VERBOSE_UNSUPPORTED_DT);
    VDISPATCH_ELTWISE(IMPLICATION(data_md()->data_type == data_type::bf16,
                              mayiuse(avx512_core)),
            VERBOSE_ISA_DT_MISMATCH);
    VDISPATCH_ELTWISE(IMPLICATION(data_md()->data_type == data_type::f16,
                              mayiuse(avx512_core_fp16)),
            VERBOSE_ISA_DT_MISMATCH);
    VDISPATCH_ELTWISE(!has_zero_dim_memory(), VERBOSE_EMPTY_TENSOR, "");
    VDISPATCH_ELTWISE(set_default_formats_common(), VERBOSE_UNSUPPORTED_TAG);
    VDISPATCH_ELTWISE(data_d.is_dense(true), VERBOSE_UNSUPPORTED_SPARSE_CFG);
    VDISPATCH_ELTWISE(eltwise_injector::is_isa_supported(injector_isa),
            VERBOSE_UNSUPPORTED_ISA);
    VDISPATCH_ELTWISE(eltwise_injector::is_alg_supported(desc_.alg_kind),
            VERBOSE_BAD_ALGORITHM);
    // refer to a comment in jit_uni_kernel why this is needed
    VDISPATCH_ELTWISE(IMPLICATION(!data_d.is_dense(), is_zero_preserved()),
            VERBOSE_UNSUPPORTED_SPARSE_CFG);
    VDISPATCH_ELTWISE(data_d == memory_desc_wrapper(diff_dst_md()),
            VERBOSE_INCONSISTENT_MDS, "data", "diff_dst");
    VDISPATCH_ELTWISE(memory_desc_wrapper(diff_src_md())
                    == memory_desc_wrapper(diff_dst_md()),
            VERBOSE_INCONSISTENT_MDS, "diff_src", "diff_dst");
    VDISPATCH_ELTWISE(attr()->has_default_values(), VERBOSE_UNSUPPORTED_ATTR);

    return status::success;
}

template <cpu_isa_t isa, data_type_t d_type>
jit_uni_eltwise_bwd_t<isa, d_type>::jit_uni_eltwise_bwd_t(const pd_t *apd)
    : primitive_t(apd) {}

template <cpu_isa_t isa, data_type_t d_type>
jit_uni_eltwise_bwd_t<isa, d_type>::~jit_uni_eltwise_bwd_t() = default;

template <cpu_isa_t isa, data_type_t d_type>
status_t jit_uni_eltwise_bwd_t<isa, d_type>::init(engine_t *engine) {
    CHECK(safe_ptr_assign(kernel_, new jit_uni_kernel_t<isa>(pd())));
    return kernel_->create_kernel();
}

template <cpu_isa_t isa, data_type_t d_type>
status_t jit_uni_eltwise_bwd_t<isa, d_type>::execute(exec_ctx_t &ctx) const {
    auto src = pd()->use_dst() ? CTX_IN_MEM(const data_t *, DNNL_ARG_DST)
                               : CTX_IN_MEM(const data_t *, DNNL_ARG_SRC);
    auto diff_dst = CTX_IN_MEM(const data_t *, DNNL_ARG_DIFF_DST);
    auto diff_src = CTX_OUT_MEM(data_t *, DNNL_ARG_DIFF_SRC);

    const memory_desc_wrapper data_d(pd()->data_md());
    const memory_desc_wrapper diff_data_d(pd()->diff_src_md());
    const auto nelems = data_d.nelems(true);
    const int simd_w = 64 / data_d.data_type_size();

    src += data_d.offset0();
    diff_dst += diff_data_d.offset0();
    diff_src += diff_data_d.offset0();

    parallel(0, [&](const int ithr, const int nthr) {
        dim_t start {0}, end {0};

        balance211(utils::div_up(nelems, simd_w), nthr, ithr, start, end);
        start = nstl::min(nelems, start * simd_w);
        end = nstl::min(nelems, end * simd_w);
        if (start == end) return;

        jit_args_t args;
        args.src = src + start;
        args.dst = diff_src + start;
        args.diff_dst = diff_dst + start;
        args.work_amount = end - start;
        (*kernel_)(&args);
    });

    return status::success;
}

template struct jit_uni_eltwise_fwd_t<sse41, data_type::f32>;
template struct jit_uni_eltwise_fwd_t<avx, data_type::f32>;
template struct jit_uni_eltwise_fwd_t<avx2, data_type::f32>;
template struct jit_uni_eltwise_fwd_t<avx2_vnni_2, data_type::bf16>;
template struct jit_uni_eltwise_fwd_t<avx2_vnni_2, data_type::f16>;
template struct jit_uni_eltwise_fwd_t<avx512_core, data_type::f32>;
template struct jit_uni_eltwise_fwd_t<avx512_core, data_type::bf16>;
template struct jit_uni_eltwise_fwd_t<avx512_core_fp16, data_type::f16>;
template struct jit_uni_eltwise_fwd_t<avx512_core_amx, data_type::f8_e5m2>;
template struct jit_uni_eltwise_fwd_t<avx512_core_amx, data_type::f8_e4m3>;

template struct jit_uni_eltwise_bwd_t<sse41, data_type::f32>;
template struct jit_uni_eltwise_bwd_t<avx, data_type::f32>;
template struct jit_uni_eltwise_bwd_t<avx2, data_type::f32>;
template struct jit_uni_eltwise_bwd_t<avx512_core, data_type::f32>;
template struct jit_uni_eltwise_bwd_t<avx512_core, data_type::bf16>;
template struct jit_uni_eltwise_bwd_t<avx512_core_fp16, data_type::f16>;
template struct jit_uni_eltwise_bwd_t<avx512_core_amx, data_type::f8_e5m2>;
template struct jit_uni_eltwise_bwd_t<avx512_core_amx, data_type::f8_e4m3>;

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
