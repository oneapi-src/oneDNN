/*******************************************************************************
* Copyright 2023 Intel Corporation
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

#include "common/dnnl_thread.hpp"

#include "cpu/cpu_primitive.hpp"

#include "cpu/x64/jit_generator.hpp"
#include "cpu/x64/utils/jit_io_helper.hpp"

#include "cpu/x64/jit_uni_group_normalization.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

using namespace Xbyak;
using namespace data_type;

namespace {
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
struct kernel_t : public jit_uni_group_normalization_fwd_t::kernel_base_t,
                  public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_group_normalization_fwd_t::kernel_t);

    kernel_t(const group_normalization_pd_t *pd)
        : jit_generator(jit_name())
        , src_d_(pd->src_md())
        , dst_d_(pd->dst_md())
        , C_(pd->src_md()->dims[1])
        , simd_w_(vlen / sizeof(float))
        , axis_simd_full_(C_ / simd_w_)
        , axis_simd_tail_(C_ % simd_w_)
        , use_scale_(pd->use_scale())
        , use_shift_(pd->use_shift())
        , eps_(pd->desc()->group_norm_epsilon) {
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

    status_t create_kernel() override { return jit_generator::create_kernel(); }
    void generate() override {
        const size_t c_src_size
                = C_ * types::data_type_size(src_d_.data_type());
        const size_t c_dst_size
                = C_ * types::data_type_size(dst_d_.data_type());

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

        // load epsilon
        uni_vmovq(xmm_tmp, reg_eps);
        uni_vbroadcastss(vmm_eps, xmm_tmp);

        // load ones
        mov(reg_tmp, float2int(1.f));
        uni_vmovq(xmm_tmp, reg_tmp);
        uni_vbroadcastss(vmm_ones, xmm_tmp);

        // add block_start to block_size to define block_end
        add(reg_block_end, reg_src);

        Xbyak::Label unroll_loop, end;
        L(unroll_loop);
        {
            cmp(reg_block_end, reg_src);
            jle(end, T_NEAR);

            // precompute and broadcast scales
            uni_vmovss(xmm_tmp, dword[reg_src_scales]);
            uni_vbroadcastss(vmm_combined_scales, xmm_tmp);
            uni_vmovss(xmm_tmp, dword[reg_dst_scales]);
            uni_vbroadcastss(vmm_tmp, xmm_tmp);
            uni_vmulps(vmm_combined_scales, vmm_combined_scales, vmm_tmp);
            io_.init_saturate_f32({dst_d_.data_type()});

            // calculate dst
            compute_dst();

            add(reg_src, c_src_size);
            add(reg_dst, c_dst_size);

            jmp(unroll_loop);
        }
        L(end);

        postamble();
    }

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

protected:
    using Vmm = typename cpu_isa_traits<isa>::Vmm;
    const Xbyak::AddressFrame &vmmword = (isa == sse41) ? xword
            : (isa == avx2)                             ? yword
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
    const dim_t C_;
    const size_t simd_w_;
    const dim_t axis_simd_full_;
    const dim_t axis_simd_tail_;
    const bool use_scale_ = false;
    const bool use_shift_ = false;
    const float eps_;

    void compute_dst_body(size_t offt_elems, bool tail = false) {
        if (use_scale_) {
            io_[f32]->load(scale_ptr(offt_elems), vmm_scale, tail);
        }
        if (use_shift_) {
            io_[f32]->load(shift_ptr(offt_elems), vmm_shift, tail);
        }
        io_[src_d_.data_type()]->load(src_ptr(offt_elems), vmm_dst, tail);

        io_[f32]->load(mean_ptr(offt_elems), vmm_mean, tail);
        io_[f32]->load(var_ptr(offt_elems), vmm_inv_sqrtvar, tail);

        // calculate inv_sqrtvar
        uni_vaddps(vmm_inv_sqrtvar, vmm_inv_sqrtvar, vmm_eps);
        uni_vsqrtps(vmm_inv_sqrtvar, vmm_inv_sqrtvar);
        uni_vdivps(vmm_inv_sqrtvar, vmm_ones, vmm_inv_sqrtvar, vmm_tmp);

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

    void compute_dst() {
        for (dim_t i = 0; i < axis_simd_full_; i++)
            compute_dst_body(i * simd_w_);
        if (axis_simd_tail_) compute_dst_body(axis_simd_full_ * simd_w_, true);
    }

    Xbyak::Address src_ptr(size_t offt = 0) {
        return vmmword[reg_src + offt * src_d_.data_type_size()];
    }

    Xbyak::Address dst_ptr(size_t offt = 0) {
        return vmmword[reg_dst + offt * dst_d_.data_type_size()];
    }

    Xbyak::Address mean_ptr(size_t offt = 0) {
        return vmmword[reg_mean + offt * sizeof(float)];
    }

    Xbyak::Address var_ptr(size_t offt = 0) {
        return vmmword[reg_var + offt * sizeof(float)];
    }

    Xbyak::Address scale_ptr(size_t offt = 0) {
        return vmmword[reg_scale + offt * sizeof(float)];
    }

    Xbyak::Address shift_ptr(size_t offt = 0) {
        return vmmword[reg_shift + offt * sizeof(float)];
    }

    const Xbyak::Reg64 reg_param = abi_param1;
    const Xbyak::Reg64 reg_src = rdx;
    const Xbyak::Reg64 reg_dst = rax;
    const Xbyak::Reg64 reg_mean = rbx;
    const Xbyak::Reg64 reg_scale = r8;
    const Xbyak::Reg64 reg_block_end = r9;
    const Xbyak::Reg64 reg_eps = r10;
    const Xbyak::Reg64 reg_tmp = r11;
    const Xbyak::Reg64 reg_shift = r12;
    const Xbyak::Reg64 reg_var = r13;
    const Xbyak::Reg64 reg_src_scales = r14;
    const Xbyak::Reg64 reg_dst_scales = r15;

    const Vmm vmm_tail_mask = Vmm(0);
    const Vmm vmm_zero = Vmm(5); // In unroll range, safe for dst compute.
    const Vmm vmm_saturation_ubound
            = Vmm(6); // In unroll range, safe for dst compute.
    const Vmm vmm_combined_scales = Vmm(7);
    const Vmm vmm_scale = Vmm(8); // In unroll range, safe for dst compute.
    const Vmm vmm_shift = Vmm(9); // In unroll range, safe for dst compute.
    const Vmm vmm_ones = Vmm(10);
    const Vmm vmm_eps = Vmm(11);
    const Vmm vmm_mean = Vmm(12);
    const Vmm vmm_inv_sqrtvar = Vmm(13);
    const Vmm vmm_dst = Vmm(14);
    const Vmm vmm_tmp = Vmm(15);
    const Xbyak::Xmm xmm_tmp = Xbyak::Xmm(15);

    const int bf16_emu_zmm_1_idx = 28;
    const int bf16_emu_zmm_2_idx = 29;
    const int bf16_emu_zmm_3_idx = 30;
    const int bf16_emu_zmm_4_idx = 31;
    const int tail_opmask_idx = 1;
};

template struct kernel_t<avx2>;
template struct kernel_t<avx512_core>;

template <cpu_isa_t isa>
struct kernel_stat_t
    : public jit_uni_group_normalization_fwd_t::kernel_stat_base_t,
      public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(
            jit_uni_group_normalization_fwd_t::kernel_stat_t);

    kernel_stat_t(const group_normalization_pd_t *pd, bool compute_var = false)
        : jit_generator(jit_name())
        , src_d_(pd->src_md())
        , compute_var_(compute_var)
        , C_(pd->src_md()->dims[1])
        , simd_w_(vlen / sizeof(float))
        , axis_simd_tail_(C_ % simd_w_)
        , unroll_c_(compute_var_ ? 6 : 12) // Vmm3-14
        , c_block_(unroll_c_ * simd_w_)
        , nc_blocks_(C_ / c_block_)
        , c_block_tail_((C_ % c_block_) - axis_simd_tail_)
        , unroll_c_tail_(c_block_tail_ / simd_w_) {

        io::io_conf_t io_conf;
        io::io_tail_conf_t io_tail_conf(simd_w_, axis_simd_tail_,
                tail_opmask_idx, vmm_tail_mask.getIdx(), reg_tmp);
        io::io_emu_bf16_conf_t io_bf16_conf(bf16_emu_zmm_1_idx,
                bf16_emu_zmm_2_idx, bf16_emu_zmm_3_idx, reg_tmp,
                bf16_emu_zmm_4_idx);
        const auto io_isa
                = get_io_isa(isa, utils::one_of(f16, src_d_.data_type()),
                        utils::one_of(bf16, src_d_.data_type()));
        io_ = io::jit_io_multi_dt_helper_t<Vmm>(this, io_isa,
                {src_d_.data_type(), f32 /* stats */}, io_conf, io_tail_conf,
                io_bf16_conf);
    }
    status_t create_kernel() override { return jit_generator::create_kernel(); }
    void generate() override {
        preamble();

        io_.init_bf16();
        if (axis_simd_tail_) io_.prepare_tail_mask();

#define PARAM_OFF(x) offsetof(ker_args_t, x)
        mov(reg_mean, ptr[reg_param + PARAM_OFF(mean)]);
        if (compute_var_) mov(reg_var, ptr[reg_param + PARAM_OFF(var)]);
        mov(reg_src_start, ptr[reg_param + PARAM_OFF(src)]);
#undef PARAM_OFF

        if (nc_blocks_) {
            xor_(reg_nc_block, reg_nc_block);
            Xbyak::Label c_blk_loop, c_blk_loop_end;
            L(c_blk_loop);
            {

                cmp(reg_nc_block, nc_blocks_);
                je(c_blk_loop_end, T_NEAR);

                // calculate mean
                compute_stat_block(unroll_c_);

                add(reg_src_start,
                        c_block_ * types::data_type_size(src_d_.data_type()));
                add(reg_mean, c_block_ * sizeof(float));
                if (compute_var_) add(reg_var, c_block_ * sizeof(float));
                add(reg_nc_block, 1);

                jmp(c_blk_loop);
            }
            L(c_blk_loop_end);
        }

        if (unroll_c_tail_) {
            compute_stat_block(unroll_c_tail_);
            add(reg_src_start,
                    c_block_tail_ * types::data_type_size(src_d_.data_type()));
            add(reg_mean, c_block_tail_ * sizeof(float));
            if (compute_var_) add(reg_var, c_block_tail_ * sizeof(float));
        }

        if (axis_simd_tail_) compute_stat_block(1, true);

        postamble();
    }

    void operator()(
            const void *src, float *mean, size_t block_size) const override {
        ker_args_t args;
        args.src = src;
        args.mean = mean;
        args.block_size
                = block_size * C_ * types::data_type_size(src_d_.data_type());

        jit_generator::operator()(&args);
    }

    void operator()(const void *src, const float *mean, float *var,
            size_t block_size) const override {
        ker_args_t args;
        args.src = src;
        args.mean = mean;
        args.var = var;
        args.block_size
                = block_size * C_ * types::data_type_size(src_d_.data_type());

        jit_generator::operator()(&args);
    }

protected:
    using Vmm = typename cpu_isa_traits<isa>::Vmm;
    const Xbyak::AddressFrame &vmmword = (isa == sse41) ? xword
            : (isa == avx2)                             ? yword
                                                        : zword;
    const int vlen = cpu_isa_traits<isa>::vlen;

    struct ker_args_t {
        const void *src;
        const float *mean;
        const float *var;
        size_t block_size;
    };

    io::jit_io_multi_dt_helper_t<Vmm> io_;
    const memory_desc_wrapper src_d_;
    bool compute_var_;
    const dim_t C_;
    const size_t simd_w_;
    const dim_t axis_simd_tail_;
    const dim_t unroll_c_;
    const dim_t c_block_;
    const dim_t nc_blocks_;
    const dim_t c_block_tail_;
    const dim_t unroll_c_tail_;

    void compute_mean_block(size_t unroll, bool tail = false) {
        const size_t c_src_size
                = C_ * types::data_type_size(src_d_.data_type());
#define PARAM_OFF(x) offsetof(ker_args_t, x)
        mov(reg_sp_block_end, ptr[reg_param + PARAM_OFF(block_size)]);
#undef PARAM_OFF
        for (size_t ur = 0; ur < unroll; ur++) {
            uni_vpxor(Vmm_mean(ur), Vmm_mean(ur), Vmm_mean(ur));
        }

        mov(reg_src, reg_src_start);
        // add block_start to block_size to define block_end
        add(reg_sp_block_end, reg_src);

        Xbyak::Label sp_blk_loop, sp_blk_loop_end;
        L(sp_blk_loop);
        {
            cmp(reg_sp_block_end, reg_src);
            jle(sp_blk_loop_end, T_NEAR);

            for (size_t ur = 0; ur < unroll; ur++) {
                io_[src_d_.data_type()]->load(
                        src_ptr(ur * simd_w_), vmm_src, tail);
                uni_vaddps(Vmm_mean(ur), Vmm_mean(ur), vmm_src);
            }

            add(reg_src, c_src_size);
            jmp(sp_blk_loop);
        }
        L(sp_blk_loop_end);

        for (size_t ur = 0; ur < unroll; ur++) {
            io_[data_type::f32]->store(
                    Vmm_mean(ur), mean_ptr(ur * simd_w_), tail);
        }
    }

    void compute_var_block(size_t unroll, bool tail = false) {
        const size_t c_src_size
                = C_ * types::data_type_size(src_d_.data_type());
#define PARAM_OFF(x) offsetof(ker_args_t, x)
        mov(reg_sp_block_end, ptr[reg_param + PARAM_OFF(block_size)]);
#undef PARAM_OFF
        for (size_t ur = 0; ur < unroll; ur++) {
            uni_vpxor(Vmm_var(ur), Vmm_var(ur), Vmm_var(ur));
            io_[data_type::f32]->load(
                    mean_ptr(ur * simd_w_), Vmm_mean(ur), tail);
        }

        mov(reg_src, reg_src_start);
        // add block_start to block_size to define block_end
        add(reg_sp_block_end, reg_src);

        Xbyak::Label sp_blk_loop, sp_blk_loop_end;
        L(sp_blk_loop);
        {
            cmp(reg_sp_block_end, reg_src);
            jle(sp_blk_loop_end, T_NEAR);

            for (size_t ur = 0; ur < unroll; ur++) {
                io_[src_d_.data_type()]->load(
                        src_ptr(ur * simd_w_), vmm_src, tail);
                uni_vsubps(vmm_src, vmm_src, Vmm_mean(ur));
                uni_vfmadd231ps(Vmm_var(ur), vmm_src, vmm_src);
            }

            add(reg_src, c_src_size);
            jmp(sp_blk_loop);
        }
        L(sp_blk_loop_end);

        for (size_t ur = 0; ur < unroll; ur++) {
            io_[data_type::f32]->store(
                    Vmm_var(ur), var_ptr(ur * simd_w_), tail);
        }
    }
    void compute_stat_block(size_t unroll, bool tail = false) {
        if (compute_var_)
            compute_var_block(unroll, tail);
        else
            compute_mean_block(unroll, tail);
    }

    Vmm Vmm_mean(size_t ur = 0) { return Vmm(3 + ur); }
    Vmm Vmm_var(size_t ur = 0) { return Vmm(9 + ur); }

    Xbyak::Address src_ptr(size_t offt = 0) {
        return vmmword[reg_src + offt * src_d_.data_type_size()];
    }

    Xbyak::Address mean_ptr(size_t offt = 0) {
        return vmmword[reg_mean + offt * sizeof(float)];
    }

    Xbyak::Address var_ptr(size_t offt = 0) {
        return vmmword[reg_var + offt * sizeof(float)];
    }

    const Xbyak::Reg64 reg_param = abi_param1;
    const Xbyak::Reg64 reg_src = rdx;
    const Xbyak::Reg64 reg_src_start = rax;
    const Xbyak::Reg64 reg_mean = rbx;
    const Xbyak::Reg64 reg_sp_block_end = r9;
    const Xbyak::Reg64 reg_nc_block = r10;
    const Xbyak::Reg64 reg_tmp = r11;
    const Xbyak::Reg64 reg_var = r12;

    const Vmm vmm_tail_mask = Vmm(0);
    const Vmm vmm_zero = Vmm(1);
    const Vmm vmm_src = Vmm(2);
    const Vmm vmm_tmp = Vmm(15);
    const Xbyak::Xmm xmm_tmp = Xbyak::Xmm(15);

    const int bf16_emu_zmm_1_idx = 28;
    const int bf16_emu_zmm_2_idx = 29;
    const int bf16_emu_zmm_3_idx = 30;
    const int bf16_emu_zmm_4_idx = 31;
    const int tail_opmask_idx = 1;
};

template struct kernel_stat_t<avx2>;
template struct kernel_stat_t<avx512_core>;

} // namespace

jit_uni_group_normalization_fwd_t::kernel_base_t *
jit_uni_group_normalization_fwd_t::kernel_base_t::create(
        const group_normalization_pd_t *pd) {
    if (mayiuse(avx512_core)) {
        return new kernel_t<avx512_core>(pd);
    } else if (mayiuse(avx2)) {
        return new kernel_t<avx2>(pd);
    } else {
        assert(!"kernel is empty.");
        return nullptr;
    }
}

jit_uni_group_normalization_fwd_t::kernel_stat_base_t *
jit_uni_group_normalization_fwd_t::kernel_stat_base_t::create(
        const group_normalization_pd_t *apd, bool compute_var) {
    if (mayiuse(avx512_core)) {
        return new kernel_stat_t<avx512_core>(apd, compute_var);
    } else if (mayiuse(avx2)) {
        return new kernel_stat_t<avx2>(apd, compute_var);
    } else {
        assert(!"kernel is empty.");
        return nullptr;
    }
}

status_t jit_uni_group_normalization_fwd_t::execute_forward(
        const exec_ctx_t &ctx) const {
    using namespace memory_tracking::names;

    const auto src = CTX_IN_MEM(const void *, DNNL_ARG_SRC);
    auto dst = CTX_OUT_MEM(void *, DNNL_ARG_DST);

    auto scale = CTX_IN_MEM(const float *, DNNL_ARG_SCALE);
    auto shift = CTX_IN_MEM(const float *, DNNL_ARG_SHIFT);

    auto scratchpad = ctx.get_scratchpad_grantor();
    auto stat_reduction = scratchpad.template get<float>(key_gnorm_reduction);
    auto tmp_mean = scratchpad.template get<float>(key_gnorm_tmp_mean);
    auto tmp_var = scratchpad.template get<float>(key_gnorm_tmp_var);

    float *mean {nullptr}, *variance {nullptr};
    mean = pd()->stats_is_src()
            ? const_cast<float *>(CTX_IN_MEM(const float *, DNNL_ARG_MEAN))
            : pd()->is_training() ? CTX_OUT_MEM(float *, DNNL_ARG_MEAN)
                                  : tmp_mean;
    variance = pd()->stats_is_src()
            ? const_cast<float *>(CTX_IN_MEM(const float *, DNNL_ARG_VARIANCE))
            : pd()->is_training() ? CTX_OUT_MEM(float *, DNNL_ARG_VARIANCE)
                                  : tmp_var;

    DEFINE_ARG_SCALES_BUFFER(src_scales, DNNL_ARG_SRC);
    DEFINE_ARG_SCALES_BUFFER(dst_scales, DNNL_ARG_DST);

    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper dst_d(pd()->dst_md());

    const dim_t N = src_d.dims()[0];
    const dim_t C_padded = src_d.padded_dims()[1];
    const dim_t C = src_d.dims()[1];
    const dim_t D = pd()->D();
    const dim_t H = pd()->H();
    const dim_t W = pd()->W();
    const dim_t C_PER_G = 1;
    const dim_t SP = D * H * W;

    const bool calculate_stats = !pd()->stats_is_src();
    const int nthr = pd()->nthr_;

    if (calculate_stats) {
        auto reduce = [&](float *stat, const float *tmp_stat) {
            for (dim_t n = 0; n < N; ++n) {
                const float *loc_stat = tmp_stat + n * nthr * C;
                for (dim_t c = 0; c < C; ++c)
                    stat[c] = loc_stat[c];

                for (int ithr_sp = 1; ithr_sp < nthr; ++ithr_sp) {
                    loc_stat += C;
                    for (dim_t c = 0; c < C; ++c)
                        stat[c] += loc_stat[c];
                }

                for (dim_t c = 0; c < C; ++c)
                    stat[c] /= C_PER_G * SP;
                stat += C;
            }
        };
        // compute mean
        parallel(nthr, [&](const int ithr, const int nthr) {
            dim_t SP_start = 0, SP_end = 0;
            balance211(SP, nthr, ithr, SP_start, SP_end);
            const int block_size = SP_end - SP_start;
            for (int n = 0; n < N; ++n) {
                float *local_mean = stat_reduction + n * nthr * C + ithr * C;
                const size_t s_off
                        = (size_t)n * SP * C_padded + SP_start * C_padded;
                const char *__restrict local_src
                        = static_cast<const char *>(src)
                        + s_off * src_d.data_type_size();
                (*kernel_mean_)(local_src, local_mean, block_size);
            }
        });
        reduce(mean, stat_reduction);
        // compute variance
        parallel(nthr, [&](const int ithr, const int nthr) {
            dim_t SP_start = 0, SP_end = 0;
            balance211(SP, nthr, ithr, SP_start, SP_end);
            const dim_t block_size = SP_end - SP_start;
            for (dim_t n = 0; n < N; ++n) {
                float *local_mean = mean + n * C;
                float *local_var = stat_reduction + n * nthr * C + ithr * C;
                const size_t s_off
                        = (size_t)n * SP * C_padded + SP_start * C_padded;
                const char *__restrict local_src
                        = static_cast<const char *>(src)
                        + s_off * src_d.data_type_size();
                (*kernel_var_)(local_src, local_mean, local_var, block_size);
            }
        });
        reduce(variance, stat_reduction);
    }

    parallel(nthr, [&](const int ithr, const int nthr) {
        dim_t SP_start = 0, SP_end = 0;
        balance211(SP, nthr, ithr, SP_start, SP_end);
        for (dim_t n = 0; n < N; ++n) {
            const size_t data_off = n * SP * C_padded + SP_start * C_padded;
            const char *const __restrict src_ptr
                    = reinterpret_cast<const char *>(src)
                    + data_off * src_d.data_type_size();
            char *const __restrict dst_ptr = reinterpret_cast<char *>(dst)
                    + data_off * dst_d.data_type_size();
            const dim_t block_size = SP_end - SP_start;

            (*kernel_)(src_ptr, dst_ptr, scale, shift, &mean[n * C_padded],
                    &variance[n * C_padded], src_scales, dst_scales,
                    block_size);
        }
    });

    return status::success;
}

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
