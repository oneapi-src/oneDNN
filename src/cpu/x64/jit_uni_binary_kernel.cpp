/*******************************************************************************
* Copyright 2022-2023 Intel Corporation
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

#include "cpu/x64/jit_avx512_core_bf16cvt.hpp"
#include "cpu/x64/jit_uni_binary_kernel.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

#define PARAM_OFF(x) offsetof(jit_binary_call_s, x)

static bcast_set_t get_supported_postops_bcast_strategies() {
    return {broadcasting_strategy_t::scalar, broadcasting_strategy_t::per_oc,
            broadcasting_strategy_t::per_oc_spatial,
            broadcasting_strategy_t::no_broadcast};
}

static bool is_xf16(const data_type_t dtype) {
    return utils::one_of(dtype, data_type::bf16, data_type::f16);
}
static bool is_ne_xf16_supported(cpu_isa_t isa, const data_type_t dtype) {
    return isa == avx2_vnni_2 && is_xf16(dtype);
}

binary_kernel_t::binary_kernel_t(const size_t vlen, const binary_pd_t *pd,
        const jit_binary_conf_t conf, const char *name, bool tail_kernel)
    : jit_generator(name)
    , vlen_(vlen)
    , simd_w_(vlen / sizeof(float))
    , pd_(pd)
    , conf_(conf)
    , is_tail_kernel_(tail_kernel)
    , is_src1_outer_dims_tail_(
              conf_.is_src_different_layouts && conf_.outer_dims % simd_w_)
    , tail_size_(get_tail_size())
    , padding_tail_size_(
              pd->src_md(0)->padded_dims[1] - pd->src_md(0)->dims[1]) {}

size_t binary_kernel_t::get_tail_size() const {
    memory_desc_wrapper src0_d(pd_->src_md(0));
    const auto &dims = src0_d.dims();
    const auto &ndims = src0_d.ndims();

    dim_t nelems = 0;

    if (ndims == 1)
        nelems = dims[0];
    else if (is_src1_outer_dims_tail_)
        nelems = conf_.outer_dims;
    else if (!conf_.is_i8 && conf_.op_type == op_t::c_blocked
            && (is_tail_kernel_ || conf_.bcast_type == bcast_t::per_w))
        nelems = dims[1];
    else if (conf_.bcast_type == bcast_t::none
            && !conf_.postops_per_oc_broadcast_exists)
        nelems = src0_d.nelems(true);
    else if (conf_.bcast_type == bcast_t::per_batch
            && !conf_.postops_per_oc_broadcast_exists)
        nelems = src0_d.nelems(true) / dims[0];
    else {
        if (conf_.op_type == op_t::n_spatial_c)
            nelems = dims[1];
        else if (conf_.op_type == op_t::n_c_spatial && ndims >= 3)
            nelems = conf_.bcast_type == bcast_t::per_w
                    ? utils::array_product(
                            dims + (ndims - conf_.not_bcasted_sp_dims),
                            conf_.not_bcasted_sp_dims)
                    : utils::array_product(dims + 2, ndims - 2);
    }
    // it's float due to for bfloat16 we still load 16 elements, not 32.
    return nelems % simd_w_;
}

template <cpu_isa_t isa, typename Vmm>
jit_uni_binary_kernel_t<isa, Vmm>::jit_uni_binary_kernel_t(
        const binary_pd_t *pd, const jit_binary_conf_t conf, bool tail_kernel)
    : binary_kernel_t(vreg_traits<Vmm>::vlen, pd, conf, jit_name(), tail_kernel)
    , offt_src0_(vlen_ / ((conf_.is_bf16 || conf_.is_f16) ? 2 : 1))
    , offt_src1_(conf_.use_stride_src1 ? offt_src0_ : 0)
    , io_(this, isa, {conf_.src0_type, conf_.src1_type, conf_.dst_type},
              {false},
              io::io_tail_conf_t {simd_w_, tail_size_, tail_opmask_,
                      vmm_tail_vmask_.getIdx(), reg_tmp_},
              io::io_emu_bf16_conf_t {vreg_bf16_emu_1_, vreg_bf16_emu_2_,
                      vreg_bf16_emu_3_, reg_tmp_, vreg_bf16_emu_4_},
              create_saturation_vmm_map(),
              io::io_gather_conf_t {simd_w_, full_mask_,
                      vmm_full_mask_.getIdx(), reg_tmp_, reg_tmp1_,
                      vmm_tmp_gather_.getIdx()}) {
    init();
}

template <cpu_isa_t isa, typename Vmm>
std::map<data_type_t, io::io_saturation_conf_t>
jit_uni_binary_kernel_t<isa, Vmm>::create_saturation_vmm_map() const {

    std::map<data_type_t, io::io_saturation_conf_t> saturation_map {};

    if (conf_.is_i8 || conf_.dst_type == data_type::s32)
        saturation_map.emplace(conf_.dst_type,
                io::io_saturation_conf_t {vreg_zero_.getIdx(),
                        vreg_saturation_ubound_.getIdx(), reg_tmp_});

    return saturation_map;
}

template <cpu_isa_t isa, typename Vmm>
void jit_uni_binary_kernel_t<isa, Vmm>::init() {
    if (conf_.with_postops) init_post_ops_injector();
}

template <cpu_isa_t isa, typename Vmm>
void jit_uni_binary_kernel_t<isa, Vmm>::init_post_ops_injector() {
    const memory_desc_wrapper dst_d(pd_->dst_md(0));
    const auto &po = pd_->attr()->post_ops_;

    const eltwise_injector::static_params_t esp(true /*save_state*/,
            reg_elt_inj_table_, elt_inj_opmask_, true /*is_fwd*/,
            false /*use_dst*/);
    const binary_injector::rhs_arg_static_params_t rhs_arg_bsp {10, reg_tmp_,
            reg_elt_inj_table_, r13, true /*preserve gpr*/,
            true /*preserve vmm*/, PARAM_OFF(post_ops_binary_rhs_arg_vec),
            PARAM_OFF(dst_orig), dst_d, tail_size_, tail_opmask_,
            false /*use_exact_tail_scalar_bcast*/};
    const binary_injector::static_params_t bsp(this->param1,
            get_supported_postops_bcast_strategies(), rhs_arg_bsp);

    postops_injector_ = utils::make_unique<
            injector::jit_uni_postops_injector_t<inject_isa, Vmm>>(
            this, po, bsp, esp);
}

template <cpu_isa_t isa, typename Vmm>
void jit_uni_binary_kernel_t<isa, Vmm>::apply_postops(int unroll, bool tail) {
    const auto sum_ne_xf16_injector = [&]() {
        const Vmm vreg_dst_even_tmp = conf_.is_src_different_layouts
                ? vmm_gathered_src_
                : Vmm(unroll + vmm_start_idx_);
        const Vmm vreg_dst_odd_tmp = Vmm(unroll + 1 + vmm_start_idx_);
        const Vmm vmm_aux = vreg_saturation_ubound_;
        for (int i = 0; i < unroll; i += 2) {
            const bool can_load_two_simdw = unroll - i >= 2;
            const int offt_base = simd_w_ * i;
            const Vmm vreg_tmp_even_src0 = Vmm(i + vmm_start_idx_);
            const Vmm vreg_tmp_odd_src0 = Vmm(i + 1 + vmm_start_idx_);
            if (can_load_two_simdw) {
                io_.at(conf_.dst_type)
                        ->load_two_simdw_xf16(dst_ptr(offt_base
                                                      * types::data_type_size(
                                                              conf_.dst_type)),
                                vreg_dst_even_tmp, vreg_dst_odd_tmp);
                io_.at(conf_.dst_type)
                        ->merge_interleaved_to_plain(
                                vreg_dst_even_tmp, vreg_dst_odd_tmp, vmm_aux);
            } else
                io_.at(conf_.dst_type)
                        ->load(dst_ptr(offt_base
                                       * types::data_type_size(conf_.dst_type)),
                                vreg_dst_even_tmp, false);
            uni_vfmadd231ps(
                    vreg_tmp_even_src0, vreg_dst_even_tmp, vreg_sum_scale_);
            if (can_load_two_simdw)
                uni_vfmadd231ps(
                        vreg_tmp_odd_src0, vreg_dst_odd_tmp, vreg_sum_scale_);
        }
    };

    const auto sum_injector = [&]() {
        for (int i = 0; i < unroll; i++) {
            const int offt = simd_w_ * i;
            const Vmm vreg_tmp_src0 = Vmm(i + vmm_start_idx_);
            const Vmm vreg_tmp = conf_.is_src_different_layouts
                    ? vmm_gathered_src_
                    : Vmm(unroll + i + vmm_start_idx_);
            io_.at(conf_.dst_type)
                    ->load(dst_ptr(offt
                                   * types::data_type_size(conf_.dst_type)),
                            vreg_tmp, tail);
            uni_vfmadd231ps(vreg_tmp_src0, vreg_tmp, vreg_sum_scale_);
        }
    };

    if (conf_.do_sum) {
        if (is_ne_xf16_supported(isa, conf_.dst_type) && !tail)
            postops_injector_->set_lambda_injector(
                    primitive_kind::sum, sum_ne_xf16_injector);
        else
            postops_injector_->set_lambda_injector(
                    primitive_kind::sum, sum_injector);
    }

    if (conf_.with_binary) {
        binary_injector::rhs_arg_dynamic_params_t rhs_arg_params;
        const Reg64 &reg_offt_dst
                = conf_.is_i8 ? reg_offt_dst_ : reg_offt_src0_;

        const injector_utils::register_preserve_guard_t register_guard {
                this, {reg_tmp1_}};

        mov(reg_tmp1_, reg_dst_);
        add(reg_tmp1_, reg_offt_dst);

        for (int vmm_idx = 1; vmm_idx < unroll + vmm_start_idx_; vmm_idx++) {
            rhs_arg_params.vmm_idx_to_out_reg.emplace(vmm_idx, reg_tmp1_);
            rhs_arg_params.vmm_idx_to_out_elem_off_val.emplace(vmm_idx,
                    (vmm_idx - vmm_start_idx_) * simd_w_
                            * types::data_type_size(conf_.dst_type));
            if (tail) rhs_arg_params.vmm_tail_idx_.emplace(vmm_idx);
        }
        postops_injector_->compute_vector_range(
                1, unroll + vmm_start_idx_, rhs_arg_params);
    } else
        postops_injector_->compute_vector_range(1, unroll + vmm_start_idx_);
}

template <cpu_isa_t isa, typename Vmm>
void jit_uni_binary_kernel_t<isa, Vmm>::load_kernel_params() {
    mov(reg_tmp_, float2int(conf_.sum_scale));
    uni_vmovq(xreg_sum_scale_, reg_tmp_);
    uni_vbroadcastss(vreg_sum_scale_, xreg_sum_scale_);
    if (is_src1_outer_dims_tail_)
        mov(reg_outer_dims_range_,
                ptr[reg_param_ + PARAM_OFF(spat_offt_count)]);
    else
        mov(reg_reverse_spat_offt_,
                ptr[reg_param_ + PARAM_OFF(spat_offt_count)]);
    mov(reg_src0_, ptr[reg_param_ + PARAM_OFF(src0)]);
    mov(reg_src1_, ptr[reg_param_ + PARAM_OFF(src1)]);
    mov(reg_dst_, ptr[reg_param_ + PARAM_OFF(dst)]);
    if (conf_.is_src_different_layouts) {
        mov(reg_tmp_, ptr[reg_param_ + PARAM_OFF(indices)]);
        uni_vmovdqu(vmm_indices_, ptr[reg_tmp_]);

        mov(reg_src1_stride_range_,
                ptr[reg_param_ + PARAM_OFF(src1_stride_range)]);
        mov(reg_reverse_src1_stride_range_, reg_src1_stride_range_);
    }
    if (conf_.do_scale_src0)
        mov(reg_scales_src0_, ptr[reg_param_ + PARAM_OFF(scales_src0)]);
    if (conf_.do_scale_src1)
        mov(reg_scales_src1_, ptr[reg_param_ + PARAM_OFF(scales_src1)]);
}

template <cpu_isa_t isa, typename Vmm>
Address jit_uni_binary_kernel_t<isa, Vmm>::src0_ptr(size_t offt) {
    return vmmword[reg_src0_ + reg_offt_src0_ + offt];
}

template <cpu_isa_t isa, typename Vmm>
Address jit_uni_binary_kernel_t<isa, Vmm>::src1_ptr(size_t offt) {
    return vmmword[reg_src1_ + reg_offt_src1_ + offt];
}

template <cpu_isa_t isa, typename Vmm>
Address jit_uni_binary_kernel_t<isa, Vmm>::dst_ptr(size_t offt) {
    const Reg64 &reg_offt_dst = conf_.is_i8 ? reg_offt_dst_ : reg_offt_src0_;
    return vmmword[reg_dst_ + reg_offt_dst + offt];
}

template <cpu_isa_t isa, typename Vmm>
unsigned int jit_uni_binary_kernel_t<isa, Vmm>::cmp_predicate(alg_kind_t alg) {
    using namespace alg_kind;
    switch (alg) {
        case binary_ge: return _cmp_nlt_us;
        case binary_gt: return _cmp_nle_us;
        case binary_le: return _cmp_le_os;
        case binary_lt: return _cmp_lt_os;
        case binary_eq: return _cmp_eq_oq;
        case binary_ne: return _cmp_neq_uq;
        default: assert(!"not supported operation!"); return -1;
    }
}

template <cpu_isa_t isa, typename Vmm>
void jit_uni_binary_kernel_t<isa, Vmm>::perform_op(
        const Vmm &v0, const Vmm &v1, const Vmm &s_src0, const Vmm &s_src1) {
    using namespace alg_kind;
    const auto alg = pd_->desc()->alg_kind;
    const bool cmp_op = utils::one_of(alg, alg_kind::binary_ge,
            alg_kind::binary_gt, alg_kind::binary_le, alg_kind::binary_lt,
            alg_kind::binary_eq, alg_kind::binary_ne);
    if (conf_.do_scale_src0) uni_vmulps(v0, v0, s_src0);
    if (conf_.do_scale_src1 && offt_src1_ != 0 && !conf_.broadcast_src1_value)
        uni_vmulps(v1, v1, s_src1);

    if (alg == binary_add)
        uni_vaddps(v0, v0, v1);
    else if (alg == binary_mul)
        uni_vmulps(v0, v0, v1);
    else if (alg == binary_max)
        uni_vmaxps(v0, v0, v1);
    else if (alg == binary_min)
        uni_vminps(v0, v0, v1);
    else if (alg == binary_div)
        uni_vdivps(v0, v0, v1);
    else if (alg == binary_sub)
        uni_vsubps(v0, v0, v1);
    else if (cmp_op) {
        const unsigned int predicate = cmp_predicate(alg);
        if (is_avx512) {
            vcmpps(cmp_mask, v0, v1, predicate);
            vmovups(v0 | cmp_mask | T_z, vreg_one_);
        } else {
            uni_vcmpps(v0, v0, v1, predicate);
            uni_vminps(v0, v0, vreg_one_);
        }
    } else
        assert(!"not supported operation!");
}

template <cpu_isa_t isa, typename Vmm>
void jit_uni_binary_kernel_t<isa, Vmm>::prepare_isa_kernel() {
    if (conf_.is_bf16) io_.init_bf16();
    if (tail_size_ > 0) io_.prepare_tail_mask();
    if (conf_.is_src_different_layouts && is_superset(isa, avx2)) {
        io_.init_full_mask();
        io_.prepare_full_mask();
    }
}

template <cpu_isa_t isa, typename Vmm>
void jit_uni_binary_kernel_t<isa, Vmm>::compute_bcast(bool tail) {
    if (conf_.broadcast_src1_value) {
        if (conf_.is_i8)
            uni_vpxor(xreg_bcast_src1_, xreg_bcast_src1_, xreg_bcast_src1_);
        io_.at(conf_.src1_type)->broadcast(src1_ptr(), vreg_bcast_src1_);
    } else if (!conf_.is_i8 && offt_src1_ == 0) {
        io_.at(conf_.src1_type)->load(src1_ptr(), vreg_bcast_src1_, tail);
    }
}

template <cpu_isa_t isa, typename Vmm>
void jit_uni_binary_kernel_t<isa, Vmm>::load_src1(
        const Vmm &vreg_src1, const int offt, bool tail) {
    if (conf_.is_src_different_layouts) {
        // if different layouts, gather data with strides
        // after getting to stride range, offset is restored and
        // increased
        io_.at(conf_.src1_type)
                ->gather(reg_src1_, vmm_indices_, vreg_src1, tail);
        // gather is using register instead of operand to read address
        // use reg_src1_ directly, without offset stored in second
        // register
        add(reg_src1_,
                types::data_type_size(conf_.src1_type) * conf_.src1_stride
                        * simd_w_);
        sub(reg_reverse_src1_stride_range_,
                types::data_type_size(conf_.src1_type) * conf_.src1_stride
                        * simd_w_);

        Label src1_stride_range_not_exceed, src1_C_tail_end;

        cmp(reg_reverse_src1_stride_range_, 0);
        jg(src1_stride_range_not_exceed, T_NEAR);
        {
            pop(reg_src1_);
            add(reg_src1_, types::data_type_size(conf_.src1_type));
            push(reg_src1_);
            mov(reg_reverse_src1_stride_range_, reg_src1_stride_range_);
        }
        L(src1_stride_range_not_exceed);
    } else
        io_.at(conf_.src1_type)
                ->load(src1_ptr(offt * types::data_type_size(conf_.src1_type)),
                        vreg_src1, tail);
}

template <cpu_isa_t isa, typename Vmm>
void jit_uni_binary_kernel_t<isa, Vmm>::store(int unroll, bool tail) {
    for (int i = 0; i < unroll; i++) {
        const Vmm vreg_tmp_src0 = Vmm(i + vmm_start_idx_);
        const int offt = simd_w_ * i;
        const auto dt_size = types::data_type_size(conf_.dst_type);

        if (is_tail_kernel_ && padding_tail_size_) {
            // apply zero-padding
            auto off_base = 0;
            auto zero_pad_left = padding_tail_size_;

            if (zero_pad_left >= simd_w_ - tail_size_) {
                vxorps(vreg_zero_, vreg_zero_, vreg_zero_);
                if (is_avx512)
                    uni_vmovups(vreg_zero_ | tail_opmask_, vreg_tmp_src0);
                else
                    uni_vblendvps(vreg_zero_, vreg_zero_, vreg_tmp_src0,
                            vmm_tail_vmask_);
                io_.at(conf_.dst_type)
                        ->store(vreg_zero_, dst_ptr(offt * dt_size), false);
                off_base = simd_w_ * dt_size;
                zero_pad_left -= simd_w_ - tail_size_;
            } else {
                io_.at(conf_.dst_type)
                        ->store(vreg_tmp_src0, dst_ptr(offt * dt_size), true);
                off_base = tail_size_ * dt_size;
            }

            if (zero_pad_left) {
                push(abi_param1);
                const Reg32 &reg_zero = eax;
                const Reg64 &reg_ptr = rdi;
                const Reg64 &reg_counter = rcx;
                const auto off_start = off_base;
                const auto off_end = off_start + zero_pad_left * dt_size;
                xor_(reg_zero, reg_zero);
                lea(reg_ptr,
                        ptr[dst_ptr(offt * dt_size).getRegExp()
                                + RegExp(off_start)]);
                mov(reg_counter, off_end - off_start);
                rep();
                stosb();
                pop(abi_param1);
            }
        } else
            io_.at(conf_.dst_type)
                    ->store(vreg_tmp_src0, dst_ptr(offt * dt_size), tail);
    }
}

template <cpu_isa_t isa, typename Vmm>
void jit_uni_binary_kernel_t<isa, Vmm>::compute_ne_xf16_dst_body(
        int unroll, bool tail) {
    const Vmm vreg_tmp = conf_.is_src_different_layouts
            ? vmm_gathered_src_
            : Vmm(unroll + vmm_start_idx_);
    const Vmm vreg_tmp_even_src1 = offt_src1_ ? vreg_tmp : vreg_bcast_src1_;
    const Vmm vreg_tmp_odd_src1 = Vmm(unroll + 1 + vmm_start_idx_);
    const Vmm vmm_aux = Vmm(unroll + 2 + vmm_start_idx_);

    for (int i = 0; i < unroll; i += 2) {
        const bool can_load_two_simdw_src0
                = is_xf16(conf_.src0_type) && unroll - i >= 2;
        const bool can_load_two_simdw_src1 = is_xf16(conf_.src1_type)
                && unroll - i >= 2 && offt_src1_
                && !conf_.is_src_different_layouts;
        const Vmm vreg_tmp_even_src0 = Vmm(i + vmm_start_idx_);
        const Vmm vreg_tmp_odd_src0 = Vmm(i + 1 + vmm_start_idx_);
        const int offt_base = simd_w_ * i;

        if (can_load_two_simdw_src0) {
            io_.at(conf_.src0_type)
                    ->load_two_simdw_xf16(
                            src0_ptr(offt_base
                                    * types::data_type_size(conf_.src0_type)),
                            vreg_tmp_even_src0, vreg_tmp_odd_src0);
            io_.at(conf_.src0_type)
                    ->merge_interleaved_to_plain(
                            vreg_tmp_even_src0, vreg_tmp_odd_src0, vmm_aux);
        }

        if (can_load_two_simdw_src1) {
            io_.at(conf_.src1_type)
                    ->load_two_simdw_xf16(
                            src1_ptr(offt_base
                                    * types::data_type_size(conf_.src1_type)),
                            vreg_tmp_even_src1, vreg_tmp_odd_src1);
            io_.at(conf_.src1_type)
                    ->merge_interleaved_to_plain(
                            vreg_tmp_even_src1, vreg_tmp_odd_src1, vmm_aux);
        }

        for (int j = 0; j < 2 && i + j < unroll; j++) {
            const Vmm vreg_tmp_src0
                    = j == 0 ? vreg_tmp_even_src0 : vreg_tmp_odd_src0;
            const Vmm vreg_tmp_src1 = j == 0 || !can_load_two_simdw_src1
                    ? vreg_tmp_even_src1
                    : vreg_tmp_odd_src1;
            const int offt = simd_w_ * j + offt_base;
            if (!can_load_two_simdw_src0)
                io_.at(conf_.src0_type)
                        ->load(src0_ptr(offt
                                       * types::data_type_size(
                                               conf_.src0_type)),
                                vreg_tmp_src0, tail);
            if (offt_src1_ && !can_load_two_simdw_src1)
                load_src1(vreg_tmp_src1, offt, tail);

            // avoid multiple multiplication on input scale for broadcasted vreg
            // not needed for different layouts
            if (!conf_.is_src_different_layouts)
                uni_vmovups(vreg_tmp, vreg_tmp_src1);
            perform_op(vreg_tmp_src0, vreg_tmp, vreg_scales_src0_,
                    vreg_scales_src1_);
        }
    }
}

template <cpu_isa_t isa, typename Vmm>
void jit_uni_binary_kernel_t<isa, Vmm>::compute_dst_body(
        int unroll, bool tail) {
    for (int i = 0; i < unroll; i++) {
        const Vmm vreg_tmp_src0 = Vmm(i + vmm_start_idx_);
        const Vmm vreg_tmp = conf_.is_src_different_layouts
                ? vmm_gathered_src_
                : Vmm(unroll + i + vmm_start_idx_);
        const Vmm vreg_tmp_src1 = offt_src1_ ? vreg_tmp : vreg_bcast_src1_;
        const int offt = simd_w_ * i;
        io_.at(conf_.src0_type)
                ->load(src0_ptr(offt * types::data_type_size(conf_.src0_type)),
                        vreg_tmp_src0, tail);
        if (offt_src1_) load_src1(vreg_tmp_src1, offt, tail);

        // avoid multiple multiplication on input scale for broadcasted vreg
        // not needed for different layouts
        if (!conf_.is_src_different_layouts)
            uni_vmovups(vreg_tmp, vreg_tmp_src1);
        perform_op(
                vreg_tmp_src0, vreg_tmp, vreg_scales_src0_, vreg_scales_src1_);
    }
}

template <cpu_isa_t isa, typename Vmm>
void jit_uni_binary_kernel_t<isa, Vmm>::compute_dst(int unroll, bool tail) {
    // When src1 supports but src0 does not support ne convert instructions
    // we only call compute_ne_xf16_dst_body() when loading src1 is needed
    if (!tail
            && IMPLICATION(is_xf16(conf_.src1_type),
                    offt_src1_ && !conf_.is_src_different_layouts)
            && (is_ne_xf16_supported(isa, conf_.src0_type)
                    || is_ne_xf16_supported(isa, conf_.src1_type)))
        compute_ne_xf16_dst_body(unroll, tail);
    else
        compute_dst_body(unroll, tail);

    if (postops_injector_) apply_postops(unroll, tail);

    store(unroll, tail);
}

template <cpu_isa_t isa, typename Vmm>
void jit_uni_binary_kernel_t<isa, Vmm>::forward() {
    Label unroll_loop, unroll_loop_tail, nelems_tail, end;

    const auto src0_type_size = types::data_type_size(conf_.src0_type);
    const auto src1_type_size = types::data_type_size(conf_.src1_type);
    const auto dst_type_size = types::data_type_size(conf_.dst_type);

    if (conf_.is_src_different_layouts) push(reg_src1_);

    // if outer dims tail, do it outside outer dims loop
    if (!is_src1_outer_dims_tail_) {
        if (conf_.is_i8 || conf_.dst_type == data_type::s32) {
            uni_vpxor(vreg_zero_, vreg_zero_, vreg_zero_);
            io_.init_saturate_f32({conf_.dst_type});
            xor_(reg_offt_dst_, reg_offt_dst_); // offt_dst to get addr of dst
        }

        xor_(reg_offt_src0_,
                reg_offt_src0_); // offt_src0 to get addr of src0/dst
        if (!conf_.is_src_different_layouts)
            xor_(reg_offt_src1_,
                    reg_offt_src1_); // offt_src1 to get addr of src1
        if (conf_.use_stride_rhs_postops && !conf_.is_i8)
            xor_(reg_off_rhs_postops_, reg_off_rhs_postops_);
    }
    const auto alg = pd_->desc()->alg_kind;

    if (utils::one_of(alg, alg_kind::binary_ge, alg_kind::binary_gt,
                alg_kind::binary_le, alg_kind::binary_lt, alg_kind::binary_eq,
                alg_kind::binary_ne)) {
        Xmm xreg_one = Xmm(vreg_one_.getIdx());
        mov(reg_tmp_, float2int(1));
        uni_vmovq(xreg_one, reg_tmp_);
        uni_vbroadcastss(vreg_one_, xreg_one);
    }

    compute_bcast(false); // bcast/load vreg just one time per a kernel call

    // used in c_blocked strategy for last blocked if tail exists
    const bool treat_each_compute_step_as_tail
            = !conf_.is_i8 && is_tail_kernel_ && tail_size_;

    if (conf_.do_scale_src0)
        uni_vbroadcastss(vreg_scales_src0_, ptr[reg_scales_src0_]);
    if (conf_.do_scale_src1) {
        uni_vbroadcastss(vreg_scales_src1_, ptr[reg_scales_src1_]);
        if (conf_.broadcast_src1_value || offt_src1_ == 0)
            uni_vmulps(vreg_bcast_src1_, vreg_bcast_src1_, vreg_scales_src1_);
    }

    L(unroll_loop);
    {
        const size_t offt = unroll_regs_ * simd_w_;
        cmp(reg_reverse_spat_offt_, offt * dst_type_size);
        jl(unroll_loop_tail, T_NEAR);

        compute_dst(unroll_regs_, treat_each_compute_step_as_tail);
        sub(reg_reverse_spat_offt_, offt * dst_type_size);
        add(reg_offt_src0_, offt * src0_type_size);
        if (conf_.is_i8) {
            if (!conf_.broadcast_src1_value && !conf_.is_src_different_layouts)
                add(reg_offt_src1_, offt * src1_type_size);
            add(reg_offt_dst_, offt);
        } else {
            if (conf_.use_stride_src1 && !conf_.is_src_different_layouts)
                add(reg_offt_src1_, offt * src1_type_size);
            if (conf_.use_stride_rhs_postops) add(reg_off_rhs_postops_, offt);
        }
        jmp(unroll_loop);
    }

    L(unroll_loop_tail);
    {
        cmp(reg_reverse_spat_offt_, simd_w_ * dst_type_size);
        jl(nelems_tail, T_NEAR);

        compute_dst(1, treat_each_compute_step_as_tail);
        sub(reg_reverse_spat_offt_, simd_w_ * dst_type_size);
        add(reg_offt_src0_, simd_w_ * src0_type_size);
        if (conf_.is_i8) {
            if (!conf_.broadcast_src1_value && !conf_.is_src_different_layouts)
                add(reg_offt_src1_, simd_w_ * src1_type_size);
            add(reg_offt_dst_, simd_w_);
        } else {
            if (conf_.use_stride_src1 && !conf_.is_src_different_layouts)
                add(reg_offt_src1_, simd_w_ * src1_type_size);
            if (conf_.use_stride_rhs_postops)
                add(reg_off_rhs_postops_, simd_w_);
        }

        jmp(unroll_loop_tail);
    }

    L(nelems_tail);
    {
        cmp(reg_reverse_spat_offt_, 1);
        jl(end, T_NEAR);

        compute_dst(1, true);
        // need to increase if forward over outer dims
        if (is_src1_outer_dims_tail_) {
            add(reg_offt_src0_, tail_size_ * src0_type_size);
            if (conf_.is_i8)
                add(reg_offt_dst_, tail_size_);
            else {
                if (conf_.use_stride_rhs_postops)
                    add(reg_off_rhs_postops_, tail_size_);
            }
        }
    }

    L(end);
    if (conf_.is_src_different_layouts) pop(reg_src1_);
}

template <cpu_isa_t isa, typename Vmm>
void jit_uni_binary_kernel_t<isa, Vmm>::forward_over_outer_dims() {
    const auto outer_dims_size
            = conf_.outer_dims * types::data_type_size(conf_.dst_type);

    if (conf_.is_i8 || conf_.dst_type == data_type::s32) {
        uni_vpxor(vreg_zero_, vreg_zero_, vreg_zero_);
        io_.init_saturate_f32({conf_.dst_type});
        xor_(reg_offt_dst_, reg_offt_dst_); // offt_dst to get addr of dst
    }

    xor_(reg_offt_src0_,
            reg_offt_src0_); // offt_src0 to get addr of src0/dst
    if (conf_.use_stride_rhs_postops && !conf_.is_i8)
        xor_(reg_off_rhs_postops_, reg_off_rhs_postops_);

    Label c_loop;
    L(c_loop);
    {
        mov(reg_reverse_spat_offt_, outer_dims_size);
        forward();
        sub(reg_outer_dims_range_, outer_dims_size);
        cmp(reg_outer_dims_range_, 0);
        jg(c_loop);
    }
}

template <cpu_isa_t isa, typename Vmm>
void jit_uni_binary_kernel_t<isa, Vmm>::generate() {
    preamble();
    load_kernel_params();
    prepare_isa_kernel();
    // if outer dims is not aligned to simd_w, iterate over it to avoid
    // modifying the gather indices
    if (is_src1_outer_dims_tail_)
        forward_over_outer_dims();
    else
        forward();
    postamble();

    if ((conf_.with_eltwise || conf_.is_i8) && postops_injector_)
        postops_injector_->prepare_table();
}

#undef PARAM_OFF

template struct jit_uni_binary_kernel_t<avx512_core_fp16, Zmm>;
template struct jit_uni_binary_kernel_t<avx512_core_fp16, Ymm>;
template struct jit_uni_binary_kernel_t<avx512_core_fp16, Xmm>;
template struct jit_uni_binary_kernel_t<avx512_core_bf16, Zmm>;
template struct jit_uni_binary_kernel_t<avx512_core_bf16, Ymm>;
template struct jit_uni_binary_kernel_t<avx512_core_bf16, Xmm>;
template struct jit_uni_binary_kernel_t<avx512_core, Zmm>;
template struct jit_uni_binary_kernel_t<avx512_core, Ymm>;
template struct jit_uni_binary_kernel_t<avx512_core, Xmm>;
template struct jit_uni_binary_kernel_t<avx2_vnni_2, Ymm>;
template struct jit_uni_binary_kernel_t<avx2_vnni_2, Xmm>;
template struct jit_uni_binary_kernel_t<avx2, Ymm>;
template struct jit_uni_binary_kernel_t<avx2, Xmm>;
template struct jit_uni_binary_kernel_t<sse41, Xmm>;

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
