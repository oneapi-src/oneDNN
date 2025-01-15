/*******************************************************************************
* Copyright 2024-2025 Intel Corporation
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

#include "common/memory_tracking.hpp"

#include "cpu/x64/cpu_isa_traits.hpp"
#include "cpu/x64/jit_avx512_core_scale_precompute.hpp"
#include "cpu/x64/jit_generator.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

using namespace Xbyak;

namespace scale_utils {

const float *precompute_scales(const memory_tracking::grantor_t &scratchpad,
        const float *src_scales, const float *wei_scales, dim_t IC, dim_t OC,
        const bool wei_scale_per_ic, const bool wei_scale_per_oc,
        const primitive_attr_t *attr,
        const jit_avx512_core_scale_precompute_t *const jit_scale_precompute,
        float scale_adjust_factor, bool req_transpose) {

    const float *scales = nullptr;
    const dim_t wei_scale_count
            = (wei_scale_per_ic ? IC : 1) * (wei_scale_per_oc ? OC : 1);
    ;
    if (jit_scale_precompute) {
        const auto &attr_scales = attr->scales_;
        const int wei_scale_mask = attr_scales.get_mask(DNNL_ARG_WEIGHTS);
        size_t size = 0;
        auto loc_scales = scratchpad.template get<float>(
                memory_tracking::names::key_precomputed_scales, &size);
        const dim_t count = nstl::min(
                static_cast<dim_t>(size / sizeof(float)), wei_scale_count);
        const auto wei_scale_stride_ic
                = wei_scale_per_ic ? wei_scale_per_oc ? OC : 1 : 0;
        const auto with_wei_scale
                = !attr_scales.has_default_values(DNNL_ARG_WEIGHTS);
        const auto wei_scale_has_groups = with_wei_scale
                && !attr_scales.get(DNNL_ARG_WEIGHTS).has_default_groups();
        const auto wei_scale_group_stride = wei_scale_has_groups
                ? wei_scale_stride_ic * sizeof(float)
                : 0;

        // JIT run-time params
        jit_call_t jrp(src_scales, wei_scales, loc_scales, count,
                wei_scale_group_stride);

        assert(req_copy_scales(attr, scale_adjust_factor));
        assert(mayiuse(avx512_core));
        assert(wei_scale_mask > 0);
        if (wei_scale_has_groups) {
            assert(count == wei_scale_count);
            const auto wei_scale_groups_ic
                    = attr_scales.get_group(DNNL_ARG_WEIGHTS, 0);
            const dim_t wei_scale_nb_ic = IC / wei_scale_groups_ic;
            const auto wei_scale_dt_sz = types::data_type_size(
                    attr_scales.get_data_type(DNNL_ARG_WEIGHTS));
            for (int nb_ic = 0; nb_ic < wei_scale_nb_ic; nb_ic++) {
                const auto offset = nb_ic * wei_scale_stride_ic;
                jrp.nelems_ = wei_scale_stride_ic;
                jrp.wei_scales_ = (char *)wei_scales + offset * wei_scale_dt_sz;
                jrp.scales_ = &loc_scales[offset * wei_scale_groups_ic];
                (*jit_scale_precompute)(&jrp);
            }
        } else
            (*jit_scale_precompute)(&jrp);

        scales = loc_scales;
        MAYBE_UNUSED(wei_scale_mask);
    } else
        scales = cpu::precompute_scales(scratchpad, src_scales, wei_scales, IC,
                OC, wei_scale_per_ic, wei_scale_per_oc, attr,
                scale_adjust_factor, req_transpose);
    return scales;
}

} // namespace scale_utils

#define GET_OFF(field) offsetof(scale_utils::jit_call_t, field)

void jit_avx512_core_scale_precompute_t::store(
        const int offset_base, const bool compute_tail) {
    mov(reg_aux_dst_scales_, reg_dst_scales_);
    const auto addr_offset = static_cast<size_t>(offset_base) * sizeof(float);
    const Vmm vmm_m_dst = compute_tail ? vmm_dst_ | ktail_f32_mask_ : vmm_dst_;
    for (size_t g = 0; g < wei_groups_ic_; g++) {
        vmovups(ptr[reg_aux_dst_scales_ + addr_offset], vmm_m_dst);
        add(reg_aux_dst_scales_, reg_groups_stride_);
    }
}

void jit_avx512_core_scale_precompute_t::cvt2ps(data_type_t type_in,
        const Vmm vmm_in, const Xbyak::Operand &op,
        const bool mask_flag = false) {
    const Vmm vmm = mask_flag ? vmm_in | ktail_f32_mask_ | T_z : vmm_in;
    switch (type_in) {
        case data_type::f32: vmovups(vmm, op); break;
        case data_type::f16: vcvtph2psx(vmm, op); break;
        case data_type::bf16:
            vpmovzxwd(vmm, op);
            vpslld(vmm_in, vmm_in, 0x10);
            break;
        default: assert(!"unsupported data type");
    }
}

void jit_avx512_core_scale_precompute_t::compute_scale(
        const int offset_base, const bool compute_tail) {
    const size_t wei_addr_offset
            = static_cast<size_t>(offset_base) * wei_scales_dsz_;
    const Vmm vmm_m_wei_scales = compute_tail
            ? vmm_wei_scales_ | ktail_f32_mask_ | T_z
            : vmm_wei_scales_;

    cvt2ps(wei_scales_dt_, vmm_wei_scales_,
            ptr[reg_wei_scales_ + wei_addr_offset], compute_tail);
    if (compute_scale_factor_)
        vmulps(vmm_m_wei_scales, vmm_scale_factor_, vmm_m_wei_scales);

    vmulps(vmm_dst_, vmm_m_wei_scales, ptr_b[reg_src_scales_]);
    store(offset_base, compute_tail);
}

void jit_avx512_core_scale_precompute_t::setup_mask() {
    mov(reg_mask_, 1);
    shl(reg_mask_, reg_tail_.cvt8());
    sub(reg_mask_, 1);
    kmovw(ktail_f32_mask_, reg_mask_);
}

void jit_avx512_core_scale_precompute_t::generate() {

    preamble();

    // get params
    mov(reg_src_scales_, ptr[abi_param1 + GET_OFF(src_scales_)]);
    mov(reg_wei_scales_, ptr[abi_param1 + GET_OFF(wei_scales_)]);
    mov(reg_dst_scales_, ptr[abi_param1 + GET_OFF(scales_)]);
    mov(reg_nelems_, ptr[abi_param1 + GET_OFF(nelems_)]);
    mov(reg_groups_stride_, ptr[abi_param1 + GET_OFF(stride_per_groups_)]);
    if (compute_scale_factor_) {
        const Xmm xmm_scale_factor_(vmm_scale_factor_.getIdx());
        mov(reg_scale_factor_, float2int(scale_adjust_factor_));
        vmovq(xmm_scale_factor_, reg_scale_factor_);
        vbroadcastss(vmm_scale_factor_, xmm_scale_factor_);
    }

    constexpr int n_unroll = 2;
    Xbyak::Label l_simd_loop[n_unroll + 2], l_done;
    for (int i = n_unroll; i >= 0; i--) {
        const int unroll = 1 << i; // 4, 2, 1
        const size_t addr_step = static_cast<size_t>(simd_w_) * unroll;
        L(l_simd_loop[i + 1]);
        {
            cmp(reg_nelems_, addr_step);
            jl(l_simd_loop[i], T_NEAR);
            for (int offset_base = 0; offset_base < unroll; offset_base++) {
                compute_scale(offset_base * simd_w_, false);
            }
            add(reg_wei_scales_, addr_step * wei_scales_dsz_);
            add(reg_dst_scales_, addr_step * sizeof(float));
            sub(reg_nelems_, addr_step);
            jmp(l_simd_loop[i + 1], T_NEAR);
        }
    }
    L(l_simd_loop[0]);

    test(reg_nelems_, reg_nelems_);
    jz(l_done, T_NEAR);

    mov(reg_tail_, reg_nelems_);
    setup_mask();

    compute_scale(0, true);

    L(l_done);

    postamble();
}

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
