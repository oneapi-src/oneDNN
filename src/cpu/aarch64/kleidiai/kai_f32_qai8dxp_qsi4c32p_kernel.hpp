
/*******************************************************************************
* Copyright 2025 Arm Ltd. and affiliates
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed tos in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/
#if defined(DNNL_EXPERIMENTAL_UKERNEL) && defined(DNNL_AARCH64_USE_KAI)

#ifndef CPU_AARCH64_KLEIDIAI_KAI_F32_QAI8DXP_QSI4C32P_KERNEL
#define CPU_AARCH64_KLEIDIAI_KAI_F32_QAI8DXP_QSI4C32P_KERNEL

#include <kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod.h>
#include <kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm.h>
#include <kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp_qsi4c32p_interface.h>
#include <kai/ukernels/matmul/pack/kai_lhs_quant_pack_qai8dxp_f32.h>
#include <kai/ukernels/matmul/pack/kai_rhs_pack_kxn_qsi4c32p_qsu4c32s1s0.h>
#include <kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0.h>
#include <unordered_map>

#include "cpu/aarch64/brgemm/brgemm_types.hpp"
#include "cpu/aarch64/kleidiai/kai_types.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

// Groupwise Kernel mapping
struct kai_f32_qa8dxp_qs4c32p_kernel_packet_t
    : public kernel_kai_common_t<
              kai_matmul_clamp_f32_qai8dxp_qsi4c32p_ukernel> {
public:
    using kernel_kai_common_t::kernel_kai_common_t;

    kai_f32_qa8dxp_qs4c32p_kernel_packet_t() = default;

    kai_f32_qa8dxp_qs4c32p_kernel_packet_t(brgemm_desc_t desc) {
        desc_ = desc;

        if (desc_.M == 1)
            ukernel_id_ = kai_ukernel_id::f32_qai8dxp_qsi4c32p_gemv;
        else
            ukernel_id_ = kai_ukernel_id::f32_qai8dxp_qsi4c32p_gemm;

        ukernel_ = groupwiseKernels_.at(ukernel_id_);
        init_ukernel_params();
    }

    kai_f32_qa8dxp_qs4c32p_kernel_packet_t(const kai_ukernel_id id) {
        ukernel_id_ = id;
        ukernel_ = groupwiseKernels_.at(ukernel_id_);
        init_ukernel_params();
    }

    kai_f32_qa8dxp_qs4c32p_kernel_packet_t(
            const kai_f32_qa8dxp_qs4c32p_kernel_packet_t *kernel) {
        ukernel_id_ = kernel->get_ukernel_id();
        ukernel_ = groupwiseKernels_.at(ukernel_id_);
        desc_ = kernel->get_desc();
        init_ukernel_params();
    }

    size_t get_lhs_packed_size() const {
        return kai_get_lhs_packed_size(desc_.M, desc_.K, mr_, kr_, sr_);
    };

    size_t get_lhs_packed_size(uint64_t M, uint64_t K) const {
        return kai_get_lhs_packed_size(M, K, mr_, kr_, sr_);
    };

    size_t get_lhs_packed_offset(uint64_t m_idx) const {
        return ukernel_.get_lhs_packed_offset(m_idx, desc_.K);
    };

    size_t get_rhs_packed_size() const {
        return kai_get_rhs_nxk_packed_size(
                desc_.N, desc_.K, nr_, kr_, sr_, desc_.BL, dt_packed_rhs_);
    }

    size_t get_rhs_packed_size(uint64_t N, uint64_t K, uint64_t BL = 32) const {
        return kai_get_rhs_nxk_packed_size(
                N, K, nr_, kr_, sr_, BL, dt_packed_rhs_);
    };

    size_t get_rhs_packed_offset(uint64_t n_idx) const {
        return ukernel_.get_rhs_packed_offset(n_idx, desc_.K, desc_.BL);
    };

    size_t get_rhs_packed_offset(
            uint64_t n_idx, uint64_t K, uint64_t BL = 32) const {
        return ukernel_.get_rhs_packed_offset(n_idx, K, BL);
    };

    void set_rhs_pack_param(uint64_t lhs_zp, uint64_t rhs_zp,
            kai_datatype scale_dt = kai_datatype::kai_dt_unknown) {
        rhs_pack_params_.lhs_zero_point = lhs_zp;
        rhs_pack_params_.rhs_zero_point = rhs_zp;

        if (scale_dt != kai_datatype::kai_dt_unknown)
            rhs_pack_params_.scale_dt = scale_dt;
    };

    void run_lhs_quant_pack(const float *lhs_ptr, void *lhs_quant_packed_ptr,
            dim_t m_idx = 0, stride_t lhs_stride = 0) const {
        if (lhs_stride == 0) lhs_stride = desc_.K * sizeof(float);

        size_t offset_lhs_quant_pack
                = ukernel_.get_lhs_packed_offset(m_idx, desc_.K);
        void *lhs_quant_packed_offset_ptr
                = (void *)((uint8_t *)lhs_quant_packed_ptr
                        + offset_lhs_quant_pack);

        kai_run_lhs_quant_pack(desc_.M, desc_.K, mr_, kr_, sr_, m_idx, lhs_ptr,
                lhs_stride, lhs_quant_packed_offset_ptr);
    };

    void run_rhs_pack(const uint8_t *rhs, const float *scales,
            const float *bias, uint8_t *rhs_packed, uint64_t num_groups = 1,
            size_t extra_bytes = 0, stride_t rhs_stride = 0,
            stride_t scales_stride = 0) const {
        if (rhs_stride == 0) { rhs_stride = kai_roundup(desc_.K, 2) / 2; }

        if (scales_stride == 0) {
            scales_stride = (kai_roundup(desc_.K, desc_.BL) / desc_.BL)
                    * sizeof(uint16_t);
        }

        kai_run_rhs_nxk_pack(num_groups, desc_.N, desc_.K, nr_, kr_, sr_,
                desc_.BL, rhs, rhs_stride, bias, scales, scales_stride,
                rhs_packed, extra_bytes,
                (kai_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0_params
                                *)&rhs_pack_params_);
    };

    void operator()(brgemm_kernel_params_t *kernel_params) const override {
        const brgemm_batch_element_t *brgemm_be_vec_ptr = kernel_params->batch;
        for (uint64_t bs_idx = 0; bs_idx < kernel_params->BS; bs_idx++) {
            execute(kernel_params->ptr_A, kernel_params->ptr_B,
                    kernel_params->ptr_C, brgemm_be_vec_ptr[bs_idx].offset.A,
                    brgemm_be_vec_ptr[bs_idx].offset.B);
        }
    };

    std::string to_string() const override {
        if (ukernel_id_ == kai_ukernel_id::f32_qai8dxp_qsi4c32p_gemv)
            return "kai_matmul_clamp_f32_qai8dxp_qsi4c32p_ukernel_gemv";
        else
            return "kai_matmul_clamp_f32_qai8dxp_qsi4c32p_ukernel_gemm";
    }

    void execute(const void *lhs_packed_base_ptr,
            const void *rhs_packed_base_ptr, void *dst_base_ptr,

            int64_t m_idx = -1, int64_t n_idx = -1, stride_t lhs_stride = 0,
            stride_t dst_stride = 0) const {
        // if 'offsets' are -1 compute MxN matrix,
        // otherwise the user has tiled based on one ukernel execution
        // with 1 ukernel execution per run (m_step x n_step output).
        uint64_t ms_per_run = (m_idx == -1)
                ? desc_.M
                : std::min(desc_.M - m_idx, (int64_t)m_step_);
        uint64_t ns_per_run = (n_idx == -1)
                ? desc_.N
                : std::min(desc_.N - n_idx, (int64_t)n_step_);
        m_idx = (m_idx == -1) ? 0 : m_idx;
        n_idx = (n_idx == -1) ? 0 : n_idx;

        if (dst_stride == 0) dst_stride = desc_.N * sizeof(float);

        const void *lhs_offset_packed_ptr
                = (const void *)((uint8_t *)lhs_packed_base_ptr
                        + ukernel_.get_lhs_packed_offset(m_idx, desc_.K));
        const void *rhs_offset_packed_ptr
                = (const void *)((uint8_t *)rhs_packed_base_ptr
                        + ukernel_.get_rhs_packed_offset(
                                n_idx, desc_.K, desc_.BL));
        float *dst_offset_ptr = (float *)((float *)dst_base_ptr
                + ukernel_.get_dst_offset(m_idx, n_idx, dst_stride));

        ukernel_.run_matmul(ms_per_run, ns_per_run, desc_.K, desc_.BL,
                lhs_offset_packed_ptr, rhs_offset_packed_ptr, dst_offset_ptr,
                dst_stride, sizeof(float), -FLT_MAX, FLT_MAX);
    };

private:
    brgemm_desc_t desc_;
    kai_datatype dt_packed_rhs_ = kai_datatype::kai_dt_bf16;
    kai_rhs_pack_param rhs_pack_params_;

    std::unordered_map<kai_ukernel_id, kai_matmul_clamp_f32_qai8dxp_qsi4c32p_ukernel> groupwiseKernels_ = {
            {kai_ukernel_id::f32_qai8dxp_qsi4c32p_gemv,
                    {kai_get_m_step_matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod,
                            kai_get_n_step_matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod,
                            kai_get_mr_matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod,
                            kai_get_nr_matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod,
                            kai_get_kr_matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod,
                            kai_get_sr_matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod,
                            kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod,
                            kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod,
                            kai_get_dst_offset_matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod,
                            kai_get_dst_size_matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod,
                            kai_run_matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod}},
            {kai_ukernel_id::f32_qai8dxp_qsi4c32p_gemm,
                    {kai_get_m_step_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm,
                            kai_get_n_step_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm,
                            kai_get_mr_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm,
                            kai_get_nr_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm,
                            kai_get_kr_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm,
                            kai_get_sr_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm,
                            kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm,
                            kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm,
                            kai_get_dst_offset_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm,
                            kai_get_dst_size_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm,
                            kai_run_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm}}};

    size_t (*kai_get_lhs_packed_size)(
            size_t m, size_t k, size_t mr, size_t kr, size_t sr)
            = (&kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f32);
    size_t (*kai_get_rhs_nxk_packed_size)(size_t n, size_t k, size_t nr,
            size_t kr, size_t sr, size_t bl, enum kai_datatype scale_dt)
            = &kai_get_rhs_packed_size_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0;

    void (*kai_run_lhs_quant_pack)(size_t m, size_t k, size_t mr, size_t kr,
            size_t sr, size_t m_idx_start, const float *lhs, size_t lhs_stride,
            void *lhs_packed)
            = &kai_run_lhs_quant_pack_qai8dxp_f32;

    void (*kai_run_rhs_nxk_pack)(size_t num_groups, size_t n, size_t k,
            size_t nr, size_t kr, size_t sr, size_t bl, const uint8_t *rhs,
            size_t rhs_stride, const float *bias, const void *scale,
            size_t scale_stride, void *rhs_packed, size_t extra_bytes,
            const struct kai_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0_params *params)
            = &kai_run_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0;

    void init_ukernel_params() {
        m_step_ = ukernel_.get_m_step();
        n_step_ = ukernel_.get_n_step();
        mr_ = ukernel_.get_mr();
        nr_ = ukernel_.get_nr();
        kr_ = ukernel_.get_kr();
        sr_ = ukernel_.get_sr();
        set_rhs_pack_param(1, 8, dt_packed_rhs_);
    };
};

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif

#endif
