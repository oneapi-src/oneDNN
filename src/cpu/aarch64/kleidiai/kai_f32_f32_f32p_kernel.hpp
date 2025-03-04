
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

#ifndef CPU_AARCH64_KLEIDIAI_KAI_F32_f32_f32P_KERNEL
#define CPU_AARCH64_KLEIDIAI_KAI_F32_f32_f32P_KERNEL

#include <kai/ukernels/matmul/matmul_clamp_f32_f32_f32p/kai_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla.h>
#include <kai/ukernels/matmul/matmul_clamp_f32_f32_f32p/kai_matmul_clamp_f32_f32_f32p_interface.h>
#include <kai/ukernels/matmul/pack/kai_rhs_pack_kxn_f32p8x1biasf32_f32_f32_neon.h>
#include <unordered_map>

#include "cpu/aarch64/brgemm/brgemm_types.hpp"
#include "cpu/aarch64/kleidiai/kai_types.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

// Channelwise Kernel mapping
struct kai_f32_f32_f32p_kernel_packet_t
    : public kernel_kai_common_t<kai_matmul_clamp_f32_f32_f32p_ukernel> {
public:
    using kernel_kai_common_t::kernel_kai_common_t;

    kai_f32_f32_f32p_kernel_packet_t() = default;

    kai_f32_f32_f32p_kernel_packet_t(brgemm_desc_t desc) {
        desc_ = desc;
        ukernel_id_ = kai_ukernel_id::f32_f32_f32p;
        ukernel_ = f32_f32_f32p_kernels_.at(ukernel_id_);
        init_ukernel_params();
    }

    kai_f32_f32_f32p_kernel_packet_t(const kai_ukernel_id id) {
        ukernel_id_ = id;
        ukernel_ = f32_f32_f32p_kernels_.at(ukernel_id_);
        init_ukernel_params();
    }

    kai_f32_f32_f32p_kernel_packet_t(
            const kai_f32_f32_f32p_kernel_packet_t *kernel) {
        ukernel_id_ = kernel->get_ukernel_id();
        ukernel_ = f32_f32_f32p_kernels_.at(ukernel_id_);
        desc_ = kernel->get_desc();
        init_ukernel_params();
    }

    std::string to_string() const override {
        return "kai_matmul_clamp_f32_f32_f32p_ukernel";
    };

    size_t get_rhs_packed_size() const {
        return kai_get_rhs_packed_size(desc_.N, desc_.K);
    };

    size_t get_rhs_packed_size(uint64_t N, uint64_t K) const {
        return kai_get_rhs_packed_size(N, K);
    };

    size_t get_rhs_packed_offset(uint64_t n_idx) const {
        return ukernel_.get_rhs_packed_offset(n_idx, desc_.K);
    };

    size_t get_rhs_packed_offset(uint64_t n_idx, uint64_t K) const {
        return ukernel_.get_rhs_packed_offset(n_idx, K);
    };

    void run_rhs_pack(const float *rhs, const float *bias, float *rhs_packed,
            stride_t rhs_stride = 0, uint64_t num_groups = 1,
            size_t extra_bytes = 0) const {
        if (rhs_stride == 0) rhs_stride = desc_.N * sizeof(float);

        float *to_use_bias;
        if (bias == nullptr) {
            to_use_bias = new float[desc_.N]();
        } else {
            to_use_bias = (float *)bias;
        }

        kai_run_rhs_pack(num_groups, desc_.N, desc_.K, nr_, kr_, sr_,
                rhs_stride, rhs, to_use_bias, nullptr, rhs_packed, extra_bytes,
                nullptr);

        if (bias == nullptr) delete[] to_use_bias;
    };

    void operator()(brgemm_kernel_params_t *kernel_params) const override {

        const brgemm_batch_element_t *brgemm_be_vec_ptr = kernel_params->batch;
        for (uint64_t bs_idx = 0; bs_idx < kernel_params->BS; bs_idx++) {
            execute(kernel_params->ptr_A, kernel_params->ptr_B,
                    kernel_params->ptr_C, brgemm_be_vec_ptr[bs_idx].offset.A,
                    brgemm_be_vec_ptr[bs_idx].offset.B);
        }
    };

    void execute(const void *lhs_base_ptr, const void *rhs_packed_base_ptr,
            void *dst_base_ptr,

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

        if (lhs_stride == 0) lhs_stride = desc_.K * sizeof(float);

        if (dst_stride == 0) dst_stride = desc_.N * sizeof(float);

        const void *lhs_offset_ptr = (const void *)((const float *)lhs_base_ptr
                + ukernel_.get_lhs_offset(m_idx, lhs_stride));
        const void *rhs_offset_packed_ptr
                = (const void *)((const float *)rhs_packed_base_ptr
                        + ukernel_.get_rhs_packed_offset(n_idx, desc_.K));
        float *dst_offset_ptr = (float *)((float *)dst_base_ptr
                + ukernel_.get_dst_offset(m_idx, n_idx, dst_stride));

        ukernel_.run_matmul(ms_per_run, ns_per_run, desc_.K, lhs_offset_ptr,
                lhs_stride, rhs_offset_packed_ptr, dst_offset_ptr, dst_stride,
                sizeof(float), -FLT_MAX, FLT_MAX);
    };

private:
    std::unordered_map<kai_ukernel_id, kai_matmul_clamp_f32_f32_f32p_ukernel>
            f32_f32_f32p_kernels_ = {{kai_ukernel_id::f32_f32_f32p,
                    {kai_get_m_step_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla,
                            kai_get_n_step_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla,
                            kai_get_nr_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla,
                            kai_get_kr_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla,
                            kai_get_sr_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla,
                            kai_get_lhs_offset_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla,
                            kai_get_rhs_packed_offset_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla,
                            kai_get_dst_offset_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla,
                            kai_get_dst_size_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla,
                            kai_run_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla}}};

    size_t (*kai_get_rhs_packed_size)(size_t n, size_t k)
            = &kai_get_rhs_packed_size_rhs_pack_kxn_f32p8x1biasf32_f32_f32_neon;

    void (*kai_run_rhs_pack)(size_t num_groups, size_t n, size_t k, size_t nr,
            size_t kr, size_t sr, size_t rhs_stride, const void *rhs,
            const void *bias, const void *scale, void *rhs_packed,
            size_t extra_bytes, const void *params)
            = &kai_run_rhs_pack_kxn_f32p8x1biasf32_f32_f32_neon;

    void init_ukernel_params() {
        m_step_ = ukernel_.get_m_step();
        n_step_ = ukernel_.get_n_step();
        nr_ = ukernel_.get_nr();
        kr_ = ukernel_.get_kr();
        sr_ = ukernel_.get_sr();
    };
};

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif

#endif
