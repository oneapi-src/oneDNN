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

#ifndef CPU_AARCH64_KLEIDIAI_KAI_TYPES_HPP
#define CPU_AARCH64_KLEIDIAI_KAI_TYPES_HPP

#include <cfloat>
#include <cstddef>
#include <cstdint>
#include "cpu/aarch64/brgemm/brgemm_types.hpp"
#include <kai/kai_common.h>

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

enum class rhs_format {
    nxk,
    kxn,
};

typedef enum {
    unknown = -1,

    // f32-f32-f32p
    f32_f32_f32p = 0,

    // f32-s8p-s4p--cw
    f32_qai8dxp_qsi4cxp_gemv = 1, // GEMV
    f32_qai8dxp_qsi4cxp_gemm = 2, // GEMM

    // f32-s8p-s4p--gw
    f32_qai8dxp_qsi4c32p_gemv = 3, // GEMV
    f32_qai8dxp_qsi4c32p_gemm = 4, // GEMM
} kai_ukernel_id;

struct kai_rhs_pack_param {
    int8_t lhs_zero_point;
    uint8_t rhs_zero_point;
    kai_datatype scale_dt;
};

template <typename KernelType>
struct kernel_kai_common_t {
public:
    kernel_kai_common_t() = default;
    virtual ~kernel_kai_common_t() {};

    virtual void operator()(brgemm_kernel_params_t *kernel_params) const {};

    virtual std::string to_string() const { return "kai_ukernel_undef"; };

    virtual brgemm_desc_t get_desc() const { return desc_; };
    // ukernel
    virtual kai_ukernel_id get_ukernel_id() const { return ukernel_id_; };
    virtual KernelType get_ukernel() const { return ukernel_; };
    virtual uint64_t get_ukernel_m_step() const { return m_step_; };
    virtual uint64_t get_ukernel_n_step() const { return n_step_; };

    virtual uint64_t get_ukernel_mr() const { return mr_; };
    virtual uint64_t get_ukernel_nr() const { return nr_; };
    virtual uint64_t get_ukernel_kr() const { return kr_; };
    virtual uint64_t get_ukernel_sr() const { return sr_; };

    virtual size_t get_lhs_offset(dim_t m_idx, size_t lhs_stride) const {
        return m_idx * lhs_stride;
    };
    virtual size_t get_dst_offset(uint64_t n_idx, uint64_t n, dim_t k,
            stride_t dst_stride = 0) const {
        if (dst_stride == 0) dst_stride = n * sizeof(float);

        return ukernel_.get_dst_offset(n_idx, k, dst_stride);
    };

protected:
    brgemm_desc_t desc_;
    kai_ukernel_id ukernel_id_;
    KernelType ukernel_;
    uint64_t m_step_ = 0, n_step_ = 0, mr_ = 0, nr_ = 0, kr_ = 0, sr_ = 0;
};

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif

#endif
