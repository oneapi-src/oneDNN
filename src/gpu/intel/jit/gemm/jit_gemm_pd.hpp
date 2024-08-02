/*******************************************************************************
* Copyright 2022-2024 Intel Corporation
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

#ifndef GPU_INTEL_JIT_GEMM_JIT_GEMM_PD_HPP
#define GPU_INTEL_JIT_GEMM_JIT_GEMM_PD_HPP

#include <vector>

#include "common/c_types_map.hpp"
#include "gpu/gpu_gemm_pd.hpp"
#include "gpu/intel/gpu_post_ops.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {

#define GEMM_MAX_PO 36

struct jit_gemm_pd_t : public gpu_gemm_pd_t {
    using gpu_gemm_pd_t::gpu_gemm_pd_t;

    struct binary_src_t {
        enum type_t { none, scales, bias, binary, prelu } type;
        int index;

        binary_src_t(type_t type_, int index_) : type(type_), index(index_) {}
    };

    static constexpr post_op::specializations_t get_post_op_specializations() {
        using mode_t = post_op::specializations_t::inline_mode_t;
        using sum_t = post_op::specializations_t::sum_t;
        // The sum scale is handled as GEMM beta argument
        return {{}, sum_t(mode_t::impl_managed(), {}), {}};
    }

    status_t init_post_ops();

    dim_t ld_binary(int idx) const;
    dim_t stride_binary(int idx, int stride = 0) const;

    const post_ops_t *post_ops() const { return &post_ops_; }
    const std::vector<binary_src_t> &binary_srcs() const {
        return binary_srcs_;
    }

    float beta_ = 0.0f;

    bool with_sum_ = false;
    bool sum_at_begin_ = false;

    bool bias_via_binary_ = false;

    post_ops_t post_ops_;
    std::vector<binary_src_t> binary_srcs_;

    memory_desc_t wei_scales_md, src_scales_md, c_scales_md, prelu_wei_md;
};

} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
