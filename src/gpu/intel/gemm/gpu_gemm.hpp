/*******************************************************************************
* Copyright 2020-2024 Intel Corporation
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

#ifndef GPU_INTEL_GEMM_GPU_GEMM_HPP
#define GPU_INTEL_GEMM_GPU_GEMM_HPP

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "gpu/intel/gemm/gpu_gemm_exec_types.hpp"
#include "gpu/intel/gpu_primitive.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {

struct gpu_gemm_t : public gpu_primitive_t {
    using gpu_primitive_t::gpu_primitive_t;
    virtual status_t execute(const gemm_exec_ctx_t &ctx) const = 0;
    status_t execute(const exec_ctx_t &ctx) const override {
        gemm_exec_args_t gemm_args;
        // TODO: we have to swap a and b because
        // - gemm primitive is created with row major desc,
        // - parameters to gemm are passed as row major
        // - but gemm implementation assumes column major
        gemm_args.a = &CTX_IN_STORAGE(DNNL_ARG_A);
        gemm_args.b = &CTX_IN_STORAGE(DNNL_ARG_B);
        gemm_args.c = &CTX_OUT_STORAGE(DNNL_ARG_C);
        gemm_args.a_zero_point
                = &CTX_IN_STORAGE(DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_A);
        gemm_args.b_zero_point
                = &CTX_IN_STORAGE(DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_B);
        gemm_args.c_zero_point
                = &CTX_IN_STORAGE(DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_C);
        gemm_args.a_scales = &CTX_IN_STORAGE(DNNL_ARG_ATTR_SCALES | DNNL_ARG_A);
        gemm_args.b_scales = &CTX_IN_STORAGE(DNNL_ARG_ATTR_SCALES | DNNL_ARG_B);
        gemm_args.c_scales = &CTX_IN_STORAGE(DNNL_ARG_ATTR_SCALES | DNNL_ARG_C);
        gemm_exec_ctx_t gemm_ctx(ctx, gemm_args);
        return execute(gemm_ctx);
    }
};

inline const gpu_gemm_t *gpu_gemm(const std::shared_ptr<impl::primitive_t> &p) {
    return utils::downcast<gpu_gemm_t *>(p.get());
}

} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
