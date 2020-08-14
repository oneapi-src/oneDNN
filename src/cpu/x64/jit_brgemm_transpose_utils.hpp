/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#ifndef CPU_X64_JIT_BRGEMM_TRANSPOSE_UTILS_HPP
#define CPU_X64_JIT_BRGEMM_TRANSPOSE_UTILS_HPP

#include "cpu/x64/jit_brgemm_primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

struct jit_brgemm_trans_wei_t {
    struct ctx_t {
        const void *src;
        const void *tr_src;

        dim_t current_gemm_batch;
        dim_t current_N, current_K;
    };

    virtual void operator()(ctx_t *ctx) = 0;
    virtual status_t create_kernel() = 0;

    jit_brgemm_trans_wei_t(const jit_brgemm_primitive_conf_t *conf)
        : conf_(conf) {}
    virtual ~jit_brgemm_trans_wei_t() {}

    const jit_brgemm_primitive_conf_t *conf_;
};

status_t create_brgemm_trans_wei(
        std::unique_ptr<jit_brgemm_trans_wei_t> &trans_ker,
        const jit_brgemm_primitive_conf_t *conf);

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
