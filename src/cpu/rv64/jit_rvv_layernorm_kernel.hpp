/*******************************************************************************
* Copyright 2026 Institute of Software, Chinese Academy of Sciences
* Copyright 2026 SpacemiT Corporation
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

#ifndef CPU_RV64_JIT_RVV_LAYERNORM_KERNEL_HPP
#define CPU_RV64_JIT_RVV_LAYERNORM_KERNEL_HPP

#include "common/c_types_map.hpp"
#include "cpu/rv64/jit_generator.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

struct jit_rvv_layernorm_fused_kernel_t : public jit_generator_t {
    struct call_params_t {
        const float *src;
        float *dst;
        const float *scale;
        const float *shift;
        dim_t len;
        float eps;
        float *mean;
        float *variance;
    };

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_rvv_layernorm_fused_kernel_t)

    jit_rvv_layernorm_fused_kernel_t(bool with_scale, bool with_shift);

    void operator()(const call_params_t *p) const {
        jit_generator_t::operator()(p);
    }

protected:
    void generate() override;

private:
    const bool with_scale_;
    const bool with_shift_;
};

struct jit_rvv_layernorm_data_kernel_t : public jit_generator_t {
    struct call_params_t {
        const float *src;
        float *dst;
        const float *scale;
        const float *shift;
        dim_t len;
        float mean;
        float inv_std;
    };

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_rvv_layernorm_data_kernel_t)

    jit_rvv_layernorm_data_kernel_t(bool with_scale, bool with_shift);

    void operator()(const call_params_t *p) const {
        jit_generator_t::operator()(p);
    }

protected:
    void generate() override;

private:
    const bool with_scale_;
    const bool with_shift_;
};

struct jit_rvv_layernorm_f16_fused_kernel_t : public jit_generator_t {
    struct call_params_t {
        const void *src; // f16
        void *dst; // f16
        const void *scale; // gamma, f16 or f32 (per weights_f16_)
        const void *shift; // beta,  f16 or f32 (per weights_f16_)
        dim_t len;
        float eps;
        float *mean; // f32 out, nullptr to skip
        float *variance; // f32 out, nullptr to skip
    };

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_rvv_layernorm_f16_fused_kernel_t)

    jit_rvv_layernorm_f16_fused_kernel_t(bool with_scale, bool with_shift,
            bool weights_f16, bool use_widening_sum);

    void operator()(const call_params_t *p) const {
        jit_generator_t::operator()(p);
    }

protected:
    void generate() override;

private:
    const bool with_scale_;
    const bool with_shift_;
    const bool weights_f16_;
    const bool use_widening_sum_;
};

struct jit_rvv_layernorm_bwd_kernel_t : public jit_generator_t {
    struct call_params_t {
        const float *src;
        const float *diff_dst;
        float *diff_src;
        const float *scale;
        float *diff_scale;
        float *diff_shift;
        dim_t len;
        float mean;
        float inv_std;
        float channels;
    };
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_rvv_layernorm_bwd_kernel_t)
    jit_rvv_layernorm_bwd_kernel_t(bool scale, bool stats, bool weights);
    status_t status() const { return status_; }
    void operator()(const call_params_t *p) const {
        jit_generator_t::operator()(p);
    }

private:
    bool scale_, stats_, weights_;
    status_t status_ = status::runtime_error;
    void generate() override;
};

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
