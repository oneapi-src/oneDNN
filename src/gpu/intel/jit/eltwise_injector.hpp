/*******************************************************************************
* Copyright 2020-2025 Intel Corporation
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

#ifndef GPU_INTEL_JIT_ELTWISE_INJECTOR_HPP
#define GPU_INTEL_JIT_ELTWISE_INJECTOR_HPP

#include <assert.h>

#include "common/c_types_map.hpp"
#include "common/utils.hpp"
#include "gpu/intel/jit/codegen/kernel.hpp"
#include "gpu/intel/jit/generator.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {

inline bool eltwise_injector_f32_is_supported(alg_kind_t alg) {
    using namespace alg_kind;
    return utils::one_of(alg, eltwise_elu, eltwise_elu_use_dst_for_bwd,
            eltwise_exp, eltwise_exp_use_dst_for_bwd, eltwise_gelu_tanh,
            eltwise_gelu_erf, eltwise_hardsigmoid, eltwise_hardswish,
            eltwise_log, eltwise_mish, eltwise_pow, eltwise_relu,
            eltwise_relu_use_dst_for_bwd, eltwise_soft_relu, eltwise_sqrt,
            eltwise_sqrt_use_dst_for_bwd, eltwise_square, eltwise_swish,
            eltwise_tanh, eltwise_tanh_use_dst_for_bwd, eltwise_abs,
            eltwise_round, eltwise_linear, eltwise_clip, eltwise_clip_v2,
            eltwise_clip_v2_use_dst_for_bwd, eltwise_logistic,
            eltwise_logistic_use_dst_for_bwd, eltwise_stochastic_round);
}

template <gpu_gen_t hw>
struct eltwise_injector_f32_t {
    eltwise_injector_f32_t(generator_t<hw> *host, alg_kind_t alg, float alpha,
            float beta, float scale, int eu_count,
            const ngen::GRFRange &scratch = ngen::GRFRange(),
            bool is_fwd = true)
        : alg_(alg)
        , alpha_(alpha)
        , beta_(beta)
        , scale_(scale)
        , is_fwd_(is_fwd)
        , eu_count_(eu_count)
        , h(host)
        , scratch_(scratch) {

        assert(eltwise_injector_f32_is_supported(alg_));
        assert(scratch_.isEmpty() || (scratch_.getLen() >= min_scratch_regs()));
    }

    int min_scratch_regs();
    int preferred_scratch_regs();
    void set_scratch(const ngen::GRFRange &scratch) { scratch_ = scratch; }

    void prepare();
    void compute(const ngen::GRF &reg) { compute(reg - reg); }
    void compute(const ngen::GRFRange &regs, int seed = -1, int seed_off = -1,
            ngen::DataType = ngen::DataType::invalid);
    void compute(const int *grfs, int ngrf, int seed = -1, int seed_off = -1,
            ngen::DataType = ngen::DataType::invalid);

private:
    const alg_kind_t alg_;
    const float alpha_;
    const float beta_;
    const float scale_;
    const bool is_fwd_;

    const int eu_count_;

    generator_t<hw> *h;

    ngen::GRFRange scratch_;

    bool is_gpu(ngen::HW arg_hw, int arg_eu_count) const {
        return (hw == arg_hw) && (eu_count_ == arg_eu_count);
    }
    bool use_tanh_compat() const { return false; }

    int max_batch_size();
    int phase_count(alg_kind_t alg);

    void relu_zero_ns_prepare_fwd();
    void relu_prepare_bwd();
    void abs_prepare_bwd();
    void clip_prepare_bwd();
    void tanh_prepare_fwd();
    void tanh_prepare_fwd_compat();

    void relu_zero_ns_compute_fwd(int simd, const ngen::GRF &r);
    void relu_compute_fwd(int simd, const ngen::GRF &r, int phase, int off);
    void abs_compute_fwd(int simd, const ngen::GRF &r);
    void exp_compute_fwd(int simd, const ngen::GRF &r, int phase);
    void elu_compute_fwd(int simd, const ngen::GRF &r, int phase, int off);
    void gelu_erf_compute_fwd(
            int simd, const ngen::GRF &r, int phase, int off, int batch);
    void hardsigmoid_compute_fwd(
            int simd, const ngen::GRF &r, int phase, int off);
    void hardswish_compute_fwd(
            int simd, const ngen::GRF &r, int phase, int off);
    void log_compute_fwd(int simd, const ngen::GRF &r, int phase);
    void mish_compute_fwd(
            int simd, const ngen::GRF &r, int phase, int off, int batch);
    void pow_compute_fwd(int simd, const ngen::GRF &r, int phase, int off);
    void soft_relu_compute_fwd_inner(int simd, const ngen::GRF &input,
            const ngen::GRF &temp, const ngen::GRF &dest, int phase, int off,
            float alpha);
    void soft_relu_compute_fwd(
            int simd, const ngen::GRF &r, int phase, int off);
    void sqrt_compute_fwd(int simd, const ngen::GRF &r);
    void square_compute_fwd(int simd, const ngen::GRF &r);
    void round_compute_fwd(int simd, const ngen::GRF &r);
    void sround_compute_fwd(int simd, const ngen::GRF &r, int phase,
            const ngen::Subregister &seed, const ngen::DataType dst_dt,
            int off);
    void philox_4x32(
            int simd, const ngen::Subregister &seed, const ngen::GRF &bias);
    void swish_compute_fwd(int simd, const ngen::GRF &r, int phase, int off);
    void tanh_compute_fwd(
            int simd, const ngen::GRF &r, int phase, int off, int batch);
    void tanh_compute_fwd_compat(
            int simd, const ngen::GRF &r, int phase, int off, int batch);
    void linear_compute_fwd(int simd, const ngen::GRF &r, int phase);
    void clip_compute_fwd(
            int simd, const ngen::GRF &r, int phase, float alpha, float beta);
    void gelu_tanh_compute_fwd(
            int simd, const ngen::GRF &r, int phase, int off);
    void logistic_compute_fwd(int simd, const ngen::GRF &r, int phase);

    void relu_compute_bwd(int simd, const ngen::GRF &r);
    void abs_compute_bwd(int simd, const ngen::GRF &r, int phase);
    void square_compute_bwd(int simd, const ngen::GRF &r);
    void linear_compute_bwd(int simd, const ngen::GRF &r);
    void clip_compute_bwd(
            int simd, const ngen::GRF &r, int phase, float alpha, float beta);
    void gelu_tanh_compute_bwd(
            int simd, const ngen::GRF &r, int phase, int off, int batch);

    const ngen::InstructionModifier le = generator_t<hw>::le;
    const ngen::InstructionModifier lt = generator_t<hw>::lt;
    const ngen::InstructionModifier ge = generator_t<hw>::ge;
    const ngen::InstructionModifier gt = generator_t<hw>::gt;
    const ngen::InstructionModifier eq = generator_t<hw>::eq;
    const ngen::InstructionModifier sat = generator_t<hw>::sat;
    const ngen::FlagRegister f0 = generator_t<hw>::f0;
};

} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif // GPU_INTEL_JIT_ELTWISE_INJECTOR_HPP
