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

#ifndef CPU_RV64_JIT_RVV_SOFTMAX_KERNEL_HPP
#define CPU_RV64_JIT_RVV_SOFTMAX_KERNEL_HPP

#include "common/bfloat16.hpp"
#include "common/c_types_map.hpp"
#include "common/float16.hpp"
#include "cpu/rv64/jit_generator.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

// The xf16 kernels below serve both 16-bit float types. They are identical
// apart from the widen/narrow pair -- Zvfh vfwcvt/vfncvt for f16, Zvfbfmin
// vfwcvtbf16/vfncvtbf16 for bf16 -- and the NaN/lowest constants, so each one
// takes the dtype at construction and the caller passes raw pointers.

struct jit_rvv_softmax_affine_kernel_t : public jit_generator_t {
    struct call_params_t {
        const float *src;
        float *dst;
        dim_t len;
        float sub;
        float mul;
    };

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_rvv_softmax_affine_kernel_t)

    jit_rvv_softmax_affine_kernel_t();

    void operator()(const call_params_t *p) const {
        jit_generator_t::operator()(p);
    }

protected:
    void generate() override;
};

struct jit_rvv_softmax_xf16_affine_kernel_t : public jit_generator_t {
    struct call_params_t {
        const void *src;
        void *dst;
        dim_t len;
        float sub;
        float mul;
    };

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_rvv_softmax_xf16_affine_kernel_t)

    jit_rvv_softmax_xf16_affine_kernel_t(bool src_f32, data_type_t dt);

    void operator()(const call_params_t *p) const {
        jit_generator_t::operator()(p);
    }

protected:
    void generate() override;

private:
    bool src_f32_;
    data_type_t dt_;
};

// Pure 16-bit gather/scatter: no arithmetic, so one kernel serves f16 and bf16.
struct jit_rvv_softmax_xf16_strided_kernel_t : public jit_generator_t {
    struct call_params_t {
        const void *src;
        void *dst;
        dim_t len;
        dim_t stride_bytes;
    };

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_rvv_softmax_xf16_strided_kernel_t)

    explicit jit_rvv_softmax_xf16_strided_kernel_t(bool gather);

    void operator()(const call_params_t *p) const {
        jit_generator_t::operator()(p);
    }

protected:
    void generate() override;

private:
    bool gather_;
};

template <bool gather>
const jit_rvv_softmax_xf16_strided_kernel_t &get_xf16_strided_kernel();

struct jit_rvv_softmax_xf16_exp_sub_sum_kernel_t : public jit_generator_t {
    struct call_params_t {
        const void *src;
        float *tmp;
        dim_t len;
        float sub;
        float *sum;
    };

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_rvv_softmax_xf16_exp_sub_sum_kernel_t)

    explicit jit_rvv_softmax_xf16_exp_sub_sum_kernel_t(data_type_t dt);

    void operator()(const call_params_t *p) const {
        jit_generator_t::operator()(p);
    }

protected:
    void generate() override;

private:
    data_type_t dt_;
};

// f32 exp/sub/sum kernel. Computes exp(src - sub), accumulates the ordered
// sum into *sum, and (when store_exp is set) writes the exp values to dst.
// softmax uses store_exp=true (dst holds the exp result for the later scale);
// logsoftmax uses store_exp=false (only the sum is required).
struct jit_rvv_softmax_f32_exp_sub_sum_kernel_t : public jit_generator_t {
    struct call_params_t {
        const float *src;
        float *dst;
        dim_t len;
        float sub;
        float *sum;
    };

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_rvv_softmax_f32_exp_sub_sum_kernel_t)

    explicit jit_rvv_softmax_f32_exp_sub_sum_kernel_t(bool store_exp);

    void operator()(const call_params_t *p) const {
        jit_generator_t::operator()(p);
    }

protected:
    void generate() override;

private:
    bool store_exp_;
};

struct jit_rvv_softmax_xf16_reduce_max_kernel_t : public jit_generator_t {
    struct call_params_t {
        const void *src;
        dim_t len;
        float *max_val;
        uint32_t *has_nan;
    };

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_rvv_softmax_xf16_reduce_max_kernel_t)

    explicit jit_rvv_softmax_xf16_reduce_max_kernel_t(data_type_t dt);

    void operator()(const call_params_t *p) const {
        jit_generator_t::operator()(p);
    }

protected:
    void generate() override;

private:
    data_type_t dt_;
};

// f32 value-only max reduction for softmax Stage 1. Computes max(src[0..len))
// with the scalar `val > max_val` semantics: seeded with -INFINITY and NaN
// lanes merged to the seed so a NaN never wins the reduction. The caller uses
// a platform-tuned crossover to choose between this kernel and the scalar loop.
struct jit_rvv_softmax_f32_reduce_max_kernel_t : public jit_generator_t {
    struct call_params_t {
        const float *src;
        dim_t len;
        float *max_val;
    };

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_rvv_softmax_f32_reduce_max_kernel_t)

    jit_rvv_softmax_f32_reduce_max_kernel_t();

    void operator()(const call_params_t *p) const {
        jit_generator_t::operator()(p);
    }

protected:
    void generate() override;
};

void jit_rvv_softmax_xf16_affine_from_xf16(data_type_t dt, const void *src,
        void *dst, dim_t len, float sub, float mul);

void jit_rvv_softmax_xf16_affine_from_f32(data_type_t dt, const float *src,
        void *dst, dim_t len, float sub, float mul);

void jit_rvv_softmax_xf16_gather(
        const void *src, void *dst, dim_t len, dim_t stride_bytes);

void jit_rvv_softmax_xf16_scatter(
        const void *src, void *dst, dim_t len, dim_t stride_bytes);

void jit_rvv_softmax_xf16_exp_sub_sum(data_type_t dt, const void *src,
        float *tmp, dim_t len, float sub, float *sum);

void jit_rvv_softmax_f32_exp_sub_sum(const float *src, float *dst, dim_t len,
        float sub, float *sum, bool store_exp);

void jit_rvv_softmax_xf16_reduce_max(data_type_t dt, const void *src, dim_t len,
        float *max_val, uint32_t *has_nan);

void jit_rvv_softmax_f32_reduce_max(
        const float *src, dim_t len, float *max_val);

struct jit_rvv_softmax_inner_kernel_t : public jit_generator_t {
    struct call_params_t {
        const float *src;
        float *dst;
        dim_t axis_len;
        dim_t stride_bytes;
        dim_t columns;
        float *max;
        float *sum;
    };

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_rvv_softmax_inner_kernel_t)
    jit_rvv_softmax_inner_kernel_t(bool apply, bool logsoftmax,
            bool inf_as_zero, int lmul, bool axis_vectorized, bool packed_axis);
    void operator()(const call_params_t *p) const {
        jit_generator_t::operator()(p);
    }
    status_t status() const { return status_; }

private:
    status_t status_ = status::runtime_error;
    const bool apply_, logsoftmax_, inf_as_zero_, axis_vectorized_,
            packed_axis_;
    const int lmul_;
    void generate() override;
};

const jit_rvv_softmax_inner_kernel_t &get_softmax_inner_kernel(bool apply,
        bool logsoftmax, bool inf_as_zero, int lmul, bool axis_vectorized,
        bool packed_axis);

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_RV64_JIT_RVV_SOFTMAX_KERNEL_HPP
