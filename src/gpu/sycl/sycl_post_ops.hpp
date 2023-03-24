/*******************************************************************************
* Copyright 2022-2023 Intel Corporation
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

#ifndef GPU_SYCL_SYCL_POST_OPS_HPP
#define GPU_SYCL_SYCL_POST_OPS_HPP

#include "common/c_types_map.hpp"
#include "common/primitive_attr.hpp"
#include "gpu/sycl/sycl_math_utils.hpp"
#include "gpu/sycl/sycl_types.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace sycl {

struct ref_eltwise_fwd_t {
    ref_eltwise_fwd_t() = default;
    ref_eltwise_fwd_t(alg_kind_t alg, float alpha, float beta, float scale)
        : alg_(alg), alpha_(alpha), beta_(beta), scale_(scale) {
        using namespace alg_kind;
        assert(utils::one_of(alg_, eltwise_relu, eltwise_linear, eltwise_clip,
                eltwise_clip_v2));
    }

    ref_eltwise_fwd_t(const post_ops_t::entry_t::eltwise_t &eltwise)
        : ref_eltwise_fwd_t(
                eltwise.alg, eltwise.alpha, eltwise.beta, eltwise.scale) {}

    float compute(float s) const {
        return compute(alg_, s, alpha_, beta_) * scale_;
    }

    template <int width>
    ::sycl::vec<float, width> compute(::sycl::vec<float, width> src_vec) const {
        ::sycl::vec<float, width> scale_vec(scale_);
        return compute(alg_, src_vec, alpha_, beta_) * scale_vec;
    }

    // Make `compute` functions static so that it can be used in an eltwise
    // primitive.
    template <int width>
    static ::sycl::vec<float, width> compute(alg_kind_t alg,
            ::sycl::vec<float, width> src_vec, float alpha, float beta) {
        using namespace alg_kind;
        using namespace math;

        constexpr ::sycl::vec<float, width> nan_vec(NAN);

        switch (alg) {
            case eltwise_relu: return relu_fwd(src_vec, alpha); ;
            case eltwise_linear: return linear_fwd(src_vec, alpha, beta);
            default: return nan_vec;
        }
    }

    static float compute(alg_kind_t alg, float s, float alpha, float beta) {
        using namespace alg_kind;
        using namespace math;

        float d = 0.f;
        switch (alg) {
            case eltwise_relu: d = relu_fwd(s, alpha); break;
            case eltwise_linear: d = linear_fwd(s, alpha, beta); break;
            case eltwise_clip: d = clip_fwd(s, alpha, beta); break;
            case eltwise_clip_v2: d = clip_v2_fwd(s, alpha, beta); break;
            default: d = ::sycl::nan(0u);
        }
        return d;
    }

private:
    alg_kind_t alg_;
    float alpha_;
    float beta_;
    float scale_;
};

struct ref_binary_op_t {
    ref_binary_op_t() = default;
    ref_binary_op_t(alg_kind_t alg) : alg_(alg) {
        using namespace alg_kind;
        assert(utils::one_of(alg_, binary_add, binary_div, binary_max,
                binary_min, binary_mul, binary_sub, binary_ge, binary_gt,
                binary_le, binary_lt, binary_eq, binary_ne));
    }

    ref_binary_op_t(const post_ops_t::entry_t::binary_t &binary)
        : ref_binary_op_t(binary.alg) {}

    float compute(float s0, float s1) const { return compute(alg_, s0, s1); }

    static float compute(alg_kind_t alg, float s0, float s1) {
        using namespace alg_kind;
        using namespace math;

        float d = 0.f;
        switch (alg) {
            case binary_add: d = s0 + s1; break;
            case binary_div: d = s0 / s1; break;
            case binary_min: d = ::sycl::min(s0, s1); break;
            case binary_max: d = ::sycl::max(s0, s1); break;
            case binary_mul: d = s0 * s1; break;
            case binary_sub: d = s0 - s1; break;
            case binary_ge: d = (float)((s0 >= s1) * -1); break;
            case binary_gt: d = (float)((s0 > s1) * -1); break;
            case binary_le: d = (float)((s0 <= s1) * -1); break;
            case binary_lt: d = (float)((s0 < s1) * -1); break;
            case binary_eq: d = (float)((s0 == s1) * -1); break;
            case binary_ne: d = (float)((s0 != s1) * -1); break;
            default: d = ::sycl::nan(0u);
        }
        return d;
    }

private:
    alg_kind_t alg_;
};

struct sycl_post_ops_t {
    // SYCL has a limitation on total size of kernel arguments.
    // This affects number of post ops, e.g. binary post op (which is not yet
    // implemented) contains sycl_md_t which is large enough to limit
    // the number of post ops.
    static constexpr int max_post_ops = 5;

    sycl_post_ops_t() = default;
    sycl_post_ops_t(const primitive_attr_t *attr) {
        using namespace primitive_kind;

        const auto &attr_po = attr->post_ops_;
        assert(attr_po.len() <= max_post_ops);

        int sum_idx = 0;
        int eltwise_idx = 0;
        int binary_idx = 0;

        for (auto i = 0; i < attr_po.len(); ++i) {
            if (attr_po.contain(sum, i)) {
                post_op_kinds_[i] = sum;
                sum_scales_[sum_idx++] = attr_po.entry_[i].sum.scale;
            } else if (attr_po.contain(eltwise, i)) {
                post_op_kinds_[i] = eltwise;
                eltwise_post_ops_[eltwise_idx++]
                        = ref_eltwise_fwd_t(attr_po.entry_[i].eltwise);
            } else if (attr_po.contain(binary, i)) {
                post_op_kinds_[i] = binary;
                binary_post_ops_[binary_idx++]
                        = ref_binary_op_t(attr_po.entry_[i].binary);
            }
        }
        n_post_ops_ = attr_po.len();
    }

    template <int width>
    ::sycl::vec<float, width> apply(::sycl::vec<float, width> acc,
            ::sycl::vec<float, width> dst) const {
        using namespace primitive_kind;
        constexpr ::sycl::vec<float, width> nan_vec(NAN);

        if (n_post_ops_ == 0) return acc;

        int sum_idx = 0;
        int eltwise_idx = 0;

        for (auto i = 0; i < n_post_ops_; ++i) {
            switch (post_op_kinds_[i]) {
                case sum: {
                    const ::sycl::vec<float, width> sum_scale(
                            sum_scales_[sum_idx++]);
                    acc += sum_scale * dst;
                    break;
                }
                case eltwise:
                    acc = eltwise_post_ops_[eltwise_idx++].compute(acc);
                    break;
                default: acc = nan_vec;
            }
        }
        return acc;
    }

    float apply(float acc, float dst) const {
        using namespace primitive_kind;

        if (n_post_ops_ == 0) return acc;

        int sum_idx = 0;
        int eltwise_idx = 0;

        for (auto i = 0; i < n_post_ops_; ++i) {
            switch (post_op_kinds_[i]) {
                case sum: acc += sum_scales_[sum_idx++] * dst; break;
                case eltwise:
                    acc = eltwise_post_ops_[eltwise_idx++].compute(acc);
                    break;
                default: acc = ::sycl::nan(0u);
            }
        }
        return acc;
    }

    template <int width>
    float apply(float acc, float dst_sum, ::sycl::vec<float, width> dst) const {
        using namespace primitive_kind;

        if (n_post_ops_ == 0) return acc;

        int binary_idx = 0;
        int sum_idx = 0;
        int eltwise_idx = 0;
        for (auto i = 0; i < n_post_ops_; ++i) {
            switch (post_op_kinds_[i]) {
                case eltwise:
                    acc = eltwise_post_ops_[eltwise_idx++].compute(acc);
                    break;
                case binary:
                    acc = binary_post_ops_[binary_idx++].compute(acc, dst[i]);
                    break;
                case sum: acc += sum_scales_[sum_idx++] * dst_sum; break;
                default: acc = ::sycl::nan(0u);
            }
        }
        return acc;
    }

    template <int width>
    float apply(float acc, ::sycl::vec<float, width> dst) const {
        using namespace primitive_kind;

        if (n_post_ops_ == 0) return acc;

        int binary_idx = 0;
        int eltwise_idx = 0;
        for (auto i = 0; i < n_post_ops_; ++i) {
            switch (post_op_kinds_[i]) {
                case eltwise:
                    acc = eltwise_post_ops_[eltwise_idx++].compute(acc);
                    break;
                case binary:
                    acc = binary_post_ops_[binary_idx++].compute(acc, dst[i]);
                    break;
                default: acc = ::sycl::nan(0u);
            }
        }
        return acc;
    }

    inline int get_post_op() const { return n_post_ops_; }

    inline primitive_kind_t get_post_op_kind(int i) const {
        return post_op_kinds_[i];
    }

    inline ref_binary_op_t get_binary_post_op(int i) const {
        return binary_post_ops_[i];
    }

private:
    float sum_scales_[max_post_ops];
    ref_binary_op_t binary_post_ops_[max_post_ops];
    ref_eltwise_fwd_t eltwise_post_ops_[max_post_ops];
    primitive_kind_t post_op_kinds_[max_post_ops];
    // Indicates the actual number of post ops.
    int n_post_ops_;
};

CHECK_SYCL_KERNEL_ARG_TYPE(ref_binary_op_t);
CHECK_SYCL_KERNEL_ARG_TYPE(ref_eltwise_fwd_t);
CHECK_SYCL_KERNEL_ARG_TYPE(sycl_post_ops_t);

} // namespace sycl
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
