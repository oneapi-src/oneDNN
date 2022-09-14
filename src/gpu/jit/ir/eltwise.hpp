/*******************************************************************************
* Copyright 2022 Intel Corporation
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

#ifndef GPU_JIT_IR_ELTWISE_HPP
#define GPU_JIT_IR_ELTWISE_HPP

#include <string>
#include <vector>

#include "gpu/jit/ir/ir.hpp"
#include "gpu/jit/utils/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

class eltwise_t : public func_impl_t {
public:
    IR_DECL_DERIVED_TYPE_ID(eltwise_t, func_impl_t)

    static func_t make(
            alg_kind_t alg_kind, float scale, float alpha, float beta) {
        return func_t(new eltwise_t(alg_kind, scale, alpha, beta));
    }

    bool is_equal(const object_impl_t &obj) const override {
        if (!obj.is<self_type>()) return false;
        auto &other = obj.as<self_type>();

        return (alg_kind == other.alg_kind) && (scale == other.scale)
                && (alpha == other.alpha) && (beta == other.beta);
    }

    size_t get_hash() const override {
        return ir_utils::get_hash(alg_kind, scale, alpha, beta);
    }

    std::string str() const override {
        switch (alg_kind) {
            case alg_kind::eltwise_relu: return "relu";
            case alg_kind::eltwise_tanh: return "tanh";
            case alg_kind::eltwise_elu: return "elu";
            case alg_kind::eltwise_square: return "square";
            case alg_kind::eltwise_abs: return "abs";
            case alg_kind::eltwise_sqrt: return "sqrt";
            case alg_kind::eltwise_swish: return "swish";
            case alg_kind::eltwise_linear: return "linear";
            case alg_kind::eltwise_soft_relu: return "soft_relu";
            case alg_kind::eltwise_logistic: return "logistic";
            case alg_kind::eltwise_mish: return "mish";
            case alg_kind::eltwise_exp: return "exp";
            case alg_kind::eltwise_log: return "log";
            case alg_kind::eltwise_clip: return "clip";
            case alg_kind::eltwise_clip_v2: return "clip_v2";
            case alg_kind::eltwise_pow: return "pow";
            case alg_kind::eltwise_gelu_tanh: return "gelu_tanh";
            case alg_kind::eltwise_gelu_erf: return "gelu_erf";
            case alg_kind::eltwise_hardswish: return "hardswish";
            case alg_kind::eltwise_relu_use_dst_for_bwd:
                return "relu_use_dst_for_bwd";
            case alg_kind::eltwise_tanh_use_dst_for_bwd:
                return "tanh_use_dst_for_bwd";
            case alg_kind::eltwise_elu_use_dst_for_bwd:
                return "elu_use_dst_for_bwd";
            case alg_kind::eltwise_sqrt_use_dst_for_bwd:
                return "sqrt_use_dst_for_bwd";
            case alg_kind::eltwise_logistic_use_dst_for_bwd:
                return "logistic_use_dst_for_bwd";
            case alg_kind::eltwise_exp_use_dst_for_bwd:
                return "exp_use_dst_for_bwd";
            case alg_kind::eltwise_clip_v2_use_dst_for_bwd:
                return "clip_v2_use_dst_for_bwd";
            case alg_kind::eltwise_round: return "round";
            default: ir_error_not_expected();
        }
        return "unknown";
    }

    IR_DEFINE_ARG_GET(elems, 0)
    IR_DEFINE_ARG_GET(data, 1)

    alg_kind_t alg_kind;
    float scale;
    float alpha;
    float beta;

private:
    eltwise_t(alg_kind_t alg_kind, float scale, float alpha, float beta)
        : func_impl_t(_type_info())
        , alg_kind(alg_kind)
        , scale(scale)
        , alpha(alpha)
        , beta(beta) {}
};

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
