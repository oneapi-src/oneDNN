/*******************************************************************************
* Copyright 2017-2020 Intel Corporation
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

#ifndef CPU_SIMPLE_Q10N_HPP
#define CPU_SIMPLE_Q10N_HPP

#include <assert.h>

#include "c_types_map.hpp"
#include "math_utils.hpp"
#include "nstl.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

template <typename out_t>
inline out_t round_and_saturate(float f) {
    return math::saturate<out_t>(math::out_round<int>(f));
}

/* Quantization with alpha == 1 and beta == 0 */
template <typename in_t, typename out_t, typename enabled = void>
struct qz_a1b0 {
    out_t operator()(in_t in) { return round_and_saturate<out_t>((float)in); }
};

template <typename in_t, typename out_t>
struct qz_a1b0<in_t, out_t,
        typename utils::enable_if<true && nstl::is_integral<in_t>::value
                && !is_subset<in_t, out_t>::value>::type> {
    out_t operator()(in_t in) { return math::saturate<out_t>(in); }
};

template <typename in_t, typename out_t>
struct qz_a1b0<in_t, out_t,
        typename utils::enable_if<is_subset<in_t, out_t>::value>::type> {
    out_t operator()(in_t in) { return (out_t)in; }
};

/* Quantization with alpha == 1 */
template <typename in_t, typename out_t>
struct qz_a1 {
    out_t operator()(in_t in, out_t out, float beta) {
        return round_and_saturate<out_t>((float)in + beta * out);
    }
};

template <typename in_t>
struct qz_a1<in_t, float> {
    float operator()(in_t in, float out, float beta) {
        return (float)in + beta * out;
    }
};

/* Quantization with beta == 0 */
template <typename in_t, typename out_t>
struct qz_b0 {
    out_t operator()(in_t in, float alpha) {
        return round_and_saturate<out_t>(alpha * in);
    }
};

template <typename in_t>
struct qz_b0<in_t, float> {
    float operator()(in_t in, float alpha) { return alpha * in; }
};

/* Quantization */
template <typename in_t, typename out_t>
struct qz {
    out_t operator()(in_t in, out_t out, float alpha, float beta) {
        return round_and_saturate<out_t>(alpha * in + (beta ? beta * out : 0));
    }
};

template <typename in_t>
struct qz<in_t, float> {
    float operator()(in_t in, float out, float alpha, float beta) {
        return alpha * in + (beta ? beta * out : 0);
    }
};

template <>
struct qz<bfloat16_t, bfloat16_t> {
    float operator()(bfloat16_t in, bfloat16_t out, float alpha, float beta) {
        return (bfloat16_t)(alpha * (float)in + (beta ? beta * (float)out : 0));
    }
};

template <>
struct qz<float, bfloat16_t> {
    float operator()(float in, bfloat16_t out, float alpha, float beta) {
        return (bfloat16_t)(alpha * in + (beta ? beta * out : 0));
    }
};

template <>
struct qz<float16_t, float16_t> {
    float operator()(float16_t in, float16_t out, float alpha, float beta) {
        return (float16_t)(alpha * (float)in + (beta ? beta * (float)out : 0));
    }
};

template <>
struct qz<float, float16_t> {
    float operator()(float in, float16_t out, float alpha, float beta) {
        return (float16_t)(alpha * in + (beta ? beta * out : 0));
    }
};

} // namespace cpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino=+2s,^=l0,\:0,N-s
#endif
