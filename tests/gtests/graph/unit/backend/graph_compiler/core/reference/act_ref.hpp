/*******************************************************************************
 * Copyright 2020-2023 Intel Corporation
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
#ifndef GRAPH_UNIT_BACKEND_GRAPH_COMPILER_CORE_REFERENCE_ACT_REF_HPP
#define GRAPH_UNIT_BACKEND_GRAPH_COMPILER_CORE_REFERENCE_ACT_REF_HPP

#include <algorithm>
#include <cfenv>
#include <cmath>
#include <stdlib.h>
#include <test_utils.hpp>
namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
template <typename T>
static void ref_relu(T *out, const T *in, size_t size) {
    test_utils::parallel_nd(static_cast<int>(size),
            [&](int64_t i) { out[i] = std::max(in[i], static_cast<T>(0)); });
}

template <typename T>
static void ref_leaky_relu(T *out, const T *in, size_t size, float alpha) {
    test_utils::parallel_nd(static_cast<int>(size), [&](int64_t i) {
        out[i] = (in[i] > 0) ? in[i] : T(alpha * in[i]);
    });
}

template <typename T>
static void ref_prelu(T *out, const T *in, const T *alpha, size_t size) {
    test_utils::parallel_nd(static_cast<int>(size), [&](int64_t i) {
        out[i] = (in[i] > 0) ? in[i] : T(alpha[i] * in[i]);
    });
}

template <typename T>
static void ref_round(T *out, const T *in, size_t size) {
    auto old_round = std::fegetround();
    std::fesetround(FE_TONEAREST);
    test_utils::parallel_nd(static_cast<int>(size),
            [&](int64_t i) { out[i] = std::nearbyint(in[i]); });
    std::fesetround(old_round);
}

template <typename T>
static void ref_sigmoid(T *out, const T *in, size_t size) {
    test_utils::parallel_nd(static_cast<int>(size),
            [&](int64_t i) { out[i] = 1.0f / (1.0f + expf(-in[i])); });
}

template <typename T>
static void ref_tanh(T *out, const T *in, size_t size) {
    test_utils::parallel_nd(
            static_cast<int>(size), [&](int64_t i) { out[i] = tanhf(in[i]); });
}

template <typename T>
static void ref_clamp(
        T *out, const T *in, size_t size, float clamp_min, float clamp_max) {
    test_utils::parallel_nd(static_cast<int>(size), [&](int64_t i) {
        out[i] = std::max(std::min(in[i], static_cast<T>(clamp_max)),
                static_cast<T>(clamp_min));
    });
}

template <typename T>
static void ref_erf(T *out, const T *in, size_t size) {
    test_utils::parallel_nd(
            static_cast<int>(size), [&](int64_t i) { out[i] = erff(in[i]); });
}

template <typename T>
static void ref_gelu_tanh(T *out, const T *in, size_t size) {
    test_utils::parallel_nd(static_cast<int>(size), [&](int64_t i) {
        float sqrt_2_over_pi = 0.79788458347320556640625;
        float fitting_const = 0.044715;
        float v = tanh(
                sqrt_2_over_pi * in[i] * (1 + fitting_const * in[i] * in[i]));
        out[i] = 0.5f * in[i] * (1.0f + v);
    });
}

template <typename T>
static void ref_gelu_erf(T *out, const T *in, size_t size) {
    test_utils::parallel_nd(static_cast<int>(size), [&](int64_t i) {
        float sqrt_2 = 1.414213562f;
        out[i] = 0.5f * in[i] * (1.0f + erff(in[i] / sqrt_2));
    });
}

template <typename T>
static void ref_reciprocal(T *out, const T *in, size_t size) {
    test_utils::parallel_nd(
            static_cast<int>(size), [&](int64_t i) { out[i] = 1.0f / in[i]; });
}

template <typename T>
static void ref_abs(T *out, const T *in, size_t size) {
    test_utils::parallel_nd(
            static_cast<int>(size), [&](int64_t i) { out[i] = fabs(in[i]); });
}

template <typename T>
static void ref_elu(T *out, const T *in, size_t size, T alpha) {
    test_utils::parallel_nd(static_cast<int>(size), [&](int64_t i) {
        out[i] = (float)in[i] >= 0.f ? in[i] : (T)(alpha * (expf(in[i]) - 1));
    });
}

template <typename T>
static void ref_hardsigmoid(T *out, const T *in, size_t size, T alpha, T beta) {
    test_utils::parallel_nd(static_cast<int>(size), [&](int64_t i) {
        out[i] = std::max(0.f, std::min(1.f, alpha * in[i] + beta));
    });
}

template <typename T>
static void ref_hardswish(T *out, const T *in, size_t size, T alpha, T beta) {
    test_utils::parallel_nd(static_cast<int>(size), [&](int64_t i) {
        out[i] = in[i] * std::max(0.f, std::min(1.f, alpha * in[i] + beta));
    });
}

template <typename T>
static void ref_linear_elementwise(
        T *out, const T *in, size_t size, T alpha, T beta) {
    test_utils::parallel_nd(static_cast<int>(size),
            [&](int64_t i) { out[i] = in[i] * alpha + beta; });
}

template <typename T>
static void ref_log(T *out, const T *in, size_t size) {
    test_utils::parallel_nd(
            static_cast<int>(size), [&](int64_t i) { out[i] = logf(in[i]); });
}

template <typename T>
static void ref_mish(T *out, const T *in, size_t size) {
    test_utils::parallel_nd(static_cast<int>(size),
            [&](int64_t i) { out[i] = in[i] * tanhf(logf(1 + expf(in[i]))); });
}

template <typename T>
static void ref_pow(T *out, const T *in, size_t size, T ex) {
    test_utils::parallel_nd(static_cast<int>(size),
            [&](int64_t i) { out[i] = powf(in[i], ex); });
}

template <typename T>
static void ref_soft_plus(T *out, const T *in, size_t size, T alpha) {
    test_utils::parallel_nd(static_cast<int>(size), [&](int64_t i) {
        out[i] = (T)1.f / alpha * logf((T)1.f + expf(in[i] * alpha));
    });
}

template <typename T>
static void ref_square(T *out, const T *in, size_t size) {
    test_utils::parallel_nd(
            static_cast<int>(size), [&](int64_t i) { out[i] = in[i] * in[i]; });
}

template <typename T>
static void ref_swish(T *out, const T *in, size_t size, T alpha) {
    test_utils::parallel_nd(static_cast<int>(size), [&](int64_t i) {
        out[i] = in[i] / ((T)1.f + expf(-alpha * in[i]));
    });
}

/** a relu bwd reference implementation.
 * @tparam T data type of inputs
 * @param gout gradient output after relu bwd, as a input to calculate gradient
 * data and weight.
 * @param gin gradient output before relu bwd, received from last layer.
 * @param out fwd output after relu this layer.
 * @param size input size.
 * */
template <typename T>
static void ref_relu_bwd(T *gout, T *gin, T *out, size_t size) {
    test_utils::parallel_nd(static_cast<int>(size), [&](int64_t i) {
        if (out[i] < 0) {
            gout[i] = 0.f;
        } else {
            gout[i] = gin[i];
        }
    });
}

/** a sigmoid bwd reference implementation.
 * @tparam T data type of inputs
 * @param gout gradient output after sigmoid bwd, as a input to calculate
 * gradient data and weight.
 * @param gin gradient output before sigmoid bwd, received from last layer.
 * @param out fwd output after sigmoid this layer.
 * @param size input size.
 * */
template <typename T>
static void ref_sigmoid_bwd(T *gout, T *gin, T *out, size_t size) {
    test_utils::parallel_nd(static_cast<int>(size),
            [&](int64_t i) { gout[i] = gin[i] * (out[i] - out[i] * out[i]); });
}

/** a gelu_tanh bwd reference implementation.
 * @tparam T data type of inputs
 * @param gout gradient output after gelu_tanh bwd, as a input to calculate
 * gradient data and weight.
 * @param gin gradient output before gelu_tanh bwd, received from last layer.
 * @param in fwd input before gelu_tanh this layer.
 * @param size input size.
 * */
template <typename T>
static void ref_gelu_tanh_bwd(T *gout, T *gin, T *in, size_t size) {
    test_utils::parallel_nd(static_cast<int>(size), [&](int64_t i) {
        float sqrt_2_over_pi = 0.79788458347320556640625f;
        float fitting_const = 0.044715f;
        float g = in[i] * sqrt_2_over_pi * (1 + fitting_const * in[i] * in[i]);
        float dg = sqrt_2_over_pi * (1 + 3 * fitting_const * in[i] * in[i]);
        float v = tanh(g);
        gout[i] = gin[i] * 0.5 * (1 + v) * (1 + in[i] * (1 - v) * dg);
    });
}

/** a gelu_erf bwd reference implementation.
 * @tparam T data type of inputs
 * @param gout gradient output after gelu_erf bwd, as a input to calculate
 * gradient data and weight.
 * @param gin gradient output before gelu_erf bwd, received from last layer.
 * @param in fwd input before gelu_erf this layer.
 * @param size input size.
 * */
template <typename T>
static void ref_gelu_erf_bwd(T *gout, T *gin, T *in, size_t size) {
    test_utils::parallel_nd(static_cast<int>(size), [&](int64_t i) {
        float sqrt_2 = 1.414213562f;
        float sqrt_2pi = 2.506628275f;
        gout[i] = gin[i]
                * (0.5f * (1.0f + erff(in[i] / sqrt_2))
                        + in[i] / sqrt_2pi * expf(-in[i] * in[i] / 2));
    });
}
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
#endif
