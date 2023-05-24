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
#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_UTIL_MATH_UTILS_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_UTIL_MATH_UTILS_HPP
#include <algorithm>
#include <vector>
#include "parallel.hpp"
#include <runtime/config.hpp>
namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace math_utils {

template <class T>
std::vector<T> vector_mul(const std::vector<T> &inputs1,
        const std::vector<T> &inputs2, bool parallel = true) {
    assert(inputs1.size() == inputs2.size() || inputs1.size() == 1UL
            || inputs2.size() == 1UL);
    size_t outsize = std::max(inputs1.size(), inputs2.size());
    std::vector<T> outputs(outsize);
    auto func = [&](uint64_t iter, uint64_t end) {
        T input1, input2;
        if (inputs1.size() == 1UL) {
            input1 = inputs1[0];
        } else {
            input1 = inputs1[iter];
        }
        if (inputs2.size() == 1UL) {
            input2 = inputs2[0];
        } else {
            input2 = inputs2[iter];
        }
        outputs[iter] = input1 * input2;
    };
    if (parallel) {
        utils::parallel(func, 0, outsize);
    } else {
        for (uint64_t i = 0; i < outsize; i++) {
            func(i, outsize);
        }
    }
    return outputs;
}

template <class T>
std::vector<T> vector_mul(const std::vector<T> &inputs1, const T &input2) {
    std::vector<T> outputs(inputs1.size());
    auto func = [&](uint64_t iter, uint64_t end) {
        outputs[iter] = inputs1[iter] * input2;
    };
    utils::parallel(func, 0, inputs1.size());
    return outputs;
}

template <class T>
T get_dims_product(const std::vector<T> &dims) {
    T ret = 1;
    for (unsigned i = 0; i < dims.size(); ++i) {
        ret *= dims[i];
    }
    assert(ret > 0 && "Overflow or non-constant shape detected");
    return ret;
}

template <typename T,
        typename dummy
        = typename std::enable_if<std::is_same<float, std::decay<T>>::value
                || std::is_same<double, std::decay<T>>::value>>
std::vector<T> vector_rcp(const std::vector<T> &inputs) {
    std::vector<T> outputs(inputs.size());
    auto func = [&](uint64_t iter, uint64_t end) {
        outputs[iter] = 1.0 / inputs[iter];
    };
    utils::parallel(func, 0, inputs.size());
    return outputs;
}

inline int nearest_power_of_2(int in) {
    if (in & (in - 1)) {
        in |= in >> 1;
        in |= in >> 2;
        in |= in >> 4;
        in |= in >> 8;
        in |= in >> 16;
        return in + 1;
    }
    return in == 0 ? 1 : in;
}

// get greatest common divisor of block_in and block_out
inline int64_t get_gcd(int64_t a, int64_t b) {
    COMPILE_ASSERT(a * b != 0, "non-zero number is expected");
    int64_t i = std::min(a, b);
    while (a % i != 0 || b % i != 0) {
        i--;
        if (i == 0) return 1;
    }
    return i;
}

} // namespace math_utils
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
#endif
