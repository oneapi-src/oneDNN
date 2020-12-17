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

#ifndef LLGA_BACKEND_DNNL_UTILS_HPP
#define LLGA_BACKEND_DNNL_UTILS_HPP

#include <algorithm>
#include <atomic>
#include <chrono>
#include <climits>
#include <cstring>
#include <dnnl.h>
#include <dnnl.hpp>
#include <iterator>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "interface/logical_tensor.hpp"

namespace llga {
namespace impl {
namespace dnnl_impl {
namespace utils {
template <typename F, typename T,
        typename U = decltype(std::declval<F>()(std::declval<T>()))>
std::vector<U> fmap(const std::vector<T> &vec, const F &f) {
    std::vector<U> result;
    std::transform(vec.begin(), vec.end(), std::back_inserter(result), f);
    return result;
}

template <typename T, typename P>
constexpr bool one_of(T val, P item) {
    return val == item;
}

template <typename T, typename P, typename... Args>
constexpr bool one_of(T val, P item, Args... item_others) {
    return val == item || one_of(val, item_others...);
}

template <typename T>
inline bool any_le(const std::vector<T> &v, T i) {
    return std::any_of(v.begin(), v.end(), [i](T k) { return k <= i; });
}

inline memory::dims get_compatible_dilates(
        const memory::dims &dilates, size_t input_size = 4) {
    if (!dilates.empty() && !any_le(dilates, static_cast<dim>(0)))
        return fmap(dilates, [](dim x) { return x - 1; });
    if (4 == input_size) {
        return {0, 0};
    } else {
        return {0, 0, 0};
    }
}

inline memory::dims group_dims(const dims &adims, dim groups) {
    auto new_dims = adims;
    new_dims.insert(new_dims.begin(), groups);
    new_dims[1] /= groups;
    return new_dims;
}

inline dnnl::algorithm rnn_kind_to_algorithm(rnn_kind rnn) {
    if (rnn == RNN_RELU || rnn == RNN_TANH) {
        return dnnl::algorithm::vanilla_rnn;
    } else if (rnn == LSTM) {
        return dnnl::algorithm::vanilla_lstm;
    } else if (rnn == GRU) {
        return dnnl::algorithm::lbr_gru;
    } else {
        return dnnl::algorithm::undef;
    }
}

inline dnnl::algorithm rnn_kind_to_activation(rnn_kind rnn) {
    if (rnn == RNN_RELU) {
        return dnnl::algorithm::eltwise_relu;
    } else if (rnn == RNN_TANH || rnn == LSTM || rnn == GRU) {
        return dnnl::algorithm::eltwise_tanh;
    } else {
        return dnnl::algorithm::undef;
    }
}

inline std::pair<std::vector<float>, std::vector<float>> compute_scales(
        float src_scale, float dst_scale, std::vector<float> weight_scales) {
    auto scale_size = weight_scales.size();
    std::vector<float> bias_scales(scale_size), op_scales(scale_size);

    for (int i = 0; i < scale_size; i++) {
        bias_scales[i] = src_scale * weight_scales[i];
        op_scales[i] = dst_scale / bias_scales[i];
    }
    return std::make_pair(std::move(bias_scales), std::move(op_scales));
}

/** sorts an array of values using @p comparator. While sorting the array
 * of value, the function permutes an array of @p keys accordingly.
 *
 * @note The arrays of @p keys can be omitted. In this case the function
 *       sorts the array of @vals only.
 */
template <typename T, typename U, typename F>
inline void simultaneous_sort(T *vals, U *keys, size_t size, F comparator) {
    if (size == 0) return;

    for (auto i = 0; i < size - 1; ++i) {
        bool swapped = false;
        for (auto j = 0; j < size - i - 1; j++) {
            if (comparator(vals[j], vals[j + 1]) > 0) {
                std::swap(vals[j], vals[j + 1]);
                if (keys) std::swap(keys[j], keys[j + 1]);
                swapped = true;
            }
        }

        if (swapped == false) break;
    }
}

template <typename T>
inline T rnd_up(const T a, const T b) {
    return (a + b - 1) / b * b;
}

inline int op_scale_mask(dim scale_size) {
    return scale_size > 1 ? 2 : 0;
}

inline int tensor_scale_mask(dim scale_size, bool grouped) {
    return scale_size > 1 ? grouped ? 3 : 1 : 0;
}

inline int tensor_zp_mask(dim zp_size) {
    return zp_size > 1 ? 1 : 0;
}

inline uintptr_t mod_ptr(void *ptr, size_t bytes) {
    return reinterpret_cast<uintptr_t>(ptr) & (bytes - 1);
}

inline bool is_aligned_ptr(void *ptr, size_t bytes) {
    return mod_ptr(ptr, bytes) == 0;
}

#define BACKEND_DNNL_TYPE_DISPATCH(type_enum, type_key, ...) \
    switch (type_enum) { \
        case data_type::f32: { \
            using type_key = float; \
            __VA_ARGS__ \
        } break; \
        default: error::wrap_c_api(dnnl_unimplemented, "Unimplemented type"); \
    }

} // namespace utils
} // namespace dnnl_impl
} // namespace impl
} // namespace llga

#endif
