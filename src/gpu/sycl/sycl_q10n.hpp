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

#ifndef GPU_SYCL_SYCL_Q10N_HPP
#define GPU_SYCL_SYCL_Q10N_HPP

#include "common/c_types_map.hpp"
#include "common/math_utils.hpp"
#include "common/utils.hpp"
#include "gpu/sycl/sycl_types.hpp"
#include "sycl/sycl_utils.hpp"

namespace dnnl {
namespace impl {
namespace sycl {

template <typename data_t, typename acc_t, int width>
inline ::sycl::vec<typename utils::enable_if<!nstl::is_integral<data_t>::value,
                           typename utils::remove_reference<acc_t>::type>::type,
        width>
saturate_vec(const ::sycl::vec<acc_t, width> &src_vec) {
    ::sycl::vec<acc_t, width> src_copy_vec = src_vec;
    return src_copy_vec;
}

template <typename data_t, typename acc_t, int width>
inline ::sycl::vec<typename utils::enable_if<nstl::is_integral<data_t>::value,
                           typename utils::remove_reference<acc_t>::type>::type,
        width>
saturate_vec(const ::sycl::vec<acc_t, width> &src_vec) {
    ::sycl::vec<acc_t, width> src_copy_vec = src_vec;

    acc_t lbound = (acc_t)std::numeric_limits<data_t>::lowest();
    acc_t ubound = (acc_t)std::numeric_limits<data_t>::max();
    ::sycl::vec<acc_t, width> lbound_vec(lbound);
    ::sycl::vec<acc_t, width> ubound_vec(ubound);

    // lbound
    auto src_vec_mask
            = ((src_vec >= lbound_vec) * -1).template convert<acc_t>();
    auto i_src_vec_mask
            = ((src_vec < lbound_vec) * -lbound).template convert<acc_t>();

    src_copy_vec *= src_vec_mask;
    src_copy_vec += i_src_vec_mask;

    // ubound
    src_vec_mask
            = ((src_copy_vec <= ubound_vec) * -1).template convert<acc_t>();
    i_src_vec_mask
            = ((src_copy_vec > ubound_vec) * -ubound).template convert<acc_t>();

    src_copy_vec *= src_vec_mask;
    src_copy_vec += i_src_vec_mask;

    return src_copy_vec;
}

template <>
inline ::sycl::vec<uint8_t, 8> saturate_vec<int8_t, uint8_t, 8>(
        const ::sycl::vec<uint8_t, 8> &src_vec) {
    ::sycl::vec<uint8_t, 8> ubound_vec(127u);

    auto src_copy_vec = src_vec;
    // Mask to nullify elements that are greater than ubound in src_vec.
    auto src_vec_mask = (src_vec <= ubound_vec) * -1;
    auto i_src_vec_mask = (src_vec > ubound_vec) * -127;
    // Nullify elements that are greater than ubound.
    src_copy_vec *= src_vec_mask.template convert<uint8_t>();
    // Combine ubound scaled elements with the elements that are lesser
    // than ubound in `src_copy_vec`.
    return src_copy_vec + i_src_vec_mask.template convert<uint8_t>();
}

template <>
inline ::sycl::vec<int8_t, 8> saturate_vec<uint8_t, int8_t, 8>(
        const ::sycl::vec<int8_t, 8> &src_vec) {
    ::sycl::vec<int8_t, 8> lbound_vec(0);

    auto src_copy_vec = src_vec;
    // Mask to nullify elements that are lesser than lbound in src_vec.
    auto src_vec_mask = (src_vec >= lbound_vec) * -1;
    // Nullify elements that are lesser than lbound.
    src_copy_vec *= src_vec_mask.template convert<int8_t>();
    // Return the `src_copy_vec`.
    return src_copy_vec;
}

template <typename data_t, typename acc_t>
inline typename utils::enable_if<!nstl::is_integral<data_t>::value,
        typename utils::remove_reference<acc_t>::type>::type
saturate(const acc_t &x) {
    acc_t v = x;
    return v;
}

template <typename data_t, typename acc_t>
inline typename utils::enable_if<nstl::is_integral<data_t>::value,
        typename utils::remove_reference<acc_t>::type>::type
saturate(const acc_t &x) {
    acc_t v = x;
    acc_t lbound = (acc_t)std::numeric_limits<data_t>::lowest();
    acc_t ubound = (acc_t)std::numeric_limits<data_t>::max();
    if (v < lbound) v = lbound;
    if (v > ubound) v = ubound;
    return v;
}

template <>
inline uint8_t saturate<int8_t, uint8_t>(const uint8_t &x) {
    return x <= 127u ? x : 127;
}

template <>
inline int8_t saturate<uint8_t, int8_t>(const int8_t &x) {
    return x >= 0 ? x : 0;
}

template <typename out_t, int width>
inline ::sycl::vec<typename utils::enable_if<nstl::is_integral<out_t>::value,
                           typename utils::remove_reference<out_t>::type>::type,
        width>
out_round_vec(::sycl::vec<float, width> src_vec) {
    return src_vec.template convert<out_t, ::sycl::rounding_mode::rte>();
}

template <typename out_t, int width>
inline ::sycl::vec<typename utils::enable_if<!nstl::is_integral<out_t>::value,
                           typename utils::remove_reference<out_t>::type>::type,
        width>
out_round_vec(::sycl::vec<float, width> src_vec) {
    return src_vec.template convert<out_t>();
}

template <typename out_t>
inline typename utils::enable_if<nstl::is_integral<out_t>::value,
        typename utils::remove_reference<out_t>::type>::type
out_round(float v) {
    return (out_t)::sycl::rint(v);
}

template <typename out_t>
inline typename utils::enable_if<!nstl::is_integral<out_t>::value,
        typename utils::remove_reference<out_t>::type>::type
out_round(float v) {
    return v;
}

template <typename out_t, typename acc_t = float>
inline out_t saturate_and_round(acc_t f) {
    return out_round<out_t>(saturate<out_t, acc_t>(f));
}

template <typename out_t, typename acc_t = float, int width = 8>
inline ::sycl::vec<out_t, width> saturate_and_round_vec(
        ::sycl::vec<acc_t, width> f_vec) {
    return out_round_vec<out_t, width>(saturate_vec<out_t, acc_t>(f_vec));
}

// Quantization with alpha == 1 and beta == 0
template <typename in_t, typename out_t, typename enabled = void>
struct qz_a1b0 {
    out_t operator()(in_t in) { return saturate_and_round<out_t>((float)in); }
};

// Quantization with alpha == 1 and beta == 0
template <typename in_t, typename out_t, int width = 8, typename enabled = void>
struct qz_a1b0_vec {
    ::sycl::vec<out_t, width> operator()(::sycl::vec<in_t, width> in) {
        return saturate_and_round_vec<out_t>(in.template convert<float>());
    }
};

template <typename in_t, typename out_t>
struct qz_a1b0<in_t, out_t,
        typename utils::enable_if<nstl::is_integral<in_t>::value
                && !is_subset<in_t, out_t>::value>::type> {
    out_t operator()(in_t in) { return saturate<out_t>(in); }
};

template <typename in_t, typename out_t>
struct qz_a1b0<in_t, out_t,
        typename utils::enable_if<is_subset<in_t, out_t>::value>::type> {
    out_t operator()(in_t in) { return (out_t)in; }
};

} // namespace sycl
} // namespace impl
} // namespace dnnl

#endif
