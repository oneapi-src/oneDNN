/*******************************************************************************
* Copyright 2022-2024 Intel Corporation
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

#ifndef GPU_GENERIC_SYCL_SYCL_UTILS_HPP
#define GPU_GENERIC_SYCL_SYCL_UTILS_HPP

#include "common/memory_desc.hpp"
#include "common/memory_desc_wrapper.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace generic {
namespace sycl {

inline bool md_dims_in_range(const dnnl::impl::memory_desc_t *desc) {
    auto wrap = dnnl::impl::memory_desc_wrapper(desc);
    for (int i = 0; i < wrap.ndims(); i++) {
        if (wrap.dims()[i] > INT_MAX) { return false; }
    }

    return true;
}

// copy from type_helpers.hpp, just without the assert
inline size_t data_type_size(data_type_t data_type) {
    using namespace data_type;
    switch ((int)data_type) {
        case f8_e5m2: return sizeof(prec_traits<f8_e5m2>::type);
        case f8_e4m3: return sizeof(prec_traits<f8_e4m3>::type);
        case f16: return sizeof(prec_traits<f16>::type);
        case bf16: return sizeof(prec_traits<bf16>::type);
        case tf32: // the tf32 type is an f32
        case f32: return sizeof(prec_traits<f32>::type);
        case f64: return sizeof(prec_traits<f64>::type);
        case s32: return sizeof(prec_traits<s32>::type);
        case s8: return sizeof(prec_traits<s8>::type);
        case u8: return sizeof(prec_traits<u8>::type);
        case s4: return sizeof(prec_traits<s4>::type);
        case u4: return sizeof(prec_traits<u4>::type);
        case boolean: return sizeof(prec_traits<boolean>::type);
    }
    return (size_t)-1; /* not supposed to be reachable */
}

} // namespace sycl
} // namespace generic
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
