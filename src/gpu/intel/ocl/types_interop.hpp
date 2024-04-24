/*******************************************************************************
 * Copyright 2024 Intel Corporation
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

#ifndef GPU_INTEL_OCL_TYPES_INTEROP_HPP
#define GPU_INTEL_OCL_TYPES_INTEROP_HPP

#include "gpu/intel/compute/kernel_arg_list.hpp"
#include "gpu/intel/ocl/types_interop.h"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace compute {

template <>
struct scalar_type_traits<int64x2_t> {
    static const auto type = scalar_type_t::_int64x3_t;
};

template <>
struct scalar_type_traits<int64x3_t> {
    static const auto type = scalar_type_t::_int64x3_t;
};

template <>
struct scalar_type_traits<int64x4_t> {
    static const auto type = scalar_type_t::_int64x4_t;
};

template <>
struct scalar_type_traits<int64x5_t> {
    static const auto type = scalar_type_t::_int64x5_t;
};

template <>
struct scalar_type_traits<int64x6_t> {
    static const auto type = scalar_type_t::_int64x6_t;
};

template <>
struct scalar_type_traits<dispatch_gws_rt_params_t> {
    static const auto type = scalar_type_t::_dispatch_gws_rt_params_t;
};

template <>
struct scalar_type_traits<zero_pad_mask_t> {
    static const auto type = scalar_type_t::_zero_pad_mask_t;
};

} // namespace compute
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
