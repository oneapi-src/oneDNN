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

#ifndef CPU_SCALE_UTILS_HPP
#define CPU_SCALE_UTILS_HPP

#include "common/c_types_map.hpp"
#include "common/memory_tracking.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

void book_precomputed_scales(memory_tracking::registrar_t &scratchpad,
        const arg_scales_t &attr_scales, size_t oc);

// By default returns the original wei_scales buffer as a dequantization scale.
// If both src_scales and wei_scales are set, returns a scratchpad memory that
// contains src_scale * wei_scale as a dequantization scale.
const float *precompute_scales(const memory_tracking::grantor_t &scratchpad,
        const float *src_scales, const float *wei_scales, size_t oc,
        const primitive_attr_t *attr);

} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_SCALE_UTILS_HPP
