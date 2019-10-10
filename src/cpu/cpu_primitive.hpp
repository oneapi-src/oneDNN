/*******************************************************************************
* Copyright 2019 Intel Corporation
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

#ifndef CPU_PRIMITIVE_HPP
#define CPU_PRIMITIVE_HPP

#include <assert.h>

#include "dnnl_types.h"

#include "common/c_types_map.hpp"
#include "common/primitive_attr.hpp"
#include "common/primitive_exec_types.hpp"
#include "common/utils.hpp"
#include "common/z_magic.hpp"

#define DEFINE_SCALES_BUFFER(scales) \
    alignas(16) float CONCAT2(scales, _buf16)[16] = {0}; \
    const float *scales; \
    if (pd()->attr()->output_scales_.defined()) { \
        scales = pd()->attr()->output_scales_.scales_; \
    } else { \
        scales = CTX_IN_MEM(const float *, DNNL_ARG_ATTR_OUTPUT_SCALES); \
        if (scales == nullptr) return status::invalid_arguments; \
        const auto scales_d = ctx.memory_mdw(DNNL_ARG_ATTR_OUTPUT_SCALES); \
        bool ok = scales_d.data_type() == data_type::f32 \
                && scales_d.ndims() == 1; \
        if (!ok) return status::invalid_arguments; \
        if (scales_d.dims()[0] == 1) { \
            utils::array_set(CONCAT2(scales, _buf16), scales[0], 16); \
            scales = CONCAT2(scales, _buf16); \
        } \
    } \
    MAYBE_UNUSED(scales);

#endif // CPU_PRIMITIVE_HPP
