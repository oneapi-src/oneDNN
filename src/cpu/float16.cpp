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

#include "common/float16.hpp"
#include "common/dnnl_thread.hpp"

#include "cpu/platform.hpp"

namespace dnnl {
namespace impl {

void cvt_float_to_float16(float16_t *out, const float *inp, size_t nelems) {
    PRAGMA_OMP_SIMD()
    for (size_t i = 0; i < nelems; ++i)
        out[i] = static_cast<float16_t>(inp[i]);
}

void cvt_float16_to_float(float *out, const float16_t *inp, size_t nelems) {
    PRAGMA_OMP_SIMD()
    for (size_t i = 0; i < nelems; ++i)
        out[i] = inp[i];
}

} // namespace impl
} // namespace dnnl
