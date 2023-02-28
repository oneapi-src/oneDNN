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
#ifndef GRAPH_UNIT_BACKEND_GRAPH_COMPILER_CORE_REFERENCE_BIAS_REF_HPP
#define GRAPH_UNIT_BACKEND_GRAPH_COMPILER_CORE_REFERENCE_BIAS_REF_HPP

#include <stdlib.h>
#include <test_utils.hpp>
namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
template <typename T>
static void ref_bias_bwd(T *out, T *in, size_t size, size_t sum_size) {
    test_utils::parallel_nd(static_cast<int>(size), [&](int64_t i) {
        out[i] = 0.f;
        for (unsigned k = 0; k < sum_size; ++k) {
            out[i] += in[k * size + i];
        }
    });
}
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
#endif
