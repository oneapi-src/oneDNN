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
#ifndef GRAPH_UNIT_BACKEND_GRAPH_COMPILER_CORE_REFERENCE_ELTWISE_REF_HPP
#define GRAPH_UNIT_BACKEND_GRAPH_COMPILER_CORE_REFERENCE_ELTWISE_REF_HPP

#include <assert.h>
#include <memory>
#include <utility>
#include <test_utils.hpp>

using dnnl::impl::graph::gc::sc_dims;

inline void compute_elementwise_ref_direct_fwd(float *src_m1, float *src_m2,
        sc_dims dims, float *dst_m = nullptr, bool inplace = true,
        char op = '+') {
    int64_t ranges = 1;
    float *ref;
    for (auto d : dims)
        ranges *= d;
    if (inplace)
        ref = src_m1;
    else
        ref = dst_m;
    std::function<void(int64_t)> thefunc;
    switch (op) {
        case '+':
            thefunc = [&](int64_t i) { ref[i] = src_m1[i] + src_m2[i]; };
            break;
        case '-':
            thefunc = [&](int64_t i) { ref[i] = src_m1[i] - src_m2[i]; };
            break;
        case '*':
            thefunc = [&](int64_t i) { ref[i] = src_m1[i] * src_m2[i]; };
            break;
        case '/':
            thefunc = [&](int64_t i) { ref[i] = src_m1[i] / src_m2[i]; };
            break;
        default:
            std::cout << "Unexpected elementwise opertaor: " << op << std::endl;
            break;
    }
    dnnl::impl::graph::gc::utils::parallel_for(0, ranges, 1, thefunc);
}

#endif
