/*******************************************************************************
* Copyright 2023 Intel Corporation
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

#include <stdio.h>
#include <stdlib.h>

#include "self/self.hpp"

#include "utils/compare.hpp"

namespace self {

static int check_compare() {
    {
        res_t res {};
        res.state = EXECUTED;
        dnnl_dims_t dims {100};
        dnn_mem_t m0(1, dims, dnnl_f32, tag::abx, get_cpu_engine());
        dnn_mem_t m1(1, dims, dnnl_f32, tag::abx, get_cpu_engine());
        compare::compare_t cmp;
        cmp.set_zero_trust_percent(100.f);
        for (int i = 0; i < dims[0]; i++) {
            m0.set_elem(i, 0);
            m1.set_elem(i, 0);
        }
        cmp.compare(m0, m1, attr_t(), &res);
        SELF_CHECK_EQ(res.state, PASSED);

        // Ensure that `compare` finds exactly 100 zeros.
        cmp.set_zero_trust_percent(99.f);
        cmp.compare(m0, m1, attr_t(), &res);
        SELF_CHECK_EQ(res.state, MISTRUSTED);
    }
    {
        res_t res {};
        res.state = EXECUTED;
        dnnl_dims_t dims {100};
        dnn_mem_t m0(1, dims, dnnl_f32, tag::abx, get_cpu_engine());
        dnn_mem_t m1(1, dims, dnnl_f32, tag::abx, get_cpu_engine());
        compare::compare_t cmp;
        cmp.set_threshold(99.f);
        for (int i = 0; i < dims[0]; i++) {
            m0.set_elem(i, 1);
            m1.set_elem(i, i + 1);
        }
        cmp.compare(m0, m1, attr_t(), &res);
        SELF_CHECK_EQ(res.state, PASSED);

        // Ensure that `compare` finds the biggest max_rdiff and max_diff.
        // (visual confirmation only).
        cmp.set_threshold(98.f);
        cmp.compare(m0, m1, attr_t(), &res);
        SELF_CHECK_EQ(res.state, FAILED);
    }
    return OK;
}

void compare() {
    RUN(check_compare());
}

} // namespace self
