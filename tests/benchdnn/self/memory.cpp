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

#include "dnnl_memory.hpp"

#include "self/self.hpp"

namespace self {

static int check_bool_operator() {
    dnnl_dim_t dims {1};
    auto md = dnn_mem_t::init_md(1, &dims, dnnl_f32, tag::abx);
    auto md0 = dnn_mem_t::init_md(0, &dims, dnnl_f32, tag::abx);
    {
        dnn_mem_t m;
        SELF_CHECK_EQ(bool(m), false);
    }
    {
        dnn_mem_t m(md, get_test_engine());
        SELF_CHECK_EQ(bool(m), true);
        dnn_mem_t n(md0, get_test_engine());
        SELF_CHECK_EQ(bool(n), false);
    }
    {
        dnn_mem_t m(1, &dims, dnnl_f32, tag::abx, get_test_engine());
        SELF_CHECK_EQ(bool(m), true);
        dnn_mem_t n(0, &dims, dnnl_f32, tag::abx, get_test_engine());
        SELF_CHECK_EQ(bool(n), false);
    }
    {
        dnn_mem_t m(1, &dims, dnnl_f32, &dims /* strides */, get_test_engine());
        SELF_CHECK_EQ(bool(m), true);
        dnn_mem_t n(0, &dims, dnnl_f32, &dims /* strides */, get_test_engine());
        SELF_CHECK_EQ(bool(n), false);
    }
    {
        dnn_mem_t m(md, dnnl_f32, tag::abx, get_test_engine());
        SELF_CHECK_EQ(bool(m), true);
        dnn_mem_t n(md0, dnnl_f32, tag::abx, get_test_engine());
        SELF_CHECK_EQ(bool(n), false);
    }
    return OK;
}

void memory() {
    RUN(check_bool_operator());
}

} // namespace self
