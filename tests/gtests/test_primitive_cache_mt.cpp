/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#include "dnnl_test_common.hpp"
#include "gtest/gtest.h"

#include "dnnl.hpp"

namespace dnnl {

TEST(primitive_cache_mt_test, TestGeneralCase) {
    using tag = memory::format_tag;
    using dt = memory::data_type;

    engine eng(get_test_engine_kind(), 0);

    int n_primitives = 12;

    dnnl::impl::parallel_nd(n_primitives, [&](int np) {
        auto relu_d = eltwise_forward::desc(prop_kind::forward_inference,
                algorithm::eltwise_relu, {{np, 1, 1, 1}, dt::f32, tag::nchw},
                0.f, 0.f);
        auto relu_pd = eltwise_forward::primitive_desc(relu_d, eng);
        auto relu = eltwise_forward(relu_pd);
    });
}

TEST(primitive_cache_mt_test, TestNestedCase) {
    using tag = memory::format_tag;
    using dt = memory::data_type;

    engine eng(get_test_engine_kind(), 0);

    int n_primitives = 12;
    int n_srcs = 32;

    dnnl::impl::parallel_nd(n_primitives, [&](int np) {
        std::vector<memory::desc> src_mds(n_srcs);
        std::vector<float> scales(n_srcs, 1.0);

        for (int ns = 0; ns < n_srcs; ++ns) {
            src_mds[ns] = memory::desc({{128, 128}, dt::f32, tag::nc});
        }
        auto sum_pd = sum::primitive_desc(scales, src_mds, eng);
        auto sum_prim = sum(sum_pd);
    });
}

} // namespace dnnl
