/*******************************************************************************
* Copyright 2016 Intel Corporation
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

#include "gtest/gtest.h"
#include "mkldnn_test_common.hpp"

#include "mkldnn.hpp"

namespace mkldnn {

template <typename data_t>
void check_relu(prop_kind aprop_kind, data_t negative_slope,
        const memory::desc &md, const data_t *src, const data_t *dst)
{
    ASSERT_EQ(md.data.ndims, 4);
    ASSERT_EQ(md.data.format, memory::convert_to_c(memory::format::nchw));
    ASSERT_EQ(md.data.data_type, memory::convert_to_c(memory::data_type::f32)); // TODO: type assert

    size_t N = md.data.dims[0];
    size_t C = md.data.dims[1];
    size_t H = md.data.dims[2];
    size_t W = md.data.dims[3];
    for (size_t i = 0; i < N * C * H * W; ++i) {
        data_t s = src[i];
        assert_eq(dst[i], s >= 0 ? s : s * negative_slope);
    }
}

template <typename data_t>
struct relu_fwd_test_params {
    data_t negative_slope;
    prop_kind aprop_kind;
    engine::kind engine_kind;
    memory::format memory_format;
    memory::dims dims;
};

template <typename data_t>
class relu_test : public ::testing::TestWithParam<relu_fwd_test_params<data_t>> {
protected:
    virtual void SetUp() {
        relu_fwd_test_params<data_t> p
            = ::testing::TestWithParam<relu_fwd_test_params<data_t>>::GetParam();

        ASSERT_EQ(p.memory_format, memory::format::nchw);
        ASSERT_TRUE(p.engine_kind == engine::kind::cpu);
        auto eng = engine(p.engine_kind, 0);

        ASSERT_EQ(p.dims.size(), 4U);
        size_t size = p.dims[0] * p.dims[1] * p.dims[2] * p.dims[3];

        // TODO: free
        auto src_nchw_data = new data_t[size];
        auto dst_nchw_data = new data_t[size];

        memory::data_type prec = data_traits<data_t>::data_type;
        auto dims = p.dims;

        auto nchw_mem_desc = memory::desc(dims, prec, memory::format::nchw);
        auto nchw_mem_prim_desc = memory::primitive_desc(nchw_mem_desc, eng);
        auto src_nchw = memory(nchw_mem_prim_desc, src_nchw_data);
        auto dst_nchw = memory(nchw_mem_prim_desc, dst_nchw_data);

        fill_data<data_t>(size, (data_t *)src_nchw.get_data_handle(),
                data_t(0), data_t(1));

        auto relu_desc = relu_forward::desc(p.aprop_kind, nchw_mem_desc,
                p.negative_slope);
        auto relu_prim_desc = relu_forward::primitive_desc(relu_desc, eng);
        auto relu = relu_forward(relu_prim_desc, src_nchw, dst_nchw);

        std::vector<primitive> pipeline;
        pipeline.push_back(relu);
        auto s = stream(stream::kind::lazy);
        s.submit(pipeline).wait();

        check_relu(p.aprop_kind, p.negative_slope,
                nchw_mem_desc, src_nchw_data, dst_nchw_data);
    }
};

using relu_forward_test_float = relu_test<float>;
using relu_fwd_test_params_float = relu_fwd_test_params<float>;

TEST_P(relu_forward_test_float, TestsReLU) { }
INSTANTIATE_TEST_CASE_P(TestReLUForward, relu_forward_test_float,
        ::testing::Values(
            relu_fwd_test_params_float{0, prop_kind::forward, engine::kind::cpu,
            memory::format::nchw, {10, 10, 10, 10}},
            relu_fwd_test_params_float{.1f, prop_kind::forward, engine::kind::cpu,
            memory::format::nchw, {256, 64, 8, 16}},
            relu_fwd_test_params_float{.1f, prop_kind::forward, engine::kind::cpu,
            memory::format::nchw, {1, 1, 1, 1}},
            relu_fwd_test_params_float{.1f, prop_kind::forward, engine::kind::cpu,
            memory::format::nchw, {3, 5, 7, 11}}));
}
