/*******************************************************************************
* Copyright 2016-2017 Intel Corporation
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

#include "mkldnn_test_common.hpp"
#include "gtest/gtest.h"

#include "mkldnn.hpp"

namespace mkldnn {

struct sum_test_params {
    const engine::kind engine_kind;
    std::vector<memory::format> srcs_format;
    memory::format dst_format;
    std::vector<memory::dims> srcs_cds;
    memory::dims dst_cds;
    std::vector<double> scale;
};


template <typename data_t>
class sum_test: public ::testing::TestWithParam<sum_test_params> {
    void check_data(const std::vector<memory> &srcs,
                    const std::vector<double> scale,
                    const memory &dst)
    {
        const data_t *dst_data = (const data_t *)dst.get_data_handle();
        const auto &dst_d = dst.get_primitive_desc().desc();
        const auto dst_dims = dst_d.data.dims;

#       pragma omp parallel for collapse(4) schedule(static)
        for (auto n = 0; n < dst_dims[0]; n++)
        for (auto c = 0; c < dst_dims[1]; c++)
        for (auto h = 0; h < dst_dims[2]; h++)
        for (auto w = 0; w < dst_dims[3]; w++) {
            auto src_sum = 0.0;
            for (size_t num = 0; num < srcs.size(); num++) {
                const data_t *src_data =
                    (const data_t *)srcs[num].get_data_handle();
                const auto &src_d = srcs[num].get_primitive_desc().desc();
                const auto src_dims = src_d.data.dims;

                auto src_idx = w
                    + src_dims[3]*h
                    + src_dims[2]*src_dims[3]*c
                    + src_dims[1]*src_dims[2]*src_dims[3]*n;
                src_sum += scale[num]*src_data[map_index(src_d, src_idx)];

            }
            auto dst_idx = w
                + dst_dims[3]
                + dst_dims[2]
                + dst_dims[1];

            EXPECT_NEAR(src_sum,
                        dst_data[map_index(dst_d, dst_idx)],
                        1e-7);
        }
    }

protected:
    virtual void SetUp() {
        sum_test_params p
            = ::testing::TestWithParam<sum_test_params>::GetParam();

        for (size_t i = 0; i < p.srcs_cds.size(); i++) {
            for (size_t dim = 0; dim < p.dst_cds.size(); dim++) {
                ASSERT_TRUE(p.srcs_cds[i][dim] == p.dst_cds[dim]);
            }
        }
        ASSERT_TRUE(p.engine_kind == engine::kind::cpu);
        auto eng = engine(p.engine_kind, 0);
        memory::data_type data_type = data_traits<data_t>::data_type;
        ASSERT_EQ(data_type, mkldnn::memory::data_type::f32);

        std::vector<memory::primitive_desc> srcs_pd;
        std::vector<memory> srcs;
        for (size_t i = 0; i < p.srcs_cds.size(); i++) {
            auto desc =
                memory::desc(p.srcs_cds[i], data_type, p.srcs_format[i]);
            auto mpd = memory::primitive_desc(desc, eng);
            auto src_memory = memory(mpd);
            const size_t sz =
                src_memory.get_primitive_desc().get_size() / sizeof(data_t);
            auto s = (data_t *)src_memory.get_data_handle();
#           pragma omp parallel for
            /* Note: size_t incompatible with MSVC++ */
            for (ptrdiff_t j = 0; j < sz; ++j) s[j] = static_cast<data_t>(i + 1);
            // fill_data<data_t>(sz, (data_t *)src_memory.get_data_handle());
            srcs_pd.push_back(mpd);
            srcs.push_back(src_memory);
        }

        auto dst_desc = memory::desc(p.dst_cds, data_type, p.dst_format);
        auto sum_pd = sum::primitive_desc(dst_desc, p.scale, srcs_pd);
        auto dst = memory(sum_pd.dst_primitive_desc());

        std::vector<primitive::at> inputs;
        for (size_t i = 0; i < p.srcs_cds.size(); i++) {
            inputs.push_back(srcs[i]);
        }
        auto c = sum(sum_pd, inputs, dst);

        ASSERT_EQ(sum_pd.dst_primitive_desc().desc().data.format,
                dst_desc.data.format);
        ASSERT_EQ(sum_pd.dst_primitive_desc().desc().data.ndims,
                dst_desc.data.ndims);

        std::vector<primitive> pipeline;
        pipeline.push_back(c);
        auto s = stream(stream::kind::eager);
        s.submit(pipeline).wait();

        check_data(srcs, p.scale, dst);
    }
};

using sum_test_float = sum_test<float>;
using sum_test_params_float = sum_test_params;

TEST_P(sum_test_float, TestsSum)
{
}

INSTANTIATE_TEST_CASE_P(TestSum, sum_test_float, ::testing::Values(

    sum_test_params_float{engine::kind::cpu,
    {memory::format::nchw, memory::format::nchw}, memory::format::nchw,
    {{2, 8, 3, 4}, {2, 8, 3, 4}}, {2, 8, 3, 4}, {1.0, 1.0}},
    sum_test_params_float{engine::kind::cpu,
    {memory::format::nChw8c, memory::format::nChw8c}, memory::format::nChw8c,
    {{2, 16, 1, 1}, {2, 16, 1, 1}}, {2, 16, 1, 1}, {1.0, 1.0}},
    sum_test_params_float{engine::kind::cpu,
    {memory::format::nchw, memory::format::nchw}, memory::format::nChw8c,
    {{2, 16, 1, 1}, {2, 16, 1, 1}}, {2, 16, 1, 1}, {1.0, 1.0}},
    sum_test_params_float{engine::kind::cpu,
    {memory::format::nChw8c, memory::format::nChw8c}, memory::format::nchw,
    {{2, 16, 1, 1}, {2, 16, 1, 1}}, {2, 16, 1, 1}, {1.0, 1.0}},

    sum_test_params_float{engine::kind::cpu,
    {memory::format::nchw, memory::format::nchw}, memory::format::nchw,
    {{2, 8, 3, 4}, {2, 8, 3, 4}}, {2, 8, 3, 4}, {2.0, 3.0}},
    sum_test_params_float{engine::kind::cpu,
    {memory::format::nChw8c, memory::format::nChw8c}, memory::format::nChw8c,
    {{2, 16, 1, 1}, {2, 16, 1, 1}}, {2, 16, 1, 1}, {2.0, 3.0}},
    sum_test_params_float{engine::kind::cpu,
    {memory::format::nchw, memory::format::nchw}, memory::format::nChw8c,
    {{2, 16, 1, 1}, {2, 16, 1, 1}}, {2, 16, 1, 1}, {2.0, 3.0}},
    sum_test_params_float{engine::kind::cpu,
    {memory::format::nChw8c, memory::format::nChw8c}, memory::format::nchw,
    {{2, 16, 1, 1}, {2, 16, 1, 1}}, {2, 16, 1, 1}, {2.0, 3.0}},

    sum_test_params_float{engine::kind::cpu,
    {memory::format::nchw, memory::format::nChw8c}, memory::format::nchw,
    {{2, 8, 3, 4}, {2, 8, 3, 4}}, {2, 8, 3, 4}, {2.0, 3.0}},
    sum_test_params_float{engine::kind::cpu,
    {memory::format::nchw, memory::format::nChw8c}, memory::format::nchw,
    {{32, 32, 13, 14}, {32, 32, 13, 14}}, {32, 32, 13, 14}, {2.0, 3.0}}
));

}
