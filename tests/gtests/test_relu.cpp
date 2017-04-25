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

#include "gtest/gtest.h"
#include "mkldnn_test_common.hpp"

#include "mkldnn.hpp"

namespace mkldnn {

template <typename data_t>
struct relu_test_params {
    engine::kind engine_kind;
    memory::format memory_format;
    data_t negative_slope;
    memory::dims dims;
};

template <typename data_t>
void check_relu_fwd(data_t negative_slope, const memory::desc &md,
        const memory &src, const memory &dst)
{
    data_t *src_data = (data_t *)src.get_data_handle();
    data_t *dst_data = (data_t *)dst.get_data_handle();

    ASSERT_EQ(md.data.ndims, 4);
    ASSERT_EQ(md.data.format, memory::convert_to_c(memory::format::nchw));
    ASSERT_EQ(md.data.data_type, memory::convert_to_c(memory::data_type::f32)); // TODO: type assert

    size_t N = md.data.dims[0];
    size_t C = md.data.dims[1];
    size_t H = md.data.dims[2];
    size_t W = md.data.dims[3];
    for (size_t i = 0; i < N * C * H * W; ++i) {
        data_t s = src_data[i];
        assert_eq(dst_data[i], s > 0 ? s : s * negative_slope);
    }
}

template <typename data_t>
void check_relu_bwd(data_t negative_slope, const memory::desc &md,
        const memory &src, const memory &diff_dst, const memory &diff_src)
{
    data_t *src_data = (data_t *)src.get_data_handle();
    data_t *diff_dst_data = (data_t *)diff_dst.get_data_handle();
    data_t *diff_src_data = (data_t *)diff_src.get_data_handle();

    ASSERT_EQ(md.data.ndims, 4);
    ASSERT_EQ(md.data.data_type, memory::convert_to_c(memory::data_type::f32)); // TODO: type assert

    size_t N = md.data.dims[0];
    size_t C = md.data.dims[1];
    size_t H = md.data.dims[2];
    size_t W = md.data.dims[3];
    for (size_t i = 0; i < N * C * H * W; ++i) {
        data_t s = src_data[i];
        data_t dd = diff_dst_data[i];
        assert_eq(diff_src_data[i], s > 0 ? dd : dd * negative_slope);
    }
}

template <typename data_t>
class relu_test : public ::testing::TestWithParam<relu_test_params<data_t>> {
private:
    std::shared_ptr<memory> src;
    std::shared_ptr<memory> diff_src;
    std::shared_ptr<memory> dst;
    std::shared_ptr<memory> workspace;
    std::shared_ptr<memory::desc> src_desc;
    std::shared_ptr<memory::desc> dst_desc;
    std::shared_ptr<relu_forward::primitive_desc> relu_prim_desc;
    relu_test_params<data_t> p;
    std::shared_ptr<engine> eng;
    memory::data_type data_type;

protected:
    virtual void SetUp() {
        p = ::testing::TestWithParam<relu_test_params<data_t>>::GetParam();

        ASSERT_EQ(p.memory_format, memory::format::nchw);
        ASSERT_TRUE(p.engine_kind == engine::kind::cpu);
        eng.reset(new engine(p.engine_kind, 0));

        ASSERT_EQ(p.dims.size(), 4U);

        data_type = data_traits<data_t>::data_type;
        ASSERT_EQ(data_type, mkldnn::memory::data_type::f32);

        Forward();
        Backward();
    }

    void Forward() {
        auto dims = p.dims;
        size_t size = p.dims[0] * p.dims[1] * p.dims[2] * p.dims[3];

        src_desc.reset(new memory::desc(dims, data_type,
            p.memory_format));
        dst_desc.reset(new memory::desc(dims, data_type,
            p.memory_format));
        src.reset(new memory({*src_desc, *eng}));
        dst.reset(new memory({*src_desc, *eng}));

        fill_data<data_t>(size, (data_t *)src->get_data_handle(),
                data_t(0), data_t(1));

        auto relu_desc = relu_forward::desc(prop_kind::forward_training,
                *src_desc, p.negative_slope);
        relu_prim_desc.reset(
                new relu_forward::primitive_desc(relu_desc, *eng));
        auto relu = relu_forward(*relu_prim_desc, *src, *dst);

        std::vector<primitive> pipeline;
        pipeline.push_back(relu);
        auto s = stream(stream::kind::lazy);
        s.submit(pipeline).wait();

        check_relu_fwd(p.negative_slope, *src_desc,
            *src, *dst);
    }

    void Backward() {
        diff_src.reset(new memory({*src_desc, *eng}));

        auto relu_bwd_desc = relu_backward::desc(*src_desc, *dst_desc,
                p.negative_slope);
        auto relu_bwd_prim_desc = relu_backward::primitive_desc(relu_bwd_desc,
                *eng, *relu_prim_desc);
        auto relu_bwd = relu_backward(relu_bwd_prim_desc, *src, *dst,
                *diff_src);

        std::vector<primitive> pipeline;
        pipeline.push_back(relu_bwd);
        auto s = stream(stream::kind::lazy);
        s.submit(pipeline).wait();

        check_relu_bwd(p.negative_slope, *src_desc,
            *src, *dst, *diff_src);
    }
};

using relu_test_float = relu_test<float>;
using relu_test_params_float = relu_test_params<float>;

TEST_P(relu_test_float, TestsReLU)
{
}

#define EXPAND_SIZES(mb, c, h, w) { mb, c, h, w }
#define EXPAND_FORMATS(data) memory::format::data

#define ENGINE engine::kind::cpu

#define PARAMS(data, ns, mb, c, h, w) \
    relu_test_params_float { ENGINE, \
    EXPAND_FORMATS(data), ns, EXPAND_SIZES(mb, c, h, w) }

#define INST_TEST_CASE(str, ...) INSTANTIATE_TEST_CASE_P( \
        str, relu_test_float, ::testing::Values(__VA_ARGS__))

INST_TEST_CASE(SimpleZeroNegativeSlope_NCHW,
    PARAMS(nchw, 0.f, 2, 8, 4, 4),
    PARAMS(nchw, 0.f, 2, 16, 4, 4),
    PARAMS(nchw, 0.f, 2, 16, 8, 8),
    PARAMS(nchw, 0.f, 2, 16, 16, 8),
    PARAMS(nchw, 0.f, 2, 16, 10, 8),
    PARAMS(nchw, 0.f, 10, 10, 10, 10),
    PARAMS(nchw, 0.f, 256, 64, 8, 16),
    PARAMS(nchw, 0.f, 1, 1, 1, 1),
    PARAMS(nchw, 0.f, 3, 5, 7, 11)
);

INST_TEST_CASE(Simple_NCHW,
    PARAMS(nchw, 0.1f, 2, 8, 4, 4),
    PARAMS(nchw, 0.1f, 2, 16, 4, 4),
    PARAMS(nchw, 0.1f, 2, 16, 8, 8),
    PARAMS(nchw, 0.1f, 2, 16, 16, 8),
    PARAMS(nchw, 0.1f, 2, 16, 10, 8),
    PARAMS(nchw, 0.1f, 10, 10, 10, 10),
    PARAMS(nchw, 0.1f, 256, 64, 8, 16),
    PARAMS(nchw, 0.1f, 1, 1, 1, 1),
    PARAMS(nchw, 0.1f, 3, 5, 7, 11)
);

INST_TEST_CASE(AlexNet_NCHW,
    PARAMS(nchw, 0.f, 2, 96, 55, 55),
    PARAMS(nchw, 0.f, 2, 256, 27, 27),
    PARAMS(nchw, 0.f, 2, 384, 13, 13)
);

}
