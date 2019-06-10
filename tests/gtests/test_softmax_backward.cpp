/*******************************************************************************
* Copyright 2016-2018 Intel Corporation
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
#include <memory>

namespace mkldnn {

template <typename data_t>
void check_softmax_bwd(memory& dst, memory& diff_dst, memory &diff_src, int axis)
{
    auto dst_ptr = map_memory<data_t>(dst);
    auto diff_dst_ptr = map_memory<data_t>(diff_dst);
    auto diff_src_ptr = map_memory<data_t>(diff_src);

    const memory::desc dst_pd = dst.get_desc();
    const memory::desc diff_dst_pd = diff_dst.get_desc();

    const mkldnn::impl::memory_desc_wrapper dst_mdw(dst_pd.data);
    const mkldnn::impl::memory_desc_wrapper diff_dst_mdw(diff_dst_pd.data);

    ASSERT_EQ(diff_dst_mdw.data_type(),
            memory::data_type::f32); // TODO: type assert

    auto ndims = diff_dst_pd.data.ndims;
    const float eps = 1e-7; //TODO: What should be the threshold?

    memory::dim OU = 1;
    for (int d = 0; d < axis; ++d) OU *= diff_dst_pd.data.dims[d];
    const int C = diff_dst_pd.data.dims[axis];
    memory::dim IN = 1;
    for (int d = axis + 1; d < ndims; ++d) IN *= diff_dst_pd.data.dims[d];

    mkldnn::impl::parallel_nd(OU, IN, [&](memory::dim ou, memory::dim in) {
        if (is_current_test_failed())
            return;

        const memory::dim idx_start = ou * C * IN + in;

        float sbr = 0.0;
        for (memory::dim c = 0; c < C; ++c) {
            auto off_d = dst_mdw.off_l(idx_start + c * IN);
            auto off_dd = diff_dst_mdw.off_l(idx_start + c * IN);
            sbr += dst_ptr[off_d] * diff_dst_ptr[off_dd];
        }

        for (memory::dim c = 0; c < C; ++c) {
            auto off_d = dst_mdw.off_l(idx_start + c * IN);
            auto off_dd = diff_dst_mdw.off_l(idx_start + c * IN);
            data_t diff_src_ref = dst_ptr[off_d] * (diff_dst_ptr[off_dd] - sbr);
            ASSERT_NEAR(diff_src_ptr[off_dd], diff_src_ref, eps);
        }
    });
}

template <typename data_t>
struct softmax_test_params {
    memory::format_tag data_memory_format;
    memory::format_tag diff_memory_format;
    memory::dims dims;
    int axis;
    bool expect_to_fail;
    mkldnn_status_t expected_status;
};

template <typename data_t>
class softmax_test : public ::testing::TestWithParam<softmax_test_params<data_t>> {
    softmax_test_params<data_t> p;
protected:
    virtual void SetUp() {
        p = ::testing::TestWithParam<softmax_test_params<data_t>>::GetParam();
        catch_expected_failures([=](){Test();}, p.expect_to_fail,
                    p.expected_status);
    }

    void Test() {
        auto eng = engine(get_test_engine_kind(), 0);
        auto strm = stream(eng);

        memory::data_type prec = data_traits<data_t>::data_type;

        auto data_mem_desc = memory::desc(p.dims, prec, p.data_memory_format);
        auto diff_mem_desc = memory::desc(p.dims, prec, p.data_memory_format);

        auto src = memory(data_mem_desc, eng);
        auto dst = memory(data_mem_desc, eng);

        auto diff_src = memory(diff_mem_desc, eng);
        auto diff_dst = memory(diff_mem_desc, eng);

        // Create softmax backward descriptor
        // before forward so its exceptions can be tested
        auto softmax_desc
            = softmax_backward::desc(diff_mem_desc, data_mem_desc, p.axis);

        // Create softmax forward (hint for backward)
        auto softmax_fwd_desc = softmax_forward::desc(prop_kind::forward_scoring,
                data_mem_desc, p.axis);
        auto softmax_fwd_pdesc = softmax_forward::primitive_desc(softmax_fwd_desc,
                eng);

        auto softmax = softmax_forward(softmax_fwd_pdesc);

        auto softmax_prim_desc
            = softmax_backward::primitive_desc(softmax_desc, eng, softmax_fwd_pdesc);
        auto softmax_bwd = softmax_backward(softmax_prim_desc);

        auto test_with_given_fill = [&](data_t mean, data_t var) {
            // Fill the softmax forward input
            fill_data<data_t>(data_mem_desc.get_size() / sizeof(data_t),
                    src, mean, var);
            check_zero_tail<data_t>(1, src);

            // Fill the softmax backward diffs
            // eg. data diff that comes from upper primitive/layer
            fill_data<data_t>(diff_mem_desc.get_size() / sizeof(data_t),
                    diff_dst, data_t(0), data_t(1));
            check_zero_tail<data_t>(1, diff_dst);

            softmax.execute(strm, {{MKLDNN_ARG_SRC, src}, {MKLDNN_ARG_DST, dst}});
            softmax_bwd.execute(strm, {
                    {MKLDNN_ARG_DST, dst},
                    {MKLDNN_ARG_DIFF_DST, diff_dst},
                    {MKLDNN_ARG_DIFF_SRC, diff_src}});
            strm.wait();

            check_softmax_bwd<data_t>(dst, diff_dst, diff_src, p.axis);
            check_zero_tail<data_t>(0, diff_src);
        };

        test_with_given_fill(-200, 1);
        test_with_given_fill(   0, 1);
        test_with_given_fill( 200, 1);
    }
};

using softmax_backward_test_float = softmax_test<float>;
using softmax_bwd_test_params_float = softmax_test_params<float>;

TEST_P(softmax_backward_test_float, TestsSoftmax) { }
INSTANTIATE_TEST_SUITE_P(TestSoftmaxBackward, softmax_backward_test_float,
        ::testing::Values(
            softmax_bwd_test_params_float{ memory::format_tag::nchw, memory::format_tag::nchw, {2, -2, 128, 256}, 0, true, mkldnn_invalid_arguments},
            softmax_bwd_test_params_float{ memory::format_tag::nchw, memory::format_tag::nchw, {2, 19, 128, 256}, 5, true, mkldnn_invalid_arguments},
            softmax_bwd_test_params_float{ memory::format_tag::nchw, memory::format_tag::nchw, {2, 0, 5, 5}, 0},
            softmax_bwd_test_params_float{ memory::format_tag::nchw, memory::format_tag::nchw, {2, 0, 5, 5}, 1},
            softmax_bwd_test_params_float{ memory::format_tag::nchw, memory::format_tag::nchw, {2, 19, 128, 256}, 0},
            softmax_bwd_test_params_float{ memory::format_tag::nchw, memory::format_tag::nchw, {2, 19, 128, 256}, 2},
            softmax_bwd_test_params_float{ memory::format_tag::nchw, memory::format_tag::nchw, {2, 19, 128, 256}, 3},
            softmax_bwd_test_params_float{ memory::format_tag::ncw, memory::format_tag::ncw, {2, 19, 128}, 0},
            softmax_bwd_test_params_float{ memory::format_tag::ncw, memory::format_tag::ncw, {2, 19, 128}, 1},
            softmax_bwd_test_params_float{ memory::format_tag::ncw, memory::format_tag::ncw, {2, 19, 128}, 2},
            softmax_bwd_test_params_float{ memory::format_tag::nc, memory::format_tag::nc, {16, 300}, 0},
            softmax_bwd_test_params_float{ memory::format_tag::nc, memory::format_tag::nc, {16, 30000}, 1},
            softmax_bwd_test_params_float{ memory::format_tag::nc, memory::format_tag::nc, {2, 1000}, 1},
            softmax_bwd_test_params_float{ memory::format_tag::nChw8c, memory::format_tag::nChw8c, {64, 1011, 1, 1}, 1},
            softmax_bwd_test_params_float{ memory::format_tag::nChw8c, memory::format_tag::nChw8c, {2, 1011, 32, 1}, 2}
));
}
