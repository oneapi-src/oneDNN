/*******************************************************************************
* Copyright 2022-2024 Intel Corporation
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

#include "oneapi/dnnl/dnnl.hpp"

namespace dnnl {

using tag = memory::format_tag;
using dt = memory::data_type;

struct shuffle_test_params_t {
    dt src_dt;
    dt dst_dt;
    tag src_tag;
    tag dst_tag;
    memory::dims dims;
    int axis;
    memory::dim group_size;
    bool expect_to_fail;
    dnnl_status_t expected_status;
};

class shuffle_test_t : public ::testing::TestWithParam<shuffle_test_params_t> {
private:
    shuffle_test_params_t p;
    std::shared_ptr<shuffle_forward::primitive_desc> pd_fwd_hint;

protected:
    void SetUp() override {
        p = ::testing::TestWithParam<shuffle_test_params_t>::GetParam();

        SKIP_IF_CUDA(true, "Shuffle primitive not supported by CUDA");
        SKIP_IF_HIP(true, "Shuffle primitive not supported by HIP");

        SKIP_IF(unsupported_data_type(p.src_dt, p.dst_dt),
                "Engine does not support this data type.");

        catch_expected_failures(
                [&]() { Test(); }, p.expect_to_fail, p.expected_status);
    }

    void Forward(prop_kind pk) {
        // shuffle specific types and values
        using pd_t = shuffle_forward::primitive_desc;

        auto eng = get_test_engine();
        auto strm = make_stream(eng);

        auto aa = allows_attr_t {false};

        auto src_md = memory::desc(p.dims, p.src_dt, p.src_tag);
        auto dst_md = memory::desc(p.dims, p.dst_dt, p.dst_tag);

        // default pd ctor
        auto pd = pd_t();
        // regular pd ctor
        pd = pd_t(eng, pk, src_md, dst_md, p.axis, p.group_size);
        // test all pd ctors
        test_fwd_pd_constructors<pd_t>(
                pd, aa, pk, src_md, dst_md, p.axis, p.group_size);
        pd_fwd_hint = std::make_shared<pd_t>(pd);

        EXPECT_ANY_THROW(shuffle_forward(pd, {}));
        // default primitive ctor
        auto shuffle = shuffle_forward();
        // regular primitive ctor
        shuffle = shuffle_forward(pd);

        // check primitive kind is shuffle
        ASSERT_TRUE(shuffle.get_kind() == primitive::kind::shuffle);
        // query for descs from pd
        const auto src_desc = pd.src_desc();
        const auto dst_desc = pd.dst_desc();
        // query for src_desc via exec arg
        ASSERT_TRUE(pd.query_md(query::exec_arg_md, DNNL_ARG_SRC) == src_desc);
        if (p.src_tag != tag::any) { ASSERT_TRUE(src_md == src_desc); }
        // query for dst_desc via exec arg
        ASSERT_TRUE(pd.query_md(query::exec_arg_md, DNNL_ARG_DST) == dst_desc);
        if (p.dst_tag != tag::any) { ASSERT_TRUE(dst_md == dst_desc); }

        // query primitive parameters
        ASSERT_EQ(pd.get_prop_kind(), pk);
        ASSERT_EQ(pd.get_axis(), p.axis);
        ASSERT_EQ(pd.get_group_size(), p.group_size);

        // check primitive returns zero_md for all rest md
        ASSERT_TRUE(pd.weights_desc().is_zero());
        ASSERT_TRUE(pd.diff_src_desc().is_zero());
        ASSERT_TRUE(pd.diff_dst_desc().is_zero());
        ASSERT_TRUE(pd.diff_weights_desc().is_zero());

        auto src = test::make_memory(src_desc, eng);
        auto dst = test::make_memory(dst_desc, eng);

        fill_data(p.src_dt, src, 1, 1);
        // test out-place mode
        shuffle.execute(strm, {{DNNL_ARG_SRC, src}, {DNNL_ARG_DST, dst}});
        strm.wait();
    }

    void Backward() {
        // shuffle specific types and values
        using pd_t = shuffle_backward::primitive_desc;
        using hint_pd_t = shuffle_forward::primitive_desc;
        allows_attr_t aa {false}; // doesn't support anything

        auto eng = get_test_engine();
        auto strm = make_stream(eng);

        auto diff_src_md = memory::desc(p.dims, p.src_dt, p.src_tag);
        auto diff_dst_md = memory::desc(p.dims, p.dst_dt, p.dst_tag);

        // default pd ctor
        auto pd = pd_t();
        // regular pd ctor
        pd = pd_t(eng, diff_src_md, diff_dst_md, p.axis, p.group_size,
                *pd_fwd_hint);
        // test all pd ctors
        test_bwd_pd_constructors<pd_t, hint_pd_t>(pd, *pd_fwd_hint, aa,
                diff_src_md, diff_dst_md, p.axis, p.group_size);

        EXPECT_ANY_THROW(shuffle_backward(pd, {}));
        // default primitive ctor
        auto shuffle = shuffle_backward();
        // regular primitive ctor
        shuffle = shuffle_backward(pd);

        // check primitive kind is shuffle
        ASSERT_TRUE(shuffle.get_kind() == primitive::kind::shuffle);

        // query for descs from pd
        const auto diff_src_desc = pd.diff_src_desc();
        const auto diff_dst_desc = pd.diff_dst_desc();
        // query for diff_src_desc via exec arg
        ASSERT_TRUE(pd.query_md(query::exec_arg_md, DNNL_ARG_DIFF_SRC)
                == diff_src_desc);
        if (p.src_tag != tag::any) {
            ASSERT_TRUE(diff_src_md == diff_src_desc);
        }
        // query for diff_dst_desc via exec arg
        ASSERT_TRUE(pd.query_md(query::exec_arg_md, DNNL_ARG_DIFF_DST)
                == diff_dst_desc);
        if (p.dst_tag != tag::any) {
            ASSERT_TRUE(diff_dst_md == diff_dst_desc);
        }
        // query primitive parameters
        ASSERT_EQ(pd.get_prop_kind(), prop_kind::backward_data);
        ASSERT_EQ(pd.get_axis(), p.axis);
        ASSERT_EQ(pd.get_group_size(), p.group_size);

        // check primitive returns zero_md for all rest md
        ASSERT_TRUE(pd.src_desc().is_zero());
        ASSERT_TRUE(pd.weights_desc().is_zero());
        ASSERT_TRUE(pd.dst_desc().is_zero());
        ASSERT_TRUE(pd.diff_weights_desc().is_zero());

        auto diff_src = test::make_memory(diff_src_desc, eng);
        auto diff_dst = test::make_memory(diff_dst_desc, eng);

        fill_data(p.dst_dt, diff_dst, 2, 2);

        // test out-place mode
        shuffle.execute(strm,
                {{DNNL_ARG_DIFF_DST, diff_dst}, {DNNL_ARG_DIFF_SRC, diff_src}});
        strm.wait();
    }

    void Test() {
        const bool is_int8 = p.src_dt == dt::s8 || p.src_dt == dt::u8;
        std::vector<prop_kind> pks = {is_int8 ? prop_kind::forward_inference
                                              : prop_kind::forward_training};

        for (auto pk : pks) {
            Forward(pk);

            bool to_continue = pk != prop_kind::forward_training;
            if (to_continue) continue;

            Backward();
        }
    }

    bool is_fwd(prop_kind pk) const {
        return pk == prop_kind::forward_training
                || pk == prop_kind::forward_inference;
    }
};

using tp = shuffle_test_params_t;

TEST_P(shuffle_test_t, TestsShuffle) {}

INSTANTIATE_TEST_SUITE_P(Test_Shuffle_EF, shuffle_test_t,
        ::testing::Values(
                // Negative dims
                tp {dt::f32, dt::f32, tag::nchw, tag::nchw, {2, -4, 128, 256},
                        1, 4, true, dnnl_invalid_arguments},
                // Non-divisible group_size
                tp {dt::f32, dt::f32, tag::nchw, tag::nchw, {2, 4, 128, 256}, 1,
                        3, true, dnnl_invalid_arguments},
                // Axis exceeds ndims
                tp {dt::f32, dt::f32, tag::nchw, tag::nchw, {2, 4, 128, 256}, 4,
                        4, true, dnnl_invalid_arguments},
                // Tag for src on forward is not specified
                tp {dt::f32, dt::f32, tag::any, tag::nchw, {2, 4, 128, 256}, 1,
                        4, true, dnnl_invalid_arguments},
                // Data type for src is not specified
                tp {dt::undef, dt::f32, tag::nchw, tag::nchw, {2, 4, 128, 256},
                        1, 4, true, dnnl_invalid_arguments},
                // Different data types are not supported
                tp {dt::f32, dt::bf16, tag::nchw, tag::nchw, {2, 4, 128, 256},
                        1, 4, true, dnnl_unimplemented},
                // Different memory formats are not supported
                tp {dt::f32, dt::f32, tag::nchw, tag::nhwc, {2, 4, 128, 256}, 1,
                        4, true, dnnl_unimplemented}));

static auto all_cases = [](dt src_dt, dt dst_dt) {
    return ::testing::Values(
            tp {src_dt, dst_dt, tag::nwc, tag::nwc, {2, 16, 10}, 1, 4},
            tp {src_dt, dst_dt, tag::ncw, tag::ncw, {2, 64, 27}, 1, 4},
            tp {src_dt, dst_dt, tag::nhwc, tag::nhwc, {2, 15, 10, 8}, 1, 3},
            tp {src_dt, dst_dt, tag::nchw, tag::nchw, {2, 64, 27, 27}, 0, 2},
            tp {src_dt, dst_dt, tag::nChw8c, tag::nChw8c, {2, 16, 16, 8}, 1, 4},
            tp {src_dt, dst_dt, tag::nChw16c, tag::nChw16c, {2, 16, 4, 4}, 1,
                    4},
            tp {src_dt, dst_dt, tag::ncdhw, tag::ncdhw, {2, 64, 7, 7, 7}, 2, 7},
            tp {src_dt, dst_dt, tag::ncdhw, tag::ncdhw, {10, 10, 10, 10, 10}, 0,
                    5},
            tp {src_dt, dst_dt, tag::nCdhw16c, tag::nCdhw16c, {4, 16, 2, 2, 2},
                    1, 4});
};

#define EXPAND_DTS(src, dst) memory::data_type::src, memory::data_type::dst

#define INST_TEST_CASE(name, suite, ...) \
    INSTANTIATE_TEST_SUITE_P(name, shuffle_test_t, suite(__VA_ARGS__));

#define CPU_INST_TEST_CASE(name, suite, ...) \
    CPU_INSTANTIATE_TEST_SUITE_P(name, shuffle_test_t, suite(__VA_ARGS__));

#define GPU_INST_TEST_CASE(name, suite, ...) \
    GPU_INSTANTIATE_TEST_SUITE_P(name, shuffle_test_t, suite(__VA_ARGS__));

INST_TEST_CASE(ShuffleSimpleF32, all_cases, EXPAND_DTS(f32, f32));
INST_TEST_CASE(ShuffleSimpleBF16, all_cases, EXPAND_DTS(bf16, bf16));
INST_TEST_CASE(ShuffleSimpleF16, all_cases, EXPAND_DTS(f16, f16));
INST_TEST_CASE(ShuffleSimpleU8, all_cases, EXPAND_DTS(u8, u8));
INST_TEST_CASE(ShuffleSimpleS8, all_cases, EXPAND_DTS(s8, s8));

} // namespace dnnl
