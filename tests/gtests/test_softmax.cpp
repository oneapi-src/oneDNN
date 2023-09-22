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

#include "dnnl_test_common.hpp"
#include "gtest/gtest.h"

#include "oneapi/dnnl/dnnl.hpp"

namespace dnnl {

using tag = memory::format_tag;
using dt = memory::data_type;

struct softmax_test_params_t {
    prop_kind aprop_kind;
    algorithm aalgorithm;
    dt src_dt; // diff_src_dt
    dt dst_dt;
    dt diff_dst_dt;
    tag src_tag; // diff_src_tag
    tag dst_tag;
    tag diff_dst_tag;
    memory::dims dims;
    int axis;
    bool expect_to_fail;
    dnnl_status_t expected_status;
};

class softmax_test_t : public ::testing::TestWithParam<softmax_test_params_t> {
private:
    softmax_test_params_t p;
    memory dst, workspace;
    std::shared_ptr<softmax_forward::primitive_desc> pd_fwd_hint;

protected:
    void SetUp() override {
        p = ::testing::TestWithParam<softmax_test_params_t>::GetParam();

        SKIP_IF_CUDA(
                !cuda_check_format_tag(p.src_tag), "Unsupported format tag");
        SKIP_IF_CUDA(
                !cuda_check_format_tag(p.dst_tag), "Unsupported format tag");

        SKIP_IF_HIP(!hip_check_format_tag(p.src_tag), "Unsupported format tag");
        SKIP_IF_HIP(!hip_check_format_tag(p.dst_tag), "Unsupported format tag");

        if (!is_fwd(p.aprop_kind)) {
            SKIP_IF_CUDA(!cuda_check_format_tag(p.diff_dst_tag),
                    "Unsupported format tag");

            SKIP_IF_HIP(!hip_check_format_tag(p.diff_dst_tag),
                    "Unsupported format tag");
        }
        SKIP_IF_CUDA((p.src_dt == dt::bf16 || p.dst_dt == dt::bf16),
                "Unsupported datatype for CUDA");
        SKIP_IF_HIP((p.src_dt == dt::bf16 || p.dst_dt == dt::bf16),
                "Unsupported datatype for HIP");
        if (!is_fwd(p.aprop_kind)) {
            SKIP_IF_CUDA((p.diff_dst_dt == dt::bf16),
                    "Unsupported datatype for CUDA");
            SKIP_IF_HIP((p.diff_dst_dt == dt::bf16),
                    "Unsupported datatype for HIP");
        }

        SKIP_IF(unsupported_data_type(p.src_dt)
                        || unsupported_data_type(p.dst_dt),
                "Engine does not support this data type.");
        if (!is_fwd(p.aprop_kind)) {
            SKIP_IF(unsupported_data_type(p.diff_dst_dt),
                    "Engine does not support this data type.");
        }

        SKIP_IF_CUDA(p.src_dt != p.dst_dt && p.src_dt != dt::undef
                        && p.dst_dt != dt::undef,
                "Unsupported different data types for source and "
                "destination");
        SKIP_IF_HIP(p.src_dt != p.dst_dt && p.src_dt != dt::undef
                        && p.dst_dt != dt::undef,
                "Unsupported different data types for source and "
                "destination");
        SKIP_IF_CUDA(!is_fwd(p.aprop_kind) && p.src_dt != p.diff_dst_dt
                        && p.src_dt != dt::undef && p.diff_dst_dt != dt::undef,
                "Unsupported different data types for diff_source and "
                "diff_destination");
        SKIP_IF_HIP(!is_fwd(p.aprop_kind) && p.src_dt != p.diff_dst_dt
                        && p.src_dt != dt::undef && p.diff_dst_dt != dt::undef,
                "Unsupported different data types for diff_source and "
                "diff_destination");

        SKIP_IF_CUDA(p.src_tag != p.dst_tag && p.src_tag != tag::any
                        && p.dst_tag != tag::any,
                "Unsupported different memory formats for source and "
                "destination");
        SKIP_IF_HIP(p.src_tag != p.dst_tag && p.src_tag != tag::any
                        && p.dst_tag != tag::any,
                "Unsupported different memory formats for source and "
                "destination");
        SKIP_IF_CUDA(!is_fwd(p.aprop_kind) && p.src_tag != p.diff_dst_tag
                        && p.src_tag != tag::any && p.diff_dst_tag != tag::any,
                "Unsupported different memory formats for diff_source and "
                "diff_destination");
        SKIP_IF_HIP(!is_fwd(p.aprop_kind) && p.src_tag != p.diff_dst_tag
                        && p.src_tag != tag::any && p.diff_dst_tag != tag::any,
                "Unsupported different memory formats for diff_source and "
                "diff_destination");
        SKIP_IF_CUDA(p.dst_dt == dt::u8 || p.dst_dt == dt::s8,
                "Unsupported int8 destination data type");
        SKIP_IF_HIP(p.dst_dt == dt::u8 || p.dst_dt == dt::s8,
                "Unsupported int8 destination data type");

        SKIP_IF_HIP(p.axis != 1, "Unsupported axis for HIP");

        catch_expected_failures(
                [&]() { Test(); }, p.expect_to_fail, p.expected_status);
    }
    bool cuda_check_format_tag(memory::format_tag tag) {
        return (tag != memory::format_tag::aBcd8b
                && tag != memory::format_tag::aBcd16b);
    }
    bool hip_check_format_tag(memory::format_tag tag) {
        return (tag == memory::format_tag::a || tag == memory::format_tag::ab
                || tag == memory::format_tag::abc
                || tag == memory::format_tag::abcd
                || tag == memory::format_tag::abcde);
    }
    void Forward() {
        // softmax specific types and values
        using pd_t = softmax_forward::primitive_desc;

        auto eng = get_test_engine();
        auto strm = make_stream(eng);
        prop_kind pk = !is_fwd(p.aprop_kind) ? prop_kind::forward_training
                                             : p.aprop_kind;

        allows_attr_t aa {false};
        if (!(is_nvidia_gpu(eng) || is_amd_gpu(eng))) {
            aa.po_eltwise = true;
            aa.po_binary = true;
        }

        bool is_int8 = impl::utils::one_of(p.src_dt, memory::data_type::s8,
                               memory::data_type::u8)
                || impl::utils::one_of(
                        p.dst_dt, memory::data_type::s8, memory::data_type::u8);

        if (is_int8 && !(is_nvidia_gpu(eng) || is_amd_gpu(eng)))
            aa.scales = true;

        // To validate backward on valid tag::any settings reuse dst tag.
        const bool src_bwd_any = !is_fwd(p.aprop_kind) && p.src_tag == tag::any;
        auto src_tag = src_bwd_any ? p.dst_tag : p.src_tag;

        auto src_md = memory::desc(p.dims, p.src_dt, src_tag);
        auto dst_md = memory::desc(p.dims, p.dst_dt, p.dst_tag);

        // default pd ctor
        auto pd = pd_t();
        // regular pd ctor
        pd = pd_t(eng, pk, p.aalgorithm, src_md, dst_md, p.axis);
        // test all pd ctors
        test_fwd_pd_constructors<pd_t>(
                pd, aa, pk, p.aalgorithm, src_md, dst_md, p.axis);
        pd_fwd_hint = std::make_shared<pd_t>(pd);

        EXPECT_ANY_THROW(softmax_forward(pd, {}));
        // default primitive ctor
        auto softmax = softmax_forward();
        // regular primitive ctor
        softmax = softmax_forward(pd);

        // check primitive kind is softmax
        ASSERT_TRUE(softmax.get_kind() == primitive::kind::softmax);
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
        ASSERT_EQ(pd.get_algorithm(), p.aalgorithm);

        // query for workspace
        const auto workspace_desc = pd.workspace_desc();

        // check primitive returns zero_md for all rest md
        ASSERT_TRUE(pd.weights_desc().is_zero());
        ASSERT_TRUE(pd.diff_src_desc().is_zero());
        ASSERT_TRUE(pd.diff_dst_desc().is_zero());
        ASSERT_TRUE(pd.diff_weights_desc().is_zero());

        auto src = test::make_memory(src_desc, eng);
        dst = test::make_memory(dst_desc, eng);
        workspace = test::make_memory(workspace_desc, eng);

        fill_data(p.src_dt, src, 1, 1);
        // test out-place mode
        softmax.execute(strm,
                {{DNNL_ARG_SRC, src}, {DNNL_ARG_DST, dst},
                        {DNNL_ARG_WORKSPACE, workspace}});
        strm.wait();

        // test in-place mode on forward
        if (p.aprop_kind != prop_kind::backward_data && p.src_tag == p.dst_tag
                && p.src_dt == p.dst_dt) {
            softmax.execute(strm,
                    {{DNNL_ARG_SRC, src}, {DNNL_ARG_DST, src},
                            {DNNL_ARG_WORKSPACE, workspace}});
            strm.wait();
        }
    }

    void Backward() {
        // softmax specific types and values
        using pd_t = softmax_backward::primitive_desc;
        using hint_pd_t = softmax_forward::primitive_desc;
        allows_attr_t aa {false}; // doesn't support anything

        auto eng = get_test_engine();
        auto strm = make_stream(eng);
        auto diff_src_md = memory::desc(p.dims, p.src_dt, p.src_tag);
        auto diff_dst_md = memory::desc(p.dims, p.diff_dst_dt, p.diff_dst_tag);
        auto dst_md = memory::desc(p.dims, p.dst_dt, p.dst_tag);

        // default pd ctor
        auto pd = pd_t();
        // regular pd ctor
        pd = pd_t(eng, p.aalgorithm, diff_src_md, diff_dst_md, dst_md, p.axis,
                *pd_fwd_hint);
        // test all pd ctors
        test_bwd_pd_constructors<pd_t, hint_pd_t>(pd, *pd_fwd_hint, aa,
                p.aalgorithm, diff_src_md, diff_dst_md, dst_md, p.axis);

        EXPECT_ANY_THROW(softmax_backward(pd, {}));
        // default primitive ctor
        auto softmax = softmax_backward();
        // regular primitive ctor
        softmax = softmax_backward(pd);

        // check primitive kind is softmax
        ASSERT_TRUE(softmax.get_kind() == primitive::kind::softmax);

        // query for descs from pd
        const auto diff_src_desc = pd.diff_src_desc();
        const auto diff_dst_desc = pd.diff_dst_desc();
        const auto dst_desc = pd.dst_desc();
        // query for diff_src_desc via exec arg
        ASSERT_TRUE(pd.query_md(query::exec_arg_md, DNNL_ARG_DIFF_SRC)
                == diff_src_desc);
        if (p.src_tag != tag::any) {
            ASSERT_TRUE(diff_src_md == diff_src_desc);
        }
        // query for diff_dst_desc via exec arg
        ASSERT_TRUE(pd.query_md(query::exec_arg_md, DNNL_ARG_DIFF_DST)
                == diff_dst_desc);
        if (p.diff_dst_tag != tag::any) {
            ASSERT_TRUE(diff_dst_md == diff_dst_desc);
        }
        // query for dst_desc via exec arg
        ASSERT_TRUE(pd.query_md(query::exec_arg_md, DNNL_ARG_DST) == dst_desc);
        if (p.dst_tag != tag::any) { ASSERT_TRUE(dst_md == dst_desc); }

        // query primitive parameters
        ASSERT_EQ(pd.get_prop_kind(), prop_kind::backward_data);
        ASSERT_EQ(pd.get_axis(), p.axis);
        ASSERT_EQ(pd.get_algorithm(), p.aalgorithm);

        // check primitive returns zero_md for all rest md
        ASSERT_TRUE(pd.src_desc().is_zero());
        ASSERT_TRUE(pd.weights_desc().is_zero());
        ASSERT_TRUE(pd.diff_weights_desc().is_zero());

        auto diff_src = test::make_memory(diff_src_desc, eng);
        auto diff_dst = test::make_memory(diff_dst_desc, eng);

        fill_data(p.diff_dst_dt, diff_dst, 2, 2);

        // test out-place mode
        softmax.execute(strm,
                {{DNNL_ARG_DST, dst}, {DNNL_ARG_DIFF_DST, diff_dst},
                        {DNNL_ARG_DIFF_SRC, diff_src},
                        {DNNL_ARG_WORKSPACE, workspace}});
        strm.wait();

        // test in-place mode
        if (p.src_tag == p.diff_dst_tag && p.src_dt == p.diff_dst_dt) {
            softmax.execute(strm,
                    {{DNNL_ARG_DST, dst}, {DNNL_ARG_DIFF_DST, diff_dst},
                            {DNNL_ARG_DIFF_SRC, diff_dst},
                            {DNNL_ARG_WORKSPACE, workspace}});
            strm.wait();
        }
    }

    void Test() {
        Forward();
        if (!is_fwd(p.aprop_kind)) Backward();
    }

    bool is_fwd(prop_kind pk) const {
        return pk == prop_kind::forward_training
                || pk == prop_kind::forward_inference;
    }
};

using tp = softmax_test_params_t;

static const auto training = prop_kind::forward_training;
static const auto inference = prop_kind::forward_inference;
static const auto backward = prop_kind::backward_data;
static const auto alg_softmax = algorithm::softmax_accurate;
static const auto alg_logsoftmax = algorithm::softmax_log;

TEST_P(softmax_test_t, TestsSoftmax) {}

INSTANTIATE_TEST_SUITE_P(Test_Softmax_EF, softmax_test_t,
        ::testing::Values(
                // Negative dims
                tp {training, alg_softmax, dt::f32, dt::f32, dt::undef,
                        tag::nchw, tag::nchw, tag::undef, {2, -2, 128, 256}, 0,
                        true, dnnl_invalid_arguments},
                // Axis exceeds ndims
                tp {training, alg_softmax, dt::f32, dt::f32, dt::undef,
                        tag::nchw, tag::nchw, tag::undef, {2, 2, 128, 256}, 10,
                        true, dnnl_invalid_arguments},
                // Not supported algorithm
                tp {training, algorithm::eltwise_relu, dt::f32, dt::f32,
                        dt::undef, tag::nchw, tag::nchw, tag::undef,
                        {2, 2, 128, 256}, 3, true, dnnl_invalid_arguments},
                // Tag for src on forward is not specified
                tp {training, alg_softmax, dt::f32, dt::f32, dt::undef,
                        tag::any, tag::nchw, tag::undef, {2, 2, 128, 256}, 3,
                        true, dnnl_invalid_arguments},
                // Tag for dst on backward is not specified
                tp {backward, alg_softmax, dt::f32, dt::f32, dt::f32, tag::nchw,
                        tag::any, tag::nchw, {2, 2, 128, 256}, 3, true,
                        dnnl_invalid_arguments},
                // Data type for src is not specified
                tp {training, alg_softmax, dt::undef, dt::f32, dt::undef,
                        tag::nchw, tag::nchw, tag::undef, {2, 2, 128, 256}, 3,
                        true, dnnl_invalid_arguments}));

INSTANTIATE_TEST_SUITE_P(Test_Softmax_Forward_Float, softmax_test_t,
        ::testing::Values(
                tp {training, alg_softmax, dt::f32, dt::f32, dt::undef,
                        tag::nchw, tag::nchw, tag::undef, {2, 0, 5, 5}, 1},
                tp {training, alg_softmax, dt::f32, dt::f32, dt::undef,
                        tag::nhwc, tag::nhwc, tag::undef, {2, 19, 16, 64}, 1},
                tp {training, alg_softmax, dt::f32, dt::f32, dt::undef,
                        tag::nchw, tag::any, tag::undef, {1, 8, 128, 1024}, 3},
                tp {inference, alg_softmax, dt::f32, dt::f32, dt::undef,
                        tag::nc, tag::nc, tag::undef, {2, 1000}, 0},
                tp {inference, alg_softmax, dt::f32, dt::f32, dt::undef,
                        tag::nc, tag::cn, tag::undef, {2, 1000}, 1},
                tp {inference, alg_softmax, dt::f32, dt::f32, dt::undef,
                        tag::nc, tag::any, tag::undef, {1, 13}, 1},
                tp {inference, alg_softmax, dt::f32, dt::f32, dt::undef,
                        tag::ncw, tag::ncw, tag::undef, {16, 257, 32}, 1},
                tp {inference, alg_logsoftmax, dt::f32, dt::f32, dt::undef,
                        tag::ncw, tag::ncw, tag::undef, {16, 257, 32}, 2},
                tp {inference, alg_softmax, dt::f32, dt::f32, dt::undef,
                        tag::nChw16c, tag::nChw16c, tag::undef,
                        {64, 1011, 1, 1}, 1},
                tp {inference, alg_softmax, dt::f32, dt::f32, dt::undef,
                        tag::nChw8c, tag::nChw8c, tag::undef, {3, 1011, 1, 1},
                        1},
                tp {inference, alg_logsoftmax, dt::f32, dt::f32, dt::undef,
                        tag::nChw8c, tag::nChw8c, tag::undef, {2, 1011, 32, 1},
                        2}));

INSTANTIATE_TEST_SUITE_P(Test_Softmax_Backward_Float, softmax_test_t,
        ::testing::Values(
                tp {backward, alg_softmax, dt::f32, dt::f32, dt::f32, tag::nchw,
                        tag::nchw, tag::nchw, {2, 0, 5, 5}, 1},
                tp {backward, alg_softmax, dt::f32, dt::f32, dt::f32, tag::nhwc,
                        tag::nhwc, tag::nhwc, {2, 19, 16, 64}, 1},
                tp {backward, alg_softmax, dt::f32, dt::f32, dt::f32, tag::any,
                        tag::nchw, tag::any, {1, 8, 128, 1024}, 3},
                tp {backward, alg_softmax, dt::f32, dt::f32, dt::f32, tag::nc,
                        tag::nc, tag::nc, {2, 1000}, 0},
                tp {backward, alg_softmax, dt::f32, dt::f32, dt::f32, tag::nc,
                        tag::cn, tag::cn, {2, 1000}, 1},
                tp {backward, alg_softmax, dt::f32, dt::f32, dt::f32, tag::any,
                        tag::nc, tag::nc, {1, 13}, 1},
                tp {backward, alg_softmax, dt::f32, dt::f32, dt::f32, tag::ncw,
                        tag::ncw, tag::ncw, {16, 257, 32}, 1},
                tp {backward, alg_logsoftmax, dt::f32, dt::f32, dt::f32,
                        tag::ncw, tag::ncw, tag::nwc, {16, 257, 32}, 2},
                tp {backward, alg_softmax, dt::f32, dt::f32, dt::f32,
                        tag::nChw16c, tag::nChw16c, tag::nChw16c,
                        {64, 1011, 1, 1}, 1},
                tp {backward, alg_softmax, dt::f32, dt::f32, dt::f32,
                        tag::nChw8c, tag::nhwc, tag::nchw, {3, 1011, 1, 1}, 1},
                tp {backward, alg_logsoftmax, dt::f32, dt::f32, dt::f32,
                        tag::nChw8c, tag::nChw8c, tag::nChw8c, {2, 1011, 32, 1},
                        2}));

INSTANTIATE_TEST_SUITE_P(Test_Softmax_Forward_Bfloat16, softmax_test_t,
        ::testing::Values(
                tp {training, alg_softmax, dt::bf16, dt::bf16, dt::undef,
                        tag::nchw, tag::nchw, tag::undef, {2, 0, 5, 5}, 1},
                tp {training, alg_softmax, dt::bf16, dt::bf16, dt::undef,
                        tag::nhwc, tag::nhwc, tag::undef, {2, 19, 16, 64}, 1},
                tp {training, alg_softmax, dt::bf16, dt::bf16, dt::undef,
                        tag::nchw, tag::any, tag::undef, {1, 8, 128, 1024}, 3},
                tp {inference, alg_softmax, dt::bf16, dt::bf16, dt::undef,
                        tag::nc, tag::nc, tag::undef, {2, 1000}, 0},
                tp {inference, alg_softmax, dt::bf16, dt::bf16, dt::undef,
                        tag::nc, tag::cn, tag::undef, {2, 1000}, 1},
                tp {inference, alg_softmax, dt::bf16, dt::bf16, dt::undef,
                        tag::nc, tag::any, tag::undef, {1, 13}, 1},
                tp {inference, alg_softmax, dt::bf16, dt::bf16, dt::undef,
                        tag::ncw, tag::ncw, tag::undef, {16, 257, 32}, 1},
                tp {inference, alg_logsoftmax, dt::bf16, dt::bf16, dt::undef,
                        tag::ncw, tag::ncw, tag::undef, {16, 257, 32}, 2},
                tp {inference, alg_softmax, dt::bf16, dt::bf16, dt::undef,
                        tag::nChw16c, tag::nChw16c, tag::undef,
                        {64, 1011, 1, 1}, 1},
                tp {inference, alg_softmax, dt::bf16, dt::bf16, dt::undef,
                        tag::nChw8c, tag::nChw8c, tag::undef, {3, 1011, 1, 1},
                        1},
                tp {inference, alg_logsoftmax, dt::bf16, dt::bf16, dt::undef,
                        tag::nChw8c, tag::nChw8c, tag::undef, {2, 1011, 32, 1},
                        2}));

INSTANTIATE_TEST_SUITE_P(Test_Softmax_Backward_Bfloat16, softmax_test_t,
        ::testing::Values(
                tp {backward, alg_softmax, dt::bf16, dt::bf16, dt::bf16,
                        tag::nchw, tag::nchw, tag::nchw, {2, 0, 5, 5}, 1},
                tp {backward, alg_softmax, dt::bf16, dt::bf16, dt::bf16,
                        tag::nhwc, tag::nhwc, tag::nhwc, {2, 19, 16, 64}, 1},
                tp {backward, alg_softmax, dt::bf16, dt::bf16, dt::bf16,
                        tag::any, tag::nchw, tag::any, {1, 8, 128, 1024}, 3},
                tp {backward, alg_softmax, dt::bf16, dt::bf16, dt::bf16,
                        tag::nc, tag::nc, tag::nc, {2, 1000}, 0},
                tp {backward, alg_softmax, dt::bf16, dt::bf16, dt::bf16,
                        tag::nc, tag::cn, tag::cn, {2, 1000}, 1},
                tp {backward, alg_softmax, dt::bf16, dt::bf16, dt::bf16,
                        tag::any, tag::nc, tag::nc, {1, 13}, 1},
                tp {backward, alg_softmax, dt::bf16, dt::bf16, dt::bf16,
                        tag::ncw, tag::ncw, tag::ncw, {16, 257, 32}, 1},
                tp {backward, alg_logsoftmax, dt::bf16, dt::bf16, dt::bf16,
                        tag::ncw, tag::ncw, tag::nwc, {16, 257, 32}, 2},
                tp {backward, alg_softmax, dt::bf16, dt::bf16, dt::bf16,
                        tag::nChw16c, tag::nChw16c, tag::nChw16c,
                        {64, 1011, 1, 1}, 1},
                tp {backward, alg_softmax, dt::bf16, dt::bf16, dt::bf16,
                        tag::nChw8c, tag::nhwc, tag::nchw, {3, 1011, 1, 1}, 1},
                tp {backward, alg_logsoftmax, dt::bf16, dt::bf16, dt::bf16,
                        tag::nChw8c, tag::nChw8c, tag::nChw8c, {2, 1011, 32, 1},
                        2}));

GPU_INSTANTIATE_TEST_SUITE_P(Test_Softmax_Forward_Half, softmax_test_t,
        ::testing::Values(
                tp {training, alg_softmax, dt::f16, dt::f16, dt::undef,
                        tag::nchw, tag::nchw, tag::undef, {2, 0, 5, 5}, 1},
                tp {training, alg_softmax, dt::f16, dt::f16, dt::undef,
                        tag::nhwc, tag::nhwc, tag::undef, {2, 19, 16, 64}, 1},
                tp {training, alg_softmax, dt::f16, dt::f16, dt::undef,
                        tag::nchw, tag::any, tag::undef, {1, 8, 128, 1024}, 3},
                tp {inference, alg_softmax, dt::f16, dt::f16, dt::undef,
                        tag::nc, tag::nc, tag::undef, {2, 1000}, 0},
                tp {inference, alg_softmax, dt::f16, dt::f16, dt::undef,
                        tag::nc, tag::cn, tag::undef, {2, 1000}, 1},
                tp {inference, alg_softmax, dt::f16, dt::f16, dt::undef,
                        tag::nc, tag::any, tag::undef, {1, 13}, 1},
                tp {inference, alg_softmax, dt::f16, dt::f16, dt::undef,
                        tag::ncw, tag::ncw, tag::undef, {16, 257, 32}, 1},
                tp {inference, alg_logsoftmax, dt::f16, dt::f16, dt::undef,
                        tag::ncw, tag::ncw, tag::undef, {16, 257, 32}, 2},
                tp {inference, alg_softmax, dt::f16, dt::f16, dt::undef,
                        tag::nChw16c, tag::nChw16c, tag::undef,
                        {64, 1011, 1, 1}, 1},
                tp {inference, alg_softmax, dt::f16, dt::f16, dt::undef,
                        tag::nChw8c, tag::nChw8c, tag::undef, {3, 1011, 1, 1},
                        1},
                tp {inference, alg_logsoftmax, dt::f16, dt::f16, dt::undef,
                        tag::nChw8c, tag::nChw8c, tag::undef, {2, 1011, 32, 1},
                        2}));

INSTANTIATE_TEST_SUITE_P(Test_Softmax_Forward_U8, softmax_test_t,
        ::testing::Values(
                tp {training, alg_softmax, dt::f32, dt::u8, dt::undef,
                        tag::nhwc, tag::nhwc, tag::undef, {2, 0, 5, 5}, 1},
                tp {training, alg_softmax, dt::f32, dt::u8, dt::undef,
                        tag::nhwc, tag::nhwc, tag::undef, {2, 19, 16, 64}, 1},
                tp {training, alg_softmax, dt::f32, dt::u8, dt::undef,
                        tag::nhwc, tag::any, tag::undef, {1, 8, 128, 1024}, 3},
                tp {inference, alg_softmax, dt::f32, dt::u8, dt::undef, tag::nc,
                        tag::nc, tag::undef, {2, 1000}, 0},
                tp {inference, alg_softmax, dt::f32, dt::u8, dt::undef, tag::nc,
                        tag::cn, tag::undef, {2, 1000}, 1},
                tp {inference, alg_softmax, dt::f32, dt::u8, dt::undef, tag::nc,
                        tag::any, tag::undef, {1, 13}, 1},
                tp {inference, alg_softmax, dt::f32, dt::u8, dt::undef,
                        tag::ncw, tag::ncw, tag::undef, {16, 257, 32}, 1},
                tp {inference, alg_logsoftmax, dt::f32, dt::u8, dt::undef,
                        tag::ncw, tag::ncw, tag::undef, {16, 257, 32}, 2},
                tp {inference, alg_softmax, dt::f32, dt::u8, dt::undef,
                        tag::nhwc, tag::nhwc, tag::undef, {64, 1011, 1, 1}, 1},
                tp {inference, alg_softmax, dt::f32, dt::u8, dt::undef,
                        tag::nhwc, tag::nhwc, tag::undef, {3, 1011, 1, 1}, 1},
                tp {inference, alg_logsoftmax, dt::f32, dt::u8, dt::undef,
                        tag::nhwc, tag::nhwc, tag::undef, {2, 1011, 32, 1},
                        2}));

INSTANTIATE_TEST_SUITE_P(Test_Softmax_Forward_S8, softmax_test_t,
        ::testing::Values(
                tp {training, alg_softmax, dt::f32, dt::s8, dt::undef,
                        tag::nhwc, tag::nhwc, tag::undef, {2, 0, 5, 5}, 1},
                tp {training, alg_softmax, dt::f32, dt::s8, dt::undef,
                        tag::nhwc, tag::nhwc, tag::undef, {2, 19, 16, 64}, 1},
                tp {training, alg_softmax, dt::f32, dt::s8, dt::undef,
                        tag::nhwc, tag::any, tag::undef, {1, 8, 128, 1024}, 3},
                tp {inference, alg_softmax, dt::f32, dt::s8, dt::undef, tag::nc,
                        tag::nc, tag::undef, {2, 1000}, 0},
                tp {inference, alg_softmax, dt::f32, dt::s8, dt::undef, tag::nc,
                        tag::cn, tag::undef, {2, 1000}, 1},
                tp {inference, alg_softmax, dt::f32, dt::s8, dt::undef, tag::nc,
                        tag::any, tag::undef, {1, 13}, 1},
                tp {inference, alg_softmax, dt::f32, dt::s8, dt::undef,
                        tag::ncw, tag::ncw, tag::undef, {16, 257, 32}, 1},
                tp {inference, alg_logsoftmax, dt::f32, dt::s8, dt::undef,
                        tag::ncw, tag::ncw, tag::undef, {16, 257, 32}, 2},
                tp {inference, alg_softmax, dt::f32, dt::s8, dt::undef,
                        tag::nhwc, tag::nhwc, tag::undef, {64, 1011, 1, 1}, 1},
                tp {inference, alg_softmax, dt::f32, dt::s8, dt::undef,
                        tag::nhwc, tag::nhwc, tag::undef, {3, 1011, 1, 1}, 1},
                tp {inference, alg_logsoftmax, dt::f32, dt::s8, dt::undef,
                        tag::nhwc, tag::nhwc, tag::undef, {2, 1011, 32, 1},
                        2}));

} // namespace dnnl
