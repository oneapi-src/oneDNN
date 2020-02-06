/*******************************************************************************
* Copyright 2019 Intel Corporation
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

#include "cpu_isa_traits.hpp"
#include "dnnl.hpp"

namespace dnnl {

using tag = memory::format_tag;

struct binary_test_params {
    std::vector<tag> srcs_format;
    tag dst_format;
    algorithm aalgorithm;
    memory::dims dims;
    bool expect_to_fail;
    dnnl_status_t expected_status;
};

template <typename src0_data_t, typename src1_data_t = src0_data_t,
        typename dst_data_t = src0_data_t>
class binary_test : public ::testing::TestWithParam<binary_test_params> {
private:
    binary_test_params p;
    memory::data_type src0_dt, src1_dt, dst_dt;

protected:
    virtual void SetUp() {
        src0_dt = data_traits<src0_data_t>::data_type;
        src1_dt = data_traits<src1_data_t>::data_type;
        dst_dt = data_traits<dst_data_t>::data_type;

        p = ::testing::TestWithParam<binary_test_params>::GetParam();

        SKIP_IF(src0_dt == memory::data_type::f16
                        && get_test_engine_kind() == engine::kind::cpu,
                "F16 not supported with CPU engine");
        SKIP_IF(src0_dt == memory::data_type::bf16
                        && !impl::cpu::mayiuse(impl::cpu::avx512_core)
                        && get_test_engine_kind() == engine::kind::cpu,
                "current ISA doesn't support bfloat16 data type");

        // TODO: remove SKIP_IF when GPU adds int8 support
        SKIP_IF((src0_dt == memory::data_type::s8
                        || src0_dt == memory::data_type::u8)
                        && get_test_engine_kind() == engine::kind::gpu,
                "GPU doesn't support int8 data type");

        catch_expected_failures(
                [=]() { Test(); }, p.expect_to_fail, p.expected_status);
    }

    void Test() {
        // binary specific types and values
        using op_desc_t = binary::desc;
        using pd_t = binary::primitive_desc;
        allows_attr_t aa {0};
        aa.po_sum = 1;
        aa.po_eltwise = 1;
        if (get_test_engine_kind() == engine::kind::cpu) { aa.scales = 1; }

        auto eng = engine(get_test_engine_kind(), 0);
        auto strm = stream(eng);

        std::vector<memory::desc> srcs_md;
        std::vector<memory> srcs;

        for (int i_case = 0;; ++i_case) {
            memory::dims dims_B = p.dims;
            if (i_case == 0) {
            } else if (i_case == 1) {
                dims_B[0] = 1;
            } else if (i_case == 2) {
                dims_B[1] = 1;
                dims_B[2] = 1;
            } else if (i_case == 3) {
                dims_B[0] = 1;
                dims_B[2] = 1;
                dims_B[3] = 1;
            } else if (i_case == 4) {
                dims_B[0] = 1;
                dims_B[1] = 1;
                dims_B[2] = 1;
                dims_B[3] = 1;
            } else {
                break;
            }

            auto desc_A = memory::desc(p.dims, src0_dt, p.srcs_format[0]);
            // TODO: try to fit "reshape" logic here.
            auto desc_B = memory::desc(dims_B, src1_dt, memory::dims());
            auto desc_C = memory::desc(p.dims, dst_dt, p.dst_format);

            // default op desc ctor
            auto op_desc = op_desc_t();
            // regular op desc ctor
            op_desc = op_desc_t(p.aalgorithm, desc_A, desc_B, desc_C);

            // default pd ctor
            auto pd = pd_t();
            // regular pd ctor
            ASSERT_NO_THROW(pd = pd_t(op_desc, eng));
            // test all pd ctors
            test_fwd_pd_constructors<op_desc_t, pd_t>(op_desc, pd, aa);

            // default primitive ctor
            auto prim = binary();
            // regular primitive ctor
            prim = binary(pd);

            // query for descs from pd
            const auto src0_desc = pd.src_desc(0);
            const auto src1_desc = pd.src_desc(1);
            const auto dst_desc = pd.dst_desc();
            const auto workspace_desc = pd.workspace_desc();

            ASSERT_TRUE(pd.query_md(query::exec_arg_md, DNNL_ARG_SRC_0)
                    == src0_desc);
            ASSERT_TRUE(pd.query_md(query::exec_arg_md, DNNL_ARG_SRC_1)
                    == src1_desc);
            ASSERT_TRUE(
                    pd.query_md(query::exec_arg_md, DNNL_ARG_DST) == dst_desc);

            // check primitive returns zero_md for all rest md
            ASSERT_TRUE(pd.weights_desc().is_zero());
            ASSERT_TRUE(pd.diff_src_desc().is_zero());
            ASSERT_TRUE(pd.diff_dst_desc().is_zero());
            ASSERT_TRUE(pd.diff_weights_desc().is_zero());

            const auto test_engine = pd.get_engine();

            auto mem_A = memory(src0_desc, test_engine);
            auto mem_B = memory(src1_desc, test_engine);
            auto mem_C = memory(dst_desc, test_engine);
            auto mem_ws = memory(workspace_desc, test_engine);

            prim.execute(strm,
                    {{DNNL_ARG_SRC_0, mem_A}, {DNNL_ARG_SRC_1, mem_B},
                            {DNNL_ARG_DST, mem_C},
                            {DNNL_ARG_WORKSPACE, mem_ws}});
            strm.wait();
        }
    }
};

static auto expected_failures = []() {
    return ::testing::Values(
            // different src0 and dst format_tags
            binary_test_params {{tag::nchw, tag::nchw}, tag::nhwc,
                    algorithm::binary_add, {1, 8, 4, 4}, true,
                    dnnl_invalid_arguments},
            // not supported alg_kind
            binary_test_params {{tag::nchw, tag::nchw}, tag::nchw,
                    algorithm::eltwise_relu, {1, 8, 4, 4}, true,
                    dnnl_invalid_arguments},
            // negative dim
            binary_test_params {{tag::nchw, tag::nchw}, tag::nchw,
                    algorithm::binary_add, {-1, 8, 4, 4}, true,
                    dnnl_invalid_arguments});
};

static auto zero_dim = []() {
    return ::testing::Values(
            binary_test_params {{tag::nchw, tag::nchw}, tag::nchw,
                    algorithm::binary_add, {0, 7, 6, 5}},
            binary_test_params {{tag::nChw8c, tag::nhwc}, tag::nChw8c,
                    algorithm::binary_mul, {5, 0, 7, 6}},
            binary_test_params {{tag::nChw16c, tag::nchw}, tag::nChw16c,
                    algorithm::binary_add, {8, 15, 0, 5}},
            binary_test_params {{tag::nhwc, tag::nChw16c}, tag::nhwc,
                    algorithm::binary_mul, {5, 16, 7, 0}});
};

static auto simple_cases = []() {
    return ::testing::Values(
            binary_test_params {{tag::nchw, tag::nchw}, tag::nchw,
                    algorithm::binary_add, {8, 7, 6, 5}},
            binary_test_params {{tag::nhwc, tag::nhwc}, tag::nhwc,
                    algorithm::binary_mul, {5, 8, 7, 6}},
            binary_test_params {{tag::nChw8c, tag::nchw}, tag::nChw8c,
                    algorithm::binary_max, {8, 15, 6, 5}},
            binary_test_params {{tag::nhwc, tag::nChw16c}, tag::any,
                    algorithm::binary_min, {5, 16, 7, 6}});
};

#define INST_TEST_CASE(test) \
    TEST_P(test, Testsbinary) {} \
    INSTANTIATE_TEST_SUITE_P(TestbinaryEF, test, expected_failures()); \
    INSTANTIATE_TEST_SUITE_P(TestbinaryZero, test, zero_dim()); \
    INSTANTIATE_TEST_SUITE_P(TestbinarySimple, test, simple_cases());

using binary_test_f32 = binary_test<float>;
using binary_test_bf16 = binary_test<bfloat16_t>;
using binary_test_f16 = binary_test<float16_t>;
using binary_test_s8 = binary_test<int8_t>;
using binary_test_u8 = binary_test<uint8_t>;
using binary_test_s8u8s8 = binary_test<int8_t, uint8_t, int8_t>;
using binary_test_u8s8u8 = binary_test<uint8_t, int8_t, uint8_t>;

INST_TEST_CASE(binary_test_f32)
INST_TEST_CASE(binary_test_bf16)
INST_TEST_CASE(binary_test_f16)
INST_TEST_CASE(binary_test_s8)
INST_TEST_CASE(binary_test_u8)
INST_TEST_CASE(binary_test_s8u8s8)
INST_TEST_CASE(binary_test_u8s8u8)

} // namespace dnnl
