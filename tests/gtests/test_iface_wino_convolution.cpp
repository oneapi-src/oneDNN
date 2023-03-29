/*******************************************************************************
* Copyright 2020-2023 Intel Corporation
* Copyright 2023 Arm Ltd. and affiliates
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

#if DNNL_CPU_RUNTIME != DNNL_RUNTIME_NONE
#include "tests/test_isa_common.hpp"
#endif
namespace dnnl {

// short names for brevity
using data_type = memory::data_type;
using tag = memory::format_tag;

class wino_conv_test_t : public ::testing::Test {
protected:
    engine eng = get_test_engine();
    struct input_data_t {
        data_type dat_dt;
        data_type wei_dt;
        bool wino_supported = false;
        bool backward_supported = false;
    } input_f32, input_f16, input_int8;

    void SetUp() override {
        input_f32.dat_dt = data_type::f32;
        input_f32.wei_dt = data_type::f32;

        input_f16.dat_dt = data_type::f16;
        input_f16.wei_dt = data_type::f16;

        input_int8.dat_dt = data_type::u8;
        input_int8.wei_dt = data_type::s8;

#if DNNL_X64 || DNNL_CPU_RUNTIME == DNNL_RUNTIME_NONE
        const bool is_gpu = get_test_engine_kind() == engine::kind::gpu;
        input_f32.wino_supported = is_gpu;
        input_f16.wino_supported = is_gpu;
#endif
    }
};

TEST_F(wino_conv_test_t, TestSmallPadding) {
    for (const auto &input : {input_f32, input_f16, input_int8}) {
        if (unsupported_data_type(input.dat_dt)
                || unsupported_data_type(input.wei_dt))
            continue;

        memory::desc src_md {{1, 16, 7, 7}, input.dat_dt, tag::any};
        memory::desc wei_md {{32, 16, 3, 3}, input.wei_dt, tag::any};
        memory::desc dst_md {{1, 32, 7, 7}, input.dat_dt, tag::any};

        if (input.wino_supported) {
            convolution_forward::primitive_desc fwd_hint;
            EXPECT_NO_THROW(
                    fwd_hint = convolution_forward::primitive_desc(eng,
                            prop_kind::forward, algorithm::convolution_winograd,
                            src_md, wei_md, dst_md, {1, 1}, {1, 1}, {1, 1}));

            if (input.backward_supported) {
                EXPECT_NO_THROW(convolution_backward_data::primitive_desc(eng,
                        algorithm::convolution_winograd, src_md, wei_md, dst_md,
                        {1, 1}, {1, 1}, {1, 1}, fwd_hint));

                EXPECT_NO_THROW(convolution_backward_weights::primitive_desc(
                        eng, algorithm::convolution_winograd, src_md, wei_md,
                        dst_md, {1, 1}, {1, 1}, {1, 1}, fwd_hint));
            }
        } else {
            EXPECT_ANY_THROW(convolution_forward::primitive_desc(eng,
                    prop_kind::forward, algorithm::convolution_winograd, src_md,
                    wei_md, dst_md, {1, 1}, {1, 1}, {1, 1}));
        }
    }
}

TEST_F(wino_conv_test_t, TestLargePadding) {
    for (const auto &input : {input_f32, input_f16, input_int8}) {
        if (unsupported_data_type(input.dat_dt)
                || unsupported_data_type(input.wei_dt))
            continue;

        memory::desc src_md {{1, 16, 7, 7}, input.dat_dt, tag::any};
        memory::desc wei_md {{32, 16, 3, 3}, input.wei_dt, tag::any};
        memory::desc dst_md {{1, 32, 9, 9}, input.dat_dt, tag::any};

        bool large_pad_is_supported
                = (get_test_engine_kind() == engine::kind::gpu);
        if (input.wino_supported && large_pad_is_supported) {
            EXPECT_NO_THROW(convolution_forward::primitive_desc(eng,
                    prop_kind::forward, algorithm::convolution_winograd, src_md,
                    wei_md, dst_md, {1, 1}, {2, 2}, {2, 2}));
        } else {
            EXPECT_ANY_THROW(convolution_forward::primitive_desc(eng,
                    prop_kind::forward, algorithm::convolution_winograd, src_md,
                    wei_md, dst_md, {1, 1}, {2, 2}, {2, 2}));
        }
    }
}

TEST_F(wino_conv_test_t, TestUnsupportedKernel) {
    SKIP_IF_HIP(true, "Unsupported test case.");
    for (const auto &input : {input_f32, input_f16, input_int8}) {
        if (unsupported_data_type(input.dat_dt)
                || unsupported_data_type(input.wei_dt))
            continue;

        memory::desc src_md {{1, 16, 5, 5}, input.dat_dt, tag::any};
        memory::desc wei_md {{32, 16, 2, 2}, input.wei_dt, tag::any};
        memory::desc dst_md {{1, 32, 6, 6}, input.dat_dt, tag::any};

        EXPECT_ANY_THROW(convolution_forward::primitive_desc(eng,
                prop_kind::forward, algorithm::convolution_winograd, src_md,
                wei_md, dst_md, {1, 1}, {1, 1}, {1, 1}));
    }
}

} // namespace dnnl
