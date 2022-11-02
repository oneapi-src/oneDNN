/*******************************************************************************
* Copyright 2019-2022 Intel Corporation
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

#include "oneapi/dnnl/dnnl.h"
#include "oneapi/dnnl/dnnl_types.h"

namespace dnnl {

class pd_test_t : public ::testing::Test {
protected:
    engine e = get_test_engine();
    memory::desc dat_md {
            {16, 16, 16, 16}, memory::data_type::f32, memory::format_tag::nhwc};
    memory::desc wht_md {
            {16, 16, 1, 1}, memory::data_type::f32, memory::format_tag::oihw};
};

TEST_F(pd_test_t, ConvTestNotEmpty) {
    bool no_exception = true;
    bool is_empty = false;

    try {
        auto default_attr = primitive_attr();
        auto pd = convolution_forward::primitive_desc {e,
                prop_kind::forward_inference, algorithm::convolution_direct,
                dat_md, wht_md, dat_md, {1, 1}, {0, 0}, {0, 0}, default_attr,
                false};
        is_empty = pd.get(true) == nullptr; // not reached if !allow_empty
    } catch (error &) { no_exception = false; }

    ASSERT_TRUE(no_exception);
    ASSERT_TRUE(!is_empty);
}

TEST_F(pd_test_t, ConvTestEmpty) {
    auto attrs = primitive_attr {};
    attrs.set_scales_mask(DNNL_ARG_SRC, 0);

    for (bool allow_empty : {true, false}) {
        bool no_exception = true;
        bool is_empty = false;

        try {
            auto pd = convolution_forward::primitive_desc {e,
                    prop_kind::forward_inference, algorithm::convolution_direct,
                    dat_md, wht_md, dat_md, {1, 1}, {0, 0}, {0, 0}, attrs,
                    allow_empty};
            is_empty = pd.get(true) == nullptr; // not reached if !allow_empty
        } catch (error &) { no_exception = false; }

        ASSERT_TRUE(no_exception == allow_empty);
        ASSERT_TRUE(is_empty == allow_empty);
    }
}

TEST_F(pd_test_t, TestOptionalQueries) {
    memory::desc a_md {
            {10, 10}, memory::data_type::f32, memory::format_tag::ab};
    memory::desc b_md {
            {10, 10}, memory::data_type::f32, memory::format_tag::ab};
    memory::desc c_md {
            {10, 10}, memory::data_type::f32, memory::format_tag::ab};

    auto pd = matmul::primitive_desc(e, a_md, b_md, c_md);

    ASSERT_TRUE(pd.get_strides().empty());
    ASSERT_TRUE(pd.get_dilations().empty());
    ASSERT_TRUE(pd.get_padding_l().empty());
    ASSERT_TRUE(pd.get_padding_r().empty());
    ASSERT_TRUE(pd.get_kernel().empty());
    ASSERT_TRUE(pd.get_factors().empty());

    ASSERT_EQ(pd.get_alpha(), 0.0f);
    ASSERT_EQ(pd.get_beta(), 0.0f);
    ASSERT_EQ(pd.get_epsilon(), 0.0f);
    ASSERT_EQ(pd.get_k(), 0.0f);
    ASSERT_EQ(pd.get_p(), 0.0f);

    ASSERT_EQ(pd.get_flags(), 0x0U);
    ASSERT_EQ(pd.get_local_size(), 0);
    ASSERT_EQ(pd.get_group_size(), 0);
    ASSERT_EQ(pd.get_axis(), -1);

    ASSERT_EQ(pd.get_algorithm(), dnnl::algorithm::undef);
    ASSERT_EQ(pd.get_cell_kind(), dnnl::algorithm::undef);
    ASSERT_EQ(pd.get_activation_kind(), dnnl::algorithm::undef);

    ASSERT_EQ(pd.get_prop_kind(), dnnl::prop_kind::undef);
}

} // namespace dnnl
