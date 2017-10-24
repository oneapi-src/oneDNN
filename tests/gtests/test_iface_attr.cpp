/*******************************************************************************
* Copyright 2017 Intel Corporation
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

class attr_test: public ::testing::Test {
protected:
    virtual void SetUp() {}
};

TEST_F(attr_test, TestIntOutputRoundMode) {
    mkldnn::primitive_attr attr;
    for (auto r: {round_nearest, round_down})
    {
        attr.set_int_output_round_mode(r);
        EXPECT_EQ(r, attr.get_int_output_round_mode());
    }
}

TEST_F(attr_test, TestIntOutputScales) {
    mkldnn::primitive_attr attr;

    int mask;
    std::vector<float> scales;

    // default scales
    attr.get_output_scales(mask, scales);
    EXPECT_EQ(mask, 0);
    EXPECT_EQ(scales.size(), 1U);
    EXPECT_EQ(scales[0], 1.);

    // single non-default scale
    attr.set_output_scales(0, {2.});
    attr.get_output_scales(mask, scales);
    EXPECT_EQ(mask, 0);
    EXPECT_EQ(scales.size(), 1U);
    EXPECT_EQ(scales[0], 2.);

    // multiple scales
    attr.set_output_scales(1 << 1, {1., 2., 3.});
    attr.get_output_scales(mask, scales);
    EXPECT_EQ(mask, 1 << 1);
    EXPECT_EQ(scales.size(), 3U);
    EXPECT_EQ(scales[0], 1.);
    EXPECT_EQ(scales[1], 2.);
    EXPECT_EQ(scales[2], 3.);
}

}
