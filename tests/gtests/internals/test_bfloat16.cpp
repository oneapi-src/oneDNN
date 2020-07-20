/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#include "src/common/bfloat16.hpp"

namespace dnnl {

TEST(test_bfloat16_plus_float, TestDenormF32) {
    const float denorm_f32 {FLT_MIN / 2.0f};
    const bfloat16_t initial_value_bf16 {FLT_MIN};

    bfloat16_t expect_bf16 {float {initial_value_bf16} + denorm_f32};
    ASSERT_GT(float {expect_bf16}, float {initial_value_bf16});

    bfloat16_t bf16_plus_equal_f32 = initial_value_bf16;
    bf16_plus_equal_f32 += denorm_f32;
    ASSERT_EQ(float {bf16_plus_equal_f32}, float {expect_bf16});

    bfloat16_t bf16_plus_f32 = initial_value_bf16 + denorm_f32;
    ASSERT_EQ(float {bf16_plus_f32}, float {expect_bf16});
}

} // namespace dnnl
