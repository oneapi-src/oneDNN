/*******************************************************************************
* Copyright 2024 Intel Corporation
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

#include "src/common/float4.hpp"
#include "src/common/int4.hpp"
#include "src/common/nstl.hpp"
#include "src/common/type_helpers.hpp"

namespace dnnl {

template <typename T>
void test_limits(float max, float lowest, float epsilon) {
    ASSERT_EQ(max, static_cast<float>(impl::nstl::numeric_limits<T>::max()));
    ASSERT_EQ(lowest,
            static_cast<float>(impl::nstl::numeric_limits<T>::lowest()));
    ASSERT_EQ(epsilon,
            static_cast<float>(impl::nstl::numeric_limits<T>::epsilon()));
}

TEST(test_limits, int4) {
    test_limits<impl::int4_t>(7.f, -8.f, 0.f);
}

TEST(test_limits, uint4) {
    test_limits<impl::uint4_t>(15.f, 0.f, 0.f);
}

TEST(test_limits, f4_e2m1) {
    test_limits<impl::float4_e2m1_t>(6.0f, -6.0f, 1.0f);
}

TEST(test_limits, f4_e3m0) {
    test_limits<impl::float4_e3m0_t>(16.0f, -16.0f, 1.0f);
}

template <typename T>
void test_conversions() {
    impl::parallel_nd(0xff, [&](uint8_t u8) {
        // Each uint8_t contains a pair of 4-bit numbers.
        // Convert T -> f32 and back again,
        // expecting bitwise identical values.
        impl::nibble2_t T_pair(u8);
        float num1 = static_cast<T>(T_pair.get(0));
        float num2 = static_cast<T>(T_pair.get(1));
        // Check that the all numbers are in the range
        float T_lowest
                = static_cast<float>(impl::nstl::numeric_limits<T>::lowest());
        float T_max = static_cast<float>(impl::nstl::numeric_limits<T>::max());
        ASSERT_TRUE(num1 >= T_lowest && num1 <= T_max);
        ASSERT_TRUE(num2 >= T_lowest && num2 <= T_max);

        // Check that the numbers are extracted in the right order
        if (u8 <= 0xf)
            ASSERT_TRUE(num2 == 0);
        else
            // only case for num2 == -num2 is with fp types and signed 0.
            ASSERT_TRUE(num2 != 0 || num2 == -num2);

        // The target value must be initialized
        impl::nibble2_t new_T_pair(static_cast<T>(T_pair.get(0)).raw_bits_,
                static_cast<T>(T_pair.get(1)).raw_bits_);
        ASSERT_EQ(T_pair.get(), new_T_pair.get());
    });
}

TEST(test_int4_conversion, int4) {
    test_conversions<impl::int4_t>();
}

TEST(test_int4_conversion, uint4) {
    test_conversions<impl::uint4_t>();
}

TEST(test_e2m1_conversion, f4_e2m1) {
    test_conversions<impl::float4_e2m1_t>();
}

TEST(test_e3m0_conversion, f4_e3m0) {
    test_conversions<impl::float4_e3m0_t>();
}

} // namespace dnnl
