/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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

#include <vector>
#include <gtest/gtest.h>

#include "interface/logical_tensor.hpp"
#include "utils.hpp"

namespace impl = dnnl::graph::impl;
namespace utils = dnnl::graph::tests::unit::utils;

TEST(logical_tensor_test, simple_create) {
    const size_t id = 123;
    impl::logical_tensor_t lt
            = utils::logical_tensor_init(id, impl::data_type::f32);

    ASSERT_EQ(lt.id, id);
    ASSERT_EQ(lt.data_type, impl::data_type::f32);
}

TEST(logical_tensor_test, create_with_shape) {
    const size_t id = 123;

    impl::logical_tensor_t lt_0
            = utils::logical_tensor_init(id, {}, impl::data_type::f32);
    ASSERT_EQ(lt_0.id, id);
    ASSERT_EQ(lt_0.ndims, -1);
    ASSERT_EQ(lt_0.data_type, impl::data_type::f32);

    impl::logical_tensor_t lt_1
            = utils::logical_tensor_init(id, {3}, impl::data_type::f32);
    ASSERT_EQ(lt_1.id, id);
    ASSERT_EQ(lt_1.ndims, 1);
    ASSERT_EQ(lt_1.data_type, impl::data_type::f32);

    impl::logical_tensor_t lt_2
            = utils::logical_tensor_init(id, {3, 4}, impl::data_type::f32);
    ASSERT_EQ(lt_2.id, id);
    ASSERT_EQ(lt_2.ndims, 2);
    ASSERT_EQ(lt_2.data_type, impl::data_type::f32);

    impl::logical_tensor_t lt_3
            = utils::logical_tensor_init(id, {3, 4, 5}, impl::data_type::f32);
    ASSERT_EQ(lt_3.id, id);
    ASSERT_EQ(lt_3.ndims, 3);
    ASSERT_EQ(lt_3.data_type, impl::data_type::f32);

    impl::logical_tensor_t lt_4 = utils::logical_tensor_init(
            id, {3, 4, 5, 6}, impl::data_type::f32);
    ASSERT_EQ(lt_4.id, id);
    ASSERT_EQ(lt_4.ndims, 4);
    ASSERT_EQ(lt_4.data_type, impl::data_type::f32);
}

TEST(logical_tensor_test, copy) {
    const size_t id = 123;

    impl::logical_tensor_t lt_1
            = utils::logical_tensor_init(id, {3, 4}, impl::data_type::f32);
    impl::logical_tensor_t lt_2(lt_1);

    ASSERT_EQ(lt_1.id, lt_2.id);
    ASSERT_EQ(lt_1.ndims, lt_2.ndims);
    ASSERT_EQ(lt_1.data_type, lt_2.data_type);
}

TEST(logical_tensor_test, assign) {
    const size_t id = 123;

    impl::logical_tensor_t lt_1
            = utils::logical_tensor_init(id, {3, 4}, impl::data_type::f32);
    impl::logical_tensor_t lt_2 = lt_1;

    ASSERT_EQ(lt_1.id, lt_2.id);
    ASSERT_EQ(lt_1.ndims, lt_2.ndims);
    ASSERT_EQ(lt_1.data_type, lt_2.data_type);
}

TEST(logical_tensor_test, push_to_vector) {
    size_t num_inputs = 3;
    std::vector<impl::dim_t> dims {1};
    std::vector<impl::logical_tensor_t> lt_vec;
    lt_vec.reserve(num_inputs);
    for (size_t i = 0; i < num_inputs; ++i) {
        lt_vec.emplace_back(
                utils::logical_tensor_init(i, dims, impl::data_type::f32));
    }

    for (size_t i = 0; i < num_inputs; ++i) {
        ASSERT_EQ(lt_vec[i].ndims, dims.size());
    }
}
