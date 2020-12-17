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

#include <gtest/gtest.h>

#include "oneapi/dnnl/dnnl_graph.hpp"

namespace {
int deletion_counter {0};

dnnl_graph_result_t destory(int *i) {
    ++deletion_counter;
    delete i;
    return dnnl_graph_result_success;
}

} // namespace

/**
 * 1. Create dnnl::graph::detail::handle to a manageable object with a custom deleter
 * 2. Destroy the handle
 * 3. Expect the deleter to be invoked
 */
TEST(api_base_test, managed_handle) {
    constexpr int expected_deletion {1};
    const int deletion_counter_before_deletion = [] {
        using handle = dnnl::graph::detail::handle<int, destory>;
        handle h {new int {}};
        return deletion_counter;
    }();
    EXPECT_EQ(deletion_counter,
            deletion_counter_before_deletion + expected_deletion);
}

/**
 * 1. Create dnnl::graph::detail::handle to an unmanageable object with a custom deleter
 * 2. Destroy the handle
 * 3. Expect the deleter to be not invoked
 */
TEST(api_base_test, unmanaged_handle) {
    constexpr int expected_deletion {0};
    const int deletion_counter_before_deletion = [] {
        using handle = dnnl::graph::detail::handle<int, destory>;
        auto h = handle(reinterpret_cast<int *>(1234), true);
        return deletion_counter;
    }();
    EXPECT_EQ(deletion_counter,
            deletion_counter_before_deletion + expected_deletion);
}
