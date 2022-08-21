/*******************************************************************************
* Copyright 2022 Intel Corporation
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

#include "interface/partition.hpp"

namespace impl = dnnl::graph::impl;

TEST(CompilationContext, Create) {
    impl::compilation_context_t ctx;

    size_t tid = 123;
    // a vector
    int content_1[5] = {1, 3, 5, 7, 10};
    ctx.set_tensor_data_handle(tid, content_1);

    ASSERT_TRUE(ctx.get_ids().size() == 1);
    ASSERT_TRUE(ctx.get_ids().count(tid) > 0);

    void *get_void_content_1 = ctx.get_tensor_data_handle(tid);
    ASSERT_TRUE(get_void_content_1 == content_1);

    int *get_real_content_1 = static_cast<int *>(get_void_content_1);
    for (size_t i = 0; i < 5; ++i) {
        ASSERT_EQ(get_real_content_1[i], content_1[i]);
    }

    // a scalar
    tid = 3;
    float content_2 = 0.463;
    ctx.set_tensor_data_handle(tid, &content_2);

    ASSERT_TRUE(ctx.get_ids().size() == 2);
    ASSERT_TRUE(ctx.get_ids().count(tid) > 0);

    void *get_void_content_2 = ctx.get_tensor_data_handle(tid);
    ASSERT_TRUE(get_void_content_2 == &content_2);

    float *get_real_content_2 = static_cast<float *>(get_void_content_2);
    ASSERT_EQ(*get_real_content_2, content_2);
}
