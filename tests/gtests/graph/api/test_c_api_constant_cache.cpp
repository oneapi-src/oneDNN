/*******************************************************************************
* Copyright 2021-2022 Intel Corporation
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

#include "oneapi/dnnl/dnnl_graph.h"

TEST(CAPI, ConstantTensorCache) {
    int flag = 0;
    ASSERT_EQ(dnnl_graph_get_constant_tensor_cache(&flag), dnnl_success);
    ASSERT_EQ(dnnl_graph_set_constant_tensor_cache(1), dnnl_success);
    ASSERT_EQ(dnnl_graph_get_constant_tensor_cache(&flag), dnnl_success);
    ASSERT_EQ(flag, 1);

    // negative test
    ASSERT_EQ(dnnl_graph_get_constant_tensor_cache(nullptr),
            dnnl_invalid_arguments);
    ASSERT_EQ(dnnl_graph_set_constant_tensor_cache(-1), dnnl_invalid_arguments);
}
