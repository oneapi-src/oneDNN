/*******************************************************************************
* Copyright 2020-2022 Intel Corporation
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

#ifndef GRAPH_API_TEST_API_COMMON_H
#define GRAPH_API_TEST_API_COMMON_H

#include <gtest/gtest.h>

#include "oneapi/dnnl/dnnl_graph.h"

#define ASSERT_EQ_SAFE(val1, val2, ...) \
    do { \
        auto result = (val1); \
        if (result != (val2)) { \
            {__VA_ARGS__} ASSERT_EQ(result, val2); \
            return; \
        } \
    } while (0)

#endif
