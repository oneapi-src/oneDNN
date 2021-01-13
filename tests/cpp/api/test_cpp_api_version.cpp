/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#include <climits>
#include <cstring>

#include <gtest/gtest.h>

#include "oneapi/dnnl/dnnl_graph.hpp"

#ifndef DNNL_GRAPH_VERSION_MAJOR
#define DNNL_GRAPH_VERSION_MAJOR INT_MAX
#endif

#ifndef DNNL_GRAPH_VERSION_MINOR
#define DNNL_GRAPH_VERSION_MINOR INT_MAX
#endif

#ifndef DNNL_GRAPH_VERSION_PATCH
#define DNNL_GRAPH_VERSION_PATCH INT_MAX
#endif

#ifndef DNNL_GRAPH_VERSION_HASH
#define DNNL_GRAPH_VERSION_HASH "N/A"
#endif

TEST(cpp_api_test, version) {
    const dnnl::graph::version_t *version = dnnl::graph::version();
    EXPECT_NE(version->major, INT_MAX);
    EXPECT_NE(version->minor, INT_MAX);
    EXPECT_NE(version->patch, INT_MAX);
    EXPECT_EQ(version->major, DNNL_GRAPH_VERSION_MAJOR);
    EXPECT_EQ(version->minor, DNNL_GRAPH_VERSION_MINOR);
    EXPECT_EQ(version->patch, DNNL_GRAPH_VERSION_PATCH);
    EXPECT_EQ(std::strcmp(version->hash, DNNL_GRAPH_VERSION_HASH), 0);
}
