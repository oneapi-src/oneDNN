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

#include "interface/stream.hpp"

#include "cpp/unit/unit_test_common.hpp"
#include "cpp/unit/utils.hpp"

#include "gtest/gtest.h"

namespace impl = dnnl::graph::impl;
namespace utils = dnnl::graph::tests::unit::utils;

TEST(Stream, DnnlGraphStreamCreate) {
    impl::stream_t *stream;
    impl::engine_t &engine = get_engine();
    if (engine.kind() == impl::engine_kind::gpu) {
        ASSERT_EQ(dnnl_graph_stream_create(&stream, &engine),
                impl::status::invalid_arguments);
    } else {
#ifdef DNNL_GRAPH_CPU_SYCL
        ASSERT_EQ(dnnl_graph_stream_create(&stream, &engine),
                impl::status::invalid_arguments);
#else
        ASSERT_EQ(dnnl_graph_stream_create(&stream, &engine),
                impl::status::success);
        ASSERT_EQ(dnnl_graph_stream_destroy(stream), impl::status::success);
#endif
    }
}
