/*******************************************************************************
* Copyright 2022-2023 Intel Corporation
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

TEST(Stream, DnnlGraphStreamCreate) {
    impl::stream_t *stream = nullptr;
    impl::engine_t &engine = get_engine();
    impl::status_t ret = impl::status::success;

    if (engine.kind() == impl::engine_kind::gpu) {
        ret = dnnl_graph_stream_create(&stream, &engine);
        if (ret != impl::status::invalid_arguments) {
            dnnl_graph_stream_destroy(stream);
            FAIL();
        }
    } else {
#ifdef DNNL_GRAPH_CPU_SYCL
        ret = dnnl_graph_stream_create(&stream, &engine);
        if (ret != impl::status::invalid_arguments) {
            dnnl_graph_stream_destroy(stream);
            FAIL();
        }
#else
        ret = dnnl_graph_stream_create(&stream, &engine);
        if (ret != impl::status::success) {
            dnnl_graph_stream_destroy(stream);
            FAIL();
        }
#endif
    }

    if (stream != nullptr) { dnnl_graph_stream_destroy(stream); }
}
