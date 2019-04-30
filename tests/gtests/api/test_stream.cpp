/*******************************************************************************
* Copyright 2019 Intel Corporation
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

#include "mkldnn_test_common.hpp"
#include "gtest/gtest.h"

#include "mkldnn.h"

namespace mkldnn {

TEST(stream_test_c, WaitNullStream) {
    mkldnn_stream_t stream = nullptr;
    mkldnn_status_t status = mkldnn_stream_wait(stream);
    ASSERT_EQ(status, mkldnn_invalid_arguments);
}

TEST(stream_test_c, Wait) {
    mkldnn_engine_t engine;
    MKLDNN_CHECK(mkldnn_engine_create(&engine, mkldnn_cpu, 0));

    mkldnn_stream_t stream;
    MKLDNN_CHECK(
            mkldnn_stream_create(&stream, engine, mkldnn_stream_default_flags));

    MKLDNN_CHECK(mkldnn_stream_wait(stream));

    MKLDNN_CHECK(mkldnn_stream_destroy(stream));
    MKLDNN_CHECK(mkldnn_engine_destroy(engine));
}

TEST(stream_test_cpp, Wait) {
    engine eng(engine::kind::cpu, 0);
    stream s(eng);
    s.wait();
}

} // namespace mkldnn
