/*******************************************************************************
* Copyright 2019-2020 Intel Corporation
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

#include "dnnl.h"

namespace dnnl {

TEST(stream_test_c, WaitNullStream) {
    dnnl_stream_t stream = nullptr;
    dnnl_status_t status = dnnl_stream_wait(stream);
    ASSERT_EQ(status, dnnl_invalid_arguments);
}

TEST(stream_test_c, Wait) {
    dnnl_engine_t engine;
    DNNL_CHECK(dnnl_engine_create(&engine, dnnl_cpu, 0));

    dnnl_stream_t stream;
    DNNL_CHECK(dnnl_stream_create(&stream, engine, dnnl_stream_default_flags));

    DNNL_CHECK(dnnl_stream_wait(stream));

    DNNL_CHECK(dnnl_stream_destroy(stream));
    DNNL_CHECK(dnnl_engine_destroy(engine));
}

TEST(stream_test_c, GetStream) {
    dnnl_engine_t engine;
    DNNL_CHECK(dnnl_engine_create(&engine, dnnl_cpu, 0));

    dnnl_stream_t stream;
    DNNL_CHECK(dnnl_stream_create(&stream, engine, dnnl_stream_default_flags));

    dnnl_engine_t stream_engine;
    DNNL_CHECK(dnnl_stream_get_engine(stream, &stream_engine));
    ASSERT_EQ(engine, stream_engine);

    DNNL_CHECK(dnnl_stream_destroy(stream));
    DNNL_CHECK(dnnl_engine_destroy(engine));
}

TEST(stream_test_cpp, Wait) {
    engine eng(engine::kind::cpu, 0);
    stream s(eng);
    engine s_eng = s.get_engine();
    s.wait();
}

} // namespace dnnl
