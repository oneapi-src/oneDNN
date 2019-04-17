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

class engine_test_c : public ::testing::TestWithParam<mkldnn_engine_kind_t>
{
};
class engine_test_cpp : public ::testing::TestWithParam<mkldnn_engine_kind_t>
{
};

static void compare_backend(
        mkldnn_engine_kind_t engine_kind, mkldnn_backend_kind_t backend_kind) {
    if (engine_kind == mkldnn_cpu) {
#if MKLDNN_CPU_BACKEND == MKLDNN_BACKEND_NATIVE
        EXPECT_EQ(backend_kind, mkldnn_backend_native);
#endif
    } else if (engine_kind == mkldnn_gpu) {
#if MKLDNN_GPU_BACKEND == MKLDNN_BACKEND_OPENCL
        EXPECT_EQ(backend_kind, mkldnn_backend_ocl);
#endif
    }
}

TEST_P(engine_test_c, GetBackend) {
    mkldnn_engine_kind_t engine_kind = GetParam();
    SKIP_IF(mkldnn_engine_get_count(engine_kind) == 0,
            "Engine kind is not supported.");

    mkldnn_engine_t engine;
    MKLDNN_CHECK(mkldnn_engine_create(&engine, engine_kind, 0));

    mkldnn_backend_kind_t backend_kind;
    MKLDNN_CHECK(mkldnn_engine_get_backend_kind(engine, &backend_kind));

    compare_backend(engine_kind, backend_kind);

    MKLDNN_CHECK(mkldnn_engine_destroy(engine));
}

TEST_P(engine_test_c, CreateWithBackend) {
    mkldnn_engine_kind_t engine_kind = GetParam();
    SKIP_IF(mkldnn_engine_get_count(engine_kind) == 0,
            "Engine kind is not supported.");

    auto backend_kinds = { mkldnn_backend_native, mkldnn_backend_ocl };
    for (auto backend_kind : backend_kinds) {
        if (engine_kind == mkldnn_cpu) {
            if (backend_kind == mkldnn_backend_ocl)
                continue;
        } else if (engine_kind == mkldnn_gpu) {
            if (backend_kind == mkldnn_backend_native)
                continue;
        }

        mkldnn_engine_t engine;
        MKLDNN_CHECK(mkldnn_engine_create_with_backend(
                &engine, engine_kind, backend_kind, 0));

        mkldnn_backend_kind_t queried_backend_kind;
        MKLDNN_CHECK(
                mkldnn_engine_get_backend_kind(engine, &queried_backend_kind));

        EXPECT_EQ(queried_backend_kind, backend_kind);

        MKLDNN_CHECK(mkldnn_engine_destroy(engine));
    }
}

TEST_P(engine_test_cpp, GetBackend) {
    engine::kind engine_kind = static_cast<engine::kind>(GetParam());
    SKIP_IF(engine::get_count(engine_kind) == 0,
            "Engine kind is not supported.");

    engine eng(engine_kind, 0);

    compare_backend(static_cast<mkldnn_engine_kind_t>(engine_kind),
            static_cast<mkldnn_backend_kind_t>(eng.get_backend_kind()));
}

namespace {
struct PrintToStringParamName {
    template <class ParamType>
    std::string operator()(
            const ::testing::TestParamInfo<ParamType> &info) const {
        return to_string(info.param);
    }
};

auto all_engine_kinds = ::testing::Values(mkldnn_cpu, mkldnn_gpu);

} // namespace

INSTANTIATE_TEST_SUITE_P(AllEngineKinds, engine_test_c, all_engine_kinds,
        PrintToStringParamName());
INSTANTIATE_TEST_SUITE_P(AllEngineKinds, engine_test_cpp, all_engine_kinds,
        PrintToStringParamName());

} // namespace mkldnn
