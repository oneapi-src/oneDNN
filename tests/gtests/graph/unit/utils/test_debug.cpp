/*******************************************************************************
* Copyright 2021-2023 Intel Corporation
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

#include <string>

#include <gtest/gtest.h>

#include "interface/c_types_map.hpp"
#include "utils/debug.hpp"

TEST(DebugUtilsDeathTest, DataType2str) {
    using namespace dnnl::impl::graph;
    using namespace dnnl::impl::graph::utils;
    EXPECT_STREQ("undef", data_type2str(data_type::undef));
    EXPECT_STREQ("f16", data_type2str(data_type::f16));
    EXPECT_STREQ("bf16", data_type2str(data_type::bf16));
    EXPECT_STREQ("f32", data_type2str(data_type::f32));
    EXPECT_STREQ("s32", data_type2str(data_type::s32));
    EXPECT_STREQ("s8", data_type2str(data_type::s8));
    EXPECT_STREQ("u8", data_type2str(data_type::u8));
    EXPECT_STREQ("boolean", data_type2str(data_type::boolean));
#ifndef NDEBUG
    EXPECT_DEATH(data_type2str(static_cast<data_type_t>(data_type::u8 + 1)),
            "unknown data_type");
#else
    EXPECT_STREQ("unknown data_type",
            data_type2str(static_cast<data_type_t>(data_type::u8 + 1)));
#endif
}

TEST(DebugUtilsDeathTest, EngineKind2str) {
    using namespace dnnl::impl::graph;
    using namespace dnnl::impl::graph::utils;
    EXPECT_STREQ("any", engine_kind2str(engine_kind::any_engine));
    EXPECT_STREQ("cpu", engine_kind2str(engine_kind::cpu));
    EXPECT_STREQ("gpu", engine_kind2str(engine_kind::gpu));
#ifndef NDEBUG
    EXPECT_DEATH(
            engine_kind2str(static_cast<engine_kind_t>(engine_kind::gpu + 1)),
            "unknown engine_kind");
#else
    EXPECT_STREQ("unknown engine_kind",
            engine_kind2str(static_cast<engine_kind_t>(engine_kind::gpu + 1)));
#endif
}

TEST(DebugUtilsDeathTest, LayoutType2str) {
    using namespace dnnl::impl::graph;
    using namespace dnnl::impl::graph::utils;
    EXPECT_STREQ("undef", layout_type2str(layout_type::undef));
    EXPECT_STREQ("any", layout_type2str(layout_type::any));
    EXPECT_STREQ("strided", layout_type2str(layout_type::strided));
    EXPECT_STREQ("opaque", layout_type2str(layout_type::opaque));
#ifndef NDEBUG
    EXPECT_DEATH(layout_type2str(
                         static_cast<layout_type_t>(layout_type::opaque + 1)),
            "unknown layout_type");
#else
    EXPECT_STREQ("unknown layout_type",
            layout_type2str(
                    static_cast<layout_type_t>(layout_type::opaque + 1)));
#endif
}

TEST(DebugUtilsDeathTest, FpmathMode2str) {
    using namespace dnnl::impl::graph;
    using namespace dnnl::impl::graph::utils;
    EXPECT_STREQ("strict", fpmath_mode2str(fpmath_mode::strict));
    EXPECT_STREQ("bf16", fpmath_mode2str(fpmath_mode::bf16));
    EXPECT_STREQ("f16", fpmath_mode2str(fpmath_mode::f16));
    EXPECT_STREQ("any", fpmath_mode2str(fpmath_mode::any));
    EXPECT_STREQ("tf32", fpmath_mode2str(fpmath_mode::tf32));
#ifndef NDEBUG
    EXPECT_DEATH(
            fpmath_mode2str(static_cast<fpmath_mode_t>(fpmath_mode::tf32 + 1)),
            "unknown fpmath_mode");
#else
    EXPECT_STREQ("unknown fpmath_mode",
            fpmath_mode2str(static_cast<fpmath_mode_t>(fpmath_mode::tf32 + 1)));
#endif
}
