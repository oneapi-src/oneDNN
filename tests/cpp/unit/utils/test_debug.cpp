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
#include <string>

#include "utils/debug.hpp"

#include "oneapi/dnnl/dnnl_graph_types.h"

#include "gtest/gtest.h"

namespace impl = dnnl::graph::impl;

TEST(DebugUtilsTest, DnnlGraphRuntime2str) {
#define CASE(runtime, s) \
    ASSERT_EQ(std::string(impl::utils::dnnl_graph_runtime2str(runtime)), \
            std::string(#s))
    CASE(DNNL_GRAPH_RUNTIME_NONE, none);
    CASE(DNNL_GRAPH_RUNTIME_SEQ, sequential);
    CASE(DNNL_GRAPH_RUNTIME_OMP, OpenMP);
    CASE(DNNL_GRAPH_RUNTIME_TBB, TBB);
    CASE(DNNL_GRAPH_RUNTIME_THREADPOOL, threadpool);
#ifdef DNNL_GRAPH_WITH_SYCL
    CASE(DNNL_GRAPH_RUNTIME_SYCL, DPC++);
#else
    CASE(-1, unknown);
#endif

#undef CASE
}

TEST(DebugUtilsTest, DataType2str) {
    using namespace dnnl::graph::impl;
    using namespace dnnl::graph::impl::utils;
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

TEST(DebugUtilsTest, EngineKind2str) {
    using namespace dnnl::graph::impl;
    using namespace dnnl::graph::impl::utils;
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

TEST(DebugUtilsTest, LayoutType2str) {
    using namespace dnnl::graph::impl;
    using namespace dnnl::graph::impl::utils;
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

TEST(DebugUtilsTest, PropertyType2str) {
    using namespace dnnl::graph::impl;
    using namespace dnnl::graph::impl::utils;
    EXPECT_STREQ("undef", property_type2str(property_type::undef));
    EXPECT_STREQ("variable", property_type2str(property_type::variable));
    EXPECT_STREQ("constant", property_type2str(property_type::constant));
#ifndef NDEBUG
    EXPECT_DEATH(property_type2str(static_cast<property_type_t>(-1)),
            "unknown property_type");
#else
    EXPECT_STREQ("unknown property_type",
            property_type2str(static_cast<property_type_t>(-1)));
#endif
}

TEST(DebugUtilsTest, PartitionKind2str) {
    using namespace dnnl::graph::impl;
    using namespace dnnl::graph::impl::utils;
    using namespace impl::partition_kind;
#define CASE(v) ASSERT_EQ(partition_kind2str(v), std::string(#v))
    CASE(undef);
    CASE(convolution_post_ops);
    CASE(convtranspose_post_ops);
    CASE(interpolate_post_ops);
    CASE(matmul_post_ops);
    CASE(reduction_post_ops);
    CASE(unary_post_ops);
    CASE(binary_post_ops);
    CASE(pooling_post_ops);
    CASE(batch_norm_post_ops);
    CASE(misc_post_ops);
    CASE(quantized_convolution_post_ops);
    CASE(quantized_convtranspose_post_ops);
    CASE(quantized_matmul_post_ops);
    CASE(quantized_unary_post_ops);
    CASE(quantized_pooling_post_ops);
    CASE(misc_quantized_post_ops);
    CASE(convolution_backprop_post_ops);
    CASE(mha);
    CASE(mlp);
    CASE(quantized_mha);
    CASE(quantized_mlp);
    CASE(residual_conv_blocks);
    CASE(quantized_residual_conv_blocks);
#undef CASE
    ASSERT_EQ(partition_kind2str(static_cast<impl::partition_kind_t>(
                      quantized_residual_conv_blocks + 1000)),
            std::string("unknown_kind"));
}

TEST(DebugUtilsTest, FpmathMode2str) {
    using namespace dnnl::graph::impl;
    using namespace dnnl::graph::impl::utils;
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
