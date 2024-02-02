/*******************************************************************************
 * Copyright 2022-2024 Intel Corporation
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

#include <algorithm>
#include <cmath>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include "compiler/dimensions.hpp"
#include "compiler/ir/graph/driver.hpp"
#include "compiler/ir/graph/graph.hpp"
#include "compiler/ir/graph/lowering.hpp"
#include "compiler/ir/graph/pass/pass.hpp"
#include "compiler/ir/graph/quantization/quantize_info.hpp"
#include "context.hpp"
#include "gtest.h"
#include "ops/fusible/pooling.hpp"
#include "reference/pool_ref.hpp"
#include "util/any_map.hpp"
#include <compiler/ir/graph/mixed_partition.hpp>
#include <compiler/ir/graph/tunable_op.hpp>
#include <compiler/ir/transform/loop_merge.hpp>
#include <compiler/ir/transform/simplify.hpp>
#include <compiler/jit/jit.hpp>
#include <ops/templates/nested_conv_fwd.hpp>
#include <runtime/dynamic_dispatch/dynamic_tensor.hpp>

using namespace dnnl::impl::graph::gc;
using nested_conv_fwd_config_t = ops::nested_conv_fwd_config_t;

template <typename Store_type, typename Compute_type = Store_type>
static void check_fusible_pooling_fwd(pooling_type_t pool_type, int N, int C,
        int H, int W, int R, int S, sc_dims strides, sc_dims padding,
        bool add_bn_relu = true, int c_block = 1, bool exclude_pad = false,
        bool round_floor = true, std::string auto_pad = auto_pad_options::none,
        bool channel_last = false) {
    REQUIRE_AVX2();
    int stride_h = strides[0], stride_w = strides[0];
    if (strides.size() > 1) stride_w = strides[1];

    sc_data_type_t dtype = sc_data_traits_t<Store_type>::type();
    sc_graph_t mgr;
    std::vector<sc_op_ptr> fuse_arg_ops;
    auto pooling_op_type_str
            = pool_type == pooling_type_t::max ? "pooling_max" : "pooling_avg";
    auto in_fmt = (c_block == 1) ? sc_data_format_t::NCHW()
                                 : sc_data_format_t::NCHWc(c_block);
    sc_dims in_shape = {N, C, H, W};
    std::string data_format = data_format_options::NCX;
    if (channel_last) {
        // when use channel_last input, the channel axis should not be blocked.
        EXPECT_EQ(c_block, 1);
        data_format = data_format_options::NXC;
        in_shape = {N, H, W, C};
    }
    auto in_a = mgr.make_input({graph_tensor::make(in_shape, in_fmt, dtype)});
    std::string rounding_type = round_floor ? rounding_type_options::floor
                                            : rounding_type_options::ceil;
    auto pooling_out
            = mgr.make(pooling_op_type_str, {in_a->get_outputs()[0]}, {},
                    {{pooling_attr_key::strides, strides},
                            {pooling_attr_key::pads_begin, padding},
                            {pooling_attr_key::pads_end, padding},
                            {pooling_attr_key::kernel, sc_dims {R, S}},
                            {pooling_attr_key::exclude_pad, exclude_pad},
                            {pooling_attr_key::rounding_type, rounding_type},
                            {pooling_attr_key::auto_pad, auto_pad},
                            {pooling_attr_key::data_format, data_format}});
    fuse_arg_ops = {
            in_a,
    };
    sc_op_ptr final_out = pooling_out;

    auto bc_axis = std::vector<int> {1};
    if (add_bn_relu) {
        auto bn_mul = mgr.make_input({graph_tensor::make({C})});
        auto bn_add = mgr.make_input({graph_tensor::make({C})});
        final_out = mgr.make("mul",
                {final_out->get_outputs()[0], bn_mul->get_outputs()[0]}, {},
                {{"bc_axis", bc_axis}});

        final_out = mgr.make("add",
                {final_out->get_outputs()[0], bn_add->get_outputs()[0]}, {},
                {{"bc_axis", bc_axis}});
        final_out = mgr.make("relu", {final_out->get_outputs()[0]}, {}, {});
        fuse_arg_ops.emplace_back(bn_mul);
        fuse_arg_ops.emplace_back(bn_add);
    }
    auto out = mgr.make_output(final_out->get_outputs());
    fuse_arg_ops.insert(fuse_arg_ops.begin(), out);

    mgr.attrs_.set(sc_graph_t::attr_key_t::is_input_plain,
            c_block == 1 && !channel_last);
    mgr.attrs_.set(sc_graph_t::attr_key_t::is_output_plain,
            c_block == 1 && !channel_last);

    int P, Q;
    auto pool = pooling_out->dyn_cast<pooling_op_t>();
    auto output_dims = pool->info_.outputs_[0]->details_.get_plain_dims();
    P = output_dims[channel_last ? 1 : 2];
    Q = output_dims[channel_last ? 2 : 3];

    // check auto pad
    const sc_dims &pads_begin = pool->pads_begin_;
    const sc_dims &pads_end = pool->pads_end_;
    if (auto_pad == auto_pad_options::valid) {
        EXPECT_TRUE(std::all_of(pads_begin.begin(), pads_begin.end(),
                [](int64_t x) { return x == 0; }));
        EXPECT_TRUE(std::all_of(pads_end.begin(), pads_end.end(),
                [](int64_t x) { return x == 0; }));
    } else if (auto_pad == auto_pad_options::same_upper
            || auto_pad == auto_pad_options::same_lower) {
        EXPECT_EQ(P, H);
        EXPECT_EQ(Q, W);
        if (auto_pad == auto_pad_options::same_upper)
            EXPECT_TRUE(std::equal(pads_begin.begin(), pads_begin.end(),
                    pads_end.begin(),
                    [](int64_t x, int64_t y) { return x <= y; }));
        else
            EXPECT_TRUE(std::equal(pads_begin.begin(), pads_begin.end(),
                    pads_end.begin(),
                    [](int64_t x, int64_t y) { return x >= y; }));
    }

    // check round type
    auto round_func = [&round_floor](double x) {
        return round_floor ? std::floor(x) : std::ceil(x);
    };
    int expected_P = round_func(
            1.0 * (H + pads_begin[0] + pads_end[0] - R) / stride_h + 1);
    int expected_Q = round_func(
            1.0 * (W + pads_begin[1] + pads_end[1] - S) / stride_w + 1);
    EXPECT_EQ(P, expected_P);
    EXPECT_EQ(Q, expected_Q);

    auto output = alloc_array<Store_type>(N * C * P * Q);
    auto input = alloc_array<Store_type>(N * C * H * W);
    auto ref_output = alloc_array<Store_type>(N * C * P * Q);
    auto bn_mul = alloc_array<float>(C);
    auto bn_add = alloc_array<float>(C);

    auto ctx = std::make_shared<context_t>(*get_test_ctx());
    ctx->flags_.use_cost_model_ = true;
    graph_driver(mgr, ctx);

    auto f = lower_graph(ctx, mgr, fuse_arg_ops);
    auto fptr = jit_engine_t::make(ctx)->get_entry_func(f);

    if (add_bn_relu) {
        fptr->call_default(&output[0], &input[0], &bn_mul[0], &bn_add[0]);
    } else {
        fptr->call_default(&output[0], &input[0]);
    }

    auto ref_input = NCHWc2NCHW(input, N, C / c_block, H, W, c_block);
    auto sc_output = NCHWc2NCHW(output, N, C / c_block, P, Q, c_block);
    if (channel_last) {
        ref_input = NHWC2NCHW(input, N, H, W, C);
        sc_output = NHWC2NCHW(output, N, P, Q, C);
    }

    auto ref_mul = std::move(bn_mul);
    auto ref_add = std::move(bn_add);
    std::string pool_type_str;
    if (pool_type == pooling_type_t::max) {
        pool_type_str = "max";
    } else if (pool_type == pooling_type_t::avg) {
        pool_type_str = "avg";
    }
    compute_pooling_ref_fwd<Store_type, Compute_type>(pool_type_str, N, C, H, W,
            P, Q, R, S, stride_h, stride_w, pads_begin[0], pads_begin[1],
            &ref_input[0], &ref_output[0], &ref_mul[0], &ref_add[0],
            add_bn_relu, exclude_pad);
    test_utils::compare_data(
            sc_output.data(), ref_output.data(), sc_output.size());
}

// maxpool
TEST(GCCore_CPU_fusible_pooling, Test_max1) {
    check_fusible_pooling_fwd<float>(
            pooling_type_t::max, 56, 256, 64, 56, 1, 1, {1, 1}, {0, 0}, false);
}
TEST(GCCore_CPU_fusible_pooling, Test_max2) {
    check_fusible_pooling_fwd<float>(
            pooling_type_t::max, 56, 64, 56, 56, 3, 3, {1, 1}, {0, 0}, false);
}
TEST(GCCore_CPU_fusible_pooling, Test_max3) {
    check_fusible_pooling_fwd<float>(
            pooling_type_t::max, 56, 64, 56, 56, 3, 3, {2, 2}, {0, 0}, false);
}
TEST(GCCore_CPU_fusible_pooling, Test_max4) {
    check_fusible_pooling_fwd<float>(
            pooling_type_t::max, 56, 64, 56, 56, 3, 3, {1, 1}, {1, 1}, false);
}
TEST(GCCore_CPU_fusible_pooling, Test_max5) {
    check_fusible_pooling_fwd<float>(
            pooling_type_t::max, 56, 64, 56, 56, 3, 3, {2, 2}, {1, 1}, false);
}

// avgpool
TEST(GCCore_CPU_fusible_pooling, Test_avg1) {
    check_fusible_pooling_fwd<float>(
            pooling_type_t::avg, 56, 256, 64, 56, 1, 1, {1, 1}, {0, 0}, false);
}
TEST(GCCore_CPU_fusible_pooling, Test_avg2) {
    check_fusible_pooling_fwd<float>(
            pooling_type_t::avg, 56, 64, 56, 56, 3, 3, {1, 1}, {0, 0}, false);
}
TEST(GCCore_CPU_fusible_pooling, Test_avg3) {
    check_fusible_pooling_fwd<float>(
            pooling_type_t::avg, 56, 64, 56, 56, 3, 3, {2, 2}, {0, 0}, false);
}
TEST(GCCore_CPU_fusible_pooling, Test_avg4) {
    check_fusible_pooling_fwd<float>(
            pooling_type_t::avg, 56, 64, 56, 56, 3, 3, {1, 1}, {1, 1}, false);
}
TEST(GCCore_CPU_fusible_pooling, Test_avg5) {
    check_fusible_pooling_fwd<float>(
            pooling_type_t::avg, 56, 64, 56, 56, 3, 3, {2, 2}, {1, 1}, false);
}

// avgpool exclude pads
TEST(GCCore_CPU_fusible_pooling, Test_avg_exclude_pad1) {
    check_fusible_pooling_fwd<float>(pooling_type_t::avg, 56, 256, 64, 56, 1, 1,
            {1, 1}, {0, 0}, false, 1, true);
}
TEST(GCCore_CPU_fusible_pooling, Test_avg_exclude_pad2) {
    check_fusible_pooling_fwd<float>(pooling_type_t::avg, 56, 64, 56, 56, 3, 3,
            {1, 1}, {0, 0}, false, 1, true);
}
TEST(GCCore_CPU_fusible_pooling, Test_avg_exclude_pad3) {
    check_fusible_pooling_fwd<float>(pooling_type_t::avg, 56, 64, 56, 56, 3, 3,
            {2, 2}, {0, 0}, false, 1, true);
}
TEST(GCCore_CPU_fusible_pooling, Test_avg_exclude_pad4) {
    check_fusible_pooling_fwd<float>(pooling_type_t::avg, 56, 64, 56, 56, 3, 3,
            {1, 1}, {1, 1}, false, 1, true);
}
TEST(GCCore_CPU_fusible_pooling, Test_avg_exclude_pad5) {
    check_fusible_pooling_fwd<float>(pooling_type_t::avg, 56, 64, 56, 56, 3, 3,
            {1, 1}, {1, 1}, false, 1, true);
}

// maxpool c_blocking
TEST(GCCore_CPU_fusible_pooling, Test_blocking_max1) {
    check_fusible_pooling_fwd<float>(pooling_type_t::max, 56, 256, 64, 56, 1, 1,
            {1, 1}, {0, 0}, false, 64);
}
TEST(GCCore_CPU_fusible_pooling, Test_blocking_max2) {
    check_fusible_pooling_fwd<float>(pooling_type_t::max, 56, 64, 56, 56, 3, 3,
            {1, 1}, {0, 0}, false, 16);
}
TEST(GCCore_CPU_fusible_pooling, Test_blocking_max3) {
    check_fusible_pooling_fwd<float>(pooling_type_t::max, 56, 64, 56, 56, 3, 3,
            {2, 2}, {0, 0}, false, 64);
}
TEST(GCCore_CPU_fusible_pooling, Test_blocking_max4) {
    check_fusible_pooling_fwd<float>(pooling_type_t::max, 56, 64, 56, 56, 3, 3,
            {1, 1}, {1, 1}, false, 64);
}
TEST(GCCore_CPU_fusible_pooling, Test_blocking_max5) {
    check_fusible_pooling_fwd<float>(pooling_type_t::max, 56, 64, 56, 56, 3, 3,
            {2, 2}, {1, 1}, false, 16);
}

// avgpool c_blocking
TEST(GCCore_CPU_fusible_pooling, Test_blocking_avg1) {
    check_fusible_pooling_fwd<float>(pooling_type_t::avg, 56, 256, 64, 56, 1, 1,
            {1, 1}, {0, 0}, false, 16);
}
TEST(GCCore_CPU_fusible_pooling, Test_blocking_avg2) {
    check_fusible_pooling_fwd<float>(pooling_type_t::avg, 56, 64, 56, 56, 3, 3,
            {1, 1}, {0, 0}, false, 8);
}
TEST(GCCore_CPU_fusible_pooling, Test_blocking_avg3) {
    check_fusible_pooling_fwd<float>(pooling_type_t::avg, 56, 64, 56, 56, 3, 3,
            {2, 2}, {0, 0}, false, 16);
}
TEST(GCCore_CPU_fusible_pooling, Test_blocking_avg4) {
    check_fusible_pooling_fwd<float>(pooling_type_t::avg, 56, 64, 56, 56, 3, 3,
            {1, 1}, {1, 1}, false, 16);
}
TEST(GCCore_CPU_fusible_pooling, Test_blocking_avg5) {
    check_fusible_pooling_fwd<float>(pooling_type_t::avg, 56, 64, 56, 56, 3, 3,
            {2, 2}, {1, 1}, false, 8);
}

// asymmetric
TEST(GCCore_CPU_fusible_pooling, Test_asymmetric1) {
    check_fusible_pooling_fwd<float>(
            pooling_type_t::max, 56, 64, 56, 56, 3, 3, {1, 2}, {2, 1}, false);
}

TEST(GCCore_CPU_fusible_pooling, Test_asymmetric2) {
    check_fusible_pooling_fwd<float>(
            pooling_type_t::max, 56, 64, 56, 56, 3, 2, {1, 2}, {2, 1}, false);
}
TEST(GCCore_CPU_fusible_pooling, Test_asymmetric3) {
    check_fusible_pooling_fwd<float>(
            pooling_type_t::max, 56, 64, 56, 56, 3, 1, {1, 2}, {2, 1}, false);
}
TEST(GCCore_CPU_fusible_pooling, Test_asymmetric4) {
    check_fusible_pooling_fwd<float>(
            pooling_type_t::avg, 56, 64, 56, 56, 3, 3, {1, 2}, {2, 1}, false);
}

TEST(GCCore_CPU_fusible_pooling, Test_asymmetric5) {
    check_fusible_pooling_fwd<float>(
            pooling_type_t::avg, 56, 64, 56, 56, 3, 2, {1, 2}, {2, 1}, false);
}
TEST(GCCore_CPU_fusible_pooling, Test_asymmetric6) {
    check_fusible_pooling_fwd<float>(
            pooling_type_t::avg, 56, 64, 56, 56, 3, 1, {1, 2}, {2, 1}, false);
}

// bn_relu_fuse
TEST(GCCore_CPU_fusible_pooling, Test_bn_relu1) {
    check_fusible_pooling_fwd<float>(
            pooling_type_t::avg, 56, 256, 64, 56, 1, 1, {1, 1}, {0, 0}, true);
}
TEST(GCCore_CPU_fusible_pooling, Test_bn_relu2) {
    check_fusible_pooling_fwd<float>(
            pooling_type_t::avg, 56, 64, 56, 56, 3, 3, {1, 1}, {0, 0}, true);
}
TEST(GCCore_CPU_fusible_pooling, Test_bn_relu3) {
    check_fusible_pooling_fwd<float>(
            pooling_type_t::avg, 56, 64, 56, 56, 3, 3, {2, 2}, {0, 0}, true);
}
TEST(GCCore_CPU_fusible_pooling, Test_bn_relu4) {
    check_fusible_pooling_fwd<float>(
            pooling_type_t::avg, 56, 64, 56, 56, 3, 3, {1, 1}, {1, 1}, true);
}
TEST(GCCore_CPU_fusible_pooling, Test_bn_relu5) {
    check_fusible_pooling_fwd<float>(
            pooling_type_t::avg, 56, 64, 56, 56, 3, 3, {2, 2}, {1, 1}, true);
}
TEST(GCCore_CPU_fusible_pooling, Test_bn_relu6) {
    check_fusible_pooling_fwd<float>(
            pooling_type_t::avg, 56, 64, 56, 56, 3, 1, {1, 2}, {2, 1}, true);
}

// auto pad
TEST(GCCore_CPU_fusible_pooling, Test_auto_pad_valid) {
    check_fusible_pooling_fwd<float>(pooling_type_t::avg, 56, 64, 56, 56, 3, 3,
            {1, 1}, {1, 1}, false, 1, false, true, auto_pad_options::valid);
}

TEST(GCCore_CPU_fusible_pooling, Test_auto_pad_same_upper) {
    check_fusible_pooling_fwd<float>(pooling_type_t::avg, 56, 64, 56, 56, 2, 2,
            {1, 1}, {1, 1}, false, 1, false, true,
            auto_pad_options::same_upper);
}

TEST(GCCore_CPU_fusible_pooling, Test_auto_pad_same_lower) {
    check_fusible_pooling_fwd<float>(pooling_type_t::avg, 56, 64, 56, 56, 2, 2,
            {1, 1}, {1, 1}, false, 1, false, true,
            auto_pad_options::same_lower);
}

// rounding type
TEST(GCCore_CPU_fusible_pooling, Test_rounding_type_floor) {
    check_fusible_pooling_fwd<float>(pooling_type_t::avg, 56, 64, 224, 224, 3,
            3, {2, 2}, {0, 0}, false, 1, false, true);
}

TEST(GCCore_CPU_fusible_pooling, Test_rounding_type_ceil) {
    check_fusible_pooling_fwd<float>(pooling_type_t::avg, 56, 64, 224, 224, 3,
            3, {2, 2}, {0, 0}, false, 1, false, false);
}

// bf16 maxpool
TEST(GCCore_CPU_fusible_pooling, Test_bf16_max1) {
    check_fusible_pooling_fwd<bf16_t, float>(pooling_type_t::max, 56, 256, 64,
            56, 1, 1, {1, 1}, {0, 0}, false, 1);
}
TEST(GCCore_CPU_fusible_pooling, Test_bf16_max2) {
    check_fusible_pooling_fwd<bf16_t, float>(pooling_type_t::max, 56, 64, 56,
            56, 3, 3, {1, 1}, {0, 0}, false, 1);
}
TEST(GCCore_CPU_fusible_pooling, Test_bf16_max3) {
    check_fusible_pooling_fwd<bf16_t, float>(pooling_type_t::max, 56, 64, 56,
            56, 3, 3, {2, 2}, {0, 0}, false, 1);
}
TEST(GCCore_CPU_fusible_pooling, Test_bf16_max4) {
    check_fusible_pooling_fwd<bf16_t, float>(pooling_type_t::max, 56, 64, 56,
            56, 3, 3, {1, 1}, {1, 1}, false, 1);
}
TEST(GCCore_CPU_fusible_pooling, Test_bf16_max5) {
    check_fusible_pooling_fwd<bf16_t, float>(pooling_type_t::max, 56, 64, 56,
            56, 3, 3, {2, 2}, {1, 1}, false, 1);
}

// NXC
TEST(GCCore_CPU_fusible_pooling, Test_NXC_max) {
    check_fusible_pooling_fwd<float>(pooling_type_t::max, 64, 32, 56, 56, 3, 3,
            {2, 2}, {1, 1}, false, 1, false, true, auto_pad_options::none,
            true);
}

TEST(GCCore_CPU_fusible_pooling, Test_NXC_avg) {
    check_fusible_pooling_fwd<float>(pooling_type_t::avg, 64, 32, 56, 56, 3, 3,
            {2, 2}, {1, 1}, false, 1, false, true, auto_pad_options::none,
            true);
}

// bf16 avgpool
TEST(GCCore_CPU_fusible_pooling, Test_bf16_avg1) {
    check_fusible_pooling_fwd<bf16_t, float>(pooling_type_t::avg, 56, 256, 64,
            56, 1, 1, {1, 1}, {0, 0}, false, 1);
}
TEST(GCCore_CPU_fusible_pooling, Test_bf16_avg2) {
    check_fusible_pooling_fwd<bf16_t, float>(pooling_type_t::avg, 56, 64, 56,
            56, 3, 3, {1, 1}, {0, 0}, false, 1);
}
TEST(GCCore_CPU_fusible_pooling, Test_bf16_avg3) {
    check_fusible_pooling_fwd<bf16_t, float>(pooling_type_t::avg, 56, 64, 56,
            56, 3, 3, {2, 2}, {0, 0}, false, 1);
}
TEST(GCCore_CPU_fusible_pooling, Test_bf16_avg4) {
    check_fusible_pooling_fwd<bf16_t, float>(pooling_type_t::avg, 56, 64, 56,
            56, 3, 3, {1, 1}, {1, 1}, false, 1);
}
TEST(GCCore_CPU_fusible_pooling, Test_bf16_avg5) {
    check_fusible_pooling_fwd<bf16_t, float>(pooling_type_t::avg, 56, 64, 56,
            56, 3, 3, {2, 2}, {1, 1}, false, 1);
}

// int8 maxpool
TEST(GCCore_CPU_fusible_pooling, Test_int8_max1) {
    check_fusible_pooling_fwd<int8_t>(pooling_type_t::max, 56, 256, 64, 56, 1,
            1, {1, 1}, {0, 0}, false, 1);
}
TEST(GCCore_CPU_fusible_pooling, Test_int8_max2) {
    check_fusible_pooling_fwd<int8_t>(pooling_type_t::max, 56, 64, 56, 56, 3, 3,
            {1, 1}, {0, 0}, false, 1);
}
TEST(GCCore_CPU_fusible_pooling, Test_int8_max3) {
    check_fusible_pooling_fwd<int8_t>(pooling_type_t::max, 56, 64, 56, 56, 3, 3,
            {2, 2}, {0, 0}, false, 1);
}
TEST(GCCore_CPU_fusible_pooling, Test_int8_max4) {
    check_fusible_pooling_fwd<int8_t>(pooling_type_t::max, 56, 64, 56, 56, 3, 3,
            {1, 1}, {1, 1}, false, 1);
}
TEST(GCCore_CPU_fusible_pooling, Test_int8_max5) {
    check_fusible_pooling_fwd<int8_t>(pooling_type_t::max, 56, 64, 56, 56, 3, 3,
            {2, 2}, {1, 1}, false, 1);
}

// int8 avgpool
TEST(GCCore_CPU_fusible_pooling, Test_int8_avg1) {
    check_fusible_pooling_fwd<int8_t, int32_t>(pooling_type_t::avg, 56, 256, 64,
            56, 1, 1, {1, 1}, {0, 0}, false, 1);
}
TEST(GCCore_CPU_fusible_pooling, Test_int8_avg2) {
    check_fusible_pooling_fwd<int8_t, int32_t>(pooling_type_t::avg, 56, 64, 56,
            56, 3, 3, {1, 1}, {0, 0}, false, 1);
}
TEST(GCCore_CPU_fusible_pooling, Test_int8_avg3) {
    check_fusible_pooling_fwd<int8_t, int32_t>(pooling_type_t::avg, 56, 64, 56,
            56, 3, 3, {2, 2}, {0, 0}, false, 1);
}
TEST(GCCore_CPU_fusible_pooling, Test_int8_avg4) {
    check_fusible_pooling_fwd<int8_t, int32_t>(pooling_type_t::avg, 56, 64, 56,
            56, 3, 3, {1, 1}, {1, 1}, false, 1);
}
TEST(GCCore_CPU_fusible_pooling, Test_int8_avg5) {
    check_fusible_pooling_fwd<int8_t, int32_t>(pooling_type_t::avg, 56, 64, 56,
            56, 3, 3, {2, 2}, {1, 1}, false, 1);
}

// test conv fusion
template <typename src_type, typename wei_type, typename dst_type>
static sc_graph_t make_conv_pooling_postops_graph(const int64_t N,
        const int64_t K, const int64_t C, const int64_t H, const int64_t W,
        const int64_t R, const int64_t S, const int64_t SH, const int64_t SW,
        const int64_t PH, const int64_t PW, pooling_type_t pool_type,
        const int64_t p_P, const int64_t p_Q, const int64_t p_SH,
        const int64_t p_SW, const int64_t p_PH, const int64_t p_PW,
        std::vector<sc_op_ptr> &fuse_arg_ops, bool exclude_pad = false,
        bool bn_relu = false,
        const std::string &auto_pad = auto_pad_options::none,
        const int64_t c_block = 1, const bool export_conv = true) {
    sc_graph_t g;
    auto in_fmt = c_block == 1 ? sc_data_format_t::NCHW()
                               : sc_data_format_t::NCHWc(c_block);
    sc_data_type_t src_dtype = datatypes::f32;
    if (std::is_same<src_type, bf16_t>::value) src_dtype = datatypes::bf16;
    sc_data_type_t weight_dtype = datatypes::f32;
    if (std::is_same<wei_type, bf16_t>::value) weight_dtype = datatypes::bf16;
    sc_op_ptr data_input = g.make_input(
            {graph_tensor::make({N, C, H, W}, in_fmt, src_dtype)});
    sc_op_ptr weight_input = g.make_input(
            {graph_tensor::make({K, C, R, S}, in_fmt, weight_dtype)});
    sc_dims paddings = {PH, PW};
    sc_op_ptr conv_op = g.make("conv_fwd_core",
            {data_input->get_outputs()[0], weight_input->get_outputs()[0]}, {},
            {{"strides", sc_dims {SH, SW}}, {"pads_begin", paddings},
                    {"pads_end", paddings}, {"data_format", "NCX"},
                    {"weights_format", "OIX"}});
    fuse_arg_ops = {data_input, weight_input};
    if (export_conv) {
        sc_op_ptr conv_out = g.make_output(conv_op->get_outputs());
        fuse_arg_ops.insert(fuse_arg_ops.begin(), conv_out);
    }
    auto pooling_op_type_str
            = pool_type == pooling_type_t::max ? "pooling_max" : "pooling_avg";
    sc_op_ptr pooling_op = g.make(pooling_op_type_str,
            {conv_op->get_outputs()[0]}, {},
            {{pooling_attr_key::strides, sc_dims {p_SH, p_SW}},
                    {pooling_attr_key::paddings, sc_dims {p_PH, p_PW}},
                    {pooling_attr_key::kernel, sc_dims {p_P, p_Q}},
                    {pooling_attr_key::exclude_pad, exclude_pad},
                    {pooling_attr_key::auto_pad, auto_pad},
                    {pooling_attr_key::data_format, data_format_options::NCX}});
    sc_op_ptr final_out = pooling_op;
    auto bc_axis = std::vector<int> {1};
    if (bn_relu) {
        sc_op_ptr bn_mul = g.make_input({graph_tensor::make({K})});
        sc_op_ptr bn_add = g.make_input({graph_tensor::make({K})});
        final_out = g.make("mul",
                {final_out->get_outputs()[0], bn_mul->get_outputs()[0]}, {},
                {{"bc_axis", bc_axis}});

        final_out = g.make("add",
                {final_out->get_outputs()[0], bn_add->get_outputs()[0]}, {},
                {{"bc_axis", bc_axis}});
        final_out = g.make("relu", {final_out->get_outputs()[0]}, {}, {});
        fuse_arg_ops.emplace_back(bn_mul);
        fuse_arg_ops.emplace_back(bn_add);
    }
    sc_op_ptr out = g.make_output(final_out->get_outputs());
    fuse_arg_ops.insert(fuse_arg_ops.begin(), out);

    g.attrs_.set(sc_graph_t::attr_key_t::is_input_plain, c_block == 1);
    g.attrs_.set(sc_graph_t::attr_key_t::is_output_plain, c_block == 1);

    return g;
}

void compute_conv_pooling_outshape(const int64_t MB, const int64_t OC,
        const int64_t IC, const int64_t IH, const int64_t IW, const int64_t KH,
        const int64_t KW, const int64_t SH, const int64_t SW, const int64_t PH,
        const int64_t PW, pooling_type_t pooling_type, const int64_t p_KH,
        const int64_t p_KW, const int64_t p_SH, const int64_t p_SW,
        int64_t &p_PH, int64_t &p_PW, const bool exclude_pad,
        const std::string &auto_pad, int64_t &conv_p, int64_t &conv_q,
        int64_t &pool_p, int64_t &pool_q) {
    conv_p = (IH + PH * 2 - KH) / SH + 1;
    conv_q = (IW + PW * 2 - KW) / SW + 1;
    if (auto_pad == auto_pad_options::same_upper
            || auto_pad == auto_pad_options::same_lower) {
        pool_q = (conv_q + p_SH - 1) / p_SH;
        pool_p = (conv_p + p_SW - 1) / p_SW;
        int64_t total_pad_h
                = std::max((pool_p - 1) * p_SH + p_KH - conv_p, int64_t(0));
        int64_t total_pad_w
                = std::max((pool_q - 1) * p_SW + p_KW - conv_q, int64_t(0));
        if (auto_pad == auto_pad_options::same_upper) {
            p_PH = total_pad_h / 2;
            p_PW = total_pad_w / 2;
        } else if (auto_pad == auto_pad_options::same_lower) {
            p_PH = (total_pad_h + 1) / 2;
            p_PW = (total_pad_w + 1) / 2;
        }
    } else if (auto_pad == auto_pad_options::valid) {
        pool_p = (conv_p - p_KH) / p_SH + 1;
        pool_q = (conv_q - p_KW) / p_SW + 1;
        p_PH = 0;
        p_PW = 0;
    } else {
        pool_p = (conv_p + p_PH * 2 - p_KH) / p_SH + 1;
        pool_q = (conv_q + p_PW * 2 - p_KW) / p_SW + 1;
    }
}

template <typename src_type, typename wei_type, typename dst_type>
static sc_graph_t make_dyn_conv_pooling_postops_graph(const int64_t N,
        const int64_t K, const int64_t C, const int64_t H, const int64_t W,
        const int64_t R, const int64_t S, const int64_t SH, const int64_t SW,
        const int64_t PH, const int64_t PW, pooling_type_t pool_type,
        const int64_t p_P, const int64_t p_Q, const int64_t p_SH,
        const int64_t p_SW, const int64_t p_PH, const int64_t p_PW,
        std::vector<sc_op_ptr> &fuse_arg_ops, nested_conv_fwd_config_t &cfg,
        bool exclude_pad = false, bool bn_relu = false,
        const std::string &auto_pad = auto_pad_options::none) {
    sc_graph_t g;
    sc_data_type_t src_dtype = datatypes::f32;
    if (std::is_same<src_type, bf16_t>::value) src_dtype = datatypes::bf16;
    sc_data_type_t weight_dtype = datatypes::f32;
    if (std::is_same<wei_type, bf16_t>::value) weight_dtype = datatypes::bf16;
    sc_op_ptr data_input = g.make_input(
            {graph_tensor::make({N, C, H, W}, sc_data_format_t(), src_dtype)});
    sc_op_ptr weight_input = g.make_input({graph_tensor::make(
            {K, C, R, S}, sc_data_format_t(), weight_dtype)});
    sc_dims paddings = {PH, PW};
    sc_op_ptr conv_op = g.make("conv_fwd_core",
            {data_input->get_outputs()[0], weight_input->get_outputs()[0]}, {},
            {{"strides", sc_dims {SH, SW}}, {"pads_begin", paddings},
                    {"pads_end", paddings}});
    conv_op->attrs_.set<std::string>("temp.test_format", "NHWC");
    auto tunop = conv_op->dyn_cast<tunable_op_t>();
    reflection::shared_general_object_t cfgptr;
    cfgptr = tunop->create_generator()->get_default_config(
            get_default_context());
    cfg = *(nested_conv_fwd_config_t *)cfgptr.get();
    tunop->set_config(cfgptr);
    tunop->get_inputs()[0]->details_.set_format(sc_data_format_t::NHWC());
    tunop->get_inputs()[1]->details_.set_format(
            sc_data_format_t::KCRSck(cfg.im_ic_block, cfg.im_oc_block));
    tunop->get_outputs()[0]->details_.set_format(sc_data_format_t::NHWC());

    fuse_arg_ops = {data_input, weight_input};
    auto pooling_op_type_str
            = pool_type == pooling_type_t::max ? "pooling_max" : "pooling_avg";
    sc_op_ptr pooling_op = g.make(pooling_op_type_str,
            {conv_op->get_outputs()[0]}, {},
            {{pooling_attr_key::strides, sc_dims {p_SH, p_SW}},
                    {pooling_attr_key::paddings, sc_dims {p_PH, p_PW}},
                    {pooling_attr_key::kernel, sc_dims {p_P, p_Q}},
                    {pooling_attr_key::exclude_pad, exclude_pad},
                    {pooling_attr_key::auto_pad, auto_pad},
                    {pooling_attr_key::data_format, data_format_options::NCX}});
    sc_op_ptr final_out = pooling_op;
    auto bc_axis = std::vector<int> {1};
    if (bn_relu) {
        sc_op_ptr bn_mul = g.make_input({graph_tensor::make({K})});
        sc_op_ptr bn_add = g.make_input({graph_tensor::make({K})});
        final_out = g.make("mul",
                {final_out->get_outputs()[0], bn_mul->get_outputs()[0]}, {},
                {{"bc_axis", bc_axis}});

        final_out = g.make("add",
                {final_out->get_outputs()[0], bn_add->get_outputs()[0]}, {},
                {{"bc_axis", bc_axis}});
        final_out = g.make("relu", {final_out->get_outputs()[0]}, {}, {});
        fuse_arg_ops.emplace_back(bn_mul);
        fuse_arg_ops.emplace_back(bn_add);
    }
    sc_op_ptr out = g.make_output(final_out->get_outputs());
    fuse_arg_ops.insert(fuse_arg_ops.begin(), out);

    g.attrs_.set(sc_graph_t::attr_key_t::is_input_plain, false);
    g.attrs_.set(sc_graph_t::attr_key_t::is_output_plain, false);

    return g;
}

template <typename src_type, typename wei_type, typename dst_type>
static void check_dyn_conv_pooling_postops_graph(
        const std::string &expected_fusion, const int64_t MB, const int64_t OC,
        const int64_t IC, const int64_t IH, const int64_t IW, const int64_t KH,
        const int64_t KW, const int64_t SH, const int64_t SW, const int64_t PH,
        const int64_t PW, pooling_type_t pooling_type, const int64_t p_KH,
        const int64_t p_KW, const int64_t p_SH, const int64_t p_SW,
        int64_t p_PH, int64_t p_PW, const int64_t REAL_MB,
        const int64_t REAL_IH, const int64_t REAL_IW,
        const std::string auto_pad = auto_pad_options::none,
        const bool exclude_pad = false, const bool bn_relu = true) {
    REQUIRE_AVX2();
    int64_t conv_p, conv_q, pool_p, pool_q;

    compute_conv_pooling_outshape(REAL_MB, OC, IC, REAL_IH, REAL_IW, KH, KW, SH,
            SW, PH, PW, pooling_type, p_KH, p_KW, p_SH, p_SW, p_PH, p_PW,
            exclude_pad, auto_pad, conv_p, conv_q, pool_p, pool_q);
    // make graph
    std::vector<sc_op_ptr> fuse_arg_ops;
    nested_conv_fwd_config_t cfg;
    sc_graph_t g
            = make_dyn_conv_pooling_postops_graph<src_type, wei_type, dst_type>(
                    MB, OC, IC, IH, IW, KH, KW, SH, SW, PH, PW, pooling_type,
                    p_KH, p_KW, p_SH, p_SW, p_PH, p_PW, fuse_arg_ops, cfg,
                    exclude_pad, bn_relu, auto_pad);

    auto ctx = std::make_shared<context_t>(*get_default_context());
    // print_graph(g, std::cout, 1);
    graph_driver(g, ctx);
    // print_graph(g, std::cout, 1);
    // check graph
    auto has_expected_fusion = false;
    for (auto &op : g.ops_) {
        if (op->op_name_.find(expected_fusion) != std::string::npos) {
            has_expected_fusion = true;
            break;
        }
    }
    EXPECT_TRUE(has_expected_fusion);

    // compute sc
    uint8_t in_mask = 0;
    if (is_dynamic_dim(MB)) { in_mask |= 1 << 0; }
    if (is_dynamic_dim(IH)) { in_mask |= 1 << 1; }
    if (is_dynamic_dim(IW)) { in_mask |= 1 << 2; }
    auto sc_output
            = alloc_array<float>(REAL_MB * OC * pool_p * pool_q, INIT_NOOP);
    auto input = alloc_array<float>(REAL_MB * IC * REAL_IH * REAL_IW);
    auto weight = alloc_array<float>(OC * IC * KH * KW);
    auto bias = alloc_array<float>(OC);
    auto bn_mul = alloc_array<float>(OC);
    auto bn_add = alloc_array<float>(OC);

    sc_dims out_dims = sc_dims {REAL_MB, OC, pool_p, pool_q};
    sc_dims data_dims = sc_dims {REAL_MB, IC, REAL_IH, REAL_IW};
    sc_dims in_weight_dims = sc_dims {OC, IC, KH, KW};
    sc_dims in_postop_dims = sc_dims {OC};
    // Define dynamic tensor
    runtime::dynamic_tensor_t dyn_output(&sc_output[0], &out_dims[0],
            out_dims.size(), uint32_t(sc_data_etype::F32), 0);
    runtime::dynamic_tensor_t dyn_input(&input[0], &data_dims[0],
            data_dims.size(), uint32_t(sc_data_etype::F32), in_mask);
    runtime::dynamic_tensor_t dyn_weight(&weight[0], &in_weight_dims[0],
            in_weight_dims.size(), uint32_t(sc_data_etype::F32), 0);
    runtime::dynamic_tensor_t dyn_bn_mul(&bn_mul[0], &in_postop_dims[0],
            in_postop_dims.size(), uint32_t(sc_data_etype::F32), 0);
    runtime::dynamic_tensor_t dyn_bn_add(&bn_add[0], &in_postop_dims[0],
            in_postop_dims.size(), uint32_t(sc_data_etype::F32), 0);

    std::vector<void *> sc_args = {&dyn_output, &dyn_input, &dyn_weight};
    if (bn_relu) {
        sc_args.emplace_back(&dyn_bn_mul);
        sc_args.emplace_back(&dyn_bn_add);
    }
    std::vector<generic_val> generic_args;
    for (unsigned i = 0; i < sc_args.size(); i++)
        generic_args.emplace_back(sc_args.at(i));

    auto f = lower_graph(ctx, g, fuse_arg_ops);
    auto fptr = jit_engine_t::make(ctx)->get_entry_func(f);
    fptr->call_generic_default(generic_args.data());

    // compute ref
    auto ref_data = NHWC2NCHW(input, REAL_MB, REAL_IH, REAL_IW, IC);
    auto ref_weight = KCRSck2KCRS(weight, OC / cfg.im_oc_block,
            IC / cfg.im_ic_block, KH, KW, cfg.im_ic_block, cfg.im_oc_block);
    auto ref_bn_mul = std::move(bn_mul);
    auto ref_bn_add = std::move(bn_add);
    auto ref_output = alloc_array<dst_type>(REAL_MB * OC * pool_p * pool_q);
    auto ref_conv_output
            = alloc_array<dst_type>(REAL_MB * OC * conv_p * conv_q);

    std::string pooling_op_type_str
            = pooling_type == pooling_type_t::max ? "max" : "avg";
    compute_conv_pooling_postops_ref<src_type, wei_type, dst_type>(REAL_MB, OC,
            IC, REAL_IH, REAL_IW, conv_p, conv_q, KH, KW, SH, SW, PH, PW,
            &ref_data[0], &ref_weight[0], &ref_conv_output[0],
            pooling_op_type_str, pool_p, pool_q, p_KH, p_KW, p_SH, p_SW, p_PH,
            p_PW, &ref_output[0], exclude_pad, bn_relu, &ref_bn_mul[0],
            &ref_bn_add[0]);

    auto sc_output_plain = NHWC2NCHW(sc_output, REAL_MB, pool_p, pool_q, OC);
    test_utils::compare_data(sc_output_plain.data(), ref_output.data(),
            sc_output_plain.size(), 1e-3f, 1e-3f);
}

template <typename src_type, typename wei_type, typename dst_type>
static void check_conv_pooling_postops_graph(const std::string &expected_fusion,
        const int64_t MB, const int64_t c_block, const int64_t OC,
        const int64_t IC, const int64_t IH, const int64_t IW, const int64_t KH,
        const int64_t KW, const int64_t SH, const int64_t SW, const int64_t PH,
        const int64_t PW, pooling_type_t pooling_type, const int64_t p_KH,
        const int64_t p_KW, const int64_t p_SH, const int64_t p_SW,
        int64_t p_PH, int64_t p_PW, const bool exclude_pad = false,
        const bool bn_relu = false,
        const std::string auto_pad = auto_pad_options::none,
        const bool check_conv_out = false) {
    REQUIRE_AVX2();
    int64_t conv_p, conv_q, pool_p, pool_q;
    compute_conv_pooling_outshape(MB, OC, IC, IH, IW, KH, KW, SH, SW, PH, PW,
            pooling_type, p_KH, p_KW, p_SH, p_SW, p_PH, p_PW, exclude_pad,
            auto_pad, conv_p, conv_q, pool_p, pool_q);

    // compute ref
    auto ref_data = alloc_array<src_type>(MB * IC * IH * IW);
    auto ref_weight = alloc_array<wei_type>(IC * OC * KH * KW);
    auto ref_conv_output = alloc_array<dst_type>(MB * OC * conv_p * conv_q);
    auto ref_output = alloc_array<dst_type>(MB * OC * pool_p * pool_q);
    auto bn_mul = alloc_array<float>(OC);
    auto bn_add = alloc_array<float>(OC);
    std::string pooling_op_type_str
            = pooling_type == pooling_type_t::max ? "max" : "avg";
    compute_conv_pooling_postops_ref<src_type, wei_type, dst_type>(MB, OC, IC,
            IH, IW, conv_p, conv_q, KH, KW, SH, SW, PH, PW, &ref_data[0],
            &ref_weight[0], &ref_conv_output[0], pooling_op_type_str, pool_p,
            pool_q, p_KH, p_KW, p_SH, p_SW, p_PH, p_PW, &ref_output[0],
            exclude_pad, bn_relu, &bn_mul[0], &bn_add[0]);

    // make graph
    std::vector<sc_op_ptr> fuse_arg_ops;
    sc_graph_t g
            = make_conv_pooling_postops_graph<src_type, wei_type, dst_type>(MB,
                    OC, IC, IH, IW, KH, KW, SH, SW, PH, PW, pooling_type, p_KH,
                    p_KW, p_SH, p_SW, p_PH, p_PW, fuse_arg_ops, exclude_pad,
                    bn_relu, auto_pad, c_block, check_conv_out);
    auto ctx = std::make_shared<context_t>(*get_default_context());
    ctx->flags_.use_cost_model_ = true;
    graph_driver(g, ctx);

    // check graph
    auto has_expected_fusion = false;
    for (auto &op : g.ops_) {
        if (op->op_name_.find(expected_fusion) != std::string::npos) {
            has_expected_fusion = true;
            break;
        }
    }
    EXPECT_TRUE(has_expected_fusion);

    // compute sc
    auto sc_data = NCHW2NCHWc(ref_data, MB, IC / c_block, IH, IW, c_block);
    auto sc_weight = NCHW2NCHWc(ref_weight, OC, IC / c_block, KH, KW, c_block);
    auto sc_conv_output = alloc_array<dst_type>(MB * OC * conv_p * conv_q);
    auto sc_output = alloc_array<dst_type>(MB * OC * pool_p * pool_q);

    std::vector<generic_val> generic_args = {&sc_output[0]};
    if (check_conv_out) { generic_args.emplace_back(&sc_conv_output[0]); }
    generic_args.emplace_back(&sc_data[0]);
    generic_args.emplace_back(&sc_weight[0]);
    if (bn_relu) {
        generic_args.emplace_back(&bn_mul[0]);
        generic_args.emplace_back(&bn_add[0]);
    }
    auto f = lower_graph(ctx, g, fuse_arg_ops);
    auto fptr = jit_engine_t::make(ctx)->get_entry_func(f);
    fptr->call_generic_default(generic_args.data());
    auto sc_output_plain
            = NCHWc2NCHW(sc_output, MB, OC / c_block, pool_p, pool_q, c_block);

    // compare ref and sc
    if (check_conv_out) {
        auto sc_conv_output_plain = NCHWc2NCHW(
                sc_conv_output, MB, OC / c_block, conv_p, conv_q, c_block);
        test_utils::compare_data(sc_conv_output_plain.data(),
                ref_conv_output.data(), ref_conv_output.size(), 1e-3f, 1e-3f);
    }
    test_utils::compare_data(sc_output_plain.data(), ref_output.data(),
            sc_output_plain.size(), 1e-3f, 1e-3f);
}

TEST(GCCore_CPU_fusible_pooling, Test_fwd_f32_conv_padding_pooling_avg_postop) {
    SET_THREADS_OR_SKIP(56);
    std::string expected_fusion = "conv_fwd_core_pooling_avg_mul_add_relu";
    check_conv_pooling_postops_graph<float, float, float>(expected_fusion, 56,
            1, 128, 128, 56, 56, 3, 3, 1, 1, 1, 1, pooling_type_t::avg, 2, 2, 2,
            2, 0, 0, false, true, auto_pad_options::none, false);
}

TEST(GCCore_CPU_fusible_pooling, Test_fwd_f32_conv_pooling_avg_postop) {
    SET_THREADS_OR_SKIP(56);
    std::string expected_fusion = "conv_fwd_core_pooling_avg_mul_add_relu";
    check_conv_pooling_postops_graph<float, float, float>(expected_fusion, 56,
            1, 128, 128, 56, 56, 3, 3, 1, 1, 0, 0, pooling_type_t::avg, 2, 2, 2,
            2, 0, 0, false, true, auto_pad_options::none, false);
}

TEST(GCCore_CPU_fusible_pooling, Test_fwd_f32_conv_padding_pooling_max_postop) {
    SET_THREADS_OR_SKIP(56);
    std::string expected_fusion = "conv_fwd_core_pooling_max_mul_add_relu";
    check_conv_pooling_postops_graph<float, float, float>(expected_fusion, 56,
            1, 48, 128, 64, 64, 3, 3, 1, 1, 1, 1, pooling_type_t::max, 3, 3, 2,
            2, 1, 1, false, true, auto_pad_options::none, false);
}

TEST(GCCore_CPU_fusible_pooling, Test_fwd_f32_conv_pooling_max_postop) {
    SET_THREADS_OR_SKIP(56);
    std::string expected_fusion = "conv_fwd_core_pooling_max_mul_add_relu";
    check_conv_pooling_postops_graph<float, float, float>(expected_fusion, 56,
            1, 48, 128, 64, 64, 3, 3, 1, 1, 0, 0, pooling_type_t::max, 3, 3, 2,
            2, 1, 1, false, true, auto_pad_options::none, false);
}

TEST(GCCore_CPU_fusible_pooling,
        Test_fwd_f32_conv_pooling_max_fuse_pooling_avg) {
    SET_THREADS_OR_SKIP(1);
    int stride_h = 1, stride_w = 1;
    int padding_h = 1, padding_w = 1;
    int R = 3, S = 3;
    sc_data_type_t dtype = datatypes::f32;
    sc_graph_t mgr;
    auto in_fmt = sc_data_format_t::NCHW();
    sc_dims input_tensor_shape = {1, 256, 7, 7};
    sc_op_ptr in_tensor;
    std::vector<std::shared_ptr<graph_tensor>> inputs;
    in_tensor = mgr.make_input(
            {graph_tensor::make(input_tensor_shape, in_fmt, dtype)});
    inputs.emplace_back(in_tensor->get_outputs()[0]);

    auto pooling_out = mgr.make("pooling_max", inputs, {},
            {{pooling_attr_key::strides, sc_dims {stride_h, stride_w}},
                    {pooling_attr_key::paddings,
                            sc_dims {padding_h, padding_w}},
                    {pooling_attr_key::kernel, sc_dims {R, S}},
                    {pooling_attr_key::src_shape, input_tensor_shape},
                    {pooling_attr_key::exclude_pad, true},
                    {pooling_attr_key::data_format, "NCX"}});
    pooling_out = mgr.make("pooling_avg", pooling_out->get_outputs(), {},
            {{pooling_attr_key::strides, sc_dims {stride_h, stride_w}},
                    {pooling_attr_key::paddings,
                            sc_dims {padding_h, padding_w}},
                    {pooling_attr_key::kernel, sc_dims {R, S}},
                    {pooling_attr_key::src_shape, input_tensor_shape},
                    {pooling_attr_key::exclude_pad, true},
                    {pooling_attr_key::data_format, "NCX"}});

    auto out = mgr.make_output(pooling_out->get_outputs());
    auto ctx = std::make_shared<context_t>(*get_test_ctx());
    ctx->flags_.use_cost_model_ = true;
    graph_driver(mgr, ctx);
    mixed_fuse_op_t *fused_op = get_mixed_op_from_graph(mgr);
    auto body = fused_op->parti_list_[0]
                        ->get_outer_loops()
                        .back()
                        ->body_.checked_as<stmts>()
                        ->seq_[1]
                        .checked_as<stmts>();
    ir_simplifier_t simp {false};
    loop_merger_t lm;
    auto sim_body = simp(lm(body));
    EXPECT_EQ(sim_body.checked_as<stmts>()->seq_.size(), 2UL);
}

template <typename src_type>
static sc_graph_t make_conv_postops_pooling_graph_int8(const int64_t N,
        const int64_t K, const int64_t C, const int64_t H, const int64_t W,
        const int64_t R, const int64_t S, const int64_t SH, const int64_t SW,
        const int64_t PH, const int64_t PW, pooling_type_t pool_type,
        const int64_t p_P, const int64_t p_Q, const int64_t p_SH,
        const int64_t p_SW, const int64_t p_PH, const int64_t p_PW,
        std::vector<sc_op_ptr> &fuse_arg_ops, std::vector<any_map_t> qinfos,
        bool exclude_pad = false,
        const std::string &auto_pad = auto_pad_options::none,
        const int64_t c_block = 1) {
    static_assert(std::is_same<src_type, int8_t>::value
                    || std::is_same<src_type, uint8_t>::value,
            "src_type should be int8_t or uint8_t");
    sc_graph_t g;
    // input
    auto in_fmt = c_block == 1 ? sc_data_format_t::NCHW()
                               : sc_data_format_t::NCHWc(c_block);
    sc_data_type_t src_dtype = sc_data_traits_t<src_type>::type();
    sc_data_type_t weight_dtype = datatypes::s8;
    // input dequantize
    sc_op_ptr data_input = g.make_input(
            {graph_tensor::make({N, C, H, W}, in_fmt, src_dtype)});
    sc_op_ptr weight_input = g.make_input(
            {graph_tensor::make({K, C, R, S}, in_fmt, weight_dtype)});
    auto deq_data
            = g.make("dequantize", data_input->get_outputs(), {}, qinfos[0]);
    auto deq_weight
            = g.make("dequantize", weight_input->get_outputs(), {}, qinfos[1]);
    // conv relu quantize
    sc_dims paddings = {PH, PW};
    sc_op_ptr conv_op = g.make("conv_fwd_core",
            {deq_data->get_outputs()[0], deq_weight->get_outputs()[0]}, {},
            {{"strides", sc_dims {SH, SW}}, {"pads_begin", paddings},
                    {"pads_end", paddings}, {"data_format", "NCX"},
                    {"weights_format", "OIX"}});

    fuse_arg_ops = {data_input, weight_input};
    sc_op_ptr relu_out = g.make("relu", {conv_op->get_outputs()[0]}, {}, {});
    auto conv_relu_quan_out
            = g.make("quantize", relu_out->get_outputs(), {}, qinfos[2]);
    // dequantize pooling quantize
    auto deq_pool_in = g.make(
            "dequantize", conv_relu_quan_out->get_outputs(), {}, qinfos[3]);
    auto pooling_op_type_str
            = pool_type == pooling_type_t::max ? "pooling_max" : "pooling_avg";
    sc_op_ptr pooling_op = g.make(pooling_op_type_str,
            {deq_pool_in->get_outputs()[0]}, {},
            {{pooling_attr_key::strides, sc_dims {p_SH, p_SW}},
                    {pooling_attr_key::paddings, sc_dims {p_PH, p_PW}},
                    {pooling_attr_key::kernel, sc_dims {p_P, p_Q}},
                    {pooling_attr_key::exclude_pad, exclude_pad},
                    {pooling_attr_key::auto_pad, auto_pad},
                    {pooling_attr_key::data_format, data_format_options::NCX}});
    auto pooling_quan_out
            = g.make("quantize", pooling_op->get_outputs(), {}, qinfos[4]);
    const sc_op_ptr &final_out = pooling_quan_out;

    sc_op_ptr out = g.make_output(final_out->get_outputs());
    fuse_arg_ops.insert(fuse_arg_ops.begin(), out);

    g.attrs_.set(sc_graph_t::attr_key_t::is_input_plain, c_block == 1);
    g.attrs_.set(sc_graph_t::attr_key_t::is_output_plain, c_block == 1);

    return g;
}

static any_map_t get_qinfo(sc_data_type_t dtype,
        std::vector<float> scales = {1.f}, bool per_channel = false,
        int channel_axis = 0, std::vector<int> zero_points = {0},
        bool asymmetric = false) {
    return {
            {attr_keys::quan_dtype, dtype},
            {attr_keys::scales, scales},
            {attr_keys::per_channel, per_channel},
            {attr_keys::channel_axis, channel_axis},
            {attr_keys::zero_points, zero_points},
            {attr_keys::asymmetric, asymmetric},
    };
}

template <typename T, typename src_type>
static std::pair<float, int> get_scale_zero(
        T rmax, T rmin, bool asymmetric = false) {
    static_assert(std::is_same<src_type, int8_t>::value
                    || std::is_same<src_type, uint8_t>::value,
            "src_type should be int8_t or uint8_t");
    int zero_point = 0;
    float scale;
    if (asymmetric) {
        int qmax = 255;
        int qmin = 0;
        if (std::is_same<src_type, int8_t>::value) {
            qmax = 127;
            qmin = -128;
        }
        scale = (rmax - rmin) / static_cast<float>(qmax - qmin);
        zero_point = std::round(qmax - rmax / scale);
    } else {
        if (std::is_same<src_type, uint8_t>::value) {
            scale = std::max(rmax, 0.f) / 255;
        } else
            scale = std::max(std::abs(rmax), std::abs(rmin)) / 127;
    }

    return {scale, zero_point};
}

template <typename T, typename src_type>
static std::pair<float, int> get_list_scale_zero(
        T *begin, T *end, bool asymmetric = false) {
    static_assert(std::is_same<src_type, int8_t>::value
                    || std::is_same<src_type, uint8_t>::value,
            "src_type should be int8_t or uint8_t");
    T rmax = *std::max_element(begin, end);
    T rmin = *std::min_element(begin, end);
    return get_scale_zero<T, src_type>(rmax, rmin, asymmetric);
}

template <typename src_type, typename dst_type = src_type>
static void check_conv_postops_pooling_graph_int8(
        const std::string &expected_fusion, const int64_t MB,
        const int64_t c_block, const int64_t OC, const int64_t IC,
        const int64_t IH, const int64_t IW, const int64_t KH, const int64_t KW,
        const int64_t SH, const int64_t SW, const int64_t PH, const int64_t PW,
        pooling_type_t pooling_type, const int64_t p_KH, const int64_t p_KW,
        const int64_t p_SH, const int64_t p_SW, int64_t p_PH, int64_t p_PW,
        const bool exclude_pad = false,
        const std::string auto_pad = auto_pad_options::none,
        const bool check_conv_out = false) {
    REQUIRE_VNNI();
    int64_t conv_p, conv_q, pool_p, pool_q;
    compute_conv_pooling_outshape(MB, OC, IC, IH, IW, KH, KW, SH, SW, PH, PW,
            pooling_type, p_KH, p_KW, p_SH, p_SW, p_PH, p_PW, exclude_pad,
            auto_pad, conv_p, conv_q, pool_p, pool_q);

    // compute ref
    float data_min = -1;
    if (std::is_same<src_type, uint8_t>::value) {
        // seems conv not support asymmetric uint8 data input
        // so set ref_data all positive.
        data_min = 0;
    }
    auto ref_data
            = alloc_array<float>(MB * IC * IH * IW, INIT_RANGE, data_min, 1);
    auto ref_weight = alloc_array<float>(IC * OC * KH * KW);
    auto ref_conv_output = alloc_array<float>(MB * OC * conv_p * conv_q);
    auto ref_output = alloc_array<float>(MB * OC * pool_p * pool_q);
    auto bn_mul = alloc_array<float>(OC, INIT_RANGE, 1, 1);
    auto bn_add = alloc_array<float>(OC, INIT_ZERO);
    std::string pooling_op_type_str
            = pooling_type == pooling_type_t::max ? "max" : "avg";
    compute_conv_postops_pooling_ref<float, float, float>(MB, OC, IC, IH, IW,
            conv_p, conv_q, KH, KW, SH, SW, PH, PW, &ref_data[0],
            &ref_weight[0], &ref_conv_output[0], pooling_op_type_str, pool_p,
            pool_q, p_KH, p_KW, p_SH, p_SW, p_PH, p_PW, &ref_output[0],
            exclude_pad, true, &bn_mul[0], &bn_add[0]);

    // make graph
    bool input_asymmetric = false;
    bool output_asymmetric = false;
    auto sz_data = get_list_scale_zero<float, src_type>(
            ref_data.begin(), ref_data.end(), input_asymmetric);
    auto sz_conv_relu_out = get_list_scale_zero<float, dst_type>(
            ref_conv_output.begin(), ref_conv_output.end(), output_asymmetric);
    auto sz_pool_out = get_list_scale_zero<float, dst_type>(
            ref_output.begin(), ref_output.end(), output_asymmetric);
    std::vector<float> scale_weights(OC);
    int64_t kernel_size = IC * KH * KW;
    for (int i = 0; i < OC; i++) {
        scale_weights[i] = get_list_scale_zero<float, int8_t>(
                ref_weight.begin() + i * kernel_size,
                ref_weight.begin() + (i + 1) * kernel_size)
                                   .first;
    }
    auto qinfos = std::vector<any_map_t> {
            get_qinfo(datatypes::f32, {sz_data.first}, false, 0,
                    {sz_data.second}, input_asymmetric),
            get_qinfo(datatypes::f32, scale_weights, true, 0, {0},
                    false), // weights should be signed int8
            get_qinfo(sc_data_traits_t<dst_type>::type(),
                    {sz_conv_relu_out.first}, false, 0,
                    {sz_conv_relu_out.second}, output_asymmetric),
            get_qinfo(datatypes::f32, {sz_conv_relu_out.first}, false, 0,
                    {sz_conv_relu_out.second}, output_asymmetric),
            get_qinfo(sc_data_traits_t<dst_type>::type(), {sz_pool_out.first},
                    false, 0, {sz_pool_out.second}, output_asymmetric),
    };
    std::vector<sc_op_ptr> fuse_arg_ops;
    sc_graph_t g = make_conv_postops_pooling_graph_int8<src_type>(MB, OC, IC,
            IH, IW, KH, KW, SH, SW, PH, PW, pooling_type, p_KH, p_KW, p_SH,
            p_SW, p_PH, p_PW, fuse_arg_ops, qinfos, exclude_pad, auto_pad,
            c_block);
    auto ctx = std::make_shared<context_t>(*get_default_context());
    ctx->flags_.use_cost_model_ = true;
    graph_driver(g, ctx);
    // check graph
    auto has_expected_fusion = false;
    for (auto &op : g.ops_) {
        if (op->op_name_.find(expected_fusion) != std::string::npos) {
            has_expected_fusion = true;
            break;
        }
    }
    EXPECT_TRUE(has_expected_fusion);

    // compute sc
    auto sc_data_plain = alloc_array<src_type>(MB * IC * IH * IW, INIT_NOOP);
    auto sc_weight_plain = alloc_array<int8_t>(OC * IC * KH * KW, INIT_NOOP);
    for (size_t i = 0; i < ref_data.size(); i++) {
        sc_data_plain[i] = static_cast<src_type>(
                ref_data[i] / sz_data.first + sz_data.second);
    }
    for (int i = 0; i < OC; i++) {
        for (int j = 0; j < kernel_size; j++) {
            sc_weight_plain[i * kernel_size + j] = static_cast<int8_t>(
                    ref_weight[i * kernel_size + j] / scale_weights[i]);
        }
    }
    auto sc_data = NCHW2NCHWc(sc_data_plain, MB, IC / c_block, IH, IW, c_block);
    auto sc_weight
            = NCHW2NCHWc(sc_weight_plain, OC, IC / c_block, KH, KW, c_block);
    auto sc_output = alloc_array<dst_type>(MB * OC * pool_p * pool_q);

    std::vector<generic_val> generic_args = {&sc_output[0]};
    generic_args.emplace_back(&sc_data[0]);
    generic_args.emplace_back(&sc_weight[0]);
    auto f = lower_graph(ctx, g, fuse_arg_ops);
    auto fptr = jit_engine_t::make(ctx)->get_entry_func(f);
    fptr->call_generic_default(generic_args.data());
    auto sc_output_plain
            = NCHWc2NCHW(sc_output, MB, OC / c_block, pool_p, pool_q, c_block);
    auto sc_output_f32 = alloc_array<float>(MB * OC * pool_p * pool_q);
    for (int i = 0; i < MB; i++) {
        for (int j = 0; j < OC; j++) {
            for (int k = 0; k < pool_p * pool_q; k++) {
                int idx = i * OC * pool_p * pool_q + j * pool_p * pool_q + k;
                sc_output_f32[idx]
                        = static_cast<float>((int32_t)sc_output_plain[idx]
                                  - sz_pool_out.second)
                        * sz_pool_out.first;
            }
        }
    }

    // compare ref and sc
    auto ref_vector = std::vector<float>(ref_output.begin(), ref_output.end());
    auto sc_f32_vector
            = std::vector<float>(sc_output_f32.begin(), sc_output_f32.end());
    EXPECT_TRUE(test_utils::cal_rmse(ref_vector, sc_f32_vector) < 3);
}

TEST(GCCore_CPU_fusible_pooling,
        Test_fwd_quantized_conv_relu_max_pooling_uint8) {
    REQUIRE_AMX();
    SET_THREADS_OR_SKIP(56);
    std::string expected_fusion
            = "quantized_conv_fwd_core_cast_mul_relu_cast_pooling_max_cast_mul_"
              "cast";
    check_conv_postops_pooling_graph_int8<uint8_t>(expected_fusion, 56, 1, 64,
            128, 56, 56, 3, 3, 1, 1, 0, 0, pooling_type_t::max, 3, 3, 2, 2, 1,
            1, false, auto_pad_options::none);
}

TEST(GCCore_CPU_fusible_pooling,
        Test_fwd_quantized_conv_relu_avg_pooling_int8) {
    REQUIRE_AMX();
    SET_THREADS_OR_SKIP(56);
    std::string expected_fusion
            = "quantized_conv_fwd_core_cast_mul_relu_cast_pooling_avg_mul_cast";
    check_conv_postops_pooling_graph_int8<int8_t, uint8_t>(expected_fusion, 56,
            1, 64, 128, 56, 56, 3, 3, 1, 1, 0, 0, pooling_type_t::avg, 3, 3, 2,
            2, 1, 1, false, auto_pad_options::none);
}

TEST(GCCore_CPU_fusible_pooling,
        Test_fwd_quantized_conv_padding_relu_avg_pooling_uint8) {
    REQUIRE_AMX();
    SET_THREADS_OR_SKIP(56);
    std::string expected_fusion
            = "quantized_conv_fwd_core_cast_mul_relu_cast_pooling_avg_mul_cast";
    check_conv_postops_pooling_graph_int8<uint8_t>(expected_fusion, 56, 1, 64,
            128, 56, 56, 3, 3, 1, 1, 1, 1, pooling_type_t::avg, 3, 3, 2, 2, 1,
            1, false, auto_pad_options::none);
}

TEST(GCCore_CPU_fusible_pooling,
        Test_fwd_quantized_conv_padding_relu_max_pooling_int8) {
    REQUIRE_AMX();
    SET_THREADS_OR_SKIP(56);
    std::string expected_fusion
            = "quantized_conv_fwd_core_cast_mul_relu_cast_pooling_max_cast_mul_"
              "cast";
    check_conv_postops_pooling_graph_int8<int8_t, uint8_t>(expected_fusion, 56,
            1, 64, 128, 56, 56, 3, 3, 1, 1, 1, 1, pooling_type_t::max, 3, 3, 2,
            2, 1, 1, false, auto_pad_options::none);
}
