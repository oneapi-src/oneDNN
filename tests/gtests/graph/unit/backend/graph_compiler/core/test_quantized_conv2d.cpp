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

#include <iostream>
#include "context.hpp"
#include "reference/conv_ref.hpp"
#include "test_utils.hpp"
#include "gtest/gtest.h"
#include <compiler/ir/graph/driver.hpp>
#include <compiler/ir/graph/fusible_op.hpp>
#include <compiler/ir/graph/lowering.hpp>
#include <compiler/ir/graph/pass/pass.hpp>
#include <compiler/ir/graph/transform/transform.hpp>
#include <compiler/ir/sc_data_type.hpp>
#include <compiler/jit/jit.hpp>
#include <ops/convolution.hpp>
#include <ops/templates/conv_fwd.hpp>
#include <ops/templates/nested_conv_fwd.hpp>
#include <util/any_map.hpp>
#include <util/reflection.hpp>
#include <util/utils.hpp>

using namespace dnnl::impl::graph::gc;
using conv_fwd_config_t = ops::conv_fwd_config_t;
using nested_conv_fwd_config_t = ops::nested_conv_fwd_config_t;

const conv_fwd_config_t cfg_fwd = {
        16, // K_block
        16, // C_block
        1, // tile_d
        2, // tile_p
        2, // tile_q
        2, // tile_os
        0, // pack_input
        1 // loop_sched
};

graph_tensor_ptr make_tensor(const sc_dims &d, sc_data_type_t dtype) {
    return graph_tensor::make(d, sc_data_format_t(), dtype);
}

template <typename src_type, typename wei_type, typename dst_type>
void check_qconv(conv_fwd_config_t cfg, int N, int K, int C, int H, int W,
        int R, int S, const sc_dims &stride, const sc_dims &dilations,
        const sc_dims &padding, bool fuse_bias = false,
        bool default_cfg = false, bool force_blocking = false,
        bool force_channel_last = false) {
    int stride_h = stride[0], stride_w = stride[0];
    if (stride.size() == 2) { stride_w = stride[1]; }
    int padding_h = padding[0], padding_w = padding[0];
    if (padding.size() == 2) { padding_w = padding[1]; }
    int dilation_h = dilations[0], dilation_w = dilations[0];
    if (dilations.size() == 2) { dilation_w = dilations[1]; }

    sc_graph_t g;

    auto src_dtype = sc_data_traits_t<src_type>::type();
    auto wei_dtype = sc_data_traits_t<wei_type>::type();
    auto g_data = g.make_input({make_tensor({N, C, H, W}, src_dtype)});
    auto g_weight = g.make_input({make_tensor({K, C, R, S}, wei_dtype)});
    auto g_conv_out = g.make("conv_fwd_core",
            {g_data->get_outputs()[0], g_weight->get_outputs()[0]}, {},
            {{"strides", stride}, {"paddings", padding}, {"use_nested", false},
                    {"dilations", dilations}});
    COMPILE_ASSERT(!force_blocking || !force_channel_last,
            "only one of force_blocking and force_channel_last allowed");
    if (force_blocking) {
        g_conv_out->attrs_.set("temp.test_format", "NCHWc");
    } else if (force_channel_last) {
        g_conv_out->attrs_.set("temp.test_format", "NHWC");
    }
    auto tunop = g_conv_out->template dyn_cast<tunable_op_t>();

    auto gen = tunop->create_generator();
    auto conv_gen = (ops::gen_conv_fwd_t *)gen.get();
    int D = 0, P = 0, Q = 0;
    std::tie(D, P, Q) = conv_gen->get_output_shape();
    reflection::shared_general_object_t cfgptr;
    if (!default_cfg) {
        cfgptr = reflection::general_object_t::make(cfg);
        tunop->set_config(cfgptr);
        auto pcfg = (conv_fwd_config_t *)cfgptr.get();
        tunop->get_inputs()[0]->details_.set_format(
                sc_data_format_t::NCHWc(pcfg->C_block));
        tunop->get_inputs()[1]->details_.set_format(
                sc_data_format_t::KCRSck4c(pcfg->C_block, pcfg->K_block));
        tunop->get_outputs()[0]->details_.set_format(
                sc_data_format_t::NCHWc(pcfg->K_block));
    } else {
        cfgptr = gen->get_default_config(get_default_context());
        cfg = *(conv_fwd_config_t *)cfgptr.get();
    }

    std::vector<sc_op_ptr> args = {g_data, g_weight};
    sc_op_ptr final_out = g_conv_out;
    auto bc_axis = std::vector<int> {1};
    if (fuse_bias) {
        auto g_bias = g.make_input({make_tensor({K}, datatypes::f32)});
        final_out = g.make("add",
                {final_out->get_outputs()[0], g_bias->get_outputs()[0]}, {},
                {{"bc_axis", bc_axis}});
        args.emplace_back(g_bias);
    }
    auto g_out = g.make_output(final_out->get_outputs());
    args.insert(args.begin(), g_out);
    g.attrs_[sc_graph_t::attr_key_t::is_input_plain] = true;
    g.attrs_[sc_graph_t::attr_key_t::is_output_plain] = true;

    graph_driver(g, get_test_ctx());
    auto f = lower_graph(get_test_ctx(), g, args);
    auto fptr = jit_engine_t::make(get_test_ctx())->get_entry_func(f, true);

    auto output = alloc_array<dst_type>(N * K * P * Q);
    auto input = alloc_array<src_type>(N * C * H * W);
    auto weight = alloc_array<wei_type>(K * C * R * S);
    auto bias = alloc_array<float>(K);

    std::vector<generic_val> generic_args = {&output[0], &input[0], &weight[0]};
    if (fuse_bias) generic_args.emplace_back(&bias[0]);
    fptr->call_generic_default(generic_args.data());

    auto sc_output = std::move(output);
    auto plain_input = std::move(input);
    auto plain_weight = std::move(weight);

    test_buffer<float> plain_bias = std::move(bias);
    auto plain_output = alloc_array<dst_type>(N * K * P * Q, INIT_ZERO);

    compute_ref_direct_fwd(N, 1, K, C, H, W, P, Q, R, S, stride_h, stride_w,
            padding_h, padding_w, &plain_input[0], &plain_weight[0],
            &plain_bias[0], &plain_output[0], fuse_bias ? dir_t::FWD_B : FWD_I,
            nullptr, nullptr, false, 1, 1, 1, 0, 1, 1, dilation_h, dilation_w);

    bool correctness = equal(sc_output, plain_output, 1e-3);
    if (!correctness) {
        std::cout << "Check correctness FAIL." << std::endl;
        print_output(sc_output, plain_output, 100);
        check_sum(sc_output, plain_output);
    }
    EXPECT_TRUE(correctness);
}

template <typename src_type, typename wei_type, typename dst_type>
void check_qconv(conv_fwd_config_t cfg, int N, int K, int C, int H, int W,
        int R, int S, const sc_dims &stride, const sc_dims &padding,
        bool fuse_bias = false, bool default_cfg = false,
        bool force_blocking = false, bool force_channel_last = false) {
    check_qconv<src_type, wei_type, dst_type>(cfg, N, K, C, H, W, R, S, stride,
            sc_dims {1}, padding, fuse_bias, default_cfg, force_blocking,
            force_channel_last);
}

template <typename src_type, typename wei_type, typename dst_type>
void check_netsed_qconv(nested_conv_fwd_config_t cfg, int N, int K, int C,
        int H, int W, int R, int S, const sc_dims &stride,
        const sc_dims &padding, bool fuse_bias = false,
        bool default_cfg = false, bool force_blocking = false,
        bool force_channel_last = false) {
    int stride_h = stride[0], stride_w = stride[0];
    if (stride.size() == 2) { stride_w = stride[1]; }
    int padding_h = padding[0], padding_w = padding[0];
    if (padding.size() == 2) { padding_w = padding[1]; }

    sc_graph_t g;

    auto src_dtype = sc_data_traits_t<src_type>::type();
    auto wei_dtype = sc_data_traits_t<wei_type>::type();
    auto g_data = g.make_input({make_tensor({N, C, H, W}, src_dtype)});
    auto g_weight = g.make_input({make_tensor({K, C, R, S}, wei_dtype)});
    auto g_conv_out = g.make("conv_fwd_core",
            {g_data->get_outputs()[0], g_weight->get_outputs()[0]}, {},
            {{"strides", stride}, {"paddings", padding}});
    COMPILE_ASSERT(!force_blocking || !force_channel_last,
            "only one of force_blocking and force_channel_last allowed");
    if (force_blocking) {
        g_conv_out->attrs_.set("temp.test_format", "NCHWc");
    } else if (force_channel_last) {
        g_conv_out->attrs_.set("temp.test_format", "NHWC");
    }
    auto tunop = g_conv_out->template dyn_cast<tunable_op_t>();

    auto gen = tunop->create_generator();
    auto conv_gen = (ops::gen_conv_fwd_t *)gen.get();
    int D = 0, P = 0, Q = 0;
    std::tie(D, P, Q) = conv_gen->get_output_shape();

    std::vector<sc_op_ptr> args = {g_data, g_weight};
    sc_op_ptr final_out = g_conv_out;
    auto bc_axis = std::vector<int> {1};
    if (fuse_bias) {
        auto g_bias = g.make_input({make_tensor({K}, datatypes::f32)});
        final_out = g.make("add",
                {final_out->get_outputs()[0], g_bias->get_outputs()[0]}, {},
                {{"bc_axis", bc_axis}});
        args.emplace_back(g_bias);
    }
    auto g_out = g.make_output(final_out->get_outputs());
    args.insert(args.begin(), g_out);
    g.attrs_[sc_graph_t::attr_key_t::is_output_plain] = true;
    g.attrs_[sc_graph_t::attr_key_t::is_input_plain] = true;

    graph_driver(g, get_test_ctx());
    auto f = lower_graph(get_test_ctx(), g, args);
    auto fptr = jit_engine_t::make(get_test_ctx())->get_entry_func(f, true);

    auto output = alloc_array<dst_type>(N * K * P * Q);
    auto input = alloc_array<src_type>(N * C * H * W);
    auto weight = alloc_array<wei_type>(K * C * R * S);
    auto bias = alloc_array<float>(K);

    std::vector<generic_val> generic_args = {&output[0], &input[0], &weight[0]};
    if (fuse_bias) generic_args.emplace_back(&bias[0]);
    fptr->call_generic_default(generic_args.data());

    auto sc_output = std::move(output);

    auto plain_input = std::move(input);
    auto plain_weight = std::move(weight);

    test_buffer<float> plain_bias = std::move(bias);
    auto plain_output = alloc_array<dst_type>(N * K * P * Q, INIT_ZERO);

    compute_ref_direct_fwd(N, 1, K, C, H, W, P, Q, R, S, stride_h, stride_w,
            padding_h, padding_w, &plain_input[0], &plain_weight[0],
            &plain_bias[0], &plain_output[0], fuse_bias ? dir_t::FWD_B : FWD_I);

    bool correctness = equal(sc_output, plain_output, 1e-3);
    if (!correctness) {
        std::cout << "Check correctness FAIL." << std::endl;
        print_output(sc_output, plain_output, 100);
        check_sum(sc_output, plain_output);
    }
    EXPECT_TRUE(correctness);
}

auto partial_ow_cfg = conv_fwd_config_t {64, 64, 1, -1, -1, 8, -1, 1};
auto full_ow_cfg = conv_fwd_config_t {64, 32, 1, -1, -1, 56, -1, 3};
auto single_os_block_cfg = conv_fwd_config_t {64, 64, 1, -1, -1, 33, -1, 1};
auto single_os_block_cfg1 = conv_fwd_config_t {8, 64, 1, -1, -1, 61, -1, 1};
auto single_os_block_cfg2 = conv_fwd_config_t {16, 64, 1, -1, -1, 33, -1, 1};
auto multi_os_block_cfg = conv_fwd_config_t {64, 64, 1, -1, -1, 11, -1, 1};
TEST(GCCore_CPU_qconv2d_u8s8s32_3x3, partial_ow_NCX) {
    REQUIRE_AMX();
    check_qconv<uint8_t, int8_t, int32_t>(partial_ow_cfg, 64, 64, 64, 58, 58, 3,
            3, {1, 1}, {0, 0}, false, true, true, false);
}
TEST(GCCore_CPU_qconv2d_u8s8s32_3x3, partial_ow_NXC) {
    REQUIRE_AMX();
    check_qconv<uint8_t, int8_t, int32_t>(partial_ow_cfg, 64, 64, 64, 58, 58, 3,
            3, {1, 1}, {0, 0}, false, true, false, true);
}
TEST(GCCore_CPU_qconv2d_u8s8s32_3x3, full_ow_NCX) {
    REQUIRE_AMX();
    check_qconv<uint8_t, int8_t, int32_t>(full_ow_cfg, 64, 64, 64, 58, 58, 3, 3,
            {1, 1}, {0, 0}, false, true, true, false);
}
TEST(GCCore_CPU_qconv2d_u8s8s32_3x3, full_ow_NXC) {
    REQUIRE_AMX();
    check_qconv<uint8_t, int8_t, int32_t>(full_ow_cfg, 64, 64, 64, 58, 58, 3, 3,
            {1, 1}, {0, 0}, false, true, false, true);
}
TEST(GCCore_CPU_qconv2d_u8s8s32_3x3, single_os_block_NCX) {
    REQUIRE_AMX();
    check_qconv<uint8_t, int8_t, int32_t>(single_os_block_cfg, 4, 128, 64, 7, 7,
            3, 3, {1, 1}, {0, 0}, false, true, true, false);
}
TEST(GCCore_CPU_qconv2d_u8s8s32_3x3, single_os_block_NXC) {
    REQUIRE_AMX();
    check_qconv<uint8_t, int8_t, int32_t>(single_os_block_cfg, 4, 128, 64, 7, 7,
            3, 3, {1, 1}, {0, 0}, false, true, false, true);
}
TEST(GCCore_CPU_qconv2d_u8s8s32_3x3, single_os_block_1_NCX) {
    REQUIRE_AMX();
    check_qconv<uint8_t, int8_t, int32_t>(single_os_block_cfg1, 1, 8, 64, 9, 9,
            3, 3, {1, 1}, {0, 0}, false, true, true, false);
}
TEST(GCCore_CPU_qconv2d_u8s8s32_3x3, single_os_block_1_NXC) {
    REQUIRE_AMX();
    check_qconv<uint8_t, int8_t, int32_t>(single_os_block_cfg1, 1, 8, 64, 9, 9,
            3, 3, {1, 1}, {0, 0}, false, true, false, true);
}
TEST(GCCore_CPU_qconv2d_u8s8s32_3x3, single_os_block_2_NCX) {
    REQUIRE_AMX();
    check_qconv<uint8_t, int8_t, int32_t>(single_os_block_cfg2, 1, 16, 64, 7, 7,
            3, 3, {1, 1}, {0, 0}, false, true, true, false);
}
TEST(GCCore_CPU_qconv2d_u8s8s32_3x3, single_os_block_2_NXC) {
    REQUIRE_AMX();
    check_qconv<uint8_t, int8_t, int32_t>(single_os_block_cfg2, 1, 16, 64, 7, 7,
            3, 3, {1, 1}, {0, 0}, false, true, false, true);
}
TEST(GCCore_CPU_qconv2d_u8s8s32_3x3, multi_os_block_NCX) {
    REQUIRE_AMX();
    check_qconv<uint8_t, int8_t, int32_t>(multi_os_block_cfg, 4, 128, 64, 7, 7,
            3, 3, {1, 1}, {0, 0}, false, true, true, false);
}
TEST(GCCore_CPU_qconv2d_u8s8s32_3x3, multi_os_block_NXC) {
    REQUIRE_AMX();
    check_qconv<uint8_t, int8_t, int32_t>(multi_os_block_cfg, 4, 128, 64, 7, 7,
            3, 3, {1, 1}, {0, 0}, false, true, false, true);
}
TEST(GCCore_CPU_qconv2d, Test_2DConv_3x3_with_dilation_int8) {
    REQUIRE_AMX();
    SET_THREADS_OR_SKIP(56);
    std::vector<std::vector<int>> workload_list = {
            // prepadding
            {1, 256, 960, 38, 38, 12, 1, 0}, // deeplabv3_mobilenet
            {1, 256, 960, 62, 62, 24, 1, 0}, // deeplabv3_mobilenet
            {1, 256, 960, 86, 86, 36, 1, 0}, // deeplabv3_mobilenet
            {1, 256, 256, 32, 32, 2, 1, 0}, // deeplabv3_resnet101
            {1, 256, 256, 36, 36, 4, 1, 0}, // deeplabv3_resnet101
            {1, 1024, 512, 31, 31, 6, 1, 0}, // ssd300_vgg16
            // with padding
            {1, 256, 960, 14, 14, 12, 1, 12}, // deeplabv3_mobilenet
            {1, 256, 960, 14, 14, 24, 1, 24}, // deeplabv3_mobilenet
            {1, 256, 960, 14, 14, 36, 1, 36}, // deeplabv3_mobilenet
            {1, 256, 256, 28, 28, 2, 1, 2}, // deeplabv3_resnet101
            {1, 256, 256, 28, 28, 4, 1, 4}, // deeplabv3_resnet101
            {1, 1024, 512, 19, 19, 6, 1, 6}, // ssd300_vgg16
    }; // N, K, C, H, W, Dilation, Stride, Padding
    int R = 3, S = 3;
    for (auto workload : workload_list) {
        auto N = workload[0];
        auto K = workload[1];
        auto C = workload[2];
        auto H = workload[3];
        auto W = workload[4];
        auto dilation = workload[5];
        auto stride = workload[6];
        auto padding = workload[7];
        if (dilation * 2 + 1 > H + 2 * padding) { continue; }
        check_qconv<uint8_t, int8_t, int32_t>(conv_fwd_config_t(), N, K, C, H,
                W, R, S, {stride, stride}, {dilation, dilation},
                {padding, padding}, false, true, false, true);
        check_qconv<int8_t, int8_t, int32_t>(conv_fwd_config_t(), N, K, C, H, W,
                R, S, {stride, stride}, {dilation, dilation},
                {padding, padding}, false, true, false, true);
    }
    return;
}

TEST(GCCore_CPU_qconv2d_u8s8s32_1x1, no_padding_1_NCX) {
    REQUIRE_VNNI();
    check_qconv<uint8_t, int8_t, int32_t>(cfg_fwd, 128, 64, 64, 56, 56, 1, 1,
            {1, 1}, {0, 0}, false, true, true, false);
}
TEST(GCCore_CPU_qconv2d_u8s8s32_1x1, no_padding_1_NXC) {
    REQUIRE_VNNI();
    check_qconv<uint8_t, int8_t, int32_t>(cfg_fwd, 128, 64, 64, 56, 56, 1, 1,
            {1, 1}, {0, 0}, false, true, false, true);
}
TEST(GCCore_CPU_qconv2d_u8s8s32_1x1, no_padding_2_NCX) {
    REQUIRE_VNNI();
    check_qconv<uint8_t, int8_t, int32_t>(cfg_fwd, 32, 128, 128, 64, 64, 1, 1,
            {1, 2}, {0, 0}, false, true, true, false);
}
TEST(GCCore_CPU_qconv2d_u8s8s32_1x1, no_padding_2_NXC) {
    REQUIRE_VNNI();
    check_qconv<uint8_t, int8_t, int32_t>(cfg_fwd, 32, 128, 128, 64, 64, 1, 1,
            {1, 2}, {0, 0}, false, true, false, true);
}
TEST(GCCore_CPU_qconv2d_u8s8s32_1x1, no_padding_3_NCX) {
    REQUIRE_VNNI();
    check_qconv<uint8_t, int8_t, int32_t>(cfg_fwd, 32, 256, 128, 64, 64, 1, 1,
            {2, 3}, {0, 0}, false, true, true, false);
}
TEST(GCCore_CPU_qconv2d_u8s8s32_1x1, no_padding_3_NXC) {
    REQUIRE_VNNI();
    check_qconv<uint8_t, int8_t, int32_t>(cfg_fwd, 32, 256, 128, 64, 64, 1, 1,
            {2, 3}, {0, 0}, false, true, false, true);
}

TEST(GCCore_CPU_qconv2d_u8s8s32_3x3, no_padding_1_NCX) {
    REQUIRE_VNNI();
    check_qconv<uint8_t, int8_t, int32_t>(cfg_fwd, 128, 64, 64, 56, 56, 3, 3,
            {1, 1}, {0, 0}, false, true, true, false);
}
TEST(GCCore_CPU_qconv2d_u8s8s32_3x3, no_padding_1_NXC) {
    REQUIRE_VNNI();
    check_qconv<uint8_t, int8_t, int32_t>(cfg_fwd, 128, 64, 64, 56, 56, 3, 3,
            {1, 1}, {0, 0}, false, true, false, true);
}
TEST(GCCore_CPU_qconv2d_u8s8s32_3x3, no_padding_2_NCX) {
    REQUIRE_VNNI();
    check_qconv<uint8_t, int8_t, int32_t>(cfg_fwd, 32, 128, 128, 64, 64, 3, 3,
            {2, 1}, {0, 0}, false, true, true, false);
}
TEST(GCCore_CPU_qconv2d_u8s8s32_3x3, no_padding_2_NXC) {
    REQUIRE_VNNI();
    check_qconv<uint8_t, int8_t, int32_t>(cfg_fwd, 32, 128, 128, 64, 64, 3, 3,
            {2, 1}, {0, 0}, false, true, false, true);
}
TEST(GCCore_CPU_qconv2d_u8s8s32_3x3, no_padding_3_NCX) {
    REQUIRE_VNNI();
    check_qconv<uint8_t, int8_t, int32_t>(cfg_fwd, 32, 256, 128, 64, 64, 3, 3,
            {2, 3}, {0, 0}, false, true, true, false);
}
TEST(GCCore_CPU_qconv2d_u8s8s32_3x3, no_padding_3_NXC) {
    REQUIRE_VNNI();
    check_qconv<uint8_t, int8_t, int32_t>(cfg_fwd, 32, 256, 128, 64, 64, 3, 3,
            {2, 3}, {0, 0}, false, true, false, true);
}

TEST(GCCore_CPU_qconv2d_u8s8s32_3x3, with_padding_1_NCX) {
    REQUIRE_VNNI();
    check_qconv<uint8_t, int8_t, int32_t>(cfg_fwd, 128, 64, 64, 56, 56, 3, 3,
            {1, 2}, {1, 2}, false, true, true, false);
}

#define conv_padding_support_NXC 0

#if conv_padding_support_NXC
TEST(GCCore_CPU_qconv2d_u8s8s32_3x3, with_padding_1_NXC) {
    REQUIRE_VNNI();
    check_qconv<uint8_t, int8_t, int32_t>(cfg_fwd, 128, 64, 64, 56, 56, 3, 3,
            {1, 2}, {1, 2}, false, true, false, true);
}
#endif
TEST(GCCore_CPU_qconv2d_u8s8s32_3x3, with_padding_2_NCX) {
    REQUIRE_VNNI();
    check_qconv<uint8_t, int8_t, int32_t>(cfg_fwd, 32, 256, 128, 64, 64, 3, 3,
            {2, 3}, {1, 0}, false, true, true, false);
}
#if conv_padding_support_NXC
TEST(GCCore_CPU_qconv2d_u8s8s32_3x3, with_padding_2_NXC) {
    REQUIRE_VNNI();
    check_qconv<uint8_t, int8_t, int32_t>(cfg_fwd, 32, 256, 128, 64, 64, 3, 3,
            {2, 3}, {1, 0}, false, true, false, true);
}
#endif
// top/middle/bottom padding region, left padding only, no padding, right
// padding only
TEST(GCCore_CPU_qconv2d_u8s8s32_3x3, with_padding_3_NCX) {
    REQUIRE_AMX();
    check_qconv<uint8_t, int8_t, int32_t>(
            conv_fwd_config_t {1, 4, 1, 1, 1, 1, -1, -1}, 1, 1, 4, 6, 6, 3, 3,
            {1, 1}, {2, 2}, false, true, true, false);
}
TEST(GCCore_CPU_qconv2d_u8s8s32_3x3, with_padding_3_NXC) {
    REQUIRE_AMX();
    check_qconv<uint8_t, int8_t, int32_t>(
            conv_fwd_config_t {1, 4, 1, 1, 1, 1, -1, -1}, 1, 1, 4, 6, 6, 3, 3,
            {1, 1}, {2, 2}, false, true, false, true);
}
// top/middle/bottom padding region, left and right padding
TEST(GCCore_CPU_qconv2d_u8s8s32_3x3, with_padding_4_NCX) {
    REQUIRE_AMX();
    check_qconv<uint8_t, int8_t, int32_t>(
            conv_fwd_config_t {1, 4, 1, 1, 6, 6, -1, -1}, 1, 1, 4, 6, 6, 3, 3,
            {1, 1}, {1, 1}, false, true, true, false);
}
TEST(GCCore_CPU_qconv2d_u8s8s32_3x3, with_padding_4_NXC) {
    REQUIRE_AMX();
    check_qconv<uint8_t, int8_t, int32_t>(
            conv_fwd_config_t {1, 4, 1, 1, 6, 6, -1, -1}, 1, 1, 4, 6, 6, 3, 3,
            {1, 1}, {1, 1}, false, true, false, true);
}
// top/middle/bottom padding region, left padding only, right padding not being
// used
TEST(GCCore_CPU_qconv2d_u8s8s32_3x3, with_padding_5_NCX) {
    REQUIRE_AMX();
    check_qconv<uint8_t, int8_t, int32_t>(cfg_fwd, 1, 1, 4, 4, 4, 3, 3, {2, 2},
            {1, 1}, false, true, true, false);
}
TEST(GCCore_CPU_qconv2d_u8s8s32_3x3, with_padding_5_NXC) {
    REQUIRE_AMX();
    check_qconv<uint8_t, int8_t, int32_t>(cfg_fwd, 1, 1, 4, 4, 4, 3, 3, {2, 2},
            {1, 1}, false, true, false, true);
}

TEST(GCCore_CPU_qconv2d_nested_u8s8s32_3x3, rn50_stage1_NCX) {
    REQUIRE_AMX();
    check_netsed_qconv<int8_t, int8_t, int32_t>(nested_conv_fwd_config_t(), 56,
            64, 64, 58, 58, 3, 3, {1, 1}, {0, 0}, false, true, true, false);
}

TEST(GCCore_CPU_qconv2d_nested_u8s8s32_3x3, rn50_stage1_NXC) {
    REQUIRE_AMX();
    check_netsed_qconv<int8_t, int8_t, int32_t>(nested_conv_fwd_config_t(), 56,
            64, 64, 58, 58, 3, 3, {1, 1}, {0, 0}, false, true, false, true);
}

TEST(GCCore_CPU_qconv2d_nested_u8s8s32_3x3, rn50_stage2_NCX) {
    REQUIRE_AMX();
    check_netsed_qconv<int8_t, int8_t, int32_t>(nested_conv_fwd_config_t(), 56,
            128, 128, 30, 30, 3, 3, {1, 1}, {0, 0}, false, true, true, false);
}

TEST(GCCore_CPU_qconv2d_nested_u8s8s32_3x3, rn50_stage2_NXC) {
    REQUIRE_AMX();
    check_netsed_qconv<int8_t, int8_t, int32_t>(nested_conv_fwd_config_t(), 56,
            128, 128, 30, 30, 3, 3, {1, 1}, {0, 0}, false, true, false, true);
}

TEST(GCCore_CPU_qconv2d_nested_u8s8s32_3x3, rn50_stage3_NCX) {
    REQUIRE_AMX();
    check_netsed_qconv<int8_t, int8_t, int32_t>(nested_conv_fwd_config_t(), 56,
            256, 256, 16, 16, 3, 3, {1, 1}, {0, 0}, false, true, true, false);
}

TEST(GCCore_CPU_qconv2d_nested_u8s8s32_3x3, rn50_stage3_NXC) {
    REQUIRE_AMX();
    check_netsed_qconv<int8_t, int8_t, int32_t>(nested_conv_fwd_config_t(), 56,
            256, 256, 16, 16, 3, 3, {1, 1}, {0, 0}, false, true, false, true);
}

TEST(GCCore_CPU_qconv2d_nested_u8s8s32_3x3, rn50_stage4_NCX) {
    REQUIRE_AMX();
    check_netsed_qconv<int8_t, int8_t, int32_t>(nested_conv_fwd_config_t(), 56,
            512, 512, 9, 9, 3, 3, {1, 1}, {0, 0}, false, true, true, false);
}

TEST(GCCore_CPU_qconv2d_nested_u8s8s32_3x3, rn50_stage4_NXC) {
    REQUIRE_AMX();
    check_netsed_qconv<int8_t, int8_t, int32_t>(nested_conv_fwd_config_t(), 56,
            512, 512, 9, 9, 3, 3, {1, 1}, {0, 0}, false, true, false, true);
}
