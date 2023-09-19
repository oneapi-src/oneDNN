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
#include <ops/templates/conv_rl.hpp>
#include <ops/templates/nested_conv_fwd.hpp>
#include <runtime/dynamic_dispatch/dynamic_tensor.hpp>
#include <runtime/runtime.hpp>
#include <util/any_map.hpp>
#include <util/math_utils.hpp>
#include <util/reflection.hpp>
#include <util/utils.hpp>

using namespace dnnl::impl::graph::gc;
using conv_fwd_config_t = ops::conv_fwd_config_t;
using conv_fwd_rl_config_t = ops::conv_fwd_rl_config_t;
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

static graph_tensor_ptr make_tensor(const sc_dims &d, sc_data_type_t dtype) {
    return graph_tensor::make(d, sc_data_format_t(), dtype);
}

template <typename src_type, typename wei_type, typename dst_type>
void check_qconv(conv_fwd_config_t cfg, int N, int G, int K, int C, int H,
        int W, int R, int S, const sc_dims &stride, const sc_dims &dilations,
        const sc_dims &pads_begin, const sc_dims &pads_end,
        bool fuse_bias = false, bool default_cfg = false,
        bool force_blocking = false, bool force_channel_last = false) {
    int stride_h = stride[0], stride_w = stride[0];
    if (stride.size() == 2) { stride_w = stride[1]; }
    int padding_h = pads_begin[0], padding_w = pads_begin[0];
    if (pads_begin.size() == 2) { padding_w = pads_begin[1]; }
    int dilation_h = dilations[0], dilation_w = dilations[0];
    if (dilations.size() == 2) { dilation_w = dilations[1]; }
    COMPILE_ASSERT(C % G == 0 && K % G == 0,
            "C and K should be dividable by G, but got C("
                    << C << "), K(" << K << "), G(" << G << ").");

    auto ctx = std::make_shared<context_t>(*get_test_ctx());
    ctx->flags_.mixed_fusion_ = true;
    sc_graph_t g;

    auto src_dtype = sc_data_traits_t<src_type>::type();
    auto wei_dtype = sc_data_traits_t<wei_type>::type();
    sc_dims data_dims = {N, C, H, W};
    sc_dims weight_dims = {K, C / G, R, S};
    auto g_data = g.make_input({make_tensor(data_dims, src_dtype)});
    auto g_weight = g.make_input({make_tensor(weight_dims, wei_dtype)});
    auto g_conv_out = g.make("conv_fwd_core",
            {g_data->get_outputs()[0], g_weight->get_outputs()[0]}, {},
            {{"strides", stride}, {"pads_begin", pads_begin},
                    {"pads_end", pads_end}, {"use_nested", false},
                    {"dilations", dilations}, {"groups", G}});
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
    if (!default_cfg) {
        reflection::shared_general_object_t cfgptr
                = reflection::general_object_t::make(cfg);
        tunop->set_config(cfgptr);
        auto pcfg = (conv_fwd_config_t *)cfgptr.get();
        tunop->get_inputs()[0]->details_.set_format(
                sc_data_format_t::NCHWc(pcfg->C_block));
        tunop->get_inputs()[1]->details_.set_format(
                sc_data_format_t::KCRSck4c(pcfg->C_block, pcfg->K_block));
        tunop->get_outputs()[0]->details_.set_format(
                sc_data_format_t::NCHWc(pcfg->K_block));
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
    g.attrs_["use_rl"] = ops::rl_kind::NO_LOWERING;

    graph_driver(g, ctx);
    auto f = lower_graph(ctx, g, args);
    auto fptr = jit_engine_t::make(get_test_ctx())->get_entry_func(f, true);

    auto output = alloc_array<dst_type>(N * K * P * Q);
    auto input = alloc_array<src_type>(math_utils::get_dims_product(data_dims));
    auto weight
            = alloc_array<wei_type>(math_utils::get_dims_product(weight_dims));
    auto bias = alloc_array<float>(K);

    std::vector<generic_val> generic_args = {&output[0], &input[0], &weight[0]};
    if (fuse_bias) generic_args.emplace_back(&bias[0]);
    fptr->call_generic_default(generic_args.data());

    auto sc_output = std::move(output);
    auto plain_input = std::move(input);
    auto plain_weight = std::move(weight);

    test_buffer<float> plain_bias = std::move(bias);
    auto plain_output = alloc_array<dst_type>(N * K * P * Q, INIT_ZERO);

    compute_ref_direct_fwd(N, G, K, C, H, W, P, Q, R, S, stride_h, stride_w,
            padding_h, padding_w, &plain_input[0], &plain_weight[0],
            &plain_bias[0], &plain_output[0], fuse_bias ? dir_t::FWD_B : FWD_I,
            nullptr, nullptr, false, 1, 1, 1, 0, 1, 1, dilation_h, dilation_w);

    test_utils::compare_data(sc_output, plain_output, 1e-3f, 1e-3f);
}

template <typename src_type, typename wei_type, typename dst_type>
void check_qconv(conv_fwd_config_t cfg, int N, int K, int C, int H, int W,
        int R, int S, const sc_dims &stride, const sc_dims &dilations,
        const sc_dims &padding, bool fuse_bias = false,
        bool default_cfg = false, bool force_blocking = false,
        bool force_channel_last = false) {
    check_qconv<src_type, wei_type, dst_type>(cfg, N, 1, K, C, H, W, R, S,
            stride, dilations, padding, padding, fuse_bias, default_cfg,
            force_blocking, force_channel_last);
}

template <typename src_type, typename wei_type, typename dst_type>
void check_nested_qconv(nested_conv_fwd_config_t cfg, int N, int K, int C,
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

    test_utils::compare_data(sc_output, plain_output, 1e-3f, 1e-3f);
}

template <typename src_type, typename wei_type, typename dst_type>
void check_dynamic_netsed_qconv(nested_conv_fwd_config_t cfg, int N, int K,
        int C, int H, int W, int R, int S, const sc_dims &stride,
        const sc_dims &padding, bool fuse_bias = false,
        bool default_cfg = false, bool force_blocking = false,
        bool force_channel_last = false, int real_N = -1, int real_H = -1,
        int real_W = -1) {
    int stride_h = stride[0], stride_w = stride[0];
    if (stride.size() == 2) { stride_w = stride[1]; }
    int padding_h = padding[0], padding_w = padding[0];
    if (padding.size() == 2) { padding_w = padding[1]; }
    bool is_dynamic = N < 0 || H < 0 || W < 0;

    sc_graph_t g;
    auto src_dtype = sc_data_traits_t<src_type>::type();
    auto wei_dtype = sc_data_traits_t<wei_type>::type();
    auto g_data = g.make_input({make_tensor({N, C, H, W}, src_dtype)});
    auto g_weight = g.make_input({make_tensor({K, C, R, S}, wei_dtype)});
    auto g_conv_out = g.make("conv_fwd_core",
            {g_data->get_outputs()[0], g_weight->get_outputs()[0]}, {},
            {{"strides", stride}, {"paddings", padding}, {"no_fuse", false}});
    COMPILE_ASSERT(!force_blocking || !force_channel_last,
            "only one of force_blocking and force_channel_last allowed");
    if (force_blocking) {
        g_conv_out->attrs_.set("temp.test_format", "NCHWc");
    } else if (force_channel_last) {
        g_conv_out->attrs_.set("temp.test_format", "NHWC");
    }
    auto tunop = g_conv_out->template dyn_cast<tunable_op_t>();

    auto gen = tunop->create_generator();
    auto conv_gen = (ops::gen_nested_conv_fwd_t *)gen.get();
    int D = 0, P = 0, Q = 0;
    std::tie(D, P, Q) = conv_gen->get_output_shape();
    reflection::shared_general_object_t cfgptr;
    cfgptr = gen->get_default_config(get_test_ctx());
    cfg = *(nested_conv_fwd_config_t *)cfgptr.get();
    tunop->set_config(cfgptr);
    tunop->get_inputs()[0]->details_.set_format(sc_data_format_t::NHWC());
    tunop->get_inputs()[1]->details_.set_format(
            sc_data_format_t::KCRSck4c(cfg.im_ic_block, cfg.im_oc_block));
    tunop->get_outputs()[0]->details_.set_format(sc_data_format_t::NHWC());

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
    g.attrs_[sc_graph_t::attr_key_t::is_output_plain] = false;
    g.attrs_[sc_graph_t::attr_key_t::is_input_plain] = false;

    graph_driver(g, get_default_context());
    auto f = lower_graph(get_default_context(), g, args);
    auto fptr = jit_engine_t::make(get_default_context())
                        ->get_entry_func(f, true);

    if (is_dynamic) {
        if (is_dynamic_dim(N)) {
            assert(real_N > 0);
            N = real_N;
        }
        if (is_dynamic_dim(H)) {
            assert(real_H > 0);
            H = real_H;
        }
        if (is_dynamic_dim(W)) {
            assert(real_W > 0);
            W = real_W;
        }
        P = (H + padding_h * 2 - R) / stride_h + 1;
        Q = (W + padding_w * 2 - S) / stride_w + 1;
    }

    auto output = alloc_array<dst_type>(N * K * P * Q, INIT_NOOP);
    auto input = alloc_array<src_type>(N * C * H * W);
    auto weight = alloc_array<wei_type>(K * C * R * S);
    auto bias = alloc_array<float>(K);

    sc_dims out_dims = sc_dims {N, K, P, Q};
    sc_dims in_a_dims = sc_dims {N, C, H, W};
    sc_dims in_weight_dims = sc_dims {K, C, R, S};
    sc_dims in_postop_dims = sc_dims {K};

    // Define dynamic tensor
    runtime::dynamic_tensor_t dyn_output(&output[0], &out_dims[0],
            out_dims.size(), uint32_t(sc_data_traits_t<dst_type>::type()), 0);
    runtime::dynamic_tensor_t dyn_input(&input[0], &in_a_dims[0],
            in_a_dims.size(), uint32_t(sc_data_traits_t<src_type>::type()), 0);
    runtime::dynamic_tensor_t dyn_weight(&weight[0], &in_weight_dims[0],
            in_weight_dims.size(), uint32_t(sc_data_traits_t<wei_type>::type()),
            0);
    runtime::dynamic_tensor_t dyn_bias(&bias[0], &in_postop_dims[0],
            in_postop_dims.size(), uint32_t(datatypes::f32), 0);

    std::vector<void *> sc_args = is_dynamic
            ? std::vector<void *> {&dyn_output, &dyn_input, &dyn_weight}
            : std::vector<void *> {&output[0], &input[0], &weight[0]};
    std::vector<generic_val> generic_args;
    for (unsigned i = 0; i < sc_args.size(); i++)
        generic_args.emplace_back(sc_args.at(i));
    if (fuse_bias) {
        if (is_dynamic)
            generic_args.emplace_back(&dyn_bias);
        else
            generic_args.emplace_back(&bias[0]);
    }
    fptr->call_generic_default(generic_args.data());

    auto sc_output
            = any2NCHW(g_conv_out->get_outputs()[0]->details_.get_format(),
                    output, N, K, P, Q, cfg.im_oc_block);

    auto plain_input = any2NCHW(g_data->get_outputs()[0]->details_.get_format(),
            input, N, C, H, W, cfg.im_ic_block);
    auto plain_weight = KCRSckc2KCRS(weight, K / cfg.im_oc_block,
            utils::divide_and_ceil(C, cfg.im_ic_block), R, S,
            utils::divide_and_ceil(cfg.im_ic_block, 4), cfg.im_oc_block);

    test_buffer<float> plain_bias = std::move(bias);
    auto plain_output = alloc_array<dst_type>(N * K * P * Q, INIT_ZERO);

    compute_ref_direct_fwd(N, 1, K, C, H, W, P, Q, R, S, stride_h, stride_w,
            padding_h, padding_w, &plain_input[0], &plain_weight[0],
            &plain_bias[0], &plain_output[0], fuse_bias ? dir_t::FWD_B : FWD_I);

    test_utils::compare_data(sc_output, plain_output, 1e-3f, 1e-3f);
}

template <typename src_type, typename wei_type, typename dst_type>
void check_rl_qconv(conv_fwd_rl_config_t cfg, int N, int G, int K, int C, int H,
        int W, int R, int S, const sc_dims &stride, const sc_dims &dilations,
        const sc_dims &pads_begin, const sc_dims &pads_end,
        bool fuse_bias = false, bool default_cfg = false) {
    COMPILE_ASSERT(default_cfg, "only default cfg is supported!");
    // use new fusion manager
    auto ctx = std::make_shared<context_t>(*get_test_ctx());
    ctx->flags_.mixed_fusion_ = true;

    int stride_h = stride[0], stride_w = stride[0];
    if (stride.size() == 2) { stride_w = stride[1]; }
    int padding_h = pads_begin[0], padding_w = pads_begin[0];
    if (pads_begin.size() == 2) { padding_w = pads_begin[1]; }
    int dilation_h = dilations[0], dilation_w = dilations[0];
    if (dilations.size() == 2) { dilation_w = dilations[1]; }
    COMPILE_ASSERT(C % G == 0 && K % G == 0,
            "C and K should be dividable by G, but got C("
                    << C << "), K(" << K << "), G(" << G << ").");

    sc_graph_t g;
    auto src_shape = sc_dims {N, C, H, W};
    auto wei_shape = sc_dims {K, C / G, R, S};
    auto src_dtype = sc_data_traits_t<src_type>::type();
    auto wei_dtype = sc_data_traits_t<wei_type>::type();
    auto g_data = g.make_input({make_tensor(src_shape, src_dtype)});
    auto g_weight = g.make_input({make_tensor(wei_shape, wei_dtype)});
    auto g_conv_out = g.make("conv_fwd_core",
            {g_data->get_outputs()[0], g_weight->get_outputs()[0]}, {},
            {{"strides", stride}, {"pads_begin", pads_begin},
                    {"pads_end", pads_end}, {"use_nested", false},
                    {"dilations", dilations}, {"groups", G}});

    auto out_shape = ops::conv_fwd_core_op_t::infer_out_dims(
            g_conv_out->get_owner_graph(), src_shape, wei_shape, pads_begin,
            pads_end, stride, dilations);
    COMPILE_ASSERT(out_shape.size() == src_shape.size(),
            "out_shape is expected to be same size vs src_shape, but got "
                    << out_shape.size() << " vs " << src_shape.size());
    int P = out_shape[2], Q = out_shape[3];

    std::vector<sc_op_ptr> args = {g_data, g_weight};
    sc_op_ptr final_out = g_conv_out;
    auto bc_axis = std::vector<int> {1};
    if (fuse_bias) {
        auto g_bias = g.make_input({make_tensor({K}, datatypes::s32)});
        final_out = g.make("add",
                {final_out->get_outputs()[0], g_bias->get_outputs()[0]}, {},
                {{"bc_axis", bc_axis}});
        args.emplace_back(g_bias);
    }
    auto g_out = g.make_output(final_out->get_outputs());
    args.insert(args.begin(), g_out);
    g.attrs_[sc_graph_t::attr_key_t::is_input_plain] = true;
    g.attrs_[sc_graph_t::attr_key_t::is_output_plain] = true;

    graph_driver(g, ctx);
    auto f = lower_graph(ctx, g, args);
    int tile_col = (src_dtype == datatypes::bf16) ? 32 : 64;
    int threshold = static_cast<int>(tile_col * 0.75);
    if (C / G * S <= threshold) {
        std::stringstream ss;
        ss << f;
        EXPECT_TRUE(ss.str().find("aux_buf") != std::string::npos);
    }
    auto fptr = jit_engine_t::make(ctx)->get_entry_func(f, true);

    auto output = alloc_array<dst_type>(N * K * P * Q);
    auto input = alloc_array<src_type>(math_utils::get_dims_product(src_shape));
    auto weight
            = alloc_array<wei_type>(math_utils::get_dims_product(wei_shape));
    auto bias = alloc_array<int32_t>(K);

    std::vector<generic_val> generic_args = {&output[0], &input[0], &weight[0]};
    if (fuse_bias) generic_args.emplace_back(&bias[0]);
    fptr->call_generic_default(generic_args.data());

    auto sc_output = std::move(output);
    auto plain_input = std::move(input);
    auto plain_weight = std::move(weight);

    test_buffer<int32_t> plain_bias = std::move(bias);
    auto plain_output = alloc_array<dst_type>(N * K * P * Q, INIT_ZERO);

    compute_ref_direct_fwd(N, G, K, C, H, W, P, Q, R, S, stride_h, stride_w,
            padding_h, padding_w, &plain_input[0], &plain_weight[0],
            &plain_bias[0], &plain_output[0], fuse_bias ? dir_t::FWD_B : FWD_I,
            nullptr, nullptr, false, 1, 1, 1, 0, 1, 1, dilation_h, dilation_w,
            true);

    test_utils::compare_data(sc_output, plain_output, 1e-3f, 1e-3f);
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
            3, {1, 1}, {1, 1}, {0, 0}, false, true, true, false);
}
TEST(GCCore_CPU_qconv2d_u8s8s32_3x3, partial_ow_NXC) {
    REQUIRE_AMX();
    check_qconv<uint8_t, int8_t, int32_t>(partial_ow_cfg, 64, 64, 64, 58, 58, 3,
            3, {1, 1}, {1, 1}, {0, 0}, false, true, false, true);
}
TEST(GCCore_CPU_qconv2d_u8s8s32_3x3, full_ow_NCX) {
    REQUIRE_AMX();
    check_qconv<uint8_t, int8_t, int32_t>(full_ow_cfg, 64, 64, 64, 58, 58, 3, 3,
            {1, 1}, {1, 1}, {0, 0}, false, true, true, false);
}
TEST(GCCore_CPU_qconv2d_u8s8s32_3x3, full_ow_NXC) {
    REQUIRE_AMX();
    check_qconv<uint8_t, int8_t, int32_t>(full_ow_cfg, 64, 64, 64, 58, 58, 3, 3,
            {1, 1}, {1, 1}, {0, 0}, false, true, false, true);
}
TEST(GCCore_CPU_qconv2d_u8s8s32_3x3, single_os_block_NCX) {
    REQUIRE_AMX();
    check_qconv<uint8_t, int8_t, int32_t>(single_os_block_cfg, 4, 128, 64, 7, 7,
            3, 3, {1, 1}, {1, 1}, {0, 0}, false, true, true, false);
}
TEST(GCCore_CPU_qconv2d_u8s8s32_3x3, single_os_block_NXC) {
    REQUIRE_AMX();
    check_qconv<uint8_t, int8_t, int32_t>(single_os_block_cfg, 4, 128, 64, 7, 7,
            3, 3, {1, 1}, {1, 1}, {0, 0}, false, true, false, true);
}
TEST(GCCore_CPU_qconv2d_u8s8s32_3x3, single_os_block_1_NCX) {
    REQUIRE_AMX();
    check_qconv<uint8_t, int8_t, int32_t>(single_os_block_cfg1, 1, 8, 64, 9, 9,
            3, 3, {1, 1}, {1, 1}, {0, 0}, false, true, true, false);
}
TEST(GCCore_CPU_qconv2d_u8s8s32_3x3, single_os_block_1_NXC) {
    REQUIRE_AMX();
    check_qconv<uint8_t, int8_t, int32_t>(single_os_block_cfg1, 1, 8, 64, 9, 9,
            3, 3, {1, 1}, {1, 1}, {0, 0}, false, true, false, true);
}
TEST(GCCore_CPU_qconv2d_u8s8s32_3x3, single_os_block_2_NCX) {
    REQUIRE_AMX();
    check_qconv<uint8_t, int8_t, int32_t>(single_os_block_cfg2, 1, 16, 64, 7, 7,
            3, 3, {1, 1}, {1, 1}, {0, 0}, false, true, true, false);
}
TEST(GCCore_CPU_qconv2d_u8s8s32_3x3, single_os_block_2_NXC) {
    REQUIRE_AMX();
    check_qconv<uint8_t, int8_t, int32_t>(single_os_block_cfg2, 1, 16, 64, 7, 7,
            3, 3, {1, 1}, {1, 1}, {0, 0}, false, true, false, true);
}
TEST(GCCore_CPU_qconv2d_u8s8s32_3x3, multi_os_block_NCX) {
    REQUIRE_AMX();
    check_qconv<uint8_t, int8_t, int32_t>(multi_os_block_cfg, 4, 128, 64, 7, 7,
            3, 3, {1, 1}, {1, 1}, {0, 0}, false, true, true, false);
}
TEST(GCCore_CPU_qconv2d_u8s8s32_3x3, asymmetric_padding_3x3_NCX_1) {
    REQUIRE_AMX();
    check_qconv<uint8_t, int8_t, int32_t>(conv_fwd_config_t(), 1, 1, 64, 64,
            224, 224, 7, 7, {2, 2}, {1, 1}, {3, 3}, {2, 2}, false, true, true,
            false);
}
TEST(GCCore_CPU_qconv2d_u8s8s32_3x3, asymmetric_padding_3x3_NXC_1) {
    REQUIRE_AMX();
    check_qconv<uint8_t, int8_t, int32_t>(conv_fwd_config_t(), 1, 1, 64, 64,
            224, 224, 7, 7, {2, 2}, {1, 1}, {3, 3}, {2, 2}, false, true, false,
            true);
}
TEST(GCCore_CPU_qconv2d_u8s8s32_3x3, asymmetric_padding_3x3_NCX_2) {
    REQUIRE_AMX();
    check_qconv<uint8_t, int8_t, int32_t>(conv_fwd_config_t(), 1, 1, 64, 64,
            224, 224, 7, 7, {1, 1}, {1, 1}, {3, 3}, {2, 2}, false, true, true,
            false);
}
TEST(GCCore_CPU_qconv2d_u8s8s32_3x3, asymmetric_padding_3x3_NXC_2) {
    REQUIRE_AMX();
    check_qconv<uint8_t, int8_t, int32_t>(conv_fwd_config_t(), 1, 1, 64, 64,
            224, 224, 7, 7, {1, 1}, {1, 1}, {3, 3}, {2, 2}, false, true, false,
            true);
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
            {7, 1024, 32, 19, 19, 6, 1, 6},
            {7, 1024, 4, 6, 6, 2, 1, 2},
    }; // N, K, C, H, W, Dilation, Stride, Padding
    int R = 3, S = 3;
    int G = 1;
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
        check_qconv<uint8_t, int8_t, int32_t>(conv_fwd_config_t(), N, G, K, C,
                H, W, R, S, {stride, stride}, {dilation, dilation},
                {padding, padding}, {padding, padding}, false, true, false,
                true);
        check_qconv<int8_t, int8_t, int32_t>(conv_fwd_config_t(), N, G, K, C, H,
                W, R, S, {stride, stride}, {dilation, dilation},
                {padding, padding}, {padding, padding}, false, true, false,
                true);
    }
    return;
}
TEST(GCCore_CPU_qconv2d, Test_2DConv_3x3_with_asymmetric_dilation_int8) {
    REQUIRE_AMX();
    SET_THREADS_OR_SKIP(56);
    std::vector<std::vector<int>> workload_list = {
            {7, 1024, 32, 19, 19, 1, 6, 1, 6},
            {7, 1024, 4, 6, 6, 1, 2, 1, 2},
    }; // N, K, C, H, W, Dilation_H, Dilation_W, Stride, Padding
    int R = 3, S = 3;
    for (auto workload : workload_list) {
        auto N = workload[0];
        auto K = workload[1];
        auto C = workload[2];
        auto H = workload[3];
        auto W = workload[4];
        auto dilation_h = workload[5];
        auto dilation_w = workload[6];
        auto stride = workload[7];
        auto padding = workload[8];
        if ((dilation_h * 2 + 1 > H + 2 * padding)
                || (dilation_w * 2 + 1 > W + 2 * padding)) {
            continue;
        }
        check_qconv<uint8_t, int8_t, int32_t>(conv_fwd_config_t(), N, K, C, H,
                W, R, S, {stride, stride}, {dilation_h, dilation_w},
                {padding, padding}, false, true, false, true);
        check_qconv<int8_t, int8_t, int32_t>(conv_fwd_config_t(), N, K, C, H, W,
                R, S, {stride, stride}, {dilation_h, dilation_w},
                {padding, padding}, false, true, false, true);
    }
    return;
}
TEST(GCCore_CPU_qconv2d_u8s8s32_1x1, no_padding_1_NCX) {
    REQUIRE_VNNI();
    check_qconv<uint8_t, int8_t, int32_t>(cfg_fwd, 128, 64, 64, 56, 56, 1, 1,
            {1, 1}, {1, 1}, {0, 0}, false, true, true, false);
}
TEST(GCCore_CPU_qconv2d_u8s8s32_1x1, no_padding_1_NXC) {
    REQUIRE_VNNI();
    check_qconv<uint8_t, int8_t, int32_t>(cfg_fwd, 128, 64, 64, 56, 56, 1, 1,
            {1, 1}, {1, 1}, {0, 0}, false, true, false, true);
}
TEST(GCCore_CPU_qconv2d_u8s8s32_1x1, no_padding_2_NCX) {
    REQUIRE_VNNI();
    check_qconv<uint8_t, int8_t, int32_t>(cfg_fwd, 32, 128, 128, 64, 64, 1, 1,
            {1, 2}, {1, 1}, {0, 0}, false, true, true, false);
}
TEST(GCCore_CPU_qconv2d_u8s8s32_1x1, no_padding_2_NXC) {
    REQUIRE_VNNI();
    check_qconv<uint8_t, int8_t, int32_t>(cfg_fwd, 32, 128, 128, 64, 64, 1, 1,
            {1, 2}, {1, 1}, {0, 0}, false, true, false, true);
}
TEST(GCCore_CPU_qconv2d_u8s8s32_1x1, no_padding_3_NCX) {
    REQUIRE_VNNI();
    check_qconv<uint8_t, int8_t, int32_t>(cfg_fwd, 32, 256, 128, 64, 64, 1, 1,
            {2, 3}, {1, 1}, {0, 0}, false, true, true, false);
}
TEST(GCCore_CPU_qconv2d_u8s8s32_1x1, no_padding_3_NXC) {
    REQUIRE_VNNI();
    check_qconv<uint8_t, int8_t, int32_t>(cfg_fwd, 32, 256, 128, 64, 64, 1, 1,
            {2, 3}, {1, 1}, {0, 0}, false, true, false, true);
}

TEST(GCCore_CPU_qconv2d_u8s8s32_3x3, no_padding_1_NCX) {
    REQUIRE_VNNI();
    check_qconv<uint8_t, int8_t, int32_t>(cfg_fwd, 128, 64, 64, 56, 56, 3, 3,
            {1, 1}, {1, 1}, {0, 0}, false, true, true, false);
}
TEST(GCCore_CPU_qconv2d_u8s8s32_3x3, no_padding_1_NXC) {
    REQUIRE_VNNI();
    check_qconv<uint8_t, int8_t, int32_t>(cfg_fwd, 128, 64, 64, 56, 56, 3, 3,
            {1, 1}, {1, 1}, {0, 0}, false, true, false, true);
}
TEST(GCCore_CPU_qconv2d_u8s8s32_3x3, no_padding_2_NCX) {
    REQUIRE_VNNI();
    check_qconv<uint8_t, int8_t, int32_t>(cfg_fwd, 32, 128, 128, 64, 64, 3, 3,
            {2, 1}, {1, 1}, {0, 0}, false, true, true, false);
}
TEST(GCCore_CPU_qconv2d_u8s8s32_3x3, no_padding_2_NXC) {
    REQUIRE_VNNI();
    check_qconv<uint8_t, int8_t, int32_t>(cfg_fwd, 32, 128, 128, 64, 64, 3, 3,
            {2, 1}, {1, 1}, {0, 0}, false, true, false, true);
}
TEST(GCCore_CPU_qconv2d_u8s8s32_3x3, no_padding_3_NCX) {
    REQUIRE_VNNI();
    check_qconv<uint8_t, int8_t, int32_t>(cfg_fwd, 32, 256, 128, 64, 64, 3, 3,
            {2, 3}, {1, 1}, {0, 0}, false, true, true, false);
}
TEST(GCCore_CPU_qconv2d_u8s8s32_3x3, no_padding_3_NXC) {
    REQUIRE_VNNI();
    check_qconv<uint8_t, int8_t, int32_t>(cfg_fwd, 32, 256, 128, 64, 64, 3, 3,
            {2, 3}, {1, 1}, {0, 0}, false, true, false, true);
}

TEST(GCCore_CPU_qconv2d_u8s8s32_3x3, with_padding_1_NCX) {
    REQUIRE_VNNI();
    check_qconv<uint8_t, int8_t, int32_t>(cfg_fwd, 128, 64, 64, 56, 56, 3, 3,
            {1, 2}, {1, 1}, {1, 2}, false, true, true, false);
}

#define conv_padding_support_NXC 0

#if conv_padding_support_NXC
TEST(GCCore_CPU_qconv2d_u8s8s32_3x3, with_padding_1_NXC) {
    REQUIRE_VNNI();
    check_qconv<uint8_t, int8_t, int32_t>(cfg_fwd, 128, 64, 64, 56, 56, 3, 3,
            {1, 2}, {1, 1}, {1, 2}, false, true, false, true);
    check_qconv<uint8_t, int8_t, int32_t>(cfg_fwd, 128, 1, 64, 64, 56, 56, 3, 3,
            {1, 2}, {1, 1}, {1, 2}, {2, 1}, false, true, false, true);
}
#endif
TEST(GCCore_CPU_qconv2d_u8s8s32_3x3, with_padding_2_NCX) {
    REQUIRE_VNNI();
    check_qconv<uint8_t, int8_t, int32_t>(cfg_fwd, 32, 256, 128, 64, 64, 3, 3,
            {2, 3}, {1, 1}, {1, 0}, false, true, true, false);
    check_qconv<uint8_t, int8_t, int32_t>(cfg_fwd, 32, 1, 256, 128, 64, 64, 3,
            3, {2, 3}, {1, 1}, {1, 0}, {0, 1}, false, true, true, false);
}
#if conv_padding_support_NXC
TEST(GCCore_CPU_qconv2d_u8s8s32_3x3, with_padding_2_NXC) {
    REQUIRE_VNNI();
    check_qconv<uint8_t, int8_t, int32_t>(cfg_fwd, 32, 256, 128, 64, 64, 3, 3,
            {2, 3}, {1, 1}, {1, 0}, false, true, false, true);
    check_qconv<uint8_t, int8_t, int32_t>(cfg_fwd, 32, 1, 256, 128, 64, 64, 3,
            3, {2, 3}, {1, 1}, {1, 0}, {0, 1}, false, true, false, true);
}
#endif
// top/middle/bottom padding region, left padding only, no padding, right
// padding only
TEST(GCCore_CPU_qconv2d_u8s8s32_3x3, with_padding_3_NCX) {
    REQUIRE_AMX();
    check_qconv<uint8_t, int8_t, int32_t>(
            conv_fwd_config_t {1, 4, 1, 1, 1, 1, -1, -1}, 1, 1, 4, 6, 6, 3, 3,
            {1, 1}, {1, 1}, {2, 2}, false, true, true, false);
    check_qconv<uint8_t, int8_t, int32_t>(
            conv_fwd_config_t {1, 4, 1, 1, 1, 1, -1, -1}, 1, 1, 1, 4, 6, 6, 3,
            3, {1, 1}, {1, 1}, {2, 2}, {1, 1}, false, true, true, false);
}
TEST(GCCore_CPU_qconv2d_u8s8s32_3x3, with_padding_3_NXC) {
    REQUIRE_AMX();
    check_qconv<uint8_t, int8_t, int32_t>(
            conv_fwd_config_t {1, 4, 1, 1, 1, 1, -1, -1}, 1, 1, 4, 6, 6, 3, 3,
            {1, 1}, {1, 1}, {2, 2}, false, true, false, true);
    check_qconv<uint8_t, int8_t, int32_t>(
            conv_fwd_config_t {1, 4, 1, 1, 1, 1, -1, -1}, 1, 1, 1, 4, 6, 6, 3,
            3, {1, 1}, {1, 1}, {2, 2}, {1, 1}, false, true, false, true);
}
// top/middle/bottom padding region, left and right padding
TEST(GCCore_CPU_qconv2d_u8s8s32_3x3, with_padding_4_NCX) {
    REQUIRE_AMX();
    check_qconv<uint8_t, int8_t, int32_t>(
            conv_fwd_config_t {1, 4, 1, 1, 6, 6, -1, -1}, 1, 1, 4, 6, 6, 3, 3,
            {1, 1}, {1, 1}, {1, 1}, false, true, true, false);
    check_qconv<uint8_t, int8_t, int32_t>(
            conv_fwd_config_t {1, 4, 1, 1, 6, 6, -1, -1}, 1, 1, 1, 4, 6, 6, 3,
            3, {1, 1}, {1, 1}, {1, 1}, {0, 0}, false, true, true, false);
}
TEST(GCCore_CPU_qconv2d_u8s8s32_3x3, with_padding_4_NXC) {
    REQUIRE_AMX();
    check_qconv<uint8_t, int8_t, int32_t>(
            conv_fwd_config_t {1, 4, 1, 1, 6, 6, -1, -1}, 1, 1, 4, 6, 6, 3, 3,
            {1, 1}, {1, 1}, {1, 1}, false, true, false, true);
    check_qconv<uint8_t, int8_t, int32_t>(
            conv_fwd_config_t {1, 4, 1, 1, 6, 6, -1, -1}, 1, 1, 1, 4, 6, 6, 3,
            3, {1, 1}, {1, 1}, {1, 1}, {2, 2}, false, true, false, true);
}
// top/middle/bottom padding region, left padding only, right padding not
// being used
TEST(GCCore_CPU_qconv2d_u8s8s32_3x3, with_padding_5_NCX) {
    REQUIRE_AMX();
    check_qconv<uint8_t, int8_t, int32_t>(cfg_fwd, 1, 1, 4, 4, 4, 3, 3, {2, 2},
            {1, 1}, {1, 1}, false, true, true, false);
    check_qconv<uint8_t, int8_t, int32_t>(cfg_fwd, 1, 1, 1, 4, 4, 4, 3, 3,
            {2, 2}, {1, 1}, {1, 1}, {2, 2}, false, true, true, false);
}
TEST(GCCore_CPU_qconv2d_u8s8s32_3x3, with_padding_5_NXC) {
    REQUIRE_AMX();
    check_qconv<uint8_t, int8_t, int32_t>(cfg_fwd, 1, 1, 4, 4, 4, 3, 3, {2, 2},
            {1, 1}, {1, 1}, false, true, false, true);
    check_qconv<uint8_t, int8_t, int32_t>(cfg_fwd, 1, 1, 1, 4, 4, 4, 3, 3,
            {2, 2}, {1, 1}, {1, 1}, {2, 2}, false, true, false, true);
}

TEST(GCCore_CPU_qconv2d_u8s8s32_3x3_with_groups, no_padding_1_NCX) {
    REQUIRE_AMX();
    check_qconv<uint8_t, int8_t, int32_t>(conv_fwd_config_t(), 1, 4, 8, 8, 12,
            12, 3, 3, {2, 2}, {1, 1}, {0, 0}, {0, 0}, false, true, false,
            false);
}
TEST(GCCore_CPU_qconv2d_u8s8s32_3x3_with_groups, no_padding_1_NXC) {
    REQUIRE_AMX();
    check_qconv<uint8_t, int8_t, int32_t>(conv_fwd_config_t(), 1, 4, 8, 8, 12,
            12, 3, 3, {2, 2}, {1, 1}, {0, 0}, {0, 0}, false, true, false, true);
}
// os-blocking cases
TEST(GCCore_CPU_qconv2d_u8s8s32_3x3_with_groups, no_padding_2_NCX) {
    REQUIRE_AMX();
    check_qconv<uint8_t, int8_t, int32_t>(conv_fwd_config_t(), 14, 2, 48, 48,
            28, 28, 3, 3, {1, 1}, {1, 1}, {0, 0}, {0, 0}, false, true, false,
            false);
}
TEST(GCCore_CPU_qconv2d_u8s8s32_3x3_with_groups, no_padding_2_NXC) {
    REQUIRE_AMX();
    check_qconv<uint8_t, int8_t, int32_t>(conv_fwd_config_t(), 14, 2, 48, 48,
            28, 28, 3, 3, {1, 1}, {1, 1}, {0, 0}, {0, 0}, false, true, false,
            true);
}
TEST(GCCore_CPU_qconv2d_u8s8s32_3x3_with_groups, with_padding_1_NCX) {
    REQUIRE_AMX();
    check_qconv<uint8_t, int8_t, int32_t>(conv_fwd_config_t(), 1, 2, 8, 8, 12,
            12, 3, 3, {2, 2}, {1, 1}, {1, 1}, {1, 1}, false, true, false,
            false);
}
TEST(GCCore_CPU_qconv2d_u8s8s32_3x3_with_groups, with_padding_1_NXC) {
    REQUIRE_AMX();
    check_qconv<uint8_t, int8_t, int32_t>(conv_fwd_config_t(), 1, 2, 8, 8, 12,
            12, 3, 3, {2, 2}, {1, 1}, {1, 1}, {1, 1}, false, true, false, true);
}
TEST(GCCore_CPU_qconv2d_u8s8s32_3x3_with_groups, with_padding_2_NCX) {
    REQUIRE_AMX();
    check_qconv<uint8_t, int8_t, int32_t>(conv_fwd_config_t(), 14, 2, 48, 48,
            114, 114, 3, 3, {2, 2}, {1, 1}, {1, 1}, {1, 1}, false, true, false,
            false);
}
TEST(GCCore_CPU_qconv2d_u8s8s32_3x3_with_groups, with_padding_2_NXC) {
    REQUIRE_AMX();
    check_qconv<uint8_t, int8_t, int32_t>(conv_fwd_config_t(), 14, 2, 48, 48,
            114, 114, 3, 3, {2, 2}, {1, 1}, {1, 1}, {1, 1}, false, true, false,
            true);
}

TEST(GCCore_CPU_qconv2d_nested_u8s8s32_3x3, rn50_stage1_NCX) {
    REQUIRE_AMX();
    check_nested_qconv<int8_t, int8_t, int32_t>(nested_conv_fwd_config_t(), 12,
            64, 64, 58, 58, 3, 3, {1, 1}, {0, 0}, false, true, true, false);
}

TEST(GCCore_CPU_qconv2d_nested_u8s8s32_3x3, rn50_stage1_NXC) {
    REQUIRE_AMX();
    check_nested_qconv<int8_t, int8_t, int32_t>(nested_conv_fwd_config_t(), 12,
            64, 64, 58, 58, 3, 3, {1, 1}, {0, 0}, false, true, false, true);
}

TEST(GCCore_CPU_qconv2d_nested_u8s8s32_3x3, rn50_stage2_NCX) {
    REQUIRE_AMX();
    check_nested_qconv<int8_t, int8_t, int32_t>(nested_conv_fwd_config_t(), 12,
            128, 128, 30, 30, 3, 3, {1, 1}, {0, 0}, false, true, true, false);
}

TEST(GCCore_CPU_qconv2d_nested_u8s8s32_3x3, rn50_stage2_NXC) {
    REQUIRE_AMX();
    check_nested_qconv<int8_t, int8_t, int32_t>(nested_conv_fwd_config_t(), 12,
            128, 128, 30, 30, 3, 3, {1, 1}, {0, 0}, false, true, false, true);
}

TEST(GCCore_CPU_qconv2d_nested_u8s8s32_3x3, rn50_stage3_NCX) {
    REQUIRE_AMX();
    check_nested_qconv<int8_t, int8_t, int32_t>(nested_conv_fwd_config_t(), 12,
            256, 256, 16, 16, 3, 3, {1, 1}, {0, 0}, false, true, true, false);
}

TEST(GCCore_CPU_qconv2d_nested_u8s8s32_3x3, rn50_stage3_NXC) {
    REQUIRE_AMX();
    check_nested_qconv<int8_t, int8_t, int32_t>(nested_conv_fwd_config_t(), 12,
            256, 256, 16, 16, 3, 3, {1, 1}, {0, 0}, false, true, false, true);
}

TEST(GCCore_CPU_qconv2d_nested_u8s8s32_3x3, rn50_stage4_NCX) {
    REQUIRE_AMX();
    check_nested_qconv<int8_t, int8_t, int32_t>(nested_conv_fwd_config_t(), 12,
            512, 512, 9, 9, 3, 3, {1, 1}, {0, 0}, false, true, true, false);
}

TEST(GCCore_CPU_qconv2d_nested_u8s8s32_3x3, rn50_stage4_NXC) {
    REQUIRE_AMX();
    check_nested_qconv<int8_t, int8_t, int32_t>(nested_conv_fwd_config_t(), 12,
            512, 512, 9, 9, 3, 3, {1, 1}, {0, 0}, false, true, false, true);
}

TEST(GCCore_CPU_qconv2d_nested_u8s8s32_1x1, rn50_stage4_NXC) {
    REQUIRE_AMX();
    check_nested_qconv<uint8_t, int8_t, int32_t>(nested_conv_fwd_config_t(), 12,
            512, 512, 56, 56, 1, 1, {1, 1}, {0, 0}, false, true, false, true);
}

TEST(GCCore_CPU_qconv2d_nested_u8s8s32_3x3, oob_rn50_conv_NXC) {
    SET_THREADS_OR_SKIP(4);
    REQUIRE_AMX();
    check_nested_qconv<int8_t, int8_t, int32_t>(nested_conv_fwd_config_t(), 1,
            512, 512, 21, 21, 3, 3, {2, 2}, {0, 0}, false, true, false, true);
}

TEST(GCCore_CPU_dynamic_qconv2d_nested_u8s8s32_1x1, ut1) {
    REQUIRE_AMX();
    check_dynamic_netsed_qconv<uint8_t, int8_t, int32_t>(
            nested_conv_fwd_config_t(), -1, 512, 512, -1, -1, 1, 1, {1, 1},
            {0, 0}, false, true, false, true, /*real_N*/ 1, /*real_H*/ 56,
            /*real_W*/ 56);
}

TEST(GCCore_CPU_dynamic_qconv2d_nested_u8s8s32_1x1, ut2) {
    REQUIRE_AMX();
    check_dynamic_netsed_qconv<uint8_t, int8_t, int32_t>(
            nested_conv_fwd_config_t(), -1, 512, 512, -1, -1, 1, 1, {1, 1},
            {0, 0}, false, true, false, true, /*real_N*/ 1, /*real_H*/ 55,
            /*real_W*/ 55);
}

TEST(GCCore_CPU_dynamic_qconv2d_nested_u8s8s32_1x1, ut3) {
    REQUIRE_AMX();
    check_dynamic_netsed_qconv<uint8_t, int8_t, int32_t>(
            nested_conv_fwd_config_t(), -1, 512, 512, -1, -1, 1, 1, {1, 1},
            {0, 0}, false, true, false, true, /*real_N*/ 1, /*real_H*/ 67,
            /*real_W*/ 67);
}

TEST(GCCore_CPU_dynamic_qconv2d_nested_u8s8s32_3x3, no_padding_ut1) {
    REQUIRE_VNNI();
    check_dynamic_netsed_qconv<uint8_t, int8_t, int32_t>(
            nested_conv_fwd_config_t(), -1, 256, 64, 58, 58, 3, 3, {1, 1},
            {0, 0}, false, true, false, true, /*real_N*/ 1, /*real_H*/ 58,
            /*real_W*/ 58);
}

TEST(GCCore_CPU_dynamic_qconv2d_nested_u8s8s32_3x3, no_padding_ut2) {
    REQUIRE_VNNI();
    check_dynamic_netsed_qconv<uint8_t, int8_t, int32_t>(
            nested_conv_fwd_config_t(), -1, 256, 64, -1, -1, 3, 3, {1, 1},
            {0, 0}, false, true, false, true, /*real_N*/ 1, /*real_H*/ 58,
            /*real_W*/ 58);
}

TEST(GCCore_CPU_dynamic_qconv2d_nested_u8s8s32_3x3, no_padding_ut3) {
    REQUIRE_VNNI();
    check_dynamic_netsed_qconv<uint8_t, int8_t, int32_t>(
            nested_conv_fwd_config_t(), -1, 256, 64, -1, -1, 3, 3, {1, 1},
            {0, 0}, false, true, false, true, /*real_N*/ 8, /*real_H*/ 69,
            /*real_W*/ 69);
}
TEST(GCCore_CPU_dynamic_qconv2d_nested_u8s8s32_3x3, no_padding_ut4) {
    REQUIRE_VNNI();
    check_dynamic_netsed_qconv<uint8_t, int8_t, int32_t>(
            nested_conv_fwd_config_t(), -1, 256, 64, -1, -1, 3, 3, {1, 1},
            {0, 0}, false, true, false, true, /*real_N*/ 1, /*real_H*/ 9,
            /*real_W*/ 9);
}
TEST(GCCore_CPU_dynamic_qconv2d_nested_u8s8s32_3x3, no_padding_ut5) {
    REQUIRE_VNNI();
    check_dynamic_netsed_qconv<uint8_t, int8_t, int32_t>(
            nested_conv_fwd_config_t(), -1, 256, 64, -1, -1, 3, 3, {1, 1},
            {0, 0}, false, true, false, true, /*real_N*/ 1, /*real_H*/ 6,
            /*real_W*/ 6);
}

TEST(GCCore_CPU_dynamic_qconv2d_nested_u8s8s32_3x3, no_padding_ut6) {
    REQUIRE_VNNI();
    check_dynamic_netsed_qconv<uint8_t, int8_t, int32_t>(
            nested_conv_fwd_config_t(), -1, 256, 64, 58, 58, 3, 3, {2, 2},
            {0, 0}, false, true, false, true, /*real_N*/ 1, /*real_H*/ 58,
            /*real_W*/ 58);
}

TEST(GCCore_CPU_dynamic_qconv2d_nested_u8s8s32_3x3, no_padding_ut7) {
    REQUIRE_VNNI();
    check_dynamic_netsed_qconv<uint8_t, int8_t, int32_t>(
            nested_conv_fwd_config_t(), -1, 256, 64, -1, -1, 3, 3, {2, 2},
            {0, 0}, false, true, false, true, /*real_N*/ 8, /*real_H*/ 69,
            /*real_W*/ 69);
}

TEST(GCCore_CPU_dynamic_qconv2d_nested_u8s8s32_3x3, no_padding_ut8) {
    REQUIRE_VNNI();
    check_dynamic_netsed_qconv<uint8_t, int8_t, int32_t>(
            nested_conv_fwd_config_t(), -1, 256, 64, -1, -1, 3, 3, {3, 2},
            {0, 0}, false, true, false, true, /*real_N*/ 8, /*real_H*/ 69,
            /*real_W*/ 69);
}

TEST(GCCore_CPU_dynamic_qconv2d_nested_u8s8s32_3x3, padding_ut1) {
    REQUIRE_VNNI();
    check_dynamic_netsed_qconv<uint8_t, int8_t, int32_t>(
            nested_conv_fwd_config_t(), -1, 256, 64, 56, 56, 3, 3, {1, 1},
            {1, 1}, false, true, false, true, /*real_N*/ 8, /*real_H*/ 56,
            /*real_W*/ 56);
}

TEST(GCCore_CPU_dynamic_qconv2d_nested_u8s8s32_3x3, padding_ut2) {
    REQUIRE_VNNI();
    check_dynamic_netsed_qconv<uint8_t, int8_t, int32_t>(
            nested_conv_fwd_config_t(), -1, 256, 64, -1, -1, 3, 3, {1, 1},
            {1, 1}, false, true, false, true, /*real_N*/ 8, /*real_H*/ 56,
            /*real_W*/ 56);
}

TEST(GCCore_CPU_dynamic_qconv2d_nested_u8s8s32_3x3, padding_ut3) {
    REQUIRE_VNNI();
    check_dynamic_netsed_qconv<uint8_t, int8_t, int32_t>(
            nested_conv_fwd_config_t(), -1, 256, 64, -1, -1, 3, 3, {1, 1},
            {1, 1}, false, true, false, true, /*real_N*/ 8, /*real_H*/ 67,
            /*real_W*/ 67);
}
TEST(GCCore_CPU_dynamic_qconv2d_nested_u8s8s32_3x3, padding_ut4) {
    REQUIRE_VNNI();
    check_dynamic_netsed_qconv<uint8_t, int8_t, int32_t>(
            nested_conv_fwd_config_t(), -1, 256, 64, -1, -1, 3, 3, {1, 1},
            {1, 1}, false, true, false, true, /*real_N*/ 8, /*real_H*/ 67,
            /*real_W*/ 56);
}
TEST(GCCore_CPU_dynamic_qconv2d_nested_u8s8s32_3x3, padding_ut5) {
    REQUIRE_VNNI();
    check_dynamic_netsed_qconv<uint8_t, int8_t, int32_t>(
            nested_conv_fwd_config_t(), -1, 256, 64, -1, -1, 3, 3, {1, 1},
            {2, 2}, false, true, false, true, /*real_N*/ 8, /*real_H*/ 56,
            /*real_W*/ 56);
}

TEST(GCCore_CPU_dynamic_qconv2d_nested_u8s8s32_3x3, padding_ut6) {
    REQUIRE_VNNI();
    check_dynamic_netsed_qconv<uint8_t, int8_t, int32_t>(
            nested_conv_fwd_config_t(), -1, 256, 64, -1, -1, 3, 3, {1, 1},
            {1, 1}, false, true, false, true, /*real_N*/ 8, /*real_H*/ 7,
            /*real_W*/ 7);
}

TEST(GCCore_CPU_dynamic_qconv2d_nested_u8s8s32_3x3, padding_ut7) {
    REQUIRE_VNNI();
    check_dynamic_netsed_qconv<uint8_t, int8_t, int32_t>(
            nested_conv_fwd_config_t(), -1, 256, 64, -1, -1, 3, 3, {2, 2},
            {1, 1}, false, true, false, true, /*real_N*/ 8, /*real_H*/ 20,
            /*real_W*/ 20);
}

TEST(GCCore_CPU_dynamic_qconv2d_nested_u8s8s32_3x3, padding_ut8) {
    REQUIRE_VNNI();
    check_dynamic_netsed_qconv<uint8_t, int8_t, int32_t>(
            nested_conv_fwd_config_t(), -1, 256, 256, 12, 12, 3, 3, {3, 3},
            {1, 1}, false, true, false, true, /*real_N*/ 1, /*real_H*/ 12,
            /*real_W*/ 12);
}

/* rl conv with padding */
TEST(GCCore_CPU_qconv2d_u8s8s32_rl, padding_1) {
    // single real_pr
    REQUIRE_AMX();
    check_rl_qconv<uint8_t, int8_t, int32_t>(conv_fwd_rl_config_t(), 1, 1, 64,
            3, 224, 224, 7, 7, {2, 2}, {1, 1}, {3, 3}, {3, 3}, false, true);
    check_rl_qconv<uint8_t, int8_t, int32_t>(conv_fwd_rl_config_t(), 1, 1, 64,
            3, 224, 224, 7, 7, {2, 2}, {1, 1}, {3, 3}, {2, 2}, false, true);
}
TEST(GCCore_CPU_qconv2d_u8s8s32_rl, padding_2) {
    // double real_pr
    REQUIRE_AMX();
    check_rl_qconv<uint8_t, int8_t, int32_t>(conv_fwd_rl_config_t(), 1, 1, 16,
            3, 12, 12, 5, 5, {2, 2}, {1, 1}, {4, 4}, {4, 4}, false, true);
    check_rl_qconv<uint8_t, int8_t, int32_t>(conv_fwd_rl_config_t(), 1, 1, 16,
            3, 12, 12, 5, 5, {2, 2}, {1, 1}, {4, 4}, {3, 3}, false, true);
}
TEST(GCCore_CPU_qconv2d_u8s8s32_rl, padding_3) {
    REQUIRE_AMX();
    check_rl_qconv<uint8_t, int8_t, int32_t>(conv_fwd_rl_config_t(), 1, 1, 16,
            3, 16, 16, 7, 7, {2, 2}, {1, 1}, {4, 3}, {4, 3}, false, true);
    check_rl_qconv<uint8_t, int8_t, int32_t>(conv_fwd_rl_config_t(), 1, 1, 16,
            3, 16, 16, 7, 7, {2, 2}, {1, 1}, {4, 3},
            {
                    3,
                    4,
            },
            false, true);
}
// top/middle/bottom padding region, left padding only, no padding, right
// padding only
TEST(GCCore_CPU_qconv2d_u8s8s32_rl, padding_4) {
    REQUIRE_AMX();
    check_rl_qconv<uint8_t, int8_t, int32_t>(conv_fwd_rl_config_t {1, 1}, 1, 1,
            1, 4, 6, 6, 3, 3, {1, 1}, {1, 1}, {2, 2}, {2, 2}, false, true);
    check_rl_qconv<uint8_t, int8_t, int32_t>(conv_fwd_rl_config_t {1, 1}, 1, 1,
            1, 4, 6, 6, 3, 3, {1, 1}, {1, 1}, {2, 2}, {1, 1}, false, true);
}
// top/middle/bottom padding region, left and right padding
TEST(GCCore_CPU_qconv2d_u8s8s32_rl, padding_5) {
    REQUIRE_AMX();
    check_rl_qconv<uint8_t, int8_t, int32_t>(conv_fwd_rl_config_t {1, 1}, 1, 1,
            1, 4, 6, 6, 3, 3, {1, 1}, {1, 1}, {1, 1}, {1, 1}, false, true);
    check_rl_qconv<uint8_t, int8_t, int32_t>(conv_fwd_rl_config_t {1, 1}, 1, 1,
            1, 4, 6, 6, 3, 3, {1, 1}, {1, 1}, {1, 1}, {2, 2}, false, true);
}
// top/middle/bottom padding region, left padding only, right padding not
// being used
TEST(GCCore_CPU_qconv2d_u8s8s32_rl, padding_6) {
    REQUIRE_AMX();
    check_rl_qconv<uint8_t, int8_t, int32_t>(conv_fwd_rl_config_t {1, 1}, 1, 1,
            1, 4, 4, 4, 3, 3, {2, 2}, {1, 1}, {1, 1}, {1, 1}, false, true);
    check_rl_qconv<uint8_t, int8_t, int32_t>(conv_fwd_rl_config_t {1, 1}, 1, 1,
            1, 4, 4, 4, 3, 3, {2, 2}, {1, 1}, {1, 1}, {2, 2}, false, true);
}

TEST(GCCore_CPU_qconv2d_u8s8s32_rl, padding_7) {
    REQUIRE_AMX();
    check_rl_qconv<uint8_t, int8_t, int32_t>(conv_fwd_rl_config_t(), 1, 1, 16,
            1, 12, 12, 7, 7, {2, 2}, {1, 1}, {1, 1}, {1, 1}, false, true);
    check_rl_qconv<uint8_t, int8_t, int32_t>(conv_fwd_rl_config_t(), 1, 1, 16,
            1, 12, 12, 7, 7, {2, 2}, {1, 1}, {1, 1}, {0, 0}, false, true);
}
TEST(GCCore_CPU_qconv2d_u8s8s32_rl, padding_8) {
    REQUIRE_AMX();
    // specify num_threads(4) to cover parallel at width axis
    SET_THREADS_OR_SKIP(4);
    check_rl_qconv<uint8_t, int8_t, int32_t>(conv_fwd_rl_config_t(), 1, 1, 16,
            3, 22, 22, 3, 3, {2, 2}, {1, 1}, {1, 2}, {1, 2}, false, true);
    check_rl_qconv<uint8_t, int8_t, int32_t>(conv_fwd_rl_config_t(), 1, 1, 16,
            3, 22, 22, 3, 3, {2, 2}, {1, 1}, {1, 2}, {2, 1}, false, true);
}
TEST(GCCore_CPU_qconv2d_u8s8s32_rl, padding_9) {
    REQUIRE_AMX();
    // specify num_threads(4) to cover parallel at batch axis
    SET_THREADS_OR_SKIP(4);
    check_rl_qconv<uint8_t, int8_t, int32_t>(conv_fwd_rl_config_t(), 4, 1, 16,
            3, 22, 22, 3, 3, {2, 2}, {1, 1}, {1, 2}, {1, 2}, false, true);
    check_rl_qconv<uint8_t, int8_t, int32_t>(conv_fwd_rl_config_t(), 4, 1, 16,
            3, 22, 22, 3, 3, {2, 2}, {1, 1}, {1, 2}, {2, 1}, false, true);
}
TEST(GCCore_CPU_qconv2d_u8s8s32_rl, padding_10) {
    REQUIRE_AMX();
    // specify num_threads(4) to cover parallel at width axis with pads >
    // strides
    SET_THREADS_OR_SKIP(4);
    check_rl_qconv<uint8_t, int8_t, int32_t>(conv_fwd_rl_config_t(), 1, 1, 16,
            3, 5, 5, 5, 5, {2, 2}, {1, 1}, {3, 3}, {3, 3}, false, true);
    check_rl_qconv<uint8_t, int8_t, int32_t>(conv_fwd_rl_config_t(), 1, 1, 16,
            3, 5, 5, 5, 5, {2, 2}, {1, 1}, {3, 3}, {2, 2}, false, true);
}

TEST(GCCore_CPU_qconv2d_u8s8s32_rl_bias, padding_1) {
    REQUIRE_AMX();
    check_rl_qconv<uint8_t, int8_t, int32_t>(conv_fwd_rl_config_t(), 1, 1, 16,
            3, 12, 12, 5, 5, {2, 2}, {1, 1}, {4, 4}, {4, 4}, true, true);
    check_rl_qconv<uint8_t, int8_t, int32_t>(conv_fwd_rl_config_t(), 1, 1, 16,
            3, 12, 12, 5, 5, {2, 2}, {1, 1}, {4, 4}, {3, 3}, true, true);
}

/* rl conv without padding */
TEST(GCCore_CPU_qconv2d_u8s8s32_rl, no_padding_1) {
    REQUIRE_AMX();
    check_rl_qconv<uint8_t, int8_t, int32_t>(conv_fwd_rl_config_t(), 1, 1, 64,
            3, 230, 230, 7, 7, {2, 2}, {1, 1}, {0, 0}, {0, 0}, false, true);

    // specify odd num_threads to cover different parallelism at width axis
    SET_THREADS_OR_SKIP(7);
    check_rl_qconv<uint8_t, int8_t, int32_t>(conv_fwd_rl_config_t(), 1, 1, 64,
            3, 230, 230, 7, 7, {2, 2}, {1, 1}, {0, 0}, {0, 0}, false, true);
}
TEST(GCCore_CPU_qconv2d_u8s8s32_rl, no_padding_2) {
    REQUIRE_AMX();
    check_rl_qconv<uint8_t, int8_t, int32_t>(conv_fwd_rl_config_t(), 1, 1, 64,
            3, 17, 17, 7, 7, {2, 2}, {1, 1}, {0, 0}, {0, 0}, false, true);
}
TEST(GCCore_CPU_qconv2d_u8s8s32_rl, no_padding_3) {
    REQUIRE_AMX();
    check_rl_qconv<uint8_t, int8_t, int32_t>(conv_fwd_rl_config_t(), 8, 1, 64,
            3, 16, 16, 7, 7, {2, 2}, {1, 1}, {0, 0}, {0, 0}, false, true);
}
TEST(GCCore_CPU_qconv2d_u8s8s32_rl, no_padding_4) {
    REQUIRE_AMX();
    check_rl_qconv<uint8_t, int8_t, int32_t>(conv_fwd_rl_config_t(), 1, 1, 64,
            3, 12, 12, 7, 7, {2, 2}, {1, 1}, {0, 0}, {0, 0}, false, true);
}
TEST(GCCore_CPU_qconv2d_u8s8s32_rl, no_padding_5) {
    REQUIRE_AMX();
    check_rl_qconv<uint8_t, int8_t, int32_t>(conv_fwd_rl_config_t(), 1, 1, 16,
            3, 13, 13, 7, 7, {2, 2}, {1, 1}, {0, 0}, {0, 0}, false, true);
}

/*  rl conv with groups */
TEST(GCCore_CPU_qconv2d_u8s8s32_rl_with_groups, no_padding_1) {
    REQUIRE_AMX();
    check_rl_qconv<uint8_t, int8_t, int32_t>(conv_fwd_rl_config_t(), 1, 4, 8, 8,
            12, 12, 3, 3, {2, 2}, {1, 1}, {0, 0}, {0, 0}, false, true);
}
TEST(GCCore_CPU_qconv2d_u8s8s32_rl_with_groups, no_padding_2) {
    REQUIRE_AMX();
    check_rl_qconv<uint8_t, int8_t, int32_t>(conv_fwd_rl_config_t(), 14, 4, 48,
            48, 28, 28, 3, 3, {1, 1}, {1, 1}, {0, 0}, {0, 0}, false, true);
}
TEST(GCCore_CPU_qconv2d_u8s8s32_rl_with_groups, no_padding_3) {
    REQUIRE_AMX();
    check_rl_qconv<uint8_t, int8_t, int32_t>(conv_fwd_rl_config_t(), 1, 2, 8, 8,
            14, 14, 3, 3, {2, 2}, {1, 1}, {0, 0}, {0, 0}, false, true);
}
TEST(GCCore_CPU_qconv2d_u8s8s32_rl_with_groups, no_padding_4) {
    REQUIRE_AMX();
    check_rl_qconv<uint8_t, int8_t, int32_t>(conv_fwd_rl_config_t(), 14, 4, 36,
            36, 116, 116, 5, 5, {2, 2}, {1, 1}, {0, 0}, {0, 0}, false, true);
}
TEST(GCCore_CPU_qconv2d_u8s8s32_rl_with_groups, with_padding_1) {
    REQUIRE_AMX();
    check_rl_qconv<uint8_t, int8_t, int32_t>(conv_fwd_rl_config_t(), 1, 2, 8, 8,
            12, 12, 3, 3, {2, 2}, {1, 1}, {1, 1}, {1, 1}, false, true);
}
TEST(GCCore_CPU_qconv2d_u8s8s32_rl_with_groups, with_padding_2) {
    REQUIRE_AMX();
    check_rl_qconv<uint8_t, int8_t, int32_t>(conv_fwd_rl_config_t(), 14, 4, 36,
            36, 114, 114, 5, 5, {2, 2}, {1, 1}, {1, 1}, {1, 1}, false, true);
}

/* kl lowering without padding */
TEST(GCCore_CPU_qconv2d_u8s8s32_kl, no_padding_1) {
    REQUIRE_AMX();
    check_rl_qconv<uint8_t, int8_t, int32_t>(conv_fwd_rl_config_t(), 1, 1, 16,
            32, 14, 14, 3, 3, {2, 2}, {1, 1}, {0, 0}, {0, 0}, false, true);
}
TEST(GCCore_CPU_qconv2d_u8s8s32_kl, no_padding_2) {
    REQUIRE_AMX();
    check_rl_qconv<uint8_t, int8_t, int32_t>(conv_fwd_rl_config_t(), 1, 1, 16,
            24, 13, 13, 3, 3, {2, 2}, {1, 1}, {0, 0}, {0, 0}, false, true);
}
TEST(GCCore_CPU_qconv2d_u8s8s32_kl, no_padding_3) {
    REQUIRE_AMX();
    check_rl_qconv<uint8_t, int8_t, int32_t>(conv_fwd_rl_config_t(), 1, 1, 16,
            12, 13, 13, 5, 5, {1, 1}, {1, 1}, {0, 0}, {0, 0}, false, true);
}
/* kl lowering with padding */
TEST(GCCore_CPU_qconv2d_u8s8s32_kl, padding_1) {
    REQUIRE_AMX();
    check_rl_qconv<uint8_t, int8_t, int32_t>(conv_fwd_rl_config_t(), 1, 1, 16,
            32, 13, 13, 3, 3, {2, 2}, {1, 1}, {1, 1}, {1, 1}, false, true);
}
TEST(GCCore_CPU_qconv2d_u8s8s32_kl, padding_2) {
    REQUIRE_AMX();
    check_rl_qconv<uint8_t, int8_t, int32_t>(conv_fwd_rl_config_t(), 1, 1, 16,
            24, 13, 13, 3, 3, {2, 2}, {1, 1}, {2, 3}, {3, 2}, false, true);
}
TEST(GCCore_CPU_qconv2d_u8s8s32_kl, padding_3) {
    REQUIRE_AMX();
    check_rl_qconv<uint8_t, int8_t, int32_t>(conv_fwd_rl_config_t(), 1, 1, 16,
            12, 13, 13, 5, 5, {1, 1}, {1, 1}, {4, 4}, {4, 4}, false, true);
}
/* kl lowering with group and without padding */
TEST(GCCore_CPU_qconv2d_u8s8s32_kl_with_groups, no_padding_1) {
    REQUIRE_AMX();
    check_rl_qconv<uint8_t, int8_t, int32_t>(conv_fwd_rl_config_t(), 1, 2, 16,
            64, 14, 14, 3, 3, {2, 2}, {1, 1}, {0, 0}, {0, 0}, false, true);
}
TEST(GCCore_CPU_qconv2d_u8s8s32_kl_with_groups, no_padding_2) {
    REQUIRE_AMX();
    check_rl_qconv<uint8_t, int8_t, int32_t>(conv_fwd_rl_config_t(), 1, 3, 18,
            72, 13, 13, 3, 3, {2, 2}, {1, 1}, {0, 0}, {0, 0}, false, true);
}
TEST(GCCore_CPU_qconv2d_u8s8s32_kl_with_groups, no_padding_3) {
    REQUIRE_AMX();
    check_rl_qconv<uint8_t, int8_t, int32_t>(conv_fwd_rl_config_t(), 1, 2, 16,
            24, 13, 13, 5, 5, {1, 1}, {1, 1}, {0, 0}, {0, 0}, false, true);
}
/* kl lowering with group and padding */
TEST(GCCore_CPU_qconv2d_u8s8s32_kl_with_groups, padding_1) {
    REQUIRE_AMX();
    check_rl_qconv<uint8_t, int8_t, int32_t>(conv_fwd_rl_config_t(), 1, 2, 16,
            64, 13, 13, 3, 3, {2, 2}, {1, 1}, {1, 1}, {1, 1}, false, true);
}
TEST(GCCore_CPU_qconv2d_u8s8s32_kl_with_groups, padding_2) {
    REQUIRE_AMX();
    check_rl_qconv<uint8_t, int8_t, int32_t>(conv_fwd_rl_config_t(), 1, 4, 16,
            96, 13, 13, 3, 3, {2, 2}, {1, 1}, {2, 3}, {3, 2}, false, true);
}
TEST(GCCore_CPU_qconv2d_u8s8s32_kl_with_groups, padding_3) {
    REQUIRE_AMX();
    check_rl_qconv<uint8_t, int8_t, int32_t>(conv_fwd_rl_config_t(), 1, 3, 18,
            36, 13, 13, 5, 5, {1, 1}, {1, 1}, {4, 4}, {4, 4}, false, true);
}
