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
#include "gtest/gtest.h"
#include <compiler/ir/graph/driver.hpp>
#include <compiler/ir/graph/fusible_op.hpp>
#include <compiler/ir/graph/lowering.hpp>
#include <compiler/ir/graph/pass/pass.hpp>
#include <compiler/ir/graph/transform/transform.hpp>
#include <compiler/jit/jit.hpp>
#include <ops/convolution.hpp>
#include <ops/templates/conv_fwd.hpp>
#include <util/reflection.hpp>
using namespace dnnl::impl::graph::gc;

using conv_fwd_config_t = ops::conv_fwd_config_t;

static inline graph_tensor_ptr make_tensor(const sc_dims &shape,
        const sc_data_type_t dtype = datatypes::f32,
        const sc_data_format_t &fmt = sc_data_format_t()) {
    return std::make_shared<graph_tensor>(nullptr, fmt, shape, dtype);
}

template <typename src_type, typename wei_type, typename dst_type>
void check_conv_fwd_correctness(conv_fwd_config_t cfg,
        const sc_dims &input_dims, const sc_dims &weight_dims,
        const sc_dims &stride, const sc_dims &padding, bool fuse_bias = false,
        bool default_cfg = false) {
    REQUIRE_AVX2();
    COMPILE_ASSERT(input_dims.size() == 5,
            "input_dims is expected to be 5D tensor, but got "
                    << input_dims.size() << "D.");
    COMPILE_ASSERT(weight_dims.size() == 5,
            "weight_dims is expected to be 5D tensor, but got "
                    << weight_dims.size() << "D.");
    int mb = input_dims[0], ic = input_dims[1], id = input_dims[2],
        ih = input_dims[3], iw = input_dims[4];
    int oc = weight_dims[0], kd = weight_dims[2], kh = weight_dims[3],
        kw = weight_dims[4];

    int sd = stride[0], sh = stride[0], sw = stride[0];
    if (stride.size() > 1) {
        COMPILE_ASSERT(stride.size() == 3,
                "stride is expected to be 3D tensor, but got " << stride.size()
                                                               << "D.");
        sh = stride[1];
        sw = stride[2];
    }
    int pd = padding[0], ph = padding[0], pw = padding[0];
    if (padding.size() > 1) {
        COMPILE_ASSERT(padding.size() == 3,
                "padding is expected to be 3D tensor, but got "
                        << padding.size() << "D.");
        ph = padding[1];
        pw = padding[2];
    }

    sc_graph_t g;
    auto src_dtype = sc_data_traits_t<src_type>::type();
    auto wei_dtype = sc_data_traits_t<wei_type>::type();
    std::vector<sc_op_ptr> fuse_arg_ops;
    auto in_data = g.make_input({make_tensor(input_dims, src_dtype)});
    auto in_weight = g.make_input({make_tensor(weight_dims, wei_dtype)});
    auto conv_out = g.make("conv_fwd_core",
            {in_data->get_outputs()[0], in_weight->get_outputs()[0]}, {},
            {{"strides", stride}, {"paddings", padding}});
    auto tunop = conv_out->template dyn_cast<tunable_op_t>();
    int od = 0, oh = 0, ow = 0;
    {
        auto gen = tunop->create_generator();
        auto conv_gen = (ops::gen_conv_fwd_t *)gen.get();
        std::tie(od, oh, ow) = conv_gen->get_output_shape();
        reflection::shared_general_object_t cfgptr;
        if (!default_cfg) {
            cfgptr = reflection::general_object_t::make(cfg);
        } else {
            cfgptr = gen->get_default_config(get_default_context());
            cfg = *(conv_fwd_config_t *)cfgptr.get();
        }
        tunop->set_config(cfgptr);
        auto pcfg = (conv_fwd_config_t *)cfgptr.get();
        tunop->get_inputs()[0]->details_.set_format(
                sc_data_format_t::NCDHWc(pcfg->C_block));
        if (std::is_same<wei_type, int8_t>::value) {
            tunop->get_inputs()[1]->details_.set_format(
                    sc_data_format_t::KCDRSck4c(pcfg->C_block, pcfg->K_block));
        } else {
            tunop->get_inputs()[1]->details_.set_format(
                    sc_data_format_t::KCDRSck(pcfg->C_block, pcfg->K_block));
        }
        tunop->get_outputs()[0]->details_.set_format(
                sc_data_format_t::NCDHWc(pcfg->K_block));
    }
    fuse_arg_ops = {in_data, in_weight};
    sc_op_ptr final_out = conv_out;
    auto bc_axis = std::vector<std::pair<int, sc_dims>> {{1, {}}};
    if (fuse_bias) {
        auto bias_in = g.make_input({make_tensor({oc})});
        final_out = g.make("add",
                {final_out->get_outputs()[0], bias_in->get_outputs()[0]}, {},
                {{"bc_axis", bc_axis}});
        fuse_arg_ops.emplace_back(bias_in);
    }
    auto out = g.make_output(final_out->get_outputs());
    fuse_arg_ops.insert(fuse_arg_ops.begin(), out);
    auto f = lower_graph(get_test_ctx(), g, fuse_arg_ops);

    auto fptr = jit_engine_t::make(get_test_ctx())->get_entry_func(f, true);
    auto output = alloc_array<dst_type>(mb * oc * od * oh * ow, INIT_NOOP);
    auto input = alloc_array<src_type>(mb * ic * id * ih * iw);
    auto weight = alloc_array<wei_type>(oc * ic * kd * kh * kw);
    auto bias = alloc_array<float>(oc);

    std::vector<generic_val> sc_args = {&output[0], &input[0], &weight[0]};

    if (fuse_bias) sc_args.emplace_back(&bias[0]);
    std::vector<generic_val> generic_args;
    for (unsigned i = 0; i < sc_args.size(); i++)
        generic_args.emplace_back(sc_args.at(i));
    fptr->call_generic_default(generic_args.data());

    auto sc_output = NCDHWc2NCDHW(
            output, mb, oc / cfg.K_block, od, oh, ow, cfg.K_block);
    auto ref_input = NCDHWc2NCDHW(
            input, mb, ic / cfg.C_block, id, ih, iw, cfg.C_block);
    test_buffer<wei_type> ref_weight;

    if (std::is_same<wei_type, int8_t>::value) {
        ref_weight = KCDRSckc2KCDRS(weight, oc / cfg.K_block, ic / cfg.C_block,
                kd, kh, kw, utils::divide_and_ceil(cfg.C_block, 4), cfg.K_block,
                4);
    } else {
        ref_weight = KCDRSck2KCDRS(weight, oc / cfg.K_block, ic / cfg.C_block,
                kd, kh, kw, cfg.C_block, cfg.K_block);
    }

    auto ref_bias = std::move(bias);
    test_buffer<dst_type> ref_output(mb * oc * od * oh * ow);

    compute_ref_direct_fwd(mb, 1, oc, ic, ih, iw, oh, ow, kh, kw, sh, sw, ph,
            pw, &ref_input[0], &ref_weight[0], &ref_bias[0], &ref_output[0],
            fuse_bias ? dir_t::FWD_B : FWD_I, nullptr, nullptr, false, od, id,
            sd, pd, kd);

    test_utils::compare_data(sc_output, ref_output, 1e-3f, 1e-3f);
}

void check_conv_bwd_d_correctness(int N, int K, int C, int D, int H, int W,
        int KD, int R, int S, int stride, int padding) {
    REQUIRE_AVX2();
    sc_graph_t mgr;
    std::vector<sc_op_ptr> fuse_arg_ops;
    sc_dims stride_arr = {stride, stride, stride};
    sc_dims padding_arr = {padding, padding, padding};
    int O = (D + 2 * padding - KD) / stride + 1;
    int P = (H + 2 * padding - R) / stride + 1;
    int Q = (W + 2 * padding - S) / stride + 1;
    auto in_a = mgr.make_input({graph_tensor::make({N, K, O, P, Q})});
    auto in_weight = mgr.make_input({graph_tensor::make({K, C, KD, R, S})});
    auto conv_out = mgr.make("conv_bwd_data_core",
            {in_a->get_outputs()[0], in_weight->get_outputs()[0]},
            {graph_tensor::make({N, C, D, H, W})},
            {{"strides", stride_arr}, {"paddings", padding_arr},
                    {"dst_shape", sc_dims {N, C, D, H, W}}});

    fuse_arg_ops = {in_a, in_weight};
    const sc_op_ptr &final_out = conv_out;
    auto out = mgr.make_output(final_out->get_outputs());
    fuse_arg_ops.insert(fuse_arg_ops.begin(), out);

    graph_driver(mgr, get_test_ctx());

    auto f = lower_graph(get_test_ctx(), mgr, fuse_arg_ops);

    auto fptr = jit_engine_t::make(get_test_ctx())->get_entry_func(f, true);
    auto grad = alloc_array<float>(N * P * O * Q * K);
    auto grad_data = alloc_array<float>(N * H * D * W * C);
    auto weight = alloc_array<float>(K * C * KD * R * S);
    test_buffer<float> bias(K);
    bias.zeroout();

    std::vector<float *> sc_args = {&grad_data[0], &grad[0], &weight[0]};
    std::vector<generic_val> generic_args;
    for (unsigned i = 0; i < sc_args.size(); i++)
        generic_args.emplace_back(sc_args.at(i));
    fptr->call_generic_default(generic_args.data());

    auto mkldnn_grad = std::move(grad);
    auto mkldnn_weight = std::move(weight);
    auto mkldnn_bias = std::move(bias);
    test_buffer<float> mkldnn_grad_data(N * C * D * H * W);
    compute_ref_direct_bwd_d(N, 1, K, C, H, W, P, Q, R, S, stride, stride,
            padding, padding, &mkldnn_grad_data[0], &mkldnn_weight[0],
            &mkldnn_bias[0], &mkldnn_grad[0], dir_t::BWD_D, O, D, stride,
            padding, KD);

    test_utils::compare_data(grad_data, mkldnn_grad_data, 1e-3f, 1e-4f);
}

void check_conv_bwd_w_correctness(int N, int K, int C, int D, int H, int W,
        int KD, int R, int S, int stride, int padding) {
    REQUIRE_AVX2();
    sc_graph_t mgr;
    std::vector<sc_op_ptr> fuse_arg_ops;
    sc_dims stride_arr = {stride, stride, stride};
    sc_dims padding_arr = {padding, padding, padding};
    int O = (D + 2 * padding - KD) / stride + 1;
    int P = (H + 2 * padding - R) / stride + 1;
    int Q = (W + 2 * padding - S) / stride + 1;
    auto in_data = mgr.make_input({graph_tensor::make({N, C, D, H, W})});
    auto in_diff_dst = mgr.make_input({graph_tensor::make({N, K, O, P, Q})});
    auto conv_out = mgr.make("conv_bwd_weight_core",
            {in_data->get_outputs()[0], in_diff_dst->get_outputs()[0]},
            {graph_tensor::make({K, C, KD, R, S})},
            {{"strides", stride_arr}, {"paddings", padding_arr},
                    {"weights_shape", sc_dims {K, C, KD, R, S}}});

    fuse_arg_ops = {in_data, in_diff_dst};
    const sc_op_ptr &final_out = conv_out;
    auto out = mgr.make_output(final_out->get_outputs());
    fuse_arg_ops.insert(fuse_arg_ops.begin(), out);

    graph_driver(mgr, get_test_ctx());

    auto f = lower_graph(get_test_ctx(), mgr, fuse_arg_ops);

    auto fptr = jit_engine_t::make(get_test_ctx())->get_entry_func(f, true);
    auto data = alloc_array<float>(N * D * H * W * C);
    auto grad = alloc_array<float>(N * O * P * Q * K);
    auto grad_weight = alloc_array<float>(K * C * KD * R * S);

    std::vector<float *> sc_args = {&grad_weight[0], &data[0], &grad[0]};
    std::vector<generic_val> generic_args;
    for (unsigned i = 0; i < sc_args.size(); i++)
        generic_args.emplace_back(sc_args.at(i));
    fptr->call_generic_default(generic_args.data());

    auto mkldnn_grad = std::move(grad);
    auto mkldnn_data = std::move(data);
    test_buffer<float> mkldnn_grad_weight(K * C * KD * R * S);

    compute_ref_bwd_weights(N, 1, K, C, H, W, P, Q, R, S, stride, stride,
            padding, padding, &mkldnn_data[0], &mkldnn_grad_weight[0],
            &mkldnn_grad[0], dir_t::BWD_W, O, D, stride, padding, KD);
    test_utils::compare_data(grad_weight, mkldnn_grad_weight, 1e-3f, 5e-3f);
}

TEST(GCCore_CPU_conv3d_fwd, CONV3D_3x3_1) {
    check_conv_fwd_correctness<float, float, float>(conv_fwd_config_t(),
            {1, 1, 5, 5, 5}, {1, 1, 3, 3, 3}, {1, 1, 1}, {0, 0, 0}, false,
            true);
}

TEST(GCCore_CPU_conv3d_fwd, CONV3D_3x3_2) {
    check_conv_fwd_correctness<float, float, float>(conv_fwd_config_t(),
            {16, 64, 56, 56, 56}, {64, 64, 3, 3, 3}, {2, 2, 2}, {0, 0, 0},
            false, true);
}

TEST(GCCore_CPU_conv3d_fwd, CONV3D_3x3_PAD_1) {
    check_conv_fwd_correctness<float, float, float>(conv_fwd_config_t(),
            {1, 1, 9, 9, 9}, {1, 1, 3, 3, 3}, {2, 2, 2}, {1, 1, 1}, false,
            true);
}

TEST(GCCore_CPU_conv3d_fwd, CONV3D_3x3_PAD_2) {
    check_conv_fwd_correctness<float, float, float>(conv_fwd_config_t(),
            {1, 1, 9, 9, 9}, {1, 1, 3, 3, 3}, {2, 2, 2}, {2, 2, 2}, false,
            true);
}

TEST(GCCore_CPU_conv3d_fwd, CONV3D_3x3_PAD_3) {
    check_conv_fwd_correctness<float, float, float>(conv_fwd_config_t(),
            {16, 16, 28, 28, 28}, {16, 16, 3, 3, 3}, {1, 1, 2}, {2, 2, 1},
            false, true);
}

TEST(GCCore_CPU_conv3d_fwd, CONV3D_3x3_PAD_4) {
    check_conv_fwd_correctness<float, float, float>(conv_fwd_config_t(),
            {16, 16, 28, 28, 28}, {16, 16, 3, 3, 3}, {2, 1, 2}, {1, 2, 2},
            false, true);
}

TEST(GCCore_CPU_conv3d_fwd, CONV3D_3x3_PAD_5) {
    check_conv_fwd_correctness<float, float, float>(conv_fwd_config_t(),
            {1, 1, 5, 5, 5}, {1, 1, 3, 3, 3}, {2, 1, 2}, {3, 4, 4}, false,
            true);
}

TEST(GCCore_CPU_conv3d_fwd, CONV3D_3x3_PAD_6) {
    check_conv_fwd_correctness<float, float, float>(conv_fwd_config_t(),
            {16, 16, 28, 28, 28}, {16, 16, 3, 3, 3}, {2, 1, 2}, {3, 4, 4},
            false, true);
}

TEST(GCCore_CPU_conv3d_fwd, QCONV3D_3X3_PAD_1) {
    REQUIRE_AMX();
    check_conv_fwd_correctness<uint8_t, int8_t, int32_t>(conv_fwd_config_t(),
            {1, 4, 5, 5, 5}, {4, 4, 3, 3, 3}, {1, 1, 1}, {1, 1, 1}, false,
            true);
}

TEST(GCCore_CPU_conv3d_fwd, QCONV3D_3X3_PAD_2) {
    REQUIRE_AMX();
    check_conv_fwd_correctness<uint8_t, int8_t, int32_t>(conv_fwd_config_t(),
            {16, 16, 28, 28, 28}, {16, 16, 3, 3, 3}, {2, 2, 2}, {1, 2, 1},
            false, true);
}

TEST(GCCore_CPU_conv3d_fwd, QCONV3D_3X3_PAD_3) {
    REQUIRE_AMX();
    check_conv_fwd_correctness<uint8_t, int8_t, int32_t>(conv_fwd_config_t(),
            {1, 16, 28, 28, 28}, {16, 16, 3, 3, 3}, {1, 1, 2}, {3, 4, 4}, false,
            true);
}

TEST(GCCore_CPU_conv3d_fwd, QCONV3D_3x3_1) {
    REQUIRE_VNNI();
    check_conv_fwd_correctness<uint8_t, int8_t, int32_t>(conv_fwd_config_t(),
            {1, 4, 5, 5, 5}, {4, 4, 3, 3, 3}, {1, 1, 1}, {0, 0, 0}, false,
            true);
}

TEST(GCCore_CPU_conv3d_fwd, QCONV3D_3x3_2) {
    REQUIRE_VNNI();
    check_conv_fwd_correctness<uint8_t, int8_t, int32_t>(conv_fwd_config_t(),
            {16, 64, 56, 56, 56}, {64, 64, 3, 3, 3}, {2, 2, 2}, {0, 0, 0},
            false, true);
}

TEST(GCCore_CPU_conv3d_bwd_d, CONV3D_1x1_1) {
    check_conv_bwd_d_correctness(1, 4, 8, 2, 5, 5, 1, 1, 1, 1, 0);
}

TEST(GCCore_CPU_conv3d_bwd_d, CONV3D_1x1_2) {
    check_conv_bwd_d_correctness(1, 4, 8, 2, 5, 5, 1, 1, 1, 2, 0);
}

TEST(GCCore_CPU_conv3d_bwd_d, CONV3D_1x1_3) {
    check_conv_bwd_d_correctness(16, 64, 64, 8, 28, 28, 1, 1, 1, 2, 0);
}

TEST(GCCore_CPU_conv3d_bwd_d, CONV3D_1x1_4) {
    check_conv_bwd_d_correctness(16, 64, 64, 8, 28, 28, 1, 1, 1, 1, 1);
}

TEST(GCCore_CPU_conv3d_bwd_w, CONV3D_1x1_1) {
    check_conv_bwd_w_correctness(1, 4, 8, 2, 5, 5, 1, 1, 1, 1, 0);
}

TEST(GCCore_CPU_conv3d_bwd_w, CONV3D_1x1_2) {
    check_conv_bwd_w_correctness(1, 4, 8, 2, 5, 5, 1, 1, 1, 2, 0);
}

TEST(GCCore_CPU_conv3d_bwd_w, CONV3D_1x1_3) {
    check_conv_bwd_w_correctness(16, 64, 64, 8, 28, 28, 1, 1, 1, 2, 0);
}

TEST(GCCore_CPU_conv3d_bwd_w, CONV3D_1x1_4) {
    check_conv_bwd_w_correctness(16, 64, 64, 8, 28, 28, 1, 1, 1, 1, 1);
}
