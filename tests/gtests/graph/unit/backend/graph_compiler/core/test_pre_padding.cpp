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

#include <iostream>
#include "context.hpp"
#include "reference/conv_ref.hpp"
#include "reference/eltwise_ref.hpp"
#include "reference/padding_ref.hpp"
#include "gtest/gtest.h"
#include <compiler/ir/easy_build.hpp>
#include <compiler/ir/graph/driver.hpp>
#include <compiler/ir/graph/fusible_op.hpp>
#include <compiler/ir/graph/lowering.hpp>
#include <compiler/ir/graph/mixed_partition.hpp>
#include <compiler/ir/graph/pass/pass.hpp>
#include <compiler/ir/graph/transform/transform.hpp>
#include <compiler/ir/graph/tunable_op.hpp>
#include <compiler/ir/graph/visitor.hpp>
#include <compiler/ir/ir_comparer.hpp>
#include <compiler/ir/transform/buffer_schedule.hpp>
#include <compiler/ir/transform/loop_transform.hpp>
#include <compiler/jit/jit.hpp>
#include <ops/convolution.hpp>
#include <ops/fusible/padding.hpp>
#include <ops/templates/conv_fwd.hpp>
#include <util/any_map.hpp>
#include <util/reflection.hpp>
#include <util/utils.hpp>

using namespace dnnl::impl::graph::gc;

TEST(GCCore_CPU_pre_padding_test, TestPre_Padding_Standalone) {
    REQUIRE_AVX2();
    auto ctx = std::make_shared<context_t>(*get_default_context());
    ctx->flags_.mixed_fusion_ = true;
    sc_graph_t g;
    auto input_shape = sc_dims {64, 256, 56, 56};
    auto paddings = sc_dims {4, 4};
    auto output_shape = {input_shape[0], input_shape[1],
            input_shape[2] + paddings[0] * 2, input_shape[3] + paddings[0] * 2};

    auto input = g.make_input({graph_tensor::make(
            input_shape, sc_data_format_t::NCHW(), datatypes::f32)});

    auto padding = g.make("padding", {input->get_outputs()[0]}, {},
            {{"pads_begin", paddings}, {"pads_end", paddings}});

    auto out = g.make_output(padding->get_outputs());

    graph_driver(g, ctx);
    std::vector<sc_op_ptr> arg_list = {input, out};
    auto f = lower_graph(ctx, g, arg_list);
    auto input_data = alloc_array<float>(test_utils::product(input_shape));
    auto sc_output = alloc_array<float>(test_utils::product(output_shape));
    auto fptr = jit_engine_t::make(ctx)->get_entry_func(f);
    fptr->call_default(&input_data[0], &sc_output[0]);

    auto ref_output = alloc_array<float>(test_utils::product(output_shape));
    ref_padding_2d(
            &ref_output[0], &input_data[0], output_shape, paddings, paddings);

    test_utils::compare_data(sc_output, ref_output, 1e-4f, 1e-5f);
}

TEST(GCCore_CPU_pre_padding_test, TestPre_Padding_Graph) {
    REQUIRE_AMX();
    REQUIRE_AVX2();
    auto ctx = std::make_shared<context_t>(*get_default_context());
    ctx->flags_.mixed_fusion_ = true;
    sc_graph_t g;
    auto input = g.make_input({graph_tensor::make(
            {9, 256, 56, 56}, sc_data_format_t::NCHW(), datatypes::bf16)});

    auto conv_1 = g.make("conv_fwd_core",
            {input->get_outputs()[0],
                    g.make_input({graph_tensor::make({64, 256, 1, 1},
                                         sc_data_format_t::KCRS(),
                                         datatypes::bf16)},
                             {{"constant", const_kind::local_const}})
                            ->get_outputs()[0]},
            {},
            {{"strides", sc_dims {1, 1}}, {"pads_begin", sc_dims {0, 0}},
                    {"pads_end", sc_dims {0, 0}}});

    conv_1 = g.make("cast", {conv_1->get_outputs()[0]}, {},
            {{"dtype", datatypes::bf16}});

    auto conv_2 = g.make("conv_fwd_core",
            {conv_1->get_outputs()[0],
                    g.make_input({graph_tensor::make({64, 64, 3, 3},
                                         sc_data_format_t::KCRS(),
                                         datatypes::bf16)},
                             {{"constant", const_kind::local_const}})
                            ->get_outputs()[0]},
            {},
            {{"strides", sc_dims {1, 1}}, {"pads_begin", sc_dims {1, 1}},
                    {"pads_end", sc_dims {1, 1}}});

    conv_2 = g.make("cast", {conv_2->get_outputs()[0]}, {},
            {{"dtype", datatypes::bf16}});
    auto conv_3 = g.make("conv_fwd_core",
            {conv_2->get_outputs()[0],
                    g.make_input({graph_tensor::make({64, 64, 1, 1},
                                         sc_data_format_t::KCRS(),
                                         datatypes::bf16)},
                             {{"constant", const_kind::local_const}})
                            ->get_outputs()[0]},
            {},
            {{"strides", sc_dims {1, 1}}, {"pads_begin", sc_dims {0, 0}},
                    {"pads_end", sc_dims {0, 0}}});

    auto add = g.make("add",
            {conv_3->get_outputs()[0], conv_3->get_outputs()[0]}, {}, {});
    auto out = g.make_output(add->get_outputs());
    graph_driver(g, ctx);
    std::stringstream ss;
    print_graph(g, ss, true);
    EXPECT_TRUE(ss.str().find("padding") != std::string::npos);
}

TEST(GCCore_CPU_pre_padding_test, TestPre_Padding_NoInplace) {
    SET_THREADS_OR_SKIP(32);
    auto ctx = std::make_shared<context_t>(*get_test_ctx());
    ctx->flags_.mixed_fusion_ = true;
    sc_graph_t g;
    auto input = g.make_input({graph_tensor::make(
            {64, 56, 56}, sc_data_format_t(), datatypes::u8)});

    auto relu0 = g.make("relu", input->get_outputs(), {}, {});
    // used for query buffer later
    auto cache_gt = relu0->get_outputs()[0];
    // `relu0` is shared with `relu1`(marked as `break_pre_fuse`) and `tv0`, as
    // the result, `padding0` could not inplace output buffer of `relu0` in
    // avoid of potential `tensorptr` node occuring on function argument
    auto relu1 = g.make("relu", relu0->get_outputs(), {},
            {{op_attr_key::break_pre_fuse, true}});
    auto out0 = g.make_output(relu1->get_outputs());
    auto tv0 = g.make("tensor_view", relu0->get_outputs(), {},
            {{"shape", sc_dims {32, 2, 56, 56}}});
    auto padding0 = g.make("padding", tv0->get_outputs(), {},
            {{"pads_begin", sc_dims {1, 1}}, {"pads_end", sc_dims {1, 1}}});
    auto relu2 = g.make("relu", padding0->get_outputs(), {}, {});
    auto out1 = g.make_output(relu2->get_outputs());
    mixed_partition(g, ctx);
    mixed_fuse_op_t *fused_op = get_mixed_op_from_graph(g);
    ASSERT_TRUE(fused_op && fused_op->parti_list_.size() == 1);
    auto parti = fused_op->parti_list_[0];
    // The output buffer of `relu0` is expected to be a `tensor` node rather
    // than `tensorptr`
    EXPECT_TRUE(parti->buf_alloc_.g2b_map_.get(cache_gt).isa<tensor>());
}

TEST(GCCore_CPU_pre_padding_test, TestPre_Padding_Conv_Padding_Reorder) {
    REQUIRE_AVX2();
    auto ctx = std::make_shared<context_t>(*get_default_context());
    ctx->flags_.mixed_fusion_ = true;
    sc_graph_t g;
    auto input_shape = sc_dims {1, 64, 14, 14};
    auto weight_shape_1 = sc_dims {64, 64, 1, 1};
    auto strides = sc_dims {1};

    const sc_dims pads_begin = {1, 1};
    const sc_dims pads_end = {1, 1};

    auto conv1_output_shape = sc_dims {input_shape[0], weight_shape_1[0],
            (input_shape[2] - weight_shape_1[2]) / strides[0] + 1,
            (input_shape[3] - weight_shape_1[3]) / strides[0] + 1};

    auto padding_output_shape = sc_dims {input_shape[0], weight_shape_1[0],
            conv1_output_shape[2] + pads_begin[0] + pads_end[0],
            conv1_output_shape[3] + pads_begin[1] + pads_end[1]};

    auto out_shape = padding_output_shape;

    auto input = g.make_input({graph_tensor::make(
            input_shape, sc_data_format_t::NCHW(), datatypes::f32)});
    auto weight_1 = g.make_input({std::make_shared<graph_tensor>(nullptr,
            sc_data_format_t::KCRS(), weight_shape_1, datatypes::f32)});

    auto conv_1 = g.make("conv_fwd_core",
            {input->get_outputs()[0], weight_1->get_outputs()[0]}, {},
            {{"strides", sc_dims {strides[0], strides[0]}},
                    {"pads_begin", sc_dims {0, 0}},
                    {"pads_end", sc_dims {0, 0}}});

    auto padding = g.make("padding", {conv_1->get_outputs()[0]}, {},
            {{"pads_begin", pads_begin}, {"pads_end", pads_end}});

    auto reorder = g.make("reorder", padding->get_outputs(), {},
            {{"out_format", sc_data_format_t::NCHWc(16)}, {"internal", true}});

    auto out = g.make_output(reorder->get_outputs());
    g.attrs_["is_input_plain"] = true;
    g.attrs_["is_output_plain"] = false;
    graph_driver(g, ctx);
    //     print_graph(g, std::cout,1);
    std::vector<sc_op_ptr> arg_list = {input, weight_1, out};
    auto f = lower_graph(ctx, g, arg_list);
    auto input_data = alloc_array<float>(test_utils::product(input_shape));
    auto weight_1_data
            = alloc_array<float>(test_utils::product(weight_shape_1));
    auto conv1_output
            = alloc_array<float>(test_utils::product(conv1_output_shape));
    auto sc_output = alloc_array<float>(test_utils::product(out_shape));
    auto ref_output = alloc_array<float>(test_utils::product(out_shape));

    auto fptr = jit_engine_t::make(ctx)->get_entry_func(f);
    fptr->call_default(&input_data[0], &weight_1_data[0], &sc_output[0]);

    auto padding_output
            = alloc_array<float>(test_utils::product(padding_output_shape));

    compute_ref_direct_fwd(input_shape[0], 1, weight_shape_1[0], input_shape[1],
            input_shape[2], input_shape[3], conv1_output_shape[2],
            conv1_output_shape[3], weight_shape_1[2], weight_shape_1[3],
            strides[0], strides[0], 0, 0, &input_data[0], &weight_1_data[0],
            static_cast<float *>(nullptr), &conv1_output[0], FWD_I);

    ref_padding_2d(&padding_output[0], &conv1_output[0], padding_output_shape,
            pads_begin, pads_end);
    ref_output = NCHW2NCHWc(padding_output, 1, 4, 16, 16, 16);
    test_utils::compare_data(sc_output, ref_output, 1e-3f, 1e-3f);
}

TEST(GCCore_CPU_pre_padding_test, TestPre_Padding_Conv_Padding) {
    REQUIRE_AVX2();
    auto ctx = std::make_shared<context_t>(*get_default_context());
    ctx->flags_.mixed_fusion_ = true;
    sc_graph_t g;
    auto input_shape = sc_dims {16, 374, 66, 66};
    auto weight_shape_1 = sc_dims {52, 374, 1, 1};
    auto weight_shape_2 = sc_dims {279, 52, 5, 5};
    auto strides = sc_dims {1, 3};

    const sc_dims pads_begin = {4, 4};
    const sc_dims pads_end = {4, 4};

    auto conv1_output_shape = sc_dims {input_shape[0], weight_shape_1[0],
            (input_shape[2] - weight_shape_1[2]) / strides[0] + 1,
            (input_shape[3] - weight_shape_1[3]) / strides[0] + 1};

    auto padding_output_shape = sc_dims {input_shape[0], weight_shape_1[0],
            conv1_output_shape[2] + pads_begin[0] + pads_end[0],
            conv1_output_shape[3] + pads_begin[1] + pads_end[1]};

    auto out_shape = sc_dims {padding_output_shape[0], weight_shape_2[0],
            (padding_output_shape[2] - weight_shape_2[2]) / strides[1] + 1,
            (padding_output_shape[3] - weight_shape_2[3]) / strides[1] + 1};

    auto input = g.make_input({graph_tensor::make(
            input_shape, sc_data_format_t::NCHW(), datatypes::f32)});
    auto weight_1 = g.make_input({std::make_shared<graph_tensor>(nullptr,
            sc_data_format_t::KCRS(), weight_shape_1, datatypes::f32)});
    auto weight_2 = g.make_input({std::make_shared<graph_tensor>(nullptr,
            sc_data_format_t::KCRS(), weight_shape_2, datatypes::f32)});

    auto conv_1 = g.make("conv_fwd_core",
            {input->get_outputs()[0], weight_1->get_outputs()[0]}, {},
            {{"strides", sc_dims {strides[0], strides[0]}},
                    {"pads_begin", sc_dims {0, 0}},
                    {"pads_end", sc_dims {0, 0}}});

    auto padding = g.make("padding", {conv_1->get_outputs()[0]}, {},
            {{"pads_begin", pads_begin}, {"pads_end", pads_end}});

    auto conv_2 = g.make("conv_fwd_core",
            {padding->get_outputs()[0], weight_2->get_outputs()[0]}, {},
            {{"strides", sc_dims {strides[1], strides[1]}},
                    {"pads_begin", sc_dims {0, 0}},
                    {"pads_end", sc_dims {0, 0}}});

    auto out = g.make_output(conv_2->get_outputs());
    graph_driver(g, ctx);
    std::vector<sc_op_ptr> arg_list = {input, weight_1, weight_2, out};
    auto f = lower_graph(ctx, g, arg_list);

    auto input_data = alloc_array<float>(test_utils::product(input_shape));

    auto weight_1_data
            = alloc_array<float>(test_utils::product(weight_shape_1));
    auto weight_2_data
            = alloc_array<float>(test_utils::product(weight_shape_2));

    auto conv1_output
            = alloc_array<float>(test_utils::product(conv1_output_shape));

    auto sc_output = alloc_array<float>(test_utils::product(out_shape));
    auto ref_output = alloc_array<float>(test_utils::product(out_shape));

    auto fptr = jit_engine_t::make(ctx)->get_entry_func(f);
    fptr->call_default(&input_data[0], &weight_1_data[0], &weight_2_data[0],
            &sc_output[0]);

    auto padding_output
            = alloc_array<float>(test_utils::product(padding_output_shape));

    compute_ref_direct_fwd(input_shape[0], 1, weight_shape_1[0], input_shape[1],
            input_shape[2], input_shape[3], conv1_output_shape[2],
            conv1_output_shape[3], weight_shape_1[2], weight_shape_1[3],
            strides[0], strides[0], 0, 0, &input_data[0], &weight_1_data[0],
            static_cast<float *>(nullptr), &conv1_output[0], FWD_I);

    ref_padding_2d(&padding_output[0], &conv1_output[0], padding_output_shape,
            pads_begin, pads_end);

    compute_ref_direct_fwd(padding_output_shape[0], 1, weight_shape_2[0],
            padding_output_shape[1], padding_output_shape[2],
            padding_output_shape[3], out_shape[2], out_shape[3],
            weight_shape_2[2], weight_shape_2[3], strides[1], strides[1], 0, 0,
            &padding_output[0], &weight_2_data[0],
            static_cast<float *>(nullptr), &ref_output[0], FWD_I);

    test_utils::compare_data(sc_output, ref_output, 1e-3f, 1e-3f);
}

TEST(GCCore_CPU_pre_padding_test, TestPre_Padding_Conv_Asym_Padding) {
    REQUIRE_AVX2();
    auto ctx = std::make_shared<context_t>(*get_default_context());
    ctx->flags_.mixed_fusion_ = true;
    sc_graph_t g;
    auto input_shape = sc_dims {1, 64, 224, 224};
    auto weight_shape_1 = sc_dims {64, 64, 1, 1};
    auto weight_shape_2 = sc_dims {64, 64, 7, 7};
    auto strides = sc_dims {1, 2};

    const sc_dims pads_begin = {3, 3};
    const sc_dims pads_end = {2, 2};

    auto conv1_output_shape = sc_dims {input_shape[0], weight_shape_1[0],
            (input_shape[2] - weight_shape_1[2]) / strides[0] + 1,
            (input_shape[3] - weight_shape_1[3]) / strides[0] + 1};

    auto padding_output_shape = sc_dims {input_shape[0], weight_shape_1[0],
            conv1_output_shape[2] + pads_begin[0] + pads_end[0],
            conv1_output_shape[3] + pads_begin[1] + pads_end[1]};

    auto out_shape = sc_dims {padding_output_shape[0], weight_shape_2[0],
            (padding_output_shape[2] - weight_shape_2[2]) / strides[1] + 1,
            (padding_output_shape[3] - weight_shape_2[3]) / strides[1] + 1};

    auto input = g.make_input({graph_tensor::make(
            input_shape, sc_data_format_t::NCHW(), datatypes::f32)});
    auto weight_1 = g.make_input({std::make_shared<graph_tensor>(nullptr,
            sc_data_format_t::KCRS(), weight_shape_1, datatypes::f32)});
    auto weight_2 = g.make_input({std::make_shared<graph_tensor>(nullptr,
            sc_data_format_t::KCRS(), weight_shape_2, datatypes::f32)});

    auto conv_1 = g.make("conv_fwd_core",
            {input->get_outputs()[0], weight_1->get_outputs()[0]}, {},
            {{"strides", sc_dims {strides[0], strides[0]}},
                    {"pads_begin", sc_dims {0, 0}},
                    {"pads_end", sc_dims {0, 0}}});

    auto padding = g.make("padding", {conv_1->get_outputs()[0]}, {},
            {{"pads_begin", pads_begin}, {"pads_end", pads_end}});

    auto conv_2 = g.make("conv_fwd_core",
            {padding->get_outputs()[0], weight_2->get_outputs()[0]}, {},
            {{"strides", sc_dims {strides[1], strides[1]}},
                    {"pads_begin", sc_dims {0, 0}},
                    {"pads_end", sc_dims {0, 0}}});

    auto out = g.make_output(conv_2->get_outputs());
    graph_driver(g, ctx);
    std::vector<sc_op_ptr> arg_list = {input, weight_1, weight_2, out};
    auto f = lower_graph(ctx, g, arg_list);

    auto input_data = alloc_array<float>(test_utils::product(input_shape));

    auto weight_1_data
            = alloc_array<float>(test_utils::product(weight_shape_1));
    auto weight_2_data
            = alloc_array<float>(test_utils::product(weight_shape_2));

    auto conv1_output
            = alloc_array<float>(test_utils::product(conv1_output_shape));

    auto sc_output = alloc_array<float>(test_utils::product(out_shape));
    auto ref_output = alloc_array<float>(test_utils::product(out_shape));

    auto fptr = jit_engine_t::make(ctx)->get_entry_func(f);
    fptr->call_default(&input_data[0], &weight_1_data[0], &weight_2_data[0],
            &sc_output[0]);

    auto padding_output
            = alloc_array<float>(test_utils::product(padding_output_shape));

    compute_ref_direct_fwd(input_shape[0], 1, weight_shape_1[0], input_shape[1],
            input_shape[2], input_shape[3], conv1_output_shape[2],
            conv1_output_shape[3], weight_shape_1[2], weight_shape_1[3],
            strides[0], strides[0], 0, 0, &input_data[0], &weight_1_data[0],
            static_cast<float *>(nullptr), &conv1_output[0], FWD_I);

    ref_padding_2d(&padding_output[0], &conv1_output[0], padding_output_shape,
            pads_begin, pads_end);

    compute_ref_direct_fwd(padding_output_shape[0], 1, weight_shape_2[0],
            padding_output_shape[1], padding_output_shape[2],
            padding_output_shape[3], out_shape[2], out_shape[3],
            weight_shape_2[2], weight_shape_2[3], strides[1], strides[1], 0, 0,
            &padding_output[0], &weight_2_data[0],
            static_cast<float *>(nullptr), &ref_output[0], FWD_I);

    test_utils::compare_data(sc_output, ref_output, 1e-3f, 1e-3f);
}

TEST(GCCore_CPU_pre_padding_test, TestPre_Padding_Fuse) {
    REQUIRE_AVX2();
    auto ctx = std::make_shared<context_t>(*get_default_context());
    ctx->flags_.mixed_fusion_ = true;
    sc_graph_t g;
    auto input_shape = sc_dims {1, 1, 4, 4};
    auto weight_shape = sc_dims {1, 1, 3, 3};
    const sc_dims stride = {1, 1};
    const sc_dims padding_conv = {0, 0};
    const sc_dims padding_pad = {1, 1};

    auto in_data = g.make_input({std::make_shared<graph_tensor>(
            nullptr, sc_data_format_t::NCHW(), input_shape, datatypes::f32)});
    auto in_weight = g.make_input({std::make_shared<graph_tensor>(
            nullptr, sc_data_format_t::KCRS(), weight_shape, datatypes::f32)});
    auto conv_out = g.make("conv_fwd_core",
            {in_data->get_outputs()[0], in_weight->get_outputs()[0]}, {},
            {{"strides", stride}, {"pads_begin", padding_conv},
                    {"pads_end", padding_conv}});
    auto pad_out = g.make("padding", {conv_out->get_outputs()[0]}, {},
            {{"pads_begin", padding_pad}, {"pads_end", padding_pad}});
    auto out = g.make_output(pad_out->get_outputs());
    g.attrs_["is_input_plain"] = true;
    g.attrs_["is_output_plain"] = true;
    graph_driver(g, ctx);

    std::vector<sc_op_ptr> arg_list = {in_data, in_weight, out};
    auto f = lower_graph(ctx, g, arg_list);

    const auto input_size = test_utils::product(input_shape);
    auto input_data = alloc_array<float>(input_size);
    for (size_t i = 0; i < input_size; i++) {
        input_data[i] = 1.0 * i;
    }

    const auto weight_size = test_utils::product(weight_shape);
    auto weight_data = alloc_array<float>(weight_size);
    for (size_t i = 0; i < weight_size; i++) {
        weight_data[i] = 1.0;
    }

    int out_size = 4 * 4;
    auto sc_output = alloc_array<float>(out_size);

    auto fptr = jit_engine_t::make(ctx)->get_entry_func(f);
    fptr->call_default(&input_data[0], &weight_data[0], &sc_output[0]);

    std::vector<float> expected
            = {0, 0, 0, 0, 0, 45, 54, 0, 0, 81, 90, 0, 0, 0, 0, 0};
    test_utils::compare_data(sc_output, expected, 1e-4f, 1e-5f);
}
