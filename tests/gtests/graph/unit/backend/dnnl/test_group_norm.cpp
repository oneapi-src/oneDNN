/*******************************************************************************
* Copyright 2024-2025 Intel Corporation
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

#include <random>
#include "gtest/gtest.h"

#include "graph/unit/backend/dnnl/dnnl_test_common.hpp"
#include "graph/unit/unit_test_common.hpp"
#include "graph/unit/utils.hpp"

namespace graph = dnnl::impl::graph;
namespace utils = dnnl::graph::tests::unit::utils;

TEST(test_group_norm_execute, GroupnormTraining) {
    graph::engine_t *eng = get_engine();

    // src shape: (2, 2, 2, 3) nxc
    std::vector<float> src(24);
    for (size_t i = 0; i < src.size(); i++) {
        src[i] = i * i;
    }
    std::vector<float> scale {1.0, 2.0, 3.0};
    std::vector<float> shift {0.0, 1.0, 2.0};
    std::vector<float> dst(src.size(), 0.0);
    std::vector<double> ref_dst = {-1.0, -0.714286, 0.142857, 1.57143, -1.51237,
            -0.0535733, 1.7294, 3.83654, -1.8794, 0.518774, 3.19909, 6.16154,
            -1.3078, -0.480077, 0.41386, 1.37401, -1.63131, 0.0547714, 1.84304,
            3.7335, -1.96166, 0.596261, 3.27896, 6.08644};
    std::vector<float> ref_mean {3.5, 31.5, 91.5, 183.5, 307.5, 463.5};
    std::vector<float> ref_var {
            12.25, 152.25, 452.25, 912.25, 1532.25, 2312.25};
    std::vector<float> mean(ref_mean.size(), 0.0);
    std::vector<float> var(ref_var.size(), 0.0);

    graph::op_t groupnorm_op(graph::op_kind::GroupNorm);

    groupnorm_op.set_attr<float>(graph::op_attr::epsilon, 0);
    groupnorm_op.set_attr<bool>(graph::op_attr::use_affine, true); //inference
    groupnorm_op.set_attr<bool>(graph::op_attr::keep_stats, true); //inference
    groupnorm_op.set_attr<std::string>(graph::op_attr::data_format, "NXC");
    groupnorm_op.set_attr<int64_t>(graph::op_attr::groups, 3);

    graph::logical_tensor_t src_lt = utils::logical_tensor_init(
            0, {2, 2, 2, 3}, {12, 2, 1, 4}, graph::data_type::f32);
    graph::logical_tensor_t scale_lt
            = utils::logical_tensor_init(1, {3}, graph::data_type::f32);
    graph::logical_tensor_t shift_lt
            = utils::logical_tensor_init(2, {3}, graph::data_type::f32);
    graph::logical_tensor_t dst_lt = utils::logical_tensor_init(
            3, {2, 2, 2, 3}, {12, 2, 1, 4}, graph::data_type::f32);
    graph::logical_tensor_t mean_lt
            = utils::logical_tensor_init(4, {2, 3}, graph::data_type::f32);
    graph::logical_tensor_t variance_lt
            = utils::logical_tensor_init(5, {2, 3}, graph::data_type::f32);

    graph::engine_t *engine = get_engine();
    graph::graph_t g(engine->kind());

    groupnorm_op.add_input(src_lt);
    groupnorm_op.add_input(scale_lt);
    groupnorm_op.add_input(shift_lt);
    groupnorm_op.add_output(dst_lt);
    groupnorm_op.add_output(mean_lt);
    groupnorm_op.add_output(variance_lt);

    ASSERT_EQ(g.add_op(&groupnorm_op), graph::status::success);
    g.finalize();

    graph::pass::pass_base_ptr apass = get_pass("gn_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile
    graph::partition_t p;
    p.init(part);
    graph::compiled_partition_t cp(p);

    std::vector<const graph::logical_tensor_t *> inputs {
            &src_lt, &scale_lt, &shift_lt};
    std::vector<const graph::logical_tensor_t *> outputs {
            &dst_lt, &mean_lt, &variance_lt};

    ASSERT_EQ(p.compile(&cp, inputs, outputs, engine), graph::status::success);

    test_tensor_t src_ts(src_lt, eng, src);
    test_tensor_t scale_ts(scale_lt, eng, scale);
    test_tensor_t shift_ts(shift_lt, eng, shift);
    test_tensor_t dst_ts(dst_lt, eng, dst);
    test_tensor_t mean_ts(mean_lt, eng, mean);
    test_tensor_t var_ts(variance_lt, eng, var);

    graph::stream_t *strm = get_stream();
    cp.execute(strm, {src_ts.get(), scale_ts.get(), shift_ts.get()},
            {dst_ts.get(), mean_ts.get(), var_ts.get()});
    strm->wait();

    dst = dst_ts.as_vec_type<float>();
    mean = mean_ts.as_vec_type<float>();
    var = var_ts.as_vec_type<float>();

    for (size_t i = 0; i < ref_dst.size(); ++i) {
        ASSERT_LE(std::abs(dst[i] - ref_dst[i]), 1e-4);
    }

    for (size_t i = 0; i < ref_mean.size(); ++i) {
        ASSERT_FLOAT_EQ(mean[i], ref_mean[i]);
        ASSERT_FLOAT_EQ(var[i], ref_var[i]);
    }
}

TEST(test_group_norm_execute, GroupnormInference) {
    graph::engine_t *eng = get_engine();

    // src shape: (2, 3, 2, 2) ncx
    std::vector<float> src(24);
    for (size_t i = 0; i < src.size(); i++) {
        src[i] = i * i;
    }
    std::vector<float> scale {1.0, 2.0, 3.0};
    std::vector<float> shift {0.0, 1.0, 2.0};
    std::vector<float> dst(src.size(), 0.0);
    std::vector<double> ref_dst = {-1.0, -0.714286, 0.142857, 1.57143, -1.51237,
            -0.0535733, 1.7294, 3.83654, -1.8794, 0.518774, 3.19909, 6.16154,
            -1.3078, -0.480077, 0.41386, 1.37401, -1.63131, 0.0547714, 1.84304,
            3.7335, -1.96166, 0.596261, 3.27896, 6.08644};

    graph::op_t groupnorm_op(graph::op_kind::GroupNorm);

    groupnorm_op.set_attr<float>(graph::op_attr::epsilon, 0);
    groupnorm_op.set_attr<bool>(graph::op_attr::use_affine, true); //inference
    groupnorm_op.set_attr<bool>(graph::op_attr::keep_stats, false); //inference
    groupnorm_op.set_attr<std::string>(graph::op_attr::data_format, "NCX");
    groupnorm_op.set_attr<int64_t>(graph::op_attr::groups, 3);

    graph::logical_tensor_t src_lt = utils::logical_tensor_init(
            0, {2, 3, 2, 2}, graph::data_type::f32);
    graph::logical_tensor_t scale_lt
            = utils::logical_tensor_init(1, {3}, graph::data_type::f32);
    graph::logical_tensor_t shift_lt
            = utils::logical_tensor_init(2, {3}, graph::data_type::f32);
    graph::logical_tensor_t dst_lt = utils::logical_tensor_init(
            3, {2, 3, 2, 2}, graph::data_type::f32);

    graph::engine_t *engine = get_engine();
    graph::graph_t g(engine->kind());

    groupnorm_op.add_input(src_lt);
    groupnorm_op.add_input(scale_lt);
    groupnorm_op.add_input(shift_lt);
    groupnorm_op.add_output(dst_lt);

    ASSERT_EQ(g.add_op(&groupnorm_op), graph::status::success);
    g.finalize();

    graph::pass::pass_base_ptr apass = get_pass("gn_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile
    graph::partition_t p;
    p.init(part);
    graph::compiled_partition_t cp(p);

    std::vector<const graph::logical_tensor_t *> inputs {
            &src_lt, &scale_lt, &shift_lt};
    std::vector<const graph::logical_tensor_t *> outputs {&dst_lt};

    ASSERT_EQ(p.compile(&cp, inputs, outputs, engine), graph::status::success);

    test_tensor_t src_ts(src_lt, eng, src);
    test_tensor_t scale_ts(scale_lt, eng, scale);
    test_tensor_t shift_ts(shift_lt, eng, shift);
    test_tensor_t dst_ts(dst_lt, eng, dst);

    graph::stream_t *strm = get_stream();
    cp.execute(strm, {src_ts.get(), scale_ts.get(), shift_ts.get()},
            {dst_ts.get()});
    strm->wait();

    dst = dst_ts.as_vec_type<float>();
    for (size_t i = 0; i < dst.size(); ++i) {
        ASSERT_LE(std::abs(dst[i] - ref_dst[i]), 1e-4);
    }
}

TEST(test_group_norm_execute, GroupnormSwishTypecastQuant) {
    graph::engine_t *eng = get_engine();

    static auto isa = dnnl_get_effective_cpu_isa();
    SKIP_IF((isa < dnnl_cpu_isa_avx512_core)
                    && eng->kind() == graph::engine_kind::cpu,
            "Skip bf16 tests for systems that do not support avx512_core.");

    // src shape: (2, 3, 2, 2) ncx
    std::vector<bfloat16_t> src(24);
    for (size_t i = 0; i < src.size(); i++) {
        src[i] = i * i;
    }
    std::vector<float> scale {1.0, 2.0, 3.0};
    std::vector<float> shift {0.0, 1.0, 2.0};

    graph::op_t groupnorm_op(0, graph::op_kind::GroupNorm, "groupnorm_op");

    groupnorm_op.set_attr<float>(graph::op_attr::epsilon, 0);
    groupnorm_op.set_attr<bool>(graph::op_attr::use_affine, true); //inference
    groupnorm_op.set_attr<bool>(graph::op_attr::keep_stats, false); //inference
    groupnorm_op.set_attr<std::string>(graph::op_attr::data_format, "NCX");
    groupnorm_op.set_attr<int64_t>(graph::op_attr::groups, 3);

    graph::op_t mul_op {1, graph::op_kind::Multiply, "mul_op"};
    graph::op_t sigmoid_op {2, graph::op_kind::Sigmoid, "sigmoid_op"};

    graph::op_t typecast(3, graph::op_kind::TypeCast, "typecast");
    graph::op_t quantize(4, graph::op_kind::Quantize, "quantize");
    quantize.set_attr<std::vector<float>>(graph::op_attr::scales, {0.1f});
    quantize.set_attr<std::vector<int64_t>>(graph::op_attr::zps, {9});
    quantize.set_attr<std::string>(graph::op_attr::qtype, "per_tensor");

    std::vector<int64_t> groupnorm_shape {2, 3, 2, 2};
    graph::logical_tensor_t src_lt = utils::logical_tensor_init(
            0, groupnorm_shape, graph::data_type::bf16);
    graph::logical_tensor_t scale_lt
            = utils::logical_tensor_init(1, {3}, graph::data_type::f32);
    graph::logical_tensor_t shift_lt
            = utils::logical_tensor_init(2, {3}, graph::data_type::f32);
    graph::logical_tensor_t groupnorm_dst_lt = utils::logical_tensor_init(
            3, groupnorm_shape, graph::data_type::bf16);
    graph::logical_tensor_t sigmoid_dst_lt = utils::logical_tensor_init(
            4, groupnorm_shape, graph::data_type::bf16);
    graph::logical_tensor_t mul_dst_lt = utils::logical_tensor_init(
            5, groupnorm_shape, graph::data_type::bf16);
    graph::logical_tensor_t tc_dst_lt = utils::logical_tensor_init(
            6, groupnorm_shape, graph::data_type::f32);
    graph::logical_tensor_t quant_dst_lt = utils::logical_tensor_init(
            7, groupnorm_shape, graph::data_type::u8);

    graph::engine_t *engine = get_engine();
    graph::graph_t g(engine->kind());

    groupnorm_op.add_input(src_lt);
    groupnorm_op.add_input(scale_lt);
    groupnorm_op.add_input(shift_lt);
    groupnorm_op.add_output(groupnorm_dst_lt);
    sigmoid_op.add_input(groupnorm_dst_lt);
    sigmoid_op.add_output(sigmoid_dst_lt);
    mul_op.add_input(groupnorm_dst_lt);
    mul_op.add_input(sigmoid_dst_lt);
    mul_op.add_output(mul_dst_lt);
    typecast.add_input(mul_dst_lt);
    typecast.add_output(tc_dst_lt);
    quantize.add_input(tc_dst_lt);
    quantize.add_output(quant_dst_lt);

    ASSERT_EQ(g.add_op(&groupnorm_op), graph::status::success);
    ASSERT_EQ(g.add_op(&sigmoid_op), graph::status::success);
    ASSERT_EQ(g.add_op(&mul_op), graph::status::success);
    ASSERT_EQ(g.add_op(&typecast), graph::status::success);
    ASSERT_EQ(g.add_op(&quantize), graph::status::success);
    ASSERT_EQ(g.finalize(), graph::status::success);

    graph::pass::pass_base_ptr apass = get_pass("groupnorm_post_ops_fusion");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile
    graph::partition_t p;
    p.init(part);
    graph::compiled_partition_t cp(p);

    std::vector<const graph::logical_tensor_t *> inputs {
            &src_lt, &scale_lt, &shift_lt};
    std::vector<const graph::logical_tensor_t *> outputs {&quant_dst_lt};

    ASSERT_EQ(p.compile(&cp, inputs, outputs, engine), graph::status::success);

    test_tensor_t src_ts(src_lt, eng, src);
    test_tensor_t scale_ts(scale_lt, eng, scale);
    test_tensor_t shift_ts(shift_lt, eng, shift);
    test_tensor_t dst_ts(quant_dst_lt, eng);
    test_tensor_t ref_dst_ts(quant_dst_lt, eng);

    graph::stream_t *strm = get_stream();
    cp.execute(strm, {src_ts.get(), scale_ts.get(), shift_ts.get()},
            {dst_ts.get()});
    strm->wait();
    ASSERT_EQ(run_graph(g, {src_ts, scale_ts, shift_ts}, {ref_dst_ts}, *engine,
                      *strm),
            graph::status::success);
    auto ref_data = ref_dst_ts.as_vec_type<uint8_t>();
    auto dst_data = dst_ts.as_vec_type<uint8_t>();
    for (size_t i = 0; i < dst_data.size(); ++i) {
        ASSERT_LE(ref_data[i] - dst_data[i], 1);
    }
}
