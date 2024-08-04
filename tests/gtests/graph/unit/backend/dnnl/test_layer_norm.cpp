/*******************************************************************************
* Copyright 2020-2024 Intel Corporation
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

// primitive only support 2-5D data tensor for layernorm,
TEST(test_layer_norm_execute, LayernormNDimCheck) {
    graph::engine_t *engine = get_engine();

    std::vector<std::vector<int64_t>> src_shapes {
            {}, {2}, {2, 3, 4, 5, 6, 7}, {2, 3}};
    std::vector<size_t> expected_partition_num {0, 0, 0, 1, 1};

    graph::logical_tensor_t scale_lt
            = utils::logical_tensor_init(1, graph::data_type::f32);
    graph::logical_tensor_t shift_lt
            = utils::logical_tensor_init(2, graph::data_type::f32);
    graph::logical_tensor_t dst_lt
            = utils::logical_tensor_init(3, graph::data_type::f32);
    graph::logical_tensor_t mean_lt
            = utils::logical_tensor_init(4, graph::data_type::f32);
    graph::logical_tensor_t variance_lt
            = utils::logical_tensor_init(5, graph::data_type::f32);

    // the last ndim is DNNL_GRAPH_UNKNOWN_NDIMS
    for (size_t i = 0; i < src_shapes.size() + 1; i++) {
        graph::logical_tensor_t src_lt;
        if (i == src_shapes.size())
            // ndim is DNNL_GRAPH_UNKNOWN_NDIMS
            src_lt = utils::logical_tensor_init(0, graph::data_type::f32);
        else
            src_lt = utils::logical_tensor_init(
                    0, src_shapes[i], graph::data_type::f32);
        graph::graph_t g(engine->kind());

        graph::op_t layernorm_op(graph::op_kind::LayerNorm);
        layernorm_op.add_input(src_lt);
        layernorm_op.add_input(scale_lt);
        layernorm_op.add_input(shift_lt);
        layernorm_op.add_output(dst_lt);
        layernorm_op.add_output(mean_lt);
        layernorm_op.add_output(variance_lt);

        ASSERT_EQ(g.add_op(&layernorm_op), graph::status::success);
        g.finalize();

        graph::pass::pass_base_ptr apass = get_pass("ln_pass");
        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), expected_partition_num[i]);
    }
}

TEST(test_layer_norm_execute, LayernormTraining) {
    graph::engine_t *eng = get_engine();

    std::vector<float> src {2.0, 4.0, 5.0, 2.0, 3.0, 5.0};
    std::vector<float> scale {1.0, 2.0};
    std::vector<float> shift {0.0, 1.0};
    std::vector<float> ref_dst {-1.0, 3.0, 1.0, -1.0, -1.0, 3.0};
    std::vector<float> ref_mean {3.0, 3.5, 4.0};
    std::vector<float> ref_var {1.0, 2.25, 1.0};
    std::vector<float> dst(src.size(), 0.0);
    std::vector<float> mean(ref_mean.size(), 0.0);
    std::vector<float> var(ref_var.size(), 0.0);

    graph::op_t layernorm_op(graph::op_kind::LayerNorm);

    layernorm_op.set_attr<float>(graph::op_attr::epsilon, 0);

    graph::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 3, 2}, graph::data_type::f32);
    graph::logical_tensor_t scale_lt
            = utils::logical_tensor_init(1, {2}, graph::data_type::f32);
    graph::logical_tensor_t shift_lt
            = utils::logical_tensor_init(2, {2}, graph::data_type::f32);
    graph::logical_tensor_t dst_lt
            = utils::logical_tensor_init(3, {1, 3, 2}, graph::data_type::f32);
    graph::logical_tensor_t mean_lt
            = utils::logical_tensor_init(4, {1, 3}, graph::data_type::f32);
    graph::logical_tensor_t variance_lt
            = utils::logical_tensor_init(5, {1, 3}, graph::data_type::f32);

    graph::engine_t *engine = get_engine();
    graph::graph_t g(engine->kind());

    layernorm_op.add_input(src_lt);
    layernorm_op.add_input(scale_lt);
    layernorm_op.add_input(shift_lt);
    layernorm_op.add_output(dst_lt);
    layernorm_op.add_output(mean_lt);
    layernorm_op.add_output(variance_lt);

    ASSERT_EQ(g.add_op(&layernorm_op), graph::status::success);
    g.finalize();

    graph::pass::pass_base_ptr apass = get_pass("ln_pass");
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

    test_tensor src_ts(src_lt, eng, src);
    test_tensor scale_ts(scale_lt, eng, scale);
    test_tensor shift_ts(shift_lt, eng, shift);
    test_tensor dst_ts(dst_lt, eng, dst);
    test_tensor mean_ts(mean_lt, eng, mean);
    test_tensor var_ts(variance_lt, eng, var);

    graph::stream_t *strm = get_stream();
    cp.execute(strm, {src_ts.get(), scale_ts.get(), shift_ts.get()},
            {dst_ts.get(), mean_ts.get(), var_ts.get()});
    strm->wait();

    dst = dst_ts.as_vec_type<float>();
    mean = mean_ts.as_vec_type<float>();
    var = var_ts.as_vec_type<float>();

    for (size_t i = 0; i < ref_dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }

    for (size_t i = 0; i < ref_mean.size(); ++i) {
        ASSERT_FLOAT_EQ(mean[i], ref_mean[i]);
        ASSERT_FLOAT_EQ(var[i], ref_var[i]);
    }
}

TEST(test_layer_norm_execute, LayernormInference) {
    graph::engine_t *eng = get_engine();

    std::vector<float> src {2.0, 4.0, 3.0, 5.5, 5.0, 4.0, 1.0, 2.5};
    std::vector<float> scale {1.0, 2.0};
    std::vector<float> shift {0.0, 1.0};
    std::vector<float> ref_dst {-1.0, 3.0, -1.0, 3.0, 1.0, -1.0, -1.0, 3.0};
    std::vector<float> dst(src.size(), 0.0);

    graph::op_t layernorm_op(graph::op_kind::LayerNorm);

    layernorm_op.set_attr<float>(graph::op_attr::epsilon, 0);
    layernorm_op.set_attr<bool>(graph::op_attr::keep_stats, false); //inference

    graph::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {2, 2, 2}, graph::data_type::f32);
    graph::logical_tensor_t scale_lt
            = utils::logical_tensor_init(1, {2}, graph::data_type::f32);
    graph::logical_tensor_t shift_lt
            = utils::logical_tensor_init(2, {2}, graph::data_type::f32);
    graph::logical_tensor_t dst_lt
            = utils::logical_tensor_init(3, {2, 2, 2}, graph::data_type::f32);

    graph::engine_t *engine = get_engine();
    graph::graph_t g(engine->kind());

    layernorm_op.add_input(src_lt);
    layernorm_op.add_input(scale_lt);
    layernorm_op.add_input(shift_lt);
    layernorm_op.add_output(dst_lt);

    ASSERT_EQ(g.add_op(&layernorm_op), graph::status::success);
    g.finalize();

    graph::pass::pass_base_ptr apass = get_pass("ln_pass");
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

    test_tensor src_ts(src_lt, eng, src);
    test_tensor scale_ts(scale_lt, eng, scale);
    test_tensor shift_ts(shift_lt, eng, shift);
    test_tensor dst_ts(dst_lt, eng, dst);

    graph::stream_t *strm = get_stream();
    cp.execute(strm, {src_ts.get(), scale_ts.get(), shift_ts.get()},
            {dst_ts.get()});
    strm->wait();

    dst = dst_ts.as_vec_type<float>();
    for (size_t i = 0; i < ref_dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(test_layer_norm_execute, LayernormInferenceWithoutScaleShift) {
    graph::engine_t *eng = get_engine();

    std::vector<float> src {2.0, 4.0, 3.0, 5.5, 5.0, 4.0, 1.0, 2.5};
    std::vector<float> ref_dst {-1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0};
    std::vector<float> dst(src.size(), 0.0);

    graph::op_t layernorm_op(graph::op_kind::LayerNorm);

    layernorm_op.set_attr<float>(graph::op_attr::epsilon, 0);
    layernorm_op.set_attr<bool>(graph::op_attr::keep_stats, false); //inference
    layernorm_op.set_attr<bool>(graph::op_attr::use_affine, false);

    graph::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {2, 2, 2}, graph::data_type::f32);
    graph::logical_tensor_t dst_lt
            = utils::logical_tensor_init(3, {2, 2, 2}, graph::data_type::f32);

    graph::engine_t *engine = get_engine();
    graph::graph_t g(engine->kind());

    layernorm_op.add_input(src_lt);
    layernorm_op.add_output(dst_lt);

    ASSERT_EQ(g.add_op(&layernorm_op), graph::status::success);
    g.finalize();

    graph::pass::pass_base_ptr apass = get_pass("ln_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile
    graph::partition_t p;
    p.init(part);
    graph::compiled_partition_t cp(p);

    std::vector<const graph::logical_tensor_t *> inputs {&src_lt};
    std::vector<const graph::logical_tensor_t *> outputs {&dst_lt};

    ASSERT_EQ(p.compile(&cp, inputs, outputs, engine), graph::status::success);

    test_tensor src_ts(src_lt, eng, src);
    test_tensor dst_ts(dst_lt, eng, dst);

    graph::stream_t *strm = get_stream();
    cp.execute(strm, {src_ts.get()}, {dst_ts.get()});
    strm->wait();
    dst = dst_ts.as_vec_type<float>();
    for (size_t i = 0; i < ref_dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(test_layer_norm_execute, LayerNormBackwardFp32) {
    using dims = graph::dnnl_impl::dims;

    graph::engine_t *engine = get_engine();
    graph::stream_t *strm = get_stream();

    const graph::dim_t T = 2;
    const graph::dim_t N = 2;
    const graph::dim_t C = 2;
    const dims data_dims {T, N, C};
    const dims diff_data_dims {T, N, C};
    const dims stats_dims {T, N};
    const dims scaleshift_dims {C};

    // inputs data
    std::vector<float> src_data {
            1.0f, -1.0f, 2.0f, 0.0f, 1.0f, -1.0f, 2.0f, 0.0f};
    std::vector<float> diff_dst_data {
            12.0f, -2.0f, 6.0f, -1.0f, 3.0f, -0.5f, 1.5f, -0.25f};
    std::vector<float> diff_dst_data_no_affine {
            1.5f, -0.5f, 0.75f, -0.25f, 0.375f, -0.125f, 0.1875f, -0.0625f};
    std::vector<float> mean_data {0.0f, 1.0f, 0.0f, 1.0f};
    std::vector<float> var_data {3.9375f, 0.9375f, 0.1875f, 3.9375f};
    std::vector<float> scale_data {0.125f, 0.25f};

    // outputs data
    std::vector<float> diff_src_data(src_data.size());
    std::vector<float> diff_scale_data(scale_data.size());
    std::vector<float> diff_shift_data(scale_data.size());

    // expected outputs data
    std::vector<float> ref_diff_src_data {
            0.375f, -0.375f, 0.0f, 0.0f, -1.5f, 1.5f, 0.046875f, -0.046875f};
    std::vector<float> ref_diff_scale_data {18.75f, 3.125f};
    std::vector<float> ref_diff_shift_data {22.5f, -3.75f};

    const float epsilon {0.0625f};

    const std::vector<bool> use_affine_flags {true, false};
    for (const auto use_affine : use_affine_flags) {
        graph::op_t ln_bwd_op(graph::op_kind::LayerNormBackward);
        ln_bwd_op.set_attr<float>(graph::op_attr::epsilon, epsilon);
        ln_bwd_op.set_attr<bool>(graph::op_attr::use_affine, use_affine);

        graph::logical_tensor_t src = utils::logical_tensor_init(0, data_dims,
                graph::data_type::f32, graph::layout_type::strided);
        graph::logical_tensor_t diff_dst
                = utils::logical_tensor_init(1, diff_data_dims,
                        graph::data_type::f32, graph::layout_type::strided);
        graph::logical_tensor_t mean = utils::logical_tensor_init(2, stats_dims,
                graph::data_type::f32, graph::layout_type::strided);
        graph::logical_tensor_t var = utils::logical_tensor_init(3, stats_dims,
                graph::data_type::f32, graph::layout_type::strided);
        graph::logical_tensor_t scale
                = utils::logical_tensor_init(4, scaleshift_dims,
                        graph::data_type::f32, graph::layout_type::strided);
        graph::logical_tensor_t diff_src
                = utils::logical_tensor_init(5, diff_data_dims,
                        graph::data_type::f32, graph::layout_type::strided);
        graph::logical_tensor_t diff_scale
                = utils::logical_tensor_init(6, scaleshift_dims,
                        graph::data_type::f32, graph::layout_type::strided);
        graph::logical_tensor_t diff_shift
                = utils::logical_tensor_init(7, scaleshift_dims,
                        graph::data_type::f32, graph::layout_type::strided);

        ln_bwd_op.add_input(src);
        ln_bwd_op.add_input(diff_dst);
        ln_bwd_op.add_input(mean);
        ln_bwd_op.add_input(var);
        ln_bwd_op.add_input(scale);
        ln_bwd_op.add_output(diff_src);
        if (use_affine) {
            ln_bwd_op.add_output(diff_scale);
            ln_bwd_op.add_output(diff_shift);
        }

        graph::graph_t g(engine->kind());
        g.add_op(&ln_bwd_op);
        g.finalize();

        graph::pass::pass_base_ptr apass = get_pass("ln_bw_pass");
        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 1U);
        auto part = g.get_partitions()[0];

        graph::partition_t p;
        p.init(part);
        graph::compiled_partition_t cp(p);

        std::vector<const graph::logical_tensor_t *> inputs {
                &src, &diff_dst, &mean, &var, &scale};
        std::vector<const graph::logical_tensor_t *> outputs {&diff_src};
        if (use_affine) {
            outputs.push_back(&diff_scale);
            outputs.push_back(&diff_shift);
        }

        ASSERT_EQ(p.compile(&cp, inputs, outputs, engine),
                graph::status::success);

        auto inplace_pairs = cp.get_inplace_pairs();
        ASSERT_EQ(inplace_pairs.size(), 1U);
        ASSERT_EQ(inplace_pairs[0].input_id, diff_dst.id);
        ASSERT_EQ(inplace_pairs[0].output_id, diff_src.id);

        test_tensor src_ts(src, engine, src_data);
        test_tensor diff_dst_ts(diff_dst, engine,
                use_affine ? diff_dst_data : diff_dst_data_no_affine);
        test_tensor mean_ts(mean, engine, mean_data);
        test_tensor var_ts(var, engine, var_data);
        test_tensor scale_ts(scale, engine, scale_data);
        test_tensor diff_src_ts(diff_src, engine, diff_src_data);
        test_tensor diff_scale_ts(diff_scale, engine, diff_scale_data);
        test_tensor diff_shift_ts(diff_shift, engine, diff_shift_data);

        std::vector<test_tensor> inputs_ts {
                src_ts, diff_dst_ts, mean_ts, var_ts, scale_ts};
        std::vector<test_tensor> outputs_ts {diff_src_ts};
        if (use_affine) {
            outputs_ts.push_back(diff_scale_ts);
            outputs_ts.push_back(diff_shift_ts);
        }

        cp.execute(strm, test_tensor::to_graph_tensor(inputs_ts),
                test_tensor::to_graph_tensor(outputs_ts));
        strm->wait();

        const float abs_err {0.001f};
        diff_src_data = diff_src_ts.as_vec_type<float>();
        diff_scale_data = diff_scale_ts.as_vec_type<float>();
        diff_shift_data = diff_shift_ts.as_vec_type<float>();
        for (size_t i = 0; i < diff_src_data.size(); ++i) {
            ASSERT_NEAR(ref_diff_src_data[i], diff_src_data[i], abs_err);
        }
        if (use_affine) {
            for (size_t i = 0; i < diff_scale_data.size(); ++i) {
                ASSERT_NEAR(
                        ref_diff_scale_data[i], diff_scale_data[i], abs_err);
            }
            for (size_t i = 0; i < diff_shift_data.size(); ++i) {
                ASSERT_NEAR(
                        ref_diff_shift_data[i], diff_shift_data[i], abs_err);
            }
        }
    }
}

TEST(test_layer_norm_execute_subgraph_int8, LayernormTypecastQuant_CPU) {
    graph::engine_t *engine = get_engine();
    graph::stream_t *strm = get_stream();
    static auto isa = dnnl_get_effective_cpu_isa();
    SKIP_IF((isa < dnnl_cpu_isa_avx512_core)
                    && engine->kind() == graph::engine_kind::cpu,
            "Skip bf16 tests for systems that do not support avx512_core.");
    // some cases can't pass correctness check on GPU
    SKIP_IF(engine->kind() == graph::engine_kind::gpu,
            "Skip for GPU - not supported yet");

    std::vector<int64_t> layernorm_shape {2, 2, 2};
    std::vector<int64_t> scale_lt_shape {2};
    std::vector<int64_t> shift_lt_shape {2};
    std::vector<bfloat16_t> src_data(product(layernorm_shape));

    // random seed = 7
    std::default_random_engine generator(7);
    std::uniform_real_distribution<float> src_distribution(0.f, 1.f);
    std::generate(src_data.begin(), src_data.end(),
            [&]() { return src_distribution(generator); });

    graph::op_t layernorm_op(0, graph::op_kind::LayerNorm, "layernorm");
    layernorm_op.set_attr<float>(graph::op_attr::epsilon, 0);
    layernorm_op.set_attr<bool>(graph::op_attr::keep_stats, false); //inference

    graph::op_t typecast(1, graph::op_kind::TypeCast, "typecast");
    graph::op_t quantize(2, graph::op_kind::Quantize, "quantize");
    quantize.set_attr<std::vector<float>>(graph::op_attr::scales, {0.1f});
    quantize.set_attr<std::vector<int64_t>>(graph::op_attr::zps, {0});
    quantize.set_attr<std::string>(graph::op_attr::qtype, "per_tensor");

    // prepare logical tensor
    graph::logical_tensor_t src = utils::logical_tensor_init(
            0, layernorm_shape, graph::data_type::bf16);
    graph::logical_tensor_t scale_lt = utils::logical_tensor_init(
            1, scale_lt_shape, graph::data_type::f32);
    graph::logical_tensor_t shift_lt = utils::logical_tensor_init(
            2, shift_lt_shape, graph::data_type::f32);

    graph::logical_tensor_t layernorm_dst = utils::logical_tensor_init(
            3, layernorm_shape, graph::data_type::bf16);
    graph::logical_tensor_t tc_dst = utils::logical_tensor_init(
            4, layernorm_shape, graph::data_type::f32);
    graph::logical_tensor_t quant_dst = utils::logical_tensor_init(
            5, layernorm_shape, graph::data_type::u8);

    layernorm_op.add_input(src);
    layernorm_op.add_input(scale_lt);
    layernorm_op.add_input(shift_lt);
    layernorm_op.add_output(layernorm_dst);
    typecast.add_input(layernorm_dst);
    typecast.add_output(tc_dst);
    quantize.add_input(tc_dst);
    quantize.add_output(quant_dst);

    graph::graph_t g(engine->kind());
    ASSERT_EQ(g.add_op(&layernorm_op), graph::status::success);
    ASSERT_EQ(g.add_op(&typecast), graph::status::success);
    ASSERT_EQ(g.add_op(&quantize), graph::status::success);
    ASSERT_EQ(g.finalize(), graph::status::success);

    graph::pass::pass_base_ptr apass
            = get_pass("layernorm_post_ops_fusion_cpu");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile
    graph::partition_t p;
    p.init(part);

    graph::compiled_partition_t cp(p);

    std::vector<const graph::logical_tensor_t *> lt_ins {
            &src, &scale_lt, &shift_lt};
    std::vector<const graph::logical_tensor_t *> lt_outs {&quant_dst};

    ASSERT_EQ(p.compile(&cp, lt_ins, lt_outs, engine), graph::status::success);

    std::vector<float> scale(product(scale_lt_shape));
    std::vector<float> shift(product(shift_lt_shape));

    test_tensor src_ts(src, engine, src_data);
    test_tensor scale_ts(scale_lt, engine, scale);
    test_tensor shift_ts(shift_lt, engine, shift);
    test_tensor dst_ts(quant_dst, engine);
    test_tensor ref_ts(quant_dst, engine);

    ASSERT_EQ(run_graph(g, {src_ts, scale_ts, shift_ts}, {ref_ts}, *engine,
                      *strm),
            graph::status::success);
    ASSERT_EQ(cp.execute(strm, {src_ts.get(), scale_ts.get(), shift_ts.get()},
                      {dst_ts.get()}),
            graph::status::success);
    strm->wait();
    auto dst_data = dst_ts.as_vec_type<uint8_t>();
    auto ref_data = ref_ts.as_vec_type<uint8_t>();
    for (size_t i = 0; i < ref_data.size(); ++i) {
        ASSERT_FLOAT_EQ(ref_data[i], dst_data[i]);
    }
}
