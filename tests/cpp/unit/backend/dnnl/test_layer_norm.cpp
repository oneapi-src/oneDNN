/*******************************************************************************
* Copyright 2020-2022 Intel Corporation
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

#include "cpp/unit/backend/dnnl/dnnl_test_common.hpp"

TEST(Execute, LayernormTraining) {
    impl::engine_t &eng = get_engine();

    test::vector<float> src {2.0, 4.0, 5.0, 2.0, 3.0, 5.0};
    test::vector<float> scale {1.0, 2.0};
    test::vector<float> shift {0.0, 1.0};
    test::vector<float> ref_dst {-1.0, 3.0, 1.0, -1.0, -1.0, 3.0};
    test::vector<float> ref_mean {3.0, 3.5, 4.0};
    test::vector<float> ref_var {1.0, 2.25, 1.0};
    test::vector<float> dst(src.size(), 0.0);
    test::vector<float> mean(ref_mean.size(), 0.0);
    test::vector<float> var(ref_var.size(), 0.0);

    impl::op_t layernorm_op(impl::op_kind::LayerNorm);

    layernorm_op.set_attr<float>(impl::op_attr::epsilon, 0);

    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 3, 2}, impl::data_type::f32);
    impl::logical_tensor_t scale_lt
            = utils::logical_tensor_init(1, {2}, impl::data_type::f32);
    impl::logical_tensor_t shift_lt
            = utils::logical_tensor_init(2, {2}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(3, {1, 3, 2}, impl::data_type::f32);
    impl::logical_tensor_t mean_lt
            = utils::logical_tensor_init(4, {1, 3}, impl::data_type::f32);
    impl::logical_tensor_t variance_lt
            = utils::logical_tensor_init(5, {1, 3}, impl::data_type::f32);

    impl::engine_t &engine = get_engine();
    impl::graph_t g(engine.kind());

    layernorm_op.add_input(src_lt);
    layernorm_op.add_input(scale_lt);
    layernorm_op.add_input(shift_lt);
    layernorm_op.add_output(dst_lt);
    layernorm_op.add_output(mean_lt);
    layernorm_op.add_output(variance_lt);

    ASSERT_EQ(g.add_op(&layernorm_op), impl::status::success);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("ln_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    // compile
    impl::partition_t p;
    p.init(part);
    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> inputs {
            &src_lt, &scale_lt, &shift_lt};
    std::vector<const impl::logical_tensor_t *> outputs {
            &dst_lt, &mean_lt, &variance_lt};

    ASSERT_EQ(p.compile(&cp, inputs, outputs, &engine), impl::status::success);

    impl::tensor_t src_ts(src_lt, &eng, src.data());
    impl::tensor_t scale_ts(scale_lt, &eng, scale.data());
    impl::tensor_t shift_ts(shift_lt, &eng, shift.data());
    impl::tensor_t dst_ts(dst_lt, &eng, dst.data());
    impl::tensor_t mean_ts(mean_lt, &eng, mean.data());
    impl::tensor_t var_ts(variance_lt, &eng, var.data());

    impl::stream_t &strm = get_stream();
    cp.execute(&strm, {src_ts, scale_ts, shift_ts}, {dst_ts, mean_ts, var_ts});
    strm.wait();

    for (size_t i = 0; i < ref_dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }

    for (size_t i = 0; i < ref_mean.size(); ++i) {
        ASSERT_FLOAT_EQ(mean[i], ref_mean[i]);
        ASSERT_FLOAT_EQ(var[i], ref_var[i]);
    }
}

TEST(Execute, LayernormInference) {
    impl::engine_t &eng = get_engine();

    test::vector<float> src {2.0, 4.0, 3.0, 5.5, 5.0, 4.0, 1.0, 2.5};
    test::vector<float> scale {1.0, 2.0};
    test::vector<float> shift {0.0, 1.0};
    test::vector<float> ref_dst {-1.0, 3.0, -1.0, 3.0, 1.0, -1.0, -1.0, 3.0};
    test::vector<float> dst(src.size(), 0.0);

    impl::op_t layernorm_op(impl::op_kind::LayerNorm);

    layernorm_op.set_attr<float>(impl::op_attr::epsilon, 0);
    layernorm_op.set_attr<bool>(impl::op_attr::keep_stats, false); //inference

    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {2, 2, 2}, impl::data_type::f32);
    impl::logical_tensor_t scale_lt
            = utils::logical_tensor_init(1, {2}, impl::data_type::f32);
    impl::logical_tensor_t shift_lt
            = utils::logical_tensor_init(2, {2}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(3, {2, 2, 2}, impl::data_type::f32);

    impl::engine_t &engine = get_engine();
    impl::graph_t g(engine.kind());

    layernorm_op.add_input(src_lt);
    layernorm_op.add_input(scale_lt);
    layernorm_op.add_input(shift_lt);
    layernorm_op.add_output(dst_lt);

    ASSERT_EQ(g.add_op(&layernorm_op), impl::status::success);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("ln_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    // compile
    impl::partition_t p;
    p.init(part);
    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> inputs {
            &src_lt, &scale_lt, &shift_lt};
    std::vector<const impl::logical_tensor_t *> outputs {&dst_lt};

    ASSERT_EQ(p.compile(&cp, inputs, outputs, &engine), impl::status::success);

    impl::tensor_t src_ts(src_lt, &eng, src.data());
    impl::tensor_t scale_ts(scale_lt, &eng, scale.data());
    impl::tensor_t shift_ts(shift_lt, &eng, shift.data());
    impl::tensor_t dst_ts(dst_lt, &eng, dst.data());

    impl::stream_t &strm = get_stream();
    cp.execute(&strm, {src_ts, scale_ts, shift_ts}, {dst_ts});
    strm.wait();

    for (size_t i = 0; i < ref_dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(Execute, LayernormInferenceWithoutScaleShift) {
    impl::engine_t &eng = get_engine();

    test::vector<float> src {2.0, 4.0, 3.0, 5.5, 5.0, 4.0, 1.0, 2.5};
    test::vector<float> ref_dst {-1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0};
    test::vector<float> dst(src.size(), 0.0);

    impl::op_t layernorm_op(impl::op_kind::LayerNorm);

    layernorm_op.set_attr<float>(impl::op_attr::epsilon, 0);
    layernorm_op.set_attr<bool>(impl::op_attr::keep_stats, false); //inference
    layernorm_op.set_attr<bool>(impl::op_attr::use_affine, false);

    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {2, 2, 2}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(3, {2, 2, 2}, impl::data_type::f32);

    impl::engine_t &engine = get_engine();
    impl::graph_t g(engine.kind());

    layernorm_op.add_input(src_lt);
    layernorm_op.add_output(dst_lt);

    ASSERT_EQ(g.add_op(&layernorm_op), impl::status::success);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("ln_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    // compile
    impl::partition_t p;
    p.init(part);
    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> inputs {&src_lt};
    std::vector<const impl::logical_tensor_t *> outputs {&dst_lt};

    ASSERT_EQ(p.compile(&cp, inputs, outputs, &engine), impl::status::success);

    impl::tensor_t src_ts(src_lt, &eng, src.data());
    impl::tensor_t dst_ts(dst_lt, &eng, dst.data());

    impl::stream_t &strm = get_stream();
    cp.execute(&strm, {src_ts}, {dst_ts});
    strm.wait();

    for (size_t i = 0; i < ref_dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(Execute, LayerNormBackpropFp32) {
    using dims = impl::dnnl_impl::dims;

    impl::engine_t &engine = get_engine();
    impl::stream_t &strm = get_stream();

    const impl::dim_t T = 2;
    const impl::dim_t N = 2;
    const impl::dim_t C = 2;
    const dims data_dims {T, N, C};
    const dims diff_data_dims {T, N, C};
    const dims stats_dims {T, N};
    const dims scaleshift_dims {C};

    // inputs data
    test::vector<float> src_data {
            1.0f, -1.0f, 2.0f, 0.0f, 1.0f, -1.0f, 2.0f, 0.0f};
    test::vector<float> diff_dst_data {
            12.0f, -2.0f, 6.0f, -1.0f, 3.0f, -0.5f, 1.5f, -0.25f};
    test::vector<float> diff_dst_data_no_affine {
            1.5f, -0.5f, 0.75f, -0.25f, 0.375f, -0.125f, 0.1875f, -0.0625f};
    test::vector<float> mean_data {0.0f, 1.0f, 0.0f, 1.0f};
    test::vector<float> var_data {3.9375f, 0.9375f, 0.1875f, 3.9375f};
    test::vector<float> scale_data {0.125f, 0.25f};

    // outputs data
    test::vector<float> diff_src_data(src_data.size());
    test::vector<float> diff_scale_data(scale_data.size());
    test::vector<float> diff_shift_data(scale_data.size());

    // expected outputs data
    test::vector<float> ref_diff_src_data {
            0.375f, -0.375f, 0.0f, 0.0f, -1.5f, 1.5f, 0.046875f, -0.046875f};
    test::vector<float> ref_diff_scale_data {18.75f, 3.125f};
    test::vector<float> ref_diff_shift_data {22.5f, -3.75f};

    const float epsilon {0.0625f};

    const std::vector<bool> use_affine_flags {true, false};
    for (const auto use_affine : use_affine_flags) {
        impl::op_t ln_bwd_op(impl::op_kind::LayerNormBackprop);
        ln_bwd_op.set_attr<float>(impl::op_attr::epsilon, epsilon);
        ln_bwd_op.set_attr<bool>(impl::op_attr::use_affine, use_affine);

        impl::logical_tensor_t src = utils::logical_tensor_init(
                0, data_dims, impl::data_type::f32, impl::layout_type::strided);
        impl::logical_tensor_t diff_dst
                = utils::logical_tensor_init(1, diff_data_dims,
                        impl::data_type::f32, impl::layout_type::strided);
        impl::logical_tensor_t mean = utils::logical_tensor_init(2, stats_dims,
                impl::data_type::f32, impl::layout_type::strided);
        impl::logical_tensor_t var = utils::logical_tensor_init(3, stats_dims,
                impl::data_type::f32, impl::layout_type::strided);
        impl::logical_tensor_t scale
                = utils::logical_tensor_init(4, scaleshift_dims,
                        impl::data_type::f32, impl::layout_type::strided);
        impl::logical_tensor_t diff_src
                = utils::logical_tensor_init(5, diff_data_dims,
                        impl::data_type::f32, impl::layout_type::strided);
        impl::logical_tensor_t diff_scale
                = utils::logical_tensor_init(6, scaleshift_dims,
                        impl::data_type::f32, impl::layout_type::strided);
        impl::logical_tensor_t diff_shift
                = utils::logical_tensor_init(7, scaleshift_dims,
                        impl::data_type::f32, impl::layout_type::strided);

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

        impl::graph_t g(engine.kind());
        g.add_op(&ln_bwd_op);
        g.build_graph();

        impl::pass::pass_base_ptr apass = get_pass("ln_bw_pass");
        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 1);
        auto part = g.get_partitions()[0];

        impl::partition_t p;
        p.init(part);
        impl::compiled_partition_t cp(p);

        std::vector<const impl::logical_tensor_t *> inputs {
                &src, &diff_dst, &mean, &var, &scale};
        std::vector<const impl::logical_tensor_t *> outputs {&diff_src};
        if (use_affine) {
            outputs.push_back(&diff_scale);
            outputs.push_back(&diff_shift);
        }

        ASSERT_EQ(p.compile(&cp, inputs, outputs, &engine),
                impl::status::success);

        auto inplace_pairs = cp.get_inplace_pairs();
        ASSERT_EQ(inplace_pairs.size(), 1);
        ASSERT_EQ(inplace_pairs[0].input_id, diff_dst.id);
        ASSERT_EQ(inplace_pairs[0].output_id, diff_src.id);

        impl::tensor_t src_ts(src, &engine, src_data.data());
        impl::tensor_t diff_dst_ts(diff_dst, &engine,
                use_affine ? diff_dst_data.data()
                           : diff_dst_data_no_affine.data());
        impl::tensor_t mean_ts(mean, &engine, mean_data.data());
        impl::tensor_t var_ts(var, &engine, var_data.data());
        impl::tensor_t scale_ts(scale, &engine, scale_data.data());
        impl::tensor_t diff_src_ts(diff_src, &engine, diff_src_data.data());
        impl::tensor_t diff_scale_ts(
                diff_scale, &engine, diff_scale_data.data());
        impl::tensor_t diff_shift_ts(
                diff_shift, &engine, diff_shift_data.data());

        std::vector<impl::tensor_t> inputs_ts {
                src_ts, diff_dst_ts, mean_ts, var_ts, scale_ts};
        std::vector<impl::tensor_t> outputs_ts {diff_src_ts};
        if (use_affine) {
            outputs_ts.push_back(diff_scale_ts);
            outputs_ts.push_back(diff_shift_ts);
        }

        cp.execute(&strm, inputs_ts, outputs_ts);
        strm.wait();

        const float abs_err {0.001f};
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
