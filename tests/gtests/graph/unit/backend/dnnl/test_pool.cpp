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

#include "graph/unit/backend/dnnl/dnnl_test_common.hpp"

TEST(ExecuteSubgraphInt8, PoolAdd) {
    using dims = impl::dnnl_impl::dims;
    using config_t = std::tuple<impl::op_kind_t, bool>;

    impl::engine_t *engine = get_engine();
    impl::stream_t *strm = get_stream();

    const std::vector<config_t> confs {config_t {impl::op_kind::AvgPool, true},
            config_t {impl::op_kind::AvgPool, false},
            config_t {impl::op_kind::MaxPool, true},
            config_t {impl::op_kind::MaxPool, false}};
    const std::vector<std::string> qtypes {"symmetric", "asymmetric"};
    const std::vector<bool> swap_add_ins {true, false};

    for_(const auto swap_add_in : swap_add_ins)
    for_(const auto &qtype : qtypes)
    for (const auto &conf : confs) {
        if (engine->kind() == impl::engine_kind::gpu && qtype == "asymmetric")
            continue;

        impl::op_kind_t base_op = impl::op_kind::Wildcard;
        bool per_channel_broadcast = false;
        std::tie(base_op, per_channel_broadcast) = conf;

        const std::string data_format {"NCX"};
        const int64_t channels = 2;
        std::vector<int64_t> src_shape {2, channels, 4, 4};
        std::vector<int64_t> dst_shape {2, channels, 2, 2};
        std::vector<int64_t> other_shape {1, 1, 1, 1};
        if (per_channel_broadcast) other_shape[1] = channels;

        test::vector<int8_t> src_s8_data(product(src_shape));
        test::vector<int8_t> other_s8_data(product(other_shape));
        test::vector<int8_t> case1_dst_s8_data(product(dst_shape));
        test::vector<int8_t> case2_dst_s8_data(product(dst_shape));

        std::default_random_engine generator(7);
        std::uniform_real_distribution<float> s8_distribution(-127.0f, 128.0f);
        std::generate(src_s8_data.begin(), src_s8_data.end(), [&]() {
            return static_cast<uint8_t>(s8_distribution(generator));
        });
        std::generate(other_s8_data.begin(), other_s8_data.end(), [&]() {
            return static_cast<uint8_t>(s8_distribution(generator));
        });

        const float scale_src = 5 / 127.f;
        const float scale_out = 10 / 127.f;
        const float scale_other = 2 / 127.f;
        const int64_t zp_src = (qtype == "symmetric") ? 0 : -2;
        const int64_t zp_out = (qtype == "symmetric") ? 0 : -2;
        const int64_t zp_other = (qtype == "symmetric") ? 0 : 4;

        impl::op_t dqdata_op(0, impl::op_kind::Dequantize, "dqdata_op");
        dqdata_op.set_attr<std::string>(impl::op_attr::qtype, "per_tensor");
        dqdata_op.set_attr<std::vector<int64_t>>(impl::op_attr::zps, {zp_src});
        dqdata_op.set_attr<std::vector<float>>(
                impl::op_attr::scales, {scale_src});
        dqdata_op.set_attr<int64_t>(impl::op_attr::axis, 1);

        impl::op_t pool_op(1, base_op, "pool_op");
        size_t spatial_size = src_shape.size() - 2;
        pool_op.set_attr<dims>(impl::op_attr::strides, dims(spatial_size, 2));
        pool_op.set_attr<dims>(impl::op_attr::kernel, dims(spatial_size, 2));
        pool_op.set_attr<dims>(
                impl::op_attr::pads_begin, dims(spatial_size, 0));
        pool_op.set_attr<dims>(impl::op_attr::pads_end, dims(spatial_size, 0));
        pool_op.set_attr<std::string>(impl::op_attr::data_format, data_format);
        if (base_op == impl::op_kind::AvgPool)
            pool_op.set_attr<bool>(impl::op_attr::exclude_pad, false);
        else
            // MaxPool
            pool_op.set_attr<dims>(
                    impl::op_attr::dilations, dims(spatial_size, 1));

        impl::op_t qout_op(2, impl::op_kind::Quantize, "qout_op");
        qout_op.set_attr<std::string>(impl::op_attr::qtype, "per_tensor");
        qout_op.set_attr<std::vector<int64_t>>(impl::op_attr::zps, {zp_out});
        qout_op.set_attr<std::vector<float>>(
                impl::op_attr::scales, {scale_out});
        qout_op.set_attr<int64_t>(impl::op_attr::axis, 1);

        impl::op_t dqother_op(3, impl::op_kind::Dequantize, "dqother_op");
        dqother_op.set_attr<std::string>(impl::op_attr::qtype, "per_tensor");
        dqother_op.set_attr<std::vector<int64_t>>(
                impl::op_attr::zps, {zp_other});
        dqother_op.set_attr<std::vector<float>>(
                impl::op_attr::scales, {scale_other});
        dqother_op.set_attr<int64_t>(impl::op_attr::axis, 1);

        impl::op_t add_op(4, impl::op_kind::Add, "add_op");

        auto src_s8
                = utils::logical_tensor_init(0, src_shape, impl::data_type::s8);
        auto src_f32_dq = utils::logical_tensor_init(
                1, src_shape, impl::data_type::f32);
        auto dst_f32 = utils::logical_tensor_init(
                2, dst_shape, impl::data_type::f32);
        auto dst_s8
                = utils::logical_tensor_init(3, dst_shape, impl::data_type::s8);
        auto other_s8 = utils::logical_tensor_init(
                4, other_shape, impl::data_type::s8);
        auto other_f32_dq = utils::logical_tensor_init(
                5, other_shape, impl::data_type::f32);
        auto dst_add_f32 = utils::logical_tensor_init(
                6, dst_shape, impl::data_type::f32);

        dqdata_op.add_input(src_s8);
        dqdata_op.add_output(src_f32_dq);

        pool_op.add_input(src_f32_dq);
        pool_op.add_output(dst_f32);

        dqother_op.add_input(other_s8);
        dqother_op.add_output(other_f32_dq);

        if (swap_add_in) {
            add_op.add_input(other_f32_dq);
            add_op.add_input(dst_f32);
        } else {
            add_op.add_input(dst_f32);
            add_op.add_input(other_f32_dq);
        }
        add_op.add_output(dst_add_f32);

        qout_op.add_input(dst_add_f32);
        qout_op.add_output(dst_s8);

        impl::graph_t g(engine->kind());
        g.add_op(&dqdata_op);
        g.add_op(&pool_op);
        g.add_op(&dqother_op);
        g.add_op(&add_op);
        g.add_op(&qout_op);
        g.build_graph();

        impl::tensor_t src_s8_ts(src_s8, engine, src_s8_data.data());
        impl::tensor_t other_s8_ts(other_s8, engine, other_s8_data.data());
        impl::tensor_t case1_dst_s8_ts(
                dst_s8, engine, case1_dst_s8_data.data());
        impl::tensor_t case2_dst_s8_ts(
                dst_s8, engine, case2_dst_s8_data.data());

        // -------------------------case 1----------------------------------
        ASSERT_EQ(run_graph(g, {src_s8_ts, other_s8_ts}, {case1_dst_s8_ts},
                          *engine, *strm),
                impl::status::success);

        // -------------------------case 2----------------------------------
        impl::pass::pass_base_ptr apass
                = get_pass(engine->kind() == impl::engine_kind::gpu
                                ? "int8_pool_binary_fusion_gpu"
                                : "int8_pool_binary_fusion_cpu");
        apass->run(g);

        ASSERT_EQ(g.get_num_partitions(), 1U);
        auto part = g.get_partitions()[0];

        impl::partition_t p;
        p.init(part);

        impl::compiled_partition_t cp(p);
        std::vector<const impl::logical_tensor_t *> lt_ins {&src_s8, &other_s8};
        std::vector<const impl::logical_tensor_t *> lt_outs {&dst_s8};
        p.compile(&cp, lt_ins, lt_outs, engine);

        cp.execute(strm, {src_s8_ts, other_s8_ts}, {case2_dst_s8_ts});
        strm->wait();

        static auto isa = dnnl_get_effective_cpu_isa();
        if (isa < dnnl_cpu_isa_avx512_core_vnni)
            ASSERT_TRUE(allclose(case1_dst_s8_data, case2_dst_s8_data,
                    /*rtol*/ 0.1f,
                    /*atol*/ 1.f));
        else
            ASSERT_TRUE(allclose(case1_dst_s8_data, case2_dst_s8_data,
                    /*rtol*/ 0.01f,
                    /*atol*/ 1.f));
    }
}

TEST(ExecuteSubgraphFp32, Pool3Postops) {
    impl::engine_t *engine = get_engine();
    impl::stream_t *strm = get_stream();

    std::vector<int64_t> pool_src_shape {2, 2, 4, 4};
    std::vector<int64_t> pool_dst_shape {2, 2, 2, 2};

    std::vector<test::vector<float>> src_datas {};
    src_datas.emplace_back(product(pool_src_shape));
    // at most 3 additional input tensors
    for (size_t i = 0; i < 3; ++i)
        src_datas.emplace_back(product(pool_dst_shape));

    std::default_random_engine generator(7);
    std::uniform_real_distribution<float> f32_distribution(0.0f, 1.0f);
    for (auto &data : src_datas)
        std::generate(data.begin(), data.end(),
                [&]() { return f32_distribution(generator); });

    std::vector<impl::logical_tensor_t> lt_vec;
    lt_vec.emplace_back(utils::logical_tensor_init(
            0, pool_src_shape, impl::data_type::f32));
    // at most 7 tensors in the whole graph
    for (size_t i = 0; i < 7; ++i)
        lt_vec.emplace_back(utils::logical_tensor_init(
                i + 1, pool_dst_shape, impl::data_type::f32));

    const std::vector<impl::op_kind_t> pool_op_ts {
            impl::op_kind::AvgPool, impl::op_kind::MaxPool};
    const std::vector<impl::op_kind_t> binary_op_ts {impl::op_kind::Add,
            impl::op_kind::Divide, impl::op_kind::Maximum,
            impl::op_kind::Minimum, impl::op_kind::Multiply,
            impl::op_kind::Subtract};
    const std::vector<std::vector<impl::op_kind_t>> post_op_t_seqs {
            {impl::op_kind::Add, impl::op_kind::Subtract},
            {impl::op_kind::Minimum, impl::op_kind::Multiply,
                    impl::op_kind::Maximum},
            {impl::op_kind::Divide, impl::op_kind::Add, impl::op_kind::Add}};

    for_(auto &pool_op_t : pool_op_ts)
    for (auto &post_op_ts : post_op_t_seqs) {
        size_t lt_idx = 0;
        std::vector<size_t> input_lts {};
        std::vector<size_t> output_lts {};

        std::vector<int64_t> strides = {2, 2};
        std::vector<int64_t> pads_begin = {0, 0};
        std::vector<int64_t> pads_end = {0, 0};
        std::vector<int64_t> kernel = {2, 2};
        std::vector<int64_t> dilations = {1, 1};
        impl::op_t pool_op {0, pool_op_t, "pooling op"};
        pool_op.set_attr(impl::op_attr::strides, strides);
        pool_op.set_attr(impl::op_attr::pads_begin, pads_begin);
        pool_op.set_attr(impl::op_attr::pads_end, pads_end);
        pool_op.set_attr(impl::op_attr::kernel, kernel);
        pool_op.set_attr<std::string>(impl::op_attr::data_format, "NCX");
        if (pool_op_t == impl::op_kind::AvgPool)
            pool_op.set_attr<bool>(impl::op_attr::exclude_pad, false);
        else if (pool_op_t == impl::op_kind::MaxPool)
            pool_op.set_attr(impl::op_attr::dilations, dilations);
        pool_op.add_input(lt_vec[lt_idx]);
        input_lts.push_back(lt_idx);
        pool_op.add_output(lt_vec[++lt_idx]);

        std::vector<impl::op_t> post_ops {};
        for (size_t i = 0; i < post_op_ts.size(); ++i) {
            auto pop_t = post_op_ts[i];
            post_ops.emplace_back(impl::op_t {i + 1, pop_t, "post op"});

            post_ops.back().add_input(lt_vec[lt_idx]);
            if (std::find(binary_op_ts.begin(), binary_op_ts.end(), pop_t)
                    != binary_op_ts.end()) {
                post_ops.back().add_input(lt_vec[++lt_idx]);
                input_lts.push_back(lt_idx);
            }
            post_ops.back().add_output(lt_vec[++lt_idx]);
        }

        output_lts.push_back(lt_idx);

        impl::graph_t g(engine->kind());
        g.add_op(&pool_op);
        for (const auto &pop : post_ops)
            g.add_op(&pop);
        g.build_graph();

        std::vector<impl::tensor_t> src_tss {};
        for (size_t i = 0; i < input_lts.size(); ++i)
            src_tss.emplace_back(
                    lt_vec[input_lts[i]], engine, src_datas[i].data());

        // -------------------------case 1----------------------------------
        test::vector<float> case1_out_data(product(pool_dst_shape));
        impl::tensor_t case1_dst_ts(
                lt_vec[lt_idx], engine, case1_out_data.data());

        ASSERT_EQ(run_graph(g, src_tss, {case1_dst_ts}, *engine, *strm),
                impl::status::success);

        // -------------------------case 2----------------------------------
        impl::pass::pass_base_ptr apass = get_pass("pool_post_ops_fusion");
        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 1U);
        auto part = g.get_partitions()[0];

        impl::partition_t p;
        p.init(part);

        impl::compiled_partition_t cp(p);

        std::vector<const impl::logical_tensor_t *> lt_ins(input_lts.size());
        std::transform(input_lts.begin(), input_lts.end(), lt_ins.begin(),
                [&](size_t idx) -> impl::logical_tensor_t * {
                    return &lt_vec[idx];
                });
        std::vector<const impl::logical_tensor_t *> lt_outs {&lt_vec[lt_idx]};

        p.compile(&cp, lt_ins, lt_outs, engine);

        test::vector<float> case2_out_data(product(pool_dst_shape));
        impl::tensor_t case2_dst_ts(
                lt_vec[lt_idx], engine, case2_out_data.data());

        cp.execute(strm, src_tss, {case2_dst_ts});
        strm->wait();

        std::vector<std::pair<float, float>> out_data;
        for (size_t i = 0; i < case1_out_data.size(); ++i)
            out_data.emplace_back(case1_out_data[i], case2_out_data[i]);
        for (size_t i = 0; i < case1_out_data.size(); ++i) {
            ASSERT_FLOAT_EQ(case1_out_data[i], case2_out_data[i]);
        }
    }
}

TEST(Execute, AvgPoolExcludePad) {
    using dims = impl::dnnl_impl::dims;
    impl::engine_t *eng = get_engine();

    test::vector<float> src {-2.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0,
            3.0, -1.0, 0, 3.0, -2.0, -1.0, 4.0};
    test::vector<float> ref_dst {
            -2.0, 0.25, 0.5, 0.75, 0.5, 0.75, 3.0, -1.5, 4.0};
    test::vector<float> dst(ref_dst.size(), 0.0);

    impl::op_t avg_pool_op(0, impl::op_kind::AvgPool, "avgpool");
    avg_pool_op.set_attr<dims>(impl::op_attr::strides, {2, 2});
    avg_pool_op.set_attr<dims>(impl::op_attr::kernel, {2, 2});
    avg_pool_op.set_attr<dims>(impl::op_attr::pads_begin, {1, 1});
    avg_pool_op.set_attr<dims>(impl::op_attr::pads_end, {1, 1});
    avg_pool_op.set_attr<bool>(impl::op_attr::exclude_pad, true);
    avg_pool_op.set_attr<std::string>(impl::op_attr::data_format, "NCX");

    // prepare logical tensor
    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 1, 4, 4}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
            1, {1, 1, 3, 3}, impl::data_type::f32, impl::layout_type::any);

    avg_pool_op.add_input(src_lt);
    avg_pool_op.add_output(dst_lt);

    impl::graph_t g(eng->kind());
    g.add_op(&avg_pool_op);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("avg_pool_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile
    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> inputs {&src_lt};
    std::vector<const impl::logical_tensor_t *> outputs {&dst_lt};

    p.compile(&cp, inputs, outputs, eng);

    impl::logical_tensor_t lt;
    cp.query_logical_tensor(dst_lt.id, &lt);
    ASSERT_EQ(lt.layout_type, impl::layout_type::strided);

    impl::tensor_t src_ts(src_lt, eng, src.data());
    impl::tensor_t dst_ts(dst_lt, eng, dst.data());

    impl::stream_t *strm = get_stream();
    cp.execute(strm, {src_ts}, {dst_ts});
    strm->wait();

    for (size_t i = 0; i < dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(Execute, AvgPoolIncludePad) {
    using dims = impl::dnnl_impl::dims;
    impl::engine_t *eng = get_engine();

    test::vector<float> src {-2.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0,
            3.0, -1.0, 0, 3.0, -2.0, -1.0, 4.0};
    test::vector<float> ref_dst {
            -0.5, 0.125, 0.125, 0.375, 0.5, 0.375, 0.75, -0.75, 1.0};
    test::vector<float> dst(ref_dst.size(), 0.0);

    impl::op_t avg_pool_op(0, impl::op_kind::AvgPool, "avgpool");
    avg_pool_op.set_attr<dims>(impl::op_attr::strides, {2, 2});
    avg_pool_op.set_attr<dims>(impl::op_attr::kernel, {2, 2});
    avg_pool_op.set_attr<dims>(impl::op_attr::pads_begin, {1, 1});
    avg_pool_op.set_attr<dims>(impl::op_attr::pads_end, {1, 1});
    avg_pool_op.set_attr<std::string>(impl::op_attr::data_format, "NCX");
    avg_pool_op.set_attr<bool>(impl::op_attr::exclude_pad, false);

    // prepare logical tensor
    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 1, 4, 4}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
            1, {1, 1, 3, 3}, impl::data_type::f32, impl::layout_type::any);

    avg_pool_op.add_input(src_lt);
    avg_pool_op.add_output(dst_lt);

    impl::graph_t g(eng->kind());
    g.add_op(&avg_pool_op);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("avg_pool_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile
    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> inputs {&src_lt};
    std::vector<const impl::logical_tensor_t *> outputs {&dst_lt};

    p.compile(&cp, inputs, outputs, eng);

    impl::logical_tensor_t lt;
    cp.query_logical_tensor(dst_lt.id, &lt);
    ASSERT_EQ(lt.layout_type, impl::layout_type::strided);

    impl::tensor_t src_ts(src_lt, eng, src.data());
    impl::tensor_t dst_ts(dst_lt, eng, dst.data());

    impl::stream_t *strm = get_stream();
    cp.execute(strm, {src_ts}, {dst_ts});
    strm->wait();

    for (size_t i = 0; i < dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(Execute, AvgPoolBackwardExcludePad) {
    using dims = dnnl::impl::graph::dnnl_impl::dims;
    impl::engine_t *eng = get_engine();

    test::vector<float> ref_diff_src {-1.0, 1.5, 1.5, 10.0, 2.0, 4.0, 4.0, 4.0,
            2.0, 4.0, 4.0, 4.0, 12.0, -2.5, -2.5, -3.0};
    test::vector<float> diff_dst {
            -1.0, 3.0, 10.0, 4.0, 16.0, 8.0, 12.0, -5.0, -3.0};
    test::vector<float> diff_src(ref_diff_src.size(), 0.0);

    impl::op_t avg_pool_bwd_op(impl::op_kind::AvgPoolBackprop);
    avg_pool_bwd_op.set_attr<dims>(impl::op_attr::strides, {2, 2});
    avg_pool_bwd_op.set_attr<dims>(impl::op_attr::kernel, {2, 2});
    avg_pool_bwd_op.set_attr<dims>(impl::op_attr::pads_begin, {1, 1});
    avg_pool_bwd_op.set_attr<dims>(impl::op_attr::pads_end, {1, 1});
    avg_pool_bwd_op.set_attr<bool>(impl::op_attr::exclude_pad, true);
    avg_pool_bwd_op.set_attr<std::string>(impl::op_attr::data_format, "NCX");
    avg_pool_bwd_op.set_attr<std::vector<int64_t>>(
            impl::op_attr::input_shape, std::vector<int64_t> {1, 1, 4, 4});

    // prepare logical tensor
    impl::logical_tensor_t diff_dst_lt
            = utils::logical_tensor_init(1, {1, 1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t diff_src_lt = utils::logical_tensor_init(
            2, {1, 1, 4, 4}, impl::data_type::f32, impl::layout_type::any);

    avg_pool_bwd_op.add_input(diff_dst_lt);
    avg_pool_bwd_op.add_output(diff_src_lt);
    impl::graph_t g(eng->kind());
    ASSERT_EQ(g.add_op(&avg_pool_bwd_op), impl::status::success);
    g.build_graph();
    impl::pass::pass_base_ptr apass = get_pass("avg_pool_bw_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile
    impl::partition_t p;
    p.init(part);
    impl::compiled_partition_t cp(p);
    std::vector<const impl::logical_tensor_t *> inputs {&diff_dst_lt};
    std::vector<const impl::logical_tensor_t *> outputs {&diff_src_lt};
    ASSERT_EQ(p.compile(&cp, inputs, outputs, eng), impl::status::success);
    impl::logical_tensor_t lt;
    cp.query_logical_tensor(diff_src_lt.id, &lt);
    ASSERT_EQ(lt.layout_type, impl::layout_type::strided);

    impl::tensor_t diff_dst_ts(diff_dst_lt, eng, diff_dst.data());
    impl::tensor_t diff_src_ts(lt, eng, diff_src.data());

    impl::stream_t *strm = get_stream();
    cp.execute(strm, {diff_dst_ts}, {diff_src_ts});
    strm->wait();
    for (size_t i = 0; i < diff_src.size(); ++i) {
        ASSERT_FLOAT_EQ(diff_src[i], ref_diff_src[i]);
    }
}

TEST(Execute, AvgPoolBackwardIncludePad) {
    using dims = dnnl::impl::graph::dnnl_impl::dims;
    impl::engine_t *eng = get_engine();

    test::vector<float> ref_diff_src {-0.25, 0.75, 0.75, 2.5, 1.0, 4.0, 4.0,
            2.0, 1.0, 4.0, 4.0, 2.0, 3.0, -1.25, -1.25, -3.0 / 4};
    test::vector<float> diff_dst {
            -1.0, 3.0, 10.0, 4.0, 16.0, 8.0, 12.0, -5.0, -3.0};
    test::vector<float> diff_src(ref_diff_src.size(), 0.0);

    impl::op_t avg_pool_bwd_op(impl::op_kind::AvgPoolBackprop);
    avg_pool_bwd_op.set_attr<dims>(impl::op_attr::strides, {2, 2});
    avg_pool_bwd_op.set_attr<dims>(impl::op_attr::kernel, {2, 2});
    avg_pool_bwd_op.set_attr<dims>(impl::op_attr::pads_begin, {1, 1});
    avg_pool_bwd_op.set_attr<dims>(impl::op_attr::pads_end, {1, 1});
    avg_pool_bwd_op.set_attr<bool>(impl::op_attr::exclude_pad, false);
    avg_pool_bwd_op.set_attr<std::string>(impl::op_attr::data_format, "NCX");
    avg_pool_bwd_op.set_attr<std::vector<int64_t>>(
            impl::op_attr::input_shape, std::vector<int64_t> {1, 1, 4, 4});

    // prepare logical tensor
    impl::logical_tensor_t diff_dst_lt
            = utils::logical_tensor_init(1, {1, 1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t diff_src_lt = utils::logical_tensor_init(
            2, {1, 1, 4, 4}, impl::data_type::f32, impl::layout_type::any);

    avg_pool_bwd_op.add_input(diff_dst_lt);
    avg_pool_bwd_op.add_output(diff_src_lt);
    impl::graph_t g(eng->kind());
    ASSERT_EQ(g.add_op(&avg_pool_bwd_op), impl::status::success);
    g.build_graph();
    impl::pass::pass_base_ptr apass = get_pass("avg_pool_bw_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile
    impl::partition_t p;
    p.init(part);
    impl::compiled_partition_t cp(p);
    std::vector<const impl::logical_tensor_t *> inputs {&diff_dst_lt};
    std::vector<const impl::logical_tensor_t *> outputs {&diff_src_lt};
    ASSERT_EQ(p.compile(&cp, inputs, outputs, eng), impl::status::success);
    impl::logical_tensor_t lt;
    cp.query_logical_tensor(diff_src_lt.id, &lt);
    ASSERT_EQ(lt.layout_type, impl::layout_type::strided);

    impl::tensor_t diff_dst_ts(diff_dst_lt, eng, diff_dst.data());
    impl::tensor_t diff_src_ts(lt, eng, diff_src.data());

    impl::stream_t *strm = get_stream();
    cp.execute(strm, {diff_dst_ts}, {diff_src_ts});
    strm->wait();
    for (size_t i = 0; i < diff_src.size(); ++i) {
        ASSERT_FLOAT_EQ(diff_src[i], ref_diff_src[i]);
    }
}

TEST(ExecuteSubgraphInt8, Avgpool) {
    // compare results between:
    // case 1: [quantize] - [dequantize] - [fp32_avgpool] - [quantize]
    // case 2: [quantize] - [int8_avgpool]
    using dims = impl::dnnl_impl::dims;
    impl::engine_t *engine = get_engine();
    impl::stream_t *strm = get_stream();

    std::vector<std::string> data_formats {"NCX", "NXC"};
    std::vector<std::vector<int64_t>> src_shapes {
            {3, 3, 4, 4, 4}, {3, 3, 4, 4}, {3, 3, 4}};
    std::vector<std::vector<int64_t>> dst_shapes {
            {3, 3, 2, 2, 2}, {3, 3, 2, 2}, {3, 3, 2}};
    for_(const auto &data_format : data_formats)
    for (size_t i = 0; i < src_shapes.size(); ++i) {
        // prepare fp32 data
        std::vector<int64_t> src_shape = src_shapes[i];
        std::vector<int64_t> dst_shape = dst_shapes[i];

        if (data_format == "NXC") {
            src_shape.emplace_back(src_shape[1]);
            src_shape.erase(src_shape.begin() + 1);
            dst_shape.emplace_back(dst_shape[1]);
            dst_shape.erase(dst_shape.begin() + 1);
        }

        test::vector<uint8_t> src_u8_data(product(src_shape));
        test::vector<int8_t> case1_out_data(product(dst_shape));
        test::vector<int8_t> case2_out_data(product(dst_shape));

        // random generate src, weight and bias data
        // random seed = 7
        std::default_random_engine generator(7);
        std::uniform_real_distribution<float> u8_distribution(0.0f, 255.0f);
        std::generate(src_u8_data.begin(), src_u8_data.end(), [&]() {
            return static_cast<uint8_t>(u8_distribution(generator));
        });

        float scale_src = 1 / 127.f;
        float scale_out = 1 / 127.f;
        int64_t zp_src = 0;
        int64_t zp_out = 0;

        impl::op_t dqdata_op(1, impl::op_kind::Dequantize, "dqdata_op");
        dqdata_op.set_attr<std::string>(impl::op_attr::qtype, "per_tensor");
        dqdata_op.set_attr<std::vector<int64_t>>(impl::op_attr::zps, {zp_src});
        dqdata_op.set_attr<std::vector<float>>(
                impl::op_attr::scales, {scale_src});
        dqdata_op.set_attr<int64_t>(impl::op_attr::axis, 0);

        impl::op_t avgpool_op(2, impl::op_kind::AvgPool, "avgpool_op");
        size_t spatial_size = src_shape.size() - 2;
        avgpool_op.set_attr<dims>(
                impl::op_attr::strides, dims(spatial_size, 2));
        avgpool_op.set_attr<dims>(impl::op_attr::kernel, dims(spatial_size, 2));
        avgpool_op.set_attr<dims>(
                impl::op_attr::pads_begin, dims(spatial_size, 0));
        avgpool_op.set_attr<dims>(
                impl::op_attr::pads_end, dims(spatial_size, 0));
        avgpool_op.set_attr<std::string>(
                impl::op_attr::data_format, data_format);
        avgpool_op.set_attr<bool>(impl::op_attr::exclude_pad, false);

        impl::op_t qout_op(3, impl::op_kind::Quantize, "qout_op");
        qout_op.set_attr<std::string>(impl::op_attr::qtype, "per_tensor");
        qout_op.set_attr<std::vector<int64_t>>(impl::op_attr::zps, {zp_out});
        qout_op.set_attr<std::vector<float>>(
                impl::op_attr::scales, {scale_out});
        qout_op.set_attr<int64_t>(impl::op_attr::axis, 0);

        // prepare logical tensor
        impl::logical_tensor_t src_u8
                = utils::logical_tensor_init(1, src_shape, impl::data_type::u8);
        impl::logical_tensor_t src_f32_dq = utils::logical_tensor_init(
                2, src_shape, impl::data_type::f32);
        impl::logical_tensor_t dst_f32 = utils::logical_tensor_init(
                3, dst_shape, impl::data_type::f32);
        impl::logical_tensor_t dst_u8
                = utils::logical_tensor_init(4, dst_shape, impl::data_type::u8);

        dqdata_op.add_input(src_u8);
        dqdata_op.add_output(src_f32_dq);

        avgpool_op.add_input(src_f32_dq);
        avgpool_op.add_output(dst_f32);

        qout_op.add_input(dst_f32);
        qout_op.add_output(dst_u8);

        impl::graph_t g(engine->kind());
        g.add_op(&dqdata_op);
        g.add_op(&avgpool_op);
        g.add_op(&qout_op);
        g.build_graph();

        impl::tensor_t src_u8_ts(src_u8, engine, src_u8_data.data());
        impl::tensor_t dst_u8_ts(dst_u8, engine, case1_out_data.data());
        impl::tensor_t dst_u8_case2_ts(dst_u8, engine, case2_out_data.data());

        // -------------------------case 1----------------------------------
        ASSERT_EQ(run_graph(g, {src_u8_ts}, {dst_u8_ts}, *engine, *strm),
                impl::status::success);

        // -------------------------case 2----------------------------------
        impl::pass::pass_base_ptr apass
                = get_pass(engine->kind() == impl::engine_kind::gpu
                                ? "int8_pool_binary_fusion_gpu"
                                : "int8_pool_binary_fusion_cpu");
        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 1U);
        auto part = g.get_partitions()[0];

        // compile
        impl::partition_t p;
        p.init(part);

        impl::compiled_partition_t cp(p);

        std::vector<const impl::logical_tensor_t *> lt_ins {&src_u8};
        std::vector<const impl::logical_tensor_t *> lt_outs {&dst_u8};

        p.compile(&cp, lt_ins, lt_outs, engine);

        cp.execute(strm, {src_u8_ts}, {dst_u8_case2_ts});
        strm->wait();

        static auto isa = dnnl_get_effective_cpu_isa();
        if (isa < dnnl_cpu_isa_avx512_core_vnni)
            ASSERT_TRUE(allclose(case1_out_data, case2_out_data, /*rtol*/ 0.1f,
                    /*atol*/ 1.f));
        else
            ASSERT_TRUE(allclose(case1_out_data, case2_out_data, /*rtol*/ 0.01f,
                    /*atol*/ 1.f));
    }
}

TEST(Execute, MaxPool) {
    using dims = impl::dnnl_impl::dims;
    impl::engine_t *eng = get_engine();

    test::vector<float> src {-2.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0,
            2.5, -1.0, 0, 3.0, -2.0, -1.0, 4.0};
    test::vector<float> ref_dst {-0.5, 2.0, 3.0, 4.0};
    test::vector<float> dst(ref_dst.size(), 0.0);

    impl::op_t max_pool_op(0, impl::op_kind::MaxPool, "maxpool");
    max_pool_op.set_attr<dims>(impl::op_attr::strides, {2, 2});
    max_pool_op.set_attr<dims>(impl::op_attr::kernel, {2, 2});
    max_pool_op.set_attr<dims>(impl::op_attr::pads_begin, {0, 0});
    max_pool_op.set_attr<dims>(impl::op_attr::pads_end, {0, 0});
    max_pool_op.set_attr<std::string>(impl::op_attr::data_format, "NCX");
    max_pool_op.set_attr<dims>(impl::op_attr::dilations, {1, 1});

    // prepare logical tensor
    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 1, 4, 4}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
            1, {1, 1, 2, 2}, impl::data_type::f32, impl::layout_type::any);

    max_pool_op.add_input(src_lt);
    max_pool_op.add_output(dst_lt);

    impl::graph_t g(eng->kind());
    g.add_op(&max_pool_op);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("max_pool_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile
    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> inputs {&src_lt};
    std::vector<const impl::logical_tensor_t *> outputs {&dst_lt};

    p.compile(&cp, inputs, outputs, eng);

    impl::logical_tensor_t lt;
    cp.query_logical_tensor(dst_lt.id, &lt);
    ASSERT_EQ(lt.layout_type, impl::layout_type::strided);

    impl::tensor_t src_ts(src_lt, eng, src.data());
    impl::tensor_t dst_ts(dst_lt, eng, dst.data());

    impl::stream_t *strm = get_stream();
    cp.execute(strm, {src_ts}, {dst_ts});
    strm->wait();

    for (size_t i = 0; i < dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(Execute, MaxPoolWithOpaqueInput) {
    // dequantize - maxpool
    using dims = impl::dnnl_impl::dims;
    impl::engine_t *eng = get_engine();

    // prepare ops
    impl::op_t dequantize(0, impl::op_kind::Dequantize, "dq");
    dequantize.set_attr<std::vector<float>>(impl::op_attr::scales, {0.1f});
    int64_t zps = eng->kind() == impl::engine_kind::gpu ? 0 : 10;

    dequantize.set_attr<std::vector<int64_t>>(impl::op_attr::zps, {zps});
    dequantize.set_attr<std::string>(impl::op_attr::qtype, "per_tensor");
    dequantize.set_attr<int64_t>(impl::op_attr::axis, 0);

    impl::op_t maxpool(1, impl::op_kind::MaxPool, "maxpool");
    maxpool.set_attr<dims>(impl::op_attr::strides, {2, 2});
    maxpool.set_attr<dims>(impl::op_attr::kernel, {2, 2});
    maxpool.set_attr<dims>(impl::op_attr::pads_begin, {0, 0});
    maxpool.set_attr<dims>(impl::op_attr::pads_end, {0, 0});
    maxpool.set_attr<std::string>(impl::op_attr::data_format, "NXC");
    maxpool.set_attr<dims>(impl::op_attr::dilations, {1, 1});

    // prepare input/output logical tensor
    impl::logical_tensor_t dq_src_lt = utils::logical_tensor_init(
            0, {1, 2, 2, 1}, impl::data_type::u8, impl::layout_type::strided);
    impl::logical_tensor_t dq_dst_lt = utils::logical_tensor_init(
            1, {1, 2, 2, 1}, impl::data_type::f32, impl::layout_type::any);
    impl::logical_tensor_t mp_dst_lt = utils::logical_tensor_init(
            2, {1, 1, 1, 1}, impl::data_type::f32, impl::layout_type::any);

    dequantize.add_input(dq_src_lt);
    dequantize.add_output(dq_dst_lt);
    maxpool.add_input(dq_dst_lt);
    maxpool.add_output(mp_dst_lt);

    impl::graph_t g(eng->kind());
    g.add_op(&dequantize);
    g.add_op(&maxpool);
    g.build_graph();

    impl::pass::pass_base_ptr apass1 = get_pass("dequant_pass");
    impl::pass::pass_base_ptr apass2 = get_pass("max_pool_pass");
    apass1->run(g);
    apass2->run(g);
    ASSERT_EQ(g.get_num_partitions(), 2U);
    auto dq_part = g.get_partitions()[0];
    auto mp_part = g.get_partitions()[1];

    // compile
    impl::partition_t dq_p, mp_p;
    dq_p.init(dq_part);
    mp_p.init(mp_part);

    impl::compiled_partition_t dq_cp(dq_p);
    impl::compiled_partition_t mp_cp(mp_p);

    std::vector<const impl::logical_tensor_t *> dq_inputs {&dq_src_lt};
    std::vector<const impl::logical_tensor_t *> dq_outputs {&dq_dst_lt};
    ASSERT_EQ(dq_p.compile(&dq_cp, dq_inputs, dq_outputs, eng),
            impl::status::success);

    impl::logical_tensor_t lt;
    dq_cp.query_logical_tensor(dq_dst_lt.id, &lt);
    ASSERT_EQ(lt.layout_type, impl::layout_type::strided);

    std::vector<const impl::logical_tensor_t *> mp_inputs {&lt};
    std::vector<const impl::logical_tensor_t *> mp_outputs {&mp_dst_lt};
    ASSERT_EQ(mp_p.compile(&mp_cp, mp_inputs, mp_outputs, eng),
            impl::status::success);

    mp_cp.query_logical_tensor(mp_dst_lt.id, &lt);
    ASSERT_EQ(lt.layout_type, impl::layout_type::strided);
}

TEST(Execute, MaxPoolBackwardWithIncides) {
    using dims = dnnl::impl::graph::dnnl_impl::dims;
    impl::engine_t *eng = get_engine();

    test::vector<float> src {-2.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0,
            3.0, -1.0, 0.0, 1.0, -2.0, -1.0, 4.0};
    test::vector<float> diff_src(src.size(), 0.0);
    test::vector<float> ref_diff_src {0.0, 0.0, 16.0, 0.0, 4.0, 0.0, 0.0, 0.0,
            0.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 12.0};

    test::vector<float> diff_dst {4.0, 16.0, 8.0, 12.0};

    void *indices_data = nullptr;
    test::vector<int32_t> s32_indices {2, 0, 1, 3};
    test::vector<uint8_t> u8_indices {2, 0, 1, 3};
    if (get_test_engine_kind() == impl::engine_kind::cpu) {
        indices_data = u8_indices.data();
    } else {
        indices_data = s32_indices.data();
    }

    impl::op_t max_pool_bwd_op(impl::op_kind::MaxPoolBackprop);

    max_pool_bwd_op.set_attr<dims>(impl::op_attr::strides, dims {2, 2});
    max_pool_bwd_op.set_attr<dims>(impl::op_attr::kernel, dims {2, 2});
    max_pool_bwd_op.set_attr<dims>(impl::op_attr::pads_begin, dims {0, 0});
    max_pool_bwd_op.set_attr<dims>(impl::op_attr::pads_end, dims {0, 0});
    max_pool_bwd_op.set_attr<dims>(impl::op_attr::dilations, {1, 1});
    max_pool_bwd_op.set_attr<std::string>(impl::op_attr::data_format, "NCX");

    // prepare logical tensor
    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 1, 4, 4}, impl::data_type::f32);
    impl::logical_tensor_t diff_src_lt = utils::logical_tensor_init(
            1, {1, 1, 4, 4}, impl::data_type::f32, impl::layout_type::any);
    impl::logical_tensor_t diff_dst_lt
            = utils::logical_tensor_init(3, {1, 1, 2, 2}, impl::data_type::f32);
    impl::logical_tensor_t indices_lt;
    if (get_test_engine_kind() == impl::engine_kind::cpu) {
        indices_lt = utils::logical_tensor_init(
                4, {1, 1, 2, 2}, impl::data_type::u8);
    } else {
        indices_lt = utils::logical_tensor_init(
                4, {1, 1, 2, 2}, impl::data_type::s32);
    }

    max_pool_bwd_op.add_input(src_lt);
    max_pool_bwd_op.add_input(diff_dst_lt);
    max_pool_bwd_op.add_input(indices_lt);
    max_pool_bwd_op.add_output(diff_src_lt);
    impl::graph_t g(eng->kind());
    ASSERT_EQ(g.add_op(&max_pool_bwd_op), impl::status::success);
    g.build_graph();
    impl::pass::pass_base_ptr apass = get_pass("max_pool_bw_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile
    impl::partition_t p;
    p.init(part);
    impl::compiled_partition_t cp(p);
    std::vector<const impl::logical_tensor_t *> inputs {
            &src_lt, &diff_dst_lt, &indices_lt};
    std::vector<const impl::logical_tensor_t *> outputs {&diff_src_lt};
    ASSERT_EQ(p.compile(&cp, inputs, outputs, eng), impl::status::success);
    impl::logical_tensor_t lt;
    cp.query_logical_tensor(diff_src_lt.id, &lt);
    ASSERT_EQ(lt.layout_type, impl::layout_type::strided);

    impl::tensor_t src_ts(src_lt, eng, src.data());
    impl::tensor_t diff_dst_ts(diff_dst_lt, eng, diff_dst.data());
    impl::tensor_t diff_src_ts(lt, eng, diff_src.data());
    impl::tensor_t indices_ts(indices_lt, eng, indices_data);

    impl::stream_t *strm = get_stream();
    cp.execute(strm, {src_ts, diff_dst_ts, indices_ts}, {diff_src_ts});
    strm->wait();

    for (size_t i = 0; i < diff_src.size(); ++i) {
        ASSERT_FLOAT_EQ(diff_src[i], ref_diff_src[i]);
    }
}

TEST(Execute, MaxPoolBackwardWithoutIncides) {
    using dims = dnnl::impl::graph::dnnl_impl::dims;
    impl::engine_t *eng = get_engine();

    test::vector<float> src {-2.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0,
            3.0, -1.0, 0.0, 1.0, -2.0, -1.0, 4.0};
    test::vector<float> diff_src(src.size(), 0.0);
    test::vector<float> ref_diff_src {0.0, 0.0, 16.0, 0.0, 4.0, 0.0, 0.0, 0.0,
            0.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 12.0};

    test::vector<float> diff_dst {4.0, 16.0, 8.0, 12.0};

    impl::op_t max_pool_bwd_op(impl::op_kind::MaxPoolBackprop);

    max_pool_bwd_op.set_attr<dims>(impl::op_attr::strides, dims {2, 2});
    max_pool_bwd_op.set_attr<dims>(impl::op_attr::kernel, dims {2, 2});
    max_pool_bwd_op.set_attr<dims>(impl::op_attr::pads_begin, dims {0, 0});
    max_pool_bwd_op.set_attr<dims>(impl::op_attr::pads_end, dims {0, 0});
    max_pool_bwd_op.set_attr<dims>(impl::op_attr::dilations, {1, 1});
    max_pool_bwd_op.set_attr<std::string>(impl::op_attr::data_format, "NCX");

    // prepare logical tensor
    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 1, 4, 4}, impl::data_type::f32);
    impl::logical_tensor_t diff_src_lt = utils::logical_tensor_init(
            1, {1, 1, 4, 4}, impl::data_type::f32, impl::layout_type::any);
    impl::logical_tensor_t diff_dst_lt
            = utils::logical_tensor_init(3, {1, 1, 2, 2}, impl::data_type::f32);

    max_pool_bwd_op.add_input(src_lt);
    max_pool_bwd_op.add_input(diff_dst_lt);
    max_pool_bwd_op.add_output(diff_src_lt);
    impl::graph_t g(eng->kind());
    ASSERT_EQ(g.add_op(&max_pool_bwd_op), impl::status::success);
    g.build_graph();
    impl::pass::pass_base_ptr apass = get_pass("max_pool_bw_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile
    impl::partition_t p;
    p.init(part);
    impl::compiled_partition_t cp(p);
    std::vector<const impl::logical_tensor_t *> inputs {&src_lt, &diff_dst_lt};
    std::vector<const impl::logical_tensor_t *> outputs {&diff_src_lt};
    ASSERT_EQ(p.compile(&cp, inputs, outputs, eng), impl::status::success);
    impl::logical_tensor_t lt;
    cp.query_logical_tensor(diff_src_lt.id, &lt);
    ASSERT_EQ(lt.layout_type, impl::layout_type::strided);

    impl::tensor_t src_ts(src_lt, eng, src.data());
    impl::tensor_t diff_dst_ts(diff_dst_lt, eng, diff_dst.data());
    impl::tensor_t diff_src_ts(lt, eng, diff_src.data());

    impl::stream_t *strm = get_stream();
    cp.execute(strm, {src_ts, diff_dst_ts}, {diff_src_ts});
    strm->wait();

    for (size_t i = 0; i < diff_src.size(); ++i) {
        ASSERT_FLOAT_EQ(diff_src[i], ref_diff_src[i]);
    }
}

TEST(Execute, MaxPoolBackwardWithoutIncidesPlainGrad) {
    using dims = dnnl::impl::graph::dnnl_impl::dims;
    impl::engine_t *eng = get_engine();

    impl::op_t max_pool_bwd_op(impl::op_kind::MaxPoolBackprop);

    max_pool_bwd_op.set_attr<dims>(impl::op_attr::strides, dims {2, 2});
    max_pool_bwd_op.set_attr<dims>(impl::op_attr::kernel, dims {2, 2});
    max_pool_bwd_op.set_attr<dims>(impl::op_attr::pads_begin, dims {0, 0});
    max_pool_bwd_op.set_attr<dims>(impl::op_attr::pads_end, dims {0, 0});
    max_pool_bwd_op.set_attr<dims>(impl::op_attr::dilations, {1, 1});
    max_pool_bwd_op.set_attr<std::string>(impl::op_attr::data_format, "NCX");

    dims input_dims = {1, 8, 4, 4};
    dims input_stride = {128, 1, 32, 8};
    dims output_dims = {1, 8, 2, 2};
    // prepare logical tensor
    impl::logical_tensor_t src_lt = utils::logical_tensor_init(
            0, input_dims, input_stride, impl::data_type::f32);
    impl::logical_tensor_t diff_src_lt = utils::logical_tensor_init(
            1, input_dims, impl::data_type::f32, impl::layout_type::any);
    impl::logical_tensor_t diff_dst_lt
            = utils::logical_tensor_init(3, output_dims, impl::data_type::f32);

    max_pool_bwd_op.add_input(src_lt);
    max_pool_bwd_op.add_input(diff_dst_lt);
    max_pool_bwd_op.add_output(diff_src_lt);
    impl::graph_t g(eng->kind());
    ASSERT_EQ(g.add_op(&max_pool_bwd_op), impl::status::success);
    g.build_graph();
    impl::pass::pass_base_ptr apass = get_pass("max_pool_bw_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile
    impl::partition_t p;
    p.init(part);
    impl::compiled_partition_t cp(p);
    std::vector<const impl::logical_tensor_t *> inputs {&src_lt, &diff_dst_lt};
    std::vector<const impl::logical_tensor_t *> outputs {&diff_src_lt};
    ASSERT_EQ(p.compile(&cp, inputs, outputs, eng), impl::status::success);
    impl::logical_tensor_t lt;
    cp.query_logical_tensor(diff_src_lt.id, &lt);
    ASSERT_EQ(lt.layout_type, impl::layout_type::strided);

    test::vector<float> src(product(input_dims), 1);
    test::vector<float> diff_dst(product(output_dims), 1);
    test::vector<float> diff_src(product(input_dims), 1);

    impl::tensor_t src_ts(src_lt, eng, src.data());
    impl::tensor_t diff_dst_ts(diff_dst_lt, eng, diff_dst.data());
    impl::tensor_t diff_src_ts(lt, eng, diff_src.data());

    impl::stream_t *strm = get_stream();
    cp.execute(strm, {src_ts, diff_dst_ts}, {diff_src_ts});
    strm->wait();
}

TEST(ExecuteSubgraphInt8, Maxpool) {
    // compare results between:
    // case 1: [quantize] - [dequantize] - [fp32_maxpool] - [quantize]
    // case 2: [quantize] - [int8_maxpool]
    using dims = impl::dnnl_impl::dims;
    impl::engine_t *engine = get_engine();
    impl::stream_t *strm = get_stream();

    std::vector<std::string> data_formats {"NCX", "NXC"};
    std::vector<std::vector<int64_t>> src_shapes {
            {3, 3, 4, 4, 4}, {3, 3, 4, 4}, {3, 3, 4}};
    std::vector<std::vector<int64_t>> dst_shapes {
            {3, 3, 2, 2, 2}, {3, 3, 2, 2}, {3, 3, 2}};
    for_(const auto &data_format : data_formats)
    for (size_t i = 0; i < src_shapes.size(); ++i) {
        // prepare fp32 data
        std::vector<int64_t> src_shape = src_shapes[i];
        std::vector<int64_t> dst_shape = dst_shapes[i];

        if (data_format == "NXC") {
            src_shape.emplace_back(src_shape[1]);
            src_shape.erase(src_shape.begin() + 1);
            dst_shape.emplace_back(dst_shape[1]);
            dst_shape.erase(dst_shape.begin() + 1);
        }

        test::vector<uint8_t> src_u8_data(product(src_shape));
        test::vector<int8_t> case1_out_data(product(dst_shape));
        test::vector<int8_t> case2_out_data(product(dst_shape));

        // random generate src, weight and bias data
        // random seed = 7
        std::default_random_engine generator(7);
        std::uniform_real_distribution<float> u8_distribution(0.0f, 255.0f);
        std::generate(src_u8_data.begin(), src_u8_data.end(), [&]() {
            return static_cast<uint8_t>(u8_distribution(generator));
        });

        float scale_src = 1 / 127.f;
        float scale_out = 1 / 127.f;
        int64_t zp_src = 0;
        int64_t zp_out = 0;

        impl::op_t dqdata_op(1, impl::op_kind::Dequantize, "dqdata_op");
        dqdata_op.set_attr<std::string>(impl::op_attr::qtype, "per_tensor");
        dqdata_op.set_attr<std::vector<int64_t>>(impl::op_attr::zps, {zp_src});
        dqdata_op.set_attr<std::vector<float>>(
                impl::op_attr::scales, {scale_src});
        dqdata_op.set_attr<int64_t>(impl::op_attr::axis, 0);

        impl::op_t maxpool_op(2, impl::op_kind::MaxPool, "maxpool_op");
        size_t spatial_size = src_shape.size() - 2;
        maxpool_op.set_attr<dims>(
                impl::op_attr::strides, dims(spatial_size, 2));
        maxpool_op.set_attr<dims>(impl::op_attr::kernel, dims(spatial_size, 2));
        maxpool_op.set_attr<dims>(
                impl::op_attr::pads_begin, dims(spatial_size, 0));
        maxpool_op.set_attr<dims>(
                impl::op_attr::pads_end, dims(spatial_size, 0));
        maxpool_op.set_attr<std::string>(
                impl::op_attr::data_format, data_format);
        maxpool_op.set_attr<dims>(
                impl::op_attr::dilations, dims(spatial_size, 1));

        impl::op_t qout_op(3, impl::op_kind::Quantize, "qout_op");
        qout_op.set_attr<std::string>(impl::op_attr::qtype, "per_tensor");
        qout_op.set_attr<std::vector<int64_t>>(impl::op_attr::zps, {zp_out});
        qout_op.set_attr<std::vector<float>>(
                impl::op_attr::scales, {scale_out});
        qout_op.set_attr<int64_t>(impl::op_attr::axis, 0);

        // prepare logical tensor
        impl::logical_tensor_t src_u8
                = utils::logical_tensor_init(1, src_shape, impl::data_type::u8);
        impl::logical_tensor_t src_f32_dq = utils::logical_tensor_init(
                2, src_shape, impl::data_type::f32);
        impl::logical_tensor_t dst_f32 = utils::logical_tensor_init(
                3, dst_shape, impl::data_type::f32);
        impl::logical_tensor_t dst_u8
                = utils::logical_tensor_init(4, dst_shape, impl::data_type::u8);

        dqdata_op.add_input(src_u8);
        dqdata_op.add_output(src_f32_dq);

        maxpool_op.add_input(src_f32_dq);
        maxpool_op.add_output(dst_f32);

        qout_op.add_input(dst_f32);
        qout_op.add_output(dst_u8);

        impl::graph_t g(engine->kind());
        g.add_op(&dqdata_op);
        g.add_op(&maxpool_op);
        g.add_op(&qout_op);
        g.build_graph();

        impl::tensor_t src_u8_ts(src_u8, engine, src_u8_data.data());
        impl::tensor_t dst_u8_ts(dst_u8, engine, case1_out_data.data());
        impl::tensor_t dst_u8_case2_ts(dst_u8, engine, case2_out_data.data());

        // -------------------------case 1----------------------------------
        ASSERT_EQ(run_graph(g, {src_u8_ts}, {dst_u8_ts}, *engine, *strm),
                impl::status::success);

        // -------------------------case 2----------------------------------
        impl::pass::pass_base_ptr apass
                = get_pass(engine->kind() == impl::engine_kind::gpu
                                ? "int8_pool_binary_fusion_gpu"
                                : "int8_pool_binary_fusion_cpu");
        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 1U);
        auto part = g.get_partitions()[0];

        // compile
        impl::partition_t p;
        p.init(part);

        impl::compiled_partition_t cp(p);

        std::vector<const impl::logical_tensor_t *> lt_ins {&src_u8};
        std::vector<const impl::logical_tensor_t *> lt_outs {&dst_u8};

        p.compile(&cp, lt_ins, lt_outs, engine);

        cp.execute(strm, {src_u8_ts}, {dst_u8_case2_ts});
        strm->wait();

        static auto isa = dnnl_get_effective_cpu_isa();
        if (isa < dnnl_cpu_isa_avx512_core_vnni)
            ASSERT_TRUE(allclose(case1_out_data, case2_out_data, /*rtol*/ 0.1f,
                    /*atol*/ 1.f));
        else
            ASSERT_TRUE(allclose(case1_out_data, case2_out_data, /*rtol*/ 0.01f,
                    /*atol*/ 1.f));
    }
}

struct pool_binary_params_t {
    impl::op_kind_t pool_kind;
    impl::op_kind_t binary_kind;
};

class pool_binary_t : public ::testing::TestWithParam<pool_binary_params_t> {
public:
    void TestPoolBinary() {
        const auto params
                = ::testing::TestWithParam<pool_binary_params_t>::GetParam();
        using dims = impl::dnnl_impl::dims;

        impl::engine_t *eng = get_engine();

        std::vector<std::string> data_formats {"NCX", "NXC"};
        std::vector<bool> with_channel_broadcast_flags {true, false};
        std::vector<impl::data_type_t> data_types {
                impl::data_type::f32, impl::data_type::bf16};

        for_(const auto dt : data_types)
        for_(const auto &data_format : data_formats)
        for (const auto c_broadcast : with_channel_broadcast_flags) {
            static auto isa = dnnl_get_effective_cpu_isa();
            if (dt == impl::data_type::bf16 && isa < dnnl_cpu_isa_avx512_core
                    && eng->kind() == impl::engine_kind::cpu) {
                continue;
            }

            std::vector<int64_t> src_shape {3, 3, 4, 4, 4};
            std::vector<int64_t> dst_shape {3, 3, 2, 2, 2};
            const size_t spatial_size = src_shape.size() - 2;
            std::vector<int64_t> post_src_shape {1, 1, 1, 1, 1};

            if (c_broadcast) { post_src_shape[1] = src_shape[1]; }
            if (data_format == "NXC") {
                src_shape.emplace_back(src_shape[1]);
                src_shape.erase(src_shape.begin() + 1);
                dst_shape.emplace_back(dst_shape[1]);
                dst_shape.erase(dst_shape.begin() + 1);
                post_src_shape.emplace_back(post_src_shape[1]);
                post_src_shape.erase(post_src_shape.begin() + 1);
            }

            test::vector<float> src(product(src_shape), 4.0);
            test::vector<float> dst(product(dst_shape), 0.0);
            test::vector<float> post_src(product(post_src_shape), 2.0);

            impl::op_t pool_op(0, params.pool_kind, "pool");
            pool_op.set_attr<dims>(
                    impl::op_attr::strides, dims(spatial_size, 2));
            pool_op.set_attr<dims>(
                    impl::op_attr::kernel, dims(spatial_size, 2));
            pool_op.set_attr<dims>(
                    impl::op_attr::pads_begin, dims(spatial_size, 0));
            pool_op.set_attr<dims>(
                    impl::op_attr::pads_end, dims(spatial_size, 0));
            pool_op.set_attr<std::string>(
                    impl::op_attr::data_format, data_format);
            if (params.pool_kind == impl::op_kind::AvgPool) {
                pool_op.set_attr<bool>(impl::op_attr::exclude_pad, false);
            } else {
                pool_op.set_attr<dims>(
                        impl::op_attr::dilations, dims(spatial_size, 1));
            }

            impl::op_t binary_op(1, params.binary_kind, "binary");

            impl::logical_tensor_t src_lt
                    = utils::logical_tensor_init(0, src_shape, dt);
            impl::logical_tensor_t dst_lt
                    = utils::logical_tensor_init(1, dst_shape, dt);
            impl::logical_tensor_t post_src_lt
                    = utils::logical_tensor_init(2, post_src_shape, dt);
            impl::logical_tensor_t add_dst_lt
                    = utils::logical_tensor_init(3, dst_shape, dt);

            pool_op.add_input(src_lt);
            pool_op.add_output(dst_lt);
            binary_op.add_input(dst_lt);
            binary_op.add_input(post_src_lt);
            binary_op.add_output(add_dst_lt);

            impl::graph_t g(eng->kind());
            g.add_op(&pool_op);
            g.add_op(&binary_op);
            g.build_graph();

            impl::pass::pass_base_ptr apass = get_pass("pool_post_ops_fusion");
            apass->run(g);
            ASSERT_EQ(g.get_num_partitions(), 1U);
            auto part = g.get_partitions()[0];

            // compile
            impl::partition_t p;
            p.init(part);

            impl::compiled_partition_t cp(p);

            std::vector<const impl::logical_tensor_t *> inputs {
                    &src_lt, &post_src_lt};
            std::vector<const impl::logical_tensor_t *> outputs {&add_dst_lt};

            ASSERT_EQ(p.compile(&cp, inputs, outputs, eng),
                    impl::status::success);

            impl::logical_tensor_t lt;
            cp.query_logical_tensor(add_dst_lt.id, &lt);
            ASSERT_EQ(lt.layout_type, impl::layout_type::strided);

            impl::tensor_t src_ts(src_lt, eng, src.data());
            impl::tensor_t post_src_ts(post_src_lt, eng, post_src.data());
            impl::tensor_t add_dst_ts(add_dst_lt, eng, dst.data());

            impl::stream_t *strm = get_stream();
            ASSERT_EQ(cp.execute(strm, {src_ts, post_src_ts}, {add_dst_ts}),
                    impl::status::success);
            strm->wait();
        }
    }
};

TEST_P(pool_binary_t, TestPoolBinary) {
    TestPoolBinary();
}

INSTANTIATE_TEST_SUITE_P(Execute, pool_binary_t,
        ::testing::Values(pool_binary_params_t {impl::op_kind::AvgPool,
                                  impl::op_kind::Add},
                pool_binary_params_t {
                        impl::op_kind::MaxPool, impl::op_kind::Add},
                pool_binary_params_t {
                        impl::op_kind::AvgPool, impl::op_kind::Divide},
                pool_binary_params_t {
                        impl::op_kind::AvgPool, impl::op_kind::Maximum},
                pool_binary_params_t {
                        impl::op_kind::MaxPool, impl::op_kind::Minimum},
                pool_binary_params_t {
                        impl::op_kind::AvgPool, impl::op_kind::Multiply},
                pool_binary_params_t {
                        impl::op_kind::MaxPool, impl::op_kind::Subtract}));
