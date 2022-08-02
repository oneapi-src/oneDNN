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

TEST(ExecuteSubgraphInt8, BmmU8u8f32) {
    impl::engine_t &engine = get_engine();
    impl::stream_t &strm = get_stream();

    // u8 to s8 shift by using a reroder with 128 zps is not supported on gpu
    SKIP_IF(engine.kind() == impl::engine_kind::gpu, "skip on gpu");

    std::string qtype = "per_tensor";
    // prepare fp32 data
    std::vector<int64_t> src_shape = {1, 4, 16, 8};
    std::vector<int64_t> weight_shape = {1, 4, 8, 16};
    std::vector<int64_t> dst_shape = {1, 4, 16, 16};

    test::vector<uint8_t> src_data(product(src_shape));
    test::vector<uint8_t> weight_data(product(weight_shape));

    // random generate src, weight data
    // random seed = 7
    std::default_random_engine generator(7);
    std::uniform_real_distribution<float> u8_distribution(0.0f, 255.0f);
    std::generate(src_data.begin(), src_data.end(),
            [&]() { return static_cast<uint8_t>(u8_distribution(generator)); });
    std::generate(weight_data.begin(), weight_data.end(),
            [&]() { return static_cast<uint8_t>(u8_distribution(generator)); });
    float scale_src = 1 / 255.f; // map to 0~255
    int64_t zp_src = 110;

    size_t scales_wei_sizes = qtype == "per_tensor" ? 1 : dst_shape.back();
    std::vector<float> scale_wei(scales_wei_sizes, 1 / 255.f);
    std::vector<int64_t> zp_wei(scales_wei_sizes, 114);

    // -------------------------case 1----------------------------------
    impl::op_t dqdata_op(1, impl::op_kind::Dequantize, "dqdata_op");
    dqdata_op.set_attr<std::string>(impl::op_attr::qtype, "per_tensor");
    dqdata_op.set_attr<std::vector<int64_t>>(impl::op_attr::zps, {zp_src});
    dqdata_op.set_attr<std::vector<float>>(impl::op_attr::scales, {scale_src});
    dqdata_op.set_attr<int64_t>(impl::op_attr::axis, 0);

    impl::op_t dqweight_op(2, impl::op_kind::Dequantize, "dqweight_op");
    dqweight_op.set_attr<std::string>(impl::op_attr::qtype, qtype);
    dqweight_op.set_attr<std::vector<int64_t>>(impl::op_attr::zps, zp_wei);
    dqweight_op.set_attr<std::vector<float>>(impl::op_attr::scales, scale_wei);
    dqweight_op.set_attr<int64_t>(impl::op_attr::axis, 1);

    impl::op_t matmul_op(3, impl::op_kind::MatMul, "matmul_op");
    matmul_op.set_attr<bool>(impl::op_attr::transpose_a, false);
    matmul_op.set_attr<bool>(impl::op_attr::transpose_b, false);

    // prepare logical tensor
    impl::logical_tensor_t src_u8
            = utils::logical_tensor_init(1, src_shape, impl::data_type::u8);
    impl::logical_tensor_t src_f32_dq
            = utils::logical_tensor_init(2, src_shape, impl::data_type::f32);
    impl::logical_tensor_t weight_u8
            = utils::logical_tensor_init(4, weight_shape, impl::data_type::u8);
    impl::logical_tensor_t weight_f32_dq
            = utils::logical_tensor_init(5, weight_shape, impl::data_type::f32);
    impl::logical_tensor_t dst_f32
            = utils::logical_tensor_init(7, dst_shape, impl::data_type::f32);

    dqdata_op.add_input(src_u8);
    dqdata_op.add_output(src_f32_dq);

    dqweight_op.add_input(weight_u8);
    dqweight_op.add_output(weight_f32_dq);

    matmul_op.add_input(src_f32_dq);
    matmul_op.add_input(weight_f32_dq);
    matmul_op.add_output(dst_f32);

    impl::graph_t g(engine.kind());
    g.add_op(&dqdata_op);
    g.add_op(&dqweight_op);
    g.add_op(&matmul_op);
    g.build_graph();

    impl::tensor_t src_u8_ts(src_u8, &engine, src_data.data());
    impl::tensor_t weight_u8_ts(weight_u8, &engine, weight_data.data());
    // -------------------------case 1----------------------------------
    test::vector<float> case1_out_data(product(dst_shape));
    impl::tensor_t dst_f32_ts(dst_f32, &engine, case1_out_data.data());
    ASSERT_EQ(
            run_graph(g, {src_u8_ts, weight_u8_ts}, {dst_f32_ts}, engine, strm),
            impl::status::success);
    // -------------------------case 2----------------------------------
    impl::pass::pass_base_ptr apass
            = get_pass("int8_matmul_post_ops_fusion_cpu");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    // compile
    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> lt_ins {&src_u8, &weight_u8};
    std::vector<const impl::logical_tensor_t *> lt_outs {&dst_f32};

    p.compile(&cp, lt_ins, lt_outs, &engine);

    test::vector<float> case2_out_data(product(dst_shape));
    impl::tensor_t dst_f32_case2_ts(dst_f32, &engine, case2_out_data.data());
    cp.execute(&strm, {src_u8_ts, weight_u8_ts}, {dst_f32_case2_ts});
    strm.wait();

    static auto isa = dnnl_get_effective_cpu_isa();
    if (isa >= dnnl_cpu_isa_avx512_core_vnni) {
        ASSERT_TRUE(allclose(case1_out_data, case2_out_data, /*rtol*/ 0.01f,
                /*atol*/ 1.f));
    }
}

TEST(ExecuteSubgraphInt8, BmmDivU8u8f32) {
    impl::engine_t &engine = get_engine();
    impl::stream_t &strm = get_stream();

    // u8 to s8 shift by using a reroder with 128 zps is not supported on gpu
    SKIP_IF(engine.kind() == impl::engine_kind::gpu, "skip on gpu");

    std::string qtype = "per_tensor";
    // prepare fp32 data
    std::vector<int64_t> src_shape = {1, 4, 16, 8};
    std::vector<int64_t> weight_shape = {1, 4, 8, 16};
    std::vector<int64_t> dst_shape = {1, 4, 16, 16};

    test::vector<uint8_t> src_data(product(src_shape));
    test::vector<uint8_t> weight_data(product(weight_shape));

    // random generate src, weight data
    // random seed = 7
    std::default_random_engine generator(7);
    std::uniform_real_distribution<float> u8_distribution(0.0f, 255.0f);
    std::generate(src_data.begin(), src_data.end(),
            [&]() { return static_cast<uint8_t>(u8_distribution(generator)); });
    std::generate(weight_data.begin(), weight_data.end(),
            [&]() { return static_cast<uint8_t>(u8_distribution(generator)); });
    test::vector<float> div_src1_data {0.5f};
    float scale_src = 1 / 255.f; // map to 0~255
    int64_t zp_src = 110;

    size_t scales_wei_sizes = qtype == "per_tensor" ? 1 : dst_shape.back();
    std::vector<float> scale_wei(scales_wei_sizes, 1 / 255.f);
    std::vector<int64_t> zp_wei(scales_wei_sizes, 114);

    // -------------------------case 1----------------------------------
    impl::op_t dqdata_op(1, impl::op_kind::Dequantize, "dqdata_op");
    dqdata_op.set_attr<std::string>(impl::op_attr::qtype, "per_tensor");
    dqdata_op.set_attr<std::vector<int64_t>>(impl::op_attr::zps, {zp_src});
    dqdata_op.set_attr<std::vector<float>>(impl::op_attr::scales, {scale_src});
    dqdata_op.set_attr<int64_t>(impl::op_attr::axis, 0);

    impl::op_t dqweight_op(2, impl::op_kind::Dequantize, "dqweight_op");
    dqweight_op.set_attr<std::string>(impl::op_attr::qtype, qtype);
    dqweight_op.set_attr<std::vector<int64_t>>(impl::op_attr::zps, zp_wei);
    dqweight_op.set_attr<std::vector<float>>(impl::op_attr::scales, scale_wei);
    dqweight_op.set_attr<int64_t>(impl::op_attr::axis, 1);

    impl::op_t matmul_op(3, impl::op_kind::MatMul, "matmul_op");
    matmul_op.set_attr<bool>(impl::op_attr::transpose_a, false);
    matmul_op.set_attr<bool>(impl::op_attr::transpose_b, false);

    impl::op_t binary_op(5, impl::op_kind::Divide, "binary_div");
    binary_op.set_attr<std::string>(impl::op_attr::auto_broadcast, "numpy");

    // prepare logical tensor
    impl::logical_tensor_t src_u8
            = utils::logical_tensor_init(1, src_shape, impl::data_type::u8);
    impl::logical_tensor_t src_f32_dq
            = utils::logical_tensor_init(2, src_shape, impl::data_type::f32);
    impl::logical_tensor_t weight_u8
            = utils::logical_tensor_init(4, weight_shape, impl::data_type::u8);
    impl::logical_tensor_t weight_f32_dq
            = utils::logical_tensor_init(5, weight_shape, impl::data_type::f32);
    impl::logical_tensor_t dst_f32
            = utils::logical_tensor_init(7, dst_shape, impl::data_type::f32);
    impl::logical_tensor_t div_src1
            = utils::logical_tensor_init(8, {1}, impl::data_type::f32);
    impl::logical_tensor_t div_f32
            = utils::logical_tensor_init(9, dst_shape, impl::data_type::f32);

    dqdata_op.add_input(src_u8);
    dqdata_op.add_output(src_f32_dq);

    dqweight_op.add_input(weight_u8);
    dqweight_op.add_output(weight_f32_dq);

    matmul_op.add_input(src_f32_dq);
    matmul_op.add_input(weight_f32_dq);
    matmul_op.add_output(dst_f32);

    binary_op.add_input(dst_f32);
    binary_op.add_input(div_src1);
    binary_op.add_output(div_f32);

    impl::graph_t g(engine.kind());
    g.add_op(&dqdata_op);
    g.add_op(&dqweight_op);
    g.add_op(&matmul_op);
    g.add_op(&binary_op);
    g.build_graph();

    impl::pass::pass_base_ptr apass
            = get_pass(engine.kind() == impl::engine_kind::gpu
                            ? "int8_matmul_post_ops_fusion_gpu"
                            : "int8_matmul_post_ops_fusion_cpu");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    // compile
    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> lt_ins {
            &src_u8, &weight_u8, &div_src1};
    std::vector<const impl::logical_tensor_t *> lt_outs {&div_f32};

    p.compile(&cp, lt_ins, lt_outs, &engine);

    impl::tensor_t src_u8_ts(src_u8, &engine, src_data.data());
    impl::tensor_t weight_u8_ts(weight_u8, &engine, weight_data.data());
    impl::tensor_t div_src1_ts(div_src1, &engine, div_src1_data.data());
    test::vector<float> case2_out_data(product(dst_shape));
    impl::tensor_t dst_f32_case2_ts(div_f32, &engine, case2_out_data.data());
    cp.execute(
            &strm, {src_u8_ts, weight_u8_ts, div_src1_ts}, {dst_f32_case2_ts});
    strm.wait();
}

TEST(ExecuteSubgraphInt8, BmmDivAddU8u8f32) {
    impl::engine_t &engine = get_engine();
    impl::stream_t &strm = get_stream();

    // u8 to s8 shift by using a reroder with 128 zps is not supported on gpu
    SKIP_IF(engine.kind() == impl::engine_kind::gpu, "skip on gpu");

    std::string qtype = "per_tensor";
    // prepare fp32 data
    std::vector<int64_t> src_shape = {1, 4, 16, 8};
    std::vector<int64_t> weight_shape = {1, 4, 8, 16};
    std::vector<int64_t> dst_shape = {1, 4, 16, 16};
    std::vector<int64_t> post_add_shape = {1, 1, 1, 16};

    test::vector<uint8_t> src_data(product(src_shape));
    test::vector<uint8_t> weight_data(product(weight_shape));
    test::vector<uint8_t> add_src1_data(product(post_add_shape));

    // random generate src, weight data
    // random seed = 7
    std::default_random_engine generator(7);
    std::uniform_real_distribution<float> u8_distribution(0.0f, 255.0f);
    std::generate(src_data.begin(), src_data.end(),
            [&]() { return static_cast<uint8_t>(u8_distribution(generator)); });
    std::generate(weight_data.begin(), weight_data.end(),
            [&]() { return static_cast<uint8_t>(u8_distribution(generator)); });
    std::generate(add_src1_data.begin(), add_src1_data.end(),
            [&]() { return u8_distribution(generator); });
    test::vector<float> div_src1_data {0.5f};
    float scale_src = 1 / 255.f; // map to 0~255
    int64_t zp_src = 110;

    size_t scales_wei_sizes = qtype == "per_tensor" ? 1 : dst_shape.back();
    std::vector<float> scale_wei(scales_wei_sizes, 1 / 255.f);
    std::vector<int64_t> zp_wei(scales_wei_sizes, 114);

    // -------------------------case 1----------------------------------
    impl::op_t dqdata_op(1, impl::op_kind::Dequantize, "dqdata_op");
    dqdata_op.set_attr<std::string>(impl::op_attr::qtype, "per_tensor");
    dqdata_op.set_attr<std::vector<int64_t>>(impl::op_attr::zps, {zp_src});
    dqdata_op.set_attr<std::vector<float>>(impl::op_attr::scales, {scale_src});
    dqdata_op.set_attr<int64_t>(impl::op_attr::axis, 0);

    impl::op_t dqweight_op(2, impl::op_kind::Dequantize, "dqweight_op");
    dqweight_op.set_attr<std::string>(impl::op_attr::qtype, qtype);
    dqweight_op.set_attr<std::vector<int64_t>>(impl::op_attr::zps, zp_wei);
    dqweight_op.set_attr<std::vector<float>>(impl::op_attr::scales, scale_wei);
    dqweight_op.set_attr<int64_t>(impl::op_attr::axis, 1);

    impl::op_t matmul_op(3, impl::op_kind::MatMul, "matmul_op");
    matmul_op.set_attr<bool>(impl::op_attr::transpose_a, false);
    matmul_op.set_attr<bool>(impl::op_attr::transpose_b, false);

    impl::op_t binary_op(5, impl::op_kind::Divide, "binary_div");
    binary_op.set_attr<std::string>(impl::op_attr::auto_broadcast, "numpy");

    impl::op_t binary_op2(6, impl::op_kind::Add, "binary_add");
    binary_op2.set_attr<std::string>(impl::op_attr::auto_broadcast, "numpy");

    // prepare logical tensor
    impl::logical_tensor_t src_u8
            = utils::logical_tensor_init(1, src_shape, impl::data_type::u8);
    impl::logical_tensor_t src_f32_dq
            = utils::logical_tensor_init(2, src_shape, impl::data_type::f32);
    impl::logical_tensor_t weight_u8
            = utils::logical_tensor_init(4, weight_shape, impl::data_type::u8);
    impl::logical_tensor_t weight_f32_dq
            = utils::logical_tensor_init(5, weight_shape, impl::data_type::f32);
    impl::logical_tensor_t dst_f32
            = utils::logical_tensor_init(7, dst_shape, impl::data_type::f32);
    impl::logical_tensor_t div_src1
            = utils::logical_tensor_init(8, {1}, impl::data_type::f32);
    impl::logical_tensor_t div_f32
            = utils::logical_tensor_init(9, dst_shape, impl::data_type::f32);
    impl::logical_tensor_t add_src1 = utils::logical_tensor_init(
            10, post_add_shape, impl::data_type::f32);
    impl::logical_tensor_t add_f32
            = utils::logical_tensor_init(11, dst_shape, impl::data_type::f32);

    dqdata_op.add_input(src_u8);
    dqdata_op.add_output(src_f32_dq);

    dqweight_op.add_input(weight_u8);
    dqweight_op.add_output(weight_f32_dq);

    matmul_op.add_input(src_f32_dq);
    matmul_op.add_input(weight_f32_dq);
    matmul_op.add_output(dst_f32);

    binary_op.add_input(dst_f32);
    binary_op.add_input(div_src1);
    binary_op.add_output(div_f32);

    binary_op2.add_input(div_f32);
    binary_op2.add_input(add_src1);
    binary_op2.add_output(add_f32);

    impl::graph_t g(engine.kind());
    g.add_op(&dqdata_op);
    g.add_op(&dqweight_op);
    g.add_op(&matmul_op);
    g.add_op(&binary_op);
    g.add_op(&binary_op2);
    g.build_graph();

    impl::pass::pass_base_ptr apass
            = get_pass("int8_matmul_div_add_fusion_cpu");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    // compile
    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> lt_ins {
            &src_u8, &weight_u8, &div_src1, &add_src1};
    std::vector<const impl::logical_tensor_t *> lt_outs {&add_f32};

    p.compile(&cp, lt_ins, lt_outs, &engine);

    impl::tensor_t src_u8_ts(src_u8, &engine, src_data.data());
    impl::tensor_t weight_u8_ts(weight_u8, &engine, weight_data.data());
    impl::tensor_t div_src1_ts(div_src1, &engine, div_src1_data.data());
    impl::tensor_t add_src1_ts(add_src1, &engine, add_src1_data.data());
    test::vector<float> case2_out_data(product(dst_shape));
    impl::tensor_t dst_f32_case2_ts(add_f32, &engine, case2_out_data.data());
    cp.execute(&strm, {src_u8_ts, weight_u8_ts, div_src1_ts, add_src1_ts},
            {dst_f32_case2_ts});
    strm.wait();
}

TEST(ExecuteSubgraphInt8, BmmX8x8bf16) {
    impl::engine_t &engine = get_engine();
    impl::stream_t &strm = get_stream();

    std::vector<std::string> dtypes = {"uint8", "int8"};

    std::vector<int64_t> src_shape = {1, 4, 16, 8};
    std::vector<int64_t> weight_shape = {1, 4, 8, 16};
    std::vector<int64_t> dst_shape = {1, 4, 16, 16};

    test::vector<uint8_t> src_data(product(src_shape));
    test::vector<uint8_t> weight_data(product(weight_shape));

    for (auto &src_dtype : dtypes) {
        for (auto &weight_dtype : dtypes) {
            auto src_lt_dtype = (src_dtype == "uint8") ? impl::data_type::u8
                                                       : impl::data_type::s8;
            auto weight_lt_dtype = (weight_dtype == "uint8")
                    ? impl::data_type::u8
                    : impl::data_type::s8;
            float src_range = (src_dtype == "uint8") ? 255.f : 127.f;
            float weight_range = (weight_dtype == "uint8") ? 255.f : 127.f;

            // u8 2 s8 shift by using reorder with -128 zps is not supported on
            // GPU
            if (weight_dtype == "uint8"
                    && engine.kind() == impl::engine_kind::gpu)
                continue;

            // random generate src, weight data
            // random seed = 7
            std::default_random_engine generator(7);
            std::uniform_real_distribution<float> src_distribution(
                    0.0f, src_range);
            std::uniform_real_distribution<float> weight_distribution(
                    0.0f, weight_range);
            std::generate(src_data.begin(), src_data.end(),
                    [&]() { return src_distribution(generator); });
            std::generate(weight_data.begin(), weight_data.end(),
                    [&]() { return weight_distribution(generator); });
            float scale_src = 1 / src_range;
            float scale_wei = 1 / weight_range;
            int64_t zp_src = 110;
            int64_t zp_wei = 114;

            impl::op_t dqdata_op(0, impl::op_kind::Dequantize, "dqdata_op");
            dqdata_op.set_attr<std::string>(impl::op_attr::qtype, "per_tensor");
            dqdata_op.set_attr<std::vector<int64_t>>(
                    impl::op_attr::zps, {zp_src});
            dqdata_op.set_attr<std::vector<float>>(
                    impl::op_attr::scales, {scale_src});
            dqdata_op.set_attr<int64_t>(impl::op_attr::axis, 0);

            impl::op_t dqweight_op(1, impl::op_kind::Dequantize, "dqweight_op");
            dqweight_op.set_attr<std::string>(
                    impl::op_attr::qtype, "per_tensor");
            dqweight_op.set_attr<std::vector<int64_t>>(
                    impl::op_attr::zps, {zp_wei});
            dqweight_op.set_attr<std::vector<float>>(
                    impl::op_attr::scales, {scale_wei});
            dqweight_op.set_attr<int64_t>(impl::op_attr::axis, 1);

            impl::op_t tcdata_op {2, impl::op_kind::TypeCast, "typecast_data"};
            impl::op_t tcweight_op {
                    3, impl::op_kind::TypeCast, "typecast_weight"};

            impl::op_t matmul_op(4, impl::op_kind::MatMul, "matmul_op");
            matmul_op.set_attr<bool>(impl::op_attr::transpose_a, false);
            matmul_op.set_attr<bool>(impl::op_attr::transpose_b, false);

            // prepare logical tensor
            impl::logical_tensor_t src
                    = utils::logical_tensor_init(0, src_shape, src_lt_dtype);
            impl::logical_tensor_t src_f32_dq = utils::logical_tensor_init(
                    1, src_shape, impl::data_type::f32);
            impl::logical_tensor_t src_bf16 = utils::logical_tensor_init(
                    2, src_shape, impl::data_type::bf16);
            impl::logical_tensor_t weight = utils::logical_tensor_init(
                    3, weight_shape, weight_lt_dtype);
            impl::logical_tensor_t weight_f32_dq = utils::logical_tensor_init(
                    4, weight_shape, impl::data_type::f32);
            impl::logical_tensor_t weight_bf16 = utils::logical_tensor_init(
                    5, weight_shape, impl::data_type::bf16);
            impl::logical_tensor_t dst_bf16 = utils::logical_tensor_init(
                    6, dst_shape, impl::data_type::bf16);

            dqdata_op.add_input(src);
            dqdata_op.add_output(src_f32_dq);

            dqweight_op.add_input(weight);
            dqweight_op.add_output(weight_f32_dq);

            tcdata_op.add_input(src_f32_dq);
            tcdata_op.add_output(src_bf16);

            tcweight_op.add_input(weight_f32_dq);
            tcweight_op.add_output(weight_bf16);

            matmul_op.add_input(src_bf16);
            matmul_op.add_input(weight_bf16);
            matmul_op.add_output(dst_bf16);

            impl::graph_t g(engine.kind());
            g.add_op(&dqdata_op);
            g.add_op(&dqweight_op);
            g.add_op(&matmul_op);
            g.add_op(&tcdata_op);
            g.add_op(&tcweight_op);
            g.build_graph();

            impl::pass::pass_base_ptr apass
                    = get_pass(engine.kind() == impl::engine_kind::gpu
                                    ? "int8_bf16_matmul_post_ops_fusion_gpu"
                                    : "int8_bf16_matmul_post_ops_fusion_cpu");
            apass->run(g);
            ASSERT_EQ(g.get_num_partitions(), 1);
            auto part = g.get_partitions()[0];

            // compile
            impl::partition_t p;
            p.init(part);

            impl::compiled_partition_t cp(p);

            std::vector<const impl::logical_tensor_t *> lt_ins {&src, &weight};
            std::vector<const impl::logical_tensor_t *> lt_outs {&dst_bf16};

            p.compile(&cp, lt_ins, lt_outs, &engine);

            test::vector<float> div_src1_data(1);
            test::vector<float> dst_data(product(dst_shape));
            impl::tensor_t src_ts(src, &engine, src_data.data());
            impl::tensor_t weight_ts(weight, &engine, weight_data.data());
            impl::tensor_t dst_ts(dst_bf16, &engine, dst_data.data());
            cp.execute(&strm, {src_ts, weight_ts}, {dst_ts});
            strm.wait();
        }
    }
}

TEST(ExecuteSubgraphInt8, BmmDivX8x8bf16) {
    impl::engine_t &engine = get_engine();
    impl::stream_t &strm = get_stream();

    std::vector<std::string> dtypes = {"uint8", "int8"};

    std::vector<int64_t> src_shape = {1, 4, 16, 8};
    std::vector<int64_t> weight_shape = {1, 4, 8, 16};
    std::vector<int64_t> dst_shape = {1, 4, 16, 16};

    test::vector<uint8_t> src_data(product(src_shape));
    test::vector<uint8_t> weight_data(product(weight_shape));

    for (auto &src_dtype : dtypes) {
        for (auto &weight_dtype : dtypes) {
            auto src_lt_dtype = (src_dtype == "uint8") ? impl::data_type::u8
                                                       : impl::data_type::s8;
            auto weight_lt_dtype = (weight_dtype == "uint8")
                    ? impl::data_type::u8
                    : impl::data_type::s8;
            float src_range = (src_dtype == "uint8") ? 255.f : 127.f;
            float weight_range = (weight_dtype == "uint8") ? 255.f : 127.f;

            // u8 2 s8 shift by using reorder with -128 zps is not supported on
            // GPU
            if (weight_dtype == "uint8"
                    && engine.kind() == impl::engine_kind::gpu)
                continue;

            // random generate src, weight data
            // random seed = 7
            std::default_random_engine generator(7);
            std::uniform_real_distribution<float> src_distribution(
                    0.0f, src_range);
            std::uniform_real_distribution<float> weight_distribution(
                    0.0f, weight_range);
            std::generate(src_data.begin(), src_data.end(),
                    [&]() { return src_distribution(generator); });
            std::generate(weight_data.begin(), weight_data.end(),
                    [&]() { return weight_distribution(generator); });
            float scale_src = 1 / src_range;
            float scale_wei = 1 / weight_range;
            int64_t zp_src = 110;
            int64_t zp_wei = 114;

            impl::op_t dqdata_op(0, impl::op_kind::Dequantize, "dqdata_op");
            dqdata_op.set_attr<std::string>(impl::op_attr::qtype, "per_tensor");
            dqdata_op.set_attr<std::vector<int64_t>>(
                    impl::op_attr::zps, {zp_src});
            dqdata_op.set_attr<std::vector<float>>(
                    impl::op_attr::scales, {scale_src});
            dqdata_op.set_attr<int64_t>(impl::op_attr::axis, 0);

            impl::op_t dqweight_op(1, impl::op_kind::Dequantize, "dqweight_op");
            dqweight_op.set_attr<std::string>(
                    impl::op_attr::qtype, "per_tensor");
            dqweight_op.set_attr<std::vector<int64_t>>(
                    impl::op_attr::zps, {zp_wei});
            dqweight_op.set_attr<std::vector<float>>(
                    impl::op_attr::scales, {scale_wei});
            dqweight_op.set_attr<int64_t>(impl::op_attr::axis, 1);

            impl::op_t tcdata_op {2, impl::op_kind::TypeCast, "typecast_data"};
            impl::op_t tcweight_op {
                    3, impl::op_kind::TypeCast, "typecast_weight"};

            impl::op_t matmul_op(4, impl::op_kind::MatMul, "matmul_op");
            matmul_op.set_attr<bool>(impl::op_attr::transpose_a, false);
            matmul_op.set_attr<bool>(impl::op_attr::transpose_b, false);

            impl::op_t binary_op(5, impl::op_kind::Divide, "binary_div");
            binary_op.set_attr<std::string>(
                    impl::op_attr::auto_broadcast, "numpy");

            // prepare logical tensor
            impl::logical_tensor_t src
                    = utils::logical_tensor_init(0, src_shape, src_lt_dtype);
            impl::logical_tensor_t src_f32_dq = utils::logical_tensor_init(
                    1, src_shape, impl::data_type::f32);
            impl::logical_tensor_t src_bf16 = utils::logical_tensor_init(
                    2, src_shape, impl::data_type::bf16);
            impl::logical_tensor_t weight = utils::logical_tensor_init(
                    3, weight_shape, weight_lt_dtype);
            impl::logical_tensor_t weight_f32_dq = utils::logical_tensor_init(
                    4, weight_shape, impl::data_type::f32);
            impl::logical_tensor_t weight_bf16 = utils::logical_tensor_init(
                    5, weight_shape, impl::data_type::bf16);
            impl::logical_tensor_t dst_bf16 = utils::logical_tensor_init(
                    6, dst_shape, impl::data_type::bf16);
            impl::logical_tensor_t div_src1
                    = utils::logical_tensor_init(7, {1}, impl::data_type::bf16);
            impl::logical_tensor_t div_bf16 = utils::logical_tensor_init(
                    8, dst_shape, impl::data_type::bf16);

            dqdata_op.add_input(src);
            dqdata_op.add_output(src_f32_dq);

            dqweight_op.add_input(weight);
            dqweight_op.add_output(weight_f32_dq);

            tcdata_op.add_input(src_f32_dq);
            tcdata_op.add_output(src_bf16);

            tcweight_op.add_input(weight_f32_dq);
            tcweight_op.add_output(weight_bf16);

            matmul_op.add_input(src_bf16);
            matmul_op.add_input(weight_bf16);
            matmul_op.add_output(dst_bf16);

            binary_op.add_input(dst_bf16);
            binary_op.add_input(div_src1);
            binary_op.add_output(div_bf16);

            impl::graph_t g(engine.kind());
            ASSERT_EQ(g.add_op(&dqdata_op), impl::status::success);
            ASSERT_EQ(g.add_op(&dqweight_op), impl::status::success);
            ASSERT_EQ(g.add_op(&matmul_op), impl::status::success);
            ASSERT_EQ(g.add_op(&tcdata_op), impl::status::success);
            ASSERT_EQ(g.add_op(&tcweight_op), impl::status::success);
            ASSERT_EQ(g.add_op(&binary_op), impl::status::success);
            ASSERT_EQ(g.build_graph(), impl::status::success);

            impl::pass::pass_base_ptr apass
                    = get_pass(engine.kind() == impl::engine_kind::gpu
                                    ? "int8_bf16_matmul_post_ops_fusion_gpu"
                                    : "int8_bf16_matmul_post_ops_fusion_cpu");
            apass->run(g);
            ASSERT_EQ(g.get_num_partitions(), 1);
            auto part = g.get_partitions()[0];

            // compile
            impl::partition_t p;
            p.init(part);

            impl::compiled_partition_t cp(p);

            std::vector<const impl::logical_tensor_t *> lt_ins {
                    &src, &weight, &div_src1};
            std::vector<const impl::logical_tensor_t *> lt_outs {&div_bf16};

            p.compile(&cp, lt_ins, lt_outs, &engine);

            test::vector<float> div_src1_data(1);
            test::vector<float> dst_data(product(dst_shape));
            impl::tensor_t src_ts(src, &engine, src_data.data());
            impl::tensor_t weight_ts(weight, &engine, weight_data.data());
            impl::tensor_t div_src1_ts(div_src1, &engine, div_src1_data.data());
            impl::tensor_t dst_ts(div_bf16, &engine, dst_data.data());
            cp.execute(&strm, {src_ts, weight_ts, div_src1_ts}, {dst_ts});
            strm.wait();
        }
    }
}

TEST(ExecuteSubgraphInt8, BmmDivBlockedX8x8bf16) {
    impl::engine_t &engine = get_engine();
    impl::stream_t &strm = get_stream();

    // The below case will be skipped on GPU:
    // benchdnn --matmul --reset --allow-enum-tags-only=0 --engine=gpu
    // --cfg=u8s8bf16  --stag=acbd --wtag=adbc --dtag=abcd
    // --attr-post-ops=binary_div:bf16:common --attr-oscale=common:3.087849e-05
    // --attr-zero-points=src:common:110+wei:common:114 --attr-scratchpad=user
    // 1x4x16x8:1x4x8x16:1x4x16x16
    SKIP_IF(engine.kind() == impl::engine_kind::gpu, "skip on gpu");

    std::vector<std::string> dtypes = {"uint8", "int8"};

    std::vector<int64_t> src_shape = {1, 4, 16, 8};
    std::vector<int64_t> src_stride = {512, 8, 32, 1};
    std::vector<int64_t> weight_shape = {1, 4, 8, 16};
    std::vector<int64_t> weight_stride = {512, 8, 1, 32};
    std::vector<int64_t> dst_shape = {1, 4, 16, 16};

    test::vector<uint8_t> src_data(product(src_shape));
    test::vector<uint8_t> weight_data(product(weight_shape));

    for (auto &src_dtype : dtypes) {
        for (auto &weight_dtype : dtypes) {
            auto src_lt_dtype = (src_dtype == "uint8") ? impl::data_type::u8
                                                       : impl::data_type::s8;
            auto weight_lt_dtype = (weight_dtype == "uint8")
                    ? impl::data_type::u8
                    : impl::data_type::s8;
            float src_range = (src_dtype == "uint8") ? 255.f : 127.f;
            float weight_range = (weight_dtype == "uint8") ? 255.f : 127.f;

            // u8 2 s8 shift by using reorder with -128 zps is not supported on
            // GPU
            if (weight_dtype == "uint8"
                    && engine.kind() == impl::engine_kind::gpu)
                continue;

            // random generate src, weight data
            // random seed = 7
            std::default_random_engine generator(7);
            std::uniform_real_distribution<float> src_distribution(
                    0.0f, src_range);
            std::uniform_real_distribution<float> weight_distribution(
                    0.0f, weight_range);
            std::generate(src_data.begin(), src_data.end(),
                    [&]() { return src_distribution(generator); });
            std::generate(weight_data.begin(), weight_data.end(),
                    [&]() { return weight_distribution(generator); });
            float scale_src = 1 / src_range;
            float scale_wei = 1 / weight_range;
            int64_t zp_src = 110;
            int64_t zp_wei = 114;

            impl::op_t dqdata_op(0, impl::op_kind::Dequantize, "dqdata_op");
            dqdata_op.set_attr<std::string>(impl::op_attr::qtype, "per_tensor");
            dqdata_op.set_attr<std::vector<int64_t>>(
                    impl::op_attr::zps, {zp_src});
            dqdata_op.set_attr<std::vector<float>>(
                    impl::op_attr::scales, {scale_src});
            dqdata_op.set_attr<int64_t>(impl::op_attr::axis, 0);

            impl::op_t dqweight_op(1, impl::op_kind::Dequantize, "dqweight_op");
            dqweight_op.set_attr<std::string>(
                    impl::op_attr::qtype, "per_tensor");
            dqweight_op.set_attr<std::vector<int64_t>>(
                    impl::op_attr::zps, {zp_wei});
            dqweight_op.set_attr<std::vector<float>>(
                    impl::op_attr::scales, {scale_wei});
            dqweight_op.set_attr<int64_t>(impl::op_attr::axis, 1);

            impl::op_t tcdata_op {2, impl::op_kind::TypeCast, "typecast_data"};
            impl::op_t tcweight_op {
                    3, impl::op_kind::TypeCast, "typecast_weight"};

            impl::op_t matmul_op(4, impl::op_kind::MatMul, "matmul_op");
            matmul_op.set_attr<bool>(impl::op_attr::transpose_a, false);
            matmul_op.set_attr<bool>(impl::op_attr::transpose_b, false);

            impl::op_t binary_op(5, impl::op_kind::Divide, "binary_div");
            binary_op.set_attr<std::string>(
                    impl::op_attr::auto_broadcast, "numpy");

            // prepare logical tensor
            impl::logical_tensor_t src = utils::logical_tensor_init(
                    0, src_shape, src_stride, src_lt_dtype);
            impl::logical_tensor_t src_f32_dq = utils::logical_tensor_init(
                    1, src_shape, src_stride, impl::data_type::f32);
            impl::logical_tensor_t src_bf16 = utils::logical_tensor_init(
                    2, src_shape, src_stride, impl::data_type::bf16);
            impl::logical_tensor_t weight = utils::logical_tensor_init(
                    3, weight_shape, weight_stride, weight_lt_dtype);
            impl::logical_tensor_t weight_f32_dq = utils::logical_tensor_init(
                    4, weight_shape, weight_stride, impl::data_type::f32);
            impl::logical_tensor_t weight_bf16 = utils::logical_tensor_init(
                    5, weight_shape, weight_stride, impl::data_type::bf16);
            impl::logical_tensor_t dst_bf16 = utils::logical_tensor_init(
                    6, dst_shape, impl::data_type::bf16);
            impl::logical_tensor_t div_src1
                    = utils::logical_tensor_init(7, {1}, impl::data_type::bf16);
            impl::logical_tensor_t div_bf16 = utils::logical_tensor_init(
                    8, dst_shape, impl::data_type::bf16);

            dqdata_op.add_input(src);
            dqdata_op.add_output(src_f32_dq);

            dqweight_op.add_input(weight);
            dqweight_op.add_output(weight_f32_dq);

            tcdata_op.add_input(src_f32_dq);
            tcdata_op.add_output(src_bf16);

            tcweight_op.add_input(weight_f32_dq);
            tcweight_op.add_output(weight_bf16);

            matmul_op.add_input(src_bf16);
            matmul_op.add_input(weight_bf16);
            matmul_op.add_output(dst_bf16);

            binary_op.add_input(dst_bf16);
            binary_op.add_input(div_src1);
            binary_op.add_output(div_bf16);

            impl::graph_t g(engine.kind());
            g.add_op(&dqdata_op);
            g.add_op(&dqweight_op);
            g.add_op(&matmul_op);
            g.add_op(&tcdata_op);
            g.add_op(&tcweight_op);
            g.add_op(&binary_op);
            g.build_graph();

            impl::pass::pass_base_ptr apass
                    = get_pass("int8_bf16_matmul_post_ops_fusion_cpu");
            apass->run(g);
            ASSERT_EQ(g.get_num_partitions(), 1);
            auto part = g.get_partitions()[0];

            // compile
            impl::partition_t p;
            p.init(part);

            impl::compiled_partition_t cp(p);

            std::vector<const impl::logical_tensor_t *> lt_ins {
                    &src, &weight, &div_src1};
            std::vector<const impl::logical_tensor_t *> lt_outs {&div_bf16};

            p.compile(&cp, lt_ins, lt_outs, &engine);

            test::vector<float> div_src1_data(1);
            test::vector<float> dst_data(product(dst_shape));
            impl::tensor_t src_ts(src, &engine, src_data.data());
            impl::tensor_t weight_ts(weight, &engine, weight_data.data());
            impl::tensor_t div_src1_ts(div_src1, &engine, div_src1_data.data());
            impl::tensor_t dst_ts(div_bf16, &engine, dst_data.data());
            cp.execute(&strm, {src_ts, weight_ts, div_src1_ts}, {dst_ts});
            strm.wait();
        }
    }
}

TEST(ExecuteSubgraphInt8, BmmDivAddX8x8bf16) {
    impl::engine_t &engine = get_engine();
    impl::stream_t &strm = get_stream();

    static auto isa = dnnl_get_effective_cpu_isa();
    SKIP_IF(isa < dnnl_cpu_isa_avx512_core
                    && engine.kind() == impl::engine_kind::cpu,
            "Skip bf16 examples for systems that do not support avx512_core.");

    std::vector<std::string> dtypes = {"uint8", "int8"};

    std::vector<int64_t> src_shape = {8, 4, 16, 8};
    std::vector<int64_t> weight_shape = {8, 4, 8, 16};
    std::vector<int64_t> post_div_shape = {1};
    std::vector<int64_t> post_add_shape = {8, 1, 1, 16};
    std::vector<int64_t> dst_shape = {8, 4, 16, 16};

    test::vector<uint8_t> src_data(product(src_shape));
    test::vector<uint8_t> weight_data(product(weight_shape));

    for (auto &src_dtype : dtypes) {
        for (auto &weight_dtype : dtypes) {
            auto src_lt_dtype = (src_dtype == "uint8") ? impl::data_type::u8
                                                       : impl::data_type::s8;
            auto weight_lt_dtype = (weight_dtype == "uint8")
                    ? impl::data_type::u8
                    : impl::data_type::s8;
            float src_range = (src_dtype == "uint8") ? 255.f : 127.f;
            float weight_range = (weight_dtype == "uint8") ? 255.f : 127.f;

            // u8 2 s8 shift by using reorder with -128 zps is not supported on
            // GPU
            if (weight_dtype == "uint8"
                    && engine.kind() == impl::engine_kind::gpu)
                continue;

            // random generate src, weight data
            // random seed = 7
            std::default_random_engine generator(7);
            std::uniform_real_distribution<float> src_distribution(
                    0.0f, src_range);
            std::uniform_real_distribution<float> weight_distribution(
                    0.0f, weight_range);
            std::generate(src_data.begin(), src_data.end(),
                    [&]() { return src_distribution(generator); });
            std::generate(weight_data.begin(), weight_data.end(),
                    [&]() { return weight_distribution(generator); });
            float scale_src = 1 / src_range;
            float scale_wei = 1 / weight_range;
            int64_t zp_src = 110;
            int64_t zp_wei = 114;

            impl::op_t dqdata_op(0, impl::op_kind::Dequantize, "dqdata_op");
            dqdata_op.set_attr<std::string>(impl::op_attr::qtype, "per_tensor");
            dqdata_op.set_attr<std::vector<int64_t>>(
                    impl::op_attr::zps, {zp_src});
            dqdata_op.set_attr<std::vector<float>>(
                    impl::op_attr::scales, {scale_src});
            dqdata_op.set_attr<int64_t>(impl::op_attr::axis, 0);

            impl::op_t dqweight_op(1, impl::op_kind::Dequantize, "dqweight_op");
            dqweight_op.set_attr<std::string>(
                    impl::op_attr::qtype, "per_tensor");
            dqweight_op.set_attr<std::vector<int64_t>>(
                    impl::op_attr::zps, {zp_wei});
            dqweight_op.set_attr<std::vector<float>>(
                    impl::op_attr::scales, {scale_wei});
            dqweight_op.set_attr<int64_t>(impl::op_attr::axis, 1);

            impl::op_t tcdata_op {2, impl::op_kind::TypeCast, "typecast_data"};
            impl::op_t tcweight_op {
                    3, impl::op_kind::TypeCast, "typecast_weight"};

            impl::op_t matmul_op(4, impl::op_kind::MatMul, "matmul_op");
            matmul_op.set_attr<bool>(impl::op_attr::transpose_a, false);
            matmul_op.set_attr<bool>(impl::op_attr::transpose_b, false);

            impl::op_t binary_op(5, impl::op_kind::Divide, "binary_div");
            binary_op.set_attr<std::string>(
                    impl::op_attr::auto_broadcast, "numpy");

            impl::op_t binary_add_op(6, impl::op_kind::Add, "binary_add");
            binary_add_op.set_attr<std::string>(
                    impl::op_attr::auto_broadcast, "numpy");

            // prepare logical tensor
            impl::logical_tensor_t src
                    = utils::logical_tensor_init(0, src_shape, src_lt_dtype);
            impl::logical_tensor_t src_f32_dq = utils::logical_tensor_init(
                    1, src_shape, impl::data_type::f32);
            impl::logical_tensor_t src_bf16 = utils::logical_tensor_init(
                    2, src_shape, impl::data_type::bf16);
            impl::logical_tensor_t weight = utils::logical_tensor_init(
                    3, weight_shape, weight_lt_dtype);
            impl::logical_tensor_t weight_f32_dq = utils::logical_tensor_init(
                    4, weight_shape, impl::data_type::f32);
            impl::logical_tensor_t weight_bf16 = utils::logical_tensor_init(
                    5, weight_shape, impl::data_type::bf16);
            impl::logical_tensor_t dst_bf16 = utils::logical_tensor_init(
                    6, dst_shape, impl::data_type::bf16);
            impl::logical_tensor_t div_src1 = utils::logical_tensor_init(
                    7, post_div_shape, impl::data_type::bf16);
            impl::logical_tensor_t div_bf16 = utils::logical_tensor_init(
                    8, dst_shape, impl::data_type::bf16);
            impl::logical_tensor_t add_src1 = utils::logical_tensor_init(
                    9, post_add_shape, impl::data_type::bf16);
            impl::logical_tensor_t add_bf16 = utils::logical_tensor_init(
                    10, dst_shape, impl::data_type::bf16);

            dqdata_op.add_input(src);
            dqdata_op.add_output(src_f32_dq);

            dqweight_op.add_input(weight);
            dqweight_op.add_output(weight_f32_dq);

            tcdata_op.add_input(src_f32_dq);
            tcdata_op.add_output(src_bf16);

            tcweight_op.add_input(weight_f32_dq);
            tcweight_op.add_output(weight_bf16);

            matmul_op.add_input(src_bf16);
            matmul_op.add_input(weight_bf16);
            matmul_op.add_output(dst_bf16);

            binary_op.add_input(dst_bf16);
            binary_op.add_input(div_src1);
            binary_op.add_output(div_bf16);

            binary_add_op.add_input(div_bf16);
            binary_add_op.add_input(add_src1);
            binary_add_op.add_output(add_bf16);

            impl::graph_t g(engine.kind());
            g.add_op(&dqdata_op);
            g.add_op(&dqweight_op);
            g.add_op(&matmul_op);
            g.add_op(&tcdata_op);
            g.add_op(&tcweight_op);
            g.add_op(&binary_op);
            g.add_op(&binary_add_op);
            g.build_graph();

            impl::pass::pass_base_ptr apass
                    = get_pass(engine.kind() == impl::engine_kind::gpu
                                    ? "int8_bf16_matmul_div_add_fusion_gpu"
                                    : "int8_bf16_matmul_div_add_fusion_cpu");
            apass->run(g);
            ASSERT_EQ(g.get_num_partitions(), 1);
            auto part = g.get_partitions()[0];

            // compile
            impl::partition_t p;
            p.init(part);

            impl::compiled_partition_t cp(p);

            std::vector<const impl::logical_tensor_t *> lt_ins {
                    &src, &weight, &div_src1, &add_src1};
            std::vector<const impl::logical_tensor_t *> lt_outs {&add_bf16};

            p.compile(&cp, lt_ins, lt_outs, &engine);

            test::vector<float> div_src1_data(product(post_div_shape));
            test::vector<float> add_src1_data(product(post_add_shape));
            test::vector<float> dst_data(product(dst_shape));
            impl::tensor_t src_ts(src, &engine, src_data.data());
            impl::tensor_t weight_ts(weight, &engine, weight_data.data());
            impl::tensor_t div_src1_ts(div_src1, &engine, div_src1_data.data());
            impl::tensor_t add_src1_ts(add_src1, &engine, add_src1_data.data());
            impl::tensor_t dst_ts(add_bf16, &engine, dst_data.data());
            cp.execute(&strm, {src_ts, weight_ts, div_src1_ts, add_src1_ts},
                    {dst_ts});
            strm.wait();
        }
    }
}

TEST(ExecuteSubgraphInt8, U8u8bf16DivBmm) {
    impl::engine_t &engine = get_engine();
    impl::stream_t &strm = get_stream();

    std::vector<int64_t> src_shape = {1, 12, 128, 64};
    std::vector<int64_t> weight_shape = {1, 12, 64, 128};
    std::vector<int64_t> dst_shape = {1, 12, 128, 128};

    test::vector<uint8_t> src_data(product(src_shape));
    test::vector<uint8_t> weight_data(product(weight_shape));

    // u8 2 s8 shift by using reorder with -128 zps is not supported on
    // GPU
    SKIP_IF(engine.kind() == impl::engine_kind::gpu, "Skip on GPU device.");

    // random generate src, weight data
    // random seed = 7
    std::default_random_engine generator(7);
    std::uniform_real_distribution<float> src_distribution(0.0f, 255.f);
    std::uniform_real_distribution<float> weight_distribution(0.0f, 255.f);
    std::generate(src_data.begin(), src_data.end(),
            [&]() { return src_distribution(generator); });
    std::generate(weight_data.begin(), weight_data.end(),
            [&]() { return weight_distribution(generator); });
    float scale_src = 1 / 255.f;
    float scale_wei = 1 / 255.f;
    int64_t zp_src = 110;
    int64_t zp_wei = 114;

    impl::op_t dqdata_op(0, impl::op_kind::Dequantize, "dqdata_op");
    dqdata_op.set_attr<std::string>(impl::op_attr::qtype, "per_tensor");
    dqdata_op.set_attr<std::vector<int64_t>>(impl::op_attr::zps, {zp_src});
    dqdata_op.set_attr<std::vector<float>>(impl::op_attr::scales, {scale_src});
    dqdata_op.set_attr<int64_t>(impl::op_attr::axis, 0);

    impl::op_t dqweight_op(1, impl::op_kind::Dequantize, "dqweight_op");
    dqweight_op.set_attr<std::string>(impl::op_attr::qtype, "per_tensor");
    dqweight_op.set_attr<std::vector<int64_t>>(impl::op_attr::zps, {zp_wei});
    dqweight_op.set_attr<std::vector<float>>(
            impl::op_attr::scales, {scale_wei});
    dqweight_op.set_attr<int64_t>(impl::op_attr::axis, 1);

    impl::op_t tcdata_op {2, impl::op_kind::TypeCast, "typecast_data"};
    impl::op_t tcweight_op {3, impl::op_kind::TypeCast, "typecast_weight"};

    impl::op_t matmul_op(4, impl::op_kind::MatMul, "matmul_op");
    matmul_op.set_attr<bool>(impl::op_attr::transpose_a, false);
    matmul_op.set_attr<bool>(impl::op_attr::transpose_b, false);

    impl::op_t binary_op(5, impl::op_kind::Divide, "binary_div");
    binary_op.set_attr<std::string>(impl::op_attr::auto_broadcast, "numpy");

    // prepare logical tensor
    impl::logical_tensor_t src
            = utils::logical_tensor_init(0, src_shape, impl::data_type::u8);
    impl::logical_tensor_t src_f32_dq
            = utils::logical_tensor_init(1, src_shape, impl::data_type::f32);
    impl::logical_tensor_t src_bf16
            = utils::logical_tensor_init(2, src_shape, impl::data_type::bf16);
    impl::logical_tensor_t weight
            = utils::logical_tensor_init(3, weight_shape, impl::data_type::u8);
    impl::logical_tensor_t weight_f32_dq
            = utils::logical_tensor_init(4, weight_shape, impl::data_type::f32);
    impl::logical_tensor_t weight_bf16 = utils::logical_tensor_init(
            5, weight_shape, impl::data_type::bf16);
    impl::logical_tensor_t dst_bf16
            = utils::logical_tensor_init(6, dst_shape, impl::data_type::bf16);
    impl::logical_tensor_t div_src1
            = utils::logical_tensor_init(7, {1}, impl::data_type::bf16);
    impl::logical_tensor_t div_bf16
            = utils::logical_tensor_init(8, src_shape, impl::data_type::bf16);

    dqdata_op.add_input(src);
    dqdata_op.add_output(src_f32_dq);

    dqweight_op.add_input(weight);
    dqweight_op.add_output(weight_f32_dq);

    tcdata_op.add_input(src_f32_dq);
    tcdata_op.add_output(src_bf16);

    tcweight_op.add_input(weight_f32_dq);
    tcweight_op.add_output(weight_bf16);

    binary_op.add_input(src_bf16);
    binary_op.add_input(div_src1);
    binary_op.add_output(div_bf16);

    matmul_op.add_input(div_bf16);
    matmul_op.add_input(weight_bf16);
    matmul_op.add_output(dst_bf16);

    impl::graph_t g(engine.kind());
    ASSERT_EQ(g.add_op(&dqdata_op), impl::status::success);
    ASSERT_EQ(g.add_op(&dqweight_op), impl::status::success);
    ASSERT_EQ(g.add_op(&matmul_op), impl::status::success);
    ASSERT_EQ(g.add_op(&tcdata_op), impl::status::success);
    ASSERT_EQ(g.add_op(&tcweight_op), impl::status::success);
    ASSERT_EQ(g.add_op(&binary_op), impl::status::success);
    ASSERT_EQ(g.build_graph(), impl::status::success);

    impl::pass::pass_base_ptr apass
            = get_pass("x8x8bf16_div_matmul_fusion_cpu");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    // compile
    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> lt_ins {
            &src, &weight, &div_src1};
    std::vector<const impl::logical_tensor_t *> lt_outs {&dst_bf16};

    ASSERT_EQ(p.compile(&cp, lt_ins, lt_outs, &engine), impl::status::success);

    test::vector<float> div_src1_data(1);
    test::vector<float> dst_data(product(dst_shape));
    impl::tensor_t src_ts(src, &engine, src_data.data());
    impl::tensor_t weight_ts(weight, &engine, weight_data.data());
    impl::tensor_t div_src1_ts(div_src1, &engine, div_src1_data.data());
    impl::tensor_t dst_ts(dst_bf16, &engine, dst_data.data());
    ASSERT_EQ(cp.execute(&strm, {src_ts, weight_ts, div_src1_ts}, {dst_ts}),
            impl::status::success);
    strm.wait();
}
