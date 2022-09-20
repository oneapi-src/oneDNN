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

#include <functional>
#include <random>

#include "interface/c_types_map.hpp"

#include "gtest/gtest.h"

#include "graph/unit/backend/dnnl/dnnl_test_common.hpp"
#include "graph/unit/unit_test_common.hpp"
#include "graph/unit/utils.hpp"

namespace graph = dnnl::impl::graph;
namespace utils = dnnl::graph::tests::unit::utils;

TEST(Compile, TestReduce) {
    const auto apply_keep_dims_attr = [](const std::vector<int64_t> &shape,
                                              const std::vector<int64_t> &axes,
                                              const bool keep_dims) {
        if (keep_dims) return shape;
        std::vector<size_t> excluded_axes;
        excluded_axes.reserve(axes.size());
        for (const auto axis : axes) {
            excluded_axes.push_back(static_cast<size_t>(
                    (axis < 0) ? shape.size() + axis : axis));
        }
        std::vector<int64_t> new_shape;
        for (size_t i = 0; i < shape.size(); ++i) {
            const auto excluded
                    = std::find(excluded_axes.begin(), excluded_axes.end(), i)
                    != excluded_axes.end();
            if (!excluded) new_shape.push_back(shape[i]);
        }
        return new_shape;
    };

    graph::engine_t *engine = get_engine();
    graph::stream_t *strm = get_stream();

    std::vector<int64_t> reduce_src_shape {2, 2, 2, 2};
    std::vector<int64_t> reduce_dst_shape {2, 1, 2, 1};
    std::vector<bool> keep_dims_vals {true, false};
    std::vector<int64_t> axes {-3, 3};

    test::vector<float> src_data(product(reduce_src_shape));
    std::default_random_engine generator(7);
    std::uniform_real_distribution<float> f32_distribution(0.0f, 1.0f);
    std::generate(src_data.begin(), src_data.end(),
            [&]() { return f32_distribution(generator); });

    const std::vector<graph::op_kind_t> op_infos {graph::op_kind::ReduceL1,
            graph::op_kind::ReduceL2, graph::op_kind::ReduceMax,
            graph::op_kind::ReduceMean, graph::op_kind::ReduceMin,
            graph::op_kind::ReduceProd, graph::op_kind::ReduceSum};

    for_(auto &op_kind : op_infos)
    for (auto keep_dims : keep_dims_vals) {
        auto new_reduce_dst_shape
                = apply_keep_dims_attr(reduce_dst_shape, axes, keep_dims);

        graph::op_t reduce {0, op_kind, "reduce"};
        reduce.set_attr<std::vector<int64_t>>(graph::op_attr::axes, axes);
        reduce.set_attr<bool>(graph::op_attr::keep_dims, keep_dims);

        graph::logical_tensor_t reduce_src = utils::logical_tensor_init(
                0, reduce_src_shape, graph::data_type::f32);
        graph::logical_tensor_t reduce_dst = utils::logical_tensor_init(
                1, new_reduce_dst_shape, graph::data_type::f32);

        reduce.add_input(reduce_src);
        reduce.add_output(reduce_dst);

        graph::graph_t g(engine->kind());
        g.add_op(&reduce);
        g.finalize();

        graph::pass::pass_base_ptr apass = get_pass("reduce_pass");
        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 1U);
        auto part = g.get_partitions()[0];

        graph::partition_t p;
        p.init(part);

        graph::compiled_partition_t cp(p);

        std::vector<const graph::logical_tensor_t *> lt_ins {&reduce_src};
        std::vector<const graph::logical_tensor_t *> lt_outs {&reduce_dst};

        p.compile(&cp, lt_ins, lt_outs, engine);

        test::vector<float> case1_out_data(product(new_reduce_dst_shape));
        test::vector<float> case2_out_data(product(new_reduce_dst_shape));
        graph::tensor_t reduce_src_ts(reduce_src, engine, src_data.data());
        graph::tensor_t reduce_dst_ts1(
                reduce_dst, engine, case1_out_data.data());
        graph::tensor_t reduce_dst_ts2(
                reduce_dst, engine, case2_out_data.data());

        ASSERT_EQ(
                run_graph(g, {reduce_src_ts}, {reduce_dst_ts1}, *engine, *strm),
                graph::status::success);

        cp.execute(strm, {reduce_src_ts}, {reduce_dst_ts2});
        strm->wait();

        for (size_t i = 0; i < case1_out_data.size(); ++i) {
            ASSERT_FLOAT_EQ(case1_out_data[i], case2_out_data[i]);
        }
    }
}

TEST(ExecuteSubgraphFp32, ReduceAdd) {
    graph::engine_t *engine = get_engine();
    graph::stream_t *strm = get_stream();

    std::vector<int64_t> reduce_src_shape {2, 2, 2, 2};
    std::vector<int64_t> base_reduce_dst_shape {1, 2, 2, 1};
    std::vector<int64_t> base_add_src1_shape {1, 2, 1, 1};
    std::vector<int64_t> axes {0, 3};

    test::vector<float> src_data(product(reduce_src_shape));

    std::default_random_engine generator(7);
    std::uniform_real_distribution<float> f32_distribution(0.0f, 1.0f);
    std::generate(src_data.begin(), src_data.end(),
            [&]() { return f32_distribution(generator); });

    std::vector<bool> keep_dims_infos {true, false};
    std::vector<bool> with_sum_infos {true, false};

    const std::vector<graph::op_kind_t> op_infos {graph::op_kind::ReduceL1,
            graph::op_kind::ReduceL2, graph::op_kind::ReduceMax,
            graph::op_kind::ReduceMean, graph::op_kind::ReduceMin,
            graph::op_kind::ReduceProd, graph::op_kind::ReduceSum};

    for_(bool keep_dims : keep_dims_infos)
    for_(bool with_sum : with_sum_infos)
    for (auto &akind : op_infos) {
        std::vector<int64_t> reduce_dst_shape = base_reduce_dst_shape;
        std::vector<int64_t> add_src1_shape = base_add_src1_shape;
        if (with_sum) { add_src1_shape[2] *= 2; }
        if (!keep_dims) {
            reduce_dst_shape.erase(reduce_dst_shape.begin());
            reduce_dst_shape.erase(reduce_dst_shape.end() - 1);
            add_src1_shape.erase(add_src1_shape.begin());
            add_src1_shape.erase(add_src1_shape.end() - 1);
        }

        test::vector<float> src1_data(product(add_src1_shape));
        std::generate(src1_data.begin(), src1_data.end(),
                [&]() { return f32_distribution(generator); });

        graph::op_t reduce {0, akind, "reduce"};
        reduce.set_attr(graph::op_attr::keep_dims, keep_dims);
        reduce.set_attr(graph::op_attr::axes, axes);

        graph::op_t add {1, graph::op_kind::Add, "add"};

        graph::logical_tensor_t reduce_src = utils::logical_tensor_init(
                0, reduce_src_shape, graph::data_type::f32);
        graph::logical_tensor_t reduce_dst = utils::logical_tensor_init(
                1, reduce_dst_shape, graph::data_type::f32);

        graph::logical_tensor_t add_src1 = utils::logical_tensor_init(
                2, add_src1_shape, graph::data_type::f32);
        graph::logical_tensor_t add_dst = utils::logical_tensor_init(
                3, reduce_dst_shape, graph::data_type::f32);

        reduce.add_input(reduce_src);
        reduce.add_output(reduce_dst);

        add.add_input(reduce_dst);
        add.add_input(add_src1);
        add.add_output(add_dst);

        graph::graph_t g(engine->kind());
        g.add_op(&reduce);
        g.add_op(&add);
        g.finalize();

        graph::tensor_t reduce_src_ts(reduce_src, engine, src_data.data());
        graph::tensor_t add_src1_ts(add_src1, engine, src1_data.data());

        // -------------------------case 1----------------------------------
        test::vector<float> case1_out_data(product(reduce_dst_shape));
        graph::tensor_t add_dst_ts(add_dst, engine, case1_out_data.data());

        ASSERT_EQ(run_graph(g, {reduce_src_ts, add_src1_ts}, {add_dst_ts},
                          *engine, *strm),
                graph::status::success);

        // -------------------------case 2----------------------------------
        graph::pass::pass_base_ptr apass
                = get_pass("reduction_post_ops_fusion");
        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 1U);
        auto part = g.get_partitions()[0];

        graph::partition_t p;
        p.init(part);

        graph::compiled_partition_t cp(p);

        std::vector<const graph::logical_tensor_t *> lt_ins {
                &reduce_src, &add_src1};
        std::vector<const graph::logical_tensor_t *> lt_outs {&add_dst};

        p.compile(&cp, lt_ins, lt_outs, engine);

        test::vector<float> case2_out_data(product(reduce_dst_shape));
        graph::tensor_t add_dst_ts2(add_dst, engine, case2_out_data.data());

        cp.execute(strm, {reduce_src_ts, add_src1_ts}, {add_dst_ts2});
        strm->wait();

        for (size_t i = 0; i < case1_out_data.size(); ++i) {
            ASSERT_FLOAT_EQ(case1_out_data[i], case2_out_data[i]);
        }
    }
}

TEST(ExecuteSubgraphFp32, ReduceRelu) {
    graph::engine_t *engine = get_engine();
    graph::stream_t *strm = get_stream();

    std::vector<int64_t> reduce_src_shape {2, 2, 2, 2};
    std::vector<int64_t> base_reduce_dst_shape {1, 2, 2, 1};
    std::vector<int64_t> axes {0, 3};

    test::vector<float> src_data(product(reduce_src_shape));

    std::default_random_engine generator(7);
    std::uniform_real_distribution<float> f32_distribution(0.0f, 1.0f);
    std::generate(src_data.begin(), src_data.end(),
            [&]() { return f32_distribution(generator); });

    std::vector<bool> keep_dims_infos {true, false};
    const std::vector<graph::op_kind_t> op_infos {graph::op_kind::ReduceL1,
            graph::op_kind::ReduceL2, graph::op_kind::ReduceMax,
            graph::op_kind::ReduceMean, graph::op_kind::ReduceMin,
            graph::op_kind::ReduceProd, graph::op_kind::ReduceSum};

    for_(bool keep_dims : keep_dims_infos)
    for (auto &akind : op_infos) {
        std::vector<int64_t> reduce_dst_shape = base_reduce_dst_shape;
        if (!keep_dims) {
            reduce_dst_shape.erase(reduce_dst_shape.begin());
            reduce_dst_shape.erase(reduce_dst_shape.end() - 1);
        }

        graph::op_t reduce {0, akind, "reduce"};
        reduce.set_attr(graph::op_attr::keep_dims, keep_dims);
        reduce.set_attr(graph::op_attr::axes, axes);

        graph::op_t relu {1, graph::op_kind::ReLU, "relu"};

        graph::logical_tensor_t reduce_src = utils::logical_tensor_init(
                0, reduce_src_shape, graph::data_type::f32);
        graph::logical_tensor_t reduce_dst = utils::logical_tensor_init(
                1, reduce_dst_shape, graph::data_type::f32);

        graph::logical_tensor_t relu_dst = utils::logical_tensor_init(
                3, reduce_dst_shape, graph::data_type::f32);

        reduce.add_input(reduce_src);
        reduce.add_output(reduce_dst);

        relu.add_input(reduce_dst);
        relu.add_output(relu_dst);

        graph::graph_t g(engine->kind());
        g.add_op(&reduce);
        g.add_op(&relu);
        g.finalize();

        graph::tensor_t reduce_src_ts(reduce_src, engine, src_data.data());

        // -------------------------case 1----------------------------------
        test::vector<float> case1_out_data(product(reduce_dst_shape));
        graph::tensor_t relu_dst_ts(relu_dst, engine, case1_out_data.data());

        ASSERT_EQ(run_graph(g, {reduce_src_ts}, {relu_dst_ts}, *engine, *strm),
                graph::status::success);

        // -------------------------case 2----------------------------------
        graph::pass::pass_base_ptr apass
                = get_pass("reduction_post_ops_fusion");
        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 1U);
        auto part = g.get_partitions()[0];

        graph::partition_t p;
        p.init(part);

        graph::compiled_partition_t cp(p);

        std::vector<const graph::logical_tensor_t *> lt_ins {&reduce_src};
        std::vector<const graph::logical_tensor_t *> lt_outs {&relu_dst};

        p.compile(&cp, lt_ins, lt_outs, engine);

        test::vector<float> case2_out_data(product(reduce_dst_shape));
        graph::tensor_t relu_dst_ts2(relu_dst, engine, case2_out_data.data());

        cp.execute(strm, {reduce_src_ts}, {relu_dst_ts2});
        strm->wait();

        for (size_t i = 0; i < case1_out_data.size(); ++i) {
            ASSERT_FLOAT_EQ(case1_out_data[i], case2_out_data[i]);
        }
    }
}

TEST(ExecuteSubgraphFp32, ReduceSwish) {
    graph::engine_t *engine = get_engine();
    graph::stream_t *strm = get_stream();

    std::vector<int64_t> reduce_src_shape {2, 2, 2, 2};
    std::vector<int64_t> base_reduce_dst_shape {1, 2, 2, 1};
    std::vector<int64_t> axes {0, 3};

    test::vector<float> src_data(product(reduce_src_shape));

    std::default_random_engine generator(7);
    std::uniform_real_distribution<float> f32_distribution(0.0f, 1.0f);
    std::generate(src_data.begin(), src_data.end(),
            [&]() { return f32_distribution(generator); });

    std::vector<bool> keep_dims_infos {true, false};
    const std::vector<graph::op_kind_t> op_infos {graph::op_kind::ReduceL1,
            graph::op_kind::ReduceL2, graph::op_kind::ReduceMax,
            graph::op_kind::ReduceMean, graph::op_kind::ReduceMin,
            graph::op_kind::ReduceProd, graph::op_kind::ReduceSum};

    for_(bool keep_dims : keep_dims_infos)
    for (auto &akind : op_infos) {
        std::vector<int64_t> reduce_dst_shape = base_reduce_dst_shape;
        if (!keep_dims) {
            reduce_dst_shape.erase(reduce_dst_shape.begin());
            reduce_dst_shape.erase(reduce_dst_shape.end() - 1);
        }

        graph::op_t reduce {0, akind, "reduce"};
        reduce.set_attr(graph::op_attr::keep_dims, keep_dims);
        reduce.set_attr(graph::op_attr::axes, axes);

        graph::op_t sigmoid {1, graph::op_kind::Sigmoid, "sigmoid"};
        graph::op_t multiply {2, graph::op_kind::Multiply, "multiply"};

        graph::logical_tensor_t reduce_src = utils::logical_tensor_init(
                0, reduce_src_shape, graph::data_type::f32);
        graph::logical_tensor_t reduce_dst = utils::logical_tensor_init(
                1, reduce_dst_shape, graph::data_type::f32);
        graph::logical_tensor_t sigmoid_dst = utils::logical_tensor_init(
                2, reduce_dst_shape, graph::data_type::f32);
        graph::logical_tensor_t mul_dst = utils::logical_tensor_init(
                3, reduce_dst_shape, graph::data_type::f32);

        reduce.add_input(reduce_src);
        reduce.add_output(reduce_dst);
        sigmoid.add_input(reduce_dst);
        sigmoid.add_output(sigmoid_dst);
        multiply.add_input(reduce_dst);
        multiply.add_input(sigmoid_dst);
        multiply.add_output(mul_dst);

        graph::graph_t g(engine->kind());
        g.add_op(&reduce);
        g.add_op(&sigmoid);
        g.add_op(&multiply);
        g.finalize();

        graph::tensor_t reduce_src_ts(reduce_src, engine, src_data.data());

        // -------------------------case 1----------------------------------
        test::vector<float> case1_out_data(product(reduce_dst_shape));
        graph::tensor_t mul_dst_ts(mul_dst, engine, case1_out_data.data());

        ASSERT_EQ(run_graph(g, {reduce_src_ts}, {mul_dst_ts}, *engine, *strm),
                graph::status::success);

        // -------------------------case 2----------------------------------
        graph::pass::pass_base_ptr apass
                = get_pass("reduction_post_ops_fusion");
        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 1U);
        auto part = g.get_partitions()[0];

        graph::partition_t p;
        p.init(part);

        graph::compiled_partition_t cp(p);

        std::vector<const graph::logical_tensor_t *> lt_ins {&reduce_src};
        std::vector<const graph::logical_tensor_t *> lt_outs {&mul_dst};

        p.compile(&cp, lt_ins, lt_outs, engine);

        test::vector<float> case2_out_data(product(reduce_dst_shape));
        graph::tensor_t mul_dst_ts2(mul_dst, engine, case2_out_data.data());

        cp.execute(strm, {reduce_src_ts}, {mul_dst_ts2});
        strm->wait();

        for (size_t i = 0; i < case1_out_data.size(); ++i) {
            ASSERT_FLOAT_EQ(case1_out_data[i], case2_out_data[i]);
        }
    }
}

TEST(ExecuteSubgraphFp32, ReduceWith3PostOps) {
    graph::engine_t *engine = get_engine();
    graph::stream_t *strm = get_stream();

    // some cases with Reduce are MISTRUSTED on GPU:
    // ./benchdnn --reduction --engine=gpu --sdt=bf16 --ddt=bf16 --dtag=axb
    // --alg=min --attr-post-ops=relu 64x20x7x7:64x20x1x1
    SKIP_IF(engine->kind() == graph::engine_kind::gpu, "skip on gpu");

    std::vector<int64_t> reduce_src_shape {2, 2, 2, 2};
    std::vector<int64_t> reduce_dst_shape {1, 2, 2, 1};
    std::vector<int64_t> axes {0, 3};

    test::vector<float> src_data(product(reduce_src_shape));
    test::vector<float> max_src_data(product(reduce_dst_shape));
    test::vector<float> mul_src_data(product(reduce_dst_shape));

    std::default_random_engine generator(7);
    std::uniform_real_distribution<float> f32_distribution(0.0f, 1.0f);
    std::generate(src_data.begin(), src_data.end(),
            [&]() { return f32_distribution(generator); });
    std::generate(max_src_data.begin(), max_src_data.end(),
            [&]() { return f32_distribution(generator); });
    std::generate(mul_src_data.begin(), mul_src_data.end(),
            [&]() { return f32_distribution(generator); });

    const std::vector<graph::op_kind_t> op_infos {graph::op_kind::ReduceL1,
            graph::op_kind::ReduceL2, graph::op_kind::ReduceMax,
            graph::op_kind::ReduceMean, graph::op_kind::ReduceMin,
            graph::op_kind::ReduceProd, graph::op_kind::ReduceSum};

    for (auto &akind : op_infos) {
        graph::op_t reduce {0, akind, "reduce"};
        reduce.set_attr(graph::op_attr::keep_dims, true);
        reduce.set_attr(graph::op_attr::axes, axes);

        graph::op_t sigmoid {1, graph::op_kind::Sigmoid, "sigmoid"};
        graph::op_t maximum {2, graph::op_kind::Maximum, "maximum"};
        graph::op_t multiply {3, graph::op_kind::Multiply, "multiply"};

        graph::logical_tensor_t reduce_src = utils::logical_tensor_init(
                0, reduce_src_shape, graph::data_type::f32);
        graph::logical_tensor_t reduce_dst = utils::logical_tensor_init(
                1, reduce_dst_shape, graph::data_type::f32);
        graph::logical_tensor_t sigmoid_dst = utils::logical_tensor_init(
                2, reduce_dst_shape, graph::data_type::f32);
        graph::logical_tensor_t max_src = utils::logical_tensor_init(
                3, reduce_dst_shape, graph::data_type::f32);
        graph::logical_tensor_t max_dst = utils::logical_tensor_init(
                4, reduce_dst_shape, graph::data_type::f32);
        graph::logical_tensor_t mul_src = utils::logical_tensor_init(
                5, reduce_dst_shape, graph::data_type::f32);
        graph::logical_tensor_t mul_dst = utils::logical_tensor_init(
                6, reduce_dst_shape, graph::data_type::f32);

        reduce.add_input(reduce_src);
        reduce.add_output(reduce_dst);
        sigmoid.add_input(reduce_dst);
        sigmoid.add_output(sigmoid_dst);

        maximum.add_input(max_src);
        maximum.add_input(sigmoid_dst);
        maximum.add_output(max_dst);
        multiply.add_input(max_dst);
        multiply.add_input(mul_src);
        multiply.add_output(mul_dst);

        graph::graph_t g(engine->kind());
        g.add_op(&reduce);
        g.add_op(&sigmoid);
        g.add_op(&maximum);
        g.add_op(&multiply);
        g.finalize();

        graph::tensor_t reduce_src_ts(reduce_src, engine, src_data.data());
        graph::tensor_t max_src_ts(max_src, engine, max_src_data.data());
        graph::tensor_t mul_src_ts(mul_src, engine, mul_src_data.data());

        // -------------------------case 1----------------------------------
        test::vector<float> case1_out_data(product(reduce_dst_shape));
        graph::tensor_t mul_dst_ts(mul_dst, engine, case1_out_data.data());

        ASSERT_EQ(run_graph(g, {reduce_src_ts, max_src_ts, mul_src_ts},
                          {mul_dst_ts}, *engine, *strm),
                graph::status::success);

        // -------------------------case 2----------------------------------
        graph::pass::pass_base_ptr apass
                = get_pass("reduction_post_ops_fusion");
        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 1U);
        auto part = g.get_partitions()[0];

        graph::partition_t p;
        p.init(part);

        graph::compiled_partition_t cp(p);

        std::vector<const graph::logical_tensor_t *> lt_ins {
                &reduce_src, &max_src, &mul_src};
        std::vector<const graph::logical_tensor_t *> lt_outs {&mul_dst};

        p.compile(&cp, lt_ins, lt_outs, engine);

        test::vector<float> case2_out_data(product(reduce_dst_shape));
        graph::tensor_t mul_dst_ts2(mul_dst, engine, case2_out_data.data());

        cp.execute(
                strm, {reduce_src_ts, max_src_ts, mul_src_ts}, {mul_dst_ts2});
        strm->wait();

        for (size_t i = 0; i < case1_out_data.size(); ++i) {
            ASSERT_FLOAT_EQ(case1_out_data[i], case2_out_data[i]);
        }
    }
}

TEST(Execute, ReduceMeanOutputDims) {
    const auto apply_keep_dims_attr
            = [](const std::vector<int64_t> &shape, const bool keep_dims) {
                  if (keep_dims) return shape;
                  std::vector<int64_t> new_shape;
                  return new_shape;
              };
    graph::engine_t *engine = get_engine();
    graph::stream_t *strm = get_stream();

    test::vector<float> src0_data {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f};
    test::vector<float> ref_dst_data {4.f};
    test::vector<float> dst_data(ref_dst_data.size(), 0.0);
    std::vector<int64_t> src0_shape {7};
    std::vector<int64_t> dst_shape {1};

    std::vector<bool> keep_dims_vals {true, false};
    for (auto keep_dims : keep_dims_vals) {
        graph::op_t reducemean_op(graph::op_kind::ReduceMean);
        reducemean_op.set_attr<std::vector<int64_t>>(graph::op_attr::axes, {0});
        reducemean_op.set_attr<bool>(graph::op_attr::keep_dims, keep_dims);

        // prepare logical tensor
        graph::logical_tensor_t src0 = utils::logical_tensor_init(
                0, src0_shape, graph::data_type::f32);
        auto new_dst_shape = apply_keep_dims_attr(dst_shape, keep_dims);
        graph::logical_tensor_t dst = utils::logical_tensor_init(
                1, new_dst_shape, graph::data_type::f32);
        reducemean_op.add_input(src0);
        reducemean_op.add_output(dst);
        graph::graph_t g(engine->kind());
        g.add_op(&reducemean_op);
        g.finalize();
        graph::pass::pass_base_ptr apass = get_pass("reduce_pass");
        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 1U);
        auto part = g.get_partitions()[0];
        // compile
        graph::partition_t p;
        p.init(part);
        graph::compiled_partition_t cp(p);
        std::vector<const graph::logical_tensor_t *> inputs {&src0};
        std::vector<const graph::logical_tensor_t *> outputs {&dst};
        ASSERT_EQ(p.compile(&cp, inputs, outputs, engine),
                graph::status::success);
        // if keep_dims attr is false and the intput a one dimension tensor,
        // the output should be a scalar (ndims=0, layout_type=strided).
        // if keep_dims attr is true, the output's ndims is equal to
        // input's ndims.
        graph::logical_tensor_t dst_lt;
        cp.query_logical_tensor(dst.id, &dst_lt);
        ASSERT_EQ(dst_lt.layout_type, graph::layout_type::strided);
        ASSERT_EQ(dst_lt.ndims, static_cast<int>(new_dst_shape.size()));
        graph::tensor_t src0_ts(src0, engine, src0_data.data());
        graph::tensor_t dst_ts(dst_lt, engine, dst_data.data());

        ASSERT_EQ(
                cp.execute(strm, {src0_ts}, {dst_ts}), graph::status::success);
        strm->wait();
        for (size_t i = 0; i < ref_dst_data.size(); ++i) {
            ASSERT_FLOAT_EQ(dst_data[i], ref_dst_data[i]);
        }
    }
}
