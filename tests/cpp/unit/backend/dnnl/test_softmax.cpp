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

#include <random>
#include "gtest/gtest.h"

#include "cpp/unit/backend/dnnl/dnnl_test_common.hpp"
#include "cpp/unit/backend/dnnl/ref_func.hpp"
#include "cpp/unit/unit_test_common.hpp"
#include "cpp/unit/utils.hpp"

namespace impl = dnnl::graph::impl;
namespace utils = dnnl::graph::tests::unit::utils;

TEST(Execute, Softmax) {
    impl::engine_t &eng = get_engine();

    impl::op_t softmax_op(impl::op_kind::SoftMax);
    softmax_op.set_attr<int64_t>(impl::op_attr::axis, 0);

    test::vector<float> src_data {0.0, 1.0, 2.0, 0.0, 1.0, 2.0};
    test::vector<float> ref_dst_data {0.5, 0.5, 0.5, 0.5, 0.5, 0.5};
    test::vector<float> dst_data(ref_dst_data.size(), 0.0);

    // prepare logical tensor
    impl::logical_tensor_t src
            = utils::logical_tensor_init(0, {2, 3}, impl::data_type::f32);
    impl::logical_tensor_t dst = utils::logical_tensor_init(
            1, {2, 3}, impl::data_type::f32, impl::layout_type::any);

    softmax_op.add_input(src);
    softmax_op.add_output(dst);

    impl::graph_t g(eng.kind());
    g.add_op(&softmax_op);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("softmax_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];
    impl::partition_t p;
    p.init(part);

    // compile
    std::vector<const impl::logical_tensor_t *> inputs {&src};
    std::vector<const impl::logical_tensor_t *> outputs {&dst};

    impl::compiled_partition_t cp(p);
    ASSERT_EQ(p.compile(&cp, inputs, outputs, &eng), impl::status::success);

    impl::logical_tensor_t lt;
    cp.query_logical_tensor(dst.id, &lt);
    ASSERT_EQ(lt.layout_type, impl::layout_type::strided);

    impl::tensor_t src_ts(src, &eng, src_data.data());
    impl::tensor_t dst_ts(lt, &eng, dst_data.data());

    impl::stream_t &strm = get_stream();
    ASSERT_EQ(cp.execute(&strm, {src_ts}, {dst_ts}), impl::status::success);
    strm.wait();
    for (size_t i = 0; i < ref_dst_data.size(); ++i) {
        ASSERT_FLOAT_EQ(dst_data[i], ref_dst_data[i]);
    }
}

TEST(Execute, SoftmaxWithLastDim) {
    impl::engine_t &eng = get_engine();

    impl::op_t softmax_op(impl::op_kind::SoftMax);
    softmax_op.set_attr<int64_t>(impl::op_attr::axis, -1);

    test::vector<float> src_data {3.0, 3.0, 1.0, 1.0};
    test::vector<float> ref_dst_data {0.5, 0.5, 0.5, 0.5};
    test::vector<float> dst_data(ref_dst_data.size(), 0.0);

    // prepare logical tensor
    impl::logical_tensor_t src
            = utils::logical_tensor_init(0, {2, 2}, impl::data_type::f32);
    impl::logical_tensor_t dst = utils::logical_tensor_init(
            1, {2, 2}, impl::data_type::f32, impl::layout_type::any);

    softmax_op.add_input(src);
    softmax_op.add_output(dst);

    impl::graph_t g(eng.kind());
    g.add_op(&softmax_op);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("softmax_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];
    impl::partition_t p;
    p.init(part);

    // compile
    std::vector<const impl::logical_tensor_t *> inputs {&src};
    std::vector<const impl::logical_tensor_t *> outputs {&dst};

    impl::compiled_partition_t cp(p);
    ASSERT_EQ(p.compile(&cp, inputs, outputs, &eng), impl::status::success);

    impl::logical_tensor_t lt;
    cp.query_logical_tensor(dst.id, &lt);
    ASSERT_EQ(lt.layout_type, impl::layout_type::strided);

    impl::tensor_t src_ts(src, &eng, src_data.data());
    impl::tensor_t dst_ts(lt, &eng, dst_data.data());

    impl::stream_t &strm = get_stream();
    ASSERT_EQ(cp.execute(&strm, {src_ts}, {dst_ts}), impl::status::success);
    strm.wait();
    for (size_t i = 0; i < ref_dst_data.size(); ++i) {
        ASSERT_FLOAT_EQ(dst_data[i], ref_dst_data[i]);
    }
}

static inline void test_softmax_bwd_common(const impl::op_kind_t op_kind,
        test::vector<float> &dst, test::vector<float> &diff_dst,
        test::vector<float> &ref_diff_src, const impl::dims &dims,
        const bool plain_grad) {
    impl::op_t softmax_bwd_op(op_kind);
    impl::engine_t &eng = get_engine();

    softmax_bwd_op.set_attr<int64_t>(impl::op_attr::axis, 1);

    test::vector<float> diff_src(ref_diff_src.size(), 0.0);

    impl::logical_tensor_t diff_dst_lt;
    // prepare logical tensor
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(0, dims, impl::data_type::f32);
    if (plain_grad) {
        diff_dst_lt = utils::logical_tensor_init(1, dims, impl::data_type::f32);
    } else {
        diff_dst_lt = utils::logical_tensor_init(
                1, dims, impl::data_type::f32, impl::layout_type::any);
    }
    impl::logical_tensor_t diff_src_lt = utils::logical_tensor_init(
            2, impl::data_type::f32, impl::layout_type::any);

    softmax_bwd_op.add_input(diff_dst_lt);
    softmax_bwd_op.add_input(dst_lt);
    softmax_bwd_op.add_output(diff_src_lt);

    impl::graph_t g(eng.kind());
    g.add_op(&softmax_bwd_op);
    g.build_graph();

    impl::pass::pass_base_ptr apass = op_kind == impl::op_kind::SoftMaxBackprop
            ? get_pass("softmax_bwd_pass")
            : get_pass("logsoftmax_bwd_pass");

    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];
    impl::partition_t p;
    p.init(part);

    std::vector<const impl::logical_tensor_t *> inputs {&diff_dst_lt, &dst_lt};
    std::vector<const impl::logical_tensor_t *> outputs {&diff_src_lt};

    impl::compiled_partition_t cp(p);
    ASSERT_EQ(p.compile(&cp, inputs, outputs, &eng), impl::status::success);

    impl::logical_tensor_t q_lt;
    cp.query_logical_tensor(diff_src_lt.id, &q_lt);
    ASSERT_EQ(q_lt.layout_type, impl::layout_type::strided);

    for (auto i = 0; i < dst_lt.ndims; i++)
        ASSERT_EQ(q_lt.dims[i], dst_lt.dims[i]);

    impl::tensor_t dst_ts(dst_lt, &eng, dst.data());
    impl::tensor_t diff_dst_ts(diff_dst_lt, &eng, diff_dst.data());
    impl::tensor_t diff_src_ts(q_lt, &eng, diff_src.data());

    impl::stream_t &strm = get_stream();
    ASSERT_EQ(cp.execute(&strm, {diff_dst_ts, dst_ts}, {diff_src_ts}),
            impl::status::success);
    strm.wait();
    for (size_t i = 0; i < ref_diff_src.size(); ++i) {
        ASSERT_FLOAT_EQ(diff_src[i], ref_diff_src[i]);
    }
}

TEST(Execute, SoftMaxBackward) {
    test::vector<float> dst {0.5, 0.5, 0.5, 0.5};
    test::vector<float> diff_dst {1.0, 5.0, 2.0, 4.0};
    test::vector<float> ref_diff_src {-1, 1, -0.5, 0.5};

    const impl::dims dims {2, 2};

    const bool plain_grad = false;

    test_softmax_bwd_common(impl::op_kind::SoftMaxBackprop, dst, diff_dst,
            ref_diff_src, dims, plain_grad);
}

TEST(Execute, SoftMaxBackwardPlainGrad) {
    test::vector<float> dst {0.5, 0.5, 0.5, 0.5};
    test::vector<float> diff_dst {1.0, 5.0, 2.0, 4.0};
    test::vector<float> ref_diff_src {-1, 1, -0.5, 0.5};

    const impl::dims dims {2, 2};

    const bool plain_grad = true;

    test_softmax_bwd_common(impl::op_kind::SoftMaxBackprop, dst, diff_dst,
            ref_diff_src, dims, plain_grad);
}

TEST(Execute, LogSoftmax) {
    impl::engine_t &eng = get_engine();

    impl::op_t logsoftmax_op(impl::op_kind::LogSoftmax);
    logsoftmax_op.set_attr<int64_t>(impl::op_attr::axis, 0);

    test::vector<float> src_data {0.0, 1.0, 2.0, 0.0, 1.0, 2.0};
    test::vector<float> ref_dst_data {-0.6931472f, -0.6931472f, -0.6931472f,
            -0.6931472f, -0.6931472f, -0.6931472f};
    test::vector<float> dst_data(ref_dst_data.size(), 0.0);

    // prepare logical tensor
    impl::logical_tensor_t src
            = utils::logical_tensor_init(0, {2, 3}, impl::data_type::f32);
    impl::logical_tensor_t dst = utils::logical_tensor_init(
            1, {2, 3}, impl::data_type::f32, impl::layout_type::any);

    logsoftmax_op.add_input(src);
    logsoftmax_op.add_output(dst);

    impl::graph_t g(eng.kind());
    g.add_op(&logsoftmax_op);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("logsoftmax_pass");
    ASSERT_TRUE(apass);
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];
    impl::partition_t p;
    p.init(part);

    // compile
    std::vector<const impl::logical_tensor_t *> inputs {&src};
    std::vector<const impl::logical_tensor_t *> outputs {&dst};

    impl::compiled_partition_t cp(p);
    ASSERT_EQ(p.compile(&cp, inputs, outputs, &eng), impl::status::success);

    impl::logical_tensor_t lt;
    cp.query_logical_tensor(dst.id, &lt);
    ASSERT_EQ(lt.layout_type, impl::layout_type::strided);

    impl::tensor_t src_ts(src, &eng, src_data.data());
    impl::tensor_t dst_ts(lt, &eng, dst_data.data());

    impl::stream_t &strm = get_stream();
    ASSERT_EQ(cp.execute(&strm, {src_ts}, {dst_ts}), impl::status::success);
    strm.wait();
    for (size_t i = 0; i < ref_dst_data.size(); ++i) {
        ASSERT_FLOAT_EQ(dst_data[i], ref_dst_data[i]);
    }
}

TEST(Execute, LogSoftmaxBackward) {
    test::vector<float> dst {
            -0.6931472f, -0.6931472f, -0.6931472f, -0.6931472f};
    test::vector<float> diff_dst {1.0, 5.0, 2.0, 4.0};
    test::vector<float> ref_diff_src {-2, 2, -1, 1};

    const impl::dims dims {2, 2};

    const bool plain_grad = false;

    test_softmax_bwd_common(impl::op_kind::LogSoftmaxBackprop, dst, diff_dst,
            ref_diff_src, dims, plain_grad);
}

TEST(Execute, LogSoftmaxBackwardPlainGrad) {
    test::vector<float> dst {
            -0.6931472f, -0.6931472f, -0.6931472f, -0.6931472f};
    test::vector<float> diff_dst {1.0, 5.0, 2.0, 4.0};
    test::vector<float> ref_diff_src {-2, 2, -1, 1};

    const impl::dims dims {2, 2};

    const bool plain_grad = true;

    test_softmax_bwd_common(impl::op_kind::LogSoftmaxBackprop, dst, diff_dst,
            ref_diff_src, dims, plain_grad);
}

static inline void test_softmax_bwd_get_inplace_pair_common(
        impl::op_kind_t op_kind) {
    impl::engine_t &eng = get_engine();
    impl::op_t softmax_bwd_op(op_kind);

    softmax_bwd_op.set_attr<int64_t>(impl::op_attr::axis, 1);

    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(0, {2, 2}, impl::data_type::f32);
    impl::logical_tensor_t diff_dst_lt
            = utils::logical_tensor_init(1, {2, 2}, impl::data_type::f32);
    impl::logical_tensor_t diff_src_lt
            = utils::logical_tensor_init(2, {2, 2}, impl::data_type::f32);

    softmax_bwd_op.add_input(diff_dst_lt);
    softmax_bwd_op.add_input(dst_lt);
    softmax_bwd_op.add_output(diff_src_lt);

    impl::graph_t g(eng.kind());
    g.add_op(&softmax_bwd_op);
    g.build_graph();

    impl::pass::pass_base_ptr apass = op_kind == impl::op_kind::SoftMaxBackprop
            ? get_pass("softmax_bwd_pass")
            : get_pass("logsoftmax_bwd_pass");
    apass->run(g);

    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];
    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);

    // compile reorder_add partition
    std::vector<const impl::logical_tensor_t *> inputs {&diff_dst_lt, &dst_lt};
    std::vector<const impl::logical_tensor_t *> outputs {&diff_src_lt};
    ASSERT_EQ(p.compile(&cp, inputs, outputs, &eng), impl::status::success);

    std::vector<impl::inplace_pair_t> inplace_pairs = cp.get_inplace_pairs();
    ASSERT_EQ(inplace_pairs.size(), 1);
    ASSERT_EQ(inplace_pairs[0].input_id, diff_dst_lt.id);
    ASSERT_EQ(inplace_pairs[0].output_id, diff_src_lt.id);
}

TEST(ExecuteSubgraphInt8, SoftmaxTypecastQuant) {
    impl::engine_t &engine = get_engine();
    impl::stream_t &strm = get_stream();
    static auto isa = dnnl_get_effective_cpu_isa();
    SKIP_IF(isa < dnnl_cpu_isa_avx512_core
                    && engine.kind() == impl::engine_kind::cpu,
            "Skip bf16 test for systems that do not support avx512_core.");

    std::vector<int64_t> softmax_shape {2, 2, 2};
    test::vector<float> src_data(product(softmax_shape));

    // random seed = 7
    std::default_random_engine generator(7);
    std::uniform_real_distribution<float> src_distribution(0.f, 1.f);
    std::generate(src_data.begin(), src_data.end(),
            [&]() { return src_distribution(generator); });

    impl::op_t softmax_op(0, impl::op_kind::SoftMax, "softmax");
    softmax_op.set_attr<int64_t>(impl::op_attr::axis, 2);
    impl::op_t typecast(1, impl::op_kind::TypeCast, "typecast");
    impl::op_t quantize(2, impl::op_kind::Quantize, "quantize");
    quantize.set_attr<std::vector<float>>(impl::op_attr::scales, {0.1f});
    quantize.set_attr<std::vector<int64_t>>(impl::op_attr::zps, {0});
    quantize.set_attr<std::string>(impl::op_attr::qtype, "per_tensor");

    // prepare logical tensor
    impl::logical_tensor_t src = utils::logical_tensor_init(
            0, softmax_shape, impl::data_type::bf16);
    impl::logical_tensor_t softmax_dst = utils::logical_tensor_init(
            1, softmax_shape, impl::data_type::bf16);
    impl::logical_tensor_t tc_dst = utils::logical_tensor_init(
            2, softmax_shape, impl::data_type::f32);
    impl::logical_tensor_t quant_dst
            = utils::logical_tensor_init(3, softmax_shape, impl::data_type::u8);

    softmax_op.add_input(src);
    softmax_op.add_output(softmax_dst);
    typecast.add_input(softmax_dst);
    typecast.add_output(tc_dst);
    quantize.add_input(tc_dst);
    quantize.add_output(quant_dst);

    impl::graph_t g(engine.kind());
    ASSERT_EQ(g.add_op(&softmax_op), impl::status::success);
    ASSERT_EQ(g.add_op(&typecast), impl::status::success);
    ASSERT_EQ(g.add_op(&quantize), impl::status::success);
    ASSERT_EQ(g.build_graph(), impl::status::success);

    impl::pass::pass_base_ptr apass = get_pass("softmax_post_ops_fusion");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    // compile
    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> lt_ins {&src};
    std::vector<const impl::logical_tensor_t *> lt_outs {&quant_dst};

    ASSERT_EQ(p.compile(&cp, lt_ins, lt_outs, &engine), impl::status::success);

    test::vector<float> dst_data(product(softmax_shape));
    test::vector<float> ref_data(product(softmax_shape));
    impl::tensor_t src_ts(src, &engine, src_data.data());
    impl::tensor_t dst_ts(quant_dst, &engine, dst_data.data());
    impl::tensor_t ref_ts(quant_dst, &engine, ref_data.data());

    ASSERT_EQ(run_graph(g, {src_ts}, {ref_ts}, engine, strm),
            impl::status::success);
    ASSERT_EQ(cp.execute(&strm, {src_ts}, {dst_ts}), impl::status::success);
    strm.wait();

    for (size_t i = 0; i < ref_data.size(); ++i) {
        ASSERT_FLOAT_EQ(ref_data[i], dst_data[i]);
    }
}

TEST(Compile, SoftMaxBackwardGetInplacePair) {
    test_softmax_bwd_get_inplace_pair_common(impl::op_kind::SoftMaxBackprop);
}

TEST(Compile, LogSoftmaxBackwardGetInplacePair) {
    test_softmax_bwd_get_inplace_pair_common(impl::op_kind::LogSoftmaxBackprop);
}

TEST(Compile, SoftmaxGetInplacePair) {
    impl::engine_t &eng = get_engine();

    impl::op_t softmax_op(impl::op_kind::SoftMax);
    softmax_op.set_attr<int64_t>(impl::op_attr::axis, 0);

    // prepare logical tensor
    impl::logical_tensor_t src
            = utils::logical_tensor_init(0, {2, 3}, impl::data_type::f32);
    impl::logical_tensor_t dst
            = utils::logical_tensor_init(1, {2, 3}, impl::data_type::f32);

    softmax_op.add_input(src);
    softmax_op.add_output(dst);

    impl::graph_t g(eng.kind());
    g.add_op(&softmax_op);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("softmax_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];
    impl::partition_t p;
    p.init(part);

    // compile
    std::vector<const impl::logical_tensor_t *> inputs {&src};
    std::vector<const impl::logical_tensor_t *> outputs {&dst};

    impl::compiled_partition_t cp(p);
    ASSERT_EQ(p.compile(&cp, inputs, outputs, &eng), impl::status::success);

    auto pairs = cp.get_inplace_pairs();
    ASSERT_EQ(pairs.size(), 1);
    ASSERT_EQ(pairs[0].input_id, 0);
    ASSERT_EQ(pairs[0].output_id, 1);
}
