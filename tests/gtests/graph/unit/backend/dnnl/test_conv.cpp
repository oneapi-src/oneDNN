/*******************************************************************************
* Copyright 2020-2023 Intel Corporation
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
#include "graph/unit/backend/dnnl/ref_func.hpp"
#include "graph/unit/unit_test_common.hpp"
#include "graph/unit/utils.hpp"

namespace graph = dnnl::impl::graph;
namespace utils = dnnl::graph::tests::unit::utils;

struct eltwise_param_t {
    std::string pass_name;
    test::vector<float> bias;
    test::vector<float> ref_dst;
    graph::op_kind_t op_kind;
    std::string op_name;
    std::vector<std::pair<graph::op_attr_t, float>> attrs;
};

TEST(Compile, ConvolutionFp32) {
    using dims = graph::dnnl_impl::dims;

    graph::engine_t *engine = get_engine();

    graph::op_t conv_op(graph::op_kind::Convolution);
    conv_op.set_attr<dims>(graph::op_attr::strides, dims {1, 1});
    conv_op.set_attr<dims>(graph::op_attr::dilations, dims {1, 1});
    conv_op.set_attr<dims>(graph::op_attr::pads_begin, dims {0, 0});
    conv_op.set_attr<dims>(graph::op_attr::pads_end, dims {0, 0});
    conv_op.set_attr<int64_t>(graph::op_attr::groups, 1);
    conv_op.set_attr<std::string>(graph::op_attr::data_format, "NCX");
    conv_op.set_attr<std::string>(graph::op_attr::weights_format, "OIX");

    // prepare logical tensor
    graph::logical_tensor_t src = utils::logical_tensor_init(
            0, {8, 3, 224, 224}, graph::data_type::f32);
    graph::logical_tensor_t weight = utils::logical_tensor_init(
            1, {16, 3, 3, 3}, graph::data_type::f32);
    graph::logical_tensor_t dst = utils::logical_tensor_init(2,
            {8, 16, 222, 222}, graph::data_type::f32, graph::layout_type::any);

    conv_op.add_input(src);
    conv_op.add_input(weight);
    conv_op.add_output(dst);

    graph::graph_t g(engine->kind());
    g.add_op(&conv_op);
    g.finalize();

    graph::pass::pass_base_ptr apass = get_pass("conv_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile
    graph::partition_t p;
    p.init(part);

    graph::compiled_partition_t cp(p);

    std::vector<const graph::logical_tensor_t *> inputs {&src, &weight};
    std::vector<const graph::logical_tensor_t *> outputs {&dst};

    p.compile(&cp, inputs, outputs, engine);

    graph::logical_tensor_t lt;
    cp.query_logical_tensor(dst.id, &lt);
    ASSERT_EQ(lt.layout_type,
            engine->kind() == graph::engine_kind::gpu
                    ? graph::layout_type::opaque
                    : graph::layout_type::strided);
}

TEST(Compile, ConvolutionBackwardDataFp32) {
    using dims = dnnl::impl::graph::dnnl_impl::dims;

    graph::engine_t *eng = get_engine();

    graph::op_t conv_op(graph::op_kind::ConvolutionBackwardData);
    conv_op.set_attr<dims>(graph::op_attr::strides, dims {1, 1});
    conv_op.set_attr<dims>(graph::op_attr::dilations, dims {1, 1});
    conv_op.set_attr<dims>(graph::op_attr::pads_begin, dims {0, 0});
    conv_op.set_attr<dims>(graph::op_attr::pads_end, dims {0, 0});
    // according to spec, group should be greater than 0
    conv_op.set_attr<int64_t>(graph::op_attr::groups, 1);
    conv_op.set_attr<std::string>(graph::op_attr::data_format, "NCX");
    conv_op.set_attr<std::string>(graph::op_attr::weights_format, "OIX");
    conv_op.set_attr<dims>(graph::op_attr::dst_shape, dims {8, 3, 224, 224});

    // prepare logical tensor
    graph::logical_tensor_t diff_src = utils::logical_tensor_init(
            0, {8, 3, 224, 224}, graph::data_type::f32);
    graph::logical_tensor_t weights = utils::logical_tensor_init(
            1, {16, 3, 3, 3}, graph::data_type::f32);
    graph::logical_tensor_t diff_dst = utils::logical_tensor_init(
            2, {8, 16, 222, 222}, graph::data_type::f32);

    conv_op.add_input(diff_dst);
    conv_op.add_input(weights);
    conv_op.add_output(diff_src);

    graph::graph_t g(eng->kind());
    g.add_op(&conv_op);
    g.finalize();

    graph::pass::pass_base_ptr apass = get_pass("conv_data_bw_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile
    graph::partition_t p;
    p.init(part);
    graph::compiled_partition_t cp(p);

    std::vector<const graph::logical_tensor_t *> inputs {&diff_dst, &weights};
    std::vector<const graph::logical_tensor_t *> outputs {&diff_src};
    ASSERT_EQ(p.compile(&cp, inputs, outputs, eng), graph::status::success);

    graph::logical_tensor_t lt;
    cp.query_logical_tensor(diff_src.id, &lt);
    ASSERT_EQ(lt.layout_type, graph::layout_type::strided);
}

TEST(Compile, ConvolutionBackwardFilterFp32) {
    using dims = dnnl::impl::graph::dnnl_impl::dims;

    graph::engine_t *eng = get_engine();

    graph::op_t conv_op(graph::op_kind::ConvolutionBackwardWeights);
    conv_op.set_attr<dims>(graph::op_attr::strides, dims {1, 1});
    conv_op.set_attr<dims>(graph::op_attr::dilations, dims {1, 1});
    conv_op.set_attr<dims>(graph::op_attr::pads_begin, dims {0, 0});
    conv_op.set_attr<dims>(graph::op_attr::pads_end, dims {0, 0});
    conv_op.set_attr<std::string>(graph::op_attr::data_format, "NXC");
    conv_op.set_attr<std::string>(graph::op_attr::weights_format, "XIO");
    conv_op.set_attr<dims>(graph::op_attr::weights_shape, dims {3, 3, 64, 64});

    // prepare logical tensor
    graph::logical_tensor_t src = utils::logical_tensor_init(
            0, {1, 224, 224, 64}, graph::data_type::f32);
    graph::logical_tensor_t diff_dst = utils::logical_tensor_init(
            1, {1, 222, 222, 64}, graph::data_type::f32);
    graph::logical_tensor_t diff_weight = utils::logical_tensor_init(
            2, {3, 3, 64, 64}, graph::data_type::f32, graph::layout_type::any);

    conv_op.add_input(src);
    conv_op.add_input(diff_dst);
    conv_op.add_output(diff_weight);

    graph::graph_t g(eng->kind());
    g.add_op(&conv_op);
    g.finalize();

    graph::pass::pass_base_ptr apass = get_pass("conv_filter_bw_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile
    graph::partition_t p;
    p.init(part);
    graph::compiled_partition_t cp(p);

    std::vector<const graph::logical_tensor_t *> inputs {&src, &diff_dst};
    std::vector<const graph::logical_tensor_t *> outputs {&diff_weight};
    ASSERT_EQ(p.compile(&cp, inputs, outputs, eng), graph::status::success);
}

TEST(Compile, ConvolutionBackwardWeightsWithGroupsAndFiltersAnyLayout) {
    using dims = graph::dnnl_impl::dims;

    const dims src_dims {2, 4, 2};
    const dims diff_dst_dims {2, 4, 2};
    const dims diff_wei_dims {4, 2, 3};

    const dims strides {1};
    const dims pads_begin {1};
    const dims pads_end {1};
    const dims dilations {1};
    const int64_t groups {2};
    const std::string auto_pad {"None"};
    const std::string data_format {"NCX"};
    const std::string filter_format {"OIX"};

    graph::op_t conv_op(graph::op_kind::ConvolutionBackwardWeights);
    conv_op.set_attr(graph::op_attr::strides, strides)
            .set_attr(graph::op_attr::pads_begin, pads_begin)
            .set_attr(graph::op_attr::pads_end, pads_end)
            .set_attr(graph::op_attr::dilations, dilations)
            .set_attr(graph::op_attr::groups, groups)
            .set_attr(graph::op_attr::auto_pad, auto_pad)
            .set_attr(graph::op_attr::data_format, data_format)
            .set_attr(graph::op_attr::weights_format, filter_format);

    graph::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, src_dims, graph::data_type::f32);
    graph::logical_tensor_t diff_dst_lt = utils::logical_tensor_init(
            1, diff_dst_dims, graph::data_type::f32);
    graph::logical_tensor_t diff_wei_lt = utils::logical_tensor_init(
            2, diff_wei_dims, graph::data_type::f32, graph::layout_type::any);

    conv_op.add_input(src_lt);
    conv_op.add_input(diff_dst_lt);
    conv_op.add_output(diff_wei_lt);

    graph::engine_t *eng = get_engine();
    graph::graph_t g(eng->kind());
    g.add_op(&conv_op);
    g.finalize();

    graph::pass::pass_base_ptr apass = get_pass("conv_filter_bw_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    graph::partition_t p;
    p.init(part);

    graph::compiled_partition_t cp(p);

    std::vector<const graph::logical_tensor_t *> inputs {&src_lt, &diff_dst_lt};
    std::vector<const graph::logical_tensor_t *> outputs {&diff_wei_lt};

    ASSERT_EQ(p.compile(&cp, inputs, outputs, eng), graph::status::success);

    graph::logical_tensor_t lt;
    cp.query_logical_tensor(diff_wei_lt.id, &lt);
    // if layout queried from the primitive will make descriptor impossible
    // to reshape (with groups -> no groups), we make it strided (via reorder)
    ASSERT_TRUE(lt.layout_type == graph::layout_type::opaque
            || lt.layout_type == graph::layout_type::strided);
}

TEST(Execute, ConvolutionNcxOix) {
    using dims = graph::dnnl_impl::dims;

    // default engine kind is cpu.
    graph::engine_t *eng = get_engine();
    test::vector<float> src {-3.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0,
            2.5, -1.0, 0, 3.0, -2.0, -1.0, 4.0};
    test::vector<float> weight {1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0};
    test::vector<float> ref_dst {-1.0, 2.5, 5.0, 1.5};
    test::vector<float> dst {0.0, 0.0, 0.0, 0.0};
    graph::op_t conv_op(graph::op_kind::Convolution);
    conv_op.set_attr<dims>(graph::op_attr::strides, dims {1, 1});
    conv_op.set_attr<dims>(graph::op_attr::dilations, dims {1, 1});
    conv_op.set_attr<dims>(graph::op_attr::pads_begin, dims {0, 0});
    conv_op.set_attr<dims>(graph::op_attr::pads_end, dims {0, 0});
    conv_op.set_attr<int64_t>(graph::op_attr::groups, 1);
    conv_op.set_attr<std::string>(graph::op_attr::data_format, "NCX");
    conv_op.set_attr<std::string>(graph::op_attr::weights_format, "OIX");

    // prepare logical tensor
    graph::logical_tensor_t src_lt = utils::logical_tensor_init(
            0, {1, 1, 4, 4}, graph::data_type::f32);
    graph::logical_tensor_t weight_lt = utils::logical_tensor_init(
            1, {1, 1, 3, 3}, graph::data_type::f32);
    graph::logical_tensor_t dst_lt = utils::logical_tensor_init(
            2, {1, 1, 2, 2}, graph::data_type::f32);

    conv_op.add_input(src_lt);
    conv_op.add_input(weight_lt);
    conv_op.add_output(dst_lt);

    graph::graph_t g(eng->kind());
    g.add_op(&conv_op);
    g.finalize();

    graph::pass::pass_base_ptr apass = get_pass("conv_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile
    graph::partition_t p;
    p.init(part);

    graph::compiled_partition_t cp(p);

    std::vector<const graph::logical_tensor_t *> inputs {&src_lt, &weight_lt};
    std::vector<const graph::logical_tensor_t *> outputs {&dst_lt};

    p.compile(&cp, inputs, outputs, eng);

    graph::logical_tensor_t lt;
    cp.query_logical_tensor(dst_lt.id, &lt);
    ASSERT_EQ(lt.layout_type, graph::layout_type::strided);

    graph::tensor_t src_ts(src_lt, eng, src.data());
    graph::tensor_t weight_ts(weight_lt, eng, weight.data());
    graph::tensor_t dst_ts(dst_lt, eng, dst.data());

    graph::stream_t *strm = get_stream();
    cp.execute(strm, {src_ts, weight_ts}, {dst_ts});
    strm->wait();
    for (size_t i = 0; i < dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(Execute, ConvtransposeWithGroups) {
    using dims = graph::dnnl_impl::dims;

    // default engine kind is cpu.
    graph::engine_t *eng = get_engine();
    test::vector<float> src {1.0, 2.0, 3.0, 4.0};
    test::vector<float> weight {1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
            0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0};
    test::vector<float> ref_dst {3.0, 0.0, 3.0, 0.0, 7.0, 0.0, 7.0, 0.0};
    test::vector<float> dst(ref_dst.size(), 0);
    graph::op_t convtranspose_op(graph::op_kind::ConvTranspose);
    convtranspose_op.set_attr<dims>(graph::op_attr::strides, dims {1, 1});
    convtranspose_op.set_attr<dims>(graph::op_attr::dilations, dims {1, 1});
    convtranspose_op.set_attr<dims>(graph::op_attr::pads_begin, dims {0, 0});
    convtranspose_op.set_attr<dims>(graph::op_attr::pads_end, dims {0, 0});
    convtranspose_op.set_attr<int64_t>(graph::op_attr::groups, 2);
    convtranspose_op.set_attr<std::string>(graph::op_attr::data_format, "NCX");
    convtranspose_op.set_attr<std::string>(
            graph::op_attr::weights_format, "IOX");

    // prepare logical tensor
    graph::logical_tensor_t src_lt = utils::logical_tensor_init(
            0, {1, 4, 1, 1}, graph::data_type::f32);
    graph::logical_tensor_t weight_lt = utils::logical_tensor_init(
            1, {4, 4, 1, 1}, graph::data_type::f32);
    graph::logical_tensor_t dst_lt = utils::logical_tensor_init(
            2, {1, 8, 1, 1}, graph::data_type::f32);

    convtranspose_op.add_input(src_lt);
    convtranspose_op.add_input(weight_lt);
    convtranspose_op.add_output(dst_lt);

    graph::graph_t g(eng->kind());
    g.add_op(&convtranspose_op);
    g.finalize();

    graph::pass::pass_base_ptr apass = get_pass("convtranspose_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile
    graph::partition_t p;
    p.init(part);

    graph::compiled_partition_t cp(p);

    std::vector<const graph::logical_tensor_t *> inputs {&src_lt, &weight_lt};
    std::vector<const graph::logical_tensor_t *> outputs {&dst_lt};

    ASSERT_EQ(p.compile(&cp, inputs, outputs, eng), graph::status::success);
    ASSERT_EQ(dst_lt.layout_type, graph::layout_type::strided);

    graph::tensor_t src_ts(src_lt, eng, src.data());
    graph::tensor_t weight_ts(weight_lt, eng, weight.data());
    graph::tensor_t dst_ts(dst_lt, eng, dst.data());

    graph::stream_t *strm = get_stream();
    ASSERT_EQ(cp.execute(strm, {src_ts, weight_ts}, {dst_ts}),
            graph::status::success);
    strm->wait();
    for (size_t i = 0; i < dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(Execute, Convolution3DNcxOix) {
    using dims = std::vector<int64_t>;

    // default engine kind is cpu.
    graph::engine_t *eng = get_engine();
    test::vector<float> src {-3.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0,
            2.5, -1.0, 0, 3.0, -2.0, -1.0, 4.0};
    test::vector<float> weight {1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0};
    test::vector<float> ref_dst {-1.0, 2.5, 5.0, 1.5};
    test::vector<float> dst {0.0, 0.0, 0.0, 0.0};
    graph::op_t conv_op(graph::op_kind::Convolution);
    conv_op.set_attr<dims>(graph::op_attr::strides, dims {1, 1, 1});
    conv_op.set_attr<dims>(graph::op_attr::dilations, dims {1, 1, 1});
    conv_op.set_attr<dims>(graph::op_attr::pads_begin, dims {0, 0, 0});
    conv_op.set_attr<dims>(graph::op_attr::pads_end, dims {0, 0, 0});
    conv_op.set_attr<int64_t>(graph::op_attr::groups, 1);
    conv_op.set_attr<std::string>(graph::op_attr::data_format, "NCX");
    conv_op.set_attr<std::string>(graph::op_attr::weights_format, "OIX");

    // prepare logical tensor
    graph::logical_tensor_t src_lt = utils::logical_tensor_init(
            0, {1, 1, 1, 4, 4}, graph::data_type::f32);
    graph::logical_tensor_t weight_lt = utils::logical_tensor_init(
            1, {1, 1, 1, 3, 3}, graph::data_type::f32);
    graph::logical_tensor_t dst_lt = utils::logical_tensor_init(
            2, {1, 1, 1, 2, 2}, graph::data_type::f32);

    conv_op.add_input(src_lt);
    conv_op.add_input(weight_lt);
    conv_op.add_output(dst_lt);

    graph::graph_t g(eng->kind());
    g.add_op(&conv_op);
    g.finalize();

    graph::pass::pass_base_ptr apass = get_pass("conv_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile
    graph::partition_t p;
    p.init(part);

    graph::compiled_partition_t cp(p);

    std::vector<const graph::logical_tensor_t *> inputs {&src_lt, &weight_lt};
    std::vector<const graph::logical_tensor_t *> outputs {&dst_lt};

    p.compile(&cp, inputs, outputs, eng);

    graph::logical_tensor_t lt;
    cp.query_logical_tensor(dst_lt.id, &lt);
    ASSERT_EQ(lt.layout_type, graph::layout_type::strided);

    graph::tensor_t src_ts(src_lt, eng, src.data());
    graph::tensor_t weight_ts(weight_lt, eng, weight.data());
    graph::tensor_t dst_ts(dst_lt, eng, dst.data());

    graph::stream_t *strm = get_stream();
    cp.execute(strm, {src_ts, weight_ts}, {dst_ts});
    strm->wait();
    for (size_t i = 0; i < dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(Execute, ConvolutionNcxXio) {
    using dims = graph::dnnl_impl::dims;

    // default engine kind is cpu.
    graph::engine_t *eng = get_engine();
    test::vector<float> src {-3.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0,
            2.5, -1.0, 0, 3.0, -2.0, -1.0, 4.0};
    test::vector<float> weight {1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0};
    test::vector<float> ref_dst {-1.0, 2.5, 5.0, 1.5};
    test::vector<float> dst {0.0, 0.0, 0.0, 0.0};
    graph::op_t conv_op(graph::op_kind::Convolution);
    conv_op.set_attr<dims>(graph::op_attr::strides, dims {1, 1});
    conv_op.set_attr<dims>(graph::op_attr::dilations, dims {1, 1});
    conv_op.set_attr<dims>(graph::op_attr::pads_begin, dims {0, 0});
    conv_op.set_attr<dims>(graph::op_attr::pads_end, dims {0, 0});
    conv_op.set_attr<int64_t>(graph::op_attr::groups, 1);
    conv_op.set_attr<std::string>(graph::op_attr::data_format, "NCX");
    conv_op.set_attr<std::string>(graph::op_attr::weights_format, "XIO");

    // prepare logical tensor
    graph::logical_tensor_t src_lt = utils::logical_tensor_init(
            0, {1, 1, 4, 4}, graph::data_type::f32);
    graph::logical_tensor_t weight_lt = utils::logical_tensor_init(
            1, {3, 3, 1, 1}, graph::data_type::f32);
    graph::logical_tensor_t dst_lt = utils::logical_tensor_init(
            2, {1, 1, 2, 2}, graph::data_type::f32);

    conv_op.add_input(src_lt);
    conv_op.add_input(weight_lt);
    conv_op.add_output(dst_lt);

    graph::graph_t g(eng->kind());
    g.add_op(&conv_op);
    g.finalize();

    graph::pass::pass_base_ptr apass = get_pass("conv_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile
    graph::partition_t p;
    p.init(part);

    graph::compiled_partition_t cp(p);

    std::vector<const graph::logical_tensor_t *> inputs {&src_lt, &weight_lt};
    std::vector<const graph::logical_tensor_t *> outputs {&dst_lt};

    p.compile(&cp, inputs, outputs, eng);

    graph::logical_tensor_t lt;
    cp.query_logical_tensor(dst_lt.id, &lt);
    ASSERT_EQ(lt.layout_type, graph::layout_type::strided);

    graph::tensor_t src_ts(src_lt, eng, src.data());
    graph::tensor_t weight_ts(weight_lt, eng, weight.data());
    graph::tensor_t dst_ts(dst_lt, eng, dst.data());

    graph::stream_t *strm = get_stream();
    cp.execute(strm, {src_ts, weight_ts}, {dst_ts});
    strm->wait();
    for (size_t i = 0; i < dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(Execute, Convolution3DNcxXio) {
    using dims = graph::dnnl_impl::dims;

    // default engine kind is cpu.
    graph::engine_t *eng = get_engine();
    test::vector<float> src {-3.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0,
            2.5, -1.0, 0, 3.0, -2.0, -1.0, 4.0};
    test::vector<float> weight {1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0};
    test::vector<float> ref_dst {-1.0, 2.5, 5.0, 1.5};
    test::vector<float> dst {0.0, 0.0, 0.0, 0.0};
    graph::op_t conv_op(graph::op_kind::Convolution);
    conv_op.set_attr<dims>(graph::op_attr::strides, dims {1, 1, 1});
    conv_op.set_attr<dims>(graph::op_attr::dilations, dims {1, 1, 1});
    conv_op.set_attr<dims>(graph::op_attr::pads_begin, dims {0, 0, 0});
    conv_op.set_attr<dims>(graph::op_attr::pads_end, dims {0, 0, 0});
    conv_op.set_attr<int64_t>(graph::op_attr::groups, 1);
    conv_op.set_attr<std::string>(graph::op_attr::data_format, "NCX");
    conv_op.set_attr<std::string>(graph::op_attr::weights_format, "XIO");

    // prepare logical tensor
    graph::logical_tensor_t src_lt = utils::logical_tensor_init(
            0, {1, 1, 1, 4, 4}, graph::data_type::f32);
    graph::logical_tensor_t weight_lt = utils::logical_tensor_init(
            1, {1, 3, 3, 1, 1}, graph::data_type::f32);
    graph::logical_tensor_t dst_lt = utils::logical_tensor_init(
            2, {1, 1, 1, 2, 2}, graph::data_type::f32);

    conv_op.add_input(src_lt);
    conv_op.add_input(weight_lt);
    conv_op.add_output(dst_lt);

    graph::graph_t g(eng->kind());
    g.add_op(&conv_op);
    g.finalize();

    graph::pass::pass_base_ptr apass = get_pass("conv_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile
    graph::partition_t p;
    p.init(part);

    graph::compiled_partition_t cp(p);

    std::vector<const graph::logical_tensor_t *> inputs {&src_lt, &weight_lt};
    std::vector<const graph::logical_tensor_t *> outputs {&dst_lt};

    p.compile(&cp, inputs, outputs, eng);

    graph::logical_tensor_t lt;
    cp.query_logical_tensor(dst_lt.id, &lt);
    ASSERT_EQ(lt.layout_type, graph::layout_type::strided);

    graph::tensor_t src_ts(src_lt, eng, src.data());
    graph::tensor_t weight_ts(weight_lt, eng, weight.data());
    graph::tensor_t dst_ts(dst_lt, eng, dst.data());

    graph::stream_t *strm = get_stream();
    cp.execute(strm, {src_ts, weight_ts}, {dst_ts});
    strm->wait();
    for (size_t i = 0; i < dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(Execute, ConvolutionNxcXio) {
    using dims = graph::dnnl_impl::dims;

    // default engine kind is cpu.
    graph::engine_t *eng = get_engine();
    test::vector<float> src {
            -3.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0, 2.5, -1.0, 0};
    test::vector<float> weight {
            1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0};
    test::vector<float> ref_dst {0.5};
    test::vector<float> dst {0.0};
    graph::op_t conv_op(graph::op_kind::Convolution);
    conv_op.set_attr<dims>(graph::op_attr::strides, dims {1, 1});
    conv_op.set_attr<dims>(graph::op_attr::dilations, dims {1, 1});
    conv_op.set_attr<dims>(graph::op_attr::pads_begin, dims {0, 0});
    conv_op.set_attr<dims>(graph::op_attr::pads_end, dims {0, 0});
    conv_op.set_attr<int64_t>(graph::op_attr::groups, 1);
    conv_op.set_attr<std::string>(graph::op_attr::data_format, "NXC");
    conv_op.set_attr<std::string>(graph::op_attr::weights_format, "XIO");

    // prepare logical tensor
    graph::logical_tensor_t src_lt = utils::logical_tensor_init(
            0, {1, 3, 4, 1}, graph::data_type::f32);
    graph::logical_tensor_t weight_lt = utils::logical_tensor_init(
            1, {3, 4, 1, 1}, graph::data_type::f32);
    graph::logical_tensor_t dst_lt = utils::logical_tensor_init(
            2, {1, 1, 1, 1}, graph::data_type::f32);

    conv_op.add_input(src_lt);
    conv_op.add_input(weight_lt);
    conv_op.add_output(dst_lt);

    graph::graph_t g(eng->kind());
    g.add_op(&conv_op);
    g.finalize();

    graph::pass::pass_base_ptr apass = get_pass("conv_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile
    graph::partition_t p;
    p.init(part);

    graph::compiled_partition_t cp(p);

    std::vector<const graph::logical_tensor_t *> inputs {&src_lt, &weight_lt};
    std::vector<const graph::logical_tensor_t *> outputs {&dst_lt};

    p.compile(&cp, inputs, outputs, eng);

    graph::logical_tensor_t lt;
    cp.query_logical_tensor(dst_lt.id, &lt);
    ASSERT_EQ(lt.layout_type, graph::layout_type::strided);

    graph::tensor_t src_ts(src_lt, eng, src.data());
    graph::tensor_t weight_ts(weight_lt, eng, weight.data());
    graph::tensor_t dst_ts(dst_lt, eng, dst.data());

    graph::stream_t *strm = get_stream();
    cp.execute(strm, {src_ts, weight_ts}, {dst_ts});
    strm->wait();
    for (size_t i = 0; i < dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(Execute, Convolution3DNxcXio) {
    using dims = graph::dnnl_impl::dims;

    // default engine kind is cpu.
    graph::engine_t *eng = get_engine();
    test::vector<float> src {
            -3.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0, 2.5, -1.0, 0};
    test::vector<float> weight {
            1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0};
    test::vector<float> ref_dst {0.5};
    test::vector<float> dst {0.0};
    graph::op_t conv_op(graph::op_kind::Convolution);
    conv_op.set_attr<dims>(graph::op_attr::strides, dims {1, 1, 1});
    conv_op.set_attr<dims>(graph::op_attr::dilations, dims {1, 1, 1});
    conv_op.set_attr<dims>(graph::op_attr::pads_begin, dims {0, 0, 0});
    conv_op.set_attr<dims>(graph::op_attr::pads_end, dims {0, 0, 0});
    conv_op.set_attr<int64_t>(graph::op_attr::groups, 1);
    conv_op.set_attr<std::string>(graph::op_attr::data_format, "NXC");
    conv_op.set_attr<std::string>(graph::op_attr::weights_format, "XIO");

    // prepare logical tensor
    graph::logical_tensor_t src_lt = utils::logical_tensor_init(
            0, {1, 1, 3, 4, 1}, graph::data_type::f32);
    graph::logical_tensor_t weight_lt = utils::logical_tensor_init(
            1, {1, 3, 4, 1, 1}, graph::data_type::f32);
    graph::logical_tensor_t dst_lt = utils::logical_tensor_init(
            2, {1, 1, 1, 1, 1}, graph::data_type::f32);

    conv_op.add_input(src_lt);
    conv_op.add_input(weight_lt);
    conv_op.add_output(dst_lt);

    graph::graph_t g(eng->kind());
    g.add_op(&conv_op);
    g.finalize();

    graph::pass::pass_base_ptr apass = get_pass("conv_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile
    graph::partition_t p;
    p.init(part);

    graph::compiled_partition_t cp(p);

    std::vector<const graph::logical_tensor_t *> inputs {&src_lt, &weight_lt};
    std::vector<const graph::logical_tensor_t *> outputs {&dst_lt};

    p.compile(&cp, inputs, outputs, eng);

    graph::logical_tensor_t lt;
    cp.query_logical_tensor(dst_lt.id, &lt);
    ASSERT_EQ(lt.layout_type, graph::layout_type::strided);

    graph::tensor_t src_ts(src_lt, eng, src.data());
    graph::tensor_t weight_ts(weight_lt, eng, weight.data());
    graph::tensor_t dst_ts(dst_lt, eng, dst.data());

    graph::stream_t *strm = get_stream();
    cp.execute(strm, {src_ts, weight_ts}, {dst_ts});
    strm->wait();
    for (size_t i = 0; i < dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(Execute, ConvolutionNxcOix) {
    using dims = graph::dnnl_impl::dims;

    // default engine kind is cpu.
    graph::engine_t *eng = get_engine();
    test::vector<float> src {-3.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0,
            2.5, -1.0, 0, 3.0, -2.0, -1.0, 4.0};
    test::vector<float> weight {1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0};
    test::vector<float> ref_dst {-1.0, 2.5, 5.0, 1.5};
    test::vector<float> dst {0.0, 0.0, 0.0, 0.0};
    graph::op_t conv_op(graph::op_kind::Convolution);
    conv_op.set_attr<dims>(graph::op_attr::strides, dims {1, 1});
    conv_op.set_attr<dims>(graph::op_attr::dilations, dims {1, 1});
    conv_op.set_attr<dims>(graph::op_attr::pads_begin, dims {0, 0});
    conv_op.set_attr<dims>(graph::op_attr::pads_end, dims {0, 0});
    conv_op.set_attr<int64_t>(graph::op_attr::groups, 1);
    conv_op.set_attr<std::string>(graph::op_attr::data_format, "NXC");
    conv_op.set_attr<std::string>(graph::op_attr::weights_format, "OIX");

    // prepare logical tensor
    graph::logical_tensor_t src_lt = utils::logical_tensor_init(
            0, {1, 4, 4, 1}, graph::data_type::f32);
    graph::logical_tensor_t weight_lt = utils::logical_tensor_init(
            1, {1, 1, 3, 3}, graph::data_type::f32);
    graph::logical_tensor_t dst_lt = utils::logical_tensor_init(
            2, {1, 2, 2, 1}, graph::data_type::f32);

    conv_op.add_input(src_lt);
    conv_op.add_input(weight_lt);
    conv_op.add_output(dst_lt);

    graph::graph_t g(eng->kind());
    g.add_op(&conv_op);
    g.finalize();

    graph::pass::pass_base_ptr apass = get_pass("conv_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile
    graph::partition_t p;
    p.init(part);

    graph::compiled_partition_t cp(p);

    std::vector<const graph::logical_tensor_t *> inputs {&src_lt, &weight_lt};
    std::vector<const graph::logical_tensor_t *> outputs {&dst_lt};

    p.compile(&cp, inputs, outputs, eng);

    graph::logical_tensor_t lt;
    cp.query_logical_tensor(dst_lt.id, &lt);
    ASSERT_EQ(lt.layout_type, graph::layout_type::strided);

    graph::tensor_t src_ts(src_lt, eng, src.data());
    graph::tensor_t weight_ts(weight_lt, eng, weight.data());
    graph::tensor_t dst_ts(dst_lt, eng, dst.data());

    graph::stream_t *strm = get_stream();
    cp.execute(strm, {src_ts, weight_ts}, {dst_ts});
    strm->wait();
    for (size_t i = 0; i < dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(Execute, Convolution3DNxcOix) {
    using dims = graph::dnnl_impl::dims;

    // default engine kind is cpu.
    graph::engine_t *eng = get_engine();
    test::vector<float> src {-3.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0,
            2.5, -1.0, 0, 3.0, -2.0, -1.0, 4.0};
    test::vector<float> weight {1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0};
    test::vector<float> ref_dst {-1.0, 2.5, 5.0, 1.5};
    test::vector<float> dst {0.0, 0.0, 0.0, 0.0};
    graph::op_t conv_op(graph::op_kind::Convolution);
    conv_op.set_attr<dims>(graph::op_attr::strides, dims {1, 1, 1});
    conv_op.set_attr<dims>(graph::op_attr::dilations, dims {1, 1, 1});
    conv_op.set_attr<dims>(graph::op_attr::pads_begin, dims {0, 0, 0});
    conv_op.set_attr<dims>(graph::op_attr::pads_end, dims {0, 0, 0});
    conv_op.set_attr<int64_t>(graph::op_attr::groups, 1);
    conv_op.set_attr<std::string>(graph::op_attr::data_format, "NXC");
    conv_op.set_attr<std::string>(graph::op_attr::weights_format, "OIX");

    // prepare logical tensor
    graph::logical_tensor_t src_lt = utils::logical_tensor_init(
            0, {1, 1, 4, 4, 1}, graph::data_type::f32);
    graph::logical_tensor_t weight_lt = utils::logical_tensor_init(
            1, {1, 1, 1, 3, 3}, graph::data_type::f32);
    graph::logical_tensor_t dst_lt = utils::logical_tensor_init(
            2, {1, 1, 2, 2, 1}, graph::data_type::f32);

    conv_op.add_input(src_lt);
    conv_op.add_input(weight_lt);
    conv_op.add_output(dst_lt);

    graph::graph_t g(eng->kind());
    g.add_op(&conv_op);
    g.finalize();

    graph::pass::pass_base_ptr apass = get_pass("conv_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile
    graph::partition_t p;
    p.init(part);

    graph::compiled_partition_t cp(p);

    std::vector<const graph::logical_tensor_t *> inputs {&src_lt, &weight_lt};
    std::vector<const graph::logical_tensor_t *> outputs {&dst_lt};

    p.compile(&cp, inputs, outputs, eng);

    graph::logical_tensor_t lt;
    cp.query_logical_tensor(dst_lt.id, &lt);
    ASSERT_EQ(lt.layout_type, graph::layout_type::strided);

    graph::tensor_t src_ts(src_lt, eng, src.data());
    graph::tensor_t weight_ts(weight_lt, eng, weight.data());
    graph::tensor_t dst_ts(dst_lt, eng, dst.data());

    graph::stream_t *strm = get_stream();
    cp.execute(strm, {src_ts, weight_ts}, {dst_ts});
    strm->wait();
    for (size_t i = 0; i < dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(Execute, ConvolutionF16F16F16) {
    using dims = graph::dnnl_impl::dims;

    graph::engine_t *eng = get_engine();
    SKIP_IF(eng->kind() != graph::engine_kind::gpu,
            "Skip fp16 test for non-GPU device.");
    test::vector<float> src {-3.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0,
            2.5, -1.0, 0, 3.0, -2.0, -1.0, 4.0};
    test::vector<float> weight {1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0};
    test::vector<float> ref_dst {-1.0, 2.5, 5.0, 1.5};
    test::vector<float> dst {0.0, 0.0, 0.0, 0.0};
    graph::op_t conv_op(graph::op_kind::Convolution);
    conv_op.set_attr<dims>(graph::op_attr::strides, dims {1, 1});
    conv_op.set_attr<dims>(graph::op_attr::dilations, dims {1, 1});
    conv_op.set_attr<dims>(graph::op_attr::pads_begin, dims {0, 0});
    conv_op.set_attr<dims>(graph::op_attr::pads_end, dims {0, 0});
    conv_op.set_attr<int64_t>(graph::op_attr::groups, 1);
    conv_op.set_attr<std::string>(graph::op_attr::data_format, "NCX");
    conv_op.set_attr<std::string>(graph::op_attr::weights_format, "OIX");

    // prepare logical tensor
    graph::logical_tensor_t src_lt = utils::logical_tensor_init(
            0, {1, 1, 4, 4}, graph::data_type::f16);
    graph::logical_tensor_t weight_lt = utils::logical_tensor_init(
            1, {1, 1, 3, 3}, graph::data_type::f16);
    graph::logical_tensor_t dst_lt = utils::logical_tensor_init(
            2, {1, 1, 2, 2}, graph::data_type::f16);

    conv_op.add_input(src_lt);
    conv_op.add_input(weight_lt);
    conv_op.add_output(dst_lt);

    graph::graph_t g(eng->kind());
    g.add_op(&conv_op);
    g.finalize();

    graph::pass::pass_base_ptr apass = get_pass("conv_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile
    graph::partition_t p;
    p.init(part);

    graph::compiled_partition_t cp(p);

    std::vector<const graph::logical_tensor_t *> inputs {&src_lt, &weight_lt};
    std::vector<const graph::logical_tensor_t *> outputs {&dst_lt};

    p.compile(&cp, inputs, outputs, eng);

    graph::logical_tensor_t lt;
    cp.query_logical_tensor(dst_lt.id, &lt);
    ASSERT_EQ(lt.layout_type, graph::layout_type::strided);

    graph::tensor_t src_ts(src_lt, eng, src.data());
    graph::tensor_t weight_ts(weight_lt, eng, weight.data());
    graph::tensor_t dst_ts(dst_lt, eng, dst.data());

    graph::stream_t *strm = get_stream();
    cp.execute(strm, {src_ts, weight_ts}, {dst_ts});
    strm->wait();
}

TEST(Execute, ConvolutionBf16Bf16Bf16) {
    using dims = graph::dnnl_impl::dims;

    graph::engine_t *eng = get_engine();

    static auto isa = dnnl_get_effective_cpu_isa();
    SKIP_IF((isa < dnnl_cpu_isa_avx512_core)
                    && eng->kind() == graph::engine_kind::cpu,
            "Skip bf16 tests for systems that do not support avx512_core.");

    test::vector<float> src {-3.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0,
            2.5, -1.0, 0, 3.0, -2.0, -1.0, 4.0};
    test::vector<float> weight {1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0};
    test::vector<float> ref_dst {-1.0, 2.5, 5.0, 1.5};
    test::vector<float> dst {0.0, 0.0, 0.0, 0.0};
    graph::op_t conv_op(graph::op_kind::Convolution);
    conv_op.set_attr<dims>(graph::op_attr::strides, dims {1, 1});
    conv_op.set_attr<dims>(graph::op_attr::dilations, dims {1, 1});
    conv_op.set_attr<dims>(graph::op_attr::pads_begin, dims {0, 0});
    conv_op.set_attr<dims>(graph::op_attr::pads_end, dims {0, 0});
    conv_op.set_attr<int64_t>(graph::op_attr::groups, 1);
    conv_op.set_attr<std::string>(graph::op_attr::data_format, "NCX");
    conv_op.set_attr<std::string>(graph::op_attr::weights_format, "OIX");

    // prepare logical tensor
    graph::logical_tensor_t src_lt = utils::logical_tensor_init(
            0, {1, 1, 4, 4}, graph::data_type::bf16);
    graph::logical_tensor_t weight_lt = utils::logical_tensor_init(
            1, {1, 1, 3, 3}, graph::data_type::bf16);
    graph::logical_tensor_t dst_lt = utils::logical_tensor_init(
            2, {1, 1, 2, 2}, graph::data_type::bf16);

    conv_op.add_input(src_lt);
    conv_op.add_input(weight_lt);
    conv_op.add_output(dst_lt);

    graph::graph_t g(eng->kind());
    g.add_op(&conv_op);
    g.finalize();

    graph::pass::pass_base_ptr apass = get_pass("conv_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile
    graph::partition_t p;
    p.init(part);

    graph::compiled_partition_t cp(p);

    std::vector<const graph::logical_tensor_t *> inputs {&src_lt, &weight_lt};
    std::vector<const graph::logical_tensor_t *> outputs {&dst_lt};

    p.compile(&cp, inputs, outputs, eng);

    graph::logical_tensor_t lt;
    cp.query_logical_tensor(dst_lt.id, &lt);
    ASSERT_EQ(lt.layout_type, graph::layout_type::strided);

    graph::tensor_t src_ts(src_lt, eng, src.data());
    graph::tensor_t weight_ts(weight_lt, eng, weight.data());
    graph::tensor_t dst_ts(dst_lt, eng, dst.data());

    graph::stream_t *strm = get_stream();
    cp.execute(strm, {src_ts, weight_ts}, {dst_ts});
    strm->wait();
}

TEST(Compile, ConvAddSharedInputs) {
    /*      /\  /
           / Conv
           \  /
           Add
    */
    using dims = graph::dnnl_impl::dims;

    // prepare logical tensor
    graph::logical_tensor_t src_lt = utils::logical_tensor_init(
            0, {1, 1, 4, 4}, graph::data_type::f32);
    graph::logical_tensor_t weight_lt = utils::logical_tensor_init(
            1, {1, 1, 1, 1}, graph::data_type::f32);
    graph::logical_tensor_t conv_dst_lt = utils::logical_tensor_init(
            2, {1, 1, 4, 4}, graph::data_type::f32);
    graph::logical_tensor_t add_dst_lt = utils::logical_tensor_init(
            3, {1, 1, 4, 4}, graph::data_type::f32);

    // create op conv
    graph::op_t conv_op(0, graph::op_kind::Convolution, "Convolution");
    conv_op.set_attr<dims>(graph::op_attr::strides, dims {1, 1});
    conv_op.set_attr<dims>(graph::op_attr::dilations, dims {1, 1});
    conv_op.set_attr<dims>(graph::op_attr::pads_begin, dims {0, 0});
    conv_op.set_attr<dims>(graph::op_attr::pads_end, dims {0, 0});
    conv_op.set_attr<int64_t>(graph::op_attr::groups, 1);
    conv_op.set_attr<std::string>(graph::op_attr::data_format, "NCX");
    conv_op.set_attr<std::string>(graph::op_attr::weights_format, "OIX");
    conv_op.add_input(src_lt);
    conv_op.add_input(weight_lt);
    conv_op.add_output(conv_dst_lt);
    // create op add
    graph::op_t add_op(1, graph::op_kind::Add, "Add");
    add_op.add_input(conv_dst_lt);
    add_op.add_input(src_lt);
    add_op.add_output(add_dst_lt);
    // build graph
    graph::engine_t *eng = get_engine();
    graph::graph_t g(eng->kind());
    g.add_op(&conv_op);
    g.add_op(&add_op);
    g.finalize();

    // run pass
    graph::pass::pass_base_ptr apass = get_pass("fp_conv_post_ops");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile conv+add partition
    graph::partition_t p;
    p.init(part);
    graph::compiled_partition_t cp(p);
    std::vector<const graph::logical_tensor_t *> inputs {
            &src_lt, &weight_lt, &src_lt};
    std::vector<const graph::logical_tensor_t *> outputs {&add_dst_lt};
    ASSERT_EQ(p.compile(&cp, inputs, outputs, eng), graph::status::success);

    // check inplace pairs
    std::vector<graph::inplace_pair_t> inplace_pairs = cp.get_inplace_pairs();
    ASSERT_EQ(inplace_pairs.size(), 0U);
}

TEST(Compile, ConvAddInplace) {
    /*      \  /
             Conv
           \  /
           Add
    */
    using dims = graph::dnnl_impl::dims;

    // TODO(qun): re-enable this test once library and bridge align the inplace
    // logic
    SKIP_IF(true, "library and bridge have different inplace logic");

    // prepare logical tensor
    graph::logical_tensor_t src_lt = utils::logical_tensor_init(
            0, {1, 1, 4, 4}, graph::data_type::f32);
    graph::logical_tensor_t weight_lt = utils::logical_tensor_init(
            1, {1, 1, 1, 1}, graph::data_type::f32);
    graph::logical_tensor_t conv_dst_lt = utils::logical_tensor_init(
            2, {1, 1, 4, 4}, graph::data_type::f32);
    graph::logical_tensor_t add_src_lt = utils::logical_tensor_init(
            3, {1, 1, 4, 4}, graph::data_type::f32);
    graph::logical_tensor_t add_dst_lt = utils::logical_tensor_init(
            4, {1, 1, 4, 4}, graph::data_type::f32);

    // create op conv
    graph::op_t conv_op(0, graph::op_kind::Convolution, "Convolution");
    conv_op.set_attr<dims>(graph::op_attr::strides, dims {1, 1});
    conv_op.set_attr<dims>(graph::op_attr::dilations, dims {1, 1});
    conv_op.set_attr<dims>(graph::op_attr::pads_begin, dims {0, 0});
    conv_op.set_attr<dims>(graph::op_attr::pads_end, dims {0, 0});
    conv_op.set_attr<int64_t>(graph::op_attr::groups, 1);
    conv_op.set_attr<std::string>(graph::op_attr::data_format, "NCX");
    conv_op.set_attr<std::string>(graph::op_attr::weights_format, "OIX");
    conv_op.add_input(src_lt);
    conv_op.add_input(weight_lt);
    conv_op.add_output(conv_dst_lt);
    // create op add
    graph::op_t add_op(1, graph::op_kind::Add, "Add");
    add_op.add_input(add_src_lt);
    add_op.add_input(conv_dst_lt);
    add_op.add_output(add_dst_lt);
    // build graph
    graph::engine_t *eng = get_engine();
    graph::graph_t g(eng->kind());
    g.add_op(&conv_op);
    g.add_op(&add_op);
    g.finalize();

    // run pass
    graph::pass::pass_base_ptr apass = get_pass("fp_conv_post_ops");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile conv+add partition
    graph::partition_t p;
    p.init(part);
    graph::compiled_partition_t cp(p);
    // arbitrary order of inputs
    std::vector<const graph::logical_tensor_t *> inputs {
            &add_src_lt, &weight_lt, &src_lt};
    std::vector<const graph::logical_tensor_t *> outputs {&add_dst_lt};
    ASSERT_EQ(p.compile(&cp, inputs, outputs, eng), graph::status::success);

    // check inplace pairs
    std::vector<graph::inplace_pair_t> inplace_pairs = cp.get_inplace_pairs();
    ASSERT_EQ(inplace_pairs.size(), 1U);
    ASSERT_EQ(inplace_pairs[0].input_id, add_src_lt.id);
    ASSERT_EQ(inplace_pairs[0].output_id, add_dst_lt.id);
}

TEST(Execute, GroupConvolution) {
    using dims = graph::dnnl_impl::dims;

    // default engine kind is cpu.
    graph::engine_t *eng = get_engine();

    graph::op_t conv_op(graph::op_kind::Convolution);
    conv_op.set_attr<dims>(graph::op_attr::strides, dims {1, 1});
    conv_op.set_attr<dims>(graph::op_attr::dilations, dims {1, 1});
    conv_op.set_attr<dims>(graph::op_attr::pads_begin, dims {0, 0});
    conv_op.set_attr<dims>(graph::op_attr::pads_end, dims {0, 0});
    conv_op.set_attr<int64_t>(graph::op_attr::groups, 4);
    conv_op.set_attr<std::string>(graph::op_attr::data_format, "NCX");
    conv_op.set_attr<std::string>(graph::op_attr::weights_format, "OIX");

    // prepare logical tensor
    graph::logical_tensor_t src_lt = utils::logical_tensor_init(
            0, {8, 32, 16, 16}, graph::data_type::f32);
    graph::logical_tensor_t weight_lt = utils::logical_tensor_init(
            1, {32, 8, 1, 1}, graph::data_type::f32);
    graph::logical_tensor_t dst_lt = utils::logical_tensor_init(
            2, {8, 32, 16, 16}, graph::data_type::f32);

    conv_op.add_input(src_lt);
    conv_op.add_input(weight_lt);
    conv_op.add_output(dst_lt);

    graph::graph_t g(eng->kind());
    g.add_op(&conv_op);
    g.finalize();

    graph::pass::pass_base_ptr apass = get_pass("conv_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile
    graph::partition_t p;
    p.init(part);

    graph::compiled_partition_t cp(p);

    std::vector<const graph::logical_tensor_t *> inputs {&src_lt, &weight_lt};
    std::vector<const graph::logical_tensor_t *> outputs {&dst_lt};

    p.compile(&cp, inputs, outputs, eng);

    graph::logical_tensor_t lt;
    cp.query_logical_tensor(dst_lt.id, &lt);
    ASSERT_EQ(lt.layout_type, graph::layout_type::strided);

    test::vector<float> src(8 * 32 * 16 * 16, 1);
    test::vector<float> weight(32 * 8 * 1 * 1, 1);
    test::vector<float> dst(8 * 32 * 16 * 16, 1);

    graph::tensor_t src_ts(src_lt, eng, src.data());
    graph::tensor_t weight_ts(weight_lt, eng, weight.data());
    graph::tensor_t dst_ts(dst_lt, eng, dst.data());

    graph::stream_t *strm = get_stream();
    cp.execute(strm, {src_ts, weight_ts}, {dst_ts});
    strm->wait();
    for (size_t i = 0; i < dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], 8);
    }
}

TEST(Execute, ConvolutionBackwardData) {
    using dims = dnnl::impl::graph::dnnl_impl::dims;

    graph::engine_t *eng = get_engine();

    test::vector<float> src {-3.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0,
            2.5, -1.0, 0, 3.0, -2.0, -1.0, 4.0};
    test::vector<float> ref_diff_src {0.0, 1.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0,
            0.0, 3.0, 3.0, 1.0, 2.0, 3.0, 2.0, 3.0};
    test::vector<float> weight {1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0};
    test::vector<float> diff_dst {0.0, 1.0, 2.0, 3.0};
    test::vector<float> diff_src(src.size(), 0.0);

    graph::op_t conv_op(graph::op_kind::ConvolutionBackwardData);
    conv_op.set_attr<dims>(graph::op_attr::strides, dims {1, 1});
    conv_op.set_attr<dims>(graph::op_attr::dilations, dims {1, 1});
    conv_op.set_attr<dims>(graph::op_attr::pads_begin, dims {0, 0});
    conv_op.set_attr<dims>(graph::op_attr::pads_end, dims {0, 0});
    conv_op.set_attr<int64_t>(graph::op_attr::groups, 1);
    conv_op.set_attr<std::string>(graph::op_attr::data_format, "NCX");
    conv_op.set_attr<std::string>(graph::op_attr::weights_format, "OIX");
    conv_op.set_attr<dims>(graph::op_attr::dst_shape, dims {1, 1, 4, 4});

    // prepare logical tensor
    graph::logical_tensor_t diff_src_lt = utils::logical_tensor_init(
            1, {1, 1, 4, 4}, graph::data_type::f32);
    graph::logical_tensor_t weight_lt = utils::logical_tensor_init(
            2, {1, 1, 3, 3}, graph::data_type::f32);
    graph::logical_tensor_t diff_dst_lt = utils::logical_tensor_init(
            3, {1, 1, 2, 2}, graph::data_type::f32);

    conv_op.add_input(diff_dst_lt);
    conv_op.add_input(weight_lt);
    conv_op.add_output(diff_src_lt);

    graph::graph_t g(eng->kind());
    g.add_op(&conv_op);
    g.finalize();

    graph::pass::pass_base_ptr apass = get_pass("conv_data_bw_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile
    graph::partition_t p;
    p.init(part);
    graph::compiled_partition_t cp(p);

    std::vector<const graph::logical_tensor_t *> inputs {
            &diff_dst_lt, &weight_lt};
    std::vector<const graph::logical_tensor_t *> outputs {&diff_src_lt};
    ASSERT_EQ(p.compile(&cp, inputs, outputs, eng), graph::status::success);

    graph::logical_tensor_t lt;
    cp.query_logical_tensor(diff_src_lt.id, &lt);
    ASSERT_EQ(lt.layout_type, graph::layout_type::strided);

    graph::tensor_t weight_ts(weight_lt, eng, weight.data());
    graph::tensor_t diff_dst_ts(diff_dst_lt, eng, diff_dst.data());
    graph::tensor_t diff_src_ts(diff_src_lt, eng, diff_src.data());

    graph::stream_t *strm = get_stream();
    cp.execute(strm, {diff_dst_ts, weight_ts}, {diff_src_ts});
    strm->wait();
    for (size_t i = 0; i < diff_src.size(); ++i) {
        ASSERT_FLOAT_EQ(diff_src[i], ref_diff_src[i]);
    }
}

TEST(Execute, ConvolutionBnFp32) {
    using dims = graph::dnnl_impl::dims;

    // default engine kind is cpu.
    graph::engine_t *eng = get_engine();
    graph::stream_t *strm = get_stream();

    graph::op_t conv_op(0, graph::op_kind::Convolution, "conv");
    conv_op.set_attr<dims>(graph::op_attr::strides, dims {1, 1});
    conv_op.set_attr<dims>(graph::op_attr::dilations, dims {1, 1});
    conv_op.set_attr<dims>(graph::op_attr::pads_begin, dims {0, 0});
    conv_op.set_attr<dims>(graph::op_attr::pads_end, dims {0, 0});
    conv_op.set_attr<int64_t>(graph::op_attr::groups, 1);
    conv_op.set_attr<std::string>(graph::op_attr::data_format, "NCX");
    conv_op.set_attr<std::string>(graph::op_attr::weights_format, "OIX");

    graph::op_t bn_op(1, graph::op_kind::BatchNormInference, "bn");
    bn_op.set_attr<std::string>(graph::op_attr::data_format, "NCX");
    bn_op.set_attr(graph::op_attr::epsilon, 1e-6f);

    // prepare logical tensor
    graph::logical_tensor_t conv_src_lt = utils::logical_tensor_init(
            0, {8, 32, 16, 16}, graph::data_type::f32);
    graph::logical_tensor_t conv_weight_lt = utils::logical_tensor_init(
            1, {32, 32, 1, 1}, graph::data_type::f32);
    graph::logical_tensor_t conv_dst_lt = utils::logical_tensor_init(
            2, {8, 32, 16, 16}, graph::data_type::f32);
    graph::logical_tensor_t gamma_lt
            = utils::logical_tensor_init(3, {32}, graph::data_type::f32);
    graph::logical_tensor_t beta_lt
            = utils::logical_tensor_init(4, {32}, graph::data_type::f32);
    graph::logical_tensor_t scale_lt
            = utils::logical_tensor_init(5, {32}, graph::data_type::f32);
    graph::logical_tensor_t shift_lt
            = utils::logical_tensor_init(6, {32}, graph::data_type::f32);
    graph::logical_tensor_t bn_dst_lt = utils::logical_tensor_init(
            7, {8, 32, 16, 16}, graph::data_type::f32);

    test::vector<float> conv_src(8 * 32 * 16 * 16);
    test::vector<float> conv_weight(32 * 32 * 1 * 1);
    test::vector<float> bn_gamma(32);
    test::vector<float> bn_beta(32);
    test::vector<float> bn_scale(32);
    test::vector<float> bn_shift(32);
    test::vector<float> bn_dst(8 * 32 * 16 * 16);

    // Initialize
    std::default_random_engine generator;
    std::normal_distribution<float> distribution(0.0f, 0.1f);

    std::generate(conv_src.begin(), conv_src.end(),
            [&]() { return distribution(generator); });
    std::generate(conv_weight.begin(), conv_weight.end(),
            [&]() { return distribution(generator); });
    std::generate(bn_gamma.begin(), bn_gamma.end(),
            [&]() { return distribution(generator); });
    std::generate(bn_beta.begin(), bn_beta.end(),
            [&]() { return distribution(generator); });
    std::generate(bn_scale.begin(), bn_scale.end(),
            [&]() { return distribution(generator); });
    std::generate(bn_shift.begin(), bn_shift.end(),
            [&]() { return distribution(generator); });

    graph::tensor_t conv_src_ts(conv_src_lt, eng, conv_src.data());
    graph::tensor_t conv_weight_ts(conv_weight_lt, eng, conv_weight.data());
    graph::tensor_t bn_gamma_ts(gamma_lt, eng, bn_gamma.data());
    graph::tensor_t bn_beta_ts(beta_lt, eng, bn_beta.data());
    graph::tensor_t bn_scale_ts(scale_lt, eng, bn_scale.data());
    graph::tensor_t bn_shift_ts(shift_lt, eng, bn_shift.data());
    graph::tensor_t bn_dst_ts(bn_dst_lt, eng, bn_dst.data());

    conv_op.add_input(conv_src_lt);
    conv_op.add_input(conv_weight_lt);
    conv_op.add_output(conv_dst_lt);
    bn_op.add_input(conv_dst_lt);
    bn_op.add_input(gamma_lt);
    bn_op.add_input(beta_lt);
    bn_op.add_input(scale_lt);
    bn_op.add_input(shift_lt);
    bn_op.add_output(bn_dst_lt);

    graph::graph_t g(eng->kind());
    g.add_op(&conv_op);
    g.add_op(&bn_op);
    g.finalize();

    // run unfused graph to compute the reference
    ASSERT_EQ(run_graph(g,
                      {conv_src_ts, conv_weight_ts, bn_gamma_ts, bn_beta_ts,
                              bn_scale_ts, bn_shift_ts},
                      {bn_dst_ts}, *eng, *strm),
            graph::status::success);

    // run fusion partition
    graph::pass::pass_base_ptr apass = get_pass("fp_conv_post_ops");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile
    graph::partition_t p;
    p.init(part);

    graph::compiled_partition_t cp(p);

    std::vector<const graph::logical_tensor_t *> inputs {&conv_src_lt,
            &conv_weight_lt, &gamma_lt, &beta_lt, &scale_lt, &shift_lt};
    std::vector<const graph::logical_tensor_t *> outputs {&bn_dst_lt};

    p.compile(&cp, inputs, outputs, eng);

    test::vector<float> convbn_dst(8 * 32 * 16 * 16, 0.0);
    graph::tensor_t convbn_dst_ts(bn_dst_lt, eng, convbn_dst.data());

    cp.execute(strm,
            {conv_src_ts, conv_weight_ts, bn_gamma_ts, bn_beta_ts, bn_scale_ts,
                    bn_shift_ts},
            {convbn_dst_ts});
    strm->wait();

    float max_diff = 0;
    for (size_t i = 0; i < bn_dst.size(); ++i) {
        max_diff = std::max(max_diff, std::abs(bn_dst[i] - convbn_dst[i]));
    }
    ASSERT_LT(max_diff, 1e-6f);
}

TEST(Compile, ConvBnSharedInputs) {
    // bn has shared gamma/beta/mean/var
    using dims = graph::dnnl_impl::dims;

    // default engine kind is cpu.
    graph::engine_t *eng = get_engine();
    graph::stream_t *strm = get_stream();

    graph::op_t conv_op(0, graph::op_kind::Convolution, "conv");
    conv_op.set_attr<dims>(graph::op_attr::strides, dims {1, 1});
    conv_op.set_attr<dims>(graph::op_attr::dilations, dims {1, 1});
    conv_op.set_attr<dims>(graph::op_attr::pads_begin, dims {0, 0});
    conv_op.set_attr<dims>(graph::op_attr::pads_end, dims {0, 0});
    conv_op.set_attr<int64_t>(graph::op_attr::groups, 1);
    conv_op.set_attr<std::string>(graph::op_attr::data_format, "NCX");
    conv_op.set_attr<std::string>(graph::op_attr::weights_format, "OIX");

    graph::op_t bn_op(1, graph::op_kind::BatchNormInference, "bn");
    bn_op.set_attr<std::string>(graph::op_attr::data_format, "NCX");
    bn_op.set_attr(graph::op_attr::epsilon, 1e-6f);

    // prepare logical tensor
    graph::logical_tensor_t conv_src_lt = utils::logical_tensor_init(
            0, {8, 32, 16, 16}, graph::data_type::f32);
    graph::logical_tensor_t conv_weight_lt = utils::logical_tensor_init(
            1, {32, 32, 1, 1}, graph::data_type::f32);
    graph::logical_tensor_t conv_dst_lt = utils::logical_tensor_init(
            2, {8, 32, 16, 16}, graph::data_type::f32);
    graph::logical_tensor_t shared_lt
            = utils::logical_tensor_init(3, {32}, graph::data_type::f32);
    graph::logical_tensor_t bn_dst_lt = utils::logical_tensor_init(
            7, {8, 32, 16, 16}, graph::data_type::f32);

    test::vector<float> conv_src(8 * 32 * 16 * 16);
    test::vector<float> conv_weight(32 * 32 * 1 * 1);
    test::vector<float> bn_shared_input(32);
    test::vector<float> bn_dst(8 * 32 * 16 * 16);

    // Initialize
    std::default_random_engine generator;
    std::normal_distribution<float> distribution(0.0f, 0.1f);

    std::generate(conv_src.begin(), conv_src.end(),
            [&]() { return distribution(generator); });
    std::generate(conv_weight.begin(), conv_weight.end(),
            [&]() { return distribution(generator); });
    std::generate(bn_shared_input.begin(), bn_shared_input.end(),
            [&]() { return distribution(generator); });

    graph::tensor_t conv_src_ts(conv_src_lt, eng, conv_src.data());
    graph::tensor_t conv_weight_ts(conv_weight_lt, eng, conv_weight.data());
    graph::tensor_t bn_shared_input_ts(shared_lt, eng, bn_shared_input.data());
    graph::tensor_t bn_dst_ts(bn_dst_lt, eng, bn_dst.data());

    conv_op.add_input(conv_src_lt);
    conv_op.add_input(conv_weight_lt);
    conv_op.add_output(conv_dst_lt);
    bn_op.add_input(conv_dst_lt);
    bn_op.add_input(shared_lt);
    bn_op.add_input(shared_lt);
    bn_op.add_input(shared_lt);
    bn_op.add_input(shared_lt);
    bn_op.add_output(bn_dst_lt);

    graph::graph_t g(eng->kind());
    g.add_op(&conv_op);
    g.add_op(&bn_op);
    g.finalize();

    // run unfused graph to compute the reference
    ASSERT_EQ(run_graph(g,
                      {conv_src_ts, conv_weight_ts, bn_shared_input_ts,
                              bn_shared_input_ts, bn_shared_input_ts,
                              bn_shared_input_ts},
                      {bn_dst_ts}, *eng, *strm),
            graph::status::success);

    // run fusion partition
    graph::pass::pass_base_ptr apass = get_pass("fp_conv_post_ops");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile
    graph::partition_t p;
    p.init(part);

    graph::compiled_partition_t cp(p);

    std::vector<const graph::logical_tensor_t *> inputs {&conv_src_lt,
            &conv_weight_lt, &shared_lt, &shared_lt, &shared_lt, &shared_lt};
    std::vector<const graph::logical_tensor_t *> outputs {&bn_dst_lt};

    p.compile(&cp, inputs, outputs, eng);

    test::vector<float> convbn_dst(8 * 32 * 16 * 16, 0.0);
    graph::tensor_t convbn_dst_ts(bn_dst_lt, eng, convbn_dst.data());

    cp.execute(strm,
            {conv_src_ts, conv_weight_ts, bn_shared_input_ts,
                    bn_shared_input_ts, bn_shared_input_ts, bn_shared_input_ts},
            {convbn_dst_ts});
    strm->wait();
    ASSERT_TRUE(allclose(bn_dst, convbn_dst, /*rtol*/ 0.1f, 1e-6f));
}

TEST(Execute, ConvAdd) {
    using dims = graph::dnnl_impl::dims;

    // default engine kind is cpu.
    graph::engine_t *eng = get_engine();
    test::vector<float> src {-3.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0,
            2.5, -1.0, 0, 3.0, -2.0, -1.0, 4.0};
    test::vector<float> weight {1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0};
    test::vector<float> post_src {1.0, 2.0, 3.0, 4.0};
    test::vector<float> ref_dst {0.0, 4.5, 8.0, 5.5};
    test::vector<float> dst {0.0, 0.0, 0.0, 0.0};

    std::vector<bool> swaps {false, true};

    for (auto swap : swaps) {
        graph::op_t conv_op(1, graph::op_kind::Convolution, "Convolution");
        conv_op.set_attr<dims>(graph::op_attr::strides, dims {1, 1});
        conv_op.set_attr<dims>(graph::op_attr::dilations, dims {1, 1});
        conv_op.set_attr<dims>(graph::op_attr::pads_begin, dims {0, 0});
        conv_op.set_attr<dims>(graph::op_attr::pads_end, dims {0, 0});
        conv_op.set_attr<int64_t>(graph::op_attr::groups, 1);
        conv_op.set_attr<std::string>(graph::op_attr::data_format, "NCX");
        conv_op.set_attr<std::string>(graph::op_attr::weights_format, "OIX");
        graph::op_t add_op(2, graph::op_kind::Add, "Add");

        // prepare logical tensor
        graph::logical_tensor_t src_lt = utils::logical_tensor_init(
                0, {1, 1, 4, 4}, graph::data_type::f32);
        graph::logical_tensor_t weight_lt = utils::logical_tensor_init(
                1, {1, 1, 3, 3}, graph::data_type::f32);
        graph::logical_tensor_t post_src_lt = utils::logical_tensor_init(
                2, {1, 1, 2, 2}, graph::data_type::f32);
        graph::logical_tensor_t dst_lt = utils::logical_tensor_init(
                3, {1, 1, 2, 2}, graph::data_type::f32);
        graph::logical_tensor_t add_dst_lt = utils::logical_tensor_init(
                4, {1, 1, 2, 2}, graph::data_type::f32);

        graph::op_t in_op(0, graph::op_kind::Wildcard, "Wildcard");

        in_op.add_output(post_src_lt);
        conv_op.add_input(src_lt);
        conv_op.add_input(weight_lt);
        conv_op.add_output(dst_lt);
        if (swap) {
            add_op.add_input(post_src_lt);
            add_op.add_input(dst_lt);
        } else {
            add_op.add_input(dst_lt);
            add_op.add_input(post_src_lt);
        }
        add_op.add_output(add_dst_lt);

        graph::graph_t g(eng->kind());
        g.add_op(&in_op);
        g.add_op(&conv_op);
        g.add_op(&add_op);
        g.finalize();

        graph::pass::pass_base_ptr apass = get_pass("fp_conv_post_ops");
        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 1U);
        auto part = g.get_partitions()[0];

        // compile
        graph::partition_t p;
        p.init(part);

        graph::compiled_partition_t cp(p);

        std::vector<const graph::logical_tensor_t *> inputs {
                &src_lt, &weight_lt, &post_src_lt};
        std::vector<const graph::logical_tensor_t *> outputs {&add_dst_lt};

        p.compile(&cp, inputs, outputs, eng);

        graph::logical_tensor_t lt;
        cp.query_logical_tensor(add_dst_lt.id, &lt);
        ASSERT_EQ(lt.layout_type, graph::layout_type::strided);

        graph::tensor_t src_ts(src_lt, eng, src.data());
        graph::tensor_t weight_ts(weight_lt, eng, weight.data());
        graph::tensor_t post_src_ts(post_src_lt, eng, post_src.data());
        graph::tensor_t add_dst_ts(add_dst_lt, eng, dst.data());

        graph::stream_t *strm = get_stream();
        cp.execute(strm, {src_ts, weight_ts, post_src_ts}, {add_dst_ts});
        strm->wait();
        for (size_t i = 0; i < dst.size(); ++i) {
            ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
        }
    }
}

TEST(Execute, ConvAddPerTensorBroadcast) {
    using dims = graph::dnnl_impl::dims;

    // default engine kind is cpu.
    graph::engine_t *eng = get_engine();
    test::vector<float> src {-3.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0,
            2.5, -1.0, 0, 3.0, -2.0, -1.0, 4.0};
    test::vector<float> weight {1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0};
    test::vector<float> post_src {3.0};
    test::vector<float> ref_dst {2.0, 5.5, 8.0, 4.5};
    test::vector<float> dst {0.0, 0.0, 0.0, 0.0};
    graph::op_t conv_op(1, graph::op_kind::Convolution, "Convolution");
    conv_op.set_attr<dims>(graph::op_attr::strides, {1, 1});
    conv_op.set_attr<dims>(graph::op_attr::dilations, {1, 1});
    conv_op.set_attr<dims>(graph::op_attr::pads_begin, {0, 0});
    conv_op.set_attr<dims>(graph::op_attr::pads_end, {0, 0});
    conv_op.set_attr<int64_t>(graph::op_attr::groups, 1);
    conv_op.set_attr<std::string>(graph::op_attr::data_format, "NCX");
    conv_op.set_attr<std::string>(graph::op_attr::weights_format, "OIX");
    graph::op_t add_op(2, graph::op_kind::Add, "Add");

    // prepare logical tensor
    graph::logical_tensor_t src_lt = utils::logical_tensor_init(
            0, {1, 1, 4, 4}, graph::data_type::f32);
    graph::logical_tensor_t weight_lt = utils::logical_tensor_init(
            1, {1, 1, 3, 3}, graph::data_type::f32);
    // post_src will first be unsequeeze to {1,1,1,1} and then broadcast
    // to {1,1,2,2}
    graph::logical_tensor_t post_src_lt
            = utils::logical_tensor_init(2, {1}, graph::data_type::f32);
    graph::logical_tensor_t dst_lt = utils::logical_tensor_init(
            3, {1, 1, 2, 2}, graph::data_type::f32);
    graph::logical_tensor_t add_dst_lt = utils::logical_tensor_init(
            4, {1, 1, 2, 2}, graph::data_type::f32);

    graph::op_t in_op(0, graph::op_kind::Wildcard, "Wildcard");

    in_op.add_output(post_src_lt);
    conv_op.add_input(src_lt);
    conv_op.add_input(weight_lt);
    conv_op.add_output(dst_lt);
    add_op.add_input(dst_lt);
    add_op.add_input(post_src_lt);
    add_op.add_output(add_dst_lt);

    graph::graph_t g(eng->kind());
    g.add_op(&in_op);
    g.add_op(&conv_op);
    g.add_op(&add_op);
    g.finalize();

    graph::pass::pass_base_ptr apass = get_pass("fp_conv_post_ops");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile
    graph::partition_t p;
    p.init(part);

    graph::compiled_partition_t cp(p);

    std::vector<const graph::logical_tensor_t *> inputs {
            &src_lt, &weight_lt, &post_src_lt};
    std::vector<const graph::logical_tensor_t *> outputs {&add_dst_lt};

    p.compile(&cp, inputs, outputs, eng);
    graph::logical_tensor_t lt;
    cp.query_logical_tensor(add_dst_lt.id, &lt);
    ASSERT_EQ(lt.layout_type, graph::layout_type::strided);

    graph::tensor_t src_ts(src_lt, eng, src.data());
    graph::tensor_t weight_ts(weight_lt, eng, weight.data());
    graph::tensor_t post_src_ts(post_src_lt, eng, post_src.data());
    graph::tensor_t dst_ts(dst_lt, eng, dst.data());

    graph::stream_t *strm = get_stream();
    cp.execute(strm, {src_ts, weight_ts, post_src_ts}, {dst_ts});
    strm->wait();
    for (size_t i = 0; i < dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(Execute, ConvAddExpandedPerTensorBroadcast) {
    using dims = graph::dnnl_impl::dims;

    // default engine kind is cpu.
    graph::engine_t *eng = get_engine();
    test::vector<float> src {-3.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0,
            2.5, -1.0, 0, 3.0, -2.0, -1.0, 4.0};
    test::vector<float> weight {1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0};
    test::vector<float> post_src {3.0};
    test::vector<float> ref_dst {2.0, 5.5, 8.0, 4.5};
    test::vector<float> dst {0.0, 0.0, 0.0, 0.0};
    graph::op_t conv_op(1, graph::op_kind::Convolution, "Convolution");
    conv_op.set_attr<dims>(graph::op_attr::strides, {1, 1});
    conv_op.set_attr<dims>(graph::op_attr::dilations, {1, 1});
    conv_op.set_attr<dims>(graph::op_attr::pads_begin, {0, 0});
    conv_op.set_attr<dims>(graph::op_attr::pads_end, {0, 0});
    conv_op.set_attr<int64_t>(graph::op_attr::groups, 1);
    conv_op.set_attr<std::string>(graph::op_attr::data_format, "NCX");
    conv_op.set_attr<std::string>(graph::op_attr::weights_format, "OIX");
    graph::op_t add_op(2, graph::op_kind::Add, "Add");

    // prepare logical tensor
    graph::logical_tensor_t src_lt = utils::logical_tensor_init(
            0, {1, 1, 4, 4}, graph::data_type::f32);
    graph::logical_tensor_t weight_lt = utils::logical_tensor_init(
            1, {1, 1, 3, 3}, graph::data_type::f32);
    graph::logical_tensor_t post_src_lt = utils::logical_tensor_init(
            2, {1, 1, 1, 1}, graph::data_type::f32);
    graph::logical_tensor_t dst_lt = utils::logical_tensor_init(
            3, {1, 1, 2, 2}, graph::data_type::f32);
    graph::logical_tensor_t add_dst_lt = utils::logical_tensor_init(
            4, {1, 1, 2, 2}, graph::data_type::f32);

    graph::op_t in_op(0, graph::op_kind::Wildcard, "Wildcard");

    in_op.add_output(post_src_lt);
    conv_op.add_input(src_lt);
    conv_op.add_input(weight_lt);
    conv_op.add_output(dst_lt);
    add_op.add_input(dst_lt);
    add_op.add_input(post_src_lt);
    add_op.add_output(add_dst_lt);

    graph::graph_t g(eng->kind());
    g.add_op(&in_op);
    g.add_op(&conv_op);
    g.add_op(&add_op);
    g.finalize();

    graph::pass::pass_base_ptr apass = get_pass("fp_conv_post_ops");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile
    graph::partition_t p;
    p.init(part);

    graph::compiled_partition_t cp(p);

    std::vector<const graph::logical_tensor_t *> inputs {
            &src_lt, &weight_lt, &post_src_lt};
    std::vector<const graph::logical_tensor_t *> outputs {&add_dst_lt};

    p.compile(&cp, inputs, outputs, eng);
    graph::logical_tensor_t lt;
    cp.query_logical_tensor(add_dst_lt.id, &lt);
    ASSERT_EQ(lt.layout_type, graph::layout_type::strided);

    graph::tensor_t src_ts(src_lt, eng, src.data());
    graph::tensor_t weight_ts(weight_lt, eng, weight.data());
    graph::tensor_t post_src_ts(post_src_lt, eng, post_src.data());
    graph::tensor_t dst_ts(dst_lt, eng, dst.data());

    graph::stream_t *strm = get_stream();
    cp.execute(strm, {src_ts, weight_ts, post_src_ts}, {dst_ts});
    strm->wait();
    for (size_t i = 0; i < dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(Execute, ConvAddPerChannelBroadcast) {
    using dims = graph::dnnl_impl::dims;

    // default engine kind is cpu.
    graph::engine_t *eng = get_engine();
    test::vector<float> src {-3.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0,
            2.5, -1.0, 0, 3.0, -2.0, -1.0, 4.0};
    test::vector<float> weight {1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
            1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0};
    test::vector<float> post_src {3.0, 3.0};
    test::vector<float> ref_dst {2.0, 5.5, 8.0, 4.5, 2.0, 5.5, 8.0, 4.5};
    test::vector<float> dst {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    graph::op_t conv_op(1, graph::op_kind::Convolution, "Convolution");
    conv_op.set_attr<dims>(graph::op_attr::strides, {1, 1});
    conv_op.set_attr<dims>(graph::op_attr::dilations, {1, 1});
    conv_op.set_attr<dims>(graph::op_attr::pads_begin, {0, 0});
    conv_op.set_attr<dims>(graph::op_attr::pads_end, {0, 0});
    conv_op.set_attr<int64_t>(graph::op_attr::groups, 1);
    conv_op.set_attr<std::string>(graph::op_attr::data_format, "NCX");
    conv_op.set_attr<std::string>(graph::op_attr::weights_format, "OIX");
    graph::op_t add_op(2, graph::op_kind::Add, "Add");

    // prepare logical tensor
    graph::logical_tensor_t src_lt = utils::logical_tensor_init(
            0, {1, 1, 4, 4}, graph::data_type::f32);
    graph::logical_tensor_t weight_lt = utils::logical_tensor_init(
            1, {2, 1, 3, 3}, graph::data_type::f32);
    graph::logical_tensor_t post_src_lt = utils::logical_tensor_init(
            2, {1, 2, 1, 1}, graph::data_type::f32);
    graph::logical_tensor_t dst_lt = utils::logical_tensor_init(
            3, {1, 2, 2, 2}, graph::data_type::f32);
    graph::logical_tensor_t add_dst_lt = utils::logical_tensor_init(
            4, {1, 2, 2, 2}, graph::data_type::f32);

    graph::op_t in_op(0, graph::op_kind::Wildcard, "Wildcard");

    in_op.add_output(post_src_lt);
    conv_op.add_input(src_lt);
    conv_op.add_input(weight_lt);
    conv_op.add_output(dst_lt);
    add_op.add_input(dst_lt);
    add_op.add_input(post_src_lt);
    add_op.add_output(add_dst_lt);

    graph::graph_t g(eng->kind());
    g.add_op(&in_op);
    g.add_op(&conv_op);
    g.add_op(&add_op);
    g.finalize();

    graph::pass::pass_base_ptr apass = get_pass("fp_conv_post_ops");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile
    graph::partition_t p;
    p.init(part);

    graph::compiled_partition_t cp(p);

    std::vector<const graph::logical_tensor_t *> inputs {
            &src_lt, &weight_lt, &post_src_lt};
    std::vector<const graph::logical_tensor_t *> outputs {&add_dst_lt};

    p.compile(&cp, inputs, outputs, eng);
    graph::logical_tensor_t lt;
    cp.query_logical_tensor(add_dst_lt.id, &lt);
    ASSERT_EQ(lt.layout_type, graph::layout_type::strided);

    graph::tensor_t src_ts(src_lt, eng, src.data());
    graph::tensor_t weight_ts(weight_lt, eng, weight.data());
    graph::tensor_t post_src_ts(post_src_lt, eng, post_src.data());
    graph::tensor_t dst_ts(dst_lt, eng, dst.data());

    graph::stream_t *strm = get_stream();
    cp.execute(strm, {src_ts, weight_ts, post_src_ts}, {dst_ts});
    strm->wait();
    for (size_t i = 0; i < dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(Execute, ConvAddPerChannelBroadcastNxc) {
    using dims = graph::dnnl_impl::dims;

    // default engine kind is cpu.
    graph::engine_t *eng = get_engine();
    test::vector<float> src {-3.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0,
            2.5, -1.0, 0, 3.0, -2.0, -1.0, 4.0};
    test::vector<float> weight {1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0,
            1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0};
    test::vector<float> post_src {3.0, 3.0};
    test::vector<float> ref_dst {2.0, 2.0, 5.5, 5.5, 8.0, 8.0, 4.5, 4.5};
    test::vector<float> dst {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    graph::op_t conv_op(1, graph::op_kind::Convolution, "Convolution");
    conv_op.set_attr<dims>(graph::op_attr::strides, {1, 1});
    conv_op.set_attr<dims>(graph::op_attr::dilations, {1, 1});
    conv_op.set_attr<dims>(graph::op_attr::pads_begin, {0, 0});
    conv_op.set_attr<dims>(graph::op_attr::pads_end, {0, 0});
    conv_op.set_attr<int64_t>(graph::op_attr::groups, 1);
    conv_op.set_attr<std::string>(graph::op_attr::data_format, "NXC");
    conv_op.set_attr<std::string>(graph::op_attr::weights_format, "XIO");
    graph::op_t add_op(2, graph::op_kind::Add, "Add");

    // prepare logical tensor
    graph::logical_tensor_t src_lt = utils::logical_tensor_init(
            0, {1, 4, 4, 1}, graph::data_type::f32);
    graph::logical_tensor_t weight_lt = utils::logical_tensor_init(
            1, {3, 3, 1, 2}, graph::data_type::f32);
    graph::logical_tensor_t post_src_lt = utils::logical_tensor_init(
            2, {1, 1, 1, 2}, graph::data_type::f32);
    graph::logical_tensor_t dst_lt = utils::logical_tensor_init(
            3, {1, 2, 2, 2}, graph::data_type::f32);
    graph::logical_tensor_t add_dst_lt = utils::logical_tensor_init(
            4, {1, 2, 2, 2}, graph::data_type::f32);

    graph::op_t in_op(0, graph::op_kind::Wildcard, "Wildcard");

    in_op.add_output(post_src_lt);
    conv_op.add_input(src_lt);
    conv_op.add_input(weight_lt);
    conv_op.add_output(dst_lt);
    add_op.add_input(dst_lt);
    add_op.add_input(post_src_lt);
    add_op.add_output(add_dst_lt);

    graph::graph_t g(eng->kind());
    g.add_op(&in_op);
    g.add_op(&conv_op);
    g.add_op(&add_op);
    g.finalize();

    graph::pass::pass_base_ptr apass = get_pass("fp_conv_post_ops");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile
    graph::partition_t p;
    p.init(part);

    graph::compiled_partition_t cp(p);

    std::vector<const graph::logical_tensor_t *> inputs {
            &src_lt, &weight_lt, &post_src_lt};
    std::vector<const graph::logical_tensor_t *> outputs {&add_dst_lt};

    p.compile(&cp, inputs, outputs, eng);
    graph::logical_tensor_t lt;
    cp.query_logical_tensor(add_dst_lt.id, &lt);
    ASSERT_EQ(lt.layout_type, graph::layout_type::strided);

    graph::tensor_t src_ts(src_lt, eng, src.data());
    graph::tensor_t weight_ts(weight_lt, eng, weight.data());
    graph::tensor_t post_src_ts(post_src_lt, eng, post_src.data());
    graph::tensor_t dst_ts(dst_lt, eng, dst.data());

    graph::stream_t *strm = get_stream();
    cp.execute(strm, {src_ts, weight_ts, post_src_ts}, {dst_ts});
    strm->wait();
    for (size_t i = 0; i < dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(Compile, ConvAddBroadcast) {
    using dims = graph::dnnl_impl::dims;

    // default engine kind is cpu.
    graph::engine_t *eng = get_engine();
    test::vector<float> src {-3.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0,
            2.5, -1.0, 0, 3.0, -2.0, -1.0, 4.0};
    test::vector<float> weight {1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0};
    test::vector<float> post_src {3.0, 3.0};
    graph::op_t conv_op(1, graph::op_kind::Convolution, "Convolution");
    conv_op.set_attr<dims>(graph::op_attr::strides, {1, 1});
    conv_op.set_attr<dims>(graph::op_attr::dilations, {1, 1});
    conv_op.set_attr<dims>(graph::op_attr::pads_begin, {0, 0});
    conv_op.set_attr<dims>(graph::op_attr::pads_end, {0, 0});
    conv_op.set_attr<int64_t>(graph::op_attr::groups, 1);
    conv_op.set_attr<std::string>(graph::op_attr::data_format, "NCX");
    conv_op.set_attr<std::string>(graph::op_attr::weights_format, "OIX");
    graph::op_t add_op(2, graph::op_kind::Add, "Add");

    // prepare logical tensor
    graph::logical_tensor_t src_lt = utils::logical_tensor_init(
            0, {1, 1, 4, 4}, graph::data_type::f32);
    graph::logical_tensor_t weight_lt = utils::logical_tensor_init(
            1, {1, 1, 3, 3}, graph::data_type::f32);
    graph::logical_tensor_t post_src_lt
            = utils::logical_tensor_init(2, {2}, graph::data_type::f32);
    graph::logical_tensor_t dst_lt = utils::logical_tensor_init(
            3, {1, 1, 2, 2}, graph::data_type::f32);
    graph::logical_tensor_t add_dst_lt = utils::logical_tensor_init(
            4, {1, 1, 2, 2}, graph::data_type::f32);

    graph::op_t in_op(0, graph::op_kind::Wildcard, "Wildcard");

    in_op.add_output(post_src_lt);
    conv_op.add_input(src_lt);
    conv_op.add_input(weight_lt);
    conv_op.add_output(dst_lt);
    add_op.add_input(dst_lt);
    add_op.add_input(post_src_lt);
    add_op.add_output(add_dst_lt);

    graph::graph_t g(eng->kind());
    g.add_op(&in_op);
    g.add_op(&conv_op);
    g.add_op(&add_op);
    g.finalize();

    graph::pass::pass_base_ptr apass = get_pass("fp_conv_post_ops");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile
    graph::partition_t p;
    p.init(part);

    graph::compiled_partition_t cp(p);

    std::vector<const graph::logical_tensor_t *> inputs {
            &src_lt, &weight_lt, &post_src_lt};
    std::vector<const graph::logical_tensor_t *> outputs {&add_dst_lt};

    ASSERT_EQ(p.compile(&cp, inputs, outputs, eng), graph::status::success);
}

TEST(Execute, ConvAddRelu) {
    using dims = graph::dnnl_impl::dims;

    // default engine kind is cpu.
    graph::engine_t *eng = get_engine();
    test::vector<float> src {-3.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0,
            2.5, -1.0, 0, 3.0, -2.0, -1.0, 4.0};
    test::vector<float> weight {1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0};
    test::vector<float> post_src {-1.0, -2.0, -3.0, -4.0};
    test::vector<float> ref_dst {0.0, 0.5, 2.0, 0.0};
    test::vector<float> dst {0.0, 0.0, 0.0, 0.0};
    graph::op_t in_op(0, graph::op_kind::Wildcard, "Wildcard");

    graph::op_t conv_op(1, graph::op_kind::Convolution, "Convolution");

    conv_op.set_attr<dims>(graph::op_attr::strides, {1, 1});
    conv_op.set_attr<dims>(graph::op_attr::dilations, {1, 1});
    conv_op.set_attr<dims>(graph::op_attr::pads_begin, {0, 0});
    conv_op.set_attr<dims>(graph::op_attr::pads_end, {0, 0});
    conv_op.set_attr<int64_t>(graph::op_attr::groups, 1);
    conv_op.set_attr<std::string>(graph::op_attr::data_format, "NCX");
    conv_op.set_attr<std::string>(graph::op_attr::weights_format, "OIX");
    graph::op_t add_op(2, graph::op_kind::Add, "Add");
    graph::op_t relu_op(3, graph::op_kind::ReLU, "ReLU");

    // prepare logical tensor
    graph::logical_tensor_t src_lt = utils::logical_tensor_init(
            0, {1, 1, 4, 4}, graph::data_type::f32);
    graph::logical_tensor_t weight_lt = utils::logical_tensor_init(
            1, {1, 1, 3, 3}, graph::data_type::f32);
    graph::logical_tensor_t post_lt = utils::logical_tensor_init(
            2, {1, 1, 2, 2}, graph::data_type::f32);
    graph::logical_tensor_t dst_lt = utils::logical_tensor_init(
            3, {1, 1, 2, 2}, graph::data_type::f32);
    graph::logical_tensor_t add_dst_lt = utils::logical_tensor_init(
            4, {1, 1, 2, 2}, graph::data_type::f32);
    graph::logical_tensor_t relu_dst_lt = utils::logical_tensor_init(
            5, {1, 1, 2, 2}, graph::data_type::f32);

    in_op.add_output(post_lt);
    conv_op.add_input(src_lt);
    conv_op.add_input(weight_lt);
    conv_op.add_output(dst_lt);
    add_op.add_input(dst_lt);
    add_op.add_input(post_lt);
    add_op.add_output(add_dst_lt);
    relu_op.add_input(add_dst_lt);
    relu_op.add_output(relu_dst_lt);

    graph::graph_t g(eng->kind());
    g.add_op(&in_op);
    g.add_op(&conv_op);
    g.add_op(&add_op);
    g.add_op(&relu_op);
    g.finalize();

    graph::pass::pass_base_ptr apass = get_pass("fp_conv_post_ops");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile
    graph::partition_t p;
    p.init(part);

    graph::compiled_partition_t cp(p);

    std::vector<const graph::logical_tensor_t *> inputs {
            &src_lt, &weight_lt, &post_lt};
    std::vector<const graph::logical_tensor_t *> outputs {&relu_dst_lt};

    p.compile(&cp, inputs, outputs, eng);

    graph::logical_tensor_t lt;
    cp.query_logical_tensor(relu_dst_lt.id, &lt);
    ASSERT_EQ(lt.layout_type, graph::layout_type::strided);

    graph::tensor_t src_ts(src_lt, eng, src.data());
    graph::tensor_t weight_ts(weight_lt, eng, weight.data());
    graph::tensor_t post_src_ts(post_lt, eng, post_src.data());
    graph::tensor_t relu_dst_ts(relu_dst_lt, eng, dst.data());

    graph::stream_t *strm = get_stream();
    cp.execute(strm, {src_ts, weight_ts, post_src_ts}, {relu_dst_ts});
    strm->wait();
    for (size_t i = 0; i < dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(Execute, ConvMultiplePostOps) {
    using dims = graph::dnnl_impl::dims;

    // default engine kind is cpu.
    graph::engine_t *eng = get_engine();
    test::vector<float> src {-3.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0,
            2.5, -1.0, 0, 3.0, -2.0, -1.0, 4.0};
    test::vector<float> weight {1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0};
    test::vector<float> bias {1.0};
    test::vector<float> mul_other {2.0, 2.0, 2.0, 2.0};
    test::vector<float> sum_other {-1.0, -2.0, -3.0, -4.0};
    test::vector<float> add_other {1.0};
    test::vector<float> ref_dst {0.0, 6.0, 10.0, 2.0};
    test::vector<float> dst {0.0, 0.0, 0.0, 0.0};

    graph::op_t conv_op(1, graph::op_kind::Convolution, "Convolution");
    conv_op.set_attr<dims>(graph::op_attr::strides, {1, 1});
    conv_op.set_attr<dims>(graph::op_attr::dilations, {1, 1});
    conv_op.set_attr<dims>(graph::op_attr::pads_begin, {0, 0});
    conv_op.set_attr<dims>(graph::op_attr::pads_end, {0, 0});
    conv_op.set_attr<int64_t>(graph::op_attr::groups, 1);
    std::string data_format = "NXC";
    std::string filter_format = "XIO";
    conv_op.set_attr<std::string>(graph::op_attr::data_format, data_format);
    conv_op.set_attr<std::string>(
            graph::op_attr::weights_format, filter_format);

    graph::op_t mul_op(2, graph::op_kind::Multiply, "Mul");
    graph::op_t sum_op(3, graph::op_kind::Add, "Sum");
    graph::op_t add_op(4, graph::op_kind::Add, "Add");

    std::vector<int64_t> src_dims {1, 1, 4, 4};
    std::vector<int64_t> weight_dims {1, 1, 3, 3};
    std::vector<int64_t> dst_dims {1, 1, 2, 2};
    if (data_format == "NXC") {
        src_dims = {1, 4, 4, 1};
        dst_dims = {1, 2, 2, 1};
    }
    if (filter_format == "XIO") weight_dims = {3, 3, 1, 1};

    // prepare logical tensor
    graph::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, src_dims, graph::data_type::f32);
    graph::logical_tensor_t weight_lt
            = utils::logical_tensor_init(1, weight_dims, graph::data_type::f32);
    graph::logical_tensor_t bias_lt
            = utils::logical_tensor_init(2, {1}, graph::data_type::f32);
    graph::logical_tensor_t mul_other_lt
            = utils::logical_tensor_init(3, dst_dims, graph::data_type::f32);
    graph::logical_tensor_t mul_dst_lt
            = utils::logical_tensor_init(4, dst_dims, graph::data_type::f32);
    graph::logical_tensor_t sum_other_lt
            = utils::logical_tensor_init(5, dst_dims, graph::data_type::f32);
    graph::logical_tensor_t dst_lt
            = utils::logical_tensor_init(6, dst_dims, graph::data_type::f32);
    graph::logical_tensor_t sum_dst_lt
            = utils::logical_tensor_init(7, dst_dims, graph::data_type::f32);
    graph::logical_tensor_t add_other_lt
            = utils::logical_tensor_init(8, {1}, graph::data_type::f32);
    graph::logical_tensor_t add_dst_lt
            = utils::logical_tensor_init(9, dst_dims, graph::data_type::f32);

    conv_op.add_input(src_lt);
    conv_op.add_input(weight_lt);
    conv_op.add_input(bias_lt);
    conv_op.add_output(dst_lt);
    mul_op.add_input(dst_lt);
    mul_op.add_input(mul_other_lt);
    mul_op.add_output(mul_dst_lt);
    sum_op.add_input(mul_dst_lt);
    sum_op.add_input(sum_other_lt);
    sum_op.add_output(sum_dst_lt);
    add_op.add_input(sum_dst_lt);
    add_op.add_input(add_other_lt);
    add_op.add_output(add_dst_lt);

    graph::graph_t g(eng->kind());
    g.add_op(&conv_op);
    g.add_op(&mul_op);
    g.add_op(&sum_op);
    g.add_op(&add_op);
    g.finalize();

    graph::pass::pass_base_ptr apass = get_pass("fp_conv_post_ops");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile
    graph::partition_t p;
    p.init(part);

    graph::compiled_partition_t cp(p);

    std::vector<const graph::logical_tensor_t *> inputs {&src_lt, &weight_lt,
            &bias_lt, &mul_other_lt, &sum_other_lt, &add_other_lt};
    std::vector<const graph::logical_tensor_t *> outputs {&add_dst_lt};

    ASSERT_EQ(p.compile(&cp, inputs, outputs, eng), graph::status::success);

    std::vector<graph::inplace_pair_t> inplace_pairs = cp.get_inplace_pairs();
    ASSERT_EQ(inplace_pairs.size(), 1U);
    ASSERT_EQ(inplace_pairs[0].input_id, sum_other_lt.id);
    ASSERT_EQ(inplace_pairs[0].output_id, add_dst_lt.id);

    graph::logical_tensor_t lt;
    cp.query_logical_tensor(add_dst_lt.id, &lt);
    ASSERT_EQ(lt.layout_type, graph::layout_type::strided);

    graph::tensor_t src_ts(src_lt, eng, src.data());
    graph::tensor_t weight_ts(weight_lt, eng, weight.data());
    graph::tensor_t bias_ts(bias_lt, eng, bias.data());
    graph::tensor_t mul_other_ts(mul_other_lt, eng, mul_other.data());
    graph::tensor_t sum_other_ts(sum_other_lt, eng, sum_other.data());
    graph::tensor_t add_other_ts(add_other_lt, eng, add_other.data());
    graph::tensor_t add_dst_ts(add_dst_lt, eng, dst.data());

    graph::stream_t *strm = get_stream();
    cp.execute(strm,
            {src_ts, weight_ts, bias_ts, mul_other_ts, sum_other_ts,
                    add_other_ts},
            {add_dst_ts});
    strm->wait();
    for (size_t i = 0; i < dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(Execute, ConvBiasEltwise) {
    using dims = dnnl::impl::graph::dnnl_impl::dims;

    // default engine kind is cpu.
    graph::engine_t *eng = get_engine();
    test::vector<float> src {-3.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0,
            2.5, -1.0, 0.0, 3.0, -2.0, -1.0, 4.0};
    test::vector<float> weight {1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0};
    test::vector<float> dst {0.0, 0.0, 0.0, 0.0};

    std::vector<eltwise_param_t> params1 = {
            eltwise_param_t {"fp_conv_post_ops", {-1.0}, {2.0, 1.5, 4.0, 0.5},
                    graph::op_kind::Abs, "Abs", {}},
            eltwise_param_t {"fp_conv_post_ops", {-1.0},
                    {static_cast<float>(exp(-2) - 1), 1.5, 4.0, 0.5},
                    graph::op_kind::Elu, "Elu", {{graph::op_attr::alpha, 1.f}}},
            eltwise_param_t {"fp_conv_post_ops", {-1.0},
                    hardsigmoid_func({-2.0, 1.5, 4.0, 0.5}, 1.f / 6, 0.5f),
                    graph::op_kind::HardSigmoid, "HardSigmoid",
                    {{graph::op_attr::alpha, 1.f / 6},
                            {graph::op_attr::beta, 0.5f}}},
            eltwise_param_t {"fp_conv_post_ops", {-1.0},
                    {-0.04f, 1.5f, 4.0f, 0.5f}, graph::op_kind::LeakyReLU,
                    "LeakyReLU", {{graph::op_attr::alpha, 0.02f}}},
            eltwise_param_t {"fp_conv_post_ops", {-1.0},
                    mish_func({-2.0, 1.5, 4.0, 0.5}), graph::op_kind::Mish,
                    "Mish", {}},
            eltwise_param_t {"fp_conv_post_ops", {-1.0}, {0.0, 1.5, 3.0, 0.5},
                    graph::op_kind::Clamp, "Clamp",
                    {{graph::op_attr::min, 0.f}, {graph::op_attr::max, 3.f}}},
            eltwise_param_t {"fp_conv_post_ops", {-1.0},
                    sigmoid_func({-2.0, 1.5, 4.0, 0.5}),
                    graph::op_kind::Sigmoid, "Sigmoid", {}},
            eltwise_param_t {"fp_conv_post_ops", {-1.0},
                    {4.0, 2.25, 16.0, 0.25}, graph::op_kind::Square, "Square",
                    {}},
            eltwise_param_t {"fp_conv_post_ops", {-1.0},
                    tanh_func({-2.0, 1.5, 4.0, 0.5}), graph::op_kind::Tanh,
                    "Tanh", {}},
            eltwise_param_t {"fp_conv_post_ops", {1.0},
                    sqrt_func({0.0, 3.5, 6.0, 2.5}), graph::op_kind::Sqrt,
                    "Sqrt", {}},
    };

    for (auto &param : params1) {
        graph::op_t conv_op(1, graph::op_kind::Convolution, "Convolution");
        conv_op.set_attr<dims>(graph::op_attr::strides, {1, 1});
        conv_op.set_attr<dims>(graph::op_attr::dilations, {1, 1});
        conv_op.set_attr<dims>(graph::op_attr::pads_begin, {0, 0});
        conv_op.set_attr<dims>(graph::op_attr::pads_end, {0, 0});
        conv_op.set_attr<int64_t>(graph::op_attr::groups, 1);
        conv_op.set_attr<std::string>(graph::op_attr::data_format, "NCX");
        conv_op.set_attr<std::string>(graph::op_attr::weights_format, "OIX");
        graph::op_t eltwise_op(2, param.op_kind, param.op_name);
        for (auto &attr : param.attrs) {
            eltwise_op.set_attr<float>(attr.first, attr.second);
        }

        // prepare logical tensor
        graph::logical_tensor_t src_lt = utils::logical_tensor_init(
                0, {1, 1, 4, 4}, graph::data_type::f32);
        graph::logical_tensor_t weight_lt = utils::logical_tensor_init(
                1, {1, 1, 3, 3}, graph::data_type::f32);
        graph::logical_tensor_t bias_lt = utils::logical_tensor_init(
                2, dims {1}, graph::data_type::f32);
        graph::logical_tensor_t dst_lt = utils::logical_tensor_init(
                4, {1, 1, 2, 2}, graph::data_type::f32);
        graph::logical_tensor_t eltwise_dst_lt = utils::logical_tensor_init(
                5, {1, 1, 2, 2}, graph::data_type::f32);

        conv_op.add_input(src_lt);
        conv_op.add_input(weight_lt);
        conv_op.add_input(bias_lt);
        conv_op.add_output(dst_lt);
        eltwise_op.add_input(dst_lt);
        eltwise_op.add_output(eltwise_dst_lt);

        graph::graph_t g(eng->kind());
        g.add_op(&conv_op);
        g.add_op(&eltwise_op);
        g.finalize();

        graph::pass::pass_base_ptr apass = get_pass(param.pass_name);
        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 1U);
        auto part = g.get_partitions()[0];

        // compile
        graph::partition_t p;
        p.init(part);

        graph::compiled_partition_t cp(p);

        std::vector<const graph::logical_tensor_t *> inputs {
                &src_lt, &weight_lt, &bias_lt};
        std::vector<const graph::logical_tensor_t *> outputs {&eltwise_dst_lt};

        p.compile(&cp, inputs, outputs, eng);

        graph::logical_tensor_t lt;
        cp.query_logical_tensor(eltwise_dst_lt.id, &lt);
        ASSERT_EQ(lt.layout_type, graph::layout_type::strided);

        graph::tensor_t src_ts(src_lt, eng, src.data());
        graph::tensor_t weight_ts(weight_lt, eng, weight.data());
        graph::tensor_t bias_ts(bias_lt, eng, param.bias.data());
        graph::tensor_t eltwise_dst_ts(eltwise_dst_lt, eng, dst.data());

        graph::stream_t *strm = get_stream();
        cp.execute(strm, {src_ts, weight_ts, bias_ts}, {eltwise_dst_ts});
        strm->wait();
        for (size_t i = 0; i < dst.size(); ++i) {
            ASSERT_FLOAT_EQ(dst[i], param.ref_dst[i]);
        }
    }
}

TEST(Execute, ConvBiasAddEltwise) {
    using dims = dnnl::impl::graph::dnnl_impl::dims;

    // default engine kind is cpu.
    graph::engine_t *eng = get_engine();
    test::vector<float> src {-3.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0,
            2.5, -1.0, 0.0, 3.0, -2.0, -1.0, 4.0};
    test::vector<float> weight {1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0};
    test::vector<float> post_src {-2.0, 1.0, -1.0, 0.0};
    test::vector<float> dst {0.0, 0.0, 0.0, 0.0};

    std::vector<eltwise_param_t> params2 = {
            eltwise_param_t {"fp_conv_post_ops", {-1.0},
                    {static_cast<float>(exp(-4.0) - 1), 2.5, 3.0, 0.5},
                    graph::op_kind::Elu, "Elu", {{graph::op_attr::alpha, 1.f}}},
            eltwise_param_t {"fp_conv_post_ops", {-1.0},
                    {-0.08f, 2.5f, 3.0f, 0.5f}, graph::op_kind::LeakyReLU,
                    "LeakyReLU", {{graph::op_attr::alpha, 0.02f}}},
            eltwise_param_t {"fp_conv_post_ops", {-1.0},
                    mish_func({-4.0f, 2.5f, 3.0f, 0.5f}), graph::op_kind::Mish,
                    "Mish", {}},
            eltwise_param_t {"fp_conv_post_ops", {3.0}, {0.0, 6.f, 6.f, 4.5},
                    graph::op_kind::Clamp, "ReLU6",
                    {{graph::op_attr::min, 0.f}, {graph::op_attr::max, 6.f}}},
    };

    for (auto &param : params2) {
        graph::op_t in_op(0, graph::op_kind::Wildcard, "Wildcard");

        graph::op_t conv_op(1, graph::op_kind::Convolution, "Convolution");

        conv_op.set_attr<dims>(graph::op_attr::strides, {1, 1});
        conv_op.set_attr<dims>(graph::op_attr::dilations, {1, 1});
        conv_op.set_attr<dims>(graph::op_attr::pads_begin, {0, 0});
        conv_op.set_attr<dims>(graph::op_attr::pads_end, {0, 0});
        conv_op.set_attr<int64_t>(graph::op_attr::groups, 1);
        conv_op.set_attr<std::string>(graph::op_attr::data_format, "NCX");
        conv_op.set_attr<std::string>(graph::op_attr::weights_format, "OIX");

        graph::op_t add_op(2, graph::op_kind::Add, "Add");
        graph::op_t eltwise_op(3, param.op_kind, param.op_name);
        for (auto &attr : param.attrs) {
            eltwise_op.set_attr<float>(attr.first, attr.second);
        }

        // prepare logical tensor
        graph::logical_tensor_t src_lt = utils::logical_tensor_init(
                0, {1, 1, 4, 4}, graph::data_type::f32);
        graph::logical_tensor_t weight_lt = utils::logical_tensor_init(
                1, {1, 1, 3, 3}, graph::data_type::f32);
        graph::logical_tensor_t bias_lt = utils::logical_tensor_init(
                2, dims {1}, graph::data_type::f32);
        graph::logical_tensor_t post_lt = utils::logical_tensor_init(
                3, {1, 1, 2, 2}, graph::data_type::f32);
        graph::logical_tensor_t dst_lt = utils::logical_tensor_init(
                4, {1, 1, 2, 2}, graph::data_type::f32);
        graph::logical_tensor_t add_dst_lt = utils::logical_tensor_init(
                5, {1, 1, 2, 2}, graph::data_type::f32);
        graph::logical_tensor_t eltwise_dst_lt = utils::logical_tensor_init(
                6, {1, 1, 2, 2}, graph::data_type::f32);

        in_op.add_output(post_lt);
        conv_op.add_input(src_lt);
        conv_op.add_input(weight_lt);
        conv_op.add_input(bias_lt);
        conv_op.add_output(dst_lt);
        add_op.add_input(dst_lt);
        add_op.add_input(post_lt);
        add_op.add_output(add_dst_lt);
        eltwise_op.add_input(add_dst_lt);
        eltwise_op.add_output(eltwise_dst_lt);

        graph::graph_t g(eng->kind());
        g.add_op(&in_op);
        g.add_op(&conv_op);
        g.add_op(&add_op);
        g.add_op(&eltwise_op);
        g.finalize();

        graph::pass::pass_base_ptr apass = get_pass(param.pass_name);
        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 1U);
        auto part = g.get_partitions()[0];

        // compile
        graph::partition_t p;
        p.init(part);

        graph::compiled_partition_t cp(p);

        std::vector<const graph::logical_tensor_t *> inputs {
                &src_lt, &weight_lt, &bias_lt, &post_lt};
        std::vector<const graph::logical_tensor_t *> outputs {&eltwise_dst_lt};

        p.compile(&cp, inputs, outputs, eng);

        graph::logical_tensor_t lt;
        cp.query_logical_tensor(eltwise_dst_lt.id, &lt);
        ASSERT_EQ(lt.layout_type, graph::layout_type::strided);

        graph::tensor_t src_ts(src_lt, eng, src.data());
        graph::tensor_t weight_ts(weight_lt, eng, weight.data());
        graph::tensor_t bias_ts(bias_lt, eng, param.bias.data());
        graph::tensor_t post_src_ts(post_lt, eng, post_src.data());
        graph::tensor_t eltwise_dst_ts(eltwise_dst_lt, eng, dst.data());

        graph::stream_t *strm = get_stream();
        cp.execute(strm, {src_ts, weight_ts, bias_ts, post_src_ts},
                {eltwise_dst_ts});
        strm->wait();
        for (size_t i = 0; i < dst.size(); ++i) {
            // We noticed mish test has slight accuracy issue on GPU or AArch64
            // CPU or SNB.
            if (eng->kind() == graph::engine_kind::gpu
                    || (eng->kind() == graph::engine_kind::cpu
                            && dnnl_get_effective_cpu_isa()
                                    <= dnnl_cpu_isa_avx)) {
                ASSERT_NEAR(dst[i], param.ref_dst[i], 1e-6);
            } else {
                ASSERT_FLOAT_EQ(dst[i], param.ref_dst[i]);
            }
        }
    }
}

TEST(Execute, ConvAddEltwise) {
    using dims = dnnl::impl::graph::dnnl_impl::dims;

    // default engine kind is cpu.
    graph::engine_t *eng = get_engine();
    test::vector<float> src {-3.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0,
            2.5, -1.0, 0.0, 3.0, -2.0, -1.0, 4.0};
    test::vector<float> weight {1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0};
    test::vector<float> post_src {-2.0, 1.0, -1.0, 0.0};
    test::vector<float> ref_dst {-3.0, 3.5, 4.0, 1.5};
    test::vector<float> dst {0.0, 0.0, 0.0, 0.0};

    std::vector<eltwise_param_t> params = {
            eltwise_param_t {"fp_conv_post_ops", {0.0},
                    {static_cast<float>(exp(-3.0) - 1), 3.5, 4.0, 1.5},
                    graph::op_kind::Elu, "Elu", {{graph::op_attr::alpha, 1.f}}},
            eltwise_param_t {"fp_conv_post_ops", {0.0},
                    {-0.06f, 3.5f, 4.0f, 1.5f}, graph::op_kind::LeakyReLU,
                    "LeakyReLU", {{graph::op_attr::alpha, 0.02f}}},
            eltwise_param_t {"fp_conv_post_ops", {0.0},
                    mish_func({-3.0f, 3.5f, 4.0f, 1.5f}), graph::op_kind::Mish,
                    "Mish", {}},
            eltwise_param_t {"fp_conv_post_ops", {0.0}, {0.0, 3.5, 4.f, 1.5},
                    graph::op_kind::Clamp, "ReLU6",
                    {{graph::op_attr::min, 0.f}, {graph::op_attr::max, 6.f}}},
    };

    for (auto &param : params) {
        graph::op_t in_op(0, graph::op_kind::Wildcard, "Wildcard");
        graph::op_t conv_op(1, graph::op_kind::Convolution, "Convolution");
        conv_op.set_attr<dims>(graph::op_attr::strides, {1, 1});
        conv_op.set_attr<dims>(graph::op_attr::dilations, {1, 1});
        conv_op.set_attr<dims>(graph::op_attr::pads_begin, {0, 0});
        conv_op.set_attr<dims>(graph::op_attr::pads_end, {0, 0});
        conv_op.set_attr<int64_t>(graph::op_attr::groups, 1);
        conv_op.set_attr<std::string>(graph::op_attr::data_format, "NCX");
        conv_op.set_attr<std::string>(graph::op_attr::weights_format, "OIX");
        graph::op_t add_op(2, graph::op_kind::Add, "Add");
        graph::op_t eltwise_op(3, param.op_kind, param.op_name);
        for (auto &attr : param.attrs) {
            eltwise_op.set_attr<float>(attr.first, attr.second);
        }

        // prepare logical tensor
        graph::logical_tensor_t src_lt = utils::logical_tensor_init(
                0, {1, 1, 4, 4}, graph::data_type::f32);
        graph::logical_tensor_t weight_lt = utils::logical_tensor_init(
                1, {1, 1, 3, 3}, graph::data_type::f32);
        graph::logical_tensor_t post_lt = utils::logical_tensor_init(
                2, {1, 1, 2, 2}, graph::data_type::f32);
        graph::logical_tensor_t dst_lt = utils::logical_tensor_init(
                3, {1, 1, 2, 2}, graph::data_type::f32);
        graph::logical_tensor_t add_dst_lt = utils::logical_tensor_init(
                4, {1, 1, 2, 2}, graph::data_type::f32);
        graph::logical_tensor_t eltwise_dst_lt = utils::logical_tensor_init(
                5, {1, 1, 2, 2}, graph::data_type::f32);

        in_op.add_output(post_lt);
        conv_op.add_input(src_lt);
        conv_op.add_input(weight_lt);
        conv_op.add_output(dst_lt);
        add_op.add_input(dst_lt);
        add_op.add_input(post_lt);
        add_op.add_output(add_dst_lt);
        eltwise_op.add_input(add_dst_lt);
        eltwise_op.add_output(eltwise_dst_lt);

        graph::graph_t g(eng->kind());
        g.add_op(&in_op);
        g.add_op(&conv_op);
        g.add_op(&add_op);
        g.add_op(&eltwise_op);
        g.finalize();

        graph::pass::pass_base_ptr apass = get_pass(param.pass_name);
        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 1U);
        auto part = g.get_partitions()[0];

        // compile
        graph::partition_t p;
        p.init(part);

        graph::compiled_partition_t cp(p);

        std::vector<const graph::logical_tensor_t *> inputs {
                &src_lt, &weight_lt, &post_lt};
        std::vector<const graph::logical_tensor_t *> outputs {&eltwise_dst_lt};

        p.compile(&cp, inputs, outputs, eng);

        graph::logical_tensor_t lt;
        cp.query_logical_tensor(eltwise_dst_lt.id, &lt);
        ASSERT_EQ(lt.layout_type, graph::layout_type::strided);

        graph::tensor_t src_ts(src_lt, eng, src.data());
        graph::tensor_t weight_ts(weight_lt, eng, weight.data());
        graph::tensor_t post_src_ts(post_lt, eng, post_src.data());
        graph::tensor_t eltwise_dst_ts(eltwise_dst_lt, eng, dst.data());

        graph::stream_t *strm = get_stream();
        cp.execute(strm, {src_ts, weight_ts, post_src_ts}, {eltwise_dst_ts});
        strm->wait();
        for (size_t i = 0; i < dst.size(); ++i) {
            ASSERT_NEAR(dst[i], param.ref_dst[i], 0.0001f);
        }
    }
}

TEST(ExecuteSubgraphFp32, ConvDepthwise) {
    graph::engine_t *engine = get_engine();
    graph::stream_t *strm = get_stream();

    SKIP_IF(engine->kind() == graph::engine_kind::gpu,
            "Skip for GPU - not supported yet.");

    // N, IC, IH, IW
    std::vector<int64_t> conv_src_shape {4, 4, 4, 4};
    // OC, IC/G, KH, KW
    std::vector<int64_t> conv_wei_shape {4, 4, 1, 1};
    // N, OC, OH, OW
    std::vector<int64_t> conv_dst_shape {4, 4, 4, 4};
    // OC, IC/G, KH, KW
    std::vector<int64_t> dw_wei_shape {4, 1, 3, 3};
    // N, OC, OH, OW
    std::vector<int64_t> dw_dst_shape {4, 4, 2, 2};

    std::string dw_type {"k3s2p1"};

    test::vector<float> src_data(product(conv_src_shape));
    test::vector<float> wei_data(product(conv_wei_shape));
    test::vector<float> dw_wei_data(product(dw_wei_shape));

    std::default_random_engine generator(7);
    std::uniform_real_distribution<float> f32_distribution(0.0f, 1.0f);
    std::generate(src_data.begin(), src_data.end(),
            [&]() { return f32_distribution(generator); });
    std::generate(wei_data.begin(), wei_data.end(),
            [&]() { return f32_distribution(generator); });
    std::generate(dw_wei_data.begin(), dw_wei_data.end(),
            [&]() { return f32_distribution(generator); });

    graph::op_t conv {0, graph::op_kind::Convolution, "conv"};
    utils::set_conv_dw_base_op_attr(conv);

    graph::op_t depthwise {1, graph::op_kind::Convolution, "depthwise"};
    utils::set_conv_dw_post_op_attr(depthwise, dw_type);

    graph::logical_tensor_t conv_src = utils::logical_tensor_init(
            0, conv_src_shape, graph::data_type::f32);
    graph::logical_tensor_t conv_wei = utils::logical_tensor_init(
            1, conv_wei_shape, graph::data_type::f32);
    graph::logical_tensor_t conv_dst = utils::logical_tensor_init(
            2, conv_dst_shape, graph::data_type::f32);

    graph::logical_tensor_t dw_wei = utils::logical_tensor_init(
            3, dw_wei_shape, graph::data_type::f32);
    graph::logical_tensor_t dw_dst = utils::logical_tensor_init(
            4, dw_dst_shape, graph::data_type::f32);

    conv.add_input(conv_src);
    conv.add_input(conv_wei);
    conv.add_output(conv_dst);

    depthwise.add_input(conv_dst);
    depthwise.add_input(dw_wei);
    depthwise.add_output(dw_dst);

    graph::graph_t g(engine->kind());
    g.add_op(&conv);
    g.add_op(&depthwise);
    g.finalize();

    graph::tensor_t conv_src_ts(conv_src, engine, src_data.data());
    graph::tensor_t conv_wei_ts(conv_wei, engine, wei_data.data());
    graph::tensor_t dw_wei_ts(dw_wei, engine, dw_wei_data.data());

    // -------------------------case 1----------------------------------
    test::vector<float> case1_out_data(product(dw_dst_shape));
    graph::tensor_t dw_dst_ts(dw_dst, engine, case1_out_data.data());

    ASSERT_EQ(run_graph(g, {conv_src_ts, conv_wei_ts, dw_wei_ts}, {dw_dst_ts},
                      *engine, *strm),
            graph::status::success);

    // -------------------------case 2----------------------------------
    graph::pass::pass_base_ptr apass = get_pass("fp_conv_depthwise_cpu");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    graph::partition_t p;
    p.init(part);

    graph::compiled_partition_t cp(p);

    std::vector<const graph::logical_tensor_t *> lt_ins {
            &conv_src, &conv_wei, &dw_wei};
    std::vector<const graph::logical_tensor_t *> lt_outs {&dw_dst};

    ASSERT_EQ(p.compile(&cp, lt_ins, lt_outs, engine), graph::status::success);

    test::vector<float> case2_out_data(product(dw_dst_shape));
    graph::tensor_t dw_dst_ts2(dw_dst, engine, case2_out_data.data());

    cp.execute(strm, {conv_src_ts, conv_wei_ts, dw_wei_ts}, {dw_dst_ts2});
    strm->wait();

    for (size_t i = 0; i < case1_out_data.size(); ++i) {
        ASSERT_FLOAT_EQ(case1_out_data[i], case2_out_data[i]);
    }
}

TEST(ExecuteSubgraphInt8, Conv1dConv2dConv3d) {
    using dims = graph::dnnl_impl::dims;

    graph::engine_t *engine = get_engine();
    graph::stream_t *strm = get_stream();

    std::vector<size_t> nds = {1, 2, 3};
    std::vector<int64_t> groups = {1};
    std::vector<bool> with_biases = {true, false};
    std::vector<std::string> weight_qtypes = {"per_tensor", "per_channel"};
    std::vector<std::string> src_qtypes = {"symmetric", "asymmetric"};

    static auto isa = dnnl_get_effective_cpu_isa();

    for_(const auto &nd : nds)
    for_(const auto &g : groups)
    for_(const auto with_bias : with_biases)
    for_(const auto &src_qtype : src_qtypes)
    for (const auto &wei_qtype : weight_qtypes) {
        if (engine->kind() == graph::engine_kind::gpu
                && (src_qtype == "asymmetric" || nd == 1))
            continue;

        // prepare data
        int64_t in_channel = 8, out_channel = 8;
        int64_t kernel_size = 3;
        std::vector<int64_t> src_shape = nd == 1
                ? std::vector<int64_t> {1, in_channel, 12}
                : nd == 2 ? std::vector<int64_t> {1, in_channel, 12, 12}
                          : std::vector<int64_t> {1, in_channel, 12, 12, 12};
        std::vector<int64_t> weight_shape = nd == 1
                ? std::vector<int64_t> {out_channel, in_channel / g,
                        kernel_size}
                : nd == 2 ? std::vector<int64_t> {out_channel, in_channel / g,
                          kernel_size, kernel_size}
                          : std::vector<int64_t> {out_channel, in_channel / g,
                                  kernel_size, kernel_size, kernel_size};
        std::vector<int64_t> bias_shape {out_channel};
        std::vector<int64_t> dst_shape = nd == 1
                ? std::vector<int64_t> {1, out_channel, 10}
                : nd == 2 ? std::vector<int64_t> {1, out_channel, 10, 10}
                          : std::vector<int64_t> {1, out_channel, 10, 10, 10};

        test::vector<uint8_t> src_u8_data(product(src_shape));
        test::vector<int8_t> weight_s8_data(product(weight_shape));
        size_t bias_size = with_bias ? product(bias_shape) : 0;
        test::vector<float> bias_data(bias_size);
        test::vector<int8_t> case1_out_data(product(dst_shape));
        test::vector<int8_t> case2_out_data(product(dst_shape));

        std::default_random_engine generator(7);
        std::uniform_real_distribution<float> u8_distribution(0.0f, 255.0f);
        std::uniform_real_distribution<float> s8_distribution(-127.0f, 128.0f);
        std::uniform_real_distribution<float> f32_distribution(0.0f, 1.0f);
        std::generate(src_u8_data.begin(), src_u8_data.end(), [&]() {
            return static_cast<uint8_t>(u8_distribution(generator));
        });
        std::generate(weight_s8_data.begin(), weight_s8_data.end(), [&]() {
            return static_cast<int8_t>(s8_distribution(generator));
        });
        if (with_bias) {
            std::generate(bias_data.begin(), bias_data.end(),
                    [&]() { return f32_distribution(generator); });
        }

        float scale_src = 1 / 255.f; // map to 0~255
        float scale_out = 1;
        int64_t zp_src = src_qtype == "symmetric" ? 0 : 128;
        int64_t zp_out = 78;

        size_t scale_size = wei_qtype == "per_tensor" ? 1 : out_channel;
        std::vector<float> scale_wei(scale_size, 1 / 127.f);
        std::vector<int64_t> zp_wei(scale_size, 0);

        graph::op_t dqdata_node(1, graph::op_kind::Dequantize, "dqdata_node");
        SET_Q_DQ_DATA_ATTR(dqdata_node)

        graph::op_t dqweight_node(
                3, graph::op_kind::Dequantize, "dqweight_node");
        SET_Q_DQ_WEIGHT_ATTR(dqweight_node, 0)

        graph::op_t conv_node(4, graph::op_kind::Convolution, "conv_node");
        SET_CONV_ATTR(conv_node, nd)

        graph::op_t qout_node(5, graph::op_kind::Quantize, "qout_node");
        SET_Q_DQ_OUT_ATTR(qout_node)

        graph::logical_tensor_t src_u8 = utils::logical_tensor_init(
                1, src_shape, graph::data_type::u8);
        graph::logical_tensor_t src_f32_dq = utils::logical_tensor_init(
                2, src_shape, graph::data_type::f32);
        graph::logical_tensor_t weight_s8 = utils::logical_tensor_init(
                4, weight_shape, graph::data_type::s8);
        graph::logical_tensor_t weight_f32_dq = utils::logical_tensor_init(
                5, weight_shape, graph::data_type::f32);
        graph::logical_tensor_t dst_f32 = utils::logical_tensor_init(
                7, dst_shape, graph::data_type::f32);
        graph::logical_tensor_t dst_s8 = utils::logical_tensor_init(
                8, dst_shape, graph::data_type::s8);
        graph::logical_tensor_t bias_f32;
        if (with_bias) {
            bias_f32 = utils::logical_tensor_init(
                    6, bias_shape, graph::data_type::f32);
        }

        dqdata_node.add_input(src_u8);
        dqdata_node.add_output(src_f32_dq);

        dqweight_node.add_input(weight_s8);
        dqweight_node.add_output(weight_f32_dq);

        conv_node.add_input(src_f32_dq);
        conv_node.add_input(weight_f32_dq);
        if (with_bias) conv_node.add_input(bias_f32);
        conv_node.add_output(dst_f32);

        qout_node.add_input(dst_f32);
        qout_node.add_output(dst_s8);

        graph::graph_t g(engine->kind());
        g.add_op(&dqdata_node);
        g.add_op(&dqweight_node);
        g.add_op(&conv_node);
        g.add_op(&qout_node);
        g.finalize();

        graph::tensor_t src_u8_ts(src_u8, engine, src_u8_data.data());
        graph::tensor_t weight_s8_ts(weight_s8, engine, weight_s8_data.data());
        graph::tensor_t bias_f32_ts;
        if (with_bias) {
            bias_f32_ts = graph::tensor_t(bias_f32, engine, bias_data.data());
        }
        graph::tensor_t dst_s8_ts(dst_s8, engine, case1_out_data.data());
        graph::tensor_t dst_s8_case2_ts(dst_s8, engine, case2_out_data.data());

        // -------------------------case 1----------------------------------
        ASSERT_EQ(run_graph(g, {src_u8_ts, weight_s8_ts, bias_f32_ts},
                          {dst_s8_ts}, *engine, *strm),
                graph::status::success);

        // -------------------------case 2----------------------------------
        graph::pass::pass_base_ptr apass = get_pass("x8s8x_conv_post_ops");
        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 1U);
        auto part = g.get_partitions()[0];

        // compile
        graph::partition_t p;
        p.init(part);

        graph::compiled_partition_t cp(p);

        std::vector<const graph::logical_tensor_t *> lt_ins;
        if (with_bias)
            lt_ins = {&src_u8, &weight_s8, &bias_f32};
        else
            lt_ins = {&src_u8, &weight_s8};
        std::vector<const graph::logical_tensor_t *> lt_outs {&dst_s8};

        ASSERT_EQ(p.compile(&cp, lt_ins, lt_outs, engine),
                graph::status::success);

        if (with_bias)
            cp.execute(strm, {src_u8_ts, weight_s8_ts, bias_f32_ts},
                    {dst_s8_case2_ts});
        else
            cp.execute(strm, {src_u8_ts, weight_s8_ts}, {dst_s8_case2_ts});
        strm->wait();

        if (engine->kind() == graph::engine_kind::cpu
                && isa < dnnl_cpu_isa_avx512_core_vnni)
            ASSERT_TRUE(allclose(case1_out_data, case2_out_data, /*rtol*/ 0.1f,
                    /*atol*/ 1.f));
        else
            ASSERT_TRUE(allclose(case1_out_data, case2_out_data, /*rtol*/ 0.01f,
                    /*atol*/ 1.f));
    }
}

static inline void quantized_conv2d_eltwise(
        graph::op_kind_t eltwise, const float *alpha, const float *beta) {
    using dims = graph::dnnl_impl::dims;

    graph::engine_t *engine = get_engine();
    graph::stream_t *strm = get_stream();

    std::vector<int64_t> groups = {1, 4};
    std::vector<bool> with_biases = {true, false};
    std::vector<std::string> weight_qtypes = {"per_tensor", "per_channel"};

    static auto isa = dnnl_get_effective_cpu_isa();

    for_(const auto &g : groups)
    for_(const auto with_bias : with_biases)
    for (const auto &wei_qtype : weight_qtypes) {
        // prepare data
        int64_t in_channel = 8, out_channel = 8;
        int64_t kernel_size = 3;
        std::vector<int64_t> src_shape {1, in_channel, 112, 112};
        std::vector<int64_t> weight_shape {
                out_channel, in_channel / g, kernel_size, kernel_size};
        std::vector<int64_t> bias_shape {out_channel};
        std::vector<int64_t> dst_shape {1, out_channel, 110, 110};

        test::vector<uint8_t> src_u8_data(product(src_shape));
        test::vector<int8_t> weight_s8_data(product(weight_shape));
        size_t bias_size = with_bias ? product(bias_shape) : 0;
        test::vector<float> bias_data(bias_size);
        test::vector<int8_t> case1_out_data(product(dst_shape));
        test::vector<int8_t> case2_out_data(product(dst_shape));

        // random generate src, weight and bias data random seed = 7
        std::default_random_engine generator(7);
        std::uniform_real_distribution<float> u8_distribution(0.0f, 255.0f);
        std::uniform_real_distribution<float> s8_distribution(-127.0f, 128.0f);
        std::uniform_real_distribution<float> f32_distribution(0.0f, 1.0f);
        std::generate(src_u8_data.begin(), src_u8_data.end(), [&]() {
            return static_cast<uint8_t>(u8_distribution(generator));
        });
        std::generate(weight_s8_data.begin(), weight_s8_data.end(), [&]() {
            return static_cast<int8_t>(s8_distribution(generator));
        });
        if (with_bias) {
            std::generate(bias_data.begin(), bias_data.end(),
                    [&]() { return f32_distribution(generator); });
        }

        float scale_src = 1 / 255.f; // map to 0~255
        float scale_out = 1;
        int64_t zp_src = 0;
        int64_t zp_out = 78;

        size_t scale_size = wei_qtype == "per_tensor" ? 1 : out_channel;
        std::vector<float> scale_wei(scale_size, 1 / 127.f);
        std::vector<int64_t> zp_wei(scale_size, 0);

        graph::op_t dqdata_node(1, graph::op_kind::Dequantize, "dqdata_node");
        SET_Q_DQ_DATA_ATTR(dqdata_node)

        graph::op_t dqweight_node(
                3, graph::op_kind::Dequantize, "dqweight_node");
        SET_Q_DQ_WEIGHT_ATTR(dqweight_node, 0)

        graph::op_t conv_node(4, graph::op_kind::Convolution, "conv_node");
        SET_CONV_ATTR(conv_node, 2)

        graph::op_t eltwise_node(5, eltwise, "eltwise_node");
        if (alpha) eltwise_node.set_attr<float>(graph::op_attr::alpha, *alpha);
        if (beta) eltwise_node.set_attr<float>(graph::op_attr::beta, *beta);

        graph::op_t qout_node(6, graph::op_kind::Quantize, "qout_node");
        SET_Q_DQ_OUT_ATTR(qout_node)

        // prepare logical tensor
        graph::logical_tensor_t src_u8 = utils::logical_tensor_init(
                1, src_shape, graph::data_type::u8);
        graph::logical_tensor_t src_f32_dq = utils::logical_tensor_init(
                2, src_shape, graph::data_type::f32);
        graph::logical_tensor_t weight_s8 = utils::logical_tensor_init(
                4, weight_shape, graph::data_type::s8);
        graph::logical_tensor_t weight_f32_dq = utils::logical_tensor_init(
                5, weight_shape, graph::data_type::f32);
        graph::logical_tensor_t dst_f32 = utils::logical_tensor_init(
                7, dst_shape, graph::data_type::f32);
        graph::logical_tensor_t dst_relu_f32 = utils::logical_tensor_init(
                8, dst_shape, graph::data_type::f32);
        graph::logical_tensor_t dst_s8 = utils::logical_tensor_init(
                9, dst_shape, graph::data_type::s8);
        graph::logical_tensor_t bias_f32;
        if (with_bias) {
            bias_f32 = utils::logical_tensor_init(
                    6, bias_shape, graph::data_type::f32);
        }

        dqdata_node.add_input(src_u8);
        dqdata_node.add_output(src_f32_dq);

        dqweight_node.add_input(weight_s8);
        dqweight_node.add_output(weight_f32_dq);

        conv_node.add_input(src_f32_dq);
        conv_node.add_input(weight_f32_dq);
        if (with_bias) conv_node.add_input(bias_f32);
        conv_node.add_output(dst_f32);

        eltwise_node.add_input(dst_f32);
        eltwise_node.add_output(dst_relu_f32);

        qout_node.add_input(dst_relu_f32);
        qout_node.add_output(dst_s8);

        graph::graph_t g(engine->kind());
        g.add_op(&dqdata_node);
        g.add_op(&dqweight_node);
        g.add_op(&conv_node);
        g.add_op(&eltwise_node);
        g.add_op(&qout_node);
        g.finalize();

        graph::tensor_t src_u8_ts(src_u8, engine, src_u8_data.data());
        graph::tensor_t weight_s8_ts(weight_s8, engine, weight_s8_data.data());
        graph::tensor_t bias_f32_ts;
        if (with_bias) {
            bias_f32_ts = graph::tensor_t(bias_f32, engine, bias_data.data());
        }
        graph::tensor_t dst_s8_ts(dst_s8, engine, case1_out_data.data());
        graph::tensor_t dst_s8_case2_ts(dst_s8, engine, case2_out_data.data());

        // -------------------------case 1----------------------------------
        ASSERT_EQ(run_graph(g, {src_u8_ts, weight_s8_ts, bias_f32_ts},
                          {dst_s8_ts}, *engine, *strm),
                graph::status::success);

        // -------------------------case 2----------------------------------
        graph::pass::pass_base_ptr apass = get_pass("x8s8x_conv_post_ops");
        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 1U);
        auto part = g.get_partitions()[0];

        // compile
        graph::partition_t p;
        p.init(part);

        graph::compiled_partition_t cp(p);

        std::vector<const graph::logical_tensor_t *> lt_ins;
        if (with_bias)
            lt_ins = {&src_u8, &weight_s8, &bias_f32};
        else
            lt_ins = {&src_u8, &weight_s8};
        std::vector<const graph::logical_tensor_t *> lt_outs {&dst_s8};

        p.compile(&cp, lt_ins, lt_outs, engine);

        if (with_bias)
            cp.execute(strm, {src_u8_ts, weight_s8_ts, bias_f32_ts},
                    {dst_s8_case2_ts});
        else
            cp.execute(strm, {src_u8_ts, weight_s8_ts}, {dst_s8_case2_ts});
        strm->wait();

        if (engine->kind() == graph::engine_kind::cpu
                && isa < dnnl_cpu_isa_avx512_core_vnni)
            ASSERT_TRUE(allclose(case1_out_data, case2_out_data, /*rtol*/ 0.1f,
                    /*atol*/ 1.f));
        else
            ASSERT_TRUE(allclose(case1_out_data, case2_out_data, /*rtol*/ 0.01f,
                    /*atol*/ 1.f));
    }
}

TEST(ExecuteSubgraphInt8, Conv2dRelu) {
    const graph::op_kind_t opk = graph::op_kind::ReLU;
    quantized_conv2d_eltwise(opk, nullptr, nullptr);
}

TEST(ExecuteSubgraphInt8, Conv2dLeakyRelu) {
    const graph::op_kind_t opk = graph::op_kind::LeakyReLU;
    const float alpha = 0.02f;
    quantized_conv2d_eltwise(opk, &alpha, nullptr);
}

TEST(ExecuteSubgraphInt8, Conv2dMish) {
    const graph::op_kind_t opk = graph::op_kind::Mish;
    quantized_conv2d_eltwise(opk, nullptr, nullptr);
}

TEST(ExecuteSubgraphInt8, Conv2dSumRelu) {
    using dims = graph::dnnl_impl::dims;

    graph::engine_t *engine = get_engine();
    graph::stream_t *strm = get_stream();

    static auto isa = dnnl_get_effective_cpu_isa();

    std::vector<int64_t> groups = {1, 4};
    std::vector<bool> with_biases = {true, false};
    // swap add's two inputs
    std::vector<bool> swaps = {true, false};
    std::vector<std::string> weight_qtypes = {"per_tensor", "per_channel"};
    std::vector<std::string> other_qtypes = {"symmetric", "asymmetric"};

    for_(const auto &g : groups)
    for_(const auto with_bias : with_biases)
    for_(const auto swap : swaps)
    for_(const auto &other_qtype : other_qtypes)
    for (const auto &wei_qtype : weight_qtypes) {
        // skip on gpu for unsupported per_channel group conv
        if (engine->kind() == graph::engine_kind::gpu
                && wei_qtype == "per_channel" && g == 4)
            continue;
        // prepare data
        int64_t in_channel = 8, out_channel = 8;
        int64_t kernel_size = 3;
        std::vector<int64_t> src_shape {1, in_channel, 112, 112};
        std::vector<int64_t> weight_shape {
                out_channel, in_channel / g, kernel_size, kernel_size};
        std::vector<int64_t> bias_shape {out_channel};
        std::vector<int64_t> dst_shape {1, out_channel, 110, 110};

        test::vector<uint8_t> src_u8_data(product(src_shape));
        test::vector<int8_t> weight_s8_data(product(weight_shape));
        test::vector<int8_t> other_s8_data(product(dst_shape));
        size_t bias_size = with_bias ? product(bias_shape) : 0;
        test::vector<float> bias_data(bias_size);
        test::vector<int8_t> case1_out_data(product(dst_shape));
        test::vector<int8_t> case2_out_data(product(dst_shape));

        // random generate src, weight and bias data random seed = 7
        std::default_random_engine generator(7);
        std::uniform_real_distribution<float> u8_distribution(0.0f, 255.0f);
        std::uniform_real_distribution<float> s8_distribution(-127.0f, 128.0f);
        std::uniform_real_distribution<float> f32_distribution(0.0f, 1.0f);
        std::generate(src_u8_data.begin(), src_u8_data.end(), [&]() {
            return static_cast<uint8_t>(u8_distribution(generator));
        });
        std::generate(weight_s8_data.begin(), weight_s8_data.end(), [&]() {
            return static_cast<int8_t>(s8_distribution(generator));
        });
        std::generate(other_s8_data.begin(), other_s8_data.end(), [&]() {
            return static_cast<int8_t>(s8_distribution(generator));
        });
        if (with_bias) {
            std::generate(bias_data.begin(), bias_data.end(),
                    [&]() { return f32_distribution(generator); });
        }

        float scale_src = 1 / 255.f; // map to 0~255
        float scale_other = 1 / 127.f;
        float scale_out = 1;
        int64_t zp_src = 0;
        // post-sum didn't support zps on GPU
        int64_t zp_other = other_qtype == "symmetric"
                        || engine->kind() == graph::engine_kind::gpu
                ? 0
                : 128;
        int64_t zp_out = 78;

        size_t scale_size = wei_qtype == "per_tensor" ? 1 : out_channel;

        std::vector<float> scale_wei(scale_size, 1 / 127.f);
        std::vector<int64_t> zp_wei(scale_size, 0);

        graph::op_t dqdata_node(1, graph::op_kind::Dequantize, "dqdata_node");
        SET_Q_DQ_DATA_ATTR(dqdata_node)

        graph::op_t dqweight_node(
                3, graph::op_kind::Dequantize, "dqweight_node");
        SET_Q_DQ_WEIGHT_ATTR(dqweight_node, 0)

        graph::op_t conv_node(4, graph::op_kind::Convolution, "conv_node");
        SET_CONV_ATTR(conv_node, 2)

        graph::op_t relu_node(5, graph::op_kind::ReLU, "relu_node");

        graph::op_t qout_node(6, graph::op_kind::Quantize, "qout_node");
        SET_Q_DQ_OUT_ATTR(qout_node)

        graph::op_t dqother_node(8, graph::op_kind::Dequantize, "dqother_node");
        dqother_node.set_attr<std::string>(graph::op_attr::qtype, "per_tensor");
        dqother_node.set_attr<std::vector<int64_t>>(
                graph::op_attr::zps, {zp_other});
        dqother_node.set_attr<std::vector<float>>(
                graph::op_attr::scales, {scale_other});
        dqother_node.set_attr<int64_t>(graph::op_attr::axis, 0);

        graph::op_t add_node(9, graph::op_kind::Add, "add_node");

        // prepare logical tensor
        auto src_u8 = utils::logical_tensor_init(1, graph::data_type::u8);
        auto src_f32_dq = utils::logical_tensor_init(2, graph::data_type::f32);
        auto weight_s8 = utils::logical_tensor_init(4, graph::data_type::s8);
        auto weight_f32_dq
                = utils::logical_tensor_init(5, graph::data_type::f32);
        auto dst_f32 = utils::logical_tensor_init(7, graph::data_type::f32);
        auto dst_relu_f32
                = utils::logical_tensor_init(8, graph::data_type::f32);
        auto dst_s8 = utils::logical_tensor_init(9, graph::data_type::s8);
        auto other_s8 = utils::logical_tensor_init(11, graph::data_type::s8);
        auto other_f32_dq
                = utils::logical_tensor_init(12, graph::data_type::f32);
        auto dst_add_f32
                = utils::logical_tensor_init(13, graph::data_type::f32);
        graph::logical_tensor_t bias_f32;
        if (with_bias) {
            bias_f32 = utils::logical_tensor_init(6, graph::data_type::f32);
        }

        dqdata_node.add_input(src_u8);
        dqdata_node.add_output(src_f32_dq);

        dqweight_node.add_input(weight_s8);
        dqweight_node.add_output(weight_f32_dq);

        conv_node.add_input(src_f32_dq);
        conv_node.add_input(weight_f32_dq);
        if (with_bias) conv_node.add_input(bias_f32);
        conv_node.add_output(dst_f32);

        dqother_node.add_input(other_s8);
        dqother_node.add_output(other_f32_dq);
        if (swap) {
            add_node.add_input(dst_f32);
            add_node.add_input(other_f32_dq);
        } else {
            add_node.add_input(other_f32_dq);
            add_node.add_input(dst_f32);
        }
        add_node.add_output(dst_add_f32);

        relu_node.add_input(dst_add_f32);
        relu_node.add_output(dst_relu_f32);

        qout_node.add_input(dst_relu_f32);
        qout_node.add_output(dst_s8);

        graph::graph_t g(engine->kind());
        g.add_op(&dqdata_node);
        g.add_op(&dqweight_node);
        g.add_op(&conv_node);
        g.add_op(&dqother_node);
        g.add_op(&add_node);
        g.add_op(&relu_node);
        g.add_op(&qout_node);
        g.finalize();

        // prepare in/out with full shape
        src_u8 = utils::logical_tensor_init(1, src_shape, graph::data_type::u8);
        weight_s8 = utils::logical_tensor_init(
                4, weight_shape, graph::data_type::s8);
        other_s8 = utils::logical_tensor_init(
                11, dst_shape, graph::data_type::s8);
        if (with_bias) {
            bias_f32 = utils::logical_tensor_init(
                    6, bias_shape, graph::data_type::f32);
        }
        dst_s8 = utils::logical_tensor_init(9, dst_shape, graph::data_type::s8);

        graph::tensor_t src_u8_ts(src_u8, engine, src_u8_data.data());
        graph::tensor_t weight_s8_ts(weight_s8, engine, weight_s8_data.data());
        graph::tensor_t other_s8_ts(other_s8, engine, other_s8_data.data());
        graph::tensor_t bias_f32_ts;
        if (with_bias) {
            bias_f32_ts = graph::tensor_t(bias_f32, engine, bias_data.data());
        }
        graph::tensor_t dst_s8_ts(dst_s8, engine, case1_out_data.data());
        graph::tensor_t dst_s8_case2_ts(dst_s8, engine, case2_out_data.data());

        // -------------------------case 1----------------------------------
        ASSERT_EQ(run_graph(g,
                          {src_u8_ts, weight_s8_ts, bias_f32_ts, other_s8_ts},
                          {dst_s8_ts}, *engine, *strm),
                graph::status::success);

        // -------------------------case 2----------------------------------
        graph::pass::pass_base_ptr apass
                = get_pass(engine->kind() == graph::engine_kind::gpu
                                ? "x8s8x8_conv_add_post_ops_gpu"
                                : "x8s8x8_conv_add_post_ops_cpu");

        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 1U);
        auto part = g.get_partitions()[0];

        // compile
        graph::partition_t p;
        p.init(part);

        graph::compiled_partition_t cp(p);

        std::vector<const graph::logical_tensor_t *> lt_ins;
        if (with_bias)
            lt_ins = {&src_u8, &weight_s8, &bias_f32, &other_s8};
        else
            lt_ins = {&src_u8, &weight_s8, &other_s8};
        std::vector<const graph::logical_tensor_t *> lt_outs {&dst_s8};

        p.compile(&cp, lt_ins, lt_outs, engine);

        if (with_bias)
            cp.execute(strm,
                    {src_u8_ts, weight_s8_ts, bias_f32_ts, other_s8_ts},
                    {dst_s8_case2_ts});
        else
            cp.execute(strm, {src_u8_ts, weight_s8_ts, other_s8_ts},
                    {dst_s8_case2_ts});
        strm->wait();

        if (engine->kind() == graph::engine_kind::cpu
                && isa < dnnl_cpu_isa_avx512_core_vnni)
            ASSERT_TRUE(allclose(case1_out_data, case2_out_data, /*rtol*/ 0.1f,
                    /*atol*/ 1.f));
        else
            ASSERT_TRUE(allclose(case1_out_data, case2_out_data, /*rtol*/ 0.01f,
                    /*atol*/ 1.f));
    }
}

TEST(ExecuteSubgraphInt8, Conv2dSumReluWithDifferentSrc1AndDstTypeOnGPU) {
    using dims = graph::dnnl_impl::dims;

    graph::engine_t *engine = get_engine();
    graph::stream_t *strm = get_stream();

    // we will use blocked layout only on gpu devices, so this focus test
    // targets gpu devices
    SKIP_IF(engine->kind() != graph::engine_kind::gpu,
            "this focus test targets gpu devices");

    int64_t g = 1;
    std::string wei_qtype = "per_channel";

    // prepare data
    int64_t in_channel = 32, out_channel = 32;
    int64_t kernel_size = 3;
    std::vector<int64_t> src_shape {1, in_channel, 12, 12};
    std::vector<int64_t> weight_shape {
            out_channel, in_channel, kernel_size, kernel_size};
    std::vector<int64_t> bias_shape {out_channel};
    std::vector<int64_t> dst_shape {1, out_channel, 10, 10};

    test::vector<uint8_t> src_u8_data(product(src_shape));
    test::vector<int8_t> weight_s8_data(product(weight_shape));
    test::vector<int8_t> other_s8_data(product(dst_shape));
    size_t bias_size = product(bias_shape);
    test::vector<float> bias_data(bias_size);
    test::vector<uint8_t> case1_out_data(product(dst_shape));
    test::vector<uint8_t> case2_out_data(product(dst_shape));

    // random generate src, weight and bias data random seed = 7
    std::default_random_engine generator(7);
    std::uniform_real_distribution<float> u8_distribution(0.0f, 255.0f);
    std::uniform_real_distribution<float> s8_distribution(-127.0f, 128.0f);
    std::uniform_real_distribution<float> f32_distribution(0.0f, 1.0f);
    std::generate(src_u8_data.begin(), src_u8_data.end(),
            [&]() { return static_cast<uint8_t>(u8_distribution(generator)); });
    std::generate(weight_s8_data.begin(), weight_s8_data.end(),
            [&]() { return static_cast<int8_t>(s8_distribution(generator)); });
    std::generate(other_s8_data.begin(), other_s8_data.end(),
            [&]() { return static_cast<int8_t>(s8_distribution(generator)); });

    std::generate(bias_data.begin(), bias_data.end(),
            [&]() { return f32_distribution(generator); });

    float scale_src = 1 / 255.f; // map to 0~255
    float scale_other = 1 / 127.f;
    float scale_out = 1;
    int64_t zp_src = 0;
    // post-sum didn't support zps on GPU
    int64_t zp_other = 0;
    // The following cmd will be skiped by benchdnn, since oneDNN didn't
    // support reorder with zps on GPU: "./tests/benchdnn/benchdnn --reorder
    // --engine=gpu --mode=C --sdt=f32 --ddt=s8
    // --attr-zero-points=dst:common:78 --stag=aBc8b --dtag=abc 1x8x10"
    int64_t zp_out = 0;

    size_t scale_size = out_channel;

    std::vector<float> scale_wei(scale_size, 1 / 127.f);
    std::vector<int64_t> zp_wei(scale_size, 0);

    graph::op_t dqdata_node(1, graph::op_kind::Dequantize, "dqdata_node");
    SET_Q_DQ_DATA_ATTR(dqdata_node)

    graph::op_t dqweight_node(3, graph::op_kind::Dequantize, "dqweight_node");
    SET_Q_DQ_WEIGHT_ATTR(dqweight_node, 0)

    graph::op_t conv_node(4, graph::op_kind::Convolution, "conv_node");
    SET_CONV_ATTR(conv_node, 2)

    graph::op_t relu_node(5, graph::op_kind::ReLU, "relu_node");

    graph::op_t qout_node(6, graph::op_kind::Quantize, "qout_node");
    SET_Q_DQ_OUT_ATTR(qout_node)

    graph::op_t dqother_node(8, graph::op_kind::Dequantize, "dqother_node");
    dqother_node.set_attr<std::string>(graph::op_attr::qtype, "per_tensor");
    dqother_node.set_attr<std::vector<int64_t>>(
            graph::op_attr::zps, {zp_other});
    dqother_node.set_attr<std::vector<float>>(
            graph::op_attr::scales, {scale_other});
    dqother_node.set_attr<int64_t>(graph::op_attr::axis, 0);

    graph::op_t add_node(9, graph::op_kind::Add, "add_node");

    // prepare logical tensor
    auto src_u8 = utils::logical_tensor_init(1, graph::data_type::u8);
    auto src_f32_dq = utils::logical_tensor_init(2, graph::data_type::f32);
    auto weight_s8 = utils::logical_tensor_init(4, graph::data_type::s8);
    auto weight_f32_dq = utils::logical_tensor_init(5, graph::data_type::f32);
    auto dst_f32 = utils::logical_tensor_init(7, graph::data_type::f32);
    auto dst_relu_f32 = utils::logical_tensor_init(8, graph::data_type::f32);
    auto dst_u8 = utils::logical_tensor_init(9, graph::data_type::u8);
    auto other_s8 = utils::logical_tensor_init(11, graph::data_type::s8);
    auto other_f32_dq = utils::logical_tensor_init(12, graph::data_type::f32);
    auto dst_add_f32 = utils::logical_tensor_init(13, graph::data_type::f32);
    graph::logical_tensor_t bias_f32
            = utils::logical_tensor_init(6, graph::data_type::f32);

    dqdata_node.add_input(src_u8);
    dqdata_node.add_output(src_f32_dq);

    dqweight_node.add_input(weight_s8);
    dqweight_node.add_output(weight_f32_dq);

    conv_node.add_input(src_f32_dq);
    conv_node.add_input(weight_f32_dq);
    conv_node.add_input(bias_f32);
    conv_node.add_output(dst_f32);

    dqother_node.add_input(other_s8);
    dqother_node.add_output(other_f32_dq);

    add_node.add_input(dst_f32);
    add_node.add_input(other_f32_dq);

    add_node.add_output(dst_add_f32);

    relu_node.add_input(dst_add_f32);
    relu_node.add_output(dst_relu_f32);

    qout_node.add_input(dst_relu_f32);
    qout_node.add_output(dst_u8);

    graph::graph_t agraph(engine->kind());
    agraph.add_op(&dqdata_node);
    agraph.add_op(&dqweight_node);
    agraph.add_op(&conv_node);
    agraph.add_op(&dqother_node);
    agraph.add_op(&add_node);
    agraph.add_op(&relu_node);
    agraph.add_op(&qout_node);
    agraph.finalize();

    // prepare in/out with full shape
    src_u8 = utils::logical_tensor_init(1, src_shape, graph::data_type::u8);
    weight_s8
            = utils::logical_tensor_init(4, weight_shape, graph::data_type::s8);
    other_s8 = utils::logical_tensor_init(11, dst_shape, graph::data_type::s8);
    bias_f32 = utils::logical_tensor_init(6, bias_shape, graph::data_type::f32);
    dst_u8 = utils::logical_tensor_init(9, dst_shape, graph::data_type::u8);

    graph::tensor_t src_u8_ts(src_u8, engine, src_u8_data.data());
    graph::tensor_t weight_s8_ts(weight_s8, engine, weight_s8_data.data());
    graph::tensor_t other_s8_ts(other_s8, engine, other_s8_data.data());
    graph::tensor_t bias_f32_ts
            = graph::tensor_t(bias_f32, engine, bias_data.data());
    graph::tensor_t dst_u8_ts(dst_u8, engine, case1_out_data.data());
    graph::tensor_t dst_u8_case2_ts(dst_u8, engine, case2_out_data.data());

    // -------------------------case 1----------------------------------
    ASSERT_EQ(run_graph(agraph,
                      {src_u8_ts, weight_s8_ts, bias_f32_ts, other_s8_ts},
                      {dst_u8_ts}, *engine, *strm),
            graph::status::success);

    // -------------------------case 2----------------------------------
    graph::pass::pass_base_ptr apass
            = get_pass(engine->kind() == graph::engine_kind::gpu
                            ? "x8s8x8_conv_add_post_ops_gpu"
                            : "x8s8x8_conv_add_post_ops_cpu");
    ASSERT_NE(apass, nullptr);

    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1U);
    auto part = agraph.get_partitions()[0];

    // compile
    graph::partition_t p;
    p.init(part);

    graph::compiled_partition_t cp(p);

    std::vector<const graph::logical_tensor_t *> lt_ins {
            &src_u8, &weight_s8, &bias_f32, &other_s8};
    std::vector<const graph::logical_tensor_t *> lt_outs {&dst_u8};

    p.compile(&cp, lt_ins, lt_outs, engine);

    cp.execute(strm, {src_u8_ts, weight_s8_ts, bias_f32_ts, other_s8_ts},
            {dst_u8_case2_ts});
    strm->wait();

    ASSERT_TRUE(allclose(case1_out_data, case2_out_data, /*rtol*/ 0.01f,
            /*atol*/ 1.f));
}

TEST(ExecuteSubgraphInt8, Conv2dSumReluNxc) {
    using dims = graph::dnnl_impl::dims;

    graph::engine_t *engine = get_engine();
    graph::stream_t *strm = get_stream();

    static auto isa = dnnl_get_effective_cpu_isa();

    std::vector<int64_t> groups = {1, 4};
    std::vector<bool> with_biases = {true, false};
    std::vector<std::string> weight_qtypes = {"per_tensor", "per_channel"};

    for_(const auto &g : groups)
    for_(const auto with_bias : with_biases)
    for (const auto &wei_qtype : weight_qtypes) {
        // prepare fp32 data
        int64_t in_channel = 8, out_channel = 8;
        int64_t kernel_size = 3;
        std::vector<int64_t> src_shape {1, 12, 12, in_channel};
        std::vector<int64_t> weight_shape {
                kernel_size, kernel_size, in_channel / g, out_channel};
        std::vector<int64_t> bias_shape {out_channel};
        std::vector<int64_t> dst_shape {1, 10, 10, out_channel};

        test::vector<uint8_t> src_u8_data(product(src_shape));
        test::vector<int8_t> weight_s8_data(product(weight_shape));
        test::vector<int8_t> other_s8_data(product(dst_shape));
        size_t bias_size = with_bias ? product(bias_shape) : 0;
        test::vector<float> bias_data(bias_size);
        test::vector<int8_t> case1_out_data(product(dst_shape));
        test::vector<int8_t> case2_out_data(product(dst_shape));

        // random generate src, weight and bias data random seed = 7
        std::default_random_engine generator(7);
        std::uniform_real_distribution<float> u8_distribution(0.0f, 255.0f);
        std::uniform_real_distribution<float> s8_distribution(-127.0f, 128.0f);
        std::uniform_real_distribution<float> f32_distribution(0.0f, 1.0f);
        std::generate(src_u8_data.begin(), src_u8_data.end(), [&]() {
            return static_cast<uint8_t>(u8_distribution(generator));
        });
        std::generate(weight_s8_data.begin(), weight_s8_data.end(), [&]() {
            return static_cast<int8_t>(s8_distribution(generator));
        });
        std::generate(other_s8_data.begin(), other_s8_data.end(), [&]() {
            return static_cast<int8_t>(s8_distribution(generator));
        });
        if (with_bias) {
            std::generate(bias_data.begin(), bias_data.end(),
                    [&]() { return f32_distribution(generator); });
        }

        float scale_src = 1 / 255.f; // map to 0~255
        float scale_other = 1 / 127.f;
        float scale_out = 1;
        int64_t zp_src = 0;
        int64_t zp_other = 0;
        int64_t zp_out = 78;

        size_t scale_size = wei_qtype == "per_tensor" ? 1 : out_channel;

        std::vector<float> scale_wei(scale_size, 1 / 127.f);
        std::vector<int64_t> zp_wei(scale_size, 0);

        graph::op_t dqdata_node(1, graph::op_kind::Dequantize, "dqdata_node");
        SET_Q_DQ_DATA_ATTR(dqdata_node)

        graph::op_t dqweight_node(
                3, graph::op_kind::Dequantize, "dqweight_node");
        SET_Q_DQ_WEIGHT_ATTR(dqweight_node, 0)
        dqweight_node.set_attr<int64_t>(
                graph::op_attr::axis, wei_qtype == "per_tensor" ? 0 : 3);

        graph::op_t conv_node(4, graph::op_kind::Convolution, "conv_node");
        SET_CONV_ATTR(conv_node, 2)
        conv_node.set_attr<std::string>(graph::op_attr::data_format, "NXC");
        conv_node.set_attr<std::string>(graph::op_attr::weights_format, "XIO");

        graph::op_t relu_node(5, graph::op_kind::ReLU, "relu_node");

        graph::op_t qout_node(6, graph::op_kind::Quantize, "qout_node");
        SET_Q_DQ_OUT_ATTR(qout_node)

        graph::op_t dqother_node(8, graph::op_kind::Dequantize, "dqother_node");
        dqother_node.set_attr<std::string>(graph::op_attr::qtype, "per_tensor");
        dqother_node.set_attr<std::vector<int64_t>>(
                graph::op_attr::zps, {zp_other});
        dqother_node.set_attr<std::vector<float>>(
                graph::op_attr::scales, {scale_other});
        dqother_node.set_attr<int64_t>(graph::op_attr::axis, 0);

        graph::op_t add_node(9, graph::op_kind::Add, "add_node");

        // prepare logical tensor
        auto src_u8 = utils::logical_tensor_init(1, graph::data_type::u8);
        auto src_f32_dq = utils::logical_tensor_init(2, graph::data_type::f32);
        auto weight_s8 = utils::logical_tensor_init(4, graph::data_type::s8);
        auto weight_f32_dq
                = utils::logical_tensor_init(5, graph::data_type::f32);
        auto dst_f32 = utils::logical_tensor_init(7, graph::data_type::f32);
        auto dst_relu_f32
                = utils::logical_tensor_init(8, graph::data_type::f32);
        auto dst_s8 = utils::logical_tensor_init(9, graph::data_type::s8);
        auto other_s8 = utils::logical_tensor_init(11, graph::data_type::s8);
        auto other_f32_dq
                = utils::logical_tensor_init(12, graph::data_type::f32);
        auto dst_add_f32
                = utils::logical_tensor_init(13, graph::data_type::f32);
        graph::logical_tensor_t bias_f32;
        if (with_bias) {
            bias_f32 = utils::logical_tensor_init(6, graph::data_type::f32);
        }

        // -------------------------case 2----------------------------------
        dqdata_node.add_input(src_u8);
        dqdata_node.add_output(src_f32_dq);

        dqweight_node.add_input(weight_s8);
        dqweight_node.add_output(weight_f32_dq);

        conv_node.add_input(src_f32_dq);
        conv_node.add_input(weight_f32_dq);
        if (with_bias) conv_node.add_input(bias_f32);
        conv_node.add_output(dst_f32);

        dqother_node.add_input(other_s8);
        dqother_node.add_output(other_f32_dq);

        add_node.add_input(dst_f32);
        add_node.add_input(other_f32_dq);
        add_node.add_output(dst_add_f32);

        relu_node.add_input(dst_add_f32);
        relu_node.add_output(dst_relu_f32);

        qout_node.add_input(dst_relu_f32);
        qout_node.add_output(dst_s8);

        graph::graph_t g(engine->kind());
        g.add_op(&dqdata_node);
        g.add_op(&dqweight_node);
        g.add_op(&conv_node);
        g.add_op(&dqother_node);
        g.add_op(&add_node);
        g.add_op(&relu_node);
        g.add_op(&qout_node);
        g.finalize();

        // prepare in/out with full shape
        src_u8 = utils::logical_tensor_init(1, src_shape, graph::data_type::u8);
        weight_s8 = utils::logical_tensor_init(
                4, weight_shape, graph::data_type::s8);
        other_s8 = utils::logical_tensor_init(
                11, dst_shape, graph::data_type::s8);
        if (with_bias) {
            bias_f32 = utils::logical_tensor_init(
                    6, bias_shape, graph::data_type::f32);
        }
        dst_s8 = utils::logical_tensor_init(9, dst_shape, graph::data_type::s8);

        graph::tensor_t src_u8_ts(src_u8, engine, src_u8_data.data());
        graph::tensor_t weight_s8_ts(weight_s8, engine, weight_s8_data.data());
        graph::tensor_t other_s8_ts(other_s8, engine, other_s8_data.data());
        graph::tensor_t bias_f32_ts;
        if (with_bias) {
            bias_f32_ts = graph::tensor_t(bias_f32, engine, bias_data.data());
        }
        graph::tensor_t dst_s8_ts(dst_s8, engine, case1_out_data.data());
        graph::tensor_t dst_s8_case2_ts(dst_s8, engine, case2_out_data.data());

        // -------------------------case 1----------------------------------
        ASSERT_EQ(run_graph(g,
                          {src_u8_ts, weight_s8_ts, bias_f32_ts, other_s8_ts},
                          {dst_s8_ts}, *engine, *strm),
                graph::status::success);

        // -------------------------case 2----------------------------------
        graph::pass::pass_base_ptr apass
                = get_pass(engine->kind() == graph::engine_kind::gpu
                                ? "x8s8x8_conv_add_post_ops_gpu"
                                : "x8s8x8_conv_add_post_ops_cpu");

        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 1U);
        auto part = g.get_partitions()[0];

        // compile
        graph::partition_t p;
        p.init(part);

        graph::compiled_partition_t cp(p);

        std::vector<const graph::logical_tensor_t *> lt_ins;
        if (with_bias)
            lt_ins = {&src_u8, &weight_s8, &bias_f32, &other_s8};
        else
            lt_ins = {&src_u8, &weight_s8, &other_s8};
        std::vector<const graph::logical_tensor_t *> lt_outs {&dst_s8};

        p.compile(&cp, lt_ins, lt_outs, engine);

        if (with_bias)
            cp.execute(strm,
                    {src_u8_ts, weight_s8_ts, bias_f32_ts, other_s8_ts},
                    {dst_s8_case2_ts});
        else
            cp.execute(strm, {src_u8_ts, weight_s8_ts, other_s8_ts},
                    {dst_s8_case2_ts});
        strm->wait();

        if (engine->kind() == graph::engine_kind::cpu
                && isa < dnnl_cpu_isa_avx512_core_vnni)
            ASSERT_TRUE(allclose(case1_out_data, case2_out_data, /*rtol*/ 0.1f,
                    /*atol*/ 1.f));
        else
            ASSERT_TRUE(allclose(case1_out_data, case2_out_data, /*rtol*/ 0.01f,
                    /*atol*/ 1.f));
    }
}

TEST(ExecuteSubgraphInt8, Conv1d2d3dX8s8f32) {
    using dims = graph::dnnl_impl::dims;

    graph::engine_t *engine = get_engine();
    graph::stream_t *strm = get_stream();

    std::vector<size_t> nds = {1, 2, 3};
    std::vector<int64_t> groups = {1};
    std::vector<bool> with_biases = {true, false};
    std::vector<std::string> weight_qtypes = {"per_tensor", "per_channel"};
    std::vector<std::string> src_qtypes = {"symmetric", "asymmetric"};

    static auto isa = dnnl_get_effective_cpu_isa();
    SKIP_IF(isa < dnnl_cpu_isa_avx512_core_vnni
                    && engine->kind() == graph::engine_kind::cpu,
            "Skip the test for systems that do not support "
            "avx512_core_vnni.");

    for_(const auto &nd : nds)
    for_(const auto &g : groups)
    for_(const auto with_bias : with_biases)
    for_(const auto &src_qtype : src_qtypes)
    for (const auto &wei_qtype : weight_qtypes) {
        if (engine->kind() == graph::engine_kind::gpu
                && (src_qtype == "asymmetric" || nd == 1))
            continue;

        // prepare fp32 data
        int64_t in_channel = 8, out_channel = 8;
        int64_t kernel_size = 3;
        std::vector<int64_t> src_shape = nd == 1
                ? std::vector<int64_t> {1, in_channel, 12}
                : nd == 2 ? std::vector<int64_t> {1, in_channel, 12, 12}
                          : std::vector<int64_t> {1, in_channel, 12, 12, 12};
        std::vector<int64_t> weight_shape = nd == 1
                ? std::vector<int64_t> {out_channel, in_channel / g,
                        kernel_size}
                : nd == 2 ? std::vector<int64_t> {out_channel, in_channel / g,
                          kernel_size, kernel_size}
                          : std::vector<int64_t> {out_channel, in_channel / g,
                                  kernel_size, kernel_size, kernel_size};
        std::vector<int64_t> bias_shape {out_channel};
        std::vector<int64_t> dst_shape = nd == 1
                ? std::vector<int64_t> {1, out_channel, 10}
                : nd == 2 ? std::vector<int64_t> {1, out_channel, 10, 10}
                          : std::vector<int64_t> {1, out_channel, 10, 10, 10};

        test::vector<uint8_t> src_u8_data(product(src_shape));
        test::vector<int8_t> weight_s8_data(product(weight_shape));
        size_t bias_size = with_bias ? product(bias_shape) : 0;
        test::vector<float> bias_data(bias_size);

        // random generate src, weight and bias data random seed = 7
        std::default_random_engine generator(7);
        std::uniform_real_distribution<float> u8_distribution(0.0f, 255.0f);
        std::uniform_real_distribution<float> s8_distribution(-127.0f, 128.0f);
        std::uniform_real_distribution<float> f32_distribution(0.0f, 1.0f);
        std::generate(src_u8_data.begin(), src_u8_data.end(), [&]() {
            return static_cast<uint8_t>(u8_distribution(generator));
        });
        std::generate(weight_s8_data.begin(), weight_s8_data.end(), [&]() {
            return static_cast<int8_t>(s8_distribution(generator));
        });
        if (with_bias) {
            std::generate(bias_data.begin(), bias_data.end(),
                    [&]() { return f32_distribution(generator); });
        }

        float scale_src = 1 / 255.f; // map to 0~255
        int64_t zp_src = src_qtype == "symmetric" ? 0 : 128;

        size_t scale_size = wei_qtype == "per_tensor" ? 1 : out_channel;
        std::vector<float> scale_wei(scale_size, 1 / 127.f);
        std::vector<int64_t> zp_wei(scale_size, 0);

        graph::op_t dqdata_node(1, graph::op_kind::Dequantize, "dqdata_node");
        SET_Q_DQ_DATA_ATTR(dqdata_node)
        graph::op_t dqweight_node(
                3, graph::op_kind::Dequantize, "dqweight_node");
        SET_Q_DQ_WEIGHT_ATTR(dqweight_node, 0)
        graph::op_t conv_node(4, graph::op_kind::Convolution, "conv_node");
        SET_CONV_ATTR(conv_node, nd)

        // prepare logical tensor
        graph::logical_tensor_t src_u8 = utils::logical_tensor_init(
                1, src_shape, graph::data_type::u8);
        graph::logical_tensor_t src_f32_dq = utils::logical_tensor_init(
                2, src_shape, graph::data_type::f32);
        graph::logical_tensor_t weight_s8 = utils::logical_tensor_init(
                4, weight_shape, graph::data_type::s8);
        graph::logical_tensor_t weight_f32_dq = utils::logical_tensor_init(
                5, weight_shape, graph::data_type::f32);
        graph::logical_tensor_t dst_f32 = utils::logical_tensor_init(
                7, dst_shape, graph::data_type::f32);
        graph::logical_tensor_t bias_f32;
        if (with_bias) {
            bias_f32 = utils::logical_tensor_init(
                    6, bias_shape, graph::data_type::f32);
        }

        dqdata_node.add_input(src_u8);
        dqdata_node.add_output(src_f32_dq);

        dqweight_node.add_input(weight_s8);
        dqweight_node.add_output(weight_f32_dq);

        conv_node.add_input(src_f32_dq);
        conv_node.add_input(weight_f32_dq);
        if (with_bias) conv_node.add_input(bias_f32);
        conv_node.add_output(dst_f32);

        graph::graph_t g(engine->kind());
        g.add_op(&dqdata_node);
        g.add_op(&dqweight_node);
        g.add_op(&conv_node);
        g.finalize();

        graph::tensor_t src_u8_ts(src_u8, engine, src_u8_data.data());
        graph::tensor_t weight_s8_ts(weight_s8, engine, weight_s8_data.data());
        graph::tensor_t bias_f32_ts;
        if (with_bias) {
            bias_f32_ts = graph::tensor_t(bias_f32, engine, bias_data.data());
        }

        // -------------------------case 1----------------------------------
        test::vector<float> case1_out_data(product(dst_shape));
        graph::tensor_t dst_f32_ts(dst_f32, engine, case1_out_data.data());

        ASSERT_EQ(run_graph(g, {src_u8_ts, weight_s8_ts, bias_f32_ts},
                          {dst_f32_ts}, *engine, *strm),
                graph::status::success);

        // -------------------------case 2----------------------------------
        graph::pass::pass_base_ptr apass = get_pass("x8s8x_conv_post_ops");
        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 1U);
        auto part = g.get_partitions()[0];

        // compile
        graph::partition_t p;
        p.init(part);

        graph::compiled_partition_t cp(p);

        std::vector<const graph::logical_tensor_t *> lt_ins;
        if (with_bias)
            lt_ins = {&src_u8, &weight_s8, &bias_f32};
        else
            lt_ins = {&src_u8, &weight_s8};
        std::vector<const graph::logical_tensor_t *> lt_outs {&dst_f32};

        p.compile(&cp, lt_ins, lt_outs, engine);

        test::vector<float> case2_out_data(product(dst_shape));
        graph::tensor_t dst_f32_case2_ts(
                dst_f32, engine, case2_out_data.data());
        if (with_bias)
            cp.execute(strm, {src_u8_ts, weight_s8_ts, bias_f32_ts},
                    {dst_f32_case2_ts});
        else
            cp.execute(strm, {src_u8_ts, weight_s8_ts}, {dst_f32_case2_ts});
        strm->wait();

        if (engine->kind() == graph::engine_kind::cpu
                && isa < dnnl_cpu_isa_avx512_core_vnni)
            ASSERT_TRUE(allclose(case1_out_data, case2_out_data, /*rtol*/ 0.1f,
                    /*atol*/ 1.f));
        else
            ASSERT_TRUE(allclose(case1_out_data, case2_out_data, /*rtol*/ 0.01f,
                    /*atol*/ 1.f));
    }
}

TEST(ExecuteSubgraphInt8, Conv2dReluX8s8f32) {
    using dims = graph::dnnl_impl::dims;

    graph::engine_t *engine = get_engine();
    graph::stream_t *strm = get_stream();

    std::vector<int64_t> groups = {1, 4};
    std::vector<bool> with_biases = {true, false};
    std::vector<std::string> weight_qtypes = {"per_tensor", "per_channel"};

    static auto isa = dnnl_get_effective_cpu_isa();
    SKIP_IF(isa < dnnl_cpu_isa_avx512_core_vnni
                    && engine->kind() == graph::engine_kind::cpu,
            "Skip the test for systems that do not support "
            "avx512_core_vnni.");

    for_(const auto &g : groups)
    for_(const auto with_bias : with_biases)
    for (const auto &wei_qtype : weight_qtypes) {
        // prepare data
        int64_t in_channel = 8, out_channel = 8;
        int64_t kernel_size = 3;
        std::vector<int64_t> src_shape {1, in_channel, 112, 112};
        std::vector<int64_t> weight_shape {
                out_channel, in_channel / g, kernel_size, kernel_size};
        std::vector<int64_t> bias_shape {out_channel};
        std::vector<int64_t> dst_shape {1, out_channel, 110, 110};

        test::vector<uint8_t> src_u8_data(product(src_shape));
        test::vector<int8_t> weight_s8_data(product(weight_shape));
        size_t bias_size = with_bias ? product(bias_shape) : 0;
        test::vector<float> bias_data(bias_size);
        test::vector<float> case1_out_data(product(dst_shape));
        test::vector<float> case2_out_data(product(dst_shape));

        // random generate src, weight and bias data random seed = 7
        std::default_random_engine generator(7);
        std::uniform_real_distribution<float> u8_distribution(0.0f, 255.0f);
        std::uniform_real_distribution<float> s8_distribution(-127.0f, 128.0f);
        std::uniform_real_distribution<float> f32_distribution(0.0f, 1.0f);
        std::generate(src_u8_data.begin(), src_u8_data.end(), [&]() {
            return static_cast<uint8_t>(u8_distribution(generator));
        });
        std::generate(weight_s8_data.begin(), weight_s8_data.end(), [&]() {
            return static_cast<int8_t>(s8_distribution(generator));
        });
        if (with_bias) {
            std::generate(bias_data.begin(), bias_data.end(),
                    [&]() { return f32_distribution(generator); });
        }

        float scale_src = 1 / 255.f; // map to 0~255
        int64_t zp_src = 0;

        size_t scale_size = wei_qtype == "per_tensor" ? 1 : out_channel;
        std::vector<float> scale_wei(scale_size, 1 / 127.f);
        std::vector<int64_t> zp_wei(scale_size, 0);

        graph::op_t dqdata_node(1, graph::op_kind::Dequantize, "dqdata_node");
        SET_Q_DQ_DATA_ATTR(dqdata_node)

        graph::op_t dqweight_node(
                3, graph::op_kind::Dequantize, "dqweight_node");
        SET_Q_DQ_WEIGHT_ATTR(dqweight_node, 0)

        graph::op_t conv_node(4, graph::op_kind::Convolution, "conv_node");
        SET_CONV_ATTR(conv_node, 2)

        graph::op_t relu_node(5, graph::op_kind::ReLU, "relu_node");

        // prepare logical tensor
        graph::logical_tensor_t src_u8 = utils::logical_tensor_init(
                1, src_shape, graph::data_type::u8);
        graph::logical_tensor_t src_f32_dq = utils::logical_tensor_init(
                2, src_shape, graph::data_type::f32);
        graph::logical_tensor_t weight_s8 = utils::logical_tensor_init(
                4, weight_shape, graph::data_type::s8);
        graph::logical_tensor_t weight_f32_dq = utils::logical_tensor_init(
                5, weight_shape, graph::data_type::f32);
        graph::logical_tensor_t dst_f32 = utils::logical_tensor_init(
                7, dst_shape, graph::data_type::f32);
        graph::logical_tensor_t dst_relu_f32 = utils::logical_tensor_init(
                8, dst_shape, graph::data_type::f32);
        graph::logical_tensor_t bias_f32;
        if (with_bias) {
            bias_f32 = utils::logical_tensor_init(
                    6, bias_shape, graph::data_type::f32);
        }

        dqdata_node.add_input(src_u8);
        dqdata_node.add_output(src_f32_dq);

        dqweight_node.add_input(weight_s8);
        dqweight_node.add_output(weight_f32_dq);

        conv_node.add_input(src_f32_dq);
        conv_node.add_input(weight_f32_dq);
        if (with_bias) conv_node.add_input(bias_f32);
        conv_node.add_output(dst_f32);

        relu_node.add_input(dst_f32);
        relu_node.add_output(dst_relu_f32);

        graph::graph_t g(engine->kind());
        g.add_op(&dqdata_node);
        g.add_op(&dqweight_node);
        g.add_op(&conv_node);
        g.add_op(&relu_node);
        g.finalize();

        graph::tensor_t src_u8_ts(src_u8, engine, src_u8_data.data());
        graph::tensor_t weight_s8_ts(weight_s8, engine, weight_s8_data.data());
        graph::tensor_t bias_f32_ts;
        if (with_bias) {
            bias_f32_ts = graph::tensor_t(bias_f32, engine, bias_data.data());
        }
        graph::tensor_t dst_relu_f32_ts(
                dst_relu_f32, engine, case1_out_data.data());
        graph::tensor_t dst_f32_case2_ts(
                dst_relu_f32, engine, case2_out_data.data());

        // -------------------------case 1----------------------------------
        ASSERT_EQ(run_graph(g, {src_u8_ts, weight_s8_ts, bias_f32_ts},
                          {dst_relu_f32_ts}, *engine, *strm),
                graph::status::success);

        // -------------------------case 2----------------------------------
        graph::pass::pass_base_ptr apass = get_pass("x8s8x_conv_post_ops");
        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 1U);
        auto part = g.get_partitions()[0];

        // compile
        graph::partition_t p;
        p.init(part);

        graph::compiled_partition_t cp(p);

        std::vector<const graph::logical_tensor_t *> lt_ins;
        if (with_bias)
            lt_ins = {&src_u8, &weight_s8, &bias_f32};
        else
            lt_ins = {&src_u8, &weight_s8};
        std::vector<const graph::logical_tensor_t *> lt_outs {&dst_relu_f32};

        p.compile(&cp, lt_ins, lt_outs, engine);

        if (with_bias)
            cp.execute(strm, {src_u8_ts, weight_s8_ts, bias_f32_ts},
                    {dst_f32_case2_ts});
        else
            cp.execute(strm, {src_u8_ts, weight_s8_ts}, {dst_f32_case2_ts});
        strm->wait();

        if (engine->kind() == graph::engine_kind::cpu
                && isa < dnnl_cpu_isa_avx512_core_vnni)
            ASSERT_TRUE(allclose(case1_out_data, case2_out_data, /*rtol*/ 0.1f,
                    /*atol*/ 1.f));
        else
            ASSERT_TRUE(allclose(case1_out_data, case2_out_data, /*rtol*/ 0.01f,
                    /*atol*/ 1.f));
    }
}

TEST(ExecuteSubgraphInt8, Conv2dSumReluGetInplacePair) {
    using dims = graph::dnnl_impl::dims;

    graph::engine_t *engine = get_engine();

    std::vector<int64_t> groups = {1, 4};
    std::vector<std::string> weight_qtypes = {"per_tensor", "per_channel"};
    std::vector<std::string> other_qtypes = {"symmetric", "asymmetric"};

    for_(const auto &g : groups)
    for_(const auto &other_qtype : other_qtypes)
    for (const auto &wei_qtype : weight_qtypes) {
        int64_t in_channel = 8, out_channel = 8;
        int64_t kernel_size = 3;
        std::vector<int64_t> src_shape {1, in_channel, 112, 112};
        std::vector<int64_t> weight_shape {
                out_channel, in_channel / g, kernel_size, kernel_size};
        std::vector<int64_t> dst_shape {1, out_channel, 110, 110};

        float scale_src = 1 / 255.f; // map to 0~255
        float scale_other = 1 / 127.f;
        float scale_out = 1;
        int64_t zp_src = 0;
        // post-sum didn't support zps on GPU
        int64_t zp_other = other_qtype == "symmetric"
                        || engine->kind() == graph::engine_kind::gpu
                ? 0
                : 128;
        int64_t zp_out = 78;

        size_t scale_size = wei_qtype == "per_tensor" ? 1 : out_channel;

        std::vector<float> scale_wei(scale_size, 1 / 127.f);
        std::vector<int64_t> zp_wei(scale_size, 0);

        graph::op_t dqdata_node(1, graph::op_kind::Dequantize, "dqdata_node");
        SET_Q_DQ_DATA_ATTR(dqdata_node)

        graph::op_t dqweight_node(
                3, graph::op_kind::Dequantize, "dqweight_node");
        SET_Q_DQ_WEIGHT_ATTR(dqweight_node, 0)

        graph::op_t conv_node(4, graph::op_kind::Convolution, "conv_node");
        SET_CONV_ATTR(conv_node, 2)

        graph::op_t relu_node(5, graph::op_kind::ReLU, "relu_node");

        graph::op_t qout_node(6, graph::op_kind::Quantize, "qout_node");
        SET_Q_DQ_OUT_ATTR(qout_node)

        graph::op_t dqother_node(8, graph::op_kind::Dequantize, "dqother_node");
        dqother_node.set_attr<std::string>(graph::op_attr::qtype, "per_tensor");
        dqother_node.set_attr<std::vector<int64_t>>(
                graph::op_attr::zps, {zp_other});
        dqother_node.set_attr<std::vector<float>>(
                graph::op_attr::scales, {scale_other});
        dqother_node.set_attr<int64_t>(graph::op_attr::axis, 0);

        graph::op_t add_node(9, graph::op_kind::Add, "add_node");

        graph::op_t dqdata_node2(10, graph::op_kind::Dequantize, "dqdata_node");
        SET_Q_DQ_DATA_ATTR(dqdata_node2)

        graph::op_t dqweight_node2(
                11, graph::op_kind::Dequantize, "dqweight_node");
        SET_Q_DQ_WEIGHT_ATTR(dqweight_node2, 0)

        graph::op_t conv_node2(12, graph::op_kind::Convolution, "conv_node");
        SET_CONV_ATTR(conv_node2, 2)

        graph::op_t relu_node2(13, graph::op_kind::ReLU, "relu_node");

        graph::op_t qout_node2(14, graph::op_kind::Quantize, "qout_node");
        SET_Q_DQ_OUT_ATTR(qout_node2)

        // prepare logical tensor
        graph::logical_tensor_t src_u8 = utils::logical_tensor_init(
                1, src_shape, graph::data_type::u8);
        graph::logical_tensor_t src_f32_dq = utils::logical_tensor_init(
                2, src_shape, graph::data_type::f32);
        graph::logical_tensor_t weight_s8 = utils::logical_tensor_init(
                4, weight_shape, graph::data_type::s8);
        graph::logical_tensor_t weight_f32_dq = utils::logical_tensor_init(
                5, weight_shape, graph::data_type::f32);
        graph::logical_tensor_t dst_f32 = utils::logical_tensor_init(
                7, dst_shape, graph::data_type::f32);
        graph::logical_tensor_t dst_relu_f32 = utils::logical_tensor_init(
                8, dst_shape, graph::data_type::f32);
        graph::logical_tensor_t dst_s8 = utils::logical_tensor_init(
                9, dst_shape, graph::data_type::s8);
        graph::logical_tensor_t other_f32_dq = utils::logical_tensor_init(
                12, dst_shape, graph::data_type::f32);
        graph::logical_tensor_t dst_add_f32 = utils::logical_tensor_init(
                13, dst_shape, graph::data_type::f32);

        graph::logical_tensor_t src_u8_2 = utils::logical_tensor_init(
                14, src_shape, graph::data_type::u8);
        graph::logical_tensor_t src_f32_dq_2 = utils::logical_tensor_init(
                15, src_shape, graph::data_type::f32);
        graph::logical_tensor_t weight_s8_2 = utils::logical_tensor_init(
                16, weight_shape, graph::data_type::s8);
        graph::logical_tensor_t weight_f32_dq_2 = utils::logical_tensor_init(
                17, weight_shape, graph::data_type::f32);
        graph::logical_tensor_t dst_f32_2 = utils::logical_tensor_init(
                18, dst_shape, graph::data_type::f32);
        graph::logical_tensor_t dst_s8_2 = utils::logical_tensor_init(
                19, dst_shape, graph::data_type::s8);

        dqdata_node.add_input(src_u8);
        dqdata_node.add_output(src_f32_dq);

        dqweight_node.add_input(weight_s8);
        dqweight_node.add_output(weight_f32_dq);

        conv_node.add_input(src_f32_dq);
        conv_node.add_input(weight_f32_dq);
        conv_node.add_output(dst_f32);

        dqother_node.add_input(dst_s8_2);
        dqother_node.add_output(other_f32_dq);

        add_node.add_input(dst_f32);
        add_node.add_input(other_f32_dq);
        add_node.add_output(dst_add_f32);

        relu_node.add_input(dst_add_f32);
        relu_node.add_output(dst_relu_f32);

        qout_node.add_input(dst_relu_f32);
        qout_node.add_output(dst_s8);

        dqdata_node2.add_input(src_u8_2);
        dqdata_node2.add_output(src_f32_dq_2);

        dqweight_node2.add_input(weight_s8_2);
        dqweight_node2.add_output(weight_f32_dq_2);

        conv_node2.add_input(src_f32_dq_2);
        conv_node2.add_input(weight_f32_dq_2);
        conv_node2.add_output(dst_f32_2);

        qout_node2.add_input(dst_f32_2);
        qout_node2.add_output(dst_s8_2);

        graph::graph_t g(engine->kind());
        g.add_op(&dqdata_node);
        g.add_op(&dqweight_node);
        g.add_op(&conv_node);
        g.add_op(&dqother_node);
        g.add_op(&add_node);
        g.add_op(&relu_node);
        g.add_op(&qout_node);
        g.add_op(&dqdata_node2);
        g.add_op(&dqweight_node2);
        g.add_op(&conv_node2);
        g.add_op(&qout_node2);
        g.finalize();

        graph::pass::pass_base_ptr apass1
                = get_pass(engine->kind() == graph::engine_kind::gpu
                                ? "x8s8x8_conv_add_post_ops_gpu"
                                : "x8s8x8_conv_add_post_ops_cpu");
        graph::pass::pass_base_ptr apass2 = get_pass("x8s8x_conv_post_ops");

        apass1->run(g);
        apass2->run(g);

        ASSERT_EQ(g.get_num_partitions(), 2U);
        auto part2 = g.get_partitions()[0]; // int8_conv_sum_relu
        auto part1 = g.get_partitions()[1]; // int8_conv

        // compile
        graph::partition_t p1, p2;
        p1.init(part1);
        p2.init(part2);

        graph::compiled_partition_t cp1(p1);
        graph::compiled_partition_t cp2(p2);

        std::vector<const graph::logical_tensor_t *> lt_ins1;
        lt_ins1 = {&src_u8_2, &weight_s8_2};

        dst_s8_2.layout_type = graph::layout_type::any;
        std::vector<const graph::logical_tensor_t *> lt_outs1 {&dst_s8_2};

        p1.compile(&cp1, lt_ins1, lt_outs1, engine);

        cp1.query_logical_tensor(dst_s8_2.id, &dst_s8_2);

        std::vector<const graph::logical_tensor_t *> lt_ins;
        lt_ins = {&src_u8, &weight_s8, &dst_s8_2};

        dst_s8.layout_type = graph::layout_type::any;
        std::vector<const graph::logical_tensor_t *> lt_outs {&dst_s8};

        p2.compile(&cp2, lt_ins, lt_outs, engine);

        std::vector<graph::inplace_pair_t> inplace_pairs
                = cp2.get_inplace_pairs();

        ASSERT_EQ(inplace_pairs.size(), 1U);
        ASSERT_EQ(inplace_pairs[0].input_id, dst_s8_2.id);
        ASSERT_EQ(inplace_pairs[0].output_id, dst_s8.id);
    }
}

TEST(ExecuteSubgraphInt8, ConvolutionBiasU8s8u8MixBf16) {
    using dims = dnnl::impl::graph::dnnl_impl::dims;
    graph::engine_t *engine = get_engine();
    graph::stream_t *strm = get_stream();

    std::string qtype = "per_channel";
    std::vector<int64_t> groups = {1, 4};
    for (const auto &g_ : groups) {
        int64_t in_channel = 8, out_channel = 8;
        std::vector<int64_t> src_shape = {1, in_channel, 12, 12};
        std::vector<int64_t> weight_shape
                = {out_channel, in_channel / g_, 3, 3};
        std::vector<int64_t> bias_shape = {out_channel};
        std::vector<int64_t> dst_shape = {1, out_channel, 10, 10};

        test::vector<uint8_t> src_data(product(src_shape));
        test::vector<int8_t> weight_data(product(weight_shape));
        test::vector<float> bias_data(product(bias_shape));

        // random generate src, weight data
        // random seed = 7
        std::default_random_engine generator(7);
        std::uniform_real_distribution<float> distribution(0.0f, 255.0f);
        std::generate(src_data.begin(), src_data.end(), [&]() {
            return static_cast<uint8_t>(distribution(generator));
        });
        std::uniform_real_distribution<float> distribution2(-127.0f, 127.0f);
        std::generate(weight_data.begin(), weight_data.end(), [&]() {
            return static_cast<int8_t>(distribution2(generator));
        });
        std::uniform_real_distribution<float> distribution3(0.0f, 20.0f);
        std::generate(bias_data.begin(), bias_data.end(),
                [&]() { return distribution3(generator); });
        float scale_src = 1 / 255.f; // map to 0~255
        int64_t zp_src = 110;

        size_t scales_wei_sizes = qtype == "per_tensor" ? 1 : dst_shape.back();
        std::vector<float> scale_wei(scales_wei_sizes, 1 / 127.f);
        std::vector<int64_t> zp_wei(scales_wei_sizes, 0);

        float scale_out = 1 / 255.f; // map to 0~255
        int64_t zp_out = engine->kind() == graph::engine_kind::gpu ? 0 : 110;

        graph::op_t dqdata_op(0, graph::op_kind::Dequantize, "dqdata_op");
        dqdata_op.set_attr<std::string>(graph::op_attr::qtype, "per_tensor");
        dqdata_op.set_attr<std::vector<int64_t>>(graph::op_attr::zps, {zp_src});
        dqdata_op.set_attr<std::vector<float>>(
                graph::op_attr::scales, {scale_src});
        dqdata_op.set_attr<int64_t>(graph::op_attr::axis, 0);

        graph::op_t dqweight_op(1, graph::op_kind::Dequantize, "dqweight_op");
        dqweight_op.set_attr<std::string>(graph::op_attr::qtype, qtype);
        dqweight_op.set_attr<std::vector<int64_t>>(graph::op_attr::zps, zp_wei);
        dqweight_op.set_attr<std::vector<float>>(
                graph::op_attr::scales, scale_wei);
        dqweight_op.set_attr<int64_t>(graph::op_attr::axis, 1);

        graph::op_t tcdata_op {2, graph::op_kind::TypeCast, "typecast_data"};
        graph::op_t tcweight_op {
                3, graph::op_kind::TypeCast, "typecast_weight"};

        graph::op_t conv_op(4, graph::op_kind::Convolution, "conv_op");
        conv_op.set_attr<dims>(graph::op_attr::strides, dims(2, 1));
        conv_op.set_attr<dims>(graph::op_attr::dilations, dims(2, 1));
        conv_op.set_attr<dims>(graph::op_attr::pads_begin, dims(2, 0));
        conv_op.set_attr<dims>(graph::op_attr::pads_end, dims(2, 0));
        conv_op.set_attr<int64_t>(graph::op_attr::groups, g_);
        conv_op.set_attr<std::string>(graph::op_attr::data_format, "NCX");
        conv_op.set_attr<std::string>(graph::op_attr::weights_format, "OIX");

        graph::op_t tcdst_op {5, graph::op_kind::TypeCast, "typecast_dst"};

        graph::op_t qout_op(6, graph::op_kind::Quantize, "qdout_op");
        qout_op.set_attr<std::string>(graph::op_attr::qtype, "per_tensor");
        qout_op.set_attr<std::vector<int64_t>>(graph::op_attr::zps, {zp_out});
        qout_op.set_attr<std::vector<float>>(
                graph::op_attr::scales, {scale_out});
        qout_op.set_attr<int64_t>(graph::op_attr::axis, 1);

        // prepare logical tensor
        graph::logical_tensor_t src_u8 = utils::logical_tensor_init(
                0, src_shape, graph::data_type::u8);
        graph::logical_tensor_t src_f32_dq = utils::logical_tensor_init(
                1, src_shape, graph::data_type::f32);
        graph::logical_tensor_t src_bf16 = utils::logical_tensor_init(
                2, src_shape, graph::data_type::bf16);
        graph::logical_tensor_t weight_s8 = utils::logical_tensor_init(
                3, weight_shape, graph::data_type::s8);
        graph::logical_tensor_t weight_f32_dq = utils::logical_tensor_init(
                4, weight_shape, graph::data_type::f32);
        graph::logical_tensor_t weight_bf16 = utils::logical_tensor_init(
                5, weight_shape, graph::data_type::bf16);
        graph::logical_tensor_t bias_bf16 = utils::logical_tensor_init(
                6, bias_shape, graph::data_type::bf16);
        graph::logical_tensor_t conv_bf16 = utils::logical_tensor_init(
                7, dst_shape, graph::data_type::bf16);
        graph::logical_tensor_t conv_f32 = utils::logical_tensor_init(
                9, dst_shape, graph::data_type::f32);
        graph::logical_tensor_t dst_u8 = utils::logical_tensor_init(
                10, dst_shape, graph::data_type::u8);

        dqdata_op.add_input(src_u8);
        dqdata_op.add_output(src_f32_dq);

        dqweight_op.add_input(weight_s8);
        dqweight_op.add_output(weight_f32_dq);

        tcdata_op.add_input(src_f32_dq);
        tcdata_op.add_output(src_bf16);

        tcweight_op.add_input(weight_f32_dq);
        tcweight_op.add_output(weight_bf16);

        conv_op.add_input(src_bf16);
        conv_op.add_input(weight_bf16);
        conv_op.add_input(bias_bf16);
        conv_op.add_output(conv_bf16);

        tcdst_op.add_input(conv_bf16);
        tcdst_op.add_output(conv_f32);

        qout_op.add_input(conv_f32);
        qout_op.add_output(dst_u8);

        graph::graph_t g(engine->kind());
        ASSERT_EQ(g.add_op(&dqdata_op), graph::status::success);
        ASSERT_EQ(g.add_op(&dqweight_op), graph::status::success);
        ASSERT_EQ(g.add_op(&conv_op), graph::status::success);
        ASSERT_EQ(g.add_op(&tcdata_op), graph::status::success);
        ASSERT_EQ(g.add_op(&tcweight_op), graph::status::success);
        ASSERT_EQ(g.add_op(&tcdst_op), graph::status::success);
        ASSERT_EQ(g.add_op(&qout_op), graph::status::success);
        g.finalize();

        graph::pass::pass_base_ptr apass = get_pass("x8s8x_tc_conv_post_ops");
        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 1U);
        auto part = g.get_partitions()[0];

        // compile
        graph::partition_t p;
        p.init(part);

        graph::compiled_partition_t cp(p);

        std::vector<const graph::logical_tensor_t *> lt_ins {
                &src_u8, &weight_s8, &bias_bf16};
        std::vector<const graph::logical_tensor_t *> lt_outs {&dst_u8};

        ASSERT_EQ(p.compile(&cp, lt_ins, lt_outs, engine),
                graph::status::success);

        test::vector<uint8_t> dst_data(product(dst_shape));
        graph::tensor_t src_u8_ts(src_u8, engine, src_data.data());
        graph::tensor_t weight_s8_ts(weight_s8, engine, weight_data.data());
        graph::tensor_t bias_bf16_ts(bias_bf16, engine, bias_data.data());
        graph::tensor_t dst_ts(dst_u8, engine, dst_data.data());
        cp.execute(strm, {src_u8_ts, weight_s8_ts, bias_bf16_ts}, {dst_ts});
        strm->wait();
    }
}

TEST(ExecuteSubgraphInt8, ConvolutionBiasaddU8s8u8MixBf16) {
    using dims = dnnl::impl::graph::dnnl_impl::dims;
    graph::engine_t *engine = get_engine();
    graph::stream_t *strm = get_stream();

    std::string qtype = "per_channel";
    std::vector<int64_t> groups = {1, 4};
    for (const auto &g_ : groups) {
        int64_t in_channel = 8, out_channel = 8;
        std::vector<int64_t> src_shape = {1, in_channel, 12, 12};
        std::vector<int64_t> weight_shape
                = {out_channel, in_channel / g_, 3, 3};
        std::vector<int64_t> bias_shape = {out_channel};
        std::vector<int64_t> dst_shape = {1, out_channel, 10, 10};

        test::vector<uint8_t> src_data(product(src_shape));
        test::vector<float> weight_data(product(weight_shape));
        test::vector<float> bias_data(product(bias_shape));

        // random generate src, weight data
        // random seed = 7
        std::default_random_engine generator(7);
        std::uniform_real_distribution<float> distribution(0.0f, 255.0f);
        std::generate(src_data.begin(), src_data.end(), [&]() {
            return static_cast<uint8_t>(distribution(generator));
        });
        std::uniform_real_distribution<float> distribution2(-1.f, 1.f);
        std::generate(weight_data.begin(), weight_data.end(),
                [&]() { return distribution2(generator); });
        std::uniform_real_distribution<float> distribution3(0.0f, 20.0f);
        std::generate(bias_data.begin(), bias_data.end(),
                [&]() { return distribution3(generator); });
        float scale_src = 1 / 255.f; // map to 0~255
        int64_t zp_src = 110;

        size_t scales_wei_sizes = qtype == "per_tensor" ? 1 : dst_shape.back();
        std::vector<float> scale_wei(scales_wei_sizes, 1 / 127.f);
        std::vector<int64_t> zp_wei(scales_wei_sizes, 0);

        float scale_out = 1 / 255.f; // map to 0~255
        int64_t zp_out = engine->kind() == graph::engine_kind::gpu ? 0 : 110;

        graph::op_t dqdata_op(0, graph::op_kind::Dequantize, "dqdata_op");
        dqdata_op.set_attr<std::string>(graph::op_attr::qtype, "per_tensor");
        dqdata_op.set_attr<std::vector<int64_t>>(graph::op_attr::zps, {zp_src});
        dqdata_op.set_attr<std::vector<float>>(
                graph::op_attr::scales, {scale_src});
        dqdata_op.set_attr<int64_t>(graph::op_attr::axis, 0);

        graph::op_t qweight_op(10, graph::op_kind::Quantize, "qweight_op");
        qweight_op.set_attr<std::string>(graph::op_attr::qtype, qtype);
        qweight_op.set_attr<std::vector<int64_t>>(graph::op_attr::zps, zp_wei);
        qweight_op.set_attr<std::vector<float>>(
                graph::op_attr::scales, scale_wei);
        qweight_op.set_attr<int64_t>(graph::op_attr::axis, 1);

        graph::op_t dqweight_op(1, graph::op_kind::Dequantize, "dqweight_op");
        dqweight_op.set_attr<std::string>(graph::op_attr::qtype, qtype);
        dqweight_op.set_attr<std::vector<int64_t>>(graph::op_attr::zps, zp_wei);
        dqweight_op.set_attr<std::vector<float>>(
                graph::op_attr::scales, scale_wei);
        dqweight_op.set_attr<int64_t>(graph::op_attr::axis, 1);

        graph::op_t tcdata_op {2, graph::op_kind::TypeCast, "typecast_data"};
        graph::op_t tcweight_op {
                3, graph::op_kind::TypeCast, "typecast_weight"};

        graph::op_t conv_op(4, graph::op_kind::Convolution, "conv_op");
        conv_op.set_attr<dims>(graph::op_attr::strides, dims(2, 1));
        conv_op.set_attr<dims>(graph::op_attr::dilations, dims(2, 1));
        conv_op.set_attr<dims>(graph::op_attr::pads_begin, dims(2, 0));
        conv_op.set_attr<dims>(graph::op_attr::pads_end, dims(2, 0));
        conv_op.set_attr<int64_t>(graph::op_attr::groups, g_);
        conv_op.set_attr<std::string>(graph::op_attr::data_format, "NCX");
        conv_op.set_attr<std::string>(graph::op_attr::weights_format, "OIX");

        graph::op_t tc_bias_op {5, graph::op_kind::TypeCast, "typecast_bias"};

        graph::op_t biasadd_op {6, graph::op_kind::BiasAdd, "biasadd_op"};
        biasadd_op.set_attr<std::string>(graph::op_attr::data_format, "NCX");

        graph::op_t tcdst_op {8, graph::op_kind::TypeCast, "typecast_dst"};

        graph::op_t qout_op(9, graph::op_kind::Quantize, "qdout_op");
        qout_op.set_attr<std::string>(graph::op_attr::qtype, "per_tensor");
        qout_op.set_attr<std::vector<int64_t>>(graph::op_attr::zps, {zp_out});
        qout_op.set_attr<std::vector<float>>(
                graph::op_attr::scales, {scale_out});
        qout_op.set_attr<int64_t>(graph::op_attr::axis, 1);

        // prepare logical tensor
        graph::logical_tensor_t src_u8 = utils::logical_tensor_init(
                0, src_shape, graph::data_type::u8);
        graph::logical_tensor_t src_f32_dq = utils::logical_tensor_init(
                1, src_shape, graph::data_type::f32);
        graph::logical_tensor_t src_bf16 = utils::logical_tensor_init(
                2, src_shape, graph::data_type::bf16);
        graph::logical_tensor_t weight_f32 = utils::logical_tensor_init(
                30, weight_shape, graph::data_type::f32);
        graph::logical_tensor_t weight_s8 = utils::logical_tensor_init(
                3, weight_shape, graph::data_type::s8);
        graph::logical_tensor_t weight_f32_dq = utils::logical_tensor_init(
                4, weight_shape, graph::data_type::f32);
        graph::logical_tensor_t weight_bf16 = utils::logical_tensor_init(
                5, weight_shape, graph::data_type::bf16);
        graph::logical_tensor_t conv_bf16 = utils::logical_tensor_init(
                6, dst_shape, graph::data_type::bf16);
        graph::logical_tensor_t bias_f32 = utils::logical_tensor_init(
                7, bias_shape, graph::data_type::f32);
        graph::logical_tensor_t bias_bf16 = utils::logical_tensor_init(
                8, bias_shape, graph::data_type::bf16);
        graph::logical_tensor_t bias_out_bf16 = utils::logical_tensor_init(
                9, dst_shape, graph::data_type::bf16);
        graph::logical_tensor_t conv_f32 = utils::logical_tensor_init(
                11, dst_shape, graph::data_type::f32);
        graph::logical_tensor_t dst_u8 = utils::logical_tensor_init(
                12, dst_shape, graph::data_type::u8);

        dqdata_op.add_input(src_u8);
        dqdata_op.add_output(src_f32_dq);

        qweight_op.add_input(weight_f32);
        qweight_op.add_output(weight_s8);

        dqweight_op.add_input(weight_s8);
        dqweight_op.add_output(weight_f32_dq);

        tcdata_op.add_input(src_f32_dq);
        tcdata_op.add_output(src_bf16);

        tcweight_op.add_input(weight_f32_dq);
        tcweight_op.add_output(weight_bf16);

        conv_op.add_input(src_bf16);
        conv_op.add_input(weight_bf16);
        conv_op.add_output(conv_bf16);

        tc_bias_op.add_input(bias_f32);
        tc_bias_op.add_output(bias_bf16);

        biasadd_op.add_input(conv_bf16);
        biasadd_op.add_input(bias_bf16);
        biasadd_op.add_output(bias_out_bf16);

        tcdst_op.add_input(bias_out_bf16);
        tcdst_op.add_output(conv_f32);

        qout_op.add_input(conv_f32);
        qout_op.add_output(dst_u8);

        graph::graph_t g(engine->kind());
        ASSERT_EQ(g.add_op(&dqdata_op), graph::status::success);
        ASSERT_EQ(g.add_op(&qweight_op), graph::status::success);
        ASSERT_EQ(g.add_op(&dqweight_op), graph::status::success);
        ASSERT_EQ(g.add_op(&conv_op), graph::status::success);
        ASSERT_EQ(g.add_op(&biasadd_op), graph::status::success);
        ASSERT_EQ(g.add_op(&tcdata_op), graph::status::success);
        ASSERT_EQ(g.add_op(&tcweight_op), graph::status::success);
        ASSERT_EQ(g.add_op(&tc_bias_op), graph::status::success);
        ASSERT_EQ(g.add_op(&tcdst_op), graph::status::success);
        ASSERT_EQ(g.add_op(&qout_op), graph::status::success);
        g.finalize();

        graph::pass::pass_base_ptr apass = get_pass("x8s8x_tc_conv_post_ops");
        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 1U);
        auto part = g.get_partitions()[0];

        // compile
        graph::partition_t p;
        p.init(part);

        graph::compiled_partition_t cp(p);

        std::vector<const graph::logical_tensor_t *> lt_ins {
                &src_u8, &weight_f32, &bias_f32};
        std::vector<const graph::logical_tensor_t *> lt_outs {&dst_u8};

        ASSERT_EQ(p.compile(&cp, lt_ins, lt_outs, engine),
                graph::status::success);

        test::vector<uint8_t> dst_data(product(dst_shape));
        graph::tensor_t src_u8_ts(src_u8, engine, src_data.data());
        graph::tensor_t weight_f32_ts(weight_f32, engine, weight_data.data());
        graph::tensor_t bias_f32_ts(bias_f32, engine, bias_data.data());
        graph::tensor_t dst_ts(dst_u8, engine, dst_data.data());
        cp.execute(strm, {src_u8_ts, weight_f32_ts, bias_f32_ts}, {dst_ts});
        strm->wait();
    }
}

TEST(ExecuteSubgraphInt8, ConvolutionBiasGeluU8s8u8MixBf16) {
    using dims = dnnl::impl::graph::dnnl_impl::dims;
    graph::engine_t *engine = get_engine();
    graph::stream_t *strm = get_stream();

    std::string qtype = "per_channel";
    std::vector<int64_t> groups = {1, 4};
    std::vector<float> scales = {1.f, 1 / 255.f};
    std::vector<int64_t> zps = {0, 110};
    for_(const auto &scale : scales)
    for_(const auto &zp : zps)
    for (const auto &g_ : groups) {
        int64_t in_channel = 8, out_channel = 8;
        std::vector<int64_t> src_shape = {1, in_channel, 12, 12};
        std::vector<int64_t> weight_shape
                = {out_channel, in_channel / g_, 3, 3};
        std::vector<int64_t> bias_shape = {out_channel};
        std::vector<int64_t> dst_shape = {1, out_channel, 10, 10};

        test::vector<uint8_t> src_data(product(src_shape));
        test::vector<int8_t> weight_data(product(weight_shape));
        test::vector<float> bias_data(product(bias_shape));

        // random generate src, weight data
        // random seed = 7
        std::default_random_engine generator(7);
        std::uniform_real_distribution<float> distribution(0.0f, 255.0f);
        std::generate(src_data.begin(), src_data.end(), [&]() {
            return static_cast<uint8_t>(distribution(generator));
        });
        std::uniform_real_distribution<float> distribution2(-127.0f, 127.0f);
        std::generate(weight_data.begin(), weight_data.end(), [&]() {
            return static_cast<int8_t>(distribution2(generator));
        });
        std::uniform_real_distribution<float> distribution3(0.0f, 20.0f);
        std::generate(bias_data.begin(), bias_data.end(),
                [&]() { return distribution3(generator); });
        float scale_src = scale; // map to 0~255
        int64_t zp_src = zp;

        size_t scales_wei_sizes = qtype == "per_tensor" ? 1 : dst_shape.back();
        std::vector<float> scale_wei(scales_wei_sizes, 1 / 127.f);
        std::vector<int64_t> zp_wei(scales_wei_sizes, 0);

        float scale_out = scale; // map to 0~255
        int64_t zp_out = engine->kind() == graph::engine_kind::gpu ? 0 : zp;

        graph::op_t dqdata_op(0, graph::op_kind::Dequantize, "dqdata_op");
        dqdata_op.set_attr<std::string>(graph::op_attr::qtype, "per_tensor");
        dqdata_op.set_attr<std::vector<int64_t>>(graph::op_attr::zps, {zp_src});
        dqdata_op.set_attr<std::vector<float>>(
                graph::op_attr::scales, {scale_src});
        dqdata_op.set_attr<int64_t>(graph::op_attr::axis, 0);

        graph::op_t dqweight_op(1, graph::op_kind::Dequantize, "dqweight_op");
        dqweight_op.set_attr<std::string>(graph::op_attr::qtype, qtype);
        dqweight_op.set_attr<std::vector<int64_t>>(graph::op_attr::zps, zp_wei);
        dqweight_op.set_attr<std::vector<float>>(
                graph::op_attr::scales, scale_wei);
        dqweight_op.set_attr<int64_t>(graph::op_attr::axis, 1);

        graph::op_t tcdata_op {2, graph::op_kind::TypeCast, "typecast_data"};
        graph::op_t tcweight_op {
                3, graph::op_kind::TypeCast, "typecast_weight"};

        graph::op_t conv_op(4, graph::op_kind::Convolution, "conv_op");
        conv_op.set_attr<dims>(graph::op_attr::strides, dims(2, 1));
        conv_op.set_attr<dims>(graph::op_attr::dilations, dims(2, 1));
        conv_op.set_attr<dims>(graph::op_attr::pads_begin, dims(2, 0));
        conv_op.set_attr<dims>(graph::op_attr::pads_end, dims(2, 0));
        conv_op.set_attr<int64_t>(graph::op_attr::groups, g_);
        conv_op.set_attr<std::string>(graph::op_attr::data_format, "NCX");
        conv_op.set_attr<std::string>(graph::op_attr::weights_format, "OIX");

        graph::op_t gelu_op {5, graph::op_kind::GELU, "gelu_op"};

        graph::op_t tcdst_op {6, graph::op_kind::TypeCast, "typecast_dst"};

        graph::op_t qout_op(7, graph::op_kind::Quantize, "qdout_op");
        qout_op.set_attr<std::string>(graph::op_attr::qtype, "per_tensor");
        qout_op.set_attr<std::vector<int64_t>>(graph::op_attr::zps, {zp_out});
        qout_op.set_attr<std::vector<float>>(
                graph::op_attr::scales, {scale_out});
        qout_op.set_attr<int64_t>(graph::op_attr::axis, 1);

        // prepare logical tensor
        graph::logical_tensor_t src_u8 = utils::logical_tensor_init(
                0, src_shape, graph::data_type::u8);
        graph::logical_tensor_t src_f32_dq = utils::logical_tensor_init(
                1, src_shape, graph::data_type::f32);
        graph::logical_tensor_t src_bf16 = utils::logical_tensor_init(
                2, src_shape, graph::data_type::bf16);
        graph::logical_tensor_t weight_s8 = utils::logical_tensor_init(
                3, weight_shape, graph::data_type::s8);
        graph::logical_tensor_t weight_f32_dq = utils::logical_tensor_init(
                4, weight_shape, graph::data_type::f32);
        graph::logical_tensor_t weight_bf16 = utils::logical_tensor_init(
                5, weight_shape, graph::data_type::bf16);
        graph::logical_tensor_t bias_bf16 = utils::logical_tensor_init(
                6, bias_shape, graph::data_type::bf16);
        graph::logical_tensor_t conv_bf16 = utils::logical_tensor_init(
                7, dst_shape, graph::data_type::bf16);
        graph::logical_tensor_t gelu_out_bf16 = utils::logical_tensor_init(
                8, dst_shape, graph::data_type::bf16);
        graph::logical_tensor_t conv_f32 = utils::logical_tensor_init(
                9, dst_shape, graph::data_type::f32);
        graph::logical_tensor_t dst_u8 = utils::logical_tensor_init(
                10, dst_shape, graph::data_type::u8);

        dqdata_op.add_input(src_u8);
        dqdata_op.add_output(src_f32_dq);

        dqweight_op.add_input(weight_s8);
        dqweight_op.add_output(weight_f32_dq);

        tcdata_op.add_input(src_f32_dq);
        tcdata_op.add_output(src_bf16);

        tcweight_op.add_input(weight_f32_dq);
        tcweight_op.add_output(weight_bf16);

        conv_op.add_input(src_bf16);
        conv_op.add_input(weight_bf16);
        conv_op.add_input(bias_bf16);
        conv_op.add_output(conv_bf16);

        gelu_op.add_input(conv_bf16);
        gelu_op.add_output(gelu_out_bf16);

        tcdst_op.add_input(gelu_out_bf16);
        tcdst_op.add_output(conv_f32);

        qout_op.add_input(conv_f32);
        qout_op.add_output(dst_u8);

        graph::graph_t g(engine->kind());
        ASSERT_EQ(g.add_op(&dqdata_op), graph::status::success);
        ASSERT_EQ(g.add_op(&dqweight_op), graph::status::success);
        ASSERT_EQ(g.add_op(&conv_op), graph::status::success);
        ASSERT_EQ(g.add_op(&gelu_op), graph::status::success);
        ASSERT_EQ(g.add_op(&tcdata_op), graph::status::success);
        ASSERT_EQ(g.add_op(&tcweight_op), graph::status::success);
        ASSERT_EQ(g.add_op(&tcdst_op), graph::status::success);
        ASSERT_EQ(g.add_op(&qout_op), graph::status::success);
        g.finalize();

        graph::pass::pass_base_ptr apass = get_pass("x8s8x_tc_conv_post_ops");
        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 1U);
        auto part = g.get_partitions()[0];

        // compile
        graph::partition_t p;
        p.init(part);

        graph::compiled_partition_t cp(p);

        std::vector<const graph::logical_tensor_t *> lt_ins {
                &src_u8, &weight_s8, &bias_bf16};
        std::vector<const graph::logical_tensor_t *> lt_outs {&dst_u8};

        ASSERT_EQ(p.compile(&cp, lt_ins, lt_outs, engine),
                graph::status::success);

        test::vector<uint8_t> dst_data(product(dst_shape));
        graph::tensor_t src_u8_ts(src_u8, engine, src_data.data());
        graph::tensor_t weight_s8_ts(weight_s8, engine, weight_data.data());
        graph::tensor_t bias_bf16_ts(bias_bf16, engine, bias_data.data());
        graph::tensor_t dst_ts(dst_u8, engine, dst_data.data());
        cp.execute(strm, {src_u8_ts, weight_s8_ts, bias_bf16_ts}, {dst_ts});
        strm->wait();
    }
}

TEST(ExecuteSubgraphInt8, ConvolutionBiasaddGeluU8s8u8MixBf16) {
    using dims = dnnl::impl::graph::dnnl_impl::dims;
    graph::engine_t *engine = get_engine();
    graph::stream_t *strm = get_stream();

    std::string qtype = "per_channel";
    std::vector<int64_t> groups = {1, 4};
    for (const auto &g_ : groups) {
        int64_t in_channel = 8, out_channel = 8;
        std::vector<int64_t> src_shape = {1, in_channel, 12, 12};
        std::vector<int64_t> weight_shape
                = {out_channel, in_channel / g_, 3, 3};
        std::vector<int64_t> bias_shape = {out_channel};
        std::vector<int64_t> dst_shape = {1, out_channel, 10, 10};

        test::vector<uint8_t> src_data(product(src_shape));
        test::vector<float> weight_data(product(weight_shape));
        test::vector<float> bias_data(product(bias_shape));

        // random generate src, weight data
        // random seed = 7
        std::default_random_engine generator(7);
        std::uniform_real_distribution<float> distribution(0.0f, 255.0f);
        std::generate(src_data.begin(), src_data.end(), [&]() {
            return static_cast<uint8_t>(distribution(generator));
        });
        std::uniform_real_distribution<float> distribution2(-1.f, 1.f);
        std::generate(weight_data.begin(), weight_data.end(),
                [&]() { return distribution2(generator); });
        std::uniform_real_distribution<float> distribution3(0.0f, 20.0f);
        std::generate(bias_data.begin(), bias_data.end(),
                [&]() { return distribution3(generator); });
        float scale_src = 1 / 255.f; // map to 0~255
        int64_t zp_src = 110;

        size_t scales_wei_sizes = qtype == "per_tensor" ? 1 : dst_shape.back();
        std::vector<float> scale_wei(scales_wei_sizes, 1 / 127.f);
        std::vector<int64_t> zp_wei(scales_wei_sizes, 0);

        float scale_out = 1 / 255.f; // map to 0~255
        int64_t zp_out = engine->kind() == graph::engine_kind::gpu ? 0 : 110;

        graph::op_t dqdata_op(0, graph::op_kind::Dequantize, "dqdata_op");
        dqdata_op.set_attr<std::string>(graph::op_attr::qtype, "per_tensor");
        dqdata_op.set_attr<std::vector<int64_t>>(graph::op_attr::zps, {zp_src});
        dqdata_op.set_attr<std::vector<float>>(
                graph::op_attr::scales, {scale_src});
        dqdata_op.set_attr<int64_t>(graph::op_attr::axis, 0);

        graph::op_t qweight_op(10, graph::op_kind::Quantize, "qweight_op");
        qweight_op.set_attr<std::string>(graph::op_attr::qtype, qtype);
        qweight_op.set_attr<std::vector<int64_t>>(graph::op_attr::zps, zp_wei);
        qweight_op.set_attr<std::vector<float>>(
                graph::op_attr::scales, scale_wei);
        qweight_op.set_attr<int64_t>(graph::op_attr::axis, 1);

        graph::op_t dqweight_op(1, graph::op_kind::Dequantize, "dqweight_op");
        dqweight_op.set_attr<std::string>(graph::op_attr::qtype, qtype);
        dqweight_op.set_attr<std::vector<int64_t>>(graph::op_attr::zps, zp_wei);
        dqweight_op.set_attr<std::vector<float>>(
                graph::op_attr::scales, scale_wei);
        dqweight_op.set_attr<int64_t>(graph::op_attr::axis, 1);

        graph::op_t tcdata_op {2, graph::op_kind::TypeCast, "typecast_data"};
        graph::op_t tcweight_op {
                3, graph::op_kind::TypeCast, "typecast_weight"};

        graph::op_t conv_op(4, graph::op_kind::Convolution, "conv_op");
        conv_op.set_attr<dims>(graph::op_attr::strides, dims(2, 1));
        conv_op.set_attr<dims>(graph::op_attr::dilations, dims(2, 1));
        conv_op.set_attr<dims>(graph::op_attr::pads_begin, dims(2, 0));
        conv_op.set_attr<dims>(graph::op_attr::pads_end, dims(2, 0));
        conv_op.set_attr<int64_t>(graph::op_attr::groups, g_);
        conv_op.set_attr<std::string>(graph::op_attr::data_format, "NCX");
        conv_op.set_attr<std::string>(graph::op_attr::weights_format, "OIX");

        graph::op_t tc_bias_op {5, graph::op_kind::TypeCast, "typecast_bias"};

        graph::op_t biasadd_op {6, graph::op_kind::BiasAdd, "biasadd_op"};
        biasadd_op.set_attr<std::string>(graph::op_attr::data_format, "NCX");

        graph::op_t gelu_op {7, graph::op_kind::GELU, "gelu_op"};

        graph::op_t tcdst_op {8, graph::op_kind::TypeCast, "typecast_dst"};

        graph::op_t qout_op(9, graph::op_kind::Quantize, "qdout_op");
        qout_op.set_attr<std::string>(graph::op_attr::qtype, "per_tensor");
        qout_op.set_attr<std::vector<int64_t>>(graph::op_attr::zps, {zp_out});
        qout_op.set_attr<std::vector<float>>(
                graph::op_attr::scales, {scale_out});
        qout_op.set_attr<int64_t>(graph::op_attr::axis, 1);

        // prepare logical tensor
        graph::logical_tensor_t src_u8 = utils::logical_tensor_init(
                0, src_shape, graph::data_type::u8);
        graph::logical_tensor_t src_f32_dq = utils::logical_tensor_init(
                1, src_shape, graph::data_type::f32);
        graph::logical_tensor_t src_bf16 = utils::logical_tensor_init(
                2, src_shape, graph::data_type::bf16);
        graph::logical_tensor_t weight_f32 = utils::logical_tensor_init(
                30, weight_shape, graph::data_type::f32);
        graph::logical_tensor_t weight_s8 = utils::logical_tensor_init(
                3, weight_shape, graph::data_type::s8);
        graph::logical_tensor_t weight_f32_dq = utils::logical_tensor_init(
                4, weight_shape, graph::data_type::f32);
        graph::logical_tensor_t weight_bf16 = utils::logical_tensor_init(
                5, weight_shape, graph::data_type::bf16);
        graph::logical_tensor_t conv_bf16 = utils::logical_tensor_init(
                6, dst_shape, graph::data_type::bf16);
        graph::logical_tensor_t bias_f32 = utils::logical_tensor_init(
                7, bias_shape, graph::data_type::f32);
        graph::logical_tensor_t bias_bf16 = utils::logical_tensor_init(
                8, bias_shape, graph::data_type::bf16);
        graph::logical_tensor_t bias_out_bf16 = utils::logical_tensor_init(
                9, dst_shape, graph::data_type::bf16);
        graph::logical_tensor_t gelu_out_bf16 = utils::logical_tensor_init(
                10, dst_shape, graph::data_type::bf16);
        graph::logical_tensor_t conv_f32 = utils::logical_tensor_init(
                11, dst_shape, graph::data_type::f32);
        graph::logical_tensor_t dst_u8 = utils::logical_tensor_init(
                12, dst_shape, graph::data_type::u8);

        dqdata_op.add_input(src_u8);
        dqdata_op.add_output(src_f32_dq);

        qweight_op.add_input(weight_f32);
        qweight_op.add_output(weight_s8);

        dqweight_op.add_input(weight_s8);
        dqweight_op.add_output(weight_f32_dq);

        tcdata_op.add_input(src_f32_dq);
        tcdata_op.add_output(src_bf16);

        tcweight_op.add_input(weight_f32_dq);
        tcweight_op.add_output(weight_bf16);

        conv_op.add_input(src_bf16);
        conv_op.add_input(weight_bf16);
        conv_op.add_output(conv_bf16);

        tc_bias_op.add_input(bias_f32);
        tc_bias_op.add_output(bias_bf16);

        biasadd_op.add_input(conv_bf16);
        biasadd_op.add_input(bias_bf16);
        biasadd_op.add_output(bias_out_bf16);

        gelu_op.add_input(bias_out_bf16);
        gelu_op.add_output(gelu_out_bf16);

        tcdst_op.add_input(gelu_out_bf16);
        tcdst_op.add_output(conv_f32);

        qout_op.add_input(conv_f32);
        qout_op.add_output(dst_u8);

        graph::graph_t g(engine->kind());
        ASSERT_EQ(g.add_op(&dqdata_op), graph::status::success);
        ASSERT_EQ(g.add_op(&qweight_op), graph::status::success);
        ASSERT_EQ(g.add_op(&dqweight_op), graph::status::success);
        ASSERT_EQ(g.add_op(&conv_op), graph::status::success);
        ASSERT_EQ(g.add_op(&biasadd_op), graph::status::success);
        ASSERT_EQ(g.add_op(&gelu_op), graph::status::success);
        ASSERT_EQ(g.add_op(&tcdata_op), graph::status::success);
        ASSERT_EQ(g.add_op(&tcweight_op), graph::status::success);
        ASSERT_EQ(g.add_op(&tc_bias_op), graph::status::success);
        ASSERT_EQ(g.add_op(&tcdst_op), graph::status::success);
        ASSERT_EQ(g.add_op(&qout_op), graph::status::success);
        g.finalize();

        graph::pass::pass_base_ptr apass = get_pass("x8s8x_tc_conv_post_ops");
        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 1U);
        auto part = g.get_partitions()[0];

        // compile
        graph::partition_t p;
        p.init(part);

        graph::compiled_partition_t cp(p);

        std::vector<const graph::logical_tensor_t *> lt_ins {
                &src_u8, &weight_f32, &bias_f32};
        std::vector<const graph::logical_tensor_t *> lt_outs {&dst_u8};

        ASSERT_EQ(p.compile(&cp, lt_ins, lt_outs, engine),
                graph::status::success);

        test::vector<uint8_t> dst_data(product(dst_shape));
        graph::tensor_t src_u8_ts(src_u8, engine, src_data.data());
        graph::tensor_t weight_f32_ts(weight_f32, engine, weight_data.data());
        graph::tensor_t bias_f32_ts(bias_f32, engine, bias_data.data());
        graph::tensor_t dst_ts(dst_u8, engine, dst_data.data());
        cp.execute(strm, {src_u8_ts, weight_f32_ts, bias_f32_ts}, {dst_ts});
        strm->wait();
    }
}

TEST(ExecuteSubgraphInt8, ConvolutionAddU8s8u8MixBf16) {
    using dims = dnnl::impl::graph::dnnl_impl::dims;
    graph::engine_t *engine = get_engine();
    graph::stream_t *strm = get_stream();

    std::string qtype = "per_channel";
    int64_t in_channel = 8, out_channel = 8;
    std::vector<int64_t> src_shape = {1, in_channel, 12, 12};
    std::vector<int64_t> weight_shape = {out_channel, in_channel, 3, 3};
    std::vector<int64_t> bias_shape = {1, out_channel, 1, 1};
    std::vector<int64_t> dst_shape = {1, out_channel, 10, 10};

    test::vector<uint8_t> src_data(product(src_shape));
    test::vector<float> weight_data(product(weight_shape));
    test::vector<float> bias_data(product(bias_shape));

    // random generate src, weight data
    // random seed = 7
    std::default_random_engine generator(7);
    std::uniform_real_distribution<float> distribution(0.0f, 255.0f);
    std::generate(src_data.begin(), src_data.end(),
            [&]() { return static_cast<uint8_t>(distribution(generator)); });
    std::uniform_real_distribution<float> distribution2(-1.f, 1.f);
    std::generate(weight_data.begin(), weight_data.end(),
            [&]() { return distribution2(generator); });
    std::uniform_real_distribution<float> distribution3(0.0f, 20.0f);
    std::generate(bias_data.begin(), bias_data.end(),
            [&]() { return distribution3(generator); });
    float scale_src = 1 / 255.f; // map to 0~255
    int64_t zp_src = 110;

    size_t scales_wei_sizes = qtype == "per_tensor" ? 1 : dst_shape.back();
    std::vector<float> scale_wei(scales_wei_sizes, 1 / 127.f);
    std::vector<int64_t> zp_wei(scales_wei_sizes, 0);

    graph::op_t dqdata_op(0, graph::op_kind::Dequantize, "dqdata_op");
    dqdata_op.set_attr<std::string>(graph::op_attr::qtype, "per_tensor");
    dqdata_op.set_attr<std::vector<int64_t>>(graph::op_attr::zps, {zp_src});
    dqdata_op.set_attr<std::vector<float>>(graph::op_attr::scales, {scale_src});
    dqdata_op.set_attr<int64_t>(graph::op_attr::axis, 0);

    graph::op_t qweight_op(10, graph::op_kind::Quantize, "qweight_op");
    qweight_op.set_attr<std::string>(graph::op_attr::qtype, qtype);
    qweight_op.set_attr<std::vector<int64_t>>(graph::op_attr::zps, zp_wei);
    qweight_op.set_attr<std::vector<float>>(graph::op_attr::scales, scale_wei);
    qweight_op.set_attr<int64_t>(graph::op_attr::axis, 1);

    graph::op_t dqweight_op(1, graph::op_kind::Dequantize, "dqweight_op");
    dqweight_op.set_attr<std::string>(graph::op_attr::qtype, qtype);
    dqweight_op.set_attr<std::vector<int64_t>>(graph::op_attr::zps, zp_wei);
    dqweight_op.set_attr<std::vector<float>>(graph::op_attr::scales, scale_wei);
    dqweight_op.set_attr<int64_t>(graph::op_attr::axis, 1);

    graph::op_t tcdata_op {2, graph::op_kind::TypeCast, "typecast_data"};
    graph::op_t tcweight_op {3, graph::op_kind::TypeCast, "typecast_weight"};

    graph::op_t conv_op(4, graph::op_kind::Convolution, "conv_op");
    conv_op.set_attr<dims>(graph::op_attr::strides, dims(2, 1));
    conv_op.set_attr<dims>(graph::op_attr::dilations, dims(2, 1));
    conv_op.set_attr<dims>(graph::op_attr::pads_begin, dims(2, 0));
    conv_op.set_attr<dims>(graph::op_attr::pads_end, dims(2, 0));
    conv_op.set_attr<int64_t>(graph::op_attr::groups, 1);
    conv_op.set_attr<std::string>(graph::op_attr::data_format, "NCX");
    conv_op.set_attr<std::string>(graph::op_attr::weights_format, "OIX");

    graph::op_t tc_bias_op {5, graph::op_kind::TypeCast, "typecast_bias"};

    graph::op_t add_op {6, graph::op_kind::Add, "add_op"};

    // prepare logical tensor
    graph::logical_tensor_t src_u8
            = utils::logical_tensor_init(0, src_shape, graph::data_type::u8);
    graph::logical_tensor_t src_f32_dq
            = utils::logical_tensor_init(1, src_shape, graph::data_type::f32);
    graph::logical_tensor_t src_bf16
            = utils::logical_tensor_init(2, src_shape, graph::data_type::bf16);
    graph::logical_tensor_t weight_f32 = utils::logical_tensor_init(
            30, weight_shape, graph::data_type::f32);
    graph::logical_tensor_t weight_s8
            = utils::logical_tensor_init(3, weight_shape, graph::data_type::s8);
    graph::logical_tensor_t weight_f32_dq = utils::logical_tensor_init(
            4, weight_shape, graph::data_type::f32);
    graph::logical_tensor_t weight_bf16 = utils::logical_tensor_init(
            5, weight_shape, graph::data_type::bf16);
    graph::logical_tensor_t conv_bf16
            = utils::logical_tensor_init(6, dst_shape, graph::data_type::bf16);
    graph::logical_tensor_t bias_f32
            = utils::logical_tensor_init(7, bias_shape, graph::data_type::f32);
    graph::logical_tensor_t bias_bf16
            = utils::logical_tensor_init(8, bias_shape, graph::data_type::bf16);
    graph::logical_tensor_t bias_out_bf16
            = utils::logical_tensor_init(9, dst_shape, graph::data_type::bf16);

    dqdata_op.add_input(src_u8);
    dqdata_op.add_output(src_f32_dq);

    qweight_op.add_input(weight_f32);
    qweight_op.add_output(weight_s8);

    dqweight_op.add_input(weight_s8);
    dqweight_op.add_output(weight_f32_dq);

    tcdata_op.add_input(src_f32_dq);
    tcdata_op.add_output(src_bf16);

    tcweight_op.add_input(weight_f32_dq);
    tcweight_op.add_output(weight_bf16);

    conv_op.add_input(src_bf16);
    conv_op.add_input(weight_bf16);
    conv_op.add_output(conv_bf16);

    tc_bias_op.add_input(bias_f32);
    tc_bias_op.add_output(bias_bf16);

    add_op.add_input(conv_bf16);
    add_op.add_input(bias_bf16);
    add_op.add_output(bias_out_bf16);

    graph::graph_t g(engine->kind());
    ASSERT_EQ(g.add_op(&dqdata_op), graph::status::success);
    ASSERT_EQ(g.add_op(&qweight_op), graph::status::success);
    ASSERT_EQ(g.add_op(&dqweight_op), graph::status::success);
    ASSERT_EQ(g.add_op(&conv_op), graph::status::success);
    ASSERT_EQ(g.add_op(&add_op), graph::status::success);
    ASSERT_EQ(g.add_op(&tcdata_op), graph::status::success);
    ASSERT_EQ(g.add_op(&tcweight_op), graph::status::success);
    ASSERT_EQ(g.add_op(&tc_bias_op), graph::status::success);
    g.finalize();

    graph::pass::pass_base_ptr apass = get_pass("x8s8x_tc_conv_post_ops");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile
    graph::partition_t p;
    p.init(part);

    graph::compiled_partition_t cp(p);

    std::vector<const graph::logical_tensor_t *> lt_ins {
            &src_u8, &weight_f32, &bias_f32};
    std::vector<const graph::logical_tensor_t *> lt_outs {&bias_out_bf16};

    ASSERT_EQ(p.compile(&cp, lt_ins, lt_outs, engine), graph::status::success);

    test::vector<float> dst_data(product(dst_shape));
    graph::tensor_t src_u8_ts(src_u8, engine, src_data.data());
    graph::tensor_t weight_f32_ts(weight_f32, engine, weight_data.data());
    graph::tensor_t bias_f32_ts(bias_f32, engine, bias_data.data());
    graph::tensor_t dst_ts(bias_out_bf16, engine, dst_data.data());
    cp.execute(strm, {src_u8_ts, weight_f32_ts, bias_f32_ts}, {dst_ts});
    strm->wait();
}

TEST(Execute, ConvSumSum) {
    using dims = dnnl::impl::graph::dnnl_impl::dims;
    // default engine kind is cpu.
    graph::engine_t *eng = get_engine();
    test::vector<float> src1(1 * 8 * 112 * 112, 1.25f);
    test::vector<float> weight(8 * 8 * 3 * 3, 1.25f);
    test::vector<float> src2(1 * 8 * 110 * 110, 1.3f);
    test::vector<float> src3(1 * 8 * 110 * 110, 1.6f);
    test::vector<float> ref_dst(1 * 8 * 110 * 110, 115.4f);
    test::vector<float> dst(1 * 8 * 110 * 110, 0.f);

    graph::op_t in_op1(0, graph::op_kind::Wildcard, "Wildcard");
    graph::op_t in_op2(1, graph::op_kind::Wildcard, "Wildcard");
    graph::op_t in_op3(2, graph::op_kind::Wildcard, "Wildcard");
    graph::op_t conv_op(3, graph::op_kind::Convolution, "Convolution");
    conv_op.set_attr<dims>(graph::op_attr::strides, dims {1, 1});
    conv_op.set_attr<dims>(graph::op_attr::dilations, dims {1, 1});
    conv_op.set_attr<dims>(graph::op_attr::pads_begin, dims {0, 0});
    conv_op.set_attr<dims>(graph::op_attr::pads_end, dims {0, 0});
    conv_op.set_attr<int64_t>(graph::op_attr::groups, 1);
    conv_op.set_attr<std::string>(graph::op_attr::data_format, "NCX");
    conv_op.set_attr<std::string>(graph::op_attr::weights_format, "OIX");
    graph::op_t add_op1(4, graph::op_kind::Add, "Add");
    add_op1.set_attr<std::string>(graph::op_attr::auto_broadcast, "none");
    graph::op_t add_op2(5, graph::op_kind::Add, "Add");
    add_op2.set_attr<std::string>(graph::op_attr::auto_broadcast, "none");

    // prepare logical tensor
    graph::logical_tensor_t src1_lt = utils::logical_tensor_init(
            0, {1, 8, 112, 112}, graph::data_type::f32);
    graph::logical_tensor_t weight_lt = utils::logical_tensor_init(
            1, {8, 8, 3, 3}, graph::data_type::f32);

    graph::logical_tensor_t conv_lt = utils::logical_tensor_init(
            2, {1, 8, 110, 110}, graph::data_type::f32);
    graph::logical_tensor_t src2_lt = utils::logical_tensor_init(
            3, {1, 8, 110, 110}, graph::data_type::f32);

    graph::logical_tensor_t add1_lt = utils::logical_tensor_init(
            4, {1, 8, 110, 110}, graph::data_type::f32);
    graph::logical_tensor_t src3_lt = utils::logical_tensor_init(
            5, {1, 8, 110, 110}, graph::data_type::f32);

    graph::logical_tensor_t add2_lt = utils::logical_tensor_init(
            6, {1, 8, 110, 110}, graph::data_type::f32);

    in_op1.add_output(src1_lt);
    conv_op.add_input(src1_lt);
    conv_op.add_input(weight_lt);
    conv_op.add_output(conv_lt);

    add_op1.add_input(conv_lt);
    add_op1.add_input(src2_lt);
    add_op1.add_output(add1_lt);

    add_op2.add_input(add1_lt);
    add_op2.add_input(src3_lt);
    add_op2.add_output(add2_lt);

    graph::graph_t g(eng->kind());
    g.add_op(&in_op1);
    g.add_op(&in_op2);
    g.add_op(&in_op3);
    g.add_op(&conv_op);
    g.add_op(&add_op1);
    g.add_op(&add_op2);

    g.finalize();

    run_all_passes(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile
    graph::partition_t p;
    p.init(part);

    graph::compiled_partition_t cp(p);

    std::vector<const graph::logical_tensor_t *> inputs {
            &src1_lt, &weight_lt, &src2_lt, &src3_lt};
    std::vector<const graph::logical_tensor_t *> outputs {&add2_lt};

    p.compile(&cp, inputs, outputs, eng);

    graph::logical_tensor_t lt;
    cp.query_logical_tensor(add2_lt.id, &lt);
    ASSERT_EQ(lt.layout_type, graph::layout_type::strided);

    graph::tensor_t src1_ts(src1_lt, eng, src1.data());
    graph::tensor_t src2_ts(src2_lt, eng, src2.data());
    graph::tensor_t src3_ts(src3_lt, eng, src3.data());
    graph::tensor_t weight_ts(weight_lt, eng, weight.data());
    graph::tensor_t add2_ts(add2_lt, eng, dst.data());

    graph::stream_t *strm = get_stream();
    cp.execute(strm, {src1_ts, weight_ts, src2_ts, src3_ts}, {add2_ts});
    strm->wait();
}

TEST(Execute, ConvolutionBf16InFp32Out) {
    using dims = graph::dnnl_impl::dims;

    // default engine kind is cpu.
    graph::engine_t *eng = get_engine();
    graph::stream_t *strm = get_stream();

    static auto isa = dnnl_get_effective_cpu_isa();
    SKIP_IF((isa < dnnl_cpu_isa_avx512_core)
                    && eng->kind() == graph::engine_kind::cpu,
            "Skip bf16 examples for systems that do not support "
            "avx512_core.");

    graph::op_t conv_op(0, graph::op_kind::Convolution, "conv");
    conv_op.set_attr<dims>(graph::op_attr::strides, dims {1, 1});
    conv_op.set_attr<dims>(graph::op_attr::dilations, dims {1, 1});
    conv_op.set_attr<dims>(graph::op_attr::pads_begin, dims {0, 0});
    conv_op.set_attr<dims>(graph::op_attr::pads_end, dims {0, 0});
    conv_op.set_attr<int64_t>(graph::op_attr::groups, 1);
    conv_op.set_attr<std::string>(graph::op_attr::data_format, "NCX");
    conv_op.set_attr<std::string>(graph::op_attr::weights_format, "OIX");

    graph::op_t add_op(1, graph::op_kind::Add, "add");
    graph::op_t tc_op(2, graph::op_kind::TypeCast, "tc");

    // prepare logical tensor
    graph::logical_tensor_t conv_src_lt = utils::logical_tensor_init(
            0, {8, 32, 16, 16}, graph::data_type::bf16);
    graph::logical_tensor_t conv_weight_lt = utils::logical_tensor_init(
            1, {32, 32, 1, 1}, graph::data_type::bf16);
    graph::logical_tensor_t conv_dst_lt = utils::logical_tensor_init(
            2, {8, 32, 16, 16}, graph::data_type::bf16);
    graph::logical_tensor_t post_src_lt = utils::logical_tensor_init(
            3, {8, 32, 16, 16}, graph::data_type::bf16);
    graph::logical_tensor_t add_dst_lt = utils::logical_tensor_init(
            4, {8, 32, 16, 16}, graph::data_type::bf16);
    graph::logical_tensor_t tc_dst_lt = utils::logical_tensor_init(
            5, {8, 32, 16, 16}, graph::data_type::f32);

    // Initialize
    test::vector<uint16_t> conv_src(8 * 32 * 16 * 16, 3);
    test::vector<uint16_t> conv_weight(32 * 32 * 1 * 1, 2);
    test::vector<uint16_t> post_src(8 * 32 * 16 * 16, 1);
    test::vector<float> tc_dst(8 * 32 * 16 * 16, 0.0);

    graph::tensor_t conv_src_ts(conv_src_lt, eng, conv_src.data());
    graph::tensor_t conv_weight_ts(conv_weight_lt, eng, conv_weight.data());
    graph::tensor_t post_src_ts(post_src_lt, eng, post_src.data());
    graph::tensor_t tc_dst_ts(tc_dst_lt, eng, tc_dst.data());

    conv_op.add_input(conv_src_lt);
    conv_op.add_input(conv_weight_lt);
    conv_op.add_output(conv_dst_lt);
    add_op.add_input(conv_dst_lt);
    add_op.add_input(post_src_lt);
    add_op.add_output(add_dst_lt);
    tc_op.add_input(add_dst_lt);
    tc_op.add_output(tc_dst_lt);

    graph::graph_t g(eng->kind());
    g.add_op(&conv_op);
    g.add_op(&add_op);
    g.add_op(&tc_op);
    g.finalize();

    //run unfused graph to compute the reference
    ASSERT_EQ(run_graph(g, {conv_src_ts, conv_weight_ts, post_src_ts},
                      {tc_dst_ts}, *eng, *strm),
            graph::status::success);

    // run fusion partition
    graph::pass::pass_base_ptr apass = get_pass("fp_conv_post_ops");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];
    ASSERT_EQ(part->get_ops().size(), 3U);

    // compile
    graph::partition_t p;
    p.init(part);

    graph::compiled_partition_t cp(p);

    std::vector<const graph::logical_tensor_t *> inputs {
            &conv_src_lt, &conv_weight_lt, &post_src_lt};
    std::vector<const graph::logical_tensor_t *> outputs {&tc_dst_lt};

    p.compile(&cp, inputs, outputs, eng);

    test::vector<float> convtc_dst(8 * 32 * 16 * 16, 0.0);
    graph::tensor_t convtc_dst_ts(tc_dst_lt, eng, convtc_dst.data());

    cp.execute(
            strm, {conv_src_ts, conv_weight_ts, post_src_ts}, {convtc_dst_ts});
    strm->wait();

    for (size_t i = 0; i < tc_dst.size(); ++i) {
        ASSERT_NEAR(tc_dst[i], convtc_dst[i], 1e-10);
    }
}

TEST(ExecuteSubgraphInt8, QuantWeiConv2dSumRelu) {
    using dims = graph::dnnl_impl::dims;

    graph::engine_t *engine = get_engine();
    graph::stream_t *strm = get_stream();

    std::vector<int64_t> groups = {1, 4};
    std::vector<bool> with_biases = {true, false};
    std::vector<std::string> weight_qtypes = {"per_tensor", "per_channel"};
    std::vector<float> scales = {1.f, 1 / 127.f};
    std::vector<int64_t> zps = {0, 110};

    for_(const auto &scale : scales)
    for_(const auto &zp : zps)
    for_(const auto &g : groups)
    for_(const auto with_bias : with_biases)
    for (const auto &wei_qtype : weight_qtypes) {
        if (engine->kind() == graph::engine_kind::gpu
                && wei_qtype == "per_channel" && g == 4)
            continue;
        // prepare fp32 data
        int64_t in_channel = 8, out_channel = 8;
        int64_t kernel_size = 3;
        std::vector<int64_t> src_shape {1, in_channel, 112, 112};
        std::vector<int64_t> weight_shape {
                out_channel, in_channel / g, kernel_size, kernel_size};
        std::vector<int64_t> bias_shape {out_channel};
        std::vector<int64_t> dst_shape {1, out_channel, 110, 110};

        test::vector<uint8_t> src_u8_data(product(src_shape));
        test::vector<float> weight_f32_data(product(weight_shape));
        test::vector<int8_t> other_s8_data(product(dst_shape));
        size_t bias_size = with_bias ? product(bias_shape) : 0;
        test::vector<float> bias_data(bias_size);
        test::vector<int8_t> case1_out_data(product(dst_shape));
        test::vector<int8_t> case2_out_data(product(dst_shape));

        // random generate src, weight and bias data
        // random seed = 7
        std::default_random_engine generator(7);
        std::uniform_real_distribution<float> u8_distribution(0.0f, 127.0f);
        std::uniform_real_distribution<float> s8_distribution(0.0f, 128.0f);
        std::uniform_real_distribution<float> f32_distribution(0.0f, 1.0f);
        std::generate(src_u8_data.begin(), src_u8_data.end(), [&]() {
            return static_cast<uint8_t>(u8_distribution(generator));
        });
        std::generate(weight_f32_data.begin(), weight_f32_data.end(),
                [&]() { return f32_distribution(generator); });
        std::generate(other_s8_data.begin(), other_s8_data.end(), [&]() {
            return static_cast<int8_t>(s8_distribution(generator));
        });
        if (with_bias) {
            std::generate(bias_data.begin(), bias_data.end(),
                    [&]() { return f32_distribution(generator); });
        }

        float scale_src = scale; // map to 0~255
        float scale_other = scale;
        float scale_out = scale;
        int64_t zp_src = zp;
        int64_t zp_other = engine->kind() == graph::engine_kind::gpu ? 0 : zp;
        int64_t zp_out = zp;

        size_t scale_size = wei_qtype == "per_tensor" ? 1 : out_channel;

        std::vector<float> scale_wei(scale_size, scale);
        std::vector<int64_t> zp_wei(scale_size, 0);

        graph::op_t dqdata_node(1, graph::op_kind::Dequantize, "dqdata_node");
        SET_Q_DQ_DATA_ATTR(dqdata_node)

        graph::op_t qweight_node(2, graph::op_kind::Quantize, "qweight_node");
        SET_Q_DQ_WEIGHT_ATTR(qweight_node, 0)

        graph::op_t dqweight_node(
                3, graph::op_kind::Dequantize, "dqweight_node");
        SET_Q_DQ_WEIGHT_ATTR(dqweight_node, 0)

        graph::op_t conv_node(4, graph::op_kind::Convolution, "conv_node");
        SET_CONV_ATTR(conv_node, 2)

        graph::op_t relu_node(5, graph::op_kind::ReLU, "relu_node");

        graph::op_t qout_node(6, graph::op_kind::Quantize, "qout_node");
        SET_Q_DQ_OUT_ATTR(qout_node)

        graph::op_t dqother_node(8, graph::op_kind::Dequantize, "dqother_node");
        dqother_node.set_attr<std::string>(graph::op_attr::qtype, "per_tensor");
        dqother_node.set_attr<std::vector<int64_t>>(
                graph::op_attr::zps, {zp_other});
        dqother_node.set_attr<std::vector<float>>(
                graph::op_attr::scales, {scale_other});
        dqother_node.set_attr<int64_t>(graph::op_attr::axis, 0);

        graph::op_t add_node(9, graph::op_kind::Add, "add_node");

        // prepare logical tensor
        auto src_u8 = utils::logical_tensor_init(1, graph::data_type::u8);
        auto src_f32_dq = utils::logical_tensor_init(2, graph::data_type::f32);
        auto weight_f32 = utils::logical_tensor_init(3, graph::data_type::f32);
        auto weight_s8 = utils::logical_tensor_init(4, graph::data_type::s8);
        auto weight_f32_dq
                = utils::logical_tensor_init(5, graph::data_type::f32);
        auto dst_f32 = utils::logical_tensor_init(7, graph::data_type::f32);
        auto dst_relu_f32
                = utils::logical_tensor_init(8, graph::data_type::f32);
        auto dst_s8 = utils::logical_tensor_init(9, graph::data_type::s8);
        auto other_s8 = utils::logical_tensor_init(11, graph::data_type::s8);
        auto other_f32_dq
                = utils::logical_tensor_init(12, graph::data_type::f32);
        auto dst_add_f32
                = utils::logical_tensor_init(13, graph::data_type::f32);
        graph::logical_tensor_t bias_f32;
        if (with_bias) {
            bias_f32 = utils::logical_tensor_init(6, graph::data_type::f32);
        }

        dqdata_node.add_input(src_u8);
        dqdata_node.add_output(src_f32_dq);

        qweight_node.add_input(weight_f32);
        qweight_node.add_output(weight_s8);

        dqweight_node.add_input(weight_s8);
        dqweight_node.add_output(weight_f32_dq);

        conv_node.add_input(src_f32_dq);
        conv_node.add_input(weight_f32_dq);
        if (with_bias) conv_node.add_input(bias_f32);
        conv_node.add_output(dst_f32);

        dqother_node.add_input(other_s8);
        dqother_node.add_output(other_f32_dq);

        add_node.add_input(dst_f32);
        add_node.add_input(other_f32_dq);
        add_node.add_output(dst_add_f32);

        relu_node.add_input(dst_add_f32);
        relu_node.add_output(dst_relu_f32);

        qout_node.add_input(dst_relu_f32);
        qout_node.add_output(dst_s8);

        graph::graph_t g(engine->kind());
        g.add_op(&dqdata_node);
        g.add_op(&qweight_node);
        g.add_op(&dqweight_node);
        g.add_op(&conv_node);
        g.add_op(&dqother_node);
        g.add_op(&add_node);
        g.add_op(&relu_node);
        g.add_op(&qout_node);
        g.finalize();

        // prepare in/out with full shape
        src_u8 = utils::logical_tensor_init(1, src_shape, graph::data_type::u8);
        weight_f32 = utils::logical_tensor_init(
                3, weight_shape, graph::data_type::f32);
        // set weight to be constant
        weight_f32.property = graph::property_type::constant;
        dst_s8 = utils::logical_tensor_init(9, dst_shape, graph::data_type::s8);
        other_s8 = utils::logical_tensor_init(
                11, dst_shape, graph::data_type::s8);
        if (with_bias) {
            bias_f32 = utils::logical_tensor_init(
                    6, bias_shape, graph::data_type::f32);
            // set bias to be constant
            bias_f32.property = graph::property_type::constant;
        }

        graph::tensor_t src_u8_ts(src_u8, engine, src_u8_data.data());
        graph::tensor_t weight_f32_ts(
                weight_f32, engine, weight_f32_data.data());
        graph::tensor_t other_s8_ts(other_s8, engine, other_s8_data.data());
        graph::tensor_t bias_f32_ts;
        if (with_bias) {
            bias_f32_ts = graph::tensor_t(bias_f32, engine, bias_data.data());
        }
        graph::tensor_t dst_s8_ts(dst_s8, engine, case1_out_data.data());
        graph::tensor_t dst_s8_case2_ts(dst_s8, engine, case2_out_data.data());

        // -------------------------case 1----------------------------------
        ASSERT_EQ(run_graph(g,
                          {src_u8_ts, weight_f32_ts, bias_f32_ts, other_s8_ts},
                          {dst_s8_ts}, *engine, *strm),
                graph::status::success);

        // -------------------------case 2----------------------------------
        graph::pass::pass_base_ptr apass
                = get_pass(engine->kind() == graph::engine_kind::gpu
                                ? "x8s8x8_conv_add_post_ops_gpu"
                                : "x8s8x8_conv_add_post_ops_cpu");

        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 1U);
        auto part = g.get_partitions()[0];

        // compile
        graph::partition_t p;
        p.init(part);

        graph::compiled_partition_t cp(p);

        std::vector<const graph::logical_tensor_t *> lt_ins;
        if (with_bias)
            lt_ins = {&src_u8, &weight_f32, &bias_f32, &other_s8};
        else
            lt_ins = {&src_u8, &weight_f32, &other_s8};
        std::vector<const graph::logical_tensor_t *> lt_outs {&dst_s8};

        graph::status_t ret = p.compile(&cp, lt_ins, lt_outs, engine);
        ASSERT_EQ(ret, graph::status::success);

        // single thread
        for (size_t iter = 0; iter < 5; iter++) {
            if (with_bias)
                cp.execute(strm,
                        {src_u8_ts, weight_f32_ts, bias_f32_ts, other_s8_ts},
                        {dst_s8_case2_ts});
            else
                cp.execute(strm, {src_u8_ts, weight_f32_ts, other_s8_ts},
                        {dst_s8_case2_ts});
            strm->wait();

            static auto isa = dnnl_get_effective_cpu_isa();
            if (engine->kind() == graph::engine_kind::cpu
                    && isa < dnnl_cpu_isa_avx512_core_vnni)
                ASSERT_TRUE(allclose(case1_out_data, case2_out_data,
                        /*rtol*/ 0.1f,
                        /*atol*/ 1.f));
            else
                ASSERT_TRUE(allclose(case1_out_data, case2_out_data,
                        /*rtol*/ 0.01f,
                        /*atol*/ 1.f));
        }
    }
}

TEST(ExecuteSubgraphInt8, QuantWeiConv2dSumS8Relu) {
    static auto isa = dnnl_get_effective_cpu_isa();
    using dims = graph::dnnl_impl::dims;

    graph::engine_t *engine = get_engine();
    graph::stream_t *strm = get_stream();

    SKIP_IF(isa < dnnl_cpu_isa_avx512_core_vnni
                    && engine->kind() == graph::engine_kind::cpu,
            "Skip u8+s8 examples for systems that do not support "
            "avx512_core_vnni.");

    std::vector<int64_t> groups = {1};
    std::vector<bool> with_biases = {true};
    std::vector<std::string> weight_qtypes = {"per_channel"};

    for_(const auto &g : groups)
    for_(const auto with_bias : with_biases)
    for (const auto &wei_qtype : weight_qtypes) {
        // prepare fp32 data
        int64_t in_channel = 8, out_channel = 8;
        int64_t kernel_size = 3;
        std::vector<int64_t> src_shape {1, in_channel, 112, 112};
        std::vector<int64_t> weight_shape {
                out_channel, in_channel / g, kernel_size, kernel_size};
        std::vector<int64_t> bias_shape {out_channel};
        std::vector<int64_t> dst_shape {1, out_channel, 110, 110};
        std::vector<int64_t> nxc_stride {
                out_channel * 110 * 110, 1, out_channel * 110, out_channel};

        test::vector<uint8_t> src_u8_data(product(src_shape));
        test::vector<float> weight_f32_data(product(weight_shape));
        test::vector<int8_t> other_s8_data(product(dst_shape));
        size_t bias_size = with_bias ? product(bias_shape) : 0;
        test::vector<float> bias_data(bias_size);
        test::vector<uint8_t> case1_out_data(product(dst_shape));
        test::vector<uint8_t> case2_out_data(product(dst_shape));

        // random generate src, weight and bias data
        // random seed = 7
        std::default_random_engine generator(7);
        std::uniform_real_distribution<float> u8_distribution(0.0f, 255.0f);
        std::uniform_real_distribution<float> s8_distribution(-127.0f, 128.0f);
        std::uniform_real_distribution<float> f32_distribution(0.0f, 1.0f);
        std::generate(src_u8_data.begin(), src_u8_data.end(), [&]() {
            return static_cast<uint8_t>(u8_distribution(generator));
        });
        std::generate(weight_f32_data.begin(), weight_f32_data.end(),
                [&]() { return f32_distribution(generator); });
        std::generate(other_s8_data.begin(), other_s8_data.end(), [&]() {
            return static_cast<int8_t>(s8_distribution(generator));
        });
        if (with_bias) {
            std::generate(bias_data.begin(), bias_data.end(),
                    [&]() { return f32_distribution(generator); });
        }

        float scale_src = 1 / 255.f; // map to 0~255
        float scale_other = 1 / 127.f;
        float scale_out = 1 / 255.f;
        int64_t zp_src = 0;
        int64_t zp_other = 0;
        int64_t zp_out = 78;

        size_t scale_size = wei_qtype == "per_tensor" ? 1 : out_channel;

        std::vector<float> scale_wei(scale_size, 1 / 127.f);
        std::vector<int64_t> zp_wei(scale_size, 0);

        graph::op_t dqdata_node(1, graph::op_kind::Dequantize, "dqdata_node");
        SET_Q_DQ_DATA_ATTR(dqdata_node)

        graph::op_t qweight_node(2, graph::op_kind::Quantize, "qweight_node");
        SET_Q_DQ_WEIGHT_ATTR(qweight_node, 0)

        graph::op_t dqweight_node(
                3, graph::op_kind::Dequantize, "dqweight_node");
        SET_Q_DQ_WEIGHT_ATTR(dqweight_node, 0)

        graph::op_t conv_node(4, graph::op_kind::Convolution, "conv_node");
        SET_CONV_ATTR(conv_node, 2)

        graph::op_t relu_node(5, graph::op_kind::ReLU, "relu_node");

        graph::op_t qout_node(6, graph::op_kind::Quantize, "qout_node");
        SET_Q_DQ_OUT_ATTR(qout_node)

        graph::op_t dqother_node(8, graph::op_kind::Dequantize, "dqother_node");
        dqother_node.set_attr<std::string>(graph::op_attr::qtype, "per_tensor");
        dqother_node.set_attr<std::vector<int64_t>>(
                graph::op_attr::zps, {zp_other});
        dqother_node.set_attr<std::vector<float>>(
                graph::op_attr::scales, {scale_other});
        dqother_node.set_attr<int64_t>(graph::op_attr::axis, 0);

        graph::op_t add_node(9, graph::op_kind::Add, "add_node");

        // prepare logical tensor
        auto src_u8 = utils::logical_tensor_init(1, graph::data_type::u8);
        auto src_f32_dq = utils::logical_tensor_init(2, graph::data_type::f32);
        auto weight_f32 = utils::logical_tensor_init(3, graph::data_type::f32);
        auto weight_s8 = utils::logical_tensor_init(4, graph::data_type::s8);
        auto weight_f32_dq
                = utils::logical_tensor_init(5, graph::data_type::f32);
        auto dst_f32 = utils::logical_tensor_init(7, graph::data_type::f32);
        auto dst_relu_f32
                = utils::logical_tensor_init(8, graph::data_type::f32);
        auto dst_u8 = utils::logical_tensor_init(9, graph::data_type::u8);
        auto other_s8 = utils::logical_tensor_init(11, graph::data_type::s8);
        auto other_f32_dq
                = utils::logical_tensor_init(12, graph::data_type::f32);
        auto dst_add_f32
                = utils::logical_tensor_init(13, graph::data_type::f32);
        graph::logical_tensor_t bias_f32;
        if (with_bias) {
            bias_f32 = utils::logical_tensor_init(6, graph::data_type::f32);
        }

        dqdata_node.add_input(src_u8);
        dqdata_node.add_output(src_f32_dq);

        qweight_node.add_input(weight_f32);
        qweight_node.add_output(weight_s8);

        dqweight_node.add_input(weight_s8);
        dqweight_node.add_output(weight_f32_dq);

        conv_node.add_input(src_f32_dq);
        conv_node.add_input(weight_f32_dq);
        if (with_bias) conv_node.add_input(bias_f32);
        conv_node.add_output(dst_f32);

        dqother_node.add_input(other_s8);
        dqother_node.add_output(other_f32_dq);

        add_node.add_input(dst_f32);
        add_node.add_input(other_f32_dq);
        add_node.add_output(dst_add_f32);

        relu_node.add_input(dst_add_f32);
        relu_node.add_output(dst_relu_f32);

        qout_node.add_input(dst_relu_f32);
        qout_node.add_output(dst_u8);

        graph::graph_t g(engine->kind());
        g.add_op(&dqdata_node);
        g.add_op(&qweight_node);
        g.add_op(&dqweight_node);
        g.add_op(&conv_node);
        g.add_op(&dqother_node);
        g.add_op(&add_node);
        g.add_op(&relu_node);
        g.add_op(&qout_node);
        g.finalize();

        // prepare in/out with full shape
        src_u8 = utils::logical_tensor_init(1, src_shape, graph::data_type::u8);
        weight_f32 = utils::logical_tensor_init(
                3, weight_shape, graph::data_type::f32);
        // set weight to be constant
        weight_f32.property = graph::property_type::constant;
        dst_u8 = utils::logical_tensor_init(
                9, dst_shape, nxc_stride, graph::data_type::u8);
        other_s8 = utils::logical_tensor_init(
                11, dst_shape, nxc_stride, graph::data_type::s8);
        if (with_bias) {
            bias_f32 = utils::logical_tensor_init(
                    6, bias_shape, graph::data_type::f32);
            // set bias to be constant
            bias_f32.property = graph::property_type::constant;
        }

        graph::tensor_t src_u8_ts(src_u8, engine, src_u8_data.data());
        graph::tensor_t weight_f32_ts(
                weight_f32, engine, weight_f32_data.data());
        graph::tensor_t other_s8_ts(other_s8, engine, other_s8_data.data());
        graph::tensor_t bias_f32_ts;
        if (with_bias) {
            bias_f32_ts = graph::tensor_t(bias_f32, engine, bias_data.data());
        }
        graph::tensor_t dst_u8_ts(dst_u8, engine, case1_out_data.data());
        graph::tensor_t dst_u8_case2_ts(dst_u8, engine, case2_out_data.data());

        // -------------------------case 1----------------------------------
        ASSERT_EQ(run_graph(g,
                          {src_u8_ts, weight_f32_ts, bias_f32_ts, other_s8_ts},
                          {dst_u8_ts}, *engine, *strm),
                graph::status::success);

        // -------------------------case 2----------------------------------
        graph::pass::pass_base_ptr apass
                = get_pass(engine->kind() == graph::engine_kind::gpu
                                ? "x8s8x8_conv_add_post_ops_gpu"
                                : "x8s8x8_conv_add_post_ops_cpu");

        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 1U);
        auto part = g.get_partitions()[0];

        // compile
        graph::partition_t p;
        p.init(part);

        graph::compiled_partition_t cp(p);

        std::vector<const graph::logical_tensor_t *> lt_ins;
        if (with_bias)
            lt_ins = {&src_u8, &weight_f32, &bias_f32, &other_s8};
        else
            lt_ins = {&src_u8, &weight_f32, &other_s8};
        std::vector<const graph::logical_tensor_t *> lt_outs {&dst_u8};

        graph::status_t ret = p.compile(&cp, lt_ins, lt_outs, engine);
        ASSERT_EQ(ret, graph::status::success);

        std::vector<graph::inplace_pair_t> inplace_pairs
                = cp.get_inplace_pairs();
        ASSERT_EQ(inplace_pairs.size(), 1U);
        ASSERT_EQ(inplace_pairs[0].input_id, other_s8.id);
        ASSERT_EQ(inplace_pairs[0].output_id, dst_u8.id);

        // single thread
        for (size_t iter = 0; iter < 5; iter++) {
            if (with_bias)
                cp.execute(strm,
                        {src_u8_ts, weight_f32_ts, bias_f32_ts, other_s8_ts},
                        {dst_u8_case2_ts});
            else
                cp.execute(strm, {src_u8_ts, weight_f32_ts, other_s8_ts},
                        {dst_u8_case2_ts});
            strm->wait();
            ASSERT_TRUE(allclose(case1_out_data, case2_out_data, /*rtol*/ 0.01f,
                    /*atol*/ 1.f));
        }
    }
}

TEST(Execute, ConvReluUnfused) {
    using dims = graph::dnnl_impl::dims;

    // default engine kind is cpu.
    graph::engine_t *eng = get_engine();
    graph::stream_t *strm = get_stream();

    graph::op_t conv_op(0, graph::op_kind::Convolution, "conv");
    conv_op.set_attr<dims>(graph::op_attr::strides, dims {1, 1});
    conv_op.set_attr<dims>(graph::op_attr::dilations, dims {1, 1});
    conv_op.set_attr<dims>(graph::op_attr::pads_begin, dims {0, 0});
    conv_op.set_attr<dims>(graph::op_attr::pads_end, dims {0, 0});
    conv_op.set_attr<int64_t>(graph::op_attr::groups, 1);
    conv_op.set_attr<std::string>(graph::op_attr::data_format, "NCX");
    conv_op.set_attr<std::string>(graph::op_attr::weights_format, "OIX");

    graph::op_t relu_op(1, graph::op_kind::ReLU, "relu");

    // prepare logical tensor
    graph::logical_tensor_t conv_src_lt = utils::logical_tensor_init(
            0, {8, 32, 16, 16}, graph::data_type::f32);
    graph::logical_tensor_t conv_weight_lt = utils::logical_tensor_init(
            1, {32, 32, 1, 1}, graph::data_type::f32);
    graph::logical_tensor_t conv_dst_lt = utils::logical_tensor_init(
            2, {8, 32, 16, 16}, graph::data_type::f32, graph::layout_type::any);
    graph::logical_tensor_t relu_dst_lt = utils::logical_tensor_init(
            3, {8, 32, 16, 16}, graph::data_type::f32);

    conv_op.add_input(conv_src_lt);
    conv_op.add_input(conv_weight_lt);
    conv_op.add_output(conv_dst_lt);
    relu_op.add_input(conv_dst_lt);
    relu_op.add_output(relu_dst_lt);

    graph::graph_t g(eng->kind());
    g.add_op(&conv_op);
    g.add_op(&relu_op);
    g.finalize();

    test::vector<float> conv_src(8 * 32 * 16 * 16, 1);
    test::vector<float> conv_weight(32 * 32 * 1 * 1, 1);
    test::vector<float> relu_dst(8 * 32 * 16 * 16, 1);

    graph::tensor_t conv_src_ts(conv_src_lt, eng, conv_src.data());
    graph::tensor_t conv_weight_ts(conv_weight_lt, eng, conv_weight.data());
    graph::tensor_t relu_dst_ts(relu_dst_lt, eng, relu_dst.data());

    // run unfused graph to compute the reference
    ASSERT_EQ(run_graph(g, {conv_src_ts, conv_weight_ts}, {relu_dst_ts}, *eng,
                      *strm),
            graph::status::success);

    for (size_t i = 0; i < relu_dst.size(); ++i) {
        ASSERT_FLOAT_EQ(relu_dst[i], 32);
    }
}

TEST(ExecuteSubgraphInt8, ConvDepthwise) {
    graph::engine_t *eng = get_engine();
    graph::stream_t *strm = get_stream();

    std::vector<std::string> weight_qtypes = {"per_tensor", "per_channel"};
    std::vector<std::string> weight_formats = {"OIX", "XIO"};
    std::string dw_type {"k3s2p1"};

    std::vector<int64_t> dw_src_shape, dw_wei_shape, dw_dst_shape,
            reshape1_shape, reshape2_shape;
    int64_t groups = 3, axis;
    for_(const auto &weight_format : weight_formats)
    for (const auto &wei_qtype : weight_qtypes) {
        if (weight_format == "OIX") {
            // N, IC, OH, OW
            dw_src_shape = {4, 3, 8, 8};
            // OC/G, IC, KH, KW
            dw_wei_shape = {8, 3, 7, 7};
            // N, OC, OH, OW
            dw_dst_shape = {4, 24, 2, 2};
            reshape1_shape = {24, 7, 7};
            reshape2_shape = {24, 1, 7, 7};
            axis = 0;
        } else {
            // N, OH, OW, IC
            dw_src_shape = {4, 8, 8, 3};
            // KH, KW, IC, OC/G
            dw_wei_shape = {7, 7, 3, 8};
            // N, OH, OW, OC
            dw_dst_shape = {4, 2, 2, 24};
            reshape1_shape = {7, 7, 24};
            reshape2_shape = {7, 7, 1, 24};
            axis = 2;
        }
        test::vector<uint8_t> src_u8_data(product(dw_src_shape));
        test::vector<float> weight_f32_data(product(dw_wei_shape));
        size_t bias_size = 0;
        test::vector<float> bias_data(bias_size);
        test::vector<int8_t> case1_out_data(product(dw_dst_shape));
        test::vector<int8_t> case2_out_data(product(dw_dst_shape));

        // random generate src, weight and bias data random seed = 7
        std::default_random_engine generator(7);
        std::uniform_real_distribution<float> u8_distribution(0.0f, 255.0f);
        std::uniform_real_distribution<float> s8_distribution(-127.0f, 128.0f);
        std::uniform_real_distribution<float> f32_distribution(0.0f, 1.0f);
        std::generate(src_u8_data.begin(), src_u8_data.end(), [&]() {
            return static_cast<uint8_t>(u8_distribution(generator));
        });
        std::generate(weight_f32_data.begin(), weight_f32_data.end(),
                [&]() { return f32_distribution(generator); });
        float scale_src = 1 / 255.f; // map to 0~255
        float scale_out = 1;
        int64_t zp_src = 0;

        int64_t zp_out = eng->kind() == graph::engine_kind::gpu ? 0 : 78;

        size_t scale_size = wei_qtype == "per_tensor" ? 1 : 24;
        std::vector<float> scale_wei(scale_size, 1 / 127.f);
        std::vector<int64_t> zp_wei(scale_size, 0);

        graph::op_t dqdata_node(1, graph::op_kind::Dequantize, "dqdata_node");
        SET_Q_DQ_DATA_ATTR(dqdata_node)

        graph::op_t reshape1_op(
                2, graph::op_kind::StaticReshape, "reshape1_op");
        reshape1_op.set_attr(graph::op_attr::shape, reshape1_shape);
        reshape1_op.set_attr(graph::op_attr::special_zero, false);

        graph::op_t qweight_node(3, graph::op_kind::Quantize, "qweight_node");
        SET_Q_DQ_WEIGHT_ATTR(qweight_node, axis);

        graph::op_t dqweight_node(
                4, graph::op_kind::Dequantize, "dqweight_node");
        SET_Q_DQ_WEIGHT_ATTR(dqweight_node, axis);

        graph::op_t reshape2_op(
                5, graph::op_kind::StaticReshape, "reshape2_op");
        reshape2_op.set_attr(graph::op_attr::shape, reshape2_shape);
        reshape2_op.set_attr(graph::op_attr::special_zero, false);

        graph::op_t depthwise {6, graph::op_kind::Convolution, "depthwise"};
        utils::set_conv_dw_post_op_attr(depthwise, dw_type);
        depthwise.set_attr(graph::op_attr::weights_format, weight_format);
        depthwise.set_attr(graph::op_attr::groups, groups);

        graph::op_t qout_node(7, graph::op_kind::Quantize, "qout_node");
        SET_Q_DQ_OUT_ATTR(qout_node)
        if (weight_format == "XIO") {
            std::string data_format = "NXC";
            depthwise.set_attr(graph::op_attr::data_format, data_format);
            dqdata_node.set_attr<int64_t>(
                    dnnl::impl::graph::op_attr::axis, axis);
            qout_node.set_attr<int64_t>(dnnl::impl::graph::op_attr::axis, axis);
        }

        // prepare logical tensor
        graph::logical_tensor_t src_u8 = utils::logical_tensor_init(
                1, dw_src_shape, graph::data_type::u8);
        graph::logical_tensor_t src_f32_dq = utils::logical_tensor_init(
                2, dw_src_shape, graph::data_type::f32);
        graph::logical_tensor_t weight_f32 = utils::logical_tensor_init(
                3, dw_wei_shape, graph::data_type::f32);
        graph::logical_tensor_t weight_f32_reshape1
                = utils::logical_tensor_init(
                        4, reshape1_shape, graph::data_type::f32);
        graph::logical_tensor_t weight_s8_q = utils::logical_tensor_init(
                5, reshape1_shape, graph::data_type::s8);
        graph::logical_tensor_t weight_f32_dq = utils::logical_tensor_init(
                6, reshape1_shape, graph::data_type::f32);
        graph::logical_tensor_t weight_f32_reshape2
                = utils::logical_tensor_init(
                        7, reshape2_shape, graph::data_type::f32);
        graph::logical_tensor_t dst_f32 = utils::logical_tensor_init(
                8, dw_dst_shape, graph::data_type::f32);
        graph::logical_tensor_t dst_s8 = utils::logical_tensor_init(
                9, dw_dst_shape, graph::data_type::s8);

        dqdata_node.add_input(src_u8);
        dqdata_node.add_output(src_f32_dq);

        reshape1_op.add_input(weight_f32);
        reshape1_op.add_output(weight_f32_reshape1);

        qweight_node.add_input(weight_f32_reshape1);
        qweight_node.add_output(weight_s8_q);

        dqweight_node.add_input(weight_s8_q);
        dqweight_node.add_output(weight_f32_dq);

        reshape2_op.add_input(weight_f32_dq);
        reshape2_op.add_output(weight_f32_reshape2);

        depthwise.add_input(src_f32_dq);
        depthwise.add_input(weight_f32_reshape2);
        depthwise.add_output(dst_f32);

        qout_node.add_input(dst_f32);
        qout_node.add_output(dst_s8);

        graph::graph_t g(eng->kind());
        g.add_op(&dqdata_node);
        g.add_op(&reshape1_op);
        g.add_op(&qweight_node);
        g.add_op(&dqweight_node);
        g.add_op(&reshape2_op);
        g.add_op(&depthwise);
        g.add_op(&qout_node);
        g.finalize();

        graph::tensor_t src_u8_ts(src_u8, eng, src_u8_data.data());
        graph::tensor_t weight_f32_ts(weight_f32, eng, weight_f32_data.data());
        graph::tensor_t dst_s8_ts(dst_s8, eng, case1_out_data.data());
        graph::tensor_t dst_s8_case2_ts(dst_s8, eng, case2_out_data.data());

        // -------------------------case 2----------------------------------
        graph::pass::pass_base_ptr apass
                = get_pass("x8s8x_conv_reshape_post_ops");
        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 1U);

        auto part = g.get_partitions()[0];

        // compile
        graph::partition_t p;
        p.init(part);

        graph::compiled_partition_t cp(p);

        std::vector<const graph::logical_tensor_t *> lt_ins;
        lt_ins = {&src_u8, &weight_f32};
        std::vector<const graph::logical_tensor_t *> lt_outs {&dst_s8};

        p.compile(&cp, lt_ins, lt_outs, eng);
        cp.execute(strm, {src_u8_ts, weight_f32_ts}, {dst_s8_case2_ts});
        strm->wait();
    }
}

TEST(ExecuteSubgraphInt8, ShareCachedWeights) {
    using dims = graph::dnnl_impl::dims;

    graph::engine_t *engine = get_engine();
    graph::stream_t *strm = get_stream();

    int64_t g = 1;
    std::string wei_qtype = "per_channel";
    std::string src_qtype = "symmetric";

    float scale_src = 1 / 255.f;
    int64_t zp_src = 0;
    std::vector<int64_t> weight_shape {8, 8, 3, 3};

    size_t scale_size = 8;
    std::vector<float> scale_wei(scale_size, 1 / 127.f);
    std::vector<int64_t> zp_wei(scale_size, 0);

    graph::op_t dqdata_node(1, graph::op_kind::Dequantize, "dqdata_node");
    SET_Q_DQ_DATA_ATTR(dqdata_node)

    graph::op_t dqweight_node(3, graph::op_kind::Dequantize, "dqweight_node");
    SET_Q_DQ_WEIGHT_ATTR(dqweight_node, 0)

    graph::op_t conv_node(4, graph::op_kind::Convolution, "conv_node");
    SET_CONV_ATTR(conv_node, 2)

    auto src_u8 = utils::logical_tensor_init(1, graph::data_type::u8);
    auto src_f32_dq = utils::logical_tensor_init(2, graph::data_type::f32);
    auto weight_s8
            = utils::logical_tensor_init(4, weight_shape, graph::data_type::s8);
    weight_s8.property = graph::property_type::constant;
    auto weight_f32_dq = utils::logical_tensor_init(
            5, weight_shape, graph::data_type::f32);
    auto dst_f32 = utils::logical_tensor_init(7, graph::data_type::f32);

    dqdata_node.add_input(src_u8);
    dqdata_node.add_output(src_f32_dq);

    dqweight_node.add_input(weight_s8);
    dqweight_node.add_output(weight_f32_dq);

    conv_node.add_input(src_f32_dq);
    conv_node.add_input(weight_f32_dq);
    conv_node.add_output(dst_f32);

    graph::graph_t agraph(engine->kind());
    agraph.add_op(&dqdata_node);
    agraph.add_op(&dqweight_node);
    agraph.add_op(&conv_node);
    agraph.finalize();

    graph::pass::pass_base_ptr apass = get_pass("x8s8x_conv_post_ops");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1U);
    auto part = agraph.get_partitions()[0];

    graph::partition_t p;
    p.init(part);

    std::default_random_engine generator(7);
    std::uniform_real_distribution<float> u8_distribution(0.0f, 100.0f);
    std::uniform_real_distribution<float> s8_distribution(-64.0f, 64.0f);
    std::uniform_real_distribution<float> f32_distribution(0.0f, 1.0f);

    test::vector<int8_t> weight_s8_data(product(weight_shape));
    std::generate(weight_s8_data.begin(), weight_s8_data.end(),
            [&]() { return static_cast<int8_t>(s8_distribution(generator)); });
    graph::tensor_t weight_s8_ts(weight_s8, engine, weight_s8_data.data());

    std::vector<std::vector<int64_t>> src_shapes {
            {1, 8, 12, 12}, {3, 8, 12, 12}};
    std::vector<std::vector<int64_t>> dst_shapes {
            {1, 8, 10, 10}, {3, 8, 10, 10}};
    std::vector<std::shared_ptr<graph::compiled_partition_t>> cps;
    for (size_t i = 0; i < src_shapes.size(); ++i) {
        std::cout << "---------------\n";
        std::vector<int64_t> src_shape = src_shapes[i];
        std::vector<int64_t> dst_shape = dst_shapes[i];

        src_u8 = utils::logical_tensor_init(1, src_shape, graph::data_type::u8);
        dst_f32 = utils::logical_tensor_init(
                7, dst_shape, graph::data_type::f32);

        std::vector<const graph::logical_tensor_t *> lt_ins {
                &src_u8, &weight_s8};
        std::vector<const graph::logical_tensor_t *> lt_outs {&dst_f32};

        cps.push_back(std::make_shared<graph::compiled_partition_t>(p));
        auto &cp = *cps.back();
        ASSERT_EQ(p.compile(&cp, lt_ins, lt_outs, engine),
                graph::status::success);

        test::vector<uint8_t> src_u8_data(product(src_shape));
        std::generate(src_u8_data.begin(), src_u8_data.end(), [&]() {
            return static_cast<uint8_t>(u8_distribution(generator));
        });
        graph::tensor_t src_u8_ts(src_u8, engine, src_u8_data.data());

        test::vector<float> dst_data(product(dst_shape));
        graph::tensor_t dst_ts(dst_f32, engine, dst_data.data());

        cp.execute(strm, {src_u8_ts, weight_s8_ts}, {dst_ts});
        strm->wait();

        test::vector<float> ref_dst_data(product(dst_shape));
        graph::tensor_t ref_dst_ts(dst_f32, engine, ref_dst_data.data());

        ASSERT_EQ(run_graph(agraph, {src_u8_ts, weight_s8_ts}, {ref_dst_ts},
                          *engine, *strm),
                graph::status::success);

        ASSERT_TRUE(allclose(ref_dst_data, dst_data, /*rtol*/ 0.01f,
                /*atol*/ 0.01f));
    }
}
