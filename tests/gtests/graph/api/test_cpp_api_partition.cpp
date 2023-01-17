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

#include "oneapi/dnnl/dnnl_graph.hpp"

#include "test_api_common.hpp"
#include "gtest/gtest.h"

#include <cstdint>

TEST(APIPartition, PartitionTest) {
    using namespace dnnl::graph;
    dnnl::engine::kind engine_kind
            = static_cast<dnnl::engine::kind>(api_test_engine_kind);
    dnnl::engine eng = cpp_api_test_dnnl_engine_create(engine_kind);
    dnnl::engine::kind real_engine_kind = eng.get_kind();

    // when enable sycl, the real engine kind will always be gpu, because we
    // use default gpu::selector to find gpu device.
    ASSERT_EQ(real_engine_kind, engine_kind);

    graph g(engine_kind);

    std::vector<int64_t> input_dims {8, 256, 56, 56};
    std::vector<int64_t> conv_weight_dims {64, 256, 1, 1};
    std::vector<int64_t> conv_dst_dims {8, 64, 56, 56};
    std::vector<int64_t> infer_dst_dims(4, DNNL_GRAPH_UNKNOWN_DIM);

    logical_tensor lt1 {0, logical_tensor::data_type::f32, input_dims,
            logical_tensor::layout_type::undef};
    logical_tensor lt2 {1, logical_tensor::data_type::f32, conv_weight_dims,
            logical_tensor::layout_type::undef};
    logical_tensor lt3 {2, logical_tensor::data_type::f32, conv_dst_dims,
            logical_tensor::layout_type::undef};
    logical_tensor lt4 {3, logical_tensor::data_type::f32, infer_dst_dims,
            logical_tensor::layout_type::undef};

    op conv(0, op::kind::Convolution, "conv");
    op relu_op(1, op::kind::ReLU, "relu");

    conv.set_attr<std::vector<int64_t>>(op::attr::strides, {1, 1});
    conv.set_attr<std::vector<int64_t>>(op::attr::pads_begin, {0, 0});
    conv.set_attr<std::vector<int64_t>>(op::attr::pads_end, {0, 0});
    conv.set_attr<std::vector<int64_t>>(op::attr::dilations, {1, 1});
    conv.set_attr<std::string>(op::attr::data_format, "NCX");
    conv.set_attr<std::string>(op::attr::weights_format, "OIX");
    conv.set_attr<int64_t>(op::attr::groups, 1);

    conv.add_inputs({lt1, lt2});
    conv.add_output(lt3);
    relu_op.add_input(lt3);
    relu_op.add_output(lt4);

    g.add_op(conv);
    g.add_op(relu_op);
    g.finalize();

    //create_partition
    auto partitions = g.get_partitions(partition::policy::fusion);
    ASSERT_EQ(partitions.size(), 1U);

    // check partition engine kind
    ASSERT_EQ(partitions[0].get_engine_kind(), engine_kind);

    //get_ops
    std::vector<size_t> ops = partitions[0].get_ops();
    ASSERT_EQ(ops.size(), 2U);
    ASSERT_EQ(partitions[0].get_ops_num(), 2U);

    // The returned op ids in partition must be in topo order
    ASSERT_EQ(partitions[0].get_ops()[0], 0U);
    ASSERT_EQ(partitions[0].get_ops()[1], 1U);

    logical_tensor lt1_plain {0, logical_tensor::data_type::f32, input_dims,
            logical_tensor::layout_type::strided};
    logical_tensor lt2_plain {1, logical_tensor::data_type::f32,
            conv_weight_dims, logical_tensor::layout_type::strided};
    logical_tensor lt3_plain {2, logical_tensor::data_type::f32, conv_dst_dims,
            logical_tensor::layout_type::strided};
    logical_tensor lt4_any {3, logical_tensor::data_type::f32, infer_dst_dims,
            logical_tensor::layout_type::any};

    //compile partition
    std::vector<logical_tensor> in0({lt1_plain, lt2_plain, lt3_plain});
    std::vector<logical_tensor> out0({lt4_any});

    auto cp = partitions[0].compile(in0, out0, eng);
    // query logical tensor from compiled partition
    auto lt4_opaque = cp.query_logical_tensor(3);
    ASSERT_EQ(lt4_opaque.get_layout_type(),
            real_engine_kind == dnnl::engine::kind::gpu
                    ? logical_tensor::layout_type::opaque
                    : logical_tensor::layout_type::strided);

    auto cp1 = partitions[0].compile(in0, out0, eng);
    // query logical tensor from compiled partition
    auto lt5_opaque = cp1.query_logical_tensor(3);
    ASSERT_EQ(lt5_opaque.get_layout_type(),
            real_engine_kind == dnnl::engine::kind::gpu
                    ? logical_tensor::layout_type::opaque
                    : logical_tensor::layout_type::strided);
}

TEST(APIPartition, GetInputOutputIDs) {
    using namespace dnnl::graph;
    dnnl::engine::kind engine_kind
            = static_cast<dnnl::engine::kind>(api_test_engine_kind);
    dnnl::engine eng = cpp_api_test_dnnl_engine_create(engine_kind);
    dnnl::engine::kind real_engine_kind = eng.get_kind();

    // when enable sycl, the real engine kind will always be gpu, because we
    // use default gpu::selector to find gpu device.
    ASSERT_EQ(real_engine_kind, engine_kind);

    graph g(engine_kind);

    std::vector<int64_t> input_dims {8, 256, 56, 56};
    std::vector<int64_t> conv_weight_dims {64, 256, 1, 1};
    std::vector<int64_t> conv_dst_dims {8, 64, 56, 56};
    std::vector<int64_t> infer_dst_dims(4, DNNL_GRAPH_UNKNOWN_DIM);

    std::vector<size_t> input_ids {0, 1};
    std::vector<size_t> output_ids {2};

    logical_tensor lt1 {input_ids[0], logical_tensor::data_type::f32,
            input_dims, logical_tensor::layout_type::undef};
    logical_tensor lt2 {input_ids[1], logical_tensor::data_type::f32,
            conv_weight_dims, logical_tensor::layout_type::undef};
    logical_tensor lt3 {output_ids[0], logical_tensor::data_type::f32,
            conv_dst_dims, logical_tensor::layout_type::undef};

    op conv {0, op::kind::Convolution, {lt1, lt2}, {lt3}, "conv"};
    conv.set_attr<std::vector<int64_t>>(op::attr::strides, {1, 1});
    conv.set_attr<std::vector<int64_t>>(op::attr::pads_begin, {0, 0});
    conv.set_attr<std::vector<int64_t>>(op::attr::pads_end, {0, 0});
    conv.set_attr<std::vector<int64_t>>(op::attr::dilations, {1, 1});
    conv.set_attr<std::string>(op::attr::data_format, "NCX");
    conv.set_attr<std::string>(op::attr::weights_format, "OIX");
    conv.set_attr<int64_t>(op::attr::groups, 1);

    g.add_op(conv);
    g.finalize();

    // get partitions
    auto partitions = g.get_partitions(partition::policy::fusion);
    ASSERT_EQ(partitions.size(), 1U);

    // check ids of inputs
    std::vector<logical_tensor> got_inputs = partitions[0].get_input_ports();
    ASSERT_EQ(got_inputs.size(), input_ids.size());
    for (size_t i = 0; i < got_inputs.size(); ++i)
        ASSERT_EQ(got_inputs[i].get_id(), input_ids[i]);

    // check ids of outputs
    std::vector<logical_tensor> got_outputs = partitions[0].get_output_ports();
    ASSERT_EQ(got_outputs.size(), output_ids.size());
    for (size_t i = 0; i < got_outputs.size(); ++i)
        ASSERT_EQ(got_outputs[i].get_id(), output_ids[i]);

    // check partition's supporting status
    ASSERT_TRUE(partitions[0].is_supported());
}

TEST(APIPartition, UnsupportedPartitions) {
    using namespace dnnl::graph;
    dnnl::engine::kind engine_kind
            = static_cast<dnnl::engine::kind>(api_test_engine_kind);
    dnnl::engine eng = cpp_api_test_dnnl_engine_create(engine_kind);
    dnnl::engine::kind real_engine_kind = eng.get_kind();

    // when enable sycl, the real engine kind will always be gpu, because we
    // use default gpu::selector to find gpu device.
    ASSERT_EQ(real_engine_kind, engine_kind);

    graph g(engine_kind);

    std::vector<size_t> lt_ids {0, 1, 2};
    std::vector<size_t> op_ids {0, 1};

    logical_tensor input1 {lt_ids[0], logical_tensor::data_type::f32,
            logical_tensor::layout_type::undef};

    logical_tensor wildcard_dst {lt_ids[2], logical_tensor::data_type::f32,
            logical_tensor::layout_type::undef};

    op wildcard {op_ids[0], op::kind::Wildcard, {input1}, {wildcard_dst},
            "wildcard"};

    op end {op_ids[1], op::kind::End, {wildcard_dst}, {}, "end"};

    g.add_op(wildcard);
    g.add_op(end);
    g.finalize();

    std::vector<partition> partitions = g.get_partitions();
    ASSERT_EQ(partitions.size(), 2U);
    for (auto &p : partitions)
        ASSERT_FALSE(p.is_supported());
}

TEST(APIPartition, AddInferShape) {
    using namespace dnnl::graph;
    dnnl::engine::kind engine_kind
            = static_cast<dnnl::engine::kind>(api_test_engine_kind);
    graph g(engine_kind);

    std::vector<int64_t> shape_0 {1};
    std::vector<int64_t> shape_1 {8, 15, 5, 7};
    std::vector<int64_t> shape_2(4, DNNL_GRAPH_UNKNOWN_DIM);

    logical_tensor lt_0 {0, logical_tensor::data_type::f32, shape_0,
            logical_tensor::layout_type::strided};
    logical_tensor lt_1 {1, logical_tensor::data_type::f32, shape_1,
            logical_tensor::layout_type::strided};
    logical_tensor lt_2 {2, logical_tensor::data_type::f32, shape_2,
            logical_tensor::layout_type::strided};

    op add {3, op::kind::Add, "add"};
    add.add_inputs({lt_0, lt_1});
    add.add_outputs({lt_2});

    g.add_op(add);
    g.finalize();
    auto ps = g.get_partitions();
    ASSERT_EQ(ps.size(), 1U);
}

TEST(APIPartition, SingleConvPartition) {
    using namespace dnnl::graph;
    dnnl::engine::kind engine_kind
            = static_cast<dnnl::engine::kind>(api_test_engine_kind);

    std::vector<int64_t> input_dims {8, 256, 56, 56};
    std::vector<int64_t> conv_weight_dims {64, 256, 1, 1};
    std::vector<int64_t> conv_dst_dims {8, 64, 56, 56};

    logical_tensor lt1 {0, logical_tensor::data_type::f32, input_dims,
            logical_tensor::layout_type::undef};
    logical_tensor lt2 {1, logical_tensor::data_type::f32, conv_weight_dims,
            logical_tensor::layout_type::undef};
    logical_tensor lt3 {2, logical_tensor::data_type::f32, conv_dst_dims,
            logical_tensor::layout_type::undef};

    op conv(0, op::kind::Convolution, "conv");
    conv.set_attr<std::vector<int64_t>>(op::attr::strides, {1, 1});
    conv.set_attr<std::vector<int64_t>>(op::attr::pads_begin, {0, 0});
    conv.set_attr<std::vector<int64_t>>(op::attr::pads_end, {0, 0});
    conv.set_attr<std::vector<int64_t>>(op::attr::dilations, {1, 1});
    conv.set_attr<std::string>(op::attr::data_format, "NCX");
    conv.set_attr<std::string>(op::attr::weights_format, "OIX");
    conv.set_attr<int64_t>(op::attr::groups, 1);

    conv.add_inputs({lt1, lt2});
    conv.add_output(lt3);

    partition part {conv, engine_kind};

    // check partition engine kind
    ASSERT_EQ(part.get_engine_kind(), engine_kind);

    // get_ops
    std::vector<size_t> ops = part.get_ops();
    ASSERT_EQ(ops.size(), 1U);

    // supported?
    ASSERT_TRUE(part.is_supported());
}

TEST(APIPartition, CompileWildcardPartition) {
    using namespace dnnl::graph;
    dnnl::engine::kind engine_kind
            = static_cast<dnnl::engine::kind>(api_test_engine_kind);
    dnnl::engine eng = cpp_api_test_dnnl_engine_create(engine_kind);
    std::vector<int64_t> data_dims {8, 256, 56, 56};

    logical_tensor lt1 {0, logical_tensor::data_type::f32, data_dims,
            logical_tensor::layout_type::strided};
    logical_tensor lt2 {1, logical_tensor::data_type::f32, data_dims,
            logical_tensor::layout_type::strided};

    op wc(0, op::kind::Wildcard, "wildcard");
    wc.add_input(lt1);
    wc.add_output(lt2);

    partition part {wc, engine_kind};

    // get_ops
    std::vector<size_t> ops = part.get_ops();
    ASSERT_EQ(ops.size(), 1U);

    // supported?
    ASSERT_FALSE(part.is_supported());

    // compile
    EXPECT_THROW(part.compile({lt1}, {lt2}, eng), dnnl::error);
}

TEST(APIPartitionCache, GetSetCapacity) {
    ASSERT_EQ(dnnl_graph_set_compiled_partition_cache_capacity(-1),
            dnnl_invalid_arguments);
    ASSERT_NO_THROW(dnnl_graph_set_compiled_partition_cache_capacity(2));

    ASSERT_EQ(dnnl_graph_get_compiled_partition_cache_capacity(nullptr),
            dnnl_invalid_arguments);
    int c;
#ifndef DNNL_GRAPH_DISABLE_COMPILED_PARTITION_CACHE
    ASSERT_EQ((dnnl_graph_get_compiled_partition_cache_capacity(&c), c), 2);
#else
    ASSERT_EQ(
            dnnl_graph_get_compiled_partition_cache_capacity(&c), dnnl_success);
#endif
}
