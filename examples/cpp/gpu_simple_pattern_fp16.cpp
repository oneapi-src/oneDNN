/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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

/// @example gpu_simple_pattern_fp16.cpp
/// @copybrief gpu_simple_pattern_fp16_cpp
/// Annotated version: @ref gpu_simple_pattern_fp16_cpp

/// @page gpu_simple_pattern_fp16_cpp SYCL GPU example for conv+relu+conv+relu pattern
///
/// > Example code: @ref gpu_simple_pattern_fp16.cpp

#include "oneapi/dnnl/dnnl_graph.hpp"
#include "oneapi/dnnl/dnnl_graph_sycl.hpp"

#include "example_utils.hpp"

using namespace dnnl::graph;
using namespace cl::sycl;
using data_type = logical_tensor::data_type;
using layout_type = logical_tensor::layout_type;

int main(int argc, char **argv) {
    /// create a graph with a gpu device kind
    const engine::kind ekind = engine::kind::gpu;
    graph g(ekind);

    std::vector<int64_t> conv0_input_dims {8, 3, 227, 227};
    std::vector<int64_t> conv0_weight_dims {96, 3, 11, 11};
    std::vector<int64_t> conv0_bias_dims {96};
    std::vector<int64_t> conv1_weight_dims {96, 96, 1, 1};
    std::vector<int64_t> conv1_bias_dims {96};

    /// create logical tensors with f16 data type for conv0
    logical_tensor conv0_src_desc {
            0, data_type::f16, conv0_input_dims, layout_type::strided};
    logical_tensor conv0_weight_desc {
            1, data_type::f16, conv0_weight_dims, layout_type::strided};

    logical_tensor conv0_dst_desc {2, data_type::f16, 4, layout_type::strided};

    /// create conv0 operator
    op conv0 {3, op::kind::Convolution, {conv0_src_desc, conv0_weight_desc},
            {conv0_dst_desc}, "conv0"};
    conv0.set_attr<std::vector<int64_t>>("strides", {4, 4});
    conv0.set_attr<std::vector<int64_t>>("pads_begin", {0, 0});
    conv0.set_attr<std::vector<int64_t>>("pads_end", {0, 0});
    conv0.set_attr<std::vector<int64_t>>("dilations", {1, 1});
    conv0.set_attr<int64_t>("groups", 1);
    conv0.set_attr<std::string>("data_format", "NCX");
    conv0.set_attr<std::string>("filter_format", "OIX");

    logical_tensor conv0_bias_desc {
            4, data_type::f16, conv0_bias_dims, layout_type::strided};

    logical_tensor conv0_bias_add_dst_desc {
            5, data_type::f16, 4, layout_type::strided};

    /// create conv0_bias_add
    op conv0_bias_add {6, op::kind::BiasAdd, {conv0_dst_desc, conv0_bias_desc},
            {conv0_bias_add_dst_desc}, "conv0_bias_add"};

    logical_tensor relu0_dst_desc {7, data_type::f16, 4, layout_type::strided};

    /// create relu0 operator
    op relu0 {8, op::kind::ReLU, {conv0_bias_add_dst_desc}, {relu0_dst_desc},
            "relu0"};

    /// create logical tensors with f16 data type for conv1
    logical_tensor conv1_weight_desc {
            9, data_type::f16, conv1_weight_dims, layout_type::strided};
    logical_tensor conv1_bias_desc {
            10, data_type::f16, conv1_bias_dims, layout_type::strided};
    logical_tensor conv1_dst_desc {11, data_type::f16, 4, layout_type::strided};

    /// create conv1 operator
    op conv1 {12, op::kind::Convolution, {relu0_dst_desc, conv1_weight_desc},
            {conv1_dst_desc}, "conv1"};
    conv1.set_attr<std::vector<int64_t>>("strides", {1, 1});
    conv1.set_attr<std::vector<int64_t>>("pads_begin", {0, 0});
    conv1.set_attr<std::vector<int64_t>>("pads_end", {0, 0});
    conv1.set_attr<std::vector<int64_t>>("dilations", {1, 1});
    conv1.set_attr<int64_t>("groups", 1);
    conv1.set_attr<std::string>("data_format", "NCX");
    conv1.set_attr<std::string>("filter_format", "OIX");

    logical_tensor conv1_bias_add_dst_desc {
            13, data_type::f16, 4, layout_type::strided};

    op conv1_bias_add {14, op::kind::BiasAdd, {conv1_dst_desc, conv1_bias_desc},
            {conv1_bias_add_dst_desc}, "conv1_bias_add"};

    logical_tensor relu1_dst_desc {15, data_type::f16, 4, layout_type::strided};
    op relu1 {16, op::kind::ReLU, {conv1_bias_add_dst_desc}, {relu1_dst_desc},
            "relu1"};

    /// add operators to the graph
    g.add_op(conv0);
    g.add_op(conv0_bias_add);
    g.add_op(relu0);

    g.add_op(conv1);
    g.add_op(conv1_bias_add);
    g.add_op(relu1);

    /// The graph will be partitioned into two partitions:
    /// - conv0 + conv0_bias_add + relu0
    /// - conv1 + conv1_bias_add + relu1
    auto partitions = g.get_partitions(partition::policy::fusion);

    if (partitions.size() != 2) {
        throw std::runtime_error(
                "gpu_simple_pattern_f16: incorrect partition number");
    }

    /// users need to provide sycl device, sycl context, and sycl queue
    sycl::queue q(gpu_selector {}, sycl::property::queue::in_order {});
    engine eng = sycl_interop::make_engine(q.get_device(), q.get_context());
    allocator alloc = sycl_interop::make_allocator(
            sycl_malloc_wrapper, sycl_free_wrapper);
    eng.set_allocator(alloc);
    /// construct a new stream
    dnnl::graph::stream strm = sycl_interop::make_stream(eng, q);

    /// compile the first partition
    auto cp0 = partitions[0].compile(
            {conv0_src_desc, conv0_weight_desc, conv0_bias_desc},
            {relu0_dst_desc}, eng);

    /// get the output logical tensor for the first compiled partition
    logical_tensor relu0_dst_desc_q
            = cp0.query_logical_tensor(relu0_dst_desc.get_id());

    /// prepare data for the execution by allocating shared memory
    void *conv0_src_data = cl::sycl::malloc_shared(
            conv0_src_desc.get_mem_size(), q.get_device(), q.get_context());
    void *conv0_weight_data = cl::sycl::malloc_shared(
            conv0_weight_desc.get_mem_size(), q.get_device(), q.get_context());
    void *conv0_bias_data = cl::sycl::malloc_shared(
            conv0_bias_desc.get_mem_size(), q.get_device(), q.get_context());
    void *relu0_dst_data = cl::sycl::malloc_shared(
            relu0_dst_desc_q.get_mem_size(), q.get_device(), q.get_context());

    /// create tensors for cp0
    tensor conv0_src_ts {conv0_src_desc, eng, conv0_src_data};
    tensor conv0_weight_ts {conv0_weight_desc, eng, conv0_weight_data};
    tensor conv0_bias_ts {conv0_bias_desc, eng, conv0_bias_data};
    tensor relu0_dst_ts {relu0_dst_desc_q, eng, relu0_dst_data};

    /// execute the first compiled partition
    cp0.execute(strm, {conv0_src_ts, conv0_weight_ts, conv0_bias_ts},
            {relu0_dst_ts});

    /// compile the second partition
    auto cp1 = partitions[1].compile(
            {relu0_dst_desc_q, conv1_weight_desc, conv1_bias_desc},
            {relu1_dst_desc}, eng);

    /// get the output logical tensor for the second compiled partition
    logical_tensor relu1_dst_desc_q
            = cp1.query_logical_tensor(relu1_dst_desc.get_id());

    /// prepare data for the execution by allocating shared memory
    void *conv1_weight_data = cl::sycl::malloc_shared(
            conv1_weight_desc.get_mem_size(), q.get_device(), q.get_context());
    void *conv1_bias_data = cl::sycl::malloc_shared(
            conv1_bias_desc.get_mem_size(), q.get_device(), q.get_context());
    void *relu1_dst_data = cl::sycl::malloc_shared(
            relu1_dst_desc_q.get_mem_size(), q.get_device(), q.get_context());

    /// create tensors for cp1
    tensor conv1_weight_ts {conv1_weight_desc, eng, conv0_weight_data};
    tensor conv1_bias_ts {conv1_bias_desc, eng, conv0_bias_data};
    tensor relu1_dst_ts {relu1_dst_desc_q, eng, relu0_dst_data};

    /// execute the second compiled partition, the first input tensor is the
    /// output of cp0.
    cp0.execute(strm, {relu0_dst_ts, conv1_weight_ts, conv1_bias_ts},
            {relu1_dst_ts});

    strm.wait();

    /// release the allocated shared memory
    cl::sycl::free(conv0_src_data, q.get_context());
    cl::sycl::free(conv0_weight_data, q.get_context());
    cl::sycl::free(conv0_bias_data, q.get_context());
    cl::sycl::free(relu0_dst_data, q.get_context());
    cl::sycl::free(conv1_weight_data, q.get_context());
    cl::sycl::free(conv1_bias_data, q.get_context());
    cl::sycl::free(relu1_dst_data, q.get_context());

    return 0;
}
