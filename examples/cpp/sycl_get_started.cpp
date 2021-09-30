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

/// @example sycl_get_started.cpp
/// @copybrief sycl_get_started_cpp
/// > Annotated version: @ref sycl_get_started_cpp

/// @page sycl_get_started_cpp Getting started on both CPU and GPU with SYCL extensions API
/// This is an example to demonstrate how to build a simple graph and run on
/// SYCL device.
///
/// > Example code: @ref sycl_get_started.cpp
///
/// Some key take-aways included in this example:
///
/// * how to build a graph and get several partitions
/// * how to create engine, allocator and stream
/// * how to compile a partition
/// * how to execute a compiled partition
///

/// @page sycl_get_started_cpp
/// @section sycl_get_started_cpp_headers Public headers
///
/// To start using oneDNN graph, we must include the @ref dnnl_graph.hpp header file
/// into the application. If you also want to run with SYCL device, you need include
/// @ref dnnl_graph_sycl.hpp header as well. All the C++ APIs reside in namespace `dnnl::graph`.
/// @page sycl_get_started_cpp
/// @snippet sycl_get_started.cpp Headers and namespace
//[Headers and namespace]
#include "oneapi/dnnl/dnnl_graph.hpp"
#include "oneapi/dnnl/dnnl_graph_sycl.hpp"
using namespace dnnl::graph;
using namespace cl::sycl;
//[Headers and namespace]

#include "example_utils.hpp"

/// @page sycl_get_started_cpp
/// @section sycl_get_started_cpp_tutorial sycl_get_started_tutorial() function
///
void sycl_get_started_tutorial(engine::kind ekind) {
    std::vector<int64_t> conv0_input_dims {8, 3, 227, 227};
    std::vector<int64_t> conv0_weight_dims {96, 3, 11, 11};
    std::vector<int64_t> conv0_bias_dims {96};
    std::vector<int64_t> conv1_weight_dims {96, 96, 1, 1};
    std::vector<int64_t> conv1_bias_dims {96};

    /// @page sycl_get_started_cpp
    /// @subsection sycl_get_started_cpp_get_partition Build graph and get partitions
    ///
    /// In this section, we are trying to build a graph containing the pattern
    /// like `conv0->relu0->conv1->relu1`. After that, we can get all of partitions
    ///  which are determined by backend.
    ///
    /// To create a graph, #dnnl::graph::engine::kind is needed because the returned
    /// partitions maybe vary on different devices.
    /// @snippet sycl_get_started.cpp Create graph
    //[Create graph]
    graph g(ekind);
    //[Create graph]

    /// To build a graph, the connection relationship of different ops must be known.
    /// In oneDNN graph, #dnnl::graph::logical_tensor is used to express such relationship.
    /// So, next step is to create logical tensors for these ops including inputs and outputs.
    ///
    /// Create input/output #dnnl::graph::logical_tensor for first `Convolution` op.
    /// @snippet sycl_get_started.cpp Create conv's logical tensor
    //[Create conv's logical tensor]
    logical_tensor conv0_src_desc {0, logical_tensor::data_type::f32,
            conv0_input_dims, logical_tensor::layout_type::strided};
    logical_tensor conv0_weight_desc {1, logical_tensor::data_type::f32,
            conv0_weight_dims, logical_tensor::layout_type::strided};
    logical_tensor conv0_dst_desc {2, logical_tensor::data_type::f32, 4,
            logical_tensor::layout_type::strided};
    //[Create conv's logical tensor]

    /// Create first `Convolution` op (#dnnl::graph::op) and attaches attributes to it,
    /// such as `strides`, `pads_begin`, `pads_end`, `data_format`, etc.
    /// @snippet sycl_get_started.cpp Create first conv
    //[Create first conv]
    op conv0(3, op::kind::Convolution, {conv0_src_desc, conv0_weight_desc},
            {conv0_dst_desc}, "conv0");
    conv0.set_attr<std::vector<int64_t>>("strides", {4, 4});
    conv0.set_attr<std::vector<int64_t>>("pads_begin", {0, 0});
    conv0.set_attr<std::vector<int64_t>>("pads_end", {0, 0});
    conv0.set_attr<std::vector<int64_t>>("dilations", {1, 1});
    conv0.set_attr<int64_t>("groups", 1);
    conv0.set_attr<std::string>("data_format", "NCX");
    conv0.set_attr<std::string>("filter_format", "OIX");
    //[Create first conv]

    /// Create input/output logical tensors for first `BiasAdd` op.
    /// @snippet sycl_get_started.cpp Create biasadd's logical tensor
    //[Create biasadd's logical tensor]
    logical_tensor conv0_bias_desc {4, logical_tensor::data_type::f32,
            conv0_bias_dims, logical_tensor::layout_type::strided};
    logical_tensor conv0_bias_add_dst_desc {5, logical_tensor::data_type::f32,
            4, logical_tensor::layout_type::strided};
    //[Create biasadd's logical tensor]

    /// Create first `BiasAdd` op.
    /// @snippet sycl_get_started.cpp Create first bias_add
    //[Create first bias_add]
    op conv0_bias_add(6, op::kind::BiasAdd, {conv0_dst_desc, conv0_bias_desc},
            {conv0_bias_add_dst_desc}, "conv0_bias_add");
    //[Create first bias_add]

    /// Create output logical tensors for first `Relu` op.
    /// @snippet sycl_get_started.cpp Create relu's logical tensor
    //[Create relu's logical tensor]
    logical_tensor relu0_dst_desc {7, logical_tensor::data_type::f32, 4,
            logical_tensor::layout_type::strided};
    //[Create relu's logical tensor]

    /// Create first `Relu` op.
    /// @snippet sycl_get_started.cpp Create first relu
    //[Create first relu]
    op relu0(8, op::kind::ReLU, {conv0_bias_add_dst_desc}, {relu0_dst_desc},
            "relu0");
    //[Create first relu]

    /// Create input/output logical tensors for second `Convolution` op.
    /// @snippet sycl_get_started.cpp Create conv's second logical tensor
    //[Create conv's second logical tensor]
    logical_tensor conv1_weight_desc {9, logical_tensor::data_type::f32,
            conv1_weight_dims, logical_tensor::layout_type::strided};
    logical_tensor conv1_dst_desc {10, logical_tensor::data_type::f32, 4,
            logical_tensor::layout_type::strided};
    //[Create conv's second logical tensor]

    /// Create second `Convolution` op and also attaches required attributes to it.
    /// @snippet sycl_get_started.cpp Create second conv
    //[Create second conv]
    op conv1(11, op::kind::Convolution, {relu0_dst_desc, conv1_weight_desc},
            {conv1_dst_desc}, "conv1");
    conv1.set_attr<std::vector<int64_t>>("strides", {1, 1});
    conv1.set_attr<std::vector<int64_t>>("pads_begin", {0, 0});
    conv1.set_attr<std::vector<int64_t>>("pads_end", {0, 0});
    conv1.set_attr<std::vector<int64_t>>("dilations", {1, 1});
    conv1.set_attr<int64_t>("groups", 1);
    conv1.set_attr<std::string>("data_format", "NCX");
    conv1.set_attr<std::string>("filter_format", "OIX");
    //[Create second conv]

    /// Create input/output logical tensors for second `BiasAdd` op.
    /// @snippet sycl_get_started.cpp Create biasadd's second logical tensor
    //[Create biasadd's second logical tensor]
    logical_tensor conv1_bias_desc {12, logical_tensor::data_type::f32,
            conv1_bias_dims, logical_tensor::layout_type::strided};
    logical_tensor conv1_bias_add_dst_desc {13, logical_tensor::data_type::f32,
            4, logical_tensor::layout_type::strided};
    //[Create biasadd's second logical tensor]

    /// Create second `BiasAdd` op.
    /// @snippet sycl_get_started.cpp Create second bias_add
    //[Create second bias_add]
    op conv1_bias_add(14, op::kind::BiasAdd, {conv1_dst_desc, conv1_bias_desc},
            {conv1_bias_add_dst_desc}, "conv1_bias_add");
    //[Create second bias_add]

    /// Create output logical tensors for second `Relu` op.
    /// @snippet sycl_get_started.cpp Create relu's second logical tensor
    //[Create relu's second logical tensor]
    logical_tensor relu1_dst_desc {15, logical_tensor::data_type::f32, 4,
            logical_tensor::layout_type::strided};
    //[Create relu's second logical tensor]

    /// Create second `Relu` op.
    /// @snippet sycl_get_started.cpp Create second relu
    //[Create second relu]
    op relu1(16, op::kind::ReLU, {conv1_bias_add_dst_desc}, {relu1_dst_desc},
            "relu1");
    //[Create second relu]

    /// Finally, those created ops will be added into the graph. The graph
    /// internally will maintain a list to store all of these ops.
    /// @snippet sycl_get_started.cpp Add op
    //[Add op]
    g.add_op(conv0);
    g.add_op(conv0_bias_add);
    g.add_op(relu0);
    g.add_op(conv1);
    g.add_op(conv1_bias_add);
    g.add_op(relu1);
    //[Add op]

    /// After finished above operations, we can get partitions by calling
    /// #dnnl::graph::graph::get_partitions(). Here we can also specify the
    /// #dnnl::graph::partition::policy to get different partitions.
    ///
    /// In this example, the graph will be partitioned into two partitions:
    /// 1. conv0 + conv0_bias_add + relu0
    /// 2. conv1 + conv1_bias_add + relu1
    ///
    /// @snippet sycl_get_started.cpp Get partition
    //[Get partition]
    auto partitions = g.get_partitions();
    //[Get partition]

    if (partitions.size() != 2) {
        throw std::runtime_error(
                "sycl_get_started: incorrect partition number");
    }

    /// @page sycl_get_started_cpp
    /// @subsection sycl_get_started_cpp_compile Compile partition
    ///
    /// In the real case, users like framework should provide device information
    /// at this stage. But in this example, we just use a self-defined device to
    /// simulate the real behavior.
    ///
    /// Create a #dnnl::graph::allocator with two user-defined #dnnl_graph_sycl_allocate_f
    /// and #dnnl_graph_sycl_deallocate_f call-back functions.
    /// @snippet sycl_get_started.cpp Create allocator
    //[Create allocator]
    allocator alloc = sycl_interop::make_allocator(
            sycl_malloc_wrapper, sycl_free_wrapper);
    //[Create allocator]

    /// Define SYCL queue (code outside of oneDNN graph)
    /// @snippet sycl_get_started.cpp Define sycl queue
    //[Define sycl queue]
    sycl::queue q = (ekind == engine::kind::gpu)
            ? sycl::queue(gpu_selector {}, sycl::property::queue::in_order {})
            : sycl::queue(cpu_selector {}, sycl::property::queue::in_order {});
    //[Define sycl queue]

    /// Create a #dnnl::graph::engine based on SYCL device and context. Also,
    /// set a user-defined #dnnl::graph::allocator to this engine.
    ///
    /// @snippet sycl_get_started.cpp Create engine
    //[Create engine]
    engine eng = sycl_interop::make_engine(q.get_device(), q.get_context());
    eng.set_allocator(alloc);
    //[Create engine]

    /// Compile the partition 0 to generate compiled partition with the
    /// input and output logical tensors.
    /// @snippet sycl_get_started.cpp Compile partition 0
    //[Compile partition 0]
    auto cp0 = partitions[0].compile(
            {conv0_src_desc, conv0_weight_desc, conv0_bias_desc},
            {relu0_dst_desc}, eng);
    //[Compile partition 0]

    /// Get the output logical tensor from compiled partition 0. The logical
    /// tensor should contain the correct output shape and layout information.
    /// @snippet sycl_get_started.cpp Get output logical tensor of cp0
    //[Get output logical tensor of cp0]
    logical_tensor relu0_dst_desc_q
            = cp0.query_logical_tensor(relu0_dst_desc.get_id());
    //[Get output logical tensor of cp0]

    /// Compile the partition 1 to generate compiled partition with the
    /// input and output logical tensors.
    /// @snippet sycl_get_started.cpp Compile partition 1
    //[Compile partition 1]
    auto cp1 = partitions[1].compile(
            {relu0_dst_desc_q, conv1_weight_desc, conv1_bias_desc},
            {relu1_dst_desc}, eng);
    //[Compile partition 1]

    /// Get the output logical tensor from compiled partition 1. The logical
    /// tensor should contain the correct output shape and layout information.
    /// @snippet sycl_get_started.cpp Get output logical tensor of cp1
    //[Get output logical tensor of cp1]
    logical_tensor relu1_dst_desc_q
            = cp1.query_logical_tensor(relu1_dst_desc.get_id());
    //[Get output logical tensor of cp1]

    /// @page sycl_get_started_cpp
    /// @subsection sycl_get_started_cpp_execute Execute compiled partition
    ///

    /// Create a stream on the engine associated with a sycl queue.
    /// @snippet sycl_get_started.cpp Create stream
    //[Create stream]
    auto strm = sycl_interop::make_stream(eng, q);
    //[Create stream]

    /// Prepare SYCL USM buffer (code outside of oneDNN graph)
    /// @snippet sycl_get_started.cpp Prepare USM
    //[Prepare USM]
    auto conv0_src_data = (float *)malloc_shared(
            conv0_src_desc.get_mem_size(), q.get_device(), q.get_context());
    auto conv0_weight_data = (float *)malloc_shared(
            conv0_weight_desc.get_mem_size(), q.get_device(), q.get_context());
    auto conv0_bias_data = (float *)malloc_shared(
            conv0_bias_desc.get_mem_size(), q.get_device(), q.get_context());
    auto relu0_dst_data = (float *)malloc_shared(
            relu0_dst_desc_q.get_mem_size(), q.get_device(), q.get_context());
    auto conv1_weight_data = (float *)malloc_shared(
            conv1_weight_desc.get_mem_size(), q.get_device(), q.get_context());
    auto conv1_bias_data = (float *)malloc_shared(
            conv1_bias_desc.get_mem_size(), q.get_device(), q.get_context());
    auto relu1_dst_data = (float *)malloc_shared(
            relu1_dst_desc_q.get_mem_size(), q.get_device(), q.get_context());
    //[Prepare USM]

    /// Prepare the input/output tensors with the data buffer for the partition 0.
    /// @snippet sycl_get_started.cpp Prepare tensors for cp0
    //[Prepare tensors for cp0]
    tensor conv0_src_ts {conv0_src_desc, eng, conv0_src_data};
    tensor conv0_weight_ts {conv0_weight_desc, eng, conv0_weight_data};
    tensor conv0_bias_ts {conv0_bias_desc, eng, conv0_bias_data};
    tensor relu0_dst_ts {relu0_dst_desc_q, eng, relu0_dst_data};
    //[Prepare tensors for cp0]

    /// Execute the compiled partition 0 on the specified stream.
    /// @snippet sycl_get_started.cpp Execute compiled partition 0
    //[Execute compiled partition 0]
    std::vector<tensor> cp0_outputs = {relu0_dst_ts};
    sycl_interop::execute(cp0, strm,
            {conv0_src_ts, conv0_weight_ts, conv0_bias_ts}, cp0_outputs);
    //[Execute compiled partition 0]

    /// Prepare the input/output tensors with the data buffer for the partition 1.
    /// @snippet sycl_get_started.cpp Prepare tensors for cp1
    //[Prepare tensors for cp1]
    tensor conv1_weight_ts {conv1_weight_desc, eng, conv1_weight_data};
    tensor conv1_bias_ts {conv1_bias_desc, eng, conv1_bias_data};
    tensor relu1_dst_ts {relu1_dst_desc_q, eng, relu1_dst_data};
    //[Prepare tensors for cp1]

    /// Execute the compiled partition 1 on the specified stream.
    /// @snippet sycl_get_started.cpp Execute compiled partition 1
    //[Execute compiled partition 1]
    std::vector<tensor> cp1_outputs = {relu1_dst_ts};
    sycl_interop::execute(cp1, strm,
            {relu0_dst_ts, conv1_weight_ts, conv1_bias_ts}, cp1_outputs);
    strm.wait();
    //[Execute compiled partition 1]

    free(conv0_src_data, q.get_context());
    free(conv0_weight_data, q.get_context());
    free(conv0_bias_data, q.get_context());
    free(relu0_dst_data, q.get_context());
    free(conv1_weight_data, q.get_context());
    free(conv1_bias_data, q.get_context());
    free(relu1_dst_data, q.get_context());

    /// @page sycl_get_started_cpp Getting started on GPU with SYCL extensions API
}

int main(int argc, char **argv) {
    engine::kind ekind = parse_engine_kind(argc, argv);
    sycl_get_started_tutorial(ekind);
    return 0;
}
