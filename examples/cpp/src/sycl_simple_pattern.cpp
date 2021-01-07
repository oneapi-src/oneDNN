/*******************************************************************************
* Copyright 2020 Intel Corporation
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

/// @example sycl_simple_pattern.cpp
/// @copybrief sycl_simple_pattern_cpp
/// > Annotated version: @ref sycl_simple_pattern_cpp

/// @page sycl_simple_pattern_cpp Getting started on both CPU and GPU with SYCL extensions API
/// This is an example to demonstrate how to build a simple graph and run on
/// SYCL device.
///
/// > Example code: @ref sycl_simple_pattern.cpp
///
/// Some key take-aways included in this example:
///
/// * how to build a graph and get several partitions
/// * how to create engine, allocator and stream
/// * how to compile a partition
/// * how to execute a compiled partition with input tensors on a specific
///     stream

#include <assert.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

#include "common/utils.hpp"

/// @page sycl_simple_pattern_cpp
/// @section sycl_simple_pattern_cpp_headers Public headers
///
/// To start using oneDNN graph, we must include the @ref dnnl_graph.hpp header file
/// in the application. If you also want to run with SYCL device, you need include
/// @ref dnnl_graph_sycl.hpp header file as well. All the C++ APIs reside in namespace `dnnl::graph`.
/// @page sycl_simple_pattern_cpp
/// @snippet sycl_simple_pattern.cpp Headers and namespace
//[Headers and namespace]
#include "oneapi/dnnl/dnnl_graph.hpp"
#include "oneapi/dnnl/dnnl_graph_sycl.hpp"
using namespace dnnl::graph;
using namespace cl::sycl;
//[Headers and namespace]

/// @page sycl_simple_pattern_cpp
/// @section sycl_simple_pattern_cpp_tutorial sycl_simple_pattern_tutorial() function
///
void sycl_simple_pattern_tutorial(engine::kind engine_kind) {
    std::cout << "========Example: Conv->ReLU->Conv->ReLU========\n";
    // clang-format off

    std::cout << "Create logical tensors and operators-----------";
    const std::vector<size_t> logical_id {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    std::vector<int64_t> input_dims {8, 3, 227, 227};
    std::vector<int64_t> weight_dims {96, 3, 11, 11};
    std::vector<int64_t> bias_dims {96};
    std::vector<int64_t> weight1_dims {96, 96, 1, 1};
    std::vector<int64_t> bias1_dims {96};
    std::vector<int64_t> dst_dims {-1, -1, -1, -1};

    /// @page sycl_simple_pattern_cpp
    /// @subsection sycl_simple_pattern_cpp_get_partition Build graph and get partitions
    ///
    /// In this section, we are trying to build a graph containing the pattern like `conv0->relu0->conv1->relu1`. After that,
    /// we can get all of partitions which are determined by backend.
    ///
    /// To create a graph, #dnnl::graph::engine::kind is needed because the returned partitions maybe vary on different devices.
    /// @snippet sycl_simple_pattern.cpp Create graph
    //[Create graph]
    graph g(engine_kind);
    //[Create graph]

    /// To build a graph, the connection relationship of different ops must be known. In oneDNN graph, #dnnl::graph::logical_tensor is used
    /// to express such relationship. So, next step is to create logical tensors for these ops including inputs and outputs.
    ///
    /// Create input/output #dnnl::graph::logical_tensor for first `Convolution` op.
    /// @snippet sycl_simple_pattern.cpp Create conv's logical tensor
    //[Create conv's logical tensor]
    logical_tensor conv0_src_desc {logical_id[0], logical_tensor::data_type::f32, input_dims, logical_tensor::layout_type::undef};
    logical_tensor conv0_weight_desc {logical_id[1], logical_tensor::data_type::f32, weight_dims, logical_tensor::layout_type::undef};
    logical_tensor conv0_dst_desc {logical_id[2], logical_tensor::data_type::f32, dst_dims, logical_tensor::layout_type::undef};
    //[Create conv's logical tensor]

    /// Create first `Convolution` op (#dnnl::graph::op) and attaches attributes to it, such as `strides`, `pads_begin`, `pads_end`, `data_format`, etc.
    /// @snippet sycl_simple_pattern.cpp Create first conv
    //[Create first conv]
    op conv0(0, op::kind::Convolution, {conv0_src_desc, conv0_weight_desc}, {conv0_dst_desc}, "conv0");
    conv0.set_attr<std::vector<int64_t>>("strides", {4, 4});
    conv0.set_attr<std::vector<int64_t>>("pads_begin", {0, 0});
    conv0.set_attr<std::vector<int64_t>>("pads_end", {0, 0});
    conv0.set_attr<std::vector<int64_t>>("dilations", {1, 1});
    conv0.set_attr<int64_t>("groups", 1);
    conv0.set_attr<std::string>("data_format", "NCX");
    conv0.set_attr<std::string>("filter_format", "OIX");
    //[Create first conv]

    /// Create input/output logical tensors for first `BiasAdd` op.
    /// @snippet sycl_simple_pattern.cpp Create biasadd's logical tensor
    //[Create biasadd's logical tensor]
    logical_tensor conv0_bias_desc {logical_id[3], logical_tensor::data_type::f32, bias_dims, logical_tensor::layout_type::undef};
    logical_tensor conv0_bias_add_dst_desc {logical_id[4], logical_tensor::data_type::f32, dst_dims, logical_tensor::layout_type::undef};
    //[Create biasadd's logical tensor]

    /// Create first `BiasAdd` op.
    /// @snippet sycl_simple_pattern.cpp Create first bias_add
    //[Create first bias_add]
    op conv0_bias_add(1, op::kind::BiasAdd, {conv0_dst_desc, conv0_bias_desc}, {conv0_bias_add_dst_desc}, "conv0_bias_add");
    //[Create first bias_add]

    /// Create output logical tensors for first `Relu` op.
    /// @snippet sycl_simple_pattern.cpp Create relu's logical tensor
    //[Create relu's logical tensor]
    logical_tensor relu0_dst_desc {logical_id[5], logical_tensor::data_type::f32, dst_dims, logical_tensor::layout_type::undef};
    //[Create relu's logical tensor]

    /// Create first `Relu` op.
    /// @snippet sycl_simple_pattern.cpp Create first relu
    //[Create first relu]
    op relu0(2, op::kind::ReLU, {conv0_bias_add_dst_desc}, {relu0_dst_desc}, "relu0");
    //[Create first relu]

    /// Create input/output logical tensors for second `Convolution` op.
    /// @snippet sycl_simple_pattern.cpp Create conv's second logical tensor
    //[Create conv's second logical tensor]
    logical_tensor conv1_weight_desc {logical_id[6], logical_tensor::data_type::f32, weight1_dims, logical_tensor::layout_type::undef};
    logical_tensor conv1_dst_desc {logical_id[7], logical_tensor::data_type::f32, dst_dims, logical_tensor::layout_type::undef};
    //[Create conv's second logical tensor]

    /// Create second `Convolution` op and also attaches required attributes to it.
    /// @snippet sycl_simple_pattern.cpp Create second conv
    //[Create second conv]
    op conv1(3, op::kind::Convolution, {relu0_dst_desc, conv1_weight_desc}, {conv1_dst_desc}, "conv1");
    conv1.set_attr<std::vector<int64_t>>("strides", {1, 1});
    conv1.set_attr<std::vector<int64_t>>("pads_begin", {0, 0});
    conv1.set_attr<std::vector<int64_t>>("pads_end", {0, 0});
    conv1.set_attr<std::vector<int64_t>>("dilations", {1, 1});
    conv1.set_attr<int64_t>("groups", 1);
    conv1.set_attr<std::string>("data_format", "NCX");
    conv1.set_attr<std::string>("filter_format", "OIX");
    //[Create second conv]

    /// Create input/output logical tensors for second `BiasAdd` op.
    /// @snippet sycl_simple_pattern.cpp Create biasadd's second logical tensor
    //[Create biasadd's second logical tensor]
    logical_tensor conv1_bias_desc {logical_id[8], logical_tensor::data_type::f32, bias1_dims, logical_tensor::layout_type::undef};
    logical_tensor conv1_bias_add_dst_desc {logical_id[9], logical_tensor::data_type::f32, dst_dims, logical_tensor::layout_type::undef};
    //[Create biasadd's second logical tensor]

    /// Create second `BiasAdd` op.
    /// @snippet sycl_simple_pattern.cpp Create second bias_add
    //[Create second bias_add]
    op conv1_bias_add(4, op::kind::BiasAdd, {conv1_dst_desc, conv1_bias_desc}, {conv1_bias_add_dst_desc}, "conv1_bias_add");
    //[Create second bias_add]

    /// Create output logical tensors for second `Relu` op.
    /// @snippet sycl_simple_pattern.cpp Create relu's second logical tensor
    //[Create relu's second logical tensor]
    logical_tensor relu1_dst_desc {logical_id[10], logical_tensor::data_type::f32, dst_dims, logical_tensor::layout_type::undef};
    //[Create relu's second logical tensor]

    /// Create second `Relu` op.
    /// @snippet sycl_simple_pattern.cpp Create second relu
    //[Create second relu]
    op relu1(5, op::kind::ReLU, {conv1_bias_add_dst_desc}, {relu1_dst_desc}, "relu1");
    //[Create second relu]
    std::cout << "Success!\n";

    std::cout << "Add OP to graph--------------------------------";
    /// Finally, those created ops will be added into the graph. The graph inside will maintain a
    /// list to store all these ops.
    /// @snippet sycl_simple_pattern.cpp Add op
    //[Add op]
    g.add_op(conv0);
    g.add_op(relu0);
    g.add_op(conv1);
    g.add_op(relu1);
    g.add_op(conv0_bias_add);
    g.add_op(conv1_bias_add);
    //[Add op]
    std::cout << "Success!\n";

    std::cout << "Filter and get partition-----------------------";
    /// After finished above operations, we can get partitions by calling #dnnl::graph::graph::get_partitions().
    /// Here we can slao specify the #dnnl::graph::partition::policy to get different partitions.
    ///
    /// In this example, the graph will be filtered into two partitions:
    /// `conv0+relu0` and `conv1+relu1`.
    ///
    /// @note
    ///     Setting `DNNL_GRAPH_DUMP=1` can save internal graphs into dot files
    ///     before/after graph fusion.
    ///
    /// @snippet sycl_simple_pattern.cpp Get partition
    //[Get partition]
    auto partitions = g.get_partitions();
    //[Get partition]

    if (partitions.size() != 2) {
        throw std::runtime_error("wrong partition number");
    }
    std::cout << "Success!\n";

    /// @page sycl_simple_pattern_cpp
    /// @subsection sycl_simple_pattern_cpp_compile Compile partition
    ///
    /// In the real case, we assume that framework can provide device info at this stage. But in this example, we just use a
    /// self-defined device to simulate the real behavior.
    ///
    /// Create a #dnnl::graph::allocator with two user-defined #dnnl_graph_sycl_allocate_f and #dnnl_graph_sycl_deallocate_f call-back functions.
    /// @snippet sycl_simple_pattern.cpp Create allocator
    //[Create allocator]
    allocator alloc = sycl_interop::make_allocator(sycl_malloc_wrapper, sycl_free_wrapper);
    //[Create allocator]

    /// Define SYCL queue (code outside of oneDNN graph)
    /// @snippet sycl_simple_pattern.cpp Define sycl queue
    //[Define sycl queue]
    sycl::queue q = (engine_kind == engine::kind::gpu) ? sycl::queue(gpu_selector {}) : sycl::queue(cpu_selector {});
    //[Define sycl queue]

    /// Create a #dnnl::graph::engine based on SYCL device and context. Also, set a
    /// user-defined #dnnl::graph::allocator to this engine.
    ///
    /// @snippet sycl_simple_pattern.cpp Create engine
    //[Create engine]
    engine eng = sycl_interop::make_engine(q.get_device(), q.get_context());
    eng.set_allocator(alloc);
    //[Create engine]

    std::cout << "Prepare logical tensors with proper format-----";
    /// Sets proper format to the logical tensors for inputs/outputs of
    /// partition 0.
    ///
    /// @note
    ///    In this example, partition inputs(conv0)/weights/bias are created with plain layout while output has opaque layout.
    ///
    /// @snippet sycl_simple_pattern.cpp Prepare format for logical tensors 0
    //[Prepare format for logical tensors 0]
    logical_tensor conv0_src_desc_plain {logical_id[0], logical_tensor::data_type::f32, input_dims, logical_tensor::layout_type::strided};
    logical_tensor conv0_weight_desc_plain {logical_id[1], logical_tensor::data_type::f32, weight_dims, logical_tensor::layout_type::strided};
    logical_tensor conv0_bias_desc_plain {logical_id[3], logical_tensor::data_type::f32, bias_dims, logical_tensor::layout_type::strided};
    logical_tensor relu0_dst_desc_any {logical_id[5], logical_tensor::data_type::f32, dst_dims, logical_tensor::layout_type::any};
    //[Prepare format for logical tensors 0]
    std::cout << "Success!\n";

    std::cout << "Infer shape from partition 0-------------------";
    /// Infer output shape of the partition 0.
    /// @snippet sycl_simple_pattern.cpp Infer shape 0
    //[Infer shape 0]
    std::vector<logical_tensor> in0 {conv0_src_desc_plain, conv0_weight_desc_plain, conv0_bias_desc_plain};
    std::vector<logical_tensor> out0 {relu0_dst_desc_any};
    partitions[0].infer_shape(in0, out0);
    //[Infer shape 0]
    std::cout << "Success!\n";
    std::vector<int64_t> infered_relu0_dst_dims = out0[0].get_dims();
    std::cout << "Infered_shape: " << infered_relu0_dst_dims[0] << ","
              << infered_relu0_dst_dims[1] << "," << infered_relu0_dst_dims[2]
              << "," << infered_relu0_dst_dims[3] << "\n";

    std::cout << "Compile partition 0----------------------------";
    /// Compile the partition 0 to generate compiled partition with the
    /// input and output logical tensors.
    /// @snippet sycl_simple_pattern.cpp Compile partition 0
    //[Compile partition 0]
    auto cp0 = partitions[0].compile(in0, out0, eng);
    //[Compile partition 0]
    std::cout << "Success!\n";

    std::cout << "Query layout id from compiled partition 0------";
    /// Query input logical tensor with opaque layout from compiled partition 1.
    /// @snippet sycl_simple_pattern.cpp Query logical tensor 0
    //[Query logical tensor 0]
    logical_tensor conv1_src_desc_opaque = cp0.query_logical_tensor(logical_id[5]);
    //[Query logical tensor 0]

    /// Sets proper format to the logical tensors for inputs/outputs of
    /// partition 1.
    ///
    /// @note
    ///    In this example, partition inputs(conv1), weights and bias logical
    ///    tensors are created with plain layout.
    ///
    /// @snippet sycl_simple_pattern.cpp Prepare format for logical tensors 1
    //[Prepare format for logical tensors 1]
    logical_tensor conv1_weight_desc_plain {logical_id[6], logical_tensor::data_type::f32, weight1_dims, logical_tensor::layout_type::strided};
    logical_tensor conv1_bias_desc_plain {logical_id[8], logical_tensor::data_type::f32, bias1_dims, logical_tensor::layout_type::strided};
    logical_tensor relu1_dst_desc_plain {logical_id[10], logical_tensor::data_type::f32, dst_dims, logical_tensor::layout_type::strided};
    //[Prepare format for logical tensors 1]
    std::cout << "Success!\n";

    std::cout << "Infer shape from partition 1-------------------";

    /// Infers the shape of output from the partition 1.
    /// @snippet sycl_simple_pattern.cpp Infer shape 1
    //[Infer shape 1]
    std::vector<logical_tensor> in1 {conv1_src_desc_opaque, conv1_weight_desc_plain, conv1_bias_desc_plain};
    std::vector<logical_tensor> out1 {relu1_dst_desc_plain};
    partitions[1].infer_shape(in1, out1);
    //[Infer shape 1]
    std::cout << "Success!\n";
    const std::vector<int64_t> infered_relu1_dst_dims
            = out1[0].get_dims();
    std::cout << "Infered_shape: " << infered_relu1_dst_dims[0] << ","
              << infered_relu1_dst_dims[1] << "," << infered_relu1_dst_dims[2]
              << "," << infered_relu1_dst_dims[3] << "\n";

    std::cout << "Compile partition 1----------------------------";
    /// Compile the partition 1 to generate compiled partition with the
    /// input and output logical tensors.
    /// @snippet sycl_simple_pattern.cpp Compile partition 1
    //[Compile partition 1]
    auto cp1 = partitions[1].compile(in1, out1, eng);
    //[Compile partition 1]
    std::cout << "Success!\n";

    // Step 5: Prepare tensor and execute compiled partitions
    std::cout << "Prepare tensor and execute compiled partitions-";

    /// @page sycl_simple_pattern_cpp
    /// @subsection sycl_simple_pattern_cpp_execute Execute compiled partition
    ///
    /// Prepare SYCL USM buffer (code outside of oneDNN graph)
    /// @snippet sycl_simple_pattern.cpp Prepare USM
    //[Prepare USM]
    auto conv0_src_data = (float *)malloc_shared(static_cast<size_t>(product(input_dims)) * sizeof(float), q.get_device(), q.get_context());
    auto conv0_weight_data = (float *)malloc_shared(static_cast<size_t>(product(weight_dims)) * sizeof(float), q.get_device(), q.get_context());
    auto conv0_bias_data = (float *)malloc_shared(static_cast<size_t>(product(bias_dims)) * sizeof(float), q.get_device(), q.get_context());
    auto relu0_dst_data = (float *)malloc_shared(cp0.query_logical_tensor(logical_id[5]).get_mem_size(), q.get_device(), q.get_context());
    auto conv1_weight_data = (float *)malloc_shared(static_cast<size_t>(product(weight1_dims)) * sizeof(float), q.get_device(), q.get_context());
    auto conv1_bias_data = (float *)malloc_shared(static_cast<size_t>(product(bias1_dims)) * sizeof(float), q.get_device(), q.get_context());
    auto relu1_dst_data = (float *)malloc_shared(cp1.query_logical_tensor(logical_id[10]).get_mem_size(), q.get_device(), q.get_context());
    //[Prepare USM]

    fill_buffer<float>(q, conv0_src_data, product(input_dims), 1.0f);
    fill_buffer<float>(q, conv0_weight_data, product(weight_dims), 1.0f);
    fill_buffer<float>(q, conv0_bias_data, product(bias_dims), 1.0f);
    fill_buffer<float>(q, relu0_dst_data,
            cp0.query_logical_tensor(logical_id[5]).get_mem_size()
                    / sizeof(float),
            0.0f);
    fill_buffer<float>(q, conv1_weight_data, product(weight1_dims), 1.0f);
    fill_buffer<float>(q, conv1_bias_data, product(bias1_dims), 1.0f);
    fill_buffer<float>(q, relu1_dst_data,
            cp1.query_logical_tensor(logical_id[10]).get_mem_size()
                    / sizeof(float),
            0.0f);

    /// Create a stream on the engine asssociated with a sycl queue.
    /// @snippet sycl_simple_pattern.cpp Create stream
    //[Create stream]
    auto strm = sycl_interop::make_stream(eng, q);
    //[Create stream]

    /// Prepare the input/output tensors with the data buffer for the
    /// partition 0.
    /// @snippet sycl_simple_pattern.cpp Prepare tensors 0
    //[Prepare tensors 0]
    tensor conv0_src(conv0_src_desc_plain, conv0_src_data);
    tensor conv0_weight(conv0_weight_desc_plain, conv0_weight_data);
    tensor conv0_bias(conv0_bias_desc_plain, conv0_bias_data);
    logical_tensor relu0_dst_desc_opaque = cp0.query_logical_tensor(logical_id[5]);
    tensor relu0_dst(relu0_dst_desc_opaque, relu0_dst_data);
    //[Prepare tensors 0]

    /// Execute the compiled partition 0 on the specified stream.
    /// @snippet sycl_simple_pattern.cpp Execute compiled partition 0
    //[Execute compiled partition 0]
    std::vector<tensor> in_list_0 {conv0_src, conv0_weight, conv0_bias};
    std::vector<tensor> out_list_0 {relu0_dst};
    sycl_interop::execute(cp0, strm, in_list_0, out_list_0);
    //[Execute compiled partition 0]

    /// Prepare the input/output tensors with the data buffer for the
    /// partition 1.
    /// @snippet sycl_simple_pattern.cpp Prepare tensors 1
    //[Prepare tensors 1]
    tensor conv1_weight(conv1_weight_desc_plain, conv1_weight_data);
    tensor conv1_bias(conv1_bias_desc_plain, conv1_bias_data);
    logical_tensor relu1_dst_desc_plain_infered_shape = cp1.query_logical_tensor(logical_id[10]);
    tensor relu1_dst(relu1_dst_desc_plain_infered_shape, relu1_dst_data);
    //[Prepare tensors 1]

    /// Execute the compiled partition 1 on the specified stream.
    /// @snippet sycl_simple_pattern.cpp Execute compiled partition 1
    //[Execute compiled partition 1]
    std::vector<tensor> in_list_1 {relu0_dst, conv1_weight, conv1_bias};
    std::vector<tensor> out_list_1 {relu1_dst};
    sycl_interop::execute(cp1, strm, in_list_1, out_list_1);
    //[Execute compiled partition 1]
    std::cout << "Success!\n";

    std::cout << "Check correctness------------------------------";
    /// Check correctness of the output results.
    /// @snippet sycl_simple_pattern.cpp Check results
    //[Check results]
    float expected_result
            = (1 * 11 * 11 * 3 + /* conv0 bias */ 1.0f) * (1 * 1 * 96)
            + /* conv1 bias */ 1.0f;
    auto out_size = cp1.query_logical_tensor(logical_id[10]).get_mem_size()
            / sizeof(float);
    for (size_t i = 0; i < out_size; i++) {
        if ((float)expected_result != relu1_dst_data[i]) {
            throw std::runtime_error(
                    "output result is not equal to excepted "
                    "results");
        }
    }
    //[Check results]
    std::cout << "Success!\n";

    std::cout << "============Run Example Successfully===========\n";

    free(conv0_src_data, q.get_context());
    free(conv0_weight_data, q.get_context());
    free(conv0_bias_data, q.get_context());
    free(relu0_dst_data, q.get_context());
    free(conv1_weight_data, q.get_context());
    free(conv1_bias_data, q.get_context());
    free(relu1_dst_data, q.get_context());

    /// @page sycl_simple_pattern_cpp Getting started on GPU with SYCL extensions API
    // clang-format on
}

int main(int argc, char **argv) {
    engine::kind engine_kind = parse_engine_kind(argc, argv);
    sycl_simple_pattern_tutorial(engine_kind);
    return 0;
}
