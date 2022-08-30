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
#include <unordered_map>

#include "common/execution_context.hpp"
#include "common/helpers_any_layout.hpp"
#include "common/utils.hpp"

#define assertm(exp, msg) assert(((void)msg, exp))

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
#include "test_allocator.hpp"
using namespace dnnl::graph;
using namespace sycl;
//[Headers and namespace]
using data_type = logical_tensor::data_type;
using layout_type = logical_tensor::layout_type;

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
    std::vector<int64_t> dst_dims {8, 96, 55, 55};

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
    logical_tensor conv0_src_desc {logical_id[0], logical_tensor::data_type::f32, input_dims, logical_tensor::layout_type::strided};
    logical_tensor conv0_weight_desc {logical_id[1], logical_tensor::data_type::f32, weight_dims, logical_tensor::layout_type::strided};
    logical_tensor conv0_dst_desc {logical_id[2], logical_tensor::data_type::f32, dst_dims, logical_tensor::layout_type::strided};
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
    logical_tensor conv0_bias_desc {logical_id[3], logical_tensor::data_type::f32, bias_dims, logical_tensor::layout_type::strided};
    logical_tensor conv0_bias_add_dst_desc {logical_id[4], logical_tensor::data_type::f32, dst_dims, logical_tensor::layout_type::strided};
    //[Create biasadd's logical tensor]

    /// Create first `BiasAdd` op.
    /// @snippet sycl_simple_pattern.cpp Create first bias_add
    //[Create first bias_add]
    op conv0_bias_add(1, op::kind::BiasAdd, {conv0_dst_desc, conv0_bias_desc}, {conv0_bias_add_dst_desc}, "conv0_bias_add");
    //[Create first bias_add]

    /// Create output logical tensors for first `Relu` op.
    /// @snippet sycl_simple_pattern.cpp Create relu's logical tensor
    //[Create relu's logical tensor]
    logical_tensor relu0_dst_desc {logical_id[5], logical_tensor::data_type::f32, dst_dims, logical_tensor::layout_type::strided};
    //[Create relu's logical tensor]

    /// Create first `Relu` op.
    /// @snippet sycl_simple_pattern.cpp Create first relu
    //[Create first relu]
    op relu0(2, op::kind::ReLU, {conv0_bias_add_dst_desc}, {relu0_dst_desc}, "relu0");
    //[Create first relu]

    /// Create input/output logical tensors for second `Convolution` op.
    /// @snippet sycl_simple_pattern.cpp Create conv's second logical tensor
    //[Create conv's second logical tensor]
    logical_tensor conv1_weight_desc {logical_id[6], logical_tensor::data_type::f32, weight1_dims, logical_tensor::layout_type::strided};
    logical_tensor conv1_dst_desc {logical_id[7], logical_tensor::data_type::f32, dst_dims, logical_tensor::layout_type::strided};
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
    logical_tensor conv1_bias_desc {logical_id[8], logical_tensor::data_type::f32, bias1_dims, logical_tensor::layout_type::strided};
    logical_tensor conv1_bias_add_dst_desc {logical_id[9], logical_tensor::data_type::f32, dst_dims, logical_tensor::layout_type::strided};
    //[Create biasadd's second logical tensor]

    /// Create second `BiasAdd` op.
    /// @snippet sycl_simple_pattern.cpp Create second bias_add
    //[Create second bias_add]
    op conv1_bias_add(4, op::kind::BiasAdd, {conv1_dst_desc, conv1_bias_desc}, {conv1_bias_add_dst_desc}, "conv1_bias_add");
    //[Create second bias_add]

    /// Create output logical tensors for second `Relu` op.
    /// @snippet sycl_simple_pattern.cpp Create relu's second logical tensor
    //[Create relu's second logical tensor]
    logical_tensor relu1_dst_desc {logical_id[10], logical_tensor::data_type::f32, dst_dims, logical_tensor::layout_type::strided};
    //[Create relu's second logical tensor]

    /// Create second `Relu` op.
    /// @snippet sycl_simple_pattern.cpp Create second relu
    //[Create second relu]
    op relu1(5, op::kind::ReLU, {conv1_bias_add_dst_desc}, {relu1_dst_desc}, "relu1");
    //[Create second relu]
    std::cout << "Success!\n";

    std::unordered_map<size_t, op::kind> op_id_kind_map {{0, op::kind::Convolution},
        {1, op::kind::BiasAdd}, {2, op::kind::ReLU}, {3, op::kind::Convolution},
        {4, op::kind::BiasAdd}, {5, op::kind::ReLU}};

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
    std::cout << "Success!\n";

    std::cout << "Number of returned partitions: " << partitions.size() << "\n";
    for (size_t i = 0; i < partitions.size(); ++i) {
        std::cout << "Partition[" << partitions[i].get_id()
                  << "]'s supporting status: "
                  << (partitions[i].is_supported() ? "true" : "false") << "\n";
    }

    /// mark the output logical tensors of partition as ANY layout enabled
    std::unordered_set<size_t> id_to_set_any_layout;
    set_any_layout(partitions, id_to_set_any_layout);

    /// @page sycl_simple_pattern_cpp
    /// @subsection sycl_simple_pattern_cpp_compile Compile partition
    ///
    /// In the real case, we assume that framework can provide device info at this stage. But in this example, we just use a
    /// self-defined device to simulate the real behavior.
    ///
    /// Create a #dnnl::graph::allocator with two user-defined #dnnl_graph_sycl_allocate_f and #dnnl_graph_sycl_deallocate_f call-back functions.
    /// @snippet sycl_simple_pattern.cpp Create allocator
    //[Create allocator]
    allocator alloc = sycl_interop::make_allocator(dnnl::graph::testing::sycl_allocator_malloc, dnnl::graph::testing::sycl_allocator_free);
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
    // get simple sycl allocator
    const ::sycl::context &ctx = q.get_context();
    dnnl::graph::testing::simple_sycl_allocator *aallocator = dnnl::graph::testing::get_allocator(&ctx);
    engine eng = sycl_interop::make_engine(q.get_device(), ctx, alloc);
    //[Create engine]
    
    /// Create a stream on the engine associated with a sycl queue.
    /// @snippet sycl_simple_pattern.cpp Create stream
    //[Create stream]
    auto strm = sycl_interop::make_stream(eng, q);
    //[Create stream]
    
    std::vector<compiled_partition> c_partitions(partitions.size());

    // mapping from id to tensors
    // need provide queue for later buffer deallocation
    tensor_map tm {q};

    // mapping from id to queried logical tensor from compiled partition
    // used to record the logical tensors that are previously enabled with ANY layout
    std::unordered_map<size_t, logical_tensor> id_to_queried_logical_tensors;

    std::cout << "\nPartition[" << partitions[0].get_id() << "] is being processed.\n";
    std::vector<logical_tensor> inputs0 = partitions[0].get_in_ports();
    std::vector<logical_tensor> outputs0 = partitions[0].get_out_ports();

    /// replace input logical tensor with the queried one
    replace_with_queried_logical_tensors(inputs0, id_to_queried_logical_tensors);

    /// update output logical tensors with ANY layout
    update_tensors_with_any_layout(outputs0, id_to_set_any_layout);

    std::cout << "Compiling--------------------------------------";
    /// compile to generate compiled partition
    /// @snippet sycl_simple_pattern.cpp Compile partition
    //[Compile partition]
    c_partitions[0] = partitions[0].compile(inputs0, outputs0, eng);
    //[Compile partition]
    std::cout << "Success!\n";

    record_queried_logical_tensors(partitions[0].get_out_ports(), c_partitions[0],
        id_to_queried_logical_tensors);

    std::cout << "Creating tensors and allocating memory buffer--";
    std::vector<tensor> input_ts0 = tm.construct_and_initialize_tensors(inputs0, c_partitions[0], eng, 1);
    std::vector<tensor> output_ts0 = tm.construct_and_initialize_tensors(outputs0, c_partitions[0], eng, 0);
    std::cout << "Success!\n";

    std::cout << "Executing compiled partition-------------------";
    /// execute the compiled partition
    /// @snippet sycl_simple_pattern.cpp Execute compiled partition
    //[Execute compiled partition]
    auto returned_event0 = sycl_interop::execute(c_partitions[0], strm, input_ts0, output_ts0, {});
    //[Execute compiled partition]
    std::cout << "Success!\n";

    std::cout << "\nPartition[" << partitions[1].get_id() << "] is being processed.\n";
    std::vector<logical_tensor> inputs1 = partitions[1].get_in_ports();
    std::vector<logical_tensor> outputs1 = partitions[1].get_out_ports();

    /// replace input logical tensor with the queried one
    replace_with_queried_logical_tensors(inputs1, id_to_queried_logical_tensors);

    /// update output logical tensors with ANY layout
    update_tensors_with_any_layout(outputs1, id_to_set_any_layout);

    std::cout << "Compiling--------------------------------------";
    /// compile to generate compiled partition
    /// @snippet sycl_simple_pattern.cpp Compile partition
    //[Compile partition]
    c_partitions[1] = partitions[1].compile(inputs1, outputs1, eng);
    //[Compile partition]
    std::cout << "Success!\n";

    record_queried_logical_tensors(partitions[1].get_out_ports(), c_partitions[0],
        id_to_queried_logical_tensors);

    std::cout << "Creating tensors and allocating memory buffer--";
    std::vector<tensor> input_ts1 = tm.construct_and_initialize_tensors(inputs1, c_partitions[1], eng, 1);
    std::vector<tensor> output_ts1 = tm.construct_and_initialize_tensors(outputs1, c_partitions[1], eng, 0);
    std::cout << "Success!\n";

    std::cout << "Executing compiled partition-------------------";
    /// execute the compiled partition
    /// @snippet sycl_simple_pattern.cpp Execute compiled partition
    //[Execute compiled partition]
    auto returned_event1 = sycl_interop::execute(c_partitions[1], strm, input_ts1, output_ts1, {returned_event0});
    //[Execute compiled partition]
    std::cout << "Success!\n";

    // wait for all compiled partition's execution finished
    returned_event1.wait();

    std::cout << "Check correctness------------------------------";
    /// Check correctness of the output results.
    /// @snippet sycl_simple_pattern.cpp Check results
    //[Check results]
    float expected_result
            = (1 * 11 * 11 * 3 + /* conv0 bias */ 1.0f) * (1 * 1 * 96)
            + /* conv1 bias */ 1.0f;
    void *actual_output_ptr = tm.get(relu1_dst_desc.get_id()).get_data_handle();
    auto output_dims = relu1_dst_desc.get_dims();
    auto num_elem = product(output_dims);
    std::vector<float> expected_output(num_elem, expected_result);
    compare_data(expected_output.data(), reinterpret_cast<float *>(actual_output_ptr), num_elem);
    //[Check results]
    std::cout << "Success!\n";
    // do actual memory free
    aallocator->free_to_driver();

    std::cout << "============Run Example Successfully===========\n";

    /// @page sycl_simple_pattern_cpp Getting started on GPU with SYCL extensions API
    // clang-format on
}

int main(int argc, char **argv) {
    engine::kind engine_kind = parse_engine_kind(argc, argv);
    sycl_simple_pattern_tutorial(engine_kind);
    return 0;
}
