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

/// @example cpu_get_started.cpp
/// @copybrief cpu_get_started_cpp
/// > Annotated version: @ref cpu_get_started_cpp

/// @page cpu_get_started_cpp Getting started on GPU with SYCL extensions API
/// This is an example to demonstrate how to build a simple graph and run on
/// SYCL device.
///
/// > Example code: @ref cpu_get_started.cpp
///
/// Some key take-aways included in this example:
///
/// * how to build a graph and get several partitions
/// * how to create engine, allocator and stream
/// * how to compile a partition
/// * how to execute a compiled partition with input tensors on a specific
///     stream
///

#include <assert.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

#include "common/utils.hpp"

/// @page cpu_get_started_cpp
/// @section cpu_get_started_cpp_headers Public headers
///
/// To start using oneDNN graph, we must include the @ref dnnl_graph.hpp header file
/// in the application. If you also want to run with SYCL device, you need include
/// @ref dnnl_graph_sycl.hpp header file as well. All the C++ APIs reside in namespace `dnnl::graph`.
/// @page cpu_get_started_cpp
/// @snippet cpu_get_started.cpp Headers and namespace
//[Headers and namespace]
#include "oneapi/dnnl/dnnl_graph.hpp"
using namespace dnnl::graph;
//[Headers and namespace]

/// @page cpu_get_started_cpp
/// @section cpu_get_started_cpp_tutorial cpu_get_started_tutorial() function
///
void cpu_get_started_tutorial(engine::kind engine_kind) {
    std::cout << "========Example: Conv->ReLU->Conv->ReLU========\n";
    // clang-format off

    std::cout << "Create logical tensors and operators-------------------";
    const std::vector<size_t> logical_id {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    std::vector<int64_t> input_dims {8, 3, 227, 227};
    std::vector<int64_t> weight_dims {96, 3, 11, 11};
    std::vector<int64_t> bias_dims {96};
    std::vector<int64_t> weight1_dims {96, 96, 1, 1};
    std::vector<int64_t> bias1_dims {96};
    std::vector<int64_t> dst_dims {8, 96, 55, 55};
    std::vector<int64_t> dst1_dims {8, 96, 55, 55};

    /// @page cpu_get_started_cpp
    /// @subsection cpu_get_started_cpp_get_partition Build graph and get partitions
    ///
    /// In this section, we are trying to build a graph containing the pattern like `conv0->relu0->conv1->relu1`. After that,
    /// we can get all of partitions which are determined by backend.
    ///
    /// To create a graph, #dnnl::graph::engine::kind is needed because the returned partitions maybe vary on different devices.
    /// @snippet cpu_get_started.cpp Create graph
    //[Create graph]
    graph g(engine_kind);
    //[Create graph]

    /// To build a graph, the connection relationship of different ops must be known. In oneDNN graph, #dnnl::graph::logical_tensor is used
    /// to express such relationship. So, next step is to create logical tensors for these ops including inputs and outputs.
    ///
    /// Create input/output #dnnl::graph::logical_tensor for first `Convolution` op.
    /// @snippet cpu_get_started.cpp Create conv's logical tensor
    //[Create conv's logical tensor]
    logical_tensor conv0_src_desc {logical_id[0], logical_tensor::data_type::f32, input_dims, logical_tensor::layout_type::undef};
    logical_tensor conv0_weight_desc {logical_id[1], logical_tensor::data_type::f32, weight_dims, logical_tensor::layout_type::undef};
    logical_tensor conv0_dst_desc {logical_id[2], logical_tensor::data_type::f32, dst_dims, logical_tensor::layout_type::undef};
    //[Create conv's logical tensor]

    /// Create first `Convolution` op (#dnnl::graph::op) and attaches attributes to it, such as `strides`, `pads_begin`, `pads_end`, `data_format`, etc.
    /// @snippet cpu_get_started.cpp Create first conv
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
    /// @snippet cpu_get_started.cpp Create biasadd's logical tensor
    //[Create biasadd's logical tensor]
    logical_tensor conv0_bias_desc {logical_id[3], logical_tensor::data_type::f32, bias_dims, logical_tensor::layout_type::undef};
    logical_tensor conv0_bias_add_dst_desc {logical_id[4], logical_tensor::data_type::f32, dst_dims, logical_tensor::layout_type::undef};
    //[Create biasadd's logical tensor]

    /// Create first `BiasAdd` op.
    /// @snippet cpu_get_started.cpp Create first bias_add
    //[Create first bias_add]
    op conv0_bias_add(1, op::kind::BiasAdd, {conv0_dst_desc, conv0_bias_desc}, {conv0_bias_add_dst_desc}, "conv0_bias_add");
    //[Create first bias_add]

    /// Create output logical tensors for first `Relu` op.
    /// @snippet cpu_get_started.cpp Create relu's logical tensor
    //[Create relu's logical tensor]
    logical_tensor relu0_dst_desc {logical_id[5], logical_tensor::data_type::f32, dst_dims, logical_tensor::layout_type::undef};
    //[Create relu's logical tensor]

    /// Create first `Relu` op.
    /// @snippet cpu_get_started.cpp Create first relu
    //[Create first relu]
    op relu0(2, op::kind::ReLU, {conv0_bias_add_dst_desc}, {relu0_dst_desc}, "relu0");
    //[Create first relu]

    /// Create input/output logical tensors for second `Convolution` op.
    /// @snippet cpu_get_started.cpp Create conv's second logical tensor
    //[Create conv's second logical tensor]
    logical_tensor conv1_weight_desc {logical_id[6], logical_tensor::data_type::f32, weight1_dims, logical_tensor::layout_type::undef};
    logical_tensor conv1_dst_desc {logical_id[7], logical_tensor::data_type::f32, dst_dims, logical_tensor::layout_type::undef};
    //[Create conv's second logical tensor]

    /// Create second `Convolution` op and also attaches required attributes to it.
    /// @snippet cpu_get_started.cpp Create second conv
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
    /// @snippet cpu_get_started.cpp Create biasadd's second logical tensor
    //[Create biasadd's second logical tensor]
    logical_tensor conv1_bias_desc {logical_id[8], logical_tensor::data_type::f32, bias1_dims, logical_tensor::layout_type::undef};
    logical_tensor conv1_bias_add_dst_desc {logical_id[9], logical_tensor::data_type::f32, dst_dims, logical_tensor::layout_type::undef};
    //[Create biasadd's second logical tensor]

    /// Create second `BiasAdd` op.
    /// @snippet cpu_get_started.cpp Create second bias_add
    //[Create second bias_add]
    op conv1_bias_add(4, op::kind::BiasAdd, {conv1_dst_desc, conv1_bias_desc}, {conv1_bias_add_dst_desc}, "conv1_bias_add");
    //[Create second bias_add]

    /// Create output logical tensors for second `Relu` op.
    /// @snippet cpu_get_started.cpp Create relu's second logical tensor
    //[Create relu's second logical tensor]
    logical_tensor relu1_dst_desc {logical_id[10], logical_tensor::data_type::f32, dst_dims, logical_tensor::layout_type::undef};
    //[Create relu's second logical tensor]

    /// Create second `Relu` op.
    /// @snippet cpu_get_started.cpp Create second relu
    //[Create second relu]
    op relu1(5, op::kind::ReLU, {conv1_bias_add_dst_desc}, {relu1_dst_desc}, "relu1");
    //[Create second relu]
    std::cout << "Success!\n";

    std::cout << "Add OP to llga graph------------------------";
    /// Finally, those created ops will be added into the graph. The graph inside will maintain a
    /// list to store all these ops.
    /// @snippet cpu_get_started.cpp Add op
    //[Add op]
    g.add_op(conv0);
    g.add_op(relu0);
    g.add_op(conv1);
    g.add_op(relu1);
    g.add_op(conv0_bias_add);
    g.add_op(conv1_bias_add);
    //[Add op]
    std::cout << "Success!\n";

    std::cout << "Filter and get partition--------";
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
    /// @snippet cpu_get_started.cpp Get partition
    //[Get partition]
    auto partitions = g.get_partitions();
    //[Get partition]

    if (partitions.size() != 2) {
        throw std::runtime_error("wrong partition number");
    }
    std::cout << "Success!\n";

    /// @page cpu_get_started_cpp
    /// @subsection cpu_get_started_cpp_compile Compile partition
    ///
    /// In the real case, we assume that framework can provide device info at this stage. But in this example, we just use a
    /// self-defined device to simulate the real behavior.
    ///

    /// Create a #dnnl::graph::engine. Also, set a
    /// user-defined #dnnl::graph::allocator to this engine.
    ///
    /// @snippet cpu_get_started.cpp Create engine
    //[Create engine]
    const int32_t device_id = 0;
    engine eng {engine_kind, device_id};
    allocator alloc {};
    eng.set_allocator(alloc);
    //[Create engine]

    std::cout << "Prepare logical tensors with proper format-----";
    /// Sets proper format to the logical tensors for inputs/outputs of
    /// partition 0.
    ///
    /// @note
    ///    In this example, partition inputs(conv0)/weights/bias are created with plain layout while output has opaque layout.
    ///
    /// @snippet cpu_get_started.cpp Prepare format for logical tensors 0
    //[Prepare format for logical tensors 0]
    logical_tensor conv0_src_desc_plain {logical_id[0], logical_tensor::data_type::f32, input_dims, logical_tensor::layout_type::strided};
    logical_tensor conv0_weight_desc_plain {logical_id[1], logical_tensor::data_type::f32, weight_dims, logical_tensor::layout_type::strided};
    logical_tensor conv0_bias_desc_plain {logical_id[3], logical_tensor::data_type::f32, bias_dims, logical_tensor::layout_type::strided};
    logical_tensor relu0_dst_desc_plain {logical_id[5], logical_tensor::data_type::f32, dst_dims, logical_tensor::layout_type::strided};
    //[Prepare format for logical tensors 0]
    std::cout << "Success!\n";

    std::cout << "Compile partition 0----------------------------";
    /// Compile the partition 0 to generate compiled partition with the
    /// input and output logical tensors.
    /// @snippet cpu_get_started.cpp Compile partition 0
    //[Compile partition 0]
    auto cp0 = partitions[0].compile({conv0_src_desc_plain, conv0_weight_desc_plain, conv0_bias_desc_plain}, {relu0_dst_desc_plain}, eng);
    //[Compile partition 0]
    std::cout << "Success!\n";

    /// Sets proper format to the logical tensors for inputs/outputs of
    /// partition 1.
    ///
    /// @note
    ///    In this example, partition inputs(conv1), weights and bias logical
    ///    tensors are created with plain layout.
    ///
    /// @snippet cpu_get_started.cpp Prepare format for logical tensors 1
    //[Prepare format for logical tensors 1]
    logical_tensor conv1_weight_desc_plain {logical_id[6], logical_tensor::data_type::f32, weight1_dims, logical_tensor::layout_type::strided};
    logical_tensor conv1_bias_desc_plain {logical_id[8], logical_tensor::data_type::f32, bias1_dims, logical_tensor::layout_type::strided};
    logical_tensor relu1_dst_desc_plain {logical_id[10], logical_tensor::data_type::f32, dst_dims, logical_tensor::layout_type::strided};
    //[Prepare format for logical tensors 1]
    std::cout << "Success!\n";

    std::cout << "Compile partition 1----------------------------";
    /// Compile the partition 1 to generate compiled partition with the
    /// input and output logical tensors.
    /// @snippet cpu_get_started.cpp Compile partition 1
    //[Compile partition 1]
    auto cp1 = partitions[1].compile({relu0_dst_desc_plain, conv1_weight_desc_plain, conv1_bias_desc_plain}, {relu1_dst_desc_plain}, eng);
    //[Compile partition 1]
    std::cout << "Success!\n";

    // Step 5: Prepare tensor and execute compiled partitions
    std::cout << "Prepare tensor and execute compiled partitions--";

    /// @page cpu_get_started_cpp
    /// @subsection cpu_get_started_cpp_execute Execute compiled partition
    ///

    /// Create a stream on the engine asssociated with a sycl queue.
    /// @snippet cpu_get_started.cpp Create stream
    //[Create stream]
    stream strm {eng};
    //[Create stream]

    std::vector<float> conv0_src_data(static_cast<size_t>(product(input_dims)), 1.0f);
    std::vector<float> conv0_weight_data(static_cast<size_t>(product(weight_dims)), 1.0f);
    std::vector<float> conv0_bias_data(static_cast<size_t>(product(bias_dims)), 1.0f);
    std::vector<float> relu0_dst_data(static_cast<size_t>(product(dst_dims)), 0.0f);
    std::vector<float> conv1_weight_data(static_cast<size_t>(product(weight1_dims)), 1.0f);
    std::vector<float> conv1_bias_data(static_cast<size_t>(product(bias1_dims)), 1.0f);
    std::vector<float> relu1_dst_data(static_cast<size_t>(product(dst1_dims)), 0.0f);

    /// Prepare the input/output tensors with the data buffer for the
    /// partition 0.
    /// @snippet cpu_get_started.cpp Prepare tensors 0
    //[Prepare tensors 0]
    tensor conv0_src(conv0_src_desc_plain, conv0_src_data.data());
    tensor conv0_weight(conv0_weight_desc_plain, conv0_weight_data.data());
    tensor conv0_bias(conv0_bias_desc_plain, conv0_bias_data.data());
    tensor relu0_dst(relu0_dst_desc_plain, relu0_dst_data.data());
    //[Prepare tensors 0]

    /// Execute the compiled partition 0 on the specified stream.
    /// @snippet cpu_get_started.cpp Execute compiled partition 0
    //[Execute compiled partition 0]
    cp0.execute(strm, {conv0_src, conv0_weight, conv0_bias}, {relu0_dst});
    //[Execute compiled partition 0]

    /// Prepare the input/output tensors with the data buffer for the
    /// partition 1.
    /// @snippet cpu_get_started.cpp Prepare tensors 1
    //[Prepare tensors 1]
    tensor conv1_weight(conv1_weight_desc_plain, conv1_weight_data.data());
    tensor conv1_bias(conv1_bias_desc_plain, conv1_bias_data.data());
    tensor relu1_dst(relu1_dst_desc_plain, relu1_dst_data.data());
    //[Prepare tensors 1]

    /// Execute the compiled partition 1 on the specified stream.
    /// @snippet cpu_get_started.cpp Execute compiled partition 1
    //[Execute compiled partition 1]
    cp1.execute(strm, {relu0_dst, conv1_weight, conv1_bias}, {relu1_dst});
    //[Execute compiled partition 1]
    std::cout << "Success!\n";

    std::cout << "Check correctness------------------------------";
    /// Check correctness of the output results.
    /// @snippet cpu_get_started.cpp Check results
    //[Check results]
    float expected_result
            = (1 * 11 * 11 * 3 + /* conv0 bias */ 1.0f) * (1 * 1 * 96)
            + /* conv1 bias */ 1.0f;
    for (auto v : relu1_dst_data) {
        if ((float)expected_result != v) {
            throw std::runtime_error(
                    "output result is not equal to excepted "
                    "results");
        }
    }
    //[Check results]
    std::cout << "Success!\n";

    std::cout << "============Run Example Successfully===========\n";
    /// @page cpu_get_started_cpp Getting started on CPU
    // clang-format on
}

int main(int argc, char **argv) {
    engine::kind engine_kind = parse_engine_kind(argc, argv);
    cpu_get_started_tutorial(engine_kind);
    return 0;
}
