/*******************************************************************************
* Copyright 2024 Intel Corporation
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

/// @example gpu_single_op_partition.cpp
/// @copybrief graph_gpu_single_op_partition_cpp
/// > Annotated version: @ref graph_gpu_single_op_partition_cpp

/// @page graph_gpu_single_op_partition_cpp Single op partition on GPU
/// This is an example to demonstrate how to build a simple op graph and run it on gpu.
///
/// > Example code: @ref gpu_single_op_partition.cpp
///
/// Some key take-aways included in this example:
///
/// * how to build a single-op partition quickly
/// * how to create an engine, allocator and stream
/// * how to compile a partition
/// * how to execute a compiled partition
///
/// Some assumptions in this example:
///
/// * Only workflow is demonstrated without checking correctness
/// * Unsupported partitions should be handled by users themselves
///

/// @page graph_gpu_single_op_partition_cpp
/// @section graph_gpu_single_op_partition_cpp_headers Public headers
///
/// To start using oneDNN Graph, we must include the @ref dnnl_graph.hpp header
/// file in the application. All the C++ APIs reside in namespace `dnnl::graph`.
///
/// @page graph_gpu_single_op_partition_cpp
/// @snippet gpu_single_op_partition.cpp Headers and namespace
//[Headers and namespace]
#include "oneapi/dnnl/dnnl_graph.hpp"
#include "oneapi/dnnl/dnnl_graph_sycl.hpp"
#include "oneapi/dnnl/dnnl_sycl.hpp"
using namespace dnnl::graph;
using namespace sycl;

#include <assert.h>
#include <iostream>
#include <memory>
#include <vector>
#include <unordered_map>
#include <unordered_set>

#include "example_utils.hpp"
#include "graph_example_utils.hpp"

using namespace dnnl::graph;
using data_type = logical_tensor::data_type;
using layout_type = logical_tensor::layout_type;
using dim = logical_tensor::dim;
using dims = logical_tensor::dims;
//[Headers and namespace]

/// @page graph_gpu_single_op_partition_cpp
/// @section graph_gpu_single_op_partition_cpp_tutorial gpu_single_op_partition_tutorial() function
///
void gpu_single_op_partition_tutorial() {

    dim M = 32, K = 1024, N = 2048;

    dims src0_dims {M, K};
    dims src1_dims {K, N};

    /// @page graph_gpu_single_op_partition_cpp
    /// @subsection graph_gpu_single_op_partition_cpp_get_partition Build Graph and Get Partitions
    ///
    /// In this section, we are trying to create a partition containing the
    /// single op `matmul` without building a graph and getting partition.
    ///

    /// Create first `Matmul` op (#dnnl::graph::op) and attaches attributes
    /// to it, including `transpose_a` and `transpose_b`.
    /// @snippet gpu_single_op_partition.cpp Create matmul
    //[Create matmul]
    logical_tensor matmul_src0_desc {0, data_type::f32};
    logical_tensor matmul_src1_desc {1, data_type::f32};
    logical_tensor matmul_dst_desc {2, data_type::f32};
    op matmul(0, op::kind::MatMul, {matmul_src0_desc, matmul_src1_desc},
            {matmul_dst_desc}, "matmul");
    matmul.set_attr<bool>(op::attr::transpose_a, false);
    matmul.set_attr<bool>(op::attr::transpose_b, false);
    //[Create matmul]

    /// @page graph_gpu_single_op_partition_cpp
    /// @subsection graph_gpu_single_op_partition_cpp_compile Compile and Execute Partition
    ///
    /// In the real case, users like framework should provide device information
    /// at this stage. But in this example, we just use a self-defined device to
    /// simulate the real behavior.
    ///
    /// Create a #dnnl::graph::allocator with two user-defined
    /// #dnnl_graph_sycl_allocate_f and #dnnl_graph_sycl_deallocate_f
    /// call-back functions.
    ///
    /// @snippet gpu_single_op_partition.cpp Create allocator
    //[Create allocator]
    allocator alloc = sycl_interop::make_allocator(
            sycl_malloc_wrapper, sycl_free_wrapper);
    //[Create allocator]

    /// Define SYCL queue (code outside of oneDNN graph)
    /// @snippet sycl_getting_started.cpp Define sycl queue
    //[Define sycl queue]
    sycl::queue q = sycl::queue(
            sycl::gpu_selector_v, sycl::property::queue::in_order {});
    //[Define sycl queue]

    /// Create a #dnnl::engine based on SYCL device and context. Also,
    /// set a user-defined #dnnl::graph::allocator to this engine.
    ///
    /// @snippet sycl_getting_started.cpp Create engine
    //[Create engine]
    dnnl::engine eng = sycl_interop::make_engine_with_allocator(
            q.get_device(), q.get_context(), alloc);
    //[Create engine]

    /// Create a #dnnl::stream on a given engine
    ///
    /// @snippet gpu_single_op_partition.cpp Create stream
    //[Create stream]
    dnnl::stream strm = dnnl::sycl_interop::make_stream(eng, q);
    //[Create stream]

    // Memory buffers bound to the partition input/output tensors
    // that helps manage the lifetime of these tensors
    std::vector<std::shared_ptr<void>> data_buffer;

    // Mapping from logical tensor id to the concrete shapes.
    // In practical usage, concrete shapes and layouts are not given
    // until compilation stage, hence need this mapping to mock the step.
    std::unordered_map<size_t, dims> concrete_shapes {
            {0, src0_dims}, {1, src1_dims}};

    // Compile and execute the partitions, including the following steps:
    //
    // 1. Update the input/output logical tensors with concrete shape and layout
    // 2. Compile the partition
    // 3. Update the output logical tensors with queried ones after compilation
    // 4. Allocate memory and bind the data buffer for the partition
    // 5. Execute the partition
    //
    // Although they are not part of the APIs, these steps are essential for
    // the integration of Graph API., hence users need to implement similar
    // logic.

    /// Skip building graph and getting partition, and directly create
    /// the single-op partition
    ///
    /// @snippet cpu_single_op_partition.cpp Create partition
    //[Create partition]
    partition part(matmul, dnnl::engine::kind::gpu);
    //[Create partition]
    if (!part.is_supported()) {
        std::cout << "gpu_single_op_partition: Got unsupported partition, "
                     "users need to handle the operators by themselves."
                  << std::endl;
        return;
    }

    std::vector<logical_tensor> inputs = part.get_input_ports();
    std::vector<logical_tensor> outputs = part.get_output_ports();

    // Update input logical tensors with concrete shape and layout
    for (auto &input : inputs) {
        const auto id = input.get_id();
        // Create logical tensor with strided layout
        input = logical_tensor {id, input.get_data_type(), concrete_shapes[id],
                layout_type::strided};
    }

    // Update output logical tensors with concrete shape and layout
    for (auto &output : outputs) {
        const auto id = output.get_id();
        output = logical_tensor {id, output.get_data_type(),
                DNNL_GRAPH_UNKNOWN_NDIMS,
                // do not require concrete shape as the shape will be inferred
                // based on input shapes during compilation
                layout_type::strided};
    }

    /// Compile the partition to generate compiled partition with the
    /// input and output logical tensors.
    ///
    /// @snippet gpu_single_op_partition.cpp Compile partition
    //[Compile partition]
    compiled_partition cp = part.compile(inputs, outputs, eng);
    //[Compile partition]

    // Update output logical tensors with queried one
    for (auto &output : outputs) {
        const auto id = output.get_id();
        output = cp.query_logical_tensor(id);
    }

    // Allocate memory for the partition, and bind the data buffers with
    // input and output logical tensors
    std::vector<tensor> inputs_ts, outputs_ts;
    allocate_sycl_graph_mem(inputs_ts, inputs, data_buffer, q, eng);
    allocate_sycl_graph_mem(outputs_ts, outputs, data_buffer, q, eng);

    /// Execute the compiled partition on the specified stream.
    ///
    /// @snippet gpu_single_op_partition.cpp Execute compiled partition
    //[Execute compiled partition]
    cp.execute(strm, inputs_ts, outputs_ts);
    //[Execute compiled partition]

    // Wait for all compiled partition's execution finished
    strm.wait();

    /// @page graph_gpu_single_op_partition_cpp
    ///
    std::cout << "Graph:" << std::endl
              << " [matmul_src0] [matmul_src1]" << std::endl
              << "       \\       /" << std::endl
              << "         matmul" << std::endl
              << "            |" << std::endl
              << "        [matmul_dst]" << std::endl
              << "Note:" << std::endl
              << " '[]' represents a logical tensor, which refers to "
                 "inputs/outputs of the graph. "
              << std::endl;
}

int main(int argc, char **argv) {
    return handle_example_errors({validate_engine_kind(engine::kind::gpu)},
            gpu_single_op_partition_tutorial);
}
