/*******************************************************************************
* Copyright 2021 Intel Corporation
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

/// @example cpu_single_op_partition_matmul.cpp
/// @copybrief cpu_single_op_partition_matmul_cpp
/// Annotated version: @ref cpu_single_op_partition_matmul_cpp

/// @page cpu_single_op_partition_matmul_cpp CPU example for single operator partition
///
/// > Example code: @ref cpu_single_op_partition_matmul.cpp

#include "oneapi/dnnl/dnnl_graph.hpp"

#include <iostream>

using namespace dnnl::graph;
using data_type = logical_tensor::data_type;
using layout_type = logical_tensor::layout_type;

// clang-format off
int main(int argc, char **argv) {
    std::cout << "======== Example: Single operator partition ========\n";

    const engine::kind ekind = engine::kind::cpu;

    /// Create logical tensor
    std::cout << "Create logical tensors -------------------------------";

    std::vector<int64_t> src0_dims {32, 1024};
    std::vector<int64_t> src1_dims {1024, 2048};
    std::vector<int64_t> dst_dims {32, 2048};

    logical_tensor src0 {0, data_type::f32, src0_dims, layout_type::strided};
    logical_tensor src1 {1, data_type::f32, src1_dims, layout_type::strided};
    logical_tensor dst {2, data_type::f32, dst_dims, layout_type::strided};
    std::cout << "Success!\n";

    /// Create matmul operator
    std::cout << "Create matmul operator -------------------------------";
    op matmul {3, op::kind::MatMul, "matmul"};

    matmul.add_inputs({src0, src1});
    matmul.add_output(dst);

    matmul.set_attr<bool>("transpose_a", false);
    matmul.set_attr<bool>("transpose_b", false);
    std::cout << "Success!\n";

    /// Create partition
    std::cout << "Create a partition with the matmul operator ----------";
    partition part {matmul, ekind};
    std::cout << "Success!\n";

    /// Compile the partition
    std::cout << "Compile the partition --------------------------------";
    /// construct a new engine
    engine eng {ekind, 0};

    // compilation
    compiled_partition cp = part.compile({src0, src1}, {dst}, eng);
    std::cout << "Success!\n";

    /// Execute the compiled partition
    std::cout << "Execute the compiled partition -----------------------";
    // construct tensors
    std::vector<float> data0(32 * 1024);
    std::vector<float> data1(1024 * 2048);
    std::vector<float> data2(32 * 2048);

    tensor t0 {src0, data0.data()};
    tensor t1 {src1, data1.data()};
    tensor t2 {dst, data2.data()};

    /// construct a new stream
    stream stm {eng};
    /// execute the compile partition
    cp.execute(stm, {t0, t1}, {t2});
    std::cout << "Success!\n";
    std::cout << "============ Run Example Successfully ==============\n";

    return 0;
}
// clang-format on
