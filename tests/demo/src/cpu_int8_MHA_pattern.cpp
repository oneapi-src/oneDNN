/*******************************************************************************
* Copyright 2022 Intel Corporation
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

/// @example cpu_int8_MHA_pattern.cpp
/// @copybrief cpu_int8_MHA_pattern_cpp
/// Annotated version: @ref cpu_int8_MHA_pattern_cpp

/// @page cpu_int8_MHA_pattern_cpp CPU example for bert MHA pattern
///
/// > Example code: @ref cpu_int8_MHA_pattern.cpp

#include <assert.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <unordered_map>

#include "oneapi/dnnl/dnnl_graph.hpp"

#include "common/execution_context.hpp"
#include "common/helpers_any_layout.hpp"
#include "common/utils.hpp"

#define assertm(exp, msg) assert(((void)msg, exp))

using namespace dnnl::graph;
using data_type = logical_tensor::data_type;
using layout_type = logical_tensor::layout_type;

// Test MHA int8 pattern compile and execute
// clang-format off
int main(int argc, char **argv) {
    std::cout
            << "========Example: MHA pattern========\n";

    engine::kind engine_kind = parse_engine_kind(argc, argv);
    if (engine_kind == engine::kind::gpu) {
        printf("Don't support gpu now\n");
        return -1;
    }

    // Step 2: Construct a example graph: MHA
    graph g(engine_kind);

    /// Create logical tensor
    std::cout << "Create logical tensor--------------------------";

    std::vector<int64_t> input_Q_dims {16, 384, 1024};
    std::vector<int64_t> input_K_dims {16, 384, 1024};
    std::vector<int64_t> input_V_dims {16, 384, 1024};
    std::vector<int64_t> fscore_dims {1};
    std::vector<int64_t> attn_mask_dims {16, 1, 1, 384};
    std::vector<int64_t> output_dims {16, 384, 1024};

    auto &id_mgr = logical_id_manager::get();

    logical_tensor input_Q_desc {id_mgr["input_Q"], data_type::u8, input_Q_dims, layout_type::strided};
    logical_tensor input_K_desc {id_mgr["input_K"], data_type::u8, input_K_dims, layout_type::strided};
    logical_tensor input_V_desc {id_mgr["input_V"], data_type::u8, input_V_dims, layout_type::strided};
    logical_tensor fscore_desc {id_mgr["fscore"], data_type::f32, fscore_dims, layout_type::strided};
    logical_tensor attn_mask_desc {id_mgr["attn_mask"], data_type::f32, attn_mask_dims, layout_type::strided};

    std::vector<int64_t> reshape_qkv_dims {16, 384, 16, 64};
    std::vector<int64_t> transpose_qkv_dims {16, 16, 384, 64};
    std::vector<int64_t> transpose_qkv_order {0, 2, 1, 3};
    // dequant Q
    logical_tensor dequant_Q_dst_desc {id_mgr["dequant_Q_dst"], data_type::f32, input_Q_dims, layout_type::strided};
    op dequant_Q {0, op::kind::Dequantize, {input_Q_desc}, {dequant_Q_dst_desc}, "dequant_Q"};
    dequant_Q.set_attr<std::vector<float>>(op::attr::scales, {0.1f});
    dequant_Q.set_attr<std::vector<int64_t>>(op::attr::zps, {10});
    dequant_Q.set_attr<std::string>(op::attr::qtype, "per_tensor");
    // reshape Q
    logical_tensor reshape_Q_dst_desc {id_mgr["reshape_Q_dst"], data_type::f32, reshape_qkv_dims, layout_type::strided};
    op reshape_Q {1, op::kind::StaticReshape, {dequant_Q_dst_desc}, {reshape_Q_dst_desc}, "reshape_Q"};
    reshape_Q.set_attr<std::vector<int64_t>>(op::attr::shape, reshape_qkv_dims);
    reshape_Q.set_attr<bool>(op::attr::special_zero, false);
    // transpose Q
    logical_tensor transpose_Q_dst_desc{id_mgr["transpose_Q_dst"], data_type::f32, transpose_qkv_dims, layout_type::strided};
    op transpose_Q {2, op::kind::StaticTranspose, {reshape_Q_dst_desc}, {transpose_Q_dst_desc}, "transpose_Q"};
    transpose_Q.set_attr<std::vector<int64_t>>(op::attr::order, transpose_qkv_order);

    // dequant K
    logical_tensor dequant_K_dst_desc {id_mgr["dequant_K_dst"], data_type::f32, input_K_dims, layout_type::strided};
    op dequant_K {3, op::kind::Dequantize, {input_K_desc}, {dequant_K_dst_desc}, "dequant_K"};
    dequant_K.set_attr<std::vector<float>>(op::attr::scales, {0.1f});
    dequant_K.set_attr<std::vector<int64_t>>(op::attr::zps, {10});
    dequant_K.set_attr<std::string>(op::attr::qtype, "per_tensor");
    // reshape K
    logical_tensor reshape_K_dst_desc {id_mgr["reshape_K_dst"], data_type::f32, reshape_qkv_dims, layout_type::strided};
    op reshape_K {4, op::kind::StaticReshape, {dequant_K_dst_desc}, {reshape_K_dst_desc}, "reshape_K"};
    reshape_K.set_attr<std::vector<int64_t>>(op::attr::shape, reshape_qkv_dims);
    reshape_K.set_attr<bool>(op::attr::special_zero, false);
    // transpose K
    logical_tensor transpose_K_dst_desc{id_mgr["transpose_K_dst"], data_type::f32, transpose_qkv_dims, layout_type::strided};
    op transpose_K {5, op::kind::StaticTranspose, {reshape_K_dst_desc}, {transpose_K_dst_desc}, "transpose_K"};
    transpose_K.set_attr<std::vector<int64_t>>(op::attr::order, transpose_qkv_order);

    // matmul qk
    std::vector<int64_t> matmul_qk_dst_dims = {16, 16, 384, 384};
    logical_tensor matmul_QK_dst_desc {id_mgr["matmul_QK_dst"], data_type::f32, matmul_qk_dst_dims, layout_type::strided};
    op matmul_QK {6, op::kind::MatMul, {transpose_Q_dst_desc, transpose_K_dst_desc}, {matmul_QK_dst_desc}, "matmul_QK"};
    matmul_QK.set_attr<bool>(op::attr::transpose_b, true);
    // div
    logical_tensor div_dst_desc {id_mgr["div_dst"], data_type::f32, matmul_qk_dst_dims, layout_type::strided};
    op div {7, op::kind::Divide, {matmul_QK_dst_desc, fscore_desc}, {div_dst_desc}, "div"};
    // add
    logical_tensor add_dst_desc {id_mgr["add_dst"], data_type::f32, matmul_qk_dst_dims, layout_type::strided};
    op add {8, op::kind::Add, {div_dst_desc, attn_mask_desc}, {add_dst_desc}, "add"};
    // softmax
    logical_tensor softmax_dst_desc {id_mgr["softmax_dst"], data_type::f32, matmul_qk_dst_dims, layout_type::strided};
    op softmax {9, op::kind::SoftMax, {add_dst_desc}, {softmax_dst_desc}, "softmax"};
    // quant softmax
    logical_tensor quant_softmax_dst_desc {id_mgr["quant_softmax_dst"], data_type::u8, matmul_qk_dst_dims, layout_type::strided};
    op quant_softmax {10, op::kind::Quantize, {softmax_dst_desc}, {quant_softmax_dst_desc}, "quant_softmax"};
    quant_softmax.set_attr<std::vector<float>>(op::attr::scales, {0.1f});
    quant_softmax.set_attr<std::vector<int64_t>>(op::attr::zps, {10});
    quant_softmax.set_attr<std::string>(op::attr::qtype, "per_tensor"); 
    // dequant softmax
    logical_tensor dequant_softmax_dst_desc {id_mgr["dequant_softmax_dst"], data_type::f32, matmul_qk_dst_dims, layout_type::strided};
    op dequant_softmax {11, op::kind::Dequantize, {quant_softmax_dst_desc}, {dequant_softmax_dst_desc}, "dequant_softmax"};
    dequant_softmax.set_attr<std::vector<float>>(op::attr::scales, {0.1f});
    dequant_softmax.set_attr<std::vector<int64_t>>(op::attr::zps, {10});
    dequant_softmax.set_attr<std::string>(op::attr::qtype, "per_tensor"); 

    // dequant V
    logical_tensor dequant_V_dst_desc {id_mgr["dequant_V_dst"], data_type::f32, input_V_dims, layout_type::strided};
    op dequant_V {12, op::kind::Dequantize, {input_V_desc}, {dequant_V_dst_desc}, "dequant_V"};
    dequant_V.set_attr<std::vector<float>>(op::attr::scales, {0.1f});
    dequant_V.set_attr<std::vector<int64_t>>(op::attr::zps, {10});
    dequant_V.set_attr<std::string>(op::attr::qtype, "per_tensor"); 
    // reshape V
    logical_tensor reshape_V_dst_desc {id_mgr["reshape_V_dst"], data_type::f32, reshape_qkv_dims, layout_type::strided};
    op reshape_V {13, op::kind::StaticReshape, {dequant_V_dst_desc}, {reshape_V_dst_desc}, "reshape_V"};
    reshape_V.set_attr<std::vector<int64_t>>(op::attr::shape, reshape_qkv_dims);
    reshape_V.set_attr<bool>(op::attr::special_zero, false);
    // transpose V
    logical_tensor transpose_V_dst_desc{id_mgr["transpose_V_dst"], data_type::f32, transpose_qkv_dims, layout_type::strided};
    op transpose_V {14, op::kind::StaticTranspose, {reshape_V_dst_desc}, {transpose_V_dst_desc}, "transpose_V"};
    transpose_V.set_attr<std::vector<int64_t>>(op::attr::order, transpose_qkv_order);
    
    // matmul v
    logical_tensor matmul_V_dst_desc {id_mgr["matmul_V_dst"], data_type::f32, layout_type::strided};
    op matmul_V {15, op::kind::MatMul, {dequant_softmax_dst_desc, transpose_V_dst_desc}, {matmul_V_dst_desc}, "matmul_V"};
    // transpose out
    logical_tensor transpose_out_dst_desc {id_mgr["transpose_out_dst"], data_type::f32, reshape_qkv_dims, layout_type::strided};
    op transpose_out {16, op::kind::StaticTranspose, {matmul_V_dst_desc}, {transpose_out_dst_desc}, "transpose_out"};
    transpose_out.set_attr<std::vector<int64_t>>(op::attr::order, transpose_qkv_order);
    // reshape out
    logical_tensor reshape_out_dst_desc {id_mgr["reshape_out_dst"], data_type::f32, output_dims, layout_type::strided};
    op reshape_out {17, op::kind::StaticReshape, {transpose_out_dst_desc}, {reshape_out_dst_desc}, "reshape_out"};
    reshape_out.set_attr<std::vector<int64_t>>(op::attr::shape, output_dims);
    reshape_out.set_attr<bool>(op::attr::special_zero, false);
    // quant out
    logical_tensor quant_out_dst_desc {id_mgr["quant_out_dst"], data_type::u8, output_dims, layout_type::strided};
    op quant_out {18, op::kind::Quantize, {reshape_out_dst_desc}, {quant_out_dst_desc}, "quant_out"};
    quant_out.set_attr<std::vector<float>>(op::attr::scales, {0.1f});
    quant_out.set_attr<std::vector<int64_t>>(op::attr::zps, {10});
    quant_out.set_attr<std::string>(op::attr::qtype, "per_tensor"); 
    std::cout << "Success!\n";

    std::unordered_map<size_t, op::kind> op_id_kind_map {{0, op::kind::Dequantize}, 
        {1, op::kind::StaticReshape}, {2, op::kind::StaticTranspose}, {3, op::kind::Dequantize}, 
        {4, op::kind::StaticReshape}, {5, op::kind::StaticTranspose}, {6, op::kind::MatMul},
        {7, op::kind::Divide}, {8, op::kind::Add}, {9, op::kind::SoftMax}, {10, op::kind::Quantize},
        {11, op::kind::Dequantize}, {12, op::kind::Dequantize}, {13, op::kind::StaticReshape}, 
        {14, op::kind::StaticTranspose}, {15, op::kind::MatMul}, {16, op::kind::StaticTranspose}, 
        {17, op::kind::StaticReshape}, {18, op::kind::Quantize}};
    /// Add OP
    std::cout << "Add OP to graph--------------------------------";
    g.add_op(dequant_Q);
    g.add_op(reshape_Q);
    g.add_op(transpose_Q);
    g.add_op(dequant_K);
    g.add_op(reshape_K);
    g.add_op(transpose_K);
    g.add_op(matmul_QK);
    g.add_op(div);
    g.add_op(add);
    g.add_op(softmax);
    g.add_op(quant_softmax);
    g.add_op(dequant_softmax);
    g.add_op(dequant_V);
    g.add_op(reshape_V);
    g.add_op(transpose_V);
    g.add_op(matmul_V);
    g.add_op(transpose_out);
    g.add_op(reshape_out);
    g.add_op(quant_out);
    id_mgr.freeze(); // graph is built up, and the arguments set could be frozen
    std::cout << "Success!\n";

    // Step 3: Filter and get partitions
    /// Setting `DNNL_GRAPH_DUMP=1` can save internal graphs before/after graph fusion into dot files
    std::cout << "Filter and get partition-----------------------";
    auto partitions = g.get_partitions(partition::policy::fusion);
    std::cout << "Success!\n";

    std::cout << "Number of returned partitions: " << partitions.size() << "\n";
    if (partitions.size() == 1) { std::cout << "The MHA pattern hits the graph compiler backend.\n"; } 
    for (size_t i = 0; i < partitions.size(); ++i) {
        std::cout << "Partition[" << partitions[i].get_id()
                  << "]'s supporting status: "
                  << (partitions[i].is_supported() ? "true" : "false") << "\n";
    }

    /// mark the output logical tensors of partition as ANY layout enabled
    std::unordered_set<size_t> id_to_set_any_layout;
    set_any_layout(partitions, id_to_set_any_layout);

    /// construct a new engine
    engine e {engine_kind, 0};

    /// construct a new stream
    stream s {e};

    std::vector<compiled_partition> c_partitions(partitions.size());

    // mapping from id to tensors
    tensor_map tm;

    // mapping from id to queried logical tensor from compiled partition
    // used to record the logical tensors that are previously enabled with ANY layout
    std::unordered_map<size_t, logical_tensor> id_to_queried_logical_tensors;

    for (size_t i = 0; i < partitions.size(); ++i) {
        if (partitions[i].is_supported()) {
            std::cout << "\nPartition[" << partitions[i].get_id() << "] is being processed.\n";
            std::vector<logical_tensor> inputs = partitions[i].get_in_ports();
            std::vector<logical_tensor> outputs = partitions[i].get_out_ports();

            /// replace input logical tensor with the queried one
            replace_with_queried_logical_tensors(inputs, id_to_queried_logical_tensors);

            /// update output logical tensors with ANY layout
            update_tensors_with_any_layout(outputs, id_to_set_any_layout);

            std::cout << "Compiling--------------------------------------";
            /// compile to generate compiled partition
            c_partitions[i] = partitions[i].compile(inputs, outputs, e);
            std::cout << "Success!\n";

            record_queried_logical_tensors(partitions[i].get_out_ports(), c_partitions[i],
                id_to_queried_logical_tensors);

            std::cout << "Creating tensors and allocating memory buffer--";
            std::vector<tensor> input_ts = tm.construct_and_initialize_tensors(inputs, c_partitions[i], e, 1);
            std::vector<tensor> output_ts = tm.construct_and_initialize_tensors(outputs, c_partitions[i], e, 0);
            std::cout << "Success!\n";

            std::cout << "Executing compiled partition-------------------";
            /// execute the compiled partition
            c_partitions[i].execute(s, input_ts, output_ts);
            std::cout << "Success!\n";
        } else {
            std::vector<size_t> unsupported_op_ids = partitions[i].get_ops();
            assertm(unsupported_op_ids.size() == 1, "Unsupported partition only "
                "contains single op.");
            if (op_id_kind_map[unsupported_op_ids[0]] == op::kind::Wildcard) {
                std::cout << "\nWarning (actually an error): partition " << partitions[i].get_id() <<
                        " contains only a Wildcard op which cannot be computed.\n";
            } else {
                /// Users need to write implementation code by themselves.
                continue;
            }
        }
    }

    std::cout << "Check correctness------------------------------";
    std::cout << "Skipped!\n";

    std::cout << "============Run Example Successfully===========\n";

    return 0;
}
// clang-format on
