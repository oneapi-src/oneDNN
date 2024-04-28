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

#include <iostream>
#include <memory>
#include <vector>
#include <unordered_map>
#include <unordered_set>

#include <assert.h>

#include "oneapi/dnnl/dnnl_graph.hpp"
#include "oneapi/dnnl/dnnl_graph_ocl.hpp"

#include "example_utils.hpp"
#include "graph_example_utils.hpp"

using namespace dnnl::graph;
using data_type = logical_tensor::data_type;
using layout_type = logical_tensor::layout_type;
using dim = logical_tensor::dim;
using dims = logical_tensor::dims;

void gpu_float_sdpa(data_type dtype, int batch_size, int seq_len, int num_head,
        int head_dim) {
    const engine::kind ekind = engine::kind::gpu;
    allocator alloc = ocl_interop::make_allocator(ocl_malloc_shared, ocl_free);

    cl_uint num_platforms = 0;
    OCL_CHECK(clGetPlatformIDs(0, NULL, &num_platforms));
    std::vector<cl_platform_id> platforms(num_platforms);
    if (num_platforms > 0) {
        OCL_CHECK(clGetPlatformIDs(num_platforms, platforms.data(), NULL));
    } else {
        throw "Cannot find any openCL platform!";
    }

    std::vector<cl_device_id> gpu_device_ids;
    for (cl_platform_id &platform_id : platforms) {
        cl_uint num_devices;
        if (!clGetDeviceIDs(
                    platform_id, CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices)) {
            std::vector<cl_device_id> device_ids(num_devices);
            OCL_CHECK(clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU,
                    num_devices, device_ids.data(), NULL));
            gpu_device_ids.insert(
                    gpu_device_ids.end(), device_ids.begin(), device_ids.end());
        }
    }
    if (gpu_device_ids.empty()) { throw "Cannot find any OpenCL device!"; }

    cl_device_id device_id = gpu_device_ids[0];
    cl_int err = 0;
    auto ctx = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);
    OCL_CHECK(err);

// clCreateCommandQueue is deprecated in OpenCL.
#ifdef CL_VERSION_2_0
    cl_command_queue q
            = clCreateCommandQueueWithProperties(ctx, device_id, nullptr, &err);
#else
    cl_command_queue q = clCreateCommandQueue(ctx, device_id, {}, &err);
#endif
    OCL_CHECK(err);
    dnnl::engine eng = dnnl::graph::ocl_interop::make_engine_with_allocator(
            device_id, ctx, alloc);
    stream strm = dnnl::ocl_interop::make_stream(eng, q);

    int size_per_head = head_dim / num_head;
    dims qkv_input_shape = {batch_size, num_head, seq_len, size_per_head};
    dims qk_output_shape = {batch_size, num_head, seq_len, seq_len};
    dims scale_shape = {1};
    dims attention_mask_shape = {batch_size, 1, 1, seq_len};
    dims qkv_transpose_order = {0, 2, 1, 3};
    dims qkv_transposed_shape = {batch_size, seq_len, num_head, size_per_head};
    dims qkv_reshaped_shape = {batch_size * seq_len, head_dim};

    size_t lt_id = 0;

    logical_tensor query_input {
            lt_id++, dtype, qkv_input_shape, layout_type::strided};
    logical_tensor key_input {
            lt_id++, dtype, qkv_input_shape, layout_type::strided};
    logical_tensor matmul_qk_out {
            lt_id++, dtype, qk_output_shape, layout_type::strided};
    op matmul_qk {0, op::kind::MatMul, {query_input, key_input},
            {matmul_qk_out}, "matmul_qk"};
    matmul_qk.set_attr<bool>(op::attr::transpose_b, true);

    logical_tensor scale_factor {lt_id++, dtype, scale_shape,
            layout_type::strided, logical_tensor::property_type::constant};
    logical_tensor scaled_qk_out {
            lt_id++, dtype, qk_output_shape, layout_type::strided};
    op scale_div {1, op::kind::Divide, {matmul_qk_out, scale_factor},
            {scaled_qk_out}, "scale_div"};

    logical_tensor attention_mask {
            lt_id++, dtype, attention_mask_shape, layout_type::strided};
    logical_tensor masked_qk_out {
            lt_id++, dtype, qk_output_shape, layout_type::strided};
    op mask_add {2, op::kind::Add, {scaled_qk_out, attention_mask},
            {masked_qk_out}, "mask_add"};

    logical_tensor softmax_out {
            lt_id++, dtype, qk_output_shape, layout_type::strided};
    op softmax {
            3, op::kind::SoftMax, {masked_qk_out}, {softmax_out}, "softmax"};
    softmax.set_attr<int64_t>(op::attr::axis, -1);

    logical_tensor value_input {
            lt_id++, dtype, qkv_input_shape, layout_type::strided};
    logical_tensor matmul_v_out {
            lt_id++, dtype, qkv_input_shape, layout_type::strided};
    op matmul_v {4, op::kind::MatMul, {softmax_out, value_input},
            {matmul_v_out}, "matmul_v"};

    logical_tensor qkv_transposed_out {
            lt_id++, dtype, qkv_transposed_shape, layout_type::strided};
    op transpose {5, op::kind::StaticTranspose, {matmul_v_out},
            {qkv_transposed_out}, "transpose"};
    transpose.set_attr<std::vector<int64_t>>(
            op::attr::order, qkv_transpose_order);

    logical_tensor qkv_reshaped_out {
            lt_id++, dtype, qkv_reshaped_shape, layout_type::strided};
    op reshape {6, op::kind::StaticReshape, {qkv_transposed_out},
            {qkv_reshaped_out}, "reshape"};
    reshape.set_attr(op::attr::special_zero, false);
    reshape.set_attr<std::vector<int64_t>>(op::attr::shape, qkv_reshaped_shape);

    graph g(ekind);
    g.add_op(matmul_qk);
    g.add_op(scale_div);
    g.add_op(mask_add);
    g.add_op(softmax);
    g.add_op(matmul_v);
    g.add_op(transpose);
    g.add_op(reshape);
    g.finalize();

    std::vector<partition> partitions = g.get_partitions();
    // just for testing purpose. User code should not make assertion for it.
    assert(partitions.size() == 1);

    std::vector<logical_tensor> inputs = partitions[0].get_input_ports();
    std::vector<logical_tensor> outputs = partitions[0].get_output_ports();
    compiled_partition sdp_cpartition
            = partitions[0].compile(inputs, outputs, eng);

    std::vector<tensor> inputs_ts, outputs_ts;
    std::vector<std::shared_ptr<void>> data_buffer;
    std::unordered_map<size_t, tensor> global_outputs_ts_map;
    // Input/output memory should be prepared by users. This helper funciton is
    // for testing purpose and not part of API.
    allocate_ocl_graph_mem(
            inputs_ts, inputs, data_buffer, global_outputs_ts_map, eng, true);
    allocate_ocl_graph_mem(outputs_ts, outputs, data_buffer,
            global_outputs_ts_map, eng, false);

    sdp_cpartition.execute(strm, inputs_ts, outputs_ts);
    strm.wait();
}

data_type str2data_type(const std::string &v) {
    if (v == "f32")
        return data_type::f32;
    else if (v == "bf16")
        return data_type::bf16;
    else if (v == "f16")
        return data_type::f16;
    else
        return data_type::undef;
}

int main(int argc, char **argv) {
    if (argc > 2) {
        std::cout << "One parameter (dtype) is needed: f32 / bf16 / f16 \n";
        return 0;
    }
    // if dtype is not provide, use f32 as default.
    const std::string dtype_str = argc == 2 ? argv[1] : "f32";
    data_type dtype = str2data_type(dtype_str);

    int batch_size = 1;
    int seq_len = 384;
    int num_head = 16;
    int head_dim = 1024;

    std::cout << "Running SDPA with data_type: " << dtype_str
              << ", batch_size: " << batch_size << ", seq_len: " << seq_len
              << ", num_head: " << num_head << ", head_dim: " << head_dim
              << std::endl;

    int exit_code = 0;
    try {
        gpu_float_sdpa(dtype, batch_size, seq_len, num_head, head_dim);
    } catch (dnnl::error &e) {
        std::cout << "oneDNN error caught: " << std::endl
                  << "\tStatus: " << dnnl_status2str(e.status) << std::endl
                  << "\tMessage: " << e.what() << std::endl;
        exit_code = 1;
    } catch (std::exception &e) {
        std::cout << "Error in the example: " << e.what() << "." << std::endl;
        exit_code = 2;
    }

    std::cout << "Example " << (exit_code ? "failed" : "passed") << " with "
              << dtype_str << "." << std::endl;

    return exit_code;
}
