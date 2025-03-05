/*******************************************************************************
* Copyright 2024-2025 Intel Corporation
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

/// @example cpu_matmul_coo.cpp
/// > Annotated version: @ref cpu_matmul_coo_cpp
///
/// This C++ API example demonstrates how to create and execute a
/// [MatMul](@ref dev_guide_matmul) primitive that uses a source tensor
/// encoded with the COO sparse encoding.
///
/// @page cpu_matmul_coo_cpp MatMul Primitive Example
///
/// @include cpu_matmul_coo.cpp

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "dnnl.hpp"
#include "example_utils.hpp"

using namespace dnnl;

bool check_result(dnnl::memory dst_mem) {
    // clang-format off
    const std::vector<float> expected_result = {8.750000, 11.250000, 2.500000,
                                                6.000000,  2.250000, 3.750000,
                                               19.000000, 15.500000, 5.250000,
                                                4.000000,  7.000000, 3.000000};
    // clang-format on
    std::vector<float> dst_data(expected_result.size());
    read_from_dnnl_memory(dst_data.data(), dst_mem);
    return expected_result == dst_data;
}

void sparse_matmul() {
    dnnl::engine engine(engine::kind::cpu, 0);

    const memory::dim M = 4;
    const memory::dim N = 3;
    const memory::dim K = 6;

    // A sparse matrix represented in the COO format.
    std::vector<float> src_coo_values = {2.5f, 1.5f, 1.5f, 2.5f, 2.0f};
    std::vector<int32_t> src_coo_row_indices = {0, 1, 2, 2, 3};
    std::vector<int32_t> src_coo_col_indices = {0, 2, 0, 5, 1};

    // clang-format off
    std::vector<float> weights_data = {3.5f, 4.5f, 1.0f,
                                       2.0f, 3.5f, 1.5f,
                                       4.0f, 1.5f, 2.5f,
                                       3.5f, 5.5f, 4.5f,
                                       1.5f, 2.5f, 5.5f,
                                       5.5f, 3.5f, 1.5f};
    // clang-format on

    const int nnz = static_cast<int>(src_coo_values.size());

    // Create a memory descriptor for COO format by providing information
    // about number of non-zero entries and data types of metadata.
    const auto src_coo_md = memory::desc::coo(
            {M, K}, memory::data_type::f32, nnz, memory::data_type::s32);
    const auto wei_md = memory::desc(
            {K, N}, memory::data_type::f32, memory::format_tag::oi);
    const auto dst_md = memory::desc(
            {M, N}, memory::data_type::f32, memory::format_tag::nc);

    // This memory is created for the given values and metadata of COO format.
    memory src_coo_mem(src_coo_md, engine,
            {src_coo_values.data(), src_coo_row_indices.data(),
                    src_coo_col_indices.data()});
    memory wei_mem(wei_md, engine, weights_data.data());
    memory dst_mem(dst_md, engine);

    dnnl::stream stream(engine);

    auto sparse_matmul_pd
            = matmul::primitive_desc(engine, src_coo_md, wei_md, dst_md);
    auto sparse_matmul_prim = matmul(sparse_matmul_pd);

    std::unordered_map<int, memory> sparse_matmul_args;
    sparse_matmul_args.insert({DNNL_ARG_SRC, src_coo_mem});
    sparse_matmul_args.insert({DNNL_ARG_WEIGHTS, wei_mem});
    sparse_matmul_args.insert({DNNL_ARG_DST, dst_mem});

    sparse_matmul_prim.execute(stream, sparse_matmul_args);
    stream.wait();
    if (!check_result(dst_mem)) throw std::runtime_error("Unexpected output.");
}

int main(int argc, char **argv) {
    return handle_example_errors({engine::kind::cpu}, sparse_matmul);
}
