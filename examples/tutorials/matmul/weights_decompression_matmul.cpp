/*******************************************************************************
* Copyright 2023-2024 Intel Corporation
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

/// @example weights_decompression_matmul.cpp
/// > Annotated version: @ref weights_decompression_matmul_cpp
///
/// @page weights_decompression_matmul_cpp_short
/// C++ API example demonstrating how one can use
/// [MatMul](@ref dev_guide_matmul) with compressed weights.
///
/// Concepts:
/// - Asymmetric quantization
///   - Scales: dnnl::primitive_attr::set_scales()
///   - Zero points: dnnl::primitive_attr::set_zero_points()
/// - [Operation fusion](@ref dev_guide_attributes_post_ops)
/// - Create primitive once, use multiple times
/// - Weights pre-packing: use #dnnl::memory::format_tag::any
///
/// @page weights_decompression_matmul_cpp MatMul Tutorial: weights decompression
/// @copydetails weights_decompression_matmul_cpp_short
///
/// Assumptions:
/// 1. The shape of the weights (matrix \f$B(K, N)\f$) is known in advance, the
///    data type is `int8_t` and shifted from 0 (i.e. the zero point is not 0).
/// 2. The source matrix \f$A\f$ and destination matrix \f$C\f$ have floating
///    point data type.
/// 3. Scaling (re-quantization) factor specified at run-time only.
///
/// Since the shape of weights is known in advance, the MatMul weights can be
/// created with format tag #dnnl::memory::format_tag::any to enable the library
/// to choose the most appropriate layout for best performance.
///
/// @warning
/// The format tag #dnnl::memory::format_tag::any doesn't work for memory
/// descriptors that have one or more unknown dimensions and/or strides.
///
/// @include weights_decompression_matmul.cpp

#include <cassert>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

#include "oneapi/dnnl/dnnl.hpp"

#include "example_utils.hpp"

using namespace dnnl;

namespace {

void init_vector(std::vector<float> &v) {
    std::mt19937 gen;
    std::uniform_real_distribution<float> u(0, 1);
    for (auto &e : v)
        e = u(gen);
}

} // namespace

int number_of_runs = 1;

// Create a MatMul primitive descriptor for the following op:
// C_f32 = A_f32 * (B_s8 - zp_B) * sc_B[:]
//
// Here:
// - Matrices A and C are of f32 data type.
// - The B matrix is stored as int8_t, its zero point is zp_B, and all its
//   dimensions are known. This matrix can be a matrix of compressed weights
//   in an MLP topology.
// - The weights scaling values are not known at the primitive creation time.
matmul::primitive_desc matmul_pd_create(
        int64_t M, int64_t N, int64_t K, int64_t G, const engine &eng) {

    memory::desc a_md({M, K}, memory::data_type::f32, {K, 1}); // M x K layout
    memory::desc b_md({K, N}, memory::data_type::s8, memory::format_tag::any);
    memory::desc c_md({M, N}, memory::data_type::f32, {N, 1}); // M x N layout

    // Create attributes and indicate that the alpha and zero points are
    // runtime parameters
    primitive_attr attr;
    // Set scales with multiple scales along K and N dimensions and with groups along K.
    attr.set_scales(DNNL_ARG_WEIGHTS,
            /* mask */ (1 << 0) + (1 << 1), {G, 1}, memory::data_type::f32);
    // Set a single zero point with s8 data type.
    attr.set_zero_points(
            DNNL_ARG_WEIGHTS, /* mask */ 0, {}, memory::data_type::s8);
    // Set fpmath mode with `apply_to_int=true` to apply fpmath mode behavior to
    // integral primitives (in this example, matmul).
    attr.set_fpmath_mode(fpmath_mode::bf16, true);

    // Create a MatMul primitive descriptor
    return matmul::primitive_desc(eng, a_md, b_md, c_md, attr);
}

void prepare_input(memory &A_f32_mem, memory &sc_B_mem, memory &zp_B_mem) {
    int64_t M = A_f32_mem.get_desc().get_dims()[0];
    int64_t N = sc_B_mem.get_desc().get_dims()[0];
    int64_t K = A_f32_mem.get_desc().get_dims()[1];
    int64_t NUM_G = sc_B_mem.get_desc().get_dims()[1];

    std::vector<float> A_f32(M * K);
    init_vector(A_f32);

    std::vector<float> sc_B(NUM_G * N);
    init_vector(sc_B);

    int8_t zp_B = 2;

    write_to_dnnl_memory(A_f32.data(), A_f32_mem);
    write_to_dnnl_memory(&zp_B, zp_B_mem);
    write_to_dnnl_memory(sc_B.data(), sc_B_mem);
}

void infer(const matmul &matmul_p, int64_t M, int64_t N, int64_t K, int64_t G,
        const memory &B_s8_mem, const engine &eng) {
    // input of the current layer / operation
    memory A_f32_mem({{M, K}, memory::data_type::f32, {K, 1}}, eng);
    // De-quantization parameters (eg. Scale and Shift)
    const int64_t n_groups = K / G;
    memory sc_B_mem({{N, n_groups}, memory::data_type::f32, {1, N}}, eng);
    memory zp_B_mem({{1}, memory::data_type::s8, {1}}, eng);

    // the function below fills dnnl::memory with some values
    // these memories, typically, come from the previous layers / operations
    // with meaningful data inside
    prepare_input(A_f32_mem, sc_B_mem, zp_B_mem);

    // output - no initialization required
    memory C_f32_mem({{M, N}, memory::data_type::f32, {N, 1}}, eng);

    stream s(eng);
    for (int run = 0; run < number_of_runs; ++run)
        matmul_p.execute(s,
                {{DNNL_ARG_SRC, A_f32_mem}, {DNNL_ARG_WEIGHTS, B_s8_mem},
                        {DNNL_ARG_DST, C_f32_mem},
                        {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, sc_B_mem},
                        {DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS,
                                zp_B_mem}});
    s.wait();
}

void weights_decompression_matmul(engine::kind engine_kind) {
    engine eng(engine_kind, 0);

    const int64_t K = 64;
    const int64_t N = 1000;
    const int64_t M = 100;
    // Quantization Group size for scales. Must be divisible by 32.
    const int64_t G = K / 2;

    auto matmul_pd = matmul_pd_create(M, N, K, G, eng);

    // Original weights stored as float in a known format
    std::vector<float> B_f32(K * N);
    init_vector(B_f32);

    // Pre-packed weights stored as int8_t
    memory B_s8_mem(matmul_pd.weights_desc(), eng);
    {
        stream s(eng);
        memory B_f32_mem(
                {{K, N}, memory::data_type::f32, memory::format_tag::ab}, eng);
        write_to_dnnl_memory(B_f32.data(), B_f32_mem);
        reorder(B_f32_mem, B_s8_mem).execute(s, B_f32_mem, B_s8_mem);
        s.wait();
    }

    matmul matmul_p(matmul_pd);

    infer(matmul_p, M, N, K, G, B_s8_mem, eng);
}

int main(int argc, char **argv) {
    engine::kind engine_kind = parse_engine_kind(argc, argv);
    // GPU is not supported
    if (engine_kind != engine::kind::cpu) return 0;
    return handle_example_errors(weights_decompression_matmul, engine_kind);
}
