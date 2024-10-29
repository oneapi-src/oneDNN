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
/// @example int4_weights_decompression.cpp
/// > Annotated version: @ref int4_weights_decompression.cpp
///
/// @page int4_weights_decompression
/// C++ API example demonstrating how one can use
/// [MatMul](@ref dev_guide_matmul) with int4 compressed weights.
///
/// Concepts:
/// - AWQ (activation-aware quantization)
///   - Scales: dnnl::primitive_attr::set_scales()
///   - Zero points: dnnl::primitive_attr::set_zero_points()
/// - [Operation fusion](@ref dev_guide_attributes_post_ops)
/// - Create primitive once, use multiple times
/// - Weights pre-packing: use #dnnl::memory::format_tag::any
///
/// @page int4_weights_decompression_matmul_cpp MatMul Tutorial: weights
/// decompression
/// @copydetails int4_weights_decompression_matmul_cpp
///
/// Assumptions:
/// 1. The shape of the weights (matrix \f$B(K, N)\f$) is known in advance, the
///    data type is `int4` and shifted from 0 (i.e. the zero point is not 0).
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
// Comparing two vectors by calculating their L2 norms and the L2 norm of their
// difference Checking if the difference is within a calculated threshold The
// function returns 0 if the vectors are considered similar, otherwise it
// returns 1. --Rupak
int compare_vectors(const std::vector<float> &v1, const std::vector<float> &v2,
        int64_t K, const char *message) {
    double v1_l2 = 0, diff_l2 = 0;
    for (size_t n = 0; n < v1.size(); ++n) {
        float diff = v1[n] - v2[n];
        v1_l2 += v1[n] * v1[n];
        diff_l2 += diff * diff;
    }

    v1_l2 = std::sqrt(v1_l2);
    diff_l2 = std::sqrt(diff_l2);

    // Finding the reasonable (tight and accurate) threshold is quite difficult
    // problem.
    // The implementation testing might also use special data filling to
    // alleviate issues related to the finite precision arithmetic.
    // However, in simple cases the machine epsilon multiplied by log(K) should
    // work reasonably well.
    const double threshold = std::numeric_limits<float>::epsilon()
            * std::log(std::max(2., (double)K));
    bool ok = diff_l2 <= threshold * v1_l2;

    printf("%s\n\tL2 Norms"
           "\n\t\tReference matrix:%g\n\t\tError:%g\n\t\tRelative_error:%g\n"
           "\tAccuracy check: %s\n",
            message, v1_l2, diff_l2, diff_l2 / v1_l2, ok ? "OK" : "FAILED");

    return ok ? 0 : 1;
}

} // namespace

// Floating point MatMul
// Inputs:
// - Shape: M, N, K
// - Matrices A and B
// Outputs:
// - Matrix C
void ref_compute_matmul_f32(int64_t M, int64_t N, int64_t K, int64_t G,
        std::vector<float> &A_f32, std::vector<float> &B_f32,
        std::vector<float> &zp_B_f32, std::vector<float> &sc_B,
        std::vector<float> &C_f32) {
    // Perform the GEMM operation
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            for (int k = 0; k < K; ++k) {
                // Decompress the weight
                int64_t idx1 = k * N + n;
                int64_t idx2 = (k / G) * N + n;
                float decompressed_B
                        = (B_f32[idx1] - zp_B_f32[idx1]) * sc_B[idx2];
                // Perform the multiplication and accumulation
                C_f32[m * N + n] += A_f32[m * K + k] * decompressed_B;
            }
        }
    }
}

// Create a MatMul primitive descriptor for the following op:
// C_f32 = A_f32 * (B_s4 - zp_B) * sc_B[:] --Rupak
matmul::primitive_desc matmul_pd_create(
        int64_t M, int64_t N, int64_t K, int64_t G, const engine &eng) {

    memory::desc a_md({M, K}, memory::data_type::f32, {K, 1}); // M x K layout
    memory::desc b_md({K, N}, memory::data_type::s4,
            memory::format_tag::any); // K x N layout
    memory::desc c_md({M, N}, memory::data_type::f32, {N, 1}); // M x N layout

    // Create attributes and indicate that the alpha and zero points are
    // runtime parameters
    primitive_attr attr;
    // Set scales with multiple scales along K and N dimensions and with groups
    // along K.
    attr.set_scales(DNNL_ARG_WEIGHTS,
            /* mask */ (1 << 0) + (1 << 1), {G, 1}, memory::data_type::f32);

    // Set zero points with s4 data type.
    // The mask determines which dimensions the zero points are applied to.
    // Current mask value (1 << 0) + (1 << 1) means zero points are applied
    // both along K and N dimensions.
    // Changing the mask value would alter the dimensions along which the zero
    // points are applied. For example:
    // - mask = (1 << 0) would apply zero points only along the K dimension.
    // - mask = (1 << 1) would apply zero points only along the N dimension.
    int mask = (1 << 0)
            + (1 << 1); // zero points both along K and N dimensions --Rupak
    memory::dims groups = {};
    attr.set_zero_points(DNNL_ARG_WEIGHTS, mask, groups, memory::data_type::s4);

    // Set fpmath mode with `apply_to_int=true` to apply fpmath mode behavior to
    // integral primitives (in this example, matmul).
    attr.set_fpmath_mode(fpmath_mode::f16, true);

    // Create a MatMul primitive descriptor
    return matmul::primitive_desc(eng, a_md, b_md, c_md, attr);
}

// Function to perform matrix multiplication with int4 weights decompression
// using oneDNN --Rupka
void weights_decompression_matmul(int64_t M, int64_t N, int64_t K, int64_t G,
        std::vector<float> &A_f32, std::vector<float> &B_f32,
        std::vector<float> &zp_B_f32, std::vector<float> &sc_B,
        std::vector<float> &C_f32, const engine &eng) {
    auto matmul_pd = matmul_pd_create(M, N, K, G, eng);
    stream s(eng);

    // Pre-packed weights stored as int4
    memory B_s4_mem(matmul_pd.weights_desc(), eng);
    {
        memory B_f32_mem(
                {{K, N}, memory::data_type::f32, memory::format_tag::ab}, eng);
        write_to_dnnl_memory(B_f32.data(), B_f32_mem);
        reorder(B_f32_mem, B_s4_mem).execute(s, B_f32_mem, B_s4_mem);
        s.wait();
    }
    matmul matmul_p(matmul_pd);

    // input of the current layer / operation
    memory A_f32_mem({{M, K}, memory::data_type::f32, {K, 1}}, eng);
    // De-quantization parameters (eg. Scale and Shift)
    const int64_t n_groups = K / G;
    memory sc_B_mem({{N, n_groups}, memory::data_type::f32, {1, N}}, eng);

    // Pre-packed zp stored as int4
    // A unique zero point is used for each weight in this example
    // Allocates memory for zp_B_s4_mem with specified dimensions and data type.
    // --Rupak
    memory zp_B_s4_mem({{K, N}, memory::data_type::s4, {1, K}}, eng);
    {
        memory zp_B_f32_mem({{K, N}, memory::data_type::f32, {1, K}}, eng);
        write_to_dnnl_memory(zp_B_f32.data(), zp_B_f32_mem);
        reorder(zp_B_f32_mem, zp_B_s4_mem)
                .execute(s, zp_B_f32_mem, zp_B_s4_mem);
        s.wait();
    }

    write_to_dnnl_memory(A_f32.data(), A_f32_mem);
    write_to_dnnl_memory(sc_B.data(), sc_B_mem);

    // output - no initialization required
    memory C_f32_mem({{M, N}, memory::data_type::f32, {N, 1}}, eng);

    matmul_p.execute(s,
            {{DNNL_ARG_SRC, A_f32_mem}, {DNNL_ARG_WEIGHTS, B_s4_mem},
                    {DNNL_ARG_DST, C_f32_mem},
                    {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, sc_B_mem},
                    {DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS,
                            zp_B_s4_mem}});
    s.wait();
}

// Compares the results of reference matrix multiplication and oneDNN weights
// decompression. --Rupak
void compare_ref_and_weights_decompression(engine::kind engine_kind) {
    engine eng(engine_kind, 0);

    // MatMul parameters
    const int64_t M = 1, N = 4096, K = 1024;
    // Quantization Group size for scales
    const int64_t G = 64;

    // Prepare matrices
    std::vector<float> A_f32(M * K), C_ref(M * N), sc_B(K * N / G);
    std::vector<float> B_f32(K * N);
    std::vector<float> zp_B_f32(K * N);
    init_vector(A_f32);
    init_vector(B_f32);
    init_vector(sc_B);
    init_vector(zp_B_f32);
    init_vector(C_ref);
    std::vector<float> C_onednn = C_ref;

    // Compute _true_ C_ref result
    ref_compute_matmul_f32(M, N, K, G, A_f32, B_f32, zp_B_f32, sc_B, C_ref);

    // Compute _true_ C_onednn result
    weights_decompression_matmul(
            M, N, K, G, A_f32, B_f32, zp_B_f32, sc_B, C_onednn, eng);

    int rc = 0;
    rc |= compare_vectors(
            C_ref, C_onednn, K, "Compare ref vs oneDNN weights decompression");
    if (rc) throw std::logic_error("The resulting matrices diverged too much.");
}

int main(int argc, char **argv) {
    engine::kind engine_kind = parse_engine_kind(argc, argv);
    return handle_example_errors(
            compare_ref_and_weights_decompression, engine_kind);
}