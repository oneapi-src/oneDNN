/*******************************************************************************
* Copyright 2025 Arm Ltd. and affiliates
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
#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include "example_utils.hpp"
#include "oneapi/dnnl/dnnl_ukernel.hpp"

using namespace dnnl;
using namespace dnnl::ukernel;

using tag = memory::format_tag;
using dt = memory::data_type;

template <typename DataTypeA, typename DataTypeB, typename DataTypeBias,
        typename DataTypeCref>
void compute_ref(int M, int N, int K, std::vector<DataTypeA> A_vec,
        std::vector<DataTypeB> B_vec, std::vector<DataTypeBias> Bias_vec,
        std::vector<DataTypeCref> &C_ref_vec) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            C_ref_vec[m * N + n] = Bias_vec[n];
            for (int k = 0; k < K; k++) {
                C_ref_vec[m * N + n] += A_vec[m * K + k] * B_vec[k * N + n];
            }
        }
    }
}

template <typename DataTypeRef, typename DataTypeDst>
bool compare_vectors_l2_norms(const std::vector<DataTypeRef> ref,
        const std::vector<DataTypeDst> dst, int64_t K, const char *message) {
    double v1_l2 = 0, diff_l2 = 0;
    for (size_t n = 0; n < ref.size(); ++n) {
        float diff = static_cast<float>(ref[n]) - static_cast<float>(dst[n]);
        diff_l2 += diff * diff;
        v1_l2 += static_cast<float>(ref[n]) * static_cast<float>(ref[n]);
    }

    v1_l2 = std::sqrt(v1_l2);
    diff_l2 = std::sqrt(diff_l2);
    float threshold = 1e-5;
    bool failed = (diff_l2 / v1_l2) >= threshold || (diff_l2 / v1_l2) < 0;

    printf(""
           "\t%s - L2 Norm"
           "\n\tRelative_error: %g\n"
           "\n\tAccuracy check: %s\n\n",
            message, diff_l2 / v1_l2, failed ? "FAILED" : "OK");

    return failed;
}

void kleidiai_example() {
    // Create execution dnnl::engine. Needed for memory objects.
    dnnl::engine engine(engine::kind::cpu, 0);

    const memory::dim M = 32, N = 128, K = 128;

    memory::data_type a_dt = dt::f32;
    memory::data_type b_dt = dt::f32;
    memory::data_type bias_dt = dt::f32;
    memory::data_type c_dt = dt::f32;

    // Create KleidiAI ukernel object.
    brgemm brgemm_kai;

    try {
        brgemm_kai = brgemm(M, N, K, 1, K, N, N, dt::f32, dt::f32, dt::f32);
        brgemm_kai.finalize();
        brgemm_kai.generate();
    } catch (error &e) {
        if (e.status == dnnl_unimplemented)
            throw example_allows_unimplemented {
                    "Kernel is not supported on this platform.\n"};

        // on any other error just re-throw
        throw;
        return;
    }

    // Query the packing requirement from the kernel.
    const pack_type B_pack_t = brgemm::get_B_pack_type(dt::f32, dt::f32);

    // Execute
    // A, Bs, and C/D tensors dimensions.
    memory::dims A_dims = {M, K};
    memory::dims B_dims = {K, N};
    memory::dims C_dims = {M, N};
    memory::dims Bias_dims = {N};

    // Allocate buffers with user data.
    std::vector<float> A_vec(product(A_dims));
    std::vector<float> B_vec(product(B_dims));
    std::vector<float> Bias_vec(product(Bias_dims));

    std::vector<float> C_vec(product(C_dims)); // Result
    std::vector<float> C_ref_vec(product(C_dims)); // Ref. result

    // Initialize A
    std::generate(A_vec.begin(), A_vec.end(), []() {
        static float i = 12.154;
        static int sign_gen = 0;
        int sign = (sign_gen++ % 2) ? -1 : 1;
        float val = sign * (i++ / 3.14);

        return val;
    });

    // Initialize B
    std::generate(B_vec.begin(), B_vec.end(), []() {
        static float i = 36.394;
        static int sign_gen = 0;
        int sign = (sign_gen++ % 2) ? -1 : 1;
        float val = sign * (i++ / 3.14);

        return val;
    });

    // Bias
    std::generate(Bias_vec.begin(), Bias_vec.end(), []() {
        static float i = 102.128;
        static int sign_gen = 0;
        int sign = (sign_gen++ % 2) ? -1 : 1;
        float val = sign * (i++ / 3.14);

        return val;
    });

    // Create memories.
    // Note that all formats are `ab` except Bias.
    auto A_md = memory::desc(A_dims, a_dt, tag::ab);
    auto A_mem = memory(A_md, engine, A_vec.data());

    auto B_md = memory::desc(B_dims, b_dt, tag::ab);
    auto B_mem = memory(B_md, engine, B_vec.data());

    // Bias
    auto Bias_md = memory::desc(Bias_dims, bias_dt, tag::a);
    auto Bias_mem = memory(Bias_md, engine, Bias_vec.data());

    // Result
    auto C_md = memory::desc(C_dims, c_dt, tag::ab);
    auto C_mem = memory(C_md, engine, C_vec.data());

    // Pointers
    auto *A_ptr = reinterpret_cast<float *>(A_mem.get_data_handle());
    auto *B_ptr = reinterpret_cast<float *>(B_mem.get_data_handle());
    auto *Bias_ptr = reinterpret_cast<float *>(Bias_mem.get_data_handle());
    float *C_ptr = reinterpret_cast<float *>(C_mem.get_data_handle());

    // KAI ukernel execute section.
    // Setting bias argument into an attributes arguments storage.
    attr_params params;
    params.set_bias(Bias_ptr);

    transform pack_B(/* K = */ K, /* N = */ N,
            /* in_pack_type = */ B_pack_t, /* in_ld = */ N,
            /* out_ld = */ N, /* in_dt = */ b_dt, /* out_dt = */ c_dt);
    size_t packed_B_size = pack_B.get_output_size();
    float *B_packed_ptr = new float[packed_B_size];
    pack_B.execute(B_ptr, B_packed_ptr, params);

    //execute
    std::vector<std::pair<memory::dim, memory::dim>> A_B_offsets(1);
    A_B_offsets[0] = std::make_pair(-1, -1);

    brgemm_kai.execute(A_ptr, B_packed_ptr, A_B_offsets, C_ptr, nullptr);

    bool to_throw = false;
    compute_ref<float, float, float, float>(
            M, N, K, A_vec, B_vec, Bias_vec, C_ref_vec);

    to_throw = compare_vectors_l2_norms<float, float>(
            C_ref_vec, C_vec, K, "F32 kai");
    if (to_throw) {
        for (int m = 0; m < M; m++) {
            for (int n = 0; n < N; n++) {
                const float diff
                        = fabsf(C_ref_vec[m * N + n] - C_vec[m * N + n]);
                printf("Error: [%3d:%3d] Ref:%16.2f Got:%16.2f Diff:%16.2f\n",
                        m, n, C_ref_vec[m * N + n], C_vec[m * N + n], diff);
            }
        }
        throw status::runtime_error;
    }
}

int main(int argc, char **argv) {
    return handle_example_errors({dnnl::engine::kind::cpu}, kleidiai_example);
}
