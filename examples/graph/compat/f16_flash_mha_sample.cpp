/*
 * Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

//#include "f16_flash_mha_sample.h"
//#include <cudnn_frontend.h>
#include "compat_helpers.hpp"

static bool allowAllConfig(
        compat_0_x::onednnBackendDescriptor_t engine_config) {
    (void)engine_config;
    return false;
}

// Used for MHA
void generateMHAStrides(int64_t b, int64_t h, int64_t s_q, int64_t s_kv,
        int64_t d, int64_t *strideA, MHA_Layout layout, MHA_Matrix matrix) {

    // TODO: support mapping of QKV_INTERLEAVED, KV_INTERLEAVED and
    // SBH_INTERLEAVED layouts in cudnn graph
    constexpr int batch_dim_idx = 0;
    constexpr int head_dim_idx = 1;
    constexpr int seqlen_dim_idx = 2;
    constexpr int hidden_dim_idx = 3;

    constexpr int seqlen_transpose_dim_idx = 3;
    constexpr int hidden_transpose_dim_idx = 2;

    constexpr int seqlen_q_dim_idx = 2;
    constexpr int seqlen_kv_dim_idx = 3;

    switch (matrix) {
        case MHA_Matrix::Q_Matrix:
            strideA[hidden_dim_idx] = 1;
            strideA[seqlen_q_dim_idx] = d;
            strideA[head_dim_idx] = s_q * d;
            strideA[batch_dim_idx] = s_q * d * h;
            break;
        case MHA_Matrix::K_Matrix_Transpose:
            strideA[seqlen_transpose_dim_idx] = 1;
            strideA[hidden_transpose_dim_idx] = s_kv;
            strideA[head_dim_idx] = s_kv * d;
            strideA[batch_dim_idx] = s_kv * d * h;
            break;
        case MHA_Matrix::S_Matrix:
            strideA[seqlen_kv_dim_idx] = 1;
            strideA[seqlen_q_dim_idx] = s_kv;
            strideA[head_dim_idx] = s_q * s_kv;
            strideA[batch_dim_idx] = h * s_q * s_kv;
            break;
    }
}

static compat_0_x::Tensor tensor_create(compat_0_x::DataType_t type, int64_t id,
        compat_0_x::lt::dims const dim, compat_0_x::lt::dims const stride,
        bool is_virtual, bool is_value) {
    int nbDims = 4;
    auto tensor_created
            = compat_0_x::TensorBuilder()
                      .setDim(nbDims, dim)
                      .setStride(nbDims, stride)
                      .setId(id)
                      .setAlignment(
                              16) // 16B alignment is needed to run a tensor core engine
                      .setDataType(type)
                      .setVirtual(is_virtual)
                      .setByValue(is_value)
                      .build();
    //std::cout << tensor_created.describe() << std::endl;
    return tensor_created;
}

static compat_0_x::Tensor createQKBMM(int64_t b, int64_t h, int64_t s_q,
        int64_t s_kv, int64_t d, MHA_Layout layout,
        compat_0_x::DataType_t tensorType,
        std::vector<compat_0_x::Operation> &ops) {
    // Creates the necessary tensor descriptors
    int64_t q_dim[4] = {b, h, s_q, d};
    int64_t q_stride[4];
    generateMHAStrides(
            b, h, s_q, s_kv, d, q_stride, layout, MHA_Matrix::Q_Matrix);

    int64_t k_dim[4] = {b, h, d, s_kv};
    int64_t k_stride[4];
    generateMHAStrides(b, h, s_q, s_kv, d, k_stride, layout,
            MHA_Matrix::K_Matrix_Transpose);

    int64_t s_dim[4] = {b, h, s_q, s_kv};
    int64_t s_stride[4];
    generateMHAStrides(
            b, h, s_q, s_kv, d, s_stride, layout, MHA_Matrix::S_Matrix);

    //     auto qTensor
    //             = tensor_create(tensorType, Q_ID, q_dim, q_stride, false, false);
    //     auto kTransposeTensor = tensor_create(
    //             tensorType, K_ID, k_dim, k_stride, false, false); // is virtual
    //     // first GEMM output
    //     auto sTensor = tensor_create(dnnl::graph::logical_tensor::data_type::bf16,
    //             O_ID, s_dim, s_stride, true, false); // is virtual

    auto qTensor = tensor_create(tensorType, Q_ID,
            std::vector<int64_t>(q_dim, q_dim + 4),
            std::vector<int64_t>(q_stride, q_stride + 4), false, false);
    auto kTransposeTensor = tensor_create(tensorType, K_ID,
            std::vector<int64_t>(k_dim, k_dim + 4),
            std::vector<int64_t>(k_stride, k_stride + 4), false, false);
    auto sTensor = tensor_create(dnnl::graph::logical_tensor::data_type::bf16,
            O_ID, std::vector<int64_t>(s_dim, s_dim + 4),
            std::vector<int64_t>(s_stride, s_stride + 4), true, false);

    // Define the matmul 1 desc
    //     auto matmul_1_Desc = cudnn_frontend::MatMulDescBuilder()
    //                                  .setComputeType(CUDNN_DATA_FLOAT)
    //                                  .build();
    //std::cout << matmul_1_Desc.describe() << std::endl;

    auto matmul_1_Desc
            = compat_0_x::MatMulDescBuilder().setTransposeB(true).build();

    // Create a matmul 1 Node
    auto matmul_op1 = compat_0_x::OperationBuilder(compat_0_x::op::kind::MatMul)
                              .setaMatDesc(qTensor)
                              .setbMatDesc(kTransposeTensor)
                              .setcMatDesc(sTensor)
                              .setmatmulDesc(std::move(matmul_1_Desc))
                              .build();

    //std::cout << matmul_op1.describe() << std::endl;

    ops.push_back(std::move(matmul_op1));

    return sTensor;
}

void run_f16_flash_attention_fprop(int64_t b, int64_t h, int64_t s_q,
        int64_t s_kv, int64_t d, MHA_Layout layout, void *devPtrQ,
        void *devPtrK, void *devPtrO, compat_0_x::DataType_t tensorType) {
    // oneDNN handle. Unlike cuDNN, oneDNN needs to know the engine kind.
    auto handle = compat_0_x::Handle(dnnl::engine::kind::cpu, 0);

    //std::vector<compat_0_x::Operation const *> all_ops;
    std::vector<compat_0_x::Operation *> all_ops;
    std::vector<compat_0_x::Operation> ops;
    std::set<std::pair<uint64_t, void *>> data_ptrs;

    // Q * K^T
    auto sTensor = createQKBMM(b, h, s_q, s_kv, d, layout, tensorType, ops);

    std::cout << "Total ops created: " << ops.size() << std::endl;

    for (unsigned int i = 0; i < ops.size(); i++) {
        all_ops.push_back(&ops[i]);
    }

    // Create an Operation Graph
    auto opGraph = compat_0_x::OperationGraphBuilder()
                           .setHandle(handle)
                           .setOperationGraph(all_ops.size(), all_ops.data())
                           .build();

    compat_0_x::EngineConfigList filtered_configs;
    auto statuses = compat_0_x::get_heuristics_list({"heuristics_instant"},
            opGraph, ::allowAllConfig, filtered_configs, true);

    auto plan = compat_0_x::ExecutionPlanBuilder()
                        .setHandle(handle)
                        .setEngineConfig(std::move(filtered_configs[0]))
                        .build();

    data_ptrs.insert(std::pair<uint64_t, void *>(Q_ID, devPtrQ));
    data_ptrs.insert(std::pair<uint64_t, void *>(K_ID, devPtrK));
    data_ptrs.insert(std::pair<uint64_t, void *>(O_ID, devPtrO));

    compat_0_x::onednnGraphExecute(handle, plan, data_ptrs);
    handle.synchronize();
}

int main(void) {

    // oneDNN handle. Unlike cuDNN, oneDNN needs to know the engine kind.
    auto handle = compat_0_x::Handle(dnnl::engine::kind::cpu, 0);

    int64_t b = 2; // batch size
    int64_t h = 12; // head dim
    int64_t s_q = 2048; // q tensor is padded to this seq length
    int64_t s_kv = 2048; // k and v tensor is padded to this seq length
    int64_t d = 128; // hidden dim

    MHA_Layout layout = MHA_Layout::NOT_INTERLEAVED;

    std::cout << "====PARAMETERS====" << std::endl;
    std::cout << "batch is " << b << ", head dim is " << h
              << ", q sequence length is " << s_q << ", kv sequence length is "
              << s_kv << ", hidden dim is " << d << std::endl;

    void *devPtrQ = nullptr; // queries
    void *devPtrK = nullptr; // keys
    void *devPtrO = nullptr; // final output

    // prepare input/output memory
    const auto bf16 = dnnl::graph::logical_tensor::data_type::bf16;
    compat_0_x::Surface q_tensor(bf16, b * h * s_q * d, &handle);
    compat_0_x::Surface k_tensor(bf16, b * h * d * s_kv, &handle);
    compat_0_x::Surface o_tensor(bf16, b * s_q * h * d, &handle);

    devPtrQ = q_tensor.get_ptr();
    devPtrK = k_tensor.get_ptr();
    devPtrO = o_tensor.get_ptr();

    run_f16_flash_attention_fprop(
            b, h, s_q, s_kv, d, layout, devPtrQ, devPtrK, devPtrO, bf16);

    //     checkCudaErr(cudaDeviceSynchronize());
    //     checkCudaErr(cudaMemcpy(oTensor.hostPtr, oTensor.devPtr,
    //             sizeof(oTensor.hostPtr[0]) * oSize, cudaMemcpyDeviceToHost));
    //     checkCudaErr(cudaDeviceSynchronize());

    std::cout << "\n======================================================="
                 "=================================\n";
}
