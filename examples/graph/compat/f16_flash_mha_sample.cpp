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

#include "f16_flash_mha_sample.h"
#include "./utils/error_util.h"
#include <cudnn_frontend.h>

#define Q_ID 1
#define K_ID 2
#define V_ID 3
#define O_ID 4
#define S_ID 5
#define B_ID 6
#define D_CONST_ID 7
#define S_CONST_ID 8
#define Q_SEQLEN_ID 9
#define K_SEQLEN_ID 10
#define dQ_ID 11
#define dK_ID 12
#define dV_ID 13
#define dO_ID 14
#define MASK_VAL_ID 15
#define dS_ID 16
#define D_SEED_ID 17
#define D_OFFSET_ID 18
#define S_STATS_ID 19
#define S_SUM_ID 20
#define SCALE_PROB 21
#define K_TRANSPOSE_ID 22
#define dQ_ACCUM_ID 23

#define VIRTUAL_ID 30

// Used for MHA
void generateMHAStrides(int64_t b, int64_t h, int64_t s_q, int64_t s_kv,
        int64_t d, int64_t *strideA, MHA_Layout layout, MHA_Matrix matrix) {
    CUDNN_FRONTEND_UNUSED(b);
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

static bool allowAllConfig(cudnnBackendDescriptor_t engine_config) {
    (void)engine_config;
    return false;
}

#if (CUDNN_VERSION >= 8900)
static compat_0_x::Tensor tensor_create(compat_0_x::DataType_t type, int64_t id,
        int64_t const *dim, int64_t const *stride, bool is_virtual,
        bool is_value) {
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

static cudnn_frontend::Tensor tensor_create_cudnn(cudnnDataType_t type,
        int64_t id, int64_t const *dim, int64_t const *stride, bool is_virtual,
        bool is_value) {
    int nbDims = 4;
    auto tensor_created
            = cudnn_frontend::TensorBuilder()
                      .setDim(nbDims, dim)
                      .setStride(nbDims, stride)
                      .setId(id)
                      .setAlignment(
                              16) // 16B alignment is needed to run a tensor core engine
                      .setDataType(type)
                      .setVirtual(is_virtual)
                      .setByValue(is_value)
                      .build();
    std::cout << tensor_created.describe() << std::endl;
    return tensor_created;
}

static cudnn_frontend::PointWiseDesc pw_desc_create(
        cudnnDataType_t type, cudnnPointwiseMode_t mode) {
    auto pw_desc_created = cudnn_frontend::PointWiseDescBuilder()
                                   .setMode(mode)
                                   .setComputeType(type)
                                   .build();

    std::cout << pw_desc_created.describe() << std::endl;
    return pw_desc_created;
}

static cudnn_frontend::Operation unary_pw_op_create(
        cudnn_frontend::Tensor const &xDesc,
        cudnn_frontend::Tensor const &yDesc,
        cudnn_frontend::PointWiseDesc const &pwDesc) {
    auto pw_op_created = cudnn_frontend::OperationBuilder(
            CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                                 .setxDesc(xDesc)
                                 .setyDesc(yDesc)
                                 .setpwDesc(pwDesc)
                                 .build();
    std::cout << pw_op_created.describe() << std::endl;
    return pw_op_created;
}

static cudnn_frontend::Operation binary_pw_op_create(
        cudnn_frontend::Tensor const &xDesc,
        cudnn_frontend::Tensor const &bDesc,
        cudnn_frontend::Tensor const &yDesc,
        cudnn_frontend::PointWiseDesc const &pwDesc) {
    auto pw_op_created = cudnn_frontend::OperationBuilder(
            CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                                 .setxDesc(xDesc)
                                 .setbDesc(bDesc)
                                 .setyDesc(yDesc)
                                 .setpwDesc(pwDesc)
                                 .build();
    std::cout << pw_op_created.describe() << std::endl;
    return pw_op_created;
}

static cudnn_frontend::Operation ternary_pw_op_create(
        cudnn_frontend::Tensor const &xDesc,
        cudnn_frontend::Tensor const &bDesc,
        cudnn_frontend::Tensor const &tDesc,
        cudnn_frontend::Tensor const &yDesc,
        cudnn_frontend::PointWiseDesc const &pwDesc) {
    auto pw_op_created = cudnn_frontend::OperationBuilder(
            CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                                 .setxDesc(xDesc)
                                 .setbDesc(bDesc)
                                 .settDesc(tDesc)
                                 .setyDesc(yDesc)
                                 .setpwDesc(pwDesc)
                                 .build();
    std::cout << pw_op_created.describe() << std::endl;
    return pw_op_created;
}

static cudnn_frontend::Tensor createScale(int64_t b, int64_t h, int64_t s_q,
        int64_t s_kv, int64_t d, MHA_Layout layout, cudnnDataType_t tensorType,
        const cudnn_frontend::Tensor &sTensor,
        std::vector<cudnn_frontend::Operation> &ops) {
    // scale
    int64_t scale_dim[4] = {1, 1, 1, 1};
    int64_t scale_stride[4] = {1, 1, 1, 1};

    int64_t s_dim[4] = {b, h, s_q, s_kv};
    int64_t s_stride[4];
    generateMHAStrides(
            b, h, s_q, s_kv, d, s_stride, layout, MHA_Matrix::S_Matrix);

    auto scaleTensor = tensor_create(tensorType, S_CONST_ID, scale_dim,
            scale_stride, false, true); // is by value
    auto sScaleTensor = tensor_create(tensorType, VIRTUAL_ID + 2000, s_dim,
            s_stride, true, false); // is virtual

    // Define the scale descriptor
    auto scaleDesc = pw_desc_create(CUDNN_DATA_FLOAT, CUDNN_POINTWISE_MUL);

    // Create a Scale Node.
    auto scale_op = binary_pw_op_create(
            sTensor, scaleTensor, sScaleTensor, scaleDesc);

    ops.push_back(std::move(scale_op));
    return sScaleTensor;
}

static cudnn_frontend::Tensor createQKBMM(int64_t b, int64_t h, int64_t s_q,
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

    auto qTensor
            = tensor_create(tensorType, Q_ID, q_dim, q_stride, false, false);
    auto kTransposeTensor = tensor_create(
            tensorType, K_ID, k_dim, k_stride, false, false); // is virtual
    // first GEMM output
    auto sTensor = tensor_create(CUDNN_DATA_FLOAT, VIRTUAL_ID + 1, s_dim,
            s_stride, true, false); // is virtual

    // Define the matmul 1 desc
    auto matmul_1_Desc = cudnn_frontend::MatMulDescBuilder()
                                 .setComputeType(CUDNN_DATA_FLOAT)
                                 .build();
    //std::cout << matmul_1_Desc.describe() << std::endl;

    // Create a matmul 1 Node
    auto matmul_op1 = cudnn_frontend::OperationBuilder(
            CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR)
                              .setaMatDesc(qTensor)
                              .setbMatDesc(kTransposeTensor)
                              .setcMatDesc(sTensor)
                              .setmatmulDesc(matmul_1_Desc)
                              .build();

    //std::cout << matmul_op1.describe() << std::endl;

    ops.push_back(std::move(matmul_op1));

    return sTensor;
}

static cudnn_frontend::Tensor createQKBMMcuDNN(int64_t b, int64_t h,
        int64_t s_q, int64_t s_kv, int64_t d, MHA_Layout layout,
        cudnnDataType_t tensorType,
        std::vector<cudnn_frontend::Operation> &ops) {
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

    auto qTensor
            = tensor_create(tensorType, Q_ID, q_dim, q_stride, false, false);
    auto kTransposeTensor = tensor_create(
            tensorType, K_ID, k_dim, k_stride, false, false); // is virtual
    // first GEMM output
    auto sTensor = tensor_create(CUDNN_DATA_FLOAT, VIRTUAL_ID + 1, s_dim,
            s_stride, true, false); // is virtual

    // Define the matmul 1 desc
    auto matmul_1_Desc = cudnn_frontend::MatMulDescBuilder()
                                 .setComputeType(CUDNN_DATA_FLOAT)
                                 .build();
    std::cout << matmul_1_Desc.describe() << std::endl;

    // Create a matmul 1 Node
    auto matmul_op1 = cudnn_frontend::OperationBuilder(
            CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR)
                              .setaMatDesc(qTensor)
                              .setbMatDesc(kTransposeTensor)
                              .setcMatDesc(sTensor)
                              .setmatmulDesc(matmul_1_Desc)
                              .build();

    std::cout << matmul_op1.describe() << std::endl;

    ops.push_back(std::move(matmul_op1));

    return sTensor;
}

static cudnn_frontend::Tensor createCausalMask(int64_t b, int64_t h,
        int64_t s_q, int64_t s_kv, int64_t d, MHA_Layout layout,
        cudnnDataType_t tensorType, std::vector<cudnn_frontend::Operation> &ops,
        cudnn_frontend::Tensor &prevBlockOutputTensor) {
    CUDNN_FRONTEND_UNUSED(d);
    CUDNN_FRONTEND_UNUSED(layout);
    CUDNN_FRONTEND_UNUSED(tensorType);

    cudnn_frontend::throw_if(ops.size() == 0,
            "Padding Mask constructed incorrectly as the first one",
            CUDNN_STATUS_BAD_PARAM);

    // subtraction output
    int64_t afterBMM1_dim[4] = {b, h, s_q, s_kv};
    int64_t afterBMM1_stride[4] = {h * s_q * s_kv, s_q * s_kv, s_kv, 1};

    int64_t maskVal_dim[4] = {1, 1, 1, 1};
    int64_t maskVal_stride[4] = {1, 1, 1, 1};

    // mask value to put in the masked pixels
    auto maskValTensor = tensor_create(CUDNN_DATA_FLOAT, MASK_VAL_ID,
            maskVal_dim, maskVal_stride, false, true); // is by value
    // gen index row output
    auto rowIndexTensor = tensor_create(CUDNN_DATA_FLOAT, VIRTUAL_ID + 100,
            afterBMM1_dim, afterBMM1_stride, true, false); // is virtual
    // gen index column output
    auto columnIndexTensor = tensor_create(CUDNN_DATA_FLOAT, VIRTUAL_ID + 101,
            afterBMM1_dim, afterBMM1_stride, true, false); // is virtual
    // create causal mask (row >= col)
    auto causalMaskTensor = tensor_create(CUDNN_DATA_BOOLEAN, VIRTUAL_ID + 106,
            afterBMM1_dim, afterBMM1_stride, true, false); // is virtual

    // output after masking
    auto maskOutputTensor = tensor_create(CUDNN_DATA_FLOAT, VIRTUAL_ID + 107,
            afterBMM1_dim, afterBMM1_stride, true, false); // is virtual

    // Define the gen index for row descriptor
    auto genIndexRowDesc = cudnn_frontend::PointWiseDescBuilder()
                                   .setMode(CUDNN_POINTWISE_GEN_INDEX)
                                   .setAxis(2)
                                   .setComputeType(CUDNN_DATA_FLOAT)
                                   .build();
    std::cout << genIndexRowDesc.describe() << std::endl;

    // Create a gen index Node.
    auto genIndexRow_op = unary_pw_op_create(
            prevBlockOutputTensor, rowIndexTensor, genIndexRowDesc);
    std::cout << genIndexRow_op.describe() << std::endl;

    // Define the gen index for row descriptor
    auto genIndexColumnDesc = cudnn_frontend::PointWiseDescBuilder()
                                      .setMode(CUDNN_POINTWISE_GEN_INDEX)
                                      .setAxis(3)
                                      .setComputeType(CUDNN_DATA_FLOAT)
                                      .build();
    std::cout << genIndexColumnDesc.describe() << std::endl;

    // Create a gen index Node.
    auto genIndexColumn_op = unary_pw_op_create(
            prevBlockOutputTensor, columnIndexTensor, genIndexColumnDesc);

    // Define the greater than equal to comparison descriptor
    auto rowGreaterColDesc
            = pw_desc_create(CUDNN_DATA_BOOLEAN, CUDNN_POINTWISE_CMP_GE);

    // Create a greater than equal to Node.
    auto rowGreaterCol_op = binary_pw_op_create(rowIndexTensor,
            columnIndexTensor, causalMaskTensor, rowGreaterColDesc);

    /////////////////// Apply the mask //////////////////////////

    // Define the binary select to perform masking descriptor
    auto maskDesc
            = pw_desc_create(CUDNN_DATA_FLOAT, CUDNN_POINTWISE_BINARY_SELECT);

    // Create a binary select Node.
    auto mask_op = ternary_pw_op_create(prevBlockOutputTensor, maskValTensor,
            causalMaskTensor, maskOutputTensor, maskDesc);

    ops.push_back(std::move(genIndexRow_op));
    ops.push_back(std::move(genIndexColumn_op));
    ops.push_back(std::move(rowGreaterCol_op));
    ops.push_back(std::move(mask_op));

    return maskOutputTensor;
}

static cudnn_frontend::Tensor createSoftmaxForward(int64_t b, int64_t h,
        int64_t s_q, int64_t s_kv, bool isTraining,
        std::vector<cudnn_frontend::Operation> &ops,
        cudnn_frontend::Tensor &sAfterMaskTensor) {
    int64_t afterBMM1_dim[4] = {b, h, s_q, s_kv};
    int64_t afterBMM1_stride[4] = {h * s_q * s_kv, s_q * s_kv, s_kv, 1};

    int64_t afterReduction_dim[4] = {b, h, s_q, 1};
    int64_t afterReduction_stride[4] = {h * s_q, s_q, 1, 1};

    // max (x)
    auto afterMaxReductionTensor = tensor_create(CUDNN_DATA_FLOAT,
            VIRTUAL_ID + 150, afterReduction_dim, afterReduction_stride, true,
            false); // is virtual

    // x - max(x)
    auto afterSubtractionTensor
            = tensor_create(CUDNN_DATA_FLOAT, VIRTUAL_ID + 151, afterBMM1_dim,
                    afterBMM1_stride, true, false); // is virtual

    // e^(x - max(x))
    auto afterExponentTensor = tensor_create(CUDNN_DATA_FLOAT, VIRTUAL_ID + 152,
            afterBMM1_dim, afterBMM1_stride, true, false); // is virtual;

    // sum (e^(x - max(x)))
    auto afterAddReductionTensor = tensor_create(CUDNN_DATA_FLOAT,
            VIRTUAL_ID + 153, afterReduction_dim, afterReduction_stride, true,
            false); // is virtual

    // log (sum (e^(x - max(x))))
    auto afterLogLTensor = tensor_create(CUDNN_DATA_FLOAT, VIRTUAL_ID + 154,
            afterReduction_dim, afterReduction_stride, true, false);

    // M + log (sum (e^(x - max(x))))
    auto softmaxStatsTensor = tensor_create(CUDNN_DATA_FLOAT, S_STATS_ID,
            afterReduction_dim, afterReduction_stride, !isTraining,
            false); // not virtual if training is true, virtual if training is false

    // divide (e/ sum(e))
    auto afterSoftmaxTensor
            = cudnn_frontend::TensorBuilder()
                      .setDim(4, afterBMM1_dim)
                      .setStride(4, afterBMM1_stride)
                      .setId(VIRTUAL_ID + 156)
                      .setAlignment(
                              16) // 16B alignment is needed to run a tensor core engine
                      .setDataType(CUDNN_DATA_FLOAT)
                      .setVirtual(true)
                      .setByValue(false)
                      .setReorderType(
                              cudnn_frontend::TensorReordering_t::F16x16)
                      .build();

    // Define the reduction descriptor
    auto reductionMaxDesc = cudnn_frontend::ReductionDescBuilder()
                                    .setComputeType(CUDNN_DATA_FLOAT)
                                    .setReductionOp(CUDNN_REDUCE_TENSOR_MAX)
                                    .build();
    std::cout << reductionMaxDesc.describe() << std::endl;

    // Create a reduction max Node.
    auto reductionMax_op = cudnn_frontend::OperationBuilder(
            CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR)
                                   .setxDesc(sAfterMaskTensor)
                                   .setyDesc(afterMaxReductionTensor)
                                   .setreductionDesc(reductionMaxDesc)
                                   .build();
    std::cout << reductionMax_op.describe() << std::endl;

    // Define the subtract descriptor
    auto subtractDesc = pw_desc_create(CUDNN_DATA_FLOAT, CUDNN_POINTWISE_SUB);

    // Create a subtract Node.
    auto subtract_op = binary_pw_op_create(sAfterMaskTensor,
            afterMaxReductionTensor, afterSubtractionTensor, subtractDesc);

    // Define the exponent descriptor
    auto exponentDesc = pw_desc_create(CUDNN_DATA_FLOAT, CUDNN_POINTWISE_EXP);

    // Create a exponent Node.
    auto exponent_op = unary_pw_op_create(
            afterSubtractionTensor, afterExponentTensor, exponentDesc);

    // Define the reduction descriptor
    auto reductionAddDesc = cudnn_frontend::ReductionDescBuilder()
                                    .setComputeType(CUDNN_DATA_FLOAT)
                                    .setReductionOp(CUDNN_REDUCE_TENSOR_ADD)
                                    .build();
    std::cout << reductionAddDesc.describe() << std::endl;

    // Create a reduction add Node.
    auto reductionAdd_op = cudnn_frontend::OperationBuilder(
            CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR)
                                   .setxDesc(afterExponentTensor)
                                   .setyDesc(afterAddReductionTensor)
                                   .setreductionDesc(reductionAddDesc)
                                   .build();

    std::cout << reductionAdd_op.describe() << std::endl;

    // Create log descriptor
    auto logDesc = pw_desc_create(CUDNN_DATA_FLOAT, CUDNN_POINTWISE_LOG);

    // Create log node
    auto log_op = unary_pw_op_create(
            afterAddReductionTensor, afterLogLTensor, logDesc);

    // Create add descriptor
    auto addDesc = pw_desc_create(CUDNN_DATA_FLOAT, CUDNN_POINTWISE_ADD);

    // Create add node
    auto add_op = binary_pw_op_create(afterMaxReductionTensor, afterLogLTensor,
            softmaxStatsTensor, addDesc);

    // Define the division descriptor
    auto divisionDesc = pw_desc_create(CUDNN_DATA_FLOAT, CUDNN_POINTWISE_DIV);

    // Create a subtract Node.
    auto division_op = binary_pw_op_create(afterExponentTensor,
            afterAddReductionTensor, afterSoftmaxTensor, divisionDesc);

    ops.push_back(std::move(reductionMax_op));
    ops.push_back(std::move(subtract_op));
    ops.push_back(std::move(exponent_op));
    ops.push_back(std::move(reductionAdd_op));
    ops.push_back(std::move(log_op));
    ops.push_back(std::move(add_op));
    ops.push_back(std::move(division_op));

    return afterSoftmaxTensor;
}

static cudnn_frontend::Tensor createDropoutForward(int64_t b, int64_t h,
        int64_t s_q, int64_t s_kv, int64_t d, double probability,
        cudnnDataType_t tensorType, std::vector<cudnn_frontend::Operation> &ops,
        cudnn_frontend::Tensor &afterSoftmaxTensor) {
    CUDNN_FRONTEND_UNUSED(d);

    cudnn_frontend::throw_if(ops.size() == 0,
            "Dropout DAG constructed incorrectly as the first one",
            CUDNN_STATUS_BAD_PARAM);

    int64_t afterBMM1_dim[4] = {b, h, s_q, s_kv};
    int64_t afterBMM1_stride[4] = {h * s_q * s_kv, s_q * s_kv, s_kv, 1};

    int64_t scale_dim[4] = {1, 1, 1, 1};
    int64_t scale_stride[4] = {1, 1, 1, 1};

    auto dropoutSeed = tensor_create(CUDNN_DATA_INT64, D_SEED_ID, scale_dim,
            scale_stride, false, false); // not virtual
    auto dropoutOffset = tensor_create(CUDNN_DATA_INT64, D_OFFSET_ID, scale_dim,
            scale_stride, false, false); // not virtual

    // mask for the dropout
    auto dropoutMaskTensor = tensor_create(CUDNN_DATA_FLOAT, VIRTUAL_ID + 200,
            afterBMM1_dim, afterBMM1_stride, true, false); // is virtual
    // after dropout tensor
    auto afterDropoutTensor
            = cudnn_frontend::TensorBuilder()
                      .setDim(4, afterBMM1_dim)
                      .setStride(4, afterBMM1_stride)
                      .setId(VIRTUAL_ID + 201)
                      .setAlignment(
                              16) // 16B alignment is needed to run a tensor core engine
                      .setDataType(tensorType)
                      .setVirtual(true)
                      .setByValue(false)
                      .setReorderType(
                              cudnn_frontend::TensorReordering_t::F16x16)
                      .build();
    // scale after dropout
    auto scaleDropoutTensor = tensor_create(tensorType, D_CONST_ID, scale_dim,
            scale_stride, false, true); // is by value
    // after Scale
    auto afterScaleTensor = tensor_create(tensorType, VIRTUAL_ID + 202,
            afterBMM1_dim, afterBMM1_stride, true, false); // is virtual

    // Define the reduction descriptor
    auto rngDesc = cudnn_frontend::RngDescBuilder()
                           .setRngDistribution(CUDNN_RNG_DISTRIBUTION_BERNOULLI)
                           .setBernoulliDistProbability(1.0 - probability)
                           .build();
    std::cout << rngDesc.describe() << std::endl;

    // Create a rng Node.
    auto rng_op = cudnn_frontend::OperationBuilder(
            CUDNN_BACKEND_OPERATION_RNG_DESCRIPTOR)
                          .setyDesc(dropoutMaskTensor)
                          .setSeedDesc(dropoutSeed)
                          .setOffsetDesc(dropoutOffset)
                          .setRngDesc(rngDesc)
                          .build();

    std::cout << rng_op.describe() << std::endl;

    // Define the multiply mask descriptor
    auto maskMulDesc = pw_desc_create(CUDNN_DATA_FLOAT, CUDNN_POINTWISE_MUL);

    // Create a multiply mask Node.
    auto maskMul_op = binary_pw_op_create(afterSoftmaxTensor, dropoutMaskTensor,
            afterDropoutTensor, maskMulDesc);

    // Define the multiply scale descriptor
    auto scaleMulDesc = pw_desc_create(CUDNN_DATA_FLOAT, CUDNN_POINTWISE_MUL);

    // Create a multiply scale Node.
    auto scaleMul_op = binary_pw_op_create(afterDropoutTensor,
            scaleDropoutTensor, afterScaleTensor, scaleMulDesc);

    ops.push_back(std::move(rng_op));
    ops.push_back(std::move(maskMul_op));
    ops.push_back(std::move(scaleMul_op));

    return afterScaleTensor;
}

static cudnn_frontend::Tensor createDropoutBackward(int64_t b, int64_t h,
        int64_t s_q, int64_t s_kv, int64_t d, double probability,
        cudnnDataType_t tensorType, std::vector<cudnn_frontend::Operation> &ops,
        cudnn_frontend::Tensor &afterSoftmaxTensor,
        cudnn_frontend::Tensor &dropoutMaskTensor) {
    CUDNN_FRONTEND_UNUSED(d);

    cudnn_frontend::throw_if(ops.size() == 0,
            "Dropout DAG constructed incorrectly as the first one",
            CUDNN_STATUS_BAD_PARAM);

    int64_t afterBMM1_dim[4] = {b, h, s_q, s_kv};
    int64_t afterBMM1_stride[4] = {h * s_q * s_kv, s_q * s_kv, s_kv, 1};

    int64_t scale_dim[4] = {1, 1, 1, 1};
    int64_t scale_stride[4] = {1, 1, 1, 1};

    auto dropoutSeed = tensor_create(CUDNN_DATA_INT64, D_SEED_ID, scale_dim,
            scale_stride, false, false); // not virtual
    auto dropoutOffset = tensor_create(CUDNN_DATA_INT64, D_OFFSET_ID, scale_dim,
            scale_stride, false, false); // not virtual

    // after dropout tensor
    auto afterDropoutTensor
            = cudnn_frontend::TensorBuilder()
                      .setDim(4, afterBMM1_dim)
                      .setStride(4, afterBMM1_stride)
                      .setId(VIRTUAL_ID + 201)
                      .setAlignment(
                              16) // 16B alignment is needed to run a tensor core engine
                      .setDataType(tensorType)
                      .setVirtual(true)
                      .setByValue(false)
                      .setReorderType(
                              cudnn_frontend::TensorReordering_t::F16x16)
                      .build();
    // scale after dropout
    auto scaleDropoutTensor = tensor_create(tensorType, D_CONST_ID, scale_dim,
            scale_stride, false, true); // is by value
    // after Scale
    auto afterScaleTensor = tensor_create(tensorType, VIRTUAL_ID + 202,
            afterBMM1_dim, afterBMM1_stride, true, false); // is virtual

    // Define the reduction descriptor
    auto rngDesc = cudnn_frontend::RngDescBuilder()
                           .setRngDistribution(CUDNN_RNG_DISTRIBUTION_BERNOULLI)
                           .setBernoulliDistProbability(1.0 - probability)
                           .build();
    std::cout << rngDesc.describe() << std::endl;

    // Create a rng Node.
    auto rng_op = cudnn_frontend::OperationBuilder(
            CUDNN_BACKEND_OPERATION_RNG_DESCRIPTOR)
                          .setyDesc(dropoutMaskTensor)
                          .setSeedDesc(dropoutSeed)
                          .setOffsetDesc(dropoutOffset)
                          .setRngDesc(rngDesc)
                          .build();

    std::cout << rng_op.describe() << std::endl;

    // Define the multiply mask descriptor
    auto maskMulDesc = pw_desc_create(CUDNN_DATA_FLOAT, CUDNN_POINTWISE_MUL);

    // Create a multiply mask Node.
    auto maskMul_op = binary_pw_op_create(afterSoftmaxTensor, dropoutMaskTensor,
            afterDropoutTensor, maskMulDesc);

    // Define the multiply scale descriptor
    auto scaleMulDesc = pw_desc_create(CUDNN_DATA_FLOAT, CUDNN_POINTWISE_MUL);

    // Create a multiply scale Node.
    auto scaleMul_op = binary_pw_op_create(afterDropoutTensor,
            scaleDropoutTensor, afterScaleTensor, scaleMulDesc);

    ops.push_back(std::move(rng_op));
    ops.push_back(std::move(maskMul_op));
    ops.push_back(std::move(scaleMul_op));

    return afterScaleTensor;
}

static void createSVBMM(int64_t b, int64_t h, int64_t s_q, int64_t s_kv,
        int64_t d, MHA_Layout layout, cudnnDataType_t tensorType,
        std::vector<cudnn_frontend::Operation> &ops,
        cudnn_frontend::Tensor const &afterScaleDropoutTensor) {
    cudnn_frontend::throw_if(ops.size() == 0,
            "BMM2 op constructed incorrectly as the first one",
            CUDNN_STATUS_BAD_PARAM);

    int64_t v_dim[4] = {b, h, s_kv, d};
    int64_t v_stride[4];
    generateMHAStrides(
            b, h, s_q, s_kv, d, v_stride, layout, MHA_Matrix::V_Matrix);

    int64_t o_dim[4] = {b, h, s_q, d};
    int64_t o_stride[4];
    generateMHAStrides(
            b, h, s_q, s_kv, d, o_stride, layout, MHA_Matrix::O_Matrix);

    auto vTensor
            = tensor_create(tensorType, V_ID, v_dim, v_stride, false, false);
    // second GEMM output
    auto oTensor
            = tensor_create(tensorType, O_ID, o_dim, o_stride, false, false);

    // Define the matmul 2 desc
    auto matmul_2_Desc = cudnn_frontend::MatMulDescBuilder()
                                 .setComputeType(CUDNN_DATA_FLOAT)
                                 .build();
    std::cout << matmul_2_Desc.describe() << std::endl;

    // Create a matmul 2 Node
    auto matmul_op2 = cudnn_frontend::OperationBuilder(
            CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR)
                              .setaMatDesc(afterScaleDropoutTensor)
                              .setbMatDesc(vTensor)
                              .setcMatDesc(oTensor)
                              .setmatmulDesc(matmul_2_Desc)
                              .build();

    std::cout << matmul_op2.describe() << std::endl;

    ops.push_back(std::move(matmul_op2));
}

static cudnn_frontend::Tensor createSoftmaxBackward(int64_t b, int64_t h,
        int64_t s_q, int64_t s_kv, int64_t d, MHA_Layout layout,
        cudnnDataType_t tensorType, std::vector<cudnn_frontend::Operation> &ops,
        cudnn_frontend::Tensor &yTensor, cudnn_frontend::Tensor &dyTensor) {
    CUDNN_FRONTEND_UNUSED(tensorType);

    cudnn_frontend::throw_if(ops.size() == 0,
            "Softmax backward constructed incorrectly as the first one",
            CUDNN_STATUS_BAD_PARAM);

    int64_t p_dim[4] = {b, h, s_q, s_kv};
    int64_t p_stride[4];
    generateMHAStrides(
            b, h, s_q, s_kv, d, p_stride, layout, MHA_Matrix::S_Matrix);

    int64_t p_reduction_dim[4] = {b, h, s_q, 1};
    int64_t p_reduction_stride[4];

    p_reduction_stride[3] = 1;
    p_reduction_stride[2] = 1;
    p_reduction_stride[1] = s_q;
    p_reduction_stride[0] = s_q * h;

    int64_t const_dim[4] = {1, 1, 1, 1};
    int64_t const_stride[4] = {1, 1, 1, 1};

    // creating all tensors
    auto softmaxScaleTensor = tensor_create(
            CUDNN_DATA_FLOAT, S_CONST_ID, const_dim, const_stride, false, true);
    auto dyMulYTensor = tensor_create(
            CUDNN_DATA_FLOAT, VIRTUAL_ID + 250, p_dim, p_stride, true, false);
    auto dxAfterReductionTensor = tensor_create(CUDNN_DATA_FLOAT,
            VIRTUAL_ID + 251, p_reduction_dim, p_reduction_stride, true, false);
    auto dxAfterSubtractionTensor = tensor_create(
            CUDNN_DATA_FLOAT, VIRTUAL_ID + 252, p_dim, p_stride, true, false);
    auto dxUnscaleTensor = tensor_create(
            CUDNN_DATA_FLOAT, VIRTUAL_ID + 253, p_dim, p_stride, true, false);
    auto dxTensor = tensor_create(
            CUDNN_DATA_FLOAT, VIRTUAL_ID + 254, p_dim, p_stride, true, false);

    // creating all ops
    // mul (y * dy)
    auto mul_1_desc = pw_desc_create(CUDNN_DATA_FLOAT, CUDNN_POINTWISE_MUL);
    auto mul_1_op
            = binary_pw_op_create(yTensor, dyTensor, dyMulYTensor, mul_1_desc);

    // reduction add sum (y * dy)
    auto reductionAddDesc = cudnn_frontend::ReductionDescBuilder()
                                    .setComputeType(CUDNN_DATA_FLOAT)
                                    .setReductionOp(CUDNN_REDUCE_TENSOR_ADD)
                                    .build();
    std::cout << reductionAddDesc.describe() << std::endl;

    auto reductionAdd_op = cudnn_frontend::OperationBuilder(
            CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR)
                                   .setxDesc(dyMulYTensor)
                                   .setyDesc(dxAfterReductionTensor)
                                   .setreductionDesc(reductionAddDesc)
                                   .build();

    std::cout << reductionAdd_op.describe() << std::endl;

    // subtraction (dy - sum(y * dy))
    auto sub_0_desc = pw_desc_create(CUDNN_DATA_FLOAT, CUDNN_POINTWISE_SUB);
    auto sub_0_op = binary_pw_op_create(dyTensor, dxAfterReductionTensor,
            dxAfterSubtractionTensor, sub_0_desc);

    // mul (y * (dy - sum(y * dy)))
    auto mul_2_desc = pw_desc_create(CUDNN_DATA_FLOAT, CUDNN_POINTWISE_MUL);
    auto mul_2_op = binary_pw_op_create(
            yTensor, dxAfterSubtractionTensor, dxUnscaleTensor, mul_2_desc);

    // mul (scale * dx)
    auto mul_3_desc = pw_desc_create(CUDNN_DATA_FLOAT, CUDNN_POINTWISE_MUL);
    auto mul_3_op = binary_pw_op_create(
            dxUnscaleTensor, softmaxScaleTensor, dxTensor, mul_3_desc);

    ops.push_back(std::move(mul_1_op));
    ops.push_back(std::move(reductionAdd_op));
    ops.push_back(std::move(sub_0_op));
    ops.push_back(std::move(mul_2_op));
    ops.push_back(std::move(mul_3_op));

    return dxTensor;
}

void run_f16_flash_attention_fprop_cudnn(int64_t b, int64_t h, int64_t s_q,
        int64_t s_kv, int64_t d, MHA_Layout layout, float scaling_factor,
        bool isTraining, double dropout_probability, void *devPtrQ,
        void *devPtrK, void *devPtrV, void *devPtrSoftmaxStats, void *devPtrO,
        void *devPtrDropoutSeed, void *devPtrDropoutOffset,
        cudnnDataType_t tensorType) {
    // Create a unique_ptr for the cuDNN handle
    auto handle_ptr = create_cudnn_handle();
    auto handle_ = *handle_ptr;

    std::vector<cudnn_frontend::Operation const *> all_ops;
    std::vector<cudnn_frontend::Operation> ops;
    std::set<std::pair<uint64_t, void *>> data_ptrs;

    // Q * K^T
    auto sTensor = createQKBMM(b, h, s_q, s_kv, d, layout, tensorType, ops);

    // Q * K^T * bmmScale
    auto sScaleTensor = createScale(
            b, h, s_q, s_kv, d, layout, CUDNN_DATA_FLOAT, sTensor, ops);

    // Causual mask
    float negInfinity
            = -1.0E+20f; // change this if you have access to float_min
    auto sAfterMaskTensor = createCausalMask(
            b, h, s_q, s_kv, d, layout, tensorType, ops, sScaleTensor);

    cudnn_frontend::throw_if(dropout_probability != 0.0f && !isTraining,
            "Dropout probability should be 0.0f for inference mode",
            CUDNN_STATUS_BAD_PARAM);
    cudnn_frontend::throw_if(dropout_probability == 1.0f,
            "Dropout probability cannot be 1.0", CUDNN_STATUS_BAD_PARAM);

    // needs to be bf16 (Please change)
    half1 scale_dropout = cpu_float2half_rn(
            static_cast<float>(1 / (1 - dropout_probability)));

    auto softmax_output = createSoftmaxForward(
            b, h, s_q, s_kv, isTraining, ops, sAfterMaskTensor);

    // Dropout(softmax)
    auto dropout_output = createDropoutForward(b, h, s_q, s_kv, d,
            dropout_probability, tensorType, ops, softmax_output);
    createSVBMM(b, h, s_q, s_kv, d, layout, tensorType, ops, dropout_output);

    std::cout << "Total ops created: " << ops.size() << std::endl;

    for (unsigned int i = 0; i < ops.size(); i++) {
        all_ops.push_back(&ops[i]);
    }

    // Create an Operation Graph
    auto opGraph = cudnn_frontend::OperationGraphBuilder()
                           .setHandle(handle_)
                           .setOperationGraph(all_ops.size(), all_ops.data())
                           .build();

    cudnn_frontend::EngineConfigList filtered_configs;
    auto statuses
            = cudnn_frontend::get_heuristics_list<1>({"heuristics_instant"},
                    opGraph, ::allowAllConfig, filtered_configs, true);

    if (filtered_configs.size() == 0) {
        cudnn_frontend::set_error_and_throw_exception(nullptr,
                CUDNN_STATUS_NOT_SUPPORTED,
                "run_mha_fprop: No config returned by the heuristics");
    }

    auto plan = cudnn_frontend::ExecutionPlanBuilder()
                        .setHandle(handle_)
                        .setEngineConfig(filtered_configs[0], opGraph.getTag())
                        .build();

    std::cout << "Plan tag: " << plan.getTag() << std::endl;

    auto workspace_size = plan.getWorkspaceSize();
    std::cout << plan.describe() << " requires workspace " << workspace_size
              << std::endl;

    void *workspace_ptr = nullptr;
    if (workspace_size > 0) {
        checkCudaErr(cudaMalloc(&workspace_ptr, workspace_size));
    }

    // add all the data pointers to be used in the variant pack
    data_ptrs.insert(std::pair<uint64_t, void *>(Q_ID, devPtrQ));
    data_ptrs.insert(std::pair<uint64_t, void *>(K_ID, devPtrK));
    data_ptrs.insert(std::pair<uint64_t, void *>(V_ID, devPtrV));
    data_ptrs.insert(std::pair<uint64_t, void *>(MASK_VAL_ID, &negInfinity));
    data_ptrs.insert(std::pair<uint64_t, void *>(S_CONST_ID, &scaling_factor));
    data_ptrs.insert(std::pair<uint64_t, void *>(O_ID, devPtrO));
    data_ptrs.insert(std::pair<uint64_t, void *>(D_SEED_ID, devPtrDropoutSeed));
    data_ptrs.insert(
            std::pair<uint64_t, void *>(D_OFFSET_ID, devPtrDropoutOffset));
    data_ptrs.insert(std::pair<uint64_t, void *>(D_CONST_ID, &scale_dropout));

    // If training mode, we write out softmax stats
    if (isTraining) {
        data_ptrs.insert(
                std::pair<uint64_t, void *>(S_STATS_ID, devPtrSoftmaxStats));
    }

    auto variantPack = cudnn_frontend::VariantPackBuilder()
                               .setWorkspacePointer(workspace_ptr)
                               .setDataPointers(data_ptrs)
                               .build();
    std::cout << "variantPack " << variantPack.describe() << std::endl;
    cudnnStatus_t status = cudnnBackendExecute(
            handle_, plan.get_raw_desc(), variantPack.get_raw_desc());
    checkCudaErr(cudaDeviceSynchronize());
    if (workspace_size > 0) { checkCudaErr(cudaFree(workspace_ptr)); }

    cudnn_frontend::throw_if(
            [status]() { return (status != CUDNN_STATUS_SUCCESS); },
            "Plan execute error", status);
}

void run_f16_flash_attention_fprop(int64_t b, int64_t h, int64_t s_q,
        int64_t s_kv, int64_t d, MHA_Layout layout, float scaling_factor,
        bool isTraining, double dropout_probability, void *devPtrQ,
        void *devPtrK, void *devPtrV, void *devPtrSoftmaxStats, void *devPtrO,
        void *devPtrDropoutSeed, void *devPtrDropoutOffset,
        cudnnDataType_t tensorType) {
    // oneDNN handle. Unlike cuDNN, oneDNN needs to know the engine kind.
    auto handle = compat_0_x::Handle(dnnl::engine::kind::cpu, 0);

    std::vector<compat_0_x::Operation const *> all_ops;
    std::vector<compat_0_x::Operation> ops;
    std::set<std::pair<uint64_t, void *>> data_ptrs;

    // Q * K^T
    auto sTensor = createQKBMM(b, h, s_q, s_kv, d, layout, tensorType, ops);

    // Q * K^T * bmmScale
    auto sScaleTensor = createScale(
            b, h, s_q, s_kv, d, layout, CUDNN_DATA_FLOAT, sTensor, ops);

    // Causual mask
    float negInfinity
            = -1.0E+20f; // change this if you have access to float_min
    auto sAfterMaskTensor = createCausalMask(
            b, h, s_q, s_kv, d, layout, tensorType, ops, sScaleTensor);

    cudnn_frontend::throw_if(dropout_probability != 0.0f && !isTraining,
            "Dropout probability should be 0.0f for inference mode",
            CUDNN_STATUS_BAD_PARAM);
    cudnn_frontend::throw_if(dropout_probability == 1.0f,
            "Dropout probability cannot be 1.0", CUDNN_STATUS_BAD_PARAM);

    // needs to be bf16 (Please change)
    half1 scale_dropout = cpu_float2half_rn(
            static_cast<float>(1 / (1 - dropout_probability)));

    auto softmax_output = createSoftmaxForward(
            b, h, s_q, s_kv, isTraining, ops, sAfterMaskTensor);

    // Dropout(softmax)
    auto dropout_output = createDropoutForward(b, h, s_q, s_kv, d,
            dropout_probability, tensorType, ops, softmax_output);
    createSVBMM(b, h, s_q, s_kv, d, layout, tensorType, ops, dropout_output);

    std::cout << "Total ops created: " << ops.size() << std::endl;

    for (unsigned int i = 0; i < ops.size(); i++) {
        all_ops.push_back(&ops[i]);
    }

    // Create an Operation Graph
    auto opGraph = cudnn_frontend::OperationGraphBuilder()
                           .setHandle(handle_)
                           .setOperationGraph(all_ops.size(), all_ops.data())
                           .build();

    cudnn_frontend::EngineConfigList filtered_configs;
    auto statuses
            = cudnn_frontend::get_heuristics_list<1>({"heuristics_instant"},
                    opGraph, ::allowAllConfig, filtered_configs, true);

    if (filtered_configs.size() == 0) {
        cudnn_frontend::set_error_and_throw_exception(nullptr,
                CUDNN_STATUS_NOT_SUPPORTED,
                "run_mha_fprop: No config returned by the heuristics");
    }

    auto plan = cudnn_frontend::ExecutionPlanBuilder()
                        .setHandle(handle_)
                        .setEngineConfig(filtered_configs[0], opGraph.getTag())
                        .build();

    std::cout << "Plan tag: " << plan.getTag() << std::endl;

    auto workspace_size = plan.getWorkspaceSize();
    std::cout << plan.describe() << " requires workspace " << workspace_size
              << std::endl;

    void *workspace_ptr = nullptr;
    if (workspace_size > 0) {
        checkCudaErr(cudaMalloc(&workspace_ptr, workspace_size));
    }

    // add all the data pointers to be used in the variant pack
    data_ptrs.insert(std::pair<uint64_t, void *>(Q_ID, devPtrQ));
    data_ptrs.insert(std::pair<uint64_t, void *>(K_ID, devPtrK));
    data_ptrs.insert(std::pair<uint64_t, void *>(V_ID, devPtrV));
    data_ptrs.insert(std::pair<uint64_t, void *>(MASK_VAL_ID, &negInfinity));
    data_ptrs.insert(std::pair<uint64_t, void *>(S_CONST_ID, &scaling_factor));
    data_ptrs.insert(std::pair<uint64_t, void *>(O_ID, devPtrO));
    data_ptrs.insert(std::pair<uint64_t, void *>(D_SEED_ID, devPtrDropoutSeed));
    data_ptrs.insert(
            std::pair<uint64_t, void *>(D_OFFSET_ID, devPtrDropoutOffset));
    data_ptrs.insert(std::pair<uint64_t, void *>(D_CONST_ID, &scale_dropout));

    // If training mode, we write out softmax stats
    if (isTraining) {
        data_ptrs.insert(
                std::pair<uint64_t, void *>(S_STATS_ID, devPtrSoftmaxStats));
    }

    auto variantPack = cudnn_frontend::VariantPackBuilder()
                               .setWorkspacePointer(workspace_ptr)
                               .setDataPointers(data_ptrs)
                               .build();
    std::cout << "variantPack " << variantPack.describe() << std::endl;
    cudnnStatus_t status = cudnnBackendExecute(
            handle_, plan.get_raw_desc(), variantPack.get_raw_desc());
    checkCudaErr(cudaDeviceSynchronize());
    if (workspace_size > 0) { checkCudaErr(cudaFree(workspace_ptr)); }

    cudnn_frontend::throw_if(
            [status]() { return (status != CUDNN_STATUS_SUCCESS); },
            "Plan execute error", status);
}

main() {
#if (CUDNN_VERSION >= 8900)
    TEST_CASE("BF16 LLM Flash MHA Fprop sample",
            "[frontend][fusion][BF16LLMFprop]") {
        std::cout << "TEST_CASE :: BF16 LLM Flash MHA Fprop with backend API"
                  << std::endl;
        INFO("TEST_CASE ::  BF16 LLM Flash MHA Fprop with backend API");

#if (CUDART_VERSION < 12000)
        SKIP("Test requires CUDA version greater than 12.0");
#endif

        int64_t b = 2; // batch size
        int64_t h = 12; // head dim
        int64_t s_q = 2048; // q tensor is padded to this seq length
        int64_t s_kv = 2048; // k and v tensor is padded to this seq length
        int64_t d = 128; // hidden dim

        int64_t seed = 123456; // seed for generating the dropout mask

        MHA_Layout layout = MHA_Layout::
                SBH_INTERLEAVED; // layout of the tensors Q,K and V. BF16 LLM has layout [S,B,H,3,D]

        float scaling_factor = 0.5; // scale value before softmax

        bool isTraining = true; // training or inference mode
        double dropout_probability
                = 0.2f; // probability of dropout. Should be 0.0 for inference mode

        std::cout << "====PARAMETERS====" << std::endl;
        std::cout << "batch is " << b << ", head dim is " << h
                  << ", q sequence length is " << s_q
                  << ", kv sequence length is " << s_kv << ", hidden dim is "
                  << d << std::endl;

        void *devPtrQ = nullptr; // queries
        void *devPtrK = nullptr; // keys
        void *devPtrV = nullptr; // values
        void *devPtrSoftmaxStats = nullptr; // softmax stats
        void *devPtrO = nullptr; // final output

        // the setup is for the qkv interleaved layout (qkv interleaved assumes s_q = s_kv)
        int64_t qkvTensorDim[] = {s_q, b, h, 3, d};
        CUDNN_FRONTEND_UNUSED(qkvTensorDim);

        int64_t xSize = s_q * b * h * 3 * d;
        Surface<half> xTensor(xSize, false);
        devPtrQ = (void *)xTensor.devPtr; // q points to the top of qkv
        devPtrK = (void *)(xTensor.devPtr + d); // k is at an offset of d
        devPtrV = (void *)(xTensor.devPtr
                + 2 * d); // v is at an offset of 2 * d

        void *devPtrDropoutSeed = nullptr; // Seed for dropout
        void *devPtrDropoutOffset = nullptr; // Offset for dropout

        int64_t scaleSize = 1;
        Surface<int64_t> dropoutSeed(scaleSize, false, seed);
        devPtrDropoutSeed = (void *)dropoutSeed.devPtr;
        Surface<int64_t> dropoutOffset(scaleSize, false, (int64_t)1);
        devPtrDropoutOffset = (void *)dropoutOffset.devPtr;

        int64_t softmaxStatsSize = b * h * s_q;
        Surface<float> softmaxStats(softmaxStatsSize, false);
        if (isTraining) { devPtrSoftmaxStats = (void *)softmaxStats.devPtr; }

        int64_t oSize = b * s_q * h * d;
        Surface<half> oTensor(oSize, false);
        devPtrO = (void *)oTensor.devPtr;

        run_f16_flash_attention_fprop(b, h, s_q, s_kv, d, layout,
                scaling_factor, isTraining, dropout_probability, devPtrQ,
                devPtrK, devPtrV, devPtrSoftmaxStats, devPtrO,
                devPtrDropoutSeed, devPtrDropoutOffset, CUDNN_DATA_BFLOAT16);

        checkCudaErr(cudaDeviceSynchronize());
        checkCudaErr(cudaMemcpy(oTensor.hostPtr, oTensor.devPtr,
                sizeof(oTensor.hostPtr[0]) * oSize, cudaMemcpyDeviceToHost));
        checkCudaErr(cudaDeviceSynchronize());

        std::cout << "\n======================================================="
                     "=================================\n";
    }
}

main_cudnn() {
#if (CUDNN_VERSION >= 8900)
    TEST_CASE("BF16 LLM Flash MHA Fprop sample",
            "[frontend][fusion][BF16LLMFprop]") {
        std::cout << "TEST_CASE :: BF16 LLM Flash MHA Fprop with backend API"
                  << std::endl;
        INFO("TEST_CASE ::  BF16 LLM Flash MHA Fprop with backend API");

#if (CUDART_VERSION < 12000)
        SKIP("Test requires CUDA version greater than 12.0");
#endif

        int64_t b = 2; // batch size
        int64_t h = 12; // head dim
        int64_t s_q = 2048; // q tensor is padded to this seq length
        int64_t s_kv = 2048; // k and v tensor is padded to this seq length
        int64_t d = 128; // hidden dim

        int64_t seed = 123456; // seed for generating the dropout mask

        MHA_Layout layout = MHA_Layout::
                SBH_INTERLEAVED; // layout of the tensors Q,K and V. BF16 LLM has layout [S,B,H,3,D]

        float scaling_factor = 0.5; // scale value before softmax

        bool isTraining = true; // training or inference mode
        double dropout_probability
                = 0.2f; // probability of dropout. Should be 0.0 for inference mode

        std::cout << "====PARAMETERS====" << std::endl;
        std::cout << "batch is " << b << ", head dim is " << h
                  << ", q sequence length is " << s_q
                  << ", kv sequence length is " << s_kv << ", hidden dim is "
                  << d << std::endl;

        void *devPtrQ = nullptr; // queries
        void *devPtrK = nullptr; // keys
        void *devPtrV = nullptr; // values
        void *devPtrSoftmaxStats = nullptr; // softmax stats
        void *devPtrO = nullptr; // final output

        // the setup is for the qkv interleaved layout (qkv interleaved assumes s_q = s_kv)
        int64_t qkvTensorDim[] = {s_q, b, h, 3, d};
        CUDNN_FRONTEND_UNUSED(qkvTensorDim);

        int64_t xSize = s_q * b * h * 3 * d;
        Surface<half> xTensor(xSize, false);
        devPtrQ = (void *)xTensor.devPtr; // q points to the top of qkv
        devPtrK = (void *)(xTensor.devPtr + d); // k is at an offset of d
        devPtrV = (void *)(xTensor.devPtr
                + 2 * d); // v is at an offset of 2 * d

        void *devPtrDropoutSeed = nullptr; // Seed for dropout
        void *devPtrDropoutOffset = nullptr; // Offset for dropout

        int64_t scaleSize = 1;
        Surface<int64_t> dropoutSeed(scaleSize, false, seed);
        devPtrDropoutSeed = (void *)dropoutSeed.devPtr;
        Surface<int64_t> dropoutOffset(scaleSize, false, (int64_t)1);
        devPtrDropoutOffset = (void *)dropoutOffset.devPtr;

        int64_t softmaxStatsSize = b * h * s_q;
        Surface<float> softmaxStats(softmaxStatsSize, false);
        if (isTraining) { devPtrSoftmaxStats = (void *)softmaxStats.devPtr; }

        int64_t oSize = b * s_q * h * d;
        Surface<half> oTensor(oSize, false);
        devPtrO = (void *)oTensor.devPtr;

        run_f16_flash_attention_fprop(b, h, s_q, s_kv, d, layout,
                scaling_factor, isTraining, dropout_probability, devPtrQ,
                devPtrK, devPtrV, devPtrSoftmaxStats, devPtrO,
                devPtrDropoutSeed, devPtrDropoutOffset, CUDNN_DATA_BFLOAT16);

        checkCudaErr(cudaDeviceSynchronize());
        checkCudaErr(cudaMemcpy(oTensor.hostPtr, oTensor.devPtr,
                sizeof(oTensor.hostPtr[0]) * oSize, cudaMemcpyDeviceToHost));
        checkCudaErr(cudaDeviceSynchronize());

        std::cout << "\n======================================================="
                     "=================================\n";
    }
}

#endif