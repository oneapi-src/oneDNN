# Extending CUDA backend of oneDNN - Primitives (RFC) 

 

## Introduction 

The idea of this RFC is to make few primitives of oneDNN to be supported by cuDNN 

- This branch contains the support for primitives that are yet to be supported for CUDA backend. 

- Build process and compilation validation are in progress for one primitive. 

 

## Proposal 

The primitives are built using oneDNN and are the open-source DNN libraries of Intel. 

- The backend can be exposed to the user via DNNL_GPU_VENDOR=NVIDIA flag used in CMake. 

- This contribution would extend oneDNN's support for primitives and their supported post-ops and Quantization from oneDNN backend to cuDNN backend. 

 

## Supported Primitives and Implementation Limitations 

 

### Reduction 

Reduction primitive in cuDNN library can be implemented using cudnnReduceTensor. CuDNN currently supports only five modes of reduction operations via enumerators which are CUDNN_REDUCE_TENSOR_MAX, CUDNN_REDUCE_TENSOR_MIN, CUDNN_REDUCE_TENSOR_MUL, CUDNN_REDUCE_TENSOR_ADD, CUDNN_REDUCE_TENSOR_AVG 

Supported Data Types:  f32, s8 

 

#### Implementation Limitations 

No support for bf16 

Unsupported Modes: reduction_norm_lp_max, reduction_norm_lp_sum, reduction_norm_lp_power_p_max, reduction_norm_lp_power_p_sum 

 

#### Solution 

To achieve reduction_norm_lp_power_p_max, reduction_norm_lp_power_p_sum operations in cuDNN. Call CUDNN_REDUCE_TENSOR_NORM1 and give its output to CUDNN_REDUCE_TENSOR_MAX or CUDNN_REDUCE_TENSOR_ADD accordingly. 

Similarly, for reduction_norm_lp_max, reduction_norm_lp_sum  operations in cuDNN, Call CUDNN_REDUCE_TENSOR_NORM2 and give its output to CUDNN_REDUCE_TENSOR_MAX or CUDNN_REDUCE_TENSOR_ADD accordingly. 

 

### Recurrent Neural Networks 

RNN primitive in cuDNN library can be implemented using cudnnRNNForward, cudnnRNNBackward. These are used to compute forward and backward by data/weights for an RNN operation. 

Currently cuDNN supports two algorithms: CUDNN_LSTM, CUDNN_GRU. 

Supported Data Types: f16, f32 

 

#### Propagation 

Forward, BackwardData and BackwardWeights. 

 

#### Implementation Limitations 

1. No support for U8, bf16 

No support for the following algorithms in cuDNN 

- augru 

- lbr_augru 

- lbr_gru 

- vanilla_rnn 

 

#### Solution 

The algorithms,  augru, lbr_augru, lbr_gru, vaniila_rnn in cuDNN can be implemented using custom kernels based on the mathematical expression of the algorithm 

 

### PRelu 

PRelu primitive in cuDNN library can be implemented using cudnnActivationMode_t and  cudnnOpTensor(). 

cudnnActivationForward(), cudnnActivationBackward() is used to compute forward, backward for a  PRelu operation. 

Supported Data Types in cudnnOpTensor(): f32, s8, bf16 

 

#### Propagation 

Forward, Backward 

 

#### Implementation Limitations 

No support for u8, s32 

Exact mapping of the primitive not found in cuDNN 

 

#### Solution 

cuDNN supports RELU and Leaky RELU. In Leaky RELU, alpha is scalar (double) but the required alpha is tensor. Hence LEAKY RELU can be implemented but not PRelu. However, this can be accomplished using existing RELU (CUDNN_ACTIVATION_RELU) and cudnnTensorOp (CUDNN_OP_TENSOR_MUL).  

Alternatively, custom kernels implementation can be done to support the PRelu operation i.e. If tensor holds positive values, the values can be passed directly to the output tensor without any extra operations but if tensor holds negative values, tensor multiplication will be performed. 

 

### Layer Normalization : 

Layer Normalization primitive in cuDNN library  can be implemented using one of the modes in cudnnBatchNormMode_t 

cudnnBatchNormalizationForwardInference, cudnnBatchNormalizationForwardTraining, cudnnBatchNormalizationBackward is used to compute forward inference, forward training and backward for Layer Normalization. 

 Supported Data Types: f16, f32 

 

#### Propagation 

forward inference, forward training and backward 

 

#### Implementation Limitations 

Exact mapping of the primitive not found in cuDNN 

 

#### Solution 

To implement Layer Normalization in cuDNN, batch normalization can be used with batch size =1. 

The other approach is to implement custom kernels where the kernel performs normalization over the last logical axis of the data tensor. 

 

### Shuffle 

Equivalent to shuffle in cuDNN is not found. 

 

#### Propagation 

Forward, Backward 

 

#### Implementation Limitations 

Exact mapping of the primitive not found in cuDNN 

 

#### Solution 

Shuffle can be implemented using custom kernels with transpose operation on tensors. 

 


## Open Questions
The implementation is subject to change as we go through the review and the testing phases

Currently the HIP support for DPCPP(SYCL) compiler is in experimental stage, and the backend is not completely supported on AMD devices

Hence this effort will also explore any alternatives for running HIP backend on AMD platforms
