# oneDNN Primitives' Support in cuDNN (RFC)

## Introduction
The idea for this RFC is to make few primitives of oneDNN to be supported by cuDNN
- This branch contains the support for the primitives that are yet to be supported for CUDA backend.
- Build process and compilation validation are in progress for few of the primitives.
- Testing and Performance analysis are yet to be done.

## Proposal
The primitives are built using oneDNN and are the open-source DNN libraries of Intel.
- The backend can be exposed to the user via DNNL_GPU_VENDOR=AMD flag used in CMake.
- This contribution would extend oneDNN's support for primitives and their supported post-ops and Quantization from oneDNN backend to cuDNN backend.

## Supported Primitives and Implementation Limitations

### Reduction
Reduction primitive in cudDNN library is implemented through cudnnReduceTensor. cuDNN supports only 5 modes of reduction operations via enumerators: CUDNN_REDUCE_TENSOR_MAX, CUDNN_REDUCE_TENSOR_MIN, CUDNN_REDUCE_TENSOR_MUL,CUDNN_REDUCE_TENSOR_ADD,CUDNN_REDUCE_TENSOR_AVG. In oneDNN ,reduction primitive works with arbitrary data tensors,There is no special meaning associated with any of the dimensions of a tensor. In cuDNN,  All tensor formats are supported up to  8 dimensions.

#### Limitations
 Supported Data Types: supports only f32 ,int8 data types. Doesn’t support bf16.

### Recurrent Neural Networks
RNN primitive in cuDNN library is implemented through cudnnRNNForward, cudnnRNNBackward is used to compute forward, backward by data or backward by weights for a RNN operation.
Currently cuDNN supports two algorithmic implementations namely: CUDNN_LSTM, CUDNN_GRU

#### Propagation 
Forward, BackwardData and BackwardWeights.

#### Implementation Limitations

1. Supported Data Types: f16, f32. Doesn’t support U8,bf16 datatype.
2. Some of the RNN algorithms in oneDNN  are not supported in cuDNN which are mentioned below
 - augru
 - lbr_augru
 - lbr_gru
 - vanilla_rnn

### PRelu
PRelu primitive in cuDNN library is implemented through cudnnActivationMode_t. cudnnActivationMode_t type uses two
activation function i.e cudnnActivationForward(), cudnnActivationBackward().

#### Propagation 
Forward,Backward
#### Implementation Limitations
1. Supported Data Types in cudnnOpTensor(): Supports FLOAT, INT8 ,BFLOAT16 data types. Doesn’t support u8 & s32.
2. cuDNN only supports following operations: CUDNN_ACTIVATION_RELU, cudnnOpTensor().

### Shuffle

Shuffle is a primitive to shuffle a 2D tensor data along an axis (C) with the group parameter (G) . The shuffle axis is thought to be a 2D tensor of size (C/G X G ) and it is being transposed to ( G X C/G )
 
#### Propagation
Forward,Backward

#### Implementation Limitations
Exact equivalent to shuffle in cuDNN is not found


## Open Questions
The implementation is subject to change as we go through the review and the testing phases

Currently the HIP support for DPCPP(SYCL) compiler is in experimental stage, and the backend is not completely supported on AMD devices

Hence this effort will also explore any alternatives for running HIP backend on AMD platforms
