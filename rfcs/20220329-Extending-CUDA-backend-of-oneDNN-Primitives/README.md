# Extending CUDA backend of oneDNN - Primitives (RFC) 

## Introduction 
The idea of this RFC is to make few primitives of oneDNN to be supported by cuDNN.
- This branch contains the support for primitives that are yet to be supported for CUDA backend.
- Build process and compilation validation are yet to be done. 

## Proposal 
The primitives are built using oneDNN and are the open-source DNN libraries of Intel. 
 - The backend can be exposed to the user via `DNNL_GPU_VENDOR=NVIDIA` flag used in CMake.
 - This contribution would extend oneDNN's support for primitives from oneDNN backend to cuDNN backend.
 - Wherever direct mapping of primitives from oneDNN to cuDNN is not possible, custom kernels will be created for extending CUDA backend of oneDNN.  

## Supported Primitives, Implementation Limitations and Proposed Solutions 

### Reduction
Reduction primitive in cuDNN library can be implemented using cudnnReduceTensor. cuDNN currently supports only five modes of reduction operations via enumerators which are `CUDNN_REDUCE_TENSOR_MAX`, `CUDNN_REDUCE_TENSOR_MIN`, `CUDNN_REDUCE_TENSOR_MUL`, `CUDNN_REDUCE_TENSOR_ADD` and `CUDNN_REDUCE_TENSOR_AVG`.

Supported Data Types:  f32, s8
#### Implementation Limitations 
- No support for bf16 
- Unsupported Modes: `reduction_norm_lp_max`, `reduction_norm_lp_sum`, `reduction_norm_lp_power_p_max`, `reduction_norm_lp_power_p_sum`
#### Solution 
- To achieve `reduction_norm_lp_power_p_max`, `reduction_norm_lp_power_p_sum` operations in cuDNN, Call CUDNN_REDUCE_TENSOR_NORM1 and give its output to `CUDNN_REDUCE_TENSOR_MAX` or `CUDNN_REDUCE_TENSOR_ADD` accordingly. This solution works only if power `p = 1`. 
- Similarly, for `reduction_norm_lp_max`, `reduction_norm_lp_sum`  operations in cuDNN, Call `CUDNN_REDUCE_TENSOR_NORM2` and give its output to `CUDNN_REDUCE_TENSOR_MAX` or `CUDNN_REDUCE_TENSOR_ADD` accordingly. This solution works only if power `p = 2`.  

### Recurrent Neural Networks 
RNN primitive in cuDNN library can be implemented using `cudnnRNNForward` and `cudnnRNNBackward`. These are used to compute Forward, BackwardData and BackwardWeights for an RNN operation. Currently cuDNN supports two algorithms: `CUDNN_LSTM` and `CUDNN_GRU`. 

Supported Data Types: f16, f32 
#### Propagation 
Forward, BackwardData and BackwardWeights 
#### Implementation Limitations 
- No support for U8, bf16 
- No support for `augru`, `lbr_augru`, `lbr_gru`and `vanilla_rnn` algorithms in cuDNN 
#### Solution 
The algorithms,  `augru`, `lbr_augru`, `lbr_gru`and `vanilla_rnn` in cuDNN can be implemented using custom kernels based on the mathematical expression of the algorithm. 

### PRelu  
Equivalent to PRelu in cuDNN is not found. 
#### Implementation Limitations 
Exact mapping of this primitive is not found in cuDNN. 
#### Solution 
- PRelu primitive in cuDNN library can be implemented using `cudnnActivationForward()`, `cudnnActivationBackward()` along with `cudnnOpTensor()`. But only LEAKY RELU is possible as alpha is a scalar (double) and not a tensor as required. Hence LEAKY RELU can be implemented but not PRelu. However, this can be accomplished using existing RELU (`CUDNN_ACTIVATION_RELU`) and `cudnnOpTensor()` (which is `CUDNN_OP_TENSOR_MUL`).  
- Alternatively, custom kernels implementation can be done to support the PRelu operation i.e., if tensor holds positive values, the values can be passed directly to the output tensor without any extra operations but if it holds negative values, tensor multiplication will be performed.  

### Layer Normalization : 
Equivalent to Layer Normalization in cuDNN is not found. 
#### Implementation Limitations 
Exact mapping of this primitive is not found in cuDNN.
#### Solution 
- Layer Normalization primitive in cuDNN library  can be implemented using one of the modes in `cudnnBatchNormMode_t`, `cudnnBatchNormalizationForwardInference`, `cudnnBatchNormalizationForwardTraining` and `cudnnBatchNormalizationBackward` with batch size = 1.
- The other approach is to implement custom kernels where the kernel performs normalization over the last logical axis of the data tensor.  

### Shuffle 
Equivalent to Shuffle in cuDNN is not found.
#### Implementation Limitations 
Exact mapping of this primitive is not found in cuDNN.
#### Solution 
Shuffle can be implemented using custom kernels with transpose operation on tensors. 

## Open Questions
- For any of the above solution, if a better approach is available, please comment. Open to suggestions.
- The implementation is subject to change as we go through the reviews and results.
