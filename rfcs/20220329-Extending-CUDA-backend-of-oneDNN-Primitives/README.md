# Extending CUDA backend of oneDNN - Primitives (RFC) 
## Introduction 

The idea of this RFC is to extend the current CUDA backend of oneDNN library with support for new primitives.

- This branch contains the support for primitives that are yet to be supported for CUDA backend. 

- Compilation and functional validation is yet to be concluded.  

## Proposal 

The primitives are built using oneDNN and are the open-source DNN libraries of Intel. 

 - This contribution would extend oneDNN's support for more primitives in CUDA backend. 

 - Wherever direct mapping of primitives from oneDNN is not available in cuDNN, the implementation would proceed with custom kernels for emulating their functionality on the backend device. 

 

## Supported Primitives,  Implementation Limitations and Proposed Solution 

 

### Reduction 

- Reduction primitive in CUDA backend can be implemented using cudnnReduceTensor primitive of cuDNN.
- cudnnReduceTensor supports only five modes of reduction operations via enumerators which are as follows.  
    - `CUDNN_REDUCE_TENSOR_MAX` 
    - `CUDNN_REDUCE_TENSOR_MIN` 
    - `CUDNN_REDUCE_TENSOR_MUL`
    - `CUDNN_REDUCE_TENSOR_ADD` 
    - `CUDNN_REDUCE_TENSOR_AVG` 

- Supported Data Types:  f32, s8 

 

#### Implementation Limitations 

- Lack of support for the algorithms of  

    - `reduction_norm_lp_max` 

    - `reduction_norm_lp_sum` 

    - `reduction_norm_lp_power_p_max` 

    - `reduction_norm_lp_power_p_sum` 

- No support for bf16 datatype. 

 

#### Proposed Solutions 

- To achieve the functionality of `reduction_norm_lp_power_p_max` and `reduction_norm_lp_power_p_sum` algorithms in cuDNN, `CUDNN_REDUCE_TENSOR_NORM1` algorithm mode of `cudnnReduceTensor` can be used in combination with `CUDNN_REDUCE_TENSOR_MAX` and `CUDNN_REDUCE_TENSOR_ADD` respectively. 

    - However, this workaround doesn't provide the flexibility to use arbitrary value for power `p` other than 1.  

- Similarly, the functionality of `reduction_norm_lp_max` and `reduction_norm_lp_sum` algortihms can be implemented using `CUDNN_REDUCE_TENSOR_NORM1` algorithm mode of `cudnnReduceTensor` in combination with `CUDNN_REDUCE_TENSOR_MAX` and `CUDNN_REDUCE_TENSOR_ADD` respectively. 

    - However, this workaround doesn't provide the flexibility to use arbitrary value for power `p` other than 2.  

- As an alternative to the above workarounds, the functionality of the above algorithms can be implemented through a custom kernel in SYCL. 

 

### Recurrent Neural Networks 

- RNN primitive in CUDA backend can be implemented using `cudnnRNNForward` and `cudnnRNNBackward` primitives of cuDNN. 

- cuDNN's primitives only support two algorithm of RNN. 

    - `CUDNN_LSTM` 

    - `CUDNN_GRU` 

- Supported Data Types: f16, f32 

 

#### Propagation 

Supported propagation kinds: Forward Training/Inference and Backward 

 

#### Implementation Limitations 

- No support for following algorithms in cuDNN. 

    - `augru` 

    - `lbr_augru` 

    - `lbr_gru` 

    - `vanilla_rnn` 

- No support for U8, bf16 datatypes. 

 

#### Proposed Solution 

The unsupported algorithms of `augru`, `lbr_augru`, `lbr_gru`, `vanilla_rnn` in cuDNN can be implemented using custom kernels, per the mathematical expressions of the algorithms. 

 

### PRelu  

There is no equivalent primitive API in cuDNN for PRelu. 

 

#### Propagation 

Forward Training/Inference and Backward 

 

#### Implementation Limitations 

Lack of equivalent support for PRelu primitive in cuDNN. 

 

#### Proposed Solution 

- PRelu primitive can be implemented using cuDNN with `cudnnActivationForward()` and `cudnnActivationBackward()`. 

    - Using `CUDNN_ACTIVATION_RELU` algorithm mode, it is possible to implement LEAKY RELU with a constant scalar alpha value, but not PRelu with a tensor alpha value. 

    - However, the functionality of PRelu can be accomplished using `CUDNN_ACTIVATION_RELU` algorithm mode in combination with `cudnnOpTensor()`.

- As a better efficient alternative, the primitive could be implemented using custom kernel, emulating the mathematical functionality of PRelu. 

 

### Layer Normalization 

There is no equivalent primitive API In cuDNN for Layer Normalization. 

 

#### Propagation 

Forward Training/Inference, Backward and Backward_Data 

 

#### Implementation Limitations 

Lack of equivalent support for Layer Normalization primitive in cuDNN. 

 

#### Proposed Solution 

- Layer Normalization primitive can be implemented using cuDNN with `cudnnBatchNormalizationForwardInference`, `cudnnBatchNormalizationForwardTraining` and `cudnnBatchNormalizationBackward` primitive APIs.  

    - This can be achieved with the algorithm mode of `CUDNN_BATCHNORM_PER_ACTIVATION` and by limiting the batch size to 1. 

- As an alternative to above workaround, this primitive can be implemented with a custom kernel which performs normalization on the last logical axis of the input tensor. 

 

### Shuffle 

There is no equivalent primitive API in cuDNN for Shuffle. 

 

#### Propagation 

Forward Training/Inference and Backward 

#### Implementation Limitations 

Lack of equivalent support for Shuffle primitive in cuDNN. 


#### Solution 

Shuffle primitive can be implemented using a custom kernel, performing transpose operation on input tensor. 


## Open Questions
- For any of the above solution, if a better approach is available, please suggest.
- The implementation is subject to change as we go through the reviews and results.
