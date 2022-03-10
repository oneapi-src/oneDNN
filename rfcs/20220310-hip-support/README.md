# AMD backend support (RFC)

## Introduction
The idea for this RFC is to expose the AMD backend for oneDNN.
- oneDNN already has the NVIDIA backend as an experimental feature, this RFC advances to implement the support to AMD platform.
- This branch introduces HIP backend support for the primitives that are currently supported for CUDA backend.
- Testing and Performance analysis are yet to be done.
- Build process and compilation is successfully validated.

## Proposal
The primitives are built using MIOpen and hipDNN library, which are the open-source DNN libraries of AMD.
- The backend can be exposed to the user via DNNL_GPU_VENDOR=AMD flag used in CMake.
- This contribution would extend oneDNN's support for primitives and their supported post-ops and Quantization from CUDA backend to HIP backend.

Since MIOpen backend implementation is heavily inspired by cuDNN backend , the limitations or bugs from there are likely to be inherited here.

## Supported Primitives and Implementation Limitations:


## Binary:

Binary primitive in MIOpen library is implemented through miopenOpTensor, 
MIOpen supports only 4 modes of binary operations via enumerators: miopenTensorOpAdd, miopenTensorOpMul, miopenTensorOpMin, miopenTensorOpMax.

* This implementation keeps the same conditions as CUDA backend for blocking, broadcast and supported datatypes.
* Testing: This developed backend will be tested after addressing the comments from the oneDNN team and the results will be updated.

Limitations :
1. We currently do not have much information about blocking, broadcast in MIOpen library documentation.


## ETLWISE :

The miopenActivationForward and miopenActivationBackward is the equivalent of eltwise forward and eltwise backward in oneDNN respectively.
The eltwise primitive works with arbitrary data tensors. There is no special meaning associated with any logical dimensions.
A primitive to perform elementwise operations such as the rectifier linear unit (ReLU).

Propagation: 
Forward,Backward

Limitations: 
 
1. Supported Data Types: supports f32, f16 data types. Doesn’t support s8 & bf16.
2. MIOpen  only supports the following operations - RELU, ELU, TANH, LOGISTIC,  CLIPPEDRELU, ABS, POWER.




## LRN(LOCAL RESPONSE NORMALIZATION) :

The local response normalization primitive in the AMD backend is implemented with the
hipdnnLRNForward and hipdnnLRNBackward functions for forward and backward propagation respectively.

The implementation of LRNCrossChannelForward and LRNCrossChannelBackward in miopen require workspace creation and miopenOpTensor post operation in addition
miopenLrnBackward & miopenLrnForward hence we choose straight forward implementation in hipDNN.

Propagation:
Forward and Backward.

Limitations :
1. Data Types: 
supports f32, f16 data types. Doesn’t support bf16.


## SOFTMAX :

Softmax implementation algorithms :
* Softmax has three implementations of its algorithms, these are enumerated upon "miopenSoftmaxAlgorithm_t" varible,
* Currently MIOpen supports three algorithmic implementations namely:
   MIOPEN_SOFTMAX_FAST
   MIOPEN_SOFTMAX_ACCURATE
   MIOPEN_SOFTMAX_LOG
* NOTE: The MIOPEN_SOFTMAX_LOG implementation in MIOpen is the direct implementation of logsoftmax in oneDNN.

Modes of Operation :
MIOPEN_SOFTMAX_MODE_INSTANCE(selected by default)
MIOPEN_SOFTMAX_MODE_CHANNEL(can be used if there is forwardv2 or backwardv2 implementation)

Propagation:
Forward and Backward.

Limitations:

1. Datatypes are limited to fp16 and fp32.
2. Supported Datatypes:
miopenHalf  - 16-bit floating point(fp16)
miopenFloat - 32-bit floating point(fp32)



## BATCH NORMALIZATION :

The equivalent to oneDNN batch normalization are miopenBatchNormalizationForward includes miopenBatchNormalizationForwardTraining,
miopenBatchNormalizationForwardInference and miopenBatchNormalizationBackward operations.

Propagation:
Forward and Backward.

LIMITATIONS :

1. Supported Data Types: 
supports f32, f16 data types. Doesn’t support s8 & bf16.


## POOLING :

The pooling primitive in the AMD backend is implemented with the miopenPoolingForward 
and miopenPoolingBackward functions for forward and backward propagation respectively.

Propagation:
Forward and Backward.


LIMITATIONS :
1. Supported Data Types: 
supports f32, f16 ,miopenint8data types. Doesn’t support s8 & bf16,S32.




## CONVOLUTION  : 

1. This convolution primitive in AMD Backend is implemented as miopenConvolutionForward, miopenConvolutionBackward  
is used to compute forward, backward by data or backward by weights for a convolution operation.

2. As filter algo is implemented in HIPDNN as hipdnnConvolutionBackwardFilter.

The implementation of ConvolutionBackwardFilter in miopen requires additional workspace creation and as well as different implementation
based on parameter beta hence we choose straight forward implementation in hipDNN.

 
PROPOGRATION :
Forward, BackwardData and BackwardBias.

LIMITATIONS :

1. Supported Data Types:  fp16, fp32, bf16, miopenInt8. Doesn’t support  U8 datatype.
2. Some of the features of cuDNN backend implementation are not supported in hipDNN Backend because of the missing below 
        - hipdnnConvolutionBiasActivationForward 
	- hipdnnTransformTensor    
	- hipdnnGetConvolutionForwardAlgorithmMaxCount
        - hipdnnConvolutionFwdAlgoWinograd_NONFUSED.
	- hipdnnGetConvolutionBackwardDataAlgorithmMaxCount 
        - hipdnnGetConvolutionBackwardFilterAlgorithmMaxCount.



## DECONVOLUTION :

Deconvolution primitive is implemented through the convolution with  miopenConvolutionBackwardBias.

Limitations :
1. Supported Data Types:  fp16, fp32, bf16, miopenInt8
doesn’t support U8 datatype.



## Build command 
export CC=/path/to/hip/install/bin/clang    --> hip supported SYCL C compiler
export CXX=/path/to/hip/install/bin/clang++ -->  hip supported SYCL CPP compiler
mkdir build
cd build
cmake -DDNNL_CPU_RUNTIME=DPCPP -DDNNL_GPU_RUNTIME=SYCL -DDNNL_GPU_VENDOR=AMD ..

## Open Questions
The implementation is subject to change as we go through the review and the testing phases.
Currently the HIP support for DPCPP(SYCL) compiler is in experimental stage, and the backend is not completely supported on AMD devices.
Hence this effort will also explore any alternatives for running HIP backend on AMD platforms.
