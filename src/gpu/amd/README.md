# AMD backend support

## General information

Support for AMD backend is implemented via SYCL HIP backend. The feature is
disabled by default. Users must enable it at build time with a CMake option
`DNNL_GPU_VENDOR=AMD`. The AMD GPUs can be used via oneDNN engine abstraction.
The engine should be created using `dnnl::engine::kind::gpu` engine kind or the
user can provide a `sycl::device` objects that corresponds to AMD GPUs.

## Pre-requisites
* [oneAPI DPC++ Compiler with support for HIP AMD](https://github.com/intel/llvm/blob/sycl/sycl/doc/GetStartedGuide.md#build-dpc-toolchain-with-support-for-hip-amd), version [2022-12](https://github.com/intel/llvm/releases/tag/2022-12)
* [AMD ROCm](https://github.com/RadeonOpenCompute/ROCm), version 5.3 or newer
* [MIOpen](https://github.com/ROCmSoftwarePlatform/MIOpen), version 2.18 or newer (optional if AMD ROCm includes the required version of MIOpen)
* [rocBLAS](https://github.com/ROCmSoftwarePlatform/rocBLAS), version 2.45.0 or newer (optional if AMD ROCm includes the required version of rocBLAS)

## Build command

```bash
export CC=/path/to/dpcpp/install/bin/clang
export CXX=/path/to/dpcpp/install/bin/clang++
mkdir build
cd build
cmake -DDNNL_CPU_RUNTIME=SYCL -DDNNL_GPU_RUNTIME=SYCL \
      -DDNNL_GPU_VENDOR=AMD -G Ninja ..
```

If you have AMD ROCm, MIOpen or rocBLAS installed in non-standard locations or
you want to use MIOpen or rocBLAS that is not part of the AMD ROCm package then
the following CMake and environment variables can be used to specify their location:
* `MIOPENROOT`
* `HIPROOT`
* `ROCBLASROOT`

## Memory

Both buffer-based and USM-based oneDNN APIs are supported for AMD backend.

## Suported Data Types

The following table documents the supported data types. In generic this is for all primitives,
but primitive wise which datatypes are supported are mentioned under each primitive.

| Data Type | Computation Mode                      |
|-----------|---------------------------------------|
| f32       | Training, Inference                   |
| f16       | Inference                             |
| s8        | Inference (when applicable)           |
| bf16      | Training, Inference (when applicable) |


## Supported Primitives and Implementation Limitations

The AMD backend cannot provide all functionalities supported by oneDNN primitives.
because MIOpen and rocBLAS lack some features. The detailed limitations of each MIOpen
and rocBLAS based primitive are explained below.

### Binary

The `miopenOpTensor` is the equivalent of oneDNN binary primitive.

* Supported data types are `f32`, `f16`, `s32`.
* Datatypes of `SRC0`, `SRC1` and `DST` should be the same.
* Supported formats are `NCDHW`, `NCHW`, `NCW`, `NC`, `N`.
* Blocked formats are not supported.
* Only `scales` attribute is supported.
* Post-ops are not supported.
* Supported algorithms are `binary_add`, `binary_mul`, `binary_min`, `binary_max`.

### Convolution

The `miopenConvolutionForwardImmediate` is used to compute forward. 
The `miopenConvolutionBackwardDataImmediate` and `miopenConvolutionBackwardWeightsImmediate` 
are used to compute backward by data and backward by weights respectively.

The implementation supports both Forward and Backward directions:

#### Forward direction

* Supported data types combinations:
    | Source   |  Weights  |  Destination  |     Bias      |
    |-----------|-----------|---------------|---------------|
    |   f32     |    f32    |     f32       |       f32     |
    |   f16     |    f16    |     f16       |       f16     |
    |   s8      |     s8    |      s8       | Not supported |
    |   bf16    |   bf16    |     bf16      | Not supported |
* Supported formats: `NCDHW`, `NCHW`, `NCW` (with bias) and `NDHWC`, `NHWC`, `NWC` (without bias)
* Supported post-ops: `eltwise` (`eltwise_relu`, `eltwise_tanh`, `eltwise_elu`, `eltwise_logistic`) and `sum`
* Supported attributes: `scales`
* Supported algorithms : `winograd`, `direct`

#### Backward direction

* Supported data types combinations:
    |  Source   |  Weights  |  Destination  |     Bias      |
    -----------|-----------|---------------|---------------|
    |   f32     |    f32    |     f32       |       f32     |
    |   bf16     |    bf16    |     bf16       |       bf16     |
* Supported formats: `NCDHW`, `NCHW`, `NCW` (with bias) and `NDHWC`, `NHWC`, `NWC` (without bias)
* Supported algorithms : `winograd`, `direct`

#### Limitations

* Source, weights and destination tensors must have the same format
* Post-op sum scale with non-zero fractional part can lead to incorrect results
* Zero points are not supported
* Post-ops are implementated via separate operations
* Bias addition is implemented with `miopenOpTensor`

### Deconvolution

* Deconvolution primitive is implemented through the convolution with swapped input
  and output channels.
* Post-ops are not supported.

### Eltwise

The implementation supports both forward and backward directions. The 
`miopenCreateActivationDescriptor` and `miopenSetActivationDescriptor` are used
to create the activation descriptor. And the `miopenActivationForward` and 
`miopenActivationBackward` are used for the execution.

* Supported formats: `NCDHW`, `NDHWC`, `NCHW`, `NHWC`, `NCW`, `NWC`, `NC`, `N`

##### Forward Direction
* Supported algorithms: `relu`, `tanh`, `elu`, `soft_relu`, `abs`, `logistic`.
* soft_relu is only supported with `alpha = 1`.
* Supported data types are `f32` and `f16`.
* Post-ops are not supported.

##### Backward Direction
* Supported algorithms: `relu` and `soft_relu`.
* soft_relu is only supported with `alpha = 1`.
* Supported data types are `f32`.

### Softmax / Logsoftmax

The implementation supports both forward and backward directions. The primitive
was implemented using `miopenSoftmaxForward_V2` and `miopenSoftmaxBackward_V2`.

* Supported formats: `NCDHW`, `NDHWC`, `NCHW`, `NHWC`, `NCW`, `NWC`, `NC`.
* Only `axis = 1` is supported.

##### Forward Direction
* Supported data types: `f32` and `f16`.
* Post-ops are not supported.

##### Backward Direction
* Supported data types: `f32`.

### Local Response Normalization (LRN)

The implementation supports both forward and backward directions.
The `miopenCreateLRNDescriptor` and `miopenSetLRNDescriptor` are used to set the LRN desriptor.
The `miopenLRNForward` and `miopenLRNBackward` are used for the execution.

* Supported formats: `NCHW`, `NHWC`, `NCW`, `NWC`, `NC`.

##### Forward Direction
* Supported data types: `f32`, `f16`.
* Supported algorithms: `lrn_across_channels`, `lrn_within_channel`.
* `lrn_within_channel` supports only 2D spatial cases.
* Post-ops are not supported.

##### Backward Direction
* Supported data types: `f32`.
* Supported algorithms: `lrn_across_channels`, `lrn_within_channel`.
* `lrn_within_channels` supports only 2D spatial cases.

### Pooling

The Pooling primitive in the AMD backend is implemented with the following API's:
* `miopenCreatePoolingDescriptor`, `miopenSetNdPoolingDescriptor`, `miopenSetPoolingIndexType`,
and `miopenSetPoolingWorkSpaceIndexMode` are used to set the pooling descriptor
* `miopenPoolingGetWorkSpaceSizeV2` is used for getting a work space size.
* `miopenPoolingForward` and `miopenPoolingBackward` are used for the execution.

##### Forward direction

* Supported datatype for forward Training `f32`.
* Supported datatypes for forward Inference `f32`, `f16`.
* Only 1D, 2D and 3D pooling is supported.
* Only `NCDHW`, `NCHW`, `NCW` formats are supported.
* Supported algorithms are `pooling_max`, `pooling_avg_include_padding`, `pooling_avg_exclude_padding`.
* Post-ops are not supported.

##### Backward direction

* Supported datatypes are `f32`.
* Only 1D, 2D and 3D pooling is supported.
* Only `NCDHW`, `NCHW`, `NCW` formats are supported.
* Supported algorithms are `pooling_max`, `pooling_avg_include_padding`, `pooling_avg_exclude_padding`.

### Reduction

The Reduction primitive is implemented with the following API's:
* `miopenCreateReduceTensorDescriptor` and `miopenSetReduceTensorDescriptor` are used to set the reduction tensor descriptor
* `miopenGetReductionWorkspaceSize` is used for getting a workspace size.
* `miopenReduceTensor` is used for execution

* Supported datatypes are `f32`, `f16`.
* Only `NCDHW`, `NCHW`, `NCW`, `NC`, `N` formats are supported.
* Supported algorithms are `reduction_max`, `reduction_min`, `reduction_sum`, `reduction_mul`,
  `reduction_mean`, `reduction_norm_lp_sum`, `reduction_norm_lp_power_p_sum`
* `reduction_norm_lp_sum` algorithm supported only for the `p` value 2
* `reduction_norm_lp_power_p_sum` supported only for the `p` value 1
* Only `eps = 0` is supported.
* Post-opst are not supported.

### Matrix Multiplication

The matrix multiplication primitive is implemented with `rocblas_gemm_ex`
and `rocblas_gemm_strided_batched_ex` functions.

* Supported data types are `f32`, `f16`, `bf16` and `s8/s32`.
* Currently only below 5 combinations are supported:

    | Source    | Weights   | Destination   |   Bias        |
    |-----------|-----------|---------------|---------------|
    |   f32     |   f32     |   f32         |   f32         |
    |   f16     |   f16     |   f16         |   f16         |
    |   s8      |    s8     |   s32         |   s32         |
    |   bf16    |   bf16    |   bf16        | Not supported |
    |   bf16    |   bf16    |   f32         |   f32         |
    
* Blocked formats are not supported.
* Zero points are not supported.
* Scales are not supported.
* Post-op `eltwise` with `eltwise_relu`, `eltwise_tanh`, `eltwise_elu`, `eltwise_logistic` is supported
* Post-op `sum` is supported. For s8 case (for the 3rd combination in above table),
  only scales without fractional part are supported.
* Source and weights broadcasting is supported in the batched case.
* Only 1D, 2D, 3D supported.
* Supported formats are `NCW`, `NWC`, `NC`, `CN`, `N`

### Inner product

The inner product primitive is implemented with `rocblas_gemm_ex` and 
`rocblas_gemm_strided_batched_ex` functions for forward, backward data and backward weight 
and `miopenReduceTensor` for backward bias. A function called `gemm_consitency_check()`,
`dense_check()` is used to see if the backend can be used for inner product.
`reorder_check()` is used when reorder is required. `miopenActivationForward` operation is
used for eltwise operation and `miopenOpTensor` is used for bias operation. The
`beta` parameter in gemm is used for the sum scale and `alpha` parameter is used
for the output scale.

* Supported formats : `NCW`, `NC`, `CN`, `N`

##### Forward direction

* Supported data types are `f32`, `f16`, `bf16` and `s8/s32`.
* Currently only below combinations are supported:
    | Source    | Weights   | Destination   | Bias          |
    |-----------|-----------|---------------|---------------|
    |   f32     |   f32     |   f32         |   f32         |
    |   f16     |   f16     |   f16         |   f16         |
    |   s8      |   s8      |   s32         |   s32         |
    |   bf16    |   bf16    |   bf16        | Not supported |
    |   bf16    |   bf16    |   f32         |   f32         |
* Zero points support is not provided.
* Post-op eltwise with `eltwise_relu`, `eltwise_tanh`, `eltwise_elu`, `eltwise_logistic` is supported
* Post-op sum is supported. For s8 case(for third combination in above table),
  only integer sum scale values are supported
* Blocked formats are not supported.

##### Backward direction
* Supported data types are `f32`, `bf16`.
* Currently only below combinations are supported:

    | Propagation       |   Source  |  Weights  | Destination   | Bias          |
    |-------------------|-----------|-----------|---------------|---------------|
    |   Backward Data   |    f32    |   f32     |   f32         | Not supported |
    |                   |    bf16   |   bf16    |   bf16        | Not supported |
    | Backward Weights  |    f32    |   f32     |   f32         |    f32        |
    |                   |    bf16   |   bf16    |   bf16        | Not supported |
* Zero points are not supported.
* Blocked formats are not supported.

### Batch normalization

The closest equivalent to oneDNN batch normalization can be
`miopenBatchNormalizationForwardTraining`, `miopenBatchNormalizationForwardInference`
and `miopenBatchNormalizationBackward` operations. 

##### Forward direction

* When `global_stats` flag is set for batch normalization, the mean and variance
  are input only parameters. However, MIOpen does not have the option to accept
  the mean and variance as inputs in the forward training operation. Therefore,
  `miopenBatchNormalizationForwardInference` is used to match the oneDNN feature.
  Although inference is not supported without `global_stats` flags set.
* The MIOpen precision is different from that of oneDNN for Batch Normalization.
  (e.g `exp from oneDNN: 0.000629427 got from miopen: 0.000629831 diff:4.04136e-07 rdiff:0.000642069`)
* The forward training with no flags accepts mean and variance as an output.
  However, in MIOpen the mean and variance are running mean and variance
  respectably so they are both input and output variable. Therefore, they are
  required to have a sensible value (cannot be NaN). Since oneDNN will not set
  value for the mean and variance when no flag is passed, the NaN can be
  propagated as a result. To avoid NaN propagation, `hipMemsetD32Async` function is
  used to initialize the mean and variance with zero.
* MIOpen requires the values for scale and shift. When shift and scale are
  not defined in oneDNN, `hipMemsetD32Async` is used to initialize scale to 1 and shift
  to 0.
* For performance reason in the backward pass, MIOpen requires the mean and
  inverse variance to be saved in the forward pass. Therefore, when AMD
  backend is used for batch normalization, the workspace must be provided to
  save the mean and inverse variance.
* When `dnnl_fuse_norm_relu` flag is set for batch normalization, the
  `miopenActivationForward` operation is called immediately after the batch
  normalization, since MIOpen does not have a fused batch normalization with
  `RELU`. The implementation of the elementwise post operations is the same.
* When `dnnl_fuse_norm_relu` is used, the intermediate output of batch
  normalization, which is used as an input to the activation function, is saved
  in the workspace as well. This is required to compute the backward pass for
  `dnnl_fuse_norm_relu` flag.
* Forward pass supports `f32`, `f16`.
* Blocked Formats are not supported.
* Only `NCDHW`, `NCHW`, `NCW`, `NC` formats are supported.
* Elementwise post-op is supported only for eltwise_relu.

##### Backward direction

* MIOpen uses `alpha` and `beta` parameters to blend the `dy`, `shift` and
  `scale`. Since oneDNN does not have this feature, the `alpha` and `beta`
  values in the backward direction are set to 1 and 0 respectively to avoid
  blending.
* AMD backend for backward direction requires the workspace as an input
  containing the mean and inverse variance computed in the forward pass.
* The AMD backend for oneDNN does not support the backward direction for
  batch normalization when the flag is set to `global_stats`. 
* When `dnnl_fuse_norm_relu` flag is set, AMD backend requires the
  intermediate result of the batch normalization saved in the forward pass. This
  is used to compute the backward direction of the activation function used for
  `RELU`.
* Backward pass supports only `f32` data types.
* Blocked formats are not supported.
* Only `NCDHW`, `NCHW`, `NCW`, `NC` formats are supported.

### Reorder

The `miopenTransform` function is the equivalent of oneDNN reorder function.

* Per dimension scaling is not supported (a single alpha and beta value is
  accepted by the transform tensor function).
* Supported data types: `f32`
