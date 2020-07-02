# Nvidia backend implementation notes

The Nvidia backend for DNNL can be exposed to the user via the `dnnl::engine::kind::gpu` engine kind. Currently, for the case when user's system has both Intel and Nvidia GPUs, 
`DNNL_GPU_VENDOR=NVIDIA` Flag is used in CMake, since the devices are clustered based on the device vendor ID and index pattern can not be used to distinguish between Intel GPU and 
Nvidia GPU. However, Intel is working on restructuring the engine creation, so that it would be possible to choose engine kind and vendor kind at runtime.
Also, it is possible to create DNNL engines using `sycl::device` objects corresponding to Nvidia GPUs. The stream in Nvidia backend for DNNL defines an out-of-order SYCL queue by default. Similar to the existing DNNL API, user can specify an in-order queue when creating a stream if needed.

## build-time and Runtime dependency 

The Nvidia backend requires 
- Nvidia CUDA driver version  418.87.01 or 440.33.01
- cuBLAS library version  10.1 or 1.2, and 
- cuDNN library version 7.6.5.
- `LD_LIBRARY_PATH` should be set to `/Path/to/dpcpp/lib`

## build command

```bash
export CC=/path/to/dpcpp/install/bin/clang
export CXX=/path/to/dpcpp/install/bin/clang++
mkdir build
cd build
cmake -DDNNL_CPU_RUNTIME=DPCPP -DDNNL_GPU_RUNTIME=DPCPP -DDNNL_GPU_VENDOR=NVIDIA -G Ninja /path/to/oneDNN/root/folder -DOPENCLROOT=/path/to/the/root/folder/of/libOpenCL.so -DOPENCLHEADERS=/path/to/the/OpenCL-Headers/folder
```
## Memory
Currently, only the buffer-based DNNL API is supported for Nvidia backend.

## Suported Data Types

The Following table documents the supported data types.

| Data Type | Computation Mode            |
|-----------|-----------------------------|
| f32       | Training, Inference         |
| f16       | Inference                   |
| int8      | Inference (when applicable) |

## Supported Primitives

cuDNN functions are not necessarily the same as DNNL primitives due to lack of standard API for DNN.
For each primitive, the cuDNN equivalent function is added to the Nvidia backend for DNNL. However, the added backend cannot provide all functionalities supported by DNNL primitives. The detailed limitations of each cuDNN primitive are explained as follow.

### Binary

The `cudnnOpTensor` is equivalent of DNNL binary primitives.

* Only scales post-op is supported, Based on the current version of DNNL we have integrated the addition and multiplication operation. Once DNNL supports `min` and `max` operation we will integrate that.
* Blocking is only supported for `int8` and only in the C dimension with either 4 or 32 block size (same as other cuDNN primitives).

### Reorder

The `cudnnTransform` function is equivalent of DNNL reorder function. However, there are some limitations when using SYCL_API-DNN reorder on Nvidia GPU:

* Per dimension scaling is not supported (a single alpha and beta value is accepted by the transform tensor function)
* Blocking is only permitted for the channel dimension in cuDNN. This primitive currently supports block size of 4.
* Blocking is only supported when channel dimension is a multiple of the block size and the datatype is `int8`.

### Eltwise

The `cudnnActivationForward`amd `cudnnActivationBackward` is equivalent of eltwise forward and eltwise backward in DNNL respectively. There are some limitations when using Nvidia backend for eltwise primitve:

* cuDNN only supports the following operations - RELU, ELU, TANH, LOGISTIC, BRELU
* RELU is only supported with alpha = 0
* cuDNN expects x, y and dy as inputs to the backward pass, hence, only RELU and BRELU operations are supported in the backward pass.
* Forward pass supports f32, f16 and s8 data types. Although blocking is not supported for s8.
* Backward pass supports f32 and f16 data types.

### Convolution
The `cudnnConvolutionForward`, `cudnnConvolutionBackward` and `cudnnConvolutionBackwardFilter` is used to compute forward, backward or weights update for a batch convolution operation.

* Blocking is only supported for `int8` and only in the C dimension with block size of 4. Input and output tensors must have the same data type.
* For int8 (s8s8s8) with post-ops the operations are performed as s8s8f32 (due to cuDNN limitations) then reordered to s8 at the end which impacts performance.
* Direct convolution is not supported, so implicit GEMM is used in those cases. 
* Padding must be symmetrical. 
* Eltwise post-op limitations are the same as our eltwise limitation as post-ops are not fused.
* cuDNN requires padding tensors to 4 dimensions, so 1D convolutions are supported but are performed as 2D.

The following table shows the convolution status for the DNNL Nvidia backend:

#### Forward direction
| Weights Format | Winograd Supported | Supported Input Format | Supported Output Format | Supported Data Type | Limitations                                                                                                                                                                             |
|----------------|--------------------|------------------------|-------------------------|---------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 2D NCHW        | YES                | NCHW,NHWC              | NCHW, NHWC              | f32, f16            | The Winograd algorithm has limitations: <br>  * Filter size must be 3x3 or 5x5.<br>  * Dilation must be zero for all dimensions.<br>  * Horizontal and vertical filter stride must be 1. |
| 2D NHWC        | NO                 | NHWC                   | NHWC                    | f32, f16, int8      | * Dilation must be zero in all dimensions. <br> * Output feature maps must be multiple of 4 for `int8` type.                                                                               |
| 3D NCHW        | NO                 | NCHW, NHWC             | NCHW, NHWC              | f32, f16            |                                                                                                                                                                                         |

#### Backward direction
| Weights Format | Winograd Supported | Supported Input Format | Supported Output Format | Supported Data Type | Limitations                                                                                                                                                                                                                           |
|----------------|--------------------|------------------------|-------------------------|---------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 2D NCHW        | YES                | NCHW,NHWC              | NCHW                    | f32, f16            | <br>1. Dilation must be zero for all dimensions. <br> 2. The Winograd algorithm Limitation:<br>  * Filter size must be 3x3 or 5x5.<br>  * Dilation must be zero for all dimensions.<br>  * Horizontal and vertical filter stride must be 1. |
| 2D NHWC        | NO                 | NHWC                   | NHWC                    | f32, f16            |                                                                                                                                                                                                                                       |
| 3D NCHW        | NO                 | NCHW, NHWC             | NCHW                    | f32, f16            |                                                                                                                                                                                                                                       |

### Batch normalization
The closest equivalent to DNNL batch normalization can be `cudnnBatchNormalizationForward`, `cudnnBatchNormalizationBackward` operations. However, there are some difference between cuDNN and DNNL batch normalization.

#### Forward direction

* When `global_stats` flag is set for batch normalization, the mean and variance are input only parameters. However, cuDNN does not have the option to accept the mean and variance as an input in the forward training operation. Therefore,`cudnnBatchNormalizationForwardInference` is used to match the DNNL feature.

* The cuDNN precision is different from that of DNNL for Batch Normalization. (e.g `fp:0.0170898 dt:0.0170907 diff:8.27014e-07 rdiff:4.83922e-05`)

* The Forward training with no flags accepts mean and variance as an output. However, in cuDNN the mean and variance are running mean and variance respectably so they are both input and output variable. Therefore, they are required to have a sensible value (cannot be NaN). Since DNNL will not set value for the mean and variance when no flag is passed, the NaN can be propagated as a result. To avoid NaN propagation,  `cudaMemset` function is used to initialize the mean and variance with zero.

* cuDNN always requires the values for scale and shift. When shift and scale are not defined in DNNL, `cudaMemset` is used to initialize scale to 1 and shift to 0.

* For performance reason in the backward pass, cuDNN requires the mean and inverse variance to be saved in the forward pass. Therefore, when Nvidia backend is used for batch normalization, the workspace must be provided to save the mean and inverse variance.

* When ` dnnl_fuse_norm_relu` flag is set for batch normalization, The `cudnnActivationForward` operation is called immediately after the batch normalization, since cuDNN does not have a fused batch normalization with RELU. The implementation for element-wise post operations is the same.

* When ` dnnl_fuse_norm_relu` is used the intermediate output of batch normalization, which is used as an input to the activation function, is saved in the workspace as well. This is required to compute the backward pass for ` dnnl_fuse_norm_relu` flag. 

*  Forward pass supports f32, f16 and s8 data types. Although blocking is not supported for s8.

#### Backward direction

* cuDNN uses alpha and beta parameters to blend the dy, shift and scale. Since DNNL does not have this feature, the alpha and beta value in the backward direction is set to 1 and 0 respectively to avoid blending.

* Nvidia backend for backward direction requires the workspace as an input containing the mean and inverse variance computed in the forward pass.

* The Nvidia backend for DNNL does not support the backward direction for batch normalization when the flag is set to `global_stats`. This is due to the fact that DNNL will skip the

<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=$d_{y} -= \left ( \frac{\beta + \left ( \frac{src-mean}{\sqrt{\delta ^{2} + \epsilon }} \right )}{NHW} \right )$" > 
</p>

since the mean and variance are constant, however, cuDNN does not have an option to skip this operation.

* When ` dnnl_fuse_norm_relu` flag is set, Nvidia backend requires the intermediate result of the batch normalization saved in the forward pass. This is used to compute the backward direction of the activation function used for RELU.


### Inner product

The inner product primitives is an implementation of matrix multiplication + bias activation.
There are two implementation of inner product in cuDNN backend.

#### Using GEMM

The default backend for inner product is the gemm backend using `cublasGemmEx` for forward, backward data, and backward weight and `cudnnReduceTensor` for backward bias. A function called `gemm_consitency_check()`, `dense_check()` is used to see if the gemm backend can be used for inner product. reorder_check is used when reorder is required. If non of the above condition met, it falls back to the convolution backend.
`cudnnActivationForward` operation is used for eltwise operation and `cudnnAddTensor` is used for bias operation. the beta parameter in gemm is used for the sum scale and alpha parameter is used for the output scale.


#### Using convolution  

For the forward direction, this operation can be implemented by using `cudnnConvolutionBiasActivation` by converting the inner product to `1x1` convolution. For the backward direction the inner product operation will be equivalent of 
`cudnnConvolutionBackwardData` and `cudnnConvolutionBackwardWeights` and  `cudnnConvolutionBackwardBias` when applied. 
This implementation of inner product has the following restrictions and performance implications:

* The only blocked layouts are those that are supported in cuDNN - namely that the blocking is done on the C dimension, the block size is 4, and only for `int8` inference. The additional requirement is that both the input and filter must be blocked.

* The ReLU and sum are supported as a fused post-op, for other post-op we manually call eltwise primitive. So the limitation for the eltwise primitive is applied here.

* The restrictions for the convolution primitive are applied here for input and filter format. When required, the filter is internally reordered to match the convolution restriction

* For `int8`, cuDNN requires both input and output feature maps to be a multiple of 4. 

### Softmax/LogSoftmax 

The `cudnnSoftmaxForward` and `cudnnSoftmaxBackward` are used to implement the softmax primitive. For log softmax primitive the same functions will be used and the algorithm selection in cuDNN for the above mentioned functions will be changed to `CUDNN_SOFTMAX_LOG`.

* The softmax axis is supported for only the channel dimension, (i.e., axis=1)
* `int8` is not a supported datatype in DNNL for this primitive and hence no support for `int8` in cuDNN
* There is bug in cuDNN softmax for 5D tensor with format `NHWC`. When the channel size is > 1, it only applies softmax for one channel and leave the others untouched

### Concat
The concat operation uses the reorder primitive to concatenate tensors over the chosen dimension, So the same limitation as reorder applies here. 

### Pooling

The pooling primitive in the Nvidia backend is implemented with the `cudnnPoolingForward` and `cudnnPoolingBackward` functions for forward and backward propagation respectively.

* cuDNN only allows the use of symmetric padding, i.e. padding at the beginning of a dimension must be the same as the padding at the end of that dimension. DNNL doesn't have this limitation. Therefore,

    1. configurations where padding in the beginning is larger than padding at the end are supported and work as expected.

    2. for configurations where padding at the end is larger than padding in the beginning of any dimension, the primitive returns `status::unimplemented`.

* For backward propagation cuDNN requires the parameters `x`, `y`, `dx` and `dy`, while DNNL requires only `dx`, `dy` and workspace when the `MAX` algorithm is used. Hence, the workspace is used to store the `x` and `y` parameters in the forward pass for the Nvidia backend. Therefore, the workspace is always required when the Nvidia backend is used (except for forward inference).


### LRN

The lrn primitive in the Nvidia backend is implemented with the `cudnnLRNForward` and `cudnnLRNBackward` functions for forward and backward propagation respectively.
* There is a difference in the LRN algorithm used in DNNL and CuDNN which causes a mismatch when the local size is even. (See issue [#75](https://github.com/otcshare/mkl-dnn/issues/75) for a detailed explanation)
* CuDNN supports NCHW tensor formats for all valid dimensions. However, it does not support the NHWC tensor format for above 5 dimensions.

### Resampling

The `cudnnSpatialTfSamplerForward` and `cudnnSpatialTfSamplerBackward` are used to implement the resampling primitive. The nvidia's spacial sampling is based on [Spacial Transformer Network](https://papers.nips.cc/paper/5854-spatial-transformer-networks.pdf) where all the data locations are normalized between `-1 <= (xi, yi) <= 1`.
* `cuDNN` backend requires a grid of coordinates that can be sample-up/down based on theta. The grid is generated by `cudnnSpatialTfGridGeneratorForward`
* The theta is a `MB*2*3` matrix scaling factor for each coordinate and is used to generate the grid.
* The grid value must be normalized between [-1 , 1]. `cuDNN` clamps the out of bounds coordinate to zero. Therefore, we need to manually clamp the out of bound coordinate 
to edges in order to avoid incorrect result.
* Only 2d sampling with 4d tensor is supported.
* Since `cuDNN` computation is different from that of `DNNL`, the error threshold is smaller than other DNNL implementation, so we need to reduce the accuracy for `float` and `float16` types.
* The backward pass requires an output parameter for `dgrid` which cannot be `nullptr` ,however, since the grid's coordinates are not a tunable parameter in `DNNL` model, we create a dummy memory for `dgrid` and delete it in the destructor of the primitive.

### Matrix Multiplication
The matrix multiplication primitive in the Nvidia backend is implemented with `cublasGemmEx` and `cublasGemmStridedBatchedEx` functions.
* Zero_points support is not provided by cublas and hence not supported by the Nvidia backend.
