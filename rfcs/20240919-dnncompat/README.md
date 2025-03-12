# oneDNN Compatibility Layer

## Summary

The `dnncompat` compatibility layer in oneDNN is a thin, header-only wrapper
around the oneDNN API designed to provide an API that closely mirrors
vendor-specific DNN libraries. Its primary purpose is to be a drop-in
replacement for these libraries, making it useful for auto-translation tools
such as SYCLomatic as well as manual conversion of user applications written
with vendor-specific APIs to the oneDNN library.

Since oneDNN is optimized for a variety of hardware vendors, `dnncompat` will
allow users and tools to convert applications that are limited to a specific
hardware into hardware-agnostic projects. Thus, allowing the same codebase to
run on different platforms without modification.

The goals of `dnncompat` are as follows:

- To offer a light-weight, header-only wrapper around oneDNN which can easily be
  integrated into user projects and utilised by auto-translation tools.
- To deliver an API that closely resembles vendor-specific libraries, enabling
API calls to be converted primarily through find-and-replace.
- To handle differences in behaviour between oneDNN and vendor-specific
libraries internally, thus allowing users to easily convert applications without
having to debug differences between libraries

In addition, `dnncompat` leverages SYCL Unified Shared Memory (USM) support
within oneDNN to provide an API that can accept USM pointers as inputs similar
to vendor-specific library APIs which take pointers to memory as inputs. This
also allows `dnncompat` to work with memory that is natively allocated, such as
memory allocated via `cudaMalloc` making integration into native applications
possible too.

`dnncompat` will be shipped as a set of header files. When oneDNN is built from
source, these headers will be installed in the directory
`oneDNN-install-directory/include/dnncompat/`. Users will be able to integrate
`dnncompat` into their projects by including the oneDNN include directory in
their include path and adding `#include <dnncompat/dnncompat.hpp>` to their
project and linking with the oneDNN library.

## SYCLomatic Usage

At present, SYCLomatic provides its own header-only wrapper for oneDNN and
oneMKL which is used when converting applications. While this approach is
sufficient, the code produced by SYCLomatic can sometimes be challenging to
debug and understand for new oneDNN users as it relies on a mixture of custom
wrappers around oneDNN types along with direct oneDNN API calls. Additionally,
there are several differences between the supported primitives & their supported
features in oneDNN and vendor-specific libraries, making debugging incorrect
output more difficult.

To address these issues, the SYCLomatic headers can be substituted with the
`dnncompat` wrapper. By using `dnncompat`, the conversion process will be
simplified, reducing the complexity of the generated code and making it more
straightforward to understand and debug. This will ensure that developers can
convert their applications to `dnncompat` & oneDNN more easily, while also
reducing potential integration issues.

## Scope

Initially the focus will be on a specific set of operators commonly used in the
VGG-16 model. The following operators will be prioritised: Convolution, BiasAdd,
Pooling, Activation, Softmax, Local Response Normalization (LRN), and
TransformTensor (transpose). This will allow for creating a relatively small
prototype, while showing that `dnncompat` will be useful.

In this initial implementation, these operators will be supported in the forward
direction. The supported data types initially will be `f32` and `f16`. In the
future, the support for operations, data types & directions can be expanded
based on the need of SYCLomatic and community.

## Implementation Details

`dnncompat` will adopt an API design similar to vendor-specific libraries, which
are often C-based. Tensors and operations will be described using descriptors
(objects that encapsulate the necessary parameters for a given tensor or
operation). These descriptors will then be passed, together with pointers to GPU
 memory containing the tensor data, to the relevant API functions which will
execute the operations based on the provided parameters.

While `dnncompat` will be implemented with object-oriented programming (OOP)
functionality as it is a C++ library, the user-facing API will primarily
interact with these descriptors in a C-style API rather than through member
functions of classes. Since `dnncompat` is a C++ library, OOP features can be
used, but the C-style APIs will be the main user facing API to align with the
design of vendor-specific libraries, such as `cuDNN`.

The classes in `dnncompat` will be presented to users as opaque pointers, hiding
implementation details and providing API as close as possible to vendor-specific
libraries.

### Enumerations

The `dnncompat` library will include the following enumerations to handle
statuses, data types, tensor formats:

- `Status_t`: Status codes returned by operations, indicating success or
  specific types of errors.
- `DataType_t`: Specifies the data types used in tensor operations, such as
  `f32` or `f16`.
- `TensorFormat_t`: Defines the layout formats for tensors such as `NCHW` or
  `NHWC`.
- `NanPropagation_t`: Controls how NaN values are handled in operations.
- `MathType_t`: Used to enable or disable certain optimisations in
  vendor-specific libraries. Not supported in oneDNN, added for completeness.

### Non-Operation Classes

#### Library Context

Class **`detail::Handle`** serves as the context of the library, managing the
execution environment.

The `dnncompat::Handle` will internally manage the `dnnl::engine`,
`dnnl::stream`, and the underlying `sycl::queue`.

The `detail::Handle` class will have the following **private** constructors:

```c++
// Create a dnncompat::Handle with the specified engine kind
Handle(dnnl::engine::kind engine_kind) {}

// Create a dnncompat::Handle with an existing dnnl::engine
Handle(dnnl::engine &engine) {}

// Create a dnncompat::Handle with an existing sycl::queue
Handle(sycl::queue q) {}
```

To allow users to create instances of `dnncompat::Handle`, the library will
expose the following public functions. These functions will utilize the private
constructors but will return a `dnncompat::Status_t`, indicating any potential
errors during the handle's creation:

```c++
// Handle creation functions
Status_t create(dnnl::engine::kind engine_kind);
Status_t create(dnnl::engine &eng);
Status_t create(sycl::queue q);
```

Additional public functions and getters will be available for users to interact
with the handle:

```c++
// Retrieve the underlying engine, stream, or queue
dnnl::engine &getEngine();
dnnl::stream &getStream();
sycl::queue getQueue();

// Wait for operations to complete
void wait();
```

The following type alias will be exposed to users:

```c++
// Alias for user convenience
using Handle_t = dnncompat::detail::Handle *;
```

#### Tensor Descriptors

##### Base Descriptor Class

The `detail::DescriptorBase` class will be a base class, providing common
functionality for both `TensorDescriptor` and `FilterDescriptor` classes. It
will encapsulate parameters and information related to tensors such as number of
dimensions, dimension sizes, strides associated with each dimension, tensor
format, data type and an internal oneDNN memory descriptor
(`dnnl::memory::desc`). The classes `detail::TensorDescriptor` and
`detail::FilterDescriptor` (descriptor for Convolution weights) will inherit
from this class as both share similar functionality.

`DescriptorBase` provides several public getter functions, allowing users to
access various tensor properties:

```c++
// Retrieve the data type of the tensor
DataType_t getDataType() const;

// Get the tensor's format (layout)
TensorFormat_t getTensorFormat() const;

// Access the dimensions of the tensor
const std::vector<int> &getDims() const;

// Access the strides for each dimension
const std::vector<int> &getStrides() const;

// Get the number of dimensions in the tensor
size_t getNumDims() const;

// Retrieve the oneDNN memory descriptor associated with the tensor
const dnnl::memory::desc &getOneDNNDesc();
```

##### Tensor Descriptor

The `detail::TensorDescriptor` class will be exposed to users with the following
type alias:

```c++
using TensorDescriptor_t = dnncompat::detail::TensorDescriptor *;
```

List of free functions related to `TensorDescriptor`:

```c++
// Initialise a tensor descriptor
Status_t createTensorDescriptor(TensorDescriptor_t *tensorDesc);

// Destroy a tensor descriptor
Status_t destroyTensorDescriptor(TensorDescriptor_t tensorDesc);

// Set the parameters of a 4D tensor descriptor
Status_t setTensor4dDescriptor(TensorDescriptor_t tensorDesc,
        TensorFormat_t format, DataType_t dataType, int n, int c, int h,
        int w);

// Set the parameters of an n-dimensional tensor descriptor
Status_t setTensorNdDescriptor(TensorDescriptor_t tensorDesc,
        DataType_t dataType, int nbDims, const int dimA[],
        const int strideA[]);

// Set the parameters of a 4D tensor descriptor.
// This function takes strides as an argument instead of a tensor format
// which allows custom layouts
Status_t setTensor4dDescriptorEx(TensorDescriptor_t tensorDesc,
        DataType_t dataType, int n, int c, int h, int w, int nStride,
        int cStride, int hStride, int wStride);

// Get the parameters of a 4D tensor descriptor
Status_t getTensor4dDescriptor(const TensorDescriptor_t tensorDesc,
        DataType_t *dataType, int *n, int *c, int *h, int *w, int *nStride,
        int *cStride, int *hStride, int *wStride);

// Get the parameters of a n-dimensional tensor descriptor
Status_t getTensorNdDescriptor(const TensorDescriptor_t tensorDesc,
        int nbDimsRequested, DataType_t *dataType, int *nbDims, int *dimA,
        int *strideA);
```

##### Filter Descriptor

`detail::FilterDescriptor` encapsulates the parameters of the Convolution
operation's weights.

The `detail::FilterDescriptor` class will be exposed to users with the following
type alias:

```c++
using FilterDescriptor_t = dnncompat::detail::FilterDescriptor *;
```

List of free functions related to `FilterDescriptor`:

```c++
// Create a filter descriptor
Status_t createFilterDescriptor(FilterDescriptor_t *filt);

// Destroy a filter descriptor
Status_t destroyFilterDescriptor(FilterDescriptor_t filt);

// Set parameters of a 4-dimensional filter descriptor
Status_t setFilter4dDescriptor(FilterDescriptor_t filterDesc,
        DataType_t dataType, TensorFormat_t format, int k, int c, int h,
        int w);

// Set parameters of a n-dimensional filter descriptor.
Status_t setFilterNdDescriptor(FilterDescriptor_t filterDesc,
        DataType_t dataType, TensorFormat_t format, int nbDims,
        const int *filterDimA);

// Get parameters of a 4-dimensional filter descriptor
Status_t getFilter4dDescriptor(FilterDescriptor_t filterDesc,
        DataType_t *dataType, TensorFormat_t *format, int *k, int *c, int *h,
        int *w);

// Get parameters of a n-dimensional filter descriptor
Status_t getFilterNdDescriptor(FilterDescriptor_t filterDesc,
        int nbDimsRequested, DataType_t *dataType, TensorFormat_t *format,
        int *nbDims, int *filterDimA);
```

### Operations

#### Activation

The `detail::ActivationDescriptor` class encapsulates the parameters of the
activation operation: activation mode (initially `sigmoid` and `relu` will be
supported), NaN propagation mode, and coefficient (has different behaviour based
on the activation mode).

Free functions related to the Activation Operation:

```c++
// Create activation descriptor
Status_t createActivationDescriptor(ActivationDescriptor_t *activationDesc);

// Destroy activation descriptor
Status_t destroyActivationDescriptor(ActivationDescriptor_t activationDesc);

// Set the parameters of activation operation
Status_t setActivationDescriptor(ActivationDescriptor_t activationDesc,
        ActivationMode_t mode, NanPropagation_t reluNanOpt, double coef);

// Launch activation forward
Status_t activationForward(Handle_t handle,
        ActivationDescriptor_t activationDesc, const void *alpha,
        const TensorDescriptor_t xDesc, const void *x, const void *beta,
        const TensorDescriptor_t yDesc, void *y);
```

List of enumerations related to the Activation operation:

```c++
enum ActivationMode_t { ACTIVATION_SIGMOID, ACTIVATION_RELU };
```

#### AddTensor

Used for Convolution bias add. This operation does not have a separate
descriptor.

List of free functions related to BiasAdd operation:

```c++
Status_t addTensor(Handle_t handle, const void *alpha,
        const TensorDescriptor_t aDesc, const void *A, const void *beta,
        const TensorDescriptor_t cDesc, void *C);
```

#### Convolution

The `detail::ConvolutionDescriptor` class encapsulates the parameters of a
Convolution operation: number of groups, convolution mode (`convolution` or
`cross_correlation`), padding, convolution stride, dilation.

The `ConvolutionDescriptor` class will be exposed to users with the following
type alias:

```c++
using ConvolutionDescriptor_t = dnncompat::detail::ConvolutionDescriptor *;
```

List of free functions related to the Convolution operation:

```c++
Free functions

```c++
// Create convolution descriptor
Status_t createConvolutionDescriptor(ConvolutionDescriptor_t *convDesc);

// Destroy convolution descriptor
Status_t destroyConvolutionDescriptor(ConvolutionDescriptor_t convDesc);

// Set parameters of a 2-dimensional convolution operation with 4D inputs
Status_t setConvolution2dDescriptor(ConvolutionDescriptor_t desc, int pad_h,
        int pad_w, int stride_h, int stride_w, int dilation_h, int dilation_w,
        ConvolutionMode_t mode, DataType_t dataType);

// Set parameters of an n-dimensional convolution operations (with n-2 dimensional inputs)
Status_t setConvolutionNdDescriptor(ConvolutionDescriptor_t desc,
        int arrayLength, const int *padA, const int *filterStrideA,
        const int *dilationA, ConvolutionMode_t mode, DataType_t dataType);

// Set the number of groups to be used when performing convolution
Status_t setConvolutionGroupCount(
        ConvolutionDescriptor_t desc, int groupCount);

// Computes the dimensions of the output of an n-dimensional convolution
Status_t getConvolutionNdForwardOutputDim(
        const ConvolutionDescriptor_t convDesc,
        const TensorDescriptor_t inputTensorDesc,
        const FilterDescriptor_t filterDesc, int nbDims,
        int *tensorOutputDimA);

// Computes the dimensions of the output of an 2-dimensional convolution
Status_t getConvolution2dForwardOutputDim(
        const ConvolutionDescriptor_t convDesc,
        const TensorDescriptor_t inputTensorDesc,
        const FilterDescriptor_t filterDesc, int *n, int *c, int *h, int *w);

// Computes the size (in bytes) of a convolution forward operation's required workspace
Status_t getConvolutionForwardWorkspaceSize(Handle_t handle,
        const TensorDescriptor_t xDesc, const FilterDescriptor_t wDesc,
        const ConvolutionDescriptor_t convDesc, const TensorDescriptor_t yDesc,
        ConvolutionFwdAlgo_t algo, size_t *workSpaceSizeInBytes)

// Launch a convolution forward operation
Status_t convolutionForward(Handle_t handle, const void *alpha,
        const TensorDescriptor_t xDesc, const void *x,
        const FilterDescriptor_t wDesc, const void *w,
        const ConvolutionDescriptor_t convDesc, ConvolutionFwdAlgo_t algo,
        void *workSpace, size_t workSpaceSizeInBytes, const void *beta,
        const TensorDescriptor_t yDesc, void *y);

// Computes the size (in bytes) of a convolution backward data operation's required workspace
Status_t getConvolutionBackwardDataWorkspaceSize(Handle_t handle,
        const FilterDescriptor_t wDesc, const TensorDescriptor_t dyDesc,
        const ConvolutionDescriptor_t convDesc, const TensorDescriptor_t dxDesc,
        ConvolutionFwdAlgo_t algo, size_t *workSpaceSizeInBytes)

// Launch a convolution backward data operation
Status_t ConvolutionBackwardData(Handle_t handle, const void *alpha,
        const FilterDescriptor_t wDesc, const void *w,
        const TensorDescriptor_t dyDesc, const void *dy,
        const ConvolutionDescriptor_t convDesc, ConvolutionBwdDataAlgo_t algo,
        void *workSpace, size_t workSpaceSizeInBytes, const void *beta,
        const TensorDescriptor_t dxDesc, void *dx);

// Computes the size (in bytes) of a convolution backward weights operation's required workspace
Status_t getConvolutionBackwardFilterWorkspaceSize(Handle_t handle,
        const TensorDescriptor_t xDesc, const TensorDescriptor_t dyDesc,
        const ConvolutionDescriptor_t convDesc, const FilterDescriptor_t dwDesc,
        ConvolutionFwdAlgo_t algo, size_t *workSpaceSizeInBytes);

// Launch a convolution backward weights operation
Status_t ConvolutionBackwardFilter(Handle_t handle, const void *alpha,
        const TensorDescriptor_t xDesc, const void *x,
        const TensorDescriptor_t dyDesc, const void *dy,
        const ConvolutionDescriptor_t convDesc, ConvolutionBwdFilterAlgo_t algo,
        void *workSpace, size_t workSpaceSizeInBytes, const void *beta,
        const FilterDescriptor_t dwDesc, void *dw);

// The following APIs are used to find the best algorithm given parameters of
// a convolution operation. When used in dnncompat they will always return 
// `CONVOLUTION_FWD_ALGO_AUTO` which then gets mapped to 
// `dnnl::algorithm::convolution_auto` when launching the convolution operation.
// Thus, dnncompat, relies on oneDNN to select the best algorithm.
// This behaviour is consistent with vendor-specific libraries, which may return 
// fewer algorithms than the requested number.
Status_t findConvolutionForwardAlgorithm(Handle_t,
        const TensorDescriptor_t, const FilterDescriptor_t,
        const ConvolutionDescriptor_t, const TensorDescriptor_t, const int,
        int *returnedAlgoCount, ConvolutionFwdAlgoPerf_t *perfResults);

Status_t getConvolutionForwardAlgorithm_v7(Handle_t,
        const TensorDescriptor_t, const FilterDescriptor_t,
        const ConvolutionDescriptor_t, const TensorDescriptor_t, const int,
        int *returnedAlgoCount, ConvolutionFwdAlgoPerf_t *perfResults);

Status_t findConvolutionBwdDataAlgorithm(Handle_t,
        const TensorDescriptor_t, const FilterDescriptor_t,
        const ConvolutionDescriptor_t, const TensorDescriptor_t, const int,
        int *returnedAlgoCount, ConvolutionBwdDataAlgoPerf_t *perfResults);

Status_t findConvolutionBackwardFilterAlgorithm(Handle_t,
        const TensorDescriptor_t, const FilterDescriptor_t,
        const ConvolutionDescriptor_t, const TensorDescriptor_t, const int,
        int *returnedAlgoCount, ConvolutionBwdFilterAlgoPerf_t *perfResults);
```

List of enumerations related to the Convolution operation:

```c++
enum ConvolutionFwdAlgo_t {
    CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM = 0,
    CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,
    CONVOLUTION_FWD_ALGO_GEMM,
    CONVOLUTION_FWD_ALGO_DIRECT,
    CONVOLUTION_FWD_ALGO_FFT,
    CONVOLUTION_FWD_ALGO_FFT_TILING,
    CONVOLUTION_FWD_ALGO_WINOGRAD,
    CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED,
    CONVOLUTION_FWD_ALGO_AUTO,
    CONVOLUTION_FWD_ALGO_COUNT,
};

enum ConvolutionBwdDataAlgo_t {
    CONVOLUTION_BWD_DATA_ALGO_0,
    CONVOLUTION_BWD_DATA_ALGO_1,
    CONVOLUTION_BWD_DATA_ALGO_FFT,
    CONVOLUTION_BWD_DATA_ALGO_FFT_TILING,
    CONVOLUTION_BWD_DATA_ALGO_WINOGRAD,
    CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED,
    CONVOLUTION_BWD_DATA_ALGO_AUTO,
    CONVOLUTION_BWD_DATA_ALGO_DIRECT,
    CONVOLUTION_BWD_DATA_ALGO_COUNT,
};

enum ConvolutionBwdFilterAlgo_t {
    CONVOLUTION_BWD_FILTER_ALGO_0,
    CONVOLUTION_BWD_FILTER_ALGO_1,
    CONVOLUTION_BWD_FILTER_ALGO_FFT,
    CONVOLUTION_BWD_FILTER_ALGO_3,
    CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED,
    CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING,
    CONVOLUTION_BWD_FILTER_ALGO_AUTO,
    CONVOLUTION_BWD_FILTER_ALGO_DIRECT,
    CONVOLUTION_BWD_FILTER_ALGO_COUNT,
};

enum ConvolutionMode_t {
    CONVOLUTION = 0,
    CROSS_CORRELATION,
};

enum Determinism_t {
    NON_DETERMINISTIC = 0,
    DETERMINISTIC,
};
```

#### LRN

The `detail::LRNDescriptor` class encapsulates the parameters of the LRN
operation: N (the normalisation window), alpha (alpha variance scaling
parameter), beta (power parameter), K (k parameter of the LRN formula).

The `LRNDescriptor` will be exposed to users with the following type alias:

```c++
using LRNDescriptor_t = dnncompat::detail::LRNDescriptor *;
```

Free functions related to the LRN operation:

```c++
// Create LRN descriptor
Status_t createLRNDescriptor(LRNDescriptor_t *lrnDesc);

// Destroy LRN descriptor
Status_t destroyLRNDescriptor(LRNDescriptor_t lrnDesc);

// Set parameters of an LRN operation
Status_t setLRNDescriptor(LRNDescriptor_t normDesc, unsigned lrnN,
        double lrnAlpha, double lrnBeta, double lrnK);

// Launch LRN operation
Status_t lrnCrossChannelForward(Handle_t handle, const LRNDescriptor_t normDesc,
        LRNMode_t lrnMode, const void *alpha, const TensorDescriptor_t xDesc,
        const void *x, const void *beta, const TensorDescriptor_t yDesc,
        void *y);
```

List of enumerations related to the LRN operation:

```c++
enum LRNMode_t { LRN_CROSS_CHANNEL_DIM1 };
```

#### Pooling

The `detail::PoolingDescriptor` class encapsulates the parameters of the Pooling
operation: number of dimensions, mode (Max, Average including padding, Average
excluding padding), NaN propagation mode, Pooling window dimensions, padding,
pooling stride.

The `PoolingDescriptor` will be exposed to users with the following type alias:

```c++
using PoolingDescriptor_t = dnncompat::detail::PoolingDescriptor *;
```

Free functions related to the Pooling operation:

```c++
// Create descriptor
Status_t createPoolingDescriptor(PoolingDescriptor_t *poolingDesc);

// Destroy descriptor
Status_t destroyPoolingDescriptor(PoolingDescriptor_t poolingDesc);

// Set 2D Pooling descriptor attributes
Status_t setPooling2dDescriptor(PoolingDescriptor_t poolingDesc,
        PoolingMode_t mode, NanPropagation_t maxpoolingNanOpt, int windowWidth,
        int windowHeight, int verticalPadding, int horizontalPadding,
        int verticalStride, int horizontalStride);

// Set ND Pooling descriptor attributes
Status_t setPoolingNdDescriptor(PoolingDescriptor_t poolingDesc,
        PoolingMode_t mode, NanPropagation_t maxpoolingNanOpt, int nbDims,
        const int windowDimA[], const int paddingA[], const int strideA[]);

// Computes the dimensions of an 2-dimensional pooling operation's output tensor
Status_t getPooling2dForwardOutputDim(const PoolingDescriptor_t poolingDesc,
        const TensorDescriptor_t inputDesc, int *outN, int *outC, int *outH,
        int *outW);

// Computes the dimensions of an n-dimensional pooling operation's output tensor
Status_t getPoolingNdForwardOutputDim(const PoolingDescriptor_t poolingDesc,
        const TensorDescriptor_t inputDesc, int nbDims, int outDimA[]);

// Launch pooling operation
Status_t poolingForward(Handle_t handle, const PoolingDescriptor_t poolingDesc,
        const void *alpha, const TensorDescriptor_t xDesc, const void *x,
        const void *beta, const TensorDescriptor_t yDesc, void *y);
```

List of enumerations related to the Pooling operation:

```c++
enum PoolingMode_t {
    POOLING_MAX = 0,
    POOLING_AVERAGE_COUNT_INCLUDE_PADDING,
    POOLING_AVERAGE_COUNT_EXCLUDE_PADDING,
    POOLING_MAX_DETERMINISTIC
};
```

#### Softmax

The Softmax operation does not have a separate descriptor.

List of free functions related to the Softmax operation:

```c++
// Launch the Softmax forward operation
Status_t softmaxForward(Handle_t handle, SoftmaxAlgorithm_t algorithm,
        SoftmaxMode_t mode, const void *alpha, const TensorDescriptor_t xDesc,
        const void *x, const void *beta, const TensorDescriptor_t yDesc,
        void *y);
```

List of enumerations related to the Softmax operation:

```c++
// With mode INSTANCE Softmax is computed across dimensions C,H,W
// With mode CHANNEL Softmax is computed across dimensions C 
enum SoftmaxMode_t { SOFTMAX_MODE_INSTANCE, SOFTMAX_MODE_CHANNEL };

// Note: oneDNN does not have an explicit fast Softmax algorithm
enum SoftmaxAlgorithm_t { SOFTMAX_FAST, SOFTMAX_ACCURATE, SOFTMAX_LOG };
```

#### TransformTensor

The TransformTensor operation maps to the oneDNN Reorder primitive. It does not
have a separate descriptor.

List of free functions related to the TransformTensor operation:

```c++
// Launch the Transform tensor operator
Status_t transformTensor(Handle_t handle, const void *alpha,
        const TensorDescriptor_t xDesc, const void *x, const void *beta,
        const TensorDescriptor_t yDesc, void *y);
```

#### Matmul

Matrix multiplication is a fundamental operation in machine learning. While it
is commonly provided by BLAS libraries, a mapping for this operation will be
provided in `dnncompat`. Below is an outline of the necessary enumerations,
classes, and functions for this implementation.

##### Enumerations

```c++
// Enum for specifying transpose operations on matrices
enum class Operation_t {
    OP_N, // No transpose
    OP_T  // Transpose
};

// Enum defining matmul descriptor attributes, such as matrix transpose settings and epilogues
enum class MatmulDescAttributes_t {
    MATMUL_DESC_TRANSA,    // Transpose setting for matrix A
    MATMUL_DESC_TRANSB,    // Transpose setting for matrix B
    MATMUL_DESC_TRANSC,    // Transpose setting for matrix C
    MATMUL_DESC_EPILOGUE,  // Epilogue setting for post-ops, see Epilogue_t for supported epilogues
    MATMUL_DESC_BIAS_POINTER // Pointer which stores bias data
};

// Enum for matrix layout attributes, specifying the order and batching configurations
enum class MatrixLayoutAttribute_t {
    MATRIX_LAYOUT_ORDER,                 // Matrix layout
    MATRIX_LAYOUT_BATCH_COUNT,           // Number of batches for batched matrix operations
    MATRIX_LAYOUT_STRIDED_BATCH_OFFSET   // Stride between batches
};

// Enum for setting matrix layout formats, especially for IMMA
enum class Order_t {
    ORDER_COL,                // Column-major format
    ORDER_ROW,                // Row-major format
    ORDER_COL32,              // Special format for IMMA matrix A and C on Ampere architecture
    ORDER_COL4_4R2_8C,        // Special format for matrix B on Turing architecture
    ORDER_COL32_2R_4R4        // Special format for matrix B on Ampere architecture
};

// Enum for supported epilogues (post-ops)
enum class Epilogue_t {
    CUBLASLT_EPILOGUE_DEFAULT,      // Default epilogue
    CUBLASLT_EPILOGUE_RELU,         // ReLU activation
    CUBLASLT_EPILOGUE_BIAS,         // Bias addition
    CUBLASLT_EPILOGUE_RELU_BIAS     // ReLU activation with bias addition
};
```

##### Structures

```c++
// Descriptor for defining the computation and scaling data types for matmul operations
struct MatmulDesc_t {
    ComputeType_t computeType;   // Data type of computation (i.e. accumulation type)
    DataType_t scaleType;        // Data type for scaling factors
};

// Structure representing the result of a heuristic search for matmul algorithms
struct MatmulHeuristicResult_t {
    MatmulAlgo_t algo;           // Algorithm chosen for matmul operation (see notes for Algo_t mapping details)
    size_t workspaceSize;        // Required workspace size for the algorithm
    Status_t state;              // Status of the heuristic search
};

// Descriptor for matrix layouts, including the data type and dimensional properties
struct MatrixLayout_t {
    DataType type;               // Data type of matrix elements
    uint64_t rows;               // Number of rows in the matrix
    uint64_t cols;               // Number of columns in the matrix
    int64_t ld;                  // Leading dimension of the matrix
};

// Class representing a oneDNN matmul primitive descriptor
// See Notes for more details
struct MatmulAlgo_t {
    dnnl::matmul::desc::desc desc;   // OneDNN matmul descriptor
};
```

##### Functions

```c++
// Function to create a matrix layout descriptor
// IMMA datatypes are restricted to R_8I for matrices A and B, and R_8I/R_32I for D (R_32I only for CUDA >= 12)
Status_t matrixLayoutCreate(MatrixLayout_t *matLayout,
                            DataType_t type,
                            uint64_t rows,
                            uint64_t cols,
                            int64_t ld);

// Function to set attributes for a matrix layout
Status_t matrixLayoutSetAttribute(MatrixLayout_t matLayout,
                                  MatrixLayoutAttribute_t attr,
                                  const void *buf,
                                  size_t sizeInBytes);

// Function to execute the matrix multiplication operation
Status_t matmul(Handle_t handle,
                MatmulDesc_t computeDesc,
                const void *alpha,
                const void *A,
                MatrixLayout_t Adesc,
                const void *B,
                MatrixLayout_t Bdesc,
                const void *beta,
                const void *C,
                MatrixLayout_t Cdesc,
                void *D,
                MatrixLayout_t Ddesc,
                const MatmulAlgo_t *algo,
                void *workspace,
                size_t workspaceSizeInBytes,
                Stream_t stream);

// Function to initialize the matmul descriptor with specified computation and scale types
Status_t MatmulDescCreate(MatmulDesc_t *matmulDesc,
                          ComputeType_t computeType,
                          DataType_t scaleType);

// Function to set attributes for the matmul descriptor
Status_t MatmulDescSetAttribute(MatmulDesc_t matmulDesc,
                                MatmulDescAttributes_t attr,
                                const void* buf,
                                size_t sizeInBytes);

// Functions for matmul preferences and heuristic (cannot be mapped to oneDNN)
// A stub can be provided instead to make automated code translation simpler
Status_t MatmulPreferenceCreate(MatmulPreference_t* pref);
Status_t MatmulPreferenceSetAttribute(MatmulPreference_t pref,
                                      MatmulPreferenceAttributes_t attr,
                                      const void *buf,
                                      size_t sizeInBytes);

// Function to obtain heuristic results for viable matmul algorithms
// Note: in dnncompat this function will always return a single heuristic
Status_t MatmulAlgoGetHeuristic(Handle_t handle,
                                MatmulDesc_t operationDesc,
                                MatrixLayout_t Adesc,
                                MatrixLayout_t Bdesc,
                                MatrixLayout_t Cdesc,
                                MatrixLayout_t Ddesc,
                                MatmulPreference_t preference,
                                int requestedAlgoCount,
                                MatmulHeuristicResult_t heuristicResultsArray[],
                                int* returnAlgoCount);
```

###### Notes

- Several features such as `MatmulPreference_t`, `MatmulPreferenceCreate`,
  `MatmulPreferenceSetAttribute`, and `ReductionScheme_t` are not mappable to
  oneDNN as they are internal to the implementation and not exposed to users
- To work around the absence of direct `Algo_t` mapping, the heuristic result
  structure can be adapted. Instead of directly mapping `Algo_t`, a new struct
  containing the oneDNN primitive descriptor could be used. This allows the
  primitive to be initialized early, the required workspace size to be queried,
  and gives the user control over the memory allocation for the workspace.

## Caveats

While `dnncompat` aims to provide an API as close as possible to vendor-specific
libraries, there are some caveats:

- There are differences in the range of operators supported by oneDNN compared
  to vendor-specific libraries.

- There are also differences in feature support. Some operation features present
  in vendor-specific libraries are not available or may function differently in
  oneDNN and vice versa.

- For operations such as Activation and Pooling, cuDNN allows the option to
  disable `NaN` propagation. However, in oneDNN, `NaN` values are always
  propagated by default, which could lead to different outcomes in the execution
  of similar operations. In the initial implementation a warning will be printed
  when `NaN` propagation is explicitly disabled.

- Multiple Handle Support: cuDNN supports the creation and usage of multiple
  handles. In `dnncompat`, while multiple handles can be created, they must all
  be initialized with the same `sycl::queue` as the primary handle. This
  limitation will lead to a slight difference in usage of `dnncompat` in some
  applications.

- cuDNN includes a specialized FFT convolution implementation, while this
  algorithm is not supported in oneDNN.

### Scratchpad Memory

Some oneDNN primitives require a separate scratchpad memory buffer to
store temporary results during computations. By default, oneDNN handles the
allocation and management of this scratchpad memory internally, which generally
simplifies usage. However, in the case of `dnncompat` there's a performance
drawback due to how it creates and executes primitives. Specifically,
`dnncompat` creates the oneDNN primitives within the execution API itself. This
design means that scratchpad memory allocations occur at runtime, potentially
introducing a performance hit due to dynamic memory allocation during execution.

To address this performance concern, a more efficient approach is required for
handling scratchpad memory. One solution leverages the proposed [async memory
allocation extension](https://github.com/intel/llvm/pull/14800), which enables
scratchpad memory to be allocated asynchronously from a memory pool. The
proposed approach is outlined below:

- Upon initialization of the `dnncompat` library context, a large contiguous USM
  memory block can be pre-allocated to accommodate all the scratchpad memory
  needs during execution. This buffer will reduce the need for repeated dynamic
  allocations.
- Additionally, when the library context is created, the asynchronous memory
  pool can be initialized with the `use_existing_memory` property. This property
  allows the memory pool to reference the large, pre-allocated memory block
  mentioned above, ensuring that the pool uses the same contiguous memory rather
  than performing separate allocations.
- `dnncompat` will set the `dnnl::scratchpad_mode::user` mode for each executed
  primitive, which gives `dnncompat` explicit control over the allocation and
  management of scratchpad memory. In this mode, `dnncompat` will be responsible
  for allocating the scratchpad buffer and ensuring it is passed to the oneDNN
  primitives when required.
- Since the handle is passed to all execution APIs within `dnncompat`, the
  asynchronous memory pool can be reused across different primitives. When a
  primitive is executed, `dnncompat` will first query the required scratchpad
  size, allocate it from the pre-allocated pool, and then pass it to the
  primitive. This approach minimizes the overhead of scratchpad memory
  allocation and ensures that memory is handled efficiently.

### Workspace Memory

In oneDNN, certain primitives require additional workspace memory during the
training phase. This workspace serves as an extra buffer passed to the forward
training primitive, where it stores information that will be needed
for the backward pass. The same workspace is then passed to the corresponding
backward primitive as it is required for the computation. However, this
approach differs from vendor-specific libraries like cuDNN, which do not rely on
workspace memory in this way. To work around this, the following approach is
proposed to manage the API differences between oneDNN and cuDNN and avoid adding
extra parameters to the `dnncompat` APIs:

- A map or dictionary structure can be created where the key is a tensor object
  passed to both the forward and backward primitives (e.g. destination tensor),
  and the associated value is the corresponding workspace memory.
  - This map can be stored within the library handle which is passed to all
    primitive execution APIs
- During the forward pass, the `dnncompat` implementation will query the primitive
  to determine its workspace requirements. Once identified, the necessary
  workspace memory will be allocated from a memory pool (as described in the
  Scratchpad Memory subsection). The key-value pair, consisting of the tensor
  and the allocated workspace, is then stored in the map.
- When executing the backward pass, the handle, along with the associated map,
  is passed to retrieve the previously stored workspace memory. The retrieved
  memory is then passed to the backward oneDNN primitive to complete the
  computation.

While this method effectively addresses the API mismatch between oneDNN and
cuDNN, there is a potential issue: if the user copies or modifies the pointer
associated with the tensor between the forward and backward passes, the map
lookup will fail, as the modified pointer will no longer match the original key.
Therefore, this limitation should be clearly documented.

## Unit Testing

To ensure the correctness of dnncompat, unit tests will be implemented using the
`googletest`-based framework in oneDNN. The tests will run each operation with
different combinations of parameters and will compare the results with results
of oneDNN operations.

## Appendix

### Example usage of `dnncompat`

Below is an example of launching a Convolution forward operation:

```c++
using DataType = float;

dnncompat::Handle_t handle;
dnncompat::Create(&handle);

const dnncompat::DataType_t data_type = dnncompat::DataType_t::DATA_FLOAT;
const dnncompat::TensorFormat_t tensor_format
        = dnncompat::TensorFormat_t::TENSOR_NCHW;
const dnncompat::ConvolutionFwdAlgo_t algo
        = dnncompat::ConvolutionFwdAlgo_t::CONVOLUTION_FWD_ALGO_DIRECT;
const float alpha = 1.f;
const float beta = 0.f;

// input
const int in_n = 1;
const int in_c = 1;
const int in_h = 2;
const int in_w = 2;

dnncompat::TensorDescriptor_t in_desc;
dnncompat::createTensorDescriptor(&in_desc);
dnncompat::setTensor4dDescriptor(
        in_desc, tensor_format, data_type, in_n, in_c, in_h, in_w);

// filter
const int filt_k = 1;
const int filt_c = 1;
const int filt_h = 1;
const int filt_w = 1;

dnncompat::FilterDescriptor_t filt_desc;
dnncompat::createFilterDescriptor(&filt_desc);
dnncompat::setFilter4dDescriptor(filt_desc, data_type, tensor_format,
        filt_k, filt_c, filt_h, filt_w);

// convolution
const int pad_h = 0;
const int pad_w = 0;
const int str_h = 1;
const int str_w = 1;
const int dil_h = 1;
const int dil_w = 1;

dnncompat::ConvolutionDescriptor_t conv_desc;
dnncompat::createConvolutionDescriptor(&conv_desc);
dnncompat::setConvolution2dDescriptor(conv_desc, pad_h, pad_w, str_h, str_w,
        dil_h, dil_w, dnncompat::ConvolutionMode_t::CROSS_CORRELATION,
        dnncompat::DataType_t::DATA_FLOAT);

// output
int out_n = 0;
int out_c = 0;
int out_h = 0;
int out_w = 0;

dnncompat::getConvolution2dForwardOutputDim(
        conv_desc, in_desc, filt_desc, &out_n, &out_c, &out_h, &out_w);

dnncompat::TensorDescriptor_t out_desc;
dnncompat::createTensorDescriptor(&out_desc);
dnncompat::setTensor4dDescriptor(
        out_desc, tensor_format, data_type, out_n, out_c, out_h, out_w);

const int workspace_size = 0;
DataType *workspace_ptr = nullptr;

DataType *in_ptr = sycl::malloc_device<DataType>(
        in_n * in_c * in_h * in_w, handle->getQueue());
std::vector<DataType> host_in = {1, 2, 3, 4};
handle->getQueue().memcpy(
        in_ptr, host_in.data(), host_in.size() * sizeof(DataType));
DataType *filt_ptr = sycl::malloc_device<DataType>(
        filt_k * filt_c * filt_h * filt_w, handle->getQueue());
std::vector<DataType> host_filt = {5};
handle->getQueue().memcpy(
        filt_ptr, host_filt.data(), host_filt.size() * sizeof(DataType));
DataType *out_ptr = sycl::malloc_device<DataType>(
        out_n * out_c * out_h * out_w, handle->getQueue());
std::vector<DataType> host_out(4);

handle->getQueue().wait_and_throw();

auto status = dnncompat::convolutionForward(handle, &alpha, in_desc, in_ptr,
        filt_desc, filt_ptr, conv_desc, algo, workspace_ptr, workspace_size,
        &beta, out_desc, out_ptr);

if (status != dnncompat::Status_t::STATUS_SUCCESS) {
        std::cout << "Error occurred  during convolution operation.\n";
        return 1;
}

handle->wait();
```
