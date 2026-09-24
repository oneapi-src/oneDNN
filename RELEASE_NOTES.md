# Performance Optimizations

## Intel 64/AMD64 Processors
* Introduced initial support for AI Compute Extensions (ACE) instructions. This functionality is not dispatched by default and requires opt-in with environment variable `ONEDNN_MAX_CPU_ISA=AVX10_2_ACE`.
* Improved performance on future Intel Xeon processors with Intel AVX10.2 and Intel AMX instruction set support (codename Diamond Rapids).
* Improved performance of `f8` matmul with block-wise weights on future Intel Xeon processors with Intel AVX10.2 and Intel AMX instruction set support (codename Diamond Rapids).
* Improved performance of `bf16` and `f16` matmul with relaxed accumulation mode on Intel Xeon processors with Intel AMX instruction set support.
* Improved performance of strided deconvolution on Intel Xeon Scalable processors (formerly Sapphire Rapids).
* Improved performance of floating-point [Scaled Dot Product Attention (SDPA)] training forward propagation subgraph with Graph API.
* Improved performance of matmul and inner product primitives on AMD Zen architecture-based processors with ZenDNN. This optimization requires opt-in with [`ONEDNN_X64_USE_ZEN=ON`].
* Improved performance of convolution primitive on Intel Xeon processors with Intel AMX instruction set support.
* Improved performance of `f32` matmul and RNN primitives on x64 processors with Intel AVX2 instruction set support.
* Improved performance of binary post-op on x64 processors.

[Scaled Dot Product Attention (SDPA)]: https://uxlfoundation.github.io/oneDNN/v3.14/dev_guide_graph_sdpa.html
[`ONEDNN_X64_USE_ZEN=ON`]: https://uxlfoundation.github.io/oneDNN/v3.14/dev_guide_build_options.html#onednn-x64-use-zen

## Intel Graphics
* Improved performance on Intel GPU D-Series (formerly Crescent Island).
* Improved performance on future integrated GPUs based on Xe3p-LPG architecture (codename Nova Lake P).
* Improved performance of Scaled Dot Product Attention (SDPA) training forward and backpropagation subgraphs.
* Improved performance of Scaled Dot Product Attention (SDPA) subgraphs with asymmetric head sizes.
* Improved performance of grouped matmul with small group sizes.
* Improved performance of `f8` matmul with source and weights scales mask `3`.
* Improved performance of `f16` and `s8` matmul with `u4` weights and `N=1`.
* Reduced convolution and deconvolution primitives creation time.

## AArch64 Processors
* Improved performance of `u8` and `s8` matmul on platforms with SVE support.
* Improved performance of `f32` matmul on platforms with 128-bit SVE vector lengths.
* Improved performance of `f32` depthwise convolution on platforms with ASIMD support.
* Improved performance of convolutions.
* Improved performance of `eltwise_log` on platforms with ASIMD support.
* Improved performance of `bf16` eltwise for the `gelu_erf`, `swish`, `gelu_tanh`, `exp`, `log`, and `sqrt` algorithms.
* Improved performance of `f32` and `f16` PReLU.
* Improved performance of eltwise post-ops.
* Improved performance of the `logsoftmax` algorithm for the softmax primitive on platforms with ASIMD support.
* Reduced penalties on small utility functions on clang builds by changing the default stack-protection level from `all` to `strong`.

## RISC-V Processors
* Improved performance of `f32` binary, eltwise, pooling, softmax, and logsoftmax on processors with `V` extension support.
* Improved performance of `f16` matmul, eltwise, and softmax on processors with `Zvfh` extension support, including softmax with non-contiguous axes.
* Extended RVV-optimized implementations to `bf16` binary, eltwise, pooling, softmax, batch normalization, and resampling, and improved `bf16` matmul performance on processors with `Zvfbfwma` extension support.
* Introduced RVV-optimized forward resampling with nearest-neighbor and linear interpolation for `f32` and `f16` data types.
* Introduced RVV-optimized shuffle for `f32`, `s32`, `f16`, and `bf16` data types.
* Extended the RVV matmul implementation to support unsigned 8-bit source and weights.

# Functionality

## Functional API
* Introduced the [`binary_mul_inplace`] algorithm for binary post-ops. Unlike `binary_mul`, the new algorithm allows the matmul destination tensor to be used as one of its inputs. An optimized implementation is available for matmul on Intel GPUs.
* **[experimental]** Extended eltwise post-ops support in grouped matmul with all supported algorithms. Optimized implementation is available on Intel GPUs.
* **[experimental]** Extended grouped matmul with support for backpropagation cases (2D grouped by 3D dense and 2D grouped by 2D grouped) covering `f32`, `f16` and `bf16` data types. Optimized implementation is available for Intel GPUs. This is an experimental feature that requires opt-in with [`ONEDNN_EXPERIMENTAL_GROUPED_MEMORY=ON`] build option.

[`ONEDNN_EXPERIMENTAL_GROUPED_MEMORY=ON`]: https://uxlfoundation.github.io/oneDNN/v3.14/dev_guide_experimental.html#onednn-experimental-grouped-memory

[`binary_mul_inplace`]: https://uxlfoundation.github.io/oneDNN/v3.14/dev_guide_attributes_post_ops.html#in-place-binary-post-ops

## Graph API
* Extended [DynamicQuantize] and [DynamicDequantize] operations to support the new `mask` attribute.

[DynamicQuantize]: https://uxlfoundation.github.io/oneDNN/v3.14/dev_guide_op_dynamicquantize.html
[DynamicDequantize]: https://uxlfoundation.github.io/oneDNN/v3.14/dev_guide_op_dynamicdequantize.html

# Usability

## Common
* Updated MXFP downconversion implementations to saturate instead of overflowing. New behavior is consistent with OCP MX specification and aligned with preferred behavior in PyTorch.
* Version number can now be used as a passable approximation of pi.

## Intel 64/AMD64 processors
* Cleaned up implicit narrowing conversions and removed suppression of MSVC compiler warning C4244. 

## Intel Graphics
* Reduced overhead of verbose profiling for Level Zero runtime by removing synchronizations.
* Introduced support for concurrent primitive execution with the Level Zero runtime on Intel GPUs.
* **[experimental]** Introduced support for verbose profiling based on `sycl_ext_oneapi_profiling_tag` SYCL extension. This is an experimental feature that requires opt-in with [`ONEDNN_EXPERIMENTAL_ENABLE_SYCL_PROFILING_TAG=ON`] build option.

[`ONEDNN_EXPERIMENTAL_ENABLE_SYCL_PROFILING_TAG=ON`]: https://uxlfoundation.github.io/oneDNN/v3.14/dev_guide_experimental.html#onednn-experimental-enable-sycl-profiling-tag

## AArch64 Processors
* Introduced initial asynchronous runtime support to AArch64 platforms for the matmul, convolution, eltwise, binary, lnorm, and reorder primitives.
* Fixed a memory leak in convolutions on platforms with SVE support.

# Validation
* Updated benchdnn `smoke` and `CI` test sets for matmul using parameter space sampling approach.
* **[experimental]** Extended benchdnn `--grouped` knob with `balanced`, `hot`, and `decode` strategies for offset generation to generate MoE-style group distributions in grouped matmul validation.
* Extended [benchdnn graph driver] with support for removing operation attributes (`--op-attrs`), scalar tensors (`--in-shapes`), tensor property rewriting (`--tensor-property`), and operation removal (`--op-kind`).

[benchdnn graph driver]: https://github.com/uxlfoundation/oneDNN/blob/rls-v3.14/tests/benchdnn/doc/driver_graph.md
# Deprecated Functionality
* The [BLAS-like API], including `dnnl::sgemm`, `dnnl::gemm_u8s8s32`, and `dnnl::gemm_s8s8s32` functions, is deprecated and will be removed in future releases. If you are using this API consider switching to [matmul primitive].
* `f4_e3m0` data type is deprecated and will be removed in future releases.

[BLAS-like API]: https://uxlfoundation.github.io/oneDNN/v3.14/group_dnnl_api_blas.html
[matmul primitive]: https://uxlfoundation.github.io/oneDNN/v3.14/dev_guide_matmul.html

# Breaking Changes
* Removed optimizations for Intel Iris Xe MAX Graphics and Intel Graphics included with 11th-14th generation Intel Core processors. oneDNN remains functional on these platforms and dispatches a generic OpenCL implementation.
* Removed optimizations for processors with Intel SSE4.1 and Intel AVX instruction sets. oneDNN remains functional on these platforms and dispatches a generic C++ implementation.
* Removed optimizations for `tf32` `fpmath_mode` in matmul and convolution on future Intel Xeon processors with Intel AVX10.2 and Intel AMX instruction set support (codename Diamond Rapids).

# Thanks to our Contributors
This release contains contributions from the [project core team] as well as Abhishek Kumar @abhishek-iitmadras, Aditya Singh @adityasingh2400, Akihiro Tabuchi @Akihiro-Tabuchi, @AragornOfKebroyd, Aron Xu @happyaron, Ayush-SinghBaiswar @AyushSinghBaiswar, Codrut Irimie @CodrutIrimieARM, elimor01 @MorelElian, Emilio Cota @cota, Gassan Salama @gassan-arm, Ishita Shreya @ishita-shreya, Kamil Jackiewicz @kjackiew, Kamil Wieloch @kwieloch-intel, Keerthana KT @Keerthana-64, Léandre LE DUC @leduclean, Leon Kennedy @leoken01, Megha Sangtani @megha-sangtani, Mikhail Dvoretckii @mdvoretc-intel, Mohammed Bilgrami @mohbil01, Naveen Kumar @naveenthangudu, Nikhil Gupta @nikhil-arm, @PiotrReiterIntel, Pranav Singh @Pranavsingh431, Puneet Matharu @puneetmatharu, rinatr @rinatrap, Thiago Macieira @thiagomacieira, @Tiwari-Avanish, Udit Kumar Agarwal @uditagarwal97, @velonica0, Wang hongyan @ww8191201-coder, @xinghai-zh, and Yin Tong @Ag-Cu.

Each of them contributed an invaluable slice of the pi.

[project core team]: https://github.com/uxlfoundation/oneDNN/blob/rls-v3.14/MAINTAINERS.md
