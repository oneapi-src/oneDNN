# Performance Optimizations

## Intel Architecture Processors
  * Improved fp16/bf16 softmax performance with relaxed [accumulation mode](https://oneapi-src.github.io/oneDNN/dev_guide_attributes_accumulation_mode.html#doxid-dev-guide-attributes-accumulation-mode).
  * Improved performance for int8 RNN primitive on processors with Intel AVX2 and Intel AVX512 instruction set support.
  * Improved performance of convolution and matmul primitives on processors with Intel AMX support.
  * Improved performance of fp8 matmul primitives with bf16 and fp16 bias data type on processors with Intel AMX instruction set support.
  * Improved performance of int8 matmul primitive with fp16 output data type.
  * Improved performance of int8 depthwise separable convolution primitive with pre-channel zero points on processors with Intel AVX2 and Intel AVX512 instruction set support.

## Intel Graphics Products
  * Introduced initial optimizations for Intel GPUs based on Xe3 architecture.
  * Improved performance for Intel Arc Graphics for Intel Core Ultra processors (Series 2) (formerly Lunar Lake) and Intel Arc B-series discrete graphics (formerly Battlemage).
  * Improved performance of the following subgraphs with Graph API:
    * Scaled Dot-Product Attention (SDPA) [with implicit causal mask](https://oneapi-src.github.io/oneDNN/dev_guide_graph_sdpa.html#doxid-dev-guide-graph-sdpa).
    * Scaled Dot-Product Attention (SDPA) [with int8/int4 compressed key and value](https://oneapi-src.github.io/oneDNN/dev_guide_graph_sdpa_compressed_kv.html#doxid-dev-guide-graph-sdpa-compressed-kv).
  * Improved performance of convolution with source zero points by pre-packing compenstation.
  * Improved performance of backward by data convolution with strides for large filter.

## AArch64-based Processors

# Functionality

## Common
  * Introduced support for `select` algorithm in binary primitive. The functionality is optimized for Intel CPUs.
  * Extended quantization support in matmul and reorder with grouped scales and zero-points for weights. This functionality is optimized for Intel CPUs and GPUs.
  * Introduced initial support for 4-bit floating-point data types `f4_e2m1` and `f4_e3m0` in matmul and reorder, as well as `e8m0` scales data type in matmul and reorder. This functionality is available on Intel CPUs and GPUs.
  * Introduced [`Select`](https://oneapi-src.github.io/oneDNN/dev_guide_op_select.html), [`GenIndex`](https://oneapi-src.github.io/oneDNN/dev_guide_op_genindex.html) and [`GreaterEqual`](https://oneapi-src.github.io/oneDNN/dev_guide_op_greaterequal.html) operations in Graph API.
  * Graph API: Added support for quantized and non-quantized Gated MLP patterns.

## Intel Architecture Processors
  * Introduced support for fp16/bf16 compressed weights in fp32 matmul.

## Intel Graphics Products
  * Introduced stochastic rounding support for convolution, matmul and reorder based on Philox counter-based random number generator.
  * Added support for strided memory formats in convolution.

# Usability
  * With the SYCL runtime, memory objects on the CPU engine are now reference-counted and no longer need to be explicitly kept alive for the duration of the primitive execution. This aligns memory object lifetime behavior on CPU and GPU engines.
  * Improved support of large size tensors in convolution, matmul and reduction primitives on Intel GPUs.
  * Reduced scratchpad usage for NCHW convolution on Intel GPUs.
  * Improved verbose diagnostics to better identify issues during dispatching, primitive and kernel creation for CPU primitive and GPU (in case of OpenCL implementation) primitive implementations.
  * Improved verbose diagnostics for Intel GPU driver compatibility issues.
  * Enabled frame pointers support on Intel64 platforms to improve integration with profilers.
  * Added [examples](https://github.com/oneapi-src/oneDNN/tree/main/examples/graph) for Gated MLP and int4 Gated MLP

# Validation
  * Extended benchdnn with support and validation for fp8 matmul patterns for tensor tags in RNN primitive validation.
  * Extended benchdnn with support for rewriting data types in the test JSON files in the graph driver.
  * Extended benchdnn with support and validation for the number of partitions returned from the test JSON files.

# Deprecated Functionality

# Breaking Changes
  * Updated minimal supported CMake version to 3.13 (was 2.8.12).
  * Updated minimal supported GCC version to 8.0 (was 4.8).
  * Updated minimal supported Clang version to 11.0 (was 3.0).
  * Removed support for SYCL older than 2020.
  * Enforced `fp32` accumulation mode in `fp16` matmul and inner product primitives on Intel Graphics products without Intel XMX cores. Previous behavir can be enabled with relaxed [accumulation mode](https://oneapi-src.github.io/oneDNN/dev_guide_attributes_accumulation_mode.html#doxid-dev-guide-attributes-accumulation-mode).

# Thanks to these Contributors

This release contains contributions from the [project core team] as well as Aditya Tewari @aditew01, Alexandra Sidorova @a-sidorova, Atharva Dubey @AD2605, Deb Taylor @deb-intel, Dmitriy Ovchinnikov @inteldimitrius, Fadi Arafeh @fadara01, Hengyu Meng @airMeng, @hmaciak, John Osorio @kala855, Marek Michalowski @michalowski-arm, Michael Froelich @MichaelFroelich, Michał Górny @mgorny, Nikhil Sharma @nikhilfujitsu, Permanence AI Coder @Permanence-AI-Coder, @raistefintel, Ravi Pushkar @rpushkarr, Renato Barros Arantes @renato-arantes, Romain Biessy @Rbiessy, Ryo Suzuki @Ryo-not-rio, @Shreyas-fuj, Varad Ahirwadkar @varad-ahirwadkar, @vishwascm, and Ye Tao @taoye9. We would also like to thank everyone who asked questions and reported issues.

[project core team]: https://github.com/oneapi-src/oneDNN/blob/rls-v3.7/MAINTAINERS.md
