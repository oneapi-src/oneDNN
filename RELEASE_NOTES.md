# Performance Optimizations

## Intel Architecture Processors
  * Improved performance of convolution and matmul primitives on Intel Xeon processors with Intel AMX instruction set support (formerly Sapphire Rapids and Granite Rapids).
  * Improved performance of `int8` and `fp32` forward convolution primitive on processors with Intel AVX2 instruction set support.
  * Improved performance of `fp8` matmul primitives with `bf16` and `fp16` bias data type on Intel Xeon processors with Intel AMX instruction set support (formerly Sapphire Rapids and Granite Rapids).
  * Improved performance of `int8` RNN primitive on processors with Intel AVX2 and Intel AVX-512 instruction set support.
  * Improved performance of `int8` depthwise separable convolution primitive with per-channel zero points on processors with Intel AVX2 and Intel AVX-512 instruction set support.
  * Improved `fp16` and `bf16` softmax performance with relaxed [accumulation mode].
  * Improved performance of `int8` matmul primitive with `fp16` output data type.
  * Improved performance of the following subgraphs with Graph API:
    * [Gated Multi-Layer Perceptron (Gated MLP)].

[accumulation mode]: https://oneapi-src.github.io/oneDNN/v3.7/dev_guide_attributes_accumulation_mode.html#doxid-dev-guide-attributes-accumulation-mode

## Intel Graphics Products
  * Introduced initial optimizations for Intel GPUs based on Xe3 architecture.
  * Improved performance for Intel Arc Graphics for Intel Core Ultra processors (Series 2) (formerly Lunar Lake) and Intel Arc B-series discrete graphics (formerly Battlemage).
  * Improved performance of convolution with source zero points by pre-packing compenstation.
  * Improved performance of backward by data convolution with strides for large filter.
  * Improved performance of the following subgraphs with Graph API:
    * Scaled Dot-Product Attention (SDPA) with [implicit causal mask].
    * SDPA with [`int8` or `int4` compressed key and value].
    * Gated MLP.

[implicit causal mask]: https://oneapi-src.github.io/oneDNN/v3.7/dev_guide_graph_sdpa.html#doxid-dev-guide-graph-sdpa
[`int8` or `int4` compressed key and value]: https://oneapi-src.github.io/oneDNN/v3.7/dev_guide_graph_sdpa_compressed_kv.html#doxid-dev-guide-graph-sdpa-compressed-kv
[Gated Multi-Layer Perceptron (Gated MLP)]: https://oneapi-src.github.io/oneDNN/v3.7/dev_guide_graph_gated_mlp.html#doxid-dev-guide-graph-gated-mlp

## AArch64-based Processors
  * Improved `bf16` matmul performance with `fp32` destination with Arm Compute Library (ACL).
  * Improved `bf16` to `fp32` reorder performance.
  * Improved `bf16` reorder performance.
  * Improved `bf16` convolution with ACL.

## NVIDIA GPUs
  * Improved matmul performance using cuBLASLt-based implementation.

# Functionality

## Common
  * Introduced support for `select` algorithm in binary primitive. The functionality is optimized for Intel CPUs.
  * Extended quantization support in matmul and reorder with grouped scales and zero-points for weights. This functionality is optimized for Intel CPUs and GPUs.
  * Introduced initial support for 4-bit floating-point data types `f4_e2m1` and `f4_e3m0` in matmul and reorder, as well as `e8m0` scales data type in matmul and reorder. This functionality is available on Intel CPUs and GPUs.
  * Introduced [`GenIndex`], and [`GreaterEqual`] operations in Graph API.

[`GenIndex`]: https://oneapi-src.github.io/oneDNN/v3.7/dev_guide_op_genindex.html
[`GreaterEqual`]: https://oneapi-src.github.io/oneDNN/v3.7/dev_guide_op_greaterequal.html

## Intel Architecture Processors
  * Introduced support for `fp32` matmul with `fp16` and `bf16` weights.

## Intel Graphics Products
  * Introduced stochastic rounding support for convolution, matmul and reorder based on Philox counter-based random number generator.
  * Introduced support for strided memory formats in convolution.

## Generic GPU vendor
  * Introduced support for reduction primitive.
  * Introduced support for inner product primitive forward propagation.

# Usability

## Common
  * With the SYCL runtime, memory objects on the CPU engine are now reference-counted and no longer need to be explicitly kept alive for the duration of the primitive execution. This aligns memory object lifetime behavior on CPU and GPU engines.
  * Added Graph API examples for [Gated MLP] and [`int4` Gated MLP] patterns.

[Gated MLP]: https://github.com/oneapi-src/oneDNN/blob/rls-v3.7/examples/graph/gated_mlp.cpp
[`int4` Gated MLP]: https://github.com/oneapi-src/oneDNN/blob/rls-v3.7/examples/graph/gated_mlp_int4.cpp

## Intel Architecture Processors
  * Improved verbose diagnostics to better identify issues during dispatching, primitive and kernel creation for Intel CPU and Intel GPU implementations.
  * Enabled frame pointers support on Intel64 platforms to improve integration with profilers.

## Intel Processor Graphics
  * Improved verbose diagnostics for Intel GPU driver compatibility issues.
  * Improved support of large size tensors in convolution, matmul and reduction primitives on Intel GPUs.
  * Reduced scratchpad usage for NCHW convolution on Intel GPUs.

## AArch64-based Processors
  * Added support for the Arm Compute Library (ACL) thread_local scheduler via ThreadpoolScheduler.
  * Improved memory efficiency in ACL matmuls by fixing a bug where scratchpad memory was not being used.
  * Made the ACL matmul primitive thread-safe which allows concurrent execution.

# Validation
  * Extended benchdnn with support and validation for fp8 matmul patterns for tensor tags in RNN primitive validation.
  * Extended benchdnn with support for rewriting data types in the test JSON files in the graph driver.
  * Extended benchdnn with support and validation for the number of partitions returned from the test JSON files.

# Deprecated Functionality
  * Experimental [Graph Compiler] is deprecated and will be removed in future releases.

[Graph Compiler]: https://oneapi-src.github.io/oneDNN/v3.7/dev_guide_graph_compiler.html

# Breaking Changes
  * Updated minimal supported CMake version to 3.13 (was 2.8.12).
  * Updated minimal supported GCC version to 8.0 (was 4.8).
  * Updated minimal supported Clang version to 11.0 (was 3.0).
  * Updated minimal supported ACL version to 24.11.1 (was 24.09).
  * Removed support for SYCL standards preceding SYCL 2020.
  * Enforced `fp32` accumulation mode in `fp16` matmul and inner product primitives on Intel Graphics products without Intel XMX cores. Previous behavir can be enabled with relaxed [accumulation mode].

# Thanks to our Contributors

This release contains contributions from the [project core team] as well as Aditya Tewari @aditew01, Alexandra Sidorova @a-sidorova, Atharva Dubey @AD2605, Deb Taylor @deb-intel, Dmitriy Ovchinnikov @inteldimitrius, Fadi Arafeh @fadara01, Hengyu Meng @airMeng, @hmaciak, John Karasev @karasjoh000, John Osorio @kala855, Keola Wierschem @kwiersch, Marek Michalowski @michalowski-arm, Michael Froelich @MichaelFroelich, Michał Górny @mgorny, Nicolò Scipione @s-Nick, Nikhil Sharma @nikhilfujitsu, Permanence AI Coder @Permanence-AI-Coder, @raistefintel, Ravi Pushkar @rpushkarr, Renato Barros Arantes @renato-arantes, Romain Biessy @Rbiessy, Ryo Suzuki @Ryo-not-rio, @Shreyas-fuj, Tadej Ciglarič @t4c1, Varad Ahirwadkar @varad-ahirwadkar, Viktoriia Gvozdeva @vgvozdeva, @vishwascm, @yair-obodovsky, Ye Tao @taoye9. We would also like to thank everyone who asked questions and reported issues.

[project core team]: https://github.com/oneapi-src/oneDNN/blob/rls-v3.7/MAINTAINERS.md
