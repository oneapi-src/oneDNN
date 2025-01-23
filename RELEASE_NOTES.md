# Performance Optimizations
## Intel Architecture Processors
  * Improved fp16/bf16 softmax performance with relaxed [accumulation mode](https://oneapi-src.github.io/oneDNN/dev_guide_attributes_accumulation_mode.html#doxid-dev-guide-attributes-accumulation-mode).
  * Added support and improved perfomance for fp8 matmul with bf16/fp16.

## Intel Graphics Products
  * Introduced initial optimizations for GPUs based on Xe3 architecture.
  * Improved performance for convolution for Intel Arc Graphics for Intel Core Ultra processors (Series 2) (formerly Lunar Lake).
  * Improved performance of the following subgraphs with Graph API
    * Scaled dot-product Attention (SDPA) [with implicit causal mask](https://oneapi-src.github.io/oneDNN/dev_guide_graph_sdpa.html#doxid-dev-guide-graph-sdpa)
    * Scaled dot-product Attention (SDPA) [with compressed key and value](https://oneapi-src.github.io/oneDNN/dev_guide_graph_sdpa_compressed_kv.html#doxid-dev-guide-graph-sdpa-compressed-kv)
## AArch64-based Processors

# Functionality
  * Introduced support for `select` algorithm in binary primitive. The functionality is optimized for Intel CPUs.
  * Enabled support for matmul primitive with grouped quantization on weight along N dimension
  * Graph API: new [`select`](https://oneapi-src.github.io/oneDNN/dev_guide_op_select.html), [`GenIndex`](https://oneapi-src.github.io/oneDNN/dev_guide_op_genindex.html) and [`GreaterEqual`](https://oneapi-src.github.io/oneDNN/dev_guide_op_greaterequal.html) operations.
  * Introduced support for fp16/bf16 compressed weights in fp32 matmul on Intel CPUs.
  * Introduced support for grouped scales and zero points in reorder primitive.
  * Enabled support for 4d weight scale in matmul primitive.
  * Graph API: added support for Quantized and non-quantized Gated MLP pattern
  * Introduced preliminary support for 4-bit floating-point data types `f4_e2m1` and `f4_e3m0` in matmul and reorder, as well as `e8m0` scales data type in matmul and reorder.
  * [experimental] Extended microkernel API:
		Introduced int4 quantization support.
		Fpmath mode API
# Usability
  * With SYCL runtime, memory objects on CPU engine are now reference-counted and no more need to be explicitly kept alive by user for the duration of the primitive execution. This align memory object lifetime behavior on CPU and GPU engines.
  * Improve verbose diagnostic to better identify issues during dispatching, primitive and kernel creation for CPU primitive and GPU (in case of OpenCL implementation) primitive implementations.
  * Improve verbose diagnostic to simplify debugging of nGEN fallbacks.
  * Enabled frame pointers support on Intel64 platforms to improve integration with profilers.
  * Added [examples](https://github.com/oneapi-src/oneDNN/tree/main/examples/graph) for Gated MLP and int4 Gated MLP
# Validation
  * Extended benchdnn with support and validation for fp8 matmul patterns for tensor tags in RNN primitive validation.
# Deprecated Functionality

# Breaking Changes
  * Updated minimal supported CMake version to 3.13 (was 2.8.12).
  * Updated minimal supported GCC version to 8.0 (was 4.8).
  * Updated minimal supported Clang version to 11.0 (was 3.0).
  * Removed support for SYCL older than 2020
# Thanks to these Contributors

This release contains contributions from the [project core team] as well as Michał Górny @mgorny, Fadi Arafeh @fadara01, John Osorio @kala855, Ravi Pushkar @rpushkarr, Marek Michalowski @michalowski-arm, Renato Barros Arantes @renato-arantes, Ryo Suzuki @Ryo-not-rio, Varad Ahirwadkar @varad-ahirwadkar, Tadej Ciglarič @t4c1, Nikhil Sharma @nikhilfujitsu, @taoye9, @Shreyas-fuj, @raistefintel. We would also like to thank everyone who asked questions and reported issues.

[project core team]: https://github.com/oneapi-src/oneDNN/blob/rls-v3.7/MAINTAINERS.md
