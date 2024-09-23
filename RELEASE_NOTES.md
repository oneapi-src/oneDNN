oneDNN v3.6 Release Notes
=========================

# Performance Optimizations

## Intel Architecture Processors

  * Improved performance for 4th generation Intel Xeon Scalable processors
  (formerly Sapphire Rapids).
  * Improved performance for Intel Xeon 6 processors (formerly Granite Rapids).
  * Improved performance of group normalization primitive.
  * Improved `bf16` matmul performance with `int4` compressed weights on processors
  with Intel AMX instruction set support.
  * Improved performance of `fp8` matmul, pooling, and eltwise primitives on
  processors with Intel AMX instruction set support.
  * Improved `fp32` RNN primitive performance on processors with Intel AVX2
  instruction set support.
  * Improved performance of the following subgraphs with Graph API:
    - `convolution` and `binary` operation fusions with better layout selection
    in Graph API.
    - `fp8` `convolution` and `unary` or `binary` on processors with Intel AMX
    instruction set support.
    - Scaled Dot Product Attention (SDPA) without scale,
    Multi-Query Attention (MQA), and Grouped Query Attention (GQA) patterns.
    - `LayerNorm`, `GroupNorm`, and `SoftMax` with `int8` quantized output
    and zero-points.

## Intel Graphics Products

  * Improved performance for the Intel Data Center GPU Max Series (formerly
  Ponte Vecchio).
  * Introduced broad production quality optimizations for Intel Arc Graphics for
  Intel Core Ultra Processors (Series 2) (formerly Lunar Lake).
  * Introduced broad production quality optimizations for future discrete GPU
  based on Xe2 architecture (code name Battlemage).
  * Introduced support for Intel Arc Graphics for future Intel Core Ultra
  Processor (code name Arrow Lake-H).
  * Improved performance of `fp8_e5m2` primitives on Intel Data Center GPU Max
  Series (formerly Ponte Vecchio).
  * Improved matmul and inner product primitives performance for shapes relevant
  to large language models (LLMs) on GPUs with Intel XMX support.
  * Improved `int8` convolution performance with weight zero-points.
  * Reduced primitive creation time for softmax, layer normalization, and concat
  primitives via kernel reuse.
  * Improved performance of the following subgraphs with Graph API:
    - SDPA without scale, MQA, and GQA patterns. `f16` variants of these
    patterns significantly benefit from Intel(R) Xe Matrix Extensions (Intel(R)
    XMX) support.
    - `fp8`, `convolution`, and `unary` or `binary` on the Intel Data Center GPU Max
    Series.
    - `LayerNorm`, `GroupNorm`, and `SoftMax` with `int8` quantized output and
    zero-points.

## AArch64-based Processors

  * Improved `fp32` convolution backpropagation performance on processors with
  SVE support.
  * Improved reorder performance for blocked format on processors with
  SVE support.
  * Improved `bf16` softmax performance on processors with SVE support.
  * Improved batch normalization performance on processors with SVE support.
  * Improved matmul performance on processors with SVE support.
  * Improved `fp16` convolution with Arm Compute Library (ACL).
  * Improved matmul performance with ACL.
  * Switched matmul and convolution implementation with ACL to stateless API
  significantly improving primitive creation time and increasing caching
  efficiency and performance for these operators.

# Functionality

  * Introduced [generic GPU] support. This implementation relies on portable
  SYCL kernels and can be used as a starting point to enable new devices in
  oneDNN.
  * Extended functionality supported on NVIDIA GPUs and AMD GPUs with SYCL-based
  implementations.
  * Enabled support for `int8` activations with grouped scales and `int8`
  or `int4` compressed weights in matmul primitive. This functionality
  is implemented on Intel GPUs.
  * Introduces support for stochastic rounding for `fp8` data type
  functionality.
  * **[experimental]** Extended [microkernel API]:
    - Introduced `int8` quantization support.
    - Extended transform microkernel with transposition support and support for
    arbitrary strides.
    - Introduced verbose diagnostics support.
  * **[experimental]** Extended [sparse API]:
    - Introduced support for sparse memory with coordinate (COO) storage format.
    - Extended matmul primitive to work with sparse memory in COO format. This
    functionality is implemented on CPUs and Intel GPUs.
  * Introduced `int8` support in eltwise primitive with 'clip' algorithm. This
  functionality is implemented on CPUs.
  * Graph API:
    - Introduced `GroupNorm` operation and fusions in Graph API.
    - Introduced support for standalone `StaticReshape` and `StaticTranspose`
    operations.

[generic GPU]: https://github.com/oneapi-src/oneDNN/blob/rls-v3.6/src/gpu/generic/sycl/README.md
[microkernel API]: https://oneapi-src.github.io/oneDNN/v3.6/ukernels.html
[sparse API]: https://oneapi-src.github.io/oneDNN/v3.6/dev_guide_experimental.html#onednn-experimental-sparse

# Usability

  * Added [examples][Graph API examples] for SDPA, MQA, and GQA patterns
  implementation with Graph API.
  * Added [an example][deconvolution example] for deconvolution primitive.
  * Added examples for [Vanilla RNN][Vanilla RNN example] and
  [LBR GRU][LBR GRU example] RNN cells.
  * Introduced support for Intel DPC++/C++ Compiler 2025.0.
  * Introduced interoperability with [SYCL Graph] record/replay mode.
  * Removed dependency on OpenCL runtime for NVIDIA and AMD GPUs.
  * **[experimental]** Introduced [logging mechanism][spdlog] based on spdlog
  library.
  * Introduced support for `ONEDNN_ENABLE_WORKLOAD` build knob for Graph API.
  * Improved performance of `get_partitions()` function in Graph API.

[Graph API examples]: https://github.com/oneapi-src/oneDNN/tree/rls-v3.6/examples/graph
[deconvolution example]: https://github.com/oneapi-src/oneDNN/blob/rls-v3.6/examples/primitives/deconvolution.cpp
[Vanilla RNN example]: https://github.com/oneapi-src/oneDNN/blob/rls-v3.6/examples/primitives/vanilla_rnn.cpp
[LBR GRU example]: https://github.com/oneapi-src/oneDNN/blob/rls-v3.6/examples/primitives/lbr_gru.cpp
[SYCL Graph]: https://codeplay.com/portal/blogs/2024/01/22/sycl-graphs
[spdlog]: https://oneapi-src.github.io/oneDNN/v3.6/dev_guide_experimental.html#onednn-experimental-logging

# Validation

  * Introduced protection from out-of-memory scenarios in benchdnn Graph API
  driver.

# Deprecated Functionality

  * Experimental [Graph Compiler] is deprecated and will be removed in future releases.

[Graph Compiler]: https://oneapi-src.github.io/oneDNN/v3.6/dev_guide_graph_compiler.html

# Breaking Changes

  * Experimental [microkernel API] in this release is not compatible with
  [the version available][microkernel API v3.5] in oneDNN v3.5.
  * Updated minimal supported ACL version to 24.08.1 (was 24.04).

[microkernel API v3.5]: https://oneapi-src.github.io/oneDNN/v3.5/ukernels.html

# Thanks to these Contributors

This release contains contributions from the [project core team] as well as
Abdel @quickwritereader, Adam Jackson @nwnk, Aleksandr Voron @alvoron,
Alexey Makarevich @amakarev, Annop Wongwathanarat @annop-w, Daniel Kuts
@apach301, @deepeshfujitsu, Fadi Arafeh @fadara01, Fritz Heckel @fwph,
Gorokhov Dmitriy @dmitry-gorokhov, Deeksha Kasture @kasturedeeksha,
Kentaro Kawakami @kawakami-k, Marek Michalowski @michalowski-arm,
@matthias-bonne, @Menooker, Michael Froelich @MichaelFroelich,
Nicolas Miller @npmiller, Nikhil Sharma @nikhilfujitsu, @nishith-fujitsu,
Permanence AI Coder @Permanence-AI-Coder, Radu Salavat @Radu2k, Renato Barros
Arantes @renato-arantes, Robert Cohn @rscohn2, Robert Hardwick @robert-hardwick,
Ryo Suzuki @Ryo-not-rio, Shreyas-fuj @Shreyas-fuj, Shu Chen @shu1chen,
Siddhartha Menon @Sqvid, Song Jiaming @Litchilitchy, Vladimir Paramuzov
@vladimir-paramuzov, Yifei Zhang @yifeizh2. We would also like to thank everyone
who asked questions and reported issues.

[project core team]: https://github.com/oneapi-src/oneDNN/blob/rls-v3.6/MAINTAINERS.md
