Developer Guide {#dev_guide}
============================

oneDNN is an open-source performance library for deep learning applications.
The library includes basic building blocks for neural networks optimized
for Intel Architecture Processors and Intel Processor Graphics.
oneDNN is intended for deep learning applications and framework
developers interested in improving application performance
on Intel CPUs and GPUs.

# Building and Linking

 * @ref dev_guide_build
    * @ref dev_guide_build_options
 * @ref dev_guide_link

# Programming Model

 * @ref dev_guide_basic_concepts
 * @ref getting_started_cpp
 * @ref memory_format_propagation_cpp
 * @ref dev_guide_inference_and_training_aspects
   * @ref dev_guide_inference
   * @ref dev_guide_inference_int8
   * @ref dev_guide_training_bf16
 * @ref dev_guide_attributes
   * @ref dev_guide_attributes_scratchpad
   * @ref dev_guide_attributes_quantization
   * @ref dev_guide_attributes_post_ops
 * @ref dev_guide_data_types
 * @ref cross_engine_reorder_cpp
 * @ref dev_guide_c_and_cpp_apis

# Primitives

Compute intensive operations:
 * [(De-)Convolution](@ref dev_guide_convolution): Direct 1D/2D/3D, Winograd 2D
 * [Inner Product](@ref dev_guide_inner_product)
 * [Matrix Multiplication](@ref dev_guide_matmul)
 * [RNN](@ref dev_guide_rnn): LSTM, Vanilla RNN, GRU

Memory bandwidth limited operations:
 * [Batch Normalization](@ref dev_guide_batch_normalization)
 * [Binary](@ref dev_guide_binary)
 * [Concat](@ref dev_guide_concat)
 * [Elementwise](@ref dev_guide_eltwise): ReLU, Tanh, ELU, Abs, and other
 * [Layer Normalization](@ref dev_guide_layer_normalization)
 * [Local Response Normalization](@ref dev_guide_lrn)
 * [LogSoftmax](@ref dev_guide_logsoftmax)
 * [Pooling](@ref dev_guide_pooling)
 * [Resampling](@ref dev_guide_resampling)
 * [Shuffle](@ref dev_guide_shuffle)
 * [Softmax](@ref dev_guide_softmax)
 * [Sum](@ref dev_guide_sum)

Data manipulation:
 * [Reorder](@ref dev_guide_reorder)

# Performance Benchmarking and Inspection

 * @ref dev_guide_performance_settings
 * @ref dev_guide_verbose
 * @ref dev_guide_benchdnn
 * @ref dev_guide_profilers
 * @ref dev_guide_inspecting_jit
 * @ref performance_profiling_cpp
 * @ref dev_guide_cpu_dispatcher_control

# Advanced topics

 * @ref dev_guide_transition_to_v1
 * @ref dev_guide_transition_to_dnnl
 * @ref dev_guide_understanding_memory_formats
 * @ref dev_guide_int8_computations
 * @ref dev_guide_opencl_interoperability
 * @ref dev_guide_dpcpp_interoperability
 * @ref dev_guide_dpcpp_usm
 * @ref dev_guide_dpcpp_backends
 * @ref dev_guide_primitive_cache
 * @ref dev_guide_threadpool

# Examples

| Topic                          | Engine   | C++ API                                | C API                        |
| :----                          | :---     | :----                                  | :---                         |
| Tutorials                      | CPU/GPU  | @ref getting_started_cpp               |                              |
|                                | CPU/GPU  | @ref memory_format_propagation_cpp     |                              |
|                                | CPU/GPU  | @ref performance_profiling_cpp         |                              |
|                                | CPU/GPU  | @ref cross_engine_reorder_cpp          | @ref cross_engine_reorder_c  |
|                                | GPU      | @ref gpu_opencl_interop_cpp            |                              |
| f32 inference                  | CPU/GPU  | @ref cnn_inference_f32_cpp             | @ref cnn_inference_f32_c     |
|                                | CPU      | @ref cpu_rnn_inference_f32_cpp         |                              |
| int8 inference                 | CPU/GPU  | @ref cnn_inference_int8_cpp            |                              |
|                                | CPU      | @ref cpu_rnn_inference_int8_cpp        |                              |
| f32 training                   | CPU/GPU  | @ref cnn_training_f32_cpp              |                              |
|                                | CPU      |                                        | @ref cpu_cnn_training_f32_c  |
|                                | CPU/GPU  | @ref rnn_training_f32_cpp              |                              |
| bf16 training                  | CPU      | @ref cpu_cnn_training_bf16_cpp         |                              |
| SYCL interoperability          | CPU/GPU  | @ref sycl_interop_cpp                  |                              |
