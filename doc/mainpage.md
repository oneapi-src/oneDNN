Developer Guide {#dev_guide}
============================

The Intel(R) Math Kernel Library for Deep Neural Networks (Intel(R) MKL-DNN)
is an open-source C/C++ performance library for Deep Learning (DL)
applications primarily intended for acceleration of DL frameworks on Intel(R)
architecture and Intel(R) Processor Graphics Architecture. Intel MKL-DNN
includes highly optimized implementations of computational operations used in
convolutional and recurrent neural networks covering a wide range of
applications, including image recognition, object detection, semantic
segmentation, neural machine translation, and speech recognition.

# Building and Linking

 * @ref dev_guide_build
    * @ref dev_guide_build_options
 * @ref dev_guide_link

# Programming Model

 * @ref dev_guide_basic_concepts
 * @ref cpu_getting_started_cpp
 * @ref cpu_memory_format_propagation_cpp
 * @ref dev_guide_inference_and_training_aspects
   * @ref dev_guide_inference
   * @ref dev_guide_inference_int8
 * @ref dev_guide_attributes
   * @ref dev_guide_attributes_scratchpad
   * @ref dev_guide_attributes_quantization
   * @ref dev_guide_attributes_post_ops
 * @ref dev_guide_data_types
 * @ref dev_guide_c_and_cpp_apis


# Primitives

Compute intensive operations:
 * [(De-)Convolution](@ref dev_guide_convolution): Direct 1D/2D/3D, Winograd 2D
 * [Inner Product](@ref dev_guide_inner_product)
 * [RNN](@ref dev_guide_rnn): LSTM, Vanilla RNN, GRU

Memory bandwidth limited operations:
 * [Pooling](@ref dev_guide_pooling)
 * [Batch Normalization](@ref dev_guide_batch_normalization)
 * [Local Response Normalization](@ref dev_guide_lrn)
 * [Softmax](@ref dev_guide_softmax)
 * [Elementwise](@ref dev_guide_eltwise): ReLU, Tanh, ELU, Abs, and other
 * [Sum](@ref dev_guide_sum)
 * [Concat](@ref dev_guide_concat)
 * [Shuffle](@ref dev_guide_shuffle)

Data manipulation:
 * [Reorder](@ref dev_guide_reorder)


# Performance Benchmarking and Inspection

 * @ref dev_guide_verbose
 * @ref dev_guide_benchdnn
 * @ref dev_guide_vtune
 * @ref dev_guide_inspecting_jit

# Advanced topics

 * @ref dev_guide_understanding_memory_formats

# Examples

| Scenario           | Platform | C++ API                          | C API                          |
| :----              | :---     | :----                            | :---                           |
| Introduction       | CPU      | @ref cpu_getting_started_cpp     |                                |
|                    | GPU      | @ref gpu_getting_started_cpp     | @ref gpu_getting_started_c     |
| fp32 inference     | CPU      | @ref cpu_cnn_inference_fp32_cpp  | @ref cpu_cnn_inference_fp32_c  |
|                    |          | @ref cpu_rnn_inference_fp32_cpp  |                                |
| int8 inference     | CPU      | @ref cpu_cnn_inference_int8_cpp  |                                |
|                    |          | @ref cpu_rnn_inference_int8_cpp  |                                |
| training           | CPU      | @ref cpu_cnn_training_fp32_cpp   | @ref cpu_cnn_training_fp32_c   |
|                    | CPU      | @ref cpu_rnn_training_fp32_cpp   |                                |
