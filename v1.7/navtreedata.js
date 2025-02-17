/*
@ @licstart  The following is the entire license notice for the
JavaScript code in this file.

Copyright (C) 1997-2017 by Dimitri van Heesch

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with this program; if not, write to the Free Software Foundation, Inc.,
51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

@licend  The above is the entire license notice
for the JavaScript code in this file
*/
var NAVTREE =
[
  [ "oneDNN", "index.html", [
    [ "Developer Guide", "index.html", null ],
    [ "Building and Linking", "usergroup0.html", [
      [ "Build from Source", "dev_guide_build.html", null ],
      [ "Build Options", "dev_guide_build_options.html", null ],
      [ "Linking to the Library", "dev_guide_link.html", null ]
    ] ],
    [ "Programming Model", "usergroup1.html", [
      [ "Basic Concepts", "dev_guide_basic_concepts.html", null ],
      [ "Getting Started", "getting_started_cpp.html", null ],
      [ "Memory Format Propagation", "memory_format_propagation_cpp.html", null ],
      [ "Inference and Training", "dev_guide_inference_and_training_aspects.html", [
        [ "Inference", "dev_guide_inference.html", null ],
        [ "Inference Using int8", "dev_guide_inference_int8.html", null ],
        [ "Training Using bfloat16", "dev_guide_training_bf16.html", null ]
      ] ],
      [ "Primitive Attributes", "dev_guide_attributes.html", [
        [ "Managing Scratchpad", "dev_guide_attributes_scratchpad.html", null ],
        [ "Quantization", "dev_guide_attributes_quantization.html", null ],
        [ "Post-ops", "dev_guide_attributes_post_ops.html", null ]
      ] ],
      [ "Data Types", "dev_guide_data_types.html", null ],
      [ "Reorder Between CPU and GPU Engines", "cross_engine_reorder_cpp.html", null ],
      [ "C and C++ API", "dev_guide_c_and_cpp_apis.html", null ]
    ] ],
    [ "Supported Primitives", "usergroup2.html", [
      [ "(De-)Convolution", "dev_guide_convolution.html", null ],
      [ "Inner Product", "dev_guide_inner_product.html", null ],
      [ "Matrix Multiplication", "dev_guide_matmul.html", null ],
      [ "RNN", "dev_guide_rnn.html", null ],
      [ "Batch Normalization", "dev_guide_batch_normalization.html", null ],
      [ "Binary", "dev_guide_binary.html", null ],
      [ "Concat", "dev_guide_concat.html", null ],
      [ "Elementwise", "dev_guide_eltwise.html", null ],
      [ "Layer Normalization", "dev_guide_layer_normalization.html", null ],
      [ "Local Response Normalization", "dev_guide_lrn.html", null ],
      [ "Logsoftmax", "dev_guide_logsoftmax.html", null ],
      [ "Pooling", "dev_guide_pooling.html", null ],
      [ "Resampling", "dev_guide_resampling.html", null ],
      [ "Shuffle", "dev_guide_shuffle.html", null ],
      [ "Softmax", "dev_guide_softmax.html", null ],
      [ "Sum", "dev_guide_sum.html", null ],
      [ "Reorder", "dev_guide_reorder.html", null ],
      [ "Reduction", "dev_guide_reduction.html", null ]
    ] ],
    [ "Examples", "dev_guide_examples.html", null ],
    [ "Performance Profiling and Inspection", "usergroup3.html", [
      [ "Verbose Mode", "dev_guide_verbose.html", null ],
      [ "Configuring oneDNN for Benchmarking", "dev_guide_performance_settings.html", null ],
      [ "Performance Benchmark", "dev_guide_benchdnn.html", null ],
      [ "Profiling oneDNN Performance", "dev_guide_profilers.html", null ],
      [ "Inspecting JIT Code", "dev_guide_inspecting_jit.html", null ],
      [ "Performance Profiling Example", "performance_profiling_cpp.html", null ],
      [ "CPU Dispatcher Controls", "dev_guide_cpu_dispatcher_control.html", null ]
    ] ],
    [ "Advanced Topics", "usergroup4.html", [
      [ "Transition from v0.x to v1.x", "dev_guide_transition_to_v1.html", null ],
      [ "Transition from Intel MKL-DNN to oneDNN", "dev_guide_transition_to_dnnl.html", null ],
      [ "Understanding oneDNN Memory Formats", "dev_guide_understanding_memory_formats.html", null ],
      [ "Nuances of int8 computations", "dev_guide_int8_computations.html", null ],
      [ "OpenCL Interoperability", "dev_guide_opencl_interoperability.html", null ],
      [ "Primitive Cache", "dev_guide_primitive_cache.html", null ],
      [ "Using oneDNN with Threadpool-based Threading", "dev_guide_threadpool.html", null ]
    ] ],
    [ "API Reference", "usergroup5.html", [
      [ "Modules", "modules.html", "modules" ],
      [ "Namespace Members", "namespacemembers.html", [
        [ "All", "namespacemembers.html", null ],
        [ "Functions", "namespacemembers_func.html", null ],
        [ "Typedefs", "namespacemembers_type.html", null ],
        [ "Enumerations", "namespacemembers_enum.html", null ]
      ] ],
      [ "Class List", "annotated.html", [
        [ "Class List", "annotated.html", "annotated_dup" ],
        [ "Class Index", "classes.html", null ],
        [ "Class Hierarchy", "hierarchy.html", "hierarchy" ],
        [ "Class Members", "functions.html", [
          [ "All", "functions.html", "functions_dup" ],
          [ "Functions", "functions_func.html", "functions_func" ],
          [ "Variables", "functions_vars.html", null ],
          [ "Typedefs", "functions_type.html", null ],
          [ "Enumerations", "functions_enum.html", null ]
        ] ]
      ] ]
    ] ]
  ] ]
];

var NAVTREEINDEX =
[
"annotated.html",
"group__dnnl__api__engine.html#ga8a38bdce17f51616d03310a8e8764c8c",
"group__dnnl__api__primitives__common.html#ga5cf662f6dc742a3500ec4f54290b86da",
"group__dnnl__api__rnn.html#ga1915ea2d2fe94077fa30734ced88a225",
"structdnnl_1_1deconvolution__backward__data_1_1desc.html#a4aac0ae3c42454b1d9eecead6a9c316b",
"structdnnl_1_1lbr__gru__forward.html",
"structdnnl_1_1memory.html#a8e83474ec3a50e08e37af76c8c075dceaf31ee5e3824f1f5e5d206bdf3029f22b",
"structdnnl_1_1rnn__primitive__desc__base.html#aa9a4652f8f5c8d1b32d2f26deac70f92",
"structdnnl__pooling__desc__t.html#a22eca3d8d4a7f8eec128fb62614dc841"
];

var SYNCONMSG = 'click to disable panel synchronisation';
var SYNCOFFMSG = 'click to enable panel synchronisation';