# Examples {#dev_guide_examples}

## C++ API examples

| Topic                                | Engine   | Data Type | Examples                                              |
| :----                                | :---     | :---      | :----                                                 |
| conv+bias+relu                       | CPU      | FP32      | @ref cpu_simple_pattern_f32_cpp                       |
| conv+bias+relu                       | CPU      | BF16      | @ref cpu_simple_pattern_bf16_cpp                      |
| int8 conv+relu                       | CPU      | INT8      | @ref cpu_simple_pattern_int8_cpp                      |
| conv+relu+conv+relu                  | CPU/GPU  | FP32      | @ref sycl_get_started_cpp                             |
| conv+bias+relu                       | GPU      | FP16      | @ref gpu_simple_pattern_fp16_cpp                      |
| inplace computation                  | CPU      | FP32      | @ref cpu_inplace_ports_cpp                            |
| Single operator partition            | CPU      | FP32      | @ref cpu_single_op_partition_f32_cpp                  |
| Simple CNN Training                  | CPU      | FP32      | @ref cpu_cnn_training_f32_cpp                         |

## C API examples

| Topic                                | Engine   | Data Type | Examples                          |
| :----                                | :---     | :---      | :----                             |
| conv+bn+add+relu                     | CPU      | FP32      | @ref cpu_simple_pattern_c         |
| conv+bias+bn+add+relu                | CPU      | FP32      | @ref cpu_conv_bias_bn_add_relu_c  |
| conv+bias+relu+conv+bias+relu        | CPU      | FP32      | @ref cpu_multi_times_inference_c  |
| conv+bias+relu+conv+bias+relu (tiny) | CPU      | FP32      | @ref cpu_simple_pattern_tiny_c    |
