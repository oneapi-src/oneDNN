# Examples {#dev_guide_examples}

| Topic                                | Engine   | Data Type | C++ API                                    | C API                             |
| :----                                | :---     | :---      | :----                                      | :---                              |
| conv+relu                            | CPU      | FP32      | @ref cpu_conv_relu_pattern_cpp             |                                   |
| conv+relu with XIO format            | CPU      | FP32      | @ref cpu_conv_relu_pattern2_cpp            |                                   |
| conv+bias+bn+relu                    | CPU      | FP32      | @ref cpu_conv_bias_bn_relu_pattern_cpp     |                                   |
| conv+bias+relu                       | CPU      | FP32      | @ref cpu_conv_bias_relu_pattern_cpp        |                                   |
| conv+bias+relu                       | CPU      | BF16      | @ref cpu_simple_pattern_bf16_cpp           |                                   |
| conv+bn+add+relu                     | CPU      | FP32      | @ref cpu_conv_bn_add_relu_pattern_cpp      | @ref cpu_simple_pattern_c         |
| conv+relu with data conversion       | CPU      | FP32      | @ref cpu_conversion_simple_pattern_cpp     |                                   |
| conv+relu+conv+relu                  | CPU/GPU  | FP32      | @ref sycl_simple_pattern_cpp               |                                   |
| conv+relu+conv+relu                  | CPU/GPU  | BF16      | @ref sycl_simple_pattern_bf16_cpp          |                                   |
| conv+bias+relu                       | GPU      | FP16      | @ref gpu_simple_pattern_fp16_cpp           |                                   |
| conv+bias+bn+relu                    | CPU/GPU  | FP32      | @ref sycl_conv_bias_bn_relu_pattern_cpp    |                                   |
| conv+bias+bn+add+relu                | CPU      | FP32      |                                            | @ref cpu_conv_bias_bn_add_relu_c  |
| conv+bias+relu+conv+bias+relu        | CPU      | FP32      |                                            | @ref cpu_multi_times_inference_c  |
| conv+bias+relu+conv+bias+relu (tiny) | CPU      | FP32      |                                            | @ref cpu_simple_pattern_tiny_c    |
| add(conv+bias, conv+bias)            | CPU      | FP32      | @ref cpu_inplace_options_cpp               |                                   |
| matmul+relu                          | CPU      | FP32      | @ref cpu_matmul_relu_pattern_cpp           |                                   |
