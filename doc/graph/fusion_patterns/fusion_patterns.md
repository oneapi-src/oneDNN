Fusion Patterns {#dev_guide_graph_fusion_patterns}
==================================================

| Pattern | Description                  |
|:--------|:-----------------------------|
| Scaled Dot-Product Attention | Refer to @ref dev_guide_graph_sdpa for more details. |
| Grouped Query Attention | Refer to @ref dev_guide_graph_gqa for more details. |
| Scaled Dot-Product Attention with Compressed Key/Value | Refer to @ref dev_guide_graph_sdpa_compressed_kv for more details. |
| Gated Multi-Layer Perceptron (Gated-MLP) | Refer to @ref dev_guide_graph_gated_mlp for more details. |
| MatMul Fusions | Refer to @ref dev_guide_graph_matmul_fusions for more details. |
| Quantized MatMul Fusions | Refer to @ref dev_guide_graph_quantized_matmul_fusions for more details. |
| Convolution Fusions | Refer to @ref dev_guide_graph_convolution_fusions for more details. |
| Quantized Convolution Fusions | Refer to @ref dev_guide_graph_quantized_convolution_fusions for more details. |
| ConvolutionBackwardWeights Fusions | Refer to @ref dev_guide_graph_convolutionbackwardweights_fusions for more details. |
| ConvTranspose Fusions | Refer to @ref dev_guide_graph_convtranspose_fusions for more details. |
| Quantized ConvTranspose Fusions | Refer to @ref dev_guide_graph_quantized_convtranspose_fusions for more details. |
| Softmax Fusions | Refer to @ref dev_guide_graph_softmax_fusions for more details. |
| Binary Fusions | Fusions related to binary operations like Add, Divide, Maximum, Minimum, Multiply, Subtract. Refer to @ref dev_guide_graph_binary_fusions for more details. |
| Unary Fusions | Fusions related to unary operations like Abs, Clamp, Elu, Exp, GELU, HardSigmoid, HardSwish, LeakyReLU, Log, Mish, Sigmoid, SoftPlus, ReLU, Round, Sqrt, Square, Tanh. Refer to @ref dev_guide_graph_unary_fusions for more details. |
| Norm Fusions | Fusions related to norm operations like GroupNorm, LayerNorm, BatchNormInference, BatchNormForwardTraining, BatchNormTrainingBackward. Refer to @ref dev_guide_graph_norm_fusions for more details. |
| Pool Fusions | Fusions related to pool operations like MaxPool, AvgPool. Refer to @ref dev_guide_graph_pool_fusions for more details. |
| Reduction Fusions | Fusions related to reduction operations like ReduceL1, ReduceL2, ReduceMax, ReduceMean, ReduceMin, ReduceProd, ReduceSum. Refer to @ref dev_guide_graph_reduction_fusions for more details. |
| Reorder Fusions | Fusions related to reorder operations like Reorder, StaticReshape, StaticTranspose. Refer to @ref dev_guide_graph_reorder_fusions for more details. |
| Other Fusions | Fusions related to Concat, Interpolate, Reciprocal, TypeCast. Refer to @ref dev_guide_graph_other_fusions for more details. |