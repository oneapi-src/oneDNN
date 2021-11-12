# Operators and Fusion Patterns {#dev_guide_ops_and_patterns}

## Operators

Supported operation refers to operation which can be converted to oneDNN Graph
OP and thus can be part of oneDNN Graph partition. The preview supports the
following operations as part of Opset defined in oneDNN Graph Spec. It supports
FP32/FP16/BF16 data type.  For complete OP definition, please refer to
[oneDNN Graph Specification](https://spec.oneapi.com/onednn-graph/latest/ops/index.html).

- AvgPool
- AvgPoolBackprop
- BatchNormForwardTraining
- BatchNormInference
- BatchNormTrainingBackprop
- BiasAdd
- BiasAddBackprop
- Clamp
- ClampBackprop
- Concat
- Convolution
- ConvolutionBackpropData
- ConvolutionBackpropFilters
- ConvTranspose
- Dequantize
- DynamicReshape
- DynamicTranspose
- Divide
- Elu
- EluBackprop
- End
- Erf
- Exp
- GELU
- GELUBackprop
- HardTanh
- HardTanhBackprop
- Index
- Interpolate
- InterpolateBackprop
- LayerNorm
- LayerNormBackprop
- Log
- LogSoftmax
- LogSoftmaxBackprop
- MatMul
- MaxPool
- MaxPoolBackprop
- Maximum
- Minimum
- Multiply
- Pow
- PowBackprop
- PowBackpropExponent
- Quantize
- ReduceSum
- ReLU
- ReLUBackprop
- Reorder
- Round
- Sigmoid
- SigmoidBackprop
- SoftMax
- SoftMaxBackprop
- SoftPlus
- SoftPlusBackprop
- Sqrt
- SqrtBackprop
- StaticReshape
- StaticTranspose
- Square
- Tanh
- TanhBackprop
- TypeCast
- Wildcard

## Fusion Patterns

The preview depends on the oneDNN primitive post-ops feature to support fusion.
It supports a subset of the pattern capability as listed below.

- Add + Multiply
- Add + ReLU
- Add + Sigmoid
- AvgPool + Add
- BatchNormInference + ReLU
- Convolution + Add
- Convolution + Add + ReLU
- Convolution + Add + HardTanh (ReLU6)
- Convolution + Add + Elu
- Convolution + BiasAdd
- Convolution + BiasAdd + Add
- Convolution + BiasAdd + Add + ReLU
- Convolution + BiasAdd + Add + Elu
- Convolution + BiasAdd + Elu
- Convolution + BiasAdd + Sigmoid
- Convolution + BiasAdd + ReLU
- Convolution + BiasAdd + HardTanh
- Convolution + BiasAdd + Square
- Convolution + BiasAdd + Tanh
- Convolution + BiasAdd + Sqrt
- Convolution + BiasAdd + BatchNormInference
- Convolution + BiasAdd + BatchNormInference + Add
- Convolution + BiasAdd + BatchNormInference + Add + ReLU
- Convolution + BiasAdd + BatchNormInference + ReLU
- Convolution + BatchNormInference
- Convolution + BatchNormInference + Add
- Convolution + BatchNormInference + Add + ReLU
- Convolution + BatchNormInference + ReLU
- Convolution + ReLU
- ConvTranspose + BiasAdd
- MatMul + Add
- MatMul + Add + GELU
- MatMul + Add + ReLU
- MatMul + Add + Sigmoid
- Matmul + ReLU
- MatMul + Elu
- MatMul + GELU
- MatMul + Sigmoid
- MatMul + HardTanh
- MatMul + BiasAdd
- MatMul + BiasAdd + Add
- MatMul + BiasAdd + Add + ReLU
- MatMul + BiasAdd + Elu
- MatMul + BiasAdd + GELU
- MatMul + BiasAdd + HardTanh
- MatMul + BiasAdd + ReLU
- MatMul + BiasAdd + Sigmoid
- MatMul + BiasAdd + Sigmoid + Multiply (Swish)
- Maximum + Add
- Maximum + ReLU
- Maximum + Sigmoid
- MaxPool + Add
- Minimum + Add
- Minimum + ReLU
- Minimum + Sigmoid
- Multiply + Add
- Multiply + ReLU
- Multiply + Sigmoid
- Pow + Multiply + Add + Tanh (GELU)
