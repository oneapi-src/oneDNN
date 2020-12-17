# Operators and Fusion Patterns {#dev_guide_ops_and_patterns}

## Operators

All the operators defined in [oneDNN Graph Specification](https://spec.oneapi.com/onednn-graph/latest/ops/index.html) are supported. Including:

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
- Divide
- Elu
- EluBackprop
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
- ReduceSum
- ReLU
- ReLUBackprop
- Reshape
- Round
- Sigmoid
- SigmoidBackprop
- SoftMax
- SoftMaxBackprop
- SoftPlus
- SoftPlusBackprop
- Sqrt
- SqrtBackprop
- Square
- Tanh
- TanhBackprop
- Transpose
- Wildcard

## Fusion Patterns

The preview depends on the oneDNN primitive post-ops feature to support fusion. It supports a subset of the pattern capability as listed below.

- Convolution + Add
- Convolution + Add + ReLU
- Convolution + BiasAdd
- Convolution + BiasAdd + Add
- Convolution + BiasAdd + Add + ReLU
- Convolution + BiasAdd + Add + ELU
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
- Matmul + ReLU
- MatMul + Elu
- MatMul + GELU
- MatMul + Sigmoid
- MatMul + HardTanh
- MatMul + BiasAdd + Add
- BatchNormInference + ReLU
