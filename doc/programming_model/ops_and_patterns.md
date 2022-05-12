# Operators and Fusion Patterns {#dev_guide_ops_and_patterns}

## Operators

Supported operation refers to operation which can be converted to oneDNN Graph
OP and thus can be part of oneDNN Graph partition. The alpha release supports the
following operations as part of Opset defined in oneDNN Graph Spec. It supports
FP32/FP16/BF16/S8/U8 data type.  For complete OP definition, please refer to
[oneDNN Graph Specification](https://spec.oneapi.com/onednn-graph/latest/ops/index.html).

- Abs
- Add
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
- ConvTransposeBackpropData
- ConvTransposeBackpropFilters
- Dequantize
- DynamicDequantize
- DynamicQuantize
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
- HardSwish
- HardSwishBackprop
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
- Negative
- Pow
- PowBackprop
- PowBackpropExponent
- PReLU
- PReLUBackprop
- Quantize
- Reciprocal
- ReduceL1
- ReduceL2
- ReduceMax
- ReduceMean
- ReduceMin
- ReduceProd
- ReduceSum
- ReLU
- ReLUBackprop
- Reorder
- Round
- Sigmoid
- SigmoidBackprop
- Sign
- SoftMax
- SoftMaxBackprop
- SoftPlus
- SoftPlusBackprop
- Sqrt
- SqrtBackprop
- StaticReshape
- StaticTranspose
- Square
- SquaredDifference
- Subtract
- Tanh
- TanhBackprop
- TypeCast
- Wildcard

## Fusion Patterns

### oneDNN Primitive Backend

The alpha release depends on the oneDNN primitive post-ops feature to support fusion.
It supports a subset of the pattern capability as listed below.

#### 1. Floating Point Patterns

- [Abs / Clamp / Elu / Exp / GELU / HardTanh / HardSwish / Log / Sigmoid /
  Sigmoid + Multiply / SoftPlus / Pow / ReLU / Round / Sqrt / Square / Tanh] +
  [Add / Divide / Maximum / Minimum / Multiply / Subtract]
- [Add / Divide / Maximum / Minimum / Multiply / Subtract] + [Abs / Clamp / Elu /
  Exp / GELU / HardTanh / HardSwish / Log / Sigmoid+Multiply / Sigmoid /
  SoftPlus / Pow / ReLU / Round / Sqrt / Square / Tanh / Add / Multiply /
  Maximum / Minimum / Divide / Subtract]
- [AvgPool / MaxPool] + [Add / Multiply / Maximum / Minimum / Divide / Subtract]
- BatchNormInference + ReLU
- Convolution + [BiasAdd]\* + [BatchNormInference]\* + [Abs / Clamp / Elu / Exp /
  GELU / HardTanh / HardSwish / Log / Sigmoid+Multiply / Sigmoid / SoftPlus /
  Pow / ReLU / Round / Sqrt / Square / Tanh / Add / Multiply / Maximum / Minimum
  / Divide / Subtract]<sup>*[0,3]</sup>
- Convolution + ReLU + Convolution + ReLU + Convolution + Add + ReLU
- Convolution (with bias) + ReLU + Convolution (with bias) + ReLU
- Convolution (with bias) + ReLU + Convolution (with bias) + Add + ReLU
- Convolution + [BiasAdd]\* + ReLU + Convolution + [BiasAdd]\* + ReLU +
  Convolution + [BiasAdd]\* + Add + ReLU
- Convolution + [BiasAdd]\* + ReLU + Convolution + [BiasAdd]\*+ ReLU +
  Convolution + [BiasAdd]\* + Convolution + [BiasAdd]\* + Add + ReLU
- ConvolutionBackpropFilters + BiasAddBackprop
- ConvTranspose + [BiasAdd]\* + [Abs / Clamp / Elu / Exp / GELU / HardTanh /
  HardSwish / Log / Sigmoid+Multiply / Sigmoid / SoftPlus / Pow / ReLU / Round /
  Sqrt / Square / Tanh / Add / Multiply / Maximum / Minimum / Divide / Subtract]<sup>\*[0,3]</sup>
- Interpolate + [Abs / Clamp / Elu / Exp / GELU / HardTanh / HardSwish / Log /
  Sigmoid + Multiply / Sigmoid / SoftPlus / Pow / ReLU / Round / Sqrt / Square /
  Tanh / Add / Multiply / Maximum / Minimum / Divide / Subtract]
- MatMul + [BiasAdd]\* + [Abs / Clamp / Elu / Exp / GELU / HardTanh / HardSwish /
  Log / Sigmoid+Multiply / Sigmoid / SoftPlus / Pow / ReLU / Round / Sqrt /
  Square / Tanh / Add / Multiply / Maximum / Minimum / Divide / Subtract]<sup>\*[0,3]</sup>
- Reciprocal + Multiply
- [ReduceL1 / ReduceL2 / ReduceMax / ReduceMean / ReduceMin / ReduceProd /
  ReduceSum] + [Abs / Clamp / Elu / Exp / GELU / HardTanh / HardSwish / Log /
  Sigmoid + Multiply / Sigmoid / SoftPlus / Pow / ReLU / Round / Sqrt / Square /
  Tanh / Add / Multiply / Maximum / Minimum / Divide / Subtract]
- ReLUBackprop + BatchNormTrainingBackprop
- Reorder + Add

-

```text
                      |
                    StaticReshape
            |         |
    StaticReshape   StaticTranspose
            |         |
    StaticTranspose StaticTranspose
              \      /
               MatMul
                 \    /
                 Divide
                    \   /       |
                     Add     StaticReshape
                      |         |
                   Softmax   StaticTranspose
                       \       /
                         MatMul
                           |
                    StaticTranspose
                           |

  ```

-

```text
   \   /
    Pow
      \  /
    Multiply
        \__   __/
           Add
            \    /
           Multiply
              |
            Tanh
              \   /
               Add
                 \      /
                 Multiply
                     \      /
                     Multiply
                        |
  ```

-

```text
   \   /
   Divide
     |
    Erf
     \   /
      Add
        \    /
       Multiply
            \      /
            Multiply
                |

  ```

#### 2. Quantized Patterns

- Dequantize + [AvgPool/MaxPool] + Quantize
- Dequantize + ReLU + Quantize
- Dequantize + Reorder + Quantize
- TypeCast + Quantize

-

```text
            |
        Dequantize
            |               |
      AvgPool/MaxPool   Dequantize
             \____     ____/
                   Add
                    |
                 Quantize
                    |
  ```

-

```text
    |              |
  Dequantize   Dequantize
     \            /
      Convolution (with bias)
           |
          ReLU
           |
        Quantize
           |           |
      Dequantize    Dequantize
            \         /
            Convolution (with bias)
                |
               ReLU
                |
             Quantize
                |

  ```

-

```text
    |             |
  Dequantize   Dequantize
     \           /
      Convolution (with bias or BiasAdd)
           |
          ReLU
           |
        Quantize
           |           |
      Dequantize    Dequantize
            \         /
            Convolution (with bias or BiasAdd)
                |
               ReLU
                |
             Quantize
                |            |
            Dequantize    Dequantize
                 \         /                            |
                 Convolution (with bias or BiasAdd)   Dequantize
                                                 \    /
                                                   Add
                                                    |
                                                  ReLU
                                                    |
                                                 Quantize
                                                    |
  ```

-

```text
     ___________________________|__________________________
    |                                                      |
    |             |                                        |           |
  Dequantize   Dequantize                              Dequantize   Dequantize
     \           /                                          \         /
      Convolution                                            Convolution
  (with bias or BiasAdd)                               (with bias or BiasAdd)
           |                                                     |
          ReLU                                               Quantize
           |                                                     |
        Quantize                                            Dequantize
           |                                                     |
      Dequantize    Dequantize                                   |
            \         /                                         /
            Convolution (with bias or BiasAdd)                 /
                |                                             /
               ReLU                                          /
                |                                           /
             Quantize                                      /
                |                                         /
            Dequantize    Dequantize                     /
                 \         /                            /
                 Convolution (with bias or BiasAdd)    /
                                                 \    /
                                                   Add
                                                    |
                                                   ReLU
                                                    |
                                                 Quantize
                                                    |

  ```

-

```text
    |              |
  Dequantize   Dequantize
     \___      ___/
         MatMul
            \    /
            Divide
               \   /
               [Add]*
                 |
  ```

-

```text
        |
    Dequantize
        |        |
      ReLU   Dequantize
       \___   ___/
           Add
            |
        Quantize
            |

  ```

-

```text
                                        |
                                    [Quantize]*
               |                        |
          Dequantize                Dequantize
               \                      /
            Convolution/ConvTranspose/MatMul
                            |
                        [BiasAdd]*
                            |
  [Abs/Clamp/Elu/Exp/GELU/HardTanh/HardSwish/Log/Sigmoid/SoftPlus/
   Pow/ReLU/Round/Sqrt/Square/Tanh/[Dequantize+Add]*[0,1] ]*[0,3]
                            |
                        [Quantize]*
                            |
  ```

-

```text
    |              |
  Dequantize   Dequantize
    |              |
  TypeCast     TypeCast
     \___      ___/
         MatMul
            \      /
            [Divide]*
                \     /
                 [Add]*
                   |

  ```

-

```text
    |              |
  Dequantize   Dequantize
    |              |
  TypeCast     TypeCast
     \___      ___/
         MatMul
           |
         [GeLU]*
           |
        TypeCast
           |
        Quantize
           |
  ```

-

```text
    |              |
  Dequantize   Dequantize
    |              |
  TypeCast     TypeCast  Dequantize
     \___      ___/          |
         MatMul           TypeCast
            \_____        ___/
                   [Add]*
                     |

  ```

-

```text
      |
  Dequantize
      |           |
    Reorder   Dequantize
       \__   __/
          Add
           |
        Quantize
           |
  ```

-

```text
                      |
                    StaticReshape
            |         |
    StaticReshape   StaticTranspose
            |         |
    StaticTranspose StaticTranspose
            |         |
        Quantize    Quantize
            |         |
        Dequantize  Dequantize
              \      /
               MatMul
                 \    /
                 Divide
                    \   /       |
                     Add     StaticReshape
                      |         |
                   Softmax   StaticTranspose
                      |         |
                   Quantize  Quantize
                      |         |
                  Dequantize  Dequantize
                       \       /
                         MatMul
                           |
                    StaticTranspose
                           |

  ```

-

```text
                      |
                    StaticReshape
            |         |
    StaticReshape   StaticTranspose
            |         |
    StaticTranspose StaticTranspose
            |         |
        TypeCast    TypeCast
            |         |
        Quantize    Quantize
            |         |
        Dequantize  Dequantize
            |         |
        TypeCast    TypeCast
              \      /
               MatMul
                 \    /
                 Divide
                    \   /       |
                     Add     StaticReshape
                      |         |
                   Softmax   StaticTranspose
                      |         |
                   TypeCast  TypeCast
                      |         |
                   Quantize  Quantize
                      |         |
                  Dequantize Dequantize
                      |         |
                   TypeCast  TypeCast
                       \       /
                         MatMul
                           |
                    StaticTranspose
                           |
  ```

### Graph Compiler Backend

The alpha release depends on graph compiler to support additional fusion
patterns.

#### 1. Compiler Backend Floating Point Patterns

Compiler backend supports floating point patterns for MHA inference and training
as well as MLP inference and training (2-6 layers).

-

```text
                      |
                    StaticReshape
            |         |
    StaticReshape   StaticTranspose
            |         |
    StaticTranspose StaticTranspose
              \      /
               MatMul
                 \     /
         Multiply/Divide
                    \    /      |
                     Add     StaticReshape
                      |         |
                   Softmax   StaticTranspose
                       \       /
                         MatMul
                           |
                    StaticTranspose
                           |
                    [StaticReshape]*
                           |

  ```

-

```text
              \      /
               MatMul
                 \     /
         Multiply/Divide
                    \    /
                     Add
                      |
                   Softmax
                       \       /
                         MatMul
                           |
                    StaticTranspose
                           |
                        Reorder
                           |
  ```

-

```text
              \      /
               MatMul
                 \     /
         Multiply/Divide
                    \    /
                     Add
                      |
                   Softmax
                ______|______
               /             \     /
                             Multiply
                           _____|______
                          /            \      /
                                        MatMul
                                           |
                                    StaticTranspose
                                           |
                                     StaticReshape
                                           |

  ```

-

```text
                        |
                  StaticReshape
                        |
                 StaticTranspose
           \     /           \    /
            MatMul           MatMul
              |                  \     /
                                Multiply
                                /    \     /
                               /     Multiply
                              |         |
                               \     ReduceSum
                                \      /
                                Subtract
                                  \       /
                                   Multiply
                                       \         /
                                     Multiply/Divide
                          ________________|______
                          \                      \     /
                           \     /                MatMul
                            MatMul                  |
                              |
  ```

- [MatMul + ReLU/Sigmoid/GELU]<sup>[2,6]</sup>

- [MLP Backprop Unit]<sup>[2,6]</sup>

  in which MLP Backprop Unit is

  ```text
                                 \                   /
            |          ReLUBackprop/SigmoidBackprop/GELUBackprop          |
  [StaticTranspose]* ______________________|______________________  [StaticTranspose]*
             \      /                      |                      \      /
              MatMul                  [ReduceSum]*                 MatMul
                |                          |                         |
  ```

#### 2. Compiler Backend Quantized Patterns

Compiler backend supports quantized patterns for MHA inference and MLP
inference (2-6 layers).

-

```text
                      |
                    Dequantize
            |         |
      Dequantize    StaticReshape
            |         |
    StaticReshape   StaticTranspose
            |         |
    StaticTranspose StaticTranspose
              \      /
               MatMul
                 \     /
         Multiply/Divide
                    \    /
                     Add
                      |         |
                   Softmax   Dequantize
                      |         |
                  Quantize   StaticReshape
                      |         |
                Dequantize   StaticTranspose
                       \       /
                         MatMul
                           |
                    StaticTranspose
                           |
                    [StaticReshape]*
                           |
                        Quantize
                           |

  ```

-

```text
             |        |
      Dequantize    Dequantize
              \      /
               MatMul
                 \     /
         Multiply/Divide
                    \    /
                     Add
                      |
                   Softmax
                      |
                   Quantize
                      |         |
                  Dequantize  Dequantize
                       \       /
                         MatMul
                           |
                    StaticTranspose
                           |
                        Reorder
                           |
                        Quantize
                           |

  ```

-

```text
             |        |
      Dequantize    Dequantize
             |        |
        TypeCast    TypeCast
              \      /
               MatMul
                 \     /
         Multiply/Divide
                    \    /
                     Add
                      |
                   Softmax
                      |
                   TypeCast
                      |
                   Quantize
                      |         |
                  Dequantize  Dequantize
                      |         |
                  TypeCast    TypeCast
                       \       /
                         MatMul
                           |
                    StaticTranspose
                           |
                        Reorder
                           |
                        TypeCast
                           |
                        Quantize
                           |
  ```

- [Quantized MLP Unit]<sup>[2,6]</sup>

  in which Quantized MLP Unit is:

  ```text

                    |       |
            Dequantize    Dequantize
                     \     /
                      MatMul
                        |
                ReLU/Sigmoid/GELU
                        |
                     Quantize
                        |

  ```
