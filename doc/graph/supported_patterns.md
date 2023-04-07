Supported Fusion Patterns {#dev_guide_graph_fusion_patterns}
============================================================

@anchor fusion_patterns
## Fusion Patterns

The following fusion patterns are subgraphs that the oneDNN Graph API recognizes
as candidate for fusion. The patterns are described using oneDNN Graph
operation (op) names with the following convention.

@note oneDNN Graph performs limited input validation to minimize the performance
overheads. The application is responsible for sanitizing inputs passed to the
library. For large u8 or s8 inputs may lead to accumulator overflow, you can use
floating point patterns instead of quantized patterns.

`"+"` describes a chain of two ops. The preceding op produces an output tensor,
which is consumed by the following op as its first operand.

`"[]"` describes a component of the overall pattern description. For example,
it could include a subgraph or all the op choices within the bracket.

`"|"` describes choices of multiple operations, say A+[B|C] means the graph
partition contains A followed by B or C.

`","` describes a graph composed of multiple subgraphs, each subgraph marks its
output tensor explicitly, which is consumed by other subgraphs.

`Superscript` denotes the numbers of repetition pattern. For example,
A+[B|C]\f$^{3}\f$ means the graph partition contains A followed by three ops,
each of them is either B or C. The superscript could be a range of number
meaning allowing a range of repetition. If the range is between 0 and 1, we use
superscript `"?"`.

`Subscript` denotes the input and output tensors which need to explicitly mark
the producer and consumer relation within one graph partition. For example,
A\f$_{>t1}\f$+B+C\f$_{<t1}\f$ refers
to the pattern started with A followed by B and C, and C takes an implicit input
tensor from B and an extra tensor t1 output from A. `">"` refers to the output
tensor, and `"<"` for input tensor.  Input and output tensor between neighbor
ops are not explicitly marked, for example, B consumes t1 implicitly in the
example above.

Subscript `"out"` marks the output tensor of a certain op to be the output of
a graph partition. For example, in
A\f$_{>t1}\f$+B\f$_{>out}\f$+C\f$_{<t1,>out}\f$, B's output and C's output
are marked as output tensors.

Subscript `"in"` marks the input tensor of a certain op to be the input of a
graph partition. For example, in A\f$_{<in1}\f$+B\f$_{<in1}\f$ A's input and
B's second input are graph partition input, and they share the same input tensor
in1. Most input tensors of a graph partition are not explicitly marked.
For example, the input tensors of the first op are implicitly regarded as graph
partition inputs. Besides, for input tensors of other ops, if they are not
produced by any proceeding ops, they are regarded as implicit graph partition
inputs. In the example A\f$_{>t1}\f$+B+C\f$_{<t1}\f$, A's inputs are
regarded as implicit graph partition inputs, and if B is a binary operation, the
second input tensor is an implicit graph partition input.

The following categories will be used in describing fusion pattern.

Unary = [Abs | Clamp | Elu | Exp | GELU | HardSwish | LeakyReLU |
Log | Sigmoid | SoftPlus | Pow | ReLU | Round | Sqrt | Square | Tanh]

Binary = [Add | Divide | Maximum | Minimum | Multiply | Subtract]

Reduction = [ReduceL1 | ReduceL2 | ReduceMax | ReduceMean | ReduceMin |
ReduceProd | ReduceSum]

### Inference

#### Floating Point Patterns

| Pattern | Description                  |
|:--------|:-----------------------------|
| Convolution + BiasAdd\f$^?\f$ + BatchNormInference\f$^?\f$ + [Unary \| Binary]\f$^{0-3}\f$\f$_{>out}\f$ | This pattern is widely used in Convolution Neural Networks, for example ResNet, ResNext, SSD, etc. |
| ConvTranspose + BiasAdd\f$^?\f$ + [Unary \| Binary]\f$^{0-3}\f$\f$_{>out}\f$ | This pattern is widely used in Generative Adversarial Networks. |
| Interpolate + [Unary \| Binary]\f$^{0-3}\f$\f$_{>out}\f$ | This pattern is widely used for image processing. |
| MatMul + BiasAdd\f$^?\f$ + [Unary \| Binary]\f$^{0-3}\f$\f$_{>out}\f$ | This pattern is widely used in language models and recommendation models, for example BERT, DLRM, etc. |
| Reduction + [Unary \| Binary]\f$^{0-3}\f$\f$_{>out}\f$ | This pattern is widely used for data processing, for example loss reduction. |
| Unary + Binary\f$^{0-3}\f$\f$_{>out}\f$ | This pattern is widely used in Convolution Neural Networks. |
| Binary + [Unary \| Binary]\f$^{0-3}\f$\f$_{>out}\f$ | This pattern is widely used in Generative Adversarial Networks, for example ParallelWaveGAN. |
| [AvgPool \| MaxPool] + Binary\f$^{0-3}\f$\f$_{>out}\f$ | This pattern is widely used in Convolution Neural Networks. |
| BatchNormInference + ReLU\f$_{>out}\f$ | This pattern is widely used in Convolution Neural Networks, for example DenseNet. |
| Reciprocal + Multiply\f$_{>out}\f$ | N/A |
| Reorder + Add\f$_{>out}\f$ | N/A |

#### Quantized Patterns

| Pattern | Description                  |
|:--------|:-----------------------------|
| Quantize\f$^?\f$ + Dequantize\f$_{>t1}\f$, Dequantize\f$_{>t2}\f$\f$^{0-3}\f$, Dequantize + Convolution\f$_{<t1}\f$ + BiasAdd\f$^?\f$ + [Unary \| Binary\f$_{<t2}\f$]\f$^{0-3}\f$ + Quantize\f$^?\f$\f$_{>out}\f$ | N/A |
| Quantize\f$^?\f$ + Dequantize\f$_{>t1}\f$, Dequantize\f$_{>t2}\f$\f$^{0-3}\f$, Dequantize + ConvTranspose\f$_{<t1}\f$ + BiasAdd\f$^?\f$ + [Unary \| Binary\f$_{<t2}\f$]\f$^{0-3}\f$ + Quantize\f$^?\f$\f$_{>out}\f$ |N/A |
| Quantize\f$^?\f$ + Dequantize\f$_{>t1}\f$, Dequantize\f$_{>t2}\f$\f$^{0-3}\f$, Dequantize + MatMul\f$_{<t1}\f$ + BiasAdd\f$^?\f$ + [Unary \| Binary\f$_{<t2}\f$]\f$^{0-3}\f$ + Quantize\f$^?\f$\f$_{>out}\f$ |N/A |
| Dequantize + [AvgPool \| MaxPool] + Quantize\f$_{>out}\f$ |N/A |
| Dequantize\f$_{>t1}\f$, Dequantize + [AvgPool \| MaxPool] + Add\f$_{<t1}\f$ + Quantize\f$_{>out}\f$ |N/A |
| Dequantize + Reorder + Quantize\f$_{>out}\f$ |N/A |
| Dequantize\f$_{>t1}\f$, Dequantize + Reorder + Add\f$_{<t1}\f$ + Quantize\f$_{>out}\f$ |N/A |

### Training

| Pattern | Description                  |
|:--------|:-----------------------------|
| ConvolutionBackwardWeights + BiasAddBackward\f$_{>out}\f$ | N/A |
| ReLUBackward + BatchNormTrainingBackward\f$_{>out}\f$ |N/A |

All the above fusion patterns are supported by default.

## Aggressive Fusion Patterns
Aggressive fusion patterns also follow the pattern description convention
defined in the [Fusion Patterns](@ref fusion_patterns) section.

@note Aggressive fusion patterns are only supported when
[Graph Compiler](@ref dev_guide_graph_compiler) is enabled.

The following categories will also be used to describe aggressive fusion
patterns.

- ReshapeTranspose = [StaticReshape + StaticTranspose\f$^{1-2}\f$]

- Activation = [ReLU \| Sigmoid \| GELU]

- ActivationBackward = [ReLUBackward \| SigmoidBackward \| GELUBackward]

### Inference

#### Floating Point Patterns

| Pattern | Description                  |
|:--------|:-----------------------------|
| MatMul + [Multiply \| Divide] + Add + Softmax + MatMul + StaticTranspose + Reorder\f$_{>out}\f$ | Multi-head Attention. This pattern is widely used in models containing encoder-decoder structures, for example BERT. |
| ReshapeTranspose\f$_{>t1}\f$, ReshapeTranspose\f$_{>t2}\f$, ReshapeTranspose + MatMul\f$_{<t1}\f$ + [Multiply \| Divide] + Add + Softmax + MatMul\f$_{<t2}\f$ + StaticTranspose + StaticReshape\f$_{>out}\f$ | Multi-head Attention. |
| MatMul + Activation\f$_{>t1}\f$, [MatMul\f$_{<t1}\f$ + Activation\f$_{>t1}\f$]\f$^{0-4}\f$, MatMul\f$_{<t1}\f$ + Activation\f$_{>out}\f$ | Multi-layer Perceptron. This pattern is widely used in recommendation models, for example DLRM. |
| [Convolution + BiasAdd\f$^{?}\f$ + ReLU]\f$^{1-3}\f$ + Convolution + BiasAdd\f$^{?}\f$ + Add + ReLU\f$_{>out}\f$ | Identical Bottleneck. Enabled only in single thread runtime scenario. This pattern is widely used in Convolution Neural Networks, for example ResNet. |
| Convolution + BiasAdd\f$^{?}\f$\f$_{>t1}\f$, [Convolution + BiasAdd\f$^{?}\f$ + ReLU]\f$^{1-3}\f$ + Convolution + BiasAdd\f$^{?}\f$ + Add\f$_{<t1}\f$ + ReLU\f$_{>out}\f$ | Convolutional Bottleneck. Enabled only in single thread runtime scenario. This pattern is widely used in Convolution Neural Networks, for example ResNet. |

#### Quantized Patterns

| Pattern | Description                  |
|:--------|:-----------------------------|
| Dequantize\f$_{>t1}\f$, Dequantize\f$_{>t2}\f$, Dequantize + MatMul\f$_{<t1}\f$ + [Multiply \| Divide] + Add + Softmax + Quantize + Dequantize + MatMul\f$_{<t2}\f$ + StaticTranspose + Reorder + Quantize\f$_{>out}\f$ | Quantized Multi-head Attention. |
| Dequantize + ReshapeTranspose\f$_{>t1}\f$, Dequantize + ReshapeTranspose\f$_{>t2}\f$, Dequantize + MatMul\f$_{<t1}\f$ + [Multiply \| Divide] + Add + Softmax + Quantize + Dequantize + MatMul\f$_{<t2}\f$ + StaticTranspose + StaticReshape + Quantize\f$_{>out}\f$ | Quantized Multi-head Attention. |
| Dequantize\f$_{>t1}\f$, Dequantize + MatMul\f$_{<t1}\f$ + Activation + Quantize\f$_{>t2}\f$, [Dequantize\f$_{>t3}\f$, Dequantize\f$_{<t2}\f$ + MatMul\f$_{<t3}\f$ + Activation + Quantize\f$_{>t2}\f$]\f$^{0-4}\f$, Dequantize\f$_{>t4}\f$, Dequantize\f$_{<t2}\f$ + MatMul\f$_{<t4}\f$ + Activation + Quantize\f$_{>out}\f$ | Quantized Multi-layer Perceptron. |
| Dequantize\f$_{>t2}\f$, Dequantize\f$_{>t3}\f$, [Dequantize\f$_{>t1}\f$, Dequantize + Convolution\f$_{<t1}\f$ + BiasAdd\f$^{?}\f$ + ReLU + Quantize]\f$^{1-3}\f$ + Dequantize + Convolution\f$_{<t2}\f$ + BiasAdd\f$^{?}\f$ + Add\f$_{<t3}\f$ + ReLU + Quantize\f$_{>out}\f$ | Quantized Identical Bottleneck. Enabled only in single thread runtime scenario. |
| [Dequantize\f$_{>t1}\f$, Dequantize + Convolution\f$_{<t1}\f$ + BiasAdd\f$^{?}\f$ + Quantize + Dequantize]\f$_{>t2}\f$, Dequantize\f$_{>t4}\f$, [Dequantize\f$_{>t3}\f$, Dequantize + Convolution\f$_{<t3}\f$ + BiasAdd\f$^{?}\f$ + ReLU + Quantize]\f$^{1-3}\f$ + Dequantize + Convolution\f$_{<t4}\f$ + BiasAdd\f$^{?}\f$ + Add\f$_{<t2}\f$ + ReLU + Quantize\f$_{>out}\f$ | Quantized Convolutional Bottleneck. Enabled only in single thread runtime scenario. |

### Training

| Pattern | Description                  |
|:--------|:-----------------------------|
| Dequantize\f$_{>t1}\f$, Dequantize\f$_{>t2}\f$, Dequantize + MatMul\f$_{<t1}\f$ + [Multiply \| Divide] + Add + Softmax + Quantize + Dequantize + MatMul\f$_{<t2}\f$ + StaticTranspose + Reorder + Quantize\f$_{>out}\f$ | Multi-head Attention Training Forward Pattern. |
| StaticReshape + StaticTranspose\f$_{>t1}\f$ + MatMul + Multiply\f$_{>t2}\f$ + Subtract\f$_{<t3}\f$ + Multiply\f$^{?}\f$ + [Multiply \| Divide]\f$_{>t4}\f$ + MatMul\f$_{>out1}\f$, Multiply\f$_{<t2}\f$ + ReduceSum\f$_{>t3}\f$, MatMul\f$_{<t1,>out2}\f$, MatMul\f$_{<t4,>out3}\f$ | Multi-head Attention Training Backward Pattern. |
| MatMul\f$_{>out1}\f$ + Activation\f$_{>t1,>out2}\f$, [MatMul\f$_{<t1,>out3}\f$ + Activation\f$_{>t1,>out4}\f$]\f$^{0-4}\f$, MatMul\f$_{<t1,>out5}\f$ + Activation\f$_{>out6}\f$ | Multi-layer Perceptron Training Forward Pattern. |
| StaticTranspose\f$^{?}\f$\f$_{>t0}\f$, ActivationBackward\f$_{>t2}\f$ + MatMul\f$_{<t0,>t1}\f$, ReduceSum\f$^{?}\f$\f$_{<t2,>out1}\f$, StaticTranspose\f$^{?}\f$ + MatMul\f$_{<t2,>out2}\f$, [StaticTranspose\f$^{?}\f$\f$_{>t3}\f$, ActivationBackward\f$_{>t4,<t1}\f$ + MatMul\f$_{<t3,>t1}\f$, ReduceSum\f$^{?}\f$\f$_{<t4,>out3}\f$, StaticTranspose\f$^{?}\f$ + MatMul\f$_{<t4,>out4}\f$]\f$^{0-4}\f$, StaticTranspose\f$^{?}\f$\f$_{>t5}\f$, ActivationBackward\f$_{>t6,<t1}\f$ + MatMul\f$_{<t5,>out5}\f$, ReduceSum\f$^{?}\f$\f$_{<t6,>out6}\f$, StaticTranspose\f$^{?}\f$ + MatMul\f$_{<t6,>out7}\f$ | Multi-layer Perceptron Training Backward Pattern. |
| Convolution\f$_{>out1}\f$ + BatchNormForwardTraining\f$_{>out2}\f$ + ReLU\f$_{>out3}\f$ + Convolution\f$_{>out4}\f$ + BatchNormForwardTraining\f$_{>out5}\f$ + ReLU\f$_{>out6}\f$ + Convolution\f$_{>out7}\f$ + BatchNormForwardTraining\f$_{>out8}\f$ + Add + ReLU\f$_{>out9}\f$ | Identical Bottleneck Training Forward Pattern. |
| Convolution\f$_{>out1}\f$ + BatchNormForwardTraining\f$_{>t1,>out2}\f$, Convolution\f$_{>out3}\f$ + BatchNormForwardTraining\f$_{>out4}\f$ + ReLU\f$_{>out5}\f$ + Convolution\f$_{>out6}\f$ + BatchNormForwardTraining\f$_{>out7}\f$ + ReLU\f$_{>out8}\f$ + Convolution\f$_{>out9}\f$ + BatchNormForwardTraining\f$_{>out10}\f$ + Add\f$_{<t1}\f$ + ReLU\f$_{>out11}\f$ | Convolutional Bottleneck Training Forward Pattern. |
| ReLUBackward\f$_{>t1}\f$ + BatchNormTrainingBackward\f$_{>t2,>out1}\f$ + ConvolutionBackwardData + ReLUBackward + BatchNormTrainingBackward\f$_{>t3,>out2}\f$ + ConvolutionBackwardData + ReLUBackward + BatchNormTrainingBackward\f$_{>t4,>out3}\f$ + ConvolutionBackwardData + Add\f$_{<t1,>out4}\f$, ConvolutionBackwardWeights\f$_{<t2,>out5}\f$, ConvolutionBackwardWeights\f$_{<t3,>out6}\f$, ConvolutionBackwardWeights\f$_{<t4,>out7}\f$ | Identical Bottleneck Training Backward Pattern. |
| ReLUBackward\f$_{>t1}\f$ + BatchNormTrainingBackward\f$_{>t2,>out1}\f$ + ConvolutionBackwardData + ReLUBackward + BatchNormTrainingBackward\f$_{>t3,>out2}\f$ + ConvolutionBackwardData + ReLUBackward + BatchNormTrainingBackward\f$_{>t4,>out3}\f$ + ConvolutionBackwardData + Add\f$_{<t6,>out4}\f$, BatchNormTrainingBackward\f$_{<t1,>t5,>out5}\f$ + ConvolutionBackwardData\f$_{>t6}\f$, ConvolutionBackwardWeights\f$_{<t2,>out6}\f$, ConvolutionBackwardWeights\f$_{<t3,>out7}\f$, ConvolutionBackwardWeights\f$_{<t4,>out8}\f$, ConvolutionBackwardWeights\f$_{<t5,>out9}\f$ | Convolutional Bottleneck Training Backward Pattern. |
