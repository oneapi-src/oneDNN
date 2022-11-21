Supported Fusion Patterns {#dev_guide_graph_fusion_patterns}
============================================================

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

Pattern | Description
:-- | :--:
Convolution + BiasAdd\f$^?\f$ + BatchNormInference\f$^?\f$ + [Unary \| Binary]\f$^{0-3}\f$\f$_{>out}\f$ | This pattern is widely used in Convolution Neural Networks, for example ResNet, ResNext, SSD, etc.
ConvTranspose + BiasAdd\f$^?\f$ + [Unary \| Binary]\f$^{0-3}\f$\f$_{>out}\f$ | This pattern is widely used in Generative Adversarial Networks.
Interpolate + [Unary \| Binary]\f$^{0-3}\f$\f$_{>out}\f$ | This pattern is widely used for image processing.
MatMul + BiasAdd\f$^?\f$ + [Unary \| Binary]\f$^{0-3}\f$\f$_{>out}\f$ | This pattern is widely used in language models and recommendation models, for example BERT, DLRM, etc.
Reduction + [Unary \| Binary]\f$^{0-3}\f$\f$_{>out}\f$ | This pattern is widely used for data processing, for example loss reduction.
Unary + Binary\f$^{0-3}\f$\f$_{>out}\f$ | This pattern is widely used in Convolution Neural Networks.
Binary + [Unary \| Binary]\f$^{0-3}\f$\f$_{>out}\f$ | This pattern is widely used in Generative Adversarial Networks, for example ParallelWaveGAN.
[AvgPool \| MaxPool] + Binary\f$^{0-3}\f$\f$_{>out}\f$ | This pattern is widely used in Convolution Neural Networks.
BatchNormInference + ReLU\f$_{>out}\f$ | This pattern is widely used in Convolution Neural Networks, for example DenseNet.
Reciprocal + Multiply\f$_{>out}\f$ | N/A
Reorder + Add\f$_{>out}\f$ | N/A

#### Quantized Patterns

Pattern | Description
:-- | :--:
Quantize\f$^?\f$ + Dequantize\f$_{>t1}\f$, Dequantize\f$_{>t2}\f$\f$^{0-3}\f$, Dequantize + Convolution\f$_{<t1}\f$ + BiasAdd\f$^?\f$ + [Unary \| Binary\f$_{<t2}\f$]\f$^{0-3}\f$ + Quantize\f$^?\f$\f$_{>out}\f$ | N/A
Quantize\f$^?\f$ + Dequantize\f$_{>t1}\f$, Dequantize\f$_{>t2}\f$\f$^{0-3}\f$, Dequantize + ConvTranspose\f$_{<t1}\f$ + BiasAdd\f$^?\f$ + [Unary \| Binary\f$_{<t2}\f$]\f$^{0-3}\f$ + Quantize\f$^?\f$\f$_{>out}\f$ |N/A
Quantize\f$^?\f$ + Dequantize\f$_{>t1}\f$, Dequantize\f$_{>t2}\f$\f$^{0-3}\f$, Dequantize + MatMul\f$_{<t1}\f$ + BiasAdd\f$^?\f$ + [Unary \| Binary\f$_{<t2}\f$]\f$^{0-3}\f$ + Quantize\f$^?\f$\f$_{>out}\f$ |N/A
Dequantize + [AvgPool \| MaxPool] + Quantize\f$_{>out}\f$ |N/A
Dequantize\f$_{>t1}\f$, Dequantize + [AvgPool \| MaxPool] + Add\f$_{<t1}\f$ + Quantize\f$_{>out}\f$ |N/A
Dequantize + Reorder + Quantize\f$_{>out}\f$ |N/A
Dequantize\f$_{>t1}\f$, Dequantize + Reorder + Add\f$_{<t1}\f$ + Quantize\f$_{>out}\f$ |N/A

#### Training

Pattern | Description
:-- | :--:
ConvolutionBackwardWeights + BiasAddBackward\f$_{>out}\f$ | N/A
ReLUBackward + BatchNormTrainingBackward\f$_{>out}\f$ |N/A

All the above fusion patterns are supported by default.
