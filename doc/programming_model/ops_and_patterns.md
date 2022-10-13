# Operators and Fusion Patterns {#dev_guide_ops_and_patterns}

## Operators

Supported operation refers to operation which can be converted to oneDNN Graph
OP and thus can be part of oneDNN Graph partition. The alpha release supports the
following operations as part of Opset defined in oneDNN Graph Spec. It supports
FP32/FP16/BF16/S8/U8 data type.  For complete OP definition, please refer to
[documentation](../operations/).

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

### 1. Describing fusion pattern

Fusion pattern describes a graph partition which oneDNN Graph takes as input
and generates optimized code. The pattern is described using oneDNN Graph
OP names with the following convention.

`"+"` describes a chain of two OPs. The preceeding OP produces an output tensor,
which is consumed by the following OP as its first operand.

`"[]"` describes a component of the overall pattern description. For example,
it could include a subgraph or all the OP choices within the bracket.

`"|"` describes choices of multiple operations, say A+[B|C] means the graph partition
contains A followed by B or C.

`","` describes a graph composed of multiple subgraphs, each subgraph marks its output
tensor explicitly, which is consumed by other subgraphs.

`Superscript` denotes the numbers of repetition pattern. For example, A+[B|C]<sup>3</sup>
means the graph partition contains A followed by 3 OPs, each of them is either
B or C. The superscript could be a range of number meaning allowing a range of
repetition. If the range is between 0 and 1, we use superscript `"?"`.

`Subscript` denotes the input and output tensors which need to explicitly mark the
producer and consumer relation within one graph partition. For example,
A<sub>>t1</sub>+B+C<sub><t1</sub> refers
to the pattern started with A followed by B and C, and C takes an implicit input
tensor from B and an extra tensor t1 output from A. `">"` refers to the output tensor,
and `"<"` for input tensor.  Input and output tensor between neighbor ops are not
explicitly marked, for example, B consumes t1 implicitly in the example above.

Subscript `"out"` marks the output tensor of a certain OP to be the output of
a graph partition. For example, in A<sub>>t1</sub>+B<sub>>out</sub>+C<sub><t1,>out</sub>
B’s output and C’s output are marked as output tensors.

Subscript `"in"` marks the input tensor of a certain OP to be the input of a graph
partition. For example, in A<sub><in1</sub>+B<sub><in1</sub> A’s input and B's
second input are graph partition input, and they share the same input tensor in1.
Most input tensors of a graph partition are not explicitly marked.
For example, the input tensors of the first OP are implicitly regarded as graph
partition inputs. Besides, for input tensors of other OPs, if they are not produced
by any proceeding OPs, they are regarded as implicit graph partition inputs.
In the example A<sub>>t1</sub>+B+C<sub><t1</sub>, A’s inputs are regarded as implicit
graph partition inputs, and if B is a binary operation, the second input tensor
is an implicit graph partition input.

### 2. Post-ops fusion pattern

The following categories will be used in describing post-ops fusion pattern.

Unary = [Abs | Clamp | Elu | Exp | GELU | HardSwish | LeakyReLU |
Log | Sigmoid | SoftPlus | Pow | ReLU | Round | Sqrt | Square | Tanh]

Binary = [Add | Divide | Maximum | Minimum | Multiply | Subtract]

Reduction = [ReduceL1 | ReduceL2 | ReduceMax | ReduceMean | ReduceMin |
ReduceProd | ReduceSum]

#### 2.1 Inference

##### 2.1.1 Floating Point Patterns

- Convolution Post-ops
  - Pattern: Convolution + BiasAdd<sup>?</sup> +
    BatchNormInference<sup>?</sup> + [Unary | Binary]<sup>0-3</sup><sub>>out</sub>
  - Description: this pattern is widely used in Convolution Neural Networks,
    i.e. ResNet, ResNext, SSD, etc.
- ConvTranspose Post-ops
  - Pattern: ConvTranspose + BiasAdd<sup>?</sup> +
    [Unary | Binary]<sup>0-3</sup><sub>>out</sub>
  - Description: this pattern is widely used in Generative
    Adversarial Networks.
- Interpolate Post-ops
  - Pattern: Interpolate + [Unary | Binary]<sup>0-3</sup><sub>>out</sub>
  - Description: this pattern is widely used for image processing.
- MatMul Post-ops
  - Pattern: MatMul + BiasAdd<sup>?</sup> + [Unary | Binary]<sup>0-3</sup><sub>>out</sub>
  - Description: this pattern is widely used in language models and
    recommendation models, i.e. BERT, DLRM, etc.
- Reduction Post-ops
  - Pattern: Reduction + [Unary | Binary]<sup>0-3</sup><sub>>out</sub>
  - Description: this pattern is widely used for data processing, i.e.
    loss reduction.
- Unary Post-ops
  - Pattern: Unary + Binary<sup>0-3</sup><sub>>out</sub>
  - Description: this pattern is widely used in Convolution Neural Networks.
- Binary Post-ops
  - Pattern: Binary + [Unary | Binary]<sup>0-3</sup><sub>>out</sub>
  - Description: this pattern is widely used in Generative Adversarial Networks,
    i.e. ParallelWaveGAN.
- Pooling Post-ops
  - Pattern: [AvgPool | MaxPool] + Binary<sup>0-3</sup><sub>>out</sub>
  - Description: this pattern is widely used in Convolution Neurual Networks.
- Batch Normalization Post-ops
  - Pattern: BatchNormInference + ReLU<sub>>out</sub>
  - Description: this pattern is widely used in Convolution Neurual Networks,
    i.e. DenseNet.
- Misc Post-ops
  - Pattern: Reciprocal + Multiply<sub>>out</sub>
  - Pattern: Reorder + Add<sub>>out</sub>

##### 2.1.2 Quantized Patterns

- Quantized Convolution Post-ops
  - Pattern: Quantize<sup>?</sup> + Dequantize<sub>>t1</sub>,
    Dequantize<sub>>t2</sub><sup>0-3</sup>, Dequantize +
    Convolution<sub><t1</sub> +
    BiasAdd<sup>?</sup> + [Unary | Binary<sub><t2</sub>]<sup>0-3</sup> + Quantize<sup>?</sup><sub>>out</sub>
- Quantized ConvTranspose Post-ops
  - Pattern: Quantize<sup>?</sup> + Dequantize<sub>>t1</sub>,
    Dequantize<sub>>t2</sub><sup>0-3</sup>, Dequantize +
    ConvTranspose<sub><t1</sub> +
    BiasAdd<sup>?</sup> + [Unary | Binary<sub><t2</sub>]<sup>0-3</sup> + Quantize<sup>?</sup><sub>>out</sub>
- Quantized MatMul Post-ops
  - Pattern: Quantize<sup>?</sup> + Dequantize<sub>>t1</sub>,
    Dequantize<sub>>t2</sub><sup>0-3</sup>, Dequantize + MatMul<sub><t1</sub> +
    BiasAdd<sup>?</sup> + [Unary | Binary<sub><t2</sub>]<sup>0-3</sup> + Quantize<sup>?</sup><sub>>out</sub>
- Quantized Unary Post-ops
  - Pattern: Dequantize + ReLU + Quantize<sub>>out</sub>
- Quantized Pooling Post-ops
  - Pattern: Dequantize + [AvgPool | MaxPool] + Quantize<sub>>out</sub>
  - Pattern: Dequantize<sub>>t1</sub>, Dequantize + [AvgPool | MaxPool] +
    Add<sub><t1</sub> + Quantize<sub>>out</sub>
- Misc Quantized Post-ops
  - Pattern: Dequantize + Reorder + Quantize<sub>>out</sub>
  - Pattern: Dequantize<sub>>t1</sub>, Dequantize + Reorder +
    Add<sub><t1</sub> + Quantize<sub>>out</sub>

#### 2.2 Training

- ConvolutionBackpropFilters Post-ops
  - Pattern: ConvolutionBackpropFilters + BiasAddBackprop<sub>>out</sub>
- Misc Post-ops
  - Pattern: ReLUBackprop + BatchNormTrainingBackprop<sub>>out</sub>

All the post-ops fusion patterns are supported by default.

### 3. Aggressive fusion pattern

The following category will be used to describe aggressive fusion pattern.

- ReshapeTranspose = [StaticReshape + StaticTranspose<sup>1-2</sup>]

- Activation = [ReLU | Sigmoid | GELU]

- ActivationBackprop = [ReLUBackprop | SigmoidBackprop | GELUBackprop]

#### 3.1 Inference

##### 3.1.1 Floating Point Patterns

- MHA (Multi-head Attention)
  - Pattern1: MatMul + [Multiply | Divide] + Add + Softmax + MatMul +
    StaticTranspose + Reorder<sub>>out</sub>
  - Pattern2: ReshapeTranspose<sub>>t1</sub>, ReshapeTranspose<sub>>t2</sub>,
    ReshapeTranspose + MatMul<sub><t1</sub> + [Multiply | Divide] + Add +
    Softmax + MatMul<sub><t2</sub> + StaticTranspose +
    StaticReshape<sub>>out</sub>

    <img src="../images/oneDNN_graph_MHA_fp32_patterns.png" width = "85%"
    height = "85%" alt="MHA Floating Point Patterns"/>

  - Description: these patterns are used in various BERT models. The `Reorder`
    op in Pattern1 could change the physical layout of the input tensor.
    It could be translated from `torch.contiguous`.
- MLP (Multi-layer Perceptron)
  - Pattern: MatMul + Activation<sub>>t1</sub>, [MatMul<sub><t1</sub> +
    Activation<sub>>t1</sub>]<sup>0-4</sup>, MatMul<sub><t1</sub> +
    Activation<sub>>out</sub>
  - Description: this pattern is composed of multiple layers of MatMul +
    post-ops. It is used in recommendation model, e.g. DLRM.

##### 3.1.2 Quantized Patterns

- Quantized MHA
  - Pattern1: Dequantize<sub>\>t1</sub>, Dequantize<sub>\>t2</sub>,
    Dequantize + MatMul<sub>\<t1</sub> + [Multiply | Divide] + Add + Softmax +
    Quantize + Dequantize + MatMul<sub>\<t2</sub> + StaticTranspose + Reorder +
    Quantize<sub>>out</sub>
  - Pattern2: Dequantize + ReshapeTranspose<sub>\>t1</sub>,
    Dequantize + ReshapeTranspose<sub>\>t2</sub>, Dequantize +
    MatMul<sub>\<t1</sub> + [Multiply | Divide] + Add + Softmax + Quantize +
    Dequantize + MatMul<sub>\<t2</sub> + StaticTranspose + StaticReshape +
    Quantize<sub>>out</sub>

    ![MHA Quantized Patterns](../images/oneDNN_graph_MHA_quantized_patterns.png)

  - Descriptions: this pattern is used in quantized BERT models. The `Reorder`
    op in Pattern1 could change the physical layout of the input tensor.
    It could be translated from `torch.contiguous`.

- Quantized MLP
  - Pattern:

    Dequantize<sub>>t1</sub>, Dequantize + MatMul<sub><t1</sub> +
    Activation + Quantize<sub>>t2</sub>,

    [Dequantize<sub>>t3</sub>,
    Dequantize<sub><t2</sub> + MatMul<sub><t3</sub> + Activation +
    Quantize<sub>>t2</sub>]<sup>0-4</sup>,

    Dequantize<sub>>t4</sub>,
    Dequantize<sub><t2</sub> + MatMul<sub><t4</sub> + Activation +
    Quantize<sub>>out</sub>
  - Description: this pattern is used in quantized recommendation models,
    e.g. int8 DLRM.

#### 3.2 Training

- MHA
  - Forward Pattern: MatMul + [Multiply | Divide] + Add + Softmax
    <sub>>out1</sub> + Multiply<sub>>out2</sub> + MatMul + StaticTranspose +
    StaticReshape<sub>>out3</sub>
  - Backward Pattern: StaticReshape + StaticTranspose<sub>>t1</sub> + MatMul +
    Multiply<sub>>t2</sub> + Subtract<sub><t3</sub> + Multiply<sup>?</sup> +
    [Multiply | Divide]<sub>>t4</sub> + MatMul<sub>>out1</sub>,
    Multiply<sub><t2</sub> + ReduceSum<sub>>t3</sub>,
    MatMul<sub><t1,>out2</sub>, MatMul<sub><t4,>out3</sub>

    ![MHA Training Patterns](../images/oneDNN_graph_MHA_training_patterns.png)

  - Description: this pattern is used in BERT training cases. The pattern is
    similar to MHA inference, except for an extra Multiply op after Softmax.

- MLP
  - Forward Pattern:
    MatMul<sub>>out1</sub> + Activation<sub>>t1,>out2</sub>,
    [MatMul<sub><t1,>out3</sub> + Activation<sub>>t1,>out4</sub>]<sup>0-4</sup>,
    MatMul<sub><t1,>out5</sub> + Activation<sub>>out6</sub>
  - Backward Pattern:

    StaticTranspose<sup>?</sup><sub>>t0</sub>,
    ActivationBackprop<sub>>t2</sub> + MatMul<sub><t0,>t1</sub>,
    ReduceSum<sup>?</sup><sub><t2,>out1</sub>,
    StaticTranspose<sup>?</sup> + MatMul<sub><t2,>out2</sub>,

    [StaticTranspose<sup>?</sup><sub>>t3</sub>,
    ActivationBackprop<sub>>t4,<t1</sub> + MatMul<sub><t3,>t1</sub>,
    ReduceSum<sup>?</sup><sub><t4,>out3</sub>,
    StaticTranspose<sup>?</sup> + MatMul<sub><t4,>out4</sub>]<sup>0-4</sup>,

    StaticTranspose<sup>?</sup><sub>>t5</sub>,
    ActivationBackprop<sub>>t6,<t1</sub> + MatMul<sub><t5,>out5</sub>,
    ReduceSum<sup>?</sup><sub><t6,>out6</sub>,
    StaticTranspose<sup>?</sup> + MatMul<sub><t6,>out7</sub>

    ![MLP Training Patterns](../images/oneDNN_graph_MLP_training_patterns.png)

    MLP Training Patterns: MLP Forward Pattern (left); MLP Backward Pattern
    (right). Solid orange lines denote the connection between successive
    repetition layers. Inputs/outputs denoted with black solid lines are
    required in each layer of repetition. Inputs denoted with blue solid lines
    are required only in the first repetition layer, and outputs in blue
    exist only in the last repetition layer.

  - Description: this pattern is used in recommendation models, e.g. DLRM.

Aggressive patterns are supported by [oneDNN Graph Compiler](https://github.com/oneapi-src/oneDNN/tree/dev-graph/doc#onednn-graph-compiler).
