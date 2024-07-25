# Support Unsqueeze and Expand in Graph API

## Introduction & Motivation

As mentioned in the GQA RFC, if we choose to support option1, we need support
StaticUnsqueeze and StaticExpand operation in Graph API.

## Proposal

### StaticUnsqueeze
In the frameworks, there is `unsqueeze` used to add dimensions of size 1 to the
input tensor. We propose to define `StaticUnsqueeze` to map the similar
`unsqueeze` in the frameworks.[[#1]][1] [[#2]][2] [[#3]][3] [[#4]][4]

| Framework      | TensorFlow | PyTorch   | ONNX     | OpenVINO   |
|----------------|------------|-----------|----------|------------|
| op             | expand_dims| unsqueeze |Unsqueeze |Unsqueeze   |
| input          | src        | src       | data     | src        |
| input          | axis       | dim       | axes(a tensor of int)     | dim|  
| output         | dst        | dst       | dst      | dst        |

These ops in the framework are the same with only a slight difference. Only for
the ONNX `Unsuqeeze`, the second input is a tensor of int. It supports a list of
dimensions to be inserted. Both these `axis` and `dim` are in the range
[-input.dim() - 1, input.dim()].

Based on the definitions of these frameworks, we define the following operation
`StaticUnsqueeze` to map these ops.

| StaticUnsqueeze | Argument Name    | Required or Optional        | Data Type     |
|-----------|------------------|-----------------------------|---------------|
| input     | `src`            | Required                    | f32,f16,bf16* |
| attribute | `dim`            | Required                    | s64           |
| output    | `dst`            | Required                    | f32,f16,bf16* |

**Detailed description**:

It return the output with a dimension of size one inserted at the specified
`dim` position. Unsqueeze operation can return a view or copy of `src`.

`dim`: the index at which to insert the singleton dimension, which should also be in
the range `[-src.dim() - 1, src.dim()]`

For example:
when `src`'s shape is \[4\], 
1. `dim` = 0, the `dst`'s shape is [1,4]
2. `dim` = 1, the `dst`'s shape is [4,1]

### StaticExpand

In the frameworks, there are some operations similar to expand semantics. For
example, `expand, expand_as, repeat, repeat_interleave` in PyTorch, `broadcast1,
broadcast3` in openvino, `Expand-13, Expand-8` in ONNX and
etc.[[#5]][5] [[#6]][6] [[#7]][7] [[#8]][8] These OP definitions are not quite the
same, but they can all implement similar expand functions.

However, there is no operation in Graph API corresponding to the semantics in
the framework. So we can add an operation `StaticExpand` to map the similar op
from framework. It replicates data on the input to fit a given shape.

#### option1

| StaticExpand | Argument Name    | Required or Optional        | Data Type     |
|-----------|------------------|-----------------------------|---------------|
| input     | `src`            | Required                    | f32,f16,bf16* |
| attribute | `target_shape`   | Required                    | s64           |
| output    | `dst`            | Required                    | f32,f16,bf16* |

**Detailed description**:

`Expand` takes the first tensor `src` builds a new tensor with shape matching the
attribute `target_shape`. `target_shape` is a 1D integer tensor that represents
required shape of the output. It requires thar the rank of input and output are
equal.

Pros:

1. This definition is simple and easy to understand. It is convenient to map
   pytorch `unsqueeze`.

Cons:

1. It require the input's rank is equal output's rank. Don't support some op
   function.

#### option2
Add an attribute `axes_mapping` based on the option1.

| StaticExpand | Argument Name    | Required or Optional        | Data Type     |
|-----------|------------------|-----------------------------|---------------|
| input     | `src`            | Required                    | f32,f16,bf16* |
| attribute | `target_shape`   | Required                    | s64           |
| attribute | `axes_mapping`   | Optional*                     | s64           |
| output    | `dst`            | Required                    | f32,f16,bf16* |

**Detailed description**:

The attribute `axes_mapping` is a tensor of int. If this attribute is not set,
it is the same as option 1. If the attribute is set, the size of `axis_mapping`
should match the rank of input data tensor, so all axes from data tensor should
be mapped to axes of the output. For example, `axes_mapping = [1]`enables
broadcasting of a tensor with shape `[C]` to shape `[N,C,H,W]` by replication of
initial tensor along dimensions 0, 2 and 3. Another example is broadcasting of
tensor with shape `[H,W]` to shape `[N,H,W,C]` with `axes_mapping = [1, 2]`.

Pros:

1. It solve the option1's cons.

Cons:

1. Need add an attribute, which increases the difficulty of understanding. Not
   very useful in practice, the option1's cons can be solved by `unsqueeze` op.

## References

1. https://www.tensorflow.org/api_docs/python/tf/expand_dims
2. https://pytorch.org/docs/stable/generated/torch.unsqueeze.html
3. https://onnx.ai/onnx/operators/onnx__Unsqueeze.html#l-onnx-doc-unsqueeze
4. https://docs.openvino.ai/2022.3/openvino_docs_ops_shape_Unsqueeze_1.html
5. https://pytorch.org/docs/stable/generated/torch.Tensor.expand.html
6. https://docs.openvino.ai/2022.3/openvino_docs_ops_movement_Broadcast_3.html
7. https://onnx.ai/onnx/operators/onnx__Expand.html
8. https://www.tensorflow.org/api_docs/python/tf/broadcast_to

[1]: https://www.tensorflow.org/api_docs/python/tf/expand_dims
[2]: https://pytorch.org/docs/stable/generated/torch.unsqueeze.html
[3]: https://onnx.ai/onnx/operators/onnx__Unsqueeze.html#l-onnx-doc-unsqueeze
[4]: https://docs.openvino.ai/2022.3/openvino_docs_ops_shape_Unsqueeze_1.html
[5]: https://pytorch.org/docs/stable/generated/torch.Tensor.expand.html
[6]: https://docs.openvino.ai/2022.3/openvino_docs_ops_movement_Broadcast_3.html
[7]: https://onnx.ai/onnx/operators/onnx__Expand.html
[8]: https://www.tensorflow.org/api_docs/python/tf/broadcast_to
