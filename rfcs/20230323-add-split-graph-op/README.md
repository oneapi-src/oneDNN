# Add Split Operation in Graph API

## Background
### Motivation

`Split` operation splits an input tensor into pieces of the specific length
along some axis, which is opposite to the operation of `concat`. `Split`
operation is supported by different deep learning frameworks and toolkit, eg.
 PyTorch [1], TensorFlow [2], OpenVINO [3], and ONNX [4]. The `split`+`concat` 
 pattern is more common in most models and takes up a large amount of computing 
 time. However, since the graph API does not support `Split` op, possible 
 optimization is limited. So we propose to add the `Split` op into graph API.

### Investigation

The `Split` operation is supported by most frameworks. To split an input tensor, 
we need to specify the split dimension according to the `axis` for input tensor, 
and then output multiple small chunks. But as for how to split along specific 
dimensions, the definition of frameworks is slightly different, which usually 
includes three rules. 
- Specify the number of output tensors through an integer
- Uniformly specify the size of each output chunk through an integer, last chunk
will be smaller if the tensor size along `axis` is not divisible by this integer.
- Specify the specific size of each chunk through a list. 

The size of all output chunks in the first two rules is the same but may differ 
in the third rule. Different frameworks support one or more of the above three 
rules in different ways. For example, TensorFlow supports the first and third 
rules through `num_or_sections_split`, while Pytorch supports the second and third 
rules through `split_size_or_sections`. In addition, for different rules, the 
definitions of different frameworks are also different, for example, some allow 
divisibility while others do not. The following are the brief definitions of 
`Split` in four common frameworks. For details, please refer to the link at the 
end of this RFC.

#### operations used in Frameworks and toolkit

| Tensorflow | Argument Name         | Description                                                                                       |
|------------|-----------------------|---------------------------------------------------------------------------------------------------|
| input      | `value`               | input data                                                                                        |
| input      | `axis`                | dimension along which to split the tensor.                                                        |
| input      | `num_or_sections_split ` | number of splits(int) or list of sizes for each output tensor(list(int)).                         |
| input      | `num`                 | Optional, specify the number of outputs when it cannot be inferred from the shape of sections_split. |
| output     | `List[Tensor]`        | output data                                                                                       |

| XLA HLO[5] | Argument Name         | Description                                                                                       |
|------------|-----------------------|---------------------------------------------------------------------------------------------------|
| input      | `operand`             | input data                                                                                        |
| input      | `start_indices`       | List of N integers containing the starting indices of the slice for each dimension.               |
| input      | `limit_indices`       | List of N integers containing the ending indices (exclusive) for the slice for each dimension.    |
| input      | `strides`             | List of N integers that decides the input stride of the slice.                                    |
| output     | `output`              | output data                                                                                       |

@note: XLA doesn't directly support `Split`, but decomoses `Split` into multiple
`Slice` operations. Above is the definition of `Slice` op in XLA.

| Pytorch | Argument Name             | Description                                                             |
|---------|---------------------------|-------------------------------------------------------------------------|
| input   | `tensor`                  | input data                                                              |
| input   | `axis`                    | dimension along which to split the tensor.                              |
| input   | `split_size_or_sections ` | size of a single chunk(int) or list of sizes for each chunk.(list(int)) |
| output  | `List[Tensor]`            | output data                                                             |

| ONNX      | Argument Name  | Description                                          |
|-----------|----------------|------------------------------------------------------|
| input     | `input`        | input data                                           |
| input     | `split `       | tensor(int64): Optional size of each output.         |
| attribute | `axis`         | dimension along which to split the tensor.           |
| attribute | `num_outputs ` | Number of outputs to split parts of the tensor into. |
| output    | `outputs `     | output data                                          |

| OpenVINO  | Argument Name | Description                                          |
|-----------|---------------|------------------------------------------------------|
| input     | `input`       | input data                                           |
| input     | `axis`        | dimension along which to split the tensor.           |
| attribute | `num_splits ` | Number of outputs to split parts of the tensor into. |
| output    | `outputs `    | output data                                          |

## Proposal

To support the request from framework, we need to add `Split` operation to the 
operation set of Graph API, with 2 options below.

### Option1:

`Split` operation supports three attributes: `axis`, `num_splits` and 
`sections_split`. Both `axis` and `num_splits` should be an interger value. 
`sections_split` is a 1-D Tensor (or list). The input data is split into 
`num_splits` or len(`sections_split`) elements. The shape of the i-th element has 
the same size as the value except along dimension `axis` where the size is 
determined by `num_splits` or `sections_split` attribute. Note that Split operation
is not guaranteed to return a view or a copy of original input tensor.

| Split     | Argument Name    | Description                                                                            | Required or Optional | Value Type    |
|-----------|------------------|----------------------------------------------------------------------------------------|----------------------|---------------|
| input     | `src`            | input data                                                                             | Required             | f32,f16,bf16* |
| attribute | `axis`           | Specifies which dimension to split along.                                              | Required             | s64           |
| attribute | `num_splits`     | number of outputs into which the input tensor data will be split along axis dimension. | optional             | s64           |
| attribute | `sections_split` | list of sizes for each output alone axis dims.                                         | optional             | s64           |
| output    | `dst_i`          | output data                                                                            | Required             | f32,f16,bf16* |

@note: 
- input and outputs must have the same data type.
-The i-th output has the same shape as src input tensor except for dimension
along `axis` which is src.dims[`axis`]/`num_splits` or `sections_split`[i].
- Either `num_splits` or `sections_split` should be provided. When `num_splits` is 
used,`sections_split` will be ignored.
- If `num_splits` is specified, the dimension of input tensor data shape along 
`axis` must be evenly divisible by `num_splits`.
- If `sections_split` is specified, the sum of the values in `sections_split` must be 
equal to the src.dims[`axis`].

Pros.
1. The definition is simple and we only focus on the static shape support.

Cons.
1. We may need to add a dynamic split in the future when graph API support dynamic shape.
2. Not suitable for Pytorch integration to map `split_size_or_sections ` of `Split`
in Pytorch to `num_splits`.

### Option2: 

`Split` operation supports four attributes: `axis`, `num_splits` , `size_split`
and `sections_split`. `axis`, `num_splits` and `size_split` should be an interger 
value. `sections_split` is a 1-D Tensor (or list). The input data is split into 
`num_splits` or len(`sections_split`) elements. The shape of the i-th element has 
the same size as the value except along dimension `axis` where the size is 
determined by one of `num_splits`, `size_split`, `sections_split`. Note that Split
operation is not guaranteed to return a view or a copy of original input tensor.

| Split     | Argument Name    | Description                                                                            | Required or Optional | Value Type    |
|-----------|------------------|----------------------------------------------------------------------------------------|----------------------|---------------|
| input     | `src`            | input data                                                                             | Required             | f32,f16,bf16* |
| attribute | `axis`           | Specifies which dimension to split along.                                              | Required             | s64           |
| attribute | `num_splits`     | number of outputs into which the input tensor data will be split along axis dimension. | optional             | s64           |
| attribute | `size_split`     | size for single output tensor along axis dimension.                                    | optional             | s64           |
| attribute | `sections_split` | list of sizes for each output alone axis dimension.                                    | optional             | s64           |
| output    | `dst_i`          | output data                                                                            | Required             | f32,f16,bf16* |

@note:
- input and outputs must have the same data type.
- One of `num_splits`, `size_split`, `sections_split` must be provided. The priority
of the three attributes is `num_splits`>`size_split`>`sections_split`. When the 
attribute with higher priority is used, the rest will be ignored.
- If `num_splits` is specified, the dimension of input tensor data shape along 
`axis` must be evenly divisible by `num_splits`.
- If `size_split` is specified, the input tensor will be split into equally sized
chunks (if possible). Last chunk will be smaller if the tensor size along `axis`
is not divisible by `size_split`.
- If `sections_split` is specified, the sum of the values in `sections_split` must be 
equal to the src.dims[`axis`].

Pros.
1. Compatible with both pytorch and tensorflow. Easy to map frameworks' split into
graph API.

Cons.
1. More attributes may lead to misunderstanding of users.

### Backend support

Currently the proposal is to provide support through compiler backend.

### Documentation changes

Corresponding operation documents should be added to `doc/graph/operations/`.

## References

1. <https://pytorch.org/docs/stable/generated/torch.split.html>
2. <https://www.tensorflow.org/api_docs/python/tf/split>
3. <https://docs.openvino.ai/latest/openvino_docs_ops_movement_Split_1.html>
4. <https://onnx.ai/onnx/operators/onnx__Split.html>
5. <https://www.tensorflow.org/xla/operation_semantics#slice>
