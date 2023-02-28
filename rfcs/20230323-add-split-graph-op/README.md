# Add Split Operation in Graph API

## Motivation

`Split` operation splits an input tensor into pieces of the specific length
along some axis, which is opposite to the operation of `concat`. `Split`
operation is supported by different deep learning frameworks and toolkit, eg.
 PyTorch [1], TensorFlow [2], OpenVINO [3], and ONNX [4]. The `split`+`concat` 
 pattern is more common in most models and takes up a large amount of computing 
 time. However, since the graph API does not support `Split` op, possible 
 optimization is limited. So we propose to add the `Split` op into graph API.


## Proposal

The proposal is to add `Split` operation to the operation set of Graph API.

### Operations

`Split` operation splits one single input tensor (`src`) and generates multiple
outputs (`dst_i`) of the specific length along a scalar `axis` attribute. It
produces multiple output tensors based on `num_splits` or `size_splits` attribute.
The input and output tensors should have the same data type and support floating
point data types (f32, bf16, and f16).

#### Proposed Attributes

`Split` operation supports three attributes: `axis`, `num_splits` and `size_splits`.
Both `axis` and `num_splits` should be an interger value. `size_splits` is a 1-D
Tensor (or list). The input data is split into `num_splits` or len(`size_splits`)
elements. The shape of the i-th element has the same size as the value except along
dimension axis where the size is determined by `num_splits` or `size_splits` attribute.

Attribute Name | Description | Value Type |Supported Values | Required or Optional
-- | -- | --| --|--
`axis` | specifies which dimension to split along. |s64 |A s64 value in the range of [-r, r-1] where r = rank(src)  | Required
`num_splits` | number of outputs into which the input tensor data will be split along axis dimension. |s64 |A s64 value in the range of[1, src.dims[axis]]  | Optional
`size_splits` |  list of sizes for each output alone axis dims. |s64 |A s64 list containing positive values, none is default  | Optional

@note:

- Either `num_splits` or `size_splits` should be provided. When `num_splits` is used,`size_splits` will be ignored.
- The dimension of input tensor data shape along `axis` must be evenly divisible by `num_splits` attribute.
- The sum of the values in `size_splits` must be equal to the src.dims[`axis`].

#### Proposed Inputs and Outputs

There will be 1 input tensor and multiple output tensors for the proposed `Split` operation.

##### Inputs

| Index | Argument Name | Required or Optional |
| ----- | ------------- | -------------------- |
| 0     | `src`        | Required             |

##### Outputs

| Index | Argument Name | Required or Optional |
| ----- | ------------- | -------------------- |
| 0     | `dst_i`    | Required             |

**@note** The i-th output has the same shape as src input tensor except for dimension
along `axis` which is src.dims[`axis`]/`num_splits` or `size_splits`[i].

Example1:

```
    src shape: src [6, 12, 10, 24]
    attribute:
            num_splits = 3
            axis = 1
    dst_i shape:
            dst_0 [6, 4, 10, 24]
            dst_1 [6, 4, 10, 24]
            dst_2 [6, 4, 10, 24]
```

Example2:

```
    src shape: src [6, 12, 10, 24]
    attribute:
            size_splits = [5, 6, 6, 7]
            axis = 3
    dst_i shape:
            dst_0 [6, 12, 10, 5]
            dst_1 [6, 12, 10, 6]
            dst_2 [6, 12, 10, 6]
            dst_3 [6, 12, 10, 7]
```

### API changes

One new operation kind value will be added to `dnnl_graph_op_kind_t` and `class
op::kind::`.

```c
// dnnl_graph_types.h

typedef enum {
    // ...
    dnnl_graph_op_split,
} dnnl_graph_op_kind_t;

// dnnl_graph.hpp
// class op
enum class kind {
    // ...
    Split = dnnl_graph_op_split,
};
```

### OP definition

```c

DNNL_GRAPH_OP_SCHEMA(Split, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::optional)
                .set_num_inputs(1)
                .set_num_outputs(std::set<size_t>({1, MAX}))
                .set_input(0, "data", "input tensor", "T")
                .set_output(0, "output", "output tensor", "T")
                .set_attr(op_attr::axis,
                        "specifies which dimension to split along", true,
                        attribute_kind::i)
                .set_attr(op_attr::num_splits,
                        "specifies number of outputs into which the input tensor data will be split along axis", false,
                        attribute_kind::i)
                .set_attr(op_attr::size_splits,
                        "specifies shape value for each output along axis", false,
                        attribute_kind::is)
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_split_output_shape))

```

### Backend support

Currently the proposal is to provide support through compiler backend.

### Documentation changes

Corresponding operation documents should be added to `doc/graph/operations/`.

## References

1. <https://pytorch.org/docs/stable/generated/torch.split.html>
2. <https://www.tensorflow.org/api_docs/python/tf/split>
3. <https://docs.openvino.ai/latest/openvino_docs_ops_movement_Split_1.html>
4. <https://onnx.ai/onnx/operators/onnx__Split.html>
