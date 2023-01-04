# Add HardSigmoid Operation in Graph API

## Motivation

HardSigmoid is a common element-wise activation function used in deep neural
networks like MobileNet series where the operation is usually called after a
Convolution layer and the pattern repeats multiple times in the models.
HardSigmoid operations is supported by different deep learning frameworks and
toolkit, eg. PyTorch [1], TensorFlow [2], OpenVINO [3], and ONNX [4].
HardSigmoid operation is also supported in oneDNN primitive API through the
algorithm `hardsigmoid` of eltwise primitive [5] since v2.7 release.

Framework developers are requesting to support HardSigmoid operation and it's
fusions through oneDNN Graph API to improve the performance of framework models.
It's also important for Graph API to expose this new functionality of eltwise
primitive.

## Proposal

The proposal is to add HardSigmoid and HardSigmoidBackward operation to the
operation set of Graph API.

### Formulas

 The formulas of the operations should follow the existing definition of
`hardsigmoid` in eltwise primitive [5]. The forward formula is defined as
follows:

$$d = \text{max}(0, \text{min}(1, \alpha s + \beta))$$

And the corresponding backward formula is defined as:

$$ds = \begin{cases} dd \cdot \alpha & \text{if}\ 0 < \alpha s + \beta < 1 \\ 0 & \text{otherwise}\ \end{cases}$$

### Operations

HardSigmoid operation takes one single input tensor (`src`) and generates one
single output tensor (`dst`). The input and output tensors should have the same
data type and support floating point data types (f32, bf16, and f16).
HardSigmoid operation supports two required attributes: alpha and beta. Both
alpha and beta should be a single f32 value.

HardSigmoidBackward operation takes two input tensors (`src` and `diff_dst`) and
generates one output tensor (`diff_src`). Same as the forward operation,
HardSigmoidBackward input and output tensors should have the same data type and
support floating point data types (f32, bf16, and f16). HardSigmoidBackward
operation supports two required attributes: alpha and beta. Both alpha and beta
should be a single f32 value.

### API changes

Two new operation kind values will be added to `dnnl_graph_op_kind_t` and `class
op::kind::`.

```c
// dnnl_graph_types.h

typedef enum {
    // ...
    dnnl_graph_op_hard_sigmoid,
    dnnl_graph_op_hard_sigmoid_backward,
    dnnl_graph_op_last_symbol,
} dnnl_graph_op_kind_t;

// dnnl_graph.hpp
// class op
enum class kind {
    // ...
    HardSigmoid = dnnl_graph_op_hard_sigmoid,
    HardSigmoidBackward = dnnl_graph_op_hard_sigmoid_backward,
};
```

### Backend changes

DNNL backend will implement single operation execution through oneDNN's eltwise
primitive and implement the fusions through the eltwise post-ops of leading
operations (forward only).

### Documentation changes

Corresponding operation documents should be added to `doc/graph/operations/`.

## References

1. https://pytorch.org/docs/stable/generated/torch.nn.Hardsigmoid.html
2. https://www.tensorflow.org/api_docs/python/tf/keras/activations/hard_sigmoid
3. https://docs.openvino.ai/latest/openvino_docs_ops_activation_HardSigmoid_1.html
4. https://github.com/onnx/onnx/blob/main/docs/Operators.md#hardsigmoid
5. https://oneapi-src.github.io/oneDNN/dev_guide_eltwise.html
