# Support Swish operation in Graph API

## Background

Swish is an activation operation introduced and experimented in [[#1]][1] and
[[#2]][2]. It is also known as SiLU (Sigmoid Linear Unit) in some papers and
frameworks. In this document, we choose to call the operation Swish following
the naming convention of oneDNN. Swish operation is defined as:

$$Swish(x) = x * sigmoid(factor * x)$$

where $factor = 1.f$ by default for most real models.

### Adoption in models

Swish operation is widely adopted to improve the quality of deep learning
networks. For examples:

- EfficientNet series [[#3]][3]: Swish is used as the activation in
  Convolutional Neural Networks.
- Large language models like LLaMA [[#4]][4], Qwen [[#5]][5], etc.: Swish is
  used to construct SwiGLU [[#6]][6] by replacing the Sigmoid activation in
  typical GLU (Gated Linear Unit). SwiGLU is further used to build Gated MLP in
  the models.

### Support in frameworks and libraries

- PyTorch supports Swish via the SiLU operation [[#7]][7]. The operation does
  not support specifying `factor` in the formula.
- OpenVINO support Swish via the Swish operation [[#8]][8]. Unlike PyTorch's
  SiLU operation, OpenVINO's Swish also accepts a scalar input `Beta` as the
  multiplication `factor` for Sigmoid.
- For ONNX, a PR is working in progress to add Swish operation [[#9]][9].
- oneDNN supports Swish as an algorithm of eltwise primitive [[#10]][10] which
  accepts a scalar `alpha` in primitive descriptor creation as the
  multiplication `factor` for Sigmoid.
- cuDNN backend API supports Swish as a mode (`CUDNN_POINTWISE_SWISH_FWD`) of
  its Pointwise operation [[#11]][11] and accepts attribute
  `CUDNN_ATTR_POINTWISE_SWISH_BETA` as the multiplication `factor`.

## Proposals

### Option 1: Support Swish via Sigmoid and Multiply operation

As indicated by the formula of Swish, the proposal is to support it via the
combination of Sigmoid and Multiply operations which are already supported in
oneDNN Graph API.

- [Sigmoid operation](https://oneapi-src.github.io/oneDNN/dev_guide_op_sigmoid.html)
- [Multiply operation](https://oneapi-src.github.io/oneDNN/dev_guide_op_multiply.html)

With that, a Swish operation with default `factor` can ben programed as below:

```cpp
using namespace dnnl::graph;

graph swish = graph(engine::kind::cpu);

logical_tensor src = logical_tensor(ID_SRC, dt, shape);
logical_tensor res = logical_tensor(ID_RES, dt, shape);
logical_tensor dst = logical_tensor(ID_DST, dt, shape);

op sig = op(ID_SIG, op::kind::Sigmoid, "sig");
sig.add_input(src);
sig.add_output(res);

op mul = op(ID_MUL, op::kind::Multiply, "mul");
mul.add_inputs({src, res});
mul.add_output(dst);

swish.add_op(sig);
swish.add_op(mul);
swish.finalize();
```

Pros:

- There is no need to define and maintain a new operation in oneDNN Graph API.

Cons:

- Compared to a dedicate Swish operation, this proposal requires more users code
  (at least one more logical tensor and one more operation).
- It also requires complex logic in the backend to detect `Sigmoid + Multiply`
  and map to the existing Swish kernels in oneDNN. It requires the input of
  Sigmoid and the second input of Multiply to be the same tensor.
- Considering that SiLU is a built-in operation in PyTorch, mapping it to two
  operations in oneDNN Graph is troublesome for some integrations.
- Currently, oneDNN Graph Sigmoid operation does not support a multiplication
  `factor`. We may need to extend either the proposed Swish graph or the Sigmoid
  operation to support cases where `factor != 1.f`.

### Option 2: Support Swish as a dedicate operation

As aforementioned, main stream frameworks and libraries all support Swish as a
dedicate operation. We think that it's reasonable to add a new Swish operation
in oneDNN Graph API. The proposed operation schema is as follow:

- Operation Kind: `Swish` (C++), `dnnl_graph_op_swish` (C).
- Input/output: Single input, single output.
- Attribute: `beta` (optional). `beta = 1.f` if not provided.
- Data types: f32, bf16, f16.

With the new operation being defined, a Swish operation can be programed as
below:

```cpp
using namespace dnnl::graph;

graph swish = graph(engine::kind::cpu);

logical_tensor src = logical_tensor(ID_SRC, dt, shape);
logical_tensor dst = logical_tensor(ID_DST, dt, shape);

op swi = op(ID_SWI, op::kind::Swish, "swi");
swi.set_attr<float>(op::attr::beta, 0.5f); // optional
swi.add_input(src);
swi.add_output(dst);

swish.add_op(swi);
swish.finalize();
```

Pros:

- It simplifies the user code, especially when Swish is used to construct a
  complex fusion pattern.
- The operation can be directly dispatched to the existing Swish kernels in
  oneDNN.
- It can be integrated easily into PyTorch to optimize the SiLU operation. It
  also helps when converting cuDNN code into oneDNN code.
- Attribute `beta` is considered to support the cases where `factor != 1.f`.
- The granularity of operations is consistent within oneDNN and with other
  frameworks and libraries.

Cons:

- It adds an new operation into oneDNN Graph API which may need additional
  maintenance effort.
- To some extend, supporting all Sigmoid, Multiply, and Swish operations is kind
  of duplication.

## Conclusions

Option 2 is recommended.

oneDNN eltwise primitive and post-op can be used as the implementation of this
operation and fusions.

Benchdnn graph driver needs to be extended to support the validation of this new
operation with reusing eltwise driver as the reference.

## References

1. Swish: a Self-Gated Activation Function, https://arxiv.org/abs/1710.05941v1
2. Gaussian Error Linear Units (GELUs), https://arxiv.org/abs/1606.08415
3. EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks, https://arxiv.org/abs/1905.11946
4. LLaMA: Open and Efficient Foundation Language Models, https://arxiv.org/abs/2302.13971
5. Qwen Technical Report, https://arxiv.org/abs/2309.16609
6. GLU Variants Improve Transformer, https://arxiv.org/abs/2002.05202
7. SiLU operation in PyTorch, https://pytorch.org/docs/stable/generated/torch.nn.SiLU.html
8. Swish operation in OpenVINO, https://docs.openvino.ai/2024/documentation/openvino-ir-format/operation-sets/operation-specs/activation/swish-4.html
9. PR for Swish operation in ONNX, https://github.com/onnx/onnx/pull/5964
10. Swish in oneDNN, https://oneapi-src.github.io/oneDNN/dev_guide_eltwise.html
11. Swish in cuDNN, https://docs.nvidia.com/deeplearning/cudnn/latest/api/cudnn-graph-library.html#cudnnpointwisemode-t

[1]: https://arxiv.org/abs/1710.05941v1
[2]: https://arxiv.org/abs/1606.08415
[3]: https://arxiv.org/abs/1905.11946
[4]: https://arxiv.org/abs/2302.13971
[5]: https://arxiv.org/abs/2309.16609
[6]: https://arxiv.org/abs/2002.05202
[7]: https://pytorch.org/docs/stable/generated/torch.nn.SiLU.html
[8]: https://docs.openvino.ai/2024/documentation/openvino-ir-format/operation-sets/operation-specs/activation/swish-4.html
[9]: https://github.com/onnx/onnx/pull/5964
[10]: https://oneapi-src.github.io/oneDNN/dev_guide_eltwise.html
[11]: https://docs.nvidia.com/deeplearning/cudnn/latest/api/cudnn-graph-library.html#cudnnpointwisemode-t
