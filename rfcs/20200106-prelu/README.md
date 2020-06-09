# Introducing PReLU Support in oneDNN

## Introduction

This paper https://arxiv.org/abs/1502.01852 introduced PReLU activation
function. Request comes from Caffe users:

> PReLU is used in some topologies, such as VNET, SRGAN. It takes most of time
> if the original implementation, is not good optimized. For example VNET's
> train perf with latest Intel Caffe is 2.4s, PReLU (has optimized by OpenMP)
> spends 429ms (not including reorder time), 17% of total time. If we uses
> oneDNN ReLU to make projection of the perf, it will come to ~1.8s.

Some other data scientist in discussion mentioned that he encounter PReLU issue
when he tried to optimize the inference pipeline of a popular face-related
application - PReLU takes 47% of time in his application.

Currently oneDNN supports ReLU activation function in
[eltwise operations](https://oneapi-src.github.io/oneDNN/dev_guide_eltwise.html).
LeakyReLU can be achieved by specifying non zero alpha parameter. PReLU is
specific variant of LeakyReLU, where alpha is learnable parameter. It comes
with two flavors:
- channel-wise PReLU - where number of trained alpha parameters is equal to
  channel number
- channel-shared PReLU - where there is only one trainable alpha

All popular frameworks supports PReLU:
- [Caffe](https://caffe.berkeleyvision.org/tutorial/layers/prelu.html)
- [Keras/TensorFlow](https://www.tensorflow.org/api_docs/python/tf/keras/layers/PReLU)
- [PyTorch](https://pytorch.org/docs/master/generated/torch.nn.PReLU.html)
- [MxNet](https://beta.mxnet.io/api/gluon/_autogen/mxnet.gluon.nn.PReLU.html)

Example of calculations forward/backward can be found
[here](https://bit.ly/2TXgvzT).

## Proposal

### Option A - create new dedicated primitive PReLU.

``` cpp
typedef struct {
    /// The kind of primitive. Used for self-identifying the primitive
    /// descriptor. Must be #dnnl_prelu.
    dnnl_primitive_kind_t primitive_kind;
    /// The kind of propagation. Possible values: #dnnl_forward_training,
    /// #dnnl_forward_inference, #dnnl_backward
    dnnl_prop_kind_t prop_kind;
    /// Source and destination memory descriptor.
    dnnl_memory_desc_t data_desc;
    /// Learnable parameter memory descriptor.
    dnnl_memory_desc_t weights_desc;
    /// Source and destination gradient memory descriptor.
    dnnl_memory_desc_t diff_data_desc;
    /// Learnable parameter gradient memory descriptor.
    dnnl_memory_desc_t diff_weights_desc;
} dnnl_prelu_desc_t;
```

The learnable PReLU parameters, which typically referred as alpha, are stored
as weights tensor. In order to allow user use different PReLU algorithms
(channel-wise, channels-shared, TensorFlow/Keras shared axes) weights tensor
will support broadcast-semantics. Examples:

```
n = 256, h = 7, w = 7 c = 128
src_md = {dims={256, 128, 7, 7}, f32, tag::nhwc};
```

| Case                              | Primitive Descriptor Initialization
| :--                               | :--
| Whole-tensor (TensorFlow variant) | `prelu_forward::desc(src_md, weights_md={{256, 128, 7, 7}, tag::nhwc}})`
| Channel-wise                      | `prelu_forward::desc(src_md, weights_md={{1, 128, 1, 1}, tag::nhwc}})`
| Channel-shared                    | `prelu_forward::desc(src_md, weights_md={{1, 1, 1, 1}, tag::nhwc}})`
| Shared axes (n and c shared)      | `prelu_forward::desc(src_md, weights_md={{1, 1, 7, 7}, tag::nhwc}})`

The library will recommend user to allow PReLU primitive to choose the
appropriate weights memory format by passing weights_md with `format_tag::any`.
This might be important for the whole-tensor case, where ideally the weights
should repeat the memory format of the data tensor. However, given that the
most typical case will be channel-wise (`c` elements) or channel-shared (1
element) weights, the primitive should be optimized for the plain format that
user has. Data type will be preserved.

Performance friendly examples:
- `prelu_forward::desc(src_md, weights_md={{n, c, h, w}, f32, tag::any}})`
- `prelu_forward::desc(src_md, weights_md={{1, c, 1, 1}, f32, tag::any}})`
- `prelu_forward::desc(src_md, weights_md={{1, c, 1, 1}, f32, tag::abcd}})`
- `prelu_forward::desc(src_md, weights_md={{1, 1, 1, 1}, f32, tag::abcd}})`

Not recommended (for **full** `(n, c, h, w)` weights tensor the memory format
is forced):
- `prelu_forward::desc(src_md, weights_md={{n, c, h, w}, tag::abcd}})`


### Option B - extend existing eltwise implementation

### Recommendation

I would recommend Option A. In case of calculation gradients for alpha we must
pass memory descriptors for alpha and alpha_grad. Implementation in existing
eltwise kernel would require breaking the ABI. What is more now in eltwise we
pass float as alpha parameter. If we added `DNNL_ARG_WEIGHTS` (alpha) required
for PReLU and alpha required for LeakyReLU remained it would be ambiguous for
the user.

## Open questions

1. TensorFlow gives user greater flexibility than channel wise/shared,
   by shared_axes parameter. Should we support that as well?

   API wise yes, implementation of full functionality can be spread over time
   according to the requests and priorities.

2. Should the primitive have only cumulative `backward` pass (that includes
   backward by data and by weights), or should it support those steps
   separately?
   - Separate pass might be required, if at some point the alpha is no more
     treated as learnable parameter.
   - It is not clear if cumulative backward pass brings any value, though not
     providing it is strange, as other primitives (batch and layer
     normalization) have it.

   Current suggestion: support `backward = backward_data + backward_weights`,
   and `backward_data`.
