# Introducing PReLU support in oneDNN

## Introduction

This paper https://arxiv.org/abs/1502.01852 introduced PReLU activation function.
Request comes from Caffe users: "PReLU is used in some POR topologies, such as
VNET, SRGAN. It takes most of time if the original implementation, is not good
optimized. For example VNET's train perf with latest Intel Caffe is 2.4s, PReLU
(has optimized by openmp) spends 429ms (not including reorder time), 17% of total
time. If we uses oneDNN ReLU to make projection of the perf, it will come to
~1.8s." Some other data scientist in discussion mentioned that he encouter PReLU
issue when he tried to optimize the inference pipeline of a popular face-related
application - PReLU tooks 47 % of time in his application.

Currently in oneDNN we support ReLU activation function in eltwise operation:
https://oneapi-src.github.io/oneDNN/dev_guide_eltwise.html. LeakyReLU can be
achieved by specifying non zero alpha parameter. PReLU is specific variant of
LeakyReLU, where alpha is learnable parameter. It cames with two flavours:
- channel-wise PReLU - where number of trained alpha parameters is equal to
channel number
- channel-shared PReLU - where there is only one trainable alpha

All popular framworks supports PReLU:
- Caffe - https://caffe.berkeleyvision.org/tutorial/layers/prelu.html
- Keras/tensorflow - https://www.tensorflow.org/api_docs/python/tf/keras/layers/PReLU
- PyTorch - https://pytorch.org/docs/master/generated/torch.nn.PReLU.html
- MxNet - https://beta.mxnet.io/api/gluon/_autogen/mxnet.gluon.nn.PReLU.html

Example of calculations forward/backward can be found: https://bit.ly/2TXgvzT

## Proposal

1. Option A - create new dedicated primitive PReLU:

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

The learnable PReLU parameters, which typically referred as alpha, are stored as
weights tensor. In order to allow user use of different PReLU algorithms (
channel-wise, channels-shared, tf/keras shared axes ) weights tensor will be
broadcasted to the src tensor if it will math broadcastable definition (see
pseudcode below).

``` cpp
bool are_weights_brodcastable(const dims_t &src, const dims_t &weights) {

    if (src.size() != weights.size())
        return false;

    for (std::size_t i = 0; i < src.size(), i++)
        if (weights[i] !=1 && weights[i] != src[i])
            return false;
    return true;
}
```

Examples of algorithm selection with broadcast semantics:

n = 256, h = 7, w = 7 c=128
src_md = {{256, 128, 7, 7}, f32, nhwc};

- whole-tensor (tensorflow wariant) --
prelu_forward::desc(src_md, weights_md={{256, 128, 7, 7}, tag::nhwc}})
- channel_wise --
prelu_forward::desc(src_md, weights_md={{1, 128, 1, 1}, tag::nhwc}})
- channel_shared --
prelu_forward::desc(src_md, weights_md={{1, 1, 1, 1}, tag::nhwc}})
- shared_axes (n, c shared) --
prelu_forward::desc(src_md, weights_md={{1, 1, 7, 7}, tag::nhwc}})

The library will recommend user to allow prelu primitive to choose the
appropriate weights memory format by passing weights_md with ```format_tag::any```.
This might be important for the whole-tensor case, where ideally the weights
should repeat the memory format of the data tensor. However, given that the
most typical case will be channel_wise (C elements) or channel_shared (1 element)
weights, the primitive should be optimized for the plain format that user has.
Data type will be preserved.

Examples:
- prelu_forward::desc(src_md, weights_md={{n, c, h, w}, tag::any}}) -- ok;
- prelu_forward::desc(src_md, weights_md={{n, c, h, w}, tag::abcd}}) --
not recommended, since primitive might want to use tag::aBcd16b for src_md,
weights data format may affect primitive performance;
- prelu_forward::desc(src_md, weights_md={{1, c, 1, 1}, tag::any}}) -- ok;
- prelu_forward::desc(src_md, weights_md={{1, c, 1, 1}, tag::abcd}}) -- ok;
- prelu_forward::desc(src_md, weights_md={{1, 1, 1, 1}, tag::abcd}}) -- ok;

2. Option B - extend exisiting eltwise implementation

I would recommend option number A. In case of calculation gradients for alpha we
 must pass memory descriptors for alpha and alpha_grad. Implementation
 in existing eltwise kernel would require breaking the ABI. What is more now in
 eltwise we pass float as alpha param. If we added DNNL_ARG_WEIGHTS (alpha)
 required for PReLU and alpha required for LeakyReLU remained it
 would be ambiguous for the user.

## Open questions
1. Tensorflow gives user greater flexibility than channel wise/shared,
by shared_axes parameter. Should we support that as well ?

API wise yes, implementation of full functionality can be spread over time
according to the requests and priorities.
