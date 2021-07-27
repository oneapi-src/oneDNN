# RFC: Pooling with padding value

## Introduction

This document introduces new feature of Pooling operation that allows user to
customize padding value used in the operation.

### Motivation

The original request comes from OpenVino team. OpenVino updates definition of
MaxPool operation (more details: [MaxPool-8](https://github.com/openvinotoolkit/openvino/pull/5359)).
Part of this request is to add parameter `pads_value`:
* *pads_value*
    * **Description**: *pads_value* will be used to fill pixels added during padding.
      If not specified, the new pixels will be filled in with minimum value for the type.
    * **Range of values**: floating point values
    * **Type**: float
    * **Default value**: *lowestinity*
    * **Required**: *no*
This request downstreams to oneDNN since OpenVino uses oneDNN-based plugin.

The change is caused by an attempt to fuse `Pad + Pooling` as a performance
optimization:
```python
import tensorflow as tf
import numpy as np
a = np.array([-0.8, -0.3, 0.4,
              -0.1, -0.2, 0.3,
              -1.0, -2.0, -3.0])
a = np.reshape(a, (1, 3, 3, 1))
x = tf.constant(a)
paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
pad = tf.pad(x, paddings)
pool_argmax = tf.nn.max_pool_with_argmax(pad, ksize=[3, 3], strides=[2, 2], padding='VALID')
```

In TensorFlow [specification](https://www.tensorflow.org/api_docs/python/tf/pad)
Pad operation supports the following:
*   not only a constant padding value, but other options where padding value
    depend on the actual data in tensor;
*   padding over any dimension

OpenVino [specification](https://docs.openvinotoolkit.org/latest/openvino_docs_ops_movement_Pad_1.html) is aligned with TensorFlow.

**NOTE:** No performance data was provided. There is an OpenVino issue that is
ModelOptimizer generates single OpenVino MaxPool operation for a sequence of
TensorFlow Pad + MaxPool operations. This leads to incorrect results,
because the current MaxPool specification in OpenVino assumes that padding area
is filled with `-inf`.

### Definition of MaxPool-8 in OpenVino

```
dst(n, c, oh, ow) = max(pad_value, limits_{kh, kw} (src(n, c, oh * SH + kh * (DH + 1) - PH_L, ow * SW + kw * (DW + 1) - PW_L)))
indices(n, c, oh, ow) = index(max(pad_value, limits_{kh, kw} (src(n, c, oh * SH + kh * (DH + 1) - PH_L, ow * SW + kw * (DW + 1) - PW_L))))
```

There are a few issues with the proposed OpenVino MaxPool-8 specification:
* In case maximum value is in padding area index for this kernel is not specified.
* Padding value is implicitly converted to MaxPool computational data type.
* Definition of MaxPool-8 doesn't allow to use Pad + MaxPool fusion due to indices
  mismatch with the reference implementation
  ```python
  >>> import tensorflow as tf
  >>> import numpy as np
  >>> a = np.array([-0.8, -0.3, 0.4,
  ...               -0.1, -0.2, 0.3,
  ...               -1.0, -2.0, -3.0])
  >>> a = np.reshape(a, (1, 3, 3, 1))
  >>> x = tf.constant(a)
  >>> paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
  >>> pad = tf.pad(x, paddings)
  >>> pool_argmax = tf.nn.max_pool_with_argmax(pad, ksize=[3, 3], strides=[2, 2], padding='VALID')
  >>> pool_argmax
  MaxPoolWithArgmax(output=<tf.Tensor: shape=(1, 2, 2, 1), dtype=float64, numpy=
  array([[[[0. ],
           [0.4]],
          [[0. ],
           [0.3]]]])>, argmax=<tf.Tensor: shape=(1, 2, 2, 1), dtype=int64, numpy=
  array([[[[ 0],
           [ 8]],
          [[10],
           [13]]]])>)
  ```
  The second output which represents indices is `[0, 8, 10, 13]`. However if we
 pass unpadded data to the MaxPool-8 with `pad_value=0` then indices will be
 equal to `[undef, 2, undef, 5]`. Even if there were no issues with padding
 being the biggest value indices should be recalculated to emulate physically
 padded tensor.
* Backward pass in not taken care of.

#### Integer support

With the current pooling definition padding doesn't affect the result of padding
operation on shifted data (with zero point not equal to zero). This fact
eliminates necessity in zero points support for Pooling operation. However
adding support for custom padding value introduces this necessity, since padding
value that is not equal to `lowest` should be updated with zero point value before
computing the maximum value in a kernel.

## Proposal

Pad operation in OpenVino is a separate operation and it can't be described
semantically as part of Pooling operation. To fully enable this fusion oneDNN
team would have to introduce Pad operation and then extend it with a Pooling
post operation. As an alternative we could support only limited functionality
that will be enough for OpenVino.

### Option 1

Align oneDNN with OpenVino MaxPool-8 definition by adding float member to
support all possible pad values for inference only.

Backward pass of pooling returns `invalid_arguments` if user tries to pass
`pooling_desc` with padding value not equal to `-inf`.

Pros:
* Full coverage of OpenVino MaxPool-8.

Cons:
* Floating point padding value doesn't make sense for Pooling on integer data;
* There is no reason to have padding values other than `0` and `lowest` from
  mathematical point of view (if we exclude `Pad+Pooling` optimization).
  The original PR in OpenVino introduces padding value only to support these 2 cases.
  On the other side it complicates implementations a lot (especially corner
  cases like Pooling on integer data);
* Parameter should be ignored for average pooling until we decide to extend it as well.

#### API

```cpp
// dnnl.h

dnnl_status_t DNNL_API dnnl_pooling_v3_forward_desc_init(
        dnnl_pooling_v3_desc_t *pool_desc, dnnl_prop_kind_t prop_kind,
        dnnl_alg_kind_t alg_kind, const dnnl_memory_desc_t *src_desc,
        const dnnl_memory_desc_t *dst_desc, const dnnl_dims_t strides,
        const dnnl_dims_t kernel, const dnnl_dims_t dilation,
        const dnnl_dims_t padding_l, const dnnl_dims_t padding_r,
        float padding_value);

// This API is needed to be able to create a backward pass using
// dnnl_pooling_v3_desc_t.
dnnl_status_t DNNL_API dnnl_pooling_v3_backward_desc_init(
        dnnl_pooling_v3_desc_t *pool_desc, dnnl_alg_kind_t alg_kind,
        const dnnl_memory_desc_t *diff_src_desc,
        const dnnl_memory_desc_t *diff_dst_desc, const dnnl_dims_t strides,
        const dnnl_dims_t kernel, const dnnl_dims_t dilation,
        const dnnl_dims_t padding_l, const dnnl_dims_t padding_r);
```

### Option 2

Add padding value as part of algorithm and limit support to 2 values: `lowest` and `0`.
Similarly to the Option 1 algorithm will be supported on forward only.

Pros:
* Relatively easy implementation;

Cons:
* In case padding value other than `lowest` or `0` are needed oneDNN will have
 to extend Pooling definition again (new RFC), OpenVino will have to adapt
 integration, etc.
* Not scalable to other algorithms.

#### API

```cpp
// dnnl_types.h

/// Kinds of algorithms.
typedef enum {
    ...
    /// Max pooling with padding values equal to lowest value
    dnnl_pooling_max_pad_lowest = 0x1ff,
    /// Max pooling with padding values equal to zero
    dnnl_pooling_max_pad_zero = 0x200,
    /// Max pooling
    dnnl_pooling_max = dnnl_pooling_max_pad_lowest,
    ...
} dnnl_alg_kind_t;
```

### Option 3

Add padding value as a parameter, and limit support to 2 values: `lowest` and `0`.
Similarly to the Option 1 algorithm will be supported on forward only.

Pros:
* Relatively easy implementation;
* No corner cases.

Cons:
* In case padding value other than `lowest` or `0` are needed oneDNN will have
 to extend Pooling definition again (new RFC), OpenVino will have to adapt
 integration, etc.
* Parameter should be ignored for average pooling until we decide to extend it as well.

#### API

```cpp
// dnnl_types.h

/// Kinds of padding types in pooling.
typedef enum {
    ...
    /// Max pooling with padding values equal to lowest value
    dnnl_pooling_padding_type_lowest = 0x1,
    /// Max pooling with padding values equal to zero
    dnnl_pooling_paddyng_type_zero = 0x2,
} dnnl_pooling_padding_type_t;

// dnnl.h

dnnl_status_t DNNL_API dnnl_pooling_v3_forward_desc_init(
        dnnl_pooling_v3_desc_t *pool_desc, dnnl_prop_kind_t prop_kind,
        dnnl_alg_kind_t alg_kind, const dnnl_memory_desc_t *src_desc,
        const dnnl_memory_desc_t *dst_desc, const dnnl_dims_t strides,
        const dnnl_dims_t kernel, const dnnl_dims_t dilation,
        const dnnl_dims_t padding_l, const dnnl_dims_t padding_r,
        dnnl_pooling_padding_type_t padding_type);

// This API is needed to be able to create a backward pass using
// dnnl_pooling_v3_desc_t.
dnnl_status_t DNNL_API dnnl_pooling_v3_backward_desc_init(
        dnnl_pooling_v3_desc_t *pool_desc, dnnl_alg_kind_t alg_kind,
        const dnnl_memory_desc_t *diff_src_desc,
        const dnnl_memory_desc_t *diff_dst_desc, const dnnl_dims_t strides,
        const dnnl_dims_t kernel, const dnnl_dims_t dilation,
        const dnnl_dims_t padding_l, const dnnl_dims_t padding_r);
```

### Option 4

Traditional option is to do nothing. This is motivated by the fact that
optimization is pretty much experimental and is not fully defined yet.

Pros:

Cons:
* OpenVino might lose performance in some benchmarks (there was no data provided where the optimization is required).

## Recommendation

Option 4 seems the most reasonable at the moment:
- Adding this optimization right now might cost API changes in the future due to
 lack of clear operation semantics definition;
- If we extend API it will require a lot of work on the implementation side to
 cover all implementations on both CPU & GPU, but the real benefit of the
 optimization is questionable;
- In case this optimization will become important and useful in many models we
 can return and implement one of Options 1-3. At that moment we will have clear operation
 semantics.

## Open Questions

- Can we push OpenVino team to limit pad value definition on their side to just `lowest` and `0`?
  The optimization is intended to be with padding of type float to support more general cases.
  
  After a discussion with OpenVino team Pooling specification was updated so there is no padding_value parameter anymore.

- Is this optimization critical for OpenVino customers?

  Based on the measurements the optimization doesn't provide any significant improvement.
