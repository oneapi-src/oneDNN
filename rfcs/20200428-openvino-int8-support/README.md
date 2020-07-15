Proposal for low precision support used in OpenVino
====================================================

## 1. Introduction
### 1.1. Motivation

- OpenVino low precision pipeline supports symmetric and asymmetric
  quantization modes while oneDNN supports only symmetric mode. Asymmetric mode
  can represent the original range more accurately and improve overall model
  accuracy.

### 1.2 OpenVino low precision pipeline

#### OpenVino 8-bit integer Inference workflow

8-bit inference pipeline includes two stages (also refer to the figure below):

1. Offline stage, or model quantization. During this stage, FakeQuantize
   layers are added before most layers to have quantized tensors before layers
   in a way that low-precision accuracy drop for 8-bit integer inference
   satisfies the specified threshold. The output of this stage is a quantized
   model.  Quantized model precision is not changed, quantized tensors are in
   original precision range (fp32). FakeQuantize layer has Quantization
   Levels attribute which defines quants count. Quants count defines precision
   which is used during inference. For int8 range Quantization Levels attribute
   value has to be 255 or 256.
2. Run-time stage. This stage is an internal procedure of the CPU Plugin.
   During this stage, the pluing uses the Low Precision Transformation
   component to update the model to infer it in low precision:
   * Updates FakeQuantize layers to have quantized output tensors in low
     precision range and add dequantization layers to compensate the update.
     Dequantization layers are pushed through as many layers as possible to
     have more layers in low precision. After that, most layers have quantized
     input tensors in low precision range and can be inferred in low precision.
     Ideally, dequantization layers should be fused in next FakeQuantize or
     ScaleShift layers.
   * Weights are quantized and stored in Const layers.
   * Biases are updated to avoid shifts in dequantization layers.

[Source](https://docs.openvinotoolkit.org/latest/_docs_IE_DG_Int8Inference.html)

#### FakeQuantize layer

(!)
During run-time stage FQ layers will be split on quantization & dequantization.
Dequantization will be pushed through as many layers as possible to make them
to be executed in int8. However if it is not possible then quantization and
dequantization can be merged back into FQ and this layer will be executed
during execution. In this case we can merge it to the previous operation or
execute it as a standalone operation. In other words plugins don't remove FQ
layers to keep accuracy similar to what was achieved during calibration.

Definition:

``` python
 if x <= min(input_low, input_high):
     output = output_low
 elif x > max(input_low, input_high):
     output = output_high
 else:
     # input_low < x <= input_high
     output = round((x - input_low) / (input_high - input_low) * (levels-1))
                / (levels-1) * (output_high - output_low) + output_low
```

Inputs:

1. `X` - multidimensional input tensor of floating type to be quantized.
   Required.
2. `input_low` - minimum limit for input value. The shape must be broadcastable
   to the shape of `X`. Required.
3. `input_high` - maximum limit for input value. Can be the same as `input_low`
   for binarization. The shape must be broadcastable to the shape of `X`.
   Required.
4. `output_low` - minimum quantized value. The shape must be broadcastable to
   the shape of `X`. Required.
5. `output_high` - maximum quantized value. The shape must be broadcastable to
   the shape of `X`. Required.

Outputs:

1. `Y` - resulting tensor with shape and type matching the 1st input tensor
   `X`.

[Source](https://docs.openvinotoolkit.org/latest/_docs_ops_quantization_FakeQuantize_1.html)

#### Asymmetric activations & weights in convolution

```
q_dst - zp_dst = conv(q_src - zp_src, q_wei - zp_wei)
               = conv(q_src, q_wei) - conv(q_src, zp_wei) - conv(zp_src, q_wei) + conv(zp_src, zp_wei)
```

If zp_wei = 0:

```
q_dst - zp_dst = conv(q_src - zp_src, q_wei)
               = conv(q_src, q_wei) - conv(zp_src, q_wei)
```

### 1.3 History

Initially the work to support OpenVino int8 pipeline was done by Inference
Engine CPU plugin team. This work is available in the [fork](https://github.com/openvinotoolkit/oneDNN/tree/v0.21_for_ie_master)
and based on MKL-DNN 0.x API. The following features required for OpenVino int8
support were implemented in this fork:

* Asymmetric activations & weights support in convolution primitive. Both
  activations & weights can have per channel zero points;

    <details>
    <summary>C API</summary>

    ~~~c
    mkldnn_status_t MKLDNN_API mkldnn_primitive_attr_get_output_compensations(
            const_mkldnn_primitive_attr_t attr, int *count, int *mask,
            const int32_t **compensations);

    mkldnn_status_t MKLDNN_API mkldnn_primitive_attr_set_output_compensations(
            mkldnn_primitive_attr_t attr, int count, int mask,
            const int32_t *compensations);

    mkldnn_status_t MKLDNN_API mkldnn_primitive_attr_get_input_zero_points(
            const_mkldnn_primitive_attr_t attr, int *count, int *mask,
            const uint8_t **zero_points);

    mkldnn_status_t MKLDNN_API mkldnn_primitive_attr_set_input_zero_points(
            mkldnn_primitive_attr_t attr, int count, int mask,
            const uint8_t *zero_points);

    mkldnn_status_t MKLDNN_API mkldnn_primitive_attr_get_weights_zero_points(
            const_mkldnn_primitive_attr_t attr, int *count, int *mask,
            const float **zero_points);

    mkldnn_status_t MKLDNN_API mkldnn_primitive_attr_set_weights_zero_points(
            mkldnn_primitive_attr_t attr, int count, int mask,
            const float *zero_points);
    ~~~

    </details>

* quantization_forward primitive. This primitives supports asymmetric
  quantization, dequantization & requantization with per channel crop, scales
  and zero points support;

    <details>
    <summary>C API</summary>

    ~~~c
    mkldnn_status_t MKLDNN_API mkldnn_quantization_forward_desc_init(
            mkldnn_quantization_desc_t *quantization_desc,
            mkldnn_prop_kind_t prop_kind, mkldnn_alg_kind_t alg_kind, int axis,
            const mkldnn_memory_desc_t *src_desc,
            const mkldnn_memory_desc_t *crop_low_desc,
            const mkldnn_memory_desc_t *crop_high_desc,
            const mkldnn_memory_desc_t *input_scale_desc,
            const mkldnn_memory_desc_t *input_shift_desc,
            const mkldnn_memory_desc_t *output_scale_desc,
            const mkldnn_memory_desc_t *output_shift_desc,
            const mkldnn_memory_desc_t *dst_desc);
    ~~~

    </details>


* quantization as a post-operation to convolution, pooling & ip primitives.

    <details>
    <summary>C API</summary>

    ~~~c
    mkldnn_status_t MKLDNN_API mkldnn_post_ops_append_quantization(
            mkldnn_post_ops_t post_ops, mkldnn_alg_kind_t alg,
            int crop_low_count, const float* crop_low,
            int crop_high_count, const float* crop_high,
            int input_scale_count, const float* input_scale,
            int input_shift_count, const float* input_shift,
            int output_scale_count, const float* output_scale,
            int output_shift_count, const float* output_shift);
    ~~~

    </details>

However this work is done in the fork based on MKL-DNN 0.x and can't be reused
in GPU plugin which uses MKL-DNN 1.x.

## 2. Proposal

### 2.1. Asymmetric activation support in (de)convolution

#### Motivation

Asymmetric quantization support in (de)convolution is an essential feature to
support OpenVino low precision pipeline. This feature consists of asymmetric
activations & weights and there is no request for asymmetric weights support at
this time. While asymmetric weights support is not a part of the proposal this
functionality is covered to make sure that this can be added in the future.

#### Discussion

Asymmetric quantization in convolution can be supported using zero points
mechanism already available in the library. Zero points indicate that data is
shifted with respect to zero. Data can have multiple (per channel) zero points.

##### Implementation details: pre-computing compensation

```
q_dst - zp_dst = conv(q_src - zp_src, q_wei - zp_wei)
               = conv(q_src, q_wei) - conv(q_src, zp_wei) - conv(zp_src, q_wei) + conv(zp_src, zp_wei)
```

If zp_wei = 0:

```
q_dst - zp_dst = conv(q_src - zp_src, q_wei)
               = conv(q_src, q_wei) - conv(zp_src, q_wei)
               = conv(q_src, q_wei) - zp_src * conv(1, q_wei)
               = conv(q_src, q_wei) - zp_src * compensation
```

The second part of this equation (compensation) can be precomputed during
initial weights quantization/reordering if the following requirements are met:
* Zero points are available prior to inference;
* Weights are constant during inference;
* Zero points are representable in the src data type.

In this case asymmetric quantization support in convolution implies the
following:
1. Additional computations for padded area if zero points are provided;
2. Additional computations for compensation.

There are 3 options of implementing compensation required by the fact that
activation has zero points.
1. Compensation is computed on integration side. Result is provided to
   convolution as an input to a binary post op.
2. Compensation is computed using reduction primitive and `binary_mul` post op.
   Result is provided to convolution as an input to a binary post op.
3. Compensation is computed during oneDNN weights reorder execution and
   provided as a part of weights data to a convolution.

In the first two cases oneDNN convolution implements only part of computations
required for asymmetric quantization support but provides a mechanism to
complete it.

All 3 options can be supported. The first one was initially requested by
OpenVino CPU plugin. The final decision is to implement third option, because
in this case convolution primitive computes complete convolution operation with
zero points for src memory and its definition doesn't depend on implementation.
Also in this case zero points abstraction can be reused without changing its
semantics.

#### API

<details>
<summary>C API</summary>

~~~c
/* dnnl_types.h */
typedef enum {
    /// Indicates the weights have an additional buffer, that depends on the
    /// @p compensation_mask.
    ///
    /// For instance, in 4D case with the compensation mask equals (1 << 0)
    /// the additional buffer would consist of OC values:
    /// O[oc : 0,OC] =
    ///  SUM(ic : 0,IC; kh : 0,KH; kw : 0,KW){ weights(oc, ic, kh, kw) }
    dnnl_memory_extra_flag_compensation_conv_asymmetric_src = 0x8U,
} dnnl_memory_extra_flags_t;

/* dnnl.h */
/* This API is already available. The proposal is to reuse it for data in
 * Convolution primitive. In the future this API can be reused for
 * weights memory in case asymmetric weights support is required.
 */
dnnl_status_t DNNL_API dnnl_primitive_attr_set_zero_points(
        dnnl_primitive_attr_t attr, int arg, dnnl_dim_t count, int mask,
        const int32_t *zero_points);

dnnl_status_t DNNL_API dnnl_primitive_attr_get_zero_points(
        const_dnnl_primitive_attr_t attr, int arg, dnnl_dim_t *count,
        int *mask, const int32_t **zero_points);
~~~

</details>

### 2.2. Fake quantization (FQ) as a post op

#### Motivation

This is the next step in supporting OpenVino int8 pipeline. If there is a FQ
which was not split on quantization and dequantization this FQ will not be
removed from quantized model to keep accuracy equal to accuracy of FQ model.
Merging it into primitives is an optimization to achieve better performance.
This optimization is already available in the fork.

The following primitives should support FQ as a post op:
* (de)convolution
* pooling
* inner product
* binary

(!) Each primitive can have multiple FQ layers with other primitive between
them merged as a chain of post operations.

(!) To support partially quantized models this optimization should be supported
in primitives with f32/bf16/s8/u8 data types

#### Discussion

There are 2 ways to implement FQ as a post op:
 * Custom post op;
 * Chain of simple post ops: max + min + add + mul + round + mul + add.

The second approach is more flexible and allows to reuse these simple ops for
other purposes:
 * max, min, add & mul are implemented using binary post operation.
 * round is implemented as a new algorithm of eltwise operation. Rounding is
   done in according to mxcsr.

The final decision is to implement FQ as a chain of simple post ops.

#### API

<details>
<summary>C API</summary>

~~~c
/* dnnl_types.h */

typedef enum {
    ...
    /// Eltwise: round
    dnnl_eltwise_round = 0x40,
} dnnl_alg_kind_t;

/* There are might be more than one post op (see Section 2.2.) so
 * indexing is a way to pass arguments to multiple instances of the same post
 * operation on execution. */

/* Allocating 5 bits for multiple post ops support. This arg will be used as a
 * prefix to an actual argument. */
#define DNNL_ARG_ATTR_MULTIPLE_POST_OP 2 << 13 /* 16384 */


/* dnnl.h */

/* This is a new post op. In this case it allows user to pass compensation. */
dnnl_status_t DNNL_API dnnl_post_ops_append_binary(dnnl_post_ops_t post_ops,
        dnnl_alg_kind_t alg_kind, const dnnl_memory_desc_t *src1_desc);

dnnl_status_t DNNL_API dnnl_post_ops_get_params_binary(
        const_dnnl_post_ops_t post_ops, int index,
        dnnl_alg_kind_t *alg_kind, dnnl_memory_desc_t *src1_desc);
~~~

</details>

### 2.3. Fake quantization as a separate entity

#### Motivation

While fake quantization is not required and can be implemented on integration
side, it is a nice-to-have feature to hide all complexity and HW-specific
implementations under oneDNN API.

#### Discussion

The decision is to not implement FQ as a separate oneDNN primitive, but provide
functionality to implement it using other oneDNN primitives.

Standalone FQ uses binary primitive as a basis and the rest part is implemented
as a chain of post operations as it was described in section 2.2.

</details>
