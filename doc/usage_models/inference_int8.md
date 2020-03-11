Int8 Inference {#dev_guide_inference_int8}
==========================================

## Introduction

To push higher performance during inference computations, recent work has
focused on computing at a lower precision (that is, shrinking the size of data
for activations and weights) to achieve higher throughput. Eight-bit
computations (referred to as int8) offer improved performance over
higher-precision types because they enable packing more data into a single
instruction, at the cost of reduced (but acceptable) accuracy.

## Int8 Workflow

There are different ways to use lower precision to perform inference.  Please
go through the @ref dev_guide_attributes_quantization page to get the initial
understanding of what kind of quantization model oneDNN supports.

### Quantization Process
To operate with int8 data types from a higher-precision format (for example,
32-bit floating point), data must first be _quantized_. The quantization
process converts a given input into a lower-precision format. The precision and
accuracy factors are determined by the scaling factors.

### Scale
The scale is usually obtained from sampling the dataset of previous executions
in the original format (for example, the activations and weights from training
in fp32) and is formulated as:

+ \f$ R_{\{\alpha,w\}} = max(abs(T_{\{\alpha,w\}}))\f$

where \f$T_{\{\alpha,w\}} {}_{}\f$ is a tensor corresponding
to either the weights \f$w\f$ or the activations \f$\alpha\f$.

The purpose is to establish the range of values used in the computation,
where selecting a proper scaling factor prevents over- or underflows during
computation of the lower-precision results.

### Quantization Factor
The next step is to calculate the **quantization factor** for converting the
values into the corresponding int8 range. This is also known as the **scale**
or **scaling factor** applied to the original high-precision values and is
calculated as:

+ \f$ Q_{\alpha} = \frac{255}{R_{\alpha}}\f$ is the
quantization factor for activations with non-negative values.

+ \f$ Q_{w} = \frac{127}{R_{w}}\f$ is the quantization factor for weights.

The low-precision values, known as the **quantized** activation, weights, and
bias values, are calculated as:

+ \f$\alpha_{u8} = \lceil Q_{\alpha} \alpha_{f32} \rceil \in [0,255]\f$

+ \f$W_{s8} = \lceil Q_{w} W_{f32} \rceil \in [-127,127]\f$

+ \f$b_{s32} = \lceil Q_{\alpha} Q_{w} b_{f32} \rceil \in [-2^{31},2^{31}-1]\f$

where the function \f$ \lceil \rceil \f$ rounds to the selected rounding mode
(typically determined by the MXCSR register; the default value is
 RoundNearestEven).

When the destination value (for example, from a convolution) is stored as a
signed 32-bit integer, the result is bound to the same quantization *scaling*
factors:

+ \f$X_{s32} = W_{s8} \times \alpha{u8} + b_{s32} \approx Q_{\alpha} Q_{\omega} X_{f32}\f$

+ where \f$X_{f32} = W_{f32} \times \alpha_{f32} + b_{f32}\f$

where the approximated value is due to the rounded values.

Inversely, the dequantized value is calculated as:

+ \f$X_{f32} \approx \frac{1}{Q_{\alpha} Q_{\omega}} X_{s32} \f$

### Quantization Example
To show how the int8 parameters are obtained, suppose we first start off with a
set of arbitrary high-precision input and output values. These values come from
sampling a previously executed training run and are in their original 32-bit
floating point format as:

+ activations: \f$ T_{\alpha} = [15, 14, 15 ... 8, 11 ]\f$
  where \f$ max(abs(T_{\alpha})) = 15\f$

+ weights:\f$ T_{\omega} = [-5.1 , 6.8, ... -1.2, 9.8 ]\f$
  where \f$ max(abs(T_{\omega})) = 9.8\f$

+ bias:\f$ T_{\alpha} = [ 2.4, -5.2 ... -8 ]\f$
  where \f$ max(abs(T_{\alpha})) = 8\f$

The scaling factors are:

+ \f$ Q_{\alpha} = \frac{255}{R_{\alpha}} = \frac{255}{15} = 17 \f$

+ \f$ Q_{w} = \frac{127}{R_{w}} = \frac{127}{9.8} = 12.96\f$

Finally, the quantized input values for the 8-bit operation are calculated as:

+ \f$\alpha_{u8} = \lceil Q_{\alpha} \alpha_{f32} \rceil\f$
   \f$ = \lceil 17 \times [15, 14, ... 11 ] \rceil = [255, 238, ... 187] \f$

+ \f$W_{s8} = \lceil Q_{w} W_{f32} \rceil
    = \lceil 12.96 \times [-5.1 , 6.8, ... -1.2, 9.8 ] \rceil
    = [-66, 88, ... -15, 127] \f$

+ \f$b_{s32} = \lceil Q_{\alpha} Q_{w} b_{f32} \rceil
    = \lceil 17 \times 12.96 \times [ 2.4, -5.2 ... -8 ] \rceil
    = [528, -1145, ... -1762] \f$

These arrays are the new inputs for the int8 net.

## int8 Support

oneDNN supports low-precision computations for inference through the
int8 primitives. int8 primitives are ordinary oneDNN primitives that
have their input and output parameters configured to 8-bit types. int8
primitives are optimized for high performance on the compatible hardware
(see @ref dev_guide_data_types).

### Attributes

oneDNN primitive behavior may be extended for additional
functionalities involving output data transformation. These additional features
are configured via **primitive attributes**. The primitive attributes
definition is an opaque structure for passing extra parameters to a primitive
descriptor. These parameters include a scaling factor and fused post-ops.
All operation primitives support the attributes structure;
however, some configurations are not implemented and result in *failed
primitive creation*.

The **scaling factor**, as previously described, is known prior to the
inference operation where the values are calculated from a set of formulas. In
oneDNN, the scaling factor is applied to the output of a primitive.
Moreover, to perform input transformations (for example, source, bias, and
weights), oneDNN performs quantizing and dequantizing of data for int8
through the **Reorder Primitive**.

oneDNN has two formats for defining the output scaling factor. Depending
on the configuration set by the scaling mask, either the output is scaled
uniformly across all the dimensions (_mask = 0_) or a set of scaling values is
applied to specific dimensions, as explained below:

* A *single floating point value* shared across the tensor
![Single-value scaling format](./images/img_singlescalar.png)
* An array of floating point values each corresponding to a specific output
  channel ![Multi-value scaling format](./images/img_multiscalar.png)

The **mask** parameter determines the dimension to which the scales array is
applied, where the i<sup>th</sup>-bit(s) of mask selects the dimension or
dimensions d<sub>i</sub> (where _d_ is an _n_-dimensional output tensor with
logical dimensions as [*d0, d1, ..., dn-1*]). For example:

+ The single-scale format always has mask = 0.

+ For a 5-dimensional tensor T[g0, o1,i2,h3,w4] where the numbering indicates
  the bit-index:

  + A mask = 2 = 2<sup>1</sup> selects the output channel for scaling.

  + A mask = 3 = 2<sup>0</sup> | 2<sup>1</sup> selects the group and output channels.

Mask is always applied to the logical dimension; this is independent of
the dimension format that the primitive might select. The dimensions in
oneDNN are defined as follows:
+ 2D dimensional data the order of dimensions is always: (n, c)
+ 4D dimensional data the order is always: (n, c, h, w)
+ 5D dimensional weights the order is always: (g, oc, ic, kh, kw)

Fused **post-ops** allow chaining operations during
the primitive computation. Note that the resulting output value from
post-ops is always affected by the scaling factor. The supported operations are:

+ Accumulation where the primitive sums the resulting values from previously
  computed activations as:
  - \f$dst[ ] \leftarrow scale * dst[] + op(...)\f$, instead of
  - \f$dst[ ] \leftarrow op(...)\f$

+ Element-wise (eltwise) operation with kind, alpha and beta parameters as:
  - \f$dst[ ] \leftarrow scale * eltwise\_op ( op(...) )\f$, instead of
  - \f$dst[ ] \leftarrow op(...)\f$

The list of supported eltwise operations for int8 is currently limited to ReLU.
For instance, post-ops may only configure a convolution with accumulation
followed by eltwise (relu).

## Example

@ref cnn_inference_int8_cpp example walks through the steps of int8 inference.
