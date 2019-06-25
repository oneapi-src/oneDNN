Eltwise {#dev_guide_eltwise}
============================

>
> API reference: [C](@ref c_api_eltwise), [C++](@ref cpp_api_eltwise)
>

The eltwise primitive applies an operation to every element of the tensor:

\f[
    dst(\overline{x}) = Operation(src(\overline{x})),
\f]

where \f$\overline{x} = (x_n, .., x_0)\f$.

The following operations are supported:

| Operation    | MKL-DNN algorithm kind       | Formula
| :--          | :--                          | :--
| abs          | #mkldnn_eltwise_abs          | \f$ f(x) = \begin{cases} x & \text{if}\ x > 0 \\ \alpha -x & \text{if}\ x \leq 0 \end{cases} \f$
| bounded_relu | #mkldnn_eltwise_bounded_relu | \f$ f(x) = \begin{cases} \alpha & \text{if}\ x > \alpha \\ \alpha x & \text{if}\ x \leq \alpha \end{cases} \f$
| elu          | #mkldnn_eltwise_elu          | \f$ f(x) = \begin{cases} x & \text{if}\ x > 0 \\ \alpha (e^x - 1) & \text{if}\ x \leq 0 \end{cases} \f$
| exp          | #mkldnn_eltwise_exp          | \f$ f(x) = e^x \f$
| gelu         | #mkldnn_eltwise_gelu         | \f$ f(x) = 0.5 x (1 + tanh[\sqrt{\frac{2}{\pi}} (x + 0.044715 x^3)])\f$
| linear       | #mkldnn_eltwise_linear       | \f$ f(x) = \alpha x + \beta \f$
| logistic     | #mkldnn_eltwise_logistic     | \f$ f(x) = \frac{1}{1+e^{-x}} \f$
| relu         | #mkldnn_eltwise_relu         | \f$ f(x) = \begin{cases} x & \text{if}\ x > 0 \\ \alpha x & \text{if}\ x \leq 0 \end{cases} \f$
| soft_relu    | #mkldnn_eltwise_soft_relu    | \f$ f(x) = \log_{e}(1+e^x) \f$
| sqrt         | #mkldnn_eltwise_sqrt         | \f$ f(x) = \sqrt{x} \f$
| square       | #mkldnn_eltwise_square       | \f$ f(x) = x^2 \f$
| tanh         | #mkldnn_eltwise_tanh         | \f$ f(x) = \frac{e^z - e^{-z}}{e^z + e^{-z}} \f$

#### Difference Between [Forward Training](#mkldnn_forward_training) and [Forward Inference](#mkldnn_forward_inference)

There is no difference between the #mkldnn_forward_training and
#mkldnn_forward_inference propagation kinds.

### Backward

The backward propagation computes
\f$diff\_src(\overline{x})\f$,
based on
\f$diff\_dst(\overline{x})\f$ and \f$src(\overline{x})\f$.

## Implementation Details

### General Notes

1. All eltise primitives have a common initialization function (e.g.,
   mkldnn::eltwise_forward::desc::desc()) which takes both parameters
   \f$\alpha\f$, and \f$\beta\f$. These parameters are ignored if they are
   unused.

2. The memory format and data type for `src` and `dst` are assumed to be the
   same, and in the API are typically referred as `data` (e.g., see `data_desc`
   in mkldnn::eltwise_forward::desc::desc()). The same holds for
   `diff_src` and `diff_dst`. The corresponding memory descriptors are referred
   to as `diff_data_desc`.

3. Both forward and backward propagation support in-place operations, meaning
   that `src` can be used as input and output for forward propagation, and
   `diff_dst` can be used as input and output for backward propagation. In case
   of in-place operation, the original data will be overwritten.

4. For some operations it might be performance beneficial to compute backward
   propagation based on \f$dst(\overline{x})\f$, rather than on
   \f$src(\overline{x})\f$. However, for some other operations this is simply
   impossible. So for generality the library always requires \f$src\f$.

@note For the ReLU operation with \f$\alpha = 0\f$, \f$dst\f$ can be used
instead of \f$src\f$ and \f$dst\f$ when backward propagation is computed. This
enables several performance optimizations (see the tips below).

### Data Type Support

The eltwise primitive supports the following combinations of data types:

| Propagation        | Source / Destination | Intermediate data type
| :--                | :--                  | :--
| forward / backward | f32                  | f32
| forward            | f16                  | f16
| forward            | s32 / s8 / u8        | f32

@warning
    There might be hardware and/or implementation specific restrictions.
    Check [Implementation Limitations](@ref dg_eltwise_impl_limits) section
    below.

Here the intermediate data type means that the values coming in are first
converted to the intermediate data type, then the operation is applied, and
finally the result is converted to the output data type.

### Data Representation

The eltwise primitive works with arbitrary data tensors. There is no special
meaning associated with any logical dimensions.

### Post-ops and Attributes

The eltwise primitive doesn't support any post-ops or attributes.


@anchor dg_eltwise_impl_limits
## Implementation Limitations

1. No primitive specific limitations. Refer to @ref dev_guide_data_types for
   limitations related to data types support.


## Performance Tips

1. For backward propagation, use the same memory format for `src`, `diff_dst`,
   and `diff_src` (the format of the `diff_dst` and `diff_src` are always the
   same because of the API). Different formats are functionally supported but
   lead to highly suboptimal performance.

2. Use in-place operations whenever possible.

3. As mentioned above for the ReLU operation with \f$\alpha = 0\f$, one can use
   the \f$dst\f$ tensor instead of \f$src\f$. This enables the following
   potential optimizations for training:

    - ReLU can be safely done in-place.

    - Moreover, ReLU can be fused as a [post-op](@ref dev_guide_attributes)
      with the previous operation if that operation doesn't require its
      \f$dst\f$ to compute the backward propagation (e.g., if the convolution
      operation satisfies these conditions).
