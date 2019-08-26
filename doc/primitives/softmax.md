Softmax {#dev_guide_softmax}
============================

>
> API reference: [C](@ref c_api_softmax), [C++](@ref cpp_api_softmax)
>

The softmax primitive performs softmax along a particular axis on data with
arbitrary dimensions. All other axes are treated as independent (batch).

In general form, the operation is defined by the following formulas:

### Forward

\f[
    dst(\overline{ou}, c, \overline{in}) =
        \frac
        {e^{src(\overline{ou}, c, \overline{in}) - \nu(\overline{ou}, \overline{in})}}
        {
            \sum\limits_{ic}
                e^{src(\overline{ou}, ic, \overline{in}) - \nu(\overline{ou}, \overline{in})}
        },
\f]

where

- \f$c\f$ dimension is called a softmax axis,
- \f$\overline{ou}\f$ is the outermost indices (to the left from softmax axis),
- \f$\overline{in}\f$ is the innermost indices (to the right from softmax axis), and
- \f$\nu\f$ is used to produce more accurate results and defined as:

\f[
    \nu(\overline{ou}, \overline{in}) =
        \max\limits_{ic}
        src(\overline{ou}, ic, \overline{in})
\f]

#### Difference Between [Forward Training](#dnnl_forward_training) and [Forward Inference](#dnnl_forward_inference)

There is no difference between the #dnnl_forward_training
and #dnnl_forward_inference propagation kinds.

### Backward

The backward propagation computes
\f$diff\_src(ou, c, in)\f$,
based on
\f$diff\_dst(ou, c, in)\f$ and \f$dst(ou, c, in)\f$.

## Implementation Details

### General Notes

N/A

### Post-ops and Attributes

The softmax primitive doesn't support any post-ops or attributes.

### Data Type Support

The softmax primitive supports the following combinations of data types:

| Propagation        | Source / Destination
| :--                | :--
| forward / backward | f32
| forward            | f16

### Data Representation

#### Source, Destination, and Their Gradients

The softmax primitive works with arbitrary data tensors. There is no special
meaning associated with any logical dimensions. However, the softmax axis is
typically referred to as channels (hence in formulas we use \f$c\f$).


## Implementation Limitations

1. No primitive specific limitations. Refer to @ref dev_guide_data_types for
   limitations related to data types support.

## Performance Tips

 * Currently the softmax primitive is optimized for the cases where
   the dimension of the softmax axis is physically dense. For instance:
   - Optimized: 2D case, tensor \f$A \times B\f$,
                softmax axis 1 (B), format tag #dnnl_ab
   - Optimized: 4D case, tensor \f$A \times B \times C \times D\f$,
                softmax axis 3 (D), format tag #dnnl_abcd
   - Optimized: 4D case, tensor \f$A \times B \times C \times D\f$,
                softmax axis 1 (B), format tag #dnnl_abcd, and
                \f$C = D = 1\f$
   - Non-optimized: 2D case, tensor \f$A \times B\f$,
                    softmax axis 0 (A), format tag #dnnl_ab,
                    and \f$B \ne 1\f$
   - Non-optimized: 2D case, tensor \f$A \times B\f$,
                    softmax axis 1 (B), format tag #dnnl_ba,
                    and \f$A \ne 1\f$
   - Non-optimized: 4D case, tensor \f$A \times B\f$,
                    softmax axis 2 (C), format tag #dnnl_acdb, and
                    and \f$D \cdot B \ne 1\f$
