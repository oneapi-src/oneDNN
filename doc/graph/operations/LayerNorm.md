LayerNorm {#dev_guide_op_layernorm}
===================================

## General

LayerNorm performs a layer normalization operation on \src tensor.

The layerNorm operation performs normalization from `begin_norm_axis` to last
dimension of the data tensor. It is defined by the following formulas which is
the same as @ref dev_guide_layer_normalization.

\f[
    \dst(t, n, c) =
       \gamma(c) \cdot
       \frac{\src(t, n, c) - \mu(t, n)} {\sqrt{\sigma^2(t, n) + \epsilon}}
       + \beta(c),
\f]

where

- \f$\gamma(c), \beta(c)\f$ are optional scale and shift for a channel

- \f$\mu(t, n), \sigma^2(t, n)\f$ are mean and variance (see

- \f$\epsilon\f$ is a constant to improve numerical stability.

Mean and variance are computed at runtime or provided by a user. When mean and
variance are computed at runtime, the following formulas are used:

- \f$\mu(t, n) = \frac{1}{C} \sum\limits_{c} \src(t, n, c)_{}\f$,

- \f$\sigma^2(t, n) = \frac{1}{C} \sum\limits_{c} {}_{} (\src(t, n, c) - \mu(t, n))^2\f$.

## Operation attributes

| Attribute Name                                                 | Description                                                                                                                                                                                                                                                                                   | Value Type | Supported Values                              | Required or Optional |
|:---------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------|:----------------------------------------------|:---------------------|
| [keep_stats](@ref dnnl::graph::op::attr::keep_stats)           | Indicate whether to output mean and variance which can be later passed to backward op.                                                                                                                                                                                                        | bool       | `false`,`true` (default)                      | Optional             |
| [begin_norm_axis](@ref dnnl::graph::op::attr::begin_norm_axis) | `begin_norm_axis` is used to indicate which axis to start layer normalization. The normalization is from `begin_norm_axis` to last dimension. Negative values means indexing from right to left. This op normalizes over the last dimension by default, e.g. C in TNC for 3D and LDNC for 4D. | s64        | [-r,r-1],where r=rank(src). -1 is default     | Optional             |
| [use_affine](@ref dnnl::graph::op::attr::use_affine)           | When set to True, this module has learnable per-element affine parameters.                                                                                                                                                                                                                    | bool       | `false`, `true` (default)                     | Optional             |
| [epsilon](@ref dnnl::graph::op::attr::epsilon)                 | The constant to improve numerical stability.                                                                                                                                                                                                                                                  | f32        | Arbitrary positive f32 value, `1e-5`(default) | Optional             |

## Execution arguments

The inputs and outputs must be provided according to below index order when
constructing an operation.

### Inputs

| Index | Argument Name | Required or Optional |
|:------|:--------------|:---------------------|
| 0     | `src`         | Required             |
| 1     | `gamma`       | Optional             |
| 2     | `beta`        | Optional             |

@note `gamma` is scaling for normalized value. `beta` is the bias added to
the scaled normalized value. They are both 1D tensor with the same span as src’s
channel axis and required if attribute `use_affine` is set to True.

### Outputs

| Index | Argument Name | Required or Optional |
|:------|:--------------|:---------------------|
| 0     | `dst`         | Required             |
| 1     | `mean`        | Optional             |
| 2     | `variance`    | Optional             |

@note Both `mean` and `variance` are required if attribute `keep_stats` is set to
True.

## Supported data types

LayerNorm operation supports the following data type combinations.

| Src / Dst | Gamma / Beta / Mean / Variance |
|:----------|:-------------------------------|
| f32       | f32                            |
| bf16      | f32, bf16                      |
| f16       | f32                            |
