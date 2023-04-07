LayerNormBackward {#dev_guide_op_layernormbackward}
===================================================

## General

LayerNormBackward performs the backward of LayerNorm operation.

The backward propagation computes
\f$\diffsrc(t, n, c)\f$,
\f$\diffgamma(c)^*\f$, and \f$\diffbeta(c)^*\f$
based on
\f$\diffdst(t, n, c)\f$, \f$src(t, n, c)\f$, \f$\mu(t, n)\f$,
\f$\sigma^2(t, n)\f$, \f$\gamma(c) ^*\f$, and \f$\beta(c) ^*\f$.

The tensors marked with an asterisk are used only when the operation is
configured to use \f$\gamma(c)\f$, and \f$\beta(c)\f$

## Operation attributes

| Attribute Name                                                 | Description                                                                                                                                                                                                                                                                                   | Value Type | Supported Values                             | Required or Optional |
|:---------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------|:---------------------------------------------|:---------------------|
| [begin_norm_axis](@ref dnnl::graph::op::attr::begin_norm_axis) | `begin_norm_axis` is used to indicate which axis to start layer normalization. The normalization is from `begin_norm_axis` to last dimension. Negative values means indexing from right to left. This op normalizes over the last dimension by default, e.g. C in TNC for 3D and LDNC for 4D. | s64        | [-r,r-1],where r=rank(src). -1 is default    | Optional             |
| [use_affine](@ref dnnl::graph::op::attr::use_affine)           | When set to True, this module has learnable per-element affine parameters.                                                                                                                                                                                                                    | bool       | `false`,`true` (default)                     | Optional             |
| [epsilon](@ref dnnl::graph::op::attr::epsilon)                 | The constant to improve numerical stability.                                                                                                                                                                                                                                                  | f32        | Arbitrary positive f32 value, 1e-5 (default) | Optional             |

## Execution arguments

The inputs and outputs must be provided according to below index order when
constructing an operation.

### Inputs

| Index | Argument Name | Required or Optional |
|:------|:------------- |:---------------------|
| 0     | `src`         | Required             |
| 1     | `diff_dst`    | Required             |
| 2     | `mean`        | Required             |
| 3     | `variance`    | Required             |
| 4     | `gamma`       | Optional             |
| 5     | `beta`        | Optional             |

@note `gamma` is scaling for normalized value. `beta` is the bias added to
the scaled normalized value. They are both 1D tensor with the same span as src’s channel
axis and required if attribute `use_affine` is set to True.

### Outputs

| Index | Argument Name | Required or Optional |
|:------|:--------------|:---------------------|
| 0     | `diff_src`    | Required             |
| 1     | `diff_gamma`  | Optional             |
| 2     | `diff_beta`   | Optional             |

## Supported data types

LayerNormBackward operation supports the following data type combinations.

| Src / Diff_dst / Diff_src | Gamma / Beta / Mean / Variance / Diff_gamma / Diff_beta |
|:--------------------------|:--------------------------------------------------------|
| f32                       | f32                                                     |
| bf16                      | f32, bf16                                               |
| f16                       | f32                                                     |
