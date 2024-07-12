GroupNorm {#dev_guide_op_groupnorm}
===================================

## General
The GroupNorm operation performs the following transformation of the
input tensor:

\f[
    y =
       \gamma \cdot
       \frac{(x - mean)} {\sqrt{variance + \epsilon}}
       + \beta,
\f]


The operation is applied per batch, per group of channels. The gamma and beta
coefficients are the optional inputs to the model and need to be specified
separately for each channel. The `mean` and `variance` are calculated for each
group.


## Operation attributes

| Attribute Name                                                 | Description                                                                                                                                                                                                                                                                                   | Value Type | Supported Values                              | Required or Optional |
|:---------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------|:----------------------------------------------|:---------------------|
| [groups](@ref dnnl::graph::op::attr::groups)                 | Specifies the number of groups `G` that the channel dimension will be divided into. `groups` should be divisible by the number of channels                                                              | s64        |  between 1 and the number of channels `C` in the input tensor                                 | Required             |
| [keep_stats](@ref dnnl::graph::op::attr::keep_stats)           | Indicate whether to output mean and variance which can be later passed to backward op.                                                                                                                                                                                                        | bool       | `false`,`true` (default is true)                      | Optional             |
| [use_affine](@ref dnnl::graph::op::attr::use_affine)           | When set to True, this module has inputs `gamma` and `beta`                                                                                                                                                                                                                  | bool       | `false`, `true` (default is true)                     | Optional             |
| [epsilon](@ref dnnl::graph::op::attr::epsilon)                 | The constant to improve numerical stability.                                                                                                                                                                                                                                                  | f32        | Arbitrary positive f32 value, `1e-5`(default) | Optional             |
| [data_format](@ref dnnl::graph::op::attr::data_format)       | Controls how to interpret the shape of `src` and `dst`.                                                                                                                                   | string     | `NCX`, `NXC` (default is `NXC`)                                               | Optional             |
## Execution arguments

The inputs and outputs must be provided according to below index order when
constructing an operation.

### Inputs

| Index | Argument Name | Required or Optional |
|:------|:--------------|:---------------------|
| 0     | `src`         | Required             |
| 1     | `gamma`       | Optional             |
| 2     | `beta`        | Optional             |

@note `gamma` is scaling for the normalized value. `beta` is the bias added to
the scaled normalized value. They are both 1D tensor with the same span as src’s
channel axis and required if the attribute `use_affine` is set to True.

### Outputs

| Index | Argument Name | Required or Optional |
|:------|:--------------|:---------------------|
| 0     | `dst`         | Required             |
| 1     | `mean`        | Optional             |
| 2     | `variance`    | Optional             |

@note Both `mean` and `variance` are required if the attribute `keep_stats` is
set to `True`.

## Supported data types

GroupNorm operation supports the following data type combinations.

| Src / Dst | Gamma / Beta / Mean / Variance |
|:----------|:-------------------------------|
| f32       | f32                            |
| bf16      | f32, bf16                      |
| f16       | f32                            |
