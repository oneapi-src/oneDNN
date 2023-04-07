BatchNormInference {#dev_guide_op_batchnorminference}
=====================================================

## General

The formula is the same as
[Batch Normalization primitive](@ref dev_guide_batch_normalization) like below.

\f[
    \dst(n, c, h, w) =
       \gamma(c) \cdot
       \frac{\src(n, c, h, w) - \mu(c)} {\sqrt{\sigma^2(c) + \varepsilon}}
       + \beta(c),
\f]

where

- \f$\gamma(c), \beta(c)\f$ are required scale and shift for a channel,

- \f$\mu(c), \sigma^2(c)\f$ are mean and variance for a channel, and

- \f$\varepsilon\f$ is a constant to improve numerical stability.

## Operation attributes

| Attribute Name                                         | Description                                                     | Value Type | Supported Values       | Required or Optional |
|:-------------------------------------------------------|:----------------------------------------------------------------|:-----------|:-----------------------|:---------------------|
| [epsilon](@ref dnnl::graph::op::attr::epsilon)         | A number to be added to the variance to avoid division by zero. | f32        | A positive float value | Required             |
| [data_format](@ref dnnl::graph::op::attr::data_format) | Controls how to interpret the shape of `src` and `dst`.         | string     |`NCX`, `NXC` (default)  | Optional             |

## Execution arguments

The inputs and outputs must be provided according to below index order when
constructing an operation.

### Inputs

| Index | Argument Name               | Required or Optional |
|:------|:----------------------------|:---------------------|
| 0     | `src`                       | Required             |
| 1     | `gamma`                     | Required             |
| 2     | `beta`                      | Required             |
| 3     | `mean`                      | Required             |
| 4     | `variance` (\f$\sigma^2\f$) | Required             |

### Outputs

| Index | Argument Name | Required or Optional |
|:------|:--------------|:---------------------|
| 0     | `dst`         | Required             |

## Supported data types

BatchNormInference operation supports the following data type combinations.

| Src / Dst | Gamma / Beta / Mean / Variance |
|:----------|:-------------------------------|
| f32       | f32                            |
| bf16      | f32, bf16                      |
| f16       | f32                            |
