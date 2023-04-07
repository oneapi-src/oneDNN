BatchNormForwardTraining {#dev_guide_op_batchnormforwardtraining}
=================================================================

## General

BatchNormForwardTraining operation performs batch normalization at training mode.

Mean and variance are computed at runtime, the following formulas are used:

- \f$\mu(c) = \frac{1}{NHW} \sum\limits_{nhw} \src(n, c, h, w)_{}\f$,

- \f$\sigma^2(c) = \frac{1}{NHW} \sum\limits_{nhw} {}_{} (\src(n, c, h, w) - \mu(c))^2\f$.

## Operation attributes

| Attribute Name                                         | Description                                                         | Value Type | Supported Values       | Required or Optional |
|:-------------------------------------------------------|:--------------------------------------------------------------------|:-----------|:-----------------------|:---------------------|
| [epsilon](@ref dnnl::graph::op::attr::epsilon)         | A number to be added to the variance to avoid division by zero.     | f32        | A positive f32 value   | Required             |
| [momentum](@ref dnnl::graph::op::attr::momentum)       | A number to be used to calculate running mean and running variance. | f32        | A positive f32 value   | Optional             |
| [data_format](@ref dnnl::graph::op::attr::data_format) | Controls how to interpret the shape of `src` and `dst`.             | string     | `NCX`, `NXC` (default) | Optional             |

## Execution arguments

The inputs and outputs must be provided according to below index order when
constructing an operation.

### Inputs

| Index | Argument Name               | Required or Optional |
|:------|:----------------------------|:---------------------|
| 0     | `src`                       | Required             |
| 1     | `mean`                      | Required             |
| 2     | `variance` (\f$\sigma^2\f$) | Required             |
| 3     | `gamma`                     | Optional             |
| 4     | `beta`                      | Optional             |

@note `gamma` and `beta` should be either both provided or neither provided.

### Outputs

| Index | Argument Name      | Required or Optional |
|:------|:-------------------|:---------------------|
| 0     | `dst`              | Required             |
| 1     | `running_mean`     | Required             |
| 2     | `running_variance` | Required             |
| 3     | `batch_mean`       | Required             |
| 4     | `batch_variance`   | Required             |

## Supported data types

BatchNormInference operation supports the following data type combinations.

| Src / Dst | Gamma / Beta / Mean / Variance / Batch_mean / Batch_variance / Running_mean / Running_variance |
|:----------|:-----------------------------------------------------------------------------------------------|
| f32       | f32                                                                                            |
| bf16      | f32, bf16                                                                                      |
| f16       | f32                                                                                            |
