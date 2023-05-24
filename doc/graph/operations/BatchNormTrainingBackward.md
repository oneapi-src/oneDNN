BatchNormTrainingBackward {#dev_guide_op_batchnormtrainingbackward}
===================================================================

## General

BatchNormTrainingBackward operation calculated the gradients of input tensors.

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
| 1     | `diff_dst`                  | Required             |
| 2     | `mean`                      | Required             |
| 3     | `variance` (\f$\sigma^2\f$) | Required             |
| 4     | `gamma`                     | Optional             |

### Outputs

| Index | Argument Name | Required or Optional |
|:------|:--------------|:---------------------|
| 0     | `diff_src`    | Required             |
| 1     | `diff_gamma`  | Optional             |
| 2     | `diff_beta`   | Optional             |

@note `diff_gamma` and `diff_beta` should be either both provided or neither
provided. If neither provided, the input `gamma` will be ignored.

## Supported data types

BatchNormTrainingBackward operation supports the following data type
combinations.

| Src / Diff_dst / Diff_src | Mean / Variance / Gamma / Diff_gamma / Diff_beta |
|:--------------------------|:-------------------------------------------------|
| f32                       | f32                                              |
| bf16                      | f32, bf16                                        |
| f16                       | f32                                              |
