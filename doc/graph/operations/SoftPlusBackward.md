SoftPlusBackward {#dev_guide_op_softplusbackward}
=================================================

## General

SoftPlusBackward operation computes gradient for SoftPlus.

## Operation attributes

| Attribute Name                           | Description                         | Value Type | Supported Values                       | Required or Optional |
|:-----------------------------------------|:------------------------------------|:-----------|:---------------------------------------|:---------------------|
| [beta](@ref dnnl::graph::op::attr::beta) | Value for the SoftPlus formulation. | f32        | Arbitrary f32 value (`1.f` by default) | Optional             |

## Execution arguments

The inputs and outputs must be provided according to below index order when
constructing an operation.

### Inputs

| Index | Argument Name | Required or Optional |
|:------|:--------------|:---------------------|
| 0     | `src`         | Required             |
| 1     | `diff_dst`    | Required             |

### Outputs

| Index | Argument Name | Required or Optional |
|:------|:--------------|:---------------------|
| 0     | `diff_src`    | Required             |

## Supported data types

SoftPlusBackward operation supports the following data type combinations.

| Src  | Diff_dst | Diff_src |
|:-----|:---------|:---------|
| f32  | f32      | f32      |
| bf16 | bf16     | bf16     |
| f16  | f16      | f16      |
