HardSigmoidBackward {#dev_guide_op_hardsigmoidbackward}
=======================================================

## General

HardSigmoidBackward operation computes gradient for HardSigmoid. The formula is
defined as follows:

\f[ diff\_src = \begin{cases} diff\_dst \cdot \alpha & \text{if}\ 0 < \alpha src + \beta < 1 \\ 0 & \text{otherwise}\ \end{cases} \f]

## Operation attributes

| Attribute Name                             | Description                    | Value Type | Supported Values     | Required or Optional |
|:-------------------------------------------|:-------------------------------|:-----------|:---------------------|:---------------------|
| [alpha](@ref dnnl::graph::op::attr::alpha) | \f$ \alpha \f$ in the formula. | f32        | Arbitrary f32 value. | Required             |
| [beta](@ref dnnl::graph::op::attr::beta)   | \f$ \beta \f$ in the formula.  | f32        | Arbitrary f32 value. | Required             |

## Execution arguments

The inputs and outputs must be provided according to the index order shown below
when constructing an operation.

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

HardSigmoidBackward operation supports the following data type combinations.

| Src  | Diff_dst | Diff_src |
|:-----|:---------|:---------|
| f32  | f32      | f32      |
| f16  | f16      | f16      |
| bf16 | bf16     | bf16     |
