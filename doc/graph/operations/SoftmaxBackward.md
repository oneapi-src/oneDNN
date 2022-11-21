SoftmaxBackward {#dev_guide_op_softmaxbackward}
===============================================

## General

SoftmaxBackward operation computes gradient for Softmax.

## Operation attributes

Attribute Name | Description | Value Type | Supported Values | Required or Optional
-- | -- | -- | -- | --
[axis](@ref dnnl::graph::op::attr::axis) | Represents the axis of which the Softmax is calculated. | s64 | Arbitrary s64 value (`1` in default) | Optional

## Execution arguments

The inputs and outputs must be provided according to below index order when
constructing an operation.

### Inputs

Index | Argument Name | Required or Optional
-- | -- | --
0 | `diff_dst` | Required
1 | `src` | Required

### Outputs

Index | Argument Name | Required or Optional
-- | -- | --
0 | `diff_src` | Required

## Supported data types

SoftmaxBackward operation supports the following data type combinations.

Src | Diff_dst | Diff_src
-- | -- | --
f32 | f32 | f32
bf16 | bf16 | bf16
f16 | f16 | f16
