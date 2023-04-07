MatMul {#dev_guide_op_matmul}
=============================

## General

MatMul operation computes the product of two tensors with optional bias addition
The variable names follow the standard @ref dev_guide_conventions, typically
taking 2D input tensors as an example, the formula is below:

\f[
    \dst(m, n) =
        \sum_{k=0}^{K - 1} \left(
            \src(m, k) \cdot \weights(k, n)
        \right) +
        \bias(m, n)
\f]

In the shape of a tensor, two right-most axes are interpreted as row and column
dimensions of a matrix while all left-most axes (if present) are interpreted as
batch dimensions. The operation supports broadcasting semantics for those batch
dimensions. For example \src can be broadcasted to \weights if the corresponding
dimension in \src is `1` (and vice versa). Additionally, if ranks of \src and
\weights are different, the tensor with a smaller rank will be *unsqueezed* from
the left side of dimensions (inserting `1`) to make sure two ranks matched.

## Operation attributes

| Attribute Name                                         | Description                                                         | Value Type | Supported Values      | Required or Optional |
|:-------------------------------------------------------|:--------------------------------------------------------------------|:-----------|:----------------------|:---------------------|
| [transpose_a](@ref dnnl::graph::op::attr::transpose_a) | Controls whether to transpose the last two dimensions of `src`.     |bool        | True, False (default) | Optional             |
| [transpose_b](@ref dnnl::graph::op::attr::transpose_b) | Controls whether to transpose the last two dimensions of `weights`. |bool        | True, False (default) | Optional             |

The above transpose attribute will not in effect when rank of an input tensor is
less than 2. For example, in library implementation 1D tensor is unsqueezed
firstly before compilation. The rule is applied independently.

- For \src tensor, the rule is defined like: `[d] -> [1, d]`.

- For \weights tensor, the rule is defined like: `[d] -> [d, 1]`.

## Execution arguments

The inputs and outputs must be provided according to below index order when
constructing an operation.

### Inputs

| Index | Argument Name | Required or Optional |
|:------|:--------------|:---------------------|
| 0     | `src`         | Required             |
| 1     | `weights`     | Required             |
| 2     | `bias`        | Optional             |

### Outputs

| Index | Argument Name | Required or Optional |
|:------|:--------------|:---------------------|
| 0     | `dst`         | Required             |

## Supported data types

MatMul operation supports the following data type combinations.

| Src  | Weights | Bias | Dst  |
|:-----|:--------|:-----|:-----|
| f32  | f32     | f32  | f32  |
| bf16 | bf16    | bf16 | bf16 |
| f16  | f16     | f16  | f16  |
