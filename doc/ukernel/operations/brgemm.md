Batch-Reduce General Matrix Multiplication {#dev_guide_ukernel_brgemm}
=======================================

>
> [API Reference](@ref dnnl_api_ukernel_brgemm)
>


## General

The batch-reduce General Matrix Multiplication ukernel (BRGeMM) is an operation
that computes a small matrix multiplication batch and accumulates their results
in the same destination.

\f$C = \sum_i A_i \cdot B_i\f$

with
- \f$A_i\f$ a set of matrices of dimension \f$M \times K\f$
- \f$B_i\f$ a set of matrices of dimension \f$K \times N\f$
- \f$C\f$ matrix of dimension \f$M \times N\f$.

The BRGeMM ukernel also supports accumulation with values already present in
\f$C\f$, as well as post-operation and down-conversion to another \f$D\f$
matrix:

\f$D = \operatorname{convert}( \operatorname{post\_ops}(C + \sum_i A_i \cdot B_i, post\_ops\_args))\f$

## Data Types

In general, C represents an accumulation buffer. Hence, when computations are
carried in floating-point arithmetic, C shall be of type f32; when computation
is carried in integer arithmetic, C should be of type s32.

The BRGeMM ukernel supports the following combinations of data-types.

| A      | B      | C   | D                           |
|:-------|:-------|:----|:----------------------------|
| f32    | f32    | f32 | u8, s8, s32, f32, f16, bf16 |
| f16    | f16    | f32 | u8, s8, s32, f32, f16, bf16 |
| bf16   | bf16   | f32 | u8, s8, s32, f32, f16, bf16 |
| u8, s8 | u8, s8 | s32 | u8, s8, s32, f32, f16, bf16 |

## Data Representation

Because of hardware restrictions, the BRGeMM ukernel requires a specific data
layout. For x86-64 architecture this layout applies to a B matrix. It is
expressed through #dnnl::ukernel::pack_type which can be queried by
#dnnl::ukernel::brgemm::get_B_pack_type call. If the query returns
#dnnl::ukernel::pack_type::no_trans, then packing is not required.
Otherwise, the user is responsible for packing the data appropriately before
calling #dnnl::ukernel::brgemm::execute, either with custom code, or by
using a dedicated set of APIs: #dnnl::ukernel::transform::generate for
generating a kernel of a transform routine and
#dnnl::ukernel::transform::execute to run the generated kernel.

## Attributes

The following ukernel attributes can be set through dedicated setters.

| Type      | Operation                                                  | Description                                               | Restrictions                        |
|:----------|:-----------------------------------------------------------|:----------------------------------------------------------|:------------------------------------|
| Attribute | [Scales](@ref dnnl::primitive_attr::set_scales_mask)       | Scales the corresponding tensors by given scale factor(s) |                                     |
| Post-op   | [Eltwise](@ref dnnl::post_ops::append_eltwise)             | Applies an @ref dnnl_api_eltwise operation to the result  |                                     |
| Post-op   | [Binary](@ref dnnl::post_ops::append_binary)               | Applies a @ref dnnl_api_binary operation to the result    | General binary post-op restrictions |


@note if zero-points are passed for A/B, fpmath_mode should be set for the
computation to happen over floating-point format (so up-conversion to
floating-point format would happen before computation). If computation in
integer format is needed, BRGeMM ukernel should be configured without
zero-point, and the user should prepare a compensation term that will be passed
to the binary post-op.

## Implementation limitations

BRGeMM ukernel has no known limitations.

## Examples

[BRGeMM ukernel example](@ref cpu_brgemm_example_cpp)

@copydetails cpu_brgemm_example_cpp
