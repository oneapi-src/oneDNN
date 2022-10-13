# MatMul {#dev_guide_op_matmul}

**Versioned name**: *MatMul-1*

**Category**: Matrix multiplication

**Short description**: Generalized matrix multiplication

## Detailed description

*MatMul* operation takes two tensors and performs usual matrix-matrix
multiplication, matrix-vector multiplication or vector-matrix multiplication
depending on argument shapes. Input tensors can have any \f$rank >= 1\f$. Two
right-most axes in each tensor are interpreted as matrix rows and columns
dimensions while all left-most axes (if present) are interpreted as
multidimensional batch: [BATCH_DIM_1, BATCH_DIM_2,..., BATCH_DIM_K,
ROW_INDEX_DIM, COL_INDEX_DIM]. The operation supports usual broadcast
semantics for batch dimensions. It enables multiplication of batch of pairs of
matrices in a single shot.

Before matrix multiplication, there is an implicit shape alignment for input
arguments. It consists of the following steps:

1. Applying transpositions specified by optional ``transpose_a`` and
   ``transpose_b`` attributes. Only the two right-most dimensions are
   transposed, other dimensions remain the same. Transpose attributes are
   ignored for 1D tensors.

1. One-dimensional tensors unsqueezing is applied for each input independently.
   The axes inserted in this step are not included in the output shape.

   a. If rank of the first input is equal to 1, it is always unsqueezed to 2D
   tensor row vector (regardless of ``transpose_a``) by adding axes with
   size 1 at ROW_INDEX_DIM, to the left of the shape. For example
   \f$[S]\f$ will be reshaped to \f$[1, S]\f$.

   b. If rank of the second input is equal to 1, it is always unsqueezed to 2D
   tensor column vector (regardless of ``transpose_b``) by adding axes with
   size 1 at COL_INDEX_DIM, to the right of the shape. For example
   \f$[S]\f$ will be reshaped to \f$[S, 1]\f$.

1. If ranks of input arguments are different after steps 1 and 2, the tensor
   with a smaller rank is unsqueezed from the left side of the shape by
   necessary number of axes to make both shapes of the same rank.

1. Usual rules of the broadcasting are applied for batch dimensions.

Temporary axes inserted in step 2 are removed from the final output shape after
multiplying. After vector-matrix multiplication, the temporary axis inserted at
ROW_INDEX_DIM is removed. After matrix-vector multiplication, the temporary
axis inserted at COL_INDEX_DIM is removed. Output shape of two 1D tensors
multiplication ``[S]`` x ``[S]`` is squeezed to scalar.

Output shape inference logic examples (ND here means bigger than 1D):

* 1D x 1D: [X] x [X] -> [1, X] x [X, 1] -> [1, 1] => [] (scalar)

* 1D x ND: [X] x [B, ..., X, Y] -> [1, X] x [B, ..., X, Y] -> [B, ..., 1, Y] =>
  [B, ..., Y]

* ND x 1D: [B, ..., X, Y] x [Y] -> [B, ..., X, Y] x [Y, 1] -> [B, ..., X, 1] =>
  [B, ..., X]

* ND x ND: [B, ..., X, Y] x [B, ..., Y, Z] => [B, ..., X, Z]

* ND x MD (M > N): [B, ..., X, Y] x [A, ..., B, ..., Y, Z] =>
  [A, ..., B, ..., X, Z]

* MD x ND (M > N): [A, ..., B, ..., X, Y] x [B, ..., Y, Z] =>
  [A, ..., B, ..., X, Z]

And bias can be 1D tensor or have the same number of dimensions as output
tensor.

Two attributes, ``transpose_a`` and ``transpose_b`` specify embedded
transposition for two right-most dimensions for the first and the second input
tensors correspondingly. It implies swapping of ROW_INDEX_DIM and
COL_INDEX_DIM in the corresponding input tensor. Batch dimensions and 1D
tensors are not affected by these attributes.

## Attributes

* *transpose_a*

  * **Description**: transposes dimensions *ROW_INDEX_DIM* and *COL_INDEX_DIM*
    of the 1st input; False means no transpose, True means transpose
  * **Range of values**: False or True
  * **Type**: bool
  * **Default value**: False
  * **Required**: *no*

* *transpose_b*

  * **Description**: transposes dimensions *ROW_INDEX_DIM* and *COL_INDEX_DIM*
    of the second input; False means no transpose, True means transpose
  * **Range of values**: False or True
  * **Type**: bool
  * **Default value**: False
  * **Required**: *no*

## Inputs

* **1**: ``input_1`` - input batch of matrices A. :math:`Rank >= 1`.
  **Required.**

  * **Type**: T

* **2**: ``input_2`` - input batch of matrices B. :math:`Rank >= 1`.
  **Required.**

  * **Type**: T

* **3**: ``bias`` - input bias. :math:`Rank == 1 or Rank == Rank(output)`.
  Broadcasting is supported. **Optional.**

  * **Type**: T

## Outputs

* **1**: ``output`` - the output tensor.

  * **Type**: T

**Types**:

* **T**: f32, f16, bf16.
* **Note**: Inputs and outputs have the same data type denoted by *T*. For
  example, if input is f32 tensor, then all other tensors have f32 data type.
