# BatchMatMul or extended MatMul support in oneDNN

> Note : In this document, following terms/notations are used interchangeably:
> (src, A), (weights, B), (dst, C) and (Tensor, matrix).

## Motivation
Frameworks request for NumPy-like matmul that can support multiplication of two
Tensors with dimensions of tensors greater or equal to two. Currently, the
existing MatMul primitive supports at most one batch dim (in other words 3D
tensor).

The proposed RFC is an attempt to support following additional semantics over
the existing dnnl_matmul primitive:
  - (i) Multiple batch dimensions and
  - (ii) Broadcasting.

In general operation can be described as:

`C = A * B + bias` , where `*` denotes matrix-matrix multiplication of two inner
dimensions.

An example:
```python
import numpy as np

a = np.random.rand(4, 1, 5, 6, 6)
print(a.shape)
# Out: (4, 1, 5, 6, 6)

b = np.random.rand(3, 1, 2, 1, 6, 6)
print(b.shape)
# Out: (3, 1, 2, 1, 6, 6)

print(np.matmul(a, b).shape) # A will be auto expanded to shape (1, 4, 1, 5, 6, 6).
# Out: (3, 4, 2, 5, 6, 6)
```
For general Broadcasting rules, refer to this [excellent article](https://numpy.org/doc/1.19/user/basics.broadcasting.html).

> Note: GPU or other arch support is out of scope for this RFC.

## Options

Option 0.x are existing known workarounds to support this feature (sometimes
partially).

### Option 0.a Use MKL
Currently TensorFlow uses `cblas_?gemm_batch` [API](https://software.intel.com/content/www/us/en/develop/articles/introducing-batch-gemm-operations.html)
from MKL binary with array of pointers to matrices for A and B.

- Pros:
  - Uses existing optimized MKL implementation.
  - Optimal performance.
  - No maintenance/testing burden on oneDNN.
- Cons:
  - Dependency on MKL binary.

### Option 0.b Call GeMM in a loop
Example: A = {3 x M x K} and B = {1 x K x N}
```python
for i in range(3) :
  C[i] = GeMM(A[i], B)
```

- Pros:
  - Uses existing GeMM impl.
  - No maintenance/testing burden on oneDNN.
- Cons: Performance may not be optimal.

### Option 1 Extend MatMul primitive

This primitive is similar to oneMKL [API](https://software.intel.com/content/www/us/en/develop/documentation/mkl-developer-reference-c/top/blas-and-sparse-blas-routines/blas-like-extensions/cblas-gemm-batch-strided.html)
`cblas_?gemm_batch_strided`, as all matrices have fixed stride.

**Broadcasting**
  - Support implicit broadcasting, but keep the restriction
  `ndims(A) == ndims(B) == ndims(C) == ndims(bias)` (Note: Unfortunately, this
  means that the example at the beginning of this doc is not directly supported.
  The tensor A must be manually expanded by user to match number of dims of B).
  - Broadcasting on dims `M`, `N` and `K` is invalid and not supported.
  - The shape of `dst` only depends on `A` and `B` tensors. The `bias` cannot
  change dimensions of `dst` by broadcasting. In other words for every dim, the
  following condition must hold true `dim(bias) == dim(dst) || dim(bias) == 1`.

**Layouts**
  - Support only plain and transposed layouts. For example, in a 5D tensor only
  `abcde` and `abced` layouts are supported. Requires adding two tags (one for
  plain and one for transpose) for each dim if not already present to the
  library.

**Maximum Dimensions**
  - Currently oneDNN supports maximum of `12` dimensions for a tensor. This
  implies that the maximum number of batch dimensions can be at most 10.

**DNNL_RUNTIME_DIM_VAL and Broadcasting**
  - The broadcasting consistency check is not done for the dimensions with
  `DNNL_RUNTIME_DIM_VAL`. It is user responsibility to make sure the dims for
  the tensors are valid (consistent with current runtime behavior of M, N, K
  dims).
  - Inconsistency with dims being "fully-defined" vs "runtime-defined" is
  invalid (consistent with current behavior). Meaning, a "fully-defined" dim in
  tensor A with a corresponding dim in B set as `DNNL_RUNTIME_DIM_VAL` is not
  allowed. For example, `A` and `B` with dims set to `{3, 4, 4` and
  `{DNNL_RUNTIME_DIM_VAL, 4, 4}` respectively is invalid.

**Zero points**: As is (scalar value).

**Scales mask**: As is: (common scaling and per_oc (last dim, i.e.,
`mask = 1 << (ndim -1)`)).

**Primitive Cache**: No changes expected.

**Benchdnn testing**
We can keep the existing MatMul benchdnn driver for most part except `%desc%`.

- Existing `%desc%`: An example shape for current `%desc%` looks as
`mb2048m40n40k64`.

- New `%desc%`:
  - To support generic Tensor multiplication with support for arbitrary dims and
  broadcast we can use `2048x16x64:1x64x32[:2048x16x32]`, with optional `C`
  dims. If `C` dims are not specified, they are calculated by driver. Here `A`,
  `B` and `C` have shapes `{2048x16x64}`, `{1x64x32}` and `{2048x16x32}`
  - Runtime dims are supported using the option
  `--runtime_dims_masks=[INT][:INT]` -- a bit-mask values for `src` and
  `weights` that indicates if a dim is `DNNL_RUNTIME_DIM_VAL` (indicated as
  1-bit in corresponding dim position). Default is `0` for all dims, meaning all
  tensor dimensions are fully defined at primitive creation.

**Pros and Cons**
- Pros:
  - Can use existing primitive. No change in API.
  - Follows "Unified GeMM API" intent.
- Cons:
  - No automatic expansion of dims. This means product of Tensors `A` and `B` as
  `{d0, M, K}` and `{K, N}`  respectively are not directly supported. It is user
  responsibility to expand the dims of `B`.
  - Deviation from an earlier behavior due to broadcasting. Previously, if dims
  did not match, the library would throw an error. Now, it tries to broadcast.
  - Requires benchdnn matmul driver interface change for problem description.
  - Maintenance/Testing of an additional functionality in oneDNN.
  - Additional work is required to achieve perf parity with MKL.

### Option 2 New BLAS function "BatchGeMM"
(or any other name suggestions are welcome)

The goal of this primitive is to achieve the flexibility of oneMKL [API](https://software.intel.com/content/www/us/en/develop/documentation/mkl-developer-reference-c/top/blas-and-sparse-blas-routines/blas-like-extensions/cblas-gemm-batch.html)
`cblas_?gemm_batch`.

Initial mock-up of API by closely following oneMKL `cblas_sgemm_batch` and
`dnnl_sgemm`. Need more thorough work on the API if there is a significant
interest in this option. Also benchdnn testing.

```C++
dnnl_sgemm_batch(const char *transa_array, const char *transb_array,
  const dnnl_dim_t *M_array, const dnnl_dim_t *N_array,
  const dnnl_dim_t *K_array, const float* alpha_array, const float **A_array,
  const dnnl_dim_t *lda_array, const float **B_array,
  const dnnl_dim_t *ldb_array, const float *beta_array, float **c_array,
  const dnnl_dim_t *ldc_array, const dnnl_dim_t group_count,
  const dnnl_dim_t *group_size);

// And its siblings for other data types.
```

- Pros:
  - New API to clearly distinguish batch support.
  - Directly maps to a well known BLAS API.
  - Flexible to support combination of transpose and non-transpose GeMMs in
  single call.
  - No limitations on number of batch dims.
  - No need of additional layouts per new dim.
- Cons:
  - Needs user effort to create array of pointers.
  - Does not follow oneDNN programming model `op_desc->primitive_desc->primitive`.
  - Potential performance-hit/complexity as there is no guarantee that buffers
  are contiguous or overlapped.
    - May be add a enum/flag in the API, to get a guarantee that buffers are
    contiguous and do not overlap or alias to optimize for most common case?
  - Breaks "Unified GeMM API" (which was key aspect of MatMul primitive).
  - Maintenance/Testing of an additional primitive in oneDNN.
  - Additional work is required to achieve perf parity with MKL.

**Recommended : Option 1 (Extend with MatMul primitive).**

Rationale:
In an ideal world with infinite resources we could support both approaches like
most BLAS libraries. In this RFC we chose option-1 as it is simpler and natural
extension to an existing primitive libraries/frameworks mostly solving the
problem described in motivation section. Option-2 is quite broad, Swiss Army
Knife solution to tensor multiplication, that is best left to consider as a
separate feature.
