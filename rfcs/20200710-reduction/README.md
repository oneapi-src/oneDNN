Proposal for Reduction primitive
====================================================

# Introduction

Reduction functionality is motivated by the following requests from OpenVino team:
* Add support for Reduce operations:
  - ReduceLogicalAnd
  - ReduceLogicalOr
  - ReduceMax
  - ReduceMean
  - ReduceMin
  - ReduceProd
  - ReduceSum
* Add support for Normalize operations:
  - L-2 normalization

In addition to L-2 the following Normalize algorithms are nice-to-have:
  - L-1 normalization
  - L-infinity normalization

The following Reduce and Norm algorithms are supported by the most popular
frameworks:

| Algorithm | Formula                                    | TensorFlow | PyTorch | OpenVino |
| ---------:| ------------------------------------------:| ----------:| -------:| --------:|
| And       | reduce(lambda a, b: a & b, x)              |            |         | x        |
| Or        | reduce(lambda a, b: a \| b, x)             |            |         | x        |
| Min       | reduce(lambda a, b: a if a < b else b, x)  | x          | x       | x        |
| Max       | reduce(lambda a, b: a if a > b else b, x)  | x          | x       | x        |
| Sum       | reduce(lambda a, b: a + b, x)              | x          | x       | x        |
| Prod      | reduce(lambda a, b: a * b, x)              | x          | x       | x        |
| Mean      | sum(x) / n                                 | x          | x       | x        |
| Variance  | Mean(pow([x_i - Mean(x) for x_i in x], 2)) | x          | x       |          |
| Std       | pow(Variance(x), 0.5)                      | x          | x       |          |
| LogSumExp | log(sum([exp(x_i - max(x)) for x_i in x])) + max(x) | x | x       |          |
| Median    |                                            |            | x       |          |
| Mode      |                                            |            | x       |          |
| p-norm    | pow(sum([pow(abs(x_i), p) for x_i in x]), 1/p) | x      | x       |          |

Note: In this RFC Python syntax is used for math formulas

The following algorithms are out of scope due to their low priority:
* And;
* Or;
* Median;
* Mode.

Also there are normalize operations, which variate between frameworks:
* OpenVino:
  ``` python
  def l2_normalize(x, eps_mode, eps):
      l2_norm = pow( eps_mode( sum([pow(x_i, 2) for x_i in x]), eps), 0.5)
      return [x_i / l2_norm for x_i in x]
  ```
  where eps_mode = {add, max}
* TensorFlow:
  ``` python
  def l2_normalize(x, eps):
      l2_norm = pow( max( sum([pow(x_i, 2) for x_i in x]), eps), 0.5)
      return [x_i / l2_norm for x_i in x]
  ```
* PyTorch:
  ``` python
  def l2_normalize(x, eps):
      l2_norm = max( pow( sum([pow(x_i, 2) for x_i in x]), 0.5), eps)
      return [x_i / l2_norm for x_i in x]
  ```

# Proposal

## Common part: Dst memory descriptor.
Dst memory descriptor is required to indicate reduction dimensions and dst data
type. As a result dst memory descriptor also contains information about dst
memory layout. Ideally reduction primitive should preserve memory layout as most
oneDNN primitives to minimize amount of reorders between multiple layouts.
However mixing of reduction operation and oneDNN blocked layouts leads to the
following situations:
* Let's assume user has memory in nChw16c layout and wants to reduce it over all
 dimensions. Dst memory descriptor will have 4 dimensions similar to src, but
 will contain ones in all dimensions. The main issue here is dst layout. Since
 reduction should preserve layout dst will have nChw16c layout which is
 illogical, because 16c guarantees that size c divides by 16 and oneDNN
 internals rely on this. To address this issue dst can be padded to have 16
 elements.
* Let's assume user has memory in nChw16c layout and wants to reduce it over
 dimensions h and w. As before dst memory descriptor will have 4 dimensions
 similar to src, but will contain {n, c, 1, 1} dimensions. In this case layout
 can be either nChw16c or nchw or nhwc and really depends on what user wants to
 do with result.

The following behavior is proposed to address these issues:
* If any is provided for dst layout primitive preserves layout;
* Otherwise primitives uses provided layout.

This behavior would be aligned with binary primitive, that is typically used
along with the reduction.

For example:
``` python
# l2_norm.memory_format ~ nChw16c, with n, h, w = 1
l2_norm = reduction(src, {1, c, 1, 1})
# binary uses the optimized pass as the memory formats are the same
src = binary(div, src, l2_norm)
```

## Option 1: reduction and norms as algorithms of single primitive (Recommended)

### API
``` cpp
// dnnl_types.h
typedef enum {
    ...
    dnnl_reduction_max,
    dnnl_reduction_min,
    dnnl_reduction_norm_lp_max,
    dnnl_reduction_norm_lp_add,
    dnnl_reduction_norm_lp_power_p_max,
    dnnl_reduction_norm_lp_power_p_add,
    dnnl_reduction_sum,
    dnnl_reduction_mul,
    dnnl_reduction_mean,
    ...
} dnnl_alg_kind_t;

typedef struct {
    dnnl_primitive_kind_t primitive_kind;
    /// The kind of reduction algorithm.
    dnnl_alg_kind_t reduction_alg_kind;
    /// Source memory descriptor.
    dnnl_memory_desc_t src_desc;
    /// Destination memory descriptor.
    dnnl_memory_desc_t dst_desc;
    /// Algorithm specific parameters.
    /// Accordance table:
    /// #dnnl_reduction_max: @p p and @eps are ignored
    /// #dnnl_reduction_min: @p p and @eps are ignored
    /// #dnnl_reduction_norm_lp_max: @p p -- power, @eps -- epsilon
    /// #dnnl_reduction_norm_lp_add: @p p -- power, @eps -- epsilon
    /// #dnnl_reduction_norm_lp_power_p_max: @p p -- power, @eps -- epsilon
    /// #dnnl_reduction_norm_lp_power_p_add: @p p -- power, @eps -- epsilon
    /// #dnnl_reduction_sum: @p p and @eps are ignored
    /// #dnnl_reduction_mul: @p p and @eps are ignored
    /// #dnnl_reduction_mean: @p p and @eps are ignored
    float p, eps;
} dnnl_reduction_desc_t;

// dnnl.h
dnnl_status_t DNNL_API dnnl_reduction_desc_init(
        dnnl_reduction_desc_t *reduction_desc,
        dnnl_alg_kind_t reduction_alg_kind, float p, float eps,
        const dnnl_memory_desc_t *src_desc, const dnnl_memory_desc_t *dst_desc);
```

### Pros
* Minimizes amount of post operations required to implement framework operations,
 hence implementation is simpler.

### Cons
* New algorithms on framework side will require changes on oneDNN side.

## Option 2: reduction and norms as combination of eltwise and reduction algorithms

### API

``` cpp
// dnnl_types.h
typedef enum {
    ...
    dnnl_eltwise_abs,
    dnnl_eltwise_pow,
    ...
    dnnl_reduction_max,
    dnnl_reduction_min,
    dnnl_reduction_sum,
    dnnl_reduction_mul,
    ...
} dnnl_alg_kind_t;

typedef struct {
    dnnl_primitive_kind_t primitive_kind;
    // The kind of eltwise algorithm applied before reduction. This
    // algorithm can be undefined.
    dnnl_alg_kind_t eltwise_alg_kind;
    // The kind of reduction algorithm.
    dnnl_alg_kind_t reduction_alg_kind;
    // Eltwise algorithm specific parameters.
    float alpha, beta;
    // Source memory descriptor.
    dnnl_memory_desc_t src_desc;
    // Destination memory descriptor.
    dnnl_memory_desc_t dst_desc;
} dnnl_reduction_desc_t;

// dnnl.h
dnnl_status_t DNNL_API dnnl_reduction_desc_init(
        dnnl_reduction_desc_t *reduction_desc,
        dnnl_alg_kind_t eltwise_alg_kind,
        dnnl_alg_kind_t reduction_alg_kind,
        float alpha, float beta,
        const dnnl_memory_desc_t *src_desc,
        const dnnl_memory_desc_t *dst_desc);
```

### Pros
* Eltwise and reduction are separated hence it provides more flexibility by
 allowing using all combinations of eltwise and reduction;
* Provides a building block. Combination of this
 primitive with other oneDNN primitives allows to implement most of
 framework operations and adding new algorithms for these operations means
 changing combination of oneDNN primitives on framework side instead of
 extending oneDNN monolithic primitive (see option 1).

### Cons
* To implement TF / PyTorch version of l2_norm the following chain of operations
 is required:
  ``` python
  tmp = dnnl_reduction(dnnl_eltwise_pow, dnnl_reduction_sum, 2.0, 0, x)
  tmp = dnnl_binary(dnnl_binary_max, tmp, eps)
  l2_norm = dnnl_eltwise(dnnl_eltwise_pow, 0.5, tmp)
  ```
 In general this option requires more work on implementation side due to post
 operations required to build framework operations.

## Discussion

### Option 1 vs Option 2
From API perspective Option 2 is the best option, because in this case oneDNN
implements reduction functionality as a building block and frameworks operations
can reuse it in combination with other oneDNN primitives. Option 1 on the other
side provides functionality to implement particular framework operations and new
algorithms on framework side will require extending oneDNN reduction API.

However from implementation perspective Option 1 simplifies implementations
because in this case amount of post-operation is minimal or zero since oneDNN
reduction provides functionality for particular framework operations.
