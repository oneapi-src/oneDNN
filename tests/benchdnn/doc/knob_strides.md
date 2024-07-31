# Strides specification

## Introduction
oneDNN provides several interfaces to specify a memory format for a memory
object passed to the library. One of these interfaces is stride specification.

Strides are the distances between consecutive points along the same dimension.
For example, tensor `2x3` can be represented in multiple ways:
- If strides are `3x1`, it is the same tensor as initialized with
  `format_tag::ab`.
- If strides are `1x2`, it is the same tensor as initialized with
  `format_tag::ba`.
- If strides are `6x1`, the tensor has a notion of `format_tag::ab` but has
  padding over `dims[1]` where the first three points are meaningful, and the
  next three are just dummy spaces in memory. This means that the memory object
  has at least `2*6` elements in memory, unless there's more padding over
  `dims[0]` dimension.

## Usage
```
    --strides=SRC_DIMS[:WEI_DIMS]:DST_DIMS
```

This is a general representation of strides option support across benchdnn.
The order of arguments is always the same: source, weights, and destination.
Weights strides are required if the driver has weights support. However, they
can't be specified if the driver doesn't have a notion of weights for the
operation, and only two inputs are expected in this case.

`DIMS` generic form is \f$D_0xD_1x...xD_n\f$, where `n` should match the problem
number of dimensions.

A colon (or two, if weights are supported) must be present in the final line.
Thus, `--strides=:` is an identical syntax to `--strides=`.

Just source strides specification looks like this: `--strides=3x4x5x6:`.

`--strides` and (any-flavored) `--tag` options are mutually excluded and can't
be supported for a single argument simultaneously. However, for the example
above, specifying `--dtag=TAG` would be legit.

## Limitations

The option has limited support across drivers in benchdnn.
