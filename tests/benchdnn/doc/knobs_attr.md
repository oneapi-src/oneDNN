# Attributes

## Usage
```
    --attr-scratchpad=MODE
    --attr-fpmath=MATHMODE
    --attr-oscale=POLICY[:SCALE[*]]
    --attr-scales=ARG:POLICY[:SCALE[*]][+...]
    --attr-zero-points=ARG:POLICY:ZEROPOINT[*][+...]
    --attr-post-ops=SUM[:SCALE[:ZERO_POINT[:DATA_TYPE]]]
                    ELTWISE[:ALPHA[:BETA[:SCALE]]]
                    DW:KkSsPp[:DST_DT[:OUTPUTSCALE]]
                    BINARY:DT[:POLICY[:TAG]]
```

`--attr-scratchpad` specifies the scratchpad mode to be used for benchmarking.
`MODE` values can be `library` (the default) or `user`. Refer to
[scratchpad primitive attribute](https://oneapi-src.github.io/oneDNN/dev_guide_attributes_scratchpad.html)
for details.

`--attr-fpmath` specifies the fpmath mode to be used for benchmarking.
`MATHMODE` values can be any of `strict` (the default), `bf16`, `f16`, `tf32`,
or `any`. Refer to
[fpmath primitve attribute](https://oneapi-src.github.io/oneDNN/dev_guide_attributes_fpmath_mode.html)
for details.

`--attr-oscale` defines output scale primitive attribute. `POLICY` specifies the
way scale values will be applied to the output tensor. `SCALE` is optional
argument, parsed as a real number that specifies either a common output scale
(for `common` policy) or a starting point for a policy with non-zero mask
(e.g. `per_oc`), which uses many scales. The default scale is `1.0`. Asterisk
mark (`*`) is an optional addition to `SCALE` indicating the scales will be
passed to a primitive at run-time.

`POLICY` supported values are:
  - `none`           (the default) means no output scale is applied.
  - `common`         corresponds to `mask = 0` and means a whole tensor will be
                     multiplied by a single SCALE value.
  - `per_oc`         corresponds to `mask = 1 << 1` and means elements of dim1
                     will be multiplied by scale factors different for each
                     point. Number of scale factors is equal to dims[1].
  - `per_dim_0`      corresponds to `mask = 1 << 0` and means elements of dim0
                     will be multiplied by scale factors different for each
                     point. Number of scale factors is equal to dims[0].
  - `per_dim_1`      same as `per_oc`.
  - `per_dim_01`     corresponds to `mask = (1 << 0) + (1 << 1)` and means
                     elements of dim0 and dim1 will be multiplied by scale
                     factors different for a pair of {dim0, dim1} points.
                     Number of scale factors is equal to dims[0] * dims[1].
  - `per_dim_023`    corresponds to `mask = (1 << 0) + (1 << 2) + (1 << 3)` and
                     means elements of dim0, dim2 and dim3 will be multiplied
                     by scale factors different for {dim0, dim2, dim3} points.
                     Number of scale factors is equal to
                     dims[0] * dims[2] * dim[3]. Intended to use for 4D tensors.
  - `per_dim_23`     corresponds to `mask = (1 << 2) + (1 << 3)` and means
                     elements of dim2 and dim3 will be multiplied by scale
                     factors different for {dim2, dim3} points. Number of scale
                     factors is equal to dims[2] * dim[3]. Intended to use for
                     4D tensors.
  - `per_dim_03`     corresponds to `mask = (1 << 0) + (1 << 3)` and
                     means elements of dim0 and dim3 will be multiplied by
                     scale factors different for {dim0, dim3} points.
                     Number of scale factors is equal to dims[0] * dims[3].
                     Currently supported only in matmul primitive for 4D tensors.
  - `per_dim_2`      corresponds to `mask = 1 << 2` and means elements of dim3
                     will be multiplied by scale factors different for each
                     point. Number of scale factors is equal to dims[2].
                     Currently supported only in matmul primitive for 3D tensors.
  - `per_dim_3`      corresponds to `mask = 1 << 3` and means elements of dim3
                     will be multiplied by scale factors different for each
                     point. Number of scale factors is equal to dims[3].
                     Currently supported only in matmul primitive for 4D tensors.
  - `per_tensor`     means each element of original tensor will be multiplied
                     by a unique number. Number of scale factor is equal to
                     `nelems`. As of now supported only by binary post-ops.

`--attr-scales` defines input scales per memory argument primitive attribute.
`ARG` specifies which memory argument will be modified with input scale.
`POLICY`, `SCALE` and `*` have the same semantics and meaning as for
`--attr-oscale`. To specify more than one memory argument, plus delimiter `+` is
used.

`ARG` supported values are:
  - `src` or `src0` corresponds to `DNNL_ARG_SRC`
  - `src1` corresponds to `DNNL_ARG_SRC_1`
  - `msrci` corresponds to `DNNL_ARG_MULTIPLE_SRC + i`

`POLICY` supported values are:
  - `none`
  - `common`

`--attr-zero-points` defines zero points per memory argument primitive
attribute. This attribute is supported only for integer data types as of now.
`ARG` specifies which memory argument will be modified with zero points.
`POLICY` has the same semantics and meaning as for `--attr-oscale`. `ZEROPOINT`
is an integer value which will be subtracted from each tensor point. Asterisk
mark (`*`) is an optional addition to `ZEROPOINT` indicating the value will be
passed to a primitive at run-time. To specify more than one memory argument,
plus delimiter `+` is used.

`ARG` supported values are:
  - `src` corresponds to `DNNL_ARG_SRC`
  - `wei` corresponds to `DNNL_ARG_WEIGHTS`
  - `dst` corresponds to `DNNL_ARG_DST`

`POLICY` supported values are:
  - `common`
  - `per_dim_1` (for `src` and `dst`, at run-time only)

`--attr-post-ops` defines post operations primitive attribute. Depending on
post operations kind, the syntax differs. To specify more than one post
operation, plus delimiter `+` is used.

`SUM` post operation kind appends operation result to the output. It supports
optional arguments `SCALE` parsed as a real number, which scales the output
before appending the result, `ZERO_POINT` parsed as a integer, which shifts the
output before using the scale and `DATA_TYPE` argument which defines sum data
type parameter. If invalid or `undef` value of `DATA_TYPE` is specified, an
error will be returned.

`ELTWISE` post operation kind applies one of supported element-wise algorithms
to the operation result and then stores it. It supports optional arguments
`ALPHA` and `BETA` parsed as real numbers. To specify `BETA`, `ALPHA` must be
specified. `SCALE` has same notation and semantics as for `SUM` kind, but
requires both `ALPHA` and `BETA` to be specified. `SCALE` is applicable only
when output tensor has integer data type.

`DW:KkSsPp` post operation kind appends depthwise convolution with kernel size
of `k`, stride size of `s`, and left padding size of `p`.
These kinds are applicable only for convolution operation with kernel size of 1
as of now. They support optional argument `DST_DT`, which defines destination
tensor data type. Refer to [data types](knobs_dt.md) for details. Optional
argument `OUTPUTSCALE` defines the semantics of output scale as for
`--attr-oscale` with the same syntax. It requires `DST_DT` to be specified.

`BINARY` post operation kind applies one of supported binary algorithms to the
operation result and then stores it. It requires mandatory argument of `DT`
specifying data type of second memory operand. It supports optional argument of

`PRELU` post operation kind applies forward algorithm to the operations result
and then stores it. Weights `DT` is always implicitly f32.

`POLICY` giving a hint what are the dimensions for a second memory operand. In
case `POLICY` value is `per_tensor`, additional optional argument `TAG` is
supported, positioned after `POLICY`, to specify memory physical format. `TAG`
values use same notation as in drivers. The default value of `TAG` is `any`.
Refer to [tags](knobs_tag.md) for details.

Operations may be called in any order, e.g. apply `SUM` at first and then apply
`ELTWISE`, or vice versa - apply `ELTWISE` and then `SUM` it with destination.

`ELTWISE` supported values are:
  - Eltwise operations that support no alpha or beta:
      - `abs`
      - `exp`
      - `exp_dst`
      - `gelu_erf`
      - `gelu_tanh`
      - `log`
      - `logistic`
      - `logistic_dst`
      - `mish`
      - `round`
      - `sqrt`
      - `sqrt_dst`
      - `square`
      - `tanh`
      - `tanh_dst`
  - Eltwise operations that support only alpha:
      - `elu`
      - `elu_dst`
      - `relu`
      - `relu_dst`
      - `soft_relu`
      - `swish`
  - Eltwise operations that support both alpha and beta:
      - `clip`
      - `clip_v2`
      - `clip_v2_dst`
      - `hardsigmoid`
      - `hardswish`
      - `linear`
      - `pow`

`BINARY` supported values are:
  - `add`
  - `div`
  - `eq`
  - `ge`
  - `gt`
  - `le`
  - `lt`
  - `max`
  - `min`
  - `mul`
  - `ne`
  - `sub`

## Examples:

Run a set of f32 forward convolutions without bias appending accumulation into
destination and perform relu on the output with scale set to 0.5:
``` sh
    ./benchdnn --conv --cfg=f32 --dir=FWD_D \
               --attr-post-ops=sum+relu:0.5 --batch=shapes_tails
```

Run a 1D-spatial reorder problem with s8 input data and u8 output data in four
different physical memory layout combinations {ncw, ncw}, {ncw, nwc},
{nwc, ncw} and {nwc, nwc} applying output scale 2.5 for each output point:
``` sh
    ./benchdnn --reorder --sdt=s8 --ddt=u8 \
               --stag=ncw,nwc --dtag=ncw,nwc \
               --attr-oscale=common:2.5 2x8x8
```

Run a binary problem with s8 input data and u8 output data in nc layout
applying scales to both inputs without any post operations:
``` sh
    ./benchdnn --binary --sdt=u8:s8 --ddt=u8 --stag=nc:nc \
               --attr-scales=src:common:1.5+src1:common:2.5 \
               100x100:100x100
```

Run a 1x1 convolution fused with depthwise convolution where output scales set
to 0.5 for 1x1 convolution and 1.5 for depthwise post-op followed by a relu
post-op. The final dst datatype after the fusion in the example below is `s8`.
The weights datatype is inferred as `s8`, `f32` and `bf16` for int8, f32 and
bf16 convolutions respectively.
``` sh
  ./benchdnn --conv --cfg=u8s8u8 --attr-oscale=per_oc:0.5 \
             --attr-post-ops=relu+dw_k3s1p1:s8:per_oc:1.5+relu \
             ic16oc16ih4oh4kh1ph0
```

Run a convolution problem with binary post operation, first one adds single int
value to the destination memory, second one adds a tensor of same size with
`nhwc`-like physical memory layout with float values to the destination memory:
``` sh
  ./benchdnn --conv --attr-post-ops=add:s32:common,add:f32:per_tensor:axb \
             ic16oc16ih4oh4kh1ph0
```
