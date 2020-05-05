# Attributes

## Usage
```
    [--attr="attr_str"]
```

The attribute string *attr_str* is defined as follows (line breaks are for
readability):
```
    [oscale={none,common,per_oc}[:scale[*]];]
    [scales='arg:{none,common}[:scale][[_]...]';]
    [zero_points=arg:zero_point[*][[_]...];]
    [post_ops='eltwise[:alpha[:beta[:int8_eltwise_scale]]];sum[:sum_scale];';]
```

where `oscale` stands for output_scales. The first parameter is the policy that
is defined below. `scale` is the second optional parameter, which is a real
number that specifies either a common output scale (for `common` policy) or a
starting point for the policies with non-zero mask (e.g. `per_oc`), which uses
many scales. The default scale is `1.0`. Optional asterisk (`*`) after scale
indicates the scales should be passed to a primitive at run-time.

Known policies are:
  - `none` (default) means no output scales set (i.e. scale = 1)
  - `common` corresponds to `mask=0` with common scale factor
  - `per_oc` corresponds to `mask=1<<1` (i.e. output channels) with different
     scale factors
  - `per_dim_0`  corresponds to `mask=1<<0`. Elements corresponding to a single
                 dim0 point have same factor. Different dim0 points have
                 different scaling factors.
  - `per_dim_1`  corresponds to `mask=1<<1`. Elements corresponding to a single
                 dim1 point have same factor. Different dim1 points have
                 different scaling factors.
  - `per_dim_01` corresponds to `mask=(1<<0) + (1<<1)`. Elements corresponding
                 to a single {dim0, dim1} point have same factor. Different
                 {dim0, dim1} points have different scaling factors.


`scales` is used to specify scaling factors for a particular memory argument
of a primitive operation. The first parameter is a name of the argument to
which the scaling factors will be applied. The second parameter is a policy
that has the same semantics and supported values as for `oscale`.
The third parameter is `scale` which has the same semantics as in the `oscale`
case. Scaling factors can be set in any order.
Optional underscore (`_`) is used as a delimiter for different arguments to
improve readability.
Possible arguments:
  - `src` corresponds to `DNNL_ARG_SRC`
  - `src1` corresponds to `DNNL_ARG_SRC_1`


`zero_points` sets zero points for given memory arguments `arg`.
Possible arguments:
  - `src` corresponds to `DNNL_ARG_SRC`
  - `wei` corresponds to `DNNL_ARG_WEIGHTS`
  - `dst` corresponds to `DNNL_ARG_DST`
Optional asterisk (`*`) after zero point value indicates the zero point should
be passed to a primitive at run-time.
Optional underscore (`_`) is used as a delimiter for different arguments to
improve readability.


`post_ops` stands for post operation sequence. All post operations support
output scale (relevant for int8 operations only) which is used as a multiplier
before storing the result. Some post operations support custom alpha and beta
constants with a default value of 0. Operations may be called in any order, e.g.
apply `sum` at first and then apply `relu`, or vice versa - apply `relu` and
then `sum` it with destination. Up to 4 post operations applied is supported.

Currently supported post operations:
  - `sum` -- appends operation result to the output.

Eltwise operations that support no alpha or beta:
  - `abs`
  - `exp`
  - `gelu_tanh`
  - `log`
  - `logistic`
  - `sqrt`
  - `square`
  - `srelu`
  - `tanh`
  - `gelu_erf`

Eltwise operations that support only alpha:
  - `brelu`
  - `elu`
  - `relu`

Eltwise operations that support both alpha and beta:
  - `clip`
  - `linear`
  - `pow`

Depthwise convolution:

This post-op appends depthwise convolution as post-op. The weights of depthwise
convolution have a spatial size of 3 and padding 1. This post-op is currently
only supported for 1x1 convolution.
  - `dw_k3s1p1` -- appends depthwise post-op with stride=1
  - `dw_k3s2p1` -- appends depthwise post-op with stride=2


## Examples:

Run a set of f32 forward convolutions without bias appending accumulation into
destination and perform relu on the output with scale set to 0.5:
``` sh
    ./benchdnn --conv --cfg=f32 --dir=FWD_D \
               --attr=post_ops='sum;relu:0.5' --batch=conv_tails
```

Run a 1D-spatial reorder problem with s8 input data and u8 output data in four
different physical memory layout combinations {ncw, ncw}, {ncw, nwc},
{nwc, ncw} and {nwc, nwc} applying output scale 2.5 for each output point:
``` sh
    ./benchdnn --reorder --sdt=s8 --ddt=u8 \
               --stag=ncw,nwc --dtag=ncw,nwc \
               --attr=oscale=common:2.5 2x8x8
```

Run a binary problem with s8 input data and u8 output data in nc layout
applying scales to both inputs:
``` sh
    ./benchdnn --binary --sdt=u8:s8 --ddt=u8 --stag=nc:nc \
               --attr="scales='src:common:1.5_src1:common:2.5';" \
               100x100:100x100
```

Run a 1x1 convolution fused with depthwise convolution where output scales set
to 0.5 for 1x1 convolution and 1.5 for depthwise post-op followed by a relu
post-op. The final dst datatype after the fusion in the example below is `s8`.
The weights datatype is inferred as `s8`, `f32` and `bf16` for int8, f32 and
bf16 convolutions respectively.
``` sh
  ./benchdnn --conv --cfg=u8s8u8 \
  --attr="oscale=per_oc:0.5;post_ops='relu;dw_k3s1p1:s8:per_oc:1.5;relu';" \
  ic16oc16ih4oh4kh1ph0
```
