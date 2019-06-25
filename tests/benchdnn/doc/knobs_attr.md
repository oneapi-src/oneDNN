# Attributes

## Usage
```
    [--attr="attr_str"]
```

The attribute string *attr_str* is defined as follows (line breaks are for
readability):
```
    [oscale={none,common,per_oc}[:scale];]
    [post_ops='[{relu,sum[:sum_scale]};]...';]
```

`oscale` stands for output_scales. The first parameter is the policy that
is defined below. `scale` is the second optional parameter, which is a real
number that specifies either the one common output scale (for the `none` and
`common` polices) or a starting point for the `per_oc` policy, which uses many
scales. The default scale is `1.0`.

Known policies are:
  - `none` (default) means no output scales set (i.e. scale = 1)
  - `common` corresponds to `mask=0` with common scale factor
  - `per_oc` corresponds to `mask=1<<1` (i.e. output channels) with different
     scale factors
  - `per_dim_0` corresponds to ???. TBA.
  - `per_dim_1` corresponds to ???. TBA.
  - `per_dim_01` corresponds to ???. TBA.

`post_ops` stands for post operation sequence. Currently supported post
operations are:

#TBA about optional scale. Looks like it is not everywhere reasonable or even
supported.
  - `sum` with optional parameter scale (default is 1.0)
  - `relu` with optional parameter scale (default is 1.0)
  - `brelu` with optional parameter scale (default is 1.0)
  - `srelu` with optional parameter scale (default is 1.0)
  - `tanh` with optional parameter scale (default is 1.0)
  - `elu` with optional parameter scale (default is 1.0)
  - `abs` with optional parameter scale (default is 1.0)
  - `sqrt` with optional parameter scale (default is 1.0)
  - `linear` with optional parameter scale (default is 1.0)
  - `logistic` with optional parameter scale (default is 1.0)
  - `exp` with optional parameter scale (default is 1.0)
  - `square` with optional parameter scale (default is 1.0)

## Examples:

Run a set of f32 forward convolutions without bias appending accumulation into
destination and performs relu on the output with scale set to 0.5:
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
