# Resampling Driver

## Usage
``` sh
    ./benchdnn --resampling [benchdnn-knobs] [resampling-knobs] [resampling-desc] ...
```

where *resampling-knobs* are:

 - `--dir={FWD_D [default], BWD_D}` -- dnnl_prop_kind_t.
            Refer to [direction](knobs_dir.md) for details.
 - `--sdt={f32 [default], ...}` -- src data type.
            Refer to [data types](knobs_dt.md) for details.
 - `--ddt={f32 [default], ...}` -- dst data type.
            Refer to [data types](knobs_dt.md) for details.
 - `--tag={nchw [default], ...}` -- physical src and dst memory layout.
            Refer to [tags](knobs_tag.md) for details.
 - `--alg={nearest [default], linear}` -- resampling algorithm.
            `nearest` or `resampling_nearest` is dnnl_resampling_nearest;
            `linear` or `resampling_nearest` is dnnl_resampling_linear;
            Refer to [resampling primitive](https://oneapi-src.github.io/oneDNN/dev_guide_resampling.html)
            for details.
 - `--attr-post-ops=STRING` -- post operation primitive attribute. No post
            operations are set by default. Refer to [attributes](knobs_attr.md)
            for details.
 - `--mb=INT` -- override minibatch size specified in the problem description.
             When set to `0`, use minibatch size as defined by the individual
             problem descriptor. The default is `0`.
 - `--match=REGEX` -- skip problems not matching the regular expression in
            `REGEX`. By default no pattern is applied (run everything).
            Note: Windows may interpret only string arguments surrounded by
            double quotation marks.

and *resampling-desc* is a problem descriptor. The canonical form is:
```
    mbXicX_idXihXiwX_odXohXowX_nS
```
Refer to [descriptor](knobs_desc.md) for details.

## Essence of Testing
nearest: Fill input data with integers and expect an integer answer.
linear: Fill input data with integers and expect a float answer.


## Examples

Run a set of resamplings from an input file with the default settings:
``` sh
    ./benchdnn --resampling --batch=inputs/resampling/shapes_2d
```

Run a named problem with single precision src/dst, iterating by:
1) both blocked memory layouts, where channel blocking equals 8 and 16,
2) both forward and backward prop_kind,
3) all algorithm combinations,
4) using default minibatch of 96 and 5:
``` sh
    ./benchdnn --resampling --sdt=f32 --ddt=f32 --tag=nChw8c,nChw16c \
               --dir=FWD_D,BWD_D --alg=nearest,linear --mb=0,5 \
               mb96ic768_ih17oh34
```

More examples with different driver options can be found at
inputs/resampling/test_\*. Examples with different problem descriptors can be
found at inputs/resampling/shapes_\*. Examples with different benchdnn common
options can be found at driver_conv.md.
