# Pooling Driver

## Usage
``` sh
    ./benchdnn --pool [benchdnn-knobs] [pool-knobs] [pool-desc] ...
```

where *pool-knobs* are:

 - `--dir={FWD_D [default], FWD_I, BWD_D}` -- dnnl_prop_kind_t.
            Refer to [direction](knobs_dir.md) for details.
 - `--dt={f32:f32:f32 [default], ...}` -- source, weights and destination data
            types. Interface supports broadcasting, when a single input is
            provided, e.g., `--dt=f32`, and the value will be applied for all
            tensors. Refer to [data types](knobs_dt.md) for details.
 - `--cfg={f32 [default], ...}` -- Deprecated setting.
            Refer to ``Configurations`` below.
 - `--tag={nchw [default], ...}` -- physical src and dst memory layout.
            Refer to [tags](knobs_tag.md) for details.
 - `--alg={max [default], avg_np, avg_p}` -- pooling algorithm.
            `max` or `pooling_max` is dnnl_pooling_max;
            `avg_np` or `pooling_avg_exclude_padding` is
                    dnnl_pooling_avg_exclude_padding;
            `avg_p` or `pooling_avg_include_padding` is
                    dnnl_pooling_avg_include_padding;
            Refer to [pooling primitive](https://oneapi-src.github.io/oneDNN/dev_guide_pooling.html)
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

and *pool-desc* is a problem descriptor. The canonical form is:
```
    mbXicX_idXihXiwX_odXohXowX_kdXkhXkwX_sdXshXswX_pdXphXpwX_ddXdhXdwX_nS
```
Refer to [descriptor](knobs_desc.md) for details. Input shape and kernel size
are mandatory inputs. Output shape and padding may be deduced based on the
values provided.

## Precision Configurations

`--cfg` option specifies what [data types](knobs_dt.md) will be used for a
problem. It is implicit for the integer type saturation. This option also
defines the threshold for computation errors.

The table below shows supported name configurations for this driver:

| src   | dst   | cfg   | notes
|:---   |:---   |:---   |:---
| f32   | f32   | f32   | TBA.
| s32   | s32   | s32   |
| f16   | f16   | f16   |
| bf16  | bf16  | bf16  |
| s8    | s8    | s8    |
| u8    | u8    | u8    |
| s8    | u8    | s8u8  | Only on GPU engine
| u8    | s8    | u8s8  | Only on GPU engine
| s8    | f16   | s8f16 | Only on GPU engine
| f16   | s8    | f16s8 | Only on GPU engine
| u8    | f16   | u8f16 | Only on GPU engine
| f16   | u8    | f16u8 | Only on GPU engine
| s8    | f32   | s8f32 | Only on GPU engine
| f32   | s8    | f32s8 | Only on GPU engine
| u8    | f32   | u8f32 | Only on GPU engine
| f32   | u8    | f32u8 | Only on GPU engine


## Essence of Testing
`max` algorithm: Fill input data with integers and expect an integer answer.
`avg_p` algorithm: Fill input data with integers, divisible by the kernel size,
            and expect an integer answer.
`avg_np` algorithm: Fill input data with integers, divisible by the kernel size,
            but expect a float answer due to boarder points have different
            kernel shapes applied to the same point.


## Examples

Run a set of poolings from an input file with the default settings:
``` sh
    ./benchdnn --pool --batch=inputs/pool/shapes_2d
```

Run a named problem with single precision src/dst, iterating by:
1) both blocked memory layouts, where channel blocking equals 8 and 16,
2) both forward training, inference and backward by data prop_kind,
3) all algorithm combinations,
4) using default minibatch of 96 and 5:
``` sh
    ./benchdnn --pool --dt=f32 --tag=nChw8c,nChw16c \
               --dir=FWD_D,FWD_I,BWD_D --alg=max,avg_np,avg_p --mb=0,5 \
               mb96ic768_ih17oh17_kh3sh1ph1n"googlenet_v3:ave_pool_mixed_4_pool"
```

More examples with different driver options can be found at
inputs/pool/test_\*. Examples with different problem descriptors can be found
at inputs/pool/shapes_\*. Examples with different benchdnn common options can
be found at driver_conv.md.
