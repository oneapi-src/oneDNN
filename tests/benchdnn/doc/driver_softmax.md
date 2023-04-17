# Softmax Driver

## Usage
``` sh
    ./benchdnn --softmax [benchdnn-knobs] [softmax-knobs] [softmax-desc] ...
```

where *softmax-knobs* are:

 - `--dir={FWD_D [default], FWD_I, BWD_D}` -- dnnl_prop_kind_t.
            Refer to [direction](knobs_dir.md) for details.
 - `--sdt={f32 [default], bf16, f16, s8, u8}` -- src data type.
            Refer to [data types](knobs_dt.md) for details.
 - `--ddt={f32 [default], bf16, f16, s8, u8}` -- dst data type.
            Refer to [data types](knobs_dt.md) for details.
 - `--stag={abx [default], ...}` -- physical src memory layout.
            Refer to [tags](knobs_tag.md) for details.
 - `--dtag={any [default], ...}` -- physical dst memory layout.
            Refer to [tags](knobs_tag.md) for details.
 - `--alg={SOFTMAX [default], LOGSOFTMAX}` -- softmax algorithm.
            `SOFTMAX` or `softmax_accurate` is `dnnl_softmax_accurate`;
            `LOGSOFTMAX` or `softmax_log` is `dnnl_softmax_log`;
            Refer to [softmax primitive](https://oneapi-src.github.io/oneDNN/dev_guide_softmax.html)
            for details.
 - `--axis=INT` -- dimension on which operation will be performed.
            Default is `1`, corresponds to channels in logical memory layout.
 - `--attr-scales=STRING` -- per argument scales primitive attribute. No
            scales are set by default. Refer to [attributes](knobs_attr.md) for
            details.
 - `--attr-post-ops=STRING` -- post operation primitive attribute. No post
            operations are set by default. Refer to [attributes](knobs_attr.md)
            for details.
 - `--mb=INT` -- override minibatch size specified in the problem description.
             When set to `0`, use minibatch size as defined by the individual
             problem descriptor. The default is `0`.
 - `--inplace=BOOL` -- memory mode for the primitive. If `true`, it uses input
            memory as output, otherwise, input and output are separate.
            The default is `false`.
 - `--match=REGEX` -- skip problems not matching the regular expression in
            `REGEX`. By default no pattern is applied (run everything).
            Note: Windows may interpret only string arguments surrounded by
            double quotation marks.

and *softmax-desc* is a problem descriptor. The canonical form is:
```
    NxNxNxNxN
```
where N is an integer number. This represents a 3D spatial problem with the
following logical dimensions: N, C, D, H, W. Consider removing each `xN` from
the end to specify fewer dimensions.


## Essence of Testing
### Forward
Fill data the way it tests two modes: max_val < 0 and max_val >= 0;
Test max_val < 0 by using only negative numbers to check correct max_val
subtraction, mostly if library used signed value, not abs.
Test max_val >= 0 by exceeding `exp_overflow_arg` value to check answer does not
contain +infinity (nan) in the answer.

### Backward
Fill input data with negative integers, and expect positive output. This avoids
potential cancellation errors.


## Examples

Run the softmax set from an input file with the default settings:
``` sh
    ./benchdnn --softmax --batch=shapes_ci
```

Run a specific softmax problem with forward prop_kind, plain physical memory
layouts, f32 source and destination data types, out-place memory mode, and axis
size of 1000:
``` sh
    ./benchdnn --softmax --dir=FWD_D --sdt=f32 --ddt=f32 --stag=nc \
               --inplace=false --axis=1 256x1000
```

Run a specific logsoftmax problem with backward prop_kind, default physical
memory layouts, default data types, in-place memory mode, and axis size of 64:
``` sh
    ./benchdnn --softmax --dir=BWD_D --inplace=true \
               --alg=LOGSOFTMAX --axis=3 1x2x112x64
```

More examples with different driver options can be found at
inputs/softmax/test_\*. Examples with different benchdnn common options can be
found at driver_conv.md.
