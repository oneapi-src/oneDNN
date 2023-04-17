# Layer Normalization Driver

## Usage
``` sh
    ./benchdnn --lnorm [benchdnn-knobs] [lnorm-knobs] [lnorm-desc] ...
```

where *lnorm-knobs* are:

 - `--dir={FWD_D [default], FWD_I, BWD_D, BWD_DW}` -- dnnl_prop_kind_t.
            Refer to [direction](knobs_dir.md) for details.
 - `--dt={f32:f32 [default], ...}` -- src and dst data types.
            Refer to [data types](knobs_dt.md) for details.
 - `--tag={tnc:any [default], ...}` -- physical src and dst memory format.
            If only src memory format is provided, dst memory format will be set
            to `any`. Refer to [tags](knobs_tag.md) for details.
 - `--stat_tag={tn [default], ...}` -- physical mean and variance memory format.
            Refer to [tags](knobs_tag.md) for details.
 - `--flags=[|G|C|H]` -- layer normalization flags, default `none`; where
            multiple simultaneous flags are supported.
            `G` is dnnl_use_global_stats;
            `C` is dnnl_use_scale;
            `H` is dnnl_use_shift;
            Refer to [layer normalization primitive](https://oneapi-src.github.io/oneDNN/dev_guide_layer_normalization.html)
            for details.
 - `--attr-scales=STRING` -- per argument scales primitive attribute. No
            scales are set by default. Refer to [attributes](knobs_attr.md) for
            details.
 - `--inplace=BOOL` -- memory mode for the primitive. If `true`, it uses input
            memory as output, otherwise, input and output are separate.
            Default is `false`.
 - `--match=REGEX` -- skip problems not matching the regular expression in
            `REGEX`. By default no pattern is applied (run everything).
            Note: Windows may interpret only string arguments surrounded by
            double quotation marks.

and *lnorm-desc* is a problem descriptor. The canonical form is:
```
    NxNxN
```
where N is an integer number. This represents a problem with the
following logical dimensions: T, N, C. Consider removing each `xN` from
the end to specify fewer dimensions.

## Essence of Testing
TBA.


## Examples

Run a set of lnorms from an input file, using the default settings:
``` sh
    ./benchdnn --lnorm --batch=shapes_ci
```

Run a named problem with single precision src/dst, iterating by:
1) Src/dst memory formats
2) Statistics memory formats
3) forward training, backward by data and weights prop_kinds,
4) some flag combinations:
``` sh
    ./benchdnn --lnorm --dt=f32 --tag=tnc,ntc --stat_tag=tn,nt \
               --dir=FWD_D,BWD_DW --flags=GCH,CH 8x32x1024
```

Run the same problem as previous but with different data types for source and
destination:
``` sh
    ./benchdnn --lnorm --dt=bf16:f32 --tag=tnc,ntc --stat_tag=tn,nt \
               --dir=FWD_D,BWD_DW --flags=GCH,CH 8x32x1024
```

More examples with different driver options can be found at
inputs/lnorm/test_\*. Examples with different problem descriptors can be found
at inputs/lnorm/shapes_\*. Examples with different benchdnn common options can
be found at driver_conv.md.
