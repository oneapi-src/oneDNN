# Group Normalization Driver

## Usage
``` sh
    ./benchdnn --gnorm [benchdnn-knobs] [gnorm-knobs] [gnorm-desc] ...
```

where *gnorm-knobs* are:

 - `--dir={FWD_D [default], FWD_I, BWD_D, BWD_DW}` -- dnnl_prop_kind_t.
            Refer to [direction](knobs_dir.md) for details.
 - `--dt={f32:f32 [default], ...}` -- src and dst data types.
            Refer to [data types](knobs_dt.md) for details.
 - `--tag={abcd:any [default], ...}` -- physical src and dst memory format.
            If only src memory format is provided, dst memory format will be set
            to `any`. Refer to [tags](knobs_tag.md) for details.
 - `--flags=[|G|C|H]` -- group normalization flags, default `none`; where
            multiple simultaneous flags are supported.
            `G` is dnnl_use_global_stats;
            `C` is dnnl_use_scale;
            `H` is dnnl_use_shift;
            Refer to [group normalization primitive](https://oneapi-src.github.io/oneDNN/dev_guide_group_normalization.html)
            for details.
 - `--attr-scales=STRING` -- per argument scales primitive attribute. No
            scales are set by default. Refer to [attributes](knobs_attr.md) for
            details.
 - `--attr-post-ops=STRING` -- post operation primitive attribute. No post
            operations are set by default. Refer to [attributes](knobs_attr.md)
            for details.
 - `--inplace=BOOL` -- memory mode for the primitive. If `true`, it uses input
            memory as output, otherwise, input and output are separate.
            Default is `false`.
 - `--match=REGEX` -- skip problems not matching the regular expression in
            `REGEX`. By default no pattern is applied (run everything).
            Note: Windows may interpret only string arguments surrounded by
            double quotation marks.

and *gnorm-desc* is a problem descriptor. The canonical form is:
```
    gXmbXicX_idXihXiwX_epsF_nS
```
Refer to [descriptor](knobs_desc.md) for details. `epsF` stands for Group
Normalization epsilon value and accepts float F values. The default is `1.f/16`.


## Examples

Run a set of gnorms from an input file, using the default settings:
``` sh
    ./benchdnn --gnorm --batch=shapes_ci
```

Run a named problem with single precision src/dst, iterating by:
1) Src/dst memory formats
2) forward training, backward by data and weights prop_kinds,
3) some flag combinations:
``` sh
    ./benchdnn --gnorm --dt=f32 --tag=abc,acb --dir=FWD_D,BWD_DW \
               --flags=GCH,CH g5ic5iw10_n"gnorm_test_shape"
```

Run the same problem as previous but with different data types for source and
destination:
``` sh
    ./benchdnn --gnorm --dt=bf16:f32 --tag=abc,acb --dir=FWD_D,BWD_DW \
               --flags=GCH,CH g5ic5iw10_n"gnorm_test_shape"
```

More examples with different driver options can be found at
inputs/gnorm/test_\*. Examples with different problem descriptors can be found
at inputs/gnorm/shapes_\*. Examples with different benchdnn common options can
be found at driver_conv.md.
