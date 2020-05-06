# Layer Normalization Driver

## Usage
``` sh
    ./benchdnn --lnorm [benchdnn-knobs] [lnorm-knobs] [lnorm-desc] ...
```

where *lnorm-knobs* are:

 - `--dir={FWD_D [default], FWD_I, BWD_D, BWD_DW}` -- dnnl_prop_kind_t.
            Refer to the common glossary in README.md for details.
 - `--dt={f32 [default]}` -- src and dst data types.
            Refer to the common glossary in README.md for details.
 - `--tag={tnc [default], ...}` -- physical src and dst memory format.
            Refer to the common glossary in README.md for details.
 - `--stat_tag={tn [default], ...}` -- physical mean and variance memory format.
            Refer to the common glossary in README.md for details.
 - `--flags=[|G|S]` -- layer normalization flags, default `none`; where
            multiple simultaneous flags are supported.
            `G` is dnnl_use_global_stats;
            `S` is dnnl_use_scaleshift;
            Refer to ``doc/primitives/layer_normalization.md`` for details.
 - `--attr="attr_str"` -- primitive attributes, default `""` (no attributes).
            Refer to [attributes](knobs_attr.md) for details.
 - `--inplace=BOOL` -- memory mode for the primitive. If `true`, it uses input
            memory as output, otherwise, input and output are separate.
            Default is `true`.

and *lnorm-desc* is a problem descriptor. The canonical form is:
```
    NxNxN
```
where N is an integer number. This represents a problem with the
following logical dimensions: T, N, C. Consider removing each `xN` from
the end to specify fewer dimensions.

There are default values for some entities in case they were not specified:
 - eps = 1./16;

## Essence of Testing
TBA.


## Examples

Run a set of lnorms from an input file, using the default settings:
``` sh
    ./benchdnn --lnorm --batch=inputs/lnorm/lnorm_all
```

Run a named problem with single precision src/dst, iterating by:
1) Src/dst memory formats
2) Statistics memory formats
3) forward training, backward by data and weights prop_kinds,
4) all flag combinations:
``` sh
    ./benchdnn --lnorm --dt=f32 --tag=tnc,ntc --stat_tag=tn,nt\
               --dir=FWD_D,BWD_DW --flags=GS,S \
               8x32x1024
```

More examples with different driver options can be found at
inputs/lnorm/test_lnorm_all. Examples with different driver descriptors can be
found at inputs/lnorm/lnorm_***. Examples with different benchdnn options can be
found at driver_conv.md.
