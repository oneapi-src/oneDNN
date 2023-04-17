# Reduction Driver

## Usage
``` sh
    ./benchdnn --reduction [benchdnn-knobs] [reduction-knobs] [reduction-desc] ...
```

where *reduction-knobs* are:

 - `--sdt={f32 [default], bf16, f16, s8, u8, s32}` -- src data type.
            Refer to [data types](knobs_dt.md) for details.
 - `--ddt={f32 [default], bf16, f16, s8, u8, s32}` -- dst data type.
            Refer to [data types](knobs_dt.md) for details.
 - `--stag={nchw [default], ...}` -- physical src memory layout.
            Refer to [tags](knobs_tag.md) for details.
 - `--dtag={any [default], ...}` -- physical dst memory layout.
            Refer to [tags](knobs_tag.md) for details.
 - `--alg={sum [default], ...}` -- algorithm for reduction operations.
            Refer to [reduction primitive](https://oneapi-src.github.io/oneDNN/dev_guide_reduction.html)
            for details.
 - `--p=FLOAT` -- float value corresponding to algorithm operation.
            Refer to ``Floating point arguments`` below.
 - `--eps=FLOAT` -- float value corresponding to algorithm operation.
            Refer to ``Floating point arguments`` below.
 - `--attr-post-ops=STRING` -- post operation primitive attribute. No post
            operations are set by default. Refer to [attributes](knobs_attr.md)
            for details.
 - `--match=REGEX` -- skip problems not matching the regular expression in
            `REGEX`. By default no pattern is applied (run everything).
            Note: Windows may interpret only string arguments surrounded by
            double quotation marks.

and *reduction-desc* is a problem descriptor. The canonical form is:
```
    NxNxNxNxN:NxNxNxNxN
```
where N is an integer number. This represents a 3D spatial problem with the
following logical dimensions: N, C, D, H, W. Consider removing each `xN` from
the end to specify fewer dimensions.

## Floating point arguments
Some operations support `p` and `eps` arguments such as
`norm_lp_max`, `norm_lp_sum`, `norm_lp_power_p_max`, `norm_lp_power_p_sum`.

## Essence of Testing

## Examples

Run the set of reduction primitive problems from `inputs/reduction/shapes_ci`
with the default settings:
``` sh
    ./benchdnn --reduction --batch=inputs/reduction/shapes_ci
```

Run a specific reduction primitive problem:
- Data type is `f32` for source and destination tensors.
- Source tensor uses `acb` memory format.
- The reduce operation is sum.
``` sh
    ./benchdnn --reduction --sdt=f32 --ddt=f32 --stag=acb --alg=sum 1x2x3:1x1x3
```

More examples with different driver options can be found at
inputs/reduction/test_\*. Examples with different problem descriptors can be
found at inputs/reduction/shapes_\*. Examples with different benchdnn common
options can be found at driver_conv.md.
