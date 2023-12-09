# Reorder Driver

## Usage
``` sh
    ./benchdnn --reorder [benchdnn-knobs] [reorder-knobs] [reorder-desc] ...
```

where *reorder-knobs* are:

 - `--sdt={f32 [default], s32, s8, u8, bf16, f16, f64}` -- src data type.
            Refer to [data types](knobs_dt.md) for details.
 - `--ddt={f32 [default], s32, s8, u8, bf16, f16, f64}` -- dst data type.
            Refer to [data types](knobs_dt.md) for details.
 - `--stag={nchw [default], ...}` -- physical src memory layout.
            Refer to [tags](knobs_tag.md) for details.
 - `--dtag={nchw [default], ...}` -- physical dst memory layout.
            Refer to [tags](knobs_tag.md) for details.
 - `--strides=S_0xS_1x..xS_n:D_0xD_1x..xD_n` -- direct
            stride specification for `src` and `dst` tensors that can be
            specified as an alternative to memory formats. The syntax matches
            with dimensions descriptor where `x` is the delimiter for
            dimensions within a tensor and `:` is the delimiter for tensors in
            the order `src` and `dst` respectively. The stride for either of the
            tensors can be skipped and moreover if a separate tag
            is not provided for the skipped tensor, trivial strides based on the
            default format of the skipped tensor will be used. As long as
            `--strides` and `--tag` options refer to different tensors, they
            can be specified together.
 - `--attr-scales=STRING` -- per argument scales primitive attribute. No
            scales are set by default. See `--def-scales` for additional
            reorder specific mechanics and refer to [attributes](knobs_attr.md)
            for details.
 - `--attr-zero-points=STRING` -- zero points primitive attribute. No zero
            points are set by default. Refer to [attributes](knobs_attr.md)
            for details.
 - `--attr-post-ops=STRING` -- post operation primitive attribute. No post
            operations are set by default. Refer to [attributes](knobs_attr.md)
            for details.
- `--def-scales=FLOAT[,FLOAT...]` -- set of scales used to improve testing
            coverage. Enabled only when combined with `attr-scales` which is
            given a policy of `common` and a `SCALE` of 0. e.g.:
            `--attr-scales=src:common:0*`. The default set of scales is
            `0.125,0.25,0.5,1,2,4,8`.
            Example: `--def-scales=-3,3` replaces the default set from seven entries
            to two, but to enable it the user is required to pass
            `--attr-scales=ARG:common:0*` in addition.
 - `--oflag=FLAG:MASK[+...]` -- memory descriptor extra field specifier. By
            default `FLAG` is empty and `MASK` is `0`. Possible `FLAG` values
            are:
            `s8s8_comp` for `compensation_conv_s8s8` flag;
            `zp_comp` for `compensation_conv_asymmetric_src` flag;
            `MASK` value is a non-negative integer number.
 - `--cross-engine={none [default], cpu2gpu, gpu2cpu}` -- defines what kind of
            cross-engine reorder will be used. If `--engine` is set to `cpu`,
            `none` is the only supported value.
 - `--runtime_dim_mask=INT` -- a bit-mask that indicates whether a dimension is
            `DNNL_RUNTIME_DIM_VAL` (indicated as 1-bit in the corresponding
            dimension position). The default is `0`, meaning all tensor
            dimensions are fully defined at primitive creation.
 - `--match=REGEX` -- skip problems not matching the regular expression in
            `REGEX`. By default no pattern is applied (run everything).
            Note: Windows may interpret only string arguments surrounded by
            double quotation marks.

and *reorder-desc* is a problem descriptor. The canonical form is:
```
    NxNxNxNxN
```
where N is an integer number. This represents a 3D spatial problem with the
following logical dimensions: N, C, D, H, W. Consider removing each `xN` from
the end to specify fewer dimensions.


## Essence of Testing
TBA.


## Examples

Run the reorder set from an input file with the default settings:
``` sh
    ./benchdnn --reorder --batch=inputs/reorder/test_reorder_all
```

Run two specific reorders with s8 src and dst data type, and specific input and
output physical memory layouts. First problem without a flag; second problem
with the `s8s8_comp` flag and mask of `1`:
``` sh
    ./benchdnn --reorder --sdt=s8 --ddt=s8 --stag=hwio --dtag=OIhw4i16o4i \
               32x32x3x3 \
               --oflag=s8s8_comp:1 16x32x7x5
```

More examples with different driver options can be found at
inputs/reorder/test_\*. Examples with different problem descriptors can be
found at inputs/reorder/harness_\* and inputs/reorder/test_\*. Examples with
different benchdnn common options can be found at driver_conv.md.
