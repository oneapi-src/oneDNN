# Batch Normalization Driver

## Usage
``` sh
    ./benchdnn --bnorm [benchdnn-knobs] [bnorm-knobs] [bnorm-desc] ...
```

where *bnorm-knobs* are:

 - `--dir={FWD_D [default], FWD_I, BWD_D, BWD_DW}` -- dnnl_prop_kind_t.
            Refer to [direction](knobs_dir.md) for details.
 - `--dt={f32 [default], s8, bf16, f16}` -- src and dst data types.
            Refer to [data types](knobs_dt.md) for details.
 - `--tag={nchw [default], ...}` -- physical src and dst memory layout.
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
 - `--flags=[|G|C|H|R|A]` -- batch normalization flags, default `none`; where
            multiple simultaneous flags are supported.
            `G` is dnnl_use_global_stats;
            `C` is dnnl_use_scale;
            `H` is dnnl_use_shift;
            `R` is dnnl_fuse_norm_relu;
            `A` is dnnl_fuse_norm_add_relu;
            Refer to [batch normalization primitive](https://oneapi-src.github.io/oneDNN/dev_guide_batch_normalization.html)
            for details.
 - `--attr-post-ops=STRING` -- post operation primitive attribute. No post
            operations are set by default. Refer to [attributes](knobs_attr.md)
            for details.
 - `--inplace=BOOL` -- memory mode for the primitive. If `true`, it uses input
            memory as output, otherwise, input and output are separate.
            Default is `false`.
 - `--debug-check-ws=BOOL` -- checks if workspace has correct values. Feature is
            based on internal knowledge of a library implementation. The default
            is `false`.
 - `--mb=INT` -- override minibatch size specified in the problem description.
             When set to `0`, use minibatch size as defined by the individual
             problem descriptor. The default is `0`.
 - `--match=REGEX` -- skip problems not matching the regular expression in
            `REGEX`. By default no pattern is applied (run everything).
            Note: Windows may interpret only string arguments surrounded by
            double quotation marks.

and *bnorm-desc* is a problem descriptor. The canonical form is:
```
    mbXicX_idXihXiwX_epsF_nS
```
Refer to [descriptor](knobs_desc.md) for details. `epsF` stands for Batch
Normalization epsilon value and accepts float F values. The default is `1.f/16`.

## Essence of Testing
TBA.


## Examples

Run a set of bnorms from an input file, using the default settings:
``` sh
    ./benchdnn --bnorm --batch=inputs/bnorm/shapes_resnet_50
```

Run a named problem with single precision src/dst, skipping all problems that
use the reference implementation, iterating by:
1) blocked memory layouts, where channel blocking equals 8 and 16,
2) forward training, backward by data and weights prop_kinds,
3) some flag combinations:
``` sh
    ./benchdnn --bnorm --dt=f32 --skip-impl=ref --tag=nChw8c,nChw16c \
               --dir=FWD_D,BWD_D,BWD_DW --flags=CHR,GCH,CH \
               mb96_ic192_ih71iw71_n"googlenet_v3:conv_4_4_batchnorm"
```

Run a set of 3D spatial bnorms with s8 src/dst data_type, inference prop_kind,
plain physical memory layout with dense channels, and some flags specified:
``` sh
    ./benchdnn --bnorm --dt=s8 --tag=ndhwc --dir=FWD_I \
               --flags=GCHR --batch=shapes_3d
```

More examples with different driver options can be found at
inputs/bnorm/test_\*. Examples with different problem descriptors can be found
at inputs/bnorm/shapes_\*. Examples with different benchdnn common options can
be found at driver_conv.md.
