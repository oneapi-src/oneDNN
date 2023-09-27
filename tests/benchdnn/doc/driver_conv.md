# Convolution/Deconvolution Driver

## Usage
``` sh
    ./benchdnn --conv [benchdnn-knobs] [conv-knobs] [conv-desc] ...
    ./benchdnn --deconv [benchdnn-knobs] [conv-knobs] [conv-desc] ...
```

where *conv-knobs* are:

 - `--dir={FWD_B [default], FWD_D, FWD_I, BWD_D, BWD_W, BWD_WB}`
            -- dnnl_prop_kind_t. Refer to [direction](knobs_dir.md) for details.
 - `--dt={f32:f32:f32 [default], ...}` -- source, weights and destination data
            types. Interface supports broadcasting, when a single input is
            provided, e.g., `--dt=f32`, and the value will be applied for all
            tensors. Refer to [data types](knobs_dt.md) for details.
 - `--cfg={f32 [default], ...}` -- Deprecated setting.
            Refer to ``Configurations`` below.
 - `--stag={any [default], ...}` -- physical src memory layout.
            Refer to [tags](knobs_tag.md) for details.
 - `--wtag={any [default], ...}` -- physical wei memory layout.
            Refer to [tags](knobs_tag.md) for details.
 - `--dtag={any [default], ...}` -- physical dst memory layout.
            Refer to [tags](knobs_tag.md) for details.
 - `--alg={DIRECT [default], WINO, AUTO}` -- convolution algorithm. `WINO` is
            Winograd-based convolution. `AUTO` will pick one of `DIRECT` or
            `WINO` automatically, library-based decision.
 - `--attr-scales=STRING` -- scale primitive attribute. No scale is
            set by default. Refer to [attributes](knobs_attr.md) for details.
 - `--attr-zero-points=STRING` -- zero points primitive attribute. No zero
            points are set by default. Refer to [attributes](knobs_attr.md)
            for details.
 - `--attr-post-ops=STRING` -- post operation primitive attribute. No post
            operations are set by default. Refer to [attributes](knobs_attr.md)
            for details.
 - `--attr-fpmath=STRING` -- fpmath mode primitive attribute. `strict` math mode
            is set by default. Refer to [attributes](knobs_attr.md) for details.
 - `--mb=INT` -- override minibatch size specified in the problem description.
             When set to `0`, use minibatch size as defined by the individual
             problem descriptor. The default is `0`.
 - `--match=REGEX` -- skip problems not matching the regular expression in
            `REGEX`. By default no pattern is applied (run everything).
            Note: Windows may interpret only string arguments surrounded by
            double quotation marks.

and *conv-desc* is a problem descriptor. The canonical form is:
```
    gXmbX_icXidXihXiwX_ocXodXohXowX_kdXkhXkwX_sdXshXswX_pdXphXpwX_ddXdhXdwX_nS
```
Refer to [descriptor](knobs_desc.md) for details. Input shape and kernel size
are mandatory inputs. Output shape and padding may be deduced based on the
values provided.

## Precision Configurations

`--cfg` option specifies what [data types](knobs_dt.md) will be used for a
problem. It also defines the data filling strategy. It is implicit for the
integer type saturation. This option also defines the threshold for computation
errors.

The table below shows supported name configurations for this driver:

For data type support, refer to [data types](https://oneapi-src.github.io/oneDNN/dev_guide_data_types.html)
and [convolution primitive](https://oneapi-src.github.io/oneDNN/dev_guide_convolution.html#data-types)
documentation.

| src  | wei  | dst  | acc  | cfg             |
|:---  |:---  |:---  |:---  |:---             |
| f32  | f32  | f32  | f32  | f32             |
| f64  | f64  | f64  | f64  | f64             |
| f32  | f32  | s8   | f32  | f32f32s8        |
| u8   | s8   | f32  | s32  | u8s8f32         |
| u8   | s8   | s32  | s32  | u8s8s32         |
| u8   | s8   | s8   | s32  | u8s8s8          |
| u8   | s8   | u8   | s32  | u8s8u8          |
| s8   | s8   | f32  | s32  | s8s8f32         |
| s8   | s8   | s32  | s32  | s8s8s32         |
| s8   | s8   | s8   | s32  | s8s8s8          |
| s8   | s8   | u8   | s32  | s8s8u8          |
| f32  | f32  | f32  | f32  | f32_wino        |
| f16  | f16  | f16  | f32  | f16             |
| f16  | f16  | s8   | f32  | f16f16s8        |
| bf16 | bf16 | bf16 | f32  | bf16bf16bf16    |
| bf16 | bf16 | f32  | f32  | bf16bf16f32     |
| bf16 | f32  | bf16 | f32  | bf16f32bf16     |
| f32  | bf16 | bf16 | f32  | f32bf16bf16     |

## Essence of Testing

Since convolution problems require a significant number of accumulators for a
single output point, hitting an overflow or loss of precision issues is easy.
To deal with that, the convolution driver applies two techniques to mitigate the
above-mentioned issues: 1) uses integer values for activations and weights so
that integers can be compared to integers without dealing with floating-point
precision loss; 2) utilizes data density to control the output range of values
so that final values remain in the range of float data type representation and
no saturation happens for a lower precision integer output.

## Examples

Run the set of f32 forward convolutions from inputs/conv/set_conv_all file w/ bias and
default minibatch:
``` sh
    ./benchdnn --conv --dt=f32 --dir=FWD_B --batch=inputs/conv/set_conv_all
```

Run the same but with post_ops ReLU:
``` sh
    ./benchdnn --conv --dt=f32 --dir=FWD_B \
               --attr-post-ops=relu --batch=inputs/conv/set_conv_all
```

Run the same as previous but measures performance, not correctness check:
``` sh
    ./benchdnn --conv --mode=p --dt=f32 --dir=FWD_B \
               --attr-post-ops=relu --batch=inputs/conv/set_conv_all
```

Run a set of f32 backward convolutions wrt weights with kh=3 and
verbose level set to 2:
``` sh
    ./benchdnn --conv -v2 --dt=f32 --dir=BWD_W \
               --match='.*kh3[^0-9].*' --batch=inputs/conv/set_conv_all
```

Run a set of u8s8u8 backward convolutions wrt data but skip all
the convolutions that will use reference or gemm-based implementation:
``` sh
    ./benchdnn --conv --dt=u8:s8:u8 --dir=BWD_D \
               --skip-impl=ref,x64:gemm --batch=inputs/conv/set_conv_all
```

Run explicitly specified first forward convolution (including bias) from Alexnet
with the minibatch set to 4 and the verbose level set to 1 for two given
configurations (`u8:s8:u8` and `f32`):
``` sh
    ./benchdnn --conv -v1 --mb=4 --dir=FWD_B --dt=f32,u8:s8:u8 \
               ic3ih227iw227_oc96oh55ow55_kh11kw11_sh4sw4ph0pw0_n"alexnet:conv1"
```

Run the batch file for different algorithms (assuming the file specifies only
convolutions and does not include driver options that would override any passed
on the command line). Also ignore dnnl_unimplemented errors in case of
Winograd:
``` sh
    ./benchdnn --conv --alg=DIRECT,WINO,AUTO --batch=inputs/conv/set_conv_all
```

Run a set of u8s8u8 forward convolutions without bias, skipping
reference implementations with one common output scale set to 0.5:
``` sh
    ./benchdnn --conv --dt=u8:s8:u8 --dir=FWD_D --skip-impl=ref \
               --attr-scales=dst:common:0.5 --batch=inputs/conv/set_conv_all
```

More examples with different driver options can be found at inputs/conv/test_\*
or inputs/conv/harness_\*. Examples with different problem descriptors can be
found at inputs/conv/shapes_\*.

