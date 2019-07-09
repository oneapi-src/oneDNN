# Convolution/Deconvolution Driver

## Usage
``` sh
    ./benchdnn --conv [benchdnn-knobs] [conv-knobs] [conv-desc] ...
    ./benchdnn --deconv [benchdnn-knobs] [conv-knobs] [conv-desc] ...
```

where *conv-knobs* are:

 - `--dir={FWD_B [default], FWD_D, FWD_I, BWD_D, BWD_W, BWD_WB}`
            -- mkldnn_prop_kind_t. Refer to the common glossary in README.md for
            details.
 - `--cfg={f32 [default], u8s8u8s32, ...}` -- Refer to ``Configurations`` below.
 - `--stag={any [default], ...}` -- physical src memory layout.
            Refer to the common glossary in README.md for details.
 - `--wtag={any [default], ...}` -- physical wei memory layout.
            Refer to the common glossary in README.md for details.
 - `--dtag={any [default], ...}` -- physical dst memory layout.
            Refer to the common glossary in README.md for details.
 - `--alg={DIRECT [default], WINO, AUTO}` -- convolution algorithm. `WINO` is
            Winograd-based convolution. `AUTO` will pick one of `DIRECT` or
            `WINO` automatically, library-based decision.
 - `--attr="attr_str"` -- primitive attributes. The default is `""` (no
            attributes). Refer to knobs_attr.md for details.
 - `--mb=N` -- override minibatch size specified in the problem description.
             When set to `0`, use minibatch size as defined by the individual
             problem descriptor. The default is `0`.
 - `--match=regex` -- run only problems that match the regex. The default is
            `".*"` (everything). Note: Windows may interpret only string
            arguments surrounded by double quotation marks.

and *conv-desc* is a problem descriptor. The canonical form is:
```
    gXmbX_icXidXihXiwX_ocXodXohXowX_kdXkhXkwX_sdXshXswX_pdXphXpwX_ddXdhXdwX_nS
```
Here X is an integer number and S is a string (n stands for name).
The special symbol `_` is ignored, so it may be used as a delimiter.
Refer to the common glossary in README.md for entity name and description.

There are default values for some entities in case they were not specified:
 - g = 1;
 - mb = 2;
 - strides = 1;
 - dilations = 0;
There are also implicit rules:
 - Output shape may be deduced from the input and kernel size.
 - Values for smaller dimensions can be copied from the biggest.

## Precision Configurations

`--cfg` option specifies what data type will be used for a problem. It also
defines the data filling strategy. It is implicit for the integer type
saturation. This option also defines the threshold for computation errors.

The table below shows supported name configurations for this driver:

| src  | wei  | dst  | acc  | cfg             | notes
|:---  |:---  |:---  |:---  |:---             |:---
| f32  | f32  | f32  | f32  | f32             | inference optimized for sse4.1+, training for avx2+
| u8   | s8   | f32  | s32  | u8s8f32s32      | optimized for processors with support of avx512vl, FWD_x only.
| u8   | s8   | s32  | s32  | u8s8s32s32      | same as above
| u8   | s8   | s8   | s32  | u8s8s8s32       | same as above
| u8   | s8   | u8   | s32  | u8s8u8s32       | same as above
| s8   | s8   | f32  | s32  | s8s8f32s32      | same as above
| s8   | s8   | s32  | s32  | s8s8s32s32      | same as above
| s8   | s8   | s8   | s32  | s8s8s8s32       | same as above
| s8   | s8   | u8   | s32  | s8s8u8s32       | same as above
| f32  | f32  | f32  | f32  | f32_wino        | Winograd-based convolution.
| u8   | s8   | f32  | s32  | u8s8f32s32_wino | sabe as above
| u8   | s8   | s32  | s32  | u8s8s32s32_wino | same as above
| u8   | s8   | s8   | s32  | u8s8s8s32_wino  | same as above
| u8   | s8   | u8   | s32  | u8s8u8s32_wino  | same as above
| f16  | f16  | f16  | f16  | f16             | Only for GPU
| bf16 | bf16 | bf16 | f32  | bf16bf16bf16    | optimized for processors with support of avx512vl + VNNI
| bf16 | bf16 | f32  | f32  | bf16bf16f32     | same as above
| bf16 | f32  | bf16 | f32  | bf16f32bf16     | same as above
| f32  | bf16 | bf16 | f32  | f32bf16bf16     | same as above

## Essence of Testing

Intel MKL-DNN supports different data types, such as single-precision floating
point (`mkldnn_f32`) and signed/unsigned integer of different length
(`mkldnn_{s,u}{8,16,32}`). We need to cover all those cases with tests. It is
essential to test real convolution sizes, because Intel MKL-DNN provides
different optimizations depending on convolution parameters. There is no
single unified approach inside, so it would not be enough to test only a few
convolutions (also known as unit tests).

But even for a given convolution, the correctness convolution test is not as
simple as it might seem at first sight. One of the biggest problems we
encountered was numerical instability. For every output point, a lot of
operations may occur. For instance, on backward propagation with respect to
filter, each filter point requires `mb * oh * ow` operations (see the *Notation*
section below). That large amount of compute operations may lead to either
integer overflow or accuracy loss if initial data was chosen inadequately.

These two main issues complicate testing. **benchdnn** tries to address these
by using integers for initialization with uniform distribution in a range
`[cfg->f_min .. cfg->f_max]`, with the step `cfg->f_step` (see
`struct dt_conf_t` in conv/conv.hpp). `f_min` and `f_max` are chosen so that
most of the results would belong in the `[cfg->min .. cfg->max]` range. Also,
for floating point all integers in both ranges have exact representation (that
is, the absolute numbers are less than `2^size_of_mantissa`). Uniform
distribution leads to results that are uniformly distributed and quite small.
`f_min/f_max` keep the result in a reasonable range. Yet another trick: not all
the points are initialized with non-zero values: see `fill_{src,wei,bia,dst}` in
conv/conv.cpp.

## Examples

Run the set of f32 forward convolutions from inputs/conv_all file w/ bias and
default minibatch:
``` sh
    ./benchdnn --conv --cfg=f32 --dir=FWD_B --batch=inputs/conv_all
```

Run the same but with post_ops ReLU:
``` sh
    ./benchdnn --conv --cfg=f32 --dir=FWD_B \
               --attr="post_ops='relu'" --batch=inputs/conv_all
```

Run the same as previous but measures performance, not correctness check:
``` sh
    ./benchdnn --conv --mode=p --cfg=f32 --dir=FWD_B \
               --attr="post_ops='relu'" --batch=inputs/conv_all
```

Run a set of f32 backward convolutions wrt weights with kh=3 and
verbose level set to 2:
``` sh
    ./benchdnn --conv -v2 --cfg=f32 --dir=BWD_W \
               --match='.*kh3[^0-9].*' --batch=inputs/conv_all
```

Run a set of u8s8u8s32 backward convolutions wrt data but skip all
the convolutions that will use reference or gemm-based implementation:
``` sh
    ./benchdnn --conv --cfg=u8s8u8s32 --dir=BWD_B \
               --skip-impl='ref:gemm' --batch=inputs/conv_all
```

Run explicitly specified 1st forward convolution (including bias) from Alexnet
with the minibatch set to 4, verbose level set to 1 for two given
configurations (`u8s8u8s32` and `f32`):
``` sh
    ./benchdnn --conv -v1 --mb=4 --dir=FWD_B --cfg=f32,u8s8u8s32
               ic3ih227iw227_oc96oh55ow55_kh11kw11_sh4sw4ph0pw0_n"alexnet:conv1"
```

Run batch file for different algorithms (assuming the file specifies only
convolutions and does not include driver options that would override any
passed on the command line). Also ignore mkldnn_unimplemented errors in case of
Winograd. Before running AUTO algorithm, resel allow-unimpl value back to false:
``` sh
    ./benchdnn --conv \
               --alg=DIRECT --batch=convs.in \
               --allow-unimpl=true \
               --alg=WINO   --batch=convs.in \
               --reset \
               --alg=AUTO   --batch=convs.in
```

Run a set of u8s8u8s32 forward convolutions without bias, skipping
reference implementations and not triggering unimplemented as an error, with
one common output scale set to 0.5:
``` sh
    ./benchdnn --conv --cfg=u8s8u8s32 --dir=FWD_D \
               --skip-impl="ref" --allow-unimpl=true \
               --attr="oscale=common:.5" --batch=inputs/conv_all
```

More examples with different driver options can be found at
inputs/test_conv_***. Examples with different driver descriptors can be found at
inputs/conv_***.
