# Batch Normalization Driver

## Usage
``` sh
    ./benchdnn --bnorm [benchdnn-knobs] [bnorm-knobs] [bnorm-desc] ...
```

where *bnorm-knobs* are:

 - `--dir={FWD_D [default], FWD_I, BWD_D, BWD_DW}` -- mkldnn_prop_kind_t.
            Refer to the common glossary in README.md for details.
 - `--dt={f32 [default], s8}` -- src and dst data types.
            Refer to the common glossary in README.md for details.
 - `--tag={nchw [default], ...}` -- physical src and dst memory layout.
            Refer to the common glossary in README.md for details.
 - `--flags=[|G|S|R]` -- batch normalization flags, default `none`; where
            multiple simultaneous flags are supported.
            `G` stands for "use global stats"; TBA.
            `S` stands for "use scale shift"; TBA.
            `R` stands for "fuse with ReLU"; TBA.
 - `--attr="attr_str"` -- primitive attributes, default `""` (no attributes).
            Refer to knobs_attr.md for details.
 - `--inplace=BOOL` -- memory mode for the primitive. If `true`, it uses input
            memory as output, otherwise, input and output are separate.
            Default is `true`.
 - `--mb=N` -- override minibatch size specified in the problem description.
             When set to `0`, use minibatch size as defined by the individual
             problem descriptor. The default is `0`.
 - `--match=regex` -- run only problems that match the regex. The default is
            `".*"` (everything). Note: Windows may interpret only string
            arguments surrounded by double quotation marks.

and *bnorm-desc* is a problem descriptor. The canonical form is:
```
    mbXicXidXihXiwXepsYnS
```
Here X is an integer number, Y is a real number, and S is a string (n stands for
name). The special symbol `_` is ignored, so it may be used as a delimiter.
Refer to the common glossary in README.md for entity name and description.

There are default values for some entities in case they were not specified:
 - mb = 2;
 - eps = 1./16;
There are also implicit rules:
 - Values for smaller dimensions may be copied from the biggest.


## Essence of Testing
TBA.


## Examples

Run a set of bnorms from input file with default settings:
``` sh
    ./benchdnn --bnorm --batch=inputs/bnorm/bnorm_resnet_50
```

Run a named problem with single precision src/dst, skipping all problems that
use the reference implementation, iterating by:
1) both blocked memory layouts, where channel blocking equals to 8 and 16,
2) both forward w/o bias and  backward by data and weights prop_kind,
3) all flag combinations:
``` sh
    ./benchdnn --bnorm --dt=f32 --skip-impl=ref --tag=nChw8c,nChw16c \
               --dir=FWD_D,BWD_DW --flags=SR,GS,S \
               mb96_ic192_ih71iw71_n"googlenet_v3:conv_4_4_batchnorm"
```

Run a set of 3D spatial bnorms with s8 src/dst data_type, inference prop_kind,
plain physical memory layout with dense channels and all three flags chosen:
``` sh
    ./benchdnn --bnorm --dt=s8 --tag=ndhwc --dir=FWD_I \
               --flags=GSR --batch=bnorm_3d
```

More examples with different driver options can be found at
inputs/bnorm/test_bnorm_***. Examples with different driver descriptors can be
found at inputs/bnorm/bnorm_***. Examples with different benchdnn options can be
found at driver_conv.md.
