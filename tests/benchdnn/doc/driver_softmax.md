# Softmax Driver

## Usage
``` sh
    ./benchdnn --softmax [benchdnn-knobs] [softmax-knobs] [softmax-desc] ...
```

where *softmax-knobs* are:

 - `--dir={FWD_D [default], BWD_D}` -- dnnl_prop_kind_t. Refer to the common
            glossary in README.md for details.
 - `--dt={f32 [default], f16}` -- src and dst data type.
            Refer to the common glossary in README.md for details.
 - `--tag={nchw [default], ...}` -- physical src and dst memory layout.
            Refer to the common glossary in README.md for details.
 - `--axis=INT` -- dimension on which operation will be performed.
            Default is `1`, corresponds to channels in logical memory layout.
 - `--mb=INT` -- override minibatch size specified in the problem description.
             When set to `0`, use minibatch size as defined by the individual
             problem descriptor. The default is `0`.
 - `--inplace=BOOL` -- memory mode for the primitive. If `true`, it uses input
            memory as output, otherwise, input and output are separate.
            The default is `true`.

and *softmax-desc* is a problem descriptor. The canonical form is:
```
    NxNxNxNxN
```
where N is an integer number. This represents a 3D spatial problem with the
following logical dimensions: N, C, D, H, W. Consider removing each `xN` from
the end to specify fewer dimensions.


## Essence of Testing
Forward: Fill input data in two ways: only negative integers, or mixed signed
            integers (to check the max value finding).
Backward: Fill input data with negative integers, and expect positive output.
            This avoids potential cancellation errors.


## Examples

Run the softmax set from an input file with the default settings:
``` sh
    ./benchdnn --softmax --batch=inputs/softmax/test_softmax_2d
```

Run a specific softmax problem with forward prop_kind, plain physical memory
layout, f32 data type, out-place memory mode, and axis size of 1000:
``` sh
    ./benchdnn --softmax --dir=FWD_D --dt=f32 --tag=nchw \
               --inplace=false --axis=1 256x1000
```

More examples with different driver options can be found at
inputs/softmax/test_softmax_all. Examples with different benchdnn options can be
found at driver_conv.md.
