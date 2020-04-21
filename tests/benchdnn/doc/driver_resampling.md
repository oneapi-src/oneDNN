# Resampling Driver

## Usage
``` sh
    ./benchdnn --resampling [benchdnn-knobs] [resampling-knobs] [resampling-desc] ...
```

where *resampling-knobs* are:

 - `--dir={FWD_D [default], BWD_D}` -- dnnl_prop_kind_t.
            Refer to the common glossary in README.md for details.
 - `--dt={f32 [default], ...}` -- src and dst data type.
            Refer to the common glossary in README.md for details.
 - `--tag={nchw [default], ...}` -- physical src and dst memory layout.
            Refer to the common glossary in README.md for details.
 - `--alg={nearest [default], linear}` -- resampling algorithm.
            `nearest` is dnnl_resampling_nearest;
            `linear` is dnnl_resampling_linear;
            Refer to ``doc/primitives/resampling.md`` for details.
 - `--mb=INT` -- override minibatch size specified in the problem description.
             When set to `0`, use minibatch size as defined by the individual
             problem descriptor. The default is `0`.

and *resampling-desc* is a problem descriptor. The canonical form is:
```
    mbXicX_idXihXiwX_odXohXowX_nS
```
Here `X` is an integer number and `S` is a string literal without spaces (`n`
stands for name). The special symbol `_` is ignored, so it may be used as a
delimiter for better readability. Refer to the common glossary in README.md for
the entity name and description.

There are default values for some entities in case they were not specified:
 - mb = 2;
There are also implicit rules:
 - Values for smaller dimensions may be copied from the biggest.


## Essence of Testing
nearest: Fill input data with integers and expect an integer answer.
linear: Fill input data with integers and expect a float answer.


## Examples

Run a set of resamplings from an input file with the default settings:
``` sh
    ./benchdnn --resampling --batch=inputs/resampling/resampling_2d
```

Run a named problem with single precision src/dst, iterating by:
1) both blocked memory layouts, where channel blocking equals 8 and 16,
2) both forward and backward prop_kind,
3) all algorithm combinations,
4) using default minibatch of 96 and 5:
``` sh
    ./benchdnn --resampling --dt=f32 --tag=nChw8c,nChw16c \
               --dir=FWD_D,BWD_D --alg=nearest,linear --mb=0,5 \
               mb96ic768_ih17oh34
```

More examples with different driver options can be found at
inputs/resampling/test_resampling_all. Examples with different driver descriptors can be
found at inputs/resampling/resampling_***. Examples with different benchdnn options can be
found at driver_conv.md.
