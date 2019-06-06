# Shuffle Driver

## Usage
``` sh
    ./benchdnn --shuffle [benchdnn-knobs] [shuffle-knobs] [shuffle-desc] ...
```

where *shuffle-knobs* are:

 - `--dir={FWD_D [default], BWD_D}` -- mkldnn_prop_kind_t. Refer to the common
            glossary in README.md for details.
 - `--dt={f32 [default], s32, s8, u8, bf16, f16}` -- src and dst data type.
            Refer to the common glossary in README.md for details.
 - `--tag={nchw [default], ...}` -- physical src and dst memory layout.
            Refer to the common glossary in README.md for details.
 - `--axis=N` -- dimension on which shuffle will be performed. TBA. default `1`.
 - `--group=N` -- number of elements to shuffle in one group. TBA. default `1`.

and *shuffle-desc* is a problem descriptor. The canonical form is:
```
    NxNxNxNxN
```
where N is an integer number. This represents a 3D spatial problem with the
following logical dimensions: N, C, D, H, W. Consider removing each `xN` from
the end to specify fewer dimensions. TBA.


## Essence of Testing
TBA.


## Examples

Run the set of shuffles from an input file with the default settings:
``` sh
    ./benchdnn --shuffle --batch=inputs/shuffle/test_shuffle_axis
```

Run a specific shuffle problem with forward prop_kind and plain physical memory
layout. Group elements by 4 and over `h` dimension, iterating by all listed
data types:
``` sh
    ./benchdnn --shuffle --dir=FWD_D --dt=f32,s32,s8,u8,bf16 \
               --tag=nchw --group=4 --axis=2 1x68x56x56
```

More examples with different driver options can be found at
inputs/shuffle/test_shuffle_***. Examples with different driver descriptors can
be found at inputs/shuffle/shuffle_***. Examples with different benchdnn options
can be found at driver_conv.md.
