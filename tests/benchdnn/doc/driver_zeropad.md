# Zero Pad Driver

## Usage
``` sh
    ./benchdnn --zeropad [benchdnn-knobs] [zeropad-knobs] [zeropad-desc] ...
```

where *zeropad-knobs* are:

 - `--dt={f32 [default], bf16, f16, s32, s8}` -- data type.
            Refer to [data types](knobs_dt.md) for details.
 - `--tag={nchw [default], ...}` -- physical memory layout.
            Refer to [tags](knobs_tag.md) for details.

and *zeropad-desc* is a problem descriptor. The canonical form is:
```
    NxNxNxNxN
```
where N is an integer number. This represents a 3D spatial problem with the
following logical dimensions: N, C, D, H, W. Consider removing each `xN` from
the end to specify fewer dimensions.

## Examples

Run the zeropad set from an input file with the default settings:
``` sh
    ./benchdnn --zeropad --batch=inputs/zeropad/test_zeropad_ci
```

Run a specific zeropad problem with the f32 data type and iterating
over memory layouts:
``` sh
    ./benchdnn --zeropad --dt=f32 --tag=ABcd4a4b,nChw16c 64x3x60x60
```
