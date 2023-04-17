# Prelu Driver

## Usage
``` sh
    ./benchdnn --prelu [benchdnn-knobs] [prelu-knobs] [prelu-desc] ...
```

where *prelu-knobs* are:
 - `--dir={FWD_D [default], BWD_DW}` -- dnnl_prop_kind_t.
            Refer to [direction](knobs_dir.md) for details.
 - `--sdt={f32:f32 [default], ...}` -- src data type.
            Refer to [data types](knobs_dt.md) for details.
 - `--stag={abx:any [default], ...}` -- physical src and wei memory layout.
            Refer to [tags](knobs_tag.md) for details.
 - `--match=REGEX` -- skip problems not matching the regular expression in
            `REGEX`. By default no pattern is applied (run everything).
            Note: Windows may interpret only string arguments surrounded by
            double quotation marks.

and *prelu-desc* is a problem descriptor. The canonical form is:
```
    NxNxNxNxN:MxMxMxMxM
```
where N and M are integer numbers.

N describes source tensor dimensions. This represents tensor with the
following logical dimensions: N, C, D, H, W. Consider removing each `xN` from
the end to specify fewer dimensions. PReLU supports 1D up to 5D tensors.

M describes weights tensor dimensions containing alpha parameter for PReLU
primitive and supports broadcast-semantics. Weights tensor can be used with
format_tag::any - primitive will match it to src tensor format.

## Element broadcasting

Element broadcasting supported for a weights tensor. It allows to specify ones
instead of matching values for all dimensions at once, and for all dimnesions
but second one (known as channels). I.e. for a `8x7x6:1x7x1` problem each
element of the second tensor is broadcasted across the first and the last
dimensions when applying a prelu operation.

## Examples

Run the set of prelu primitive problems from `inputs/prelu/shapes_ci`
with the default settings:
``` sh
    ./benchdnn --prelu --batch=inputs/prelu/shapes_ci
```

Run a specific prelu primitive problem:
- Direction is `BWD_DW`
- Data types are `f32` for source and weights tensors.
- Source and weights tensors use `abx` memory format.
- Source tensor size is `256x128x7x7` and weigths tensor is `1x128x1x1`
  which is channel-wise broadcast case.
``` sh
    ./benchdnn --prelu --dir=BWD_DW --stag=abx:abx --sdt=f32:f32 \
                       256x128x7x7:1x128x1x1
```

More examples with different driver options can be found at
inputs/prelu/test_\*. Examples with different problem descriptors can be found
at inputs/prelu/shapes_\*. Examples with different benchdnn common options can
be found at driver_conv.md.
