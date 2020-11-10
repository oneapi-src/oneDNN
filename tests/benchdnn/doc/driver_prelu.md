# Prelu Driver

## Usage
``` sh
    ./benchdnn --prelu [benchdnn-knobs] [prelu-knobs] [prelu-desc] ...
```

where *prelu-knobs* are:
 - `--dir={FWD_D [default], BWD_DW}` -- dnnl_prop_kind_t.
            Refer to [direction](knobs_dir.md) for details.
 - `--dt={f32 [default], bf16}` -- src data type.
            Refer to [data types](knobs_dt.md) for details.
 - `--tag={abx [default], ...}` -- physical src and dst memory layout.
            Refer to [tags](knobs_tag.md) for details.

and *prelu-desc* is a problem descriptor. The canonical form is:
```
    NxNxNxN:MxMxMxM
```
where N and M are integer numbers.

N describes input tensor dimensions and represents a 2D spatial problem with
the following logical dimensions: N, C, H, W.
Refer to [descriptor](knobs_desc.md)for details.

M describes weights tensor dimensions containing alpha parameter for PReLU
primitive and supports broadcast-semantics. Weights tensor can be used with
format_tag::any - primitive will match it to src tensor format.
PReLU primitive also supports 1D and 3D spatial problems.

## Element broadcasting
Element broadcasting supported for the second tensor: it can have fewer
dimensions than the first one. The trailing dimensions are implicitly padded
with dimensions of size 1. For example, for a 8x7x6:1x7 problem the 1x7 tensor
dimensions are first padded to 1x7x1. Then, according to the definition of the
primitive, each element of the second tensor is broadcast across the first and
the last dimensions when applying a binary operation.

## Examples

Run the set of prelu primitive problems from `prelu/test_prelu_all`
with the default settings:
``` sh
    ./benchdnn --prelu --batch=inputs/prelu/test_prelu_all
```

Run a specific prelu primitive problem:
- Direction is `BWD_DW`
- Data type is `f32` for source and destination tensors.
- Source tensor uses `abx` memory format.
- Source tensor size is `256x128x7x7` and weigths tensor is `1x128x1x1`
  which is channel-wise
``` sh
    ./benchdnn --prelu --dir=BWD_DW --tag=abx --dt=f32  256x128x7x7:1x128x1x1
```

More examples with different driver options can be found at
inputs/prelu/test_prelu_all. Examples with different benchdnn options
can be found at driver_conv.md.
