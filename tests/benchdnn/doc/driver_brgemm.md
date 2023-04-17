# BRGeMM Driver

## Usage
``` sh
    ./benchdnn --brgemm [benchdnn-knobs] [brgemm-knobs] [brgemm-desc] ...
```

where *brgemm-knobs* are:

 - `--dt={f32:f32:f32 [default], ...}` -- source, weights and destination data
            types. Interface supports broadcasting, when a single input is
            provided, e.g., `--dt=f32`, and the value will be applied for all
            tensors. Refer to [data types](knobs_dt.md) for details.
 - `--bia_dt={undef [default], f32, s32, s8, u8}` -- bias data type.
            To run BRGEMM kernel without bias, use `undef` data type.
            Refer to [data types](knobs_dt.md) for details.
 - `--ld=LDA:LDB:LDD` -- direct leading dimension specification for src,
            weights, and dst tensors. The value for either of the tensors can be
            skipped meaning a default value will be applied.
            So far `LDA` and `LDD` do nothing and `LDB` is used only to make a
            correct kernel call when `N` < 16 since blocked format is implied
            for weights.
 - `--alpha=FLOAT` -- float value corresponding to scaling of accumulator
            result: `C = alpha * A * B`.
 - `--beta=FLOAT` -- float value corresponding to adding part of accumulator
            result: `C = A * B + beta * C`.
 - `--bs=INT` -- specifies batch size setting for a kernel. The default is `1`.
 - `--brgemm-attr=STRING` -- specifies brgemm attributes through the string.
            The format is: KEY:VALUE[+KEY:VALUE[...]] following post-ops
            notation. STRING may have `,` to iterate over multiple attribute
            settings. Refer to internal brgemm headers for more details.
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
 - `--match=REGEX` -- skip problems not matching the regular expression in
            `REGEX`. By default no pattern is applied (run everything).
            Note: Windows may interpret only string arguments surrounded by
            double quotation marks.

and *brgemm-desc* is a problem descriptor. The canonical form is:
```
    MxK:KxN[_nS]
```
Here `x` is the delimiter for dimensions within a tensor and `:` is the
delimiter for tensors in the order `src` and `weights`, and `M`, `N`, and `K`
are inner dimensions for matrix multiplication.

## Examples

Run the default validation set of BRGEMM using `inputs/brgemm/shapes_2d`
file:
``` sh
    ./benchdnn --brgemm --batch=inputs/brgemm/shapes_2d
```

Run reduced precision (int8) with batch size equal to `10`:
``` sh
    ./benchdnn --brgemm --dt=u8:s8:u8 --bs=10 10x30:30x20
```

Run single precision kernel with bias:
``` sh
    ./benchdnn --brgemm --bia_dt=f32 1x30:30x1
```

Run single precision with `ldb` equal to `16`:
``` sh
    ./benchdnn --brgemm --ld=:16: 16x16:16x2
```

More examples with different driver options can be found at
inputs/brgemm/test_\*.
