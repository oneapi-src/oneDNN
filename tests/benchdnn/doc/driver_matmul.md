# MatMul Driver

## Usage
``` sh
    ./benchdnn --matmul [benchdnn-knobs] [matmul-knobs] [matmul-desc] ...
```

where *matmul-knobs* are:

 - `--cfg={f32 [default], ...}` -- refer to ``Configurations`` in
            driver_conv.md.
 - `--stag={ab [default], any, ...}` -- memory format of the source memory.
            Refer to the common glossary in README.md for details.
 - `--wtag={ab [default], any, ...}` -- memory format of the weights memory.
            Refer to the common glossary in README.md for details.
 - `--dtag={ab [default], any, ...}` -- memory format of the destination memory.
            Refer to the common glossary in README.md for details.
 - `--runtime_mb=BOOL` -- specify whether `mb` dimension is a run-time
            parameter.
 - `--runtime_m=BOOL` -- specify whether `m` dimension is a run-time parameter.
 - `--runtime_n=BOOL` -- specify whether `n` dimension is a run-time parameter.
 - `--runtime_k=BOOL` -- specify whether `k` dimension is a run-time parameter.
 - `--attr="attr_str"` -- primitive attributes, default `""` (no attributes).
            Refer to [attributes](knobs_attr.md) for details.
 - `--bia_dt={undef [default], f32, s32, s8, u8}` -- bias data type.
            To run MatMul without bias, use `undef` data type (default).
            Refer to the common glossary in README.md for details.
 - `--bia_mask=INT` -- a bit-mask that indicates which bias dimensions are
            broadcasted. 0-bit means broadcast, 1-bit means full dimension.

and *matmul-desc* is a problem descriptor. The canonical form is:
```
    [mbX]mXnXkX_nS
```
Here `X` is an integer number and `S` is a string literal without spaces (`n`
stands for name). The special symbol `_` is ignored, so it may be used as a
delimiter for better readability. Refer to the common glossary in README.md for
the entity name and description.

The `mb` can be omitted, in which case the problem is treated as regular
2D matrix multiplication. With `mb` set to a non-zero value, batched matrix
multiplication is used.

## Examples

Run the default validation set of MatMul using `inputs/matmul/test_matmul_all`
file:
``` sh
    ./benchdnn --matmul --batch=inputs/matmul/test_matmul_all
```

Run single precision matrix multiplication with all sizes provided at run-time:
``` sh
    ./benchdnn --matmul \
               --runtime_m=true --runtime_n=true --runtime_k=true \
               m10n20k30
```

Run reduced precision (int8) matrix multiplication with asymmetric quantization
for the source and destination memory (both use `uint8_t` data type) and
symmetric quantization for weights memory (with `int8_t` data type and allowing
the library to choose the proper memory format), with zero points provided at
runtime, but sizes specified at creation time:
``` sh
    ./benchdnn --matmul \
               --cfg=u8s8u8 \
               --wtag=any \
               --attr="zero_points=src:1*_dst:-2*;" \
               m10n20k30
```

Run single precision batched matrix multiplication with bias, of which only the
full dimension is along the `n`-axis:
``` sh
    ./benchdnn --matmul \
               --bia_dt=f32 --bia_mask=4 \
               mb2m10n20k30
```

More examples with different driver options can be found at
inputs/matmul/test_matmul_all.
