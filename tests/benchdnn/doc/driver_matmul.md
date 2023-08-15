# MatMul Driver

## Usage
``` sh
    ./benchdnn --matmul [benchdnn-knobs] [matmul-knobs] [matmul-desc] ...
```

where *matmul-knobs* are:

 - `--dt={f32:f32:f32 [default], ...}` -- source, weights and destination data
            types. Interface supports broadcasting, when a single input is
            provided, e.g., `--dt=f32`, and the value will be applied for all
            tensors. Refer to [data types](knobs_dt.md) for details.
 - `--stag={ab [default], any, ...}` -- memory format of the source memory.
            Refer to [tags](knobs_tag.md) for details.
 - `--wtag={ab [default], any, ...}` -- memory format of the weights memory.
            Refer to [tags](knobs_tag.md) for details.
 - `--dtag={ab [default], any, ...}` -- memory format of the destination memory.
            Refer to [tags](knobs_tag.md) for details.
 - `--strides=S0xS1x..xS_mxS_k:W_0xW_1x..xW_kxW_n:D_0xD_1x..xD_mxD_n` -- direct
            stride specification for `src`, `weights`, and `dst` tensors that
            can be specified as an alternative to memory formats. The syntax
            matches with problem descriptor where `x` is the delimiter for
            dimensions within a tensor and `:` is the delimiter for tensors in
            the order `src`, `weights`, and `dst` respectively. The stride for
            either of the tensors can be skipped and moreover if a separate tag
            is not provided for the skipped tensor, trivial strides based on the
            default format of the skipped tensor will be used. As long as
            `--strides` and `--*tag` options refer to different tensors, they
            can be specified together.
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
 - `--bia_dt={undef [default], f32, s32, s8, u8}` -- bias data type.
            To run MatMul without bias, use `undef` data type (default).
            Refer to [data types](knobs_dt.md) for details.
 - `--bia_mask=INT` -- a bit-mask that indicates which bias dimensions are
            broadcasted. 0-bit means broadcast, 1-bit means full dimension.
 - `--runtime_dims_masks=[INT][:INT]` -- a bit-mask with values for `src` and
            `weights` that indicates whether a dimension is
            `DNNL_RUNTIME_DIM_VAL` (indicated as 1-bit in the corresponding
            dimension position). The default is `0` for all dimensions, meaning
            all tensor dimensions are fully defined at primitive creation.
- `--encoding=STRING` - sparse encodings and sparsity. No encodings are set by
            default. Refer to [encodings](knobs_encoding.md) for details. This
            is an experimental feature that must be enabled via a build time
            CMake option `DNNL_EXPERIMENTAL_SPARSE`.
 - `--match=REGEX` -- skip problems not matching the regular expression in
            `REGEX`. By default no pattern is applied (run everything).
            Note: Windows may interpret only string arguments surrounded by
            double quotation marks.

and *matmul-desc* is a problem descriptor. The canonical form is:
```
    d0xd1xd2x..xMxK:d0xd1xd2x..xKxN[:d0xd1xd2x..xMxN][nS]
```
Here `x` is the delimiter for dimensions within a tensor and `:` is the
delimiter for tensors in the order `src`, `weights`, and `dst`. The `dst` is
optional, and each of its individual dimensions are computed as
`max(src_dimension, weights_dimension)` by the driver if not provided by user.
`d0`, `d1`, `d2` and so on are dimension values of the corresponding tensor,
where `m`, `n`, and `k` are inner dimensions for matrix multiplication.

## Examples

Run the default validation set of MatMul using `inputs/matmul/shapes_2d`
file:
``` sh
    ./benchdnn --matmul --batch=inputs/matmul/shapes_2d
```

Run single precision matrix multiplication with all sizes provided at run-time
using plain `ab` format for all inputs:
``` sh
    ./benchdnn --matmul --stag=ab --wtag=ab --dtag=ab \
                        --runtime_dims_masks=3:3 10x30:30x20
```

Run reduced precision (int8) matrix multiplication with asymmetric quantization
for the source and destination memory (both use `uint8_t` data type) and
symmetric quantization for weights memory (with `int8_t` data type and allowing
the library to choose the proper memory format), with zero points provided at
runtime, but sizes specified at creation time:
``` sh
    ./benchdnn --matmul \
               --dt=u8:s8:u8 \
               --wtag=any \
               --attr-zero-points=src:common:1*+dst:common:-2* \
               10x30:30x20
```

Run single precision batched matrix multiplication with bias, of which only the
full dimension is along the `n`-axis:
``` sh
    ./benchdnn --matmul --bia_dt=f32 --bia_mask=4 2x10x30:2x30x20
```

Run single precision batched matrix multiplication with strides so that `dst` tensor
has non-dense memory layout:
``` sh
    ./benchdnn --matmul --strides=8x4x1:24x6x1:21x7x1 3x2x4:3x4x6
```

or
``` sh
    # --dtag cannot be specified here
    ./benchdnn --matmul --stag=bax --wtag=abx --strides=::8x4x1 2x2x3:2x3x2
```

More examples with different driver options can be found at
inputs/matmul/test_\*.
