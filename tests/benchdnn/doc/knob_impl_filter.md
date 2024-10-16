# Implementation filter specification

## Introduction
oneDNN operates with compute kernel abstractions wrapped into implementation
abstractions. Under a single primitive multiple implementations exist and
may provide a different level of optimizations depending on the arguments
passed to the library, hardware characteristics and many others. Several
implementations may support a single problem, their performance may be
different, and they are specified in a fixed order considered by the library
from the best to the worst performance outcome. Those implementation
abstractions have names and there are situations when it's desired to compare
different implementations for a single problem or validate a specific
implementation from the middle of the list which can't be dispatched by default.

To satisfy this desire, benchdnn provides knobs allowing to control which
implementation should be searched for or which implementations shouldn't be
considered when setting up the library objects.

## Usage
```
    --impl=STRING_NAME[,STRING_NAME...]
    --skip-impl=STRING_NAME[,STRING_NAME...]
```

`STRING_NAME` is a string literal without spaces or any quotes. Multiple names
are supported and can be specified by a comma-separated line.

`--impl` instructs benchdnn to search for an implementation containing one of
the names provided. If the first implementation fetched doesn't match any of the
names, benchdnn will use the library API to fetch the next one in the list until
it finds the one that matches or reaches the end of the list. If the end is
reached and no implementations were identified as suitable, the `SKIPPED` status
will be returned.

`--skip-impl` instructs benchdnn to avoid implementations with the names
provided, or just does the opposite to what `--impl` does.

## Details
For these options "matching" means a complete entry of `STRING_NAME` into a
string queried from the library. E.g., the library returned `marvelous:any`.
`--impl=super` will skip this implementation because there's no full entry
of `super` in `marvelous:any`, but `--impl=marv` will fetch it because
`marvelous` contains `marv`.

`-v6` provides additional information about filtering.

## Limitations

Options are logically opposite to each other. Because of this, they are
controlled by the same object internally. Thus, if both options are specified,
only the latter one will take the effect.

## Examples
This example demonstrates the exactness importance during the matching:
``` sh
benchdnn --matmul -v6 --impl=gemmx:jit:f32 64x32:32x64
create: --matmul --impl=gemmx:jit:f32 64x32:32x64
[IMPL_FILTER] Implementation skipped:: brg_matmul:avx512_core
[IMPL_FILTER] Implementation skipped:: gemm:jit:f32
[IMPL_FILTER] Implementation skipped:: brg_matmul:avx2
[IMPL_FILTER] Implementation skipped:: ref:any
All implementations were skipped!
run: --matmul --impl=gemmx:jit:f32 64x32:32x64
0:SKIPPED (Skip-impl option hit) __REPRO: --matmul --impl=gemmx:jit:f32 64x32:32x64
```
