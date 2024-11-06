# Implementation filter specification

## Introduction
oneDNN operates with compute kernel abstractions wrapped into implementation
abstractions. Under a single primitive, multiple implementations exist and
oneDNN may provide a different level of optimization depending on the arguments
passed to the library, hardware characteristics, and many others. Several
implementations may support a single problem, however, their performance may be
different. The implementations are specified in a fixed order as defined by the
library from the best to the worst performance outcome. Those implementation
abstractions have names one can compare and trigger different implementations
for a single problem or validate a specific implementation from the middle of
the list which can't be dispatched by default.

Benchdnn provides knobs allowing one to control which implementation should be
searched for and which implementations shouldn't be considered when setting up
the library objects.

## Usage
```
    --impl=STRING_NAME[,STRING_NAME...]
    --skip-impl=STRING_NAME[,STRING_NAME...]
    --global-impl=STRING_NAME[,STRING_NAME...]
    --global-skip-impl=STRING_NAME[,STRING_NAME...]
```

`STRING_NAME` is a string literal without spaces or quotes. Multiple names
are supported and can be specified by a comma-separated list.

The `--impl` option is used to search for an implementation containing one of
the names provided. If the first implementation fetched doesn't match any of the
names, benchdnn will use the library API to fetch the next one in the list until
it finds the one that matches or reaches the end of the list. If the end is
reached and no implementations were identified as suitable, the `SKIPPED` status
will be returned.

The `--skip-impl` option is used to skip implementations containing one of the
names provided. If the first implementation fetched matches any of the names,
benchdnn will use the library API to fetch the next one in the list until it
finds the one that doesn't match or reaches the end of the list. If the end is
reached and no implementations were identified as suitable, the `SKIPPED` status
will be returned.

Global counterparts act identically with the only exception - they override
values from `--impl` and `--skip-impl` options because these options are
considered as local to a specific problem. This gives the advantage of forcing
the filtering policy regardless of what policies are set in the batch files.
For example, if a batch file filters out the reference implementation, with the
`--global-impl=ref` option that restriction will be obliviated.

## Details
For the preceding options, "matching" means a complete match of the
`STRING_NAME` with the string queried from the library. For example, if the
library returns `marvelous:any`, the `--impl=super` option will skip this
implementation because `marvelous:any` doesn't contain the string `super`.
However, the `--impl=marv` option will fetch it because `marvelous` contains
`marv`.

`-v6` provides additional information about filtering.

## Limitations

The options are logically opposite to each other. Because of this, they are
controlled by the same object internally. Thus, if both options are specified,
only the latter one will take effect.

## Examples
This example demonstrates the importance of precision in the matching process:
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

This example demonstrates the effectiveness of the global option. Note that the
local option will be printed in the reproducer script:
``` sh
benchdnn --matmul -v6 --global-impl=ref --skip-impl=gemm,ref 64x64:64x32
create: --matmul --impl=ref 64x64:64x32
[IMPL_FILTER] Implementation skipped:: brg_matmul:avx512_core
[IMPL_FILTER] Implementation skipped:: gemm:jit:f32
[IMPL_FILTER] Implementation skipped:: brg_matmul:avx2
...
oneDNN implementation: ref:any
run: --matmul --impl=ref 64x64:64x32
...
0:PASSED __REPRO: --matmul --impl=ref 64x64:64x32
```
