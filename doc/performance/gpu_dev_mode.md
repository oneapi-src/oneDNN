GPU Developer Mode Controls {#dev_guide_gpu_dev_mode}
=====================================================

oneDNN provides additional controls for Intel GPU kernel development and
performance exploration. These controls are intended for oneDNN developers.
They are not part of the stable API and may change without notice.

## Build-time Controls

The controls are available only if the library is built with
`ONEDNN_DEV_MODE=ON`. In other builds they do not take any effect.

| CMake Option    | Supported values (defaults in bold) | Description                     |
|:----------------|:------------------------------------|:--------------------------------|
| ONEDNN_DEV_MODE | **OFF**, ON                         | Enables developer-only controls |

## GPU Logging

The `xe=<level>` token of the `ONEDNN_VERBOSE` environment variable enables
Intel GPU specific logs. These logs are independent of the generic
[verbose mode](@ref dev_guide_verbose) output and can be combined with other
verbose flags, e.g. `ONEDNN_VERBOSE=dispatch,xe=debug`.

| Level      | Description                                        |
|:-----------|:---------------------------------------------------|
| **off**    | GPU logs are disabled (default)                    |
| warning    | Warnings                                           |
| suggestion | Suggestions for better performance                 |
| info       | Implementation decisions, e.g. the selected kernel |
| debug      | Details of the implementation decisions            |
| perf       | Performance-related details                        |
| trace      | All messages                                       |
| all        | Same as `trace`                                    |

Each level includes the messages of the levels above it. An integer value on
the `debuginfo` scale is also accepted.

### Example: GEMM Kernel Selection

With `xe=info`, the JIT GEMM implementation prints the kernel it uses:

~~~sh
    $ ONEDNN_VERBOSE=xe=info ./benchdnn --matmul --engine=gpu --dt=f16 1024x1024:1024x1024
[ INFO][src/gpu/intel/gemm/jit.hpp:438] [ 659] use 0/31 gemm HH[SH] N@8N@8N 32 48 1 0 av64+m16@64 am32+m32@72 aB wg 8x4 xaf vav hi pt sr br sb64 bk0 sn grf256 sys afb
[ INFO][src/gpu/intel/gemm/jit.hpp:445] [ 659] final gemm HH[SH] N@16N@16N 32 48 1 0 aV64#16,0{cc}+M16,16@64{cc} aM32#16,0{cc}+M32,32@72{cc} aB#16,0{cc/bb} wg 8x4 sys kc1 xaf k64 grf256 raxehpc sn vav di sr bf cc li pt sb64 np bm524288 bn786432 bk16777216
~~~

The bracketed number is the catalog entry index. `use i/n` shows the rank of
the selected entry among the `n` sorted candidates. The `use` line contains
the kernel string, which can be passed back to the library as a
[kernel override](#gemm-kernel-override). The `final` line shows the
strategy after all internal adjustments.

With `xe=debug`, the implementation also prints each candidate it considers
with its model score and estimated time, and the reason for each candidate it
skips.

## GEMM Kernel Override

The JIT GEMM implementation can use a specified kernel instead of the
selected catalog entry. The override accepts one of these values:

| Value         | Description                                      |
|:--------------|:-------------------------------------------------|
| `N`           | Uses the candidate with rank `N` (`use N/n ...`) |
| kernel string | Uses the kernel described by the string          |

The override can be set in these ways:

| Method                                             | Scope                |
|:---------------------------------------------------|:---------------------|
| benchdnn `--gpu-kernel=STR` option (matmul driver) | One benchdnn problem |
| `dnnl_impl_gpu_intel_set_kernel_override(attr, s)` | One primitive        |
| `GEMM_KERNEL` environment variable (strings only)  | All GEMM problems    |

The benchdnn option also accepts a rank range `A-B` and `all`. Refer to the
[matmul driver](https://github.com/uxlfoundation/oneDNN/blob/main/tests/benchdnn/doc/driver_matmul.md)
documentation for details. The internal
`dnnl_impl_gpu_intel_get_kernel_count(pd, &count)` function returns the
number of candidates for a primitive descriptor. The environment variable
takes precedence over a kernel string set in the attribute.

### Example: Rank Override

~~~sh
    $ ONEDNN_VERBOSE=xe=info ./benchdnn --matmul --engine=gpu --dt=f16 --gpu-kernel=1 1024x1024:1024x1024
[ INFO][src/gpu/intel/gemm/jit.hpp:438] [ 658] use 1/31 pin gemm HH[SH] N@8N@8N 64 32 1 0 av32+m32@48 am32+S32@64 aB wg 4x8 xaf st vav hi pt sb32 bk0 sn grf256 sys sr br afb rr
...
~~~

### Example: Kernel String Override

Copy the part of the `use` line that starts at `gemm`. Edit the strategy
tokens as necessary, e.g. change the workgroup shape from `wg 8x4` to `wg 4x8`:

~~~sh
    $ ONEDNN_VERBOSE=xe=info ./benchdnn --matmul --engine=gpu --dt=f16 \
      --gpu-kernel="gemm HH[SH] N@8N@8N 32 48 1 0 av64+m16@64 am32+m32@72 aB wg 4x8 xaf vav hi pt sr br sb64 bk0 sn grf256 sys afb" \
      1024x1024:1024x1024
[ INFO][src/gpu/intel/gemm/jit.hpp:443] [override] use gemm HH[SH] N@8N@8N 32 48 1 0 av64+m16@64 am32+m32@72 aB wg 4x8 xaf vav hi pt sr br sb64 bk0 sn grf256 sys afb
...
~~~

The kernel string has the following format:

~~~
    gemm <precisions> <layouts> <unroll_m> <unroll_n> <alpha> <beta> <strategy>
~~~

Tokens before `gemm` are ignored. The override applies only to a compatible
problem:

- The external data types and the layouts of A, B, and C must match the
  problem.
- The alignments in the string must not exceed the actual buffer alignments.
- A fixed alpha or beta must match the problem. Use `-` to accept any value.

A string that does not apply or cannot be built is skipped and the next
implementation is used. Use `ONEDNN_VERBOSE=dispatch` to see the reason.

@warning
The override can change the internal data types of the computation. An
incompatible precision can produce inaccurate results.
