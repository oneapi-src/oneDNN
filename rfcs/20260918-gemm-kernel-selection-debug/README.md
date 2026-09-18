# JIT GPU GEMM Kernel Selection: Debugging and Override

## Summary

Add selection logs and per-primitive kernel overrides for oneDNN JIT GEMM on
Intel GPUs. Developers and external tuners can inspect candidates, select a
kernel by rank or strategy string, and read back the finalized strategy for
replay.

Readback uses `ONEDNN_VERBOSE=xe=info`; overrides use an internal C API or
benchdnn's `--gpu-kernel` option. These features require `ONEDNN_DEV_MODE=ON`
and are compiled out of release builds. There is no query API or ABI guarantee.

## Motivation

Existing debugging tools leave four gaps:

- Selection logs lack a shared identifier linking candidates to the chosen
  kernel and do not translate scores into estimated time.
- `GEMM_KERNEL` applies globally, accepts only a full strategy string, and
  asserts on invalid input.
- The chosen kernel is logged before finalization, so the string does not
  describe what ran and cannot be replayed.
- External tuners lack an API to override selection per problem.

## Proposal

### 1. Selection logging

Each candidate has a **global catalog index**, stable across runs using the
same catalog build, and a **rank**, assigned by the evaluator for the current
problem (`0` is the default choice).

`ONEDNN_VERBOSE=xe=<level>` accepts either a severity name or its integer value:

| Level | Integer | Output |
|-------|---------|--------|
| `info` | `150` | Finalized chosen kernel |
| `debug` | `160` | Every considered candidate, every skip, and the built kernel |

Candidate, skip, and choice lines use the same aligned catalog index. Candidate
and choice lines have these formats:

```text
[<global>] consider score:<score> [<ms>ms] <catalog strategy>
[<global>] use <rank>/<count>[ pin] <finalized strategy>
```

The score is evaluator cost in aggregate EU-cycles; lower is better. Estimated
time is `score / (eu_count * clock_MHz * 1e3)` milliseconds, omitted when the
clock is unavailable. It is an idealized comparison aid, not measured runtime.

Real output (PVC, `xe=debug`, `512x1024:1024x2048`):

```text
[DEBUG][src/gpu/intel/gemm/jit/gen_kernel.cpp:61] [ 987] consider score:1.01616e+08 [0.146ms] F gemm SSS NNN 32 16 aB16+m8@32 aS32+m16@40 aB wg 4x4 kc16 nse hi pt sb256 bk0 sn grf256 sr
[DEBUG][src/gpu/intel/gemm/jit/gen_kernel.cpp:61] [ 993] consider score:7.30621e+07 [0.105ms] F gemm SSS NNN 64 16 aB8x2+B8@8 aS8x2+S16@8 aB wg 2x4x4 kr kc8 nse hi pt sr kv sb64 bk0 sn grf256 afb
...
[ INFO][src/gpu/intel/gemm/jit.hpp:425] [ 993] use 0/10 gemm SSS NNN 64 16 1 0 aB8x2{cc}+B8,8@8{cc} aS8x2{cc}+S16,16@8{cc} aB{cc/ub} wg 2x4x4 kr kc8 k16 grf256 sn nse di sr ch kv afb pk32 li pt sb64 np bm1048576 bn262144 bk16777216 | dispatch k0=0 wgK=1
```

`consider` prints the catalog strategy; `use` prints the finalized strategy,
including cache hints and any dispatch suffix described below. The strings
therefore differ even when their catalog index matches. `pin` marks a rank
override.

### 2. Kernel override

An override on the primitive attribute forces kernel selection for that
primitive. It accepts two forms:

| Form | Meaning |
|------|---------|
| `N` | Candidate with rank `N` (`0` = default) |
| `gemm ...` | Full strategy string from a `use` line |

Rank depends on the catalog build and the problem. Persist strategy strings
for replay. Invalid overrides return `unimplemented` instead of asserting. The
generator also rejects degenerate strategies that would otherwise crash.

A full-strategy override is logged as:

```text
[override] use <finalized strategy>
```

#### Poor-man tuning

Sweep ranks from `0` to `<count> - 1`, compare performance, and capture the
fastest candidate's strategy for replay. For a problem with five candidates:

```sh
for i in 0 1 2 3 4; do
    ONEDNN_VERBOSE=xe=info ONEDNN_PRIMITIVE_CACHE_CAPACITY=0 \
        ./build/tests/benchdnn/benchdnn -v5 --engine=gpu --mode=F \
        --cold-cache=all --matmul --gpu-kernel="$i" --dt=bf16 8x1024:1024x4096
done | grep perf,gpu
```

```text
perf,gpu,jit:gemm:any,,--mode=F --engine=gpu --cold-cache=all --matmul --dt=bf16:bf16:bf16 --gpu-kernel=0 8x1024:1024x4096,0.0671089,164.855,0.01744,3847.99,0.019875,3376.54
perf,gpu,jit:gemm:any,,--mode=F --engine=gpu --cold-cache=all --matmul --dt=bf16:bf16:bf16 --gpu-kernel=1 8x1024:1024x4096,0.0671089,165.327,0.01728,3883.61,0.0198,3389.34
perf,gpu,jit:gemm:any,,--mode=F --engine=gpu --cold-cache=all --matmul --dt=bf16:bf16:bf16 --gpu-kernel=2 8x1024:1024x4096,0.0671089,25.0774,0.0144,4660.34,0.0152833,4391
perf,gpu,jit:gemm:any,,--mode=F --engine=gpu --cold-cache=all --matmul --dt=bf16:bf16:bf16 --gpu-kernel=3 8x1024:1024x4096,0.0671089,37.0898,0.0144,4660.34,0.0151889,4418.27
perf,gpu,jit:gemm:any,,--mode=F --engine=gpu --cold-cache=all --matmul --dt=bf16:bf16:bf16 --gpu-kernel=4 8x1024:1024x4096,0.0671089,689.267,0.02432,2759.41,0.0306103,2192.36
```

Compare the `perf,gpu` lines using the same performance column. Here rank `3`
reaches about 1.3x the default's throughput in the final GFLOPS column.

### 3. Reading back and replaying the kernel

Copy the finalized strategy, starting at `gemm` and including any dispatch
suffix, from the `use` line into `--gpu-kernel` or the override API. Exclude
the log prefix, indices, and optional `pin` marker.

For batch runs, `use` lines go to stdout. Associate each with the adjacent
benchdnn `N:STATUS` line; rejected cases have no preceding `use` line. Set
`ONEDNN_PRIMITIVE_CACHE_CAPACITY=0` to obtain one line per case.

#### Dispatch geometry

The evaluator derives `k0` and `wgK` outside the strategy grammar. Replaying
only the strategy can recompute different geometry, particularly for
k-parallel kernels. To preserve these values, append:

```text
<strategy> | dispatch k0=<value> wgK=<value>
```

The suffix is emitted for non-trivial geometry (`k0 > 0` or `wgK > 1`), including
non-k-parallel kernels. The parser strips the suffix before parsing the
strategy and honors its values. This preserves `k0` and `wgK`; it does not
guarantee identical execution in all circumstances.

### API and benchdnn

The internal C API sets the override on a primitive attribute. It is specific
to Intel GPUs and available only in dev-mode builds. If other backends need it,
move it to the GPU level and dispatch to backend-specific implementations.

```c
dnnl_status_t dnnl_impl_gpu_intel_set_kernel_override(
        dnnl_primitive_attr_t attr, const char *kernel);
```

The override is stored in `gpu_primitive_attr_t` and participates in the
primitive cache key.

benchdnn exposes the override through `--gpu-kernel` for GPU execution with a
dev-mode library:

| Value | Meaning |
|-------|---------|
| `N` | Rank `N` |
| `A-B` | Ranks `A` through `B`, one case per rank |
| `all` | Ranks from `0` until the library reports no matching candidate |
| `gemm ...` | Full strategy string |

Quote strategy strings because they contain spaces; generated reproduction
commands preserve the quoting. Strategy strings also contain characters such
as `,`, `:`, `|`, and `{}`. The PoC must confirm that benchdnn's command-line
and batch-file parsing does not treat them as separators.

## Scope and compatibility

- Default kernel selection is unchanged unless an override is set.
- GPU matmul accepts a GPU attribute to forward `grf_per_thread`, `use_dpas`,
  and the override to the inner GEMM. The check uses `engine_kind::gpu`, but
  only the Intel path consumes the attribute. CPU matmul continues to reject it.
