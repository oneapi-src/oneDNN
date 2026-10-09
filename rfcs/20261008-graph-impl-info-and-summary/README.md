# Graph API: Query Implementation Info and Support `--summary` in Benchdnn

## 1. Background

For primitives, oneDNN exposes the name of the dispatched implementation through
`dnnl_primitive_desc_query(pd, dnnl_query_impl_info_str, ...)` in C and
`primitive_desc_base::impl_info_str()` in C++. benchdnn uses it in all primitive
drivers to:

- append `--impl=<name>` to every `__REPRO` line, and
- print an implementation statistics table at the end of a run
  (`--summary=impl`, enabled by default; `--summary=impl-csv` for CSV).

Example from `benchdnn --conv --batch=test_conv_smoke`:

```
=========================================================================
= Implementation statistics (--summary=no-impl to disable)              =
=========================================================================
|                              brg_conv_fwd:avx10_1_512_amx : 54 (21%)  |
|                                              x64:gemm:jit : 34 (13%)  |
|                                  brg_conv_fwd:avx10_1_512 : 24 (10%)  |
|                                           jit:avx512_core : 23 (9%)   |
...
```

The graph API has no equivalent. The graph backend already knows which kernel it
picked for a compiled partition (for example `sdp_primitive_kernel_t`,
`sdp_decomp_kernel_t` or `larger_partition_kernel_t`). That name is only visible
in verbose output (`ONEDNN_VERBOSE=profile`), as the kernel field of the graph
verbose line.

Graph partitions often have several kernels that can implement them. For
example, an SDPA partition can be dispatched to:

- `sdp_primitive_kernel_t`, a fused SDPA primitive;
- `sdp_decomp_kernel_t`, a decomposed implementation;
- `larger_partition_kernel_t`, a generic sequence of primitives.

All of them produce correct results, but their performance differs a lot.

Currently, if a change in the library makes a test case silently dispatch to a
different kernel:

- the case still passes correctness checks;
- nothing fails in CI;
- nothing is reported in the benchdnn output.

But real workloads that hit that pattern may see a significant performance
change. The only way to notice is to run with verbose enabled and inspect the
logs manually.

The goals of this RFC:

1. Provide a public API to query the implementation name of a compiled
   partition, mirroring `impl_info_str` for primitives.
2. Use it in the benchdnn graph driver so that `--summary` prints the
   dispatching statistics for graph cases, and so that the `__REPRO` line
   records which kernel ran. Comparing the summary, or its CSV form, between
   CI runs makes dispatching changes visible.

## 2. Proposal

### C API

```c
/// Queries the implementation name of a compiled partition. The name
/// identifies the kernel dispatched by the backend for the compiled partition
/// and matches the kernel field reported in verbose output.
///
/// @param compiled_partition The handle of target compiled_partition.
/// @param impl_info The output implementation name.
/// @returns #dnnl_success on success or a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_graph_compiled_partition_query_impl_info(
        const_dnnl_graph_compiled_partition_t compiled_partition,
        const char **impl_info);
```

### C++ API

```cpp
class compiled_partition {
    // ...
    /// Returns the implementation name of the compiled partition, which
    /// identifies the kernel dispatched by the backend.
    const char *impl_info_str() const;
};
```

The method name and return type match `primitive_desc_base::impl_info_str()`, so
users see the same concept in both APIs.

The C API returns `dnnl_invalid_arguments` if `compiled_partition` or
`impl_info` is NULL. The C++ API will convert any C API failure into a C++
exception.

### API Semantics

- Content: The string identifies the backend kernel selected at compile time.
  Its main purpose is debugging and profiling. Like `dnnl_query_impl_info_str`,
  the exact value is implementation-defined and may change between releases.
  Users must not rely on it for functional decisions.
- Lifetime: The string is owned by the compiled partition. It is valid until
  `dnnl_graph_compiled_partition_destroy()` is called, and the user must not
  free it.
- Consistency: Repeated queries on the same compiled partition return the same
  pointer.

## 3. Implementation String Format

### Options

**Option A: graph kernel name only (recommended).**
Examples: `sdp_primitive_kernel_t`, `sdp_decomp_kernel_t`,
`larger_partition_kernel_t`, `matmul_t`, `conv_fwd_t`.

- (+) Directly answers which graph-level dispatching path was taken, which is
  where silent fallbacks happen (for example fused SDPA to decomposed or large
  partition).
- (+) Identical to the kernel field in verbose output, so there is a single
  source of truth.
- (−) Doesn't show which primitive implementation runs underneath, for
  example the ISA of a brgemm convolution behind `conv_fwd_t`.

**Option B: graph kernel name plus inner primitive implementation names.**
Examples: `sdp_primitive_kernel_t:ocl:micro:reusable`,
`conv_fwd_t:brg_conv_fwd:avx10_1_512_amx`.

- (+) Gives the same level of detail as primitive drivers.
- (−) A partition may execute many primitives. Listing all of them produces
  long, unbounded strings that change whenever any internal reorder changes.
  Listing only the main compute primitives requires a per-kernel definition of
  what counts as main.
- (−) Needs a new kernel interface to collect primitive descriptors from every
  kernel.
- (−) Primitive-level details are already available through primitive verbose
  output, which reports every primitive a partition executes.

**Option C: Option A with a backend prefix.**
Example: `dnnl_backend:sdp_primitive_kernel_t`.

- (+) Unambiguous if more backends are added.
- (−) Only one production backend exists today.

### Recommendation

The recommendation is to go with Option A. A partition may execute several
primitives, and it isn't practical to enumerate all of their implementation
names in the partition info string. The graph kernel name captures the
dispatching decision we want to monitor. Primitive-level details remain
available through verbose output. The API signature doesn't depend on the
format, so the string can be extended later, without an API change.

## 4. benchdnn Integration

### `--summary`

After all partitions of a case are compiled, the graph driver sets
`res->impl_name` from `impl_info_str()` of each compiled partition. For
multi-partition graphs, the names are joined with `+`.

With `res->impl_name` populated, the existing benchdnn infrastructure works
without driver-specific code:

- `--summary=impl` (default) prints the statistics table;
- `--summary=impl-csv` prints the CSV line;
- the `__REPRO` line includes `--impl=<name>`.

Example from `benchdnn --graph --batch=test_graph_ci`:

```
============================================================
= Implementation statistics (--summary=no-impl to disable) =
============================================================
|    larger_partition_kernel_t : 94 (40%)                  |
|                     binary_t : 19 (8%)                   |
|                eltwise_fwd_t : 17 (7%)                   |
|                   conv_fwd_t : 15 (6%)                   |
|                eltwise_bwd_t : 14 (6%)                   |
|                  reduction_t : 9 (4%)                    |
|                    reorder_t : 8 (3%)                    |
|        quantize_dequantize_t : 7 (3%)                    |
|          sdp_decomp_kernel_t : 6 (3%)                    |
| sdp_decomp_training_kernel_t : 6 (3%)                    |
|                     matmul_t : 5 (2%)                    |
...
============================================================
```

```
0:PASSED (6432 ms) __REPRO: --graph --impl=sdp_decomp_kernel_t --case=complex_fusion/mha/GQA-f32.json
```

### `--impl` / `--skip-impl`

The common `__REPRO` logic is unchanged, so graph repro lines contain
`--impl=<name>`. To keep repro lines runnable, the graph driver accepts `--impl`
and `--skip-impl` but currently ignores them, printing a warning:

```
Info: `--impl` and `--skip-impl` options are not supported by the graph driver and will be ignored.
```

Real filtering is left as a follow-up feature request. Unlike primitives, the
graph API has no `next_impl()` iterator, so filtering would most likely skip a
non-matching case rather than fetch another implementation.
