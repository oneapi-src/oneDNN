# Common Options

**Benchdnn** drivers support a set of options that are available for every
driver. Some of them represent global state settings which modify the execution
behavior. Options may be supported in any `mode` or be specific to correctness
or performance validation.

## Any mode settings

### --allow-enum-tags-only
`--allow-enum-tags-only=BOOL` instructs the driver to validate format tags
against the documented tags from `dnnl_format_tag_t` enumeration only when
`BOOL` is `true` (the default). When `BOOL` is `false`, any valid format tag
is accepted when creating a testing object.

### --batch
`--batch=FILE` instructs the driver to take options and problem descriptors
from a `FILE`. If several `--batch` options are specified, the driver reads
input files consecutively. Nested inclusion of the `--batch` option is
supported. The driver searches for a file by extracting a directory where `FILE`
is located and tries to open `dirname(FILE)/FILE`. If the file is not found, it
tries to find the file in a default path
`/path_to_benchdnn_binary/inputs/DRIVER/FILE`. If the file is still not
found, an error is reported. Note that `--batch` option doesn't change the
previous state.

### --canonical
`--canonical=BOOL` instructs the driver to print a canonical form of a
reproducer line. When `BOOL` is `false` (the default), the driver prints the
minimal reproducer line, omitting options and problem descriptor entries with
default values.

### --cpu-isa-hints
`--cpu-isa-hints=HINTS` specifies the ISA specific hints to the CPU engine.
`HINTS` values can be `none` (the default), `no_hints` or `prefer_ymm`.
`None` value respects the `DNNL_CPU_ISA_HINTS` environment variable setting,
while others override it with a chosen value. Settings other than `none` take
place immediately after the parsing and subsequent attempts to set the hints
result in a runtime error.

### --ctx-init
`--ctx-init=MAX_CONCURENCY[:CORE_TYPE[:THREADS_PER_CORE]]` specifies the
threading context for a testing object creation.
* `MAX_CONCURRENCY` is a
  positive integer value or `auto` (default) and specifies the maximum number of
  threads in the context.
* `CORE_TYPE` is a non-negative integer value or `auto` (default) that specifies
  the type of cores used in the context for hybrid CPU systems, `0` being the
  largest cores available on the system (TBB runtime only).
* `THREADS_PER_CORE` is a positive integer value or `auto` that allows users to
  enable (value `2`) or disable (value `1`) hyper-threading (TBB runtime only).

### --ctx-exe
`--ctx-exe=MAX_CONCURENCY[:CORE_TYPE[:THREADS_PER_CORE]]` specifies the
threading context for a testing object execution. The setting values follow ones
from the `ctx-init` option.

### --engine
`--engine=KIND[:INDEX]` specifies an engine kind `KIND` to be used for
benchmarking. `KIND` values can be `cpu` (the default) or `gpu`. An optional
non-negative integer value of `INDEX` may be specified followed by a colon
`:`, such as `--engine=gpu:1`, which means to use a second GPU device.
Enumeration follows the library enumeration identification. It may be
checked up with ONEDNN_VERBOSE output. By default, `INDEX` is `0`. If the
index is greater or equal to the number of devices of requested kind
discovered on a system, a runtime error occurs.

### --mem-check
`--mem-check=BOOL` instructs the driver to perform a device RAM capability
check if the problem fits the device including all service memory allocations.
The check is enabled when `BOOL` is `true` (the default) and disabled otherwise.

### --memory-kind
`--memory-kind=KIND` specifies the memory kind to test with DPC++ and OpenCL
runtimes. `KIND` values can be `usm` (default), `buffer`, `usm_device`
(to use malloc_device) or `usm_shared` (to use malloc_shared).

### --mode
`--mode=MODE` specifies **benchdnn** mode to be used for benchmarking.
`MODE` values can be:
  - `C` or `c` for correctness testing (the default)
  - `P` or `p` for performance testing
  - `F` or `f` for fast performance testing, an alias for
               `--mode=P --mode-modifier=PM --max-ms-per-prb=10`
  - `CP` or `cp` for both correctness and performance testing
  - `B` or `b` for bitwise (numerical determinism) testing
  - `R` or `r` for run mode
  - `I` or `i` for initialization mode
  - `L` or `l` for listing mode

Refer to [modes](benchdnn_general_info.md) for details.

### --mode-modifier
`--mode-modifier=MODIFIER` specifies a modifier to a selected benchmarking mode.
`MODIFIER` values can be:
  - empty for no modifiers (the default)
  - `P` or `p` for parallel backend object creation
  - `M` or `m` for disabling usage of host memory (GPU only)

Refer to [mode modifiers](benchdnn_general_info.md) for details.

Note: The `P` modifier sets the default value of scratchpad mode to `user`.
For the **benchdnn** functionality to work properly, the recommendation is to
pass this option **before** the driver name so that the modifier is processed
before the execution flow starts and can propagate a new scratchpad value. The
flow is affected when the user passes descriptors directly. When using batch
files, no difference is observed since a batch file starts a new parsing cycle
underneath, and a scratchpad value is propagated.

### --stream-kind
`--stream-kind=KIND` specifies the stream kind to test with DPC++ and OpenCL
runtimes by providing flags to the stream. The queue object is managed inside
the library. `KIND` values can be `def` (default), `in_order`, or
`out_of_order`. Refer to `dnnl_stream_flags_t` for more information.

### --repeats-per-prb
`--repeats-per-prb=N` specifies the `N` number of times to run a given problem.
The default `N` is `1`. When several problems are provided, each of them will be
executed `N` times. The option is designed to help reproduce sporadic failures
when effects like race condition or garbage values in a memory may not be
triggered from a single run but from several runs.

### --reset
`--reset` instructs the driver to reset DRIVER-OPTIONS (not COMMON-OPTIONS!) to
their default values. The only exception is `--perf-template` option which will
not be reset. COMMON-OPTIONS describe a global state and, thus, are not affected
by this option.

### --skip-impl
`--skip-impl=STR` instructs the driver to jump to the next implementation
in the list if the name of the returned one matches `STR` symbol-by-symbol.
`STR` is a string literal with no spaces. When `STR` is empty (the default), the
driver uses the first fetched implementation. `STR` supports several patterns to
be matched against through the comma `,` delimiter between patterns. The name of
a fetched implementation is searched against all specified patterns; and if any
of the patterns match any part of the implementation name string, it counts as a
hit. For example, `--skip-impl=ref,gemm` causes `ref:any` or `x64:gemm:jit`
implementations to be skipped.

### --start
`--start=N` specifies the test index `N` to start testing from. All tests
before the index `N` will be skipped.

### --verbose
`--verbose=N`, or a short form `-vN`, specifies the driver verbosity level.
Additional information is printed to the stdout depending on a level `N`. `N` is
a non-negative integer value. The default value is `0`. Refer to
[verbose](knobs_verbose.md) for details.

## Correctness mode settings

### --attr-same-pd-check
`--attr-same-pd-check=BOOL` instructs the driver to compare two primitive
descriptors - the one with user requested attributes and the one without any
attributes. When `BOOL` is `true`, the check returns an error if implementation
names mismatch for two descriptors. It indicates that appending an attribute
changes the implementation dispatching which is an undesired behavior. When
`BOOL` is `false` (the default), the check is disabled.

### --fast-ref
`--fast-ref=BOOL` instructs the driver to use an optimized implementation
from the library as a reference path for correctness comparison when `BOOL` is
`true` (the default). Refer to [additional documentation](knob_use_fast_ref.md)
for more information.

## Performance mode settings

### --cold-cache
`--cold-cache=MODE` instructs the driver to enable a cold cache measurement
mode. When `MODE` is set to `none` (the default), cold cache is disabled.
When `MODE` is set to `wei`, cold cache is enabled for weights argument
only. This mode targets forward and backward by data propagation kinds. When
`MODE` is set to `all`, cold cache is enabled for each execution argument.
This targets any propagation kind but mostly bandwidth-limited functionality
to emulate first access to data or branching cases. When `MODE` is set to
`custom`, cold cache is enabled for specified arguments, but it requires source
code adjustments. Refer to [cold cache](cold_cache.md) for more information.

### --fix-times-per-prb
`--fix-times-per-prb=N` specifies the `N` number of rounds per problem to run,
where `N` is a non-negative integer value. When `N` is set to `0` (the default),
the number of rounds will be established by the time criterion instead. For `N`
greater than `0`, the number of runs will be overridden by this setting. The
option makes performance profiling easier when a certain number of cycles is
desired or when a specific number of runs is expected.

### --max-ms-per-prb
`--max-ms-per-prb=N` specifies the `N` time limit in milliseconds per problem to
run. `N` is a positive integer value in a `[1e1, 6e4]` range. When a provided
value is out of the range, it is saturated to the board values. The default is
`3e3`, or 3 seconds. The option is useful, for example, to stabilize the
performance numbers reported for small problems on CPU.

### --num-streams
`--num-streams=N` specifies the number `N` of streams used for performance
benchmarking. The option takes place for GPU only and uses a single stream by
default.

### --perf-template
`--perf-template=STR` specifies the format of a performance report. `STR`
values can be `def` (the default), `csv` or a custom set of supported flags.
Refer to [performance report](knobs_perf_report.md) for details.
