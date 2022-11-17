# Common Options

**Benchdnn** drivers support a set of options available for every driver.
The following common options are supported:

* `--allow-enum-tags-only=BOOL` -- Instructs the driver to validate format tags
  against the documented tags from `dnnl_format_tag_t` enumeration only.  When
  BOOL is `true` (the default), the only allowed format tags are the ones from
  `dnnl_format_tag_t` enumeration.

* `--attr-same-pd-check=BOOL` -- Instructs the driver to compare two primitive
  descriptors - one with added attributes and one without them. When BOOL is
  `true`, check will return error if adding of attributes caused fallback to
  generic kernel, when optimized kernel lacked proper support.

* `--batch=FILE` -- Instructs the driver to take options and problem descriptors
  from a FILE. If several `--batch` options are specified, the driver will read
  input files consecutively. Nested inclusion of `--batch` option is supported.
  The driver searches for a file by extracting a directory where `FILE` is
  located and tries to open `dirname(FILE)/FILE`. If file was not found, it
  tries to find it in a default path
  `/path_to_benchdnn_binary/inputs/DRIVER/FILE`. If file was not found again, an
  error is reported.

* `--canonical=BOOL` -- Instructs the driver to print a canonical form of a
  reproducer line. When `BOOL` is `false` (the default), the driver prints the
  minimal reproducer line omitting options and problem descriptor entries with
  default values.

* `--cpu-isa-hints=HINTS` -- Specifies the ISA specific hints to the CPU engine.
  `HINTS` values can be `none` (the default), `no_hints` or `prefer_ymm`. `none`
  value respects the `DNNL_CPU_ISA_HINTS` environment variable setting, while
  others will override it with chosen value. The settings other than `none` take
  place immediately after the parsing and subsequent attempts to set the hints
  will result in runtime error.

* `--engine=KIND[:INDEX]` -- Specifies an engine kind KIND to be used for
  benchmarking. KIND values can be `cpu` (the default) or `gpu`. Optional
  non-negative integer value of `INDEX` may be specified followed by colon `:`.
  E.g. `--engine=gpu:1`, which means to use second GPU device. Enumeration
  follows the library enumeration identification. It may be checked up with
  ONEDNN_VERBOSE output. By default `INDEX` is `0`. If index is greater or equal
  to the number of devices of requested kind discovered on a system, runtime
  error will occur.

* `--mem-check=BOOL` -- Instructs the driver to perform a device RAM capability
  check if the problem fits the device. When BOOL is `true` (the default), the
  check is performed.

* `--memory-kind=KIND` -- Specifies the memory kind to test with DPC++ and
  OpenCL engines. `KIND` values can be `usm` (default), `buffer`, `usm_device`
  (to use malloc_device) or `usm_shared` (to use malloc_shared).

* `--mode=MODE` -- Specifies **benchdnn** mode to be used for benchmarking.
  `MODE` values can be:
    - `C` or `c` for correctness testing (the default),
    - `P` or `p` for performance testing,
    - `PO` or `po` for profiling-based performance testing (GPU only),
    - `CP` or `cp` for both correctness and performance testing,
    - `R` or `r` for run mode.
    - `I` or `i` for initialization mode.
    - `L` or `l` for listing mode.
  Refer to [modes](benchdnn_general_info.md) for details.

* `--reset` -- Instructs the driver to reset DRIVER-OPTIONS (not
  COMMON-OPTIONS!) to their default values. The only exception is
  `--perf-template` option which will not be reset.

* `--skip-impl=STR` -- Instructs the driver to jump to the next implementation
  in the list if the name of the one returned matches `STR`. `STR` is a string
  literal with no spaces. When `STR` is empty (the default), the driver behavior
  is not modified. `STR` supports several patterns to be matched against through
  `,` delimiter between patterns. A name of implementation fetched is searched
  against all patterns specified, and if any of patterns match any part of
  implementation name string, it counts as a hit. E.g. `--skip-impl=ref,gemm`
  will make `ref:any` or `x64:gemm:jit` implementations to be skipped.

* `--start=N` -- Specifies the test index `N` to start testing from. All tests
  before the index are skipped.

* `-vN`, `--verbose=N` -- Specifies the driver verbose level. It prints
  additional information depending on a level `N`. `N` is a non-negative integer
  value. The default value is `0`. Refer to [verbose](knobs_verbose.md) for
  details.

* `--ctx-init=MAX_CONCURENCY[:CORE_TYPE[:THREADS_PER_CORE]]` --
  Specifies the threading context for primitive creation.
  `MAX_CONCURRENCY` is an integer value or `auto` (default) and
  specifies the maximum number of threads in the context.
  `CORE_TYPE` is an integer value or `auto` (default) that specifies the
  type or cores used in the context for hybrid CPUs, 0 being the
  largest cores available on the system (TBB runtime only).
  `THREADS_PER_CORE` is an integer value or `auto` that allows to
  enable (value 2) or disable (value 1) hyper-threading (TBB runtime only).

* `--ctx-exe=MAX_CONCURENCY[:CORE_TYPE[:THREADS_PER_CORE]]` --
  Specifies the threading context for primitive execution.
  Accepted values are similar to the `ctx-init` option.


The following common options are applicable only for correctness mode:

* `--fast-ref-gpu=BOOL` -- Instructs the driver to use faster reference path
  when doing correctness testing if `--engine=gpu` is specified. When `BOOL`
  equals `true` (the default), the library best fit CPU implementation is used
  to compute the reference path. Designed to speed up the correctness testing
  for GPU. Currently, the option is supported by limited number of drivers.

The following common options are applicable only for a performance mode:

* `--fix-times-per-prb=N` -- Specifies the limit in rounds for performance
  benchmarking set per problem. `N` is a non-negative integer. When `N` is set
  to `0` (the default), time criterion is used for benchmarking instead. This
  option is useful for performance profiling, when certain amount of cycles is
  desired.

* `--max-ms-per-prb=N` -- Specifies the limit in milliseconds for performance
  benchmarking set per problem. `N` is an integer positive number in a range
  [1e2, 6e4]. If a value is out of the range, it will be saturated to range
  board values. The default is `3e3`. This option helps to stabilize the
  performance numbers reported for small problems.

* `--perf-template=STR` -- Specifies the format of performance report. `STR`
  values can be `def` (the default), `csv` or a custom set of supported flags.
  Refer to [performance report](knobs_perf_report.md) for details.
