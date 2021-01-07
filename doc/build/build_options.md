# Build Options {#dev_guide_build_options}

oneDNN graph supports the following build-time options.

| CMake Option                | Supported values (defaults in bold) | Description
| :---                        | :---                                | :---
| DNNL_GRAPH_CPU_RUNTIME      | **OMP**, TBB, SEQ, THREADPOOL, DPCPP| Defines the threading runtime for CPU engines
| DNNL_GRAPH_GPU_RUNTIME      | **NONE**, DPCPP                     | Defines the offload runtime for GPU engines
| DNNL_GRAPH_BUILD_EXAMPLES   | ON, **OFF**                         | Controls building the examples
| DNNL_GRAPH_BUILD_TESTS      | ON, **OFF**                         | Controls building the tests
| DNNL_GRAPH_VERBOSE          | **ON**, OFF                         | Enables verbose mode
| DNNL_GRAPH_ENABLE_ASAN      | ON, **OFF**                         | Enables sanitizer check
| DNNL_GRAPH_SUPPORT_CXX17    | ON, **OFF**                         | Enables features from c++ standard 17 (gcc/clang >= 5)

All other building options or values that can be found in CMake files are
intended for development/debug purposes and are subject to change without
notice. Please avoid using them.

## Common Options

### CPU Options

Intel Architecture Processors and compatible devices are supported by
oneDNN graph CPU engine. The CPU engine is built by default and cannot
be disabled at build time.

#### CPU Runtimes

CPU engine can use OpenMP, Threading Building Blocks (TBB) or sequential
threading runtimes. OpenMP threading is the default build mode. This behavior
is controlled by the `DNNL_GRAPH_CPU_RUNTIME` CMake option. Currently, this
option will be directly passed to DNNL build option `DNNL_CPU_RUNTIME` and not
affect the behavior of oneDNN graph itself. There are two cases:

- when `DNNL_GRAPH_GPU_RUNTIME=DPCPP`, `DNNL_GRAPH_CPU_RUNTIME` and `DNNL_CPU_RUNTIME` will be both directly set to `DPCPP`.
- when `DNNL_GRAPH_GPU_RUNTIME=NONE`, `DNNL_GRAPH_CPU_RUNTIME` will be passed to `DNNL_CPU_RUNTIME`.

##### OpenMP

oneDNN graph uses OpenMP runtime library provided by the compiler.

@warning
Because different OpenMP runtimes may not be binary-compatible, it's important
to ensure that only one OpenMP runtime is used throughout the application.
Having more than one OpenMP runtime linked to an executable may lead to
undefined behavior including incorrect results or crashes. However as long as
both the library and the application use the same or compatible compilers there
would be no conflicts.

### GPU Options

Intel Processor Graphics is supported by oneDNN graph GPU engine. GPU engine
is disabled in the default build configuration.

#### GPU Runtimes

To enable GPU support you need to specify the GPU runtime by setting
`DNNL_GRAPH_GPU_RUNTIME` CMake option. The default value is `"NONE"` which
corresponds to no GPU support in the library.
