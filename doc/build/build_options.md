# Build Options {#dev_guide_build_options}

oneDNN Graph supports the following build-time options.

| CMake Option                | Supported values (defaults in bold) | Description
| :---                        | :---                                | :---
| DNNL_GRAPH_LIBRARY_TYPE     | **SHARED**, STATIC, SDL             | Defines the resulting library type
| DNNL_GRAPH_CPU_RUNTIME      | **OMP**, TBB, SEQ, THREADPOOL, DPCPP| Defines the threading runtime for CPU engines
| DNNL_GRAPH_GPU_RUNTIME      | **NONE**, DPCPP                     | Defines the offload runtime for GPU engines
| DNNL_GRAPH_BUILD_EXAMPLES   | ON, **OFF**                         | Controls building the examples
| DNNL_GRAPH_BUILD_TESTS      | ON, **OFF**                         | Controls building the tests
| DNNL_GRAPH_VERBOSE          | **ON**, OFF                         | Enables verbose mode
| DNNL_GRAPH_SUPPORT_CXX17    | ON, **OFF**                         | Enables features from c++ standard 17 (gcc/clang >= 5)
| DNNL_GRAPH_ENABLE_COMPILED_PARTITION_CACHE | **ON**, OFF          | Enables compiled partition cache
| DNNL_GRAPH_ENABLE_DUMP      | ON, **OFF**                         | Enables graphs and pattern file dump
| DNNL_GRAPH_BUILD_COMPILER_BACKEND | ON, **OFF**                   | Enables building graph compiler backend
| DNNL_GRAPH_LLVM_CONFIG      | **AUTO**, {llvm-config EXECUTABLE}  | Defines the method for detecting/configuring LLVM

All other building options or values that can be found in CMake files are
intended for development/debug purposes and are subject to change without
notice. Please avoid using them.

Note that graph compiler backend only supports **OMP** as its threading runtime
for CPU engines so far, other threading runtimes shall be supported in future and
currently will result in fatal error.

## Build Graph Compiler

oneDNN Graph provides more aggresive operator fusion ability via compiler
technology. To enable the feature, users need to set
`DNNL_GRAPH_BUILD_COMPILER_BACKEND=ON` explicitly in the CMake command line:

~~~sh
cmake .. -DDNNL_GRAPH_BUILD_COMPILER_BACKEND=ON
~~~~

Or change the default value in [options.cmake](../../cmake/options.cmake:104) to
`ON`:

~~~sh
option(DNNL_GRAPH_BUILD_COMPILER_BACKEND "builds graph compiler backend" ON)
~~~

Option `DNNL_GRAPH_LLVM_CONFIG` is only valid when
`DNNL_GRAPH_BUILD_COMPILER_BACKEND=ON`.

## Common Options

### CPU Options

Intel Architecture Processors and compatible devices are supported by
oneDNN Graph CPU engine. The CPU engine is built by default and cannot
be disabled at build time.

#### CPU Runtimes

CPU engine can use OpenMP, Threading Building Blocks (TBB) or sequential
threading runtimes. OpenMP threading is the default build mode. This behavior
is controlled by the `DNNL_GRAPH_CPU_RUNTIME` CMake option. Currently, this
option will be directly passed to DNNL build option `DNNL_CPU_RUNTIME` and not
affect the behavior of oneDNN Graph itself.

##### OpenMP

oneDNN Graph uses OpenMP runtime library provided by the compiler.

@warning
Because different OpenMP runtimes may not be binary-compatible, it's important
to ensure that only one OpenMP runtime is used throughout the application.
Having more than one OpenMP runtime linked to an executable may lead to
undefined behavior including incorrect results or crashes. However as long as
both the library and the application use the same or compatible compilers there
would be no conflicts.

### GPU Options

Intel Processor Graphics is supported by oneDNN Graph GPU engine. GPU engine
is disabled in the default build configuration.

#### GPU Runtimes

To enable GPU support you need to specify the GPU runtime by setting
`DNNL_GRAPH_GPU_RUNTIME` CMake option. The default value is `"NONE"` which
corresponds to no GPU support in the library.
