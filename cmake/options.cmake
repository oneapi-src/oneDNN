#===============================================================================
# Copyright 2018 Intel Corporation
# Copyright 2026 Advanced Micro Devices, Inc.
# Copyright 2026 Arm Ltd. and affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#===============================================================================

# Manage different library options
#===============================================================================

if(options_cmake_included)
    return()
endif()
set(options_cmake_included true)

if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    set(DNNL_IS_MAIN_PROJECT TRUE)
else()
    set(DNNL_IS_MAIN_PROJECT FALSE)
endif()

# Defines a build option and reports when value other than default
# is set by the user.
# Optional fourth argument allows to pass value other than default
# to support backward compatibility scenarios.
function(custom_option name default docstring)
    if(${ARGC} GREATER 3)
        set(value "${ARGV3}")
    else()
        set(value "${default}")
    endif()

    # Using XOR comparison rule for boolean optons as CMake has
    # many ways to express TRUE/FALSE.
    if("${default}" MATCHES "^(ON|OFF|TRUE|FALSE|YES|NO|0|1)$")
        set(${name} ${value} CACHE BOOL "${docstring}")
        if((NOT ${${name}} AND ${default}) OR (${${name}} AND NOT ${default}))
            message(STATUS "${name}: ${${name}} (default: ${default})")
        endif()
    else()
        set(${name} "${value}" CACHE STRING "${docstring}")
        if(NOT "${${name}}" STREQUAL "${default}")
            message(STATUS "${name}: ${${name}} (default: ${default})")
        endif()
    endif()
endfunction()

# Defines a build option with ONEDNN_ prefix and handles logic
# necessary for backward compatibility with legacy variants.
function(onednn_option name default docstring)
    set(onednn_opt "ONEDNN_${name}")
    set(dnnl_opt "DNNL_${name}")

    # Read legacy DNNL_ option value if proper option is not defined.
    if(NOT DEFINED ${onednn_opt} AND DEFINED ${dnnl_opt})
        message(STATUS "Setting ${onednn_opt} from legacy ${dnnl_opt}")
        custom_option(${onednn_opt} "${default}" "${docstring}" "${${dnnl_opt}}")
    else()
        custom_option(${onednn_opt} "${default}" "${docstring}")
    endif()

    # Save `DNNL_` alias to avoid messing with macro definition in C++ code.
    # TODO: Remove after cleanup of internal macro names
    set(${dnnl_opt} "${${onednn_opt}}" CACHE INTERNAL "Alias for ${onednn_opt}" FORCE)
endfunction()

# ==============
# Common options
# ==============

# ---------------------
# Library configuration
# ---------------------

onednn_option(LIBRARY_TYPE "SHARED"
    "Specifies whether oneDNN library should be SHARED or STATIC")

onednn_option(LIBRARY_NAME "dnnl"
    "Specifies name of the library. For example, user can use this variable to
     specify custom library names for CPU and GPU configurations to safely
     include them into their CMake project via add_subdirectory")

onednn_option(ARCH_OPT_FLAGS "HostOpts"
    "Specifies compiler optimization flags (see below for more information).
    If empty default optimization level would be applied which depends on the
    compiler being used.

    - For Intel C++ Compilers the default option is `-xSSE4.1` which instructs
      the compiler to generate the code for the processors that support SSE4.1
      instructions. This option would not allow to run the library on older
      architectures.

    - For GNU* Compiler Collection and Clang, the default option is `-msse4.1` which
      behaves similarly to the description above.

    - For Clang and GCC compilers on RISC-V architecture this option accepts the
      `-march=<ISA-string>` flag. The RV64 backend emits vector kernels at runtime
      and does not use this option to include or exclude JIT sources. The default
      is the portable `-march=rv64gc` baseline; an explicit value such as
      `-march=rv64gcv` allows the compiler to generate code for that target ISA.

    - For all other cases there are no special optimizations flags.

    If the library is to be built for generic architecture (e.g. built by a
    Linux distribution maintainer) one may want to specify DNNL_ARCH_OPT_FLAGS=\"\"
    to not use any host-specific instructions")

onednn_option(BLAS_VENDOR "NONE"
    "Use an external BLAS library. Valid values:
      - NONE (default)
        Use internal BLAS implementation. Recommended in most situations.
      - ACCELERATE: Accelerate BLAS
      - ARMPL: Arm Performance Libraries
      - ANY: FindBLAS will search default library paths for a known BLAS
        installation. This vendor is supported for performance analysis
        purposes only.")

onednn_option(DPCPP_HOST_COMPILER "DEFAULT"
    "Specifies host compiler for Intel oneAPI DPC++ Compiler")

onednn_option(INSTALL_MODE "DEFAULT"
    "Specifies installation mode; supports DEFAULT and BUNDLE.
    When BUNDLE option is set oneDNN will be installed as a bundle
    which contains examples and benchdnn.") # internal
if (NOT "${DNNL_INSTALL_MODE}" MATCHES "^(DEFAULT|BUNDLE)$")
    message(FATAL_ERROR "Unsupported install mode: ${DNNL_INSTALL_MODE}")
endif()

# -------------
# Functionality
# -------------

onednn_option(BUILD_GRAPH ON "Enables Graph API and related optimizations")

onednn_option(ENABLE_CONCURRENT_EXEC OFF
    "Disables sharing a common scratchpad between primitives.
    This option must be turned ON if there is a possibility of executing
    distinct primitives concurrently.
    CAUTION: enabling this option increases memory consumption.")

onednn_option(ENABLE_PRIMITIVE_CACHE ON "Enables primitive cache.")

onednn_option(ENABLE_WORKLOAD "TRAINING"
    "Specifies a set of functionality to be available at build time. Designed to
    decrease the final memory disk footprint of the shared object or application
    statically linked against the library. Valid values:
    - TRAINING (the default). Includes all functionality to be enabled.
    - INFERENCE. Includes only forward propagation kind functionality and their
      dependencies.")
if(NOT "${DNNL_ENABLE_WORKLOAD}" MATCHES "^(TRAINING|INFERENCE)$")
    message(FATAL_ERROR "Unsupported workload type: ${DNNL_ENABLE_WORKLOAD}")
endif()

onednn_option(ENABLE_PRIMITIVE "ALL"
    "Specifies a set of primitives to be available at build time. Valid values:
    - ALL (the default). Includes all primitives to be enabled.
    - <PRIMITIVE_NAME>. Includes only the selected primitive to be enabled.
      Possible values are: BATCH_NORMALIZATION, BINARY, CONCAT, CONVOLUTION,
      DECONVOLUTION, ELTWISE, GROUP_NORMALIZATION, INNER_PRODUCT,
      LAYER_NORMALIZATION, LRN, MATMUL, POOLING, PRELU, REDUCTION, REORDER,
      RESAMPLING, RNN, SDPA, SHUFFLE, SOFTMAX, SUM.
    - <PRIMITIVE_NAME>;<PRIMITIVE_NAME>;... Includes only selected primitives to
      be enabled at build time. This is treated as CMake string, thus, semicolon
      is a mandatory delimiter between names. This is the way to specify several
      primitives to be available in the final binary.")

onednn_option(EXPERIMENTAL OFF
    "Enables experimental features in oneDNN.
    When enabled, each experimental feature has to be individually selected
    using environment variables.")

onednn_option(EXPERIMENTAL_UKERNEL OFF
    "Enables experimental functionality for ukernels. This option works
    independently from DNNL_EXPERIMENTAL.")

onednn_option(EXPERIMENTAL_GROUPED_MEMORY OFF
    "Enables experimental support for grouped memory format and grouped GEMM.
    This option works independently from DNNL_EXPERIMENTAL.")

onednn_option(EXPERIMENTAL_PROFILING OFF
    "Enables experimental profiling capabilities. This option works independently
    from DNNL_EXPERIMENTAL.")

onednn_option(EXPERIMENTAL_LOGGING OFF
    "Enables experimental functionality for logging. This option works
    independently from DNNL_EXPERIMENTAL.")

onednn_option(EXPERIMENTAL_SYCL_KERNEL_COMPILER OFF
    "Enables experimental SYCL OpenCL kernel compiler extension. This option
    works independently from DNNL_EXPERIMENTAL.")

onednn_option(EXPERIMENTAL_ENABLE_SYCL_PROFILING_TAG OFF
    "Enable use of the experimental sycl_ext_oneapi_profiling_tag extension 
    for verbose profiling. When disabled, verbose profiling requires a 
    profiling-enabled SYCL queue to function.")

# -------------------
# Debug and profiling
# -------------------

onednn_option(ENABLE_JIT_PROFILING ON
    "Enables registration of oneDNN kernels that are generated at
    runtime with VTune Profiler (on by default). Without the
    registrations, VTune Profiler would report data collected inside
    the kernels as `outside any known module`.")

onednn_option(ENABLE_ITT_TASKS ON
    "Enables ITT Tasks tagging feature and tag all primitive execution
    (on by default). VTune Profiler can group profiling results based
    on those ITT tasks and show corresponding timeline information.")

onednn_option(ENABLE_GRAPH_DUMP ON "Enables saving subgraphs defined using
    Graph API to disk when ONEDNN_GRAPH_DUMP environment variable is set.")

onednn_option(VERBOSE ON "Enables verbose mode output when ONEDNN_VERBOSE
    environment variable set")

onednn_option(DEV_MODE OFF "Enables internal tracing capabilities")

# -------------
# Documentation
# -------------

onednn_option(BUILD_DOC OFF "Enables building documentation")

onednn_option(DOC_VERSIONS_JSON "" "Location of JSON file for
    PyData Sphinx Theme version switcher. Must be a stable, persistent,
    fully resolved URL. Enables documentation version switcher when set.")

# ----------
# Validation
# ----------

onednn_option(BUILD_EXAMPLES ${DNNL_IS_MAIN_PROJECT} "Enables building examples")

onednn_option(BUILD_TESTS ${DNNL_IS_MAIN_PROJECT} "Enables building tests")

onednn_option(TEST_SET "CI"
    "Specifies the testing coverage. The variable consists of two parts:
    the set value defining the number of test cases, and the modifiers for
    testing commands. The input is expected in the CMake list style - a
    semicolon separated string, e.g., DNNL_TEST_SET=CI;NO_CORR.")

onednn_option(CODE_COVERAGE "NONE"
    "Enables code coverage instrumentation. Currently only gcov supported")
if(NOT "${DNNL_CODE_COVERAGE}" MATCHES "^(NONE|GCOV)$")
    message(FATAL_ERROR "Unsupported code coverage tool: ${DNNL_CODE_COVERAGE}")
endif()

onednn_option(USE_CLANG_SANITIZER ""
    "Instructs build system to use a Clang sanitizer. Possible values:
    Address: enables AddressSanitizer
    Leak: enables LeakSanitizer
    Memory: enables MemorySanitizer
    MemoryWithOrigin: enables MemorySanitizer with origin tracking
    Thread: enables ThreadSanitizer
    Undefined: enables UndefinedBehaviourSanitizer
    This feature is only available on Linux.")

onednn_option(USE_CLANG_TIDY "NONE"
    "Instructs build system to use clang-tidy. Valid values:
    - NONE (default)
      Clang-tidy is disabled.
    - CHECK
      Enables checks from .clang-tidy.
    - FIX
      Enables checks from .clang-tidy and fix found issues.
    This feature is only available on Linux.")

onednn_option(WERROR OFF "Enables treating warnings as errors")

onednn_option(BUILD_FOR_CI OFF
    "Specifies whether oneDNN library will use special testing environment for
    internal testing processes") # internal

onednn_option(ENABLE_MEM_DEBUG OFF "Enables memory-related debug functionality,
    such as buffer overflow (default) and underflow, using gtests and benchdnn.
    Additionally, this option enables testing of out-of-memory handling by the
    library, such as failed memory allocations, using primitive-related gtests.
    This feature is experimental and is only available on Linux.") # internal

onednn_option(ENABLE_STACK_CHECKER OFF "Enables stack checker that can be used to get
    information about stack consumption for a particular library entry point.
    This feature is only available on Linux (see src/common/stack_checker.hpp
    for more details).
    Note: This option requires enabling concurrent scratchpad
    (DNNL_ENABLE_CONCURRENT_EXEC).") # internal

custom_option(BENCHDNN_USE_RDPMC OFF
    "Enables rdpmc counter to report precise CPU frequency in benchdnn.
    CAUTION: may not work on all cpus (hence disabled by default)") # internal

# ===========
# CPU options
# ===========

# ------------------
# Common CPU options
# ------------------

onednn_option(CPU_RUNTIME "OMP"
    "Specifies the threading runtime for CPU engines;
    supports OMP (default), TBB, SYCL, SEQ, or THREADPOOL.")

if(NOT "${DNNL_CPU_RUNTIME}" MATCHES "^(NONE|OMP|TBB|SEQ|THREADPOOL|DPCPP|SYCL)$")
    message(FATAL_ERROR "Unsupported CPU runtime: ${DNNL_CPU_RUNTIME}")
endif()

custom_option(_DNNL_TEST_THREADPOOL_IMPL "STANDALONE"
    "Specifies which threadpool implementation will be used in tests when
    the library is built with DNNL_CPU_RUNTIME=THREADPOOL. Valid values: STANDALONE, EIGEN,
    EIGEN_ASYNC, TBB")
if(NOT "${_DNNL_TEST_THREADPOOL_IMPL}" MATCHES "^(STANDALONE|TBB|EIGEN|EIGEN_ASYNC)$")
    message(FATAL_ERROR
        "Unsupported threadpool implementation: ${_DNNL_TEST_THREADPOOL_IMPL}")
endif()

onednn_option(XLA_SOURCE_DIR ""
    "Path to an OpenXLA/XLA source tree used to build the minimal XLA async
    runtime subset for _DNNL_TEST_THREADPOOL_IMPL=EIGEN_ASYNC when no XLA
    CMake package is available.")

custom_option(TBBROOT ""
    "Path to Threading Building Blocks (TBB) package.")

# ---------------
# x64 CPU options
# ---------------

onednn_option(ENABLE_MAX_CPU_ISA ON
    "Enables control of CPU ISA detected by oneDNN via DNNL_MAX_CPU_ISA
    environment variable and dnnl_set_max_cpu_isa() function")

onednn_option(ENABLE_CPU_ISA_HINTS ON
    "Enables control of CPU ISA specific hints by oneDNN via DNNL_CPU_ISA_HINTS
    environment variable and dnnl_set_cpu_isa_hints() function")

onednn_option(ENABLE_PRIMITIVE_CPU_ISA "ALL"
    "Specifies a set of implementations using specific CPU ISA to be available
    at build time. Regardless of value chosen, compiler-based optimized
    implementations will always be available. Valid values:
    - ALL (the default). Includes all ISA to be enabled.
    - <ISA_NAME>. Includes selected and all \"less\" ISA to be enabled.
      Possible values are: AVX2, AVX512, AMX. The linear order is
      AVX2 < AVX512 < AMX. It means that if user selects, e.g. AVX512 ISA,
      AVX2 implementations will also be available at build time.
      Legacy values SSE41 and AVX are accepted with a warning and ignored.")

onednn_option(ENABLE_GEMM_KERNELS_ISA "ALL"
    "Specifies an ISA set of GeMM kernels residing in x64/gemm folder to be
    available at build time. Valid values:
    - ALL (the default). Includes all ISA kernels to be enabled.
    - NONE. Removes all kernels and interfaces.
    - <ISA_NAME>. Enables all ISA up to ISA_NAME included.
      Possible values are: AVX2, AVX512. The linear order is
      AVX2 < AVX512 < AMX (or ALL). It means that if user selects, e.g.
      AVX512 ISA, AVX2 kernels will also be present at build time.
      Legacy values SSE41 and AVX are accepted with a warning and ignored.")

onednn_option(SAFE_RBP OFF
    "Prohibits RBP register clobbering in JIT kernels. Use this option to enable
    runtime profiling with tools like Flame Graph.")

onednn_option(X64_USE_ZEN OFF
    "Enable ZenDNN integration. When ON, configuration requires a discoverable
    ZenDNN package (>= 6.0.0) and ONEDNN_CPU_RUNTIME=OMP, and fails otherwise.")
set(ZENDNNROOT "" CACHE STRING "Path to ZenDNN installation root")

# -------------------
# AArch64 CPU options
# -------------------

onednn_option(AARCH64_USE_ACL OFF "Enables use of AArch64 optimised functions
    from Arm Compute Library.
    This is only supported on AArch64 builds and assumes there is a
    functioning Compute Library build available at the location specified by the
    environment variable ACL_ROOT_DIR.")

# KleidiAI defaults to ON on supported platforms/configurations
set(_dnnl_aarch64_use_kai_supported ON)
if(NOT DNNL_TARGET_ARCH STREQUAL "AARCH64"
        OR DNNL_CPU_RUNTIME STREQUAL "THREADPOOL"
        OR MSVC)
    set(_dnnl_aarch64_use_kai_supported OFF)
endif()
onednn_option(AARCH64_USE_KAI ${_dnnl_aarch64_use_kai_supported}
    "Enables AArch64 KleidiAI-based implementations.")

if(NOT _dnnl_aarch64_use_kai_supported AND DNNL_AARCH64_USE_KAI)
    # Force users to rethink and reconfigure if the cached value was
    # previously ON or user explicitly requested KAI for unsupported configuration.
    message(FATAL_ERROR
        "KleidiAI is not supported for non-AArch64 targets and is not "
        "yet supported for threadpool runtime or MSVC")
endif()

# ===========
# GPU options
# ===========

# ------------------
# Common GPU options
# ------------------

onednn_option(GPU_RUNTIME "NONE"
    "Specifies the runtime to use for GPU engines.
    Can be NONE (default; no GPU engines), OCL (OpenCL GPU engines)
    or SYCL (SYCL GPU engines).

    Using OpenCL for GPU requires setting OPENCLROOT if the libraries are
    installed in a non-standard location.")

if(NOT "${DNNL_GPU_RUNTIME}" MATCHES "^(OCL|NONE|DPCPP|SYCL|ZE)$")
    message(FATAL_ERROR "Unsupported GPU runtime: ${DNNL_GPU_RUNTIME}")
endif()

onednn_option(GPU_VENDOR "INTEL"
    "Specifies target GPU vendor for GPU engines when DNNL_GPU_RUNTIME
    is not NONE. Can be INTEL (default), NVIDIA, AMD, or GENERIC.")

if(NOT "${DNNL_GPU_VENDOR}" MATCHES "^(INTEL|NVIDIA|AMD|GENERIC)$")
    message(FATAL_ERROR "Unsupported GPU vendor: ${DNNL_GPU_VENDOR}")
endif()

# -----------------
# Intel GPU options
# -----------------

onednn_option(ENABLE_PRIMITIVE_GPU_ISA "ALL"
    "Specifies a set of implementations using specific GPU ISA to be available
    at build time. Regardless of value chosen, reference OpenCL-based
    implementations will always be available. Valid values:
    - ALL (the default). Includes all ISA to be enabled.
    - <ISA_NAME>;<ISA_NAME>;... Includes only selected ISA to be enabled.
      Possible values are: XEHPG, XEHPC, XE2, XE3, XE3P.")


custom_option(OPENCLROOT ""
    "Path to OpenCL SDK.
    Use this option to specify custom location for OpenCL.")

onednn_option(OCL_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/third_party/opencl"
    "Path to OpenCL headers. Defaults to the headers bundled in
    third_party/opencl.")

onednn_option(ZE_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/third_party/level_zero"
    "Path to Level Zero headers. Defaults to the headers bundled in
    third_party/level_zero.")

onednn_option(DISABLE_GPU_REF_KERNELS OFF
    "Disables use of reference implementations on Intel GPUs.") # internal

# -----------------
# Other GPU options
# -----------------

onednn_option(AMD_SYCL_KERNELS_TARGET_ARCH ""
    "Specifies the target architecture (e.g. gfx90a when compiling on AMD MI210)
    to be used for compiling generic SYCL kernels for AMD vendor.
    When this option is set to a valid architecture (see LLVM target column in
    https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html#supported-gpus
    for supported architectures), the generic SYCL kernels will be enabled for AMD
    vendor. If not set, the SYCL kernels will not be compiled.
    Warning: This option is temporary and will be removed as soon as the compiler
    stops to require specifying the target architecture. After removing the option
    the generic SYCL kernels will always be enabled for AMD vendor.") # internal

onednn_option(INTERNAL_ENABLE_GENERIC_SYCL_KERNELS OFF
    "Enables implementations with generic SYCL kernels when DNNL_GPU_VENDOR is
    not GENERIC.")
