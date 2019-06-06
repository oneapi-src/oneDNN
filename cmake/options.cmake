#===============================================================================
# Copyright 2018 Intel Corporation
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

# ========
# Features
# ========

option(MKLDNN_VERBOSE
    "allows Intel(R) MKL-DNN be verbose whenever MKLDNN_VERBOSE
    environment variable set to 1" ON) # enabled by default

option(MKLDNN_ENABLE_CONCURRENT_EXEC
    "disables sharing a common scratchpad between primitives.
    This option must be turned on if there is a possibility of concurrent
    execution of primitives that were created in the same thread.
    CAUTION: enabling this option increases memory consumption"
    OFF) # disabled by default

# =============================
# Building properties and scope
# =============================

set(MKLDNN_LIBRARY_TYPE "SHARED" CACHE STRING
    "specifies whether Intel(R) MKL-DNN library should be SHARED or STATIC")
option(MKLDNN_BUILD_EXAMPLES "builds examples"  ON)
option(MKLDNN_BUILD_TESTS "builds tests" ON)
option(MKLDNN_BUILD_FOR_CI "specifies whether Intel(R) MKL-DNN library should be built for CI" OFF)
option(MKLDNN_WERROR "treat warnings as errors" OFF)

set(MKLDNN_INSTALL_MODE "DEFAULT" CACHE STRING
    "specifies installation mode; supports DEFAULT or BUNDLE.

    When BUNDLE option is set MKL-DNN will be installed as a bundle
    which contains examples and benchdnn.")

set(MKLDNN_THREADING "OMP" CACHE STRING
    "specifies threading type; supports OMP (default) or TBB.

    To use Intel(R) Threading Building Blocks (Intel(R) TBB) one should also
    set TBBROOT (either environment variable or CMake option) to the library
    location")


# =============
# Optimizations
# =============

set(MKLDNN_ARCH_OPT_FLAGS "HostOpts" CACHE STRING
    "specifies compiler optimization flags (see below for more information).
    If empty default optimization level would be applied which depends on the
    compiler being used.

    - For Intel(R) C++ Compilers the default option is `-xHOST` which instructs
      the compiler to generate the code for the architecture where building is
      happening. This option would not allow to run the library on older
      architectures.

    - For GNU* Compiler Collection version 5 and newer the default options are
      `-march=native -mtune=native` which behaves similarly to the description
      above.

    - For all other cases there are no special optimizations flags.

    If the library is to be built for generic architecture (e.g. built by a
    Linux distributive maintainer) one may want to specify MKLDNN_ARCH_OPT_FLAGS=\"\"
    to not use any host specific instructions")

# ======================
# Profiling capabilities
# ======================

option(MKLDNN_ENABLE_JIT_PROFILING
    "Enable registration of Intel(R) MKL-DNN kernels that are generated at
    runtime with Intel VTune Amplifier (on by default). Without the
    registrations, Intel VTune Amplifier would report data collected inside
    the kernels as `outside any known module`."
    ON)

# ===================
# Engine capabilities
# ===================

set(MKLDNN_CPU_BACKEND "NATIVE" CACHE INTERNAL
    "specifies the type of backend for CPU engines.
    Can be NATIVE only (used for internal purposes).")

set(MKLDNN_GPU_BACKEND "NONE" CACHE STRING
    "specifies the type of backend for GPU engines.
    Can be NONE (no GPU backend) or OPENCL (OpenCL GPU backend).")

set(OPENCLROOT "" CACHE STRING
    "path to Intel(R) SDK for OpenCL(TM).
    Use this option to specify custom location for OpenCL.")

# =============
# Miscellaneous
# =============

option(BENCHDNN_USE_RDPMC
    "enables rdpms counter to report precise cpu frequency in benchdnn.
     CAUTION: may not work on all cpus (hence disabled by default)"
    OFF) # disabled by default

# =========================
# Developer and debug flags
# =========================

set(MKLDNN_USE_CLANG_SANITIZER "" CACHE STRING
    "instructs build system to use a Clang sanitizer. Possible values:
    Address: enables AddressSanitizer
    Memory: enables MemorySanitizer
    MemoryWithOrigin: enables MemorySanitizer with origin tracking
    Undefined: enables UndefinedBehaviourSanitizer
    This feature is experimental and is only available on Linux.")

option(_MKLDNN_USE_MKL "use BLAS functions from Intel MKL" OFF)
