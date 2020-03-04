#===============================================================================
# Copyright 2018-2020 Intel Corporation
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

include("cmake/mkldnn_compat.cmake")

# =======================================
# CPU and JIT boring predefined constants
# =======================================

#
# These are shared to dnnl_config.h and used to "cap" the set of
# implementations available to the CPU engine using some macro magic.
#
# DNNL_CPU is directly determined from CMAKE_SYSTEM_PROCESSOR, and is
# controlled by your cmake toolchain, or your CC/CXX environment.
# Actual values agree with dnnl_types.h
set(DNNL_CPU_X86 1)
set(DNNL_CPU_VE  2)
#
#

# JIT and non-x86 feature set "cap" can be set with `cmake -DDNNL_ISA=<string>`
# All cpus support VANILLA, ANY and FULL [default] builds.
# In principle, you could cap DNNL_ISA to remove jit impls from libdnnl.
# In practice, the only important ones are VANILLA and the _FULL ones.
# Generally you should prefer using runtime CPU dispatch to run lower-level
# implementations.  Dispatch to DNNL_MAX_CPU_ISA=VANILLA is now supported.
# (Unified jit drivers might try to be clever, but probably check 'mayiuse')
set(DNNL_ISA_VANILLA 1) # ANY and FULL depend on CPU, VANILLA does not

set(DNNL_ISA_X86                10) # VANILLA + no restriction to be C/C++ ref impl
set(DNNL_ISA_SSE41              11)
set(DNNL_ISA_AVX                12)
set(DNNL_ISA_AVX2               13)
set(DNNL_ISA_AVX512_MIC         14)
set(DNNL_ISA_AVX512_MIC_4OPS    15)
set(DNNL_ISA_AVX512_CORE        16)
set(DNNL_ISA_AVX512_CORE_VNNI   17)
set(DNNL_ISA_AVX512_CORE_BF16   18)
set(DNNL_ISA_X86_FULL           18)
# ve-specific cross-compilation example: add when VANILLA is not enough
set(DNNL_ISA_VE                 30) # VANILLA + VE-specific asm, intrinsics
set(DNNL_ISA_VEDNN              31) # add external optimization library
set(DNNL_ISA_VEJIT              32) # add external JIT
set(DNNL_ISA_VE_FULL            32)
#
# Semantics:
#
# VANILLA adds the restriction of being "cross-platform"
#     Occasionally very light inline assembly might occur,
#     but code diffs should try to not impact code readability.
#     dnnl_optimize.h, dnnl_omp.h etc have some ideas for cpu-specific macros
# ANY expands on VANILLA to mean "any variety of this DNNL_CPU".
#     ANY loses the cross-platform connotation.
# FULL implies "full bells and whistles" for this DNNL_CPU.
#     (was 'all', but too easily confused with ANY)

#
# Determine target CPU (governed by available compiler / environment CC),
# and then determine what the default -DDNNL_ISA=FULL build should mean
#
# TARGET_ARCH in https://github.com/intel/mkl-dnn/pull/658 disambiguates
#     AARCH64 and ARM64 -- adjust here if xbyak and jit don't apply.
if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "^(x86_64|aarch64.*|AARCH64.*|arm64.*|ARM64.*)")
    set(DNNL_CPU ${DNNL_CPU_X86})
    set(DNNL_ISA_ANY ${DNNL_ISA_X86})
    set(DNNL_ISA_FULL ${DNNL_ISA_X86_FULL})
elseif(${CMAKE_SYSTEM_PROCESSOR} STREQUAL "aurora")
    set(DNNL_CPU ${DNNL_CPU_VE})
    set(DNNL_ISA_ANY ${DNNL_ISA_VE})
    set(DNNL_ISA_FULL ${DNNL_ISA_VE_FULL})
else()
    message(FATAL_ERROR "Unrecognized target CPU : ${CMAKE_SYSTEM_PROCESSOR}")
    # To add a new DNNL_CPU, start with ANY and FULL impls exactly the VANILLA
    # ones, until CPU-specific optimizations begin on your branch/fork.
endif()

# ===========
# CPU and JIT
# ===========

# You may specify additional CPU target info with cmake -DDNNL_ISA=<string>
# This is used for dnnl_config.h.in
#
if(DNNL_CPU EQUAL DNNL_CPU_X86)
    set(DNNL_ISA "FULL" CACHE STRING
        "All cpus must support DNNL_ISA = FULL [default] and VANILLA.

        VANILLA removes all x86 jit support and should mostly be pure C/C++.
                These impls should run on any CPU, so can be used with a
                cmake toolchain file to cross-compile libdnnl for non-x86 CPUs.
        FULL means supply all available features. x86 supports all jit targets.
             Non-x86 FULL builds on forked repos can add specialized layers to
             the basic VANILLA build to provide a cpu-specific cpu_engine.
        
        For debugging, the layer implementations included in libdnnl can be
        capped at intermediate ISA settings too (see cpu_engine.hpp macros)
        max ISA possble values: FULL (default), VANILLA, and
        x86:
           ANY SSE41 AVX AVX2 AVX512_MIC AVX512_MIC_4OPS
           AVX512_CORE AVX512_CORE_VNNI AVX512_CORE_BF16.
        VE:
        ANY VEDNN VEJIT
        In practice, ANY is pretty much the same as VANILLA.  Semantically ANY
        is CPU-specific -- it may have inline asm, cpu-specific optimizations,
        or using engine layers only available on a fork of Intel dnnl sources."
        ) # default to full support
    if(NOT ${DNNL_ISA} MATCHES "^(VANILLA|ANY|SSE41|AVX|AVX2|AVX512_MIC|AVX512_MIC_4OPS|AVX512_CORE|AVX512_CORE_VNNI|AVX512_CORE_BF16|FULL)$")
        message(FATAL_ERROR "Unsupported ${CMAKE_SYSTEM_PROCESSOR} variant: DNNL_ISA=${DNNL_ISA}")
    endif()
elseif(NECVE) # example toolchain build
    set(DNNL_ISA "FULL" CACHE STRING
        "VE cpu build styles include VANILLA, FULL(libvednn, jit,...)
        as well intermediate compile-time support for max isa VEDNN and VEJIT.
        non-x86 jit is supported via an external library.
        cpu_target.h defines TARGET_cond macros to maintain readable ref impls."
        ) # default to FULL set of bells and whistles
    if(NOT ${DNNL_ISA} MATCHES "^(VANILLA|ANY|VEDNN|VEJIT|FULL)$")
        message(FATAL_ERROR "Unsupported ${CMAKE_SYSTEM_PROCESSOR} variant: DNNL_ISA=${DNNL_ISA}")
    endif()
endif()
set(DNNL_ISA_VALUE ${DNNL_ISA_${DNNL_ISA}})
message(STATUS "Target processor ${CMAKE_SYSTEM_PROCESSOR} has DNNL_CPU=${DNNL_CPU}, and maps")
message(STATUS "  DNNL_ISA=${DNNL_ISA} to DNNL_ISA_VALUE=DNNL_ISA_${DNNL_ISA}=${DNNL_ISA_VALUE}")
# DNNL_ISA is the cmake STRING mnemonic, and
# DNNL_ISA_VALUE is a numeric constant for dnnl_config.h.in

# ========
# Features
# ========

#option(DNNL_VERBOSE
#    "allows DNNL be verbose whenever DNNL_VERBOSE
#    environment variable set to 1" ON) # enabled by default
# NEW: string (NONE or any false value) | "DEFAULT" | "EXTRA"
set(DEFAULT_DNNL_VERBOSE "DEFAULT")
if(CMAKE_BUILD_TYPE MATCHES "[Dd]ebug")
    set(DEFAULT_DNNL_VERBOSE "EXTRA")
endif()
set(DNNL_VERBOSE "${DEFAULT_DNNL_VERBOSE}" CACHE STRING
    "Possible values: NONE, DEFAULT, EXTRA.

    NONE (or any false value) removes support for DNNL Verbose Mode.
    
    DEFAULT allows DNNL to respect DNNL_VERBOSE environment
    variable or dnnl_set_verbose() calls.  Levels 0[default],
    1, 2 become active.  Adds some runtime overhead.
    Release mode build default to DEFAULT verbosity.
    
    EXTRA adds even more runtime overhead, and adds vebose levels 3,4(,5?).
    These also trace unsuccessful cpu_engine primitive creation.
    Some implementations may print a reason they were skipped.
    Debug mode builds default to EXTRA verbosity."
    )

option(DNNL_ENABLE_CONCURRENT_EXEC
    "disables sharing a common scratchpad between primitives.
    This option must be turned ON if there is a possibility of executing
    distinct primitives concurrently.
    CAUTION: enabling this option increases memory consumption."
    OFF) # disabled by default

option(DNNL_ENABLE_PRIMITIVE_CACHE "enables primitive cache.
    WARNING: the primitive cache is an experimental feature and might be
    changed without prior notification in future releases" OFF)
    # disabled by default

option(DNNL_ENABLE_MAX_CPU_ISA
    "enables control of CPU ISA detected by DNNL via DNNL_MAX_CPU_ISA
    environment variable and dnnl_set_max_cpu_isa() function" ON)

# =============================
# Building properties and scope
# =============================

set(DNNL_LIBRARY_TYPE "SHARED" CACHE STRING
    "specifies whether DNNL library should be SHARED or STATIC")
if(NOT ${DNNL_LIBRARY_TYPE} MATCHES "^(SHARED|STATIC)$")
    message(FATAL_ERROR "Unsupported ${DNNL_LIBRARY_TYPE} variant: DNNL_LIBRARY_TYPE=${DNNL_LIBRARY_TYPE}")
endif()
option(DNNL_BUILD_EXAMPLES "builds examples"  ON)
option(DNNL_BUILD_TESTS "builds tests" ON)
option(DNNL_BUILD_FOR_CI "specifies whether DNNL library should be built for CI" OFF)
option(DNNL_WERROR "treat warnings as errors" OFF)

set(DNNL_INSTALL_MODE "DEFAULT" CACHE STRING
    "specifies installation mode; supports DEFAULT or BUNDLE.

    When BUNDLE option is set DNNL will be installed as a bundle
    which contains examples and benchdnn.")
if(NOT ${DNNL_INSTALL_MODE} MATCHES "^(DEFAULT|BUNDLE)$")
    message(FATAL_ERROR "Unsupported ${DNNL_INSTALL_MODE} variant: DNNL_INSTALL_MODE=${DNNL_INSTALL_MODE}")
endif()

set(DNNL_CODE_COVERAGE "OFF" CACHE STRING
    "specifies which supported tool for code coverage will be used
    Currently only gcov supported")
if(NOT ${DNNL_CODE_COVERAGE} MATCHES "^(OFF|GCOV)$")
    message(FATAL_ERROR "Unsupported code coverage tool: ${DNNL_CODE_COVERAGE}")
endif()

set(DNNL_CODE_COVERAGE "OFF" CACHE STRING
    "specifies which supported tool for code coverage will be used
    Currently only gcov supported")
if(NOT "${DNNL_CODE_COVERAGE}" MATCHES "^(OFF|GCOV)$")
    message(FATAL_ERROR "Unsupported code coverage tool: ${DNNL_CODE_COVERAGE}")
endif()

# =============
# Optimizations
# =============

set(DNNL_ARCH_OPT_FLAGS "HostOpts" CACHE STRING
    "specifies compiler optimization flags (see below for more information).
    If empty default optimization level would be applied which depends on the
    compiler being used.

    - For Intel(R) C++ Compilers the default option is `-xSSE4.1` which instructs
      the compiler to generate the code for the processors that support SSE4.1
      instructions. This option would not allow to run the library on older
      architectures.

    - For GNU* Compiler Collection and Clang, the default option is `-msse4.1` which
      behaves similarly to the description above.

    - For all other cases there are no special optimizations flags.

    If the library is to be built for generic architecture (e.g. built by a
    Linux distributive maintainer) one may want to specify DNNL_ARCH_OPT_FLAGS=\"\"
    to not use any host specific instructions")
message(STATUS "DNNL_ARCH_OPT_FLAGS ${DNNL_ARCH_OPT_FLAGS}")

# ======================
# Profiling capabilities
# ======================

option(DNNL_ENABLE_JIT_PROFILING
    "Enable registration of DNNL kernels that are generated at
    runtime with Intel VTune Amplifier (on by default). Without the
    registrations, Intel VTune Amplifier would report data collected inside
    the kernels as `outside any known module`."
    ON)

# ===================
# Engine capabilities
# ===================

set(DNNL_CPU_RUNTIME "OMP" CACHE STRING
    "specifies the threading runtime for CPU engines;
    supports OMP (default) or TBB.

    To use Intel(R) Threading Building Blocks (Intel(R) TBB) one should also
    set TBBROOT (either environment variable or CMake option) to the library
    location.")
if(NOT "${DNNL_CPU_RUNTIME}" MATCHES "^(OMP|TBB|SEQ)$")
    message(FATAL_ERROR "Unsupported CPU runtime: ${DNNL_CPU_RUNTIME}")
endif()

set(TBBROOT "" CACHE STRING
    "path to Intel(R) Thread Building Blocks (Intel(R) TBB).
    Use this option to specify Intel(R) TBB installation locaton.")

set(DNNL_GPU_RUNTIME "NONE" CACHE STRING
    "specifies the runtime to use for GPU engines.
    Can be NONE (default; no GPU engines) or OCL (OpenCL GPU engines).

    Using OpenCL for GPU requires setting OPENCLROOT if the libraries are
    installed in a non-standard location.")
if(NOT "${DNNL_GPU_RUNTIME}" MATCHES "^(OCL|NONE)$")
    message(FATAL_ERROR "Unsupported GPU runtime: ${DNNL_GPU_RUNTIME}")
endif()

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

set(DNNL_USE_CLANG_SANITIZER "" CACHE STRING
    "instructs build system to use a Clang sanitizer. Possible values:
    Address: enables AddressSanitizer
    Memory: enables MemorySanitizer
    MemoryWithOrigin: enables MemorySanitizer with origin tracking
    Undefined: enables UndefinedBehaviourSanitizer
    This feature is experimental and is only available on Linux.")

#option(_DNNL_USE_MKL "use BLAS functions from Intel MKL" OFF)
set(DNNL_CPU_EXTERNAL_GEMM "NONE" CACHE STRING
    "NONE uses an x86 jit gemm.  For vanilla or non-x86 build a builtin
    portable gemm will be used.  Other values are: MKL, CBLAS.

    MKL can be used for performance comparisons and may support gemm
    operations with different data types/formats.

    A native CBLAS might provide a performance boost when cross-compiling.
    (For x86 vanilla build or cross-compilation, you may also investigate
    changing DNNL_ARCH_OPT_FLAGS or playing with gemm:ref tuning)")
if(NOT ${DNNL_CPU_EXTERNAL_GEMM} MATCHES "^(NONE|MKL|CBLAS)$")
    message(FATAL_ERROR "Unsupported DNNL_CPU_EXTERNAL_GEMM=${DNNL_CPU_EXTERNAL_GEMM}, expected NONE|MKL|CBLAS")
endif()
if(_DNNL_USE_MKL) # older _DNNL_USE_MKL or _MKLDNN_USE_MKL bools still accepted
    set(DNNL_CPU_EXTERNAL_GEMM MKL) # mkldnn_compat
endif()


# ==========================================
# Development work helpers (expert use only)
# ==========================================

set(DNNL_DEFAULT_ENABLE_RNN 1)
#set(DNNL_DEFAULT_ENABLE_BFLOAT16 1)  # removed, to reduce code differences
if(DNNL_CPU EQUAL DNNL_CPU_X86)
    set(TARGET_X86 TRUE)
    if(DNNL_ISA STREQUAL "VANILLA") # note: DNNL_ISA is a "string"
        message(STATUS "x86 VANILLA compile removes rnn support (for now)")
        set(DNNL_DEFAULT_ENABLE_RNN 0) # FIXME - add ref postops and try again
    else()
        set(TARGET_X86_JIT TRUE)
    endif()
elseif(DNNL_CPU EQUAL DNNL_CPU_VE)
    message(STATUS "non-x86 compile wil disable RNN support (for now)")
    #set(DNNL_DEFAULT_ENABLE_BFLOAT16 0) # initially 0 for porting
    set(DNNL_DEFAULT_ENABLE_RNN 0)      # initially 0 for porting
endif()

# Some depending on CPU & jit options, some parts of dnnl may not be ready
option(DNNL_ENABLE_RNN
    "Enable rnn support (default ${DNNL_DEFAULT_ENABLE_RNN}, for now requires x86 jit)"
    ${DNNL_DEFAULT_ENABLE_RNN})
#option(DNNL_ENABLE_BFLOAT16
#    "Enable bfloat16 support (default ${DNNL_DEFAULT_ENABLE_BFLOAT16})"
#    ${DNNL_DEFAULT_ENABLE_BFLOAT16})
# vim: ts=4 sw=4 et
