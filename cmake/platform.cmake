#===============================================================================
# Copyright 2016-2022 Intel Corporation
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

# Manage platform-specific quirks
# Leveraged from oneDNN project
# https://github.com/oneapi-src/oneDNN/blob/master/cmake/platform.cmake
#===============================================================================

if(dnnl_graph_platform_cmake_included)
    return()
endif()
set(dnnl_graph_platform_cmake_included true)

include("cmake/utils.cmake")

if (DNNL_GRAPH_LIBRARY_TYPE STREQUAL "SHARED" OR DNNL_GRAPH_LIBRARY_TYPE STREQUAL "SDL")
    add_definitions(-DDNNL_GRAPH_DLL)
endif()

set(CMAKE_CCXX_FLAGS)
set(CMAKE_CCXX_NOWARN_FLAGS)
set(CMAKE_CCXX_NOEXCEPT_FLAGS)

add_definitions(-D__STDC_LIMIT_MACROS -D__STDC_CONSTANT_MACROS)

if(DNNL_GRAPH_WITH_SYCL)
    # Clang cannot vectorize some loops with #pragma omp simd and gets
    # very upset. Tell it that it's okay and that we love it
    # unconditionally.
    append(CMAKE_CCXX_NOWARN_FLAGS "-Wno-pass-failed")
    # Suppress self-comparison warning to avoid false positives in macros such
    # as JIT_IMPL_NAME_HELPER. 
    append(CMAKE_CCXX_NOWARN_FLAGS "-Wno-tautological-compare")
    if(WIN32)
        # XXX: SYCL does not like __thiscall convention coming from TBB,
        # suppress warnings for now.
        append(CMAKE_CCXX_FLAGS "-Wno-ignored-attributes")
        # XXX: compiler always pulls in release C++ runtime by default, until
        # this is fixed we have to explicitly drop release C++ runtime for
        # debug build types.
        string(TOUPPER "${CMAKE_BUILD_TYPE}" UPPERCASE_CMAKE_BUILD_TYPE)
        if(UPPERCASE_CMAKE_BUILD_TYPE MATCHES "(DEBUG|RELWITHMDD)" AND NOT CMAKE_BASE_NAME MATCHES "(icx|icpx)")
            append(CMAKE_EXE_LINKER_FLAGS "-Xlinker /NODEFAULTLIB:msvcrt")
            append(CMAKE_SHARED_LINKER_FLAGS "-Xlinker /NODEFAULTLIB:msvcrt")
        endif()
    endif()
endif()

if(MSVC)
    set(USERCONFIG_PLATFORM "x64")
    append(CMAKE_CCXX_FLAGS "/WX")
    if(${CMAKE_CXX_COMPILER_ID} STREQUAL MSVC)
        append(CMAKE_CCXX_FLAGS "/MP")
        append(CMAKE_CCXX_FLAGS "/bigobj")
        # int -> bool
        append(CMAKE_CCXX_NOWARN_FLAGS "/wd4800")
        # unknown pragma
        append(CMAKE_CCXX_NOWARN_FLAGS "/wd4068")
        # double -> float
        append(CMAKE_CCXX_NOWARN_FLAGS "/wd4305")
        # UNUSED(func)
        append(CMAKE_CCXX_NOWARN_FLAGS "/wd4551")
        # int64_t -> int (tent)
        append(CMAKE_CCXX_NOWARN_FLAGS "/wd4244")
        # todo(xinyu): add reason
        append(CMAKE_CCXX_NOWARN_FLAGS "/wd4018")
        append(CMAKE_CCXX_NOWARN_FLAGS "/wd4996")
        # decorated name length exceeded
        append(CMAKE_CCXX_NOWARN_FLAGS "/wd4503")
    endif()
    if(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
        append(CMAKE_CCXX_FLAGS "/MP")
        append(CMAKE_CCXX_FLAGS "/bigobj")
        # disable: loop was not vectorized with "simd"
        append(CMAKE_CCXX_NOWARN_FLAGS "-Qdiag-disable:13379")
        # disable: loop was not vectorized with "simd"
        append(CMAKE_CCXX_NOWARN_FLAGS "-Qdiag-disable:15552")
        # disable: unknown pragma
        append(CMAKE_CCXX_NOWARN_FLAGS "-Qdiag-disable:3180")
        # disable: foo has been targeted for automatic cpu dispatch
        append(CMAKE_CCXX_NOWARN_FLAGS "-Qdiag-disable:15009")
        # disable: disabling user-directed function packaging (COMDATs)
        append(CMAKE_CCXX_NOWARN_FLAGS "-Qdiag-disable:11031")
        # disable: decorated name length exceeded, name was truncated
        append(CMAKE_CCXX_NOWARN_FLAGS "-Qdiag-disable:2586")
        # disable: disabling optimization; runtime debug checks enabled
        append(CMAKE_CXX_FLAGS_DEBUG "-Qdiag-disable:10182")
    endif()
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        append(CMAKE_CCXX_NOEXCEPT_FLAGS "-fno-exceptions")
        # Clang cannot vectorize some loops with #pragma omp simd and gets
        # very upset. Tell it that it's okay and that we love it
        # unconditionally.
        append(CMAKE_CCXX_FLAGS "-Wno-pass-failed")
        # Clang doesn't like the idea of overriding optimization flags.
        # We don't want to optimize jit gemm kernels to reduce compile time
        append(CMAKE_CCXX_FLAGS "-Wno-overriding-t-option")
    endif()
    if(DNNL_GRAPH_WITH_SYCL OR CMAKE_BASE_NAME STREQUAL "icx" OR CMAKE_BASE_NAME STREQUAL "icpx")
        # Default fp-model in icx and dpcpp (unlike clang) may be precise or
        # fast=1 depending on the version.
        append(CMAKE_CCXX_FLAGS "/fp:precise")
    endif()
elseif(UNIX OR MINGW)
    append(CMAKE_CCXX_FLAGS "-Wall -Werror -Wno-unknown-pragmas")
    if(DNNL_GRAPH_WITH_SYCL)
        # XXX: Intel oneAPI DPC++ Compiler generates a lot of warnings
        append(CMAKE_CCXX_FLAGS "-w -Wno-deprecated-declarations")
    endif()
    if(DNNL_GRAPH_WITH_SYCL OR CMAKE_BASE_NAME STREQUAL "icx" OR CMAKE_BASE_NAME STREQUAL "icpx")
        # Default fp-model in icx and dpcpp (unlike clang) may be precise or
        # fast=1 depending on the version.
        append(CMAKE_CCXX_FLAGS "-ffp-model=precise -fno-reciprocal-math")
    endif()
    append(CMAKE_CCXX_FLAGS "-fvisibility=internal")
    append(CMAKE_CXX_FLAGS "-fvisibility-inlines-hidden")
    append(CMAKE_CCXX_NOWARN_FLAGS "-Wno-sign-compare")
    append(CMAKE_CCXX_NOEXCEPT_FLAGS "-fno-exceptions")
    # compiler specific settings
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        append(CMAKE_CCXX_NOWARN_FLAGS "-Wno-gnu-statement-expression")
        append(CMAKE_CCXX_NOWARN_FLAGS "-Wno-string-conversion")
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
        # workaround for Intel Compiler that produces error caused
        # by pragma omp simd collapse(..)
        append(CMAKE_CCXX_NOWARN_FLAGS "-diag-disable:13379")
        append(CMAKE_CCXX_NOWARN_FLAGS "-diag-disable:15552")
        # disable `was not vectorized: vectorization seems inefficient` remark
        append(CMAKE_CCXX_NOWARN_FLAGS "-diag-disable:15335")
        # disable: foo has been targeted for automatic cpu dispatch
        append(CMAKE_CCXX_NOWARN_FLAGS "-diag-disable:15009")
        # disable: remark #11074: Inlining inhibited by limit max-size
        # disable: remark #11074: Inlining inhibited by limit max-total-size
        append(CMAKE_CCXX_NOWARN_FLAGS "-diag-disable:11074")
        # disable: remark #11076: To get full report use -qopt-report=4 -qopt-report-phase ipo
        append(CMAKE_CCXX_NOWARN_FLAGS "-diag-disable:11076")
        # disable: remark #2279: printf/scanf format not a string literal and no format arguments
        append(CMAKE_CCXX_NOWARN_FLAGS "-diag-disable:2279")
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        append(CMAKE_CCXX_NOWARN_FLAGS "-Wno-strict-overflow")
    endif()
endif()

if(UNIX OR MINGW)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
        # Link Intel libraries statically (except for iomp5)
        append(CMAKE_SHARED_LINKER_FLAGS "-liomp5")
        append(CMAKE_SHARED_LINKER_FLAGS "-static-intel")
        # Tell linker to not complain about missing static libraries
        append(CMAKE_SHARED_LINKER_FLAGS "-diag-disable:10237")
    endif()
endif()

append(CMAKE_C_FLAGS "${CMAKE_CCXX_FLAGS}")
append(CMAKE_CXX_FLAGS "${CMAKE_CCXX_FLAGS}")

if(APPLE)
    set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
    # FIXME: this is ugly but required when compiler does not add its library
    # paths to rpath (like Intel compiler...)
    foreach(_ ${CMAKE_C_IMPLICIT_LINK_DIRECTORIES})
        set(_rpath "-Wl,-rpath,${_}")
        append(CMAKE_SHARED_LINKER_FLAGS "${_rpath}")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${_rpath}")
    endforeach()
endif()
