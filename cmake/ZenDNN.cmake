#*******************************************************************************
# Copyright 2026 Advanced Micro Devices, Inc.
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
#*******************************************************************************

if(zendnn_cmake_included)
    return()
endif()
set(zendnn_cmake_included true)
include("cmake/options.cmake")

if(NOT DNNL_X64_USE_ZEN)
    add_definitions(-DDNNL_X64_USE_ZEN=0)
    return()
endif()

# ZenDNN provides x86_64 kernels only. Checked before every other requirement
# below: without it, configuration completes on a non-x64 target and the x64
# ZenDNN library is still linked into dnnl (src/CMakeLists.txt gates that on
# DNNL_X64_USE_ZEN alone), so the mismatch only surfaces at link time as an
# object machine-type conflict.
if(NOT DNNL_TARGET_ARCH STREQUAL "X64")
    message(FATAL_ERROR
        "ONEDNN_X64_USE_ZEN=ON requires an x86_64 target; ZenDNN provides "
        "x86_64 kernels only. Current DNNL_TARGET_ARCH: ${DNNL_TARGET_ARCH}. "
        "Configure with -DONEDNN_X64_USE_ZEN=OFF.")
endif()

# On Windows, ZenDNN is built against the LLVM OpenMP runtime (/openmp:llvm).
# OpenMP_RUNTIME_MSVC, which selects it here, is only honored by CMake >= 3.30;
# on older CMake it is silently ignored and this build falls back to MSVC's
# /openmp (vcomp), putting two OpenMP runtimes in one process. That corrupts
# multithreaded work partitioning rather than failing loudly.
if(WIN32 AND CMAKE_VERSION VERSION_LESS "3.30")
    message(FATAL_ERROR
        "ONEDNN_X64_USE_ZEN=ON on Windows requires CMake >= 3.30. "
        "Current CMake: ${CMAKE_VERSION}. "
        "Upgrade CMake, or configure with -DONEDNN_X64_USE_ZEN=OFF.")
endif()

# ZenDNN requires CMake >= 3.26.
if(CMAKE_VERSION VERSION_LESS "3.26")
    message(FATAL_ERROR
        "ONEDNN_X64_USE_ZEN=ON requires CMake >= 3.26. "
        "Current CMake: ${CMAKE_VERSION}. "
        "Upgrade CMake, or configure with -DONEDNN_X64_USE_ZEN=OFF.")
endif()

# ZenDNN uses the OpenMP threading runtime exclusively; any other oneDNN CPU
# runtime (TBB, SEQ, THREADPOOL, ...) is incompatible.
if(NOT DNNL_CPU_RUNTIME STREQUAL "OMP")
    message(FATAL_ERROR
        "ONEDNN_X64_USE_ZEN=ON requires ONEDNN_CPU_RUNTIME=OMP; ZenDNN only "
        "supports the OpenMP threading runtime. Current ONEDNN_CPU_RUNTIME: "
        "${DNNL_CPU_RUNTIME}. Configure with -DONEDNN_CPU_RUNTIME=OMP, or "
        "configure with -DONEDNN_X64_USE_ZEN=OFF.")
endif()

if(NOT ZENDNNROOT AND DEFINED ENV{ZENDNNROOT})
    set(ZENDNNROOT "$ENV{ZENDNNROOT}")
endif()

if(ZENDNNROOT)
    set(zendnnl_DIR "${ZENDNNROOT}/lib/cmake" CACHE PATH "Path to zendnnl CMake config files")
endif()

# Minimum supported ZenDNN version. With ONEDNN_X64_USE_ZEN=ON, a missing ZenDNN
# fails configuration (see the FATAL_ERROR below); a ZenDNN that is present but
# older than this is treated as a misconfiguration and also fails the build.
set(ZENDNN_MIN_VERSION "6.0.1")
find_package(zendnnl CONFIG)

if(NOT zendnnl_FOUND)
    message(FATAL_ERROR
        "ONEDNN_X64_USE_ZEN=ON but ZenDNN was not found. Set ZENDNNROOT (or the "
        "ZENDNNROOT environment variable) to a ZenDNN install, ensure its "
        "CMake package config is discoverable, or configure with "
        "-DONEDNN_X64_USE_ZEN=OFF.")
endif()

# zendnnl_VERSION may be unset by some package configs
if(NOT DEFINED zendnnl_VERSION OR zendnnl_VERSION STREQUAL "")
    message(FATAL_ERROR
        "ONEDNN_X64_USE_ZEN=ON requires ZenDNN >= ${ZENDNN_MIN_VERSION}, but the "
        "ZenDNN package at ${zendnnl_DIR} did not report a version. Use a "
        "ZenDNN package that provides version information, or configure with "
        "-DONEDNN_X64_USE_ZEN=OFF.")
elseif("${zendnnl_VERSION}" VERSION_LESS "${ZENDNN_MIN_VERSION}")
    message(FATAL_ERROR
        "ONEDNN_X64_USE_ZEN=ON requires ZenDNN >= ${ZENDNN_MIN_VERSION}. "
        "Found ZenDNN ${zendnnl_VERSION} at ${zendnnl_DIR}. "
        "Update ZenDNN, or configure with -DONEDNN_X64_USE_ZEN=OFF.")
endif()

# Require GCC >= 11.2, Clang >= 14, or MSVC >= 19.43.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "11.2")
        message(FATAL_ERROR
            "ONEDNN_X64_USE_ZEN=ON requires GCC >= 11.2. "
            "Current C++ compiler: ${CMAKE_CXX_COMPILER_ID} "
            "${CMAKE_CXX_COMPILER_VERSION}. "
            "Upgrade GCC, or configure with -DONEDNN_X64_USE_ZEN=OFF.")
    endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "14")
        message(FATAL_ERROR
            "ONEDNN_X64_USE_ZEN=ON requires Clang >= 14. "
            "Current C++ compiler: ${CMAKE_CXX_COMPILER_ID} "
            "${CMAKE_CXX_COMPILER_VERSION}. "
            "Upgrade Clang, or configure with -DONEDNN_X64_USE_ZEN=OFF.")
    endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "19.43")
        message(FATAL_ERROR
            "ONEDNN_X64_USE_ZEN=ON requires MSVC >= 19.43 (Visual Studio 2022 "
            "17.13), the toolset the ZenDNN Windows port is built and tested "
            "with. Current C++ compiler: ${CMAKE_CXX_COMPILER_ID} "
            "${CMAKE_CXX_COMPILER_VERSION}. "
            "Upgrade MSVC, or configure with -DONEDNN_X64_USE_ZEN=OFF.")
    endif()
else()
    message(FATAL_ERROR
        "ONEDNN_X64_USE_ZEN=ON requires GCC >= 11.2, Clang >= 14, or "
        "MSVC >= 19.43; ZenDNN does not support other compilers. "
        "Current C++ compiler: "
        "${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}. "
        "Build with GCC, Clang or MSVC, or configure with "
        "-DONEDNN_X64_USE_ZEN=OFF.")
endif()

add_definitions(-DDNNL_X64_USE_ZEN=1)
# C++17 requirement is applied per-target via target_compile_features()
# in src/cpu/{,x64/}CMakeLists.txt, not project-wide.

# ZenDNN's installed headers call std::getenv and std::strncpy, which MSVC
# deprecates (C4996); oneDNN compiles with /sdl, which promotes that to an
# error. Scoped here rather than in oneDNN's global flags so it applies only to
# ZenDNN-enabled builds -- the default ONEDNN_X64_USE_ZEN=OFF build returns
# above and keeps its CRT deprecation warnings intact.
if(MSVC)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif()

# Resolve the available ZenDNN imported target name into ${out_var}. ZenDNN may
# export either a shared (zendnnl::zendnnl) or an archive
# (zendnnl::zendnnl_archive) target depending on how it was built. This is the
# single place that encodes those target names; consumers must go through this
# (or target_link_zendnnl below) rather than hard-coding the names.
function(zendnn_resolve_target out_var)
    if(TARGET zendnnl::zendnnl)
        set(${out_var} zendnnl::zendnnl PARENT_SCOPE)
    elseif(TARGET zendnnl::zendnnl_archive)
        set(${out_var} zendnnl::zendnnl_archive PARENT_SCOPE)
    else()
        message(FATAL_ERROR
            "ONEDNN_X64_USE_ZEN=ON but neither zendnnl::zendnnl nor "
            "zendnnl::zendnnl_archive target is available. Check your "
            "ZenDNN install at ZENDNNROOT=${ZENDNNROOT}.")
    endif()
endfunction()

# Link the resolved ZenDNN target into ${target} with the given visibility
# keyword (PRIVATE/PUBLIC/INTERFACE).
function(target_link_zendnnl target visibility)
    zendnn_resolve_target(_zendnnl_target)
    target_link_libraries(${target} ${visibility} ${_zendnnl_target})
endfunction()

message(STATUS "Found ZenDNN ${zendnnl_VERSION}: ${zendnnl_DIR}")
