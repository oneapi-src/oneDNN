#===============================================================================
# Copyright 2021-2022 Intel Corporation
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

if(graph_options_cmake_included)
    return()
endif()
set(graph_options_cmake_included true)

# ============
# C++ Standard
# ============

option(DNNL_GRAPH_SUPPORT_CXX17 "uses features from C++ standard 17" OFF)

# ========
# Features
# ========

option(DNNL_GRAPH_VERBOSE
    "allows oneDNN Graph library be verbose whenever ONEDNN_GRAPH_VERBOSE
    environment variable set to 1, 2 or 3" ON)

option(DNNL_GRAPH_ENABLE_COMPILED_PARTITION_CACHE
    "enables compiled partition cache." ON) # enabled by default

option(DNNL_GRAPH_LAYOUT_DEBUG
    "allows backends in oneDNN Graph library to generate user-comprehensive
    layout id which helps debugging for layout propagation" OFF)

option(DNNL_GRAPH_ENABLE_DUMP
    "allows oneDNN Graph library to dump graphs and pattern file via ONEDNN_GRAPH_DUMP 
    environment variable" OFF)

# =============================
# Building properties and scope
# =============================

set(DNNL_GRAPH_LIBRARY_TYPE "SHARED" CACHE STRING
    "specifies whether oneDNN Graph library should be SHARED, STATIC, or
    SDL (single dynamic library).")

option(DNNL_GRAPH_BUILD_EXAMPLES "builds examples"  OFF)

option(DNNL_GRAPH_BUILD_TESTS "builds tests" OFF)

option(COVERAGE_REPORT "builds oneDNN Graph with coverage support" OFF)

# ===================
# Engine capabilities
# ===================

set(DNNL_GRAPH_CPU_RUNTIME "OMP" CACHE STRING
    "specifies the threading runtime for CPU engines;
    supports OMP (default).")

if(NOT "${DNNL_GRAPH_CPU_RUNTIME}" MATCHES "^(NONE|OMP|TBB|SEQ|DPCPP|THREADPOOL)$")
    message(FATAL_ERROR "Unsupported CPU runtime: ${DNNL_GRAPH_CPU_RUNTIME}")
endif()

set(_DNNL_GRAPH_TEST_THREADPOOL_IMPL "STANDALONE" CACHE STRING
    "specifies which threadpool implementation to use when
    DNNL_GRAPH_CPU_RUNTIME=THREADPOOL is selected. Valid values: STANDALONE")
if(NOT "${_DNNL_GRAPH_TEST_THREADPOOL_IMPL}" MATCHES "^(STANDALONE)$")
    message(FATAL_ERROR
        "Unsupported threadpool implementation: ${_DNNL_GRAPH_TEST_THREADPOOL_IMPL}")
endif()

set(DNNL_GRAPH_GPU_RUNTIME "NONE" CACHE STRING
    "specifies the runtime to use for GPU engines.
    Can be NONE (default; no GPU engines)
    or DPCPP (DPC++ GPU engines).")

if(NOT "${DNNL_GRAPH_GPU_RUNTIME}" MATCHES "^(NONE|DPCPP)$")
    message(FATAL_ERROR "Unsupported GPU runtime: ${DNNL_GRAPH_GPU_RUNTIME}")
endif()

# =========================
# Developer and debug flags
# =========================

option(DNNL_GRAPH_ENABLE_ASAN "builds oneDNN Graph with AddressSanitizer" OFF)


# ======================================
# Graph compiler backend related options
# ======================================

option(DNNL_GRAPH_BUILD_COMPILER_BACKEND "builds graph compiler backend" OFF)
if(DNNL_GRAPH_BUILD_COMPILER_BACKEND)
    add_definitions(-DDNNL_GRAPH_ENABLE_COMPILER_BACKEND)
endif()
set(DNNL_GRAPH_LLVM_CONFIG "AUTO" CACHE STRING "graph compiler backend LLVM config")

# ===================
# Testings properties
# ===================

option(DNNL_GRAPH_BUILD_FOR_CI
    "specifies whether oneDNN Graph library will use special testing enviroment
    for internal testing processes"
    OFF)

set(DNNL_GRAPH_TEST_SET "CI" CACHE STRING
    "specifies testing targets coverage. Supports CI, NIGHTLY.
    
    When CI option is set, it enables a subset of test targets to run. When
    NIGHTLY option is set, it enables a broader set of test targets to run.")

if(DNNL_GRAPH_CPU_RUNTIME STREQUAL "DPCPP")
    set(DNNL_GRAPH_CPU_SYCL true)
    add_definitions(-DDNNL_GRAPH_CPU_SYCL)
endif()

if(DNNL_GRAPH_GPU_RUNTIME STREQUAL "DPCPP")
    set(DNNL_GRAPH_GPU_SYCL true)
    add_definitions(-DDNNL_GRAPH_GPU_SYCL)
endif()

if(DNNL_GRAPH_CPU_SYCL OR DNNL_GRAPH_GPU_SYCL)
    set(DNNL_GRAPH_WITH_SYCL true)
    add_definitions(-DDNNL_GRAPH_WITH_SYCL)
endif()
