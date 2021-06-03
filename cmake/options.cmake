#===============================================================================
# Copyright 2021 Intel Corporation
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
    "allows oneDNN Graph library be verbose whenever DNNL_GRAPH_VERBOSE
    environment variable set to 1, 2 or 3" ON)

option(DNNL_GRAPH_LOGGER
    "allows oneDNN Graph library to output logging information" OFF)

# =============================
# Building properties and scope
# =============================

option(DNNL_GRAPH_BUILD_EXAMPLES "builds examples"  OFF)

option(DNNL_GRAPH_BUILD_TESTS "builds tests" OFF)

option(COVERAGE_REPORT "builds oneDNN Graph with coverage support" OFF)

# ===================
# Engine capabilities
# ===================

set(DNNL_GRAPH_CPU_RUNTIME "OMP" CACHE STRING
    "specifies the threading runtime for CPU engines;
    supports OMP (default).")

if(NOT "${DNNL_GRAPH_CPU_RUNTIME}" MATCHES "^(OMP|SEQ|DPCPP|THREADPOOL)$")
    message(FATAL_ERROR "Unsupported CPU runtime: ${DNNL_GRAPH_CPU_RUNTIME}")
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
