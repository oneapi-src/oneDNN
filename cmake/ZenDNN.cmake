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

# ZenDNN does not support Windows builds.
if(WIN32)
    message(FATAL_ERROR
        "ONEDNN_X64_USE_ZEN=ON is not supported on Windows. "
        "Build on Linux, or configure with -DONEDNN_X64_USE_ZEN=OFF.")
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

# Require GCC >= 11.2 or Clang >= 14; ZenDNN builds only with GCC/Clang.
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
else()
    message(FATAL_ERROR
        "ONEDNN_X64_USE_ZEN=ON requires GCC >= 11.2 or Clang >= 14; ZenDNN does "
        "not support other compilers. Current C++ compiler: "
        "${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}. "
        "Build with GCC or Clang, or configure with -DONEDNN_X64_USE_ZEN=OFF.")
endif()

# Fetch or locate ZenDNN source
set(ZENDNN_SOURCE_DIR "${PROJECT_SOURCE_DIR}/third_party/ZenDNN")

if(NOT EXISTS "${ZENDNN_SOURCE_DIR}/CMakeLists.txt")
    if(ONEDNN_ZENDNN_FETCH)
        message(STATUS "Fetching ZenDNN from ${ONEDNN_ZENDNN_FETCH_URL} (${ONEDNN_ZENDNN_FETCH_TAG})")
        include(FetchContent)
        FetchContent_Declare(ZenDNN
            GIT_REPOSITORY ${ONEDNN_ZENDNN_FETCH_URL}
            GIT_TAG        ${ONEDNN_ZENDNN_FETCH_TAG}
            SOURCE_DIR     ${ZENDNN_SOURCE_DIR}
            SOURCE_SUBDIR  "not-available"
        )
        FetchContent_GetProperties(ZenDNN)
        if(NOT zendnn_POPULATED)
            FetchContent_Populate(ZenDNN)
        endif()
    else()
        message(FATAL_ERROR
            "ONEDNN_X64_USE_ZEN=ON but ZenDNN source not found at ${ZENDNN_SOURCE_DIR}. "
            "Either place ZenDNN source manually at third_party/ZenDNN, or enable "
            "ONEDNN_ZENDNN_FETCH=ON to auto-download. Configure with "
            "-DONEDNN_X64_USE_ZEN=OFF to disable ZenDNN integration.")
    endif()
endif()

# Verify ZenDNN framework integration script exists
if(NOT EXISTS "${ZENDNN_SOURCE_DIR}/fwk/ZenDnnlFwkIntegrate.cmake")
    message(FATAL_ERROR
        "ZenDNN framework integration script not found at "
        "${ZENDNN_SOURCE_DIR}/fwk/ZenDnnlFwkIntegrate.cmake. "
        "Ensure ZenDNN source is complete and up-to-date.")
endif()

# Minimum supported ZenDNN version. With ONEDNN_X64_USE_ZEN=ON, a missing ZenDNN
# fails configuration (see the FATAL_ERROR above); a ZenDNN that is present but
# older than this is treated as a misconfiguration and also fails the build.
set(ZENDNN_MIN_VERSION "6.1.0")

# Extract ZenDNN version from source
file(READ "${ZENDNN_SOURCE_DIR}/CMakeLists.txt" ZENDNN_CMAKE_CONTENT)
string(REGEX MATCH "project\\([^)]*VERSION[ \t]+([0-9]+\\.[0-9]+\\.[0-9]+)"
       ZENDNN_VERSION_MATCH "${ZENDNN_CMAKE_CONTENT}")
if(CMAKE_MATCH_1)
    set(zendnnl_VERSION "${CMAKE_MATCH_1}")
else()
    set(zendnnl_VERSION "")
endif()

# zendnnl_VERSION may be unset if CMakeLists.txt doesn't follow expected format
if(NOT DEFINED zendnnl_VERSION OR zendnnl_VERSION STREQUAL "")
    message(FATAL_ERROR
        "ONEDNN_X64_USE_ZEN=ON requires ZenDNN >= ${ZENDNN_MIN_VERSION}, but the "
        "ZenDNN source at ${ZENDNN_SOURCE_DIR} did not report a version. Use a "
        "ZenDNN source that provides version information, or configure with "
        "-DONEDNN_X64_USE_ZEN=OFF.")
elseif("${zendnnl_VERSION}" VERSION_LESS "${ZENDNN_MIN_VERSION}")
    message(FATAL_ERROR
        "ONEDNN_X64_USE_ZEN=ON requires ZenDNN >= ${ZENDNN_MIN_VERSION}. "
        "Found ZenDNN ${zendnnl_VERSION} at ${ZENDNN_SOURCE_DIR}. "
        "Update ZenDNN, or configure with -DONEDNN_X64_USE_ZEN=OFF.")
endif()

message(STATUS "Using ZenDNN source at: ${ZENDNN_SOURCE_DIR}")
message(STATUS "Found ZenDNN version: ${zendnnl_VERSION}")

# Set ZenDNN framework build paths
set(ZENDNNL_SOURCE_DIR "${ZENDNN_SOURCE_DIR}" CACHE PATH "ZenDNN source directory" FORCE)
set(ZENDNNL_BINARY_DIR "${CMAKE_BINARY_DIR}/third_party/ZenDNN" CACHE PATH "ZenDNN build directory" FORCE)
set(ZENDNNL_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/zendnn_install" CACHE PATH "ZenDNN install prefix" FORCE)

# Propagate build configuration from parent to ZenDNN build
# Build type (Debug, Release, RelWithDebInfo, etc.) must match parent for debugging and sanitizers
set(ZENDNNL_BUILD_TYPE "${CMAKE_BUILD_TYPE}" CACHE STRING "ZenDNN build type" FORCE)

# Propagate compiler flags for sanitizers, debug symbols, optimization levels
set(ZENDNNL_C_FLAGS "${CMAKE_C_FLAGS}" CACHE STRING "C compiler flags for ZenDNN" FORCE)
set(ZENDNNL_CXX_FLAGS "${CMAKE_CXX_FLAGS}" CACHE STRING "C++ compiler flags for ZenDNN" FORCE)

# Configure ZenDNN dependency options
# CRITICAL: Disable oneDNN dependency to avoid circular dependency
# (oneDNN cannot link against ZenDNN that bundles oneDNN internally)
set(ZENDNNL_DEPENDS_AOCLDLP ON CACHE BOOL "ZenDNN depends on AOCL-DLP" FORCE)
set(ZENDNNL_DEPENDS_ONEDNN OFF CACHE BOOL "ZenDNN depends on oneDNN" FORCE)
set(ZENDNNL_DEPENDS_LIBXSMM OFF CACHE BOOL "ZenDNN depends on LibXSMM" FORCE)
set(ZENDNNL_DEPENDS_PARLOOPER OFF CACHE BOOL "ZenDNN depends on parlooper" FORCE)
set(ZENDNNL_DEPENDS_FBGEMM OFF CACHE BOOL "ZenDNN depends on FBGEMM" FORCE)

# Include ZenDNN framework integration (creates zendnnl::zendnnl_archive target)
include("${ZENDNN_SOURCE_DIR}/fwk/ZenDnnlFwkIntegrate.cmake")

add_definitions(-DDNNL_X64_USE_ZEN=1)

# Helper function to resolve ZenDNN target name
function(zendnn_resolve_target out_var)
    if(TARGET zendnnl::zendnnl_archive)
        set(${out_var} zendnnl::zendnnl_archive PARENT_SCOPE)
    elseif(TARGET zendnnl::zendnnl)
        set(${out_var} zendnnl::zendnnl PARENT_SCOPE)
    else()
        message(FATAL_ERROR
            "ONEDNN_X64_USE_ZEN=ON but neither zendnnl::zendnnl_archive nor "
            "zendnnl::zendnnl target is available after ZenDNN framework integration.")
    endif()
endfunction()

# Link ZenDNN target into ${target} with given visibility (PRIVATE/PUBLIC/INTERFACE)
function(target_link_zendnnl target visibility)
    zendnn_resolve_target(_zendnnl_target)
    target_link_libraries(${target} ${visibility} ${_zendnnl_target})

    # Ensure target waits for ZenDNN ExternalProject to complete
    if(TARGET fwk_zendnnl)
        add_dependencies(${target} fwk_zendnnl)
    endif()
endfunction()

message(STATUS "Found ZenDNN framework integration at ${ZENDNN_SOURCE_DIR}")
message(STATUS "ZenDNN will be built via ExternalProject at ${ZENDNNL_BINARY_DIR}")
message(STATUS "ZenDNN dependencies: AOCL-DLP=ON, oneDNN=OFF (avoid circular dep), LibXSMM=OFF, parlooper=OFF, FBGEMM=OFF")
