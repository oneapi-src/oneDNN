#===============================================================================
# Copyright 2017-2020 Intel Corporation
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

# Manage OpenMP-related compiler flags
#===============================================================================

if(OpenMP_cmake_included)
    return()
endif()
set(OpenMP_cmake_included true)
include("cmake/Threading.cmake")

if (APPLE AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    # OSX Clang doesn't have OpenMP by default.
    # But we still want to build the library.
    set(_omp_severity "WARNING")
else()
    set(_omp_severity "FATAL_ERROR")
endif()

macro(forbid_link_compiler_omp_rt)
    if (NOT WIN32)
        set_if(OpenMP_C_FOUND
            CMAKE_C_CREATE_SHARED_LIBRARY_FORBIDDEN_FLAGS
            "${OpenMP_C_FLAGS}")
        set_if(OpenMP_CXX_FOUND
            CMAKE_CXX_CREATE_SHARED_LIBRARY_FORBIDDEN_FLAGS
            "${OpenMP_CXX_FLAGS}")
        if (NOT APPLE)
            append(CMAKE_SHARED_LINKER_FLAGS "-Wl,--as-needed")
        endif()
    endif()
endmacro()

if(WIN32 AND ${CMAKE_CXX_COMPILER_ID} STREQUAL MSVC)
    add_definitions(/Qpar)
    add_definitions(/openmp)
    set(OpenMP_CXX_FOUND true)
elseif(MSVC AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    append(CMAKE_C_FLAGS "-Xclang -fopenmp")
    append(CMAKE_CXX_FLAGS "-Xclang -fopenmp")
    set(OpenMP_CXX_FOUND true)
    list(APPEND EXTRA_SHARED_LIBS ${IOMP5LIB})
else()
    find_package(OpenMP)
    #newer version for findOpenMP (>= v. 3.9)
    if(CMAKE_VERSION VERSION_LESS "3.9" AND OPENMP_FOUND)
        if(${CMAKE_MAJOR_VERSION} VERSION_LESS "3" AND ${CMAKE_CXX_COMPILER_ID} STREQUAL "Intel")
            # Override FindOpenMP flags for Intel Compiler (otherwise deprecated)
            set(OpenMP_CXX_FLAGS "-fopenmp")
            set(OpenMP_C_FLAGS "-fopenmp")
        endif()
        set(OpenMP_C_FOUND true)
        set(OpenMP_CXX_FOUND true)
    endif()
    append_if(OpenMP_C_FOUND CMAKE_SRC_CCXX_FLAGS "${OpenMP_C_FLAGS}")
endif()

if (DNNL_CPU_THREADING_RUNTIME MATCHES "OMP")
    if (OpenMP_CXX_FOUND)
        append(CMAKE_TEST_CCXX_FLAGS "${OpenMP_CXX_FLAGS}")
        append(CMAKE_EXAMPLE_CCXX_FLAGS "${OpenMP_CXX_FLAGS}")
    else()
        message(${_omp_severity} "OpenMP library could not be found. "
            "Proceeding might lead to highly sub-optimal performance.")
        # Override CPU threading to sequential if allowed to proceed
        set(DNNL_CPU_THREADING_RUNTIME "SEQ")
    endif()
else()
    # Compilation happens with OpenMP to enable `#pragma omp simd`
    # but during linkage OpenMP dependency should be avoided
    forbid_link_compiler_omp_rt()
    return()
endif()
