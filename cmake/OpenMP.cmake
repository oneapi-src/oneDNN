#===============================================================================
# Copyright 2017-2024 Intel Corporation
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

macro(set_openmp_values_for_old_cmake)
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
endmacro()

if(DNNL_DPCPP_HOST_COMPILER STREQUAL "DEFAULT")
    # XXX: workaround: when -fsycl is specified the compiler doesn't define
    # _OPENMP macro causing `find_package(OpenMP)` to fail.
    # Use -fno-sycl option to disable SYCL. The rationale: dpcpp driver sets
    # the -fsycl option by default so it has to be explicitly disabled.
    set(_omp_original_cmake_cxx_flags "${CMAKE_CXX_FLAGS}")
    string(REGEX REPLACE "-fsycl" "-fno-sycl" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

    find_package(OpenMP)
    set_openmp_values_for_old_cmake()

    set(CMAKE_CXX_FLAGS "${_omp_original_cmake_cxx_flags}")
endif()

# special case for clang-cl (not recognized by cmake up to 3.17)
if(NOT OpenMP_CXX_FOUND AND MSVC AND CMAKE_CXX_COMPILER_ID MATCHES "(Clang|IntelLLVM)")
    # clang-cl and icx will fall under this condition
    # CAVEAT: undocumented variable, may be inappropriate
    if(CMAKE_BASE_NAME STREQUAL "icx")
        # XXX: Use `-Xclang --dependent-lib=libiomp5md` to workaround an issue
        # with linking OpenMP on Windows.
        # The ICX driver doesn't link OpenMP library even if `/Qopenmp`
        # was specified.
        set(OpenMP_FLAGS "/Qopenmp -Xclang --dependent-lib=libiomp5md")
    else()
        if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "10.0")
            # version < 10 can't pass cl-style `/openmp` flag
            set(OpenMP_FLAGS "-Xclang -fopenmp")
            # ... and requires explicit linking against omp library
            set(OpenMP_CXX_LIBRARIES "libomp.lib")
        endif()
    endif()
    set(OpenMP_C_FLAGS ${OpenMP_FLAGS})
    set(OpenMP_CXX_FLAGS ${OpenMP_FLAGS})
    set(OpenMP_CXX_FOUND true)
endif()

# add flags unconditionally to always utilize openmp-simd for any threading runtime
if(OpenMP_C_FOUND)
    append(CMAKE_C_FLAGS ${OpenMP_C_FLAGS})
endif()

if(OpenMP_CXX_FOUND)
    append(CMAKE_CXX_FLAGS ${OpenMP_CXX_FLAGS})
endif()

if(DNNL_CPU_THREADING_RUNTIME MATCHES "OMP")
    if(DPCPP_HOST_COMPILER_KIND STREQUAL "")
        message(FATAL_ERROR "DPCPP_HOST_COMPILER_KIND is undefined. Please make sure that a host compiler identification is performed before this point.")
    endif()

    if(DNNL_WITH_SYCL AND DPCPP_HOST_COMPILER_KIND STREQUAL "GNU")
        # Tell DPCPP compiler to link against libgomp. By default, it links
        # against libiomp5
        append(CMAKE_SHARED_LINKER_FLAGS "-fopenmp=libgomp")
        append(CMAKE_EXE_LINKER_FLAGS "-fopenmp=libgomp")
    elseif(OpenMP_CXX_FOUND)
        if(MSVC AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
            list(APPEND EXTRA_SHARED_LIBS ${OpenMP_CXX_LIBRARIES})
        endif()
    else()
        message(${_omp_severity} "OpenMP library could not be found. "
            "Proceeding might lead to highly sub-optimal performance.")
        # Override CPU threading to sequential if allowed to proceed
        set(DNNL_CPU_THREADING_RUNTIME "SEQ")
    endif()
else()
    # Compilation happens with OpenMP to enable `#pragma omp simd`
    # but during shared objects and executables linkage OpenMP dependency should
    # be avoided.
    if (NOT WIN32 AND NOT APPLE)
        append(CMAKE_SHARED_LINKER_FLAGS "-Wl,--as-needed")
        append(CMAKE_EXE_LINKER_FLAGS "-Wl,--as-needed")
    endif()
endif()
