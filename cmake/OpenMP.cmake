#===============================================================================
# Copyright 2017 Intel Corporation
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

include("cmake/MKL.cmake")
include(CheckCXXCompilerFlag)

if(WIN32 AND ${CMAKE_CXX_COMPILER_ID} STREQUAL MSVC)
    add_definitions(/Qpar)
elseif(APPLE AND CMAKE_C_COMPILER_ID STREQUAL "Clang")
     CHECK_CXX_COMPILER_FLAG("-fopenmp" COMPLIER_SUPPORTS_FOPENMP)
     if(COMPLIER_SUPPORTS_FOPENMP)
        set(OpenMP_CXX_FLAGS "-fopenmp=libiomp5")
        set(OpenMP_C_FLAGS "-fopenmp=libiomp5") 
        set(OpenMP_C_FOUND true)
        set(OpenMP_CXX_FOUND true)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
	get_filename_component(MKLIOMP5LIBPATH ${MKLIOMP5LIB} PATH)
        set(CMAKE_MODULE_LINKER_FLAGS "-L${MKLIOMP5LIBPATH} -liomp5")
        set(CMAKE_SHARED_LINKER_FLAGS "-L${MKLIOMP5LIBPATH} -liomp5")
	set(CMAKE_EXE_LINKER_FLAGS "-L${MKLIOMP5LIBPATH} -liomp5")
        message(STATUS "openmp for llvm found -- ${MKLIOMP5LIBPATH}")
     endif()         
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
    if(OpenMP_C_FOUND)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    endif()
    if(OpenMP_CXX_FOUND)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    endif()
endif()

# Do not link with compiler-native OpenMP library if MKL is present.
# Rationale: MKL comes with Intel OpenMP library which is compatible with all
# libraries shipped with compilers that MKL-DNN supports.
if(HAVE_MKL AND NOT WIN32 AND NOT CMAKE_CXX_COMPILER_ID STREQUAL "Intel" AND NOT (APPLE AND CMAKE_C_COMPILER_ID STREQUAL "Clang"))
    if(OpenMP_C_FOUND)
        set(CMAKE_C_CREATE_SHARED_LIBRARY_FORBIDDEN_FLAGS ${OpenMP_C_FLAGS})
    endif()
    if(OpenMP_CXX_FOUND)
        set(CMAKE_CXX_CREATE_SHARED_LIBRARY_FORBIDDEN_FLAGS ${OpenMP_CXX_FLAGS})
    endif()
    list(APPEND EXTRA_LIBS ${MKLIOMP5LIB})
endif()

