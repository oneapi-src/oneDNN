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

if(WIN32 AND ${CMAKE_CXX_COMPILER_ID} STREQUAL MSVC)
    add_definitions(/Qpar)
else()
    find_package(OpenMP)
    #newer version for findOpenMP (>= v. 3.9)
    if(CMAKE_VERSION VERSION_LESS "3.9")
        if(OPENMP_FOUND)
            if(${CMAKE_MAJOR_VERSION} VERSION_LESS "3" AND ${CMAKE_CXX_COMPILER_ID} STREQUAL "Intel")
                # Override FindOpenMP flags for Intel Compiler (otherwise deprecated)
                set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
                set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")
            else()
                set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
                set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
                set(OPENMP_FLAGS "${OpenMP_CXX_FLAGS}")
            endif()
        endif()
    else()
        if(OpenMP_C_FOUND)
            set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        endif()
        if(OpenMP_CXX_FOUND)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
            set(OPENMP_FLAGS "${OpenMP_CXX_FLAGS}")
        endif()
    endif()
endif()

# Do not link with compiler-native OpenMP library if MKL is present.
# Rationale: MKL comes with Intel OpenMP library which is compatible with all
# libraries shipped with compilers that MKL-DNN supports.
if(HAVE_MKL)
    # Remove OpenMP flags from compiler link line
    set(CMAKE_C_CREATE_SHARED_LIBRARY_FORBIDDEN_FLAGS ${OPENMP_FLAGS})
    set(CMAKE_CXX_CREATE_SHARED_LIBRARY_FORBIDDEN_FLAGS ${OPENMP_FLAGS})
    if (UNIX AND NOT APPLE)
        set(LINKER_FLAGS_IOMP "-Wl,--as-needed -liomp5 -Wl,--no-as-needed")
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${LINKER_FLAGS_IOMP}")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${LINKER_FLAGS_IOMP}")
    endif()
endif()

