#===============================================================================
# Copyright 2016-2017 Intel Corporation
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

cmake_minimum_required(VERSION 2.8)

# version handling is done with old style
if(POLICY CMP0048)
    cmake_policy(SET CMP0048 OLD)
    cmake_policy(SET CMP0054 OLD)
endif()

set(PROJECT_NAME "Intel(R) MKL-DNN")
set(PROJECT_FULL_NAME "Intel(R) Math Kernel Library for Deep Neural Networks (Intel(R) MKL-DNN)")
set(PROJECT_VERSION "0.10")

set(LIB_NAME mkldnn)

project(${PROJECT_NAME} C CXX)

if("${CMAKE_BUILD_TYPE}" STREQUAL "")
    message(STATUS "CMAKE_BUILD_TYPE is unset, defaulting to Release")
    set(CMAKE_BUILD_TYPE "Release")
endif()

include("cmake/MKL.cmake")
include("cmake/Doxygen.cmake")

# sdl options
if(UNIX OR APPLE)
    set(CMAKE_CCXX_FLAGS "${CMAKE_CCXX_FLAGS} -fPIC -Wformat -Wformat-security")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -D_FORTIFY_SOURCE=2")
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -D_FORTIFY_SOURCE=2")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pie")
    if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -z noexecstack -z relro -z now")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -z noexecstack -z relro -z now")
        if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.9)
            set(CMAKE_CCXX_FLAGS "${CMAKE_CCXX_FLAGS} -fstack-protector-all")
        else()
            set(CMAKE_CCXX_FLAGS "${CMAKE_CCXX_FLAGS} -fstack-protector-strong")
        endif()
    elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
        set(CMAKE_CCXX_FLAGS "${CMAKE_CCXX_FLAGS} -fstack-protector-all")
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-z,noexecstack -Wl,-z,relro -Wl,-z,now")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-z,noexecstack -Wl,-z,relro -Wl,-z,now")
    elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
        # workaround for Intel Compiler 16.0 that produces error caused by pragma omp simd collapse(..)
        if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "17.0")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -diag-disable:13379")
        endif()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fstack-protector")
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -z noexecstack -z relro -z now")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -z noexecstack -z relro -z now")
    endif()
endif()

# UNIT8_MAX like macros are part of C99 standard and not C++ (see C99 standard 7.18.2 and 7.18.4)
add_definitions(-D__STDC_LIMIT_MACROS -D__STDC_CONSTANT_MACROS)

if(WIN32)
    set(USERCONFIG_PLATFORM "x64")
    add_definitions(-DMKLDNN_DLL -DMKLDNN_DLL_EXPORTS)
    add_definitions(/MP)
elseif(UNIX OR APPLE)
    set(CMAKE_CCXX_FLAGS "${CMAKE_CCXX_FLAGS} ${CCXX_WARN_FLAGS} -DMKLDNN_DLL -DMKLDNN_DLL_EXPORTS -fvisibility=internal")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${CMAKE_CCXX_FLAGS} -std=c99")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CCXX_FLAGS} -std=c++11 -fvisibility-inlines-hidden")
    set(CCXX_WARN_FLAGS "-Wall -Werror -Wno-unknown-pragmas")
    if(${CMAKE_CXX_COMPILER_ID} STREQUAL "Intel")
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -xHOST")
    endif()
endif()

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

if(HAVE_MKL)
    set(CMAKE_C_CREATE_SHARED_LIBRARY_FORBIDDEN_FLAGS ${OPENMP_FLAGS})
    set(CMAKE_CXX_CREATE_SHARED_LIBRARY_FORBIDDEN_FLAGS ${OPENMP_FLAGS})
    if (NOT WIN32)
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -liomp5 -Wl,--as-needed")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -liomp5 -Wl,--as-needed")
    endif()
endif()

enable_testing()

include_directories(include)

add_subdirectory(src)
add_subdirectory(examples)
add_subdirectory(tests)

install(FILES LICENSE DESTINATION share/doc/${LIB_NAME})
