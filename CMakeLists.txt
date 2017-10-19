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

include("cmake/SDL.cmake")
include("cmake/OpenMP.cmake")
include("cmake/MKL.cmake")
include("cmake/Doxygen.cmake")

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


enable_testing()

include_directories(include)

add_subdirectory(src)
add_subdirectory(examples)
add_subdirectory(tests)

install(FILES LICENSE DESTINATION share/doc/${LIB_NAME})
