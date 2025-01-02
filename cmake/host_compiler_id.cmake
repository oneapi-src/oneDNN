#===============================================================================
# Copyright 2024-2025 Intel Corporation
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

if(host_compiler_id_cmake_included)
    return()
endif()
set(host_compiler_id_cmake_included true)

# There is nothing to identify for the default host compiler.
if (DNNL_DPCPP_HOST_COMPILER STREQUAL "DEFAULT")
    set(DPCPP_HOST_COMPILER ${DNNL_DPCPP_HOST_COMPILER})
    set(DPCPP_HOST_COMPILER_KIND ${DNNL_DPCPP_HOST_COMPILER})
    set(DPCPP_HOST_COMPILER_MAJOR_VER 0)
    set(DPCPP_HOST_COMPILER_MINOR_VER 0)
    return()
endif()

if(NOT DNNL_WITH_SYCL)
    message(FATAL_ERROR "DNNL_DPCPP_HOST_COMPILER is supported only for DPCPP runtime")
endif()

if(DNNL_SYCL_CUDA)
    message(FATAL_ERROR "DNNL_DPCPP_HOST_COMPILER options is not supported for NVIDIA.")
endif()

if(DNNL_SYCL_HIP)
    message(FATAL_ERROR "DNNL_DPCPP_HOST_COMPILER options is not supported for AMD.")
endif()

if(WIN32)
    message(FATAL_ERROR "${DNNL_DPCPP_HOST_COMPILER} cannot be used on Windows")
endif()

# The code in this file does the following:
# - Checks that the provided host compiler exists
# - Identifies the host compiler kind
# - Performs nessasary checks (e.g. check for minimum version)
#
# This code also fills out the following variables that can be
# used throughout the build system:
# - DPCPP_HOST_COMPILER: an absolute path to the host compiler executable
# - DPCPP_HOST_COMPILER_KIND: a host compiler kind (e.g. GNU)
# - DPCPP_HOST_COMPILER_MAJOR_VER: a major version of the host compiler
# - DPCPP_HOST_COMPILER_MINOR_VER: a minor version of the host compiler

find_program(DPCPP_HOST_COMPILER NAMES ${DNNL_DPCPP_HOST_COMPILER})
if(NOT DPCPP_HOST_COMPILER)
    message(FATAL_ERROR "${DNNL_DPCPP_HOST_COMPILER} host compiler not found")
else()
    message(STATUS "Host compiler: ${DPCPP_HOST_COMPILER}")
endif()

function(IDENTIFY_HOST_COMPILER_KIND)
    macro(TRY_IDENTIFY KIND)
        execute_process(COMMAND ${DPCPP_HOST_COMPILER} -c -DTRY_${KIND} ${PROJECT_SOURCE_DIR}/cmake/host_compiler_id.cpp -o ${PROJECT_BINARY_DIR}/host_compiler_id.o
                        RESULT_VARIABLE EXECUTE_STATUS ERROR_VARIABLE STDERR_MESSAGE)
        if("${EXECUTE_STATUS}" STREQUAL "0")
            set(DPCPP_HOST_COMPILER_KIND "${KIND}" PARENT_SCOPE)
            set(STDERR_MESSAGE "${STDERR_MESSAGE}" PARENT_SCOPE)
            return()
        endif()
    endmacro()

    TRY_IDENTIFY(GNU)
    TRY_IDENTIFY(CLANG)

    message(FATAL_ERROR "The provided host compiler could not be identified")
endfunction()

IDENTIFY_HOST_COMPILER_KIND()

message(STATUS "Host compiler kind: ${DPCPP_HOST_COMPILER_KIND}")

# Preprocessor prints out major and minor versions of the compiler when
# compiling host_compiler_id.cpp. Using the regex below to extract
# the versions from the preprocessor message.
string(REGEX MATCH "host compiler version: ([0-9]+\\.[0-9]+)" _ "${STDERR_MESSAGE}")
set(DPCPP_HOST_COMPILER_VER ${CMAKE_MATCH_1} CACHE INTERNAL "")

string(REPLACE "." ";" DPCPP_HOST_COMPILER_VER_LIST ${DPCPP_HOST_COMPILER_VER})
list(GET DPCPP_HOST_COMPILER_VER_LIST 0 DPCPP_HOST_COMPILER_MAJOR_VER)
list(GET DPCPP_HOST_COMPILER_VER_LIST 1 DPCPP_HOST_COMPILER_MINOR_VER)

message(STATUS "Host compiler version: ${DPCPP_HOST_COMPILER_MAJOR_VER}.${DPCPP_HOST_COMPILER_MINOR_VER}")

# Check the version of the provided host compiler.
if(DPCPP_HOST_COMPILER_KIND STREQUAL "GNU")
    if(DPCPP_HOST_COMPILER_MAJOR_VER LESS 8)
        message(FATAL_ERROR "The minimum version of ${DPCPP_HOST_COMPILER_KIND} host compiler is 8.0.")
    endif()
endif()

if(DPCPP_HOST_COMPILER_KIND STREQUAL "CLANG")
    if(DPCPP_HOST_COMPILER_MAJOR_VER LESS 11)
        message(FATAL_ERROR "The minimum version of ${DPCPP_HOST_COMPILER_KIND} host compiler is 11.0.")
    endif()
endif()
