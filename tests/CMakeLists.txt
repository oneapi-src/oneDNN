#===============================================================================
# Copyright 2020-2022 Intel Corporation
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

if(NOT DNNL_GRAPH_BUILD_TESTS)
    return()
endif()

# walkaround for icx cannot init constexpr with dllimport symbols(detail::handle)
# also, suppose it's unnecessary for users to link the library with such flag
if (DNNL_GRAPH_LIBRARY_TYPE STREQUAL "SHARED" OR DNNL_GRAPH_LIBRARY_TYPE STREQUAL "SDL")
    remove_definitions(-DDNNL_GRAPH_DLL)
endif()

# propagate common flags
append(CMAKE_CXX_FLAGS "${DNNL_GRAPH_COMMON_COMPILATION_FLAGS}")
append(CMAKE_C_FLAGS "${DNNL_GRAPH_COMMON_COMPILATION_FLAGS}")

set(CMAKE_TEST_CCXX_NOWARN_FLAGS)

# propagate no warning flags
append(CMAKE_TEST_CCXX_NOWARN_FLAGS "${CMAKE_CCXX_NOWARN_FLAGS}")

if(WIN32 AND NOT MINGW)
    if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
        # 2415: unused variable
        append(CMAKE_TEST_CCXX_NOWARN_FLAGS "/Qdiag-disable:2415")
    elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
        # c4244: conversion with possible loss of data
        # c4996: unsafe / deprecated functions
        append(CMAKE_TEST_CCXX_NOWARN_FLAGS "/wd4996 /wd4244" "/wd4267")
    endif()
endif()

# propagate TEST specific flags
append(CMAKE_C_FLAGS "${CMAKE_TEST_CCXX_FLAGS}")
append(CMAKE_CXX_FLAGS "${CMAKE_TEST_CCXX_FLAGS}")

# propagate TEST specific flags
append(CMAKE_C_FLAGS "${CMAKE_TEST_CCXX_NOWARN_FLAGS}")
append(CMAKE_CXX_FLAGS "${CMAKE_TEST_CCXX_NOWARN_FLAGS}")

if(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
    append_if(WIN32 CMAKE_CXX_FLAGS "/fp:precise")
    append_if(UNIX  CMAKE_CXX_FLAGS "-fp-model precise")
    append(CMAKE_CXX_FLAGS "-Wno-unknown-pragmas")
    append(CMAKE_C_FLAGS "-Wno-unknown-pragmas")
endif()

if(WIN32)
    append(CMAKE_CXX_FLAGS "-DNOMINMAX") # to allow std::max on Windows with parentheses
endif()

if(DNNL_GRAPH_CPU_RUNTIME STREQUAL "THREADPOOL")
    # Only enable this for threadpool runtime to avoid complaints about empty
    # object files.
    set(GRAPH_TEST_THREAD ${CMAKE_CURRENT_SOURCE_DIR}/test_thread.cpp)
endif()
set(TEST_THREAD ${PROJECT_SOURCE_DIR}/third_party/oneDNN/tests/test_thread.cpp)

set(TEST_ALLOCATOR ${CMAKE_CURRENT_SOURCE_DIR}/test_allocator.cpp)

if(DNNL_GRAPH_WITH_SYCL AND DNNL_GRAPH_CPU_RUNTIME STREQUAL "NONE")
    set(DNNL_WITH_SYCL true)
    list(INSERT CMAKE_MODULE_PATH 0 ${PROJECT_SOURCE_DIR}/third_party/oneDNN/cmake)
    find_package_tbb(REQUIRED)
    list(REMOVE_AT CMAKE_MODULE_PATH 0)
    handle_tbb_target()
    unset(DNNL_WITH_SYCL)
    list(APPEND DNNL_GRAPH_EXTRA_SHARED_LIBS ${TBB_IMPORTED_TARGETS})
endif()

# add_subdirectory(cpp)
add_subdirectory(benchdnn)
add_subdirectory(benchdnn_graph)
add_subdirectory(demo)
