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

# XXX: Intel(R) oneAPI DPC++ Compiler doesn't support SEH with -fsycl option
if(WIN32 AND DNNL_GRAPH_WITH_SYCL)
    add_definitions(-DGTEST_HAS_SEH=0)
endif()

set(gtest_hide_internal_symbols ON CACHE BOOL "" FORCE)
add_subdirectory(${CMAKE_SOURCE_DIR}/tests/gtest
    ${CMAKE_BINARY_DIR}/tests/gtest)

enable_testing()

if(UNIX)
    if(APPLE)
        # TODO(wuxun): now it will fail to build with macOS in Debug mode, need
        # to enable again once the root cause is found
        #set(DNNL_GRAPH_TEST_LINK_FLAGS "-Wl,-fatal_warnings")
    else()
        set(DNNL_GRAPH_TEST_LINK_FLAGS "-Wl,--fatal-warnings")
    endif()
endif()

add_subdirectory(api)
add_subdirectory(unit)
