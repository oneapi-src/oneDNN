#===============================================================================
# Copyright 2021 Intel Corporation
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

# Manage Threadpool-related compiler flags
# Leveraged from oneDNN project
# https://github.com/oneapi-src/oneDNN/blob/master/cmake/Threadpool.cmake
#===============================================================================

if(dnnl_graph_Threadpool_cmake_included)
    return()
endif()
set(dnnl_graph_Threadpool_cmake_included true)

if("${DNNL_GRAPH_CPU_RUNTIME}" STREQUAL "THREADPOOL")
    if("${_DNNL_GRAPH_TEST_THREADPOOL_IMPL}" STREQUAL "STANDALONE")
        message(STATUS "Threadpool testing: standalone")
    endif()

    add_definitions(-DDNNL_GRAPH_TEST_THREADPOOL_USE_${_DNNL_GRAPH_TEST_THREADPOOL_IMPL})
endif()

