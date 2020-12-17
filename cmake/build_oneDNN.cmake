#===============================================================================
# Copyright 2020 Intel Corporation
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

# Build oneDNN primitive library

if(build_onednn_cmake_included)
    return()
endif()
set(build_onednn_cmake_included true)

if(DNNL_GRAPH_GPU_RUNTIME STREQUAL "DPCPP")
    include(cmake/FindDPCPP.cmake)
    set(DNNL_GRAPH_SYCL_LINK_FLAGS ${DNNL_GRAPH_SYCL_LINK_FLAGS} sycl)
    message(STATUS "DPCPP found. Compiling with SYCL support")
    set(DNNL_GRAPH_CPU_RUNTIME DPCPP CACHE INTERNAL "" FORCE)
    set(DNNL_CPU_RUNTIME DPCPP CACHE INTERNAL "" FORCE)
    set(DNNL_GPU_RUNTIME DPCPP CACHE INTERNAL "" FORCE)
    set(DNNL_GRAPH_WITH_SYCL true)
    add_definitions(-DDNNL_GRAPH_WITH_SYCL)
else()
    set(DNNL_CPU_RUNTIME ${DNNL_GRAPH_CPU_RUNTIME} CACHE INTERNAL "" FORCE)
    set(DNNL_GPU_RUNTIME NONE CACHE INTERNAL "" FORCE)
endif()

set(DNNL_BUILD_TESTS OFF CACHE INTERNAL "" FORCE)
set(DNNL_BUILD_EXAMPLES OFF CACHE INTERNAL "" FORCE)
set(DNNL_ARCH_OPT_FLAGS "" CACHE INTERNAL "" FORCE)
set(DNNL_ENABLE_CONCURRENT_EXEC ON CACHE INTERNAL "" FORCE)
set(DNNL_LIBRARY_TYPE STATIC CACHE INTERNAL "" FORCE)
set(DNNL_ENABLE_PRIMITIVE_CACHE ON CACHE INTERNAL "" FORCE)

function(build_onednn)
    set(CMAKE_CXX_STANDARD 11)
    add_subdirectory(third_party/oneDNN)
endfunction()

build_onednn()
