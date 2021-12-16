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

# Build oneDNN primitive library

if(build_onednn_cmake_included)
    return()
endif()
set(build_onednn_cmake_included true)

set(DNNL_GPU_RUNTIME ${DNNL_GRAPH_GPU_RUNTIME} CACHE INTERNAL "" FORCE)
set(DNNL_CPU_RUNTIME ${DNNL_GRAPH_CPU_RUNTIME} CACHE INTERNAL "" FORCE)
set(DNNL_BUILD_TESTS OFF CACHE INTERNAL "" FORCE)
set(DNNL_BUILD_EXAMPLES OFF CACHE INTERNAL "" FORCE)
set(DNNL_ARCH_OPT_FLAGS "" CACHE INTERNAL "" FORCE)
set(DNNL_ENABLE_CONCURRENT_EXEC ON CACHE INTERNAL "" FORCE)
set(DNNL_ENABLE_PRIMITIVE_CACHE ON CACHE INTERNAL "" FORCE)

if(DNNL_GRAPH_LIBRARY_TYPE STREQUAL "STATIC" OR DNNL_GRAPH_LIBRARY_TYPE STREQUAL "SDL")
    set(DNNL_LIBRARY_TYPE STATIC CACHE INTERNAL "" FORCE)
    list(APPEND DNNL_GRAPH_EXTRA_STATIC_LIBS dnnl)
else()
    set(DNNL_LIBRARY_TYPE ${DNNL_GRAPH_LIBRARY_TYPE} CACHE INTERNAL "" FORCE)
    list(APPEND DNNL_GRAPH_EXTRA_SHARED_LIBS dnnl)
endif()

function(build_onednn)
    # Let SYCL to choose the C++ standard it needs.
    if(NOT (DNNL_GRAPH_WITH_SYCL OR CMAKE_BASE_NAME STREQUAL "icx" OR CMAKE_BASE_NAME STREQUAL "icpx"))
        set(CMAKE_CXX_STANDARD 11)
    endif()
    add_subdirectory(third_party/oneDNN)
endfunction()

build_onednn()
