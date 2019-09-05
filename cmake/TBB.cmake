#===============================================================================
# Copyright 2018 Intel Corporation
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

# Manage TBB-related compiler flags
#===============================================================================

if(TBB_cmake_included)
    return()
endif()
set(TBB_cmake_included true)
include("cmake/Threading.cmake")

if(DNNL_CPU_RUNTIME STREQUAL "SYCL")
    if(NOT TBBROOT AND NOT DEFINED ENV{TBBROOT})
        return()
    endif()
elseif(NOT DNNL_CPU_THREADING_RUNTIME STREQUAL "TBB")
    return()
endif()

if(WIN32)
    find_package(TBB REQUIRED tbb HINTS cmake/win)
elseif(APPLE)
    find_package(TBB REQUIRED tbb HINTS cmake/mac)
elseif(UNIX)
    find_package(TBB REQUIRED tbb HINTS cmake/lnx)
endif()

include_directories(${TBB_INCLUDE_DIRS})

# XXX: workaround for SYCL. SYCL "unbundles" tbb.lib and loses its abosulte path
if(DNNL_SYCL_INTEL)
    get_target_property(tbb_lib_path TBB::tbb IMPORTED_LOCATION_RELEASE)
    get_filename_component(tbb_lib_dir "${tbb_lib_path}" PATH)
    link_directories(${tbb_lib_dir})
endif()

# XXX: this is to make "ctest" working out-of-the-box with TBB
string(REPLACE "/lib/" "/redist/" tbb_redist_dir "${tbb_lib_dir}")
append_to_windows_path_list(CTESTCONFIG_PATH "${tbb_redist_dir}")

list(APPEND EXTRA_SHARED_LIBS ${TBB_IMPORTED_TARGETS})

message(STATUS "Intel(R) TBB: ${TBBROOT}")
