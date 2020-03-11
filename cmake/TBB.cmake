#===============================================================================
# Copyright 2018-2020 Intel Corporation
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

if(NOT DNNL_CPU_THREADING_RUNTIME STREQUAL "TBB")
    return()
endif()

find_package_tbb(REQUIRED)
if(TBB_FOUND)
    include_directories(${_tbb_include_dirs})
    list(APPEND EXTRA_SHARED_LIBS ${TBB_IMPORTED_TARGETS})

    # Print TBB location
    get_filename_component(_tbb_root "${_tbb_include_dirs}" PATH)
    get_filename_component(_tbb_root "${_tbb_root}" ABSOLUTE)
    message(STATUS "TBB: ${_tbb_root}")

    unset(_tbb_include_dirs)
    unset(_tbb_root)
endif()

