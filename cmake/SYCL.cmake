#===============================================================================
# Copyright 2019 Intel Corporation
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

if(SYCL_cmake_included)
    return()
endif()
set(SYCL_cmake_included true)

if(NOT DNNL_WITH_SYCL)
    return()
endif()

set(_computecpp_flags "-Wno-sycl-undef-func -no-serial-memop")
set(COMPUTECPP_USER_FLAGS "${_computecpp_flags} ${COMPUTECPP_USER_FLAGS}"
    CACHE STRING "")

find_package(SYCL REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SYCL_FLAGS}")

# XXX: OpenCL in SYCL bundle cannot be found by FindOpenCL due to the specific
# directory layout. This workaround ensures that local OpenCL SDK doesn't
# create any conflicts with SYCL headers.
if(WIN32 AND DNNL_SYCL_DPCPP)
    include_directories(${SYCL_INCLUDE_DIRS})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -idirafter \"${OpenCL_INCLUDE_DIRS}\"")
else()
    include_directories(${SYCL_INCLUDE_DIRS} ${OpenCL_INCLUDE_DIRS})
endif()

list(APPEND EXTRA_SHARED_LIBS SYCL::SYCL)

if(DNNL_SYCL_DPCPP)
    get_target_property(sycl_lib_path SYCL::SYCL IMPORTED_LOCATION)
    get_filename_component(sycl_lib_dir "${sycl_lib_path}" PATH)

    append_to_windows_path_list(CTESTCONFIG_PATH "${sycl_lib_dir}/../bin")

    # Specify OpenCL version to avoid warnings
    add_definitions(-DCL_TARGET_OPENCL_VERSION=220)

    # Use TBB library from SYCL bundle if it is there
    if(NOT TBBROOT)
        find_path(_tbbroot
            NAMES "include/tbb/tbb.h"
            PATHS "${sycl_lib_dir}/../../tbb"
                  "${sycl_lib_dir}/../../../tbb"
            PATH_SUFFIXES "latest"
        NO_DEFAULT_PATH)
        if(_tbbroot)
            set(TBBROOT "${_tbbroot}" CACHE STRING "" FORCE)
        endif()
    endif()
endif()
