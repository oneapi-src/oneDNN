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

find_package(SYCL REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SYCL_FLAGS}")

include_directories(${SYCL_INCLUDE_DIRS} ${OpenCL_INCLUDE_DIRS})
list(APPEND EXTRA_SHARED_LIBS ${SYCL_LIBRARIES} ${OpenCL_LIBRARIES})

if(MKLDNN_SYCL_INTEL)
    # Specify OpenCL version to avoid warnings
    add_definitions(-DCL_TARGET_OPENCL_VERSION=220)
endif()
