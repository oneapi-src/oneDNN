#===============================================================================
# Copyright 2019-2020 Intel Corporation
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

# Manage OpenCL-related compiler flags
#===============================================================================

if(OpenCL_cmake_included)
    return()
endif()
set(OpenCL_cmake_included true)

if(DNNL_GPU_RUNTIME STREQUAL "OCL")
    message(STATUS "GPU support is enabled (OpenCL)")
else()
    return()
endif()

find_package(OpenCL REQUIRED)

include_directories(${OpenCL_INCLUDE_DIRS})
list(APPEND EXTRA_SHARED_LIBS OpenCL::OpenCL)

# Specify OpenCL version to avoid warnings
add_definitions(-DCL_TARGET_OPENCL_VERSION=220)
