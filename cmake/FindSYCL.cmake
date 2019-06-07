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

# The sycl_root_hint is not required to find the SYCL as SYCL will be found
# by the CC and CXX. However, it is used to set special OpenCL path. The list
# will be processed from left to right so SYCLROOT will always have priority
# against SYCL_BUNDLE_ROOT
list(APPEND sycl_root_hints
            ${SYCLROOT}
            $ENV{SYCLROOT}
            ${SYCL_BUNDLE_ROOT}
            $ENV{SYCL_BUNDLE_ROOT})

# This is used to prioritize Intel OpenCL against the system OpenCL
set(original_cmake_prefix_path ${CMAKE_PREFIX_PATH})
if(sycl_root_hints)
    list(INSERT CMAKE_PREFIX_PATH 0 ${sycl_root_hints})
endif()

include(FindPackageHandleStandardArgs)

find_package(IntelSYCL)

if(IntelSYCL_FOUND)
    set(SYCL_TARGET Intel::SYCL)
    set(SYCL_FLAGS ${INTEL_SYCL_FLAGS})
    set(SYCL_INCLUDE_DIRS ${INTEL_SYCL_INCLUDE_DIRS})
    set(SYCL_LIBRARIES ${INTEL_SYCL_LIBRARIES})
    set(MKLDNN_SYCL_INTEL true)
else()
    find_package(ComputeCpp)
    if(ComputeCpp_FOUND)
        set(SYCL_TARGET Codeplay::ComputeCpp)
        set(SYCL_FLAGS ${COMPUTECPP_FLAGS})
        set(SYCL_INCLUDE_DIRS ${COMPUTECPP_INCLUDE_DIRS})
        set(SYCL_LIBRARIES ${COMPUTECPP_LIBRARIES})
        set(MKLDNN_SYCL_COMPUTECPP true)
    endif()
endif()

find_package_handle_standard_args(
    SYCL REQUIRED_VARS SYCL_LIBRARIES SYCL_INCLUDE_DIRS)

if(SYCL_FOUND AND NOT TARGET SYCL::SYCL)
    add_library(SYCL::SYCL UNKNOWN IMPORTED)
    set_target_properties(SYCL::SYCL PROPERTIES
        INTERFACE_LINK_LIBRARIES ${SYCL_LIBRARIES})
endif()

# Reverting the CMAKE_PREFIX_PATH to its original state
set(CMAKE_PREFIX_PATH ${original_cmake_prefix_path})
