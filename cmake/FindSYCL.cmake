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
            $ENV{DPCPP_ROOT}/compiler/latest/linux
            ${SYCL_BUNDLE_ROOT}
            $ENV{SYCL_BUNDLE_ROOT})

# This is used to prioritize Intel OpenCL against the system OpenCL
set(original_cmake_prefix_path ${CMAKE_PREFIX_PATH})
if(sycl_root_hints)
    list(INSERT CMAKE_PREFIX_PATH 0 ${sycl_root_hints})
endif()

# XXX: workaround to use OpenCL from DPC++ builds
if(DEFINED ENV{DPCPP_ROOT})
    list(INSERT CMAKE_PREFIX_PATH 0
        $ENV{DPCPP_ROOT}/compiler/latest/linux/lib/clang/9.0.0)
endif()

include(FindPackageHandleStandardArgs)

find_package(IntelSYCL)

if(IntelSYCL_FOUND)
    set(SYCL_TARGET Intel::SYCL)
    set(SYCL_FLAGS ${INTEL_SYCL_FLAGS})
    set(SYCL_INCLUDE_DIRS ${INTEL_SYCL_INCLUDE_DIRS})
    set(SYCL_LIBRARIES ${INTEL_SYCL_LIBRARIES})
    set(DNNL_SYCL_INTEL true CACHE INTERNAL "" FORCE)
else()
    find_package(ComputeCpp)
    if(ComputeCpp_FOUND)
        set(SYCL_TARGET ComputeCpp::ComputeCpp)
        set(SYCL_FLAGS ${ComputeCpp_FLAGS})
        set(SYCL_INCLUDE_DIRS ${ComputeCpp_INCLUDE_DIRS})
        set(SYCL_LIBRARIES ${ComputeCpp_LIBRARIES})
        set(DNNL_SYCL_COMPUTECPP true CACHE INTERNAL "" FORCE)
    endif()
endif()

find_package_handle_standard_args(
    SYCL REQUIRED_VARS SYCL_LIBRARIES SYCL_INCLUDE_DIRS)

if(SYCL_FOUND AND NOT TARGET SYCL::SYCL)
    add_library(SYCL::SYCL UNKNOWN IMPORTED)

    get_target_property(imp_libs
        ${SYCL_TARGET} IMPORTED_LINK_INTERFACE_LIBRARIES)
    get_target_property(imp_location
        ${SYCL_TARGET} IMPORTED_LOCATION)
    get_target_property(imp_location_debug
        ${SYCL_TARGET} IMPORTED_LOCATION_DEBUG)
    get_target_property(imp_include_dirs
        ${SYCL_TARGET} INTERFACE_INCLUDE_DIRECTORIES)

    set_target_properties(SYCL::SYCL PROPERTIES
        IMPORTED_LINK_INTERFACE_LIBRARIES "${imp_libs}"
        IMPORTED_LOCATION "${imp_location}"
        IMPORTED_LOCATION_DEBUG "${imp_location_debug}"
        INTERFACE_INCLUDE_DIRECTORIES "${imp_include_dirs}")
endif()

# Reverting the CMAKE_PREFIX_PATH to its original state
set(CMAKE_PREFIX_PATH ${original_cmake_prefix_path})
