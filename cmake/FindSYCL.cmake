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

include(FindPackageHandleStandardArgs)

find_package(DPCPP)

if(DPCPP_FOUND)
    set(SYCL_TARGET DPCPP::DPCPP)
    set(SYCL_FLAGS ${DPCPP_FLAGS})
    set(SYCL_INCLUDE_DIRS ${DPCPP_INCLUDE_DIRS})
    set(SYCL_LIBRARIES ${DPCPP_LIBRARIES})
    set(DNNL_SYCL_DPCPP true CACHE INTERNAL "" FORCE)
    if(LevelZero_FOUND)
        set(DNNL_WITH_LEVEL_ZERO true CACHE INTERNAL "" FORCE)
    endif()
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
