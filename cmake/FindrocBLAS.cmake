#===============================================================================
# Copyright 2022-2023 Intel Corporation
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

find_package(HIP REQUIRED)
find_package(Threads REQUIRED)

# Prioritize ROCBLASROOT
list(APPEND rocblas_root_hints
            ${ROCBLASROOT}
            $ENV{ROCBLASROOT}
            "/opt/rocm"
            "/opt/rocm/rocblas")

find_path(
    rocBLAS_INCLUDE_DIR "rocblas.h"
    HINTS ${rocblas_root_hints}
    PATH_SUFFIXES include
)

find_library(
    rocBLAS_LIBRARY rocblas
    HINTS ${rocblas_root_hints}
    PATH_SUFFIXES lib
)

if(EXISTS "${rocBLAS_INCLUDE_DIR}/internal/rocblas-version.h")
    file(READ "${rocBLAS_INCLUDE_DIR}/internal/rocblas-version.h" rocBLAS_VERSION_CONTENT)

    string(REGEX MATCH "define ROCBLAS_VERSION_MAJOR +([0-9]+)" _ "${rocBLAS_VERSION_CONTENT}")
    set(rocBLAS_MAJOR_VERSION ${CMAKE_MATCH_1} CACHE INTERNAL "")

    string(REGEX MATCH "define ROCBLAS_VERSION_MINOR +([0-9]+)" _ "${rocBLAS_VERSION_CONTENT}")
    set(rocBLAS_MINOR_VERSION ${CMAKE_MATCH_1} CACHE INTERNAL "")

    string(REGEX MATCH "define ROCBLAS_VERSION_PATCH +([0-9]+)" _ "${rocBLAS_VERSION_CONTENT}")
    set(rocBLAS_PATCH_VERSION ${CMAKE_MATCH_1} CACHE INTERNAL "")

    set(rocBLAS_VERSION
        "${rocBLAS_MAJOR_VERSION}.${rocBLAS_MINOR_VERSION}.${rocBLAS_PATCH_VERSION}"
    )

    unset(rocBLAS_VERSION_CONTENT)
else()
    message(WARNING "rocBLAS version couldn't be identified.")
endif()

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(rocBLAS
    FOUND_VAR rocBLAS_FOUND
    REQUIRED_VARS
        rocBLAS_LIBRARY
        rocBLAS_INCLUDE_DIR
    VERSION_VAR rocBLAS_VERSION
)

if(rocBLAS_FOUND AND NOT TARGET rocBLAS::rocBLAS)
    add_library(rocBLAS::rocBLAS SHARED IMPORTED)
    set_target_properties(rocBLAS::rocBLAS PROPERTIES
        IMPORTED_LOCATION "${rocBLAS_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${rocBLAS_INCLUDE_DIR}"
        INTERFACE_LINK_LIBRARIES "HIP::HIP;Threads::Threads"
    )
endif()
