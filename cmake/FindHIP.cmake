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

find_package(Threads REQUIRED)

# Prioritize HIPROOT
list(APPEND hip_root_hints
            ${HIPROOT}
            $ENV{HIPROOT}
            "/opt/rocm"
            "/opt/rocm/hip")

find_path(
    HIP_INCLUDE_DIR "hip/hip_runtime_api.h"
    HINTS ${hip_root_hints}
    PATH_SUFFIXES include
)

find_library(
    HIP_LIBRARY amdhip64
    HINTS ${hip_root_hints}
    PATH_SUFFIXES lib
)

if(EXISTS "${HIP_INCLUDE_DIR}/hip/hip_version.h")
    file(READ "${HIP_INCLUDE_DIR}/hip/hip_version.h" HIP_VERSION_CONTENT)

    string(REGEX MATCH "define HIP_VERSION_MAJOR +([0-9]+)" _ "${HIP_VERSION_CONTENT}")
    set(HIP_MAJOR_VERSION ${CMAKE_MATCH_1} CACHE INTERNAL "")

    string(REGEX MATCH "define HIP_VERSION_MINOR +([0-9]+)" _ "${HIP_VERSION_CONTENT}")
    set(HIP_MINOR_VERSION ${CMAKE_MATCH_1} CACHE INTERNAL "")

    string(REGEX MATCH "define HIP_VERSION_PATCH +([0-9]+)" _ "${HIP_VERSION_CONTENT}")
    set(HIP_PATCH_VERSION ${CMAKE_MATCH_1} CACHE INTERNAL "")

    set(HIP_VERSION
        "${HIP_MAJOR_VERSION}.${HIP_MINOR_VERSION}.${HIP_PATCH_VERSION}"
    )

    unset(HIP_VERSION_CONTENT)
else()
    message(WARNING "HIP version couldn't be identified.")
endif()

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(HIP
    FOUND_VAR HIP_FOUND
    REQUIRED_VARS
        HIP_LIBRARY
        HIP_INCLUDE_DIR
    VERSION_VAR HIP_VERSION
)

if(HIP_FOUND AND NOT TARGET HIP::HIP)
    add_library(HIP::HIP SHARED IMPORTED)
    set_target_properties(HIP::HIP PROPERTIES
        IMPORTED_LOCATION "${HIP_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${HIP_INCLUDE_DIR}"
        INTERFACE_LINK_LIBRARIES "Threads::Threads"
        INTERFACE_COMPILE_DEFINITIONS "__HIP_PLATFORM_AMD__=1"
    )
endif()
