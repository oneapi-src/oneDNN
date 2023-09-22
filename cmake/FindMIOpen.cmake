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
find_package(rocBLAS REQUIRED)

# Rely on the standard CMake config for amd_comgr as it doesn't add redundant
# dependencies.
find_package(amd_comgr REQUIRED CONFIG
    HINTS ${COMGRROOT}/lib/cmake $ENV{COMGRROOT}/lib/cmake /opt/rocm/lib/cmake
)

# amd_comgr target adds "${COMGRROOT}/include` directory that may contain
# OpenCL headers causing conflicts with OpenCL headers from the compiler
# hence remove that path.
set_target_properties(amd_comgr PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "")

# XXX: find a way to propagate it via target properties.
list(APPEND EXTRA_SHARED_LIBS amd_comgr)

# Prioritize MIOPENROOT
list(APPEND miopen_root_hints
            ${MIOPENROOT}
            $ENV{MIOPENROOT}
            "/opt/rocm"
            "/opt/rocm/miopen"
)

find_path(
    MIOpen_INCLUDE_DIR "miopen/miopen.h"
    HINTS ${miopen_root_hints}
    PATH_SUFFIXES include
)

find_library(
    MIOpen_LIBRARY MIOpen
    HINTS ${miopen_root_hints}
    PATH_SUFFIXES lib
)

if(EXISTS "${MIOpen_INCLUDE_DIR}/miopen/version.h")
    file(READ "${MIOpen_INCLUDE_DIR}/miopen/version.h" MIOpen_VERSION_CONTENT)

    string(REGEX MATCH "define MIOPEN_VERSION_MAJOR +([0-9]+)" _ "${MIOpen_VERSION_CONTENT}")
    set(MIOpen_MAJOR_VERSION ${CMAKE_MATCH_1} CACHE INTERNAL "")

    string(REGEX MATCH "define MIOPEN_VERSION_MINOR +([0-9]+)" _ "${MIOpen_VERSION_CONTENT}")
    set(MIOpen_MINOR_VERSION ${CMAKE_MATCH_1} CACHE INTERNAL "")

    string(REGEX MATCH "define MIOPEN_VERSION_PATCH +([0-9]+)" _ "${MIOpen_VERSION_CONTENT}")
    set(MIOpen_PATCH_VERSION ${CMAKE_MATCH_1} CACHE INTERNAL "")

    set(MIOpen_VERSION
        "${MIOpen_MAJOR_VERSION}.${MIOpen_MINOR_VERSION}.${MIOpen_PATCH_VERSION}"
    )

    unset(MIOpen_VERSION_CONTENT)
else()
    message(WARNING "MIOpen version couldn't be identified.")
endif()

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(MIOpen
    FOUND_VAR MIOpen_FOUND
    REQUIRED_VARS
        MIOpen_LIBRARY
        MIOpen_INCLUDE_DIR
    VERSION_VAR MIOpen_VERSION
)

if(MIOpen_FOUND AND NOT TARGET MIOpen::MIOpen)
    add_library(MIOpen::MIOpen SHARED IMPORTED)
    set_target_properties(MIOpen::MIOpen PROPERTIES
        IMPORTED_LOCATION "${MIOpen_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${MIOpen_INCLUDE_DIR}"
        INTERFACE_LINK_LIBRARIES "HIP::HIP;rocBLAS::rocBLAS;amd_comgr;Threads::Threads"
    )
endif()
