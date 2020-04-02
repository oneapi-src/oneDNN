#===============================================================================
# Copyright 2016-2020 Intel Corporation
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

# Locate Intel MKL installation using MKLROOT
#===============================================================================

if(MKL_cmake_included)
    return()
endif()
set(MKL_cmake_included true)
include("cmake/utils.cmake")
include("cmake/options.cmake")

if (NOT _DNNL_USE_MKL)
    return()
endif()

function(detect_mkl LIBNAME)
    find_path(MKLINC mkl_cblas.h
        HINTS ${MKLROOT}/include $ENV{MKLROOT}/include)
    get_filename_component(__mkl_root "${MKLINC}" PATH)
    find_library(MKLLIB NAMES ${LIBNAME}
        PATHS ${__mkl_root}/lib ${__mkl_root}/lib/intel64
        NO_DEFAULT_PATH)

    if(WIN32)
        set(MKLREDIST ${__mkl_root}/../redist/)
        find_file(MKLDLL NAMES ${LIBNAME}.dll
            HINTS ${MKLREDIST}/mkl ${MKLREDIST}/intel64/mkl)
        if(NOT MKLDLL)
            return()
        endif()
    endif()

    if(WIN32)
        # Add paths to DLL to %PATH% on Windows
        get_filename_component(MKLDLLPATH "${MKLDLL}" PATH)
        append_to_windows_path_list(CTESTCONFIG_PATH "${MKLDLLPATH}")
        set(CTESTCONFIG_PATH "${CTESTCONFIG_PATH}" PARENT_SCOPE)
    endif()

    set(HAVE_MKL TRUE PARENT_SCOPE)
    set(MKLINC "${MKLINC}" PARENT_SCOPE)
    set(MKLLIB "${MKLLIB}" PARENT_SCOPE)
    set(MKLDLL "${MKLDLL}" PARENT_SCOPE)
endfunction()

detect_mkl("mkl_rt")

if(HAVE_MKL)
    list(APPEND EXTRA_SHARED_LIBS ${MKLLIB})
    add_definitions(-DUSE_MKL)
    include_directories(AFTER ${MKLINC})

    set(MSG "Intel MKL:")
    message(STATUS "${MSG} include ${MKLINC}")
    message(STATUS "${MSG} lib ${MKLLIB}")
    if(WIN32 AND MKLDLL)
        message(STATUS "${MSG} dll ${MKLDLL}")
    endif()
endif()
