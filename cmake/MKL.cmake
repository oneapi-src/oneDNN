#===============================================================================
# Copyright 2016-2017 Intel Corporation
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
# Locate Intel(R) MKL installation using MKLROOT or look in
# ${CMAKE_CURRENT_SOURCE_DIR}/external
#===============================================================================
function(detect_mkl LIBNAME)
    if(HAVE_MKL)
        return()
    endif()

    find_path(MKLINC mkl_cblas.h
        PATHS ${MKLROOT}/include $ENV{MKLROOT}/include)
    if(NOT MKLINC)
        file(GLOB_RECURSE MKLINC
                ${CMAKE_CURRENT_SOURCE_DIR}/external/*/mkl_cblas.h)
        if(MKLINC)
            list(LENGTH MKLINC MKLINCLEN)
            if(MKLINCLEN GREATER 1) # if user downloaded multiple external/,
                # then guess last one alphabetically is "latest" and warn
                list(SORT MKLINC)
                list(REVERSE MKLINC)
                message(STATUS "MKLINC found ${MKLINCLEN} files:")
                foreach(LOCN IN LISTS MKLINC)
                    message(STATUS "       ${LOCN}")
                endforeach()
                list(GET MKLINC 0 MKLINCLST)
                set(MKLINC "${MKLINCLST}")
                message(WARNING "MKLINC guessing... ${MKLINC}.  "
                    "Please check that above dir has the desired mkl_cblas.h")
            endif()
            get_filename_component(MKLINC ${MKLINC} PATH)
            set(MKLINC ${MKLINC} PARENT_SCOPE)
            message(STATUS "MKLINC (path) ${MKLINC}")
        endif()
    endif()

    get_filename_component(__mklinc_root "${MKLINC}" PATH)
    find_library(MKLLIB NAMES ${LIBNAME}
        PATHS   ${MKLROOT}/lib ${MKLROOT}/lib/intel64
                $ENV{MKLROOT}/lib $ENV{MKLROOT}/lib/intel64
                ${__mklinc_root}/lib ${__mklinc_root}/lib/intel64)
    if(MKLINC AND MKLLIB)
        set(HAVE_MKL TRUE PARENT_SCOPE)
        get_filename_component(MKLLIBPATH "${MKLLIB}" PATH)
        string(FIND "${MKLLIBPATH}" ${CMAKE_CURRENT_SOURCE_DIR}/external __idx)
        if(${__idx} EQUAL 0)
            install(PROGRAMS ${MKLLIB} ${MKLLIBPATH}/libiomp5.so
                    DESTINATION lib)
        endif()
    endif()
endfunction()

if(WIN32)
    detect_mkl("mklml")
    detect_mkl("mkl_rt")
elseif(UNIX)
    detect_mkl("libmklml_intel.so")
    detect_mkl("libmkl_rt.so")
endif()

set(FAIL_WITHOUT_MKL)

if(HAVE_MKL)
    add_definitions(-DUSE_MKL -DUSE_CBLAS)
    include_directories(AFTER ${MKLINC})
    list(APPEND mkldnn_LINKER_LIBS ${MKLLIB})
    message(STATUS "Intel(R) MKL found: include ${MKLINC}, lib ${MKLLIB}")
else()
    if(DEFINED ENV{FAIL_WITHOUT_MKL} OR DEFINED FAIL_WITHOUT_MKL)
        set(SEVERITY "FATAL_ERROR")
    else()
        set(SEVERITY "WARNING")
    endif()
    message(${SEVERITY} "Intel(R) MKL not found. Some performance features may not be "
        "available. Please run scripts/prepare_mkl.sh to download a minimal "
        "set of libraries or get a full version from "
        "https://software.intel.com/en-us/intel-mkl")
endif()
