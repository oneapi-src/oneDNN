#===============================================================================
# Copyright 2020-2023 Intel Corporation
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

macro(exec_cmd_and_check CMD1 CMD2 CMD3 OUT)
    execute_process(COMMAND ${CMD1} ${CMD2} ${CMD3}
        RESULT_VARIABLE _sc_exit_code
        OUTPUT_VARIABLE _sc_cmd_out)
    if(NOT ${_sc_exit_code} STREQUAL 0)
        message(FATAL_ERROR "Failed with exit code ${_sc_exit_code}: ${CMD}")
    endif()
    set(${OUT} ${_sc_cmd_out})
endmacro()

macro(find_llvm)
    if(NOT ${SC_LLVM_VERSION} STREQUAL OFF)
        # if LLVM environment is already set in parent environment, do nothing
    elseif(LLVM_FOUND)
        # if LLVM is found by cmake, use it
        set(SC_LLVM_VERSION ${LLVM_VERSION_MAJOR})
        set(SC_LLVM_INCLUDE_PATH ${LLVM_INCLUDE_DIRS})
        set(SC_LLVM_LIB_NAME "")
        foreach(__libname ${LLVM_AVAILABLE_LIBS})
            # skip dynamic LLVM libraries
            if(NOT ${__libname} MATCHES "^(LLVM|LLVMRemarks|LLVMLTO|LTO|Remarks)$")
                get_target_property(__interface_lib ${__libname} INTERFACE_LINK_LIBRARIES)
                # skip targets that depend on dynamic LLVM libraries
                list(FIND __interface_lib "LLVM" _index)
                if (NOT ${_index} GREATER -1)
                    set(LIB_VAR "__LIB_${__libname}")
                    find_library(${LIB_VAR} NAMES ${__libname} HINTS ${LLVM_LIBRARY_DIRS})
                    if(${LIB_VAR})
                        list(APPEND SC_LLVM_LIB_NAME ${__libname})
                    endif()
                endif()
            endif()
        endforeach()
    elseif(SC_LLVM_CONFIG STREQUAL "CMAKE")
        find_package(LLVM CONFIG)
        if(NOT LLVM_FOUND)
            message(FATAL_ERROR "LLVM not found.")
        else()
            set(SC_LLVM_CONFIG_RETURN_STATIC ON)
            # LLVM-10+ has a bug in llvm_map_components_to_libnames(LLVM_LIBS all)
            # bug report at https://bugs.llvm.org/show_bug.cgi?id=47003
            if(SC_LLVM_TRY_DYN_LINK OR (NOT SC_LIBRARY_TYPE STREQUAL "STATIC"))
                # find if LLVM has shared library built
                list(FIND LLVM_AVAILABLE_LIBS "LLVM" _index)
                if (${_index} GREATER -1)
                    set(LLVM_LIBS LLVM)
                    set(SC_LLVM_CONFIG_RETURN_STATIC OFF)
                else()
                    set(LLVM_LIBS ${LLVM_AVAILABLE_LIBS})
                endif()
            else()
                set(LLVM_LIBS ${LLVM_AVAILABLE_LIBS})
            endif()
            set(SC_LLVM_VERSION ${LLVM_VERSION_MAJOR})
            set(SC_LLVM_INCLUDE_PATH ${LLVM_INCLUDE_DIRS})
            set(SC_LLVM_LIB_NAME ${LLVM_LIBS})
            message(STATUS "Link static LLVM=${SC_LLVM_CONFIG_RETURN_STATIC}")
        endif()
    else()
        if(SC_LLVM_CONFIG STREQUAL "AUTO")
            foreach(__sc_llvm_ver RANGE 16 10 -1)
                find_program(__sc_llvm_config llvm-config-${__sc_llvm_ver})
                if(__sc_llvm_config)
                    SET(SC_LLVM_CONFIG ${__sc_llvm_config})
                endif()
            endforeach()
            if(SC_LLVM_CONFIG STREQUAL "AUTO")
                find_program(__sc_llvm_config llvm-config)
                if(__sc_llvm_config)
                    SET(SC_LLVM_CONFIG ${__sc_llvm_config})
                endif()
                if(SC_LLVM_CONFIG STREQUAL "AUTO")
                    message(FATAL_ERROR "Failed to find llvm-config in AUTO mode. Please specify an llvm-config executable.")
                endif()
            endif()
        endif()
        message(STATUS "Finding LLVM using ${SC_LLVM_CONFIG}")
        set(__sc_llvm_link "--link-static")
        set(SC_LLVM_CONFIG_RETURN_STATIC ON)
        if(NOT MSVC)
            if(SC_LLVM_TRY_DYN_LINK OR (NOT SC_LIBRARY_TYPE STREQUAL "STATIC"))
                # try link with shared library
                execute_process(COMMAND ${SC_LLVM_CONFIG} "--libfiles" "--link-shared"
                    RESULT_VARIABLE _sc_exit_code
                    OUTPUT_VARIABLE _sc_cmd_out ERROR_QUIET)
                if(${_sc_exit_code} STREQUAL 0)
                    set(__sc_llvm_link "--link-shared")
                    set(SC_LLVM_CONFIG_RETURN_STATIC OFF)
                endif()
            else()
                execute_process(COMMAND ${SC_LLVM_CONFIG} "--libfiles" "--link-static"
                    RESULT_VARIABLE _sc_exit_code
                    OUTPUT_VARIABLE _sc_cmd_out ERROR_QUIET)
                if(NOT ${_sc_exit_code} STREQUAL 0)
                    set(__sc_llvm_link "--link-shared")
                    set(SC_LLVM_CONFIG_RETURN_STATIC OFF)
                endif()
            endif()
        endif()

        message(STATUS "Decided to link LLVM with ${__sc_llvm_link}")

        exec_cmd_and_check(${SC_LLVM_CONFIG} "--system-libs" "${__sc_llvm_link}"
            _sc_llvm_system_libs)
        exec_cmd_and_check(${SC_LLVM_CONFIG} "--libfiles"  "${__sc_llvm_link}"
            _sc_llvm_libs)

        # libs to link
        string(STRIP ${_sc_llvm_libs} _sc_llvm_libs)
        string(STRIP ${_sc_llvm_system_libs} _sc_llvm_system_libs)
        separate_arguments(_sc_llvm_libs)
        if(SC_LLVM_CONFIG_RETURN_STATIC)
            set(SC_LLVM_LIB_NAME "")
            foreach(__libname ${_sc_llvm_libs})
                if(EXISTS ${__libname})
                    list(APPEND SC_LLVM_LIB_NAME ${__libname})
                endif()
            endforeach()
        else()
            set(SC_LLVM_LIB_NAME ${_sc_llvm_libs})
        endif()
        set(SC_LLVM_LIB_NAME "${SC_LLVM_LIB_NAME} ${_sc_llvm_system_libs}")
        separate_arguments(SC_LLVM_LIB_NAME)

        # version
        exec_cmd_and_check(${SC_LLVM_CONFIG} "--version" ""
            _sc_llvm_version)
        string(REGEX MATCH "^([^.]+)\\.([^.])+\\.[^.]+.*$" _ ${_sc_llvm_version})
        set(SC_LLVM_VERSION ${CMAKE_MATCH_1})
        string(STRIP ${SC_LLVM_VERSION} SC_LLVM_VERSION)
        
        # include directory
        exec_cmd_and_check(${SC_LLVM_CONFIG} "--includedir" "" SC_LLVM_INCLUDE_PATH)
    endif()
    if(SC_LLVM_CONFIG_RETURN_STATIC)
        set(__sc_llvm_exclude "")
        foreach(__sc_llvm_lib ${SC_LLVM_LIB_NAME})
            get_filename_component(__sc_llvm_file ${__sc_llvm_lib} NAME)
            list(APPEND __sc_llvm_exclude ${__sc_llvm_file})
        endforeach()
        string(REPLACE ";" ":" SC_LLVM_LIB_EXCLUDE "${__sc_llvm_exclude}")
    endif()
endmacro()
