#===============================================================================
# Copyright 2021 Intel Corporation
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

# Auxiliary build functions
#===============================================================================

if(dnnl_graph_utils_cmake_included)
    return()
endif()
set(dnnl_graph_utils_cmake_included true)

function(JOIN VALUES GLUE OUTPUT)
    string (REGEX REPLACE ";" "${GLUE}" _TMP_STR "${VALUES}")
    set (${OUTPUT} "${_TMP_STR}" PARENT_SCOPE)
endfunction()

# Common configuration for tests / test cases on Windows
function(maybe_configure_windows_test name kind)
    if(WIN32)
        string(REPLACE  ";" "\;" PATH "${CTESTCONFIG_PATH};$ENV{PATH}")
        set_property(${kind} ${name} PROPERTY ENVIRONMENT "PATH=${PATH}")
        if(TARGET ${name} AND CMAKE_GENERATOR MATCHES "Visual Studio")
            configure_file(${PROJECT_SOURCE_DIR}/cmake/template.vcxproj.user
                ${name}.vcxproj.user @ONLY)
        endif()
    endif()
endfunction()

# Add test (aka add_test), but possibly appends an emulator
function(add_graph_test name command)
    add_test(${name} ${DNNL_TARGET_EMULATOR} ${command} ${ARGN})
endfunction()

# Register new executable/test
#   name -- name of the executable
#   srcs -- list of source, if many must be enclosed with ""
#   test -- "test" to mark executable as a test, "" otherwise
#   arg4 -- (optional) list of extra library dependencies
function(register_exe name srcs test)
    add_executable(${name} ${srcs})
    target_link_libraries(${name} ${LIB_NAME} ${DNNL_GRAPH_EXTRA_SHARED_LIBS} ${ARGV3})
    if("x${test}" STREQUAL "xtest")
        add_graph_test(${name} ${name})
        maybe_configure_windows_test(${name} TEST)
    else()
        maybe_configure_windows_test(${name} TARGET)
    endif()
endfunction()

# Append to a variable
#   var = var + value
macro(append var value)
    set(${var} "${${var}} ${value}")
endmacro()

# Conditionally append
#   if (cond) var = var + value
macro(append_if condition var value)
    if (${condition})
        append(${var} "${value}")
    endif()
endmacro()

# search all sub-directories under current directory.
macro(listsubdir res curdir)
  file(GLOB children RELATIVE ${curdir} ${curdir}/*)
  set(dirlist "")
  foreach(child ${children})
    if(IS_DIRECTORY ${curdir}/${child})
        set(dirlist ${dirlist} ${child})
    endif()
  endforeach()
  set(${res} ${dirlist})
endmacro()

function(target_link_libraries_build target list)
    # Foreach is required for compatibility with 2.8.11 ways
    foreach(lib ${list})
        target_link_libraries(${target} LINK_PUBLIC
            "$<BUILD_INTERFACE:${lib}>")
    endforeach(lib)
endfunction()

function(target_link_libraries_install target list)
    # Foreach is required for compatibility with 2.8.11 ways
    foreach(lib ${list})
        get_filename_component(base "${lib}" NAME)
        target_link_libraries(${target} LINK_PUBLIC
            "$<INSTALL_INTERFACE:${base}>")
    endforeach(lib)
endfunction()
