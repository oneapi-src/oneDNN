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

file(GLOB_RECURSE SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/*.c
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
    )

# vanilla-x86 and non-x86 targets should ignore all x86 jit files
if(NOT TARGET_X86_JIT)
    message(STATUS "no x86 JIT: removing all '.*jit.*' source code")
    message(STATUS "\nbegin:\n${SOURCES}\n")
    #list(FILTER SOURCES EXCLUDE REGEX ".*(jit|xbyak).*") # not in older cmake
    list_filter_exclude(SOURCES ".*(jit|xbyak).*") # equiv. as macro
    message(STATUS "\nend:\n${SOURCES}\n")
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
if(TARGET_X86_JIT)
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/xbyak)
endif()

# Don't build sources from linux_perf direcotry if JIT Profiling isn't enabled.
# NOTE: On AArch64 builds. DNNL_ENABLE_JIT_PROFILING=OFF by default.
if(NOT DNNL_ENABLE_JIT_PROFILING)
    list(FILTER SOURCES EXCLUDE REGEX "linux_perf")
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
    # to make computations more stable and to align the jitted code
    # with the reference one use precise division and square root
    # by default
    file(GLOB FILES_REQUIRED_PREC_SQRT
        ${CMAKE_CURRENT_SOURCE_DIR}/*normalization*.cpp)
    file(GLOB FILES_REQUIRED_PREC_DIV
        ${CMAKE_CURRENT_SOURCE_DIR}/*resampling*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/*normalization*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/ref_rnn.cpp)
    if(WIN32)
        set_source_files_properties(${FILES_REQUIRED_PREC_SQRT}
            PROPERTIES COMPILE_FLAGS "/Qprec-sqrt")
        set_source_files_properties(${FILES_REQUIRED_PREC_DIV}
            PROPERTIES COMPILE_FLAGS "/Qprec-div")
    else()
        set_source_files_properties(${FILES_REQUIRED_PREC_SQRT}
            PROPERTIES COMPILE_FLAGS "-prec-sqrt")
        set_source_files_properties(${FILES_REQUIRED_PREC_DIV}
            PROPERTIES COMPILE_FLAGS "-prec-div")
    endif()
endif()

if(MSVC)
    file(GLOB FILES_REQUIRED_BIGOBJ
        ${CMAKE_CURRENT_SOURCE_DIR}/cpu_engine.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/cpu_reorder.cpp)
    set_source_files_properties(${FILES_REQUIRED_BIGOBJ}
        PROPERTIES COMPILE_FLAGS "/bigobj")
endif()

set(OBJ_LIB ${LIB_NAME}_cpu)
add_library(${OBJ_LIB} OBJECT ${SOURCES})
set_property(GLOBAL APPEND PROPERTY DNNL_LIB_DEPS
    $<TARGET_OBJECTS:${OBJ_LIB}>)
