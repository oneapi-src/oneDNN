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

file(GLOB_RECURSE SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/*.c
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
    )

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/xbyak
    )

# Exclude sources for JIT profiling when JIT profiling is disabled
if(NOT MKLDNN_ENABLE_JIT_PROFILING)
    set(sources_all "${SOURCES}")
    set(SOURCES)
    foreach(src ${sources_all})
        string(REGEX MATCH ".*jitprofiling.*" match ${src})
        if(NOT match)
            list(APPEND SOURCES ${src})
        endif()
    endforeach()
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
    # to make computations more stable and to align the jitted code
    # with the reference one use precise division and square root
    # by default
    file(GLOB FILES_REQUIRED_PREC_SQRT
        ${CMAKE_CURRENT_SOURCE_DIR}/*normalization*.cpp)
    file(GLOB FILES_REQUIRED_PREC_DIV
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
if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    file(GLOB FILES_REQUIRED_BIGOBJ
        ${CMAKE_CURRENT_SOURCE_DIR}/cpu_engine.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/cpu_reorder.cpp)
	if(WIN32)
		set_source_files_properties(${FILES_REQUIRED_BIGOBJ}
			PROPERTIES COMPILE_FLAGS "/bigobj")
	endif()
endif()

# XXX: Compilation with Intel(R) oneAPI DPC++ Compiler takes too long for JIT kernels
if(DNNL_SYCL_DPCPP)
    set(FILES_WITHNO_OPT)
    foreach(src in ${SOURCES})
        string(REGEX MATCH ".*jit.*kern.cpp" match ${src})
        if(match)
            list(APPEND FILES_WITHNO_OPT ${src})
        endif()
    endforeach()

    set_source_files_properties(${FILES_WITHNO_OPT}
        PROPERTIES COMPILE_FLAGS "-O0")
endif()

set(OBJ_LIB ${LIB_NAME}_cpu)
add_library(${OBJ_LIB} OBJECT ${SOURCES})
set(${LIB_NAME}_SUB_OBJS ${${LIB_NAME}_SUB_OBJS}
    $<TARGET_OBJECTS:${OBJ_LIB}> PARENT_SCOPE)
