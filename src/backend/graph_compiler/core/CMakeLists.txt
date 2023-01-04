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

### General policy inherit from DNNL ###

cmake_minimum_required(VERSION 2.8.11)

if(POLICY CMP0025)
    cmake_policy(SET CMP0025 NEW) # For AppleClang
endif()

if(POLICY CMP0022)
    cmake_policy(SET CMP0022 NEW)
endif()

if(POLICY CMP0054)
    cmake_policy(SET CMP0054 NEW)
endif()

# Enable RPATH on MacOS/OSX
if(POLICY CMP0042)
    cmake_policy(SET CMP0042 NEW)
endif()

# Do not export symbols from executables
if(POLICY CMP0065)
    cmake_policy(SET CMP0065 NEW)
endif()

# Pass all flags to try_compile
if(POLICY CMP0056)
    cmake_policy(SET CMP0056 NEW)
endif()
if(POLICY CMP0066)
    cmake_policy(SET CMP0066 NEW)
endif()

# Use <PackageName>_ROOT env. variable as a prefix
if(POLICY CMP0074)
    cmake_policy(SET CMP0074 NEW)
endif()

set(CMAKE_CXX_FLAGS_RELEASEWITHASSERT "-O3")
# Honor visibility properties for all target types.
if(POLICY CMP0063)
    cmake_policy(SET CMP0063 NEW)
endif()

if("${CMAKE_BUILD_TYPE}" STREQUAL "")
    message(STATUS "CMAKE_BUILD_TYPE is unset, defaulting to Release")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING
        "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel ...")
endif()


list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

### graph-compiler specical config ###

if (CMAKE_VERSION VERSION_LESS 3.0)
    project(${PROJECT_NAME} C CXX)
else()
    cmake_policy(SET CMP0048 NEW)
    project(${PROJECT_NAME} VERSION "${PROJECT_VERSION}" LANGUAGES C CXX)
endif()


set(SC_LIB_NAME graphcompiler)

SET(SC_MEMORY_LEAK_CHECK 0 CACHE STRING "memory leak checking. 0 = off, 1 = check numbers of instances, 2 = dump leaked objects")

set(INSTALL_DIR ${CMAKE_CURRENT_BINARY_DIR}/install)
set(CMAKE_INSTALL_BINDIR "lib" CACHE STRING "library install path")
set(CMAKE_INSTALL_LIBDIR "bin" CACHE STRING "binary install path")

set(SC_CPP_VER 11)

link_directories(${INSTALL_DIR}/lib)

if(NOT "${SC_CPU_RUNTIME}" MATCHES "^(OMP|TBB|SEQ)$")
    message(FATAL_ERROR "Unsupported CPU runtime: ${SC_CPU_RUNTIME}")
endif()

SET(DNNL_CPU_RUNTIME ${SC_CPU_RUNTIME})
SET(DNNL_CPU_THREADING_RUNTIME ${SC_CPU_RUNTIME})

if(${SC_CPU_RUNTIME} STREQUAL "OMP")
    set(SC_OMP_ENABLED ON)
    add_definitions("-DSC_CPU_THREADPOOL=1")
    find_package(OpenMP REQUIRED)
endif()

if(${SC_CPU_RUNTIME} STREQUAL "TBB")
    include("cmake/TBB.cmake")
    add_definitions("-DSC_CPU_THREADPOOL=2")
endif()

if(${SC_CPU_RUNTIME} STREQUAL "SEQ")
    add_definitions("-DSC_CPU_THREADPOOL=0")
endif()

##############################
# Enable better debug info...
##############################

if(NOT MSVC)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -ggdb -O0 -fno-inline -fno-omit-frame-pointer -gdwarf-4")
endif()

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL Clang)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-limit-debug-info")
endif()

##############################


set(DNNL_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../../third_party/oneDNN")
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_BINARY_DIR}/third_party/oneDNN/include
    ${DNNL_PATH}/src
    ${DNNL_PATH}/src/cpu/x64
    ${DNNL_PATH}/include
    ${PROJECT_BINARY_DIR}/../../../../third_party/oneDNN/include
)

file(GLOB_RECURSE SC_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.c
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp
    )

if(WIN32 OR APPLE)
    list(REMOVE_ITEM SC_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/util/fdstream.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/util/fdstream.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/util/subprocess.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/compiler/jit/cfake/cfake_jit.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/compiler/jit/cfake/cfake_jit.hpp
    )
endif()

if( SC_MEMORY_LEAK_CHECK GREATER -1 AND SC_MEMORY_LEAK_CHECK LESS 3 )
    add_definitions("-DSC_MEMORY_LEAK_CHECK=${SC_MEMORY_LEAK_CHECK}")
else()
    message(FATAL_ERROR "SC_MEMORY_LEAK_CHECK should be in 0 to 2")
endif()

####################
# Configuring LLVM
####################

message(STATUS "LLVM backend is ON. version ${SC_LLVM_VERSION}")
add_definitions("-DSC_LLVM_BACKEND=${SC_LLVM_VERSION}")
include_directories(${SC_LLVM_INCLUDE_PATH}) # include LLVM headers

####################
# Compiling objlib
####################

add_library(objlib OBJECT ${SC_SOURCES})
if (SC_LIBRARY_TYPE STREQUAL "STATIC")
    add_library(${SC_LIB_NAME} STATIC $<TARGET_OBJECTS:objlib>)
else()
    add_library(${SC_LIB_NAME} SHARED $<TARGET_OBJECTS:objlib>)
endif()

if(SC_LIBRARY_TYPE STREQUAL "SHARED")
    if(MSVC)
        set_property(TARGET objlib PROPERTY COMPILE_FLAGS "/DSC_DLL_EXPORTS")
    else()
        set_property(TARGET objlib PROPERTY COMPILE_FLAGS "-DSC_DLL_EXPORTS")
    endif()
endif()

include_directories(
    ${PROJECT_BINARY_DIR}/include
)

################
# Linking LLVM
################

set(SC_LLVM_ABS_LIBPATH ${SC_LLVM_LIB_NAME})

#expose the LLVM libraries it depends on when cmake version > 3.0
if(${CMAKE_VERSION} VERSION_GREATER "3.0.0")
    add_library(dnnl_graphcompiler_llvm_lib INTERFACE IMPORTED GLOBAL)
    set_property(TARGET dnnl_graphcompiler_llvm_lib PROPERTY INTERFACE_LINK_LIBRARIES ${SC_LLVM_LIB_NAME})
    add_library(dnnl_graphcompiler_llvm_lib_exclude_string INTERFACE IMPORTED GLOBAL)
    set_property(TARGET dnnl_graphcompiler_llvm_lib_exclude_string PROPERTY INTERFACE_LINK_LIBRARIES ${SC_LLVM_LIB_EXCLUDE})
endif()

target_link_libraries(${SC_LIB_NAME} PRIVATE ${SC_LLVM_ABS_LIBPATH})

if(NOT MSVC)
    set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/src/compiler/jit/llvm/llvm_jit_resolver.cpp PROPERTIES COMPILE_FLAGS -fno-rtti)
endif()

##########################
# Setting compiler flags
##########################

if(${SC_LLVM_VERSION} GREATER 8)
    set(SC_CPP_VER "14")
endif()

if(NOT MSVC)
    message(STATUS "C++ standard is ${SC_CPP_VER}")
    if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wmissing-field-initializers")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wmissing-field-initializers")
    endif()
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fvisibility=hidden -ffunction-sections -fdata-sections -Wall -Wno-unused-variable -Wno-unused-function -std=c99 -fPIC -Werror -DSC_HOME=\"${CMAKE_CURRENT_SOURCE_DIR}\"")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden -ffunction-sections -fdata-sections -Wall -Wno-unused-variable -Wno-unused-function -std=c++${SC_CPP_VER} -fPIC -Werror -DSC_HOME=\"${CMAKE_CURRENT_SOURCE_DIR}\"")
    # Deal with Clang < 6.0's missing-braces issue
    if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 6.0)
            SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-missing-braces")
        endif()
    endif()
    append(CMAKE_CXX_FLAGS "-fvisibility-inlines-hidden")
    if(NOT APPLE)
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--as-needed")
    else()
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-dead_strip")
    endif()
endif()

if(NOT ${KERNEL_PROFILE} STREQUAL "OFF")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DSC_KERNEL_PROFILE")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DSC_KERNEL_PROFILE")
endif()

target_link_libraries_install(${SC_LIB_NAME} "${EXTRA_SHARED_LIBS}")
if(SC_LIBRARY_TYPE STREQUAL "STATIC")
    target_link_libraries_install(${SC_LIB_NAME} "${EXTRA_STATIC_LIBS}")
endif()


set(PARENT_NAME dnnl_graph)

if(UNIX)
    target_link_libraries(${SC_LIB_NAME} PUBLIC dl)
endif()

set(LIB_CONFIG_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/${SC_LIB_NAME}")
set(LIB_EXPORT_NAME "${SC_LIB_NAME}-targets")
install(TARGETS ${SC_LIB_NAME}
    EXPORT "${LIB_EXPORT_NAME}"
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
string(TOUPPER "${SC_LIB_NAME}::" LIB_NAMESPACE)
install(EXPORT ${LIB_EXPORT_NAME}
NAMESPACE ${LIB_NAMESPACE}
DESTINATION ${LIB_CONFIG_INSTALL_DIR})

###########
# Make the runtime library target and don't need to build it by default
###########
file(GLOB_RECURSE SC_RUNTIME_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/runtime/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/runtime/*.c
    )
add_library(graphcompiler_runtime STATIC ${SC_RUNTIME_SOURCES})
target_link_libraries(graphcompiler_runtime PRIVATE dnnl_brgemm)
set_property(TARGET graphcompiler_runtime PROPERTY POSITION_INDEPENDENT_CODE ON
            CXX_VISIBILITY_PRESET "hidden"
            VISIBILITY_INLINES_HIDDEN 1
            EXCLUDE_FROM_ALL True)

###########
# Make the brgemm library target and don't need to build it by default
###########
add_definitions(-DDNNL_ENABLE_JIT_PROFILING=0)
add_definitions(-DDNNL_ENABLE_MAX_CPU_ISA=1)
add_definitions(-DDNNL_ENABLE_CPU_ISA_HINTS=1)
file(GLOB_RECURSE DNNL_SOURCES
    ${DNNL_PATH}/src/cpu/x64/brgemm/*.cpp
    ${DNNL_PATH}/src/cpu/x64/injectors/*.cpp
    ${DNNL_PATH}/src/cpu/x64/cpu_isa_traits.cpp
    ${DNNL_PATH}/src/cpu/x64/jit_avx512_core_bf16cvt.cpp
    ${DNNL_PATH}/src/cpu/x64/amx_tile_configure.[ch]pp
    ${DNNL_PATH}/src/cpu/x64/jit_uni_convert_xf16.[ch]pp
    ${DNNL_PATH}/src/cpu/jit_utils/jit_utils.cpp
    ${DNNL_PATH}/src/cpu/platform.[ch]pp
    ${DNNL_PATH}/src/cpu/bfloat16.cpp
    ${DNNL_PATH}/src/cpu/binary_injector_utils.cpp
    ${DNNL_PATH}/src/common/fpmath_mode.cpp
    ${DNNL_PATH}/src/common/utils.cpp
    ${DNNL_PATH}/src/common/bfloat16.[ch]pp
    ${DNNL_PATH}/src/common/memory_debug.cpp
    ${DNNL_PATH}/src/common/primitive_attr.cpp
    ${DNNL_PATH}/src/common/broadcast_strategy.cpp
    ${DNNL_PATH}/src/common/primitive_exec_types.cpp
    ${DNNL_PATH}/src/common/memory.cpp
    ${DNNL_PATH}/src/common/memory_zero_pad.cpp
    ${DNNL_PATH}/src/common/memory_desc_wrapper.cpp
    ${DNNL_PATH}/src/common/memory_desc.[ch]pp
    ${DNNL_PATH}/src/common/dnnl_thread.cpp
)
add_library(dnnl_brgemm ${DNNL_SOURCES})
set_property(TARGET dnnl_brgemm PROPERTY POSITION_INDEPENDENT_CODE ON
            CXX_VISIBILITY_PRESET "hidden"
            VISIBILITY_INLINES_HIDDEN 1
            EXCLUDE_FROM_ALL True)

if(DNNL_GRAPH_LIBRARY_TYPE STREQUAL "SHARED" OR MSVC)
    target_link_libraries(${SC_LIB_NAME} PRIVATE $<BUILD_INTERFACE:dnnl_brgemm>)
endif()

set(${PARENT_NAME}_EXTRA_INTERFACE ${SC_LIB_NAME} CACHE INTERNAL "set sc library")
