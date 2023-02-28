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

### graph-compiler specical config ###

set(SC_LIB_NAME dnnl_graph_backend_compiler_core)
SET(SC_MEMORY_LEAK_CHECK 0 CACHE STRING "memory leak checking. 0 = off, 1 = check numbers of instances, 2 = dump leaked objects")
set(SC_CPP_VER 11)

if(NOT "${SC_CPU_RUNTIME}" MATCHES "^(OMP|TBB|SEQ)$")
    message(FATAL_ERROR "Unsupported CPU runtime: ${SC_CPU_RUNTIME}")
endif()

# SET(DNNL_CPU_RUNTIME ${SC_CPU_RUNTIME})
# SET(DNNL_CPU_THREADING_RUNTIME ${SC_CPU_RUNTIME})

if(${SC_CPU_RUNTIME} STREQUAL "OMP")
    set(SC_OMP_ENABLED ON)
    add_definitions("-DSC_CPU_THREADPOOL=1")
    set_property(GLOBAL APPEND PROPERTY GRAPH_COMPILER_DEFINITIONS "-DSC_CPU_THREADPOOL=1")
    find_package(OpenMP REQUIRED)
endif()

if(${SC_CPU_RUNTIME} STREQUAL "TBB")
    add_definitions("-DSC_CPU_THREADPOOL=2")
    set_property(GLOBAL APPEND PROPERTY GRAPH_COMPILER_DEFINITIONS "-DSC_CPU_THREADPOOL=2")
endif()

if(${SC_CPU_RUNTIME} STREQUAL "SEQ")
    add_definitions("-DSC_CPU_THREADPOOL=0")
    set_property(GLOBAL APPEND PROPERTY GRAPH_COMPILER_DEFINITIONS "-DSC_CPU_THREADPOOL=0")
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


set(DNNL_PATH "${CMAKE_SOURCE_DIR}/src")

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/cpu/x64
)

file(GLOB_RECURSE SC_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.c
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp
    )

if(WIN32 OR APPLE OR NOT SC_CFAKE_ENABLED)
    list(REMOVE_ITEM SC_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/util/fdstream.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/util/fdstream.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/util/subprocess.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/compiler/jit/cfake/cfake_jit.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/compiler/jit/cfake/cfake_jit.hpp
    )
    add_definitions("-DSC_CFAKE_JIT_ENABLED=0")
    set_property(GLOBAL APPEND PROPERTY GRAPH_COMPILER_DEFINITIONS "-DSC_CFAKE_JIT_ENABLED=0")
else()
    add_definitions("-DSC_CFAKE_JIT_ENABLED=1")
    set_property(GLOBAL APPEND PROPERTY GRAPH_COMPILER_DEFINITIONS "-DSC_CFAKE_JIT_ENABLED=1")
endif()


if(NOT SC_BUILTIN_ENABLED)
    file(GLOB_RECURSE SC_BUILTIN_JIT_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/src/compiler/jit/xbyak/*
    )
    list(APPEND SC_BUILTIN_JIT_SRC ${CMAKE_CURRENT_SOURCE_DIR}/dummy)
    list(REMOVE_ITEM SC_SOURCES ${SC_BUILTIN_JIT_SRC})
    add_definitions("-DSC_BUILTIN_JIT_ENABLED=0")
    set_property(GLOBAL APPEND PROPERTY GRAPH_COMPILER_DEFINITIONS "-DSC_BUILTIN_JIT_ENABLED=0")
else()
    add_definitions("-DSC_BUILTIN_JIT_ENABLED=1")
    set_property(GLOBAL APPEND PROPERTY GRAPH_COMPILER_DEFINITIONS "-DSC_BUILTIN_JIT_ENABLED=1")
endif()

if(SC_LLVM_ENABLED)
    if(${SC_LLVM_VERSION} LESS 9)
        # default to c++11
    elseif(${SC_LLVM_VERSION} LESS 16)
        set(SC_CPP_VER "14")
    else()
        set(SC_CPP_VER "17")
    endif()
    if(NOT MSVC)
        include(CheckCXXCompilerFlag)
        CHECK_CXX_COMPILER_FLAG(-std=c++${SC_CPP_VER} COMPILER_SUPPORTS_LLVM_CXX_STD)
        if(NOT COMPILER_SUPPORTS_LLVM_CXX_STD)
            message(WARNING "LLVM${SC_LLVM_VERSION} requires c++${SC_CPP_VER} which is not supported by your C++ compiler. LLVM backend is disabled")
            set(SC_LLVM_ENABLED OFF)
            set(SC_CPP_VER "11")
        endif()
    endif()
endif()

if(NOT SC_LLVM_ENABLED)
    file(GLOB_RECURSE SC_LLVM_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/src/compiler/codegen/codegen_llvm.[ch]pp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/compiler/jit/llvm/*
    )
    list(APPEND SC_LLVM_SRC ${CMAKE_CURRENT_SOURCE_DIR}/dummy)
    list(REMOVE_ITEM SC_SOURCES ${SC_LLVM_SRC})
endif()

if( SC_MEMORY_LEAK_CHECK GREATER -1 AND SC_MEMORY_LEAK_CHECK LESS 3 )
    add_definitions("-DSC_MEMORY_LEAK_CHECK=${SC_MEMORY_LEAK_CHECK}")
    set_property(GLOBAL APPEND PROPERTY GRAPH_COMPILER_DEFINITIONS "-DSC_MEMORY_LEAK_CHECK=${SC_MEMORY_LEAK_CHECK}")
else()
    message(FATAL_ERROR "SC_MEMORY_LEAK_CHECK should be in 0 to 2")
endif()

####################
# Configuring LLVM
####################

if(SC_LLVM_ENABLED)
    message(STATUS "LLVM backend is ON. version ${SC_LLVM_VERSION}")
    add_definitions("-DSC_LLVM_BACKEND=${SC_LLVM_VERSION}")
    set_property(GLOBAL APPEND PROPERTY GRAPH_COMPILER_DEFINITIONS -DSC_LLVM_BACKEND=${SC_LLVM_VERSION})
    include_directories(${SC_LLVM_INCLUDE_PATH}) # include LLVM headers
endif()
####################
# Compiling objlib
####################

add_library(${SC_LIB_NAME} OBJECT ${SC_SOURCES})
set_property(GLOBAL APPEND PROPERTY DNNL_LIB_DEPS
    $<TARGET_OBJECTS:${SC_LIB_NAME}>)

if(SC_LIBRARY_TYPE STREQUAL "SHARED")
    if(MSVC)
        set_property(TARGET dnnl_graph_backend_compiler PROPERTY COMPILE_FLAGS "/DSC_DLL_EXPORTS")
    else()
        set_property(TARGET dnnl_graph_backend_compiler PROPERTY COMPILE_FLAGS "-DSC_DLL_EXPORTS")
    endif()
    add_definitions("-DSC_DLL_EXPORTS")
endif()

include_directories(
    ${PROJECT_BINARY_DIR}/include
)

################
# Linking LLVM
################

if(SC_LLVM_ENABLED)
    set(SC_LLVM_ABS_LIBPATH ${SC_LLVM_LIB_NAME})

    #expose the LLVM libraries it depends on when cmake version > 3.0
    if(${CMAKE_VERSION} VERSION_GREATER "3.0.0")
        add_library(dnnl_graphcompiler_llvm_lib INTERFACE IMPORTED GLOBAL)
        set_property(TARGET dnnl_graphcompiler_llvm_lib PROPERTY INTERFACE_LINK_LIBRARIES ${SC_LLVM_LIB_NAME})
        add_library(dnnl_graphcompiler_llvm_lib_exclude_string INTERFACE IMPORTED GLOBAL)
        set_property(TARGET dnnl_graphcompiler_llvm_lib_exclude_string PROPERTY INTERFACE_LINK_LIBRARIES ${SC_LLVM_LIB_EXCLUDE})
    endif()

    if(${SC_LLVM_CONFIG_RETURN_STATIC})
        set_property(GLOBAL APPEND PROPERTY DNNL_SUBDIR_EXTRA_STATIC_LIBS
            ${SC_LLVM_ABS_LIBPATH})
    else()
        set_property(GLOBAL APPEND PROPERTY DNNL_SUBDIR_EXTRA_SHARED_LIBS
            ${SC_LLVM_ABS_LIBPATH})
    endif()

    if(NOT MSVC)
        set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/src/compiler/jit/llvm/llvm_jit_resolver.cpp PROPERTIES COMPILE_FLAGS -fno-rtti)
    endif()
endif()

##########################
# Setting compiler flags
##########################

if(NOT MSVC)
    set_property(TARGET ${SC_LIB_NAME} PROPERTY CXX_STANDARD ${SC_CPP_VER})
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-unused-variable -Wno-unused-function -std=c99 -DSC_HOME=\"${CMAKE_CURRENT_SOURCE_DIR}\"")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-variable -Wno-unused-function -DSC_HOME=\"${CMAKE_CURRENT_SOURCE_DIR}\"")
    if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wmissing-field-initializers")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wmissing-field-initializers")
    endif()
endif()
