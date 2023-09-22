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

if(NOT "${SC_CPU_RUNTIME}" MATCHES "^(OMP|TBB|SEQ|THREADPOOL|SYCL)$")
    message(FATAL_ERROR "Unsupported CPU runtime: ${SC_CPU_RUNTIME}")
endif()

if("${SC_CPU_RUNTIME}" STREQUAL "SYCL")
    set(SC_CPU_RUNTIME "TBB")
endif()

# SET(DNNL_CPU_RUNTIME ${SC_CPU_RUNTIME})
# SET(DNNL_CPU_THREADING_RUNTIME ${SC_CPU_RUNTIME})

if(${SC_CPU_RUNTIME} STREQUAL "OMP")
    set(SC_OMP_ENABLED ON)
    add_definitions_with_host_compiler("-DSC_CPU_THREADPOOL=1")
    set_property(GLOBAL APPEND PROPERTY GRAPH_COMPILER_DEFINITIONS "-DSC_CPU_THREADPOOL=1")
endif()

if(${SC_CPU_RUNTIME} STREQUAL "TBB")
    add_definitions_with_host_compiler("-DSC_CPU_THREADPOOL=2")
    set_property(GLOBAL APPEND PROPERTY GRAPH_COMPILER_DEFINITIONS "-DSC_CPU_THREADPOOL=2")
endif()

if(${SC_CPU_RUNTIME} STREQUAL "SEQ")
    add_definitions_with_host_compiler("-DSC_CPU_THREADPOOL=0")
    set_property(GLOBAL APPEND PROPERTY GRAPH_COMPILER_DEFINITIONS "-DSC_CPU_THREADPOOL=0")
endif()

if(${SC_CPU_RUNTIME} STREQUAL "THREADPOOL")
    add_definitions_with_host_compiler("-DSC_CPU_THREADPOOL=3")
    set_property(GLOBAL APPEND PROPERTY GRAPH_COMPILER_DEFINITIONS "-DSC_CPU_THREADPOOL=3")
endif()

set(DNNL_PATH "${CMAKE_SOURCE_DIR}/src")

include_directories_with_host_compiler(
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/src/cpu/x64
)

file(GLOB_RECURSE SC_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.c
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp
    )

if(NOT DNNL_ENABLE_JIT_PROFILING)
    list(REMOVE_ITEM SC_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/compiler/jit/xbyak/debug/vtune/vtune.cpp)
else()
    add_definitions_with_host_compiler(-DSC_PROFILING=1)
endif()

if(WIN32 OR APPLE OR NOT SC_CFAKE_ENABLED)
    list(REMOVE_ITEM SC_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/util/fdstream.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/util/fdstream.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/util/subprocess.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/compiler/jit/cfake/cfake_jit.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/compiler/jit/cfake/cfake_jit.hpp
    )
    add_definitions_with_host_compiler("-DSC_CFAKE_JIT_ENABLED=0")
    set_property(GLOBAL APPEND PROPERTY GRAPH_COMPILER_DEFINITIONS "-DSC_CFAKE_JIT_ENABLED=0")
else()
    add_definitions_with_host_compiler("-DSC_CFAKE_JIT_ENABLED=1")
    set_property(GLOBAL APPEND PROPERTY GRAPH_COMPILER_DEFINITIONS "-DSC_CFAKE_JIT_ENABLED=1")
endif()


if(NOT SC_BUILTIN_ENABLED)
    file(GLOB_RECURSE SC_BUILTIN_JIT_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/src/compiler/jit/xbyak/*
    )
    list(APPEND SC_BUILTIN_JIT_SRC ${CMAKE_CURRENT_SOURCE_DIR}/dummy)
    list(REMOVE_ITEM SC_SOURCES ${SC_BUILTIN_JIT_SRC})
    add_definitions_with_host_compiler("-DSC_BUILTIN_JIT_ENABLED=0")
    set_property(GLOBAL APPEND PROPERTY GRAPH_COMPILER_DEFINITIONS "-DSC_BUILTIN_JIT_ENABLED=0")
else()
    add_definitions_with_host_compiler("-DSC_BUILTIN_JIT_ENABLED=1")
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
        ${CMAKE_CURRENT_SOURCE_DIR}/src/compiler/codegen/llvm/*
        ${CMAKE_CURRENT_SOURCE_DIR}/src/compiler/jit/llvm/*
    )
    list(APPEND SC_LLVM_SRC ${CMAKE_CURRENT_SOURCE_DIR}/dummy)
    list(REMOVE_ITEM SC_SOURCES ${SC_LLVM_SRC})
endif()

if( SC_MEMORY_LEAK_CHECK GREATER -1 AND SC_MEMORY_LEAK_CHECK LESS 3 )
    add_definitions_with_host_compiler("-DSC_MEMORY_LEAK_CHECK=${SC_MEMORY_LEAK_CHECK}")
    set_property(GLOBAL APPEND PROPERTY GRAPH_COMPILER_DEFINITIONS "-DSC_MEMORY_LEAK_CHECK=${SC_MEMORY_LEAK_CHECK}")
else()
    message(FATAL_ERROR "SC_MEMORY_LEAK_CHECK should be in 0 to 2")
endif()

####################
# Configuring LLVM
####################

if(SC_LLVM_ENABLED)
    message(STATUS "LLVM backend is ON. version ${SC_LLVM_VERSION}")
    add_definitions_with_host_compiler("-DSC_LLVM_BACKEND=${SC_LLVM_VERSION}")
    set_property(GLOBAL APPEND PROPERTY GRAPH_COMPILER_DEFINITIONS -DSC_LLVM_BACKEND=${SC_LLVM_VERSION})
    file(TO_CMAKE_PATH "${SC_LLVM_INCLUDE_PATH}" SC_LLVM_INCLUDE_PATH_CMAKE) # to convert "\" to cmake style "/" in windows path
    include_directories_with_host_compiler(${SC_LLVM_INCLUDE_PATH_CMAKE}) # include LLVM headers
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
    add_definitions_with_host_compiler("-DSC_DLL_EXPORTS")
endif()

include_directories_with_host_compiler(
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

if(CMAKE_BASE_NAME MATCHES "(icx|icpx)" AND ${SC_CPP_VER} LESS 14)
    set(NEED_SET_CPP_STD OFF)
else()
    set(NEED_SET_CPP_STD ON)
endif()

if(NEED_SET_CPP_STD)
    if(${CMAKE_VERSION} VERSION_LESS "3.1.0") 
        if(MSVC)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /std:c++${SC_CPP_VER}")
        else()
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++${SC_CPP_VER}")
        endif()
    else()
        set_property(TARGET ${SC_LIB_NAME} PROPERTY CXX_STANDARD ${SC_CPP_VER})
    endif()
endif()

if(NOT MSVC OR CMAKE_BASE_NAME MATCHES "(icx|icpx)")
    if(NOT MSVC)

    else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations")
        append_host_compiler_options(CMAKE_CXX_FLAGS "-Wno-deprecated-declarations")
    endif()
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-unused-variable -Wno-unused-function -std=c99 -DSC_HOME=\"${CMAKE_CURRENT_SOURCE_DIR}\"")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-variable -Wno-unused-function")
    append_host_compiler_options(CMAKE_CXX_FLAGS "-Wno-unused-variable -Wno-unused-function")
    if(DNNL_DPCPP_HOST_COMPILER STREQUAL "DEFAULT")
        append(CMAKE_CXX_FLAGS " -DSC_HOME=\"${CMAKE_CURRENT_SOURCE_DIR}\"")
    endif()
    if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wmissing-field-initializers")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wmissing-field-initializers")
    endif()
endif()

if("${DNNL_CPU_RUNTIME}" STREQUAL "SYCL" AND WIN32 AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations")
    append_host_compiler_options(CMAKE_CXX_FLAGS "-Wno-deprecated-declarations")
endif()

# SET(DNNL_CPU_RUNTIME ${SC_CPU_RUNTIME})
