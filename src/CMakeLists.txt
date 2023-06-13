#===============================================================================
# Copyright 2016-2023 Intel Corporation
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

file(GLOB HEADERS_ROOT
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/*.hpp
    )
file(GLOB HEADERS_SUBDIR
    ${PROJECT_BINARY_DIR}/include/oneapi/dnnl/*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/oneapi/dnnl/*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/oneapi/dnnl/*.hpp
    )
include_directories_with_host_compiler(${CMAKE_CURRENT_SOURCE_DIR})

if (DNNL_LIBRARY_TYPE STREQUAL "SHARED")
    add_definitions_with_host_compiler(-DDNNL_DLL_EXPORTS)
endif()

# propagate SRC specific flags
append(CMAKE_C_FLAGS "${CMAKE_SRC_CCXX_FLAGS}")
append(CMAKE_CXX_FLAGS "${CMAKE_SRC_CCXX_FLAGS}")

# propagate no warning flags
append(CMAKE_C_FLAGS "${CMAKE_CCXX_NOWARN_FLAGS}")
append(CMAKE_CXX_FLAGS "${CMAKE_CCXX_NOWARN_FLAGS}")

# propagate sanitizer flags
append(CMAKE_C_FLAGS "${CMAKE_CCXX_SANITIZER_FLAGS}")
append(CMAKE_CXX_FLAGS "${CMAKE_CCXX_SANITIZER_FLAGS}")

append_host_compiler_options(CMAKE_CXX_FLAGS "${DPCPP_SRC_CXX_FLAGS}")
append_host_compiler_options(CMAKE_CXX_FLAGS "${DPCPP_CXX_NOWARN_FLAGS}")

if(NOT DNNL_VERBOSE)
    add_definitions_with_host_compiler(-DDISABLE_VERBOSE)
endif()

if(DNNL_ENABLE_CONCURRENT_EXEC)
    add_definitions_with_host_compiler(-DDNNL_ENABLE_CONCURRENT_EXEC)
endif()

if(DNNL_ENABLE_PRIMITIVE_CACHE)
    message(STATUS "Primitive cache is enabled")
else()
    add_definitions_with_host_compiler(-DDNNL_DISABLE_PRIMITIVE_CACHE)
    message(STATUS "Primitive cache is disabled")
endif()

if(DNNL_ENABLE_JIT_PROFILING OR DNNL_ENABLE_ITT_TASKS)
    if (UNIX AND NOT APPLE)
        # Not every compiler adds -ldl automatically
        list(APPEND EXTRA_SHARED_LIBS "${CMAKE_DL_LIBS}")
    endif()
endif()

if(DNNL_DEV_MODE)
  add_definitions_with_host_compiler(-DDNNL_DEV_MODE)
  message(STATUS "Developer mode enabled")
endif()

if(DNNL_EXPERIMENTAL)
    message(STATUS "Experimental features are enabled")
endif()

if(DNNL_EXPERIMENTAL_SPARSE)
    message(STATUS "Experimental functionality for sparse domain is enabled")
endif()

if(DNNL_EXPERIMENTAL_PROFILING)
    message(STATUS "Experimental profiling capabilities are enabled")
endif()

if(DNNL_ENABLE_ITT_TASKS AND NOT DNNL_CPU_RUNTIME STREQUAL "NONE")
    # Only supported for certain architectures (see src/common/CMakeLists.txt)
    if(DNNL_TARGET_ARCH STREQUAL "AARCH64" OR DNNL_TARGET_ARCH STREQUAL "X64")
        add_definitions_with_host_compiler(-DDNNL_ENABLE_ITT_TASKS)
    endif()
endif()

if(DNNL_ENABLE_MAX_CPU_ISA)
    add_definitions_with_host_compiler(-DDNNL_ENABLE_MAX_CPU_ISA)
endif()

if(DNNL_ENABLE_CPU_ISA_HINTS)
    add_definitions_with_host_compiler(-DDNNL_ENABLE_CPU_ISA_HINTS)
endif()

if(WIN32)
    add_definitions_with_host_compiler(-D_WIN)
    add_definitions_with_host_compiler(-DNOMINMAX)
    add_definitions_with_host_compiler(-DWIN32_LEAN_AND_MEAN)
endif()

# Windows does not support weak/strong symbols and no guarrantees by the linker
# for out_of_memory testing to work. Not tested on macOS
if(UNIX)
    if(DNNL_ENABLE_MEM_DEBUG)
        add_definitions_with_host_compiler(-DDNNL_ENABLE_MEM_DEBUG)
    endif()
endif()

add_subdirectory(common)

if(NOT DNNL_CPU_RUNTIME STREQUAL "NONE")
    add_subdirectory(cpu)
endif()

if(NOT DNNL_GPU_RUNTIME STREQUAL "NONE")
    add_subdirectory(gpu)
endif()

if(DNNL_WITH_SYCL)
    # Enable linking SYCL kernels.
    if(DNNL_SYCL_CUDA)
        append(CMAKE_SHARED_LINKER_FLAGS "-fsycl-targets=nvptx64-nvidia-cuda")
        append(CMAKE_STATIC_LINKER_FLAGS "-fsycl-targets=nvptx64-nvidia-cuda")
    endif()
    add_subdirectory(sycl)
endif()

if(ONEDNN_BUILD_GRAPH)
    message(STATUS "Graph component is enabled")
    if (DNNL_GPU_RUNTIME STREQUAL "OCL")
        message(FATAL_ERROR "Graph API does not support OCL GPU runtime. "
            "Either disable Graph API with ONEDNN_BUILD_GRAPH=OFF or change GPU "
            "runtime to SYCL with ONEDNN_GPU_RUNTIME=SYCL.")
    endif()

    if (NOT DNNL_GPU_VENDOR STREQUAL "INTEL")
        message(FATAL_ERROR "Graph API does not support ${DNNL_GPU_VENDOR} GPU. "
            "Either disable Graph API with ONEDNN_BUILD_GRAPH=OFF or change GPU "
            "vendor to INTEL with ONEDNN_GPU_VENDOR=INTEL.")
    endif()

    if (NOT DNNL_ENABLE_PRIMITIVE STREQUAL "ALL")
        message(FATAL_ERROR "Graph API does not support selecting primitives. "
            "Either disable Graph API with ONEDNN_BUILD_GRAPH=OFF or change "
            "primitive selection with ONEDNN_ENABLE_PRIMITIVE=ALL.")
    endif()

    if (NOT DNNL_ENABLE_WORKLOAD STREQUAL "TRAINING")
        message(FATAL_ERROR "Graph API does not support selecting workloads. "
            "Either disable Graph API with ONEDNN_BUILD_GRAPH=OFF or change "
            "workload selection with ONEDNN_ENABLE_WORKLOAD=TRAINING.")
    endif()

    if(ONEDNN_EXPERIMENTAL_GRAPH_COMPILER_BACKEND)
        add_definitions_with_host_compiler(-DDNNL_ENABLE_COMPILER_BACKEND)
    endif()
    if(ONEDNN_ENABLE_GRAPH_DUMP)
        message(STATUS "Graph artifacts dump is enabled")
        add_definitions_with_host_compiler(-DDNNL_ENABLE_GRAPH_DUMP)
    endif()

    add_subdirectory(graph)
    if(ONEDNN_EXPERIMENTAL_GRAPH_COMPILER_BACKEND AND TARGET dnnl_graphcompiler_llvm_lib_exclude_string)
        get_property(GC_EXCLUDE_LIBS TARGET dnnl_graphcompiler_llvm_lib_exclude_string PROPERTY INTERFACE_LINK_LIBRARIES)
        if(DEFINED GC_EXCLUDE_LIBS AND NOT MSVC AND NOT APPLE)
            # set the LLVM symbols as hidden, or all LLVM symbols will be exported
            append(CMAKE_SHARED_LINKER_FLAGS "-Wl,--exclude-libs=${GC_EXCLUDE_LIBS}")
        endif()
    endif()
else()
    # If graph component is not built, remove the headers from build and installation.
    list(REMOVE_ITEM HEADERS_SUBDIR
    "${CMAKE_CURRENT_SOURCE_DIR}/../include/oneapi/dnnl/dnnl_graph.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/../include/oneapi/dnnl/dnnl_graph.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../include/oneapi/dnnl/dnnl_graph_types.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/../include/oneapi/dnnl/dnnl_graph_sycl.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/../include/oneapi/dnnl/dnnl_graph_sycl.hpp")
endif()

get_property(LIB_DEPS GLOBAL PROPERTY DNNL_LIB_DEPS)
get_property(STATIC_LIB_DEPS GLOBAL PROPERTY DNNL_SUBDIR_EXTRA_STATIC_LIBS)
get_property(SHARED_LIB_DEPS GLOBAL PROPERTY DNNL_SUBDIR_EXTRA_SHARED_LIBS)
add_library(${LIB_PACKAGE_NAME} ${DNNL_LIBRARY_TYPE}
    ${VERSION_RESOURCE_FILE} ${HEADERS_ROOT} ${HEADERS_SUBDIR} ${LIB_DEPS})

# LINK_PRIVATE for cmake 2.8.11 compatibility
target_link_libraries(${LIB_PACKAGE_NAME} LINK_PRIVATE ${${LIB_PACKAGE_NAME}_INTERFACE} ${STATIC_LIB_DEPS} ${SHARED_LIB_DEPS})

set_property(TARGET ${LIB_PACKAGE_NAME} PROPERTY OUTPUT_NAME ${DNNL_LIBRARY_NAME})
set_property(TARGET ${LIB_PACKAGE_NAME} PROPERTY VERSION "${DNNL_VERSION_MAJOR}.${DNNL_VERSION_MINOR}")
set_property(TARGET ${LIB_PACKAGE_NAME} PROPERTY SOVERSION "${DNNL_VERSION_MAJOR}")

target_include_directories(${LIB_PACKAGE_NAME} PUBLIC
    $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/../include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    )

target_link_libraries_build(${LIB_PACKAGE_NAME}
    "${EXTRA_SHARED_LIBS};${EXTRA_STATIC_LIBS}")
target_link_libraries_install(${LIB_PACKAGE_NAME} "${EXTRA_SHARED_LIBS}")
if(DNNL_LIBRARY_TYPE STREQUAL "STATIC")
    target_link_libraries_install(${LIB_PACKAGE_NAME} "${EXTRA_STATIC_LIBS}")
endif()

set(LIB_EXPORT_NAME "${LIB_PACKAGE_NAME}-targets")
install(TARGETS ${LIB_PACKAGE_NAME}
    EXPORT "${LIB_EXPORT_NAME}"
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

# If only Cmake could preserve the directory hierarchy...
foreach(header ${HEADERS_ROOT})
    install(FILES ${header} DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/")
endforeach()
foreach(header ${HEADERS_SUBDIR})
    install(FILES ${header} DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/oneapi/dnnl/")
endforeach()

string(TOUPPER "${LIB_PACKAGE_NAME}::" LIB_NAMESPACE)
if(DNNL_INSTALL_MODE STREQUAL "BUNDLE_V2" AND WIN32)
    # Config file for binary distribution needs to define a mapping
    # DEBUG -> RELWITHMDD so that proper library (dnnld) is picked up for the
    # DEBUG configuration.
    set(HANDLE_BUNDLE_DEBUG_SYCL_CONFIGURATION
        "set_property(TARGET ${LIB_NAMESPACE}${LIB_PACKAGE_NAME} PROPERTY \"MAP_IMPORTED_CONFIG_DEBUG\" \"RELWITHMDD\")")
endif()

# Write version and package config files
set(LIB_CONFIG_GENERATE_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
set(LIB_CONFIG_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/${LIB_PACKAGE_NAME}")
set(LIB_VERSION_FILE
    "${LIB_CONFIG_GENERATE_DIR}/${LIB_PACKAGE_NAME}-config-version.cmake")
set(LIB_CONFIG_FILE
    "${LIB_CONFIG_GENERATE_DIR}/${LIB_PACKAGE_NAME}-config.cmake")
write_basic_package_version_file(
    "${LIB_VERSION_FILE}"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion)
configure_package_config_file(
    "../cmake/config.cmake.in"
    "${LIB_CONFIG_FILE}"
    INSTALL_DESTINATION ${LIB_CONFIG_INSTALL_DIR})
install(FILES ${LIB_CONFIG_FILE} ${LIB_VERSION_FILE}
    DESTINATION ${LIB_CONFIG_INSTALL_DIR})
install(EXPORT ${LIB_EXPORT_NAME}
    NAMESPACE ${LIB_NAMESPACE}
    DESTINATION ${LIB_CONFIG_INSTALL_DIR})

# Apply a workaround to CMake config file to make it work with symlinks.
# The patched config file is only used in oneAPI binary distribution.
if(UNIX AND DNNL_INSTALL_MODE STREQUAL "BUNDLE_V2")
    install(CODE "file(READ \"${CMAKE_INSTALL_PREFIX}/${LIB_CONFIG_INSTALL_DIR}/${LIB_PACKAGE_NAME}-targets.cmake\" TARGETS_CONTENT)")
    install(CODE "string(REPLACE
                         \"get_filename_component(_IMPORT_PREFIX \\\"\\\${CMAKE_CURRENT_LIST_FILE}\\\" PATH)\"
                         \"get_filename_component(_IMPORT_PREFIX \\\"\\\${CMAKE_CURRENT_LIST_FILE}\\\" REALPATH)\\nget_filename_component(_IMPORT_PREFIX \\\"\\\${_IMPORT_PREFIX}\\\" PATH)\"
                         TARGETS_CONTENT \"\${TARGETS_CONTENT}\")")
    install(CODE "file(WRITE \"${CMAKE_INSTALL_PREFIX}/${LIB_CONFIG_INSTALL_DIR}/${LIB_PACKAGE_NAME}-targets.cmake\" \"\${TARGETS_CONTENT}\")")
endif()

# Install custom find modules for transitive dependencies
if(DNNL_CPU_THREADING_RUNTIME STREQUAL "TBB")
    if(WIN32)
        install(FILES "../cmake/win/TBBConfig.cmake" RENAME "FindTBB.cmake"
            DESTINATION ${LIB_CONFIG_INSTALL_DIR})
    elseif(APPLE)
        install(FILES "../cmake/mac/TBBConfig.cmake" RENAME "FindTBB.cmake"
            DESTINATION ${LIB_CONFIG_INSTALL_DIR})
    elseif(UNIX)
        install(FILES "../cmake/lnx/TBBConfig.cmake" RENAME "FindTBB.cmake"
            DESTINATION ${LIB_CONFIG_INSTALL_DIR})
    endif()
endif()

if(DNNL_GPU_RUNTIME STREQUAL "OCL")
    install(FILES
        "../cmake/FindOpenCL.cmake"
        DESTINATION ${LIB_CONFIG_INSTALL_DIR})
endif()

# On Windows we need to add dnnl.dll path to CTESTCONFIG_PATH which is later
# passed to ctest and Visual Studio solutions
if(WIN32)
    if(CMAKE_GENERATOR MATCHES "Visual Studio")
        foreach(BUILD_TYPE Release Debug RelWithDebInfo MinSizeRel)
            append_to_windows_path_list(CTESTCONFIG_PATH
                "${CMAKE_CURRENT_BINARY_DIR}/${BUILD_TYPE}")
        endforeach()
    else()
        append_to_windows_path_list(CTESTCONFIG_PATH
            "${CMAKE_CURRENT_BINARY_DIR}")
    endif()
    set(CTESTCONFIG_PATH "${CTESTCONFIG_PATH}" PARENT_SCOPE)
endif()
