#===============================================================================
# Copyright 2020-2022 Intel Corporation
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

# All oneDNN Graph backend sources are archivized in a single static library (dnnl_graph_static.a).
# The library is then used to link the final shared library (dnnl_graph.so) and unit level
# tests (dnnl_graph_unit_test)

file(GLOB HEADERS_ONEAPI
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/oneapi/dnnl/*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/oneapi/dnnl/*.hpp
    )

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

if (DNNL_GRAPH_LIBRARY_TYPE STREQUAL "SHARED" OR DNNL_GRAPH_LIBRARY_TYPE STREQUAL "SDL")
    add_definitions(-DDNNL_GRAPH_DLL_EXPORTS)
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
    set_source_files_properties(${SOURCES}
        PROPERTIES COMPILE_FLAGS "-prec-sqrt")
    set_source_files_properties(${SOURCES}
        PROPERTIES COMPILE_FLAGS "-prec-div")
endif()

if(WIN32)
    append(CMAKE_CXX_FLAGS "-D_WIN")
    append(CMAKE_CXX_FLAGS "-DNOMINMAX") # to allow std::max on Windows with parentheses
endif()

# propagate common compilation flags
append(CMAKE_C_FLAGS "${DNNL_GRAPH_COMMON_COMPILATION_FLAGS}")
append(CMAKE_CXX_FLAGS "${DNNL_GRAPH_COMMON_COMPILATION_FLAGS}")

# propagate SRC specific flags
append(CMAKE_C_FLAGS "${CMAKE_SRC_CCXX_FLAGS}")
append(CMAKE_CXX_FLAGS "${CMAKE_SRC_CCXX_FLAGS}")

# propagate no warning flags
append(CMAKE_C_FLAGS "${CMAKE_CCXX_NOWARN_FLAGS}")
append(CMAKE_CXX_FLAGS "${CMAKE_CCXX_NOWARN_FLAGS}")

add_subdirectory(backend)
add_subdirectory(utils)
add_subdirectory(interface)

get_property(LIB_DEPS GLOBAL PROPERTY ${LIB_NAME}_LIB_DEPS)

if(DNNL_GRAPH_LIBRARY_TYPE STREQUAL "STATIC")
    add_library(${LIB_NAME} STATIC ${HEADERS_ONEAPI} ${SOURCES} ${LIB_DEPS})
else()
    add_library(${LIB_NAME} SHARED ${HEADERS_ONEAPI} ${SOURCES} ${LIB_DEPS})
endif()

if(NOT DNNL_GRAPH_VERBOSE)
    message(STATUS "DNNL Graph verbose mode is disabled")
    target_compile_definitions(${LIB_NAME} PRIVATE DNNL_GRAPH_DISABLE_VERBOSE)
endif()

if(DNNL_GRAPH_ENABLE_COMPILED_PARTITION_CACHE)
    message(STATUS "Compiled partition cache is enabled")
else()
    target_compile_definitions(${LIB_NAME}
        PRIVATE DNNL_GRAPH_DISABLE_COMPILED_PARTITION_CACHE)
    message(STATUS "Compiled partition cache is disabled")
endif()

set_target_properties(${LIB_NAME} PROPERTIES
    VERSION "${DNNL_GRAPH_VERSION_MAJOR}.${DNNL_GRAPH_VERSION_MINOR}"
    SOVERSION "${DNNL_GRAPH_VERSION_MAJOR}"
    )

if (UNIX AND NOT APPLE)
    # Not every compiler adds -ldl automatically
    list(APPEND DNNL_GRAPH_EXTRA_SHARED_LIBS "${CMAKE_DL_LIBS}")
endif()

if(DNNL_GRAPH_LIBRARY_TYPE STREQUAL "SHARED")
    list(APPEND DNNL_GRAPH_EXTRA_SHARED_LIBS ${${LIB_NAME}_EXTRA_INTERFACE})
else()
    list(APPEND DNNL_GRAPH_EXTRA_STATIC_LIBS ${${LIB_NAME}_EXTRA_INTERFACE})
endif()

target_link_libraries_build(${LIB_NAME}
    "${DNNL_GRAPH_EXTRA_SHARED_LIBS};${DNNL_GRAPH_EXTRA_STATIC_LIBS}")
target_link_libraries_install(${LIB_NAME} "${DNNL_GRAPH_EXTRA_SHARED_LIBS}")
if(DNNL_LIBRARY_TYPE STREQUAL "STATIC")
    target_link_libraries_install(${LIB_NAME} "${DNNL_GRAPH_EXTRA_STATIC_LIBS}")
endif()

target_include_directories(${LIB_NAME}
    PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/../include>
    $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/${CMAKE_INSTALL_INCLUDEDIR}>
    )

set(DNNL_GRAPH_EXPORT_NAME "${LIB_NAME}-targets")

foreach(header ${HEADERS_ONEAPI})
    install(FILES ${header} DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/oneapi/dnnl/")
endforeach()

set(DNNL_GRAPH_CONFIG_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/${LIB_NAME}")

install(TARGETS ${LIB_NAME}
    EXPORT ${DNNL_GRAPH_EXPORT_NAME}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    PUBLIC_HEADER DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
    )

configure_package_config_file(../dnnlgraphConfig.cmake.in
    ${PROJECT_BINARY_DIR}/${LIB_NAME}-config.cmake
    INSTALL_DESTINATION ${DNNL_GRAPH_CONFIG_INSTALL_DIR}
    )

write_basic_package_version_file(
    ${PROJECT_BINARY_DIR}/${LIB_NAME}-config-version.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
    )

install(FILES
    ${PROJECT_BINARY_DIR}/${LIB_NAME}-config.cmake
    ${PROJECT_BINARY_DIR}/${LIB_NAME}-config-version.cmake
    DESTINATION ${DNNL_GRAPH_CONFIG_INSTALL_DIR}
    )
