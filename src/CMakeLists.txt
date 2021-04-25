#===============================================================================
# Copyright 2020-2021 Intel Corporation
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

file(GLOB HEADERS_ROOT
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/*.hpp
    )

file(GLOB HEADERS_ONEAPI
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/oneapi/dnnl/*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/oneapi/dnnl/*.hpp
    )

file(GLOB SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/interface/*.[ch]pp
    ${CMAKE_CURRENT_SOURCE_DIR}/backend/*.[ch]pp
    ${CMAKE_CURRENT_SOURCE_DIR}/backend/dnnl/*.[ch]pp
    ${CMAKE_CURRENT_SOURCE_DIR}/backend/fake/*.[ch]pp
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/*.[ch]pp
    ${CMAKE_CURRENT_SOURCE_DIR}/pm/*.[ch]pp
)

if(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
    set_source_files_properties(${SOURCES}
        PROPERTIES COMPILE_FLAGS "-prec-sqrt")
    set_source_files_properties(${SOURCES}
        PROPERTIES COMPILE_FLAGS "-prec-div")
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

add_library(${LIB_NAME}_objlib OBJECT ${SOURCES})

APPLY_VERSION_DEFINITIONS(${CMAKE_CURRENT_SOURCE_DIR}/utils/verbose.cpp)

if (NOT DNNL_GRAPH_LOGGER)
    message(STATUS "DNNL Graph logger is disabled")
    target_compile_definitions(${LIB_NAME}_objlib PRIVATE DNNL_GRAPH_DISABLE_LOGGER)
endif()

if(NOT DNNL_GRAPH_VERBOSE)
    message(STATUS "DNNL Graph verbose mode is disabled")
    target_compile_definitions(${LIB_NAME}_objlib PRIVATE DNNL_GRAPH_DISABLE_VERBOSE)
endif()

target_include_directories(${LIB_NAME}_objlib
    PRIVATE
    ${PROJECT_SOURCE_DIR}/third_party/oneDNN/include
    ${PROJECT_BINARY_DIR}/third_party/oneDNN/include
    PRIVATE
    ${PROJECT_SOURCE_DIR}/src
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    )

add_library(${LIB_NAME}_static STATIC $<TARGET_OBJECTS:${LIB_NAME}_objlib>)
add_library(${LIB_NAME} SHARED $<TARGET_OBJECTS:${LIB_NAME}_objlib>)

set_target_properties(${LIB_NAME}_static PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}
    )

set_target_properties(${LIB_NAME} PROPERTIES
    VERSION "${DNNL_GRAPH_VERSION_MAJOR}.${DNNL_GRAPH_VERSION_MINOR}"
    SOVERSION "${DNNL_GRAPH_VERSION_MAJOR}"
    )

if(UNIX)
    if(APPLE)
        set(DNNL_GRAPH_STATIC_PRE_LINK "-Wl,-force_load")
        set(DNNL_GRAPH_STATIC_POST_LINK "")
    else()
        set(DNNL_GRAPH_STATIC_PRE_LINK "-Wl,--whole-archive")
        set(DNNL_GRAPH_STATIC_POST_LINK "-Wl,--no-whole-archive")
        set_target_properties(dnnl_graph PROPERTIES SKIP_BUILD_RPATH 1)
    endif()
endif()

if (UNIX AND NOT APPLE)
    # Not every compiler adds -ldl automatically
    list(APPEND EXTRA_SHARED_LIBS "${CMAKE_DL_LIBS}")
endif()

target_link_libraries(${LIB_NAME}_static
    LINK_PRIVATE
    ${DNNL_GRAPH_STATIC_PRE_LINK}
    dnnl
    ${DNNL_GRAPH_STATIC_POST_LINK}
    ${DNNL_GRAPH_COMMON_LINK_FLAGS}
    )

target_include_directories(${LIB_NAME}_static
    PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/${CMAKE_INSTALL_INCLUDEDIR}>
    )

target_link_libraries(${LIB_NAME}
    LINK_PUBLIC
    ${DNNL_GRAPH_STATIC_PRE_LINK}
    dnnl
    ${EXTRA_SHARED_LIBS}
    ${DNNL_GRAPH_STATIC_POST_LINK}
    ${DNNL_GRAPH_SYCL_LINK_FLAGS}
    ${DNNL_GRAPH_COMMON_LINK_FLAGS}
    )

target_include_directories(${LIB_NAME}
    PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/${CMAKE_INSTALL_INCLUDEDIR}>
    )

set(DNNL_GRAPH_EXPORT_NAME "${LIB_NAME}-targets")
install(TARGETS ${LIB_NAME} ${LIB_NAME}_static
    EXPORT ${DNNL_GRAPH_EXPORT_NAME}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    PUBLIC_HEADER DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
    )

foreach(header ${HEADERS_ONEAPI})
    install(FILES ${header} DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/oneapi/dnnl/")
endforeach()

set(DNNL_GRAPH_CONFIG_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/${LIB_NAME}")
install(EXPORT ${DNNL_GRAPH_EXPORT_NAME}
    FILE ${LIB_NAME}-targets.cmake
    NAMESPACE ${LIB_NAME}::
    DESTINATION ${DNNL_GRAPH_CONFIG_INSTALL_DIR}
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
