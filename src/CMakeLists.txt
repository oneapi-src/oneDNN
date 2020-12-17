#===============================================================================
# Copyright 2020 Intel Corporation
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

file(GLOB HEADERS_LLGA
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/llga/*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/llga/*.hpp
    )

file(GLOB HEADERS_ONEAPI
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/oneapi/dnnl/*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/oneapi/dnnl/*.hpp
    )


file(GLOB SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/interface/*.[ch]pp
    ${CMAKE_CURRENT_SOURCE_DIR}/backend/*.[ch]pp
    ${CMAKE_CURRENT_SOURCE_DIR}/backend/dnnl/*.[ch]pp
    ${CMAKE_CURRENT_SOURCE_DIR}/backend/pass/*.[ch]pp
    ${CMAKE_CURRENT_SOURCE_DIR}/backend/pass/passes/*.[ch]pp
)

add_library(dnnl_graph_objlib OBJECT ${SOURCES})
set_target_properties(dnnl_graph_objlib PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(dnnl_graph_objlib PROPERTIES CXX_VISIBILITY_PRESET "hidden")
set_target_properties(dnnl_graph_objlib PROPERTIES VISIBILITY_INLINES_HIDDEN 1)
target_compile_options(dnnl_graph_objlib PRIVATE ${DNNL_GRAPH_COMMON_COMPILATION_FLAGS})
if (NOT DNNL_GRAPH_VERBOSE)
    target_compile_definitions(dnnl_graph_objlib PRIVATE DNNL_GRAPH_VERBOSE_DISABLE)
endif()

target_include_directories(dnnl_graph_objlib
    SYSTEM PRIVATE
    ${PROJECT_SOURCE_DIR}/third_party/oneDNN/include
    ${PROJECT_BINARY_DIR}/third_party/oneDNN/include
    PRIVATE
    ${PROJECT_SOURCE_DIR}/src
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    )

add_library(dnnl_graph_static STATIC $<TARGET_OBJECTS:dnnl_graph_objlib>)
add_library(dnnl_graph SHARED $<TARGET_OBJECTS:dnnl_graph_objlib>)

set_target_properties(dnnl_graph_static PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}
    )

set_target_properties(dnnl_graph PROPERTIES
    VERSION "${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}"
    SOVERSION "${PROJECT_VERSION_MAJOR}"
    )

if(UNIX)
if(APPLE)
    set(DNNL_GRAPH_STATIC_PRE_LINK "-Wl,-force_load")
    set(DNNL_GRAPH_STATIC_POST_LINK "")
else()
if(DNNL_GRAPH_CPU_RUNTIME STREQUAL "OMP")
    find_package(OpenMP REQUIRED)
    set(OPENMP_FLAG ${OPENMP_FLAG} OpenMP::OpenMP_CXX) # Requires CMake 3.9+
endif()
    set(DNNL_GRAPH_STATIC_PRE_LINK "-Wl,--whole-archive")
    set(DNNL_GRAPH_STATIC_POST_LINK "-Wl,--no-whole-archive")
    set_target_properties(dnnl_graph PROPERTIES SKIP_BUILD_RPATH 1)
endif()
endif()

target_link_libraries(dnnl_graph_static
    PRIVATE
    ${DNNL_GRAPH_STATIC_PRE_LINK}
    dnnl
    ${DNNL_GRAPH_STATIC_POST_LINK}
    ${OPENMP_FLAG} # Need to placed after ${DNNL_GRAPH_STATIC_POST_LINK} to avoid duplicate symbols
    ${DNNL_GRAPH_COMMON_LINK_FLAGS} # FIXME: consider target_link_options from cmake 3.13
    )

target_include_directories(dnnl_graph_static
    PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/${CMAKE_INSTALL_INCLUDEDIR}>
    )

target_link_libraries(dnnl_graph
    PUBLIC
    ${DNNL_GRAPH_STATIC_PRE_LINK}
    dnnl
    dl
    ${DNNL_GRAPH_STATIC_POST_LINK}
    ${OPENMP_FLAG} # Need to placed after ${DNNL_GRAPH_STATIC_POST_LINK} to avoid duplicate symbols
    ${DNNL_GRAPH_SYCL_LINK_FLAGS}
    ${DNNL_GRAPH_COMMON_LINK_FLAGS} # FIXME: consider target_link_options from cmake 3.13
    )

target_include_directories(dnnl_graph
    PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/${CMAKE_INSTALL_INCLUDEDIR}>
    )

foreach(header ${HEADERS_ROOT})
    install(FILES ${header} DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/")
endforeach()

foreach(header ${HEADERS_LLGA})
    install(FILES ${header} DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/llga/")
endforeach()

foreach(header ${HEADERS_ONEAPI})
    install(FILES ${header} DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/oneapi/dnnl/")
endforeach()
