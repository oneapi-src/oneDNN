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

# Configure documentation generation

if(Doxygen_cmake_included)
    return()
endif()
set(Doxygen_cmake_included true)

find_package(Doxygen)
if (NOT DOXYGEN_FOUND)
    message(STATUS "Cannot find Doxygen package")
    return()
endif()

function(build_doc)
    set(DOXYGEN_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/reference)
    set(DOXYGEN_STAMP_FILE ${CMAKE_CURRENT_BINARY_DIR}/doc.stamp)
    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/doc/Doxyfile.in
        ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
        @ONLY
    )
    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/doc/header.html.in
        ${CMAKE_CURRENT_BINARY_DIR}/header.html
        @ONLY
    )
    file(COPY
        ${CMAKE_CURRENT_SOURCE_DIR}/doc/footer.html
        DESTINATION ${CMAKE_CURRENT_BINARY_DIR}
    )
    file(COPY
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/oneDNN/doc/dnnl.js
        DESTINATION ${DOXYGEN_OUTPUT_DIR}/html/assets/mathjax/config/
    )
    file(GLOB_RECURSE HEARDERS
        ${PROJECT_SOURCE_DIR}/include/oneapi/dnnl/*.h
        ${PROJECT_SOURCE_DIR}/include/oneapi/dnnl/*.hpp
    )
    file(GLOB_RECURSE DOX
        ${PROJECT_SOURCE_DIR}/doc/*
    )
    file(GLOB_RECURSE EXAMPLES
        ${PROJECT_SOURCE_DIR}/examples/c/src/*
        ${PROJECT_SOURCE_DIR}/examples/cpp/src/*
    )
    add_custom_command(
        OUTPUT ${DOXYGEN_STAMP_FILE}
        DEPENDS ${HEARDERS} ${DOX} ${EXAMPLES}
        COMMAND ${DOXYGEN_EXECUTABLE} Doxyfile
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/third_party/oneDNN/doc/assets ${DOXYGEN_OUTPUT_DIR}/html/assets
        COMMAND ${CMAKE_COMMAND} -E touch ${DOXYGEN_STAMP_FILE}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating API documentation with Doxygen" VERBATIM)

    add_custom_target(doc DEPENDS ${DOXYGEN_STAMP_FILE})

    install(DIRECTORY ${DOXYGEN_OUTPUT_DIR}
        DESTINATION share/doc/${LIB_NAME} OPTIONAL)
endfunction()

build_doc()
