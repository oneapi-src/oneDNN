#===============================================================================
# Copyright 2021-2024 Intel Corporation
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

# Locates Sphinx and configures documentation generation
#===============================================================================

if(Sphinx_cmake_included)
    return()
endif()
set(Sphinx_cmake_included true)

find_package(Python 3.7 COMPONENTS Interpreter)
find_package(Sphinx)
if (Python_FOUND AND SPHINX_FOUND)
    set(SPHINX_GENERATOR "html" CACHE STRING "specifies generator for Sphinx")

    set(SPHINX_OUTPUT_DIR
        ${CMAKE_CURRENT_BINARY_DIR}/reference/${SPHINX_GENERATOR}
    )
    set(SPHINX_STAMP_FILE ${CMAKE_CURRENT_BINARY_DIR}/sphinx.stamp)
    set(SPHINX_SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/reference/rst)
    file(MAKE_DIRECTORY ${SPHINX_OUTPUT_DIR})
    file(GLOB_RECURSE SPHINX_IMAGES
        ${CMAKE_CURRENT_SOURCE_DIR}/doc/*.png
        ${CMAKE_CURRENT_SOURCE_DIR}/doc/*.jpg
    )
    file(COPY ${SPHINX_IMAGES} DESTINATION ${SPHINX_SOURCE_DIR})
    file(COPY
        ${CMAKE_CURRENT_SOURCE_DIR}/doc/sphinx/conf.py
        DESTINATION ${SPHINX_SOURCE_DIR}
        )
    file(COPY
        ${CMAKE_CURRENT_SOURCE_DIR}/doc/sphinx/cleanup.py
        DESTINATION ${CMAKE_CURRENT_BINARY_DIR}
        )
    add_custom_command(
        OUTPUT ${SPHINX_STAMP_FILE}
        DEPENDS ${DOXYREST_STAMP_FILE} ${SPHINX_IMAGES} ${SPHINX_CLEANUP}
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_CURRENT_SOURCE_DIR}/doc/sphinx/_static
            ${SPHINX_SOURCE_DIR}/_static
        COMMAND ${Python_EXECUTABLE}
            ${CMAKE_CURRENT_BINARY_DIR}/cleanup.py ${SPHINX_SOURCE_DIR}
        COMMAND ${CMAKE_COMMAND} -E env PROJECT_VERSION=v${PROJECT_VERSION}
            ${SPHINX_EXECUTABLE} -b ${SPHINX_GENERATOR}
            -j auto rst ${SPHINX_OUTPUT_DIR}
        COMMAND ${CMAKE_COMMAND} -E touch ${SPHINX_STAMP_FILE}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/reference
        COMMENT "Generating API documentation with Sphinx" VERBATIM)
    add_custom_target(doc_sphinx DEPENDS ${SPHINX_STAMP_FILE} doc_doxyrest)
endif(Python_FOUND AND SPHINX_FOUND)
