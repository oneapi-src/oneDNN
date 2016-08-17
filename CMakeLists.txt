#===============================================================================
# Copyright 2016 Intel Corporation
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

cmake_minimum_required(VERSION 2.8)

set(PROJECT_NAME "MKL-DNN")
set(PROJECT_VERSION, "0.0.0.1")

set(LIB_NAME mkldnn)

project(${PROJECT_NAME} C CXX)

include("cmake/MKL.cmake")

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    set(OPENMP_FLAGS "-fopenmp=libgomp")
else()
    set(OPENMP_FLAGS "-fopenmp")
endif()

if("${CMAKE_BUILD_TYPE}" STREQUAL "")
    message(WARNING "CMAKE_BUILD_TYPE is unset, use Debug by default")
    set(CMAKE_BUILD_TYPE "Debug")
endif()

set(CCXX_WARN_FLAGS "-Wall -Werror -Wno-unknown-pragmas")
set(CMAKE_CCXX_FLAGS "${OPENMP_FLAGS} ${CCXX_WARN_FLAGS} -DMKLDNN_DLL -DMKLDNN_DLL_EXPORTS -fvisibility=internal")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${CMAKE_CCXX_FLAGS} -std=c99")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CCXX_FLAGS} -std=c++11 -fvisibility-inlines-hidden")

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -xHOST")
endif()
# set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -D_FORTIFY_SOURCE=2")

enable_testing()

include_directories(api)

add_subdirectory(src)
add_subdirectory(tests)

find_package(Doxygen)
if(DOXYGEN_FOUND)
    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/doc/Doxyfile.in
        ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
        @ONLY)
    add_custom_target(doc
        ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating API documentation with Doxygen" VERBATIM)
endif(DOXYGEN_FOUND)
