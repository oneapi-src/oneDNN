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

set(CMAKE_CCXX_FLAGS "-g ${OPENMP_FLAGS} -Wall -Werror -Wno-unknown-pragmas")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${CMAKE_CCXX_FLAGS} -std=c99")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CCXX_FLAGS} -std=c++11")

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
