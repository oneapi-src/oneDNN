set(TARGET_NAME ${LIB_NAME})
file(GLOB_RECURSE HEADERS
    ${CMAKE_SOURCE_DIR}/api/*.h
    ${CMAKE_SOURCE_DIR}/api/*.hpp
    )
file(GLOB_RECURSE SOURCES
    ${CMAKE_SOURCE_DIR}/src/*.c
    ${CMAKE_SOURCE_DIR}/src/*.cpp
    ${CMAKE_SOURCE_DIR}/src/*.h
    ${CMAKE_SOURCE_DIR}/src/*.hpp
    )
include_directories(
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/common
    ${CMAKE_SOURCE_DIR}/src/cpu/xbyak
    )
add_library(${TARGET_NAME} SHARED ${HEADERS} ${SOURCES})
set_property(TARGET ${TARGET_NAME} PROPERTY CXX_STANDARD 11)
set_property(TARGET ${TARGET_NAME} PROPERTY CXX_STANDARD_REQUIRED ON)
