# Do not modify this file

set(TARGET_NAME gtest)
set(MAIN_SRC "src/gtest-all.cc")
source_group("" FILES ${MAIN_SRC})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}
                    ${CMAKE_CURRENT_SOURCE_DIR}/..
                    ${CMAKE_CURRENT_SOURCE_DIR}/gtest)
find_package(Threads REQUIRED)
add_library(${TARGET_NAME} STATIC ${MAIN_SRC})
target_link_libraries(${TARGET_NAME} ${CMAKE_THREAD_LIBS_INIT})
