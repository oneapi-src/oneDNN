find_package(Eigen3 REQUIRED)

file(GLOB RUNNER_SOURCES "*.cpp")
file(GLOB RUNNER_HEADERS "*.h")

add_library(runner STATIC ${RUNNER_SOURCES} ${RUNNER_HEADERS})

target_include_directories(runner
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${EIGEN3_INCLUDE_DIR}
)

add_executable(runner_executable parallel_loop_runner.cpp)
target_link_libraries(runner_executable 
    PRIVATE 
    runner
    Eigen3::Eigen
)