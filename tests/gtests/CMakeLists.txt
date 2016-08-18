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

add_subdirectory (gtest)

set(APP_NAME "tests_gtest")
set(MAIN_SRC_GTEST main.cpp)
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/tests/gtests
                    ${CMAKE_SOURCE_DIR}/tests/gtests/gtest
                    )
file(GLOB API_TEST_CASES_SRC api_tests/*.cpp)
file(GLOB PRIM_TEST_CASES_SRC test_*.cpp)
add_executable(${APP_NAME}
    ${MAIN_SRC_GTEST}
    ${API_TEST_CASES_SRC}
    ${PRIM_TEST_CASES_SRC}
    )
target_link_libraries(${APP_NAME} mkldnn gtest)
add_test(${APP_NAME} ${APP_NAME})

# we need to have either this or the add_test above; disabling for now...
# add_custom_command(TARGET ${APP_NAME}
#                   POST_BUILD
#                   WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
#                   COMMAND ${APP_NAME} )


