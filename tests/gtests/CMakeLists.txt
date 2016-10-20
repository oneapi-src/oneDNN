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

set(APP_NAME "gtest")
set(MAIN_SRC_GTEST main.cpp)
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/tests/gtests
                    ${CMAKE_SOURCE_DIR}/tests/gtests/gtest
                    )
# TODO: enable me!
#file(GLOB API_TEST_CASES_SRC api_tests/*.cpp)
#file(GLOB PRIM_TEST_CASES_SRC test_*.cpp)
file(GLOB PRIM_TEST_CASES_SRC
                              test_lrn_forward.cpp
                              test_lrn_backward.cpp
                              test_pooling_forward.cpp
                              test_pooling_backward.cpp
                              test_relu_forward.cpp
                              test_relu_backward.cpp
                              test_batch_normalization_forward.cpp
                              test_batch_normalization_backward.cpp
                              test_concat.cpp
                              test_inner_product.cpp
                              test_convolution_forward.cpp
                              test_convolution_relu_forward.cpp
                              test_convolution_backward_data.cpp
                              test_convolution_backward_weights.cpp
                              test_convolution_format_any.cpp
                              ) #temporary

foreach(TEST_FILE ${PRIM_TEST_CASES_SRC})
    get_filename_component (TEST_FILE_WE ${TEST_FILE} NAME_WE)
    set(CURR_APP_NAME ${TEST_FILE_WE})
    add_executable(${CURR_APP_NAME} ${MAIN_SRC_GTEST} ${TEST_FILE})
    target_link_libraries(${CURR_APP_NAME} mkldnn gtest)
    add_test(${CURR_APP_NAME} ${CURR_APP_NAME})
endforeach()

# we need to have either this or the add_test above; disabling for now...
# add_custom_command(TARGET ${APP_NAME}
#                   POST_BUILD
#                   WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
#                   COMMAND ${APP_NAME} )


