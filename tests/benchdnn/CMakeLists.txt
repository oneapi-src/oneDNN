#===============================================================================
# Copyright 2017 Intel Corporation
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

set(TARGET_NAME ${LIB_NAME})

file(GLOB_RECURSE HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/*.hpp
    )
file(GLOB_RECURSE SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp
    )
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/conv
    ${CMAKE_CURRENT_SOURCE_DIR}/ip
    )

add_executable(benchdnn ${SOURCES})
set_property(TARGET benchdnn PROPERTY CXX_STANDARD 11)
target_link_libraries(benchdnn ${LIB_NAME})
# add_test(benchdnn benchdnn)

file(COPY inputs DESTINATION .)
add_custom_target(test_conv
    COMMAND benchdnn --conv --batch=inputs/test_conv_all
    DEPENDS benchdnn
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
add_custom_target(test_conv_regression
    COMMAND benchdnn --conv --batch=inputs/test_conv_regression
    DEPENDS benchdnn
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
