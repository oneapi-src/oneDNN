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

# This file is derived from https://github.com/oneapi-src/oneDNN/blob/master/tests/gtests/gtest/CMakeLists.txt.

# Do not modify this file

set(TARGET_NAME ${LIB_NAME}_gtest)
set(MAIN_SRC "src/gtest-all.cc" "src/gtest_main.cc")
source_group("" FILES ${MAIN_SRC})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/..)

find_package(Threads REQUIRED)
add_library(${TARGET_NAME} STATIC ${MAIN_SRC})

# XXX: Link pthread explicitly: C and C++ compiler may have different flags.
# In that case FindThreads module may skip adding any flags for pthread library
# because they are not needed for C compiler but this may led to issues with
# C++ compiler which is not checked.
if(UNIX OR MINGW)
    target_link_libraries(${TARGET_NAME} pthread)
endif()
