#*******************************************************************************
# Copyright 2020 Arm Ltd. and affiliates
# Copyright 2020 FUJITSU LIMITED
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
#*******************************************************************************

file(GLOB_RECURSE SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/*.[ch]
    ${CMAKE_CURRENT_SOURCE_DIR}/*.[ch]pp
    )

if(NOT DNNL_AARCH64_USE_ACL)
    file(GLOB_RECURSE ACL_FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/acl_*.[ch]
        ${CMAKE_CURRENT_SOURCE_DIR}/acl_*.[ch]pp
        )
    list(REMOVE_ITEM SOURCES ${ACL_FILES})
endif()

set(OBJ_LIB ${LIB_NAME}_cpu_aarch64)
add_library(${OBJ_LIB} OBJECT ${SOURCES})
set_property(GLOBAL APPEND PROPERTY DNNL_LIB_DEPS
    $<TARGET_OBJECTS:${OBJ_LIB}>)
