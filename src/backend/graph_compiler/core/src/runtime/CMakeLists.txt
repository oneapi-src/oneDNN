#===============================================================================
# Copyright 2022-2023 Intel Corporation
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

file(GLOB_RECURSE SC_RUNTIME_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/*.c
    )
if(${SC_PRODUCTION} STREQUAL ON)
    list(REMOVE_ITEM SC_RUNTIME_SOURCES ${SC_SKIP_SOURCES})
endif()
add_library(graphcompiler_runtime STATIC ${SC_RUNTIME_SOURCES})
target_link_libraries(graphcompiler_runtime PRIVATE ${DNNL_LIBNAME} "${EXTRA_SHARED_LIBS}")
if(IGC_PATH)
    target_link_libraries(graphcompiler_runtime PRIVATE ze_loader)
endif()
if(DEFINED SC_SOURCES)
    install(TARGETS graphcompiler_runtime
        EXPORT graphcompiler_runtime_export
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()

