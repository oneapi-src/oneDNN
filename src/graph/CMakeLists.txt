#===============================================================================
# Copyright 2020-2022 Intel Corporation
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

if(DNNL_BUILD_GRAPH)
    message(STATUS "Graph component is enabled")
    add_definitions_with_host_compiler(-DBUILD_GRAPH)
    if(DNNL_ENABLE_GRAPH_DUMP)
        message(STATUS "Graph artifacts dump is enabled")
        add_definitions_with_host_compiler(-DDNNL_ENABLE_GRAPH_DUMP)
    endif()
endif()

add_subdirectory(interface)

if (DNNL_BUILD_GRAPH)
    add_subdirectory(backend)
    add_subdirectory(utils)
endif()
