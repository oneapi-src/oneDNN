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

# workaround for icx cannot init constexpr with dllimport symbols(detail::handle)
# also, suppose it's unnecessary for users to link the library with the flag.
if (DNNL_LIBRARY_TYPE STREQUAL "SHARED")
    remove_definitions(-DDNNL_DLL)
endif()

set(TEST_ALLOCATOR ${CMAKE_CURRENT_SOURCE_DIR}/test_allocator.cpp)

include_directories_with_host_compiler(
    ${CMAKE_CURRENT_SOURCE_DIR}
)

function(register_graph_exe exe_name srcs deps)
    add_executable(${exe_name} ${srcs})
    target_link_libraries(${exe_name} ${deps})
    
    set(sycl_rt_pattern "(SYCL|DPCPP)")
    if(${exe_name} MATCHES "_usm")
        if(DNNL_GPU_RUNTIME MATCHES ${sycl_rt_pattern})
            add_test(${exe_name}_gpu ${exe_name} "--engine=gpu")
            maybe_configure_windows_test(${exe_name}_gpu TEST)
        endif()

        if(NOT DNNL_CPU_RUNTIME STREQUAL "NONE")
            add_test(${exe_name}_cpu ${exe_name} "--engine=cpu")
            maybe_configure_windows_test(${exe_name}_cpu TEST)
        endif()
    else()
        if(NOT DNNL_CPU_RUNTIME STREQUAL "NONE")
            add_test(${exe_name} ${exe_name})
            maybe_configure_windows_test(${exe_name} TEST)
        endif()
    endif()
endfunction()

add_subdirectory(api)
add_subdirectory(unit)
