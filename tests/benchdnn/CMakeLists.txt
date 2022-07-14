#===============================================================================
# Copyright 2017-2022 Intel Corporation
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

file(GLOB_RECURSE HEADERS
    ${PROJECT_SOURCE_DIR}/third_party/oneDNN/include/*.h
    ${PROJECT_SOURCE_DIR}/third_party/oneDNN/include/*.hpp
    )

file(GLOB_RECURSE SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
    )

list(APPEND SOURCES ${TEST_THREAD})
list(APPEND SOURCES ${GRAPH_TEST_THREAD})
list(APPEND SOURCES ${TEST_ALLOCATOR})
include_directories_with_host_compiler(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/third_party/oneDNN
    ${PROJECT_SOURCE_DIR}/third_party/oneDNN/src
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/tests/
    )

if(BENCHDNN_USE_RDPMC)
    add_definitions_with_host_compiler(-DBENCHDNN_USE_RDPMC)
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
    append_if(WIN32 CMAKE_CXX_FLAGS "-Qprec-div -Qprec-sqrt")
    append_if(UNIX  CMAKE_CXX_FLAGS "-prec-div -prec-sqrt -fp-model precise")
endif()

if(UNIX AND NOT APPLE AND NOT QNXNTO)
    find_library(LIBRT rt)
elseif(QNXNTO)
    find_library(LIBREGEX regex)
    find_library(LIBSOCKET socket)
endif()
register_exe(benchdnn "${SOURCES}" "" "${LIBRT};${LIBREGEX};${LIBSOCKET}")

file(COPY inputs DESTINATION .)

if(WIN32 AND (NOT DNNL_GRAPH_BUILD_FOR_CI))
    string(REPLACE  ";" "\;" PATH "${CTESTCONFIG_PATH};$ENV{PATH}")
    configure_file(
        "${PROJECT_SOURCE_DIR}/third_party/oneDNN/cmake/run_with_env.bat.in"
        "${PROJECT_BINARY_DIR}/third_party/oneDNN/run_with_env.bat"
    )
endif()

# Set the target architecture.
if(NOT DNNL_TARGET_ARCH)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64.*|AARCH64.*|arm64.*|ARM64.*)")
        set(DNNL_TARGET_ARCH "AARCH64")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(ppc64.*|PPC64.*|powerpc64.*)")
        set(DNNL_TARGET_ARCH "PPC64")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(s390x.*|S390X.*)")
        set(DNNL_TARGET_ARCH "S390X")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(rv.*|RV.*|riscv.*|RISCV.*)")
        set(DNNL_TARGET_ARCH "RV64")
    else()
        set(DNNL_TARGET_ARCH "X64")
    endif()
endif()
message(STATUS "DNNL_TARGET_ARCH: ${DNNL_TARGET_ARCH}")

# Specify the target architecture
add_definitions_with_host_compiler(-DDNNL_${DNNL_TARGET_ARCH}=1)

function(register_benchdnn_test engine driver test_file)
    if (ARGC GREATER 3)
        message(ERROR "Incorrect use of function")
    endif()

    set(test_mode "C")
    if(DNNL_TEST_SET EQUAL DNNL_TEST_SET_CI_NO_CORR)
        set(test_mode "R")
    endif()

    string(REPLACE "test_" "test_benchdnn_" target_name ${test_file})
    set(cmd "--mode=${test_mode} -v1 --engine=${engine} --${driver} --batch=${test_file}")
    set(benchdnn_target ${target_name}_${engine})

    if(DNNL_GRAPH_BUILD_FOR_CI)
        string(REPLACE " " ";" cmd "benchdnn ${cmd}")
        add_dnnl_graph_test(${benchdnn_target} ${cmd})
    else()
        string(REPLACE " " ";" cmd "$<TARGET_FILE:benchdnn> ${cmd}")

        if(WIN32)
            set(cmd "cmd;/c;${PROJECT_BINARY_DIR}/run_with_env.bat;${cmd}")
            set(ARGV2 "cmd;/c;${PROJECT_BINARY_DIR}/run_with_env.bat;${ARGV2}")
        endif()

        add_custom_target(${benchdnn_target}
            COMMAND ${cmd}
            DEPENDS benchdnn
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        )

        set_target_properties(${benchdnn_target} PROPERTIES
            EXCLUDE_FROM_DEFAULT_BUILD TRUE)
        maybe_configure_windows_test(${benchdnn_target} TARGET)

        # Create non-suffixed target for compatibility
        if (engine STREQUAL "cpu")
            add_custom_target(${target_name} DEPENDS ${benchdnn_target})
            maybe_configure_windows_test(${target_name} TARGET)
        endif()
    endif()
endfunction()


set(stack_checker_pattern "^(bnorm|concat|conv|eltwise|ip|lrn|matmul|pool|reorder|softmax|sum)$")

function(register_all_tests engine driver test_files)
    if(DNNL_ENABLE_STACK_CHECKER AND NOT ${driver} MATCHES ${stack_checker_pattern})
        return()
    endif()

    foreach(test_file ${test_files})
        register_benchdnn_test(${engine} ${driver} ${test_file})
    endforeach()
endfunction()

# The following section is responsible to register test_benchdnn_ targets
# depending on DNNL_GRAPH_TEST_SET value.
set(has_gpu false)
if(NOT DNNL_GRAPH_GPU_RUNTIME STREQUAL "NONE")
    set(has_gpu true)
endif()
set(has_cpu false)
if(NOT DNNL_CPU_RUNTIME STREQUAL "NONE")
    set(has_cpu true)
endif()


# Very sad CMake older then 3.2 does not support continue() command.
# file(GLOB all_drivers RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/inputs inputs/*)  # TODO: open dnnl test
foreach(driver ${all_drivers})
    set(driver_dir ${CMAKE_CURRENT_SOURCE_DIR}/inputs/${driver})
    # Collect input files in groups
    file(GLOB test_files_large_cpu RELATIVE ${driver_dir}
            inputs/${driver}/test_*_large$)
    file(GLOB test_files_gpu_ci RELATIVE ${driver_dir}
            inputs/${driver}/test_*_gpu_ci)
    file(GLOB test_files_gpu RELATIVE ${driver_dir} inputs/${driver}/test_*_gpu)
    file(GLOB test_files_ci RELATIVE ${driver_dir} inputs/${driver}/test_*_ci)
    file(GLOB test_files_cpu RELATIVE ${driver_dir} inputs/${driver}/test_*)

    # Register ci input files for ci only
    if(DNNL_GRAPH_TEST_SET EQUAL DNNL_GRAPH_TEST_SET_CI)
        # gpu_ci files may happen if cpu coverage can not be used on gpu
        # Filter out gpu_ci inputs from ci
        foreach(test_file ${test_files_gpu_ci})
            string(REPLACE "${test_file}" "" test_files_ci "${test_files_ci}")
        endforeach()

        # use gpu_ci if not empty
        if(has_gpu)
            if(test_files_gpu_ci)
                register_all_tests(gpu "${driver}" "${test_files_gpu_ci}")
            else()
                register_all_tests(gpu "${driver}" "${test_files_ci}")
            endif()
        endif()
        if(has_cpu)
            register_all_tests(cpu "${driver}" "${test_files_ci}")
        endif()
    endif()

    # Register nightly input files
    if(DNNL_GRAPH_TEST_SET GREATER DNNL_GRAPH_TEST_SET_CI)
        ## Filter out gpu, large cpu and invalid inputs from cpu
        foreach(test_file ${test_files_large_cpu} ${test_files_gpu_ci}
                ${test_files_gpu} ${test_files_ci} "test_conv_all;")
            string(REPLACE "${test_file}" "" test_files_cpu "${test_files_cpu}")
        endforeach()

        if(has_cpu AND NOT DNNL_USE_CLANG_SANITIZER)
            register_all_tests(cpu "${driver}" "${test_files_large_cpu}")
        endif()
        if(has_gpu)
            register_all_tests(gpu "${driver}" "${test_files_gpu}")
        endif()
        if(has_cpu)
            register_all_tests(cpu "${driver}" "${test_files_cpu}")
        endif()
    endif()
endforeach()

# process ctest configurations of benchdnn extension for Graph API
file(COPY inputs_graph DESTINATION .)

function(register_benchdnnext_test engine bench_mode api_mode driver
        test_file)
    if (ARGC GREATER 5)
        message(FATAL_ERROR "Incorrect use of function")
    endif()

    if(bench_mode STREQUAL "P" OR bench_mode STREQUAL "p")
        set(TEST_NAME_PREFIX "perf")
    elseif(bench_mode STREQUAL "C" OR bench_mode STREQUAL "c")
        set(TEST_NAME_PREFIX "test")
    else()
        message(FATAL_ERROR "Unkown bench mode - " ${bench_mode})
    endif()

    if(api_mode STREQUAL "P" OR api_mode STREQUAL "p")
        set(TEST_NAME_PREFIX "${TEST_NAME_PREFIX}_benchdnn_")
    elseif (api_mode STREQUAL "G" OR api_mode STREQUAL "g")
        set(TEST_NAME_PREFIX "${TEST_NAME_PREFIX}_benchdnnext_")
    else()
        message(FATAL_ERROR "Unknown api mode - " ${api_mode})
    endif()

    string(REPLACE "test_" "${TEST_NAME_PREFIX}" target_name ${test_file})

    set(cmd "-v1 --engine=${engine} --mode=${bench_mode} --api=${api_mode}"
            "--${driver}"
            "--perf-template=perf,%engine%,%impl%,%name%,%prb%,%Gops%,%Gfreq%,%-time%,%-Gflops%,%0time%,%0Gflops%"
            "--batch=${test_file}")
    set(benchdnn_target ${target_name}_${engine})

    if(DNNL_GRAPH_BUILD_FOR_CI)
        string(REPLACE " " ";" cmd "benchdnn ${cmd}")
        add_dnnl_graph_test(${benchdnn_target} ${cmd})
    else()
        string(REPLACE " " ";" cmd "$<TARGET_FILE:benchdnn> ${cmd}")

        if(WIN32)
            set(cmd "cmd;/c;${PROJECT_BINARY_DIR}/run_with_env.bat;${cmd}")
            set(ARGV2 "cmd;/c;${PROJECT_BINARY_DIR}/run_with_env.bat;${ARGV2}")
        endif()

        add_custom_target(${benchdnn_target}
            COMMAND ${cmd}
            DEPENDS benchdnn
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        )

        set_target_properties(${benchdnn_target} PROPERTIES
            EXCLUDE_FROM_DEFAULT_BUILD TRUE)
        maybe_configure_windows_test(${benchdnn_target} TARGET)

        # Create non-suffixed target for compatibility
        if(engine STREQUAL "cpu")
            add_custom_target(${target_name} DEPENDS ${benchdnn_target})
            maybe_configure_windows_test(${target_name} TARGET)
        endif()
    endif()
endfunction()

function(register_all_graph_tests engine bench_mode api_mode driver test_files)
    foreach(test_file ${test_files})
        register_benchdnnext_test(${engine} ${bench_mode} ${api_mode}
                ${driver} ${test_file})
    endforeach()
endfunction()

# Register test_graph_benchdnn_ targets
file(GLOB all_graph_drivers RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/inputs_graph
        inputs_graph/*)
foreach(driver ${all_graph_drivers})
    set(driver_dir ${CMAKE_CURRENT_SOURCE_DIR}/inputs_graph/${driver})
    # Collect input files in groups.
    # TODO: At present, we don't have many test cases that's in different
    #   groups, like `large`, `ci`, `gpu`, and `cpu`.
    file(GLOB test_files_large_cpu RELATIVE ${driver_dir}
            inputs_graph/${driver}/test_*_large)
    file(GLOB test_files_gpu_ci RELATIVE ${driver_dir}
            inputs_graph/${driver}/test_*_gpu_ci)
    file(GLOB test_files_gpu RELATIVE ${driver_dir}
            inputs_graph/${driver}/test_*_gpu)
    file(GLOB test_files_ci RELATIVE ${driver_dir}
            inputs_graph/${driver}/test_*_ci)
    file(GLOB test_files_main RELATIVE ${driver_dir}
            inputs_graph/${driver}/test_graph_${driver})

    # For cpu build option only test *_ci(test_graph_${driver}) except *_gpu_ci;
    # For gpu build option only test *_gpu_ci
    if(DNNL_GRAPH_TEST_SET EQUAL DNNL_GRAPH_TEST_SET_CI)
        # gpu_ci files may happen if cpu coverage can not be used on gpu
        # Filter out gpu_ci inputs from ci
        foreach(test_file ${test_files_gpu_ci})
            string(REPLACE "${test_file}" "" test_files_ci "${test_files_ci}")
        endforeach()

        # use gpu_ci if not empty
        if(has_gpu)
            if(test_files_gpu_ci)
                register_all_graph_tests(gpu "C" "G" "${driver}" "${test_files_gpu_ci}")
            endif()
        else()
            register_all_graph_tests(cpu "C" "G" "${driver}" "${test_files_main}")
        endif()
    endif()

    # nightly: test_grpah_xxx(xxx represents driver name, include *.ci except *_gpu_ci) files used for both cpu and gpu.
    if(DNNL_GRAPH_TEST_SET GREATER DNNL_GRAPH_TEST_SET_CI)
        # special case1：conv use specific _gpu(depthwise test for cpu only) test cases for nightly.
        string(REPLACE "test_graph_conv" "test_graph_conv_gpu" test_files_main "${test_files_main}")
        # special case2：deconv use specific _gpu(exp post-ops for cpu only) test cases for nightly.
        # TODO: remove this line after onednn fix the bug.
        string(REPLACE "test_graph_deconv" "test_graph_deconv_gpu" test_files_main "${test_files_main}")

        if(has_gpu)
            register_all_graph_tests(gpu "C" "G" "${driver}" "${test_files_main}")
        endif()
        if(has_cpu)
            register_all_graph_tests(cpu "C" "G" "${driver}" "${test_files_main}")
        endif()
    endif()
endforeach()
