#===============================================================================
# Copyright 2020-2024 Intel Corporation
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

# Controls testing options values and behavior
#===============================================================================

if(testing_cmake_included)
    return()
endif()
set(testing_cmake_included true)
include("cmake/options.cmake")

# Transfer string literal into a number to support nested inclusions easier
set(DNNL_TEST_SET_SMOKE "1")
set(DNNL_TEST_SET_CI "2")
set(DNNL_TEST_SET_NIGHTLY "3")

set(DNNL_TEST_SET_COVERAGE "0")
set(DNNL_TEST_SET_COVERAGE_STR "")
set(DNNL_TEST_SET_HAS_NO_CORR "0")
set(DNNL_TEST_SET_HAS_ADD_BITWISE "0")

function(check_consistency entry)
    if(NOT DNNL_TEST_SET_COVERAGE EQUAL 0)
        message(FATAL_ERROR "Two different coverage types were specified: "
                "${DNNL_TEST_SET_COVERAGE_STR} and ${entry}. Only one can be used.")
    endif()
endfunction()

foreach(entry ${DNNL_TEST_SET})
    if(entry STREQUAL "NIGHTLY")
        check_consistency(${entry})
        set(DNNL_TEST_SET_COVERAGE ${DNNL_TEST_SET_NIGHTLY})
        set(DNNL_TEST_SET_COVERAGE_STR ${entry})
    elseif(entry STREQUAL "CI")
        check_consistency(${entry})
        set(DNNL_TEST_SET_COVERAGE ${DNNL_TEST_SET_CI})
        set(DNNL_TEST_SET_COVERAGE_STR ${entry})
    elseif(entry STREQUAL "SMOKE")
        check_consistency(${entry})
        set(DNNL_TEST_SET_COVERAGE ${DNNL_TEST_SET_SMOKE})
        set(DNNL_TEST_SET_COVERAGE_STR ${entry})
    elseif(entry STREQUAL "NO_CORR")
        set(DNNL_TEST_SET_HAS_NO_CORR "1")
    elseif(entry STREQUAL "ADD_BITWISE")
        set(DNNL_TEST_SET_HAS_ADD_BITWISE "1")
    elseif(entry STREQUAL "CI_NO_CORR") # Left here for compatibility till v4.0
        set(DNNL_TEST_SET_COVERAGE ${DNNL_TEST_SET_CI})
        set(DNNL_TEST_SET_COVERAGE_STR "CI")
        set(DNNL_TEST_SET_HAS_NO_CORR "1")
        message(WARNING
                "The 'CI_NO_CORR' value of DNNL_TEST_SET option is deprecated. "
                "Use 'CI;NO_CORR' instead.")
    else()
        message(FATAL_ERROR
                "The DNNL_TEST_SET entry ${entry} is not recognized. "
                "Supported values are:"
                "NIGHTLY, CI, SMOKE, NO_CORR, ADD_BITWISE.")
    endif()
endforeach()

message(STATUS "Enabled testing coverage: ${DNNL_TEST_SET_COVERAGE_STR}")
if(DNNL_TEST_SET_HAS_NO_CORR EQUAL 1)
    message(STATUS "Enabled testing modifier: No correctness")
endif()
if(DNNL_TEST_SET_HAS_ADD_BITWISE EQUAL 1)
    message(STATUS "Enabled testing modifier: Add bitwise validation")
endif()
