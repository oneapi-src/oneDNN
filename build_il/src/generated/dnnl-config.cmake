#===============================================================================
# Copyright 2019-2023 Intel Corporation
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


####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run ####
####### The input file was config.cmake.in                            ########

get_filename_component(PACKAGE_PREFIX_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../" ABSOLUTE)

macro(set_and_check _var _file)
  set(${_var} "${_file}")
  if(NOT EXISTS "${_file}")
    message(FATAL_ERROR "File or directory ${_file} referenced by variable ${_var} does not exist !")
  endif()
endmacro()

macro(check_required_components _NAME)
  foreach(comp ${${_NAME}_FIND_COMPONENTS})
    if(NOT ${_NAME}_${comp}_FOUND)
      if(${_NAME}_FIND_REQUIRED_${comp})
        set(${_NAME}_FOUND FALSE)
      endif()
    endif()
  endforeach()
endmacro()

####################################################################################
set(DNNL_CPU_RUNTIME "OMP")
set(DNNL_CPU_THREADING_RUNTIME "OMP")
set(DNNL_GPU_RUNTIME "NONE")

set(DNNL_BLAS_VENDOR "NONE")

if(DNNL_CPU_THREADING_RUNTIME STREQUAL "TBB")
    # Try to find TBB using a TBB-provided CMake config file.
    find_package(TBB QUIET COMPONENTS tbb)
    # If the previous `find_package` call failed then try to
    # use a TBB CMake config file that is maintained by oneDNN.
    # The reason the previous call may fail is that TBB package is
    # very old and doesn't provide a CMake config file.
    if(NOT TBB_FOUND)
        message(STATUS "TBB-provided CMake config either failed or was not found. Trying to use a custom one.")
        # Use a custom find module for transitive dependencies
        set(DNNL_ORIGINAL_CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH})
        list(INSERT CMAKE_MODULE_PATH 0 ${PACKAGE_PREFIX_DIR}/lib/cmake/dnnl)
        find_package(TBB REQUIRED COMPONENTS tbb)
        # Reverting the CMAKE_MODULE_PATH to its original state
        set(CMAKE_MODULE_PATH ${DNNL_ORIGINAL_CMAKE_MODULE_PATH})
    endif()
endif()

# Use a custom find module for transitive dependencies
set(DNNL_ORIGINAL_CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH})
list(INSERT CMAKE_MODULE_PATH 0 ${PACKAGE_PREFIX_DIR}/lib/cmake/dnnl)

if(DNNL_GPU_RUNTIME STREQUAL "OCL")
    find_package(OpenCL REQUIRED)
    set(DNNL_COMPILE_FLAGS "-DCL_TARGET_OPENCL_VERSION=220")
endif()

if(DNNL_BLAS_VENDOR STREQUAL "ACCELERATE")
    set(BLA_VENDOR "Apple")
    find_package(BLAS REQUIRED)
    unset(BLA_VENDOR)
endif()

# Reverting the CMAKE_MODULE_PATH to its original state
set(CMAKE_MODULE_PATH ${DNNL_ORIGINAL_CMAKE_MODULE_PATH})

include("${CMAKE_CURRENT_LIST_DIR}/dnnl-targets.cmake")
check_required_components("dnnl")

if(DNNL_CPU_RUNTIME STREQUAL "SYCL" OR DNNL_CPU_RUNTIME STREQUAL "DPCPP" OR
   DNNL_GPU_RUNTIME STREQUAL "SYCL" OR DNNL_GPU_RUNTIME STREQUAL "DPCPP")
    set(DNNL_COMPILE_FLAGS "-fsycl")
    
endif()
