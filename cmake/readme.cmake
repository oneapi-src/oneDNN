#===============================================================================
# Copyright 2026 Intel Corporation
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

# Generates README for binary distribution with platform-specific information
# based on DNNL_TARGET_ARCH.

if(readme_cmake_included)
    return()
endif()
set(readme_cmake_included true)

set(DNNL_ARCH_NAME "${CMAKE_SYSTEM_PROCESSOR}")
if(DNNL_TARGET_ARCH STREQUAL "X64")
    set(DNNL_ARCH_NAME "Intel 64/AMD64")
elseif(DNNL_TARGET_ARCH STREQUAL "AARCH64")
    set(DNNL_ARCH_NAME "Arm 64-bit (AArch64)")
elseif(DNNL_TARGET_ARCH STREQUAL "PPC64")
    set(DNNL_ARCH_NAME "64-bit Power ISA (PPC64)")
elseif(DNNL_TARGET_ARCH STREQUAL "S390X")
    set(DNNL_ARCH_NAME "IBMz (s390x)")
elseif(DNNL_TARGET_ARCH STREQUAL "RV64")
    set(DNNL_ARCH_NAME "RISC-V 64-bit (RV64)")
endif()

string(TOLOWER "${DNNL_TARGET_ARCH}" _dnnl_arch)
set(_dnnl_arch_readme "${PROJECT_SOURCE_DIR}/README.${_dnnl_arch}.in")
if(NOT EXISTS "${_dnnl_arch_readme}")
    set(_dnnl_arch_readme "${PROJECT_SOURCE_DIR}/README.generic.in")
endif()

set_property(DIRECTORY APPEND PROPERTY
    CMAKE_CONFIGURE_DEPENDS "${_dnnl_arch_readme}")

file(READ "${_dnnl_arch_readme}" _dnnl_arch_readme_text)
string(CONFIGURE "${_dnnl_arch_readme_text}"
    DNNL_README_SYSTEM_REQUIREMENTS @ONLY)

configure_file(
    "${PROJECT_SOURCE_DIR}/README.in"
    "${PROJECT_BINARY_DIR}/README"
)
