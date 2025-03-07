# ******************************************************************************
# Copyright 2025 Arm Limited and affiliates.
# SPDX-License-Identifier: Apache-2.0
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
# ******************************************************************************

# ----------
# FindKleidiAI
# ----------
#
# Finds KleidiAI
#
# This module defines the following variables:
#
#   KAI_INCLUDE_DIR   - include directories for KleidiAI
#   KAI_LIBRARY       - link against this library to use KleidiAI
#
# The module will also define two cache variables:
#
#   KAI_INCLUDE_DIR    - the KleidiAI include directory
#   KAI_LIBRARY        - the path to the KleidiAI library
#

find_path(KAI_INCLUDE_DIR
  NAMES kai/kai_common.h
  PATHS ENV KAI_ROOT_DIR
  )

find_library(KAI_LIBRARY
  NAMES kleidiai
  PATHS ENV KAI_ROOT_DIR
  PATH_SUFFIXES lib build
  )

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(KleidiAI DEFAULT_MSG
  KAI_INCLUDE_DIR
  KAI_LIBRARY
)

mark_as_advanced(
  KAI_LIBRARY
  KAI_INCLUDE_DIR
  )
