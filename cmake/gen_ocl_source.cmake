#===============================================================================
# Copyright 2019 Intel Corporation
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

# Generates cpp file with OpenCL code stored as string
# Parameters:
#   CL_FILE    -- path to the OpenCL source file
#   CL_INC_DIR -- include directory
#   CPP_FILE   -- path to the generated cpp file
#===============================================================================

# Read lines of OpenCL file and recursively substitute 'include'
# preprocessor directives.
#   cl_file  -- path to the OpenCL file
#   cl_lines -- list with code lines
function(read_lines cl_file cl_lines)
    file(STRINGS ${cl_file} contents NEWLINE_CONSUME)
    # Replace square brackets as they have special meaning in CMake
    string(REGEX REPLACE "\\[" "__BRACKET0__" contents "${contents}")
    string(REGEX REPLACE "\\]" "__BRACKET1__" contents "${contents}")
    # Escape backslash
    string(REGEX REPLACE "\\\\([^\n;])" "\\\\\\\\\\1" contents "${contents}")
    # Escape backslash (space is to avoid '\;' sequences after the split to a list)
    string(REGEX REPLACE "\\\\\n" "\\\\\\\\ \n" contents "${contents}")
    # Use EOL to split the contents to a list
    string(REGEX REPLACE "\n" ";" contents "${contents}")

    set(pp_lines)
    foreach(l ${contents})
        if(l MATCHES "\\s*#include \"(.*)\"")
            set(inc_file "${CL_INC_DIR}/${CMAKE_MATCH_1}")
            set(inc_lines)
            read_lines(${inc_file} inc_lines)
            list(APPEND pp_lines "${inc_lines}")
        else()
            string(REGEX REPLACE ";" "\\\\;" esc_line "${l}")
            list(APPEND pp_lines "${esc_line}")
        endif()
    endforeach()
    set(${cl_lines} "${pp_lines}" PARENT_SCOPE)
endfunction()

read_lines(${CL_FILE} cl_lines)

# Replace unescaped semicolon by EOL
string(REGEX REPLACE "([^\\]|^);" "\\1\n" cl_lines "${cl_lines}")
# Unescape semicolon
string (REGEX REPLACE "\\\\;" ";" cl_lines "${cl_lines}")
# Escape quatation marks
string(REGEX REPLACE "\"" "\\\\\"" cl_lines "${cl_lines}")
# Add EOLs
string(REGEX REPLACE " ?\n" "\\\\n\"\n\"" cl_lines "${cl_lines}")
# Replace square brackets back
string(REGEX REPLACE "__BRACKET0__" "[" cl_lines "${cl_lines}")
string(REGEX REPLACE "__BRACKET1__" "]" cl_lines "${cl_lines}")

get_filename_component(kernel_name ${CL_FILE} NAME_WE)
file(WRITE ${CPP_FILE} "const char *${kernel_name}_kernel =\"${cl_lines}\";")
