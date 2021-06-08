#===============================================================================
# Copyright 2019-2021 Intel Corporation
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

# Generates cpp file with GPU kernel or header code stored as string
# Parameters:
#   CL_FILE    -- path to the kernel source or header file
#   GEN_FILE   -- path to the generated cpp file
#===============================================================================

# Read lines of kernel or header file and escape  recursively substitute 'include'
# preprocessor directives.
#   cl_file  -- path to the kernel or header file
#   cl_file_lines -- list with code lines
function(read_lines cl_file cl_file_lines)
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
        string(REGEX REPLACE ";" "\\\\;" esc_line "${l}")
        list(APPEND pp_lines "${esc_line}")
    endforeach()
    set(${cl_file_lines} "${pp_lines}" PARENT_SCOPE)
endfunction()

read_lines(${CL_FILE} cl_file_lines)

# Replace unescaped semicolon by EOL
string(REGEX REPLACE "([^\\]|^);" "\\1\n" cl_file_lines "${cl_file_lines}")
# Unescape semicolon
string (REGEX REPLACE "\\\\;" ";" cl_file_lines "${cl_file_lines}")
# Escape quatation marks
string(REGEX REPLACE "\"" "\\\\\"" cl_file_lines "${cl_file_lines}")
# Add EOLs
string(REGEX REPLACE " ?\n" "\\\\n\",\n\"" cl_file_lines "${cl_file_lines}")
# Replace square brackets back
string(REGEX REPLACE "__BRACKET0__" "[" cl_file_lines "${cl_file_lines}")
string(REGEX REPLACE "__BRACKET1__" "]" cl_file_lines "${cl_file_lines}")

get_filename_component(cl_file_name ${CL_FILE} NAME_WE)
get_filename_component(cl_file_ext ${CL_FILE} EXT)

if(cl_file_ext STREQUAL ".cl")
    set(cl_file_contents  "const char *${cl_file_name}_kernel[] ={ \"${cl_file_lines}\", nullptr };")
elseif(cl_file_ext STREQUAL ".h")
    set(cl_file_contents  "const char *${cl_file_name}_header[] ={ \"${cl_file_lines}\", nullptr };")
else()
    message(FATAL_ERROR "Unknown file extensions: ${cl_file_ext}")
endif()

set(cl_file_contents "namespace ocl {\n${cl_file_contents}\n}")
set(cl_file_contents "namespace gpu {\n${cl_file_contents}\n}")
set(cl_file_contents "namespace impl {\n${cl_file_contents}\n}")
set(cl_file_contents "namespace dnnl {\n${cl_file_contents}\n}")
file(WRITE ${GEN_FILE} "${cl_file_contents}")
