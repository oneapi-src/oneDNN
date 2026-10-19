/*******************************************************************************
* Copyright 2025 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#ifndef GEMMSTONE_TOOLS_VT100_CODES_HPP
#define GEMMSTONE_TOOLS_VT100_CODES_HPP

#define VT_RESET        "\x1B[0m"
#define VT_BOLD         "\x1B[1m"
#define VT_RED          "\x1B[1;31m"
#define VT_YELLOW       "\x1B[1;33m"
#define VT_GREEN        "\x1B[1;32m"
#define VT_CYAN         "\x1B[1;36m"
#define VT_BLUE         "\x1B[1;34m"
#define VT_MAGENTA      "\x1B[1;35m"
#define VT_ORANGE       "\x1B[38;5;208m"
#define VT_CLEAR_LINE   "\x1B[K"
#define VT_CLEAR_SCREEN "\x1B[2J\x1B[H"
#define VT_SET_TAB      "\x1BH"
#define VT_CLEAR_TABS   "\x1B[3g"
#define VT_ALT_SCREEN_ON  "\x1B[?1049h"
#define VT_ALT_SCREEN_OFF "\x1B[?1049l"

#endif
