/*******************************************************************************
* Copyright 2016-2018 Intel Corporation
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

/**
 * @brief Macro overloading feature support
 * @file mkldnn_macros.hpp
 */

#pragma once

#define MKLDNN_MACRO_EXPAND(x) x

#define MKLDNN_MACRO_CAT_(x, y) x ## y
#define MKLDNN_MACRO_CAT(x, y) MKLDNN_MACRO_CAT_(x, y)
#define MKLDNN_MACRO_CAT3_(x, y, z) x ## y ## z
#define MKLDNN_MACRO_CAT3(x, y, z) MKLDNN_MACRO_CAT3_(x, y, z)

#define MKLDNN_MACRO_TOSTRING(...) MKLDNN_MACRO_TOSTRING_(__VA_ARGS__)
#define MKLDNN_MACRO_TOSTRING_(...) #__VA_ARGS__

#define MKLDNN_MACRO_NARG(...) MKLDNN_MACRO_EXPAND( MKLDNN_MACRO_NARG_(__VA_ARGS__, MKLDNN_MACRO_RSEQ_N()) )
#define MKLDNN_MACRO_NARG_(...) MKLDNN_MACRO_EXPAND( MKLDNN_MACRO_ARG_N(__VA_ARGS__) )
#define MKLDNN_MACRO_ARG_N(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, N, ...) N
#define MKLDNN_MACRO_RSEQ_N() 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0

#define MKLDNN_MACRO_EVAL_(NAME, N) NAME ## _ ## N
#define MKLDNN_MACRO_EVAL(NAME, N) MKLDNN_MACRO_EVAL_(NAME, N)

#define MKLDNN_MACRO_OVERLOAD(NAME, ...) MKLDNN_MACRO_EXPAND( MKLDNN_MACRO_EVAL(NAME, MKLDNN_MACRO_EXPAND( MKLDNN_MACRO_NARG(__VA_ARGS__) ))(__VA_ARGS__) )
