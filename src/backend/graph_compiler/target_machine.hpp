/*******************************************************************************
 * Copyright 2021-2022 Intel Corporation
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
#ifndef BACKEND_GRAPH_COMPILER_TARGET_MACHINE_HPP
#define BACKEND_GRAPH_COMPILER_TARGET_MACHINE_HPP

#include "core/src/compiler/config/context.hpp"
#include "runtime/config.hpp"

#define REQUIRE_AVX512_BEGIN \
    if (::sc::get_default_context()->machine_.cpu_flags_.fAVX512F) {
#define REQUIRE_VNNI_AMXINT8_BEGIN \
    if (::sc::get_default_context()->machine_.cpu_flags_.fAVX512VNNI \
            || ::sc::get_default_context() \
                       ->machine_.cpu_flags_.fAVX512AMXINT8) {
#define REQUIRE_BF16_AMXBF16_BEGIN \
    if (::sc::get_default_context()->machine_.cpu_flags_.fAVX512BF16 \
            || ::sc::get_default_context() \
                       ->machine_.cpu_flags_.fAVX512AMXBF16) {
#define REQUIRE_SINGLE_THREAD_BEGIN \
    if (sc::runtime_config_t::get().get_num_threads() == 1) {
#define REQUIRE_AVX512_END }
#define REQUIRE_VNNI_AMXINT8_END }
#define REQUIRE_BF16_AMXBF16_END }
#define REQUIRE_SINGLE_THREAD_END }
#endif
