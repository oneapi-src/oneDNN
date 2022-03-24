/*******************************************************************************
 * Copyright 2022 Intel Corporation
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
#ifndef BACKEND_GRAPH_COMPILER_CORE_SRC_UTIL_SIMPLE_MATH_HPP
#define BACKEND_GRAPH_COMPILER_CORE_SRC_UTIL_SIMPLE_MATH_HPP
#include <stddef.h>
namespace sc {
namespace utils {
static constexpr size_t divide_and_ceil(size_t x, size_t y) {
    return (x + y - 1) / y;
}

static constexpr size_t rnd_up(const size_t a, const size_t b) {
    return (divide_and_ceil(a, b) * b);
}

static constexpr size_t rnd_dn(const size_t a, const size_t b) {
    return (a / b) * b;
}
} // namespace utils
} // namespace sc
#endif
