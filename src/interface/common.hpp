/*******************************************************************************
 * Copyright 2020 Intel Corporation
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

#ifndef LLGA_INTERFACE_COMMON_HPP
#define LLGA_INTERFACE_COMMON_HPP

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "c_types_map.hpp"
#include "logger.hpp"
#include "utils/compatible.hpp"

namespace llga {
namespace impl {

inline static size_t size_of(data_type_t dtype) {
    switch (dtype) {
        case data_type::f32:
        case data_type::s32: return 4U;
        case data_type::s8:
        case data_type::u8: return 1U;
        case data_type::f16:
        case data_type::bf16: return 2U;
        default:
            throw std::runtime_error(
                    "size_of: cannot get the size of data type.");
            return 0;
    }
}

using dims = std::vector<int64_t>;

inline static size_t prod(const dims &shape) {
    if (shape.size() == 0) return 0;

    size_t p = (std::accumulate(
            shape.begin(), shape.end(), size_t(1), std::multiplies<dim_t>()));

    return p;
}

inline static size_t size_of(const dims &shape, data_type_t dtype) {
    return prod(shape) * size_of(dtype);
}

} // namespace impl
} // namespace llga

#endif
