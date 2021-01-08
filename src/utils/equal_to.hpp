/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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

#ifndef UTILS_EQUAL_TO_HPP
#define UTILS_EQUAL_TO_HPP

#include <functional>

namespace dnnl {
namespace graph {
namespace impl {
namespace utils {

template <typename type>
struct equal_to {
    constexpr equal_to(type v) noexcept : v_ {v} {}

    constexpr bool operator()(const type &v) const noexcept { return v_ == v; }

    const type v_ {};
};

template <typename type>
equal_to(type)->equal_to<type>;

} // namespace utils
} // namespace impl
} // namespace graph
} // namespace dnnl

#endif
