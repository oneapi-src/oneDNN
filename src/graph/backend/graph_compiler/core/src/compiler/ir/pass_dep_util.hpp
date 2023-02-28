/*******************************************************************************
 * Copyright 2022-2023 Intel Corporation
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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_PASS_DEP_UTIL_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_PASS_DEP_UTIL_HPP

#include "pass_id.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

template <typename T, typename... Args>
inline constexpr T make_bit_mask_impl(T oldv, int n_th, Args... args) {
    return oldv | (T(1) << n_th);
}
template <typename T>
inline constexpr T make_bit_mask_impl(T oldv) {
    return oldv;
}

template <typename T, typename... Args>
inline constexpr T make_bit_mask(Args... args) {
    return make_bit_mask_impl<T>(T(0), args...);
}

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#ifndef NDEBUG
#define SC_DECL_PASS_DEPDENCYINFO_IMPL(passname, passclass, ...) \
    void passclass::get_dependency_info(tir_pass_dependency_t &out) const { \
        using namespace tir_pass; \
        out = {passname, __VA_ARGS__}; \
    }
#else
#define SC_DECL_PASS_DEPDENCYINFO_IMPL(passname, passclass, ...)
#endif

#define SC_DECL_PASS_INFO_IMPL(passname, passclass, ...) \
    const char *passclass::get_name() const { return #passname; } \
    SC_DECL_PASS_DEPDENCYINFO_IMPL(passname, passclass, __VA_ARGS__)

// passname, SC_PASS_DEPENDS_ON(...), SC_PASS_REQUIRE_STATE(...),
// SC_PASS_REQUIRE_NOT_STATE(...), SC_PASS_SET_STATE(...),
// SC_PASS_UNSET_STATE(...)
#define SC_DECL_PASS_INFO(passname, ...) \
    SC_DECL_PASS_INFO_IMPL(passname, passname##_t, __VA_ARGS__)

#define SC_PASS_MASK(...) make_bit_mask<uint64_t>(__VA_ARGS__)
#define SC_PASS_DEPENDS_ON(...) \
    { __VA_ARGS__ }
#define SC_PASS_REQUIRE_STATE SC_PASS_MASK
#define SC_PASS_REQUIRE_NOT_STATE SC_PASS_MASK
#define SC_PASS_SET_STATE SC_PASS_MASK
#define SC_PASS_UNSET_STATE SC_PASS_MASK

#endif
