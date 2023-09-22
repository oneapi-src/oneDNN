/*******************************************************************************
 * Copyright 2023 Intel Corporation
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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_JIT_XBYAK_IR_UTIL_UTILS_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_JIT_XBYAK_IR_UTIL_UTILS_HPP

#include <assert.h>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <compiler/ir/sc_data_type.hpp>
#include <compiler/ir/sc_expr.hpp>
#include <compiler/jit/xbyak/configured_xbyak.hpp>
#include <util/utils.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace xbyak {

/**
 * If datatype is a x86 simd register type
 * */
SC_INTERNAL_API inline bool is_x86_simd(const sc_data_type_t &t) {
    return !t.is_tile()
            && (t.type_code_ == sc_data_etype::F32
                    || t.type_code_ == sc_data_etype::F16 || t.lanes_ > 1);
}

/**
 * If constant node scalar intger value exceeds 32bit
 * */
SC_INTERNAL_API inline bool const_exceed_32bit(const expr_c &v) {
    if ((utils::is_one_of(v->dtype_, datatypes::index, datatypes::generic)
                || v->dtype_.is_pointer())
            && v.isa<constant>()) {
        const auto c = v.static_as<constant_c>();
        const uint64_t x = c->value_[0].u64;
        return !Xbyak::inner::IsInInt32(x);
    }
    return false;
}

} // namespace xbyak
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
