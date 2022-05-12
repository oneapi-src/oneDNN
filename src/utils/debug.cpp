/*******************************************************************************
* Copyright 2020-2022 Intel Corporation
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

// clang-format off

#include <cassert>

#include "utils/debug.hpp"

namespace dnnl {
namespace graph {
namespace impl {
namespace utils {

const char *data_type2str(data_type_t v) {
    if (v == data_type::undef) return "undef";
    if (v == data_type::f16) return "f16";
    if (v == data_type::bf16) return "bf16";
    if (v == data_type::f32) return "f32";
    if (v == data_type::s32) return "s32";
    if (v == data_type::s8) return "s8";
    if (v == data_type::u8) return "u8";
    assert(!"unknown data_type");
    return "unknown data_type";
}

const char *engine_kind2str(engine_kind_t v) {
    if (v == engine_kind::any_engine) return "any";
    if (v == engine_kind::cpu) return "cpu";
    if (v == engine_kind::gpu) return "gpu";
    assert(!"unknown engine_kind");
    return "unknown engine_kind";
}

const char *fpmath_mode2str(fpmath_mode_t v) {
    if (v == fpmath_mode::strict) return "strict";
    if (v == fpmath_mode::bf16) return "bf16";
    if (v == fpmath_mode::f16) return "f16";
    if (v == fpmath_mode::any) return "any";
    if (v == fpmath_mode::f19) return "f19";
    assert(!"unknown fpmath_mode");
    return "unknown fpmath_mode";
}

const char *layout_type2str(layout_type_t v) {
    if (v == layout_type::undef) return "undef";
    if (v == layout_type::any) return "any";
    if (v == layout_type::strided) return "strided";
    if (v == layout_type::opaque) return "opaque";
    assert(!"unknown layout_type");
    return "unknown layout_type";
}

const char *property_type2str(property_type_t v) {
    if (v == property_type::undef) return "undef";
    if (v == property_type::variable) return "variable";
    if (v == property_type::constant) return "constant";
    assert(!"unknown property_type");
    return "unknown property_type";
}

} // namespace utils
} // namespace impl
} // namespace graph
} // namespace dnnl
