/*******************************************************************************
 * Copyright 2020-2023 Intel Corporation
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

#include <compiler/ir/sc_data_type.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
type_category get_type_category(sc_data_type_t dtype) {
    switch (dtype) {
        case datatypes::bf16:
        case datatypes::f16:
        case datatypes::f32: return CATE_FLOAT; break;
        case datatypes::s32:
        case datatypes::s8: return CATE_INT; break;
        case datatypes::index:
        case datatypes::u32:
        case datatypes::u16:
        case datatypes::u8:
        case datatypes::boolean: return CATE_UINT; break;
        default: assert(0 && "Bad type"); return CATE_INT;
    }
}

type_category get_type_category_nothrow(sc_data_type_t dtype) {
    switch (dtype) {
        case datatypes::bf16:
        case datatypes::f16:
        case datatypes::f32: return CATE_FLOAT; break;
        case datatypes::s32:
        case datatypes::s8: return CATE_INT; break;
        case datatypes::index:
        case datatypes::u32:
        case datatypes::u16:
        case datatypes::u8:
        case datatypes::boolean: return CATE_UINT; break;
        default: return CATE_OTHER;
    }
}

type_category get_etype_category(sc_data_type_t dtype) {
    switch (dtype.type_code_) {
        case sc_data_etype::BF16:
        case sc_data_etype::F16:
        case sc_data_etype::F32: return CATE_FLOAT; break;
        case sc_data_etype::S32:
        case sc_data_etype::S8: return CATE_INT; break;
        case sc_data_etype::INDEX:
        case sc_data_etype::U32:
        case sc_data_etype::U16:
        case sc_data_etype::U8:
        case sc_data_etype::BOOLEAN: return CATE_UINT; break;
        default: assert(0 && "Bad type"); return CATE_OTHER;
    }
}

type_category get_etype_category_nothrow(sc_data_type_t dtype) {
    switch (dtype.type_code_) {
        case sc_data_etype::BF16:
        case sc_data_etype::F16:
        case sc_data_etype::F32: return CATE_FLOAT; break;
        case sc_data_etype::S32:
        case sc_data_etype::S8: return CATE_INT; break;
        case sc_data_etype::INDEX:
        case sc_data_etype::U32:
        case sc_data_etype::U16:
        case sc_data_etype::U8:
        case sc_data_etype::BOOLEAN: return CATE_UINT; break;
        default: return CATE_OTHER;
    }
}

std::ostream &operator<<(std::ostream &os, sc_data_etype t) {
    switch (t) {
        case sc_data_etype::UNDEF: os << "undef"; break;
        case sc_data_etype::F16: os << "f16"; break;
        case sc_data_etype::BF16: os << "bf16"; break;
        case sc_data_etype::F32: os << "f32"; break;
        case sc_data_etype::S32: os << "s32"; break;
        case sc_data_etype::U32: os << "u32"; break;
        case sc_data_etype::U16: os << "u16"; break;
        case sc_data_etype::S8: os << "s8"; break;
        case sc_data_etype::U8: os << "u8"; break;
        case sc_data_etype::INDEX: os << "index"; break;
        case sc_data_etype::BOOLEAN: os << "bool"; break;
        case sc_data_etype::GENERIC: os << "generic_val"; break;
        case sc_data_etype::VOID_T: os << "void"; break;
        case sc_data_etype::POINTER: os << "pointer"; break;
        default:
            if (etypes::is_pointer(t)) {
                os << etypes::get_pointer_element(t);
                os << '*';
                return os;
            }
            assert(0 && "Unknown type");
    }
    return os;
}

std::ostream &operator<<(std::ostream &os, sc_data_type_t dtype) {
    os << dtype.type_code_;
    if (dtype.lanes_ > 1) {
        if (dtype.rows_ == 0) {
            os << 'x' << dtype.lanes_;
        } else {
            os << 'x' << dtype.rows_ << 'x' << dtype.lanes_ / dtype.rows_;
        }
    }
    return os;
}

namespace utils {
// Without this specialization, the generic print_vector template will
// automatically invoke `sc_data_type_t::operator uint_64t()`, resulting in
// the printed output being a list of numbers.
template <>
std::string print_vector(const std::vector<sc_data_type_t> &vec) {
    std::stringstream os;
    int cnt = 0;
    os << '[';
    for (auto &v : vec) {
        if (cnt != 0) { os << ", "; }
        os << v;
        cnt++;
    }
    os << ']';
    return os.str();
}
} // namespace utils

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

namespace std {
std::size_t hash<dnnl::impl::graph::gc::sc_data_type_t>::operator()(
        const dnnl::impl::graph::gc::sc_data_type_t &k) const {
    return hash<unsigned>()((uint64_t)k);
}
} // namespace std
