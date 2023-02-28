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

#include "content_hash.hpp"
namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

std::size_t content_hash_t<constant_c>::operator()(const constant_c &k) const {
    std::size_t ret = static_cast<uint64_t>(k->dtype_);
    if (k->dtype_.is_pointer()) {
        for (auto &v : k->value_) {
            ret = ret * 23 + v.s64;
        }
        return ret;
    }
    switch (get_etype_category(k->dtype_)) {
        case CATE_FLOAT:
            for (auto &v : k->value_) {
                ret = ret * 23 + std::hash<float>()(v.f32);
            }
            break;
        case CATE_INT:
        case CATE_UINT:
            for (auto &v : k->value_) {
                ret = ret * 23 + v.s64;
            }
            break;
        default: break;
    }
    return ret;
}

std::size_t content_hash_t<expr>::operator()(const expr &k) const {
    return content_hash_t<expr_c>()(k);
}

std::size_t content_hash_t<expr_c>::operator()(const expr_c &k) const {
    std::size_t ret;
    switch (k->node_type_) {
        case sc_expr_type::constant:
            ret = content_hash_t<constant_c>()(k.static_as<constant>());
            break;
        case sc_expr_type::tensor: ret = std::hash<expr_c>()(k); break;
        default: throw std::runtime_error("Unsupported node type!"); break;
    }
    return ret;
}

bool content_equals_t<expr_c>::operator()(
        const expr_c &a, const expr_c &b) const {
#if !SC_GNUC_VERSION_LT(7) && !defined(_MSC_VER)
    // use cached ir_comparer because it is a complex class
    // we will auto-reset after compare, so the cmper_ is unchanged after this
    // function call, as if it is "const"
    return const_cast<ir_comparer &>(cmper_).compare(a, b);
#else
    ir_comparer cmper_;
    return cmper_.compare(a, b);
#endif
}

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
