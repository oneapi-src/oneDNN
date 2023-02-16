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

#include "ssa_simplify.hpp"
#include <compiler/ir/ir_utils.hpp>
#include <compiler/ir/ssa_data.hpp>
#include <util/hash_utils.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace passlet {

expr_c ssa_simplify_t::visit(const var_c &v) {
    if (is_in_phi_) { return v; }
    if (v->ssa_data_->is_global_) { return v; }
    auto val = v->ssa_data_->get_value_of_var_nothrow();
    if (!val.defined()) { return v; }
    if (val.isa<constant>()) { return val; }
    if (val.isa<var>()) {
        if (val->ssa_data_->is_global_) { return v; }
        return visit(val.static_as<var>());
    }
    return v;
}

expr_c ssa_simplify_t::visit(const ssa_phi_c &v) {
    if (v->values_.size() == 1UL) {
        auto &ret = v->values_.front();
        if (ret.isa<var>()) { return visit(ret.static_as<var_c>()); }
        return ret;
    }
    return v;
}

} // namespace passlet
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
