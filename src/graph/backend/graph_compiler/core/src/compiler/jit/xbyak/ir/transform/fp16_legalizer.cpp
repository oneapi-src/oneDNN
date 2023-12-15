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

#include <string>
#include <utility>
#include <vector>

#include <compiler/ir/builder.hpp>
#include <compiler/ir/visitor.hpp>

#include "compiler/ir/graph/fusible_op_utils.hpp"
#include "fp16_legalizer.hpp"
#include "util/fp16.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

class fp16_legalizer_impl_t : public ir_visitor_t {
public:
    using ir_visitor_t::dispatch;
    using ir_visitor_t::visit;

    fp16_legalizer_impl_t() = default;

    expr_c visit(tensorptr_c v) override { return v; }

    stmt make_assign(const expr_c &l, const expr_c &r) {
        return builder::make_assign_unattached(l, r);
    }

    expr make_u16_indexing(const indexing_c &v) {
        auto tensor_pointer = builder::make_reinterpret(
                v->ptr_, datatypes::u16.get_pointerof());
        return copy_attr(*v, builder::make_indexing(tensor_pointer, v->idx_));
    }

    expr make_reinterpret_f16_u16(const expr_c &v) {
        assert(v->dtype_ == datatypes::f16);
        return builder::make_reinterpret(v, datatypes::u16);
    }

    expr make_reinterpret_u16_f16(const expr_c &v) {
        assert(v->dtype_ == datatypes::u16);
        return builder::make_reinterpret(v, datatypes::f16);
    }

    expr_c visit(indexing_c v) override {
        if (v->dtype_ == datatypes::f16) {
            return make_reinterpret_u16_f16(make_u16_indexing(v));
        }
        return v;
    }

    stmt_c visit(assign_c v) override {
        auto var = v->var_;
        auto value = ir_visitor_t::dispatch(v->value_);
        if (var.isa<indexing>() && var->dtype_ == datatypes::f16) {
            var = make_u16_indexing(var.static_as<indexing>());
            value = make_reinterpret_f16_u16(value);
        }
        if (!value.ptr_same(v->value_) || !var.ptr_same(v->var_)) {
            auto new_stmt = make_assign(var, value);
            return copy_attr(*v, std::move(new_stmt));
        }
        return v;
    }
};

func_c fp16_legalizer_t::operator()(func_c v) {
    if (target_machine_.cpu_flags_.fAVX512FP16) { return v; }
    fp16_legalizer_impl_t legalize_transform;
    auto ret = legalize_transform.dispatch(std::move(v));
    return ret;
}

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
