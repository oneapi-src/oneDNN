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

#include <utility>
#include <vector>
#include <unordered_map>

#include <compiler/ir/attr_keys.hpp>
#include <compiler/ir/builder.hpp>
#include <compiler/ir/ir_module.hpp>
#include <compiler/ir/ir_utils.hpp>
#include <compiler/ir/passlet/use_count_analysis.hpp>
#include <compiler/ir/ssa_data.hpp>
#include <compiler/ir/ssa_visitor.hpp>
#include <compiler/ir/transform/constant_fold.hpp>
#include <compiler/ir/transform/pointer_alias_info.hpp>
#include <compiler/ir/visitor.hpp>
#include <util/any_map.hpp>
#include <util/array_ref.hpp>

#include "indexing_transform.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace xbyak {

using namespace passlet;

class indexing_transform_impl_t : public ir_visitor_t {
private:
    constant_folder_t constant_folder_;

public:
    using ir_visitor_t::dispatch;
    using ir_visitor_t::visit;

    bool is_read_only_tensor(const expr_c &v) {
        assert(v.isa<tensor>());
        // if tesnor is read buffer and has no alias
        auto info = alias_info::get_alias_info(*v);
        return any_map_t::fetch_or_else(v->attr_.get(), "read_buffer", false)
                && (info == nullptr || info->has_no_alias());
    }

    expr_c expand_polynomial_offset(expr_c f) {
        return constant_folder_.expand_polynomial(std::move(f), 1, true);
    }

    expr_c visit(indexing_c v) override {
        auto vv = ir_visitor_t::visit(std::move(v)).checked_as<indexing_c>();
        auto &ptr = vv->ptr_;
        auto &idx = vv->idx_.back();
        auto idx_type = datatypes::index;
        auto elem_type = ptr->dtype_.get_pointer_element();
        auto elem_size = utils::get_sizeof_type(elem_type);
        auto is_var_or_const = [](const expr_c &i) {
            return i.isa<constant>() || i.isa<var>();
        };
        // set read only to the base tensor
        auto read_only = is_read_only_tensor(ptr);
        ptr->attr().set(attr_keys::read_only_tensor, read_only);
        // transform complex indexing to pointer calculation
        if (is_var_or_const(idx)) {
            // If indexing A[i] already simple, no need to transform
            return vv;
        } else if (idx.isa<add>()
                && is_var_or_const(idx.static_as<add>()->r_)) {
            // This requires canonicalized expr to work
            // The right add op must be most inner var
            auto node = idx.static_as<add>();
            auto scale = builder::make_constant({elem_size}, idx_type);
            auto offset = expand_polynomial_offset(scale * node->l_);
            auto new_ptr = builder::make_cast(
                    ptr->dtype_, builder::make_cast(idx_type, ptr) + offset);
            // transform A[l + r] to (A + (s * l))[r]
            auto new_idx = node->r_;
            return builder::make_indexing(
                    new_ptr, new_idx, vv->dtype_.lanes_, vv->mask_);
        } else {
            auto scale = builder::make_constant({elem_size}, idx_type);
            auto offset = expand_polynomial_offset(scale * idx);
            auto new_ptr = builder::make_cast(
                    ptr->dtype_, builder::make_cast(idx_type, ptr) + offset);
            // transform A[i] to (A + (s * i))[0]
            auto new_idx = builder::make_constant(UINT64_C(0));
            return builder::make_indexing(
                    new_ptr, new_idx, vv->dtype_.lanes_, vv->mask_);
        }
    }

    expr_c visit(tensorptr_c v) override {
        auto &ptr = v->base_->ptr_;
        auto &idx = v->base_->idx_.back();
        auto elem_type = ptr->dtype_.get_pointer_element();
        auto elem_size = utils::get_sizeof_type(elem_type);
        auto idx_zero = idx.cast<constant>().filter([](const constant &v) {
            uint64_t val = v->value_[0].u64;
            return (v->value_.size() == 1) && (val == 0);
        });
        // transform tensorptr &A[i] to void*(A + (s * i))
        if (idx_zero.has_value()) {
            return builder::make_cast(v->dtype_, ptr);
        } else {
            auto scale = builder::make_constant({elem_size}, idx->dtype_);
            auto offset = expand_polynomial_offset(scale * idx);
            return builder::make_cast(v->dtype_,
                    builder::make_cast(datatypes::index, ptr) + offset);
        }
        return v;
    }
};

func_c indexing_transform_t::operator()(func_c v) {
    indexing_transform_impl_t indexing_transform;
    auto vv = indexing_transform.dispatch(std::move(v));
    return vv;
}

} // namespace xbyak
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
