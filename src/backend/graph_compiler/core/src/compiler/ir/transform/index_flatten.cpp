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
#include <utility>
#include <unordered_map>

#include <vector>
#include "../builder.hpp"
#include "../util_module_passes.hpp"
#include "../visitor.hpp"
#include "index_flatten.hpp"

namespace sc {
static bool process_indexing(ir_visitor_t *ths,
        const std::vector<expr> &old_dims, const std::vector<expr> &idx,
        std::vector<expr_c> &newidx) {
    bool changed = ths->dispatch_expr_vector(idx, newidx);
    assert(!idx.empty());
    if (idx.size() == 1) {
        return changed;
    } else {
        expr_c flattened = newidx.back();
        expr_c dim = old_dims.back();
        for (int64_t i = newidx.size() - 2; i >= 0; i--) {
            flattened = builder::make_add(
                    builder::make_mul(newidx[i], dim), flattened);
            dim = builder::make_mul(old_dims[i], dim);
        }
        newidx = std::vector<expr_c> {std::move(flattened)};
        return true;
    }
}

static bool process_indexing(ir_visitor_t *ths, tensor_c &tsr,
        const std::vector<expr> &idx, std::vector<expr_c> &newidx) {
    tensor_c old = tsr;
    tsr = ths->dispatch(tsr).checked_as<tensor_c>();
    bool changed = !tsr.ptr_same(old);
    COMPILE_ASSERT(old->dims_.size() == idx.size(),
            "Unmatched dimensions of indexing: tsr= "
                    << tsr << utils::print_vector(old->dims_) << ", indexing on"
                    << utils::print_vector(idx));
    changed |= process_indexing(ths, old->dims_, idx, newidx);
    return changed;
}

static const std::vector<expr> *get_base_shape(const expr_c &ex) {
    if (ex.isa<tensor>()) { return &ex.static_as<tensor>()->dims_; }
    COMPILE_ASSERT(ex.isa<tensorptr>(), "Expecting tensorptr, got: " << ex);
    auto tptr = ex.static_as<tensorptr>();
    if (tptr->is_slice_) {
        return get_base_shape(tptr->base_->ptr_);
    } else {
        return &tptr->shape_;
    }
}

class index_flatten_t : public ir_consistent_visitor_t {
public:
    using ir_consistent_visitor_t::dispatch;
    using ir_consistent_visitor_t::visit;
    expr_c visit(indexing_c v) override {
        std::vector<expr_c> newidx;
        bool changed;
        if (v->ptr_.isa<tensor>()) {
            auto ptr = v->ptr_.static_as<tensor_c>();
            changed = process_indexing(this, ptr, v->idx_, newidx);
            if (changed) {
                return copy_attr(*v,
                        builder::make_indexing(
                                ptr, newidx, v->dtype_.lanes_, v->mask_));
            } else {
                return std::move(v);
            }
        } else {
            // indexing on a tensor_ptr
            tensorptr_c oldptr = v->ptr_.checked_as<tensorptr_c>();
            // first, flatten the indices using tensor_ptr's dimensions
            // then, add the flattened 1D index with the flattened 1D offset
            // of the base tensor_ptr
            auto ptr = dispatch(v->ptr_).checked_as<tensorptr_c>();
            assert(ptr->base_->idx_.size() == 1);
            const std::vector<expr> *shape = get_base_shape(oldptr);
            assert(!shape->empty());

            // flatten the indices using the reshaped dimensions
            process_indexing(this, *shape, v->idx_, newidx);

            return copy_attr(*v,
                    builder::make_indexing(ptr->base_->ptr_,
                            builder::make_add(ptr->base_->idx_[0], newidx[0]),
                            v->dtype_.lanes_, v->mask_));
        }
    }

    expr_c visit(tensor_c v) override {
        if (v->dims_.size() != 1) {
            expr flattened;
            for (auto &e : v->dims_) {
                if (flattened.defined()) {
                    flattened = flattened * e;
                } else {
                    flattened = e;
                }
            }
            auto ret = copy_attr(*v,
                    builder::make_tensor(v->name_, {flattened}, v->elem_dtype_,
                            v->address_space_, v->init_value_));
            return ret;
        }
        return v;
    }

    expr_c visit(tensorptr_c v) override {
        auto base = dispatch(v->base_);
        // we should remove the shape info
        if (base.ptr_same(v->base_) && v->shape_.empty()) { return v; }
        return copy_attr(*v,
                make_expr<tensorptr_node>(
                        base.remove_const().checked_as<indexing>(),
                        std::vector<expr> {}, false));
    }
};

func_c index_flattener_t::operator()(func_c f) {
    index_flatten_t pass;
    return pass.dispatch(std::move(f));
}

const_ir_module_ptr index_flattener_t::operator()(const_ir_module_ptr f) {
    index_flatten_t pass;
    return dispatch_module_on_visitor(&pass, f);
}

} // namespace sc
