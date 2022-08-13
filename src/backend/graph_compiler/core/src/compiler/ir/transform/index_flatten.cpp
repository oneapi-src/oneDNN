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
#include <utility>
#include <unordered_map>

#include <vector>
#include "../builder.hpp"
#include "../util_module_passes.hpp"
#include "../visitor.hpp"
#include "index_flatten.hpp"

namespace sc {
static bool process_indexing(ir_visitor_t *ths,
        const std::vector<expr> &old_dims, const std::vector<expr> &old_strides,
        const std::vector<expr> &idx, std::vector<expr_c> &newidx) {
    bool changed = ths->dispatch_expr_vector(idx, newidx);
    assert(!idx.empty());
    if (idx.size() == 1) {
        return changed;
    } else {
        COMPILE_ASSERT(old_strides.size() == old_dims.size(),
                "Dims and strides shall have same length.");
        expr_c flattened = builder::make_mul(newidx.back(), old_strides.back());
        for (int64_t i = newidx.size() - 2; i >= 0; i--) {
            flattened = builder::make_add(
                    builder::make_mul(newidx[i], old_strides[i]), flattened);
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
    changed |= process_indexing(ths, old->dims_, old->strides_, idx, newidx);
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

static std::vector<expr> get_dense_stride(const std::vector<expr> &shape) {
    std::vector<expr> result(shape.size(), 1);
    for (int i = shape.size() - 2; i >= 0; --i) {
        result[i] = result[i + 1] * shape[i + 1];
    }
    return result;
}

static std::vector<expr> get_base_stride(const expr_c &ex) {
    if (ex.isa<tensor>()) { return ex.static_as<tensor>()->strides_; }
    COMPILE_ASSERT(ex.isa<tensorptr>(), "Expecting tensorptr, got: " << ex);
    auto tptr = ex.static_as<tensorptr>();
    if (tptr->is_slice_) {
        COMPILE_ASSERT(tptr->base_.isa<indexing>(),
                "tptr's base should be indexing, but got: " << tptr->base_);
        return get_base_stride(tptr->base_->ptr_);
    } else {
        // when is_slice_ == false we create a dense tensor based on new shape
        return get_dense_stride(tptr->shape_);
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
            const std::vector<expr> stride = get_base_stride(oldptr);
            // 1D input might not have shape info
            assert(oldptr->base_->idx_.size() == 1 || !shape->empty());

            // flatten the indices using the reshaped dimensions
            process_indexing(this, *shape, stride, v->idx_, newidx);

            return copy_attr(*v,
                    builder::make_indexing(ptr->base_->ptr_,
                            builder::make_add(ptr->base_->idx_[0], newidx[0]),
                            v->dtype_.lanes_, v->mask_));
        }
    }

    expr_c visit(tensor_c v) override {
        COMPILE_ASSERT(v->dims_.size() == v->strides_.size(),
                "Tensor dims and strides shall have same length.");
        if (v->dims_.size() == 1 && v->strides_[0].isa<constant>()
                && get_expr_as_int(v->strides_[0]) == 1) {
            // if already flattened, return
            return v;
        } else {
            expr range = 1;
            for (size_t i = 0; i < v->dims_.size(); ++i) {
                range = builder::make_add(
                        builder::make_mul(v->dims_[i] - 1, v->strides_[i]),
                        range);
            }
            // here stride is {1} since all strided info are compressed into
            // dims
            auto ret = copy_attr(*v,
                    builder::make_stensor(v->name_, {range},
                            std::vector<expr> {UINT64_C(1)}, v->elem_dtype_,
                            v->address_space_, v->init_value_));
            return ret;
        }
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
