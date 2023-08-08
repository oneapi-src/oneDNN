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
#include "index_flatten.hpp"
#include <utility>
#include <vector>
#include "../builder.hpp"
#include "../util_module_passes.hpp"
#include "../visitor.hpp"
#include "./constant_fold.hpp"
#include <compiler/dimensions.hpp>
#include <compiler/ir/pass_dep_util.hpp>
#include <unordered_map>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

SC_DECL_PASS_INFO(index_flattener,
        SC_PASS_DEPENDS_ON(
                dyn_tensor_transformer, interface_generalizer, tensor_shrinker),
        SC_PASS_REQUIRE_STATE(), SC_PASS_REQUIRE_NOT_STATE(),
        SC_PASS_SET_STATE(), SC_PASS_UNSET_STATE(CONST_FOLDED));

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

sc_dims get_expr_to_dims(const std::vector<expr> &dims);

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

static bool is_const_expr_vector(const std::vector<expr> &vec) {
    return std::all_of(vec.begin(), vec.end(), [](const expr &x) {
        return do_cast_and_fold(x).isa<constant_c>();
    });
}

/* process indexing on tptr(slice=false) with base = tptr(slice=true)
   A = tensor[H,W,C]
   B = tensorptr(A,[0,1,1,0], [H-1,W-1,C], is_slice=true)
   C = tensorptr(B, [0,0,0,0], [S,C], is_slice=false) // S = (H-1) * (W-1)
   indexing C[s,c]
   let flatten_idx = s * C + c;
   flatten(C[s,c]) = flatten(B) + (flatten_idx / C / (W-1) % (H-1) * W * C +
                        flatten_idx / C % (W-1) * C + flatten_idx % C)

   if shape are const, we could know C = C, S = (H-1)*(W-1) and the computation
   could be simplied to
   flatten(C[s,c]) = flatten(C) + (s / (W-1) % (H-1) * W *
                        C + s % (W-1) * C + c)
*/
static void process_slice_reshape_indexing(ir_visitor_t *ths,
        const std::vector<expr> &old_dims, const std::vector<expr> &slice_shape,
        const std::vector<expr> &parent_stride, const std::vector<expr> &idx,
        std::vector<expr_c> &newidx) {
    ths->dispatch_expr_vector(idx, newidx);
    assert(!idx.empty());
    bool is_shape_all_const = is_const_expr_vector(slice_shape);
    is_shape_all_const &= is_const_expr_vector(parent_stride);
    is_shape_all_const &= is_const_expr_vector(old_dims);
    if (is_shape_all_const) {
        // if shape is all const, the computation could be simplified according
        // to shape
        expr flattened = 0;
        auto stride = get_dense_stride(old_dims);
        auto slice_const_shape = get_expr_to_dims(slice_shape);
        auto real_shape = get_expr_to_dims(old_dims);
        auto slice_dense_stride
                = get_expr_to_dims(get_dense_stride(slice_shape));

        auto acc_shape = 1;
        int acc_idx = real_shape.size() - 1;
        expr flatten_remain = 0UL;
        for (auto i = (int)slice_const_shape.size() - 1; i >= 0; i--) {
            while (acc_shape % slice_const_shape[i] != 0 && acc_idx >= 0) {
                flatten_remain
                        = idx[acc_idx] * stride[acc_idx] + flatten_remain;
                acc_shape *= real_shape[acc_idx];
                acc_idx--;
            }
            if (acc_shape == slice_const_shape[i]) {
                flattened = flatten_remain
                                / expr((uint64_t)slice_dense_stride[i])
                                * parent_stride[i]
                        + flattened;
                flatten_remain = 0UL;
                acc_shape = 1;
            } else {
                flattened = flatten_remain
                                / expr((uint64_t)slice_dense_stride[i])
                                % (uint64_t)slice_const_shape[i]
                                * parent_stride[i]
                        + flattened;
                acc_shape /= slice_const_shape[i];
            }
        }
        newidx = std::vector<expr_c> {std::move(flattened)};
    } else {
        std::vector<expr_c> flatten_idx;
        process_indexing(
                ths, old_dims, get_dense_stride(old_dims), idx, flatten_idx);
        auto flatten_remain = flatten_idx.back();
        expr flattened = 0UL;
        for (auto i = (int)slice_shape.size() - 1; i >= 0; i--) {
            flattened = (flatten_remain % slice_shape[i]) * parent_stride[i]
                    + flattened;
            flatten_remain = flatten_remain / slice_shape[i];
        }
        newidx = std::vector<expr_c> {std::move(flattened)};
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
            if (!oldptr->is_slice_ && oldptr->base_->ptr_.isa<tensorptr>()
                    && oldptr->base_->ptr_.static_as<tensorptr>()->is_slice_) {
                auto old_parent_ptr
                        = oldptr->base_->ptr_.static_as<tensorptr_c>();
                auto base_shape = oldptr->shape_;
                auto slice_shape = old_parent_ptr->shape_;
                auto parent_stride = get_base_stride(old_parent_ptr);
                process_slice_reshape_indexing(this, base_shape, slice_shape,
                        parent_stride, v->idx_, newidx);
            } else {
                const std::vector<expr> *shape = get_base_shape(oldptr);
                const std::vector<expr> stride = get_base_stride(oldptr);
                // 1D input might not have shape info
                assert(oldptr->base_->idx_.size() == 1 || !shape->empty());
                // flatten the indices using the reshaped dimensions
                process_indexing(this, *shape, stride, v->idx_, newidx);
            }
            // flatten the indices using tensor_ptr's dimensions
            // then, add the flattened 1D index with the flattened 1D offset
            // of the base tensor_ptr
            auto ptr = dispatch(v->ptr_).checked_as<tensorptr_c>();
            assert(ptr->base_->idx_.size() == 1);
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

stmt_c index_flattener_t::operator()(stmt_c f) {
    index_flatten_t pass;
    return pass.dispatch(std::move(f));
}

const_ir_module_ptr index_flattener_t::operator()(const_ir_module_ptr f) {
    index_flatten_t pass;
    return dispatch_module_on_visitor(&pass, f);
}

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
