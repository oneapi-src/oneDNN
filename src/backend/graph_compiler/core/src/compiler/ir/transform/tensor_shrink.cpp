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
#include "tensor_shrink.hpp"
#include <utility>
#include <vector>
#include "../ir_comparer.hpp"
#include "../visitor.hpp"
#include "auto_cast.hpp"
#include "constant_fold.hpp"
#include <compiler/dimensions.hpp>
#include <compiler/ir/builder.hpp>
#include <unordered_map>
#include <util/any_map.hpp>

namespace sc {

// check tensor node attr
static bool is_tensor_and_should_shrink(const expr &e) {
    return e.isa<tensor>()
            && e->attr().has_key(tensor_shrinker_attrs::should_shrink);
}
// check tensorptr node attr
static bool is_tensorptr_and_should_shrink(const expr &e) {
    return e.isa<tensorptr>() && e.static_as<tensorptr>()->base_.defined()
            && e.static_as<tensorptr>()->base_->ptr_.isa<tensor>()
            && is_tensor_and_should_shrink(
                    e.static_as<tensorptr>()->base_->ptr_.static_as<tensor>());
}
static bool is_reshaped_and_should_shrink(const expr &e) {
    return e.isa<tensorptr>() && !e.static_as<tensorptr>()->is_slice_
            && e.static_as<tensorptr>()->base_.defined()
            && e.static_as<tensorptr>()->base_->ptr_.isa<tensor>()
            && e->attr().has_key(tensor_shrinker_attrs::should_shrink)
            && e.static_as<tensorptr>()->base_->ptr_->attr().has_key(
                    tensor_shrinker_attrs::should_shrink);
}

static tensor get_real_tensor(const expr &buffer) {
    auto tsr = buffer;
    while (!tsr.isa<tensor>()) {
        COMPILE_ASSERT(
                tsr.isa<tensorptr>(), "Only tensor or tensorptr is accepted")
        auto base = tsr.static_as<tensorptr>()->base_;
        COMPILE_ASSERT(base.isa<indexing>(),
                "tensor_ptr base should be indexing, but got: " << base);
        tsr = base.static_as<indexing>()->ptr_;
    }
    return tsr.static_as<tensor>();
}

static constexpr const char *temp_shrink_tag = "tensor_shrinker.def";

/**
 * Due to in some cases, the brgemm may access discontinuous memory. If applied
 * tensor shrink, it should dynamically change leading dimension argument named
 * `LDX` in brgemm args list.
 * E.g. 1) For output[A,C,B,D], brgemm will write back
 *         it partial result into [1,C,1,D] with LDC = B*D. However, the local
 *         buffer should be shrinked into [C,D], but with new LDC = D.
 *      2) In strided writing cases, the LDC of the shrunk local buffer should
 * be updated as well. For output[A,C,D,B] and its partial result [a,c,d,b], the
 * old LDC=D*stride_w will be updated into LDC=d*stride_w.
 * */
bool check_brgemm_LDX(
        std::vector<expr> &args, const expr &buffer, int LD_arg_idx) {
    COMPILE_ASSERT(static_cast<uint64_t>(LD_arg_idx) < args.size(),
            "arg idx should not exceed args length, but got "
                    << args.size() << " args with LDX idx: " << LD_arg_idx);
    COMPILE_ASSERT(buffer.isa<tensor>() || buffer.isa<tensorptr>(),
            "tensor or tensorptr is expected for the buffer of brgemm");
    auto tsr = get_real_tensor(buffer);
    if (is_tensor_and_should_shrink(tsr)) {
        auto &shrink_info = tsr->attr_->get<tensor_shrinker_t::shrink_info_t>(
                tensor_shrinker_attrs::should_shrink);
        auto ld_arg_const
                = constant_folder_t()(auto_caster_t()(args[LD_arg_idx]));
        COMPILE_ASSERT(shrink_info.shape_.size() == tsr->dims_.size(),
                "Bad number of dimensions for indexing access");
        COMPILE_ASSERT(ld_arg_const.isa<constant>(),
                "Constant LDX is expected, but got " << ld_arg_const);
        int64_t LDX = get_expr_as_int(ld_arg_const);
        int64_t acc_orig = 1, acc_shrink = 1;
        // for conv_bwd_data stride_w > 1 cases
        args[LD_arg_idx]->attr();
        if (args[LD_arg_idx]->attr_->get_or_else("plain_init", false)) {
            acc_shrink = get_expr_as_int(shrink_info.shape_.back());
            args[LD_arg_idx]
                    = make_expr<constant_node>(acc_shrink, datatypes::s32);
            return true;
        }
        if (args[LD_arg_idx]->attr_->get_or_else("stride_w", 1) > 1) {
            auto N_axes = args[LD_arg_idx]->attr_->get_or_else(
                    "N_axes", std::vector<size_t> {});
            // plain
            if (N_axes.size() == 1) {
                COMPILE_ASSERT(N_axes[0] == tsr->dims_.size() - 1,
                        "currently only supports N is the last axis in plain "
                        "brgemm");
                acc_shrink = args[LD_arg_idx]->attr_->get<int>("stride_w")
                        * get_expr_as_int(shrink_info.shape_.back());
                args[LD_arg_idx]
                        = make_expr<constant_node>(acc_shrink, datatypes::s32);
                return true;
            }
            // blocking
            for (int64_t i
                    = static_cast<int64_t>(shrink_info.shape_.size()) - 1;
                    i >= 0; i--) {
                // when acc_orig > LDX, considering LDX that contains stride
                if (acc_orig > LDX) {
                    if (acc_shrink == acc_orig)
                        return false;
                    else {
                        args[LD_arg_idx] = make_expr<constant_node>(
                                acc_shrink, datatypes::s32);
                        return true;
                    }
                }
                acc_orig *= get_expr_as_int(tsr->dims_[i]);
                acc_shrink *= get_expr_as_int(shrink_info.shape_[i]);
            }
        } else {
            for (int64_t i
                    = static_cast<int64_t>(shrink_info.shape_.size()) - 1;
                    i >= 0; i--) {
                if (acc_orig >= LDX) {
                    if (acc_shrink == acc_orig)
                        return false;
                    else {
                        args[LD_arg_idx] = make_expr<constant_node>(
                                acc_shrink, datatypes::s32);
                        return true;
                    }
                }
                acc_orig *= get_expr_as_int(tsr->dims_[i]);
                acc_shrink *= get_expr_as_int(shrink_info.shape_[i]);
            }
        }
        COMPILE_ASSERT(0,
                "Unexpected LDX found: " << LDX
                                         << " for corresponding tensor dims: "
                                         << utils::print_vector(tsr->dims_));
    }
    return false;
}

class shrinker_impl_t : public ir_visitor_t {
public:
    using ir_visitor_t::dispatch;
    using ir_visitor_t::visit;

    std::unordered_map<expr, expr> replace_map;

    stmt_c visit(define_c v) override {
        if (is_tensor_and_should_shrink(v->var_)) {
            auto tsr = v->var_.static_as<tensor>();
            bool no_init = !tsr->init_value_
                    || tsr->init_value_
                            == tensor_node::get_zero_tensor_initializer();
            COMPILE_ASSERT(no_init && !v->init_.defined()
                            && v->linkage_ == linkage::local,
                    "The tensor to shrink should not have init value or be "
                    "re-scheduled. And it should be a local tensor: "
                            << v);
            auto &shrink_info
                    = v->var_->attr_->get<tensor_shrinker_t::shrink_info_t>(
                            tensor_shrinker_attrs::should_shrink);
            COMPILE_ASSERT(shrink_info.shape_.size() == tsr->dims_.size(),
                    "Bad shape for shrinking the tensor: "
                            << v << ", target shape = "
                            << utils::print_vector(shrink_info.shape_));
            auto replacer = copy_attr(*tsr,
                    builder::make_tensor(tsr->name_ + "_shr",
                            shrink_info.shape_, tsr->elem_dtype_,
                            tsr->address_space_, tsr->init_value_));
            replacer->attr_->remove(tensor_shrinker_attrs::should_shrink);
            replace_map[tsr] = replacer;
            auto ret = builder::make_var_tensor_def_unattached(
                    replacer, v->linkage_);
            // if the tensor definition is moved
            if (shrink_info.move_def_.defined()) {
                shrink_info.move_def_->attr()[temp_shrink_tag] = ret;
                // the tensor def is moved, return empty
                return builder::make_stmts_unattached({});
            }
            return ret;
        }
        return ir_visitor_t::visit(v);
    }

    expr_c visit(intrin_call_c v) override {
        auto intrin = ir_visitor_t::visit(v).checked_as<intrin_call_c>();
        if (v->type_ == intrin_type::brgemm
                || v->type_ == intrin_type::list_brgemm) {
            // new args
            auto args_cpy = intrin->args_;
            std::vector<std::pair<int, int>> check_LDX_list = {
                    // // Input fusion
                    // {brgemm_args::A, brgemm_args::LDA},
                    // {brgemm_args::B, brgemm_args::LDB},
                    // Output fusion
                    {brgemm_args::C, brgemm_args::LDC},
            };
            bool changed = false;
            for (auto &check_pair : check_LDX_list) {
                // need to check old args v, due to some of `attr` maybe removed
                // in new args.
                if (check_brgemm_LDX(args_cpy, v->args_[check_pair.first],
                            check_pair.second)) {
                    changed = true;
                }
            }
            if (changed) {
                return copy_attr(*intrin,
                        make_expr<intrin_call_node>(intrin->type_, args_cpy,
                                *intrin->intrin_attrs_));
            }
        }
        // TODO(xxx): add implement for list brgemm, if necessary
        // else if (v->type_ == intrin_type::list_brgemm) {}
        return intrin;
    }

    expr_c visit(tensor_c v) override {
        // if shrinked tensors should not go here, unless it is a direct use of
        // the tensor, instead of indexing
        COMPILE_ASSERT(!v->attr_
                        || !v->attr_->has_key(
                                tensor_shrinker_attrs::should_shrink),
                "The shrinked tensor is referenced without indexing: " << v);
        return ir_visitor_t::visit(std::move(v));
    }

    expr_c visit(indexing_c v) override {
        if (is_tensor_and_should_shrink(v->ptr_)) {
            auto tsr = v->ptr_.static_as<tensor>();
            COMPILE_ASSERT(v->idx_.size() == tsr->dims_.size(),
                    "Bad number of dimensions for indexing access");
            auto itr = replace_map.find(tsr);
            COMPILE_ASSERT(itr != replace_map.end(),
                    "Tensor used before definition: " << v);
            std::vector<expr> new_idx;
            bool changed = ir_visitor_t::dispatch_expr_vector(v->idx_, new_idx);
            auto &shrink_info
                    = tsr->attr_->get<tensor_shrinker_t::shrink_info_t>(
                            tensor_shrinker_attrs::should_shrink);
            // already checked that in visit(define_c v). using assert now
            assert(new_idx.size() == shrink_info.base_.size());
            for (size_t i = 0; i < new_idx.size(); i++) {
                new_idx[i] = new_idx[i] - shrink_info.base_[i];
            }
            return builder::make_indexing(
                    itr->second, new_idx, v->dtype_.lanes_, v->mask_);
        } else if (v->ptr_.isa<tensorptr>()) {
            // transform &A[0 - a, 0 - b][a, b] to &A[0, 0][0, 0] to make index
            // calculation simpler. Todo: currently we don't support expression
            // simplify: 0 - a + a => 0
            auto new_ptr = dispatch(v->ptr_);
            std::vector<expr> new_idx;
            bool changed = ir_visitor_t::dispatch_expr_vector(v->idx_, new_idx);
            changed |= !new_ptr.ptr_same(v->ptr_);
            auto new_cur_idx = v->idx_;
            auto new_cld_idx = new_ptr.static_as<tensorptr>()->base_->idx_;
            if (new_cur_idx.size() == new_cld_idx.size()) {
                bool idx_changed = false;
                for (size_t i = 0; i < new_cld_idx.size(); i++) {
                    if (new_cld_idx[i].isa<sub>()) {
                        auto &lhs = new_cld_idx[i].static_as<sub>()->l_;
                        auto &rhs = new_cld_idx[i].static_as<sub>()->r_;
                        if (rhs.ptr_same(new_cur_idx[i].remove_const())
                                || (rhs.isa<constant>()
                                        && new_cur_idx[i].isa<constant>()
                                        && rhs->equals(new_cur_idx[i]))) {
                            new_cld_idx[i] = lhs;
                            new_cur_idx[i] = 0;
                            changed = true;
                            idx_changed = true;
                        }
                    }
                }
                // remake tensorptr
                if (idx_changed) {
                    const auto &tptr = new_ptr.static_as<tensorptr>();
                    new_ptr = builder::tensor_ptr(tptr->base_->ptr_,
                            new_cld_idx, tptr->shape_, tptr->is_slice_);
                }
            }
            if (changed) {
                return copy_attr(*v,
                        builder::make_indexing(new_ptr.remove_const(),
                                new_cur_idx, v->dtype_.lanes_, v->mask_));
            }
            return v;
        }
        return ir_visitor_t::visit(v);
    }

    stmt_c visit(evaluate_c v) override {
        auto old_call_node = v->value_.static_as<call>();
        auto old_args = old_call_node->args_;
        auto evaluate = ir_visitor_t::visit(v).checked_as<evaluate_c>();
        // after this visit(v) the c->args_[0] has been shrinked
        // so old_args rather than new args_cpy can be used to
        // judge whether we "should_shrink"
        bool is_call
                = evaluate->value_.defined() && evaluate->value_.isa<call>();
        if (is_call) {
            auto c = evaluate->value_.static_as<call>();
            auto func = std::dynamic_pointer_cast<func_base>(c->func_);
            // func can be a nullptr
            if (func && func->name_ == "dnnl_brgemm_init") {
                // old arg attributes are still available on visited args
                auto args_cpy = c->args_;
                std::vector<std::pair<int, int>> check_LDX_list = {
                        {0, 3}, // {C, LDC}
                };
                bool changed = false;
                for (auto &check_pair : check_LDX_list) {
                    // need to check old_args's buffer to get the correct
                    // "should_shrink" logic
                    if (check_brgemm_LDX(args_cpy, old_args[check_pair.first],
                                check_pair.second)) {
                        changed = true;
                    }
                }
                if (changed) {
                    return copy_attr(*evaluate,
                            builder::make_evaluate_unattached(
                                    make_expr<call_node>(c->func_, args_cpy)));
                }
            }
        }
        return evaluate;
    }

    /**
     * TO deal with reshaped tensor, we need to transform both idx and shape
     * from `tensorptr(tensorptr(base,{0,..},shape,false),idx,{},true)` to
     * `tensorptr(tensorptr(base,{0,..},newshape,false),newidx,{},true)`
     * */
    expr_c visit(tensorptr_c v) override {
        // transform based reshaped tensor's shape
        if (is_reshaped_and_should_shrink(v.remove_const())) {
            auto tptr = ir_visitor_t::visit(v).checked_as<tensorptr>();
            auto &shrink_info
                    = tptr->attr_->get<tensor_shrinker_t::shrink_info_t>(
                            tensor_shrinker_attrs::should_shrink);
            return builder::tensor_ptr(tptr->base_->ptr_,
                    std::vector<expr>(tptr->base_->idx_.size(), expr(0)),
                    shrink_info.shape_, v->is_slice_);
        }
        // transform reshaped tensorptr's idx
        else if (v->base_->ptr_.isa<tensorptr>()
                && is_reshaped_and_should_shrink(
                        v->base_->ptr_.static_as<tensorptr>())) {
            // get shrink info firstly due to it will not be returned by visit
            // below
            auto &shrink_info
                    = v->base_->ptr_->attr_
                              ->get<tensor_shrinker_t::shrink_info_t>(
                                      tensor_shrinker_attrs::should_shrink);
            auto tptr = ir_visitor_t::visit(v).checked_as<tensorptr>();
            auto inner_tptr = tptr->base_->ptr_;
            std::vector<expr> newidx;
            bool changed = ir_visitor_t::dispatch_expr_vector(
                    tptr->base_->idx_, newidx);

            // already checked that in visit(define_c v). using assert now
            assert(newidx.size() == shrink_info.base_.size());
            for (size_t i = 0; i < newidx.size(); i++) {
                newidx[i] = newidx[i] - shrink_info.base_[i];
            }
            return builder::tensor_ptr(tptr->base_->ptr_, newidx, {}, true);
        }
        return ir_visitor_t::visit(v);
    }

    stmt_c visit(stmts_c s) override {
        if (s->attr_ && s->attr_->has_key(temp_shrink_tag)) {
            COMPILE_ASSERT(
                    s->seq_.empty(), "Shrink definition placeholder not empty");
            auto def = s->attr_->get<stmt>(temp_shrink_tag);
            s->attr_->as_map().clear();
            return def;
        }
        return ir_visitor_t::visit(std::move(s));
    }
};

func_c tensor_shrinker_t::operator()(func_c f) {
    shrinker_impl_t impl;
    return impl.dispatch(f);
}
stmt_c tensor_shrinker_t::operator()(stmt_c f) {
    shrinker_impl_t impl;
    return impl.dispatch(std::move(f));
}

} // namespace sc
