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

#include <string>
#include <utility>
#include <vector>
#include <unordered_map>
#include <unordered_set>

#include "dyn_tsr_transform.hpp"
#include <compiler/ir/builder.hpp>
#include <compiler/ir/builtin.hpp>
#include <compiler/ir/pass_dep_util.hpp>
#include <compiler/ir/transform/dead_write_eliminate.hpp>
#include <compiler/ir/transform/tensor2var.hpp>
#include <compiler/ir/util_module_passes.hpp>
#include <compiler/ir/visitor.hpp>
#include <runtime/dynamic_dispatch/dynamic_tensor.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

SC_DECL_PASS_INFO(dyn_tensor_transformer, SC_PASS_DEPENDS_ON(),
        SC_PASS_REQUIRE_STATE(), SC_PASS_REQUIRE_NOT_STATE(),
        SC_PASS_SET_STATE(), SC_PASS_UNSET_STATE());

class tensor_transform_impl_t : public ir_visitor_t {
public:
    using ir_visitor_t::dispatch;
    using ir_visitor_t::visit;

    // dyn var has been defined in function;
    std::unordered_set<expr> var_defined_in_func_;
    // old tsr => new dyn tsr defined in func param, used for call node inside
    // function. Vector is used for different scope.
    std::vector<std::unordered_map<expr, expr>> tsr_replace_map_;
    // for second pass callee func decl replacement
    std::unordered_map<std::string, func_t> func_decl_replace_map_;
    // def stmts to insert before call node
    std::vector<stmt> def_stmts_;
    expr create_dyn_tsr_from_tensor(
            const expr &in, std::vector<stmt> &def_stmts) {
        COMPILE_ASSERT(in.isa<tensor>() || in.isa<tensorptr>(),
                "input should be a tensor or tensor ptr node.")
        auto placeholder
                = in->attr().get_or_else("temp.dyn_placeholder", expr());
        std::string name;
        expr tsr;
        if (in.isa<tensor>()) {
            name = in.checked_as<tensor>()->name_;
            tsr = in;
        } else {
            tsr = in;
            while (tsr.isa<tensorptr>()) {
                tsr = tsr.checked_as<tensorptr>()->base_->ptr_;
            }
            name = tsr.checked_as<tensor>()->name_ + "_tptr";
        }
        auto dyn_tsr = builder::make_tensor(std::string("dyn_") + name,
                {sizeof(runtime::dynamic_tensor_t)}, datatypes::u8);
        def_stmts.push_back(builder::make_var_tensor_def_unattached(dyn_tsr));

        expr shape_tsr, dtype, dyn_mask, ndims;
        if (placeholder.defined()
                && placeholder->attr().has_key(
                        "temp.dyn_shape_of_placeholder")) {
            shape_tsr = placeholder->attr().get<expr>(
                    "temp.dyn_shape_of_placeholder");
            ndims = builder::make_read_struct(placeholder,
                    dyn_tsr_struct_t::name, dyn_tsr_struct_t::fields::ndims);
            dtype = builder::make_read_struct(placeholder,
                    dyn_tsr_struct_t::name, dyn_tsr_struct_t::fields::dtype);
            dyn_mask = builder::make_read_struct(placeholder,
                    dyn_tsr_struct_t::name, dyn_tsr_struct_t::fields::dyn_mask);
        } else {
            // should be in (tptr/tsr)
            auto plain_dims
                    = in->attr_->get<std::vector<expr>>(attr_keys::plain_dims);
            ndims = builder::make_constant({plain_dims.size()}, datatypes::s32);
            shape_tsr = builder::make_tensor(std::string("dyn_shape_") + name,
                    {plain_dims.size()}, datatypes::index);
            shape_tsr->attr().set(attr_keys::no_tensor2var, true);
            shape_tsr->attr().set(attr_keys::no_dead_write, true);
            int64_t etype = tsr->dtype_.is_etype_pointer()
                    ? tsr->dtype_.get_pointer_element().as_etype_int()
                    : tsr->dtype_.as_etype_int();
            dtype = builder::make_constant({etype}, datatypes::u32);
            dyn_mask = builder::make_var(
                    datatypes::u8, std::string("dyn_mask_") + name);
            uint64_t dyn_mask_int = 0;
            def_stmts.push_back(
                    builder::make_var_tensor_def_unattached(shape_tsr));
            for (size_t i = 0; i < plain_dims.size(); i++) {
                if (plain_dims[i].isa<constant>()
                        || var_defined_in_func_.find(plain_dims[i])
                                != var_defined_in_func_.end()) {
                    def_stmts.push_back(builder::make_assign_unattached(
                            builder::make_indexing(shape_tsr, i),
                            plain_dims[i]));
                }
                dyn_mask_int |= (uint64_t(!plain_dims[i].isa<constant>()) << i);
            }
            def_stmts.push_back(builder::make_var_tensor_def_unattached(
                    dyn_mask, linkage::local,
                    builder::make_constant({dyn_mask_int}, datatypes::u8)));
        }
        def_stmts.push_back(
                builder::make_evaluate_unattached(builder::make_write_struct(
                        dyn_tsr, in.remove_const(), dyn_tsr_struct_t::name,
                        dyn_tsr_struct_t::fields::data_ptr)));
        def_stmts.push_back(
                builder::make_evaluate_unattached(builder::make_write_struct(
                        dyn_tsr, shape_tsr, dyn_tsr_struct_t::name,
                        dyn_tsr_struct_t::fields::dim_ptr)));
        def_stmts.push_back(
                builder::make_evaluate_unattached(builder::make_write_struct(
                        dyn_tsr, ndims, dyn_tsr_struct_t::name,
                        dyn_tsr_struct_t::fields::ndims)));
        def_stmts.push_back(
                builder::make_evaluate_unattached(builder::make_write_struct(
                        dyn_tsr, dtype, dyn_tsr_struct_t::name,
                        dyn_tsr_struct_t::fields::dtype)));
        def_stmts.push_back(
                builder::make_evaluate_unattached(builder::make_write_struct(
                        dyn_tsr, dyn_mask, dyn_tsr_struct_t::name,
                        dyn_tsr_struct_t::fields::dyn_mask)));
        return dyn_tsr;
    }

    expr_c visit(intrin_call_c v) override {
        switch (v->type_) {
            case intrin_type::read_struct:
            case intrin_type::write_struct: {
                if (v->intrin_attrs_->get<std::string>(intrin_attr::struct_name)
                        == dyn_tsr_struct_t::name) {
                    std::vector<expr> new_args;
                    bool changed = dispatch_expr_vector(v->args_, new_args);
                    // only consider first arg.
                    auto &arg = v->args_[0];
                    auto it = tsr_replace_map_.back().find(arg);
                    if (it != tsr_replace_map_.back().end()) {
                        new_args[0] = it->second;
                        changed = true;
                    }
                    if (changed) {
                        return copy_attr(*v,
                                make_expr<intrin_call_node>(
                                        v->type_, new_args, *v->intrin_attrs_));
                    }
                }
                return v;
                break;
            }
            default: return v;
        }
    }
    expr_c visit(call_c v) override {
        if (v->func_->attr().get_or_else(attr_keys::forbid_trans, false)) {
            return v;
        }
        std::vector<expr> new_args(v->args_.begin(), v->args_.end());
        bool changed = false;
        func_t the_func = std::dynamic_pointer_cast<func_base>(v->func_);
        func_t proto_func = v->get_prototype();
        assert(v->args_.size() == proto_func->params_.size());
        assert(def_stmts_.empty());
        // call kernel
        bool trans_all_params = (the_func && the_func->attr_
                                        && the_func->attr_->get_or_else(
                                                attr_keys::always_trans, false))
                || (proto_func->attr_
                        && proto_func->attr_->get_or_else(
                                attr_keys::always_trans, false));
        for (size_t i = 0; i < v->args_.size(); i++) {
            auto &arg = v->args_[i];
            auto it = tsr_replace_map_.back().find(arg);
            if (it != tsr_replace_map_.back().end()) {
                new_args[i] = it->second;
                changed = true;
            } else if (arg.isa<tensor>()) {
                auto tsr_arg = arg.static_as<tensor>();
                bool should_trans = false;
                for (auto &d : tsr_arg->dims_) {
                    if (d.isa<var>()) {
                        should_trans = true;
                        break;
                    }
                }
                should_trans = should_trans || trans_all_params
                        || tsr_arg->attr().get_or_else(
                                attr_keys::always_trans, false);
                if (should_trans) {
                    auto dyn_tsr
                            = create_dyn_tsr_from_tensor(tsr_arg, def_stmts_);
                    tsr_replace_map_.back().insert(
                            std::make_pair(arg, dyn_tsr));
                    new_args[i] = dyn_tsr;
                    changed = true;
                }
            } else if (arg.isa<tensorptr>()) {
                auto tsr_arg = arg.static_as<tensorptr>();
                bool should_trans = false;
                for (auto &d : tsr_arg->shape_) {
                    if (d.isa<var>()) {
                        should_trans = true;
                        break;
                    }
                }
                should_trans = should_trans || trans_all_params
                        || tsr_arg->attr().get_or_else(
                                attr_keys::always_trans, false);
                if (should_trans) {
                    auto dyn_tsr
                            = create_dyn_tsr_from_tensor(tsr_arg, def_stmts_);
                    tsr_replace_map_.back().insert(
                            std::make_pair(arg, dyn_tsr));
                    new_args[i] = dyn_tsr;
                    changed = true;
                }
            }
        }
        if (changed) {
            if (the_func) {
                return builder::remake_call(the_func, new_args, v);
            } else {
                return copy_attr(*v, make_expr<call_node>(v->func_, new_args));
            }
        }
        return v;
    }
    stmt_c visit(stmts_c v) override {
        tsr_replace_map_.emplace_back(tsr_replace_map_.back());
        bool changed = false;
        std::vector<stmt_c> seq;
        seq.reserve(v->seq_.size());
        for (size_t i = 0; i < v->seq_.size(); i++) {
            def_stmts_.clear();
            auto &st = v->seq_[i];
            auto n = dispatch(st);
            if (!def_stmts_.empty()) {
                changed = true;
                seq.insert(seq.end(), def_stmts_.begin(), def_stmts_.end());
                def_stmts_.clear();
            }
            if (!n.ptr_same(st)) { changed = true; }
            seq.emplace_back(std::move(n));
        }
        tsr_replace_map_.pop_back();
        if (changed) {
            return copy_attr(*v, builder::make_stmts_unattached(seq));
        }
        return v;
    }

    // eliminate duplicate dynamic var definition.
    stmt_c visit(define_c v) override {
        if (var_defined_in_func_.find(v->var_) != var_defined_in_func_.end()) {
            return builder::make_stmts_unattached({});
        }
        auto ret = ir_visitor_t::visit(v).checked_as<define_c>();
        if (ret->var_.isa<var>()) { var_defined_in_func_.insert(ret->var_); }
        return ret;
    }

    func_c dispatch(func_c v) override {
        if (std::const_pointer_cast<func_base>(v)->attr().get_or_else(
                    attr_keys::forbid_trans, false)) {
            return v;
        }
        tsr_replace_map_.clear();
        tsr_replace_map_.emplace_back();
        bool all_const = true;
        std::vector<expr> new_params;
        // base and shape define stmts insert to front of body.
        assert(def_stmts_.empty());
        var_defined_in_func_.clear();
        new_params.reserve(v->params_.size());
        // kernel func should transform all tensors.
        bool trans_all_params = v->attr_
                && v->attr_->get_or_else(attr_keys::always_trans, false);
        for (auto &p : v->params_) {
            if (p.isa<tensor>()) {
                auto old_tsr = p.static_as<tensor>();
                if (old_tsr->name_ == "partial_out") {
                    int a = 1;
                    int b = 2;
                }
                std::vector<std::pair<size_t, expr>> shape_vars;
                bool cur_const = true;
                auto plain_dims
                        = old_tsr->attr().get_or_else<std::vector<expr>>(
                                attr_keys::plain_dims, std::vector<expr>());
                for (size_t i = 0; i < plain_dims.size(); i++) {
                    auto &d = plain_dims[i];
                    if (!d.isa<constant>()) {
                        assert(d.isa<var>());
                        if (var_defined_in_func_.find(d)
                                == var_defined_in_func_.end()) {
                            shape_vars.emplace_back(i, d);
                            var_defined_in_func_.insert(d);
                        }
                        cur_const = false;
                    }
                }
                if (trans_all_params
                        || p->attr().get_or_else(
                                attr_keys::always_trans, false)) {
                    cur_const = false;
                }
                if (!cur_const) {
                    expr dyn_tsr;
                    auto it = tsr_replace_map_.back().find(old_tsr);
                    if (it != tsr_replace_map_.back().end()) {
                        dyn_tsr = it->second;
                    } else {
                        dyn_tsr = builder::make_tensor(
                                std::string("dyn_") + old_tsr->name_,
                                {sizeof(runtime::dynamic_tensor_t)},
                                datatypes::u8);
                        tsr_replace_map_.back().insert(
                                std::make_pair(old_tsr, dyn_tsr));
                    }
                    new_params.push_back(dyn_tsr);

                    if (!shape_vars.empty()) {
                        // shape tensor define
                        auto dyn_shape_tsr = builder::make_tensor(
                                std::string("dyn_shape_") + old_tsr->name_,
                                {plain_dims.size()}, datatypes::index);
                        dyn_shape_tsr->attr().set(
                                attr_keys::no_tensor2var, true);
                        auto def = builder::make_var_tensor_def_unattached(
                                dyn_shape_tsr, linkage::local,
                                builder::make_read_struct(dyn_tsr,
                                        dyn_tsr_struct_t::name,
                                        dyn_tsr_struct_t::fields::dim_ptr));
                        def_stmts_.push_back(def);
                        for (auto &var_pair : shape_vars) {
                            def_stmts_.push_back(
                                    builder::make_var_tensor_def_unattached(
                                            var_pair.second, linkage::local,
                                            builder::make_indexing(
                                                    dyn_shape_tsr,
                                                    var_pair.first)));
                        }
                    }
                    // base tensor define, may be tensor ptr
                    if (old_tsr.isa<tensor>()) {
                        old_tsr->attr().set(attr_keys::no_dead_write, true);
                        old_tsr->attr().set(attr_keys::no_tensor2var, true);
                        auto def = builder::make_var_tensor_def_unattached(
                                old_tsr, linkage::local,
                                builder::make_read_struct(dyn_tsr,
                                        dyn_tsr_struct_t::name,
                                        dyn_tsr_struct_t::fields::data_ptr));
                        def_stmts_.push_back(def);
                    }
                } else {
                    new_params.emplace_back(old_tsr);
                }
                all_const &= cur_const;
            } else {
                new_params.push_back(p);
            }
        }
        std::vector<stmt> def_stmts_backup = std::move(def_stmts_);
        def_stmts_.clear();
        auto body = dispatch(v->body_);
        bool changed = !all_const || !body.ptr_same(v->body_);
        var_defined_in_func_.clear();
        if (changed) {
            if (!all_const) {
                if (body.isa<stmts>()) {
                    auto seq = body.static_as<stmts>()->seq_;
                    seq.insert(seq.begin(), def_stmts_backup.begin(),
                            def_stmts_backup.end());
                    body = copy_attr(
                            *body, make_stmt<stmts_node_t>(std::move(seq)));
                } else {
                    def_stmts_backup.push_back(body.remove_const());
                    body = make_stmt<stmts_node_t>(std::move(def_stmts_backup));
                }
            }
            auto new_func = copy_attr(*v,
                    builder::make_func(v->name_, new_params,
                            body.remove_const(), v->ret_type_));
            // save map from symbol to decl.
            func_decl_replace_map_.insert(
                    std::make_pair(v->name_, new_func->decl_));
            return std::move(new_func);
        }
        return v;
    }
};

class func_decl_replace_impl_t : public ir_visitor_t {
public:
    using ir_visitor_t::dispatch;
    using ir_visitor_t::visit;
    // inherit from tensor_transform pass
    std::unordered_map<std::string, func_t> &func_decl_replace_map_;

    expr_c visit(call_c v) override {
        func_t the_func = std::dynamic_pointer_cast<func_base>(v->func_);
        if (the_func) {
            auto it = func_decl_replace_map_.find(the_func->name_);
            if (it != func_decl_replace_map_.end()) {
                return copy_attr(*v,
                        builder::make_call(
                                std::const_pointer_cast<func_base>(it->second),
                                v->args_));
            }
        } else {
            func_t proto_func = v->get_prototype();
            auto it = func_decl_replace_map_.find(proto_func->name_);
            if (it != func_decl_replace_map_.end()) {
                v->func_->attr().set("prototype", it->second);
                return copy_attr(*v, make_expr<call_node>(v->func_, v->args_));
            }
        }
        return v;
    }
    func_decl_replace_impl_t(
            std::unordered_map<std::string, func_t> &replace_map)
        : func_decl_replace_map_(replace_map) {}
};

func_c dyn_tensor_transformer_t::operator()(func_c f) {
    tensor_transform_impl_t trans_pass;
    f = trans_pass.dispatch(f);
    func_decl_replace_impl_t replace_pass(trans_pass.func_decl_replace_map_);
    return replace_pass.dispatch(f);
}

const_ir_module_ptr dyn_tensor_transformer_t::operator()(
        const_ir_module_ptr f) {
    tensor_transform_impl_t trans_pass;
    f = dispatch_module_on_visitor(&trans_pass, f);
    func_decl_replace_impl_t replace_pass(trans_pass.func_decl_replace_map_);
    f = dispatch_module_on_visitor(&replace_pass, f);
    return f;
}
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
