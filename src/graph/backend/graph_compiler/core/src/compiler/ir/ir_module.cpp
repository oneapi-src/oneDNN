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

#include "ir_module.hpp"
#include <atomic>
#include <utility>
#include "builder.hpp"
#include "pass/func_dependency.hpp"
#include "visitor.hpp"
#include <compiler/ir/graph/dynamic_utils.hpp>
#include <compiler/ir/pass/ir_copy.hpp>
#include <compiler/ir/pass/printer.hpp>
#include <runtime/dynamic_dispatch/op_dispatch_tables.hpp>
#include <unordered_set>
#include <util/any_map.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

static std::atomic<int> rename_cnt = {0};
// this pass will
// break the link from call_nodes to function nodes with bodies
class func_unlinker_t : public ir_inplace_visitor_t {
public:
    using ir_inplace_visitor_t::dispatch_impl;
    expr visit_impl(call c) override {
        auto ret = ir_inplace_visitor_t::visit_impl(std::move(c)).as<call>();
        func_t callee = std::dynamic_pointer_cast<func_base>(ret->func_);
        if (!callee) { return ret; }
        // if the callee has a function body, unlink the call_node with the
        // function with body. Use the decl_ instead
        if (callee->body_.defined()) {
            assert(callee->decl_);
            callee = callee->decl_;
            return builder::remake_call(callee, ret->args_, ret);
        } else {
            assert(!callee->decl_);
        }
        // if the call is to a decl_, no changes
        return ret;
    }

    expr visit_impl(func_addr c) override {
        auto ret = ir_inplace_visitor_t::visit_impl(std::move(c))
                           .as<func_addr>();
        func_t callee = std::dynamic_pointer_cast<func_base>(ret->func_);
        if (!callee) { return ret; }
        // if the callee has a function body, unlink the call_node with the
        // function with body. Use the decl_ instead
        if (callee->body_.defined()) {
            assert(callee->decl_);
            ret->func_ = callee->decl_;
            return ret;
        } else {
            assert(!callee->decl_);
        }
        // if the call is to a decl_, no changes
        return ret;
    }

    func_t dispatch_impl(func_t f) override {
        if (f->decl_) {
            if (f->decl_->name_ != f->name_) { f->decl_->name_ = f->name_; }
        }
        return ir_inplace_visitor_t::dispatch_impl(std::move(f));
    }
};

std::shared_ptr<ir_module_t> ir_module_t::from_entry_func(
        context_ptr ctx, func_t funct) {
    auto ret = from_entry_func(
            std::move(ctx), std::vector<func_t> {std::move(funct)});
    ret->entry_func_idx_ = 0;
    return ret;
}

// resolve function dependency, returns funcs with their dependency
static std::vector<func_t> resolve_func(const std::vector<func_t> &funcs) {
    std::vector<func_t> dep;
    std::unordered_set<func_t> set;
    func_dependency_finder_t finder(dep);
    for (auto &f : funcs) {
        if (set.find(f) != set.end()) { continue; }
        dep.emplace_back(f);
        set.insert(f);
        size_t old_size = 0;
        do {
            size_t last_size = old_size;
            old_size = set.size();
            // run dependency_finder on newly added functions
            for (size_t i = last_size; i < dep.size(); i++) {
                finder(dep[i], set);
            }
            assert(set.size() == dep.size());
            // if old_size == set.size(), no dependency added, can exit
        } while (old_size != set.size());
    }
    return dep;
}

std::shared_ptr<ir_module_t> ir_module_t::from_entry_func(
        context_ptr ctx, const std::vector<func_t> &funcs) {
    auto ret = std::make_shared<ir_module_t>(std::move(ctx));
    ret->add_resolved_func(resolve_func(funcs));
    return ret;
}

ir_module_t *ir_module_t::merge(const ir_module_t &m) {
    assert(m.ctx_ == ctx_);
    add_resolved_func(m.get_contents());
    for (auto &v : m.module_vars_) {
        add_global_var(v);
    }
    for (auto &kv : m.op_table_map_) {
        add_op_table(kv);
    }
    return this;
}

ir_module_t *ir_module_t::merge(const std::vector<ir_module_ptr> &list) {
    for (auto &m : list) {
        merge(*m);
    }
    return this;
}

func_t ir_module_t::get_func(const std::string &name) const {
    auto itr = symbols_.find(name);
    if (itr != symbols_.end()) { return contents_.at(itr->second); }
    return func_t();
}

ir_module_ptr ir_module_t::copy() const {
    return std::make_shared<ir_module_t>(*this);
}

ir_module_ptr ir_module_t::copy_and_remove_funcs(
        const std::vector<bool> &mask) const {
    auto ret = std::make_shared<ir_module_t>(*this);
    auto entry_func = ret->entry_func_idx_;
    ret->entry_func_idx_ = -1;
    ret->symbols_ = {};
    std::vector<func_t> newcontents;
    for (size_t old_id = 0; old_id < ret->contents_.size(); old_id++) {
        if (mask.at(old_id)) {
            auto new_id = newcontents.size();
            auto the_func = ret->contents_[old_id];
            newcontents.emplace_back(the_func);
            ret->symbols_[the_func->name_] = new_id;
            if ((int)old_id == entry_func) { ret->entry_func_idx_ = new_id; }
        }
    }
    ret->contents_ = std::move(newcontents);
    return ret;
}

ir_module_ptr ir_module_t::deep_copy() const {
    auto ret = copy();
    std::unordered_map<expr_c, expr> replacer;
    for (auto &kv : ret->var_symbols_) {
        ir_copier_t cpy {replacer, true};
        kv.second = cpy(kv.second).remove_const().checked_as<define>();
    }
    for (auto &def : ret->module_vars_) {
        def = ret->var_symbols_[get_node_name(def->var_)];
    }
    for (auto &f : ret->contents_) {
        ir_copier_t cpy {replacer, true};
        f = std::const_pointer_cast<func_base>(cpy(f));
    }
    for (auto &kv : ret->op_table_map_) {
        kv.second = std::make_shared<op_dispatch_tables_t>(*kv.second);
    }
    return ret;
}

void ir_module_t::add_func(const std::vector<func_t> &funcs) {
    add_resolved_func(resolve_func(funcs));
}

var ir_module_t::make_global_var(sc_data_type_t dtype, const std::string &name,
        linkage linkage, expr init) {
    var ret = builder::make_var(dtype, name).static_as<var>();
    auto def = builder::make_var_tensor_def_unattached(
            ret, linkage, std::move(init))
                       .static_as<define>();
    add_global_var(std::move(def));
    return ret;
}

tensor ir_module_t::make_global_tensor(sc_data_type_t dtype,
        const std::string &name, const std::vector<expr> &dims,
        linkage linkage) {
    tensor ret = builder::make_tensor(name, dims, dtype).static_as<tensor>();
    auto def = builder::make_var_tensor_def_unattached(ret, linkage)
                       .static_as<define>();
    add_global_var(std::move(def));
    return ret;
}

tensor ir_module_t::make_global_stensor(sc_data_type_t dtype,
        const std::string &name, const std::vector<expr> &dims,
        const std::vector<expr> &strides, linkage linkage, stmt *out_def_node) {
    tensor ret = builder::make_stensor(name, dims, strides, dtype)
                         .static_as<tensor>();
    auto def = builder::make_var_tensor_def_unattached(ret, linkage)
                       .static_as<define>();
    if (out_def_node) { *out_def_node = def; }
    add_global_var(std::move(def));
    return ret;
}

void ir_module_t::add_global_var(define def) {
    auto &name = def->var_.isa<var>() ? def->var_.static_as<var>()->name_
                                      : def->var_.checked_as<tensor>()->name_;
    auto itr = var_symbols_.find(name);
    if (itr != var_symbols_.end()) {
        COMPILE_ASSERT(def->linkage_ != linkage::public_global,
                "Global var redefinition: " << name);
        std::string new_name;
        do {
            new_name = name + '_' + std::to_string(++rename_cnt);
            itr = var_symbols_.find(new_name);
        } while (itr != var_symbols_.end());
        name = new_name;
    }
    var_symbols_.insert(std::make_pair(name, def));
    module_vars_.emplace_back(std::move(def));
}

void ir_module_t::add_resolved_func(const std::vector<func_t> &funcs) {
    func_unlinker_t replacer;
    // all dependencies found, now check if any function name duplications
    for (auto &f : funcs) {
        // skip function decl_
        if (!f->body_.defined()) { continue; }
        auto itr = symbols_.find(f->name_);
        if (itr != symbols_.end()) {
            // if the function is already added to the module, skip
            if (contents_.at(itr->second) == f) { continue; }

            // try to rename the function if we are allowed to
            COMPILE_ASSERT(!f->attr_ || !f->attr_->has_key("entry_func"),
                    "The function "
                            << f->name_
                            << " is duplicated and is marked \'entry_func\'.")
            std::string name;
            do {
                name = f->name_ + '_' + std::to_string(++rename_cnt);
                itr = symbols_.find(name);
            } while (itr != symbols_.end());
            f->name_ = name;
            assert(f->decl_);
        }
        symbols_.insert(std::make_pair(f->name_, contents_.size()));
        replacer.dispatch_impl(f);
        contents_.emplace_back(f);
    }
}

void ir_module_t::add_op_table(
        const std::pair<std::string, op_dispatch_tables_ptr> &tb) {
    op_table_map_.insert(tb);
}

void ir_module_t::run_pass(function_pass_t &pass) {
    for (auto &f : contents_) {
        f = std::const_pointer_cast<func_base>(pass(f));
    }
}

func_t ir_module_t::make_init_func() const {
    if (!module_vars_.empty()) {
        stmts seq = make_stmt<stmts_node_t>(std::vector<stmt>());
        for (auto &v : module_vars_) {
            assert(v->linkage_ == linkage::private_global
                    || v->linkage_ == linkage::public_global);
            if (v->init_.defined()) {
                seq->seq_.emplace_back(
                        builder::make_assign_unattached(v->var_, v->init_));
            }
        }
        if (seq->seq_.empty()) return func_t();
        auto ret = builder::make_func("__sc_init__", std::vector<expr_c>(),
                std::move(seq), datatypes::void_t);
        return ret;
    }
    return func_t();
}
ostream &operator<<(ostream &os, const ir_module_t &m) {
    ir_printer_t p {os};
    return p.do_dispatch(m);
}
ostream &operator<<(ostream &os, const const_ir_module_ptr &m) {
    return (os << *m);
}

ostream &operator<<(ostream &os, const ir_module_ptr &m) {
    return (os << const_ir_module_ptr(m));
}

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
