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
#include "module_globals_resolve.hpp"
#include <atomic>
#include <memory.h>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <compiler/ir/builder.hpp>
#include <compiler/ir/pass_dep_util.hpp>
#include <compiler/ir/transform/cpu/closurize.hpp>
#include <compiler/ir/visitor.hpp>
#include <runtime/config.hpp>
#include <unordered_map>
#include <util/any_map.hpp>

SC_MODULE(pass.module_globals_resolve);
namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

SC_DECL_PASS_INFO(module_globals_resolver,
        SC_PASS_DEPENDS_ON(closurizer_cpu, kernel_lowering_cpu),
        SC_PASS_REQUIRE_STATE(), SC_PASS_REQUIRE_NOT_STATE(),
        SC_PASS_SET_STATE(), SC_PASS_UNSET_STATE());

static bool is_expr_nullptr(const expr &e) {
    if (e->dtype_.is_pointer() && e.isa<constant>()) {
        return e.checked_as<constant>()->value_.front().u64 == 0;
    }
    return false;
}

class module_globals_resolver_impl_t : public ir_visitor_t {
public:
    using ir_visitor_t::dispatch;
    std::unordered_map<std::string, func_t> *map;
    expr current_base;
    expr current_rtl_ctx;
    bool is_top_level = true;
    // change the global symbol to a local variable
    std::vector<expr> global_symbol_local_def_;
    std::unordered_map<expr_c, expr> global_symbol_replace_map_;

    expr_c visit(call_c v) override {
        func_t the_func = std::dynamic_pointer_cast<func_base>(v->func_);
        expr the_expr;
        if (!the_func) {
            the_expr = expr(std::dynamic_pointer_cast<expr_base>(v->func_));
            assert(the_expr.defined() && the_expr->attr_->has_key("prototype"));
            the_func = the_expr->attr_->get<func_t>("prototype");
        }
        auto itr = map->find(the_func->name_);
        if (itr == map->end()) {
            // if is parallel-call function
            if (v->func_ == get_parallel_call_with_env_func(true)
                    || v->func_ == get_parallel_call_with_env_func(false)) {
                std::vector<expr> ret;
                dispatch_expr_vector(v->args_, ret);
                assert(ret.size() == 8UL);
                ret[2] = current_rtl_ctx;
                ret[3] = current_base;
                return copy_attr(*v, make_expr<call_node>(v->func_, ret));
            } else if (v->func_->attr_
                    && v->func_->attr_->get_or_else(
                            "is_brgemm_func_with_stream", false)) {
                COMPILE_ASSERT(!v->args_.empty()
                                && v->args_.back()->dtype_
                                        == datatypes::pointer,
                        "The last arg of brgemm function should be a "
                        "pointer, got "
                                << v);
                if (is_expr_nullptr(v->args_.back())) {
                    std::vector<expr> newargs;
                    dispatch_expr_vector(v->args_, newargs);
                    newargs.back() = current_rtl_ctx;
                    return copy_attr(*v, builder::make_call(the_func, newargs));
                }
            }
            return ir_visitor_t::visit(v);
        }
        std::vector<expr> ret;
        dispatch_expr_vector(v->args_, ret);
        ret.insert(ret.begin(), 2, expr());
        ret[0] = current_rtl_ctx;
        ret[1] = current_base;
        if (the_expr.defined()) {
            auto new_expr = the_expr->remake();
            new_expr->attr_->set("prototype", itr->second->decl_);
            return copy_attr(*v, make_expr<call_node>(new_expr, ret));
        }
        return copy_attr(*v, builder::make_call(itr->second->decl_, ret));
    }

    expr_c visit(func_addr_c v) override {
        auto itr = map->find(v->func_->name_);
        if (itr == map->end()) { return ir_visitor_t::visit(v); }
        return copy_attr(*v, builder::make_func_addr(itr->second->decl_));
    }

    expr_c get_or_create_new_var(const expr_c &old) {
        auto itr = global_symbol_replace_map_.find(old);
        if (itr != global_symbol_replace_map_.end()) { return itr->second; }
        auto ret = old->remake();
        global_symbol_local_def_.emplace_back(ret);
        global_symbol_replace_map_.insert(std::make_pair(old, ret));
        return ret;
    }

    // for var and tensor, if they are global, change them to local definitions
    expr_c visit(tensor_c v) override {
        if (v->attr_ && v->attr_->has_key(attr_keys::module_global_offset)) {
            return get_or_create_new_var(v);
        }
        return ir_visitor_t::visit(v);
    }

    expr_c visit(var_c v) override {
        if (v->attr_ && v->attr_->has_key(attr_keys::module_global_offset)) {
            return get_or_create_new_var(v);
        }
        return ir_visitor_t::visit(v);
    }

    // converts the global vars used in function to local vars defined in
    // top-level scope
    stmt_c visit(stmts_c v) override {
        if (is_top_level) {
            is_top_level = false;
            auto ret = ir_visitor_t::visit(v).checked_as<stmts>();
            if (global_symbol_local_def_.empty()) { return ret; }
            std::vector<stmt_c> seq;
            for (auto &v : global_symbol_local_def_) {
                expr init;
                if (v.isa<tensor>()) {
                    init = builder::tensor_ptr(current_base,
                            {v->attr().get<size_t>(
                                    attr_keys::module_global_offset)});
                }
                seq.emplace_back(builder::make_var_tensor_def_unattached(
                        v, linkage::local, init));
            }
            seq.insert(seq.end(), ret->seq_.begin(), ret->seq_.end());
            return copy_attr(*v, builder::make_stmts_unattached(seq));
        } else {
            return ir_visitor_t::visit(v);
        }
    }
};

static size_t align_to_64(size_t v) {
    return utils::divide_and_ceil(v, 64) * 64;
}

static size_t get_tensor_size(const tensor &tsr, const define &def) {
    COMPILE_ASSERT(tsr->dims_.size() == 1 && tsr->dims_[0].isa<constant>(),
            "The global tensor should be 1D and the dims should be "
            "constant, got "
                    << def);
    auto size = get_const_as_int(tsr->dims_[0].static_as<constant>())
            * utils::get_sizeof_type(tsr->elem_dtype_);
    return size;
}

static size_t update_allocated_size(
        const tensor &tsr, size_t size, size_t allocated_size) {
    if (size >= 64) { allocated_size = align_to_64(allocated_size); }
    tsr->attr()[attr_keys::module_global_offset] = allocated_size;
    allocated_size += size;
    return allocated_size;
}

const_ir_module_ptr module_globals_resolver_t::operator()(
        const_ir_module_ptr m) {
    auto ret = std::make_shared<ir_module_t>(*m);
    // first, plan the memory layout for the static buffer to be used in an IR
    // module:
    // ------------
    // global vars
    // ------------
    // initialized global tensors
    // ------------
    // uninitialized global tensors
    // ------------
    size_t allocated_size = 0;
    for (auto &def : ret->get_module_vars()) {
        if (def->var_.isa<var>()) {
            auto size = utils::get_sizeof_type(def->var_->dtype_);
            // natural alignment of the address
            allocated_size
                    = utils::divide_and_ceil(allocated_size, size) * size;
            def->var_->attr()[attr_keys::module_global_offset] = allocated_size;
            allocated_size += utils::get_sizeof_type(def->var_->dtype_);
        }
    }
    // initialized tensors
    for (auto &def : ret->get_module_vars()) {
        if (def->var_.isa<tensor>()) {
            auto tsr = def->var_.static_as<tensor>();
            if (tsr->init_value_) {
                auto size = get_tensor_size(tsr, def);
                COMPILE_ASSERT(tsr->init_value_->size_ == size,
                        "The size of the global tensor ("
                                << tsr->init_value_->size_
                                << ") does not match its "
                                   "initializer ("
                                << size << "): " << def);
                allocated_size
                        = update_allocated_size(tsr, size, allocated_size);
            }
        }
    }
    size_t init_size = allocated_size;
    for (auto &def : ret->get_module_vars()) {
        if (def->var_.isa<tensor>()) {
            auto tsr = def->var_.static_as<tensor>();
            if (!tsr->init_value_) {
                auto size = get_tensor_size(tsr, def);
                allocated_size
                        = update_allocated_size(tsr, size, allocated_size);
            }
        }
    }

    std::shared_ptr<statics_table_t> sym_table
            = std::make_shared<statics_table_t>(
                    aligned_buffer_t(allocated_size, m->ctx_->engine_));
    sym_table->initialized_size_ = init_size;
    for (auto &def : ret->get_module_vars()) {
        auto offset = def->var_->attr().get<size_t>(
                attr_keys::module_global_offset);
        if (def->var_.isa<var>()) {
            sym_table->add(def->var_.static_as<var>()->name_, offset);
        } else {
            auto tsr = def->var_.checked_as<tensor>();
            sym_table->add(tsr->name_, offset);
            if (tsr->init_value_) {
                memcpy(reinterpret_cast<char *>(sym_table->data_.data_)
                                + offset,
                        tsr->init_value_->data_, tsr->init_value_->size_);
            }
        }
    }
    ret->attr_[ir_module_t::attr_key_t::MODULE_DATA_BUFFERS]
            = std::move(sym_table);
    auto &funcs = ret->get_contents();
    // second, pass the base pointer of the global buffer from "main entry" func
    // to all IR functions. We also need to replace the call nodes and append a
    // new parameter to pass down the global buffer pointer
    std::unordered_map<std::string, func_t> replace_map;
    for (auto &f : funcs) {
        if (!f->body_.defined()) { continue; }
        if (f->attr_
                && f->attr_->get_or_else(function_attrs::low_level, false)) {
            continue;
        }
        auto params = f->params_;
        // insert two placeholders
        params.insert(params.begin(), 2, expr());
        params[0] = builder::make_var(datatypes::pointer, "__stream");
        params[1] = builder::make_tensor("__module_data", {0UL}, datatypes::s8);
        auto retf = builder::make_func(f->name_, params,
                builder::make_stmts_unattached({}), f->ret_type_);
        if (f->attr_) {
            // we added new args, we need to update the comments
            if (auto comments = f->attr_->get_or_null<std::vector<std::string>>(
                        "comments")) {
                if (comments->size() >= 2
                        && utils::string_startswith((*comments)[1], "@param")) {
                    std::vector<std::string> new_comments = {comments->at(0),
                            "@param __stream the stream pointer, usually "
                            "get_default_stream()",
                            "@param __module_data the module global data"};
                    for (size_t i = 1; i < comments->size(); i++) {
                        auto &comment = (*comments)[i];
                        new_comments.emplace_back(comment);
                    }
                    retf->decl_->attr()["comments"] = new_comments;
                    retf->attr()["comments"] = std::move(new_comments);
                }
            }
        }
        auto *funcp = retf.get();
        replace_map[f->name_] = copy_attr(*f, std::move(retf));
        funcp->decl_ = copy_attr(*f, std::move(funcp->decl_));
    }

    for (unsigned i = 0; i < funcs.size(); i++) {
        if (!funcs[i]->body_.defined()) { continue; }
        if (funcs[i]->attr_
                && funcs[i]->attr_->get_or_else(
                        function_attrs::low_level, false)) {
            continue;
        }
        module_globals_resolver_impl_t impl;
        impl.map = &replace_map;
        auto funcp = replace_map[funcs[i]->name_];
        assert(funcp);
        impl.current_rtl_ctx = funcp->params_[0];
        impl.current_base = funcp->params_[1];
        funcp->body_ = impl.dispatch(funcs[i]->body_).remove_const();
        funcs[i] = funcp;
    }
    return ret;
}

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
