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
#include <compiler/ir/attr_keys.hpp>
#include <compiler/ir/builder.hpp>
#include <compiler/ir/graph/pass/graph_constant_cache.hpp>
#include <compiler/ir/pass_dep_util.hpp>
#include <compiler/ir/transform/cpu/closurize.hpp>
#include <compiler/ir/visitor.hpp>
#include <runtime/config.hpp>
#include <runtime/const_cache_wrapper.hpp>
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

static const char *shared_const_handle_name = "__shared_const_handle";

class module_globals_resolver_impl_t : public ir_visitor_t {
public:
    using ir_visitor_t::dispatch;
    std::unordered_map<std::string, func_t> *map;
    expr current_base;
    expr current_rtl_ctx;
    bool is_top_level = true;
    uintptr_t absolute_base_ = 0;
    // change the global symbol to a local variable
    std::vector<expr> global_symbol_local_def_;
    std::unordered_map<expr_c, expr> global_symbol_replace_map_;
    std::vector<expr> shared_const_base_tsr_;

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
            the_expr->attr_->set("prototype", itr->second->decl_);
            return copy_attr(*v, make_expr<call_node>(the_expr, ret));
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
        // if using absolute base, the module_global_offset is the pointer of
        // the global var
        if (absolute_base_ && ret.isa<var>()) {
            auto &attr = ret->attr_->get_any(attr_keys::module_global_offset);
            void *ptr = reinterpret_cast<void *>(
                    attr.get<size_t>() + absolute_base_);
            attr = ptr;
        }
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

    stmt_c visit(define_c v) override {
        if (!v->init_.defined()) {
            if (auto buf = any_map_t::fetch_or_else<
                        std::shared_ptr<cached_const_graph_tensor>>(
                        v->var_->attr_.get(), attr_keys::shared_const,
                        nullptr)) {
                auto idx = v->attr_->get<size_t>("shared_const_handle_idx");
                auto newvar = dispatch(v->var_);
                COMPILE_ASSERT(!v->init_.defined(),
                        "Bad shared const tensor with init");
                return copy_attr(*v,
                        builder::make_var_tensor_def_unattached(newvar,
                                linkage::local,
                                builder::tensor_ptr(
                                        shared_const_base_tsr_.at(idx),
                                        {buf->offset_})));
            }
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
                    auto anyoffset = v->attr().get_any(
                            attr_keys::module_global_offset);
                    if (auto poffset = anyoffset.get_or_null<size_t>()) {
                        auto offset = *poffset;
                        if (absolute_base_) {
                            init = make_expr<constant_node>(
                                    (uint64_t)(absolute_base_ + offset),
                                    datatypes::s8.get_pointerof());
                        } else {
                            init = builder::tensor_ptr(current_base, {offset});
                        }
                    } else {
                        init = make_expr<constant_node>(
                                (uint64_t)(anyoffset.get<void *>()),
                                datatypes::s8.get_pointerof());
                    }
                }
                seq.emplace_back(builder::make_var_tensor_def_unattached(
                        v, linkage::local, init));
            }
            auto old_seq_size = seq.size();
            if (shared_const_base_tsr_.empty()) {
                seq.insert(seq.end(), ret->seq_.begin(), ret->seq_.end());
            } else {
                // insert shared_const_base_tsr definition after definition and
                // initialization of is_init
                bool found_init = false;
                bool shared_const_base_inserted = false;
                for (auto &olds : ret->seq_) {
                    if (!shared_const_base_inserted) {
                        if (any_map_t::fetch_or_else(olds->attr_.get(),
                                    attr_keys::is_shared_const_init_stmt,
                                    false)) {
                            found_init = true;
                        } else {
                            if (found_init) {
                                for (auto &tsr : shared_const_base_tsr_) {
                                    // the function name is too long for clang
                                    // format to break
                                    // clang-format off
                                    seq.emplace_back(
                                        builder::make_var_tensor_def_unattached(
                                                            tsr,
                                                            linkage::local));
                                    // clang-format on
                                }
                                shared_const_base_inserted = true;
                            }
                        }
                    }
                    seq.emplace_back(olds);
                }
                if (!shared_const_base_inserted) {
                    for (auto &tsr : shared_const_base_tsr_) {
                        seq.insert(seq.begin() + old_seq_size,
                                builder::make_var_tensor_def_unattached(
                                        tsr, linkage::local));
                    }
                }
            }
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
    if (auto absptr = any_map_t::fetch_or_else<void *>(
                tsr->attr_.get(), attr_keys::static_global, nullptr)) {
        tsr->attr()[attr_keys::module_global_offset] = absptr;
        return allocated_size;
    }
    if (size >= 64) { allocated_size = align_to_64(allocated_size); }
    tsr->attr()[attr_keys::module_global_offset] = allocated_size;
    allocated_size += size;
    return allocated_size;
}

/**
 * shared const tensors are special local tensors. We collect all shared const
 * tensors in main-entry and assign a "handle" for each of the shared const
 * tensors. The handles are pointers to runtime::const_cache_proxy. To keep them
 * alive, we put them in statics_table_t in the final jit_module
 * */
static std::vector<void *> process_shared_const_tensors(const ir_module_ptr &m,
        std::vector<std::shared_ptr<cached_const_graph_tensor>> &out_graph_tsr,
        std::vector<expr> &out_base_tsr) {
    auto mainf = m->get_entry_func();
    if (!mainf) { return {}; }
    std::vector<void *> collected_base;
    auto &seq = mainf->body_.checked_as<stmts>()->seq_;
    auto bases = m->attr_.get_or_null<
            std::vector<std::shared_ptr<runtime::const_cache_proxy>>>(
            ir_module_t::attr_key_t::SHARED_CONST_BASES);
    if (bases) {
        out_base_tsr.reserve(bases->size());
        for (auto &base : *bases) {
            out_base_tsr.emplace_back(
                    builder::make_tensor("__shared_const_base_"
                                    + std::to_string(collected_base.size()),
                            {base->size_}, datatypes::u8));
            out_base_tsr.back()->attr()[attr_keys::shared_const_base_idx]
                    = static_cast<size_t>(collected_base.size());
            // collect the list of the buffer handle used. If the buffer is not
            // lazy, then it will be never evicted. We can directly use the
            // buffer pointer instead of the handle
            void *handle = (base->is_lazy_) ? base.get()
                                            : base->get_buffer_if_not_lazy();
            collected_base.emplace_back(handle);
        }
    }
    for (auto &s : seq) {
        // if the stmt is a define node with cached_const_graph_tensor
        if (auto const_graph_tsr
                = s.cast<define_c>()
                          .flat_map([](const define_c &v) {
                              return v->var_.cast<tensor>();
                          })
                          .map([](const tensor &v) {
                              return any_map_t::fetch_or_else<std::shared_ptr<
                                      cached_const_graph_tensor>>(
                                      v->attr_.get(), attr_keys::shared_const,
                                      nullptr);
                          })
                          .get_or_else(nullptr)) {
            out_graph_tsr.emplace_back(const_graph_tsr);
            const auto &base = const_graph_tsr->buf_base_;

            void *handle = (base->is_lazy_) ? base.get()
                                            : base->get_buffer_if_not_lazy();
            COMPILE_ASSERT(
                    bases, "Expecting SHARED_CONST_BASES attr in the module");
            auto itr = std::find(bases->begin(), bases->end(), base);
            COMPILE_ASSERT(itr != bases->end(),
                    "Cannot find the shared const's base buffer in "
                    "SHARED_CONST_BASES of the module");
            auto idx = static_cast<size_t>(itr - bases->begin());
            s->attr()["shared_const_handle_idx"] = idx;
            if (!out_base_tsr[idx]->attr().has_key(attr_keys::shared_const)) {
                out_base_tsr[idx]->attr()[attr_keys::shared_const]
                        = const_graph_tsr;
            }
        }
    }
    if (!collected_base.empty()) {
        auto tsr = m->make_global_tensor(datatypes::index,
                shared_const_handle_name, {uint64_t(collected_base.size())});
    }
    return collected_base;
}

const_ir_module_ptr module_globals_resolver_t::operator()(
        const_ir_module_ptr m) {
    auto ret = std::make_shared<ir_module_t>(*m);
    std::vector<std::shared_ptr<cached_const_graph_tensor>> out_graph_tsr;
    std::vector<expr> shared_const_base_tsr;
    auto base_cache_tsr = process_shared_const_tensors(
            ret, out_graph_tsr, shared_const_base_tsr);
    // first, plan the memory layout for the static buffer to be used in an
    // IR module:
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
    sym_table->shared_tensors_ = std::move(out_graph_tsr);
    for (auto &def : ret->get_module_vars()) {
        auto &anyoffset
                = def->var_->attr().get_any(attr_keys::module_global_offset);
        if (def->var_.isa<var>()) {
            sym_table->add(
                    def->var_.static_as<var>()->name_, anyoffset.get<size_t>());
        } else {
            auto tsr = def->var_.checked_as<tensor>();
            if (auto offset = anyoffset.get_or_null<size_t>()) {
                sym_table->add(tsr->name_, *offset);
                if (tsr->init_value_) {
                    memcpy(reinterpret_cast<char *>(sym_table->data_.data_)
                                    + *offset,
                            tsr->init_value_->data_, tsr->init_value_->size_);
                }
            }
        }
    }
    // fill the shared_const_handles
    if (!base_cache_tsr.empty()) {
        if (auto handles = sym_table->get_or_null(shared_const_handle_name)) {
            memcpy(handles, base_cache_tsr.data(),
                    sizeof(base_cache_tsr[0]) * base_cache_tsr.size());
        } else {
            throw std::runtime_error(
                    "Cannot find the shared_const_handle in module data");
        }
    }
    uintptr_t baseptr = (uintptr_t)sym_table->data_.data_;
    bool is_static_global = m->attr_.get_or_else(
            ir_module_t::attr_key_t::STATIC_GLOBALS, false);
    ret->attr_[ir_module_t::attr_key_t::MODULE_DATA_BUFFERS]
            = std::move(sym_table);
    auto &funcs = ret->get_contents();
    // second, pass the base pointer of the global buffer from "main entry"
    // func to all IR functions. We also need to replace the call nodes and
    // append a new parameter to pass down the global buffer pointer
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
        if ((int)i == ret->get_entry_func_idx()) {
            impl.shared_const_base_tsr_ = std::move(shared_const_base_tsr);
            // if it is main func, make sure shared_const_handle is used
            auto handle_tsr
                    = ret->get_var_def_from_symbol(shared_const_handle_name);
            if (handle_tsr.defined()) {
                impl.get_or_create_new_var(handle_tsr->var_);
            }
        }
        impl.absolute_base_ = is_static_global ? baseptr : 0;
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
