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
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "buffer_schedule.hpp"
#include "pointer_alias_info.hpp"
#include "tensor_inplace.hpp"
#include <compiler/ir/pass_dep_util.hpp>
#include <compiler/ir/transform/tensor_inplace.hpp>
#include <compiler/ir/viewer.hpp>
#include <unordered_set>
#include <util/any_map.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

SC_DECL_PASS_INFO(tensor_inplace,
        SC_PASS_DEPENDS_ON(
                constant_folder, index_flattener, validator, auto_caster),
        SC_PASS_REQUIRE_STATE(), SC_PASS_REQUIRE_NOT_STATE(),
        SC_PASS_SET_STATE(), SC_PASS_UNSET_STATE());

class func_finder_t : public ir_viewer_t {
public:
    using ir_viewer_t::dispatch;
    std::unordered_set<std::shared_ptr<func_base>> funcs_;

    void view(call_c v) override {
        ir_viewer_t::view(v);
        auto funct = std::dynamic_pointer_cast<func_base>(v->func_);
        if (funct) { funcs_.insert(funct); }
    }
};

using inplace_hint_t
        = std::vector<std::pair<int, std::vector<tensor_inplace_info_t>>>;

static void filter_and_sync_inplace_hint(const func_t &f) {
    if (!f->attr_) { return; }
    auto hint_in_def = f->attr_->get_or_null<inplace_hint_t>(
            function_attrs::inplace_hint);
    if (!hint_in_def) { return; }
    auto &params = f->decl_->params_;
    // the alias id is the in-place result selected by
    // buffer_scheduler_t, try to use it to filter the
    // inplace_hint
    for (auto itrkv = hint_in_def->begin(); itrkv != hint_in_def->end();) {
        auto out_arg_idx = itrkv->first;
        auto &hints = itrkv->second;
        if (auto out_alias_id
                = alias_info::get_alias_info(*params.at(out_arg_idx))) {
            for (auto itr_in = hints.begin(); itr_in != hints.end();) {
                auto in_arg_idx = itr_in->used_arg_idx_;
                if (auto in_alias_id
                        = alias_info::get_alias_info(*params.at(in_arg_idx))) {
                    if (out_alias_id->is_alias_of(in_alias_id)) {
                        // if tensors are alias, the inplace happens. continue
                        ++itr_in;
                        continue;
                    }
                }
                itr_in = hints.erase(itr_in);
            }
            ++itrkv;
        } else {
            itrkv = hint_in_def->erase(itrkv);
        }
    }
    f->decl_->attr()[function_attrs::inplace_hint] = *hint_in_def;
}

const_ir_module_ptr tensor_inplace_t::operator()(const_ir_module_ptr f) {
    auto ret = std::make_shared<ir_module_t>(*f);
    auto main_entry = f->get_entry_func();
    for (auto &entry_f : ret->get_contents()) {
        // skip functions that 1) are not main_entry function and 2) are not top
        // level functions
        bool *top_level = nullptr;
        if (entry_f->attr_) {
            top_level = entry_f->attr_->get_or_null<bool>(
                    function_attrs::top_level);
        }
        bool should_run = true;
        // if the func is explicitly marked top-level/not-top-level, follow that
        // instruction
        if (top_level) {
            should_run = *top_level;
        } else {
            // else, if the func is main entry
            should_run = (entry_f == main_entry);
        }
        if (!should_run) { continue; }
        // find all calls to func decl, and sync the inplace_hint with decl and
        // definition
        func_finder_t finder;
        finder.dispatch(entry_f);
        for (auto &funct : finder.funcs_) {
            if (!funct->body_.defined()) {
                // if the func is decl
                auto func_def = f->get_func(funct->name_);
                if (func_def && func_def->attr_) {
                    if (auto hint
                            = func_def->attr_->get_or_null<inplace_hint_t>(
                                    function_attrs::inplace_hint)) {
                        funct->attr()[function_attrs::inplace_hint] = *hint;
                    }
                }
            }
        }
        buffer_scheduler_t scheduler {ctx_, true, true};
        auto new_func = scheduler(entry_f);
        // if no changes, continue
        if (new_func == entry_f) { continue; }
        // sync back alias group info to callee functions
        for (auto &funct : finder.funcs_) {
            if (!funct->body_.defined()) {
                // if the callee func is decl
                auto func_def = f->get_func(funct->name_);
                if (func_def) {
                    filter_and_sync_inplace_hint(func_def);
                    for (size_t arg_id = 0; arg_id < funct->params_.size();
                            arg_id++) {
                        auto &arg_in_decl = funct->params_[arg_id];
                        auto &arg_in_def = func_def->params_.at(arg_id);
                        if (auto alias_id
                                = alias_info::get_alias_info(*arg_in_decl)) {
                            // sync the alias info
                            arg_in_def->attr()[attr_keys::pointer_alias]
                                    = arg_in_decl->attr_->get_any(
                                            attr_keys::pointer_alias);
                        }
                    }
                }
            }
        }
        // drop the new_func instead of replacing it with entry_f. Because
        // parallel_merge pass needs to run before buffer scheduling
    }

    return ret;
}

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
