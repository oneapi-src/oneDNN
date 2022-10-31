/*******************************************************************************
 * Copyright 2022 Intel Corporation
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

namespace sc {

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

const_ir_module_ptr tensor_inplace_t::operator()(const_ir_module_ptr f) {
    auto ret = std::make_shared<ir_module_t>(*f);
    auto main_entry = f->get_entry_func();
    for (auto &entry_f : ret->get_contents()) {
        // skip functions that 1) are not main_entry function and 2) are not top
        // level functions
        if (entry_f != main_entry
                && (!entry_f->attr_
                        || !entry_f->attr_->get_or_else(
                                function_attrs::top_level, false))) {
            continue;
        }
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
                            = func_def->attr_
                                      ->get_or_null<std::vector<std::pair<int,
                                              std::vector<
                                                      tensor_inplace_info_t>>>>(
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
                    for (size_t arg_id = 0; arg_id < funct->params_.size();
                            arg_id++) {
                        auto &arg_in_decl = funct->params_[arg_id];
                        auto &arg_in_def = func_def->params_.at(arg_id);
                        if (arg_in_decl->attr_
                                && arg_in_decl->attr_->has_key(
                                        attr_keys::pointer_alias)) {
                            arg_in_def->attr()[attr_keys::pointer_alias]
                                    = arg_in_decl->attr_->get_any(
                                            attr_keys::pointer_alias);
                        }
                    }
                }
            }
        }
        entry_f = std::const_pointer_cast<func_base>(new_func);
    }

    return ret;
}

} // namespace sc
