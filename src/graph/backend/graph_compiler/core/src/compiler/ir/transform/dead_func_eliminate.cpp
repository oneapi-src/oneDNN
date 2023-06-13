/*******************************************************************************
 * Copyright 2023 Intel Corporation
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
#include "dead_func_eliminate.hpp"
#include <memory>
#include <string>
#include <vector>
#include <compiler/ir/pass_dep_util.hpp>
#include <compiler/ir/pass_id.hpp>
#include <compiler/ir/viewer.hpp>
#include <unordered_set>

SC_MODULE(pass.dead_func_elim);

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
SC_DECL_PASS_INFO(dead_func_eliminate, SC_PASS_DEPENDS_ON(),
        SC_PASS_REQUIRE_STATE(), SC_PASS_REQUIRE_NOT_STATE(),
        SC_PASS_SET_STATE(), SC_PASS_UNSET_STATE());

class func_reference_finder_t : public ir_viewer_t {
public:
    using ir_viewer_t::dispatch;
    // use pointer to workaround a gcc4 bug
    const ir_module_t *mod_;
    std::unordered_set<std::shared_ptr<func_base>> *funcs_;
    std::vector<std::shared_ptr<func_base>> *new_found_;

    func_reference_finder_t(const ir_module_t &mod,
            std::unordered_set<std::shared_ptr<func_base>> &funcs,
            std::vector<std::shared_ptr<func_base>> &new_found)
        : mod_ {&mod}, funcs_(&funcs), new_found_(&new_found) {}

    void view(call_c v) override {
        ir_viewer_t::view(v);
        auto funct = std::dynamic_pointer_cast<func_base>(v->func_);
        if (funct) {
            auto funcdef = mod_->get_func(funct->name_);
            if (funcdef) {
                bool is_inserted = funcs_->insert(funcdef).second;
                if (is_inserted) { new_found_->emplace_back(funcdef); }
            }
        }
    }

    void view(func_addr_c v) override {
        ir_viewer_t::view(v);
        auto funcdef = mod_->get_func(v->func_->name_);
        if (funcdef) {
            bool is_inserted = funcs_->insert(funcdef).second;
            if (is_inserted) { new_found_->emplace_back(funcdef); }
        }
    }
};

const_ir_module_ptr dead_func_eliminate_t::operator()(const_ir_module_ptr f) {
    std::vector<std::shared_ptr<func_base>> gc_roots;
    for (auto &func : f->get_contents()) {
        if (!func->attr_
                || !func->attr_->get_or_else(function_attrs::private_, false)) {
            gc_roots.emplace_back(func);
        }
    }
    std::unordered_set<std::shared_ptr<func_base>> met {
            gc_roots.begin(), gc_roots.end()};
    if (auto mainf = f->get_entry_func()) {
        if (met.count(mainf) == 0) {
            gc_roots.emplace_back(mainf);
            met.insert(mainf);
        }
    }
    while (!gc_roots.empty()) {
        std::vector<std::shared_ptr<func_base>> new_found;
        func_reference_finder_t finder {*f, met, new_found};
        for (auto &func : gc_roots) {
            finder.dispatch(func);
        }
        gc_roots = std::move(new_found);
    }
    if (met.size() == f->get_contents().size()) { return f; }
    std::vector<bool> mask;
    mask.reserve(f->get_contents().size());
    for (auto &func : f->get_contents()) {
        auto keep = met.count(func) != 0
                || (func->attr_
                        && func->attr_->get_or_else(
                                attr_keys::keep_func, false));
        mask.push_back(keep);
        if (!keep) { SC_MODULE_INFO << "removing func " << func->name_; }
    }
    return f->copy_and_remove_funcs(mask);
}

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
