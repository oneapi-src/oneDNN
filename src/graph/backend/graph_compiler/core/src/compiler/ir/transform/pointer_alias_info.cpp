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
#include "pointer_alias_info.hpp"
#include <compiler/ir/sc_expr.hpp>
#include <unordered_set>
#include <util/any_map.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

namespace alias_info {

bool tensor_alias_identity_t::has_no_alias() const {
    if (alias_cliques_.empty()) return true;
    for (auto &v : alias_cliques_) {
        assert(v->set_.size() >= 1UL);
        if (v->set_.size() > 1) { return false; }
    }
    return true;
}

bool tensor_alias_identity_t::is_alias_of(
        tensor_alias_identity_t *other) const {
    for (auto &v : alias_cliques_) {
        if (v->set_.has(other->shared_from_this())) { return true; }
    }
    return false;
}

void tensor_alias_identity_t::add_to_clique(
        const std::shared_ptr<alias_set_t> &v) {
    v->set_.insert(shared_from_this());
    alias_cliques_.emplace_back(v);
}

tensor_alias_identity_t *get_alias_info(const expr_base &v) {
    if (!v.attr_) { return nullptr; }
    auto ret = v.attr_->get_or_null<std::shared_ptr<tensor_alias_identity_t>>(
            attr_keys::pointer_alias);
    if (ret) { return ret->get(); }
    return nullptr;
}

std::shared_ptr<tensor_alias_identity_t> get_or_create_alias_info(
        expr_base &v) {
    auto &attr = v.attr();
    if (!attr.has_key(attr_keys::pointer_alias)) {
        auto ret = std::make_shared<tensor_alias_identity_t>();
        attr[attr_keys::pointer_alias] = ret;
        return ret;
    }
    return attr.get<std::shared_ptr<tensor_alias_identity_t>>(
            attr_keys::pointer_alias);
}

std::shared_ptr<alias_set_t> alias_set_t::copy() const {
    auto cur_clique = std::make_shared<alias_info::alias_set_t>(*this);
    for (auto p : set_) {
        auto aid = p.lock();
        // if the tensor is removed, skip
        if (!aid) { continue; }
        aid->add_to_clique(cur_clique);
    }
    return cur_clique;
}

} // namespace alias_info

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
