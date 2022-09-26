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
#include "pointer_alias_info.hpp"
#include <compiler/ir/sc_expr.hpp>
#include <unordered_set>
#include <util/any_map.hpp>

namespace sc {

namespace alias_info {

const std::shared_ptr<alias_set_t> &tensor_alias_identity_t::get_alias_set() {
    if (alias_set_) { return alias_set_; }
    alias_set_ = std::make_shared<alias_set_t>();
    alias_set_->set_.insert(shared_from_this());
    return alias_set_;
}

bool tensor_alias_identity_t::has_no_alias() const {
    assert(alias_set_->set_.size() >= 1UL);
    return alias_set_->set_.size() == 1;
}

void tensor_alias_identity_t::add_alias(
        const std::shared_ptr<tensor_alias_identity_t> &v) {
    // hold a reference of the alias shared ptr
    auto other_set = v->get_alias_set();
    get_alias_set();
    if (other_set == this->alias_set_) { return; }
    for (auto ptr : other_set->set_) {
        auto identity = ptr.lock();
        COMPILE_ASSERT(
                identity, "shared ptr of tensor_alias_identity_t expired.");
        // merge the alias set
        this->alias_set_->set_.insert(identity);
        // reset the alias set it belongs to
        identity->alias_set_ = this->alias_set_;
    }
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

} // namespace alias_info

} // namespace sc
