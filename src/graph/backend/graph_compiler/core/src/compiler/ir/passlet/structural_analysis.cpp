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

#include "structural_analysis.hpp"
#include <functional>
#include <utility>
#include <util/weakptr_utils.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace passlet {

void structural_analysis_t::view(const stmt_c &v, pass_phase phase) {
    if (phase == pass_phase::PRE_VISIT) {
        cur_parent_.emplace_back(v.get());
        return;
    }
    cur_parent_.pop_back();
    auto cur = get_result(v.get());
    cur->parent_ = cur_parent_.back() ? cur_parent_.back()->shared_from_this()
                                      : nullptr;
    cur->cur_node_ = v.impl;
}

const stmt_base_t *structural_result_t::get_raw_parent() const {
    return utils::get_raw_from_weakptr(parent_);
}

const stmt_base_t *structural_result_t::get_raw_cur_node() const {
    return utils::get_raw_from_weakptr(cur_node_);
}

stmt structural_result_t::get_parent_node() const {
    auto parent_raw = utils::get_raw_from_weakptr(parent_);
    return parent_raw
            ? stmt {const_cast<stmt_base_t *>(parent_raw)->shared_from_this()}
            : stmt();
}

void structural_result_t::reset_parent_node(const stmt_c &parent) {
    parent_ = parent.impl;
}

bool structural_result_t::is_parent_of(const structural_result_t &other,
        const typed_addresser_t &addresser, bool allow_across_for,
        bool allow_across_if,
        const structural_result_t **out_second_level_parent) const {
    const stmt_base_t *cur = other.get_raw_cur_node();
    const structural_result_t *cur_info = &other;
    if (out_second_level_parent) { *out_second_level_parent = nullptr; }
    for (;;) {
        if (cur_info == this) { return true; }
        if (out_second_level_parent) { *out_second_level_parent = cur_info; }
        if (!allow_across_for) {
            if (cur->node_type_ == sc_stmt_type::for_loop) { return false; }
        }
        if (!allow_across_if) {
            if (cur->node_type_ == sc_stmt_type::if_else) { return false; }
        }
        cur = cur_info->get_raw_parent();
        if (!cur) { return false; }
        cur_info = addresser(nullptr, cur);
    }
}

const stmt_base_t *structural_result_t::find_shared_parent(
        const structural_result_t &other, const typed_addresser_t &addresser,
        bool allow_across_for, bool allow_across_if) const {
    const stmt_base_t *cur = get_raw_cur_node();
    const structural_result_t *cur_info = this;
    for (;;) {
        if (cur_info->is_parent_of(
                    other, addresser, allow_across_for, allow_across_if)) {
            return cur;
        }
        if (!allow_across_for) {
            if (cur->node_type_ == sc_stmt_type::for_loop) { return nullptr; }
        }
        if (!allow_across_if) {
            if (cur->node_type_ == sc_stmt_type::if_else) { return nullptr; }
        }
        cur = cur_info->get_raw_parent();
        if (!cur) { return nullptr; }
        cur_info = addresser(nullptr, cur);
    }
}

} // namespace passlet
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
