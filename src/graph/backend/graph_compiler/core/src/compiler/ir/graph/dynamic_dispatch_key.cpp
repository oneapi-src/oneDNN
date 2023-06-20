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

#include <algorithm>
#include <set>
#include <vector>
#include "dynamic_dispatch_key.hpp"
#include "dynamic_internal_info.hpp"
#include "dynamic_lower_info.hpp"
#include "dynamic_utils.hpp"
#include <compiler/ir/graph/fused_op.hpp>
#include <compiler/ir/graph/tunable_op.hpp>
#include <compiler/ir/sc_data_format.hpp>
#include <ops/fusible/memory_movement.hpp>
#include <runtime/dynamic_dispatch/ops/impl_type.hpp>
#include <runtime/logging.hpp>
#include <util/assert.hpp>
namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
#define DISPATCH_KEY_MAX_THRESHOLD UINT64_C(64)
SC_MODULE(graph.dynamic_dispatch_key)

bool op_dispatch_key_t::operator==(const op_dispatch_key_t &other) const {
    return var_block_ == other.var_block_
            && in_out_formats_ == other.in_out_formats_ && impl_ == other.impl_;
}

bool op_dispatch_key_t::operator!=(const op_dispatch_key_t &other) const {
    return !(*this == other);
}

void op_dispatch_key_t::set_op_dispatch_key(
        const sc_op_ptr &node, const context_ptr &ctx) const {
    if (auto tunable_node = node->dyn_cast<tunable_op_t>()) {
        tunable_node->set_config_by_key(*this, ctx);
    }
    auto &inputs = node->get_inputs();
    auto &outputs = node->get_outputs();
    int idx = 0;
    for (auto &in : inputs) {
        in->details_.set_format(in_out_formats_[idx++]);
    }
    for (auto &out : outputs) {
        out->details_.set_format(in_out_formats_[idx++]);
    }
    node->info_.cur_impl_ = impl_;
}

std::vector<runtime::dispatch_key>
op_dispatch_key_t::convert_to_runtime_format_vec() const {
    std::vector<runtime::dispatch_key> outs(in_out_formats_.size());
    bool var_block_empty = var_block_.empty();
    assert(var_block_empty || var_block_.size() == in_out_formats_.size());
    for (size_t i = 0; i < in_out_formats_.size(); i++) {
        sc_dim block0 = 0, block1 = 0;
        if (!var_block_empty && !var_block_[i].empty()) {
            block0 = var_block_[i][0];
            block1 = var_block_[i][1];
        } else {
            block0 = in_out_formats_[i].blocks_[0];
            block1 = in_out_formats_[i].blocks_[1];
        }
        outs[i] = runtime::dispatch_key(
                static_cast<uint64_t>(in_out_formats_[i].format_code_), block0,
                block1, impl_, in_out_formats_[i].is_plain());
    }
    return outs;
}

void impl_op_dispatch_key_t::set_op_dispatch_key(
        const sc_op_ptr &node, const context_ptr &ctx) const {
    if (auto tun_node = node->dyn_cast<tunable_op_t>()) {
        tun_node->set_internal_config_by_key(*this, ctx);
    } else {
        throw std::runtime_error("Unimplement");
    }
}

std::vector<runtime::dispatch_key>
impl_op_dispatch_key_t::convert_to_runtime_format_vec() const {
    std::vector<runtime::dispatch_key> outs(
            repeat_, runtime::dispatch_key(UINT64_C(0), 0, 0, impl_, false));
    return outs;
}

std::vector<runtime::dispatch_key>
combined_op_dispatch_key_t::convert_to_runtime_format_vec() const {
    std::vector<runtime::dispatch_key> ret;
    for (auto &key : keys_) {
        auto v = key.convert_to_runtime_format_vec();
        ret.insert(ret.end(), v.begin(), v.end());
    }
    return ret;
}

bool combined_op_dispatch_key_t::operator==(
        const combined_op_dispatch_key_t &other) const {
    return keys_ == other.keys_;
}

bool combined_op_dispatch_key_t::operator!=(
        const combined_op_dispatch_key_t &other) const {
    return !(*this == other);
}

void combined_op_dispatch_key_t::set_op_dispatch_key(
        const sc_op_ptr &node, const context_ptr &ctx) const {
    assert(node->isa<fused_op_t>() || node->isa<mixed_fuse_op_t>());
    if (node->isa<fused_op_t>()) {
        node->stc_cast<fused_op_t>()->update_internal_graph_format(*this, ctx);
    } else {
        node->stc_cast<mixed_fuse_op_t>()->update_internal_graph_format(
                *this, ctx);
    }
}

bool dispatch_key_cmper_t::operator()(
        const op_dispatch_key_t &key0, const op_dispatch_key_t &key1) const {
    if (key0.impl_ != key1.impl_) { return key0.impl_ < key1.impl_; }
    assert(key0.in_out_formats_.size() == key1.in_out_formats_.size());
    for (size_t i = 0; i < key0.in_out_formats_.size(); i++) {
        if (key0.in_out_formats_[i].format_code_
                != key1.in_out_formats_[i].format_code_) {
            return key0.in_out_formats_[i].format_code_
                    < key1.in_out_formats_[i].format_code_;
        }
        if (key0.in_out_formats_[i].blocks_
                != key1.in_out_formats_[i].blocks_) {
            return key0.in_out_formats_[i].blocks_
                    < key1.in_out_formats_[i].blocks_;
        }
    }
    assert(key0.var_block_.size() == key1.var_block_.size());
    for (size_t i = 0; i < key0.var_block_.size(); i++) {
        assert(key0.var_block_[i].size() == key1.var_block_[i].size());
        for (size_t j = 0; j < key0.var_block_[i].size(); j++) {
            if (key0.var_block_[i][j] != key1.var_block_[i][j]) {
                return key0.var_block_[i][j] < key1.var_block_[i][j];
            }
        }
    }
    // equal
    return false;
}

bool impl_dispatch_key_cmper_t::operator()(const impl_op_dispatch_key_t &key0,
        const impl_op_dispatch_key_t &key1) const {
    if (key0.impl_ != key1.impl_) { return key0.impl_ < key1.impl_; }
    // equal
    return false;
}

combined_dispatch_key_set_t::combined_dispatch_key_set_t(
        const std::vector<sc_op_ptr> &inputs, const sc_op_ptr &modified_inp) {
    if (!inputs.empty()) {
        auto dispatch_sets = get_dispatch_set_vec_from_ops(inputs);
        internal_construct(dispatch_sets, inputs, modified_inp);
    }
}

combined_dispatch_key_set_t::combined_dispatch_key_set_t(
        const std::vector<dispatch_set_ptr> &dispatch_sets) {
    if (!dispatch_sets.empty()) { internal_construct(dispatch_sets); }
}

void recursive_construct(combined_dispatch_key_set_t::inner_set_t &set,
        combined_op_dispatch_key_t &cur_combined_key,
        op_layout_link_vec_t &op_link_relations,
        const std::vector<dispatch_set_ptr> &dispatch_sets,
        const std::vector<sc_op_ptr> &inputs, size_t cur_op_idx, size_t len_key,
        int linked_reorder_impl) {
    if (cur_op_idx == len_key) {
        if (cur_combined_key.size() == len_key) {
            set.insert(cur_combined_key);
        }
        return;
    }
    auto &inner_set = dispatch_sets[cur_op_idx]->get_inner_set();
    for (auto it = inner_set.begin(); it != inner_set.end(); it++) {
        bool is_valid = true;
        auto cur_key = *it;
        if (!op_link_relations.empty()) {
            for (size_t j = 0; j < op_link_relations[cur_op_idx].size(); j++) {
                auto &link_pair = op_link_relations[cur_op_idx][j];
                if (link_pair.first != no_link_idx
                        && !is_linked_layout(it->in_out_formats_[j],
                                cur_combined_key[link_pair.first]
                                        .in_out_formats_[link_pair.second])) {
                    is_valid = false;
                    break;
                }
            }
        }
        // In order to reduce the number of combined dispatch key, all reorder
        // inside subgraph follows same impl algorithm(normal/no_padding).
        // no_padding is available when all reorders has no padding at runtime.
        if (!inputs.empty() && inputs[cur_op_idx]->isa<reorder_op_t>()) {
            if (it->impl_ != linked_reorder_impl) {
                // static reorder in dynamic pattern
                if (!can_op_be_dispatched(inputs[cur_op_idx])) {
                    cur_key.impl_ = linked_reorder_impl;
                } else {
                    is_valid = false;
                }
            }
        }
        if (!is_valid) { continue; }
        cur_combined_key.keys_.emplace_back(cur_key);
        recursive_construct(set, cur_combined_key, op_link_relations,
                dispatch_sets, inputs, cur_op_idx + 1, len_key,
                linked_reorder_impl);
        cur_combined_key.keys_.pop_back();
    }
}

static bool validate_dispatch_key_sets(
        const std::vector<dispatch_set_ptr> &dispatch_sets,
        const std::vector<sc_op_ptr> &inputs) {
    // todo: add tunable dispatch check
    // reorder impl kind check.
    int cur_impl_num = -1;
    for (size_t i = 0; i < inputs.size(); i++) {
        if (inputs[i]->isa<reorder_op_t>()) {
            if (dispatch_sets[i]->size() == 1) { continue; }
            auto get_impl_kind_num = [](const dispatch_set_ptr &dispatch_set) {
                auto &set = dispatch_set->get_inner_set();
                bool has_normal = false, has_padding = false;
                for (auto &it : set) {
                    if (it.impl_ == impl_kind_t::normal) {
                        has_normal = true;
                    } else if (it.impl_ == impl_kind_t::no_padding) {
                        has_padding = true;
                    } else {
                        throw std::runtime_error(
                                "Wrong impl kind for reorder.");
                    }
                    if (has_normal && has_padding) { return 2; }
                }
                if (has_normal || has_padding) { return 1; }
                return 0;
            };
            if (cur_impl_num == -1) {
                cur_impl_num = get_impl_kind_num(dispatch_sets[i]);
            } else if (cur_impl_num != get_impl_kind_num(dispatch_sets[i])) {
                return false;
            }
        }
    }
    return true;
}

void combined_dispatch_key_set_t::internal_construct(
        const std::vector<dispatch_set_ptr> &dispatch_sets,
        const std::vector<sc_op_ptr> &inputs, const sc_op_ptr &modified_inp) {
    op_layout_link_vec_t op_link_relations;
    auto len_key = dispatch_sets.size();
    auto num_keys = UINT64_C(1);
    for (size_t i = len_key; i > 0; i--) {
        num_keys = num_keys * dispatch_sets[i - 1]->size();
    }
    // no need to dispatch
    if (num_keys == 0) { return; }
    bool is_specific = false;
    if (!inputs.empty()) {
        op_link_relations = get_op_layout_link_relationships(
                inputs, dispatch_sets, modified_inp);
        is_specific = is_dyn_specific_graph(inputs[0]->get_owner_graph());
    }
    COMPILE_ASSERT(validate_dispatch_key_sets(dispatch_sets, inputs),
            "Wrong dispatch key sets, could not construct combined "
            "dispatch key.");
    combined_op_dispatch_key_t cur_combined_key;
    cur_combined_key.keys_.reserve(len_key);
    auto reorder_impl_candidates = get_default_impl_dispatch_candidates();
    for (auto &linked_reorder_impl : reorder_impl_candidates) {
        recursive_construct(set_, cur_combined_key, op_link_relations,
                dispatch_sets, inputs, 0, len_key, linked_reorder_impl);
    }
    COMPILE_ASSERT(!set_.empty(), "Empty linked combined dispatch key set!");
    if (!is_specific && set_.size() > DISPATCH_KEY_MAX_THRESHOLD) {
        SC_MODULE_WARN << "Number of dispatch key set " << set_.size()
                       << " has exceeded threshold "
                       << DISPATCH_KEY_MAX_THRESHOLD;
    }
}

bool combined_dispatch_key_cmper_t::operator()(
        const combined_op_dispatch_key_t &key0,
        const combined_op_dispatch_key_t &key1) const {
    assert(key0.size() == key1.size());
    dispatch_key_cmper_t cmper;
    for (size_t i = 0; i < key0.size(); i++) {
        if (key0[i] != key1[i]) { return cmper(key0[i], key1[i]); }
    }
    // equal
    return false;
}

void dispatch_key_set_t::for_each_key_process(
        const std::function<void(const op_dispatch_key_base_t *)> &callback) {
    for (auto &key : set_) {
        callback(&key);
    }
}

std::set<op_dispatch_key_t, dispatch_key_cmper_t> &
dispatch_key_set_t::get_inner_set() {
    return set_;
}

dispatch_set_ptr dispatch_key_set_t::copy() const {
    return std::make_shared<dispatch_key_set_t>(set_);
}

void impl_dispatch_key_set_t::for_each_key_process(
        const std::function<void(const op_dispatch_key_base_t *)> &callback) {
    for (auto &key : set_) {
        callback(&key);
    }
}

std::set<op_dispatch_key_t, dispatch_key_cmper_t> &
impl_dispatch_key_set_t::get_inner_set() {
    throw std::runtime_error(
            "Impl disaptch key set can not get its inner set.");
}

dispatch_set_ptr impl_dispatch_key_set_t::copy() const {
    return std::make_shared<impl_dispatch_key_set_t>(set_);
}

void combined_dispatch_key_set_t::for_each_key_process(
        const std::function<void(const op_dispatch_key_base_t *)> &callback) {
    for (auto &key : set_) {
        callback(&key);
    }
}

std::set<op_dispatch_key_t, dispatch_key_cmper_t> &
combined_dispatch_key_set_t::get_inner_set() {
    throw std::runtime_error(
            "Combined dispatch key set can not get its inner set");
}

dispatch_set_ptr combined_dispatch_key_set_t::copy() const {
    return std::make_shared<combined_dispatch_key_set_t>(set_);
}

std::vector<int> get_default_impl_dispatch_candidates() {
    static std::vector<int> default_impl_candidates
            = {impl_kind_t::normal, impl_kind_t::no_padding};
    return default_impl_candidates;
}

std::vector<int> get_dynamic_impl_dispatch_candidates(
        tunable_op_t *op, const context_ptr &ctx) {
    auto configs = op->get_dynamic_config_candidates(ctx);
    std::vector<int> ret;
    ret.reserve(configs.size());
    for (int i = 0; i < static_cast<int>(configs.size()); i++) {
        ret.push_back(i);
    }
    return ret;
}

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
