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

#include <algorithm>
#include <set>
#include <vector>
#include "dynamic_dispatch_key.hpp"
#include "dynamic_utils.hpp"
#include <compiler/ir/graph/fused_op.hpp>
#include <compiler/ir/graph/tunable_op.hpp>
#include <compiler/ir/sc_data_format.hpp>
#include <runtime/dynamic_dispatch/ops/impl_type.hpp>
#include <runtime/logging.hpp>
#include <util/assert.hpp>
namespace sc {
#define DISPATCH_KEY_MAX_THRESHOLD UINT64_C(64)
SC_MODULE(graph.dynamic_dispatch_key)

bool op_dispatch_key_t::operator==(const op_dispatch_key_t &other) const {
    return var_block_ == other.var_block_
            && in_out_formats_ == other.in_out_formats_ && impl_ == other.impl_;
}

bool op_dispatch_key_t::operator!=(const op_dispatch_key_t &other) const {
    return !(*this == other);
}

void op_dispatch_key_t::set_op_dispatch_key(const sc_op_ptr &node) const {
    if (auto tunable_node = node->dyn_cast<tunable_op_t>()) {
        tunable_node->set_config_by_key(*this);
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
        if (!var_block_empty) {
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

std::vector<runtime::dispatch_key>
combined_op_dispatch_key_t::convert_to_runtime_format_vec() const {
    std::vector<runtime::dispatch_key> ret;
    for (auto &key : *this) {
        auto v = key.convert_to_runtime_format_vec();
        ret.insert(ret.end(), v.begin(), v.end());
    }
    return ret;
}

bool combined_op_dispatch_key_t::operator==(
        const combined_op_dispatch_key_t &other) const {
    return static_cast<std::vector<op_dispatch_key_t>>(*this)
            == static_cast<std::vector<op_dispatch_key_t>>(other);
}

bool combined_op_dispatch_key_t::operator!=(
        const combined_op_dispatch_key_t &other) const {
    return !(*this == other);
}

void combined_op_dispatch_key_t::set_op_dispatch_key(
        const sc_op_ptr &node) const {
    assert(node->isa<fused_op_t>());
    node->stc_cast<fused_op_t>()->update_internal_graph_format(*this);
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

void combined_dispatch_key_set_t::internal_construct(
        const std::vector<dispatch_set_ptr> &dispatch_sets,
        const std::vector<sc_op_ptr> &inputs, const sc_op_ptr &modified_inp) {
    op_layout_link_vec_t op_link_relations;
    auto len_key = dispatch_sets.size();
    size_t num_keys = 1;
    std::vector<size_t> accum_size;
    accum_size.reserve(len_key);
    for (size_t i = len_key; i > 0; i--) {
        accum_size.emplace_back(num_keys);
        num_keys = num_keys * dispatch_sets[i - 1]->size();
    }
    // no need to dispatch
    if (num_keys == 0) { return; }
    if (!inputs.empty()) {
        op_link_relations = get_op_layout_link_relationships(
                inputs, dispatch_sets, modified_inp);
    }
    std::reverse(accum_size.begin(), accum_size.end());
    for (size_t cur_idx = 0; cur_idx < num_keys; cur_idx++) {
        combined_op_dispatch_key_t cur_combined_key;
        cur_combined_key.reserve(len_key);
        for (size_t i = 0; i < len_key; i++) {
            size_t offset = cur_idx;
            if (i) { offset = offset % accum_size[i - 1]; }
            offset = offset / accum_size[i];
            auto it = dispatch_sets[i]->get_inner_set().begin();
            std::advance(it, offset);
            bool is_valid = true;
            if (!op_link_relations.empty()) {
                for (size_t j = 0; j < op_link_relations[i].size(); j++) {
                    auto &link_pair = op_link_relations[i][j];
                    if (link_pair.first != no_link_idx
                            && !is_linked_layout(it->in_out_formats_[j],
                                    cur_combined_key[link_pair.first]
                                            .in_out_formats_
                                                    [link_pair.second])) {
                        is_valid = false;
                        break;
                    }
                }
            }
            if (!is_valid) { break; }
            cur_combined_key.emplace_back(*it);
        }
        if (cur_combined_key.size() == len_key) {
            set_.insert(cur_combined_key);
        }
    }
    COMPILE_ASSERT(!set_.empty(), "Empty linked combined dispatch key set!");
    if (set_.size() > DISPATCH_KEY_MAX_THRESHOLD) {
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

std::vector<int> get_default_impl_dispatch_candidates() {
    static std::vector<int> default_impl_candidates
            = {impl_kind_t::normal, impl_kind_t::no_padding};
    return default_impl_candidates;
}

} // namespace sc
