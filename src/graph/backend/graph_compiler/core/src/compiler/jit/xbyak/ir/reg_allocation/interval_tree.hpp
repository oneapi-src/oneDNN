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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_JIT_XBYAK_IR_REG_ALLOCATION_INTERVAL_TREE_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_JIT_XBYAK_IR_REG_ALLOCATION_INTERVAL_TREE_HPP

#include <algorithm>
#include <functional>
#include <set>
#include <vector>

#include "virtual_reg.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace xbyak {

/* *
 * Non-overlapping balanced interval tree implemetation based on std::set.
 * */
class interval_tree_t {
public:
    // constructor
    interval_tree_t() = default;
    // destructor
    virtual ~interval_tree_t() = default;

    bool empty() { return node_map_.empty(); }

    // insert new interval
    void insert(stmt_index_t start, stmt_index_t end, virtual_reg_t *virt_reg) {
        node_map_.insert(node_t(start, end, virt_reg));
    }

    // remove existing interval
    void remove(stmt_index_t start, stmt_index_t end, virtual_reg_t *virt_reg) {
        erase_nodes(start, end, virt_reg, [](node_t node) {});
    }

    // divide existing interval using cut range
    void divide(stmt_index_t start, stmt_index_t end, virtual_reg_t *virt_reg) {
        std::vector<node_t> erased_nodes;
        auto node_func = [&](node_t node) { erased_nodes.push_back(node); };
        erase_nodes(start, end, virt_reg, node_func);
        for (auto &node : erased_nodes) {
            // front
            stmt_index_t start_front = node.start_;
            stmt_index_t end_front = std::min(node.end_, start);
            if (start_front < end_front) {
                insert(start_front, end_front, virt_reg);
            }
            // back
            stmt_index_t start_back = std::max(node.start_, end);
            stmt_index_t end_back = node.end_;
            if (start_back < end_back) {
                insert(start_back, end_back, virt_reg);
            }
        }
    }

    // search interval for overlap
    bool search(stmt_index_t start, stmt_index_t end) {
        auto iter = node_map_.lower_bound(node_t(start, end, nullptr));
        if (iter != node_map_.begin()) { iter--; }
        while (iter != node_map_.end()) {
            auto &node = *iter;
            if (end <= node.start_) { break; }
            if (node.intersects(start, end)) { return true; }
            iter++;
        }
        return false;
    }

    // query interval for overlap
    void query(stmt_index_t start, stmt_index_t end,
            std::function<void(virtual_reg_t *)> func) {
        auto iter = node_map_.lower_bound(node_t(start, end, nullptr));
        if (iter != node_map_.begin()) { iter--; }
        while (iter != node_map_.end()) {
            auto &node = *iter;
            if (end <= node.start_) { break; }
            if (node.intersects(start, end)) { func(node.virtual_reg_); }
            iter++;
        }
    }

private:
    // Internal node
    struct node_t {
        stmt_index_t start_;
        stmt_index_t end_;
        virtual_reg_t *virtual_reg_;

        bool intersects(stmt_index_t start, stmt_index_t end) const {
            return std::max(start_, start) < std::min(end_, end);
        }

        bool operator<(const node_t &b) const { return start_ < b.start_; }

        node_t(stmt_index_t start, stmt_index_t end, virtual_reg_t *virt_reg)
            : start_(start), end_(end), virtual_reg_(virt_reg) {
            // must contain valid range
            assert(start < end);
        }
    };
    // Internal RB-tree
    std::set<node_t> node_map_;

    // erase interval node
    void erase_nodes(stmt_index_t start, stmt_index_t end,
            virtual_reg_t *virt_reg, std::function<void(node_t)> func) {
        auto iter = node_map_.lower_bound(node_t(start, end, nullptr));
        if (iter != node_map_.begin()) { iter--; }
        while (iter != node_map_.end()) {
            auto &node = *iter;
            if (end <= node.start_) { break; }
            if (node.virtual_reg_ == virt_reg) {
                func(node);
                iter = node_map_.erase(iter);
            } else {
                iter++;
            }
        }
    }
};

} // namespace xbyak
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
