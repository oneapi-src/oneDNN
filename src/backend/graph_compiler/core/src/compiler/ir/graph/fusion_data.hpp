/*******************************************************************************
 * Copyright 2020-2022 Intel Corporation
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

#ifndef BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_GRAPH_FUSION_DATA_HPP
#define BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_GRAPH_FUSION_DATA_HPP

#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "fusible_op.hpp"
#include "graph_map.hpp"
#include "tensor_slice.hpp"
#include <compiler/ir/builder.hpp>
#include <compiler/ir/sc_expr.hpp>
#include <unordered_map>
#include <unordered_set>

namespace sc {

using slice_range_list = std::vector<slice_range>;
using slice_range_map = std::unordered_map<int, slice_range_list>;

enum class infer_status_code : int {
    OK = 0, // Successful
    RETRY, // Need retry another anchor
    FAIL, // Could not infer
    UNKNOWN, // Unknown
    END,
};

expr do_cast_and_fold(const expr &);
struct infer_status_map_t {
private:
    context_ptr ctx_;
    bool recursive_mode_;
    std::vector<std::unordered_set<sc_op_ptr>> inf_stat_map_;

public:
    infer_status_map_t(context_ptr ctx = nullptr, bool recursive_mode = true)
        : ctx_(ctx), recursive_mode_(recursive_mode) {
        inf_stat_map_.resize(static_cast<int>(infer_status_code::END));
    }

    std::unordered_set<sc_op_ptr> &get_ops_by_status(infer_status_code code) {
        COMPILE_ASSERT(code != infer_status_code::END, "END code found");
        return inf_stat_map_[static_cast<int>(code)];
    }

    const bool is_ok() {
        return get_ops_by_status(infer_status_code::RETRY).empty()
                && get_ops_by_status(infer_status_code::FAIL).empty()
                && get_ops_by_status(infer_status_code::UNKNOWN).empty();
    }

    const bool is_fail() {
        return !get_ops_by_status(infer_status_code::FAIL).empty();
    }

    const bool is_retry() {
        return !get_ops_by_status(infer_status_code::RETRY).empty();
    }

    const bool is_unknown() {
        return !get_ops_by_status(infer_status_code::UNKNOWN).empty();
    }

    void append_ops_by_status(sc_op *cur, infer_status_code code) {
        COMPILE_ASSERT(code != infer_status_code::END, "END code found");
        inf_stat_map_[static_cast<int>(code)].insert(cur->shared_from_this());
    }

    void remove_ops_by_status(sc_op *cur, infer_status_code code) {
        COMPILE_ASSERT(code == infer_status_code::UNKNOWN,
                "remove_ops_by_status temporarily only supports remove "
                "unknown_status to avoid potential misuse.");
        auto pos = inf_stat_map_[static_cast<int>(code)].find(
                cur->shared_from_this());
        COMPILE_ASSERT(pos != inf_stat_map_[static_cast<int>(code)].end(),
                "Op not found in unknown_status map.");
        inf_stat_map_[static_cast<int>(code)].erase(pos);
    }

    void clear() {
        for (auto &ops : inf_stat_map_)
            ops.clear();
    }

    const bool is_recursive_mode() const { return recursive_mode_; }

    const context_ptr get_context() const { return ctx_; }

    static std::vector<sc_op_ptr> stat_map_to_vector(
            const std::unordered_set<sc_op_ptr> &stat_map) {
        std::vector<sc_op_ptr> result;
        result.reserve(stat_map.size());
        for (auto itr : stat_map) {
            result.push_back(itr);
        }
        return result;
    }
};

inline std::vector<expr> get_slice_idx(const slice_range &range) {
    std::vector<expr> ret;
    for (auto &r : range) {
        ret.emplace_back(do_cast_and_fold(r.first));
    }
    return ret;
}

inline std::vector<expr> get_slice_shape(const slice_range &range) {
    std::vector<expr> ret;
    for (auto &r : range) {
        ret.emplace_back(do_cast_and_fold(r.second));
    }
    return ret;
}

inline slice_range gen_slice_by_dims(const sc_dims &dims) {
    slice_range ret;
    for (auto &r : dims) {
        ret.emplace_back(std::make_pair(expr(0), dim2unsigned(r)));
    }
    return ret;
}

bool is_reshaped_tensor(const expr &tsr);

expr transform_tsr2stsr_with_range(const expr &tsr, const slice_range &range);

expr transform_tsl2stsr(const tensor_slice &tsl);

expr transform_tsr2tptr_with_range(const expr &tsr, const slice_range &range);

expr transform_tptr2stsr(const expr &tptr);

struct buffer_reuse_identity;
using tsr_reuse_vec = std::vector<expr>;
using buffer_identity_count
        = std::unordered_map<buffer_reuse_identity, tsr_reuse_vec>;

// fusion_data_t is related to buffer and slice range info of fusible op
struct fusion_data_t {
    // the number of uses of the tensor in the graph. TODO: Should be replaced
    // by uses_.size()
    int use_count_ = 0;
    bool need_alloc_ = true;
    fusion_data_t() = default;
    fusion_data_t(const fusion_data_t &) = delete;

    fusion_data_t(fusion_data_t &&moved)
        : use_count_(moved.use_count_)
        , need_alloc_(moved.need_alloc_)
        , buffer_(moved.buffer_) {}
    bool buffer_allocated() const { return buffer_.defined(); }
    void set_buffer(bool is_dynamic, const expr &buf);
    const expr &get_buffer() const { return buffer_; };
    tensor get_real_tensor() const;

private:
    expr buffer_; /*tensor or tensorptr*/
};

struct fuse_state_t {
    fdata_map fdmap_;
    std::vector<fslice_map> fsmap_list_;
    fuse_state_t() = default;
};

struct fuse_anchor_t {
    stmts anchor_position_;
    std::pair<std::vector<tensor_slice>, std::vector<tensor_slice>>
            anchor_slice_;
    fuse_anchor_t() = default;
    fuse_anchor_t(stmts pos,
            std::pair<std::vector<tensor_slice>, std::vector<tensor_slice>>
                    slice)
        : anchor_position_(std::move(pos)), anchor_slice_(std::move(slice)) {};
};

struct iter_fuse_anchor_t {
    stmts anchor_position_;
    expr iter_;
    expr tsr_;
    slice_range_list slice_list_;
    iter_fuse_anchor_t() = default;
    iter_fuse_anchor_t(
            stmts pos, expr iter, expr tsr, slice_range_list slice_list)
        : anchor_position_(pos)
        , iter_(iter)
        , tsr_(tsr)
        , slice_list_(slice_list) {}
    bool defined() const { return anchor_position_.defined(); }
};

struct fuse_anchor_map_t {
    stmts anchor_position_;
    fslice_map fsmap_;

    std::shared_ptr<fuse_anchor_map_t> parent_;
    std::unordered_set<graph_tensor_ptr> blocked_gt_set_;
    std::unordered_map<graph_tensor_ptr, std::shared_ptr<fuse_anchor_map_t>>
            borrowed_fanchor_map_;
    // content includes op and anchor
    std::unordered_map<void *, size_t> content_number_map_;

    fuse_anchor_map_t() = default;
    fuse_anchor_map_t(stmts pos, fslice_map fsmap,
            std::shared_ptr<fuse_anchor_map_t> parent = nullptr)
        : anchor_position_(std::move(pos))
        , fsmap_(std::move(fsmap))
        , parent_(parent) {
        if (parent) { parent_->append_anchor(this); }
    };
    bool defined() const { return anchor_position_.defined(); }
    void commit_stmts(stmts &ss) {
        anchor_position_->seq_.insert(
                anchor_position_->seq_.end(), ss->seq_.begin(), ss->seq_.end());
    }
    void commit_stmt(stmt &s) { anchor_position_->seq_.emplace_back(s); }
    void append_op(sc_op *op) {
        content_number_map_.insert(
                std::make_pair(op, content_number_map_.size()));
    }
    void append_anchor(fuse_anchor_map_t *fanchor) {
        content_number_map_.insert(
                std::make_pair(fanchor, content_number_map_.size()));
    }
    void attach_parent_anchor(const std::shared_ptr<fuse_anchor_map_t> &parent,
            const std::shared_ptr<fuse_anchor_map_t> &repl_parent) {
        if (!parent) return;
        auto root = this;
        while (root->parent_ && (root->parent_ != repl_parent)) {
            COMPILE_ASSERT(root != root->parent_.get(),
                    "Ring parent anchor relationship found");
            root = root->parent_.get();
        }
        if (root == parent.get()) return;
        root->parent_ = parent;
        parent->append_anchor(root);
    }

    bool is_parent_for(const fuse_anchor_map_t *cur) {
        if (!cur) return false;
        while (cur->parent_) {
            cur = cur->parent_.get();
            if (cur == this) return true;
        }
        return false;
    }

    void merge(const std::shared_ptr<fuse_anchor_map_t> &other) {
        fsmap_.datamap_.insert(
                other->fsmap_.datamap_.begin(), other->fsmap_.datamap_.end());
        blocked_gt_set_.insert(
                other->blocked_gt_set_.begin(), other->blocked_gt_set_.end());
        borrowed_fanchor_map_.insert(other->borrowed_fanchor_map_.begin(),
                other->borrowed_fanchor_map_.end());
        auto contents_size = content_number_map_.size();
        for (auto &cont_numb_pair : other->content_number_map_) {
            content_number_map_.insert(std::make_pair(cont_numb_pair.first,
                    cont_numb_pair.second + contents_size));
        }
    }
};

using fuse_anchor_map_ptr = std::shared_ptr<fuse_anchor_map_t>;

} // namespace sc

namespace std {
template <>
struct hash<sc::buffer_reuse_identity> {
    std::size_t operator()(const sc::buffer_reuse_identity &in) const;
};
} // namespace std
#endif
