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
    stmt dispatch_helper_;
    iter_fuse_anchor_t() = default;
    iter_fuse_anchor_t(stmts pos, expr iter, expr tsr,
            slice_range_list slice_list, stmt dispatch_helper)
        : anchor_position_(pos)
        , iter_(iter)
        , tsr_(tsr)
        , slice_list_(slice_list)
        , dispatch_helper_(dispatch_helper) {}
    bool defined() const { return anchor_position_.defined(); }
};

struct fuse_anchor_map_t : std::enable_shared_from_this<fuse_anchor_map_t> {
    stmts anchor_position_;
    fslice_map fsmap_;
    // parent anchor
    std::shared_ptr<fuse_anchor_map_t> parent_;
    // blocked graph tensor set, the reason why not use empty gt for judgement
    // is to distinguish non-visited gt and visited-but-failed gt
    std::unordered_set<graph_tensor_ptr> blocked_gt_set_;
    // borrowed fanchor map
    std::unordered_map<graph_tensor_ptr, std::shared_ptr<fuse_anchor_map_t>>
            borrowed_fanchor_map_;
    // content inferred under current fusion anchor scope, includes op and
    // anchor
    std::unordered_map<void *, size_t> content_number_map_;

    fuse_anchor_map_t() = default;
    fuse_anchor_map_t(stmts pos, const fslice_map &fsmap,
            const std::shared_ptr<fuse_anchor_map_t> &parent = nullptr)
        : anchor_position_(std::move(pos))
        , fsmap_(std::move(fsmap))
        , parent_(parent) {
        if (parent) { parent_->append_anchor(this); }
    };
    bool defined() const { return anchor_position_.defined(); }

    // commit `stmt` to anchor and bind parent node to commited anchor
    virtual void commit_stmt(stmt &s) {
        add_parent_node(s, anchor_position_);
        anchor_position_->seq_.emplace_back(s);
    }

    // commit `stmts` to anchor and bind parent node to commited anchor
    virtual void commit_stmts(stmts &ss) {
        for (auto &s : ss->seq_) {
            commit_stmt(s);
        }
    }

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

    fuse_anchor_map_t *get_root() const {
        auto root = this;
        while (root->parent_) {
            COMPILE_ASSERT(root != root->parent_.get(),
                    "Ring parent anchor relationship found");
            root = root->parent_.get();
        }
        return const_cast<fuse_anchor_map_t *>(root);
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

    template <typename T>
    bool isa() const {
        static_assert(is_base_of_t<fuse_anchor_map_t, T>::value,
                "T is not a subclass of fuse_anchor_map.");
        return dynamic_cast<const T *>(this);
    }

    template <typename T>
    T *dyn_cast() {
        return dynamic_cast<T *>(this);
    }

    virtual ~fuse_anchor_map_t() = default;

    // This function will find the nearest parent 'for_loop' node for fusion
    // anchor
    stmt get_parent_loop() const {
        stmt node = anchor_position_;
        while (node->attr().has_key("builder.parent_node")) {
            if (node.isa<for_loop>()) { return node; }
            node = get_parent_node(node);
        }
        return node;
    }
    // This function will return parent body scope
    stmts get_parent_scope() const {
        auto loop = get_parent_loop();
        return loop.isa<for_loop>()
                ? loop.static_as<for_loop>()->body_.checked_as<stmts>()
                : loop.checked_as<stmts>();
    }

    /**
     * What is parent relationship between anchor A and B
     * { // anchor A
     *    for(){
     *      //
     *      { // anchor B
     *      }
     *    }
     * }
     * */
    bool is_parent_for(const fuse_anchor_map_t *cur) const {
        if (!cur) return false;
        while (cur->parent_) {
            cur = cur->parent_.get();
            if (cur == this) return true;
        }
        return false;
    }

    bool is_parent_for(const std::shared_ptr<fuse_anchor_map_t> &cur) const {
        return is_parent_for(cur.get());
    }

    /**
     * What is sibling relationship between anchor A and B
     * for(){
     *    for(){
     *      //
     *      { // anchor A
     *      }
     *    }
     *    { // anchor B
     *    }
     * }
     * */
    bool is_sibling_for(const fuse_anchor_map_t *other) const {
        if (is_parent_for(other)) return false;
        auto this_loop = get_parent_loop();
        auto other_loop = other->get_parent_loop();

        while (other_loop->attr().has_key("builder.parent_node")) {
            other_loop = get_parent_node(other_loop);
            if (this_loop.ptr_same(other_loop)) { return true; }
        }
        return false;
    }

    bool is_sibling_for(const std::shared_ptr<fuse_anchor_map_t> &other) const {
        return is_sibling_for(other.get());
    }

    /**
     * What is cousin relationship between anchor A and B
     * for(){
     *    for(){
     *      //
     *      { // anchor A
     *      }
     *    }
     *    for(){
     *      //
     *      { // anchor B
     *      }
     *    }
     * }
     * */
    bool is_cousin_for(const fuse_anchor_map_t *cur) const {
        return !(this->is_parent_for(cur) || cur->is_parent_for(this)
                       || this->is_sibling_for(cur)
                       || cur->is_sibling_for(this))
                && (cur->get_root() == this->get_root());
    }

    bool is_cousin_for(const std::shared_ptr<fuse_anchor_map_t> &cur) const {
        return is_cousin_for(cur.get());
    }
};

using fuse_anchor_map_ptr = std::shared_ptr<fuse_anchor_map_t>;

/**
 * iter_anchor represents irregular slice range which is binded with loop iter.
 * @param iter_: loop var
 * @param cached_iter_anchor_: real multi anchor used to commit code
 * */
struct fuse_iter_anchor_map_t : fuse_anchor_map_t {
    // iterated var
    expr iter_;
    size_t iter_size_;
    std::vector<stmts> cached_iter_anchor_;
    stmt dispatch_helper_;
    size_t iter_cnt_;

    fuse_iter_anchor_map_t(expr iter_var, stmts pos, const fslice_map &fsmap,
            size_t iter_size, stmt dispatch_helper = stmt(),
            const fuse_anchor_map_ptr &parent = nullptr)
        : fuse_anchor_map_t(pos, fsmap, parent)
        , iter_(std::move(iter_var))
        , iter_size_(iter_size)
        , dispatch_helper_(std::move(dispatch_helper)) {
        iter_cnt_ = 0;
        cached_iter_anchor_.reserve(iter_size_);
    }

    // iter anchor special inner-build `commit_`
    void commit_(stmt s) {
        if (cached_iter_anchor_.empty()) {
            if (dispatch_helper_.isa<stmts>()) {
                anchor_position_->seq_.insert(anchor_position_->seq_.end(),
                        dispatch_helper_.static_as<stmts>()->seq_.begin(),
                        dispatch_helper_.static_as<stmts>()->seq_.end());
            } else {
                anchor_position_->seq_.emplace_back(dispatch_helper_);
            }
        }
        // create cached_iter_anchor_ if necessary
        if (cached_iter_anchor_.size() < iter_size_) {
            stmts ss = s.isa<stmts>()
                    ? s.static_as<stmts>()
                    : builder::make_stmts_unattached({s}).checked_as<stmts>();
            anchor_position_->seq_.emplace_back(
                    make_stmt<if_else_node_t>(iter_ == iter_cnt_, ss, stmt()));
            cached_iter_anchor_.emplace_back(ss);
        }
        // commit into cached_iter_anchor_
        else {
            auto cached_anchor = cached_iter_anchor_.at(iter_cnt_);
            if (s.isa<stmts>()) {
                cached_anchor->seq_.insert(cached_anchor->seq_.end(),
                        s.static_as<stmts>()->seq_.begin(),
                        s.static_as<stmts>()->seq_.end());
            } else {
                cached_anchor->seq_.emplace_back(s);
            }
        }
        iter_cnt_++;
        if (iter_cnt_ == iter_size_) iter_cnt_ = 0;
    }

    // override commit `stmt` to anchor
    void commit_stmt(stmt &s) override { commit_(s); }

    // override commit `stmts` to anchor
    void commit_stmts(stmts &ss) override { commit_(ss); }
};

} // namespace sc

namespace std {
template <>
struct hash<sc::buffer_reuse_identity> {
    std::size_t operator()(const sc::buffer_reuse_identity &in) const;
};
} // namespace std
#endif
