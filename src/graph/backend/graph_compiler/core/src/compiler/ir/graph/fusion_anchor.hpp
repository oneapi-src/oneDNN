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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_GRAPH_FUSION_ANCHOR_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_GRAPH_FUSION_ANCHOR_HPP

#include <memory>
#include <utility>
#include <vector>
#include "graph.hpp"
#include "graph_map.hpp"
#include <compiler/ir/builder.hpp>
#include <compiler/ir/sc_expr.hpp>
#include <unordered_map>
#include <unordered_set>
#include <util/variant.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

struct mixed_parti_t;

using anchor_content_t = variant<sc_op *, fuse_anchor_map_t *>;
struct op_or_fuse_anchor_map_hasher {
    size_t operator()(const anchor_content_t &v) const {
        return std::hash<void *>()(v.cast<void *>());
    }
};

struct op_or_fuse_anchor_map_cmper {
    bool operator()(
            const anchor_content_t &v, const anchor_content_t &v2) const {
        return v.cast<void *>() == v2.cast<void *>();
    }
};

struct fuse_anchor_map_t : std::enable_shared_from_this<fuse_anchor_map_t> {
private:
    // control whether the fusion anchor is output anchor or input, default is
    // output anchor.
    bool is_input_anchor_;
    mixed_parti_t *binded_mxp_ = nullptr;

public:
    friend struct mixed_parti_t;
    stmts anchor_position_;
    fslice_map fsmap_;

    // parent anchor
    std::shared_ptr<fuse_anchor_map_t> parent_;

    /* Updated when inferring */
    // blocked graph tensor set, the reason why not use empty gt for judgement
    // is to distinguish non-visited gt and visited-but-failed gt
    std::unordered_set<graph_tensor_ptr> blocked_gt_set_;
    // borrowed fanchor map, must be the parent for current anchor
    std::unordered_map<graph_tensor_ptr, std::shared_ptr<fuse_anchor_map_t>>
            borrowed_fanchor_map_;

    /* Updated when committing */
    // content-to-number mapping under current fusion anchor scope, includes
    // either op and anchor
    std::unordered_map<anchor_content_t, size_t, op_or_fuse_anchor_map_hasher,
            op_or_fuse_anchor_map_cmper>
            content_number_map_;

    fuse_anchor_map_t() = default;
    fuse_anchor_map_t(stmts pos, const fslice_map &fsmap,
            const std::shared_ptr<fuse_anchor_map_t> &parent = nullptr,
            bool is_input_anchor = false)
        : is_input_anchor_(is_input_anchor)
        , anchor_position_(std::move(pos))
        , fsmap_(std::move(fsmap))
        , parent_(parent) {
        if (parent) { parent_->append_content(this); }
    };

    mixed_parti_t *const get_binded_mxp() const { return binded_mxp_; }

    bool defined() const { return anchor_position_.defined(); }

    bool is_input_anchor() const { return is_input_anchor_; };

    // commit `stmt` to anchor and bind parent node to commited anchor
    virtual void commit_stmt(const stmt &s) {
        add_parent_node(s, anchor_position_);
        anchor_position_->seq_.emplace_back(s);
    }

    // commit `stmts` to anchor and bind parent node to commited anchor
    virtual void commit_stmts(const stmts &ss) {
        for (auto &s : ss->seq_) {
            commit_stmt(s);
        }
    }

    // get all contents inside fusion anchor
    std::vector<anchor_content_t> get_contents() const {
        std::vector<anchor_content_t> ret;
        for (auto &mp : content_number_map_) {
            ret.emplace_back(mp.first);
        }
        return ret;
    }

    // append list of contents by the given number id
    void append_contents(
            std::vector<anchor_content_t> contents, size_t num_id) {
        for (auto &content : contents) {
            content_number_map_.insert(std::make_pair(content, num_id));
        }
    }

    // append content including either fusion anchor or sc op
    void append_content(anchor_content_t content);

    void attach_parent_anchor(const std::shared_ptr<fuse_anchor_map_t> &parent,
            const std::shared_ptr<fuse_anchor_map_t> &repl_parent);

    fuse_anchor_map_t *get_root() const {
        auto root = this;
        while (root->parent_) {
            COMPILE_ASSERT(root != root->parent_.get(),
                    "Ring parent anchor relationship found");
            root = root->parent_.get();
        }
        return const_cast<fuse_anchor_map_t *>(root);
    }

    void merge(const std::shared_ptr<fuse_anchor_map_t> &other);

    template <typename T>
    bool isa() const {
        static_assert(is_base_of_t<fuse_anchor_map_t, T>::value,
                "T is not a subclass of fuse_anchor_map.");
        return dynamic_cast<const T *>(this);
    }

    template <typename T>
    T *stc_cast() {
        return static_cast<T *>(this);
    }

    template <typename T>
    T *dyn_cast() {
        return dynamic_cast<T *>(this);
    }

    virtual ~fuse_anchor_map_t() = default;

    // This function will find the nearest parent 'for_loop' node for fusion
    // anchor
    virtual stmt get_parent_loop() const;

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
        if (is_input_anchor_ != cur->is_input_anchor_) return false;
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
        if (is_input_anchor_ != other->is_input_anchor_) return false;
        if (is_parent_for(other)) return false;
        auto this_loop = get_parent_loop();
        auto other_loop = other->get_parent_loop();
        // get parent loop
        auto parent_loop = get_parent_node(other_loop);
        while (parent_loop.defined()) {
            if (this_loop.ptr_same(parent_loop)) { return true; }
            parent_loop = get_parent_node(parent_loop);
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
        if (is_input_anchor_ != cur->is_input_anchor_) return false;
        return !(this->is_parent_for(cur) || cur->is_parent_for(this)
                       || this->is_sibling_for(cur)
                       || cur->is_sibling_for(this))
                && (cur->get_root() == this->get_root()
                        || get_common_parent_node(
                                anchor_position_, cur->anchor_position_)
                                   .defined());
    }

    bool is_cousin_for(const std::shared_ptr<fuse_anchor_map_t> &cur) const {
        return is_cousin_for(cur.get());
    }

    // check this anchor whether has view of given op
    bool has_view_of(sc_op *op);

    // check inputs for op
    bool check_input_for_op(
            const sc_op *op, std::unordered_set<graph_tensor_ptr> &known_gt);

    // validate inferred slice range of inputs for op, excluded known gt
    bool validate_input_for_op(const sc_op *op,
            const std::unordered_set<graph_tensor_ptr> &known_gt);

    // validate inferred slice range of outputs for op, usually for advanced
    // fusion anchor
    virtual bool validate_output_for_op(const sc_op *op) {
        auto ths = this;
        return !op->is_dynamic()
                || std::all_of(op->get_outputs().begin(),
                        op->get_outputs().end(),
                        [&ths](const graph_tensor_ptr &out) {
                            // dynamic case does not support multi-slice
                            // currently
                            return ths->fsmap_.get(out).size() == 1;
                        });
    }

    // forbid the given op in current anchor, set all gt(excluding known gt) to
    // blocking gt set
    void forbid_op(const sc_op *op,
            const std::unordered_set<graph_tensor_ptr> &known_gt);

    // check the depedency for the given op in current anchor
    bool check_dep_for_op(const sc_op *op);

    // query op committed into current fusion anchor whether is small workload
    bool is_small_op_workload(const sc_op *op);
};

using fuse_anchor_map_ptr = std::shared_ptr<fuse_anchor_map_t>;

/**
 * iter_anchor represents irregular slice range which is binded with loop iter.
 * @param iter_: loop var
 * @param iter_size: iteration size
 * @param cached_iter_anchor_: real multi anchor used to commit code
 * @param dispatch_helper_: the helper for IR dispatch
 * @param iter_cnt_: built-in iter counter
 * */
struct fuse_iter_anchor_map_t : fuse_anchor_map_t {
private:
    // iterated var
    expr iter_;
    size_t iter_size_;
    std::vector<stmts> cached_iter_anchor_;
    stmt dispatch_helper_;
    size_t iter_cnt_;

public:
    fuse_iter_anchor_map_t(expr iter_var, stmts pos, const fslice_map &fsmap,
            stmt dispatch_helper = stmt(),
            const fuse_anchor_map_ptr &parent = nullptr,
            bool is_input_anchor = false)
        : fuse_anchor_map_t(pos, fsmap, parent, is_input_anchor)
        , iter_(std::move(iter_var))
        , dispatch_helper_(std::move(dispatch_helper)) {
        COMPILE_ASSERT(
                !fsmap.empty(), "iterated fusion anchor init slice not found")
        iter_size_ = fsmap.datamap_.begin()->second.size();
        COMPILE_ASSERT(std::all_of(fsmap.datamap_.begin(), fsmap.datamap_.end(),
                               [&](const std::pair<graph_tensor *,
                                       slice_range_list> &p) {
                                   return p.second.size() == iter_size_;
                               }),
                "all init slice size of iterated fusion anchor should be equal")
        iter_cnt_ = 0;
        cached_iter_anchor_.reserve(iter_size_);
    }

    // iter anchor special inner-build `commit_`
    void commit(const stmt &s);

    // override commit `stmt` to anchor
    void commit_stmt(const stmt &s) override { commit(s); }

    // override commit `stmts` to anchor
    void commit_stmts(const stmts &ss) override { commit(ss); }

    const size_t get_iter_size() const { return iter_size_; }

    bool validate_output_for_op(const sc_op *op) override {
        auto ths = this;
        return std::all_of(op->get_outputs().begin(), op->get_outputs().end(),
                [&ths](const graph_tensor_ptr &out) {
                    return ths->get_iter_size() == ths->fsmap_.get(out).size();
                });
    }
};

/**
 * grouped_anchor contains grouped of basic anchors but under unified management
 * and committed simultaneously.
 * @param group_size_: total group amount
 * @param group_pos_: real group anchor position used to commit code
 * @param group_cnt_: built-in group counter
 * */
struct fuse_grouped_anchor_map_t : fuse_anchor_map_t {
private:
    size_t group_size_;
    size_t group_cnt_;

public:
    friend class fusion_anchor_mgr_t;

    fuse_grouped_anchor_map_t(stmts pos, const fslice_map &fsmap,
            const fuse_anchor_map_ptr &parent = nullptr,
            bool is_input_anchor = false)
        : fuse_anchor_map_t(pos, fsmap, parent, is_input_anchor) {
        COMPILE_ASSERT(
                !fsmap.empty(), "grouped fusion anchor init slice not found")
        group_size_ = fsmap.datamap_.begin()->second.size();
        COMPILE_ASSERT(std::all_of(fsmap.datamap_.begin(), fsmap.datamap_.end(),
                               [&](const std::pair<graph_tensor *,
                                       slice_range_list> &p) {
                                   return p.second.size() == group_size_;
                               }),
                "all init slice size of grouped fusion anchor should be equal")
        COMPILE_ASSERT(pos->seq_.size() == group_size_,
                "grouped anchor position " << pos->seq_.size()
                                           << " should be equal to group size "
                                           << group_size_)
        group_cnt_ = 0;
    }

    // group anchor special inner-build `commit_`
    void commit(const stmt &s);

    // override commit `stmt` to anchor
    void commit_stmt(const stmt &s) override { commit(s); }

    // override commit `stmts` to anchor
    void commit_stmts(const stmts &ss) override { commit(ss); }

    // grouped anchor should override `get_parent_loop` method due to it owns
    // multi anchor positions
    stmt get_parent_loop() const override;

    const size_t get_group_size() const { return group_size_; }

    bool validate_output_for_op(const sc_op *op) override {
        auto ths = this;
        return std::all_of(op->get_outputs().begin(), op->get_outputs().end(),
                [&ths](const graph_tensor_ptr &out) {
                    return ths->get_group_size() == ths->fsmap_.get(out).size();
                });
    }
};

using slice_map = std::unordered_map<graph_tensor *, slice_range_list>;

class fusion_anchor_mgr_t {
private:
    std::unordered_map<int, std::shared_ptr<fuse_grouped_anchor_map_t>>
            grouped_id_map_;

public:
    fusion_anchor_mgr_t() = default;
    // maintain fusion anchor list
    std::vector<fuse_anchor_map_ptr> fanchor_list_;

    // create basic fusion anchor
    void create_fusion_anchor(const slice_map &fsmap,
            const fuse_anchor_map_ptr &parent = nullptr,
            bool is_input_anchor = false);

    // create iterated fusion anchor
    void create_fusion_anchor(const expr &iter_var, const slice_map &fsmap,
            const stmt &dispatch_helper = stmt(),
            const fuse_anchor_map_ptr &parent = nullptr,
            bool is_input_anchor = false);

    // create grouped fusion anchor
    void create_fusion_anchor(int group_id, const slice_map &fsmap,
            const fuse_anchor_map_ptr &parent = nullptr,
            bool is_input_anchor = false);
};

// split common anchor into grouped anchor
fuse_anchor_map_ptr try_convert_anchor(
        const context_ptr &ctx, const fuse_anchor_map_ptr &fanchor);

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
