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

#ifndef BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_GRAPH_MIXED_PARTITION_HPP
#define BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_GRAPH_MIXED_PARTITION_HPP
#include <memory>
#include <tuple>
#include <utility>
#include <vector>
#include "cost_model.hpp"
#include "fused_op.hpp"
#include "fusible_op.hpp"
#include "fusion_data.hpp"
#include "fusion_mgr.hpp"
#include "visitor.hpp"
#include <compiler/ir/visitor.hpp>
#include <unordered_map>
#include <unordered_set>

namespace sc {

struct mixed_parti_t;

namespace mixed_partition_hint {
// pointer: partition owner
constexpr const char *parti = "partition";
// Boolean: is cut buffer hint
constexpr const char *cut_buffer = "cut_buffer";
// the pointer: sub graph address
constexpr const char *sub_graph_ptr = "sub_graph_ptr";
// Boolean: dont inplace hint
constexpr const char *no_inplace = "no_inplace";
// Boolean: is retried graph
constexpr const char *retried_graph = "retried_graph";
} // namespace mixed_partition_hint

class mxp_replacer_t : public ir_inplace_visitor_t {
private:
    std::unordered_map<expr, expr> expr_map_;

public:
    mxp_replacer_t(std::unordered_map<expr, expr> &expr_map)
        : expr_map_(expr_map) {}

    using ir_inplace_visitor_t::dispatch_impl;
    using ir_inplace_visitor_t::visit_impl;
    expr visit_impl(var v) override {
        auto itr = expr_map_.find(v);
        if (itr != expr_map_.end()) {
            changed_ = true;
            return itr->second;
        }
        return v;
    }

    expr visit_impl(tensor v) override {
        auto itr = expr_map_.find(v);
        if (itr != expr_map_.end()) {
            changed_ = true;
            return itr->second;
        }
        return v;
    }

    void replace_func(func_t &func) {
        if (func) { dispatch_impl(func); }
    }
    void replace_anchor(const std::vector<fuse_anchor_map_ptr> &fanchors);
};

struct mxp_buffer_allocator {
    gt2buf_map g2b_map_; // record graph tensor to ir tensor/tensorptr
            // mapping(maybe n-to-one)
    std::unordered_map<expr, fuse_anchor_map_ptr>
            tsr_anch_map_; // real tensor-to-anchor mapping
    std::unordered_map<expr, graph_tensor_ptr>
            b2g_map_; // buffer-to-gt mapping(one-to-one)

    mixed_parti_t *binded_mxp_;

    // support inplace logic, allocate buffer including either tensor or
    // tensorptr
    void allocate_buffer(sc_op *op);
    // get allocated buffer
    std::tuple<std::vector<expr>, std::vector<expr>> get_buffer(sc_op *op);
    // update input buffer info
    void update_input_buffer_info(sc_op *op);
    // update output buffer info
    void update_output_buffer_info(sc_op *op);
    // set shrink info
    void declare_and_shrink_tensor();
    /** merge two buffer allocator
     * @param common_anchor_pair: the common anchor overlapped when two
     * partition merged. `first` comes from this partition, and `second` comes
     * from other one.
     * */
    void merge(mxp_buffer_allocator &other,
            std::unordered_map<expr, expr> &buffer_map,
            const std::pair<fuse_anchor_map_ptr, fuse_anchor_map_ptr>
                    &common_buffer_anchor_pair);
    // clear buffer allocator
    void clear();
    // check buffer_allocator whether empty
    const bool empty() const { return g2b_map_.empty(); };
    // initilize tensor
    void tensor_initialize();
    // replace the specific buffer
    void replace_buffer(graph_tensor *gt, expr &new_buffer);
    // calculate total allocated buffer size
    size_t get_total_allocated_buffer_size() const;
    // get real anchor for the specfic buffer
    fuse_anchor_map_ptr get_real_anchor_for_buffer(const expr &buffer) const;
    // get shrinked info for buffer
    slice_range get_shrinked_info(const expr &buffer) const;
    // query buffer inplace and set hint for IR pass
    void query_buffer_inplace();
};

struct outerloop_axis_binder {
    bound_axis_map bd_ax_map_;
    graph_tensor_ptr base_gt_;
    bound_axis init_axis_;

    // init ax_binder
    void init(const graph_tensor_ptr &base_gt, const bound_axis &axis) {
        base_gt_ = base_gt;
        init_axis_ = axis;
    }

    // clear
    void clear() {
        bd_ax_map_.clear();
        base_gt_ = nullptr;
        init_axis_.clear();
    }

    // only clear bd_ax_map
    void reset() { bd_ax_map_.clear(); }

    // auto infer axis binding information for the whole graph
    void run(int real_axis_size);

    /** algin with another axis binder according the given checking range.
     * @param other: another axis binder, alignment object
     * @param check_axis_size: checking range for outer loop, from left to right
     * @return int: the aligned loop size
     **/
    int align_with(outerloop_axis_binder &other, int check_axis_size);

    // split init axis, usually called when outer loops are split
    void split_init_axis(size_t loop_idx) {
        COMPILE_ASSERT(loop_idx < init_axis_.size(),
                "loop idx: " << loop_idx << " is larger than init axis size: "
                             << init_axis_.size())
        // copy original axis from the loop_idx init_axis
        auto orig_axis = init_axis_[loop_idx];
        // insert to init_axis
        init_axis_.emplace(init_axis_.begin() + loop_idx + 1, orig_axis);
    }
};

struct mixed_parti_t : fusion_partition_t {
    /* related to Graph */
    // different from ops_ in base class, it records the sequence of committed
    // ops in current partition
    std::vector<sc_op_ptr> committed_ops_;
    // binding axis information from GIR with outer loops in TIR
    outerloop_axis_binder ax_binder_;
    // dep matrix
    dep_mat_ptr dep_m_;

    /* related to IR */
    context_ptr ctx_;
    // the body of func_ will be updated once the new op is committed into
    // current partition, but the name and argument maybe not confirmed until
    // final mixed_fused_op created.
    func_t func_;
    // the fanchor only manage the shared pointer of fuse_anchor_map struct,
    // during the whole lifetime of mixed_parti_t, it will not copy any
    // fuse_anchor_map struct.
    std::vector<fuse_anchor_map_ptr> fanchors_;
    // manage graph tensor to real tensor mapping
    mxp_buffer_allocator buf_alloc_;
    // record the anchor to op mapping
    std::unordered_map<sc_op *, fuse_anchor_map_ptr> op_anchor_map_;

    // Cost Model
    cost_model cost_;

    using ptr = std::shared_ptr<mixed_parti_t>;

    // append fusion anchor
    void append_fusion_anchor(const fuse_anchor_map_ptr &fanchor) {
        fanchor->binded_mxp_ = this;
        fanchors_.emplace_back(fanchor);
    }

    void append_fusion_anchor(
            const std::vector<fuse_anchor_map_ptr> &fanchors) {
        for (auto &fanchor : fanchors) {
            append_fusion_anchor(fanchor);
        }
    }

    /**
     * The mixed partition merge will override base merge method, including
     * following several steps:
     * 1. It will firstly check two partition dependency and decide which one is
     * `to_merge` and another is `be_merged`.
     * 2. extract `outer_loops` from each one and compute greatest common outer
     * loops.
     * 3. commit inner loop body from `be_merged` to the largest used fanchor of
     * `to_merged`.
     * 4. update outer and inner fusion anchor map.
     * 5. replace expr iter/tensor/tensorptr in `func_`, `buf_alloc_` and
     * `fanchors_`.
     * 6. call base class `merge` method to do disjoint-set merge.
     * */
    void merge(const ptr &other);

    mixed_parti_t(const context_ptr &ctx, const sc_op_ptr &op,
            const dep_mat_ptr &dep_m);

    bool is_ok_to_add(sc_op *op, const op_dep_matrix_t &g);

    void add(const sc_op_ptr &op);

    void remove(const sc_op_ptr &op) {
        throw std::runtime_error("remove method is not implemented");
    }

    // if current partition contains no op or those ops generating no
    // codes(like tensorview op), return True.
    bool empty() const {
        if (merged_to) { return get_root()->empty(); }
        return (ops.empty() || buf_alloc_.empty());
    }

    mixed_parti_t *get_root() const {
        return static_cast<mixed_parti_t *>(fusion_partition_t::get_root());
    }

    // get outer loops of which body(stmts) contains only one stmt or two with
    // the second one is empty fanchor
    std::vector<for_loop> get_outer_loops(
            fuse_anchor_map_ptr fanchor = nullptr);

    void try_split_outermost_loop(int64_t block);

    // return op whether in op_anchor_map_
    bool ready_for_op(sc_op *op) const;

    // look up fanchor by op
    fuse_anchor_map_ptr lookup_anchor_map(sc_op *op) const;

    // look up fanchor by stmts
    fuse_anchor_map_ptr lookup_anchor_map(const stmts &ss) const;

    // look up sub fanchor by parent fanchor
    std::vector<fuse_anchor_map_ptr> lookup_sub_anchor_map(
            const fuse_anchor_map_ptr &parent_fanchor) const;

    // get anchor inside given loop
    fuse_anchor_map_ptr get_anchor_inside_loop(const for_loop &loop) const;

    /// get next inner loop including anchor
    for_loop get_next_inner_loop_with_anchor(const for_loop &cur_loop,
            const fuse_anchor_map_ptr &target_fanchor = nullptr) const;

    // clear all contents of given fanchor, but not erase it from
    // fanchor list
    void clear_fanchor(fuse_anchor_map_ptr &fanchor);

    // clear all unused fanchor, and erase them from fanchor list
    void clear_fanchors();

    // try to bind given op with given fanchor, if suitable fanchor exists, it
    // will compare two fanchor and select smaller one
    void set_anchor_for_op(sc_op *op, const fuse_anchor_map_ptr &fanchor_map);

    // schedule buffer
    void buffer_schedule();

    // judge whether the given graph tensor node is the input of the whole
    // partition
    bool is_parti_inp(const graph_tensor_ptr &gt);
    bool is_parti_inp(const graph_tensor *gt);

    // judge whether the given graph tensor node is the output of the whole
    // partition
    bool is_parti_out(const graph_tensor_ptr &gt);
    bool is_parti_out(const graph_tensor *gt);

    bool is_parti_cut(const graph_tensor_ptr &gt) {
        return is_parti_inp(gt) || is_parti_out(gt);
    }
    bool is_parti_cut(const graph_tensor *gt) {
        return is_parti_inp(gt) || is_parti_out(gt);
    }

    // count op number with given type
    template <typename T>
    size_t count_op_with_type() const;

    // query partition whether contains op with given type
    template <typename T>
    bool contain_op_with_type() const;

    // query partition whether contains op with conv type
    bool is_conv_workload() const;

    // query partition whether contains tunable op
    bool contain_tunable_op() const;

    // query partition whether contains only elementwise op
    bool contain_elemwise_op_only() const;

    // clear all contents of partition object
    void clear();

    // call inner-build cost model to evaluate current partition, return the
    // scores
    float evaluate_perf();
};

void extract_anchor_from_fmgr_to_parti(fusion_manager *fmgr,
        mixed_parti_t *parti, std::vector<expr> ir_tsrs,
        std::vector<graph_tensor_ptr> gtsrs,
        const fuse_anchor_map_ptr &parent_fanchor = nullptr);

void search_op_anchor_in_parti(sc_op *op, mixed_parti_t *parti);

void do_mixed_partition(const context_ptr &ctx, sc_graph_t &graph);

} // namespace sc
#endif
