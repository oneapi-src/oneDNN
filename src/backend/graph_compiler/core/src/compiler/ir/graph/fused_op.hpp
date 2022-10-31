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
#ifndef BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_GRAPH_FUSED_OP_HPP
#define BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_GRAPH_FUSED_OP_HPP

#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "graph_op.hpp"
#include "visitor.hpp"
#include <compiler/ir/graph/dynamic_dispatch_key.hpp>
#include <compiler/ir/graph/trait/may_inplace.hpp>
#include <unordered_map>
#include <unordered_set>

namespace sc {
namespace horizontal_merge_type {
constexpr int no_merge = 0;
}
class fusion_manager;
struct mixed_parti_t;

struct fusion_partition_t : std::enable_shared_from_this<fusion_partition_t> {
    // the fusible ops in the partition. Not including base tunable op
    std::unordered_set<sc_op_ptr> ops;
    // valid only after phase 3
    sc_op_ptr main_tunable_op;
    // valid only after phase 3, out tensors of old ops to be fused => new
    // fused_op out tensors
    std::unordered_map<graph_tensor_ptr, graph_tensor_ptr> output_replace_map;
    using ptr = std::shared_ptr<fusion_partition_t>;
    ptr merged_to;
    // if "this" is not merged, return this. If "this" is merged to another
    // partition, return the partition
    fusion_partition_t *get_root() const;

    virtual ~fusion_partition_t() = default;

    /**
     * Checks if an `op` node can be added to the partition. We check that for
     * each input op x in op.get_inputs(), if x is not in the partition, then x
     * must not depend on any op in the current partition.
     * We add this check to avoid potential cycles in the graph after fusion.
     * e.g., think the following graph (something like the basic block of
     * resnet)
     * v0=conv(v1,v2)
     * v3=relu(v0)
     * v4=conv(v3,v6)
     * v7=add(v3,v4)
     *
     * We first add relu_v3 to the partition. Note that the add_v7 depends on
     * our partition by 2 paths: v3->v7 and v3->v4->v7. For add_v7's
     * dependencies, relu_v3 is in the partition and conv_v4 is not. If we fuse
     * conv_v0, relu_v3 and add_v7, there will be a cycle in the graph:
     * conv_v4 depends on v3, which is the output of the fused op. But the fused
     * op depends on v4, which is the output of conv_v4
     */
    bool is_ok_to_add(sc_op *op, const op_dep_matrix_t &g) const;

    bool contains(sc_op *op) const;

    // merge the ops in "other" to "this"
    void merge(const ptr &other) const;
};

// inputs: base op inputs, additional args inputs (should be in the same order
// of the input ops in fmgr)
// outputs: If need to keep base op output, base op output will be the first
// element in the outs. Then the output of fmgr
class fused_op_t : public graph_op_t,
                   public op_traits::copyable_t,
                   public op_traits::batchwise_shrinkable_t {
public:
    std::shared_ptr<fusion_manager> mgr_;
    sc_graph_t main_op_;
    std::vector<bool> keep_outputs_ = {false};
    // for dispatch
    sc_op_ptr main_dispatch_op_;
    op_traits::post_fusion_acceptable_t *get_main_op() const;
    fused_op_t(const std::string &name, sc_graph_t &&main_op,
            std::shared_ptr<fusion_manager> fuse_mgr,
            const std::vector<graph_tensor_ptr> &ins,
            const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs);
    void get_graph_impl(std::shared_ptr<sc_graph_t> &graph) override;
    ir_module_ptr get_func(context_ptr ctx) override;
    bool is_valid(const context_ptr &) override;
    bool compare_contents(const sc_op *other) const override;
    size_t hash_contents() const override;
    ir_module_ptr try_get_func(const context_ptr &ctx, bool just_check,
            std::vector<sc_op_ptr> &out_failed);

    sc_op_ptr copy(const std::vector<graph_tensor_ptr> &ins, // NOLINT
            const std::vector<graph_tensor_ptr> &outs,
            sc_graph_t &mgr) override;

    sc_dims get_bwise_fuse_shrink_dims() override;
    sc_op_ptr bw_shrinked_copy(
            gt2gt_map &bw_lt_map, sc_graph_t &shrinked_graph) override;
    void collect_shrinked_lt_map(int bw_size, gt2gt_map &bw_lt_map) override;
    void collect_shrinked_axes_map(
            int bw_size, gt2axes_map &bw_axes_map) override;

    // dynamic related
    virtual const dispatch_set_ptr &get_dispatch_key_set() const override;
    virtual dispatch_set_ptr &get_dispatch_key_set() override;
    // Return vector of dispatch key set of inner ops like tunable op/reorder
    // op(dispatch alg). total_key_num is the pointer to number of input/output
    // dispatch key inside(option). The return value is mainly used for combined
    // dispatch key construction.
    virtual std::vector<dispatch_set_ptr> get_inner_dispatch_key_sets(
            int *total_key_num);
    void update_internal_graph_format(const combined_op_dispatch_key_t &key);
    ir_module_ptr get_dynamic_query_func(const context_ptr &ctx);
    // return the impl alg candidates vector, element is int(not enum) because
    // different ops have different impl algs.
    std::vector<int> get_impl_dispatch_candidates() const override;
};

class horizontal_fused_op_t : public graph_op_t {
public:
    horizontal_fused_op_t(const std::string &name,
            const std::vector<sc_op_ptr> &ops_to_merge,
            const std::vector<graph_tensor_ptr> &ins,
            const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs);

    std::vector<sc_op_ptr> ops_to_merge_;
    ir_module_ptr get_func(context_ptr ctx) override;
    void get_graph_impl(std::shared_ptr<sc_graph_t> &graph) override;
    void schedule_loops(const stmt &body);
};

class batchwise_fused_op_t : public graph_op_t {
public:
    batchwise_fused_op_t(const std::string &name, const sc_dims &bw_dims,
            const sc_graph_t &bw_graph,
            const std::vector<graph_tensor_ptr> &ins,
            const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs);
    sc_dims bw_dims_;
    sc_graph_t bw_graph_;
    ir_module_ptr get_func(context_ptr ctx) override;
    void get_graph_impl(std::shared_ptr<sc_graph_t> &graph) override;
    void schedule_loops(const stmt &body);
};

class mixed_fuse_op_t : public graph_op_t, public op_traits::may_inplace_t {
public:
    mixed_fuse_op_t(const std::string &name,
            const std::shared_ptr<mixed_parti_t> &parti,
            const sc_graph_t &graph, const std::vector<graph_tensor_ptr> &ins,
            const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs);
    sc_graph_t sub_graph_;
    std::shared_ptr<mixed_parti_t> parti_;
    ir_module_ptr get_func(context_ptr ctx) override;
    void get_graph_impl(std::shared_ptr<sc_graph_t> &) override;
    void schedule_loops(const stmt &body);
    std::vector<std::pair<int, std::vector<tensor_inplace_info_t>>>
    get_inplace_map() override;
};

void schedule_loop_body(
        const stmt &body, std::unordered_map<expr, expr> *expr_remap = nullptr);
} // namespace sc

#endif
