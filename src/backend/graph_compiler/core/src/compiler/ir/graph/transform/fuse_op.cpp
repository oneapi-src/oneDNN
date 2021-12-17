/*******************************************************************************
 * Copyright 2020-2021 Intel Corporation
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

#include "../fused_op.hpp"
#include "../fusible_op.hpp"
#include "../tunable_op.hpp"
#include "../visitor.hpp"
#include <compiler/ir/graph/fusion_mgr.hpp>
#include <compiler/ir/graph/pass/pass.hpp>
#include <unordered_set>

namespace sc {

SC_MODULE(graph.fuse_op);

static bool is_single_output_single_use(const sc_op_ptr &cur_node) {
    if (cur_node->get_outputs().size() != 1) { return false; }
    if (cur_node->get_outputs()[0]->uses_.size() != 1) { return false; }
    return true;
}

static sc_op_ptr get_single_output(const sc_op_ptr &cur_node) {
    return cur_node->get_outputs()[0]->uses_[0].second;
}

static constexpr const char *attr_key_partition = "fuse_op.partition";
static constexpr const char *attr_key_orig_op = "fuse_op.original_op";

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
    fusion_partition_t *get_root() const {
        if (merged_to) {
            assert(ops.empty());
            return merged_to->get_root();
        }
        return const_cast<fusion_partition_t *>(this);
    }

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
    bool is_ok_to_add(sc_op *op, const op_dep_matrix_t &g) const {
        if (merged_to) { return get_root()->is_ok_to_add(op, g); }
        for (auto &in : op->get_inputs()) {
            // if the input is in the partition, don't need to check
            if (contains(in->producer_owner_)) { continue; }
            // if "in" depends on an op in the partition
            if (main_tunable_op
                    && g.lookup(main_tunable_op->logical_op_id_,
                               in->producer_owner_->logical_op_id_)
                            == 1) {
                return false;
            }
            for (auto &op_in_set : ops) {
                auto result = g.lookup(op_in_set->logical_op_id_,
                        in->producer_owner_->logical_op_id_);
                // if "in" depends on an op in the partition
                if (result == 1) { return false; }
            }
        }
        return true;
    }

    bool contains(sc_op *op) const {
        if (merged_to) { return get_root()->contains(op); }
        return op == main_tunable_op.get()
                || ops.find(op->shared_from_this()) != ops.end();
    }

    // merge the ops in "other" to "this"
    void merge(const ptr &other) const {
        auto ths_root = get_root();
        auto other_root = other->get_root();
        if (ths_root == other_root) { return; }
        ths_root->ops.insert(other_root->ops.begin(), other_root->ops.end());
        assert(other_root->main_tunable_op == nullptr);
        other_root->ops.clear();
        other_root->merged_to = ths_root->shared_from_this();
    }
};

/**
 * Split the graph into partitions. It will walk through the ops in topology
 * order. For each fusible op, it will:
 * 1. Check the parent input nodes and find out their partitions. We will check
 * if the op can be added to the parent partition
 * 2. If there are multiple parent partitions that are ok to add to, try to
 * merge them.
 *  a) Any two partitions without base tunable op can be merged with each other
 *  b) If there is any partition with base tunable op, other partitions without
 * base tunable op can be merge to it.
 *  c) Do not merge two partitions both with different base tunable ops.
 *  d) If no parent partitions is avaliable, create a new partition and put the
 * current fusible op to it.
 *
 * For tunable op, we simply create a new partition for it.
 * @param g
 * @param dep
 * @param op_2_partition Outputs the op_id => fusion_partition in it
 * @param op_mask If empty, run partitioning on the whole graph. Otherwise, run
 * partitioning only on ops with op_mask[op_id]==true
 * */
static void do_partition(sc_graph_t &g, const op_dep_matrix_t &dep,
        std::vector<fusion_partition_t::ptr> &op_2_partition,
        const std::vector<bool> &op_mask) {
    // a DFS visitor which only visits masked ops and skips unmasked ones
    op_visitor_t visitor = op_visitor_t::dfs_topology_sort(g.ops_.size());
    // the checker to find if an op is in op_mask set
    auto is_in_subset = [&](sc_op *op) {
        if (!op_mask.empty() && !op_mask[op->logical_op_id_]) { return false; }
        return true;
    };
    visitor.visit_graph(g, [&](const sc_op_ptr &op) {
        if (!is_in_subset(op.get())) { return; }
        if (auto fusible = op->dyn_cast<fusible_op_t>()) {
            if (op->isa<input_op>() || op->isa<output_op>()) { return; }
            if (!op->isa<op_traits::copyable_t>()) {
                SC_MODULE_INFO << "Fusible op not copyable: " << op->op_name_;
                return;
            }
            if (fusible->attrs_.get_or_else(op_attr_key::no_fuse, false)) {
                return;
            }
            fusion_partition_t::ptr parent_partition;
            if (!fusible->attrs_.get_or_else(
                        op_attr_key::break_pre_fuse, false)) {
                // merge the partitons of all inputs
                for (auto &in : fusible->get_inputs()) {
                    if (!is_in_subset(in->producer_owner_)) { continue; }
                    auto &cur_in_partition
                            = op_2_partition[in->producer_owner_
                                                     ->logical_op_id_];
                    // if an input is fusible and is not "break_post_fuse"
                    if (cur_in_partition
                            && !in->producer_owner_->attrs_.get_or_else(
                                    op_attr_key::break_post_fuse, false)
                            && cur_in_partition->is_ok_to_add(fusible, dep)) {
                        if (parent_partition) {
                            // merge the parent partitions
                            if (!cur_in_partition->main_tunable_op) {
                                // if {parent_partition is post fusion after
                                // tunable op and cur_in_partition is not} or
                                // {parent_partition is not post fusion after
                                // tunable op and cur_in_partition is not},
                                // merge to parent_partition
                                parent_partition->merge(cur_in_partition);
                            } else if (!parent_partition->main_tunable_op
                                    && cur_in_partition->main_tunable_op) {
                                // if {parent_partition is not post fusion after
                                // tunable op and cur_in_partition is}, merge to
                                // cur_in_partition
                                cur_in_partition->merge(parent_partition);
                                parent_partition = cur_in_partition;
                            }
                            // if both parent_partition and cur_in_partition is
                            // post fusion after tunable op, do not merge these
                            // two partitions.
                        } else {
                            parent_partition = cur_in_partition;
                        }
                    }
                }
            }
            if (!parent_partition) {
                parent_partition = std::make_shared<fusion_partition_t>();
            }
            parent_partition->get_root()->ops.insert(op);
            op_2_partition[fusible->logical_op_id_] = parent_partition;
        } else if (auto tunable = op->dyn_cast<tunable_op_t>()) {
            if (tunable->attrs_.get_or_else(op_attr_key::no_fuse, false)) {
                return;
            }
            auto parent_partition = std::make_shared<fusion_partition_t>();
            parent_partition->main_tunable_op = op;
            op_2_partition[op->logical_op_id_] = parent_partition;
        }
    });

    for (auto &parti : op_2_partition) {
        if (parti) { parti = parti->get_root()->shared_from_this(); }
    }
}

// run partitioning on a subset of the graph, specified by the `partition`
static void repartition(sc_graph_t &g, const op_dep_matrix_t &dep,
        const fusion_partition_t::ptr &partition,
        std::vector<fusion_partition_t::ptr> &op_2_partition) {
    std::vector<bool> op_mask;
    op_mask.resize(op_2_partition.size());
    assert(partition);
    auto the_partition = partition; // NOLINT
    // no lint since partition may be destroyed
    if (the_partition->main_tunable_op) {
        auto opid = the_partition->main_tunable_op->logical_op_id_;
        op_2_partition[opid] = nullptr;
        op_mask[opid] = true;
    }
    // reset the op->partition map and set the mask
    for (auto &op : the_partition->ops) {
        op_2_partition[op->logical_op_id_] = nullptr;
        op_mask[op->logical_op_id_] = true;
    }
    do_partition(g, dep, op_2_partition, op_mask);
}

/**
 * Copies the partition to the graph in fusion manager.
 * @param base_op_out the output tensor of the base op. It will be mapped to the
 * input0 of the graph in fmgr
 * @param fmgr
 * @param partition
 * @param op_name it will append the fused op names to this string
 * @param out_output_tsr outputs the out tensors for the new fused op
 * @return the additional inputs besides the inputs of the original base op for
 * the fused op
 * */
static std::vector<graph_tensor_ptr> copy_partition_to_fmgr(sc_graph_t &g,
        const graph_tensor_ptr &base_op_out, fusion_manager *fmgr,
        fusion_partition_t &partition, std::string &op_name, bool keep_output,
        std::vector<graph_tensor_ptr> &out_output_tsr) {
    // the mapping for original LT in original ops to fuse => the LT in the
    // graph of fmgr
    std::unordered_map<graph_tensor_ptr, graph_tensor_ptr> orig_2_fmgr_graph;
    // if there is a base op, add an input_op for it in fmgr
    if (base_op_out) {
        orig_2_fmgr_graph[base_op_out]
                = fmgr->make<input_op>(base_op_out->details_)->get_outputs()[0];
    }
    if (keep_output) {
        assert(base_op_out);
        out_output_tsr.emplace_back(
                std::make_shared<graph_tensor>(nullptr, base_op_out->details_));
        // save the mapping of the tensor to be replaced => new tensor
        partition.output_replace_map[base_op_out] = out_output_tsr.back();
    }

    auto get_or_create_fmgr_tsr = [&](const graph_tensor_ptr &orig_lr) {
        auto itr = orig_2_fmgr_graph.find(orig_lr);
        if (itr != orig_2_fmgr_graph.end()) { return itr->second; }
        auto ret = std::make_shared<graph_tensor>(nullptr, orig_lr->details_);
        orig_2_fmgr_graph.insert(std::make_pair(orig_lr, ret));
        return ret;
    };
    // the additional args for the fused op in the original graph
    std::vector<graph_tensor_ptr> additional_args;
    auto visitor = op_visitor_t::dfs_topology_sort(g.ops_.size());
    std::unordered_set<graph_tensor_ptr> additional_args_set;
    visitor.visit_graph(g, [&](const sc_op_ptr &op) {
        if (partition.ops.find(op) == partition.ops.end()) { return; }
        std::vector<graph_tensor_ptr> fmgr_in, fmgr_out;
        for (auto &in : op->get_inputs()) {
            fmgr_in.emplace_back(get_or_create_fmgr_tsr(in));
            if (in != base_op_out && !partition.contains(in->producer_owner_)
                    && additional_args_set.find(in)
                            == additional_args_set.end()) {
                // if the input is not included in the partition, make an input
                // node
                fmgr->make_input({fmgr_in.back()});
                // add the input in the args of the fused op in orig graph
                additional_args.emplace_back(in);
                additional_args_set.insert(in);
            }
        }
        for (auto &out : op->get_outputs()) {
            fmgr_out.emplace_back(get_or_create_fmgr_tsr(out));
            // if the output is a "cut" - an edge across the partition and
            // outside of the partition
            bool is_cut = false;
            for (auto &use : out->uses_) {
                if (!partition.contains(use.second.get())) {
                    is_cut = true;
                    break;
                }
            }
            if (is_cut) {
                // if there is a use outside of the partition, the tensor should
                // be marked "output"
                const auto &outtsr = fmgr_out.back();
                fmgr->make<output_op>(outtsr);
                // make a new output tensor for the fused_op_t in the original
                // graph
                out_output_tsr.emplace_back(
                        std::make_shared<graph_tensor>(nullptr, out->details_));
                // save the mapping of the tensor to be replaced => new tensor
                partition.output_replace_map[out] = out_output_tsr.back();
            }
        }
        auto copyable = op->dyn_cast<op_traits::copyable_t>();
        assert(copyable);
        auto copied = copyable->copy(fmgr_in, fmgr_out, fmgr->get_graph());
        copied->attrs_[attr_key_orig_op] = op;

        // build the  fused op name
        if (!op_name.empty()) op_name += '_';
        op_name += copied->op_name_;
    });

    return additional_args;
}

// checks if a partition can be fused with base op. returns the fused op if
// success. Otherwise, returns null and returns the fusible_op_t that causes the
// problem in out_failed_ops
static sc_op_ptr check_partition_with_base_op(sc_graph_t &g,
        const sc_op_ptr &op,
        const fusion_partition_t::ptr &post_fusion_partition,
        std::vector<fusion_partition_t::ptr> &op_2_partition,
        std::vector<sc_op_ptr> &out_failed_ops) {
    auto fmgr = std::make_shared<fusion_manager>();
    std::string op_name;
    std::vector<graph_tensor_ptr> fused_op_in;
    sc_graph_t copied_op_graph;
    std::vector<graph_tensor_ptr> fused_op_out;
    bool multi_use = false;
    if (op) {
        // check if all uses of the base op is in the target partition
        // if there is a use outside of the partition, we should keep the
        // original output
        if (op->get_outputs()[0]->uses_.size() != 1) {
            for (auto &use : op->get_outputs()[0]->uses_) {
                if (!post_fusion_partition->contains(use.second.get())) {
                    multi_use = true;
                    break;
                }
            }
        }

        auto copyable = op->dyn_cast<op_traits::copyable_t>();
        COMPILE_ASSERT(
                copyable, "Expecting copyable base op: " << op->op_name_);
        op_name = op->op_name_;
        auto fused_op_addtional_in = copy_partition_to_fmgr(g,
                op->get_outputs()[0], fmgr.get(), *post_fusion_partition,
                op_name, multi_use, fused_op_out);
        fused_op_in = op->get_inputs();
        fused_op_in.insert(fused_op_in.end(), fused_op_addtional_in.begin(),
                fused_op_addtional_in.end());
        auto dummy_inop = copied_op_graph.make_input(
                copy_logical_tsr(op->get_inputs()));
        auto copy_of_main_op = copyable->copy(dummy_inop->get_outputs(),
                copy_logical_tsr(op->get_outputs()), copied_op_graph);
    } else {
        fused_op_in = copy_partition_to_fmgr(g, nullptr, fmgr.get(),
                *post_fusion_partition, op_name, multi_use, fused_op_out);
    }

    // if the last reorder is vnni format, do not fuse
    for (auto &outop : fmgr->get_graph().get_output_ops()) {
        for (auto &lastop : outop->get_inputs()) {
            if (auto reo_op
                    = lastop->producer_owner_->dyn_cast<reorder_op_t>()) {
                auto out_fmt = reo_op->get_output_format();
                // check VNNI format reorder
                auto block_size = out_fmt.get_blocks_size();
                auto blocks = out_fmt.blocks_;
                if (block_size > 2
                        && (blocks[block_size - 1] == 2
                                || blocks[block_size - 1] == 4))
                    out_failed_ops.emplace_back(
                            lastop->producer_owner_->shared_from_this());
            }
        }
    }
    if (!out_failed_ops.empty()) { return nullptr; }

    // if reshape is at the begining and end of the fusion pattern, do not fuse
    // it todo: we can discover this earlier
    for (auto &inop : fmgr->get_graph().get_input_ops()) {
        for (auto &firstop : inop->get_outputs()) {
            for (auto &user : firstop->uses_) {
                if (auto tv = user.second->dyn_cast<tensor_view_op_t>()) {
                    if (!op
                            || tv->attrs_[attr_key_orig_op]
                                            .get<sc_op_ptr>()
                                            ->get_inputs()[0]
                                            ->producer_owner_
                                    != op.get()) {
                        out_failed_ops.emplace_back(
                                user.second->shared_from_this());
                    }
                }
            }
        }
    }
    if (!out_failed_ops.empty()) { return nullptr; }
    for (auto &outop : fmgr->get_graph().get_output_ops()) {
        for (auto &lastop : outop->get_inputs()) {
            if (lastop->producer_owner_->isa<tensor_view_op_t>()) {
                out_failed_ops.emplace_back(
                        lastop->producer_owner_->shared_from_this());
            }
        }
    }
    if (!out_failed_ops.empty()) { return nullptr; }
    auto fused_op_ptr = std::make_shared<fused_op_t>(op_name,
            std::move(copied_op_graph), fmgr,
            /*ins*/ fused_op_in,
            /*outs*/
            fused_op_out, any_map_t {});
    if (multi_use) { fused_op_ptr->keep_outputs_[0] = true; }
    // todo: cache the get func result
    fused_op_ptr->try_get_func(get_default_context(), true, out_failed_ops);
    if (out_failed_ops.empty()) {
        return fused_op_ptr;
    } else {
        return nullptr;
    }
}

// checks if a partition can be fused with base op. If true, returns the
// fused_op. If false, returns null and repartitions the post_fusion_partition
static sc_op_ptr check_and_repartition(sc_graph_t &g,
        const op_dep_matrix_t &dep, const sc_op_ptr &op,
        const fusion_partition_t::ptr &post_fusion_partition,
        std::vector<fusion_partition_t::ptr> &op_2_partition) {
    std::vector<sc_op_ptr> out_failed_ops;
    auto ret = check_partition_with_base_op(
            g, op, post_fusion_partition, op_2_partition, out_failed_ops);
    if (!ret) {
        for (auto &failed_op : out_failed_ops) {
            auto &orig_op
                    = failed_op->attrs_[attr_key_orig_op].get<sc_op_ptr>();
            if (failed_op->attrs_.has_key(op_attr_key::fused_mode_hint)) {
                orig_op->attrs_.set(failed_op->attrs_.get<std::string>(
                                            op_attr_key::fused_mode_hint),
                        true);
                // clear hint if necessary in avoid of possible update.
                orig_op->attrs_.remove(op_attr_key::fused_mode_hint);
            } else {
                // use default no fuse flag
                orig_op->attrs_.set(op_attr_key::no_fuse, true);
            }
        }
        repartition(g, dep, post_fusion_partition, op_2_partition);
        return nullptr;
    } else {
        return ret;
    }
}

SC_INTERNAL_API void fuse_ops(sc_graph_t &g, const context_ptr &ctx) {
    if (!g.attrs_.get_or_else("temp.fuse", 1)) { return; }
    // phase 0: add attrs to some specific ops
    for (auto &op : g.ops_) {
        if (op->isa<reorder_op_t>()) {
            assert(op->get_inputs().size() == 1
                    && op->get_outputs().size() == 1);
            // Could not fuse other ops after padding reorder
            if (op->dyn_cast<reorder_op_t>()->check_padding())
                op->attrs_.set(op_attr_key::no_fuse, true);
        }
    }
    // mapping from op id => partition
    std::vector<fusion_partition_t::ptr> op_2_partition;
    op_2_partition.resize(g.ops_.size());
    op_dep_matrix_t dep(g);
    // phase 1, initial partitioning
    do_partition(g, dep, op_2_partition, {});

    std::vector<sc_op_ptr> fused_ops;

    constexpr int MAX_ITER = 20;
    // phase 2, try fusion partitions with base ops. May re-partition the graph
    // if fusion failed
    for (auto &op : g.ops_) {
        if (op->isa<tunable_op_t>()) {
            for (int retry = 0; retry < MAX_ITER; retry++) {
                const auto &post_fusion_partition
                        = op_2_partition[op->logical_op_id_];
                if (!post_fusion_partition) { break; }
                assert(post_fusion_partition->main_tunable_op == op);
                // skip single no fused tunable op
                if (post_fusion_partition->ops.empty()) { break; }
                auto fused_op = check_and_repartition(
                        g, dep, op, post_fusion_partition, op_2_partition);
                if (fused_op) {
                    fused_ops.emplace_back(fused_op);
                    fused_op->attrs_[attr_key_partition]
                            = std::weak_ptr<fusion_partition_t>(
                                    post_fusion_partition);
                    break;
                }
                // if fusion failed, the graph is re-partitioned and we will try
                // another time
            }
        }
    }

    // phase 3, try the partitions without base tunable ops. May re-partition
    for (auto parti : op_2_partition) { // NOLINT
        // here we use NOLINT on the line above due to futher modification
        if (!parti || parti->main_tunable_op
                || !parti->output_replace_map.empty()
                || parti->ops.size() < 2) {
            // if a partition has been processed or it is a single op, skip
            continue;
        }
        for (int retry = 0; retry < MAX_ITER; retry++) {
            auto fused_op = check_and_repartition(
                    g, dep, nullptr, parti, op_2_partition);
            if (fused_op) {
                fused_ops.emplace_back(fused_op);
                fused_op->attrs_[attr_key_partition]
                        = std::weak_ptr<fusion_partition_t>(parti);
                break;
            }
            // if fusion failed, the graph is re-partitioned and we will try
            // another time
        }
    }

    // phase 4, fused_ops is ready, rewrite the graph to replace the fused
    // ops with fused_ops
    std::unordered_map<graph_tensor_ptr, graph_tensor_ptr> tsr_replace_map;
    for (auto &fused_op : fused_ops) {
        auto partition = fused_op->attrs_[attr_key_partition]
                                 .get<std::weak_ptr<fusion_partition_t>>()
                                 .lock();
        assert(partition);
        fused_op->attrs_.remove(attr_key_partition);
        for (auto &old_new : partition->output_replace_map) {
            auto &old = old_new.first;
            auto &newv = old_new.second;
            old->replace_with(newv);
            assert(tsr_replace_map.find(old) == tsr_replace_map.end());
            tsr_replace_map.insert(old_new);
        }
        for (auto &in : fused_op->info_.inputs_) {
            // if an input is replaced by other fused_op node, update it
            auto itr = tsr_replace_map.find(in);
            if (itr != tsr_replace_map.end()) { in = itr->second; }
        }
        g.add(fused_op);
        // remove the original op mapping tag
        auto fused_op_ptr = fused_op->dyn_cast<::sc::fused_op_t>();
        for (auto &op : fused_op_ptr->mgr_->get_graph().ops_) {
            if (op->attrs_.has_key(attr_key_orig_op)) {
                op->attrs_.remove(attr_key_orig_op);
            }
        }
        for (auto &op : partition->ops) {
            op->remove();
        }
        if (partition->main_tunable_op) {
            partition->main_tunable_op->remove();
        }
    }
    g.reset_op_ids();
}

} // namespace sc
