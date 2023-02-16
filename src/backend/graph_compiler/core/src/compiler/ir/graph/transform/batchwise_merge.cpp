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

#include "../fusible_op.hpp"
#include "../graph_op.hpp"
#include "../pass/pass.hpp"
#include "../visitor.hpp"
#include "transform.hpp"
#include <compiler/ir/graph/fused_op.hpp>
#include <compiler/ir/graph/fusible_op_utils.hpp>
#include <ops/fusible/memory_movement.hpp>
#include <runtime/config.hpp>
#include <unordered_map>
#include <unordered_set>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

SC_MODULE(graph.batchwise_merge);

static constexpr const char *bw_attr_key_partition = "bw_fuse_op.partition";
static constexpr const char *bw_attr_key_orig_op = "bw_fuse_op.original_op";

static sc_dims extract_batchwise_dims(const sc_op_ptr &op) {
    sc_dims bw_dims;
    if (auto unstable_op = op->dyn_cast<op_traits::batchwise_shrinkable_t>()) {
        bw_dims = unstable_op->get_bwise_fuse_shrink_dims();
    }
    return bw_dims;
}

static sc_dims get_common_dims(const sc_dims &A, const sc_dims &B) {
    sc_dims ret;
    size_t common_size = std::min(A.size(), B.size());
    for (size_t i = 0; i < common_size; i++) {
        if (A[i] != B[i]) return ret;
        ret.emplace_back(A[i]);
    }
    return ret;
}

using bw_merge_map = std::unordered_map<sc_op_ptr, sc_dims>;

static void init_batchwise_merge(
        const context_ptr &ctx, sc_graph_t &graph, bw_merge_map &bw_map) {
    op_visitor_t vis = op_visitor_t::bfs();
    vis.visit_graph(graph, [&](op_visitor_t *vis, const sc_op_ptr &node) {
        // Due to this pass is behind `fuse_op` pass, if reorder op is found, it
        // is `no_fused` op. It will decide `use_output_loop` which also affect
        // batchwise dims.
        if (node->isa<reorder_op_t>())
            node->attrs_.set(op_attr_key::no_fuse, true);
        bw_map[node] = extract_batchwise_dims(node);
    });
}

sc_dims bw_merger(
        const sc_dims &common_dims, const sc_op_ptr &op, bw_merge_map &bw_map) {
    COMPILE_ASSERT(bw_map.find(op) != bw_map.end(),
            op->op_name_ << "is not initlized, please check it")
    auto bw_dims = bw_map[op];
    // no batchwise dim found
    if (bw_dims.empty()) { return bw_dims; };
    if (bw_dims != common_dims) {
        // try to get greatest common dims
        return get_common_dims(common_dims, bw_dims);
    }
    return common_dims;
}

struct bw_fusion_partition_t : fusion_partition_t {
    using ptr = std::shared_ptr<bw_fusion_partition_t>;
    sc_dims bw_dims_;

    void merge(const ptr &other) {
        fusion_partition_t::merge(other);
        bw_dims_ = get_common_dims(bw_dims_, other->bw_dims_);
    }
    bool is_ok_to_add(sc_op *op, const op_dep_matrix_t &g, bw_merge_map &bw_map,
            int least_shrink_ndim) const {
        if (!fusion_partition_t::is_ok_to_add(op, g)) return false;
        auto new_dims = bw_merger(bw_dims_, op->shared_from_this(), bw_map);
        if (static_cast<int>(new_dims.size()) < least_shrink_ndim) return false;
        sc_dim prod = get_dims_product(new_dims);
        if (prod == 1) return false;
        const int run_threads = runtime_config_t::get().get_num_threads();
        bool parallelism = (prod / run_threads > 8
                || (prod % run_threads == 0 && prod >= run_threads));
        if (!parallelism)
            SC_MODULE_INFO << "Considering parallelism, do not batchwised "
                              "merge op(or pattern): "
                           << op->op_name_;
        return parallelism;
    }
    bw_fusion_partition_t(sc_dims bw_dims) : bw_dims_(std::move(bw_dims)) {}
    bw_fusion_partition_t *get_root() const {
        return static_cast<bw_fusion_partition_t *>(
                fusion_partition_t::get_root());
    }

    void add(const sc_op_ptr &op, bw_merge_map &bw_map) {
        ops.insert(op);
        bw_dims_ = bw_merger(bw_dims_, op, bw_map);
    }
};

static bool do_partition(sc_graph_t &g, const op_dep_matrix_t &dep,
        std::vector<bw_fusion_partition_t::ptr> &op_2_partition,
        bw_merge_map &bw_map, int least_shrink_ndim,
        const std::vector<bool> &op_mask) {
    // a DFS visitor which only visits masked ops and skips unmasked ones
    op_visitor_t visitor = op_visitor_t::dfs_topology_sort(g.ops_.size());

    visitor.visit_graph(g, [&](op_visitor_t *visitor, const sc_op_ptr &op) {
        if (op->isa<input_op>() || op->isa<output_op>()
                || op->attrs_.get_or_else(op_attr_key::bwise_no_fuse, false)
                || op->attrs_.get_or_else(op_attr_key::bwise_skip_fuse, false)
                || op_mask[op->logical_op_id_])
            return;

        bw_fusion_partition_t::ptr parent_partition;
        if (!op->attrs_.get_or_else(op_attr_key::bwise_break_pre_fuse, false)) {
            // merge the partitons of all inputs
            for (auto &in : op->get_inputs()) {
                auto &cur_in_partition
                        = op_2_partition[in->producer_owner_->logical_op_id_];
                // if an input is fusible and is not "break_post_fuse"
                if (cur_in_partition
                        && !in->producer_owner_->attrs_.get_or_else(
                                op_attr_key::bwise_break_post_fuse, false)
                        && cur_in_partition->is_ok_to_add(
                                op.get(), dep, bw_map, least_shrink_ndim)
                        && in->producer_owner_->attrs_.get_or_else(
                                   "constant", const_kind::not_const)
                                == const_kind::not_const) {
                    if (parent_partition) {
                        parent_partition->merge(cur_in_partition);

                    } else {
                        parent_partition = cur_in_partition;
                    }
                }
            }
        }
        if (!parent_partition) {
            COMPILE_ASSERT(bw_map.find(op) != bw_map.end(),
                    op->op_name_ << "is not initlized, please check it")
            parent_partition
                    = std::make_shared<bw_fusion_partition_t>(bw_map[op]);
        }
        parent_partition->get_root()->add(op, bw_map);
        op_2_partition[op->logical_op_id_] = parent_partition;
    });

    auto check_partition = [&bw_map](bw_fusion_partition_t::ptr &parti,
                                   std::unordered_set<sc_op_ptr> &retry_ops) {
        if (!parti || parti->ops.empty() || parti->ops.size() < 2) return;
        for (auto &op : parti->ops) {
            COMPILE_ASSERT(bw_map.find(op) != bw_map.end(),
                    op->op_name_ << "is not initlized, please check it")
            COMPILE_ASSERT(
                    !op->attrs_.has_key(op_attr_key::bwise_no_strided_dims)
                            || (op->attrs_.get_or_else(
                                        op_attr_key::bwise_break_pre_fuse,
                                        false)
                                    || op->attrs_.get_or_else(
                                            op_attr_key::bwise_break_post_fuse,
                                            false)),
                    "If the op has been set batch-wise no stride dims, it "
                    "means it "
                    "may cause bwise break fusion");
            if (op->isa<tensor_view_op_t>()) {
                for (auto &user : op->get_outputs()[0]->uses_) {
                    if (parti->ops.find(user.second.get_shared())
                            == parti->ops.end()) {
                        op->attrs_[op_attr_key::bwise_skip_fuse] = true;
                        retry_ops.insert(op);
                        break;
                    }
                }
                if (parti->ops.find(
                            op->get_inputs()[0]
                                    ->producer_owner_->shared_from_this())
                        == parti->ops.end()) {
                    op->attrs_[op_attr_key::bwise_skip_fuse] = true;
                    retry_ops.insert(op);
                }
            } else if (op->isa<reorder_op_t>()
                    && op->attrs_.has_key(op_attr_key::bwise_no_strided_dims)) {
                auto no_strided_dims = op->attrs_.get<sc_dims>(
                        op_attr_key::bwise_no_strided_dims);
                if (parti->bw_dims_.size() <= no_strided_dims.size()) {
                    // use no strided dims insteadly
                    bw_map[op] = no_strided_dims;
                    if (op->attrs_.has_key(op_attr_key::bwise_break_pre_fuse))
                        op->attrs_.remove(op_attr_key::bwise_break_pre_fuse);
                    if (op->attrs_.has_key(op_attr_key::bwise_break_post_fuse))
                        op->attrs_.remove(op_attr_key::bwise_break_post_fuse);
                    op->attrs_.remove(op_attr_key::bwise_no_strided_dims);
                } else {
                    op->attrs_[op_attr_key::bwise_skip_fuse] = true;
                }
                retry_ops.insert(op);
            } else if (op->isa<fused_op_t>()
                    && op->attrs_.has_key(op_attr_key::bwise_no_strided_dims)) {
                auto no_strided_dims = op->attrs_.get<sc_dims>(
                        op_attr_key::bwise_no_strided_dims);
                if (parti->bw_dims_.size() <= no_strided_dims.size()) {
                    // use no strided dims insteadly
                    bw_map[op] = no_strided_dims;
                    if (op->attrs_.has_key(op_attr_key::bwise_break_pre_fuse))
                        op->attrs_.remove(op_attr_key::bwise_break_pre_fuse);
                    if (op->attrs_.has_key(op_attr_key::bwise_break_post_fuse))
                        op->attrs_.remove(op_attr_key::bwise_break_post_fuse);
                    op->attrs_.remove(op_attr_key::bwise_no_strided_dims);
                    retry_ops.insert(op);
                }
            }
        }
    };

    std::unordered_set<sc_op_ptr> retry_ops;
    for (auto &parti : op_2_partition) {
        check_partition(parti, retry_ops);
    }
    if (!retry_ops.empty()) return false;

    for (auto &parti : op_2_partition) {
        if (parti) {
            parti = std::static_pointer_cast<bw_fusion_partition_t>(
                    parti->get_root()->shared_from_this());
        }
    }
    return true;
}

/**
 * Copies the partition to the graph in new sub graph.
 * @param graph
 * @param partition
 * @param op_name it will append the fused op names to this string
 * @param out_output_tsr outputs the out tensors for the new fused op
 * @return the additional inputs besides the inputs of the original base op for
 * the fused op
 * */
static std::vector<graph_tensor_ptr> copy_partition_to_graph(sc_graph_t &g,
        sc_graph_t &bw_graph, bw_fusion_partition_t &partition,
        std::string &op_name, std::vector<graph_tensor_ptr> &out_output_tsr) {
    // the mapping for original LT in original ops to fuse => the LT in the
    // graph.
    std::unordered_map<graph_tensor_ptr, graph_tensor_ptr> orig_2_graph;

    auto get_or_create_graph_tsr = [&](const graph_tensor_ptr &orig_lr) {
        auto itr = orig_2_graph.find(orig_lr);
        if (itr != orig_2_graph.end()) { return itr->second; }
        auto ret = std::make_shared<graph_tensor>(nullptr, orig_lr->details_);
        orig_2_graph.insert(std::make_pair(orig_lr, ret));
        return ret;
    };
    // the additional args for the fused op in the original graph
    std::vector<graph_tensor_ptr> input_tsr;
    auto visitor = op_visitor_t::dfs_topology_sort(g.ops_.size());
    std::unordered_set<graph_tensor_ptr> input_tsr_set;
    visitor.visit_graph(g, [&](op_visitor_t *visitor, const sc_op_ptr &op) {
        if (partition.ops.find(op) == partition.ops.end()) { return; }
        std::vector<graph_tensor_ptr> new_graph_in, new_graph_ou;
        for (auto &in : op->get_inputs()) {
            new_graph_in.emplace_back(get_or_create_graph_tsr(in));
            if (!partition.contains(in->producer_owner_)
                    && input_tsr_set.find(in) == input_tsr_set.end()) {
                // if the input is not included in the partition, make an input
                // node
                bw_graph.make_input({new_graph_in.back()});
                // add the input in the args of the fused op in orig graph
                input_tsr.emplace_back(in);
                input_tsr_set.insert(in);
            }
        }
        for (auto &out : op->get_outputs()) {
            new_graph_ou.emplace_back(get_or_create_graph_tsr(out));
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
                const auto &outtsr = new_graph_ou.back();
                bw_graph.make_output({outtsr});
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
        auto copied = copyable->copy(new_graph_in, new_graph_ou, bw_graph);
        copied->attrs_[bw_attr_key_orig_op] = op;

        // build the fused op name
        if (!op_name.empty()) op_name += '_';
        std::string *name = &copied->op_name_;
        if (auto layer_name = copied->attrs_.get_or_null<std::string>(
                    op_attr_key::layer_name)) {
            name = layer_name;
        }
        op_name += *name;
    });

    return input_tsr;
}

static std::string print_dims(const sc_dims &vec) {
    std::stringstream os;
    int cnt = 0;
    for (auto &v : vec) {
        if (cnt != 0) { os << "X"; }
        os << v;
        cnt++;
    }
    return os.str();
}

static void do_batchwise_merge(
        const context_ptr &ctx, sc_graph_t &graph, bw_merge_map &bw_map) {
    auto init_partition = [&graph, &bw_map]() {
        std::vector<bw_fusion_partition_t::ptr> partition;
        partition.reserve(graph.ops_.size());
        std::transform(graph.ops_.begin(), graph.ops_.end(),
                std::back_inserter(partition), [&bw_map](const sc_op_ptr &cur) {
                    COMPILE_ASSERT(bw_map.find(cur) != bw_map.end(),
                            cur->op_name_
                                    << "is not initlized, please check it")
                    return std::make_shared<bw_fusion_partition_t>(bw_map[cur]);
                });
        return partition;
    };

    op_dep_matrix_t dep(graph);
    // mapping from op id => partition
    auto op_2_partition = init_partition();
    // initial partitioning
    constexpr int bw_merge_top_level = 3, maxiter = 10;
    std::vector<bool> op_mask(graph.ops_.size(), false);
    for (int bw_merge_level = bw_merge_top_level; bw_merge_level > 0;
            bw_merge_level--) {
        // mapping partition in cur bwise merge level
        std::vector<bw_fusion_partition_t::ptr> cur_op_2_partition;
        for (int i = 0; i < maxiter; i++) {
            cur_op_2_partition = init_partition();
            if (do_partition(graph, dep, cur_op_2_partition, bw_map,
                        bw_merge_level, op_mask))
                break;
        }
        // copy cur to final partition
        for (size_t i = 0; i < op_mask.size(); i++) {
            if (!op_mask[i]) op_2_partition[i] = cur_op_2_partition[i];
        }
        // set op_mask
        std::transform(op_2_partition.begin(), op_2_partition.end(),
                op_mask.begin(), [](const bw_fusion_partition_t::ptr &parti) {
                    return (parti && parti->ops.size() >= 2);
                });
        // clear temporarily attr
        for (auto &op : graph.ops_) {
            if (op->attrs_.has_key(op_attr_key::bwise_skip_fuse))
                op->attrs_.remove(op_attr_key::bwise_skip_fuse);
        }
    }
    // remove unused attr
    for (auto &op : graph.ops_) {
        if (op->attrs_.has_key(op_attr_key::bwise_no_fuse))
            op->attrs_.remove(op_attr_key::bwise_no_fuse);
        if (op->attrs_.has_key(op_attr_key::bwise_break_pre_fuse))
            op->attrs_.remove(op_attr_key::bwise_break_pre_fuse);
        if (op->attrs_.has_key(op_attr_key::bwise_break_post_fuse))
            op->attrs_.remove(op_attr_key::bwise_break_post_fuse);
        if (op->attrs_.has_key(op_attr_key::bwise_no_strided_dims))
            op->attrs_.remove(op_attr_key::bwise_no_strided_dims);
    }
    std::vector<sc_op_ptr> fused_ops;
    for (auto &parti : op_2_partition) {
        if (!parti || !parti->output_replace_map.empty()
                || parti->ops.size() < 2) {
            // if a partition has been processed or it is a single op, skip
            continue;
        }
        COMPILE_ASSERT(
                !parti->bw_dims_.empty(), "batchwise dims should not be empty")
        sc_graph_t bw_graph;
        std::vector<graph_tensor_ptr> fused_op_in;
        std::vector<graph_tensor_ptr> fused_op_out;
        std::string graph_name
                = "batchwise_" + print_dims(parti->bw_dims_) + "_fused";
        fused_op_in = copy_partition_to_graph(
                graph, bw_graph, *parti, graph_name, fused_op_out);
        auto fused_op = std::make_shared<batchwise_fused_op_t>(graph_name,
                parti->bw_dims_, bw_graph,
                /*ins*/ fused_op_in,
                /*outs*/
                fused_op_out, any_map_t {});
        // print_graph(bw_graph, std::cout, 1);
        fused_op->attrs_[bw_attr_key_partition]
                = std::weak_ptr<bw_fusion_partition_t>(parti);
        fused_ops.emplace_back(fused_op);
    }

    std::unordered_map<graph_tensor_ptr, graph_tensor_ptr> tsr_replace_map;
    for (auto &fused_op : fused_ops) {
        auto partition = fused_op->attrs_[bw_attr_key_partition]
                                 .get<std::weak_ptr<bw_fusion_partition_t>>()
                                 .lock();
        assert(partition);
        fused_op->attrs_.remove(bw_attr_key_partition);
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
        graph.add(fused_op);
        // remove the original op mapping tag
        auto fused_op_ptr = fused_op->dyn_cast<batchwise_fused_op_t>();
        for (auto &op : fused_op_ptr->bw_graph_.ops_) {
            if (op->attrs_.has_key(bw_attr_key_orig_op)) {
                op->attrs_.remove(bw_attr_key_orig_op);
            }
        }
        for (auto &op : partition->ops) {
            op->remove();
        }
        if (partition->main_tunable_op) {
            partition->main_tunable_op->remove();
        }
    }
    graph.reset_op_ids();
}

void batchwise_merge(sc_graph_t &graph, const context_ptr &ctx) {
    // todo: find out how to support this pass in dynamic
    if (graph.is_dynamic()) { return; }
    if (graph.is_non_dense()) { return; }
    if (!graph.attrs_.get_or_else("temp.fuse", 1)) { return; }
    bw_merge_map bw_map;
    init_batchwise_merge(ctx, graph, bw_map);
    do_batchwise_merge(ctx, graph, bw_map);
}

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
