/*******************************************************************************
 * Copyright 2020-2023 Intel Corporation
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

#include "fusion_mgr.hpp"
#include <algorithm>
#include <assert.h>
#include <atomic>
#include <iostream>
#include <limits>
#include <string>
#include <utility>
#include "../easy_build.hpp"
#include "fusible_op.hpp"
#include "fusion_mgr_utils.hpp"
#include <compiler/ir/builtin.hpp>
#include <compiler/ir/graph/fusible_op_utils.hpp>
#include <compiler/ir/graph/pass/pass.hpp>
#include <compiler/ir/graph/visitor.hpp>
#include <compiler/ir/ir_utils.hpp>
#include <compiler/ir/transform/auto_cast.hpp>
#include <compiler/ir/transform/buffer_schedule.hpp>
#include <compiler/ir/transform/constant_fold.hpp>
#include <compiler/ir/transform/tensor_shrink.hpp>
#include <compiler/ir/visitor.hpp>
#include <ops/fusible/memory_movement.hpp>
#include <ops/fusible/reduce.hpp>
#include <ops/fusible/unary_elemwise.hpp>
#include <runtime/config.hpp>
#include <unordered_set>
#include <util/any_map.hpp>

SC_MODULE(graph.fusion);

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
fusion_mgr_ptr fusion_manager::copy() const {
    fusion_mgr_ptr new_fmgr = std::make_shared<fusion_manager>();
    auto &graph = get_graph();
    for (auto &op : graph.ops_) {
        if (!op->is_removed_ && !op->dyn_cast<input_op>()
                && !op->dyn_cast<output_op>()
                && !op->dyn_cast<op_traits::copyable_t>()) {
            return new_fmgr;
        }
    }
    op_visitor_t vis = op_visitor_t::bfs_topology_sort(graph.ops_.size());
    std::unordered_map<graph_tensor_ptr, graph_tensor_ptr> old_new_lt_map;
    std::unordered_map<sc_op_ptr, int> op_id_map;
    vis.visit_graph(graph, [&](op_visitor_t *vis, const sc_op_ptr &node) {
        sc_op_ptr new_node;
        if (node->dyn_cast<input_op>()) {
            new_node = new_fmgr->make_input(
                    copy_logical_tsr(node->get_outputs()));

            // "unique_id" for integration
            new_node->attrs_ = node->attrs_;
        } else {
            std::vector<graph_tensor_ptr> ins;
            ins.reserve(node->get_inputs().size());
            for (auto &t : node->get_inputs()) {
                ins.emplace_back(old_new_lt_map.at(t));
            }
            if (node->dyn_cast<output_op>()) {
                const auto &outtsr = ins[0];
                new_node = new_fmgr->make<output_op>(outtsr);
                // "unique_id" for integration
                new_node->attrs_ = node->attrs_;
            } else {
                new_node = node->dyn_cast<op_traits::copyable_t>()->copy(ins,
                        copy_logical_tsr(node->get_outputs()),
                        new_fmgr->get_graph());
                new_node->info_.cur_impl_ = node->info_.cur_impl_;
            }
        }
        // recording old graph_tensor->new graph_tensor
        for (size_t i = 0; i < new_node->get_outputs().size(); ++i) {
            old_new_lt_map[node->get_outputs()[i]] = new_node->get_outputs()[i];
        }
        op_id_map[new_node] = node->logical_op_id_;
    });
    new_fmgr->get_graph().attrs_ = graph.attrs_;
    new_fmgr->get_graph().resort_op_ids(op_id_map);
    return new_fmgr;
}

int fusion_manager::get_input_idx(sc_op *v) const {
    auto itr = input_idx_map_.find(v);
    assert(itr != input_idx_map_.end());
    return itr->second;
}
int fusion_manager::get_output_idx(sc_op *v) const {
    auto itr = output_idx_map_.find(v);
    assert(itr != output_idx_map_.end());
    return itr->second;
}

template <>
std::shared_ptr<input_op> fusion_manager::make<input_op>() {
    auto ret = std::make_shared<input_op>();
    graph_.add(ret);
    input_idx_map_[ret.get()] = input_op_count_++;
    return ret;
}
template <>
std::shared_ptr<input_op> fusion_manager::make<input_op>(
        sc_dims &&dims, const sc_data_type_t &dtype) {
    auto ret = std::make_shared<input_op>(dims, dtype);
    graph_.add(ret);
    input_idx_map_[ret.get()] = input_op_count_++;
    return ret;
}

template <>
std::shared_ptr<input_op> fusion_manager::make<input_op>(
        const logical_tensor_t &lt) {
    auto ret = std::make_shared<input_op>(lt);
    graph_.add(ret);
    input_idx_map_[ret.get()] = input_op_count_++;
    return ret;
}

template <>
std::shared_ptr<input_op> fusion_manager::make<input_op>(logical_tensor_t &lt) {
    return make<input_op>(const_cast<const logical_tensor_t &>(lt));
}

sc_op_ptr fusion_manager::make_input(const std::vector<graph_tensor_ptr> &tsr) {
    auto ret = std::make_shared<input_op>(tsr);
    graph_.add(ret);
    input_idx_map_[ret.get()] = input_op_count_++;
    return ret;
}

template <>
std::shared_ptr<input_op> fusion_manager::make<input_op>(
        logical_tensor_t &&lt) {
    return make<input_op>(static_cast<const logical_tensor_t &>(lt));
}

template <>
std::shared_ptr<output_op> fusion_manager::make<output_op>(
        const graph_tensor_ptr &arg) {
    auto ret = std::make_shared<output_op>(arg);
    graph_.add(ret);
    output_idx_map_[ret.get()] = output_op_count_++;
    return ret;
}

void fusion_manager::add(fusion_op_ptr node) {
    graph_.add(std::move(node));
}

void fusion_manager::put_input_first(input_op *inp) {
    COMPILE_ASSERT(input_idx_map_.find(inp) != input_idx_map_.end(),
            "Cound not found given input op in current fusion manager graph");
    int inp_idx = input_idx_map_.find(inp)->second;
    for (auto &cur_inp : graph_.get_input_ops()) {
        if (input_idx_map_[cur_inp.get()] < inp_idx) {
            input_idx_map_[cur_inp.get()]++;
        } else if (cur_inp.get() == inp) {
            input_idx_map_[cur_inp.get()] = 0;
        }
    }
}

const sc_op *fusion_manager::get_first_input() const {
    for (auto &m : input_idx_map_) {
        if (m.second == 0) return m.first;
    }
    return nullptr;
}

void fusion_data_t::set_buffer(bool is_dynamic, const expr &buf) {
    if (buf.isa<tensorptr>()) {
        COMPILE_ASSERT(!buf.static_as<tensorptr>()->is_slice_,
                "tensorptr is only used for tensorview op inside fmgr")
        auto base = buf.static_as<tensorptr>()->base_;
        COMPILE_ASSERT(base.isa<indexing>(),
                "tensor_ptr base should be indexing, but got: " << base);
        auto offset = base.static_as<indexing>()->idx_;
        for (auto &of : offset) {
            COMPILE_ASSERT(of.isa<constant_c>()
                            && get_const_as_int(of.static_as<constant_c>())
                                    == 0,
                    "tensorptr used for tensorview op should have all-zero "
                    "offset")
        }
        COMPILE_ASSERT(base.static_as<indexing>()->ptr_.isa<tensor>(),
                "Nested tensorptr is not expected inside fmgr")
        if (!is_dynamic) {
            auto dims = get_expr_to_dims(base.static_as<indexing>()
                                                 ->ptr_.static_as<tensor>()
                                                 ->dims_);
            auto shape = get_expr_to_dims(buf.static_as<tensorptr>()->shape_);
            COMPILE_ASSERT(get_dims_product(dims) == get_dims_product(shape),
                    "Unexpected reshaped tensor found");
        }
    }
    buffer_ = buf;
}

tensor fusion_data_t::get_real_tensor() const {
    auto buf = buffer_;
    COMPILE_ASSERT(buf.isa<tensor>() || buf.isa<tensorptr>(),
            "Only tensor or tensorptr is accepted")
    if (buf.isa<tensorptr>()) {
        auto base = buf.static_as<tensorptr>()->base_;
        COMPILE_ASSERT(base.isa<indexing>(),
                "tensor_ptr base should be indexing, but got: " << base);
        buf = base.static_as<indexing>()->ptr_;
    }
    COMPILE_ASSERT(buf.isa<tensor>(), "Tensor type is expected")
    return buf.static_as<tensor>();
};

bool fusion_manager::is_allocated_tensor(const tensor &tsr) {
    std::vector<tensor>::iterator tensor_iter
            = std::find_if(allocated_tensors_.begin(), allocated_tensors_.end(),
                    [&tsr](tensor &t) { return t.ptr_same(tsr); });
    return tensor_iter != allocated_tensors_.end();
}

fusion_manager::fusion_manager(fusion_manager &&other)
    : input_op_count_(other.input_op_count_)
    , output_op_count_(other.output_op_count_)
    , alloc_tensor_count_(other.alloc_tensor_count_)
    , graph_(std::move(other.graph_))
    , allocated_tensors_(std::move(other.allocated_tensors_))
    , input_idx_map_(std::move(other.input_idx_map_))
    , output_idx_map_(std::move(other.output_idx_map_)) {}

void fusion_manager::bind_graph(sc_graph_t *graph) {
    COMPILE_ASSERT(graph_.empty(),
            "current graph in fmgr is not empty, could not bind new graph")
    graph_ = copy_graph(*graph);
}

/**
 * This function pre-allocate tensor, whether used indeedly is decided in
 * schedule tensor pass
 * */
expr fusion_manager::allocate_tensor(
        graph_tensor_ptr output, fdata_map &fdmap) {
    auto &fdetail = fdmap.get(output);
    fusible_op_t *fop = output->producer_owner_->dyn_cast<fusible_op_t>();
    tensor tsr;
    bool is_dynamic = graph_.is_dynamic();
    auto allocate_ = [&](const std::string &name, const logical_tensor_t &lt,
                             address_space addrspace = address_space::automatic,
                             const std::shared_ptr<static_data_t> &init_value
                             = nullptr,
                             bool global = false) {
        auto shapes = lt.get_blocking_dims_expr(graph_);
        auto strides = dims_to_dense_stride(shapes);
        tsr = builder::make_stensor(
                name + std::to_string(alloc_tensor_count_++), shapes, strides,
                output->details_.dtype_, addrspace, init_value)
                      .checked_as<tensor>();
        if (global) {
            auto def = builder::make_var_tensor_def_unattached(
                    tsr, linkage::private_global)
                               .static_as<define>();
            global_defines_.emplace_back(def);
        } else {
            allocated_tensors_.emplace_back(tsr);
        }

        fdetail.set_buffer(is_dynamic, tsr);
        return tsr;
    };
    // TODO(xxx): remove this reorder judgement
    if (auto const_op = fop->dyn_cast<constant_op_t>()) {
        auto const_value = const_op->get_constant_values();
        return allocate_("_const_buf_", output->details_,
                address_space::automatic, const_value, true);
    } else {
        return allocate_("_" + fop->op_name_ + "_buf_", output->details_);
    }
}

std::vector<sc_op_ptr> fusion_manager::dispatch_fusion_anchor(
        std::vector<fslice_map> &fsmap_list, const context_ptr &ctx) {
    if (graph_.empty()) { return {}; }
    COMPILE_ASSERT(!sorted_ops_.empty(),
            "sorted ops are expected to be ready, please initilize it first");
    COMPILE_ASSERT(!fanchor_list_.empty(),
            "no output anchor found, please create them before commit");

    infer_status_map_t stat_map;
    for (size_t anchor_id = 0; anchor_id < fanchor_list_.size(); anchor_id++) {
        stat_map.clear();
        // record max anchor fusion manager use.
        max_anchor_ = anchor_id;
        auto &fsmap = fsmap_list[anchor_id];
        do_infer_slice_ranges(fsmap, anchor_id, stat_map);
        if (stat_map.is_fail()) {
            auto &failes_ops
                    = stat_map.get_ops_by_status(infer_status_code::FAIL);
            assert(!failes_ops.empty());
            SC_MODULE_INFO << "Dispatch fusion anchor failed. Anchor id: "
                           << anchor_id << ", Op Name: "
                           << failes_ops.begin()->get()->op_name_;
            return infer_status_map_t::stat_map_to_vector(failes_ops);
        }
        // set anchor
        for (auto &cur : stat_map.get_ops_by_status(infer_status_code::OK)) {
            if (cur->dyn_cast<fusible_op_t>()->anchor_id_ == -1) {
                if (cur->attrs_.has_key(op_attr_key::fused_mode_hint)) {
                    cur->attrs_.remove(op_attr_key::fused_mode_hint);
                }
                cur->dyn_cast<fusible_op_t>()->anchor_id_ = anchor_id;
            }
        }
        if (stat_map.is_ok()) return {};
    }
    auto &retry_ops = stat_map.get_ops_by_status(infer_status_code::RETRY);
    auto &unknown_ops = stat_map.get_ops_by_status(infer_status_code::UNKNOWN);
    std::unordered_set<sc_op_ptr> ret_ops = retry_ops;
    ret_ops.insert(unknown_ops.begin(), unknown_ops.end());
    assert(!ret_ops.empty());

    SC_MODULE_INFO << "Could not find suitable anchor for "
                   << ret_ops.begin()->get()->op_name_ << " to commit in total "
                   << fanchor_list_.size() << " anchors";
    return infer_status_map_t::stat_map_to_vector(ret_ops);
}

void fusion_manager::do_infer_slice_ranges(
        fslice_map &fsmap, int anchor_id, infer_status_map_t &stat_map) {
    if (graph_.empty()) { return; }
    COMPILE_ASSERT(!sorted_ops_.empty(),
            "sorted ops are expected to be ready, please initilize it first");

    // set input slice firstly
    for (auto &cur : sorted_ops_) {
        if (auto input_cur = cur->dyn_cast<input_op>()) {
            auto src = fanchor_list_[anchor_id].anchor_slice_.first;
            int input_idx = get_input_idx(input_cur);
            // set input slice range
            if (input_idx < static_cast<int>(src.size()))
                fsmap.get(cur->get_outputs()[0])
                        = {src[input_idx].get_ranges()};
            stat_map.append_ops_by_status(cur.get(), infer_status_code::OK);
        }
    }
    // To enable pre-op fusion, it is allowed that infer ops list maybe not a
    // topology sequence.
    auto infer_ops_list = op_sorting_visitor_t::sort_by_rules(
            graph_, {op_sorting_visitor_t::sort_rule::preop_fusion});
    for (auto &cur : infer_ops_list) {
        auto fusible_cur = cur->dyn_cast<fusible_op_t>();
        fusible_cur->infer_slice_ranges(fsmap, stat_map);
        if (stat_map.is_fail()) return;
        if (stat_map.is_retry()) {
            auto &retry_list
                    = stat_map.get_ops_by_status(infer_status_code::RETRY);
            if (retry_list.end() != retry_list.find(cur)) { continue; }
        }
        if (stat_map.is_unknown()) {
            auto &unknown_list
                    = stat_map.get_ops_by_status(infer_status_code::UNKNOWN);
            if (unknown_list.end() != unknown_list.find(cur)) { continue; }
        }
        stat_map.append_ops_by_status(cur.get(), infer_status_code::OK);
    }

    auto first_round_unknown_list
            = stat_map.get_ops_by_status(infer_status_code::UNKNOWN);
    for (auto &cur : first_round_unknown_list) {
        // infer slice ranges for those UNKNOWN status fusible ops, which use
        // search_known_slice_ranges() in their infer_slice_ranges() functions
        // (currently only reshape_op_t do not use search_known_slice_ranges).
        auto fusible_cur = cur->dyn_cast<fusible_op_t>();
        COMPILE_ASSERT(!fusible_cur->isa<reshape_op_t>(),
                "Op with unknown status shall not be a reshape op.");
        // remove the inferred status of the current op to avoid redundancy
        stat_map.remove_ops_by_status(cur.get(), infer_status_code::UNKNOWN);
        fusible_cur->infer_slice_ranges(fsmap, stat_map);
        if (stat_map.is_fail()) return;
        if (stat_map.is_retry()) {
            auto &retry_list
                    = stat_map.get_ops_by_status(infer_status_code::RETRY);
            if (retry_list.end() != retry_list.find(cur)) { continue; }
        }
        if (stat_map.is_unknown()) {
            auto &unknown_list
                    = stat_map.get_ops_by_status(infer_status_code::UNKNOWN);
            if (unknown_list.end() != unknown_list.find(cur)) { continue; }
        }
        stat_map.append_ops_by_status(cur.get(), infer_status_code::OK);
    }

    // set output slice lastly
    for (auto &cur : sorted_ops_) {
        if (auto output_cur = cur->dyn_cast<output_op>()) {
            auto dst = fanchor_list_[anchor_id].anchor_slice_.second;
            int output_idx = get_output_idx(output_cur);
            // set output slice range
            if (!dst.empty()) {
                fsmap.get(cur->get_inputs()[0])
                        = {dst[output_idx].get_ranges()};
            } else {
                auto &details = output_cur->get_inputs()[0]->details_;
                std::vector<expr> out_dims
                        = details.get_blocking_dims_expr(graph_);
                slice_range out_ranges;
                for (auto &d : out_dims) {
                    out_ranges.emplace_back(std::make_pair(0, d));
                }
                if (fsmap.get(cur->get_inputs()[0]).empty())
                    fsmap.get(cur->get_inputs()[0]) = {out_ranges};
            }
            stat_map.append_ops_by_status(cur.get(), infer_status_code::OK);
        }
    }
}

void fusion_manager::do_prepare_fusion_data(fdata_map &fdmap) {
    if (graph_.empty()) { return; }
    COMPILE_ASSERT(!sorted_ops_.empty(),
            "sorted ops are expected to be ready, please initilize it first");
    for (auto &cur : sorted_ops_) {
        cur->dyn_cast<fusible_op_t>()->prepare_fusion_data(fdmap);
    }
}

struct buffer_reuse_identity {
    buffer_reuse_identity() = default;
    buffer_reuse_identity(
            const sc_data_type_t &dtype, const std::vector<expr> &shapes)
        : dtype_(dtype), shapes_(shapes) {}
    sc_data_type_t dtype_;
    std::vector<expr> shapes_;
    bool operator==(const buffer_reuse_identity &other) const {
        if (shapes_.size() != other.shapes_.size()) { return false; }
        for (size_t i = 0; i < shapes_.size(); i++) {
            if (!do_cast_and_fold(shapes_[i])
                            ->equals(do_cast_and_fold(other.shapes_[i]))) {
                return false;
            }
        }
        return dtype_ == other.dtype_;
    }
};

// reset first access of buffer reuse hint for special cases like the input of
// tunable op and connection tensors between two anchors.
void reset_buffer_reuse_first_access_hint(
        const expr &tsr, int64_t reset_tick = special_ticks::HINT_IN_LOOP) {
    if (tsr.defined()
            && tsr->attr().has_key(attr_keys::hint_first_access_tick)) {
        tsr->attr().set(attr_keys::hint_first_access_tick, reset_tick);
    }
}

// Set buffer reuse hint(hint_first_access_tick and hint_last_access_tick) in
// tensor attribute for buffer schedule based on graph of fusion manager. When
// hint_first_access_tick and hint_last_access_tick are set, the `tsr` can be
// reused and its lifetime is between [hint_first_access_tick,
// hint_last_access_tick]. first_access of a tensor is defined as tick of tensor
// create. last_access of a tensor is defined as maximum tick of its uses.
// access_has_updated is a boolean, used when tensor_slice_list.size() > 1
void set_buffer_reuse_hint(int64_t &hint_tick, fdata_map &fdmap,
        const sc_op_ptr &node, const expr &tsr,
        bool access_has_updated = false) {
    // tick self add, if node is input op, last access == 1
    if (!access_has_updated) { hint_tick++; }
    // update inp tsrs' last access to maximum last access
    for (auto &in : node->get_inputs()) {
        auto &in_detail = fdmap.get(in);
        if (in_detail.buffer_allocated()) {
            auto in_tsr = in_detail.get_buffer();
            while (in_tsr.isa<tensorptr>()) {
                in_tsr = in_tsr.static_as<tensorptr>()->base_->ptr_;
            }
            in_tsr->attr().set(attr_keys::hint_last_access_tick, hint_tick);
        }
    }
    // skip complex cases, don't do schedule.
    // todo: refactor based on rule.
    if (!(node->isa<input_op>()
                || (node->isa<unary_elementwise_op_t>()
                        && !node->isa<cast_op_t>())
                || node->isa<binary_elementwise_op_t>())) {
        if (tsr.defined()) {
            tsr->attr().set(attr_keys::hint_first_access_tick,
                    special_ticks::HINT_IN_LOOP);
            tsr->attr().set(attr_keys::hint_last_access_tick,
                    special_ticks::HINT_IN_LOOP);
        }
        return;
    }

    if (tsr.defined()) {
        // current tsr's first access == last access
        tsr->attr().set(attr_keys::hint_first_access_tick, hint_tick);
        tsr->attr().set(attr_keys::hint_last_access_tick, hint_tick);
    }
}

// We limit the tensors for buffer reuse in fusion to the set with most identity
// count.
void reset_buffer_hint_by_count(buffer_identity_count &buf_cnt_map) {
    if (buf_cnt_map.size() <= 1) { return; }
    std::vector<expr> *tsr_with_most_count = nullptr;
    std::vector<std::vector<expr> *> all_tsr_vec;
    all_tsr_vec.reserve(buf_cnt_map.size());
    size_t max_count = 0;
    for (auto &it : buf_cnt_map) {
        if (it.second.size() > max_count) {
            max_count = it.second.size();
            tsr_with_most_count = &it.second;
        }
        all_tsr_vec.push_back(&it.second);
    }
    for (auto &vec : all_tsr_vec) {
        if (vec != tsr_with_most_count) {
            for (auto &tsr : *vec) {
                reset_buffer_reuse_first_access_hint(tsr);
            }
        }
    }
}

void do_rd_op_init(expr &tsr, reduce_operator rd_op) {
    switch (rd_op) {
        case reduce_operator::mul:
            tsr.checked_as<tensor>()->init_value_
                    = tensor_node::make_tensor_initializer(1.0f);
            break;
        case reduce_operator::max:
            tsr.checked_as<tensor>()->init_value_
                    = tensor_node::make_tensor_initializer(
                            -std::numeric_limits<float>::infinity());
            break;
        case reduce_operator::min:
            tsr.checked_as<tensor>()->init_value_
                    = tensor_node::make_tensor_initializer(
                            std::numeric_limits<float>::infinity());
            break;
        default:
            tsr.checked_as<tensor>()->init_value_
                    = tensor_node::get_zero_tensor_initializer();
    }
}

void fusion_manager::do_allocate_tensor(fdata_map &fdmap,
        const std::vector<expr> &outs, const std::vector<expr> &inargs) {
    if (graph_.empty()) { return; }
    COMPILE_ASSERT(!sorted_ops_.empty(),
            "sorted ops are expected to be ready, please initilize it first");
    bool is_dynamic = graph_.is_dynamic();
    std::function<void(const sc_op_ptr &cur_node,
            const std::vector<tensor_slice> &src,
            const std::vector<tensor_slice> &dst, int64_t &hint_tick)>
            alloc_func;
    alloc_func = [&](const sc_op_ptr &cur_node,
                         const std::vector<tensor_slice> &src,
                         const std::vector<tensor_slice> &dst,
                         int64_t &hint_tick) {
        auto cur = cur_node->dyn_cast<fusible_op_t>();
        if (auto input_cur = cur->dyn_cast<input_op>()) {
            // if it is the input node
            int input_idx = get_input_idx(input_cur);
            int arg_idx = input_idx - static_cast<int>(src.size());
            // if there is only one input node and only one output node, we need
            // to copy the src tensor to dst tensor, not inplemented
            assert(graph_.ops_.size() > 2);
            // for input node, use the input tensor
            if (arg_idx >= 0) {
                expr tsr;
                // query if inargs is given by outside
                if (!inargs.empty()) {
                    tsr = inargs[arg_idx];
                } else {
                    auto blocking_dims
                            = input_cur->get_outputs()[0]
                                      ->details_.get_blocking_dims_expr(graph_);
                    auto strides = dims_to_dense_stride(blocking_dims);
                    auto arg_tsr = builder::make_stensor(
                            std::string("arg_tsr_") + std::to_string(arg_idx),
                            blocking_dims, strides,
                            input_cur->get_outputs()[0]->details_.dtype_);
                    tsr = arg_tsr;
                }
                fdmap.get(cur->get_outputs()[0]).set_buffer(is_dynamic, tsr);
            } else {
                auto buf = src[input_idx].get_real_tensor();
                // may reuse input buffer, e.g. originalout
                set_buffer_reuse_hint(hint_tick, fdmap, cur_node, buf);
                reset_buffer_reuse_first_access_hint(buf, 0);
                fdmap.get(cur->get_outputs()[0]).set_buffer(is_dynamic, buf);
            }
            return;
        } else if (dynamic_cast<output_op *>(cur)) {
            // if it is the output node

            // if there is only one input node and only one output node, we need
            // to copy the src tensor to dst tensor, not inplemented
            assert(graph_.ops_.size() > 2);
            auto output_cur = static_cast<output_op *>(cur);
            auto &output_cur_in_detail = fdmap.get(output_cur->get_inputs()[0]);
            if (output_cur_in_detail.buffer_allocated()) return;
            int output_idx = get_output_idx(output_cur);

            // for output node, use the output tensor
            if (!dst.empty()) {
                // TODO(xxx): need to consider multi-slice from create_anchor
                // stage
                output_cur_in_detail.set_buffer(
                        is_dynamic, dst[output_idx].get_real_tensor());
            } else {
                expr tsr;
                // query if outs is given by outside
                if (!outs.empty()) {
                    tsr = outs[get_output_idx(output_cur)];
                } else {
                    // Here, it will not be pushed into allocated_tensors_,
                    // because
                    // it will be replaced in final IR generation
                    auto dims
                            = cur->get_inputs()[0]
                                      ->details_.get_blocking_dims_expr(graph_);
                    auto strides = dims_to_dense_stride(dims);
                    auto arg_tsr = builder::make_stensor(
                            std::string("output") + std::to_string(output_idx),
                            dims, strides,
                            cur->get_inputs()[0]->details_.dtype_);
                    tsr = arg_tsr;
                }

                // output will inplace the last buffer, so we need to clear it
                // firstly.
                output_cur_in_detail.set_buffer(is_dynamic, tsr);

                auto owner_op = cur->get_inputs()[0]->producer_owner_;
                if (auto coll_op = owner_op->dyn_cast<reduce_collect_op_t>()) {
                    if (coll_op->is_place_holder_op()) {
                        do_rd_op_init(tsr, coll_op->get_rd_op());
                        fdmap.get(owner_op->get_inputs()[0])
                                .set_buffer(is_dynamic, tsr);
                    }
                } else if (auto comp_op
                        = owner_op->dyn_cast<reduce_compute_op_t>()) {
                    do_rd_op_init(tsr, comp_op->get_rd_op());
                }
            }
            // update tensors' last_access
            set_buffer_reuse_hint(hint_tick, fdmap, cur_node, expr());
            return;
        } else if (dynamic_cast<tensor_view_op_t *>(cur)) {
            auto &cur_in_detail = fdmap.get(cur->get_inputs()[0]);
            COMPILE_ASSERT(
                    !cur->share_gt_with_op<output_op>(cur->get_inputs()[0]),
                    "tensor view op could not share same input buffer with "
                    "output op");
            auto &cur_out_detail = fdmap.get(cur->get_outputs()[0]);
            auto reshape_cur = static_cast<tensor_view_op_t *>(cur);
            cur_out_detail.need_alloc_ = false;
            auto base_tsr = cur_in_detail.get_real_tensor();

            // use tensorptr to represent reshaped tensor
            cur_out_detail.set_buffer(is_dynamic,
                    builder::tensor_ptr(base_tsr,
                            std::vector<expr>(base_tsr->dims_.size(), 0),
                            reshape_cur->get_outputs()[0]
                                    ->details_.get_blocking_dims_expr(graph_)));

            bool access_has_updated = false;
            set_buffer_reuse_hint(hint_tick, fdmap, cur_node,
                    cur_out_detail.need_alloc_ ? cur_out_detail.get_buffer()
                                               : expr(),
                    access_has_updated);
            return;
        } else if (dynamic_cast<constant_op_t *>(cur)) {
            auto &cur_out_detail = fdmap.get(cur->get_outputs()[0]);
            allocate_tensor(cur->get_outputs()[0], fdmap);
            return;
        } else {
            // for every sub-node that the current node can share buffer
            // with
            auto share_map = cur->get_info().tensor_share_info_;
            auto &outputs = cur->get_outputs();
            auto &inputs = cur->get_inputs();
            for (unsigned i = 0; i < outputs.size(); i++) {
                auto &out_i_detail = fdmap.get(outputs[i]);
                // check whether the output is one of the users
                for (auto &user : outputs[i]->uses_) {
                    if (user.second->isa<output_op>()) {
                        // if it is the output node
                        bool access_has_updated = false;
                        set_buffer_reuse_hint(hint_tick, fdmap, cur_node,
                                expr(), access_has_updated);
                        alloc_func(
                                user.second.get_shared(), src, dst, hint_tick);
                        break;
                    }
                }
                if (out_i_detail.buffer_allocated()) continue;

                for (auto j : share_map[i]) {
                    auto &in_j_detail = fdmap.get(inputs[j]);
                    if (in_j_detail.use_count_ == 1
                            && !dynamic_cast<constant_op_t *>(
                                    inputs[j]->producer_owner_)) {
                        if (auto inp = inputs[j]
                                               ->producer_owner_
                                               ->dyn_cast<input_op>()) {
                            if (inp->is_arg_input()
                                    || in_j_detail.get_buffer()
                                               ->attr()
                                               .get_or_else(
                                                       "read_buffer", false))
                                continue;
                        } else if (cur->share_gt_with_op<output_op>(
                                           inputs[j])) {
                            continue;
                        }
                        // if the subnode is used only once, we can reuse
                        // its buffer
                        out_i_detail.need_alloc_ = false;
                        // inplace tsr_slice_list_
                        out_i_detail.set_buffer(
                                is_dynamic, in_j_detail.get_buffer());
                        break;
                    }
                }
                // no node can share buffer with the current node
                if (out_i_detail.need_alloc_) {
                    auto tsr = allocate_tensor(outputs[i], fdmap);
                    if (cur->isa<reduce_compute_op_t>()) {
                        auto cur_op = cur->dyn_cast<reduce_compute_op_t>();
                        do_rd_op_init(tsr, cur_op->get_rd_op());
                    }
                }
                assert(out_i_detail.buffer_allocated());
                // set buffer reuse hint for loop lifetime analysis
                // out_i_detail.tsr_slice_list may be empty when next op is
                // output op
                bool access_has_updated = false;
                set_buffer_reuse_hint(hint_tick, fdmap, cur_node,
                        out_i_detail.need_alloc_ ? out_i_detail.get_buffer()
                                                 : expr(),
                        access_has_updated);
            }
            return;
        }
    };
    // reset hint_tick before next schedule
    // start hint tick is 0, and when encounter an op, tick increases by 1.
    int64_t hint_tick = 0;
    for (auto &cur : sorted_ops_) {
        // allocate tensor stage is not anchor-senstive
        auto src = fanchor_list_[0].anchor_slice_.first;
        auto dst = fanchor_list_[0].anchor_slice_.second;
        alloc_func(cur, src, dst, hint_tick);
    }
    for (auto &op : sorted_ops_) {
        if (dynamic_cast<constant_op_t *>(op.get())) { continue; }
        for (auto &out : op->get_outputs()) {
            assert(fdmap.get(out).buffer_allocated());
        }
    }
}

/** This function will find the nearest parent 'for_loop' node body for tensor
 *  declaration. If no loop found, it will just push statements in outer-most
 *  stmt.
 *  @param anchor: tensor should belong to which anchor
 * */
static stmt get_parent_loop_body(stmt anchor) {
    auto node = std::move(anchor);
    auto parent = get_parent_node(node);
    while (parent.defined()) {
        if (parent.isa<for_loop>()) {
            auto ret = parent.static_as<for_loop>();
            return ret->body_;
        }
        node = parent;
        parent = get_parent_node(parent);
    }
    return node;
}

void fusion_manager::do_declare_tensor(fuse_state_t &fstate) {
    if (graph_.empty()) { return; }
    auto &fdmap = fstate.fdmap_;
    auto &fsmap_list = fstate.fsmap_list_;
    buffer_identity_count buf_cnt_map;
    COMPILE_ASSERT(!sorted_ops_.empty(),
            "sorted ops are expected to be ready, please initilize it first");
    auto get_or_set_buffer_reuse
            = [&](const expr &tsr, std::vector<expr> &shapes) {
                  auto tsr1 = tsr.checked_as<tensor>();
                  sc_data_type_t dtype = tsr1->elem_dtype_;
                  buffer_reuse_identity id {dtype, shapes};
                  auto it = buf_cnt_map.find(id);
                  if (it != buf_cnt_map.end()) {
                      it->second.push_back(tsr);
                  } else {
                      buf_cnt_map[id] = tsr_reuse_vec {tsr};
                  }
              };
    // declare real tensor in fdata, and put it at the beginning of ss
    auto declare_tensor_ = [&](const graph_tensor_ptr &gt_ptr,
                                   std::vector<stmt> &ss, int anchor_id) {
        auto tsr = fdmap.get(gt_ptr).get_real_tensor();
        COMPILE_ASSERT(tsr.isa<tensor>(), "allocated tensor not found");
        if (tsr->attr_ && tsr->attr_->has_key("temp.declared")) return;

        // Only allocated tensor need to be declared
        if (is_allocated_tensor(tsr)) {
            ss.emplace(
                    ss.begin(), builder::make_var_tensor_def_unattached(tsr));
            tsr->attr()["temp.declared"] = true;
        }
    };

    auto set_shrink_info_ = [&](const graph_tensor_ptr &gt_ptr, int anchor_id) {
        // buffer is tensor or tensorptr
        auto &buf = fdmap.get(gt_ptr).get_buffer();
        // automatically skip
        if (buf->attr().has_key(tensor_shrinker_attrs::should_shrink)
                || (buf.isa<tensor>()
                        && !is_allocated_tensor(buf.static_as<tensor>()))) {
            return;
        }
        int shrink_anchor = anchor_id;
        // used for reshaped tensor
        if (buf.isa<tensor>()) {
            shrink_anchor = buf->attr().get_or_else(
                    "temp.reshaped_tensor_anchor", shrink_anchor);
        }
        // if buf is not tensor, only shrink for tensorview output
        else if (buf.isa<tensorptr>()) {
            auto ptr = buf.static_as<tensorptr>()
                               ->base_.checked_as<indexing>()
                               ->ptr_;
            auto &att = ptr->attr();
            if (!att.has_key("temp.reshaped_tensor_anchor")) {
                att["temp.reshaped_tensor_anchor"] = anchor_id;
            } else {
                shrink_anchor = att.get<int>("temp.reshaped_tensor_anchor");
            }
        }

        auto range_list = fsmap_list[shrink_anchor].get(gt_ptr);
        COMPILE_ASSERT(!range_list.empty(), "empty range list found")
        if (range_list.size() != 1) return;
        auto shapes_expr = get_slice_shape(range_list[0]);
        buf->attr()[tensor_shrinker_attrs::should_shrink]
                = tensor_shrinker_t::shrink_info_t {
                        /*base*/ get_slice_idx(range_list[0]),
                        /*shape*/ shapes_expr, stmts()};
        if (buf.isa<tensor>()) { get_or_set_buffer_reuse(buf, shapes_expr); }
    };

    for (int i = static_cast<int>(sorted_ops_.size()) - 1; i >= 0; i--) {
        auto cur_op = sorted_ops_[i]->dyn_cast<fusible_op_t>();
        // TODO(xxx): when constant op support tensor mode, it can be removed
        // here.
        if (cur_op->isa<output_op>() || cur_op->isa<constant_op_t>()) continue;
        int anchor_id = cur_op->anchor_id_;
        if (cur_op->isa<input_op>()) {
            auto &out = cur_op->get_outputs()[0];

            // shrink tensor from the input of fusion manager may also need to
            // move its definition place, we need to provide an anchor to it.

            auto tsr = fdmap.get(out).get_buffer();
            if (tsr->attr().get_or_else(
                        tensor_shrinker_attrs::may_shrink, false)) {
                // search real input anchor
                int real_anchor = anchor_id;
                for (int64_t j = static_cast<int64_t>(sorted_ops_.size()) - 1;
                        j >= 0; j--) {
                    auto tmp_op = sorted_ops_[j]->dyn_cast<fusible_op_t>();
                    if (tmp_op->isa<input_op>() || tmp_op->isa<output_op>()
                            || tmp_op->isa<constant_op_t>())
                        continue;
                    int tmp_anchor = tmp_op->anchor_id_;
                    bool found = false;
                    for (auto &ins : tmp_op->get_inputs()) {
                        if (fdmap.get(ins).get_buffer().ptr_same(tsr)) {
                            real_anchor = tmp_anchor;
                            found = true;
                            break;
                        }
                        if (found) break;
                    }
                    if (found) break;
                    for (auto &out : tmp_op->get_outputs()) {
                        if (fdmap.get(out).get_buffer().ptr_same(tsr)) {
                            real_anchor = tmp_anchor;
                            found = true;
                            break;
                        }
                        if (found) break;
                    }
                    if (found) break;
                }

                // set input tensor shrink info
                auto range_list = fsmap_list[real_anchor].get(out);
                COMPILE_ASSERT(range_list.size() == 1,
                        "Currently only single slice is supported for input op")
                auto range = range_list[0];
                auto shapes_expr = get_slice_shape(range);
                tsr->attr()[tensor_shrinker_attrs::should_shrink]
                        = tensor_shrinker_t::shrink_info_t {
                                /*base*/ get_slice_idx(range),
                                /*shape*/ shapes_expr, stmts()};
                if (tsr.isa<tensor>()) {
                    get_or_set_buffer_reuse(tsr, shapes_expr);
                }
                // set declare info
                auto &decl_anchor_stmt
                        = fanchor_list_.at(real_anchor).anchor_position_;
                stmts decl_body = get_parent_loop_body(decl_anchor_stmt)
                                          .checked_as<stmts>();
                stmts place_holder = builder::make_stmts_unattached({})
                                             .checked_as<stmts>();
                place_holder
                        ->attr()[tensor_shrinker_attrs::tensor_for_placerholder]
                        = std::weak_ptr<expr_base>(tsr.impl);
                decl_body->seq_.emplace(decl_body->seq_.begin(), place_holder);
                auto &shrink_info
                        = tsr->attr_->get<tensor_shrinker_t::shrink_info_t>(
                                tensor_shrinker_attrs::should_shrink);
                shrink_info.move_def_ = place_holder;
                tsr->attr()["temp.declared"] = true;
            }
            continue;
        }

        auto &anchor = fanchor_list_.at(anchor_id).anchor_position_;
        // get decl_body, if anchor_id=0, just put it in
        // beginning of the first anchor. Otherwise, we need to get current
        // anchor nearest parent loop body.
        stmts decl_body = anchor_id == 0
                ? anchor
                : get_parent_loop_body(anchor).checked_as<stmts>();
        // get parent loop sequence, tensor declaration will be placed at the
        // beginning of this sequence.
        auto &ss = decl_body->seq_;
        for (auto &ins : cur_op->get_inputs()) {
            declare_tensor_(ins, ss, anchor_id);
            set_shrink_info_(ins, anchor_id);
        }

        for (auto &out : cur_op->get_outputs()) {
            declare_tensor_(out, ss, anchor_id);
            set_shrink_info_(out, anchor_id);
        }
    }
    reset_buffer_hint_by_count(buf_cnt_map);
    allocated_tensors_.clear();
}

void fusion_manager::do_compute_block(
        const context_ptr &ctx, fuse_state_t &fstate) {
    if (graph_.empty()) { return; }
    COMPILE_ASSERT(!sorted_ops_.empty(),
            "sorted ops are expected to be ready, please initilize it first");

    /** this wrapper function is expected to deal with multi-slice condition
     * TODO(xxx): Theoretically, if one op correctly implement its
     * 'infer_slice_range' function and write 'compute_block' function from
     * the view of tensor_slice, it will be able to deal with multi-slice
     * condition by this 'compute_wrapper'. As the result, although movement op
     * share the common interface to deal with multi-slice, they only can
     * address single slice. (@note: reorder can produce multi slice outputs,
     * but not inputs)
     * */
    auto compute_wrapper_ = [&](const context_ptr &ctx, fusible_op_t *cur,
                                    fdata_map &fdmap, fslice_map &fsmap) {
        auto dst = get_output_tsr_slices_list(cur, fdmap, fsmap);
        auto inputs = get_input_tsr_slices_list(cur, fdmap, fsmap);
        COMPILE_ASSERT(
                !inputs.empty(), "Op " << cur->op_name_ << "has no input");

        auto in_slice_size = inputs[0].size();
        for (auto &in : inputs) {
            COMPILE_ASSERT(in.size() == in_slice_size,
                    "[" << cur->op_name_
                        << "]: inputs slice size should be equal but got "
                        << in.size() << " and " << in_slice_size);
        }

        // Currently, except reorder op, other kinds of op should ensure their
        // output have same slice size with input
        if (!cur->isa<reorder_op_t>()) {
            for (auto &out : dst) {
                COMPILE_ASSERT(out.size() == in_slice_size,
                        "slice size of output "
                        "should be equal to "
                        "Inputs, except for reorder op, but got "
                                << out.size() << " and " << in_slice_size);
            }
        }
        // if cur op is successfully registered by brgemm post fusion, don't do
        // compute block.
        if (brg_fusion_reg_.can_register_next_) {
            // break brgemm fusion
            brg_fusion_reg_.can_register_next_ = false;
            if (auto brg_cur
                    = cur->dyn_cast<op_traits::brgemm_fusion_acceptable_t>()) {
                if (in_slice_size == 1) {
                    std::vector<const tensor_slice *> brg_inputs(inputs.size());
                    std::vector<tensor_slice *> brg_outputs(dst.size());
                    std::transform(inputs.begin(), inputs.end(),
                            brg_inputs.begin(),
                            [](std::vector<tensor_slice> &ins) {
                                return &ins[0];
                            });
                    std::transform(dst.begin(), dst.end(), brg_outputs.begin(),
                            [](std::vector<tensor_slice> &out) {
                                return &out[0];
                            });
                    if (brg_cur->register_brgemm_fusion(ctx, brg_outputs,
                                brg_inputs, brg_fusion_reg_)) {
                        brg_fusion_reg_.can_register_next_ = true;
                        return;
                    }
                }
            }
        }
        // unwrapper tensor slice, for compute_block, it just accpet single
        // tensor_slice
        for (size_t i = 0; i < in_slice_size; i++) {
            std::vector<const tensor_slice *> new_inputs_ptr(inputs.size());
            std::vector<tensor_slice *> new_outputs_ptr(dst.size());
            std::transform(inputs.begin(), inputs.end(), new_inputs_ptr.begin(),
                    [&i](std::vector<tensor_slice> &ins) { return &ins[i]; });

            std::transform(dst.begin(), dst.end(), new_outputs_ptr.begin(),
                    [&i](std::vector<tensor_slice> &out) { return &out[i]; });
            cur->compute_block(ctx, new_outputs_ptr, new_inputs_ptr);
        }
    };

    // check continuous load/store memory access
    auto check_continuous_memory_access
            = [&](const sc_op_ptr &fop, fslice_map &fsmap) -> bool {
        // check whether continuous
        auto is_continuous = [&](const graph_tensor_ptr &gt_ptr) -> bool {
            auto &inner_ranges = fsmap.get(gt_ptr);
            sc_dims buf_dims;
            auto buf = fstate.fdmap_.get(gt_ptr).get_buffer();
            if (buf.isa<tensor>()) {
                buf_dims = get_expr_to_dims(buf.static_as<tensor>()->dims_);
            } else if (buf.isa<tensorptr>()) {
                buf_dims = get_expr_to_dims(buf.static_as<tensorptr>()->shape_);
            }
            auto tsr = fstate.fdmap_.get(gt_ptr).get_real_tensor();
            size_t dtsize
                    = utils::get_sizeof_etype(tsr->elem_dtype_.type_code_);
            static constexpr int page_size = 4096;
            // will not access across pages.
            if (get_dims_product(buf_dims) * dtsize < page_size) return true;
            static constexpr int least_bytes_in_page = 512;
            for (auto &range : inner_ranges) {
                COMPILE_ASSERT(buf_dims.size() == range.size(),
                        "Unmatched slice range and dims size: "
                                << utils::print_vector(buf_dims) << " and "
                                << utils::print_pair_vector(range))
                bool expect_end = false;
                sc_dim continuous_dim = 1;
                for (int j = static_cast<int>(range.size() - 1); j >= 0; j--) {
                    if (!range[j].second.isa<constant>()) return false;
                    sc_dim cur_dim = get_expr_as_int(range[j].second);
                    if (expect_end && cur_dim > 1) {
                        /** In most cases, the inner anchor can
                         * extend more fine-grained fusion ability for
                         * following ops. However, if we take performance
                         * into consideration, the continuous load/store
                         * memory access has more higher proirty
                         * */
                        if (continuous_dim * dtsize < least_bytes_in_page)
                            return false;
                    }
                    if (!expect_end) {
                        continuous_dim *= cur_dim;
                        if (cur_dim == 1 && buf_dims[j] != 1) {
                            expect_end = true;
                        }
                    }
                }
            }
            return true;
        };

        for (auto &ins : fop->get_inputs()) {
            if (!is_continuous(ins)) { return false; }
        }
        for (auto &out : fop->get_outputs()) {
            if (!is_continuous(out)) { return false; }
        }
        return true;
    };

    /** Currently, the new workflow of compute_block can be concluded as below:
     * 1. start from anchor 0, the 0-th op generate its IR as before. but it
     * will generate inner anchor for post ops.
     * 2. from the 1-th to i-th op, it shuold insert its generated IR into inner
     * anchor above. Meanwhile, it should re-infer its slice range.
     * 3. when reach the first op of anchor 1 (j-th), it still use anchor 1
     * 4. from j-th to k-th in anchor 1, repeat 2.
     * 5. repeat 1-4 for following ops in larger anchor (more than 2).
     * */
    auto check_inner_anchor
            = [&](fslice_map &fsmap, const sc_op_ptr &begin_op) -> bool {
        if (graph_.is_dynamic()) { return false; }
        op_dep_matrix_t dep(graph_);
        infer_status_map_t stat_map;
        for (auto &in : graph_.get_input_ops()) {
            if (!in->stc_cast<input_op>()->is_arg_input()
                    && dep.lookup(in, begin_op) != 1) {
                return false;
            }
        }
        int begin_anchor_id = begin_op->dyn_cast<fusible_op_t>()->anchor_id_;
        auto begin_iter
                = std::find(sorted_ops_.begin(), sorted_ops_.end(), begin_op);
        COMPILE_ASSERT(begin_iter != sorted_ops_.end(),
                "begin op is not found in graph")

        for (auto iter = begin_iter + 1; iter != sorted_ops_.end(); iter++) {
            if ((*iter)->isa<input_op>() || (*iter)->isa<constant_op_t>())
                continue;
            if (dep.lookup(begin_op, (*iter)) == 0) return false;
            auto fop = (*iter)->dyn_cast<fusible_op_t>();
            // use output case is not supported for inner anchor
            if (auto reo = fop->dyn_cast<reorder_op_t>()) {
                if (reo->use_output_loop()) return false;
            }
            if (fop->anchor_id_ > begin_anchor_id) break;
            fop->infer_slice_ranges(fsmap, stat_map);
            if (!stat_map.is_ok()) {
                bool can_be_ignored = true;
                // if not ok op is not in current anchor id, it can be ignored
                for (auto &op :
                        stat_map.get_ops_by_status(infer_status_code::FAIL)) {
                    if (op->dyn_cast<fusible_op_t>()->anchor_id_
                            != begin_anchor_id)
                        continue;
                    else {
                        can_be_ignored = false;
                        break;
                    }
                }
                for (auto &op :
                        stat_map.get_ops_by_status(infer_status_code::RETRY)) {
                    if (op->dyn_cast<fusible_op_t>()->anchor_id_
                            != begin_anchor_id)
                        continue;
                    else {
                        can_be_ignored = false;
                        break;
                    }
                }
                if (!can_be_ignored) return false;
            }
            if (fop->isa<output_op>()) {
                auto &inner_ranges = fsmap.get(fop->get_inputs()[0]);
                auto &orig_ranges = fstate.fsmap_list_[fop->anchor_id_].get(
                        fop->get_inputs()[0]);
                if (inner_ranges.size() != orig_ranges.size()) return false;
                for (size_t i = 0; i < inner_ranges.size(); i++) {
                    if (inner_ranges[i].size() != orig_ranges[i].size())
                        return false;
                    for (size_t j = 0; j < inner_ranges[i].size(); j++) {
                        // if not modified, means outside inner anchor
                        bool modified = true;
                        if (inner_ranges[i][j].first.isa<constant>()
                                && orig_ranges[i][j].first.isa<constant>()) {
                            if (get_expr_as_int(inner_ranges[i][j].first)
                                    == get_expr_as_int(
                                            inner_ranges[i][j].first))
                                modified = false;
                        } else if (inner_ranges[i][j].first.ptr_same(
                                           orig_ranges[i][j].first)) {
                            modified = false;
                        }
                        if (modified) {
                            // add output offset
                            if (orig_ranges[i][j].first.isa<constant>()
                                    && get_expr_as_int(orig_ranges[i][j].first)
                                            == 0)
                                continue;
                            inner_ranges[i][j].first = inner_ranges[i][j].first
                                    + orig_ranges[i][j].first;
                        }
                    }
                }
            }
            // performance check
            if (!check_continuous_memory_access((*iter), fsmap)) {
                SC_MODULE_INFO
                        << "Take performance effect into consideration, "
                        << (*iter)->op_name_
                        << "could not be committed into the inner anchor "
                           "created by "
                        << begin_op->op_name_
                        << ". Please check whether its memory access is "
                           "continuous enough";
                return false;
            }
        }
        return true;
    };

    auto copy_inner_slice = [&](fslice_map &orig_fsmap, fslice_map &inner_fsmap,
                                    const sc_op_ptr &begin_op) {
        int begin_anchor_id = begin_op->dyn_cast<fusible_op_t>()->anchor_id_;
        auto begin_iter
                = std::find(sorted_ops_.begin(), sorted_ops_.end(), begin_op);
        COMPILE_ASSERT(begin_iter != sorted_ops_.end(),
                "begin op is not found in graph")

        for (auto iter = begin_iter; iter != sorted_ops_.end(); iter++) {
            if ((*iter)->isa<input_op>() || (*iter)->isa<output_op>()
                    || (*iter)->isa<constant_op_t>())
                continue;
            auto fop = (*iter)->dyn_cast<fusible_op_t>();
            if (fop->anchor_id_ > begin_anchor_id) return;
            // copy inner_fsmap to orig_fsmap for further tensor shrink
            for (auto &ins : fop->get_inputs()) {
                if (!ins->producer_owner_->isa<input_op>()
                        && ins->producer_owner_->dyn_cast<fusible_op_t>()
                                        ->anchor_id_
                                == begin_anchor_id) {
                    orig_fsmap.get(ins) = inner_fsmap.get(ins);
                }
            }
            for (auto &out : fop->get_outputs()) {
                bool can_be_rewrited = true;
                for (auto &user : out->uses_) {
                    if (user.second->dyn_cast<fusible_op_t>()->anchor_id_
                                    != begin_anchor_id
                            || user.second->isa<output_op>()) {
                        can_be_rewrited = false;
                        break;
                    }
                }
                if (can_be_rewrited) orig_fsmap.get(out) = inner_fsmap.get(out);
            }
        }
    };

    bool cross_anchor = false;
    int previous_anchor_id = -1;
    stmts inner_anchor;
    fslice_map inner_fsmap;
    for (auto &cur : sorted_ops_) {
        fusible_op_t *fop = cur->dyn_cast<fusible_op_t>();
        // Here is the op which does not overwrite compute block virtual
        // function
        if (cur->isa<input_op>() || cur->isa<output_op>()
                || cur->isa<constant_op_t>()) {
            continue;
        }
        int anchor_id = fop->anchor_id_;
        cross_anchor = (anchor_id > previous_anchor_id);
        // For any first op those cross anchor, it should try to generate its
        // inner anchor
        if (cross_anchor
                && fstate.fsmap_list_[anchor_id]
                                .get(fop->get_outputs()[0])
                                .size()
                        == 1) {
            fop->attrs_.set(op_attr_key::inner_anchor, fuse_anchor_t());
            // reset
            inner_anchor = stmts();
            inner_fsmap.clear();
        }

        builder::ir_builder_t dummy_builder;
        dummy_builder.push_scope();
        compute_wrapper_(ctx, fop, fstate.fdmap_,
                inner_anchor.defined() ? inner_fsmap
                                       : fstate.fsmap_list_[anchor_id]);
        auto s = dummy_builder.pop_scope().checked_as<stmts>();
        auto anchor_pos = inner_anchor.defined()
                ? inner_anchor
                : fanchor_list_[anchor_id].anchor_position_;
        anchor_pos->seq_.insert(
                anchor_pos->seq_.end(), s->seq_.begin(), s->seq_.end());

        // if new inner anchor was found, redirect anchor position and re-infer
        // following ops' slice range until cross-anchor case
        if (fop->attrs_.has_key(op_attr_key::inner_anchor)) {
            auto fanchor
                    = fop->attrs_.get<fuse_anchor_t>(op_attr_key::inner_anchor);
            inner_anchor = fanchor.anchor_position_;
            // inner anchor created sucessfully
            if (inner_anchor.defined()) {
                COMPILE_ASSERT(fop->get_outputs().size() == 1,
                        "Currently only support single output op")
                auto tsl = fanchor.anchor_slice_.first[0];
                auto &outranges = inner_fsmap.get(fop->get_outputs()[0]);
                outranges = slice_range_list {tsl.get_ranges()};

                if (check_inner_anchor(inner_fsmap, cur)) {
                    SC_MODULE_INFO << cur->op_name_
                                   << " is creating inner anchor\n";
                    // rewrite fsmap for tensor shrink info update
                    copy_inner_slice(
                            fstate.fsmap_list_[anchor_id], inner_fsmap, cur);
                } else {
                    // reset
                    inner_anchor = stmts();
                    inner_fsmap.clear();
                }
            }
        }

        previous_anchor_id = anchor_id;
    }
}

void fusion_manager::create_input_fusion_anchor(
        const std::vector<tensor_slice> &dst,
        const std::vector<tensor_slice> &src) {
    COMPILE_ASSERT(!dst.empty(), "No dst tensor slice is found");

    auto bld = builder::get_current_builder();
    auto s = bld->push_anchor();
    auto fanchor = fuse_anchor_t(s, std::make_pair(src, dst));
    // append to fuse anchor
    input_anchor_list_.emplace_back(std::move(fanchor));
}

void fusion_manager::create_output_fusion_anchor(
        const std::vector<tensor_slice> &src,
        const std::vector<tensor_slice> &dst) {
    COMPILE_ASSERT(!src.empty(), "No src tensor slice is found");

    auto bld = builder::get_current_builder();
    auto s = bld->push_anchor();
    auto fanchor = fuse_anchor_t(s, std::make_pair(src, dst));
    // append to fuse anchor
    fanchor_list_.emplace_back(std::move(fanchor));
}

void fusion_manager::create_output_fusion_anchor(expr iter, expr tsr,
        slice_range_list slice_list, stmt dispatch_helper) {
    auto bld = builder::get_current_builder();
    auto s = bld->push_anchor();
    iter_anchor_list_.emplace_back(iter_fuse_anchor_t(s, std::move(iter),
            std::move(tsr), std::move(slice_list), std::move(dispatch_helper)));
}

void fusion_manager::create_output_fusion_anchor(
        const std::vector<tensor_slice> &src, int group_id) {
    auto iter = grouped_anchor_map_.find(group_id);
    auto bld = builder::get_current_builder();
    auto s = bld->push_anchor();
    if (iter != grouped_anchor_map_.end()) {
        auto &group_anchor = iter->second;
        auto &es_map = group_anchor.expr_slice_map_;
        for (auto &src_i : src) {
            auto tsr = src_i.get_real_tensor();
            auto slice_iter = es_map.find(tsr);
            COMPILE_ASSERT(slice_iter != es_map.end(),
                    "grouped anchor require same tensor for each group")
            auto &sr_list = slice_iter->second;
            sr_list.emplace_back(src_i.get_ranges());
        }
        group_anchor.anchor_position_->seq_.emplace_back(s);
    } else {
        std::unordered_map<expr, slice_range_list> expr_slice_map;
        for (auto &src_i : src) {
            expr_slice_map.insert(std::make_pair(src_i.get_real_tensor(),
                    slice_range_list {src_i.get_ranges()}));
        }
        grouped_anchor_map_.insert(std::make_pair(group_id,
                grouped_fuse_anchor_t(
                        builder::make_stmts_unattached({s}).checked_as<stmts>(),
                        std::move(expr_slice_map))));
    }
}

std::vector<std::pair<stmts, std::unordered_map<expr, slice_range_list>>>
fusion_manager::unpack_src_anchor() {
    std::vector<std::pair<stmts, std::unordered_map<expr, slice_range_list>>>
            anchor_map_list;
    for (auto &anchor : fanchor_list_) {
        std::unordered_map<expr, slice_range_list> anchor_map;
        for (auto &src_tsr_slice : anchor.anchor_slice_.first) {
            anchor_map[src_tsr_slice.get_real_tensor()]
                    = slice_range_list {src_tsr_slice.get_ranges()};
        }
        anchor_map_list.emplace_back(
                std::make_pair(anchor.anchor_position_, anchor_map));
    }
    return anchor_map_list;
}

std::vector<std::pair<stmts, std::unordered_map<expr, slice_range_list>>>
fusion_manager::unpack_dst_anchor() {
    std::vector<std::pair<stmts, std::unordered_map<expr, slice_range_list>>>
            anchor_map_list;
    for (auto &anchor : input_anchor_list_) {
        std::unordered_map<expr, slice_range_list> anchor_map;
        for (auto &dst_tsr_slice : anchor.anchor_slice_.second) {
            anchor_map[dst_tsr_slice.get_real_tensor()]
                    = slice_range_list {dst_tsr_slice.get_ranges()};
        }
        anchor_map_list.emplace_back(
                std::make_pair(anchor.anchor_position_, anchor_map));
    }
    return anchor_map_list;
}

void fusion_manager::clear_anchor() {
    for (auto &fanchor : fanchor_list_) {
        auto anchor = fanchor.anchor_position_;
        stmt parent = get_parent_node(anchor);
        auto &ss_parent = parent.checked_as<stmts>()->seq_;
        // find anchor iter
        std::vector<stmt>::iterator anchor_iter
                = std::find_if(ss_parent.begin(), ss_parent.end(),
                        [anchor](stmt &s) { return s.ptr_same(anchor); });
        COMPILE_ASSERT(anchor_iter != ss_parent.end(),
                "Could not found anchor in current parent stmts");
        // move all stmts out of anchor in avoid of blank bracket
        auto &ss_in_anchor = anchor->seq_;
        ss_parent.insert(anchor_iter, ss_in_anchor.begin(), ss_in_anchor.end());
        // re-find anchor iter
        anchor_iter = std::find_if(ss_parent.begin(), ss_parent.end(),
                [anchor](stmt &s) { return s.ptr_same(anchor); });
        COMPILE_ASSERT(anchor_iter != ss_parent.end(),
                "Could not found anchor in current parent stmts");
        // remove anchor
        ss_parent.erase(anchor_iter);
    }
    // clear anchor status
    fanchor_list_.clear();
}

void fusion_manager::init_sorted_ops() {
    if (graph_.empty()) return;
    // reset
    sorted_ops_.clear();
    // default sorted by dfs_topology_sort
    op_visitor_t::dfs_topology_sort(graph_.ops_.size())
            .visit_graph(graph_, [&](op_visitor_t *vis, const sc_op_ptr &cur) {
                if (cur->isa<input_op>()) {
                    if (get_input_idx(cur.get()) >= static_cast<int>(
                                fanchor_list_[0].anchor_slice_.first.size())) {
                        cur->attrs_.set("temp.arg_input", true);
                    }
                }
                sorted_ops_.emplace_back(cur);
            });
}

void fusion_manager::do_reshedule_anchor(
        std::vector<fslice_map> &fsmap_list, bool use_one_anchor) {
    if (graph_.empty()) { return; }
    COMPILE_ASSERT(!sorted_ops_.empty(),
            "sorted ops are expected to be ready, please initilize it first");

    for (size_t i = 0; i < sorted_ops_.size(); i++) {
        auto cur_op = sorted_ops_[i]->dyn_cast<fusible_op_t>();
        int cur_anchor = cur_op->anchor_id_;
        /**
         * TODO(xxx): For those moememnt ops, they may generate completely
         * different multi-slice in *different anchor*, it is hard to replace
         * their each tensor_slice in tensor_slice_list_ across different
         * anchors. As the workaround, we just turn on use_one_anchor to avoid
         * cross-anchor condition.
         * */
        if (!use_one_anchor && cur_op->isa<movement_op_t>()) {
            for (auto &out : cur_op->get_outputs()) {
                bool is_last_op = true;
                for (auto &user : out->uses_) {
                    if (!user.second->isa<output_op>()) { is_last_op = false; }
                }
                if (is_last_op) continue;
                if (fsmap_list[cur_anchor].get(out).size() > 1) {
                    use_one_anchor = true;
                    break;
                }
                for (auto &user : out->uses_) {
                    if (user.second->isa<output_op>()
                            || user.second->isa<constant_op_t>())
                        continue;
                    int user_anchor_id
                            = user.second->dyn_cast<fusible_op_t>()->anchor_id_;
                    if (user_anchor_id > cur_anchor
                            && fsmap_list[user_anchor_id].get(out).size() > 1) {
                        use_one_anchor = true;
                        break;
                    }
                }
                if (use_one_anchor) break;
            }
        }
    }

    int tmp_anchor = sorted_ops_[0]->dyn_cast<fusible_op_t>()->anchor_id_;
    for (size_t i = 0; i < sorted_ops_.size(); i++) {
        auto cur_op = sorted_ops_[i]->dyn_cast<fusible_op_t>();
        // if use_one_anchor is on, just set max_anchor to each op anchor
        if (use_one_anchor)
            cur_op->anchor_id_ = max_anchor_;
        else {
            if (cur_op->anchor_id_ > tmp_anchor)
                tmp_anchor = cur_op->anchor_id_;
            else if (cur_op->anchor_id_ < tmp_anchor)
                cur_op->anchor_id_ = tmp_anchor;
        }
    }
}

/**
 * (fp32) (fp32)   (fp32 bc_arg) (fp32)    (int 8) (fp32 bc_arg)
 *  Op1    Op2          Op1     Op2           Op1   Op2
 *   \    /  \           \      /              |    /
 *    \  /    \           \    /              Op3  /
 *     Op3     \            Op3                |  /
 *      \      /             |                 Op4
 *       \    /              |                  |
 *        \  /               |                  |
 *         Op4 (fp32)       Op4(fp32)        Op5(fp32)
 *      Case 1              Case 2            Case 3
 * For Case 1, output_op i have dependence with input_op j.
 * For Case 2, output_op i have different dims with input_op j.
 * For Case 3, output_op i have different dtype with input_op j.
 * TODO(xxx): this is just first stage fir inplace propagation pass. During the
 * second stage, all fusible op inside fusion pattern should be visited and
 * queried to ensure whether inplace really occurs.
 * */
std::vector<std::vector<int>> fusion_manager::query_inplace() {
    // auto skip
    if (graph_.empty()) { return {}; }
    auto output_op_list = graph_.get_output_ops();
    for (auto &op : graph_.ops_) {
        if (op->isa<reorder_op_t>() || op->isa<cast_op_t>()) {
            return std::vector<std::vector<int>>(output_op_list.size());
        }
    }
    auto input_op_list = graph_.get_input_ops();
    size_t input_op_size = input_op_list.size();
    std::vector<std::vector<int>> inplace_list;
    // firstly we will check each op in outputs op
    for (size_t i = 0; i < output_op_list.size(); i++) {
        auto cur_out = output_op_list[i]->dyn_cast<output_op>();
        std::vector<int> inp_idx = {};
        for (size_t j = 0; j < input_op_size; j++) {
            auto cur_in = input_op_list[j]->dyn_cast<input_op>();
            bool can_replace = true;
            // Case 1: output_op i have dependence with input_op j.
            if (cur_out->get_inputs()[0] == cur_in->get_outputs()[0]) {
                can_replace = false;
            }
            // Case 2: output_op i have different dims with input_op j.
            else if (cur_out->get_inputs()[0]->details_.get_blocking_dims()
                    != cur_in->get_outputs()[0]->details_.get_blocking_dims()) {
                can_replace = false;
            }
            // Case 3: output_op i have different dtype with input_op j.
            else if (cur_out->get_inputs()[0]->details_.dtype_
                    != cur_in->get_outputs()[0]->details_.dtype_) {
                can_replace = false;
            }
            if (can_replace) inp_idx.emplace_back(j);
        }
        inplace_list.emplace_back(inp_idx);
    }
    return inplace_list;
}

std::vector<sc_op_ptr> fusion_manager::prepare_and_check(
        const context_ptr &ctx, fuse_state_t &fstate) {
    if (graph_.empty()) return {};
    COMPILE_ASSERT(!fanchor_list_.empty(),
            "no output anchor found, please create them firstly");
    fstate.fsmap_list_ = std::vector<fslice_map>(fanchor_list_.size());

    // check all output_anchor_slice have same src size
    size_t common_src_size = fanchor_list_[0].anchor_slice_.first.size();
    for (auto &fanchor : fanchor_list_) {
        COMPILE_ASSERT(fanchor.anchor_slice_.first.size() == common_src_size,
                "all output_anchor_slice should have same src size");
    }

    // Init sorted_ops sequence
    init_sorted_ops();
    // dispatch suitable commit anchor for each fusible op
    auto retry_ops = dispatch_fusion_anchor(fstate.fsmap_list_, ctx);
    return retry_ops;
};

void fusion_manager::commit(const ir_module_ptr &modu, fuse_state_t &fstate,
        const std::vector<expr> &outs, const std::vector<expr> &inargs) {
    if (graph_.empty()) { return; }
    COMPILE_ASSERT(!sorted_ops_.empty(),
            "sorted ops are expected to be ready, please initilize it first");

    // sorted by rules
    sorted_ops_ = op_sorting_visitor_t::sort_by_rules(graph_,
            {op_sorting_visitor_t::sort_rule::same_kind,
                    op_sorting_visitor_t::sort_rule::fusion_anchor});

    // prepare fusion
    do_prepare_fusion_data(fstate.fdmap_);
    // allocate tensor
    do_allocate_tensor(fstate.fdmap_, outs, inargs);
    // reschedule anchor
    do_reshedule_anchor(fstate.fsmap_list_);
    // generate real code in IR
    do_compute_block(modu->ctx_, fstate);
    // define tensor in according position by anchor
    do_declare_tensor(fstate);
    // remove empty anchor in avoid to loop fuse/reorder error
    clear_anchor();
    add_to_module(modu);
}

void fusion_manager::add_to_module(const ir_module_ptr &mod) {
    for (auto &def : global_defines_) {
        mod->add_global_var(def);
    }
}

std::vector<std::vector<tensor_slice>>
fusion_manager::get_input_tsr_slices_list(
        fusible_op_t *op, fdata_map &fdmap, fslice_map &fsmap) const {
    std::vector<std::vector<tensor_slice>> result;
    for (auto &input : op->get_inputs()) {
        std::vector<tensor_slice> tmp;
        auto buf = fdmap.get(input).get_buffer();
        auto range_list = fsmap.get(input);
        COMPILE_ASSERT(!range_list.empty(),
                "empty input range found for " << op->op_name_)
        for (auto range : range_list) {
            auto tsl = tensor_slice(buf, std::move(range));
            tmp.emplace_back(tsl);
        }
        result.emplace_back(tmp);
    }
    return result;
}

std::vector<std::vector<tensor_slice>>
fusion_manager::get_output_tsr_slices_list(
        fusible_op_t *op, fdata_map &fdmap, fslice_map &fsmap) const {
    std::vector<std::vector<tensor_slice>> result;
    for (auto &output : op->get_outputs()) {
        std::vector<tensor_slice> tmp;
        auto buf = fdmap.get(output).get_buffer();
        auto range_list = fsmap.get(output);
        COMPILE_ASSERT(!range_list.empty(),
                "empty output range found for " << op->op_name_)
        for (auto range : range_list) {
            auto tsl = tensor_slice(buf, std::move(range));
            tmp.emplace_back(tsl);
        }
        result.emplace_back(std::move(tmp));
    }
    return result;
}

void fusion_manager::transform_graph(const context_ptr &ctx, bool has_main_op) {
    // we allow the transform of reduce -> (reduce_compute+collect) if the outer
    // loop will not be default
    if (graph_.is_dynamic()
            || (!has_main_op
                    && runtime_config_t::get().get_num_threads() != 1)) {
        return;
    }
    auto old_ops = graph_.ops_;
    for (auto &op : old_ops) {
        if (auto rd = op->dyn_cast<reduce_op_t>()) {
            if (rd->can_split_op()) { rd->split_op(ctx, graph_, 1); }
        }
    }
}

void fusion_manager::reset_brgemm_register_infos() {
    brg_fusion_reg_.reset();
}

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
namespace std {
std::size_t hash<dnnl::impl::graph::gc::buffer_reuse_identity>::operator()(
        const dnnl::impl::graph::gc::buffer_reuse_identity &in) const {
    size_t seed = 0;
    dnnl::impl::graph::gc::hash_combine(seed, in.dtype_);
    dnnl::impl::graph::gc::hash_combine(seed, in.shapes_);
    return seed;
}
} // namespace std
