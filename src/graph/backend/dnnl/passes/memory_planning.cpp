/*******************************************************************************
 * Copyright 2021-2023 Intel Corporation
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
#include <memory>
#include <set>
#include <vector>
#include <unordered_map>
#include <unordered_set>

#include "graph/interface/c_types_map.hpp"
#include "graph/interface/value.hpp"

#include "graph/backend/dnnl/common.hpp"
#include "graph/backend/dnnl/op_executable.hpp"

#include "graph/backend/dnnl/passes/constant_propagation.hpp"
#include "graph/backend/dnnl/passes/memory_planning.hpp"
#include "graph/backend/dnnl/passes/utils.hpp"

#include "oneapi/dnnl/dnnl.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {
using op_t = op_t;
using op_ptr = std::shared_ptr<op_t>;
using ltw = logical_tensor_wrapper_t;

struct op_inplace_pair_t {
    op_inplace_pair_t(size_t in_idx, size_t out_idx)
        : in_idx_(in_idx), out_idx_(out_idx) {}
    const size_t in_idx_; // the index, not id
    const size_t out_idx_;
};

std::vector<op_inplace_pair_t> get_op_inplace_pairs(
        op_t &op, fusion_info_mgr_t &mgr) {
    // TODO(xxx) extend the set
    const static std::set<op_kind_t> ops {op_kind::dnnl_mul_scales,
            op_kind::dnnl_add_zps, op_kind::dnnl_reorder, op_kind::dnnl_binary,
            op_kind::dnnl_eltwise, op_kind::dnnl_softmax,
            op_kind::dnnl_logsoftmax, op_kind::dnnl_softmax_bwd,
            op_kind::dnnl_logsoftmax_bwd};
    std::vector<op_inplace_pair_t> pairs;

    // Make post-sum inplace has higher priority since it affects both
    // performance and memory footprint
    if (op.has_attr(op_attr::fusion_info_key)
            && op.get_attr<int64_t>(op_attr::fusion_info_key) != -1) {
        // sum post ops support inplace
        int64_t key = op.get_attr<int64_t>(op_attr::fusion_info_key);
        const auto &pops = mgr.get_info(key).get_post_ops();

        // the post-ops input offset
        size_t index = 1;
        if (op.get_kind() == op_kind::dnnl_convolution
                || op.get_kind() == op_kind::dnnl_matmul
                || op.get_kind() == op_kind::dnnl_convtranspose) {
            index = op.has_attr(op_attr::with_bias)
                            && op.get_attr<bool>(op_attr::with_bias)
                    ? 3 // src, wei, bias
                    : 2; // src, wei
            if (mgr.get_info(key).with_runtime_scales(true, 0)) { index += 1; }
            if (mgr.get_info(key).with_runtime_scales(true, 1)) { index += 1; }
            if (mgr.get_info(key).with_runtime_zero_points(true, 0)) {
                index += 1;
            }
            if (mgr.get_info(key).with_runtime_zero_points(true, 1)) {
                index += 1;
            }
        } else if (op.get_kind() == op_kind::dnnl_binary) {
            index = 2;
        } else {
            // do nothing
        }

        std::shared_ptr<value_t> post_sum_input;
        for (size_t i = 0; i < pops.size(); i++) {
            if (pops[i]->is_post_sum()) {
                post_sum_input = op.get_input_value(index);
                break; // assume only one post sum
            } else if (pops[i]->get_op()->get_kind() == op_kind::dnnl_binary) {
                index++;
            } else if (pops[i]->get_op()->get_kind()
                    == op_kind::dnnl_convolution) {
                // FIXME(xx) fused conv may have bias
                index++;
            } else {
                // For eltwise post-ops cases. We just do nothing for such
                // cases.
            }
        }

        if (post_sum_input) {
            bool can_inplace = false;
            auto post_sum_input_lt = post_sum_input->get_logical_tensor();
            auto output_lt = op.get_output_value(0)->get_logical_tensor();
            auto post_sum_input_desc = make_dnnl_memory_desc(post_sum_input_lt);
            auto output_desc = make_dnnl_memory_desc(output_lt);
            // allow inplace for conv(u8)+sum(s8)
            if (op.get_kind() == op_kind::dnnl_convolution
                    && post_sum_input_lt.data_type == data_type::s8
                    && output_lt.data_type == data_type::u8) {
                auto format_tag = get_format_tag_str(post_sum_input_desc);
                const auto &dims = post_sum_input_desc.get_dims();
                dnnl_memory_desc_t temp_md;
                dnnl_memory_desc_create_with_string_tag(&temp_md,
                        static_cast<int>(dims.size()), dims.data(),
                        static_cast<dnnl_data_type_t>(output_lt.data_type),
                        format_tag.data());
                can_inplace = output_desc == temp_md;
            } else {
                can_inplace = output_desc == post_sum_input_desc;
            }
            if (can_inplace) { pairs.emplace_back(index, 0); }
        }
    } else if (ops.count(op.get_kind())) {
        auto in0 = op.get_input_value(0)->get_logical_tensor();
        auto out0 = op.get_output_value(0)->get_logical_tensor();
        // always assume in0 and out0 may inplace here, please swap inputs for
        // binary operators to broadcast on src1 and inplace on src0
        const bool can_inplace
                = make_dnnl_memory_desc(in0) == make_dnnl_memory_desc(out0);
        if (can_inplace) { pairs.emplace_back(0, 0); }
    } else if (op.get_kind() == op_kind::dnnl_layernorm_bwd) {
        auto diff_dst = op.get_input_value(1)->get_logical_tensor();
        auto diff_src = op.get_output_value(0)->get_logical_tensor();
        const bool can_inplace = make_dnnl_memory_desc(diff_dst)
                == make_dnnl_memory_desc(diff_src);
        if (can_inplace) { pairs.emplace_back(1, 0); }
    } else {
        // Do nothing
    }

    return pairs;
}

std::shared_ptr<execution_args_set_t> execution_args_set_t::clone() const {
    auto ret = std::make_shared<execution_args_set_t>();

    // clone
    ret->value_mem_map_.reserve(value_mem_map_.size());
    for (auto &val_mem : value_mem_map_) {
        memory cloned_mem(val_mem.second.get_desc(),
                val_mem.second.get_engine(), nullptr);
        ret->value_mem_map_.insert({val_mem.first, cloned_mem});
    }

    auto find_val = [&](const memory &mem) -> value_t * {
        auto pos = std::find_if(value_mem_map_.begin(), value_mem_map_.end(),
                [&](const std::pair<value_t *, memory> &val_mem) {
                    return val_mem.second.get() == mem.get();
                });
        assertm(pos != value_mem_map_.end(), "can't find such mem");
        if (pos != value_mem_map_.end())
            return pos->first;
        else
            return nullptr;
    };

    // copy alias
    ret->mems_use_external_inputs_.reserve(mems_use_external_inputs_.size());
    for (const auto &mem_idx : mems_use_external_inputs_) {
        ret->mems_use_external_inputs_.emplace_back(
                ret->value_mem_map_.at(find_val(mem_idx.first)),
                mem_idx.second);
    }

    ret->mems_use_external_outputs_.reserve(mems_use_external_outputs_.size());
    for (const auto &mem_idx : mems_use_external_outputs_) {
        ret->mems_use_external_outputs_.emplace_back(
                ret->value_mem_map_.at(find_val(mem_idx.first)),
                mem_idx.second);
    }

    ret->mems_use_internal_temporary_.reserve(
            mems_use_internal_temporary_.size());
    for (const auto &mem_offkey : mems_use_internal_temporary_) {
        ret->mems_use_internal_temporary_.emplace_back(
                ret->value_mem_map_.at(find_val(mem_offkey.first)),
                mem_offkey.second);
    }

    ret->mems_use_internal_persistent_.reserve(
            mems_use_internal_persistent_.size());
    for (const auto &mem_offkey : mems_use_internal_persistent_) {
        ret->mems_use_internal_persistent_.emplace_back(
                ret->value_mem_map_.at(find_val(mem_offkey.first)),
                mem_offkey.second);
    }

    ret->topo_ordered_exec_args_.reserve(topo_ordered_exec_args_.size());
    for (const auto &args : topo_ordered_exec_args_) {
        std::unordered_map<int, memory> new_args;
        for (auto &kv : args) {
            int idx = kv.first;
            const memory &mem = kv.second;
            new_args.insert({idx, ret->value_mem_map_.at(find_val(mem))});
        }
        ret->topo_ordered_exec_args_.emplace_back(new_args);
    }

    return ret;
}

void execution_args_set_t::clear() {
    mems_use_external_inputs_.clear();
    mems_use_external_outputs_.clear();
    mems_use_internal_temporary_.clear();
    mems_use_internal_persistent_.clear();
    value_mem_map_.clear();
    topo_ordered_exec_args_.clear();
}

void alias_analyzer_t::clear() {
    alias_map_.clear();
    reverse_alias_map_.clear();
}

status_t alias_analyzer_t::run(std::shared_ptr<subgraph_t> &sg) {
    clear();
    // find alias values
    for (auto &cur_op : sg->get_ops()) {
        if (!is_preprocess_op(*cur_op)) continue;
        value_t *out = cur_op->get_output_value(0).get();
        value_t *in = cur_op->get_input_value(0).get();
        alias_map_.insert({out, in});
        reverse_alias_map_.insert({in, out});
    }
    return status::success;
}

// one input can alias to multiple output
std::vector<const value_t *> alias_analyzer_t::get_alias_outputs(
        const value_t *input) const {
    std::vector<const value_t *> alias_output;
    for (const auto &in_out : reverse_alias_map_) {
        if (in_out.first != input) continue;
        alias_output.emplace_back(in_out.second);
    }
    return alias_output;
}

// a output can alias to only one input
const value_t *alias_analyzer_t::get_alias_input(const value_t *output) const {
    if (alias_map_.count(output)) { return alias_map_.at(output); }
    return nullptr;
}

std::vector<const value_t *> alias_analyzer_t::get_all_aliases(
        const value_t *val) const {
    std::queue<const value_t *> q;
    std::set<const value_t *> visited;

    q.push(val);
    visited.insert(val);
    while (!q.empty()) {
        auto temp = q.front();
        q.pop();
        // visit all alias outputs
        auto alias_outputs = get_alias_outputs(temp);
        for (const auto &alias : alias_outputs) {
            if (visited.count(alias)) continue;
            q.push(alias);
            visited.insert(alias);
        }
        // visit alias input
        auto alias_input = get_alias_input(temp);
        if (alias_input && !visited.count(alias_input)) {
            q.push(alias_input);
            visited.insert(alias_input);
        }
    }

    std::vector<const value_t *> ret;
    ret.reserve(visited.size() - 1);
    for (auto &alias : visited) {
        if (alias == val) continue;
        ret.emplace_back(alias);
    }
    return ret;
}

// Assign partition's input edges to user given external inputs buffer. Those
// external inputs buffers may be used by other partition (which is under the
// control of user), so we can't reuse them.
// Note: Because those external inputs buffers may be used by preprocess op, so
// we also find the edges that share the same buffers and assign the same buffer
// to them.
status_t memory_planner_t::assign_external_inputs_buffer(
        std::shared_ptr<subgraph_t> &sg,
        const std::vector<logical_tensor_t> &inputs) {
    // Remove duplicated input values
    auto sg_ins = sg->get_input_values();
    std::sort(sg_ins.begin(), sg_ins.end());
    sg_ins.erase(std::unique(sg_ins.begin(), sg_ins.end()), sg_ins.end());

    // Assign external input buffer to subgraph's inputs and their alias
    for (auto &val : sg_ins) {
        for (size_t i = 0; i < inputs.size(); i++) {
            if (val->get_logical_tensor().id == inputs[i].id) {
                assign_info_t info(external_input, i);
                buffer_assignments_.insert(std::make_pair(val, info));
                // assign alias
                auto aliases = alias_analyzer_.get_all_aliases(val);
                for (auto &alias : aliases) {
                    assertm(!buffer_assignments_.count(alias),
                            "alias of input has been assigned buffer");
                    buffer_assignments_.insert(std::make_pair(alias, info));
                }
                break;
            }
        }
    }

    // Get the live range of external inputs
    size_t time_point = 0;
    status_t ret;
    ret = topo_order_visit(sg->get_output_ops(), [&](op_t *op) {
        auto in_vals = op->get_input_values();
        for (auto &in_val : in_vals) {
            if (!buffer_assignments_.count(in_val.get())) continue;
            const auto &info = buffer_assignments_.at(in_val.get());
            if (info.kind_ != external_input) continue;
            external_inputs_live_range_[&info] = time_bound_t {0, time_point};
        }
        time_point++;
        return status::success;
    });

    return ret;
}

// Assign partition's output edges to user given external outputs buffer. Those
// external outputs buffers may contain valid content (for example the inplace
// scenarios, partition's output share same buffer with inputs. This is under
// the control of user, the library can't know this in compilation), so we can't
// reuse them.
// Note: Because those external outputs buffers may be used by preprocess op, so
// we also find the edges that share the same buffers and assign the same buffer
// to them.
status_t memory_planner_t::assign_external_outputs_buffer(
        std::shared_ptr<subgraph_t> &sg,
        const std::vector<logical_tensor_t> &outputs, fusion_info_mgr_t &mgr) {
    for (auto &val : sg->get_output_values()) {
        for (size_t i = 0; i < outputs.size(); i++) {
            if (val->get_logical_tensor().id == outputs[i].id) {
                assign_info_t orig_info = buffer_assignments_.at(val);
                assign_info_t updated_info(external_output, i);
                std::queue<const value_t *> q;
                std::set<const value_t *> visited;
                q.push(val);
                while (!q.empty()) {
                    auto cur_val = q.front();
                    q.pop();
                    if (visited.count(cur_val)) continue;

                    // update the assigned buffer to external buffer
                    buffer_assignments_[cur_val] = updated_info;
                    visited.insert(cur_val);

                    // push the alias to queue for next visit
                    auto aliases = alias_analyzer_.get_all_aliases(cur_val);
                    for (const value_t *alias : aliases) {
                        q.push(alias);
                    }

                    // push the inplaced input to queue for next visit
                    auto &producer = cur_val->get_producer();
                    auto op_inplace_pairs = get_op_inplace_pairs(producer, mgr);
                    for (auto &pair : op_inplace_pairs) {
                        if (pair.out_idx_ != cur_val->get_offset()) continue;
                        auto in_val = producer.get_input_value(pair.in_idx_);
                        if (buffer_assignments_.at(in_val.get()) != orig_info)
                            continue;
                        q.push(in_val.get());
                    }
                }
            }
        }
    }
    return status::success;
}

// Assign internal constant edges (such as the const weight reorder's output) to
// persistent buffer. Those persistent buffers will be cached to the global
// constant tensor cache, so they can't be reused anymore.
// Note: Not all constant edges' buffer should be cached. We will find the final
// output edges of the constant block (a block of ops who output constant
// tensor), and only cache the constant block's outputs' buffer. Because those
// outputs may be produced by inplace op, so we also find the edges that share
// the same buffers and assign the same buffer to them. This can be regarded as
// a kind of constant folding, with which the cached buffer can be reduced.
status_t memory_planner_t::assign_internal_persistent_buffer(
        std::shared_ptr<subgraph_t> &sg, fusion_info_mgr_t &mgr) {
    for (auto &val : get_constant_block_output_values(sg)) {
        assign_info_t orig_info = buffer_assignments_.at(val);
        if (orig_info.kind_ != internal_temporary) continue;

        size_t idx = persistent_buffer_assigner_.request(
                make_dnnl_memory_desc(val->get_logical_tensor()).get_size());
        assign_info_t updated_info(internal_persistent, idx);
        std::queue<const value_t *> q;
        std::set<const value_t *> visited;
        q.push(val);
        while (!q.empty()) {
            auto cur_val = q.front();
            q.pop();
            if (visited.count(cur_val) || !cur_val->has_producer()) continue;

            // update the assigned buffer to external buffer
            buffer_assignments_[cur_val] = updated_info;
            visited.insert(cur_val);

            // push the alias to queue for next visit
            auto aliases = alias_analyzer_.get_all_aliases(cur_val);
            for (const value_t *alias : aliases) {
                q.push(alias);
            }

            // push the inplaced input to queue for next visit
            auto &producer = cur_val->get_producer();
            auto op_inplace_pairs = get_op_inplace_pairs(producer, mgr);
            for (auto &pair : op_inplace_pairs) {
                if (pair.out_idx_ != cur_val->get_offset()) continue;
                auto in_val = producer.get_input_value(pair.in_idx_);
                if (buffer_assignments_.at(in_val.get()) != orig_info) continue;
                q.push(in_val.get());
            }
        }
    }
    return status::success;
}

// Assign internal non constant edges (such as src reorder output in conv
// pattern) to temporary buffer. Those temporary buffer will be dynamically
// allocated/freed during execution. In order to reduce memory footprint, we
// introduce two kind of memory optimization:
// - Inplace:  if the op support inplace computation, the output results can be
//   written into input buffer
// - Standard Memory Sharing: if a edge's all consumers have been computed, then
//   the buffer of this edge can be reused by other edge.
// TODO(qun) Consider more situations (for example, a tensor can also be reused
// even if its consumer is not computed, as long as it consumer only need the
// tensor's metadata instead of content)
status_t memory_planner_t::assign_internal_temporary_buffer(
        std::shared_ptr<subgraph_t> &sg,
        const std::unordered_map<value_t *, size_t> &edge_ref_count,
        fusion_info_mgr_t &mgr, bool enable_standard_sharing) {
    std::unordered_map<size_t, size_t> temporary_buffer_ref_count;

    auto func = [&](op_t *op) {
        // Handle alias first
        auto inputs = op->get_input_values();
        for (auto &in : inputs) {
            auto alias_outputs = alias_analyzer_.get_alias_outputs(in.get());
            for (auto &alias : alias_outputs) {
                if (buffer_assignments_.count(alias)) { continue; }
                assign_info_t info = buffer_assignments_.at(in.get());
                buffer_assignments_.insert(std::make_pair(alias, info));
                temporary_buffer_ref_count[info.index_]
                        += edge_ref_count.at(const_cast<value_t *>(alias));
            }
        }

        // Handle inplace
        auto op_inplace_pairs = get_op_inplace_pairs(*op, mgr);
        if (!op_inplace_pairs.empty()) {
            for (const auto &pair : op_inplace_pairs) {
                value_t *in = op->get_input_value(pair.in_idx_).get();
                assign_info_t info = buffer_assignments_.at(in);
                if (info.kind_ != internal_temporary) continue;

                bool reuse_in_buffer
                        = temporary_buffer_ref_count[info.index_] == 1;
                if (reuse_in_buffer) {
                    value_t *out = op->get_output_value(pair.out_idx_).get();
                    if (!buffer_assignments_.count(out)) {
                        buffer_assignments_.insert(std::make_pair(out, info));
                        temporary_buffer_ref_count[info.index_]
                                += edge_ref_count.at(out);
                    }
                }
            }
        }

        // Allocate outputs
        for (auto &out : op->get_output_values()) {
            // already assigned buffer, skip it
            if (buffer_assignments_.count(out.get())) continue;

            // this output need a new buffer, record it
            auto lt = out->get_logical_tensor();
            size_t idx = temporary_buffer_assigner_.request(
                    make_dnnl_memory_desc(lt).get_size());
            buffer_assignments_.insert(std::make_pair(
                    out.get(), assign_info_t(internal_temporary, idx)));
            temporary_buffer_ref_count[idx] = edge_ref_count.at(out.get());
        }

        // Free inputs
        for (auto &in : op->get_input_values()) {
            assign_info_t info = buffer_assignments_.at(in.get());
            if (info.kind_ != internal_temporary) continue;

            --temporary_buffer_ref_count[info.index_];
            // if we decrease it to zero, we are ready to release
            if (enable_standard_sharing
                    && temporary_buffer_ref_count[info.index_] == 0) {
                temporary_buffer_assigner_.release(info.index_);
            }
        }

        // Free outputs that have no consumer (such as scratchpad)
        for (auto &out : op->get_output_values()) {
            assign_info_t info = buffer_assignments_.at(out.get());
            if (info.kind_ != internal_temporary) continue;

            auto consumers = out->get_consumers();
            if (consumers.empty()) {
                --temporary_buffer_ref_count[info.index_];
                if (enable_standard_sharing) {
                    temporary_buffer_assigner_.release(info.index_);
                }
            }
        }

        return status::success;
    };

    return topo_order_visit(sg->get_output_ops(), func);
}

status_t memory_planner_t::prepare_subgraph_inplace_pairs(
        std::shared_ptr<subgraph_t> &sg, bool enable_standard_sharing) {
    size_t time_point = 0;
    status_t ret;
    ret = topo_order_visit(sg->get_output_ops(), [&](op_t *cur_op) {
        auto out_vals = cur_op->get_output_values();
        for (auto &out_val : out_vals) {
            auto out_buf = buffer_assignments_.at(out_val.get());
            if (out_buf.kind_ != external_output) continue;
            logical_tensor_t out_lt = sg->outs_[out_buf.index_];
            logical_tensor_t in_lt = zero_logical_tensor();

            // check if can inplaced sharing external input buffer
            bool inplace_shared = false;
            auto op_inplace_pairs
                    = get_op_inplace_pairs(*cur_op, sg->fusion_info_mgr_);
            for (const auto &pair : op_inplace_pairs) {
                if (pair.out_idx_ != out_val->get_offset()) continue;

                auto in_val = cur_op->get_input_value(pair.in_idx_);
                auto in_buf = buffer_assignments_.at(in_val.get());
                if (in_buf.kind_ != external_input) continue;

                in_lt = sg->ins_[in_buf.index_];
                inplace_shared = true;
                break;
            }

            // check if can standard sharing external input. note: from library
            // side, it's standard sharing, but from FWK side, it's inplace
            // sharing
            bool standard_shared = false;
            if (enable_standard_sharing && !inplace_shared) {
                std::vector<logical_tensor_t> candidates;
                for (auto &ex_in : external_inputs_live_range_) {
                    // external buffer is still in use
                    if (ex_in.second.end_ >= time_point) continue;

                    // different memory size, can't reuse
                    auto in_md = make_dnnl_memory_desc(
                            sg->ins_[ex_in.first->index_]);
                    auto out_md
                            = make_dnnl_memory_desc(sg->outs_[out_buf.index_]);
                    if (in_md.get_size() != out_md.get_size()) continue;

                    candidates.emplace_back(sg->ins_[ex_in.first->index_]);
                }

                // There may be multiple external input buffers that can be
                // shared with the external output buffer. we decided to only
                // report one pair now. To not break existing tests, we prefer
                // to choose the one whose logical tensor id is larger (the
                // post-src in test). We can change this criteria if we have any
                // real cases or requests in the future.
                if (!candidates.empty()) {
                    in_lt = candidates[0];
                    for (auto &tmp : candidates) {
                        if (tmp.id > in_lt.id) { in_lt = tmp; }
                    }
                    standard_shared = true;
                }
            }

            // No sharing
            if (!inplace_shared && !standard_shared) continue;

            // Have shared, not re-do
            bool have_shared = false;
            for (auto &pair : inplace_pairs_) {
                if (pair.output_id == out_lt.id || pair.input_id == in_lt.id)
                    have_shared = true;
            }
            if (have_shared) continue;

            // TODO(qun) we didn't report inplace pair if two lts have different
            // layout type because of frontend users didn't process this
            // situation at this moment. In the future, we need to fix this for
            // more inplace opportunities.
            ltw in_ltw(in_lt), out_ltw(out_lt);
            bool can_share = in_ltw.property_type() != property_type::constant
                    && in_ltw.layout_type() == out_ltw.layout_type();
            if (can_share)
                inplace_pairs_.push_back({in_ltw.id(), out_ltw.id()});
        }
        time_point++;
        return status::success;
    });

    return ret;
}

status_t memory_planner_t::book_buffers(std::shared_ptr<subgraph_t> &sg) {
    // collect all values. Note: here we use vector to ensure that the collected
    // values are in certain order. then we can book buffer from registrar in
    // the certain order. Book buffer in uncertain order may lead to problems,
    // for example:
    //
    // there are 2 persistent buffers: buf0[64], buf1[128] one possible buffer
    // booking order:
    // - book buf0[64] -> offset0 = 0
    // - book buf1[128] -> offset1 = 64
    //
    // another possible buffer booking order:
    // - book buf1[128] -> offset1 = 0
    // - book buf0[64] -> offset0 = 128
    //
    // When two cps (cp0, cp1) compiled from same partition book these two
    // buffers in different order, their persistent memory may have different
    // offsets. In that case, if these two cps share same cached constant
    // tensor, then cp1 will read cached buf0 from the offset 128, but cp0
    // actually cache the buf0 to the offset 0. So, cp1 will read wrong data,
    // and cause accuracy issue.
    std::vector<value_t *> to_be_booked;
    topo_order_visit(sg->get_output_ops(), [&](op_t *op) {
        for (auto &in : op->get_input_values()) {
            to_be_booked.emplace_back(in.get());
        }
        for (auto &out : op->get_output_values()) {
            to_be_booked.emplace_back(out.get());
        }
        return status::success;
    });

    registrar_t temporary_registrar = temporary_registry_.registrar();
    registrar_t persistent_registrar = persistent_registry_.registrar();
    for (const value_t *val : to_be_booked) {
        const assign_info_t &info = buffer_assignments_.at(val);
        switch (info.kind_) {
            // external input and output buffer will be allocated by users, we
            // don't need to book their buffers
            case external_input:
            case external_output: break;
            // book buffers for internal temporary and persistent
            case internal_temporary:
                temporary_registrar.book(info.index_,
                        temporary_buffer_assigner_.query_size(info.index_));
                break;
            case internal_persistent:
                persistent_registrar.book(info.index_,
                        persistent_buffer_assigner_.query_size(info.index_));
                break;
            default: return status::unimplemented;
        }
    }
    return status::success;
}

status_t memory_planner_t::prepare_execution_args_set(
        std::shared_ptr<subgraph_t> &sg, const dnnl::engine &p_engine,
        fusion_info_mgr_t &mgr) {
    status_t ret;

    auto classify_mem = [&, this](const dnnl::memory &mem, const value_t *val) {
        const assign_info_t &info = buffer_assignments_.at(val);
        switch (info.kind_) {
            case external_input:
                exec_args_set_.add_mem_use_external_inputs({mem, info.index_});
                break;
            case external_output:
                exec_args_set_.add_mem_use_external_outputs({mem, info.index_});
                break;
            case internal_temporary:
                exec_args_set_.add_mem_use_internal_temporary(
                        {mem, info.index_});
                break;
            case internal_persistent:
                exec_args_set_.add_mem_use_internal_persistent(
                        {mem, info.index_});
                break;
            default: break;
        }
    };

    // create memory object for each value, and classify the memory objects into
    // different categories
    std::unordered_set<value_t *> prepared;
    ret = topo_order_visit(sg->get_output_ops(), [&](op_t *op) {
        for (auto &in : op->get_input_values()) {
            if (prepared.count(in.get())) continue;
            auto md = make_dnnl_memory_desc(in->get_logical_tensor());
            auto mem = make_dnnl_memory(md, p_engine, nullptr);
            exec_args_set_.add_value_mem_map({in.get(), mem});
            classify_mem(mem, in.get());
            prepared.insert(in.get());
        }

        for (auto &out : op->get_output_values()) {
            auto md = make_dnnl_memory_desc(out->get_logical_tensor());
            auto mem = make_dnnl_memory(md, p_engine, nullptr);
            exec_args_set_.add_value_mem_map({out.get(), mem});
            classify_mem(mem, out.get());
            prepared.insert(out.get());
        }
        return status::success;
    });
    if (ret != status::success) return ret;

    // construct the dnnl execution args for each op
    ret = topo_order_visit(sg->get_output_ops(), [&](op_t *op) {
        const op_schema_t *opm
                = op_schema_registry_t::get_op_schema(op->get_kind());
        if (!opm) {
            assertm(false, "no schema for current op");
            return status::invalid_graph_op;
        }

        if (!opm->has_additional_item("arg_indices_getter")) {
            assertm(false, "no arg indices getter in this op schema");
            return status::invalid_graph_op;
        }

        auto getter = opm->get_additional_item<arg_indices_getter_func>(
                "arg_indices_getter");

        auto arg_indices = getter(op, mgr);

        exec_args dnnl_exec_args;
        for (auto arg_idx : arg_indices) {
            int dnnl_arg = arg_idx.first;
            indices_t::type_t type = arg_idx.second.type_;
            size_t index = arg_idx.second.value_;

            // get the value by index
            value_t *val = type == indices_t::type_t::input
                    ? op->get_input_value(index).get()
                    : op->get_output_value(index).get();

            // find the corresponding memory object
            dnnl::memory mem;
            if (!exec_args_set_.find_value_mem_map(val, mem)) {
                return status::invalid_arguments;
            }

            dnnl_exec_args.insert({dnnl_arg, mem});
        }

        exec_args_set_.add_exec_args(dnnl_exec_args);
        return status::success;
    });

    return ret;
}

// In this function, we will do the following things:
// - Build the alias map. both the key and value in the map are edges. the key
//   is the alias of value.
// - Count the reference count of each edges. the reference count will be used
//   during assign temporary buffer to determine which edge's buffer can be
//   reused since it ref count reduce to zero.
// - Assign external user given inputs/outputs buffer to corresponding edges
// - Assign internal allocated temporary buffer to corresponding edges.
// - Assign internal allocated persistent buffer to corresponding edges.
// - Prepare the memory objects which will be used in execution.
status_t memory_planner_t::run(std::shared_ptr<subgraph_t> &sg) {
    status_t ret;

    auto &mgr = sg->fusion_info_mgr_;
    const auto &p_engine = *(sg->p_engine_);
    const auto &inputs = sg->ins_;
    const auto &outputs = sg->outs_;

    clear(); // clear state to make the method be reentrant

    alias_analyzer_.run(sg);

    // get the reference count of each edge
    std::unordered_map<value_t *, size_t> edge_ref_count;
    for (auto &cur_op : sg->get_ops()) {
        auto in_vals = cur_op->get_input_values();
        for (auto &val : in_vals) {
            edge_ref_count[val.get()]++;
        }
    }
    for (auto &val : sg->get_output_values()) {
        edge_ref_count[val]++;
    }

    // By default, memory reuse is enabled. We can use this internal env
    // var to disable it. The env var is for debugging purpose only and may
    // be removed without any prior notice.
    bool enable_memory_sharing
            = graph::utils::getenv_int_internal("ENABLE_MEM_REUSE", 1) > 0;
    if (!enable_memory_sharing) {
        // if not enable memory sharing, we add additional 1 to edge reference
        // count, so that tensors will not be reused
        for (auto &val_count : edge_ref_count) {
            val_count.second++;
        }
    }

    // Assign external_input buffers to subgraph's inputs and their alias
    ret = assign_external_inputs_buffer(sg, inputs);
    if (ret != status::success) return ret;

    // Assign internal temporary buffer for all other edges
    ret = assign_internal_temporary_buffer(sg, edge_ref_count, mgr, false);
    if (ret != status::success) return ret;

    // Replace some internal temporary buffers to user given external output
    // buffer
    ret = assign_external_outputs_buffer(sg, outputs, mgr);
    if (ret != status::success) return ret;

    // Replace some internal temporary buffers to cached persistent buffer
    ret = assign_internal_persistent_buffer(sg, mgr);
    if (ret != status::success) return ret;

    // Reset the unreplaced internal temporary buffer
    temporary_buffer_assigner_.clear();
    for (auto it = buffer_assignments_.begin();
            it != buffer_assignments_.end();) {
        if (it->second.kind_ == internal_temporary) {
            it = buffer_assignments_.erase(it);
        } else {
            it++;
        }
    }

    // Re-assign internal temporary buffer for reset ones (will re-do memory
    // sharing between temporary buffers)
    ret = assign_internal_temporary_buffer(sg, edge_ref_count, mgr, true);
    if (ret != status::success) return ret;

    // Check which input/output pair of the subgraph can be inplaced
    ret = prepare_subgraph_inplace_pairs(sg, false);
    if (ret != status::success) return ret;

    ret = book_buffers(sg);
    if (ret != status::success) return ret;

    // Bind memory object to each value
    ret = prepare_execution_args_set(sg, p_engine, mgr);
    if (ret != status::success) return ret;

    return status::success;
}

} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl
