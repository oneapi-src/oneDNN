/*******************************************************************************
 * Copyright 2021 Intel Corporation
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

#include <memory>
#include <vector>
#include <unordered_map>

#include "interface/c_types_map.hpp"
#include "interface/value.hpp"

#include "backend/dnnl/common.hpp"
#include "backend/dnnl/passes/constant_propagation.hpp"
#include "backend/dnnl/passes/memory_planning.hpp"
#include "backend/dnnl/passes/op_executable.hpp"
#include "backend/dnnl/passes/utils.hpp"

#include "dnnl.hpp"

namespace dnnl {
namespace graph {
namespace impl {
namespace dnnl_impl {
using op_t = impl::op_t;
using op_ptr = std::shared_ptr<impl::op_t>;
using ltw = impl::logical_tensor_wrapper_t;

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

void memory_planner_t::prepare_args_for_conv_and_matmul(op_t *op,
        const dnnl::engine &p_engine, primitive_attr_mgr_t &prm_attr_mgr) {
    exec_args args;

    memory mem;
    size_t index = 0;

    // add input args
    exec_args_set_.find_value_mem_map(op->get_input_value(index++).get(), mem);
    args.insert({DNNL_ARG_SRC, mem});

    exec_args_set_.find_value_mem_map(op->get_input_value(index++).get(), mem);
    args.insert({DNNL_ARG_WEIGHTS, mem});

    if (op->has_attr("with_bias") && op->get_attr<bool>("with_bias")) {
        exec_args_set_.find_value_mem_map(
                op->get_input_value(index++).get(), mem);
        args.insert({DNNL_ARG_BIAS, mem});
    }

    dnnl::primitive_attr prm_attr = op->has_attr("primitive_attr_key")
            ? prm_attr_mgr.get_attr(op->get_attr<int64_t>("primitive_attr_key"))
            : dnnl::primitive_attr();
    dnnl::post_ops pops = prm_attr.get_post_ops();
    for (int i = 0; i < pops.len(); i++) {
        if (pops.kind(i) == dnnl::primitive::kind::sum) {
            exec_args_set_.find_value_mem_map(
                    op->get_input_value(index++).get(), mem);
            args.insert({DNNL_GRAPH_ARG_POST_SRC, mem});
        } else if (pops.kind(i) == dnnl::primitive::kind::binary) {
            exec_args_set_.find_value_mem_map(
                    op->get_input_value(index++).get(), mem);
            args.insert(
                    {DNNL_ARG_ATTR_MULTIPLE_POST_OP(i) | DNNL_ARG_SRC_1, mem});
        } else if (pops.kind(i) == dnnl::primitive::kind::convolution) {
            exec_args_set_.find_value_mem_map(
                    op->get_input_value(index++).get(), mem);
            args.insert({DNNL_ARG_ATTR_POST_OP_DW | DNNL_ARG_WEIGHTS, mem});
        } else {
        }
    }

    // add output args
    exec_args_set_.find_value_mem_map(op->get_output_value(0).get(), mem);
    args.insert({DNNL_ARG_DST, mem});

    if (op->num_outputs() > 1) {
        exec_args_set_.find_value_mem_map(op->get_output_value(1).get(), mem);
        args.insert({DNNL_ARG_SCRATCHPAD, mem});
    }

    exec_args_set_.add_exec_args(args);
}

void memory_planner_t::prepare_args_for_binary(op_t *op,
        const dnnl::engine &p_engine, primitive_attr_mgr_t &prm_attr_mgr) {
    exec_args args;

    memory mem;
    size_t index = 0;

    // add input args
    exec_args_set_.find_value_mem_map(op->get_input_value(index++).get(), mem);
    args.insert({DNNL_ARG_SRC_0, mem});

    exec_args_set_.find_value_mem_map(op->get_input_value(index++).get(), mem);
    args.insert({DNNL_ARG_SRC_1, mem});

    dnnl::primitive_attr prm_attr = op->has_attr("primitive_attr_key")
            ? prm_attr_mgr.get_attr(op->get_attr<int64_t>("primitive_attr_key"))
            : dnnl::primitive_attr();
    dnnl::post_ops pops = prm_attr.get_post_ops();
    for (int i = 0; i < pops.len(); i++) {
        if (pops.kind(i) == dnnl::primitive::kind::sum) {
            exec_args_set_.find_value_mem_map(
                    op->get_input_value(index++).get(), mem);
            args.insert({DNNL_GRAPH_ARG_POST_SRC, mem});
        } else if (pops.kind(i) == dnnl::primitive::kind::binary) {
            exec_args_set_.find_value_mem_map(
                    op->get_input_value(index++).get(), mem);
            args.insert(
                    {DNNL_ARG_ATTR_MULTIPLE_POST_OP(i) | DNNL_ARG_SRC_1, mem});
        } else {
        }
    }

    // add output args
    exec_args_set_.find_value_mem_map(op->get_output_value(0).get(), mem);
    args.insert({DNNL_ARG_DST, mem});

    if (op->num_outputs() > 1) {
        exec_args_set_.find_value_mem_map(op->get_output_value(1).get(), mem);
        args.insert({DNNL_ARG_SCRATCHPAD, mem});
    }

    exec_args_set_.add_exec_args(args);
}

// for single-input-single-output op
void memory_planner_t::prepare_args_for_siso_op(op_t *op,
        const dnnl::engine &p_engine, primitive_attr_mgr_t &prm_attr_mgr,
        bool need_scratchpad, bool need_workspace) {
    exec_args args;

    memory mem;
    size_t index = 0;

    exec_args_set_.find_value_mem_map(op->get_input_value(index++).get(), mem);
    args.insert({DNNL_ARG_FROM, mem});

    dnnl::primitive_attr prm_attr = op->has_attr("primitive_attr_key")
            ? prm_attr_mgr.get_attr(op->get_attr<int64_t>("primitive_attr_key"))
            : dnnl::primitive_attr();
    dnnl::post_ops pops = prm_attr.get_post_ops();
    for (int i = 0; i < pops.len(); i++) {
        if (pops.kind(i) == dnnl::primitive::kind::sum) {
            exec_args_set_.find_value_mem_map(
                    op->get_input_value(index++).get(), mem);
            args.insert({DNNL_GRAPH_ARG_POST_SRC, mem});
        } else if (pops.kind(i) == dnnl::primitive::kind::binary) {
            exec_args_set_.find_value_mem_map(
                    op->get_input_value(index++).get(), mem);
            args.insert(
                    {DNNL_ARG_ATTR_MULTIPLE_POST_OP(i) | DNNL_ARG_SRC_1, mem});
        } else {
        }
    }

    exec_args_set_.find_value_mem_map(op->get_output_value(0).get(), mem);
    args.insert({DNNL_ARG_TO, mem});

    if (need_scratchpad && op->num_outputs() > 1) {
        exec_args_set_.find_value_mem_map(op->get_output_value(1).get(), mem);
        args.insert({DNNL_ARG_SCRATCHPAD, mem});
    }

    if (need_workspace && op->num_outputs() > 2) {
        exec_args_set_.find_value_mem_map(op->get_output_value(2).get(), mem);
        args.insert({DNNL_ARG_WORKSPACE, mem});
    }

    exec_args_set_.add_exec_args(args);
}

// for multiple-inputs-single-output op
void memory_planner_t::prepare_args_for_miso_op(op_t *op,
        const dnnl::engine &p_engine, primitive_attr_mgr_t &prm_attr_mgr) {
    UNUSED(prm_attr_mgr);
    exec_args args;

    memory mem;

    for (int i = 0; i < op->num_inputs(); ++i) {
        exec_args_set_.find_value_mem_map(
                op->get_input_value(static_cast<size_t>(i)).get(), mem);
        args.insert({DNNL_ARG_MULTIPLE_SRC + i, mem});
    }

    exec_args_set_.find_value_mem_map(op->get_output_value(0).get(), mem);
    args.insert({DNNL_ARG_DST, mem});

    exec_args_set_.add_exec_args(args);
}

void memory_planner_t::bind_memory_for_bn_folding(
        op_t *op, const dnnl::engine &p_engine) {
    exec_args args;
    memory mem;

    bool with_bias = op->get_attr<bool>("with_bias");

#define INSERT_ARGS(key, val_offset, direction) \
    exec_args_set_.find_value_mem_map( \
            op->get_##direction##_value(val_offset).get(), mem); \
    args.insert({key, mem});

    // bind input memory
    size_t in_idx = 0;
    INSERT_ARGS(DNNL_ARG_WEIGHTS, in_idx++, input); // weight
    if (with_bias) {
        INSERT_ARGS(DNNL_ARG_BIAS, in_idx++, input); // bias
    }
    INSERT_ARGS(DNNL_ARG_WEIGHTS_1, in_idx++, input); // scale
    INSERT_ARGS(DNNL_ARG_WEIGHTS_2, in_idx++, input); // shift
    INSERT_ARGS(DNNL_ARG_MEAN, in_idx++, input); // mean
    INSERT_ARGS(DNNL_ARG_VARIANCE, in_idx++, input); // variance

    // bind output memory
    size_t out_idx = 0;
    INSERT_ARGS(DNNL_ARG_DST_0, out_idx++, output); // updated weight
    INSERT_ARGS(DNNL_ARG_DST_1, out_idx++, output); // updated bias
    INSERT_ARGS(DNNL_ARG_SCRATCHPAD, out_idx++, output); // scratchpad

#undef INSERT_ARGS
    exec_args_set_.add_exec_args(args);
}

void memory_planner_t::bind_memory_for_conv_bwd_data(op_t *op,
        const dnnl::engine &p_engine, primitive_attr_mgr_t &prm_attr_mgr) {
    memory mem;
    size_t index = 0;
    exec_args args;

    // bind mem for inputs
    exec_args_set_.find_value_mem_map(op->get_input_value(index++).get(), mem);
    args.insert({DNNL_ARG_DIFF_DST, mem});

    exec_args_set_.find_value_mem_map(op->get_input_value(index++).get(), mem);
    args.insert({DNNL_ARG_WEIGHTS, mem});

    // bind mem for outputs
    exec_args_set_.find_value_mem_map(op->get_output_value(0).get(), mem);
    args.insert({DNNL_ARG_DIFF_SRC, mem});

    if (op->num_outputs() > 1) {
        exec_args_set_.find_value_mem_map(op->get_output_value(1).get(), mem);
        args.insert({DNNL_ARG_SCRATCHPAD, mem});
    }

    exec_args_set_.add_exec_args(args);
}

// Assign partition's input edges to user given external inputs buffer. Those
// external inputs buffers may be used by other partition (which is under the
// control of user), so we can't reuse them.
// Note: Because those external inputs buffers may be used by preprocess op, so
// we also find the edges that share the same buffers and assign the same buffer
// to them.
impl::status_t memory_planner_t::assign_external_inputs_buffer(
        const std::vector<op_ptr> &subgraph,
        const std::vector<impl::logical_tensor_t> &inputs) {
    std::queue<std::pair<const value_t *, assign_info_t>> q;

    for (auto &val : impl::graph_t(subgraph).get_input_values()) {
        for (size_t i = 0; i < inputs.size(); i++) {
            if (val->get_logical_tensor().id == inputs[i].id) {
                assign_info_t info(external_input, i);
                buffer_assignments_.insert(std::make_pair(val, info));
                q.push(std::make_pair(val, info));
            }
        }
    }

    // assign alias
    while (!q.empty()) {
        auto val_info = q.front();
        q.pop();
        if (reverse_alias_map_.count(val_info.first)) {
            const value_t *alias = reverse_alias_map_.at(val_info.first);
            buffer_assignments_.insert(std::make_pair(alias, val_info.second));
            q.push(std::make_pair(alias, val_info.second));
        }
    }
    return status::success;
}

// Assign partition's output edges to user given external outputs buffer. Those
// external outputs buffers may contain valid content (for example the inplace
// scenarios, partition's output share same buffer with inputs. This is under
// the control of user, the library can't know this in compilation), so we can't
// reuse them.
// Note: Because those external outputs buffers may be used by preprocess op, so
// we also find the edges that share the same buffers and assign the same buffer
// to them.
impl::status_t memory_planner_t::assign_external_outputs_buffer(
        const std::vector<op_ptr> &subgraph,
        const std::vector<impl::logical_tensor_t> &outputs) {
    std::queue<std::pair<const value_t *, assign_info_t>> q;

    for (auto &val : impl::graph_t(subgraph).get_output_values()) {
        for (size_t i = 0; i < outputs.size(); i++) {
            if (val->get_logical_tensor().id == outputs[i].id) {
                assign_info_t info(external_output, i);
                buffer_assignments_.insert(std::make_pair(val, info));
                q.push(std::make_pair(val, info));
            }
        }
    }

    // assign alias
    while (!q.empty()) {
        auto val_info = q.front();
        q.pop();
        if (alias_map_.count(val_info.first)) {
            const value_t *alias = alias_map_.at(val_info.first);
            buffer_assignments_.insert(std::make_pair(alias, val_info.second));
            q.push(std::make_pair(alias, val_info.second));
        }
    }
    return status::success;
}

// Assign internal constant edges (such as the const weight reorder's output) to
// persistent buffer. Those persistent buffers will be cached to the global
// constant cache, so they can't be reused anymore.
// Note: Not all constant edges' buffer should be cached. We will find the final
// output edges of the constant block (a block of ops who output constant
// tensor), and only cache the constant block's outputs' buffer. Because those
// outputs may be produced by inplace op, so we also find the edges that share
// the same buffers and assign the same buffer to them. This can be regarded as
// a kind of constant folding, with which the cached buffer can be reduced.
impl::status_t memory_planner_t::assign_internal_persistent_buffer(
        const std::vector<op_ptr> &subgraph,
        const std::unordered_map<value_t *, size_t> &edge_ref_count) {
    UNUSED(edge_ref_count);
    std::queue<std::pair<const value_t *, assign_info_t>> q;

    for (auto &val : get_constant_block_output_values(subgraph)) {
        if (buffer_assignments_.count(val)) continue;
        size_t idx = persistent_buffer_assigner_.request(
                make_dnnl_memory_desc(val->get_logical_tensor()).get_size());
        assign_info_t info(internal_persistent, idx);
        buffer_assignments_.insert(std::make_pair(val, info));
        q.push(std::make_pair(val, info));
    }

    // assign their alias
    while (!q.empty()) {
        auto val_info = q.front();
        q.pop();
        if (alias_map_.count(val_info.first)) {
            const value_t *alias = alias_map_.at(val_info.first);
            if (buffer_assignments_.count(alias)) continue;

            buffer_assignments_.insert(std::make_pair(alias, val_info.second));
            q.push(std::make_pair(alias, val_info.second));
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
impl::status_t memory_planner_t::assign_internal_temporary_buffer(
        const std::vector<op_ptr> &subgraph,
        const std::unordered_map<value_t *, size_t> &edge_ref_count) {
    auto func = [&](impl::op_t *op) {
        // Handle inplace outputs
        // TODO(qun) At this moment, we only consider inplace for SISO op. Need
        // to extend to use inplace pair of ops
        if (is_inplace(*op)) {
            value_t *in = op->get_input_value(0).get();
            assign_info_t info = buffer_assignments_.at(in);
            bool reuse_in_buffer = info.kind_ == internal_temporary
                    && (temporary_buffer_ref_count_[info.index_] == 1
                            || is_preprocess_op(*op));
            if (reuse_in_buffer) {
                value_t *out = op->get_output_value(0).get();
                buffer_assignments_.insert(std::make_pair(out, info));
                temporary_buffer_ref_count_[info.index_]
                        += edge_ref_count.at(out);
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
            temporary_buffer_ref_count_[idx] = edge_ref_count.at(out.get());
        }

        // Free inputs
        for (auto &in : op->get_input_values()) {
            assign_info_t info = buffer_assignments_.at(in.get());
            if (info.kind_ != internal_temporary) continue;

            --temporary_buffer_ref_count_[info.index_];
            // if we decrease it to zero, we are ready to release
            if (temporary_buffer_ref_count_[info.index_] == 0) {
                temporary_buffer_assigner_.release(info.index_);
            }
        }

        // Free outputs that have no consumer (such as scratchpad)
        for (auto &out : op->get_output_values()) {
            assign_info_t info = buffer_assignments_.at(out.get());
            if (info.kind_ != internal_temporary) continue;

            auto consumers = out->get_consumers();
            if (consumers.empty()) {
                --temporary_buffer_ref_count_[info.index_];
                temporary_buffer_assigner_.release(info.index_);
            }
        }

        return impl::status::success;
    };

    return impl::topo_order_visit(
            impl::graph_t(subgraph).get_output_ops(), func);
}

impl::status_t memory_planner_t::prepare_execution_args_set(
        const std::vector<op_ptr> &subgraph, const dnnl::engine &p_engine,
        primitive_attr_mgr_t &prm_attr_mgr) {
    // bind memory object to each value
    for (value_t *in : impl::graph_t(subgraph).get_input_values()) {
        exec_args_set_.add_value_mem_map({in,
                make_dnnl_memory(
                        make_dnnl_memory_desc(in->get_logical_tensor()),
                        p_engine, nullptr)});
    }

    status_t ret = impl::topo_order_visit(
            impl::graph_t(subgraph).get_output_ops(), [&](impl::op_t *op) {
                for (auto &out : op->get_output_values()) {
                    exec_args_set_.add_value_mem_map({out.get(),
                            make_dnnl_memory(make_dnnl_memory_desc(
                                                     out->get_logical_tensor()),
                                    p_engine, nullptr)});
                }
                return impl::status::success;
            });
    if (ret != status::success) return ret;

    registrar_t temporary_registrar = temporary_registry_.registrar();
    registrar_t persistent_registrar = persistent_registry_.registrar();

    // classify binded memory objects and their index to buffer
    for (const auto &it : exec_args_set_.get_value_mem_map()) {
        value_t *val = it.first;
        const dnnl::memory &mem = it.second;
        const assign_info_t &info = buffer_assignments_.at(val);
        switch (info.kind_) {
            case external_input:
                exec_args_set_.add_mem_use_external_inputs({mem, info.index_});
                break;
            case external_output:
                exec_args_set_.add_mem_use_external_outputs({mem, info.index_});
                break;
            case internal_temporary:
                temporary_registrar.book(info.index_,
                        temporary_buffer_assigner_.query_size(info.index_));
                exec_args_set_.add_mem_use_internal_temporary(
                        {mem, info.index_});
                break;
            case internal_persistent:
                persistent_registrar.book(info.index_,
                        persistent_buffer_assigner_.query_size(info.index_));
                exec_args_set_.add_mem_use_internal_persistent(
                        {mem, info.index_});
                break;
            default: return status::unknown;
        }
    }

    // Prepare exec args for each op by using binded memories
    // TODO(qun) define each in/output's semantics in op def. Because the
    // semantics should be fixed and a part of IR
    ret = impl::topo_order_visit(
            impl::graph_t(subgraph).get_output_ops(), [&](impl::op_t *op) {
                if (op->get_kind() == impl::op_kind::Convolution
                        || op->get_kind() == op_kind::dnnl_convolution
                        || op->get_kind() == impl::op_kind::MatMul
                        || op->get_kind() == impl::op_kind::ConvTranspose
                        || op->get_kind() == op_kind::dnnl_convtranspose
                        || op->get_kind() == op_kind::conv_depthwise) {
                    prepare_args_for_conv_and_matmul(
                            op, p_engine, prm_attr_mgr);
                } else if (op->get_kind() == impl::op_kind::MaxPool
                        || op->get_kind() == impl::op_kind::AvgPool
                        || op->get_kind() == op_kind::dnnl_pool) {
                    const bool is_training = op->has_attr("is_training")
                            ? op->get_attr<bool>("is_training")
                            : false;
                    prepare_args_for_siso_op(
                            op, p_engine, prm_attr_mgr, true, is_training);
                } else if (is_eltwise_kind(op->get_kind())
                        || op->get_kind() == impl::op_kind::Reorder
                        || op->get_kind() == op_kind::mul_scales
                        || op->get_kind() == op_kind::permute
                        || op->get_kind() == op_kind::to_group
                        || op->get_kind() == op_kind::expand
                        || op->get_kind() == op_kind::squeeze
                        || op->get_kind() == op_kind::dnnl_u8_to_s8
                        || op->get_kind() == impl::op_kind::StaticReshape) {
                    prepare_args_for_siso_op(op, p_engine, prm_attr_mgr);
                } else if (op->get_kind() == op_kind::dnnl_bn_folding) {
                    bind_memory_for_bn_folding(op, p_engine);
                } else if (op->get_kind() == op_kind::dnnl_conv_bwd_data) {
                    bind_memory_for_conv_bwd_data(op, p_engine, prm_attr_mgr);
                } else if (op->get_kind() == op_kind::dnnl_sum) {
                    prepare_args_for_miso_op(op, p_engine, prm_attr_mgr);
                } else if (op->get_kind() == op_kind::dnnl_binary) {
                    prepare_args_for_binary(op, p_engine, prm_attr_mgr);
                } else {
                    assertm(false, "memory planning: unsupported op");
                    return impl::status::compile_fail;
                }
                return impl::status::success;
            });
    if (ret != status::success) return ret;

    return status::success;
}

// In this function, we will do the following things:
// - Build the alias map. both the key and value in the map are edges. the key
//   is the alias of value.
// - Count the reference count of each edges. the reference count will be used
//   during assign temporary buffer to determine which edge's buffer can be
//   reused since it ref count reduce to zero.
// - Assign external user given inputs/outputs buffer to corresponding edges
// - Assign internal allocated temporary buffer to corresponding edges.
// - Assign internal allocated persistent buffer to conresponding edges.
// - Prepare the memory objects which will be used in execution.
impl::status_t memory_planner_t::run(std::shared_ptr<subgraph_t> &sg) {
    status_t ret;

    auto &subgraph = sg->get_mutable_ops();
    auto &prm_attr_mgr = sg->prm_attr_mgr_;
    const auto &p_engine = *(sg->p_engine_);
    const auto &inputs = sg->ins_;
    const auto &outputs = sg->outs_;

    clear(); // clear state to make the method be reentrant

    // find which output is the alias of input
    // TODO(qun) according to op's inplace pair
    for (auto &cur_op : subgraph) {
        if (!is_preprocess_op(*cur_op)) continue;
        value_t *out = cur_op->get_output_value(0).get();
        value_t *in = cur_op->get_input_value(0).get();
        alias_map_.insert({out, in});
        reverse_alias_map_.insert({in, out});
    }

    // get the reference count of each edge
    std::unordered_map<value_t *, size_t> edge_ref_count;
    for (auto &cur_op : subgraph) {
        auto in_vals = cur_op->get_input_values();
        for (auto &val : in_vals) {
            edge_ref_count[val.get()]++;
        }
    }
    for (auto &val : impl::graph_t(subgraph).get_output_values()) {
        edge_ref_count[val]++;
    }

    // Add 1 to subgraph's inputs ref_count since inputs buffer is given by
    // users and can't be reused
    for (auto &val : impl::graph_t(subgraph).get_input_values()) {
        edge_ref_count[val]++;
    }

    // if not enable memory sharing, we plus additional 1 to edge reference
    // count, so that tensors will not be freed and memories will not be reused
    if (!enable_memory_sharing_) {
        for (auto &val_count : edge_ref_count) {
            val_count.second++;
        }
    }

    // Assign subgraph's inputs/outputs and their alias to user given buffers
    ret = assign_external_inputs_buffer(subgraph, inputs);
    if (ret != status::success) return ret;

    ret = assign_external_outputs_buffer(subgraph, outputs);
    if (ret != status::success) return ret;

    // Assign constant block's outputs and their alias to persistent buffer
    // (these buffers will be cached to global constant cache, so should not be
    // reused or freed)
    ret = assign_internal_persistent_buffer(subgraph, edge_ref_count);
    if (ret != status::success) return ret;

    // Assign other edges to internal variable buffers.
    ret = assign_internal_temporary_buffer(subgraph, edge_ref_count);
    if (ret != status::success) return ret;

    // Bind memory object to each value
    ret = prepare_execution_args_set(subgraph, p_engine, prm_attr_mgr);
    if (ret != status::success) return ret;

    return impl::status::success;
}

} // namespace dnnl_impl
} // namespace impl
} // namespace graph
} // namespace dnnl
