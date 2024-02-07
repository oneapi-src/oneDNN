/*******************************************************************************
 * Copyright 2023-2024 Intel Corporation
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

#include <assert.h>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "binary_backward.hpp"
#include "compiler/dimensions.hpp"
#include "compiler/ir/graph/tensor_slice.hpp"
#include <compiler/ir/builder.hpp>
#include <compiler/ir/graph/brgemm_fusion.hpp>
#include <compiler/ir/graph/fusible_op.hpp>
#include <compiler/ir/graph/fusible_op_utils.hpp>
#include <runtime/dynamic_dispatch/ops/impl_type.hpp>
#include <runtime/microkernel/cpu/brgemm_alg_kind.hpp>
#include <unordered_map>
#include <util/utils.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

static std::atomic<int> idx = {0};

std::vector<std::pair<int, std::vector<tensor_inplace_info_t>>>
binary_backward_op_impl_t::get_inplace_map() {
    std::vector<tensor_inplace_info_t> ret;
    auto &inp = get_inputs();
    auto &out_dim = get_outputs()[0]->details_.get_plain_dims();
    for (size_t i = 0; i < inp.size(); i++) {
        if (inp[i]->details_.get_plain_dims() == out_dim) {
            ret.emplace_back(tensor_inplace_info_t {
                    static_cast<int>(i), inplace_kind::ZERO_OFFSET});
        }
    }
    if (ret.empty()) { return {}; }
    return {{0, std::move(ret)}, {1, std::move(ret)}};
}

binary_backward_op_impl_t::binary_backward_op_impl_t(
        const std::vector<graph_tensor_ptr> &ins,
        const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs,
        const binary_backward_operator &backward_opt) {
    info_.inputs_ = ins;
    COMPILE_ASSERT(ins.size() == 3, "Binary backward op shall have 3 inputs.");
    COMPILE_ASSERT(ins[0]->details_.get_plain_dims()
                            == ins[1]->details_.get_plain_dims()
                    && ins[1]->details_.get_plain_dims()
                            == ins[2]->details_.get_plain_dims(),
            "Binary backward op's all inputs should have the same shape ");
    backward_op_type = backward_opt;
    auto &input_1 = info_.inputs_[0]->details_.get_plain_dims();
    auto &input_2 = info_.inputs_[1]->details_.get_plain_dims();
    auto &input_3 = info_.inputs_[2]->details_.get_plain_dims();

    auto &output_shape = input_1;
    auto set_out_shape_format = [this, &output_shape]() {
        info_.outputs_.emplace_back(std::make_shared<graph_tensor>(this));
        size_t sz = info_.outputs_.size();
        auto pos = sz - 1;
        info_.outputs_[pos]->details_.set_plain_dims(output_shape);
        auto output_format = info_.inputs_[0]->details_.get_format();
        info_.outputs_[pos]->details_.set_format(output_format);
        info_.outputs_[pos]->details_.dtype_
                = info_.inputs_[0]->details_.dtype_;
    };
    if (outs.empty()) {
        // out[0]
        set_out_shape_format();
        // out[1]
        set_out_shape_format();
    } else {
        info_.outputs_ = outs;
        COMPILE_ASSERT(
                outs.size() == 2, "Binary backward op shall have 2 outpus.");
    }
    COMPILE_ASSERT(info_.outputs_[0]->details_.get_plain_dims() == output_shape,
            "binary backward op's output shape is not set correctly.");
    attrs_ = attrs;
}

static void compute_binary_backward_vectorized_op(const context_ptr &ctx,
        sc_graph_t &graph, const std::vector<const tensor_slice *> &src,
        const std::vector<tensor_slice *> &dst, sc_op_info_t &info,
        const vectorized_info_t &vx_info,
        const mask_compute_func_t &compute_lanes,
        const mask_compute_func_t &compute_scalar, any_map_t &attrs,
        const graph_tensor_ptr &expand_gt, size_t wkld, bool use_mask) {
    auto *dst_1 = dst[0];
    auto *dst_2 = dst[1];
    // In order to support non-stride test, we add dense_stride flag.
    // If it is non-stride shape, we just use step = 1 to do
    // this.
    int graph_input_size = info.inputs_.size();
    bool dense_stride = std::all_of(info.inputs_.begin(), info.inputs_.end(),
            [](const graph_tensor_ptr &in) { return in->details_.is_dense(); });
    bool is_blocking_shape = is_op_input_blocking_shape(info);
    // nested loop vars
    std::vector<expr> iter_vars;
    // the indices for multiple inputs. First dim: the input, Second dim:
    // the dimemsions in the tensor
    std::vector<std::vector<expr>> src_indices_floor(src.size());
    std::vector<std::vector<expr>> src_indices_tail(src.size());
    // the indices for the output tensor
    std::vector<expr> dst_idx_floor;
    std::vector<expr> dst_idx_tail;
    for (unsigned i = 0; i < dst_1->nslice_dims(); i++) {
        // make the loop var for the for-loop
        iter_vars.emplace_back(range_from_outer_loop(dst_1->get_ranges()[i])
                        ? expr(0)
                        : builder::make_var(datatypes::index,
                                std::string("_fuseiter")
                                        + std::to_string(idx++)));
        // for each input tensor
        for (size_t j = 0; j < src.size(); j++) {
            auto &src_idx_floor = src_indices_floor.at(j);
            auto &src_idx_tail = src_indices_tail.at(j);
            // push an index
            src_idx_floor.emplace_back(iter_vars.back());
            src_idx_tail.emplace_back(iter_vars.back());
        }
        // push an index for output tensor
        dst_idx_floor.emplace_back(iter_vars.back());
        dst_idx_tail.emplace_back(iter_vars.back());
    }
    auto tail_var = builder::make_var(
            datatypes::index, std::string("_fuseiter") + std::to_string(idx++));
    for (size_t j = 0; j < src.size(); j++) {
        auto &src_idx_tail = src_indices_tail.at(j);
        src_idx_tail[vx_info.axis] = tail_var;
    }
    dst_idx_tail[vx_info.axis] = tail_var;
    expr indexed_target_floor_1, indexed_target_floor_2;

    auto bld = builder::get_current_builder();
    bld->push_scope();
    COMPILE_ASSERT(bld, "No active builder is set");
    int lanes = static_cast<int>(vx_info.lanes);
    auto slice_len = dst_1->get_shape().at(vx_info.axis);
    auto floor = do_cast_and_fold(slice_len / lanes * lanes);
    auto tail = do_cast_and_fold(slice_len % lanes);
    int floor_int = 0;
    int tail_int = 0;
    if (floor.isa<constant>()) {
        floor_int = get_expr_as_int(floor);
        tail_int = get_expr_as_int(tail);
        COMPILE_ASSERT((floor_int + tail_int), "Don't support shape len = 0.");
    }
    const int INVALID_AXIS_MASK = -64;
    int last_axis_mask = INVALID_AXIS_MASK;
    std::unordered_map<expr, std::pair<expr, expr>> conditions;
    if (use_mask) {
        compute_mask_and_generate_condition(graph, src,
                info.inputs_[0]->details_.get_plain_dims(),
                info.inputs_[0]->details_.get_format(), iter_vars,
                vx_info.lanes, conditions, last_axis_mask);
    }
    if (last_axis_mask != INVALID_AXIS_MASK && floor_int > 0) {
        COMPILE_ASSERT(tail_int == 0,
                "Currently we only support mask in vectorize compute not "
                "tail.");
    }
    std::vector<stmt_c> tcur;
    stmt cur;
    int loop_size = static_cast<int>(dst_1->get_shape().size());
    bool tail_threshold = tail.isa<constant>() && tail_int <= 1;
    bool use_scalar = tail_threshold || lanes == 1 || !dense_stride;

    // generate inner loop assign value IR
    if (dense_stride && (!floor.isa<constant>() || floor_int)) {
        // if the shape is less than lanes, we don't use mask to
        // process.

        // dst_1
        indexing_from_diff_cond(false, false, *dst_1, dst_idx_floor, lanes,
                indexed_target_floor_1, slice_len, iter_vars.at(vx_info.axis),
                floor);
        // dst_2
        indexing_from_diff_cond(false, false, *dst_2, dst_idx_floor, lanes,
                indexed_target_floor_2, slice_len, iter_vars.at(vx_info.axis),
                floor);
        std::vector<expr> indexed_input_floor;
        expr input_floor_idx;
        for (unsigned j = 0; j < src.size(); j++) {
            indexed_input_floor.emplace_back(
                    indexing_from_diff_cond(false, false, *src.at(j),
                            src_indices_floor.at(j), lanes, input_floor_idx,
                            slice_len, iter_vars.at(vx_info.axis), floor));
        }
        std::vector<expr::lvalue_proxy_t> target_floor
                = {expr::lvalue_proxy_t(indexed_target_floor_1, false),
                        expr::lvalue_proxy_t(indexed_target_floor_2, false)};
        auto cond_it = conditions.find(iter_vars[vx_info.axis]);
        if (cond_it != conditions.end()) {
            assert(last_axis_mask != INVALID_AXIS_MASK);
            cur = compute_lanes(indexed_input_floor, target_floor,
                    cond_it->second.first, cond_it->second.second,
                    vx_info.lanes);
        } else {
            cur = compute_lanes(indexed_input_floor, target_floor);
        }
        cur->attr()[op_traits::workload_computable_t::workload_number] = wkld;
        if (iter_vars.at(vx_info.axis).isa<var>()) {
            cur = make_stmt<for_loop_node_t>(iter_vars.at(vx_info.axis),
                    expr(0), floor, expr(lanes), cur, true, for_type::NORMAL);
            bind_loop_axis(expand_gt, cur, vx_info.axis, true);
        }
        tcur.emplace_back(cur);
    }
    if (((!tail.isa<constant>() && !is_blocking_shape) || tail_int)
            || !dense_stride) {
        std::vector<expr> indexed_input_tail;
        expr mask;
        if (!use_scalar) {
            mask = last_dim_generate_mask(
                    tail_var, floor, slice_len, lanes, true);
        }
        expr indexed_target_tail_1 = builder::make_indexing(
                dst_1->tptr_, dst_idx_tail, use_scalar ? 1 : lanes, mask);
        expr indexed_target_tail_2 = builder::make_indexing(
                dst_2->tptr_, dst_idx_tail, use_scalar ? 1 : lanes, mask);
        for (unsigned j = 0; j < src.size(); j++) {
            indexed_input_tail.emplace_back(builder::make_indexing(
                    src.at(j)->tptr_, src_indices_tail.at(j),
                    use_scalar ? 1 : lanes, mask));
        }
        std::vector<expr::lvalue_proxy_t> target_tail
                = {expr::lvalue_proxy_t(indexed_target_tail_1, false),
                        expr::lvalue_proxy_t(indexed_target_tail_2, false)};
        if (use_scalar) {
            cur = compute_scalar(indexed_input_tail, target_tail);
        } else {
            cur = compute_lanes(indexed_input_tail, target_tail);
        }
        cur->attr()[op_traits::workload_computable_t::workload_number] = wkld;
        cur = make_stmt<for_loop_node_t>(tail_var,
                !dense_stride ? expr(0) : floor, slice_len,
                use_scalar ? expr(1) : expr(lanes), cur, true,
                for_type::NORMAL);
        bind_loop_axis(expand_gt, cur, vx_info.axis, true);
        tcur.emplace_back(cur);
        // create fusible output anchor as demand
        std::vector<int> anchor_pos_in_loop(1);
        anchor_pos_in_loop.emplace_back(vx_info.axis);
    }
    cur = builder::make_stmts_unattached(tcur);
    // recover schedule loop
    for (int i = loop_size - 1; i >= 0; i--) {
        if (i != vx_info.axis) {
            stmt body;
            if (iter_vars.at(i).isa<var>()) {
                body = make_stmt<stmts_node_t>(
                        std::vector<stmt> {std::move(cur)});
                // address special condition, like temp_buffer is used
                cur = make_stmt<for_loop_node_t>(std::move(iter_vars.at(i)),
                        expr(0), dst_1->get_shape().at(i),
                        vx_info.axis == i ? lanes : expr(1), std::move(body),
                        true, for_type::NORMAL);
                bind_loop_axis(expand_gt, cur, i, true);
            }
        }
    }
    cur->attr()[stmt_attr_key::merge_loop] = true;
    bld->emit(cur);
}

void binary_backward_op_impl_t::compute_block(context_ptr ctx,
        const std::vector<tensor_slice *> &dst,
        const std::vector<const tensor_slice *> &inputs) {
    COMPILE_ASSERT(info_.inputs_[0]->details_.get_plain_dims()
                    == info_.outputs_[0]->details_.get_plain_dims(),
            "Wrong op output shapes.\n");
    size_t wkld = compute_fusible_workload(ctx, dst, inputs);
    // set default vectorized information
    vx_info_.axis = dst[0]->get_shape().size() - 1;

    for (int64_t i = dst[0]->nslice_dims() - 1; i >= 0; --i) {
        auto cur_dim = dst[0]->get_shape()[i];
        if (!cur_dim.isa<constant>()
                || get_const_as_int(cur_dim.checked_as<constant>()) > 1) {
            vx_info_.axis = i;
            break;
        }
    }
    vx_info_.lanes
            = vectorize_step(ctx, info_.inputs_[0]->details_.dtype_.type_code_);
    bool use_mask = attrs_.get_or_else(op_attr_key::use_padded_mask, true);
    if (get_owner_graph().is_dynamic()) {
        use_mask &= info_.cur_impl_ != impl_kind_t::no_padding;
    }
    auto func = [&](const std::vector<expr> &in,
                        const std::vector<expr::lvalue_proxy_t> &out) -> stmt {
        auto out_dtype = out[0]->dtype_;
        expr in0, in1, in2;
        in0 = in[0], in1 = in[1], in2 = in[2];
        if (in[0]->dtype_ != out_dtype) {
            in0 = builder::make_cast(out_dtype, in[0]);
        }
        if (in[1]->dtype_ != out_dtype) {
            in1 = builder::make_cast(out_dtype, in[1]);
        }
        if (in[2]->dtype_ != out_dtype) {
            in2 = builder::make_cast(out_dtype, in[2]);
        }

        switch (backward_op_type) {
            case binary_backward_operator::PRELU_BWD: {
                // src diff
                expr res_out0 = builder::make_select(
                        in0 > make_expr<constant_node>(0.f, in0->dtype_), in2,
                        builder::make_mul(in1, in2));
                // weight diff
                expr res_out1 = builder::make_mul(
                        builder::make_min(in0,
                                make_expr<constant_node>(0.f, in0->dtype_)),
                        in2);
                auto assign_out0
                        = builder::make_assign_unattached(out[0], res_out0);
                auto assign_out1
                        = builder::make_assign_unattached(out[1], res_out1);
                return builder::make_stmts_unattached(
                        {assign_out0, assign_out1});
            } break;
            default: {
                COMPILE_ASSERT(false,
                        "Unsupport binary backward op "
                        "found.\n");
                return stmt();
            } break;
        }
        return stmt();
    };
    compute_binary_backward_vectorized_op(ctx, get_owner_graph(), inputs, dst,
            info_, vx_info_, mask_compute_func_t(func),
            mask_compute_func_t(func), attrs_, get_outputs()[0], wkld,
            use_mask);
}

void binary_backward_op_impl_t::query_format(context_ptr ctx,
        std::vector<std::vector<format_stride_pair>> &supported_ins,
        std::vector<std::vector<format_stride_pair>> &supported_outs) {
    std::vector<std::vector<sc_data_format_t>> in_formats, out_formats;
    const auto &in0_format = info_.inputs_[0]->details_.get_format();
    const auto &in1_format = info_.inputs_[1]->details_.get_format();
    const auto &in2_format = info_.inputs_[2]->details_.get_format();

    in_formats.push_back({in0_format});
    in_formats.push_back({in1_format});
    in_formats.push_back({in2_format});
    out_formats.push_back({in0_format});
    out_formats.push_back({in0_format});

    format_to_dense_format_stride_pair(
            in_formats, out_formats, supported_ins, supported_outs);
}

infer_status_code binary_backward_op_impl_t::infer_slice_ranges(
        const context_ptr &ctx, fslice_map &fsmap) {
    COMPILE_ASSERT(get_inputs().size() == 3,
            "Binary backward op is expected 3 inputs");
    // search known ranges from any input of cur fusbile op
    slice_range_map known_ranges_map = search_known_input_slice(this, fsmap);
    if (known_ranges_map.empty()) return infer_status_code::RETRY;
    // double-check all known case
    if (known_ranges_map.size() == get_inputs().size()) {
        auto erase_datamap = [&known_ranges_map, this, &fsmap](
                                     int32_t p1, int32_t p2) {
            if (known_ranges_map[p1].size() != known_ranges_map[p2].size()) {
                // try to align with smaller one and erase bigger one
                int erase_input_id = known_ranges_map[p1].size()
                                < known_ranges_map[p2].size()
                        ? p2
                        : p1;
                known_ranges_map.erase(erase_input_id);
                fsmap.datamap_.erase(get_inputs()[erase_input_id].get());
            }
        };
        erase_datamap(0, 1);
        erase_datamap(1, 2);
    }
    // if unkown slice ranges exist.
    if (known_ranges_map.size() < get_inputs().size()) {
        int32_t iter = get_inputs().size();
        std::vector<int32_t> arr_pos;
        arr_pos.reserve(get_inputs().size());
        int32_t know_pos = 0;
        while (iter--) {
            bool miss = known_ranges_map.find(iter) == known_ranges_map.end();
            if (miss) {
                arr_pos.emplace_back(iter);
            } else {
                know_pos = iter;
            }
        }
        for (auto &x : arr_pos) {
            known_ranges_map[x] = known_ranges_map[know_pos];
        }
        // set the other unknown slice range by achieved known_ranges_list
        set_unknown_input_slice(this, known_ranges_map, fsmap);
    }
    auto &outslice_1 = fsmap.get(get_outputs()[0]);
    auto &outslice_2 = fsmap.get(get_outputs()[1]);

    outslice_1 = known_ranges_map[0];
    outslice_2 = known_ranges_map[0];
    return infer_status_code::OK;
}

void binary_backward_op_impl_t::infer_binding_axis(binding_axis_map &bdax_map) {
    // search known axis from any input of cur fusbile op
    auto known_axis_map = search_known_input_axis(this, bdax_map);
    if (!bdax_map.get(get_outputs()[0]).empty()
            && !bdax_map.get(get_outputs()[1]).empty())
        return;

    if (known_axis_map.size() < get_inputs().size()) {
        int32_t iter = get_inputs().size();
        std::vector<int32_t> arr_pos;
        arr_pos.reserve(get_inputs().size());
        int32_t know_pos = 0;
        while (iter--) {
            bool miss = known_axis_map.find(iter) == known_axis_map.end();
            if (miss) {
                arr_pos.emplace_back(iter);
            } else {
                know_pos = iter;
            }
        }
        for (auto x : arr_pos) {
            known_axis_map[x] = known_axis_map[know_pos];
        }
    }
    if (bdax_map.get(get_outputs()[0]).empty()) {
        bdax_map.get(get_outputs()[0]) = known_axis_map[0];
    }
    if (bdax_map.get(get_outputs()[1]).empty()) {
        bdax_map.get(get_outputs()[1]) = known_axis_map[0];
    }
    // set the other unknown slice range by achieved known_ranges_list
    set_unknown_binding_axis(this, known_axis_map, bdax_map);
}

infer_status_code binary_backward_op_impl_t::pre_infer_slice_ranges(
        const context_ptr &ctx, fslice_map &fsmap) {
    auto &outslice_1 = fsmap.get(get_outputs()[0]);
    auto &outslice_2 = fsmap.get(get_outputs()[1]);
    if (outslice_1.empty() && outslice_2.empty()) {
        return infer_status_code::RETRY;
    }
    for (size_t i = 0; i < get_inputs().size(); i++) {
        auto &input = get_inputs()[i];
        auto &inpslice = fsmap.get(input);
        if (inpslice.empty()) {
            inpslice = outslice_1.empty() ? outslice_2 : outslice_1;
        }
    }
    return infer_status_code::OK;
}

void binary_backward_op_impl_t::pre_infer_binding_axis(
        binding_axis_map &bdax_map) {
    const int first_output = 0, second_output = 1;
    auto &outaxis0 = bdax_map.get(get_outputs()[first_output]);
    auto &outaxis1 = bdax_map.get(get_outputs()[second_output]);
    COMPILE_ASSERT(!outaxis0.empty() || !outaxis1.empty(),
            "Unknown output axis found, could not pre bind axis")
    // just current output user do this
    auto users_infer_bindmap = [this, &bdax_map](const int idx) {
        for (auto &user : get_outputs()[idx]->uses_) {
            if (auto bd_op = user.second->dyn_cast<
                             op_traits::mixed_partition_acceptable>()) {
                bd_op->infer_binding_axis(bdax_map);
            }
        }
    };
    if (outaxis0.empty()) {
        bdax_map.get(get_outputs()[first_output]) = outaxis1;
        users_infer_bindmap(first_output);
    }
    if (outaxis1.empty()) {
        bdax_map.get(get_outputs()[second_output]) = outaxis0;
        users_infer_bindmap(second_output);
    }
    for (size_t i = 0; i < get_inputs().size(); i++) {
        auto &input = get_inputs()[i];
        auto &inpaxis = bdax_map.get(input);
        if (inpaxis.empty()) {
            inpaxis = outaxis0.empty() ? outaxis1 : outaxis0;
            if (auto bd_op = input->producer_owner_->dyn_cast<
                             op_traits::mixed_partition_acceptable>()) {
                bd_op->pre_infer_binding_axis(bdax_map);
            }
        }
    }
}

shape_rl_vec binary_backward_op_impl_t::get_dynamic_shape_relations() const {
    shape_rl_vec ret;
    auto &in0_plain_dims = get_inputs()[0]->details_.get_plain_dims();
    auto &in1_plain_dims = get_inputs()[1]->details_.get_plain_dims();
    auto &in2_plain_dims = get_inputs()[2]->details_.get_plain_dims();
    auto &out_plain_dims_0 = get_outputs()[0]->details_.get_plain_dims();
    auto &out_plain_dims_1 = get_outputs()[1]->details_.get_plain_dims();
    assert(in0_plain_dims.size() == in1_plain_dims.size()
            || in0_plain_dims.size() == 1 || in1_plain_dims.size() == 1);
    if (in0_plain_dims.size() == in1_plain_dims.size()) {
        for (size_t i = 0; i < in0_plain_dims.size(); i++) {
            if ((is_dynamic_dim(in0_plain_dims[i])
                        || is_dynamic_dim(in1_plain_dims[i]))
                    && in0_plain_dims[i] != 1 && in1_plain_dims[i] != 1) {
                ret.emplace_back(in0_plain_dims[i], in1_plain_dims[i]);
            }
        }
    }
    auto add_rls_func = [&in0_plain_dims, &in1_plain_dims, &in2_plain_dims,
                                &ret](const sc_dims &out_plain_dims) {
        auto condition_add = [&ret, &out_plain_dims](
                                     size_t i, const sc_dims &in_plain_dims) {
            if (i < in_plain_dims.size() && in_plain_dims[i] != 1) {
                ret.emplace_back(in_plain_dims[i], out_plain_dims[i]);
            }
        };
        for (size_t i = 0; i < out_plain_dims.size(); i++) {
            if (is_dynamic_dim(out_plain_dims[i])) {
                condition_add(i, in0_plain_dims);
                condition_add(i, in1_plain_dims);
                condition_add(i, in2_plain_dims);
            }
        }
    };
    add_rls_func(out_plain_dims_0);
    add_rls_func(out_plain_dims_1);
    return ret;
}

OP_REGISTER(prelu_bwd_op_t, prelu_bwd)
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
