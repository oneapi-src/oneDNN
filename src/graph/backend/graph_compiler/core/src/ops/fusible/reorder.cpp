/*******************************************************************************
 * Copyright 2022-2024 Intel Corporation
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
#include <string>
#include <utility>
#include "memory_movement.hpp"
#include "reorder.hpp"
#include <compiler/ir/builder.hpp>
#include <compiler/ir/graph/anchor_loop_generator.hpp>
#include <compiler/ir/graph/dynamic_dispatch_key.hpp>
#include <compiler/ir/graph/fusible_op_utils.hpp>
#include <compiler/ir/graph/mixed_partition.hpp>
#include <compiler/ir/graph/tunable_op.hpp>
#include <compiler/ir/graph/utils.hpp>
#include <compiler/ir/transform/auto_cast.hpp>
#include <compiler/ir/transform/constant_fold.hpp>
#include <runtime/dynamic_dispatch/ops/impl_type.hpp>
#include <unordered_map>
#include <unordered_set>
#include <util/exceptions.hpp>
#include <util/math_utils.hpp>
#include <util/utils.hpp>
namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

SC_MODULE(ops.reorder)

static bool has_output_use(const sc_op *op) {
    for (auto &use : op->get_outputs()[0]->uses_) {
        if (use.second->isa<output_op>()) { return true; }
    }
    return false;
}

static bool is_dynamic_reorder_inplace(sc_op *op, const context_ptr &ctx) {
    COMPILE_ASSERT(
            ctx->machine_.device_type_ == runtime::target_machine_t::type::cpu,
            "Currently support cpu only.");
    return op->get_owner_graph().is_dynamic() && op->isa<reorder_op_t>()
            && !has_output_use(op)
            && op->get_inputs()[0]->details_.get_format()
            == op->get_outputs()[0]->details_.get_format()
            && op->get_inputs()[0]->details_.get_strides()
            == op->get_outputs()[0]->details_.get_strides();
}

// if the reorder is tensor view in dynamic, does not need fusion manager,
// but do inplace itself.
ir_module_ptr inplaced_reorder_get_func(sc_op *op, const context_ptr &ctx) {
    auto modu = std::make_shared<ir_module_t>(ctx);

    std::vector<expr> ins;
    // real_outs are the output tensors in the function arguments
    std::vector<expr> real_outs;
    auto func = graph::create_func_decl_for_op(op, ins, real_outs);
    builder::ir_builder_t bld;
    bld.push_scope();
    bld.push_evaluate(builder::make_write_struct(real_outs[0],
            builder::make_read_struct(ins[0], dyn_tsr_struct_t::name,
                    dyn_tsr_struct_t::fields::data_ptr),
            dyn_tsr_struct_t::name, dyn_tsr_struct_t::fields::data_ptr));
    bld.push_returns(true);
    auto body = bld.pop_scope();
    func->body_ = std::move(body);
    modu->add_func({func});
    modu->set_entry_func_idx(0);
    return modu;
}

ir_module_ptr reorder_op_t::get_func(context_ptr ctx) {
    attrs_.set(op_attr_key::no_fuse, true);
    // if the reorder is tensor view in dynamic, do inplacement.
    if (is_dynamic_reorder_inplace(this, ctx)) {
        return inplaced_reorder_get_func(this, ctx);
    }
    return fusible_op_get_func(this, ctx);
}

void reorder_op_t::query_format(context_ptr ctx,
        std::vector<std::vector<format_stride_pair>> &supported_ins,
        std::vector<std::vector<format_stride_pair>> &supported_outs) {
    // todo: currently reorder from the frontend is contiguous only, so we
    // penetrate it here. For single internal reorder op test, please mark it as
    // "internal".
    supported_ins.push_back(std::vector<format_stride_pair> {
            std::make_pair(info_.inputs_[0]->details_.get_format(),
                    info_.inputs_[0]->details_.get_strides())});

    auto out = info_.outputs_[0]->details_;
    if (!attrs_.get_or_else("internal", false)) {
        out.set_format(info_.inputs_[0]->details_.get_format());
        supported_outs.push_back(std::vector<format_stride_pair> {
                std::make_pair(out.get_format(), out.get_strides())});
    } else {
        supported_outs.push_back(std::vector<format_stride_pair> {
                std::make_pair(out.get_format(), out.get_strides())});
    }
    // when call layout propagation before kernel lower with concrete dispatch
    // key, set break_pre_fuse attr here.
    // reset attrs_ first
    auto &graph = get_owner_graph();
    if (graph.is_dynamic()
            && !graph.attrs_.get_or_else("insert_reorder", true)) {
        attrs_.set(op_attr_key::break_pre_fuse, false);
        attrs_.set(op_attr_key::break_post_fuse, false);
        if (use_output_loop()) {
            attrs_.set(op_attr_key::break_pre_fuse, true);
        } else if (check_padding()) {
            // Use input loop and has padding.
            attrs_.set(op_attr_key::break_post_fuse, true);
        }
    }
}

reorder_op_t::reorder_op_t(const std::vector<graph_tensor_ptr> &ins,
        const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs) {
    for (auto &in : ins) {
        info_.inputs_.emplace_back(in);
    }
    if (outs.empty()) {
        auto plain_dims = info_.inputs_[0]->details_.get_plain_dims();
        auto dtype = info_.inputs_[0]->details_.dtype_;
        auto format = attrs.get<sc_data_format_t>("out_format");
        sc_dims strides;
        if (attrs.has_key("out_stride")) {
            strides = attrs.get<sc_dims>("out_stride");
        }
        info_.outputs_.emplace_back(std::make_shared<graph_tensor>(
                this, format, plain_dims, dtype, strides));
    } else {
        info_.outputs_ = outs;
        gc::graph::check_logical_tensor_shape_dtype_identical(
                info_.inputs_[0]->details_, info_.outputs_[0]->details_);
    }
    op_name_ = "reorder";
    attrs_ = attrs;
    plain_dims_ = ins[0]->details_.get_plain_dims();
    COMPILE_ASSERT(info_.inputs_[0]->details_.get_format().is_convertible(
                           info_.outputs_[0]->details_.get_format()),
            "input format " << info_.inputs_[0]->details_.get_format()
                            << " can not convert to "
                            << info_.outputs_[0]->details_.get_format() << ".");

    update_fuse_attr();
    // currently we don't fuse reorder in dynamic as it should query next op.
    if (is_dynamic()) {
        if (info_.inputs_[0]->details_.get_format().is_blocking()
                && info_.outputs_[0]->details_.get_format().is_blocking()) {
            attrs_.set(op_attr_key::no_fuse, true);
        }
    }
}

reorder_op_t::reorder_op_t(graph_tensor_ptr v, sc_data_format_t input_format,
        sc_data_format_t output_format)
    : reorder_op_t(
            {std::move(v)}, {}, any_map_t {{"out_format", output_format}}) {}

// Update fuse attr of reorder in time once format is modified.
void reorder_op_t::update_fuse_attr() {
    // For safety, we only deal with static senario temporarily
    if (!is_dynamic()) {
        // reset attr firstly
        attrs_.set(op_attr_key::break_pre_fuse, false);
        if (use_output_loop()) {
            attrs_.set(op_attr_key::break_pre_fuse, true);
        }
    }
}

// This function will try to merge multi slice range
static void merge_multi_slice(slice_range &src_slice) {
    bool all_const_start = true;
    for (auto &r : src_slice) {
        if (!r.first.isa<constant_c>()) {
            all_const_start = false;
            break;
        }
    }
    if (all_const_start) {
        std::sort(src_slice.begin(), src_slice.end(),
                [](const std::pair<expr, expr> &a,
                        const std::pair<expr, expr> &b) {
                    return get_const_as_int(a.first.checked_as<constant_c>())
                            < get_const_as_int(
                                    b.first.checked_as<constant_c>());
                });
    }
    slice_range dst_slice;
    std::pair<int, int> temp_range = {0, 0};
    for (auto &r : src_slice) {
        if (r.first.isa<constant_c>() && r.second.isa<constant_c>()) {
            int start = get_const_as_int(r.first.checked_as<constant_c>());
            int length = get_const_as_int(r.second.checked_as<constant_c>());
            // concat
            if (temp_range.second == 0
                    || start == temp_range.first + temp_range.second) {
                if (temp_range.second == 0) temp_range.first = start;
                temp_range.second += length;
            }
            // push and reset
            else {
                dst_slice.emplace_back(temp_range);
                temp_range.first = start;
                temp_range.second = length;
            }
        } else {
            if (temp_range.second > 0) {
                dst_slice.emplace_back(temp_range);
                temp_range.second = 0;
            }
            dst_slice.emplace_back(std::move(r));
        }
    }
    if (temp_range.second > 0) {
        dst_slice.emplace_back(temp_range);
        temp_range.second = 0;
    }
    src_slice = std::move(dst_slice);
}

// Get block to plain ranges
slice_range get_block2plain_ranges(const expr &block_num_start,
        const expr &block_num_length, const expr &block_size_start,
        const expr &block_size_length, int blocks) {
    auto folded_block_size_length
            = constant_folder_t()(auto_caster_t()(block_size_length));
    COMPILE_ASSERT(folded_block_size_length.isa<constant_c>(),
            "constant length is expected, but got "
                    << folded_block_size_length);

    int block_size_length_int = get_const_as_int(
            folded_block_size_length.checked_as<constant_c>());

    std::vector<std::pair<expr, expr>> plain_range_list;
    if (block_size_length_int == blocks) {
        // when block size is equal to blocks, reorder will generate
        // consequent slice in output
        auto plain_range = std::make_pair(
                do_cast_and_fold(block_num_start * blocks + block_size_start),
                do_cast_and_fold(block_num_length * block_size_length_int));
        plain_range_list = {plain_range};
    } else {
        if (!block_num_length.isa<constant>()) { return plain_range_list; }
        // multi plain ranges
        int block_num_length_int
                = get_const_as_int(block_num_length.checked_as<constant_c>());
        for (int i = 0; i < block_num_length_int; i++) {
            auto plain_range = std::make_pair(
                    do_cast_and_fold((block_num_start + expr(i)) * blocks
                            + block_size_start),
                    block_size_length);
            plain_range_list.emplace_back(plain_range);
        }
    }
    // try to merge multi slice
    merge_multi_slice(plain_range_list);
    return plain_range_list;
}

// Get plain to block ranges
std::vector<std::pair<std::pair<expr, expr>, std::pair<expr, expr>>>
get_plain2block_ranges(const expr &start, const expr &length, int blocks) {
    std::vector<std::pair<std::pair<expr, expr>, std::pair<expr, expr>>> ret;

    auto folder = constant_folder_t();
    auto caster = auto_caster_t();
    expr folded_start = folder(caster(start)).remove_const();
    expr folded_length = folder(caster(length)).remove_const();
    // Case 1: the most commone case.
    if (folded_start.isa<constant>() && get_expr_as_int(folded_start) == 0) {
        if (folded_length.isa<constant>()) {
            int ilength
                    = get_const_as_int(folded_length.checked_as<constant_c>());
            if (ilength >= blocks) {
                auto block_num_range = std::make_pair(0, ilength / blocks);
                auto block_size_range = std::make_pair(0, blocks);
                ret.emplace_back(std::make_pair(std::move(block_num_range),
                        std::move(block_size_range)));
            }
            if (ilength % blocks != 0) {
                auto block_num_range = std::make_pair(ilength / blocks, 1);
                auto block_size_range = std::make_pair(0, ilength % blocks);
                ret.emplace_back(std::make_pair(std::move(block_num_range),
                        std::move(block_size_range)));
            }
        } else {
            // dynamic case, fixed block so no tail
            std::pair<expr, expr> block_num_range = std::make_pair(0,
                    folder(caster(divide_and_ceil(folded_length, blocks)))
                            .remove_const());
            std::pair<expr, expr> block_size_range = std::make_pair(0, blocks);
            ret.emplace_back(std::make_pair(
                    std::move(block_num_range), std::move(block_size_range)));
        }
    } else {
        COMPILE_ASSERT(folded_length.isa<constant_c>(),
                "constant length is expected, but got " << folded_length);
        int ilength = get_const_as_int(folded_length.checked_as<constant_c>());

        // Case 2: gcd case.
        if (folded_start->node_type_ == sc_expr_type::mul) {
            auto r = constant_folding::get_operand_from_binary(folded_start)
                             .second;
            if (r.isa<constant>()) {
                auto multiple = get_expr_as_int(r);
                if (multiple % blocks == 0) {
                    if (ilength >= blocks) {
                        auto block_num_range = std::make_pair(
                                folded_start / blocks, ilength / blocks);
                        auto block_size_range = std::make_pair(0, blocks);
                        ret.emplace_back(
                                std::make_pair(std::move(block_num_range),
                                        std::move(block_size_range)));
                    }
                    if (ilength % blocks != 0) {
                        auto block_num_range = std::make_pair(
                                folded_start / blocks + ilength / blocks, 1);
                        auto block_size_range
                                = std::make_pair(0, ilength % blocks);
                        ret.emplace_back(
                                std::make_pair(std::move(block_num_range),
                                        std::move(block_size_range)));
                    }
                } else {
                    int gcd = math_utils::get_gcd(multiple, blocks);
                    for (int i = 0; i < ilength / gcd; i++) {
                        auto block_num_range = std::make_pair(
                                (folded_start + i * gcd) / blocks, 1);
                        auto block_size_range = std::make_pair(
                                (folded_start + i * gcd) % blocks, gcd);
                        ret.emplace_back(
                                std::make_pair(std::move(block_num_range),
                                        std::move(block_size_range)));
                    }
                    if (ilength % gcd != 0) {
                        auto block_num_range = std::make_pair(
                                (folded_start + ilength / gcd) / blocks, 1);
                        auto block_size_range = std::make_pair(
                                (folded_start + ilength / gcd) % blocks,
                                ilength % gcd);
                        ret.emplace_back(
                                std::make_pair(std::move(block_num_range),
                                        std::move(block_size_range)));
                    }
                }
            }
        }
    }

    // Case 3: fallback to multi one-length slice
    if (ret.empty()) {
        COMPILE_ASSERT(folded_length.isa<constant_c>(),
                "constant length is expected, but got " << folded_length);
        int ilength = get_const_as_int(folded_length.checked_as<constant_c>());

        for (int i = 0; i < ilength; i++) {
            auto block_num_range
                    = std::make_pair((folded_start + i) / blocks, 1);
            auto block_size_range
                    = std::make_pair((folded_start + i) % blocks, 1);
            ret.emplace_back(std::make_pair(
                    std::move(block_num_range), std::move(block_size_range)));
        }
    }
    return ret;
}

void infer_stride2plain_reorder(slice_range &input_slice,
        sc_data_format_t input_format, sc_data_format_t output_format,
        slice_range &output_slice) {
    output_slice = input_slice;
    for (int i = 0; i < input_format.format_code_.norig_dims(); i++) {
        int plain_axis = input_format.format_code_.get(i);
        output_slice[plain_axis] = input_slice[i];
    }
}

void infer_plain2stride_reorder(slice_range &input_slice,
        sc_data_format_t input_format, sc_data_format_t output_format,
        slice_range &output_slice) {
    output_slice = input_slice;
    for (int i = 0; i < output_format.format_code_.norig_dims(); i++) {
        int plain_axis = output_format.format_code_.get(i);
        output_slice[i] = input_slice[plain_axis];
    }
}

void infer_stride2stride_reorder(slice_range_list &input_slice_list,
        sc_data_format_t input_format, sc_data_format_t output_format,
        slice_range_list &output_slice_list) {
    for (auto &input_slice : input_slice_list) {
        slice_range plain_slice, reorder_ranges;
        infer_stride2plain_reorder(
                input_slice, input_format, output_format, plain_slice);
        infer_plain2stride_reorder(
                plain_slice, input_format, output_format, reorder_ranges);
        output_slice_list.emplace_back(reorder_ranges);
    }
}

/**
 * For [AxBxCxD], if B and D, have more than one slice ranges, such as two, it
 * needs to dispatch them to four (2x2) slice ranges. Note that: if B and D come
 * from same plain axis, it only generate two slice ranges instead of four.
 * */
void dispatch_reorder_ranges(slice_range_list &total_ranges_list,
        slice_range_list &reorder_ranges_list,
        sc_data_format_t output_format = sc_data_format_t()) {
    std::vector<int> acc_size_list;
    int total_size = 1;
    for (size_t i = 0; i < total_ranges_list.size(); i++) {
        total_size *= total_ranges_list.at(i).size();
        acc_size_list.emplace_back(total_size);
    }

    // this check is aim to avoid too many compile time, it will also break
    // this kind reorder post fusion
    const int max_slice_size = 16;

    for (int i = 0; i < total_size; i++) {
        // set remainder
        int rmd = i;
        std::vector<int> index_list;
        bool valid = true;
        for (size_t j = 0; j < acc_size_list.size(); j++) {
            int size = (total_size / acc_size_list[j]);
            index_list.emplace_back(rmd / size);
            if (!output_format.is_any()
                    && output_format.format_code_.get(j)
                            < static_cast<int>(index_list.size())
                    && index_list[output_format.format_code_.get(j)]
                            != index_list[j]) {
                valid = false;
                break;
            }
            rmd = rmd % size;
        }
        if (!valid) continue;
        slice_range reorder_range;
        for (size_t j = 0; j < index_list.size(); j++) {
            reorder_range.emplace_back(total_ranges_list[j][index_list[j]]);
        }
        reorder_ranges_list.emplace_back(reorder_range);

        if (reorder_ranges_list.size() > max_slice_size) {
            reorder_ranges_list.clear();
            return;
        }
    }
}

size_t throw_if_negative(int dim) {
    if (dim < 0) { throw std::runtime_error("Bad format"); }
    return dim;
}

// infer plain format to blocking format generally
void infer_stride2block_reorder(slice_range_list &input_slice_list,
        sc_data_format_t input_format, sc_data_format_t output_format,
        slice_range_list &output_slice_list) {
    auto out_kind = output_format.format_code_;
    for (auto &input_slice : input_slice_list) {
        slice_range_list reorder_ranges_list;
        // stride->plain
        slice_range plain_slice;
        infer_stride2plain_reorder(input_slice, input_format,
                output_format.to_plain(), plain_slice);

        /**
         * block_slice_dict is the map, in which:
         * 1. key: represents plain pos
         * 2. value: is the vector of slice_range, e.g.
         *                               block_1_1  block_1_1
         *             block_1  block_1
         *   plain_1                     block_1_2  block_1_2
         *             block_2  block_2
         *                               block_2_1  block_2_2
         *
         *  the vector may look like as below:
         *  {block_1, block_1_1, block1_1}
         *  {block_1, block_1_2, block1_2}
         *  {block_2, block_2_1, block2_1}
         * */
        std::unordered_map<int, slice_range_list> block_slice_dict;

        // the first is plain index, the second is block cnt
        std::unordered_map<int, std::vector<int>> block_cnt_dict;

        for (int i = 0; i < out_kind.ndims(); i++) {
            int plain_pos = out_kind.get(i);
            block_cnt_dict[plain_pos].emplace_back(i);
            if (block_slice_dict[plain_pos].empty()) {
                block_slice_dict[plain_pos].emplace_back(
                        slice_range {plain_slice[plain_pos]});
            } else {
                slice_range_list update_block_slice;
                for (auto &block_range_list : block_slice_dict[plain_pos]) {
                    auto cur_plain_range = block_range_list.back();
                    auto cur_block
                            = output_format.blocks_
                                      [out_kind.collect_blocking_index(
                                                       plain_pos)
                                                      .at(block_cnt_dict[plain_pos] // NOLINT
                                                                      .size()
                                                              - 2)];
                    auto cur_block_ranges
                            = get_plain2block_ranges(cur_plain_range.first,
                                    cur_plain_range.second, cur_block);

                    for (auto &range_pair : cur_block_ranges) {
                        slice_range cpy_last_range_list = block_range_list;
                        cpy_last_range_list.back() = range_pair.first;
                        cpy_last_range_list.emplace_back(range_pair.second);
                        update_block_slice.emplace_back(cpy_last_range_list);
                    }
                }
                if (!update_block_slice.empty())
                    block_slice_dict[plain_pos] = update_block_slice;
            }
        }

        std::vector<slice_range> total_range_list(
                throw_if_negative(out_kind.ndims()));
        // collect all blocking slice
        for (auto &mp : block_slice_dict) {
            int plain_pos = mp.first;
            for (auto &range_list : mp.second) {
                for (size_t i = 0; i < range_list.size(); i++) {
                    int block_pos = block_cnt_dict[plain_pos].at(i);
                    total_range_list[block_pos].emplace_back(range_list[i]);
                }
            }
        }

        dispatch_reorder_ranges(
                total_range_list, reorder_ranges_list, output_format);

        output_slice_list.insert(output_slice_list.end(),
                reorder_ranges_list.begin(), reorder_ranges_list.end());
    }
}

// infer blocking format to plain format generally
void infer_block2stride_reorder(slice_range_list &input_slice_list,
        sc_data_format_t input_format, sc_data_format_t output_format,
        slice_range_list &output_slice_list) {
    auto out_kind = output_format.format_code_.to_plain();
    auto in_kind = input_format.format_code_;
    for (auto &input_slice : input_slice_list) {
        slice_range_list reorder_ranges_list;
        std::unordered_map<int, slice_range_list> plain_slice_dict;
        // from right to left
        for (int i = in_kind.ndims() - 1; i >= 0; i--) {
            int plain_pos = in_kind.get(i);
            if (plain_slice_dict[plain_pos].empty()) {
                plain_slice_dict[plain_pos].emplace_back(
                        slice_range {input_slice[i]});
            } else {
                std::pair<expr, expr> cur_block_num_range = input_slice[i];
                slice_range res;
                for (auto &cur_block_size_range :
                        plain_slice_dict[plain_pos].back()) {
                    auto cur_blocks = in_kind.collect_blocking_index(plain_pos);
                    int cur_block = input_format.blocks_[cur_blocks.at(
                            cur_blocks.size()
                            - plain_slice_dict[plain_pos].size())];
                    slice_range cur_plain_ranges_list
                            = get_block2plain_ranges(cur_block_num_range.first,
                                    cur_block_num_range.second,
                                    cur_block_size_range.first,
                                    cur_block_size_range.second, cur_block);
                    res.insert(res.end(), cur_plain_ranges_list.begin(),
                            cur_plain_ranges_list.end());
                }
                plain_slice_dict[plain_pos].emplace_back(std::move(res));
            }
        }

        std::vector<slice_range> total_range_list;
        for (int i = 0; i < out_kind.ndims(); i++) {
            total_range_list.emplace_back(plain_slice_dict[i].back()); // NOLINT
        }

        dispatch_reorder_ranges(total_range_list, reorder_ranges_list,
                output_format.to_plain());

        for (auto &range : reorder_ranges_list) {
            // plain -> stride
            slice_range stride_range;
            infer_plain2stride_reorder(range, output_format.to_plain(),
                    output_format, stride_range);
            range = std::move(stride_range);
        }
        output_slice_list.insert(output_slice_list.end(),
                reorder_ranges_list.begin(), reorder_ranges_list.end());
    }
}

// infer blocking format to blocking format generally
void infer_block2block_reorder(slice_range_list &input_slice_list,
        sc_data_format_t input_format, sc_data_format_t output_format,
        slice_range_list &output_slice_list) {
    slice_range_list plain_ranges_list;
    infer_block2stride_reorder(input_slice_list, input_format,
            input_format.to_plain(), plain_ranges_list);

    infer_stride2block_reorder(plain_ranges_list, input_format.to_plain(),
            output_format, output_slice_list);
}

static inline bool is_not_blocking(sc_data_format_t format) {
    return !format.is_blocking() && !format.is_any();
}

// generally infer reorder slice for any format (except padding cases)
void infer_reorder_slice(slice_range_list &input_slice_list,
        sc_data_format_t input_format, sc_data_format_t output_format,
        slice_range_list &output_slice_list) {
    if (is_not_blocking(input_format) && is_not_blocking(output_format)) {
        infer_stride2stride_reorder(input_slice_list, input_format,
                output_format, output_slice_list);
    } else if (is_not_blocking(input_format) && output_format.is_blocking()) {
        infer_stride2block_reorder(input_slice_list, input_format,
                output_format, output_slice_list);
    } else if (input_format.is_blocking() && is_not_blocking(output_format)) {
        infer_block2stride_reorder(input_slice_list, input_format,
                output_format, output_slice_list);
    } else if (input_format.is_blocking() && output_format.is_blocking()) {
        infer_block2block_reorder(input_slice_list, input_format, output_format,
                output_slice_list);
    } else {
        std::ostringstream ss;
        ss << "Unsupported data format. in = " << input_format
           << ", out = " << output_format;
        SC_MODULE_WARN << ss.str();
        throw tuner_recoverable_exception_t(ss.str());
    }
    for (auto &reorder_range : output_slice_list) {
        for (auto &r : reorder_range) {
            r.first = do_cast_and_fold(r.first);
        }
    }
}

void infer_padding_reorder_slice(slice_range_list &input_slice_list,
        const sc_dims &input_dims, const sc_dims &output_dims,
        slice_range_list &output_slice_list) {
    sc_dims inp_lead_dims = {input_dims.begin(), input_dims.end() - 1};
    sc_dims out_lead_dims = {output_dims.begin(), output_dims.end() - 1};
    // current, we can only infer padding reorder with last dim padded. E.g.
    // [1,16,1,1,32] ====> [1,16,1,1,48]
    if (inp_lead_dims != out_lead_dims) return;
    output_slice_list = input_slice_list;
    int inp_rank = input_dims.size();
    for (size_t i = 0; i < input_slice_list.size(); i++) {
        if (!slice_full_on_axis(
                    input_dims, input_slice_list[i], {inp_rank - 1})) {
            // could not infer this case, clear and return
            output_slice_list = {};
            return;
        }
        // set new full range on output slice
        output_slice_list[i].back()
                = std::make_pair(expr(0), dim2unsigned(output_dims.back()));
    }
}

bool check_required_slice(const graph_tensor_ptr &gt,
        const slice_range_list &range_list, int required_axis_from_end) {
    auto gt_dims = gt->details_.get_blocking_dims();
    std::vector<int> required_axis;
    int cur_len = gt_dims.size() - required_axis_from_end;
    for (size_t i = std::max(cur_len, 0); i < gt_dims.size(); i++) {
        required_axis.emplace_back(i);
    }
    return range_list.size() == 1
            && slice_full_on_axis(gt_dims, range_list[0], required_axis);
}

/**
 * @brief find the axis closest to the last which could be vectorized.
 * @param blocking_dims_expr blocking expr dim
 * @param format format
 * @param last_origin_axis original last axis
 * @param origin_axis_vectorized finded axis closed to the last
that can be vectorized
 * */
void find_vectorized_axis(std::vector<expr> const &blocking_dims_expr,
        sc_data_format_t const &format, int &last_origin_axis,
        int &origin_axis_vectorized) {
    origin_axis_vectorized = format.format_code_.ndims() - 1;
    // find not 1 dim in the last, if in dynamic cases, it will be as
    // original logic
    for (int i = origin_axis_vectorized; i >= 0; i--) {
        if (!blocking_dims_expr[i].isa<constant>()) { break; }
        if (get_expr_as_int(blocking_dims_expr[i]) > 1) {
            origin_axis_vectorized = i;
            break;
        }
    }
    last_origin_axis = format.format_code_.get(origin_axis_vectorized);
}

/**
 * @brief find the axis closest to the last which could be vectorized.
 * @param tsl tensor slice
 * @param format format
 * @param last_origin_axis original last axis
 * @param origin_axis_vectorized finded axis closed to the last
that can be vectorized
 * */
void find_vectorized_axis(const tensor_slice &tsl,
        sc_data_format_t const &format, int &last_origin_axis,
        int &origin_axis_vectorized) {
    origin_axis_vectorized = format.format_code_.ndims() - 1;
    // find not 1 dim in the last, if in dynamic cases, it will be as
    // original logic
    for (int i = origin_axis_vectorized; i >= 0; i--) {
        if (!tsl.get_shape()[i].isa<constant>()) { break; }
        if (get_expr_as_int(tsl.get_shape()[i]) > 1) {
            origin_axis_vectorized = i;
            break;
        }
    }
    last_origin_axis = format.format_code_.get(origin_axis_vectorized);
}
/**
 * @brief Calculate the total number of elements in a certain axis in the shape.
 * @param blocking_dims blocking dims
 * @param axis certain axis
 * */
int collect_axis_shape_size(
        sc_dims &blocking_dims, const std::vector<int> &axis) {
    int ret = 1;
    std::unordered_set<int> set;
    set.insert(axis.begin(), axis.end());
    for (size_t i = 0; i < blocking_dims.size(); i++) {
        if (set.find(i) != set.end()) { ret *= blocking_dims[i]; }
    }
    assert(ret > 0);
    return ret;
};

#define SLICE_RAGNE_CHECK_INIT_DATA() \
    bool use_out_loop = use_output_loop(); \
    sc_data_format_t target_format \
            = use_out_loop ? output_format : input_format; \
    auto dtype = info_.inputs_[0]->details_.dtype_; \
    auto blocking_exprs = get_blocking_shapes_expr( \
            get_owner_graph(), plain_dims_, target_format); \
    auto block_axis = target_format.format_code_.get( \
            target_format.format_code_.ndims() - 1); \
    int origin_axis_vectorized = target_format.format_code_.ndims() - 1; \
    auto toy_inp = builder::make_tensor( \
            std::string("dummy_inp"), blocking_exprs, dtype); \
    auto inp_slice = tensor_slice(toy_inp); \
    find_vectorized_axis( \
            inp_slice, target_format, block_axis, origin_axis_vectorized); \
    int len_from_last \
            = target_format.format_code_.ndims() - 1 - origin_axis_vectorized; \
    bool must_recheck = len_from_last > 0; \
    bool optimized_slice_check = support_optimized_kernel(ctx); \
    bool special_slice_check \
            = (support_optimized_kernel(ctx) || must_recheck); \
    len_from_last += optimized_slice_check \
            ? meet_vnni_reorder_require(ctx) ? 3 : 2 \
            : 1;

// infer reorder slice according input_slice
infer_status_code reorder_op_t::infer_slice_ranges(
        const context_ptr &ctx, fslice_map &fsmap) {
    // has been pre-inferred, skip
    if (!fsmap.get(get_outputs()[0]).empty()) return infer_status_code::OK;
    auto &input_format = get_input_format();
    auto &output_format = get_output_format();
    COMPILE_ASSERT(input_format.is_convertible(output_format),
            "Can not convert input format "
                    << input_format << " to output format " << output_format
                    << ".");
    // search known ranges from any input of cur fusbile op
    slice_range_map known_ranges_map = search_known_input_slice(this, fsmap);
    if (known_ranges_map.empty()) return infer_status_code::RETRY;

    auto input_slice_list = known_ranges_map[0];
    std::vector<int> required_axis(
            get_inputs()[0]->details_.get_blocking_dims().size(), 0);
    for (auto i = 0UL; i < get_inputs()[0]->details_.get_blocking_dims().size();
            i++) {
        required_axis[i] = i;
    }
    for (auto &src_range : input_slice_list) {
        if (!get_inputs()[0]->details_.is_dynamic()
                && !slice_divisible_on_axis(
                        get_inputs()[0]->details_.get_blocking_dims(),
                        src_range, required_axis)) {
            return infer_status_code::RETRY;
        }
    }

    SLICE_RAGNE_CHECK_INIT_DATA()
    if (special_slice_check
            && !check_required_slice(
                    get_inputs()[0], input_slice_list, len_from_last)) {
        return infer_status_code::RETRY;
    }

    slice_range_list reorder_ranges_list;
    // infer reorder slice only makes sense for non-padding cases in new fusion
    // mgr
    if (!check_padding() || is_dynamic()) {
        infer_reorder_slice(input_slice_list, input_format, output_format,
                reorder_ranges_list);
    } else {
        // infer padding reorder
        infer_padding_reorder_slice(input_slice_list,
                get_inputs()[0]->details_.get_blocking_dims(),
                get_outputs()[0]->details_.get_blocking_dims(),
                reorder_ranges_list);
    }

    if (!reorder_ranges_list.empty() && optimized_slice_check
            && !check_required_slice(
                    get_outputs()[0], reorder_ranges_list, len_from_last)) {
        return infer_status_code::RETRY;
    }
    fsmap.get(get_outputs()[0]) = reorder_ranges_list;
    return infer_status_code::OK;
}

// pre-infer reorder slice according output_slice
infer_status_code reorder_op_t::pre_infer_slice_ranges(
        const context_ptr &ctx, fslice_map &fsmap) {
    slice_range_list known_ranges_list = fsmap.get(get_outputs()[0]);
    auto &input_format = get_input_format();
    auto &output_format = get_output_format();
    SLICE_RAGNE_CHECK_INIT_DATA()
    // deal with begining reorder op, which use output loop
    if (fsmap.datamap_.size() == 1) {
        if (!use_output_loop()
                || (special_slice_check
                        && !attrs_.get_or_else(
                                mixed_partition_hint::pre_fuse_begin_op, false)
                        && !check_required_slice(get_outputs()[0],
                                known_ranges_list, len_from_last))) {
            return infer_status_code::RETRY;
        }
        return infer_status_code::OK;
    }
    if (is_dynamic()) { return infer_status_code::RETRY; }
    if (fsmap.get(get_inputs()[0]).empty()) {
        if (check_padding()) {
            if (!use_output_loop()) { return infer_status_code::RETRY; }
            return infer_status_code::OK;
        }
        slice_range_list input_slice_list;
        infer_reorder_slice(known_ranges_list, get_output_format(),
                get_input_format(), input_slice_list);
        if (input_slice_list.size() != 1 || !support_output_loop()) {
            return infer_status_code::RETRY;
        }
        fsmap.get(get_inputs()[0]) = input_slice_list;
    }
    return infer_status_code::OK;
}

void reorder_op_t::infer_binding_axis(binding_axis_map &bdax_map) {
    infer_identical_binding_axis(this, bdax_map);
}

void reorder_op_t::pre_infer_binding_axis(binding_axis_map &bdax_map) {
    pre_infer_identical_binding_axis(this, bdax_map);
}

std::vector<expr> get_reorder_stride2stride_indexes(
        const std::vector<expr> &in_indexes, const sc_data_format_t &in_format,
        const sc_data_format_t &out_format, const sc_dims &plain_dims) {
    if (in_indexes.empty()) { return std::vector<expr>(); }
    COMPILE_ASSERT(in_format.format_code_ != format_kinds::any
                    && out_format.format_code_ != format_kinds::any,
            "format can not be any in reorder op, please check it in layout "
            "propagation.");
    size_t base_out_dim = 0;
    assert(in_format.format_code_.ndims() == out_format.format_code_.ndims());
    size_t num_plain_dims = in_format.format_code_.norig_dims();
    size_t num_out_dims = num_plain_dims;
    std::vector<expr> ret(num_out_dims, 0);
    COMPILE_ASSERT(in_indexes.size() == num_plain_dims,
            "Wrong number of dimensions for format: "
                    << in_format
                    << ", real shape = " << utils::print_vector(in_indexes));

    COMPILE_ASSERT(in_indexes.size() <= sc_data_format_kind_t::MAX_DIMS,
            "Too many dims in plain shapes");
    std::vector<std::vector<int>> out_axis_map
            = out_format.format_code_.collect_p2b_mapping();
    // format index
    for (auto inp_idx = base_out_dim; inp_idx < num_out_dims; inp_idx++) {
        auto orig_dim = in_format.format_code_.get(inp_idx - base_out_dim);
        assert(orig_dim < static_cast<int>(out_axis_map.size())
                && out_axis_map[orig_dim].size() == 1);
        auto out_idx = out_axis_map[orig_dim][0];
        assert(base_out_dim + out_idx < ret.size());
        ret[base_out_dim + out_idx] = in_indexes[inp_idx];
    }
    return ret;
}

std::vector<expr> get_reorder_block2plain_indexes(sc_graph_t &graph,
        const std::vector<expr> &in_indexes, const sc_data_format_t &format,
        const sc_dims &plain_dims, expr &condition, expr &last_axis_offset,
        expr &other_axis_condition, const int target_axis) {
    if (in_indexes.empty()) { return std::vector<expr>(); }
    COMPILE_ASSERT(format.format_code_ != format_kinds::any,
            "format can not be any_t in reorder op, please check it in layout "
            "propagation.");
    size_t base_out_dim = 0;
    size_t num_plain_dims = throw_if_negative(format.format_code_.norig_dims());
    size_t num_format_dims = throw_if_negative(format.format_code_.ndims());
    size_t num_out_dims = num_plain_dims;
    std::vector<expr> ret(num_out_dims, 0);
    COMPILE_ASSERT(in_indexes.size() == num_format_dims,
            "Wrong number of dimensions for format: "
                    << format
                    << ", real shape = " << utils::print_vector(in_indexes));

    COMPILE_ASSERT(in_indexes.size() <= sc_data_format_kind_t::MAX_DIMS,
            "Too many dims in plain shapes");
    condition = true;
    other_axis_condition = true;
    std::unordered_map<int, int>
            axis2blocks; // plain_axis to block idx, idx++ after an access
    auto last_orig_axis
            = format.format_code_.get(static_cast<int>(num_format_dims) - 1);
    if (target_axis != TARGET_AXIS_NOT_DEFINE
            && last_orig_axis != target_axis) {
        last_orig_axis = format.format_code_.get(target_axis);
    }
    // format index
    for (int inp_idx = static_cast<int>(num_format_dims + base_out_dim) - 1;
            inp_idx >= static_cast<int>(base_out_dim); inp_idx--) {
        auto orig_axis = format.format_code_.get(inp_idx - base_out_dim);
        auto blocks = format.format_code_.collect_blocking_index(orig_axis);
        if (axis2blocks.find(orig_axis) == axis2blocks.end()) {
            axis2blocks[orig_axis] = static_cast<int>(blocks.size());
        }
        if (axis2blocks[orig_axis] == static_cast<int>(blocks.size())) {
            ret[base_out_dim + orig_axis]
                    = ret[base_out_dim + orig_axis] + in_indexes[inp_idx];
        } else {
            ret[base_out_dim + orig_axis] = ret[base_out_dim + orig_axis]
                    + in_indexes[inp_idx]
                            * format.blocks_[blocks[axis2blocks[orig_axis]]];
            if (axis2blocks[orig_axis] == 0) {
                auto new_condition
                        = ret[base_out_dim + orig_axis] < graph.dim_to_expr(
                                  plain_dims[base_out_dim + orig_axis]);
                condition = condition && new_condition;
                if (orig_axis == last_orig_axis) {
                    auto last_axis_offset_tmp
                            = cast_to_s32(graph.dim_to_expr(
                                      plain_dims[base_out_dim + orig_axis]))
                            - cast_to_s32(ret[base_out_dim + orig_axis]);
                    last_axis_offset = last_axis_offset.defined()
                            ? builder::make_min(
                                    last_axis_offset_tmp, last_axis_offset)
                            : last_axis_offset_tmp;
                } else {
                    other_axis_condition
                            = other_axis_condition && new_condition;
                }
            } else {
                auto new_condition = ret[base_out_dim + orig_axis]
                        < format.blocks_[blocks[axis2blocks[orig_axis] - 1]];
                condition = condition && new_condition;
                if (orig_axis == last_orig_axis) {
                    last_axis_offset
                            = cast_to_s32(format.blocks_[blocks
                                              [axis2blocks[orig_axis] - 1]])
                            - cast_to_s32(ret[base_out_dim + orig_axis]);
                } else {
                    other_axis_condition
                            = other_axis_condition && new_condition;
                }
            }
        }
        axis2blocks[orig_axis]--; // next block
    }
    return ret;
}

std::vector<expr> get_reorder_plain2block_indexes(
        const std::vector<expr> &in_indexes, const sc_data_format_t &format) {
    if (in_indexes.empty()) { return std::vector<expr>(); }
    COMPILE_ASSERT(format.format_code_ != format_kinds::any,
            "format can not be any in reorder op, please check it in layout "
            "propagation.");
    size_t base_out_dim = 0;
    size_t num_plain_dims = throw_if_negative(format.format_code_.norig_dims());
    size_t num_format_dims = throw_if_negative(format.format_code_.ndims());
    size_t num_out_dims = num_format_dims;
    std::vector<expr> ret(num_out_dims, 0);
    COMPILE_ASSERT(in_indexes.size() == num_plain_dims,
            "Wrong number of dimensions for format: "
                    << format
                    << ", real shape = " << utils::print_vector(in_indexes));

    COMPILE_ASSERT(in_indexes.size() <= sc_data_format_kind_t::MAX_DIMS,
            "Too many dims in plain shapes");
    std::unordered_map<int, int>
            axis2blocks; // plain_axis to block idx, idx++ after an access
    std::unordered_map<int, expr> axis2index; // current index of blocking
    for (auto inp_idx = base_out_dim; inp_idx < num_plain_dims + base_out_dim;
            inp_idx++) {
        axis2index[inp_idx - base_out_dim] = in_indexes[inp_idx];
    }
    // format index
    for (auto out_idx = base_out_dim; out_idx < num_format_dims + base_out_dim;
            out_idx++) {
        auto orig_axis = format.format_code_.get(out_idx - base_out_dim);
        if (axis2blocks.find(orig_axis) == axis2blocks.end()) {
            axis2blocks[orig_axis] = 0;
        }
        auto blocks = format.format_code_.collect_blocking_index(orig_axis);
        int cur_block = 0;
        if (axis2blocks[orig_axis] >= (int)blocks.size()) {
            ret[out_idx] = axis2index[orig_axis];
        } else {
            cur_block = format.blocks_[blocks[axis2blocks[orig_axis]]];
            ret[out_idx] = axis2index[orig_axis] / cur_block;
            axis2index[orig_axis] = axis2index[orig_axis] % cur_block;
            axis2blocks[orig_axis]++; // next block
        }
    }
    return ret;
}

static void cannot_convert_warning(const sc_data_format_t &input_format,
        const sc_data_format_t &output_format, const sc_dims &plain_dims) {
    SC_MODULE_WARN << "Can not do vectorize in reorder: " << input_format
                   << " to " << output_format
                   << " with plain dims:" << utils::print_vector(plain_dims);
}

// complex constant folding will suprisingly cause LLVM regression on
// vectorization for reorder
static void set_const_fold_bypass(const context_ptr &ctx, const stmt &v) {
#if defined(SC_LLVM_BACKEND)
    if (ctx->flags_.jit_kind_ == jit_kind::llvm) {
        v->attr()["bypass_complex_const_fold"] = true;
    }
#endif
}

constexpr const int byte = 8;
constexpr const int avx_simd_length = 128;
// inorder to vectorize u8s8x8
constexpr const int u8s8_min_simd_length = 64;
bool is_valid_step(int step) {
    return utils::is_one_of(step, 4, 8, 16, 32, 64);
}
static bool check_u8s8(sc_data_type_t dtype) {
    return dtype.is_etype(sc_data_etype::S8)
            || dtype.is_etype(sc_data_etype::U8);
}
void compute_reorder_stride2stride(sc_graph_t &graph, const context_ptr &ctx,
        const tensor_slice &src, tensor_slice &dst,
        const sc_data_format_t &input_format,
        const sc_data_format_t &output_format, sc_data_type_t dtype,
        const sc_dims &plain_dims, bool output_loop, any_map_t &attrs,
        const graph_tensor_ptr &expand_gt, size_t wkld = 0UL,
        bool is_innermost_dim_strided = false, bool is_dynamic = false,
        bool dynamic_no_padding = false) {
    auto input = src.get_real_tensor();
    auto output = dst.get_real_tensor();
    auto bld = builder::get_current_builder();
    std::vector<expr> iter_vars;
    std::vector<expr> in_indexes, out_indexes;
    std::vector<expr> loop_indexes;
    auto input_blocking_dims
            = sc_data_format_t::get_blocking_shapes(plain_dims, input_format);
    auto output_blocking_dims
            = sc_data_format_t::get_blocking_shapes(plain_dims, output_format);
    auto input_blocking_dims_expr
            = get_blocking_shapes_expr(graph, plain_dims, input_format);
    auto output_blocking_dims_expr
            = get_blocking_shapes_expr(graph, plain_dims, output_format);
    auto output_last_origin_axis = output_format.format_code_.get(
            output_format.format_code_.ndims() - 1);
    auto input_last_origin_axis = input_format.format_code_.get(
            input_format.format_code_.ndims() - 1);
    int input_origin_axis_vectorized = input_format.format_code_.ndims() - 1;
    int output_origin_axis_vectorized = output_format.format_code_.ndims() - 1;
    find_vectorized_axis(input_blocking_dims_expr, input_format,
            input_last_origin_axis, input_origin_axis_vectorized);
    find_vectorized_axis(output_blocking_dims_expr, output_format,
            output_last_origin_axis, output_origin_axis_vectorized);
    auto src_slice_shape = src.get_shape();
    auto dst_slice_shape = dst.get_shape();
    int step = static_cast<int>(vectorize_step(ctx, dtype.type_code_));
    const int max_step
            = static_cast<int>(vectorize_step(ctx, dtype.type_code_));
    if ((!output_loop
                && src_slice_shape[input_origin_axis_vectorized]
                           .isa<constant>())
            || (output_loop
                    && dst_slice_shape[output_origin_axis_vectorized]
                               .isa<constant>())) {
        step = std::min(max_step,
                (int)get_expr_as_int(!output_loop
                                ? src_slice_shape[input_origin_axis_vectorized]
                                : dst_slice_shape
                                        [output_origin_axis_vectorized]));
        step = utils::get_nearest_vector_step(step);
    }

    bool is_u8s8 = check_u8s8(dtype);
    bool can_vectorize = !is_innermost_dim_strided
            && input_last_origin_axis == output_last_origin_axis
            && step * utils::get_sizeof_type(dtype) * byte >= (uint64_t)(
                       is_u8s8 ? u8s8_min_simd_length : avx_simd_length);
    if (!can_vectorize) {
        cannot_convert_warning(input_format, output_format, plain_dims);
    }
    step = can_vectorize ? step : 1;
    bool no_padding = is_dynamic && dynamic_no_padding;
    if (!output_loop) {
        no_padding |= !is_dynamic
                && get_expr_as_int(
                           src_slice_shape[input_origin_axis_vectorized])
                                % step
                        == 0;
        for (size_t i = 0; i < plain_dims.size(); i++) {
            iter_vars.emplace_back(builder::make_var(datatypes::index,
                    std::string("_fuseiter") + fusion_create_idx()));
            loop_indexes.emplace_back(iter_vars[i]);
            in_indexes.emplace_back(iter_vars[i] + src.get_offset()[i]);
        }
        out_indexes = get_reorder_stride2stride_indexes(
                in_indexes, input_format, output_format, plain_dims);
        auto cur = builder::make_stmts_unattached({});
        expr mask;
        stmt mask_def;
        if (!no_padding && can_vectorize) {
            auto idx_len
                    = cast_to_s32(src_slice_shape[input_origin_axis_vectorized])
                    - cast_to_s32(iter_vars[input_origin_axis_vectorized]);
            auto cur_step = builder::make_min(
                    builder::make_max(builder::make_constant(0), idx_len),
                    step);
            mask = generate_mask_var_by_step(mask_def, cur_step, step);
            cur.static_as<stmts>()->seq_.emplace_back(mask_def);
        }
        auto assign = builder::make_assign_unattached(
                builder::make_indexing(output, out_indexes, step, mask),
                builder::make_indexing(src.tptr_, loop_indexes, step, mask));
        cur->attr()[op_traits::workload_computable_t::workload_number] = wkld;
        cur.static_as<stmts>()->seq_.emplace_back(assign);
        stmt body;
        std::vector<stmt> loops;

        for (int i = static_cast<int>(plain_dims.size()) - 1; i >= 0; i--) {
            body = cur.isa<stmts>() ? cur
                                    : make_stmt<stmts_node_t>(
                                            std::vector<stmt> {std::move(cur)});
            cur = make_stmt<for_loop_node_t>(std::move(iter_vars.at(i)),
                    expr(0), src.get_shape()[i],
                    can_vectorize && i == input_origin_axis_vectorized
                            ? expr(step)
                            : expr(1),
                    std::move(body), true, for_type::NORMAL);
            bind_loop_axis(expand_gt, cur, i);
            loops.push_back(cur);
        }
        std::reverse(loops.begin(), loops.end());
        for (int i = 1; i < static_cast<int>(plain_dims.size()) - 2; i++) {
            loops[0].checked_as<for_loop>()->fuse(
                    loops[i].checked_as<for_loop>());
        }
        bld->emit(cur);
        if (!can_vectorize) { set_const_fold_bypass(ctx, cur); }
    } else {
        no_padding |= !is_dynamic
                && get_expr_as_int(
                           dst_slice_shape[output_origin_axis_vectorized])
                                % step
                        == 0;
        for (size_t i = 0; i < plain_dims.size(); i++) {
            iter_vars.emplace_back(builder::make_var(datatypes::index,
                    std::string("_fuseiter") + fusion_create_idx()));
            out_indexes.emplace_back(iter_vars[i] + dst.get_offset()[i]);
        }
        in_indexes = get_reorder_stride2stride_indexes(
                out_indexes, output_format, input_format, plain_dims);
        auto cur = builder::make_stmts_unattached({});
        expr mask;
        stmt mask_def;
        if (!no_padding && can_vectorize) {
            auto idx_len
                    = cast_to_s32(
                              dst_slice_shape[output_origin_axis_vectorized])
                    - cast_to_s32(iter_vars[output_origin_axis_vectorized]);
            auto cur_step = builder::make_min(
                    builder::make_max(builder::make_constant(0), idx_len),
                    step);
            mask = generate_mask_var_by_step(mask_def, cur_step, step);
            cur.static_as<stmts>()->seq_.emplace_back(mask_def);
        }
        auto assign = builder::make_assign_unattached(
                builder::make_indexing(output, out_indexes, step, mask),
                builder::make_indexing(input, in_indexes, step, mask));
        cur.static_as<stmts>()->seq_.emplace_back(assign);
        cur->attr()[op_traits::workload_computable_t::workload_number] = wkld;
        stmt body;
        std::vector<stmt> loops;
        for (int i = static_cast<int>(plain_dims.size()) - 1; i >= 0; i--) {
            body = cur.isa<stmts>() ? cur
                                    : make_stmt<stmts_node_t>(
                                            std::vector<stmt> {std::move(cur)});
            cur = make_stmt<for_loop_node_t>(std::move(iter_vars.at(i)),
                    expr(0), dst.get_shape()[i],
                    can_vectorize && i == output_origin_axis_vectorized
                            ? expr(step)
                            : expr(1),
                    std::move(body), true, for_type::NORMAL);
            bind_loop_axis(expand_gt, cur, i);
            loops.push_back(cur);
        }
        std::reverse(loops.begin(), loops.end());
        for (int i = 1; i < static_cast<int>(plain_dims.size()) - 2; i++) {
            loops[0].checked_as<for_loop>()->fuse(
                    loops[i].checked_as<for_loop>());
        }
        bld->emit(cur);
        if (!can_vectorize) { set_const_fold_bypass(ctx, cur); }
    }
}

void compute_reorder_block2stride(sc_graph_t &graph, const context_ptr &ctx,
        const tensor_slice &src, tensor_slice &dst,
        const sc_data_format_t &input_format,
        const sc_data_format_t &output_format, sc_data_type_t dtype,
        const sc_dims &plain_dims, any_map_t &attrs,
        const graph_tensor_ptr &expand_gt, size_t wkld = 0UL,
        bool is_innermost_dim_strided = false, bool is_dynamic = false,
        bool dynamic_no_padding = false) {
    auto input = src.get_real_tensor();
    auto output = dst.get_real_tensor();
    auto bld = builder::get_current_builder();
    auto input_blocking_dims
            = sc_data_format_t::get_blocking_shapes(plain_dims, input_format);
    auto output_blocking_dims
            = sc_data_format_t::get_blocking_shapes(plain_dims, output_format);
    auto input_blocking_dims_expr
            = get_blocking_shapes_expr(graph, plain_dims, input_format);
    auto output_blocking_dims_expr
            = get_blocking_shapes_expr(graph, plain_dims, output_format);
    assert(output_format.format_code_.ndims()
            == output_format.format_code_.norig_dims());
    auto output_last_origin_axis = output_format.format_code_.get(
            output_format.format_code_.ndims() - 1);
    auto input_last_origin_axis = input_format.format_code_.get(
            input_format.format_code_.ndims() - 1);
    // block has constraint
    assert(!is_dynamic_dim(input_blocking_dims.back()));
    int input_origin_axis_vectorized = input_format.format_code_.ndims() - 1;
    int output_origin_axis_vectorized = output_format.format_code_.ndims() - 1;
    find_vectorized_axis(input_blocking_dims_expr, input_format,
            input_last_origin_axis, input_origin_axis_vectorized);
    find_vectorized_axis(output_blocking_dims_expr, output_format,
            output_last_origin_axis, output_origin_axis_vectorized);
    auto src_slice_shape = src.get_shape();
    int max_step = static_cast<int>(vectorize_step(ctx, dtype.type_code_));
    assert(!is_dynamic_dim(input_blocking_dims[input_origin_axis_vectorized]));
    int step = std::min(max_step,
            static_cast<int>(get_expr_as_int(
                    src_slice_shape[input_origin_axis_vectorized])));
    step = utils::get_nearest_vector_step(step);
    step = std::min(max_step, step);
    if (attrs.get_or_else(op_attr_key::no_fuse, false)) {
        while (step < max_step
                && get_expr_as_int(
                           src_slice_shape[input_origin_axis_vectorized])
                                % (2 * step)
                        == 0) {
            step = 2 * step;
        }
    }
    bool is_u8s8 = check_u8s8(dtype);
    bool can_vectorize = !is_innermost_dim_strided
            && input_last_origin_axis == output_last_origin_axis
            && get_expr_as_int(src_slice_shape[input_origin_axis_vectorized])
                            % step
                    == 0
            && is_valid_step(step)
            && step * utils::get_sizeof_type(dtype) * byte >= (uint64_t)(
                       is_u8s8 ? u8s8_min_simd_length : avx_simd_length);

    bool no_padding = !is_dynamic
            && sc_data_format_t::get_padded_plain_shapes(
                       input_blocking_dims, input_format)
                    == sc_data_format_t::get_padded_plain_shapes(
                            output_blocking_dims, output_format);
    no_padding |= (is_dynamic && dynamic_no_padding);

    if (!can_vectorize) {
        cannot_convert_warning(input_format, output_format, plain_dims);
    }
    step = can_vectorize ? step : 1;
    std::vector<expr> iter_vars;
    std::vector<expr> in_indexes;
    std::vector<expr> loop_indexes;
    for (size_t i = 0; i < input_blocking_dims.size(); i++) {
        iter_vars.emplace_back(range_from_outer_loop(src.get_ranges()[i])
                        ? expr(0)
                        : builder::make_var(datatypes::index,
                                std::string("_fuseiter")
                                        + fusion_create_idx()));
        in_indexes.emplace_back(iter_vars[i] + src.get_offset()[i]);
        loop_indexes.emplace_back(iter_vars[i]);
    }
    expr condition;
    expr last_axis_offset, other_axis_condition;
    std::vector<expr> tmp_out_indexes = get_reorder_block2plain_indexes(graph,
            in_indexes, input_format, plain_dims, condition, last_axis_offset,
            other_axis_condition, input_origin_axis_vectorized);
    std::vector<expr> out_indexes
            = get_reorder_stride2stride_indexes(tmp_out_indexes,
                    output_format.to_plain(), output_format, plain_dims);

    expr mask;
    stmt mask_def;
    if (!no_padding && can_vectorize) {
        // mask = min(max(0, last_dim_len - last_dim_idx),step)
        // To choose [0 ~ step] mask
        auto cur_step = builder::make_min(
                builder::make_max(builder::make_constant(0), last_axis_offset),
                step);
        // mask = other_dims_condition ? mask : 0;
        mask = generate_mask_var_by_step(
                mask_def, cur_step, step, other_axis_condition);
    }

    auto assign = builder::make_assign_unattached(
            builder::make_indexing(output, out_indexes, step, mask),
            // here, use src.tptr instead of input is aimed to avoid
            // input is tensor_view_op. Oherwisw, it will throw
            // illegal exception in index_flatten
            builder::make_indexing(expr(src.tptr_), loop_indexes, step));
    assign->attr()[op_traits::workload_computable_t::workload_number] = wkld;
    stmt cur = assign;
    if (mask_def.defined()) {
        cur = builder::make_stmts_unattached({mask_def, assign});
    }
    if (!no_padding && !can_vectorize) {
        cur = builder::make_if_else_unattached(condition, assign, stmt());
    }
    stmt body;
    for (int i = static_cast<int>(input_blocking_dims.size()) - 1; i >= 0;
            i--) {
        // Do not generate those dummy loops
        if (!iter_vars.at(i).isa<var>()) continue;
        body = cur.isa<stmts>()
                ? cur
                : make_stmt<stmts_node_t>(std::vector<stmt> {std::move(cur)});
        cur = make_stmt<for_loop_node_t>(std::move(iter_vars.at(i)), expr(0),
                src.get_shape()[i],
                can_vectorize && i == input_origin_axis_vectorized
                        ? expr(static_cast<int>(step))
                        : expr(1),
                std::move(body), true, for_type::NORMAL);
        bind_loop_axis(expand_gt, cur, i, true);
    }
    cur->attr()[stmt_attr_key::merge_loop] = true;
    bld->emit(cur);
    if (!can_vectorize) { set_const_fold_bypass(ctx, cur); }
}

void compute_reorder_stride2block(sc_graph_t &graph, const context_ptr &ctx,
        const tensor_slice &src, tensor_slice &dst,
        const sc_data_format_t &input_format,
        const sc_data_format_t &output_format, sc_data_type_t dtype,
        const sc_dims &plain_dims, bool output_loop, any_map_t &attrs,
        const graph_tensor_ptr &expand_gt, size_t wkld = 0UL,
        bool is_innermost_dim_strided = false, bool is_dynamic = false,
        bool dynamic_no_padding = false) {
    auto input = src.get_real_tensor();
    auto output = dst.get_real_tensor();
    auto bld = builder::get_current_builder();
    auto input_blocking_dims
            = sc_data_format_t::get_blocking_shapes(plain_dims, input_format);
    auto output_blocking_dims
            = sc_data_format_t::get_blocking_shapes(plain_dims, output_format);
    auto input_blocking_exprs
            = get_blocking_shapes_expr(graph, plain_dims, input_format);
    auto output_blocking_exprs
            = get_blocking_shapes_expr(graph, plain_dims, output_format);
    assert(input_format.format_code_.ndims()
            == input_format.format_code_.norig_dims());
    auto input_last_origin_axis = input_format.format_code_.get(
            input_format.format_code_.ndims() - 1);
    auto output_last_origin_axis = output_format.format_code_.get(
            output_format.format_code_.ndims() - 1);

    // block has constraint
    assert(!is_dynamic_dim(output_blocking_dims.back()));
    int input_origin_axis_vectorized = input_format.format_code_.ndims() - 1;
    int output_origin_axis_vectorized = output_format.format_code_.ndims() - 1;
    find_vectorized_axis(input_blocking_exprs, input_format,
            input_last_origin_axis, input_origin_axis_vectorized);
    find_vectorized_axis(output_blocking_exprs, output_format,
            output_last_origin_axis, output_origin_axis_vectorized);
    auto src_slice_shape = src.get_shape();
    auto dst_slice_shape = dst.get_shape();
    int max_step = static_cast<int>(vectorize_step(ctx, dtype.type_code_));
    int step = static_cast<int>(vectorize_step(ctx, dtype.type_code_));
    if ((!output_loop
                && src_slice_shape[input_origin_axis_vectorized]
                           .isa<constant>())
            || (output_loop
                    && dst_slice_shape[output_origin_axis_vectorized]
                               .isa<constant>())) {
        step = std::min(step,
                output_loop ? static_cast<int>(get_expr_as_int(
                        dst_slice_shape[output_origin_axis_vectorized]))
                            : static_cast<int>(get_expr_as_int(src_slice_shape
                                            [input_origin_axis_vectorized])));
        step = utils::get_nearest_vector_step(step);
    }
    if (attrs.get_or_else(op_attr_key::no_fuse, false)) {
        while (step < max_step
                && get_expr_as_int(
                           dst_slice_shape[output_origin_axis_vectorized])
                                % (2 * step)
                        == 0) {
            step = 2 * step;
        }
    }
    bool is_u8s8 = check_u8s8(dtype);
    bool can_vectorize = !is_innermost_dim_strided
            && input_last_origin_axis == output_last_origin_axis
            && get_expr_as_int(dst_slice_shape[output_origin_axis_vectorized])
                            % step
                    == 0
            && is_valid_step(step)
            && step * utils::get_sizeof_type(dtype) * byte >= (uint64_t)(
                       is_u8s8 ? u8s8_min_simd_length : avx_simd_length);
    // Usually use input loop means no padding in static, but not in dynamic, if
    // dynamic and use input loop, need to check the static dim with blocks.
    if (!output_loop
            && !is_dynamic_dim(
                    input_blocking_dims[input_origin_axis_vectorized])
            && get_expr_as_int(src_slice_shape[input_origin_axis_vectorized])
                            % step
                    != 0) {
        can_vectorize = false;
    }
    bool no_padding = !is_dynamic
            && sc_data_format_t::get_padded_plain_shapes(
                       output_blocking_dims, output_format)
                    == sc_data_format_t::get_padded_plain_shapes(
                            input_blocking_dims, input_format);
    no_padding |= (is_dynamic && dynamic_no_padding);
    if (!can_vectorize) {
        cannot_convert_warning(input_format, output_format, plain_dims);
    }

    step = can_vectorize ? step : 1;
    std::vector<expr> iter_vars;
    std::vector<expr> loop_indexes;
    if (!output_loop) {
        std::vector<expr> in_indexes;
        for (size_t i = 0; i < input_blocking_dims.size(); i++) {
            iter_vars.emplace_back(range_from_outer_loop(src.get_ranges()[i])
                            ? expr(0)
                            : builder::make_var(datatypes::index,
                                    std::string("_fuseiter")
                                            + fusion_create_idx()));
            in_indexes.emplace_back(iter_vars[i] + src.get_offset()[i]);
            loop_indexes.emplace_back(iter_vars[i]);
        }
        std::vector<expr> tmp_out_indexes = get_reorder_stride2stride_indexes(
                in_indexes, input_format, input_format.to_plain(), plain_dims);
        std::vector<expr> out_indexes = get_reorder_plain2block_indexes(
                tmp_out_indexes, output_format);

        auto assign = builder::make_stmts_unattached(
                {builder::make_assign_unattached(
                        builder::make_indexing(output, out_indexes, step),
                        // here, use src.tptr instead of input is aimed to avoid
                        // input is tensor_view_op. otherwise, it will throw
                        // illegal exception in index_flatten
                        builder::make_indexing(
                                src.tptr_, loop_indexes, step))});
        assign->attr()[op_traits::workload_computable_t::workload_number]
                = wkld;
        auto cur = assign;
        stmt body;
        for (int i = static_cast<int>(input_blocking_dims.size()) - 1; i >= 0;
                i--) {
            // Do not generate those dummy loops
            if (!iter_vars.at(i).isa<var>()) continue;
            body = cur.isa<stmts>() ? cur
                                    : make_stmt<stmts_node_t>(
                                            std::vector<stmt> {std::move(cur)});
            cur = make_stmt<for_loop_node_t>(std::move(iter_vars.at(i)),
                    expr(0), src.get_shape()[i],
                    can_vectorize && i == input_origin_axis_vectorized
                            ? expr(static_cast<int>(step))
                            : expr(1),
                    std::move(body), true, for_type::NORMAL);
            bind_loop_axis(expand_gt, cur, i, true);
        }
        cur->attr()[stmt_attr_key::merge_loop] = true;
        bld->emit(cur);
        if (!can_vectorize) { set_const_fold_bypass(ctx, cur); }
    } else {
        std::vector<expr> out_indexes;
        for (size_t i = 0; i < output_blocking_dims.size(); i++) {
            iter_vars.emplace_back(range_from_outer_loop(dst.get_ranges()[i])
                            ? expr(0)
                            : builder::make_var(datatypes::index,
                                    std::string("_fuseiter")
                                            + fusion_create_idx()));
            out_indexes.emplace_back(iter_vars[i] + dst.get_offset()[i]);
        }
        expr condition;
        expr last_axis_offset, other_axis_condition;
        std::vector<expr> tmp_in_indexes
                = get_reorder_block2plain_indexes(graph, out_indexes,
                        output_format, plain_dims, condition, last_axis_offset,
                        other_axis_condition, output_origin_axis_vectorized);
        std::vector<expr> in_indexes
                = get_reorder_stride2stride_indexes(tmp_in_indexes,
                        input_format.to_plain(), input_format, plain_dims);
        expr mask;
        stmt mask_def;
        if (!no_padding && can_vectorize) {
            // mask = min(max(0, last_dim_len - last_dim_idx),step)
            // To choose [0 ~ step] mask
            auto cur_step = builder::make_min(
                    builder::make_max(
                            builder::make_constant(0), last_axis_offset),
                    step);
            // mask = other_dims_condition ? mask : 0;
            mask = generate_mask_var_by_step(
                    mask_def, cur_step, step, other_axis_condition);
        }
        auto assign = builder::make_assign_unattached(
                builder::make_indexing(expr(output), out_indexes, step),
                builder::make_indexing(expr(input), in_indexes, step, mask));

        assign->attr()[op_traits::workload_computable_t::workload_number]
                = wkld;
        stmt cur = assign;
        if (mask_def.defined()) {
            cur = builder::make_stmts_unattached({mask_def, assign});
        }
        if (!no_padding && !can_vectorize) {
            auto padding = builder::make_stmts_unattached(
                    {builder::make_assign_unattached(
                            builder::make_indexing(output, out_indexes, step),
                            builder::make_constant({0UL},
                                    sc_data_type_t(dtype.type_code_, step)))});
            cur = builder::make_if_else_unattached(condition, assign, padding);
        }
        stmt body;
        std::vector<stmt> loops;
        //______________mask version
        for (int i = static_cast<int>(output_blocking_dims.size()) - 1; i >= 0;
                i--) {
            // Do not generate those dummy loops
            if (!iter_vars.at(i).isa<var>()) continue;
            body = cur.isa<stmts>() ? cur
                                    : make_stmt<stmts_node_t>(
                                            std::vector<stmt> {std::move(cur)});
            cur = make_stmt<for_loop_node_t>(std::move(iter_vars.at(i)),
                    expr(0), dst.get_shape()[i],
                    can_vectorize && i == output_origin_axis_vectorized
                            ? expr(static_cast<int>(step))
                            : expr(1),
                    std::move(body), true, for_type::NORMAL);
            bind_loop_axis(expand_gt, cur, i, true);
            loops.push_back(cur);
        }
        bld->emit(cur);
        if (!can_vectorize) { set_const_fold_bypass(ctx, cur); }
    }
}

void compute_reorder_block2block(sc_graph_t &graph, const context_ptr &ctx,
        const tensor_slice &src, tensor_slice &dst,
        const sc_data_format_t &input_format,
        const sc_data_format_t &output_format, sc_data_type_t dtype,
        const sc_dims &plain_dims1, bool output_loop, any_map_t &attrs,
        const graph_tensor_ptr &expand_gt, size_t wkld = 0UL,
        bool is_innermost_dim_strided = false, bool is_dynamic = false,
        bool dynamic_no_padding = false) {
    auto input = src.get_real_tensor();
    auto output = dst.get_real_tensor();
    auto bld = builder::get_current_builder();
    // walk around for bert fusion
    sc_dims plain_dims(plain_dims1);
    if (plain_dims.empty()) {
        sc_dims dst_blocking_dims;
        for (int64_t i = 0; i < dst.nslice_dims(); i++) {
            dst_blocking_dims.push_back(get_const_as_int(
                    dst.get_shape()[i].checked_as<constant>()));
        }
        plain_dims = sc_data_format_t::get_padded_plain_shapes(
                dst_blocking_dims, output_format);
    }
    auto input_blocking_dims
            = sc_data_format_t::get_blocking_shapes(plain_dims, input_format);
    auto output_blocking_dims
            = sc_data_format_t::get_blocking_shapes(plain_dims, output_format);
    auto input_blocking_exprs
            = get_blocking_shapes_expr(graph, plain_dims, input_format);
    auto output_blocking_exprs
            = get_blocking_shapes_expr(graph, plain_dims, output_format);
    auto input_padded_plain_dims = sc_data_format_t::get_padded_plain_shapes(
            input_blocking_dims, input_format);
    auto output_padded_plain_dims = sc_data_format_t::get_padded_plain_shapes(
            output_blocking_dims, output_format);
    // plain axis of last block
    auto input_block_axis = input_format.format_code_.get(
            input_format.format_code_.ndims() - 1);
    auto output_block_axis = output_format.format_code_.get(
            output_format.format_code_.ndims() - 1);
    int input_origin_axis_vectorized = input_format.format_code_.ndims() - 1;
    int output_origin_axis_vectorized = output_format.format_code_.ndims() - 1;
    find_vectorized_axis(input_blocking_exprs, input_format, input_block_axis,
            input_origin_axis_vectorized);
    find_vectorized_axis(output_blocking_exprs, output_format,
            output_block_axis, output_origin_axis_vectorized);
    auto src_slice_shape = src.get_shape();
    auto dst_slice_shape = dst.get_shape();
    bool no_padding = !is_dynamic
            && input_padded_plain_dims == output_padded_plain_dims;
    no_padding |= (is_dynamic && dynamic_no_padding);
    int step = std::min(static_cast<int>(vectorize_step(ctx, dtype.type_code_)),
            output_loop
                    ? static_cast<int>(get_expr_as_int(
                            dst_slice_shape[output_origin_axis_vectorized]))
                    : static_cast<int>(get_expr_as_int(
                            src_slice_shape[input_origin_axis_vectorized])));
    step = utils::get_nearest_vector_step(step);
    bool is_u8s8 = check_u8s8(dtype);
    bool can_vectorize = !is_innermost_dim_strided
            && input_block_axis == output_block_axis
            && get_expr_as_int(dst_slice_shape[output_origin_axis_vectorized])
                            % step
                    == 0
            && get_expr_as_int(src_slice_shape[input_origin_axis_vectorized])
                            % step
                    == 0
            && plain_dims[input_block_axis]
                            % input_blocking_dims[input_origin_axis_vectorized]
                    == 0
            && plain_dims[output_block_axis]
                            % output_blocking_dims
                                    [output_origin_axis_vectorized]
                    == 0
            && is_valid_step(step)
            && step * utils::get_sizeof_type(dtype) * 8 >= (uint64_t)(
                       is_u8s8 ? u8s8_min_simd_length : avx_simd_length);

    if (!can_vectorize) {
        cannot_convert_warning(input_format, output_format, plain_dims);
    }
    step = can_vectorize ? step : 1;
    std::vector<expr> iter_vars;
    // for input loops
    if (!output_loop) {
        std::vector<expr> in_indexes;
        std::vector<expr> loop_indexes;
        for (size_t i = 0; i < input_blocking_dims.size(); i++) {
            iter_vars.emplace_back(range_from_outer_loop(src.get_ranges()[i])
                            ? expr(0)
                            : builder::make_var(datatypes::index,
                                    std::string("_fuseiter")
                                            + fusion_create_idx()));
            in_indexes.emplace_back(iter_vars[i] + src.get_offset()[i]);
            loop_indexes.emplace_back(iter_vars[i]);
        }
        expr condition;
        expr last_axis_offset, other_axis_condition;
        std::vector<expr> tmp_indexes
                = get_reorder_block2plain_indexes(graph, in_indexes,
                        input_format, plain_dims, condition, last_axis_offset,
                        other_axis_condition, input_origin_axis_vectorized);
        std::vector<expr> out_indexes
                = get_reorder_plain2block_indexes(tmp_indexes, output_format);

        auto assign = builder::make_stmts_unattached(
                {builder::make_assign_unattached(
                        builder::make_indexing(output, out_indexes, step),
                        // here, use src.tptr instead of input is aimed to
                        // avoid input is tensor_view_op. Oherwisw, it will
                        // throw illegal exception in index_flatten
                        builder::make_indexing(
                                src.tptr_, loop_indexes, step))});
        assign->attr()[op_traits::workload_computable_t::workload_number]
                = wkld;
        auto cur = no_padding
                ? assign
                : builder::make_if_else_unattached(condition, assign, stmt());
        stmt body;
        for (int i = static_cast<int>(input_blocking_dims.size()) - 1; i >= 0;
                i--) {
            // Do not generate those dummy loops
            if (!iter_vars.at(i).isa<var>()) continue;
            body = cur.isa<stmts>() ? cur
                                    : make_stmt<stmts_node_t>(
                                            std::vector<stmt> {std::move(cur)});
            cur = make_stmt<for_loop_node_t>(std::move(iter_vars.at(i)),
                    expr(0), src.get_shape()[i],
                    can_vectorize && i == input_origin_axis_vectorized
                            ? expr(static_cast<int>(step))
                            : expr(1),
                    std::move(body), true, for_type::NORMAL);
            bind_loop_axis(expand_gt, cur, i, true);
        }
        cur->attr()[stmt_attr_key::merge_loop] = true;
        bld->emit(cur);
        if (!can_vectorize) { set_const_fold_bypass(ctx, cur); }
    } else {
        std::vector<expr> out_indexes;
        for (size_t i = 0; i < output_blocking_dims.size(); i++) {
            iter_vars.emplace_back(builder::make_var(datatypes::index,
                    std::string("_fuseiter") + fusion_create_idx()));
            out_indexes.emplace_back(iter_vars[i] + dst.get_offset()[i]);
        }
        expr condition;
        expr last_axis_offset, other_axis_condition;
        std::vector<expr> tmp_indexes
                = get_reorder_block2plain_indexes(graph, out_indexes,
                        output_format, plain_dims, condition, last_axis_offset,
                        other_axis_condition, output_origin_axis_vectorized);
        std::vector<expr> in_indexes
                = get_reorder_plain2block_indexes(tmp_indexes, input_format);
        auto assign = builder::make_stmts_unattached(
                {builder::make_assign_unattached(
                        builder::make_indexing(output, out_indexes, step),
                        builder::make_indexing(input, in_indexes, step))});
        assign->attr()[op_traits::workload_computable_t::workload_number]
                = wkld;
        auto padding = builder::make_stmts_unattached(
                {builder::make_assign_unattached(
                        builder::make_indexing(output, out_indexes, step),
                        builder::make_constant({0UL},
                                sc_data_type_t(dtype.type_code_, step)))});
        auto cur = no_padding
                ? assign
                : builder::make_if_else_unattached(condition, assign, padding);
        stmt body;
        std::vector<stmt> loops;
        for (int i = static_cast<int>(output_blocking_dims.size()) - 1; i >= 0;
                i--) {
            body = cur.isa<stmts>() ? cur
                                    : make_stmt<stmts_node_t>(
                                            std::vector<stmt> {std::move(cur)});
            cur = make_stmt<for_loop_node_t>(std::move(iter_vars.at(i)),
                    expr(0), dst.get_shape()[i],
                    can_vectorize && i == output_origin_axis_vectorized
                            ? expr(static_cast<int>(step))
                            : expr(1),
                    std::move(body), true, for_type::NORMAL);
            bind_loop_axis(expand_gt, cur, i, true);
            loops.push_back(cur);
        }
        std::reverse(loops.begin(), loops.end());
        for (int i = 1; i < static_cast<int>(output_blocking_dims.size()) - 2;
                i++) {
            loops[0].checked_as<for_loop>()->fuse(
                    loops[i].checked_as<for_loop>());
        }
        bld->emit(cur);
        if (!can_vectorize) { set_const_fold_bypass(ctx, cur); }
    }
}

void compute_reorder_block(sc_graph_t &graph, const context_ptr &ctx,
        const tensor_slice &src, tensor_slice &dst,
        const sc_data_format_t &input_format,
        const sc_data_format_t &output_format, sc_data_type_t dtype,
        const sc_dims &plain_dims, bool output_loop, any_map_t &attrs,
        const graph_tensor_ptr &expand_gt, size_t wkld = 0UL,
        bool is_innermost_dim_strided = false, bool is_dynamic = false,
        int impl_alg = 0) {
    COMPILE_ASSERT(input_format.is_convertible(output_format),
            "Can not convert input format "
                    << input_format << " to output format " << output_format
                    << ".");

    std::vector<int> inp_a_axis, inp_b_axis, out_a_axis, out_b_axis;

    bool is_vnni_reorder = false, vnni_usex16step = false;
    bool dynamic_no_padding = impl_alg & impl_kind_t::no_padding;
    sc_trans_kernel trans_kernel_used;
    sc_vnni_kernel vnni_kernel_used;
    if (!is_innermost_dim_strided
            && can_be_vnni_reorder(ctx, inp_a_axis, inp_b_axis, out_a_axis,
                    out_b_axis, plain_dims, input_format, output_format, src,
                    dst, dtype, is_vnni_reorder, is_dynamic, dynamic_no_padding,
                    vnni_kernel_used)) {
        compute_vnni_reorder(graph, ctx, src, dst, input_format, output_format,
                dtype, plain_dims, output_loop, attrs, inp_a_axis, inp_b_axis,
                out_a_axis, out_b_axis, expand_gt, wkld, is_vnni_reorder,
                is_dynamic, dynamic_no_padding, vnni_kernel_used);
    } else if (!is_innermost_dim_strided
            && can_be_fast_transpose(graph, ctx, inp_a_axis, inp_b_axis,
                    out_a_axis, out_b_axis, plain_dims, input_format,
                    output_format, src, dst, dtype, is_dynamic,
                    dynamic_no_padding, trans_kernel_used)) {
        compute_fast_transpose(graph, ctx, src, dst, input_format,
                output_format, dtype, plain_dims, output_loop, attrs,
                inp_a_axis, inp_b_axis, out_a_axis, out_b_axis, expand_gt, wkld,
                is_dynamic, dynamic_no_padding, trans_kernel_used);
    } else if (is_not_blocking(input_format)
            && is_not_blocking(output_format)) {
        compute_reorder_stride2stride(graph, ctx, src, dst, input_format,
                output_format, dtype, plain_dims, output_loop, attrs, expand_gt,
                wkld, is_innermost_dim_strided, is_dynamic, dynamic_no_padding);
    } else if (is_not_blocking(input_format) && output_format.is_blocking()) {
        compute_reorder_stride2block(graph, ctx, src, dst, input_format,
                output_format, dtype, plain_dims, output_loop, attrs, expand_gt,
                wkld, is_innermost_dim_strided, is_dynamic, dynamic_no_padding);
    } else if (input_format.is_blocking() && is_not_blocking(output_format)) {
        compute_reorder_block2stride(graph, ctx, src, dst, input_format,
                output_format, dtype, plain_dims, attrs, expand_gt, wkld,
                is_innermost_dim_strided, is_dynamic, dynamic_no_padding);
    } else if (input_format.is_blocking() && output_format.is_blocking()) {
        compute_reorder_block2block(graph, ctx, src, dst, input_format,
                output_format, dtype, plain_dims, output_loop, attrs, expand_gt,
                wkld, is_innermost_dim_strided, is_dynamic, dynamic_no_padding);
    } else {
        std::ostringstream ss;
        ss << "Unsupported data format. in = " << input_format
           << ", out = " << output_format;
        SC_MODULE_WARN << ss.str();
        throw tuner_recoverable_exception_t(ss.str());
    }
}

size_t reorder_op_t::compute_workload(const std::vector<shape_dtype_pair> &ins,
        const std::vector<shape_dtype_pair> &outs) {
    return fusible_op_t::compute_workload(ins, outs)
            * workload_penalty_coefficient;
}

bool reorder_op_t::check_padding() const {
    auto &input_format = get_input_format();
    auto &output_format = get_output_format();
    auto input_blocking_dims
            = sc_data_format_t::get_blocking_shapes(plain_dims_, input_format);
    auto output_blocking_dims
            = sc_data_format_t::get_blocking_shapes(plain_dims_, output_format);
    bool is_dynamic_block = is_dynamic_blocking(plain_dims_, input_format);
    is_dynamic_block |= is_dynamic_blocking(plain_dims_, output_format);
    return (!is_dynamic_block
                   && sc_data_format_t::get_padded_plain_shapes(
                              input_blocking_dims, input_format)
                           != sc_data_format_t::get_padded_plain_shapes(
                                   output_blocking_dims, output_format))
            || (is_dynamic_block
                    && !(info_.cur_impl_ & impl_kind_t::no_padding));
}

bool reorder_op_t::use_output_loop() const {
    if (attrs_.get_or_else("use_input_loop", false)) { return false; }
    if (check_padding()) {
        auto &input_format = get_input_format();
        auto &output_format = get_output_format();
        // block->stride
        if (input_format.is_blocking() && is_not_blocking(output_format))
            return false;
        else if (input_format.is_blocking() && output_format.is_blocking()) {
            // block->block: check the products of blocking dims whether
            // same?
            return (get_dims_product(
                            get_inputs()[0]->details_.get_blocking_dims())
                    < get_dims_product(
                            get_outputs()[0]->details_.get_blocking_dims()));
        }
        return true;
    }
    if (attrs_.get_or_else(op_attr_key::no_fuse, false)) {
        if (!get_input_format().is_blocking()) return true;
    }
    if (attrs_.get_or_else(op_attr_key::break_pre_fuse, false)) return true;
    return false;
}

bool reorder_op_t::support_output_loop() const {
    return is_not_blocking(get_input_format())
            || get_output_format().is_blocking();
}

#define INIT_REORDER_OP_INFO() \
    bool is_innermost_dim_strided \
            = info_.inputs_[0]->details_.get_strides().back() != 1 \
            || info_.outputs_[0]->details_.get_strides().back() != 1; \
    auto &input_format = info_.inputs_[0]->details_.get_format(); \
    auto &output_format = info_.outputs_[0]->details_.get_format(); \
    auto dtype = info_.inputs_[0]->details_.dtype_; \
    auto input_blocking_shapes_expr = get_blocking_shapes_expr( \
            get_owner_graph(), plain_dims_, input_format); \
    auto output_blocking_shapes_expr = get_blocking_shapes_expr( \
            get_owner_graph(), plain_dims_, output_format); \
    sc_graph_t g; \
    auto toy_inp_tsr = builder::make_tensor( \
            std::string("dummy_inp"), input_blocking_shapes_expr, dtype); \
    auto toy_out_tsr = builder::make_tensor( \
            std::string("dummy_out"), output_blocking_shapes_expr, dtype); \
    auto src = tensor_slice(toy_inp_tsr), dst = tensor_slice(toy_out_tsr); \
    std::vector<int> inp_a_axis, inp_b_axis, out_a_axis, out_b_axis; \
    bool is_vnni_reorder = false; \
    sc_trans_kernel trans_kernel_used; \
    sc_vnni_kernel vnni_kernel_used;

bool reorder_op_t::support_optimized_kernel(const context_ptr &ctx) const {
    INIT_REORDER_OP_INFO()
    int trans_inp_a_axis = 0, trans_inp_b_axis = 0, trans_out_a_axis = 0,
        trans_out_b_axis = 0;
    return (!is_innermost_dim_strided
                   && can_be_fast_transpose(get_owner_graph(), ctx, inp_a_axis,
                           inp_b_axis, out_a_axis, out_b_axis, plain_dims_,
                           input_format, output_format, src, dst, dtype,
                           is_dynamic(),
                           info_.cur_impl_ & impl_kind_t::no_padding,
                           trans_kernel_used))
            || can_be_vnni_reorder(ctx, inp_a_axis, inp_b_axis, out_a_axis,
                    out_b_axis, plain_dims_, input_format, output_format, src,
                    dst, dtype, is_vnni_reorder, is_dynamic(),
                    info_.cur_impl_ & impl_kind_t::no_padding,
                    vnni_kernel_used);
}

bool reorder_op_t::meet_vnni_reorder_require(const context_ptr &ctx) const {
    INIT_REORDER_OP_INFO()
    return !is_innermost_dim_strided
            && can_be_vnni_reorder(ctx, inp_a_axis, inp_b_axis, out_a_axis,
                    out_b_axis, plain_dims_, input_format, output_format, src,
                    dst, dtype, is_vnni_reorder, is_dynamic(),
                    info_.cur_impl_ & impl_kind_t::no_padding,
                    vnni_kernel_used);
}

void reorder_op_t::compute_block(context_ptr ctx,
        const std::vector<tensor_slice *> &dst,
        const std::vector<const tensor_slice *> &inputs) {
    bool is_innermost_dim_strided = !is_dynamic()
            && (info_.inputs_[0]->details_.get_strides().back() != 1
                    || info_.outputs_[0]->details_.get_strides().back() != 1);
    size_t wkld = compute_fusible_workload(ctx, dst, inputs);
    auto &input_format = info_.inputs_[0]->details_.get_format();
    auto &output_format = info_.outputs_[0]->details_.get_format();
    bool output_loop = use_output_loop();
    compute_reorder_block(get_owner_graph(), ctx, *inputs[0], *dst[0],
            input_format, output_format, info_.inputs_[0]->details_.dtype_,
            plain_dims_, output_loop, attrs_,
            output_loop ? get_outputs()[0] : get_inputs()[0], wkld,
            is_innermost_dim_strided, is_dynamic(), info_.cur_impl_);
}

std::vector<int> reorder_op_t::get_impl_dispatch_candidates(
        const context_ptr &ctx) {
    return get_default_impl_dispatch_candidates();
}

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
