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

#include <assert.h>

#include <utility>
#include "compiler/ir/builtin.hpp"
#include "memory_movement.hpp"
#include <compiler/ir/builder.hpp>
#include <compiler/ir/graph/dynamic_dispatch_key.hpp>
#include <compiler/ir/graph/fusible_op_utils.hpp>
#include <compiler/ir/graph/fusion_mgr.hpp>
#include <compiler/ir/graph/outer_loop_generator.hpp>
#include <compiler/ir/graph/tunable_op.hpp>
#include <compiler/ir/graph/utils.hpp>
#include <compiler/ir/transform/auto_cast.hpp>
#include <compiler/ir/transform/constant_fold.hpp>
#include <runtime/dynamic_dispatch/ops/impl_type.hpp>
#include <unordered_map>
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

ir_module_ptr reorder_op_t::get_func(context_ptr ctx) {
    attrs_.set(op_attr_key::no_fuse, true);
    // if the reorder is tensor view in dynamic, do inplacement.
    if (is_dynamic_reorder_inplace(this, ctx)) {
        return inplaced_reorder_get_func(this, ctx);
    }
    if (ctx->flags_.mixed_fusion_) return fusible_op_get_func(this, ctx);
    top_level_anchor_generator_t gen;
    auto ret = fusible_op_get_func(this, gen, ctx, false);
    auto func = ret->get_entry_func();
    auto body = func->body_.as<stmts>();
    COMPILE_ASSERT(body.defined(), "Expecting a body");
    COMPILE_ASSERT(body->seq_.size() <= 2, "Expecting 2 stmt in reorder body");
    auto loop = body->seq_[0].as<for_loop>();
    COMPILE_ASSERT(loop.defined(), "Expecting a for loop in reorder body");
    loop->kind_ = for_type::PARALLEL;
    return ret;
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
        // has broadcast uses, do not fuse them as their outer loop can not be
        // fused.
        for (auto &use : get_outputs()[0]->uses_) {
            if (auto may_bcst
                    = use.second->dyn_cast<op_traits::may_broadcast_t>()) {
                if (may_bcst->get_broadcast_input() != -1) {
                    attrs_.set(op_attr_key::break_post_fuse, true);
                }
            }
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
    }
    op_name_ = "reorder";
    attrs_ = attrs;
    plain_dims_ = ins[0]->details_.get_plain_dims();
    COMPILE_ASSERT(info_.inputs_[0]->details_.get_format().is_convertible(
                           info_.outputs_[0]->details_.get_format()),
            "input format " << info_.inputs_[0]->details_.get_format()
                            << " can not convert to "
                            << info_.outputs_[0]->details_.get_format() << ".");
    if (!is_dynamic()) {
        if (use_output_loop()) {
            attrs_.set(op_attr_key::break_pre_fuse, true);
        }
    }
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

void reorder_op_t::prepare_fusion_data(fdata_map &fdmap) {
    auto &in_detail0 = fdmap.get(info_.inputs_[0]);
    in_detail0.use_count_++;
}

sc_dims reorder_op_t::get_bwise_fuse_shrink_dims() {
    if (check_padding()) return {};
    bool use_out_loop = use_output_loop();
    // depends on loop mode
    auto gt = use_out_loop ? get_outputs()[0] : get_inputs()[0];
    int offset = op_traits::batchwise_shrinkable_t::get_shrinkable_offset(
            gt, false);
    auto fmt = gt->details_.get_format();
    // check aother gt legalize
    auto gt_blocks = fmt.get_blocked_axis();
    auto another_gt = use_out_loop ? get_inputs()[0] : get_outputs()[0];
    auto another_fmt = another_gt->details_.get_format();
    auto another_gt_blocks = another_fmt.get_blocked_axis();
    auto p2b_map = another_fmt.format_code_.collect_p2b_mapping();
    int cnt = 0, no_stride_cnt = 0;
    bool bwise_strided = false;
    for (; cnt < offset; cnt++) {
        auto plain_pos = fmt.format_code_.get(cnt);
        // check can shrink another gt
        if (gt_blocks[plain_pos].empty()
                && !another_gt_blocks[plain_pos].empty())
            break;
        if (!gt_blocks[plain_pos].empty()
                && !another_gt_blocks[plain_pos].empty()) {
            auto gt_remaining_dims_prod = gt_blocks[plain_pos].front();
            auto another_gt_blocks_prod = another_gt_blocks[plain_pos].front();
            if (gt_remaining_dims_prod < another_gt_blocks_prod
                    || gt_remaining_dims_prod % another_gt_blocks_prod != 0)
                break;
        }
        // check strided
        if (!bwise_strided) {
            if (p2b_map[plain_pos].front() != cnt)
                bwise_strided = true;
            else if (cnt > 0) {
                if (!gt_blocks[plain_pos].empty()
                        && another_gt_blocks[plain_pos].empty()) {
                    bwise_strided = true;
                } else if (!gt_blocks[plain_pos].empty()
                        && !another_gt_blocks[plain_pos].empty()) {
                    auto gt_remaining_dims_prod = gt_blocks[plain_pos].front();
                    auto another_gt_blocks_prod
                            = another_gt_blocks[plain_pos].front();
                    if (gt_remaining_dims_prod != another_gt_blocks_prod)
                        bwise_strided = true;
                }
            }
            if (bwise_strided) no_stride_cnt = cnt;
        }
    }
    if (bwise_strided) {
        attrs_.set(use_out_loop ? op_attr_key::bwise_break_pre_fuse
                                : op_attr_key::bwise_break_post_fuse,
                true);
        attrs_.set(op_attr_key::bwise_no_strided_dims,
                sc_dims {gt->details_.get_blocking_dims().begin(),
                        gt->details_.get_blocking_dims().begin()
                                + no_stride_cnt});
    }
    return {gt->details_.get_blocking_dims().begin(),
            gt->details_.get_blocking_dims().begin() + cnt};
};

void reorder_op_t::collect_shrinked_lt_map(int bw_size, gt2gt_map &bw_lt_map) {
    bool use_out_loop = use_output_loop();
    auto &ins = get_inputs()[0];
    auto &out = get_outputs()[0];
    op_traits::batchwise_shrinkable_t::record_shrinked_gt(
            bw_lt_map, use_out_loop ? out : ins, bw_size);
    auto &plain_dims = bw_lt_map.get(use_out_loop ? out : ins)
                               ->details_.get_plain_dims();
    // set the another one
    op_traits::batchwise_shrinkable_t::record_shrinked_gt(
            bw_lt_map, use_out_loop ? ins : out, plain_dims);
}

void reorder_op_t::collect_shrinked_axis_map(
        int bw_size, gt2axis_map &bw_axis_map) {
    bool use_out_loop = use_output_loop();
    // depends on loop mode
    auto gt = use_out_loop ? get_outputs()[0] : get_inputs()[0];
    record_shrinked_axis(bw_axis_map, gt, bw_size);
    auto another_gt = use_out_loop ? get_inputs()[0] : get_outputs()[0];
    auto fmt = gt->details_.get_format();
    auto p2b_map = another_gt->details_.get_format()
                           .format_code_.collect_p2b_mapping();
    std::vector<int> bw_axis;
    bw_axis.reserve(bw_size);
    for (int i = 0; i < bw_size; i++) {
        bw_axis.emplace_back(p2b_map[fmt.format_code_.get(i)].front());
    }
    record_shrinked_axis(bw_axis_map, another_gt, bw_axis);
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
            constant_folder_t f;
            auto_caster_t ca;
            auto plain_range
                    = std::make_pair(f(ca((block_num_start + expr(i)) * blocks
                                               + block_size_start))
                                             .remove_const(),
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

static size_t throw_if_negative(int dim) {
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
    constant_folder_t f;
    auto_caster_t ca;
    for (auto &reorder_range : output_slice_list) {
        for (auto &r : reorder_range) {
            r.first = ca(r.first).remove_const();
        }
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

#define SLICE_RAGNE_CHECK_INIT_DATA() \
    bool use_out_loop = use_output_loop(); \
    sc_data_format_t target_format \
            = use_out_loop ? output_format : input_format; \
    auto blocking_exprs = get_blocking_shapes_expr( \
            get_owner_graph(), plain_dims_, target_format); \
    auto block_axis = target_format.format_code_.get( \
            target_format.format_code_.ndims() - 1); \
    int origin_axis_vectorized = target_format.format_code_.ndims() - 1; \
    find_vectorized_axis(blocking_exprs, target_format, block_axis, \
            origin_axis_vectorized); \
    int len_from_last \
            = target_format.format_code_.ndims() - 1 - origin_axis_vectorized; \
    bool must_recheck = len_from_last > 0; \
    bool optimized_slice_check = !stat_map.is_recursive_mode() \
            && support_optimized_kernel(stat_map.get_context()); \
    bool special_slice_check = !stat_map.is_recursive_mode() \
            && (support_optimized_kernel(stat_map.get_context()) \
                    || must_recheck); \
    len_from_last += optimized_slice_check \
            ? meet_vnni_reorder_require(stat_map.get_context()) ? 3 : 2 \
            : 1;

// infer reorder slice according input_slice
void reorder_op_t::infer_slice_ranges(
        fslice_map &fsmap, infer_status_map_t &stat_map) {
    // has been pre-inferred, skip
    if (!fsmap.get(get_outputs()[0]).empty()) return;
    auto &input_format = get_input_format();
    auto &output_format = get_output_format();
    COMPILE_ASSERT(input_format.is_convertible(output_format),
            "Can not convert input format "
                    << input_format << " to output format " << output_format
                    << ".");
    // search known ranges from any input of cur fusbile op
    slice_range_map known_ranges_map
            = search_known_slice_ranges(this, fsmap, stat_map);

    std::vector<int> required_axis(
            get_inputs()[0]->details_.get_blocking_dims().size(), 0);
    for (auto i = 0UL; i < get_inputs()[0]->details_.get_blocking_dims().size();
            i++) {
        required_axis[i] = i;
    }
    for (auto &src_range : fsmap.get(get_inputs()[0])) {
        if (!get_inputs()[0]->details_.is_dynamic()
                && !slice_divisible_on_axis(
                        get_inputs()[0]->details_.get_blocking_dims(),
                        src_range, required_axis)) {
            stat_map.append_ops_by_status(this, infer_status_code::RETRY);
            return;
        }
    }

    if (known_ranges_map.empty()) return;
    auto input_slice_list = known_ranges_map[0];

    SLICE_RAGNE_CHECK_INIT_DATA()
    if (special_slice_check
            && !check_required_slice(
                    get_inputs()[0], input_slice_list, len_from_last)) {
        stat_map.append_ops_by_status(this, infer_status_code::RETRY);
        return;
    }

    slice_range_list reorder_ranges_list;
    // infer reorder slice only makes sense for non-padding cases in new fusion
    // mgr
    if (is_dynamic() || !check_padding()) {
        infer_reorder_slice(input_slice_list, input_format, output_format,
                reorder_ranges_list);
    }

    // TODO(yunfei) remove this if scope when old fusion mgr is deprecated
    if (reorder_ranges_list.empty()) {
        for (auto &user : get_outputs()[0]->uses_) {
            if (user.second->isa<output_op>()) {
                continue;
            } else if (stat_map.is_recursive_mode() && !check_padding()) {
                user.second->attrs_.set(op_attr_key::fused_mode_hint,
                        op_attr_key::break_pre_fuse);
                stat_map.append_ops_by_status(
                        user.second.get(), infer_status_code::FAIL);
                return;
            }
        }
    } else {
        if (optimized_slice_check
                && !check_required_slice(
                        get_outputs()[0], reorder_ranges_list, len_from_last)) {
            stat_map.append_ops_by_status(this, infer_status_code::RETRY);
            return;
        }
    }
    fsmap.get(get_outputs()[0]) = reorder_ranges_list;
}

// pre-infer reorder slice according output_slice
void reorder_op_t::pre_slice_ranges(
        fslice_map &fsmap, infer_status_map_t &stat_map) {
    slice_range_list known_ranges_list = fsmap.get(get_outputs()[0]);
    auto &input_format = get_input_format();
    auto &output_format = get_output_format();
    SLICE_RAGNE_CHECK_INIT_DATA()
    // deal with begining reorder op, which use output loop
    if (!stat_map.is_recursive_mode() && fsmap.datamap_.size() == 1) {
        if (!use_output_loop()
                || (special_slice_check
                        && !check_required_slice(get_outputs()[0],
                                known_ranges_list, len_from_last))) {
            stat_map.append_ops_by_status(this, infer_status_code::RETRY);
        } else {
            // set input slice for anchor check
            fsmap.get(get_inputs()[0])
                    = slice_range_list {gen_slice_by_dims_expr(
                            get_inputs()[0]->details_.get_blocking_dims_expr(
                                    get_owner_graph()))};
        }
        return;
    }
    if (is_dynamic()) {
        stat_map.append_ops_by_status(this, infer_status_code::RETRY);
        return;
    }
    if (fsmap.get(get_inputs()[0]).empty()) {
        if (check_padding()) {
            if (!use_output_loop() || stat_map.is_recursive_mode()) {
                stat_map.append_ops_by_status(this, infer_status_code::RETRY);
            }
            return;
        }
        slice_range_list input_slice_list;
        infer_reorder_slice(known_ranges_list, get_output_format(),
                get_input_format(), input_slice_list);
        if (input_slice_list.size() != 1 || !support_output_loop()) {
            stat_map.append_ops_by_status(this, infer_status_code::RETRY);
            return;
        }
        fsmap.get(get_inputs()[0]) = input_slice_list;
        // recursively pre-infer
        info_.inputs_[0]
                ->producer_owner_->dyn_cast<fusible_op_t>()
                ->pre_slice_ranges(fsmap, stat_map);
    }
}

void reorder_op_t::infer_binding_axis(bound_axis_map &bdax_map) {
    infer_identical_binding_axis(this, bdax_map);
}

void reorder_op_t::pre_binding_axis(bound_axis_map &bdax_map) {
    pre_identical_binding_axis(this, bdax_map);
}

static std::vector<expr> get_reorder_stride2stride_indexes(
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

const int TARGET_AXIS_NOT_DEFINE = -1;
static std::vector<expr> get_reorder_block2plain_indexes(sc_graph_t &graph,
        const std::vector<expr> &in_indexes, const sc_data_format_t &format,
        const sc_dims &plain_dims, expr &condition, expr &last_axis_offset,
        expr &other_axis_condition,
        const int target_axis = TARGET_AXIS_NOT_DEFINE) {
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

static std::vector<expr> get_reorder_plain2block_indexes(
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
bool is_valid_step(int step) {
    return utils::is_one_of(step, 4, 8, 16, 32, 64);
}
void compute_reorder_stride2stride(sc_graph_t &graph, const context_ptr &ctx,
        const tensor_slice &src, tensor_slice &dst,
        const sc_data_format_t &input_format,
        const sc_data_format_t &output_format, sc_data_type_t dtype,
        const sc_dims &plain_dims, bool output_loop, any_map_t &attrs,
        size_t wkld = 0UL, bool is_innermost_dim_strided = false,
        bool is_dynamic = false, bool dynamic_no_padding = false) {
    auto input = src.get_real_tensor();
    auto output = dst.get_real_tensor();
    int step = static_cast<int>(vectorize_step(ctx, dtype.type_code_));
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

    bool can_vectorize = !is_innermost_dim_strided
            && input_last_origin_axis == output_last_origin_axis
            && step * utils::get_sizeof_type(dtype) * byte >= avx_simd_length;
    if (!can_vectorize) {
        cannot_convert_warning(input_format, output_format, plain_dims);
    }
    step = can_vectorize ? step : 1;
    bool no_padding = is_dynamic && dynamic_no_padding;
    if (!output_loop) {
        no_padding |= !is_dynamic
                && input_blocking_dims[input_origin_axis_vectorized] % step
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
            auto idx_len = cast_to_s32(input_blocking_dims_expr
                                           [input_origin_axis_vectorized])
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
            loops.push_back(cur);
        }
        std::reverse(loops.begin(), loops.end());
        for (int i = 0; i < static_cast<int>(plain_dims.size()) - 2; i++) {
            loops[0].checked_as<for_loop>()->fuse(
                    loops[i].checked_as<for_loop>());
        }
        bld->emit(cur);
        if (!can_vectorize) { set_const_fold_bypass(ctx, cur); }
    } else {
        no_padding |= !is_dynamic
                && output_blocking_dims[output_origin_axis_vectorized] % step
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
            auto idx_len = cast_to_s32(output_blocking_dims_expr
                                           [output_origin_axis_vectorized])
                    - cast_to_s32(iter_vars[input_origin_axis_vectorized]);
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
            loops.push_back(cur);
        }
        std::reverse(loops.begin(), loops.end());
        for (int i = 0; i < static_cast<int>(plain_dims.size()) - 2; i++) {
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
        const sc_dims &plain_dims, any_map_t &attrs, size_t wkld = 0UL,
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

    int max_step = ctx->get_max_vector_lanes(dtype.type_code_);
    int step = std::min(static_cast<int>(vectorize_step(ctx, dtype.type_code_)),
            static_cast<int>(
                    input_blocking_dims[input_origin_axis_vectorized]));
    if (attrs.get_or_else(op_attr_key::no_fuse, false)) {
        while (step < max_step
                && input_blocking_dims[input_origin_axis_vectorized]
                                % (2 * step)
                        == 0) {
            step = 2 * step;
        }
    }
    bool can_vectorize = !is_innermost_dim_strided
            && input_last_origin_axis == output_last_origin_axis
            && input_blocking_dims[input_origin_axis_vectorized] % step == 0
            && is_valid_step(step)
            && step * utils::get_sizeof_type(dtype) * 8 >= 128;

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
        iter_vars.emplace_back(builder::make_var(datatypes::index,
                std::string("_fuseiter") + fusion_create_idx()));
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
        body = cur.isa<stmts>()
                ? cur
                : make_stmt<stmts_node_t>(std::vector<stmt> {std::move(cur)});
        cur = make_stmt<for_loop_node_t>(std::move(iter_vars.at(i)), expr(0),
                src.get_shape()[i],
                can_vectorize && i == input_origin_axis_vectorized
                        ? expr(static_cast<int>(step))
                        : expr(1),
                std::move(body), true, for_type::NORMAL);
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
        size_t wkld = 0UL, bool is_innermost_dim_strided = false,
        bool is_dynamic = false, bool dynamic_no_padding = false) {
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

    int max_step = ctx->get_max_vector_lanes(dtype.type_code_);
    int step = std::min(static_cast<int>(vectorize_step(ctx, dtype.type_code_)),
            static_cast<int>(
                    output_blocking_dims[output_origin_axis_vectorized]));
    if (attrs.get_or_else(op_attr_key::no_fuse, false)) {
        while (step < max_step
                && output_blocking_dims[output_origin_axis_vectorized]
                                % (2 * step)
                        == 0) {
            step = 2 * step;
        }
    }

    bool can_vectorize = !is_innermost_dim_strided
            && input_last_origin_axis == output_last_origin_axis
            && output_blocking_dims[output_origin_axis_vectorized] % step == 0
            && is_valid_step(step)
            && step * utils::get_sizeof_type(dtype) * 8 >= 128;
    // Usually use input loop means no padding in static, but not in dynamic, if
    // dynamic and use input loop, need to check the static dim with blocks.
    if (!output_loop && !is_dynamic_dim(input_blocking_dims.back())
            && input_blocking_dims[input_origin_axis_vectorized] % step != 0) {
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
            iter_vars.emplace_back(builder::make_var(datatypes::index,
                    std::string("_fuseiter") + fusion_create_idx()));
            in_indexes.emplace_back(iter_vars[i] + src.get_offset()[i]);
            loop_indexes.emplace_back(iter_vars[i]);
        }
        expr condition;
        std::vector<expr> tmp_out_indexes = get_reorder_stride2stride_indexes(
                in_indexes, input_format, input_format.to_plain(), plain_dims);
        std::vector<expr> out_indexes = get_reorder_plain2block_indexes(
                tmp_out_indexes, output_format);

        auto assign = builder::make_stmts_unattached(
                {builder::make_assign_unattached(
                        builder::make_indexing(output, out_indexes, step),
                        // here, use src.tptr instead of input is aimed to avoid
                        // input is tensor_view_op. Oherwisw, it will throw
                        // illegal exception in index_flatten
                        builder::make_indexing(
                                src.tptr_, loop_indexes, step))});
        assign->attr()[op_traits::workload_computable_t::workload_number]
                = wkld;
        auto cur = assign;
        stmt body;
        for (int i = static_cast<int>(input_blocking_dims.size()) - 1; i >= 0;
                i--) {
            body = cur.isa<stmts>() ? cur
                                    : make_stmt<stmts_node_t>(
                                            std::vector<stmt> {std::move(cur)});
            cur = make_stmt<for_loop_node_t>(std::move(iter_vars.at(i)),
                    expr(0), src.get_shape()[i],
                    can_vectorize && i == input_origin_axis_vectorized
                            ? expr(static_cast<int>(step))
                            : expr(1),
                    std::move(body), true, for_type::NORMAL);
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
            body = cur.isa<stmts>() ? cur
                                    : make_stmt<stmts_node_t>(
                                            std::vector<stmt> {std::move(cur)});
            cur = make_stmt<for_loop_node_t>(std::move(iter_vars.at(i)),
                    expr(0),
                    // if the offset of dst is given(commit op)
                    (no_padding || !dst.get_offset()[i].isa<constant>())
                            ? dst.get_shape()[i]
                            : output_blocking_exprs[i],
                    can_vectorize && i == output_origin_axis_vectorized
                            ? expr(static_cast<int>(step))
                            : expr(1),
                    std::move(body), true, for_type::NORMAL);
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
        size_t wkld = 0UL, bool is_innermost_dim_strided = false,
        bool is_dynamic = false, bool dynamic_no_padding = false) {
    auto input = src.get_real_tensor();
    auto output = dst.get_real_tensor();
    int step = static_cast<int>(vectorize_step(ctx, dtype.type_code_));
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
    bool no_padding = !is_dynamic
            && input_padded_plain_dims == output_padded_plain_dims;
    no_padding |= (is_dynamic && dynamic_no_padding);
    bool can_vectorize = !is_innermost_dim_strided
            && input_block_axis == output_block_axis
            && output_blocking_dims[output_origin_axis_vectorized] % step == 0
            && input_blocking_dims[input_origin_axis_vectorized] % step == 0
            && plain_dims[input_block_axis]
                            % input_blocking_dims[input_origin_axis_vectorized]
                    == 0
            && plain_dims[output_block_axis]
                            % output_blocking_dims
                                    [output_origin_axis_vectorized]
                    == 0;

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
            iter_vars.emplace_back(builder::make_var(datatypes::index,
                    std::string("_fuseiter") + fusion_create_idx()));
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
            body = cur.isa<stmts>() ? cur
                                    : make_stmt<stmts_node_t>(
                                            std::vector<stmt> {std::move(cur)});
            cur = make_stmt<for_loop_node_t>(std::move(iter_vars.at(i)),
                    expr(0), src.get_shape()[i],
                    can_vectorize && i == input_origin_axis_vectorized
                            ? expr(static_cast<int>(step))
                            : expr(1),
                    std::move(body), true, for_type::NORMAL);
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
                    expr(0),
                    (no_padding || !dst.get_offset()[i].isa<constant>())
                            ? dst.get_shape()[i]
                            : output_blocking_exprs[i],
                    can_vectorize && i == output_origin_axis_vectorized
                            ? expr(static_cast<int>(step))
                            : expr(1),
                    std::move(body), true, for_type::NORMAL);
            loops.push_back(cur);
        }
        std::reverse(loops.begin(), loops.end());
        for (int i = 0; i < static_cast<int>(output_blocking_dims.size()) - 2;
                i++) {
            loops[0].checked_as<for_loop>()->fuse(
                    loops[i].checked_as<for_loop>());
        }
        bld->emit(cur);
        if (!can_vectorize) { set_const_fold_bypass(ctx, cur); }
    }
}

static bool whole_buffer_reorder(const tensor_slice &src) {
    for (auto &offset : src.get_offset()) {
        if (!offset.isa<constant>() || get_expr_as_int(offset) != 0) {
            return false;
        }
    }
    return true;
}
// currently only support f32 8x8 and bf16 32x8
static const int trans_lanes = 8;
static const int trans_lanes_bf16 = 32;
// [..., a, ... , b] <=> [..., b, ..., a]
static bool can_be_fast_transpose(const context_ptr &ctx,
        std::vector<int> &inp_a_axis, std::vector<int> &inp_b_axis,
        std::vector<int> &out_a_axis, std::vector<int> &out_b_axis,
        const sc_dims &plain_dims, const sc_data_format_t &input_format,
        const sc_data_format_t &output_format, const tensor_slice &src,
        const tensor_slice &dst, const sc_data_type_t &dtype, bool is_dynamic,
        bool dynamic_no_padding) {
    if (!ctx->machine_.cpu_flags_.fAVX2) { return false; }
    if (!dtype.is_etype(sc_data_etype::F32)
            && !dtype.is_etype(sc_data_etype::BF16)) {
        return false;
    }
    inp_a_axis.clear();
    inp_b_axis.clear();
    out_a_axis.clear();
    out_b_axis.clear();
    int inp_idx = 0, out_idx = 0;
    auto &inp_code = input_format.format_code_;
    auto &out_code = output_format.format_code_;
    int input_ndims = input_format.format_code_.ndims();
    int output_ndims = output_format.format_code_.ndims();
    auto input_blocking_shapes
            = sc_data_format_t::get_blocking_shapes(plain_dims, input_format);
    auto output_blocking_shapes
            = sc_data_format_t::get_blocking_shapes(plain_dims, output_format);
    auto inp_b_idx = inp_code.get(input_ndims - 1);
    auto out_a_idx = out_code.get(output_ndims - 1);
    if (inp_b_idx == out_a_idx) { return false; }
    while (inp_idx < input_ndims && out_idx < output_ndims) {
        while (inp_idx < input_ndims
                && utils::is_one_of(
                        inp_code.get(inp_idx), out_a_idx, inp_b_idx)) {
            if (inp_code.get(inp_idx) == out_a_idx) {
                inp_a_axis.push_back(inp_idx);
            } else {
                inp_b_axis.push_back(inp_idx);
            }
            inp_idx++;
        }
        while (inp_idx + 1 < input_ndims
                && inp_code.get(inp_idx + 1) == inp_code.get(inp_idx)) {
            inp_idx++;
        }
        while (out_idx < output_ndims
                && utils::is_one_of(
                        out_code.get(out_idx), out_a_idx, inp_b_idx)) {
            if (out_code.get(out_idx) == out_a_idx) {
                out_a_axis.push_back(out_idx);
            } else {
                out_b_axis.push_back(out_idx);
            }
            out_idx++;
        }
        while (out_idx + 1 < output_ndims
                && out_code.get(out_idx + 1) == out_code.get(out_idx)) {
            out_idx++;
        }
        auto orig_inp_idx = inp_code.get(inp_idx);
        auto orig_out_idx = out_code.get(out_idx);
        // other axis should be in same order.
        if (orig_inp_idx != orig_out_idx) { return false; }
        inp_idx++;
        out_idx++;
    }
    // input or output not end
    if (inp_idx < input_ndims || out_idx < output_ndims) { return false; }
    // number of non-transpose axis should be equal
    if (static_cast<size_t>(input_ndims) - inp_a_axis.size() - inp_b_axis.size()
            != static_cast<size_t>(output_ndims) - out_a_axis.size()
                    - out_b_axis.size()) {
        return false;
    }
    if (input_format.is_blocking() && output_format.is_blocking()) {
        // The transpose dimension in the output shape should be an integer
        // multiple of the input shape in block2block case.
        int ix = inp_a_axis[inp_a_axis.size() - 1];
        int iy = inp_b_axis[inp_b_axis.size() - 1];
        int ox = out_a_axis[out_a_axis.size() - 1];
        int oy = out_b_axis[out_b_axis.size() - 1];
        // can't do it in dynamic cases
        if (!src.shape_[ix].isa<constant>() || !src.shape_[iy].isa<constant>()
                || !dst.shape_[ox].isa<constant>()
                || !dst.shape_[oy].isa<constant>())
            return false;
        int inp_x = get_expr_as_int(src.shape_[ix]);
        int inp_y = get_expr_as_int(src.shape_[iy]);
        int out_x = get_expr_as_int(dst.shape_[ox]);
        int out_y = get_expr_as_int(dst.shape_[oy]);

        if (out_x % inp_x != 0 || out_y % inp_y != 0) { return false; }
    }
    auto satisfy_dim_lanes = [&]() {
        int trans_lanes1
                = dtype == datatypes::bf16 ? trans_lanes_bf16 : trans_lanes;
        int trans_lanes2 = trans_lanes;
        return plain_dims[inp_b_idx] % trans_lanes2 == 0
                && plain_dims[out_a_idx] % trans_lanes1 == 0
                && get_expr_as_int(
                           src.shape_[inp_a_axis[inp_a_axis.size() - 1]])
                        % trans_lanes1
                == 0
                && get_expr_as_int(
                           dst.shape_[out_b_axis[out_b_axis.size() - 1]])
                        % trans_lanes2
                == 0
                && get_expr_as_int(src.shape_[input_blocking_shapes.size() - 1])
                        % trans_lanes2
                == 0
                && get_expr_as_int(
                           dst.shape_[output_blocking_shapes.size() - 1])
                        % trans_lanes1
                == 0;
    };
    // currently does not support tensor slice with padding.
    if (!whole_buffer_reorder(src) && (is_dynamic || !satisfy_dim_lanes())) {
        return false;
    }
    return true;
}

// unpack and interleave
#define TRANS2D_UNPACK_ASSIGN(option, dst, src1, src2, elem_bits) \
    cur_list.emplace_back(builder::make_assign_unattached(rows[((dst)-1)], \
            builder::make_unpack_##option( \
                    rows[((src1)-1)], rows[((src2)-1)], elem_bits)));
#define TRANS2D_SHUFFLE_PERMUTE_ASSIGN_F32( \
        command, dst, src1, src2, mask, cond) \
    cur_list.emplace_back(builder::make_assign_unattached(rows[((dst)-1)], \
            builder::make_##command( \
                    rows[((src1)-1)], rows[((src2)-1)], mask)));

#define TRANS2D_REG_CALCULATION_F32() \
    TRANS2D_UNPACK_ASSIGN(low, 9, 1, 2, 32) \
    TRANS2D_UNPACK_ASSIGN(high, 1, 1, 2, 32) \
    TRANS2D_UNPACK_ASSIGN(low, 10, 3, 4, 32) \
    TRANS2D_UNPACK_ASSIGN(high, 2, 3, 4, 32) \
    TRANS2D_UNPACK_ASSIGN(low, 11, 5, 6, 32) \
    TRANS2D_UNPACK_ASSIGN(high, 3, 5, 6, 32) \
    TRANS2D_UNPACK_ASSIGN(low, 12, 7, 8, 32) \
    TRANS2D_UNPACK_ASSIGN(high, 4, 7, 8, 32) \
    TRANS2D_SHUFFLE_PERMUTE_ASSIGN_F32(shuffle, 5, 9, 10, 68, 0) \
    TRANS2D_SHUFFLE_PERMUTE_ASSIGN_F32(shuffle, 6, 9, 10, 238, 1) \
    TRANS2D_SHUFFLE_PERMUTE_ASSIGN_F32(shuffle, 7, 1, 2, 68, 2) \
    TRANS2D_SHUFFLE_PERMUTE_ASSIGN_F32(shuffle, 8, 1, 2, 238, 3) \
    TRANS2D_SHUFFLE_PERMUTE_ASSIGN_F32(shuffle, 9, 11, 12, 68, 0) \
    TRANS2D_SHUFFLE_PERMUTE_ASSIGN_F32(shuffle, 10, 11, 12, 238, 1) \
    TRANS2D_SHUFFLE_PERMUTE_ASSIGN_F32(shuffle, 11, 3, 4, 68, 2) \
    TRANS2D_SHUFFLE_PERMUTE_ASSIGN_F32(shuffle, 12, 3, 4, 238, 3) \
    TRANS2D_SHUFFLE_PERMUTE_ASSIGN_F32(permute, 1, 5, 9, 32, 0) \
    TRANS2D_SHUFFLE_PERMUTE_ASSIGN_F32(permute, 2, 6, 10, 32, 1) \
    TRANS2D_SHUFFLE_PERMUTE_ASSIGN_F32(permute, 3, 7, 11, 32, 2) \
    TRANS2D_SHUFFLE_PERMUTE_ASSIGN_F32(permute, 4, 8, 12, 32, 3) \
    TRANS2D_SHUFFLE_PERMUTE_ASSIGN_F32(permute, 5, 5, 9, 49, 4) \
    TRANS2D_SHUFFLE_PERMUTE_ASSIGN_F32(permute, 6, 6, 10, 49, 5) \
    TRANS2D_SHUFFLE_PERMUTE_ASSIGN_F32(permute, 7, 7, 11, 49, 6) \
    TRANS2D_SHUFFLE_PERMUTE_ASSIGN_F32(permute, 8, 8, 12, 49, 7)

#define TRANS2D_REG_CALCULATION_BF16() \
    TRANS2D_UNPACK_ASSIGN(low, 9, 1, 2, 16) \
    TRANS2D_UNPACK_ASSIGN(high, 10, 1, 2, 16) \
    TRANS2D_UNPACK_ASSIGN(low, 11, 3, 4, 16) \
    TRANS2D_UNPACK_ASSIGN(high, 12, 3, 4, 16) \
    TRANS2D_UNPACK_ASSIGN(low, 13, 5, 6, 16) \
    TRANS2D_UNPACK_ASSIGN(high, 14, 5, 6, 16) \
    TRANS2D_UNPACK_ASSIGN(low, 15, 7, 8, 16) \
    TRANS2D_UNPACK_ASSIGN(high, 16, 7, 8, 16) \
    TRANS2D_UNPACK_ASSIGN(low, 1, 9, 11, 32) \
    TRANS2D_UNPACK_ASSIGN(high, 2, 9, 11, 32) \
    TRANS2D_UNPACK_ASSIGN(low, 3, 10, 12, 32) \
    TRANS2D_UNPACK_ASSIGN(high, 4, 10, 12, 32) \
    TRANS2D_UNPACK_ASSIGN(low, 5, 13, 15, 32) \
    TRANS2D_UNPACK_ASSIGN(high, 6, 13, 15, 32) \
    TRANS2D_UNPACK_ASSIGN(low, 7, 14, 16, 32) \
    TRANS2D_UNPACK_ASSIGN(high, 8, 14, 16, 32) \
    TRANS2D_UNPACK_ASSIGN(low, 9, 1, 5, 64) \
    TRANS2D_UNPACK_ASSIGN(high, 10, 1, 5, 64) \
    TRANS2D_UNPACK_ASSIGN(low, 11, 2, 6, 64) \
    TRANS2D_UNPACK_ASSIGN(high, 12, 2, 6, 64) \
    TRANS2D_UNPACK_ASSIGN(low, 13, 3, 7, 64) \
    TRANS2D_UNPACK_ASSIGN(high, 14, 3, 7, 64) \
    TRANS2D_UNPACK_ASSIGN(low, 15, 4, 8, 64) \
    TRANS2D_UNPACK_ASSIGN(high, 16, 4, 8, 64)

static void compute_fast_transpose(sc_graph_t &graph, const context_ptr &ctx,
        const tensor_slice &src, tensor_slice &dst,
        const sc_data_format_t &input_format,
        const sc_data_format_t &output_format, sc_data_type_t dtype,
        const sc_dims &plain_dims, bool output_loop, any_map_t &attrs,
        const std::vector<int> &inp_a_axis, const std::vector<int> &inp_b_axis,
        const std::vector<int> &out_a_axis, const std::vector<int> &out_b_axis,
        size_t wkld = 0UL, bool is_dynamic = false,
        bool dynamic_no_padding = false) {
    auto input = src.get_real_tensor();
    auto output = dst.get_real_tensor();
    int step = trans_lanes; // fixed f32x8
    auto bld = builder::get_current_builder();
    auto input_blocking_dims
            = sc_data_format_t::get_blocking_shapes(plain_dims, input_format);
    auto output_blocking_dims
            = sc_data_format_t::get_blocking_shapes(plain_dims, output_format);
    auto input_blocking_dims_expr
            = get_blocking_shapes_expr(graph, plain_dims, input_format);
    auto output_blocking_dims_expr
            = get_blocking_shapes_expr(graph, plain_dims, output_format);
    auto input_format_code = input_format.format_code_;
    std::vector<expr> rows;
    std::vector<expr> iter_vars;
    std::vector<stmt_c> cur_list;
    stmt cur, body;
    bool is_padding = false;
    int inp_a_step = dtype == datatypes::f32 ? trans_lanes : trans_lanes_bf16;
    int inp_b_step = trans_lanes;
    if ((!is_dynamic
                && (input_blocking_dims[inp_a_axis.back()] % inp_a_step
                        || input_blocking_dims[inp_b_axis.back()] % inp_b_step
                        || output_blocking_dims[out_a_axis.back()] % inp_a_step
                        || output_blocking_dims[out_b_axis.back()] % inp_b_step
                        || math_utils::get_dims_product(input_blocking_dims)
                                != math_utils::get_dims_product(
                                        output_blocking_dims)))
            || (is_dynamic && !dynamic_no_padding)) {
        is_padding = true;
    }
    if (dtype == datatypes::f32) {
        rows.resize(step + step / 2);
        for (auto i = 0; i < step + step / 2; i++) {
            rows[i] = builder::make_var(sc_data_type_t::f32(step),
                    "row" + std::to_string(i + 1) + fusion_create_var_idx());
            cur_list.emplace_back(
                    builder::make_var_tensor_def_unattached(rows[i]));
        }
    } else if (dtype == datatypes::bf16) {
        rows.resize(16); // bf16 uses 16 zmms.
        for (auto i = 0; i < 16; i++) {
            rows[i] = builder::make_var(sc_data_type_t::bf16(32),
                    "row" + std::to_string(i + 1) + fusion_create_var_idx());
            cur_list.emplace_back(
                    builder::make_var_tensor_def_unattached(rows[i]));
        }
    }

    auto determine_cur_step = [&](const std::vector<expr> &blocking_dims_expr,
                                      const std::vector<expr> &tmp_in_indexes,
                                      const std::vector<expr> &plain_indexes,
                                      expr &cur_step, expr &sup_condition,
                                      int in_axis, bool use_output_loop,
                                      int step) {
        if (!use_output_loop) {
            cur_step = builder::make_min(
                    builder::make_max(cast_to_s32(blocking_dims_expr.back())
                                    - cast_to_s32(tmp_in_indexes.back()),
                            0),
                    step);
            sup_condition
                    = tmp_in_indexes[in_axis] < blocking_dims_expr[in_axis];
        } else {
            auto tmp_plain = graph.dims_to_expr(plain_dims);
            auto input_last_dim
                    = input_format_code.get(input_format_code.ndims() - 1);
            auto input_other_dim = input_format_code.get(in_axis);
            cur_step = builder::make_min(
                    builder::make_max(cast_to_s32(tmp_plain[input_last_dim])
                                    - cast_to_s32(
                                            plain_indexes[input_last_dim]),
                            0),
                    step);
            sup_condition = plain_indexes[input_other_dim]
                    < tmp_plain[input_other_dim];
        }
    };

    auto compute_transpose_f32 = [&](const std::vector<expr> &in_indexes,
                                         const std::vector<expr> &out_indexes,
                                         const std::vector<expr>
                                                 &plain_indexes) {
        for (int i = 0; i < step; i++) {
            auto tmp_in_indexes = in_indexes;
            auto in_axis = inp_a_axis[inp_a_axis.size() - 1];
            tmp_in_indexes[in_axis]
                    = tmp_in_indexes[in_axis] + static_cast<uint64_t>(i);
            auto tmp_plain_indexes = plain_indexes;
            tmp_plain_indexes[input_format_code.get(in_axis)]
                    = tmp_plain_indexes[input_format_code.get(in_axis)]
                    + static_cast<uint64_t>(i);
            expr tmp_in = src.tptr_;
            if (output_loop) { tmp_in = input; }
            expr mask;
            if (is_padding) {
                expr cur_step, sup_condition;
                determine_cur_step(input_blocking_dims_expr, tmp_in_indexes,
                        tmp_plain_indexes, cur_step, sup_condition, in_axis,
                        output_loop, step);
                stmt mask_def;
                mask = generate_mask_var_by_step(
                        mask_def, cur_step, step, sup_condition);
                cur_list.emplace_back(mask_def);
            }
            auto assign = builder::make_assign_unattached(rows[i],
                    // here, use src.tptr instead of input is aimed to
                    // avoid input is tensor_view_op. Otherwise, it will
                    // throw illegal exception in tensor_shrink
                    builder::make_indexing(tmp_in, tmp_in_indexes, step, mask));
            assign->attr()[op_traits::workload_computable_t::workload_number]
                    = wkld;
            cur_list.emplace_back(assign);
        }

        TRANS2D_REG_CALCULATION_F32();
        for (int i = 0; i < step; i++) {
            auto tmp_out_indexes = out_indexes;
            auto out_axis = out_b_axis[out_b_axis.size() - 1];
            tmp_out_indexes[out_axis]
                    = tmp_out_indexes[out_axis] + static_cast<uint64_t>(i);
            expr mask;
            if (is_padding) {
                auto cur_step = builder::make_min(
                        builder::make_max(
                                cast_to_s32(output_blocking_dims_expr.back())
                                        - cast_to_s32(tmp_out_indexes.back()),
                                0),
                        step);
                auto sup_condition = tmp_out_indexes[out_axis]
                        < output_blocking_dims_expr[out_axis];
                // ABba(4, 4, 16, 4) => AB(16,64)
                if (input_format.is_blocking()) {
                    sup_condition = sup_condition
                            && in_indexes.back() + i
                                    < input_blocking_dims_expr.back();
                }
                stmt mask_def;
                mask = generate_mask_var_by_step(
                        mask_def, cur_step, step, sup_condition);
                cur_list.emplace_back(mask_def);
            }
            auto assign = builder::make_assign_unattached(
                    builder::make_indexing(output, tmp_out_indexes, step, mask),
                    rows[i]);
            cur_list.emplace_back(assign);
        }
    };

    auto compute_transpose_bf16 = [&](const std::vector<expr> &in_indexes,
                                          const std::vector<expr> &out_indexes,
                                          const std::vector<expr>
                                                  &plain_indexes) {
        for (int i = 0; i < step; i++) {
            for (int p = 0; p < 4; p++) {
                auto tmp_in_indexes = in_indexes;
                auto in_axis = inp_a_axis[inp_a_axis.size() - 1];
                tmp_in_indexes[in_axis] = tmp_in_indexes[in_axis]
                        + (static_cast<uint64_t>(i) + p * 8);
                auto tmp_plain_indexes = plain_indexes;
                tmp_plain_indexes[input_format_code.get(in_axis)]
                        = tmp_plain_indexes[input_format_code.get(in_axis)]
                        + (static_cast<uint64_t>(i) + p * 8);
                expr tmp_in = src.tptr_;
                if (output_loop) { tmp_in = input; }
                expr mask;
                if (is_padding) {
                    expr cur_step, sup_condition;
                    determine_cur_step(input_blocking_dims_expr, tmp_in_indexes,
                            tmp_plain_indexes, cur_step, sup_condition, in_axis,
                            output_loop, step);
                    stmt mask_def;
                    mask = generate_mask_var_by_step(
                            mask_def, cur_step, trans_lanes, sup_condition);
                    cur_list.emplace_back(mask_def);
                }
                auto brct_src = builder::make_broadcast(
                        builder::make_indexing(
                                tmp_in, tmp_in_indexes, step, mask),
                        trans_lanes_bf16);
                auto assign = builder::make_assign_unattached(rows[i],
                        // here, use src.tptr instead of input is aimed
                        // to avoid input is tensor_view_op. Otherwise,
                        // it will throw illegal exception in
                        // tensor_shrink
                        p > 0 ? builder::make_select(
                                0xff << (p * step), brct_src, rows[i])
                              : brct_src);
                assign->attr()
                        [op_traits::workload_computable_t::workload_number]
                        = wkld;
                cur_list.emplace_back(assign);
            }
        }

        TRANS2D_REG_CALCULATION_BF16();
        for (int i = 0; i < step; i++) {
            auto tmp_out_indexes = out_indexes;
            auto out_axis = out_b_axis[out_b_axis.size() - 1];
            tmp_out_indexes[out_axis]
                    = tmp_out_indexes[out_axis] + static_cast<uint64_t>(i);
            expr mask;
            if (is_padding) {
                auto cur_step = builder::make_min(
                        builder::make_max(
                                cast_to_s32(output_blocking_dims_expr.back())
                                        - cast_to_s32(tmp_out_indexes.back()),
                                0),
                        trans_lanes_bf16);
                auto sup_condition = tmp_out_indexes[out_axis]
                        < output_blocking_dims_expr[out_axis];
                // ABba(4, 4, 16, 4) => AB(16,64)
                if (input_format.is_blocking()) {
                    sup_condition = sup_condition
                            && in_indexes.back() + i
                                    < input_blocking_dims_expr.back();
                }
                stmt mask_def;
                mask = generate_mask_var_by_step(
                        mask_def, cur_step, trans_lanes_bf16, sup_condition);
                cur_list.emplace_back(mask_def);
            }
            auto assign = builder::make_assign_unattached(
                    builder::make_indexing(
                            output, tmp_out_indexes, trans_lanes_bf16, mask),
                    rows[i + 8]);
            cur_list.emplace_back(assign);
        }
    };

    auto compute_loops = [&](const std::vector<expr> &blocking_dims,
                                 const std::vector<int> &a_axis,
                                 const std::vector<int> &b_axis,
                                 const tensor_slice &tsr) {
        for (int i = static_cast<int>(blocking_dims.size()) - 1; i >= 0; i--) {
            if (utils::is_one_of(i, a_axis.back(), b_axis.back())) {
                body = cur.isa<stmts>()
                        ? cur
                        : make_stmt<stmts_node_t>(
                                std::vector<stmt> {std::move(cur)});
                int cur_step = i == a_axis.back() ? inp_a_step : inp_b_step;
                cur = make_stmt<for_loop_node_t>(std::move(iter_vars.at(i)),
                        expr(0),
                        (!is_padding || !tsr.get_offset()[i].isa<constant>()
                                || !output_loop)
                                ? tsr.get_shape()[i]
                                : blocking_dims[i],
                        expr(cur_step), std::move(body), true,
                        for_type::NORMAL);
            }
        }
        for (int i = static_cast<int>(blocking_dims.size()) - 1; i >= 0; i--) {
            if (!utils::is_one_of(i, a_axis.back(), b_axis.back())) {
                body = cur.isa<stmts>()
                        ? cur
                        : make_stmt<stmts_node_t>(
                                std::vector<stmt> {std::move(cur)});
                cur = make_stmt<for_loop_node_t>(std::move(iter_vars.at(i)),
                        expr(0),
                        (!is_padding || !tsr.get_offset()[i].isa<constant>()
                                || !output_loop)
                                ? tsr.get_shape()[i]
                                : blocking_dims[i],
                        expr(1), std::move(body), true, for_type::NORMAL);
            }
        }
    };
    if (!output_loop) {
        std::vector<expr> in_indexes, loop_indexes;
        for (size_t i = 0; i < input_blocking_dims_expr.size(); i++) {
            iter_vars.emplace_back(builder::make_var(datatypes::index,
                    std::string("_fuseiter") + fusion_create_idx()));
            in_indexes.emplace_back(iter_vars[i] + src.get_offset()[i]);
            loop_indexes.emplace_back(iter_vars[i]);
        }
        expr condition;
        expr last_axis_offset, other_axis_condition;
        std::vector<expr> tmp_indexes = get_reorder_block2plain_indexes(graph,
                in_indexes, input_format, plain_dims, condition,
                last_axis_offset, other_axis_condition,
                input_format.format_code_.get(
                        static_cast<int>(input_format.format_code_.ndims())
                        - 1));
        std::vector<expr> out_indexes
                = get_reorder_plain2block_indexes(tmp_indexes, output_format);
        if (dtype == datatypes::f32) {
            compute_transpose_f32(loop_indexes, out_indexes, tmp_indexes);
        } else {
            compute_transpose_bf16(loop_indexes, out_indexes, tmp_indexes);
        }
        cur = builder::make_stmts_unattached(cur_list);
        compute_loops(input_blocking_dims_expr, inp_a_axis, inp_b_axis, src);
        cur->attr()[stmt_attr_key::merge_loop] = true;
        bld->emit(cur);
    } else {
        std::vector<expr> out_indexes;
        for (size_t i = 0; i < output_blocking_dims_expr.size(); i++) {
            iter_vars.emplace_back(builder::make_var(datatypes::index,
                    std::string("_fuseiter") + fusion_create_idx()));
            out_indexes.emplace_back(iter_vars[i] + dst.get_offset()[i]);
        }
        expr condition;
        expr last_axis_offset, other_axis_condition;
        std::vector<expr> tmp_indexes = get_reorder_block2plain_indexes(graph,
                out_indexes, output_format, plain_dims, condition,
                last_axis_offset, other_axis_condition);
        std::vector<expr> in_indexes
                = get_reorder_plain2block_indexes(tmp_indexes, input_format);
        if (dtype == datatypes::f32) {
            compute_transpose_f32(in_indexes, out_indexes, tmp_indexes);
        } else {
            compute_transpose_bf16(in_indexes, out_indexes, tmp_indexes);
        }
        cur = builder::make_stmts_unattached(cur_list);
        compute_loops(output_blocking_dims_expr, out_a_axis, out_b_axis, dst);
        cur->attr()[stmt_attr_key::merge_loop] = true;
        bld->emit(cur);
    }
}

static bool can_be_vnni_reorder(const context_ptr &ctx,
        std::vector<int> &inp_n_axis, std::vector<int> &inp_k_axis,
        std::vector<int> &out_n_axis, std::vector<int> &out_k_axis,
        const sc_dims &plain_dims, const sc_data_format_t &input_format,
        const sc_data_format_t &output_format, const tensor_slice &src,
        const tensor_slice &dst, const sc_data_type_t &dtype,
        bool &is_vnni_reorder, bool is_dynamic, bool dynamic_no_padding) {
    // VNNI reorder only support NK2NKknk-liked format.
    // Last axis should be 2 if dytpe is bf16 and 4 if dytpe is u8/s8
    // eg. 384N 64K -> 12N 4K 8k 32n 2k
    //     384N 64K -> 12N 2K 8k 32n 4k
    //     128A 16B 32C -> 128A 2B 2C 4c 8b 4c
    // dynamic cases can't check in current condition
    auto input_blocking_dims
            = sc_data_format_t::get_blocking_shapes(plain_dims, input_format);
    auto output_blocking_dims
            = sc_data_format_t::get_blocking_shapes(plain_dims, output_format);
    bool is_padding = false;
    if ((!is_dynamic
                && math_utils::get_dims_product(input_blocking_dims)
                        != math_utils::get_dims_product(output_blocking_dims))
            || (is_dynamic && !dynamic_no_padding)) {
        is_padding = true;
    }
    if (!(ctx->machine_.cpu_flags_.fAVX512F
                || ctx->machine_.cpu_flags_.fAVX512VBMI)) {
        return false;
    }
    if (input_format.is_blocking()) return false;
    bool is_bf16 = dtype.as_etype() == sc_data_etype::BF16;
    inp_n_axis.clear();
    inp_k_axis.clear();
    out_n_axis.clear();
    out_k_axis.clear();
    if (!utils::is_one_of(dtype.as_etype(), sc_data_etype::U8,
                sc_data_etype::S8, sc_data_etype::BF16)) {
        return false;
    }
    int inp_idx = 0, out_idx = 0;
    auto &inp_code = input_format.format_code_;
    auto &out_code = output_format.format_code_;
    int input_ndims = input_format.format_code_.ndims();
    int output_ndims = output_format.format_code_.ndims();
    int vectorized_last_dims = output_ndims - 1;
    int out_dim_counts[sc_data_format_kind_t::MAX_DIMS] = {0};
    output_format.format_code_.collect_dim_count(out_dim_counts);
    int vectorized_original_axis
            = output_format.format_code_.get(vectorized_last_dims);
    // Relax the restriction that the irrelevant dimension in the output is 1
    for (int i = vectorized_last_dims; i >= 0; i--) {
        auto tmp = output_format.format_code_.get(i);
        auto target_axis_dim = dst.shape_[i];
        // can't do vnni in dynamic cases
        if (!target_axis_dim.isa<constant>()) return false;
        if (out_dim_counts[tmp] > 1 && (get_expr_as_int(target_axis_dim) > 1)) {
            vectorized_last_dims = i;
            vectorized_original_axis = tmp;
            break;
        }
    }
    if (out_dim_counts[vectorized_original_axis] < 2) { return false; }

    if (!dst.get_shape().at(vectorized_last_dims).isa<constant>()
            || get_expr_as_int(
                       dst.get_real_tensor()->strides_[vectorized_last_dims])
                    != 1) {
        return false;
    }

    auto out_k2_pos = vectorized_last_dims,
         out_n_pos = vectorized_last_dims - 1, out_k_pos = -1, out_K_pos = -1,
         out_N_pos = -1, in_K_pos = -1, in_N_pos = -1;
    auto k_idx = out_code.get(out_k2_pos);
    auto n_idx = out_code.get(out_n_pos);

    for (auto i = vectorized_last_dims - 1; i >= 0; --i) {
        if (out_code.get(i) == k_idx) {
            if (out_k_pos == -1) {
                out_k_pos = i;
            } else if (out_K_pos == -1) {
                out_K_pos = i;
            }
        }
    }

    for (auto i = vectorized_last_dims - 2; i >= 0; --i) {
        if (out_code.get(i) == n_idx) {
            if (out_N_pos == -1) { out_N_pos = i; }
        }
    }

    for (auto i = input_ndims - 1; i >= 0; --i) {
        if (inp_code.get(i) == k_idx) {
            if (in_K_pos == -1) { in_K_pos = i; }
        }
    }
    for (auto i = input_ndims - 1; i >= 0; --i) {
        if (inp_code.get(i) == n_idx) {
            if (in_N_pos == -1) { in_N_pos = i; }
        }
    }
    // our K and N dims need to be constant
    if (!(src.get_shape().at(in_K_pos).isa<constant>()
                || src.get_shape().at(in_N_pos).isa<constant>())) {
        return false;
    }
    // Relax the restriction that the irrelevant dimension in the input is 1
    // eg: ABC[16,16,1] -> ABCbab[1,1,1,8,16,2] C dim is 1, still use vnni
    // reorder
    if (!(inp_code.get(input_ndims - 1) == k_idx
                || inp_code.get(input_ndims - 1) == n_idx)) {
        int input_lastdim_max_idx = std::max(in_K_pos, in_N_pos);
        if (get_expr_as_int(
                    src.get_real_tensor()->strides_[input_lastdim_max_idx])
                != 1) {
            return false;
        }
    }

    if ((in_N_pos > in_K_pos && out_n_pos > out_k_pos)
            || (in_N_pos < in_K_pos && out_n_pos < out_k_pos)) {
        is_vnni_reorder = true;
    }
    // find axie of N K and N K k n k
    out_n_axis.emplace_back(out_N_pos);
    out_n_axis.emplace_back(out_n_pos);
    out_k_axis.emplace_back(out_K_pos);
    out_k_axis.emplace_back(out_k_pos);
    out_k_axis.emplace_back(out_k2_pos);
    inp_n_axis.emplace_back(in_N_pos);
    inp_k_axis.emplace_back(in_K_pos);

    // VNNI reorder kernel shape is 4x16 for u8/s8 and 4x8 for bf16.
    if (!is_padding) {
        if (get_expr_as_int(dst.shape_[out_k2_pos]) != (is_bf16 ? 2 : 4))
            return false;
        if (!is_vnni_reorder) {
            if (get_expr_as_int(dst.shape_[out_n_pos]) % 4 != 0) return false;
            if (get_expr_as_int(dst.shape_[out_k_pos]) % 4 != 0) return false;
        } else {
            if (get_expr_as_int(dst.shape_[out_n_pos]) % (is_bf16 ? 8 : 16)
                    != 0)
                return false;

            if ((get_expr_as_int(dst.shape_[out_k_pos])
                        * get_expr_as_int(dst.shape_[out_k2_pos]))
                            % 4
                    != 0)
                return false;
        }
    } else {
        if (output_blocking_dims[out_k2_pos] != (is_bf16 ? 2 : 4)) return false;
        if (!is_vnni_reorder) {
            if (output_blocking_dims[out_n_pos] % 4 != 0) return false;
            if (output_blocking_dims[out_k_pos] % 4 != 0) return false;
        } else {
            if (output_blocking_dims[out_n_pos] % (is_bf16 ? 8 : 16) != 0)
                return false;

            if (output_blocking_dims[out_k_pos]
                            * output_blocking_dims[out_k2_pos] % 4
                    != 0)
                return false;
        }
    }

    // new VNNI transpose currently only avaiable on spr
    if ((!ctx->machine_.cpu_flags_.fAVX512VBMI) && is_vnni_reorder)
        return false;
    return true;
}

static void do_vnni_reorder_avx512f(std::vector<stmt_c> &cur_list,
        std::vector<expr> &rows, sc_data_type_t &rows_dtype) {
    // reorder on a kernel of 4x16(u8/s8) or 4x8(bf16)
    // registers to perform reorder, should reinterpret data to f32 due to
    // intrinsic limitation
    auto xmm0 = builder::make_var(sc_data_type_t::f32(4), std::string("xmm0"));
    auto xmm1 = builder::make_var(sc_data_type_t::f32(4), std::string("xmm1"));
    auto xmm2 = builder::make_var(sc_data_type_t::f32(4), std::string("xmm2"));
    auto xmm3 = builder::make_var(sc_data_type_t::f32(4), std::string("xmm3"));
    auto xmm_tmp
            = builder::make_var(sc_data_type_t::f32(4), std::string("xmm_tmp"));
    cur_list.emplace_back(builder::make_var_tensor_def_unattached(xmm0));
    cur_list.emplace_back(builder::make_var_tensor_def_unattached(xmm1));
    cur_list.emplace_back(builder::make_var_tensor_def_unattached(xmm2));
    cur_list.emplace_back(builder::make_var_tensor_def_unattached(xmm3));
    cur_list.emplace_back(builder::make_var_tensor_def_unattached(xmm_tmp));

    // permutex2var selector
#define MAKE_IDX_U32(name, v0, v1, v2, v3) \
    auto idx##name = make_expr<constant_node>( \
            std::vector<union_val> { \
                    UINT64_C(v0), UINT64_C(v1), UINT64_C(v2), UINT64_C(v3)}, \
            sc_data_type_t::u32(4));
    MAKE_IDX_U32(0, 0x00000000, 0x00000004, 0x00000002, 0x00000006)
    MAKE_IDX_U32(1, 0x00000001, 0x00000005, 0x00000003, 0x00000007)
    MAKE_IDX_U32(2, 0x00000000, 0x00000001, 0x00000004, 0x00000005)
    MAKE_IDX_U32(3, 0x00000002, 0x00000003, 0x00000006, 0x00000007)

    stmt assign;
#define MAKE_ASSIGN_F32(dst, src) \
    assign = builder::make_assign_unattached(dst, src); \
    cur_list.emplace_back(assign);
#define MKAE_INTERPRET_F32(dst, src, attr) \
    MAKE_ASSIGN_F32(dst, \
            make_expr<intrin_call_node>( \
                    intrin_type::reinterpret, std::vector<expr> {src}, attr));

    any_map_t reinterpret_attr;
    reinterpret_attr[intrin_attr::out_dtype] = sc_data_type_t::f32(4);
    MKAE_INTERPRET_F32(xmm0, rows[0].remove_const(), reinterpret_attr)
    MKAE_INTERPRET_F32(xmm1, rows[1].remove_const(), reinterpret_attr)
    MKAE_INTERPRET_F32(xmm2, rows[2].remove_const(), reinterpret_attr)
    MKAE_INTERPRET_F32(xmm3, rows[3].remove_const(), reinterpret_attr)
    any_map_t permute_attr;
#define MAKE_PERMUTE_F32(dst, a, idx, b) \
    MAKE_ASSIGN_F32(dst, \
            make_expr<intrin_call_node>(intrin_type::permutex2var, \
                    std::vector<expr> {a, idx, b}, permute_attr))
    // do permute in any two pairs of register
    MAKE_ASSIGN_F32(xmm_tmp, xmm0)
    MAKE_PERMUTE_F32(xmm0, xmm_tmp, idx0, xmm1)
    MAKE_PERMUTE_F32(xmm1, xmm_tmp, idx1, xmm1)
    MAKE_ASSIGN_F32(xmm_tmp, xmm2)
    MAKE_PERMUTE_F32(xmm2, xmm_tmp, idx0, xmm3)
    MAKE_PERMUTE_F32(xmm3, xmm_tmp, idx1, xmm3)
    MAKE_ASSIGN_F32(xmm_tmp, xmm0)
    MAKE_PERMUTE_F32(xmm0, xmm_tmp, idx2, xmm2)
    MAKE_PERMUTE_F32(xmm2, xmm_tmp, idx3, xmm2)
    MAKE_ASSIGN_F32(xmm_tmp, xmm1)
    MAKE_PERMUTE_F32(xmm1, xmm_tmp, idx2, xmm3)
    MAKE_PERMUTE_F32(xmm3, xmm_tmp, idx3, xmm3)

    reinterpret_attr[intrin_attr::out_dtype] = rows_dtype;
    MKAE_INTERPRET_F32(rows[0], xmm0, reinterpret_attr)
    MKAE_INTERPRET_F32(rows[1], xmm1, reinterpret_attr)
    MKAE_INTERPRET_F32(rows[2], xmm2, reinterpret_attr)
    MKAE_INTERPRET_F32(rows[3], xmm3, reinterpret_attr)
}

static void do_vnni_reorder(std::vector<stmt_c> &cur_list,
        std::vector<expr> &rows, sc_data_type_t &rows_dtype,
        const bool is_vnni_reorder) {
    bool is_bf16 = rows_dtype.type_code_ == sc_data_etype::BF16;
    // reorder on a kernel of 4x16(u8/s8) or 4x8(bf16)
    // registers to perform reorder, should reinterpret data to f32 due to
    // intrinsic limitation
    auto xmm0 = builder::make_var(sc_data_type_t::u8(16), std::string("xmm0"));
    auto xmm1 = builder::make_var(sc_data_type_t::u8(16), std::string("xmm1"));
    auto xmm2 = builder::make_var(sc_data_type_t::u8(16), std::string("xmm2"));
    auto xmm3 = builder::make_var(sc_data_type_t::u8(16), std::string("xmm3"));
    auto xmm_tmp
            = builder::make_var(sc_data_type_t::u8(16), std::string("xmm_tmp"));
    cur_list.emplace_back(builder::make_var_tensor_def_unattached(xmm0));
    cur_list.emplace_back(builder::make_var_tensor_def_unattached(xmm1));
    cur_list.emplace_back(builder::make_var_tensor_def_unattached(xmm2));
    cur_list.emplace_back(builder::make_var_tensor_def_unattached(xmm3));
    cur_list.emplace_back(builder::make_var_tensor_def_unattached(xmm_tmp));

    expr idx0, idx1, idx2, idx3;
    any_map_t reinterpret_attr;
    reinterpret_attr[intrin_attr::out_dtype] = sc_data_type_t::u8(16);
    // permutex2var selector
#define MAKE_IDX(name, v0, v1, v2, v3) \
    idx##name = make_expr<intrin_call_node>(intrin_type::reinterpret, \
            std::vector<expr> {make_expr<constant_node>( \
                    std::vector<union_val> {UINT64_C(v0), UINT64_C(v1), \
                            UINT64_C(v2), UINT64_C(v3)}, \
                    sc_data_type_t::u32(4))}, \
            reinterpret_attr);
    if (!is_vnni_reorder) { // vnni
        MAKE_IDX(0, 0x03020100, 0x13121110, 0x0B0A0908, 0x1B1A1918)
        MAKE_IDX(1, 0x07060504, 0x17161514, 0x0F0E0D0C, 0x1F1E1D1C)
        MAKE_IDX(2, 0x03020100, 0x07060504, 0x13121110, 0x17161514)
        MAKE_IDX(3, 0x0B0A0908, 0x0F0E0D0C, 0x1B1A1918, 0X1F1E1D1C)
    } else if (is_bf16) {
        // vnni transpose bf16
        MAKE_IDX(0, 0x11100100, 0x13120302, 0x15140504, 0x17160706)
        MAKE_IDX(1, 0x19180908, 0x1B1A0B0A, 0x1D1C0D0C, 0x1F1E0F0E)

    } else { // vnni transpose u8/s8
        MAKE_IDX(0, 0x11011000, 0x13031202, 0x15051404, 0x17071606)
        MAKE_IDX(1, 0x19091808, 0x1B0B1A0A, 0x1D0D1C0C, 0x1F0F1E0E)
        MAKE_IDX(2, 0x11100100, 0x13120302, 0x15140504, 0x17160706)
        MAKE_IDX(3, 0x19180908, 0x1B1A0B0A, 0x1D1C0D0C, 0x1F1E0F0E)
    }

    stmt assign;
#define MAKE_ASSIGN(dst, src) \
    assign = builder::make_assign_unattached(dst, src); \
    cur_list.emplace_back(assign);
#define MKAE_INTERPRET(dst, src, attr) \
    MAKE_ASSIGN(dst, \
            make_expr<intrin_call_node>( \
                    intrin_type::reinterpret, std::vector<expr> {src}, attr));

    reinterpret_attr[intrin_attr::out_dtype] = sc_data_type_t::u8(16);
    MKAE_INTERPRET(xmm0, rows[0].remove_const(), reinterpret_attr)
    MKAE_INTERPRET(xmm1, rows[1].remove_const(), reinterpret_attr)
    MKAE_INTERPRET(xmm2, rows[2].remove_const(), reinterpret_attr)
    MKAE_INTERPRET(xmm3, rows[3].remove_const(), reinterpret_attr)

    any_map_t permute_attr;
#define MAKE_PERMUTE(dst, a, idx, b) \
    MAKE_ASSIGN(dst, \
            make_expr<intrin_call_node>(intrin_type::permutex2var, \
                    std::vector<expr> {a, idx, b}, permute_attr))
    // do permute in any two pairs of register
    if (is_vnni_reorder && is_bf16) {
        MAKE_ASSIGN(xmm_tmp, xmm0)
        MAKE_PERMUTE(xmm0, xmm_tmp, idx0, xmm1)
        MAKE_PERMUTE(xmm1, xmm_tmp, idx1, xmm1)

        MAKE_ASSIGN(xmm_tmp, xmm2)
        MAKE_PERMUTE(xmm2, xmm_tmp, idx0, xmm3)
        MAKE_PERMUTE(xmm3, xmm_tmp, idx1, xmm3)
    } else {
        MAKE_ASSIGN(xmm_tmp, xmm0)
        MAKE_PERMUTE(xmm0, xmm_tmp, idx0, xmm1)
        MAKE_PERMUTE(xmm1, xmm_tmp, idx1, xmm1)

        MAKE_ASSIGN(xmm_tmp, xmm2)
        MAKE_PERMUTE(xmm2, xmm_tmp, idx0, xmm3)
        MAKE_PERMUTE(xmm3, xmm_tmp, idx1, xmm3)

        MAKE_ASSIGN(xmm_tmp, xmm0)
        MAKE_PERMUTE(xmm0, xmm_tmp, idx2, xmm2)
        MAKE_PERMUTE(xmm2, xmm_tmp, idx3, xmm2)

        MAKE_ASSIGN(xmm_tmp, xmm1)
        MAKE_PERMUTE(xmm1, xmm_tmp, idx2, xmm3)
        MAKE_PERMUTE(xmm3, xmm_tmp, idx3, xmm3)
        if (is_vnni_reorder) {
            MAKE_ASSIGN(xmm_tmp, xmm1)
            MAKE_ASSIGN(xmm1, xmm2)
            MAKE_ASSIGN(xmm2, xmm_tmp)
        }
    }

    reinterpret_attr[intrin_attr::out_dtype] = rows_dtype;
    MKAE_INTERPRET(rows[0], xmm0, reinterpret_attr)
    MKAE_INTERPRET(rows[1], xmm1, reinterpret_attr)
    MKAE_INTERPRET(rows[2], xmm2, reinterpret_attr)
    MKAE_INTERPRET(rows[3], xmm3, reinterpret_attr)
}

static void compute_vnni_reorder(sc_graph_t &graph, const context_ptr &ctx,
        const tensor_slice &src, tensor_slice &dst,
        const sc_data_format_t &input_format,
        const sc_data_format_t &output_format, sc_data_type_t dtype,
        const sc_dims &plain_dims, bool output_loop, any_map_t &attrs,
        std::vector<int> &inp_n_axis, std::vector<int> &inp_k_axis,
        std::vector<int> &out_n_axis, std::vector<int> &out_k_axis,
        size_t wkld = 0UL, const bool &is_vnni_reorder = false,
        bool is_dynamic = false, bool dynamic_no_padding = false) {
    bool is_bf16 = dtype.as_etype() == sc_data_etype::BF16;
    auto input = src.get_real_tensor();
    auto output = dst.get_real_tensor();
    int step = 4;
    auto bld = builder::get_current_builder();
    auto input_blocking_dims
            = sc_data_format_t::get_blocking_shapes(plain_dims, input_format);
    auto input_blocking_shape_expr
            = get_blocking_shapes_expr(graph, plain_dims, input_format);
    auto output_blocking_dims
            = sc_data_format_t::get_blocking_shapes(plain_dims, output_format);
    auto input_blocking_dims_expr
            = get_blocking_shapes_expr(graph, plain_dims, input_format);
    auto output_blocking_dims_expr
            = get_blocking_shapes_expr(graph, plain_dims, output_format);
    // plain axis of last block
    auto input_last_origin_axis = input_format.format_code_.get(
            input_format.format_code_.ndims() - 1);
    auto output_last_origin_axis = output_format.format_code_.get(
            output_format.format_code_.ndims() - 1);
    int input_origin_axis_vectorized = input_format.format_code_.ndims() - 1;
    int output_origin_axis_vectorized = output_format.format_code_.ndims() - 1;
    find_vectorized_axis(input_blocking_dims_expr, input_format,
            input_last_origin_axis, input_origin_axis_vectorized);
    find_vectorized_axis(output_blocking_dims_expr, output_format,
            output_last_origin_axis, output_origin_axis_vectorized);
    bool is_padding = false;
    if ((!is_dynamic
                && math_utils::get_dims_product(input_blocking_dims)
                        != math_utils::get_dims_product(output_blocking_dims))
            || (is_dynamic && !dynamic_no_padding)) {
        is_padding = true;
    }

    std::vector<expr> rows(4);
    std::vector<expr> iter_vars;
    std::vector<stmt_c> cur_list;
    auto rows_dtype = dtype;
    rows_dtype.lanes_ = is_bf16 ? 8 : 16;
    for (auto i = 0; i < 4; i++) {
        rows[i] = builder::make_var(rows_dtype,
                "row" + std::to_string(i + 1) + fusion_create_var_idx());
        // skip bf16 elimination pass on rows. Otherwise it will be promote to
        // f32.
        rows[i]->attr()["can_promote_to_f32"] = false;
        cur_list.emplace_back(builder::make_var_tensor_def_unattached(rows[i]));
    }
    if (!output_loop) {
        std::vector<expr> in_indexes, loop_indexes;
        for (size_t i = 0; i < input_blocking_dims.size(); i++) {
            iter_vars.emplace_back(builder::make_var(datatypes::index,
                    std::string("_fuseiter") + fusion_create_idx()));
            in_indexes.emplace_back((iter_vars[i] + src.get_offset()[i]));
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
        for (int i = 0; i < step; i++) {
            auto tmp_in_indexes = loop_indexes;
            if (!is_vnni_reorder) {
                tmp_in_indexes[inp_n_axis[0]] = tmp_in_indexes[inp_n_axis[0]]
                        + static_cast<uint64_t>(i);
            } else {
                tmp_in_indexes[inp_k_axis[0]] = tmp_in_indexes[inp_k_axis[0]]
                        + static_cast<uint64_t>(i);
            }
            auto assign = builder::make_assign_unattached(rows[i],
                    // here, use src.tptr instead of input is aimed to
                    // avoid input is tensor_view_op. Otherwise, it will
                    // throw illegal exception in tensor_shrink
                    builder::make_indexing(
                            src.tptr_, tmp_in_indexes, is_bf16 ? 8 : 16));
            assign->attr()[op_traits::workload_computable_t::workload_number]
                    = wkld;
            cur_list.emplace_back(assign);
        }

        if (ctx->machine_.cpu_flags_.fAVX512VBMI) {
            do_vnni_reorder(cur_list, rows, rows_dtype, is_vnni_reorder);
        } else {
            do_vnni_reorder_avx512f(cur_list, rows, rows_dtype);
        }

        for (int i = 0; i < step; i++) {
            auto tmp_out_indexes = out_indexes;
            if (!is_vnni_reorder) { // vnni transpose
                tmp_out_indexes[out_k_axis[1]] = tmp_out_indexes[out_k_axis[1]]
                        + static_cast<uint64_t>(i);
            } else if (is_bf16) { // vnni reorder + transpose bf16
                switch (i) {
                    case 1:
                        tmp_out_indexes[out_n_axis[1]]
                                = tmp_out_indexes[out_n_axis[1]]
                                + static_cast<uint64_t>(4);
                        break;
                    case 2:
                        tmp_out_indexes[out_k_axis[1]]
                                = tmp_out_indexes[out_k_axis[1]]
                                + static_cast<uint64_t>(1);
                        break;
                    case 3:
                        tmp_out_indexes[out_n_axis[1]]
                                = tmp_out_indexes[out_n_axis[1]]
                                + static_cast<uint64_t>(4);
                        tmp_out_indexes[out_k_axis[1]]
                                = tmp_out_indexes[out_k_axis[1]]
                                + static_cast<uint64_t>(1);
                        break;
                }

            } else { // vnni reorder + transpose
                tmp_out_indexes[out_n_axis[1]] = tmp_out_indexes[out_n_axis[1]]
                        + static_cast<uint64_t>(i) * 4;
            }
            auto assign = builder::make_assign_unattached(
                    builder::make_indexing(
                            output, tmp_out_indexes, is_bf16 ? 8 : 16),
                    rows[i]);
            cur_list.emplace_back(assign);
        }
        stmt cur = builder::make_stmts_unattached(cur_list);
        stmt body;
        expr_c iter_end;
        // for-loop-transforms only support step=1
        // we can only divide iter_end_ rather than multiply step_
        for (int i = static_cast<int>(input_blocking_dims.size()) - 1; i >= 0;
                i--) {
            iter_end = src.get_shape()[i];
            expr cur_step = 1;
            if (!is_vnni_reorder) {
                if (i == inp_n_axis[0]) {
                    cur_step = 4;
                } else if (i == inp_k_axis[0]) {
                    cur_step = is_bf16 ? 8 : 16;
                }
            } else {
                if (i == inp_k_axis[0]) {
                    cur_step = 4;
                } else if (i == inp_n_axis[0]) {
                    cur_step = is_bf16 ? 8 : 16;
                }
            }
            body = cur.isa<stmts>() ? cur
                                    : make_stmt<stmts_node_t>(
                                            std::vector<stmt> {std::move(cur)});
            cur = make_stmt<for_loop_node_t>(std::move(iter_vars.at(i)),
                    expr(0), iter_end.remove_const(), cur_step, std::move(body),
                    true, for_type::NORMAL);
        }
        cur->attr()[stmt_attr_key::merge_loop] = true;
        bld->emit(cur);
    } else { // use output loop
        std::vector<expr> out_indexes;
        // create iter variable, and make index
        for (size_t i = 0; i < output_blocking_dims.size(); i++) {
            iter_vars.emplace_back(builder::make_var(datatypes::index,
                    std::string("_fuseiter") + fusion_create_idx()));
            out_indexes.emplace_back(iter_vars[i] + dst.get_offset()[i]);
        }

        // calculate the input index according to the output index
        expr condition, vnni_condition;
        expr last_axis_offset, other_axis_condition;
        std::vector<expr> tmp_indexes
                = get_reorder_block2plain_indexes(graph, out_indexes,
                        output_format, plain_dims, condition, last_axis_offset,
                        other_axis_condition, output_origin_axis_vectorized);
        std::vector<expr> in_indexes
                = get_reorder_plain2block_indexes(tmp_indexes, input_format);
        expr cur_step_var;
        stmt cur_step_var_assign;
        // load data to register
        for (int i = 0; i < step; i++) {
            auto tmp_in_indexes = in_indexes;
            if (!is_vnni_reorder) {
                tmp_in_indexes[inp_n_axis[0]] = tmp_in_indexes[inp_n_axis[0]]
                        + static_cast<uint64_t>(i);
            } else {
                tmp_in_indexes[inp_k_axis[0]] = tmp_in_indexes[inp_k_axis[0]]
                        + static_cast<uint64_t>(i);
            }
            expr mask;
            stmt mask_def;
            int real_step = is_bf16 ? 8 : 16;
            if (is_padding) {
                int tmp_in_last_dim
                        = is_vnni_reorder ? inp_n_axis[0] : inp_k_axis[0];
                int tmp_in_other_dim
                        = is_vnni_reorder ? inp_k_axis[0] : inp_n_axis[0];
                last_axis_offset
                        = cast_to_s32(
                                  input_blocking_shape_expr[tmp_in_last_dim])
                        - cast_to_s32(tmp_in_indexes[tmp_in_last_dim]);
                other_axis_condition = tmp_in_indexes[tmp_in_other_dim]
                        < input_blocking_shape_expr[tmp_in_other_dim];
                // The cur_step corresponding to each step is the same,
                // so we only need to count the first time and others
                // can be reused.
                if (i == 0) {
                    // mask = min(max(0, last_dim_len -
                    // last_dim_idx),real_step) To choose [0 ~
                    // step] mask
                    auto cur_step = builder::make_min(
                            builder::make_max(builder::make_constant(0),
                                    last_axis_offset),
                            real_step);
                    cur_step_var = builder::make_var(
                            sc_data_type_t::s32(1), "cur_step_var");
                    cur_step_var_assign
                            = builder::make_var_tensor_def_unattached(
                                    cur_step_var, linkage::local, cur_step);
                    cur_list.emplace_back(cur_step_var_assign);
                }
                // mask = other_dims_condition ? mask : 0;
                mask = generate_mask_var_by_step(mask_def, cur_step_var,
                        real_step, other_axis_condition);
                cur_list.emplace_back(mask_def);
            }

            // here, we use input is to fix the out-of-bound exception.
            // Using tptr will make the address calculation wrong in some
            // cases (479x1024..).
            auto assign = builder::make_assign_unattached(rows[i],
                    builder::make_indexing(
                            input, tmp_in_indexes, is_bf16 ? 8 : 16, mask));
            assign->attr()[op_traits::workload_computable_t::workload_number]
                    = wkld;
            cur_list.emplace_back(assign);
        }

        // vnni reorder/transpose verctorized calculation on register
        if (ctx->machine_.cpu_flags_.fAVX512VBMI) {
            do_vnni_reorder(cur_list, rows, rows_dtype, is_vnni_reorder);
        } else {
            do_vnni_reorder_avx512f(cur_list, rows, rows_dtype);
        }
        // store data from register
        for (int i = 0; i < step; i++) {
            auto tmp_out_indexes = out_indexes;
            if (!is_vnni_reorder) { // vnni transpose
                tmp_out_indexes[out_k_axis[1]] = tmp_out_indexes[out_k_axis[1]]
                        + static_cast<uint64_t>(i);
            } else if (is_bf16) { // vnni reorder + transpose bf16
                switch (i) {
                    case 1:
                        tmp_out_indexes[out_n_axis[1]]
                                = tmp_out_indexes[out_n_axis[1]]
                                + static_cast<uint64_t>(4);
                        break;
                    case 2:
                        tmp_out_indexes[out_k_axis[1]]
                                = tmp_out_indexes[out_k_axis[1]]
                                + static_cast<uint64_t>(1);
                        break;
                    case 3:
                        tmp_out_indexes[out_n_axis[1]]
                                = tmp_out_indexes[out_n_axis[1]]
                                + static_cast<uint64_t>(4);
                        tmp_out_indexes[out_k_axis[1]]
                                = tmp_out_indexes[out_k_axis[1]]
                                + static_cast<uint64_t>(1);
                        break;
                }

            } else { // vnni reorder + transpose
                tmp_out_indexes[out_n_axis[1]] = tmp_out_indexes[out_n_axis[1]]
                        + static_cast<uint64_t>(i) * 4;
            }

            auto assign = builder::make_assign_unattached(
                    builder::make_indexing(
                            output, tmp_out_indexes, is_bf16 ? 8 : 16),
                    rows[i]);
            cur_list.emplace_back(assign);
        }
        stmt cur = builder::make_stmts_unattached(cur_list);
        stmt body;

        expr iter_end;
        // for-loop-transforms only support step=1
        // we can only divide iter_end_ rather than multiply step_
        for (int i = static_cast<int>(output_blocking_dims_expr.size()) - 1;
                i >= 0; i--) {
            // if the offset of dst is given(commit op)
            if (!is_padding || !dst.get_offset()[i].isa<constant>()) {
                iter_end = dst.get_shape()[i];
            } else {
                iter_end = output_blocking_dims_expr[i];
            }
            expr cur_step = 1;
            if (!is_vnni_reorder) { // vnni transpose
                if (i == out_n_axis[1] || i == out_k_axis[1]) {
                    cur_step = 4;
                } else if (i == out_k_axis[2]) {
                    cur_step = is_bf16 ? 2 : 4;
                }
            } else { // vnni reorder
                if (i == out_n_axis[1]) {
                    cur_step = is_bf16 ? 8 : 16;
                } else if ((is_bf16 && i == out_k_axis[1])
                        || i == out_k_axis[2]) {
                    cur_step = is_bf16 ? 2 : 4;
                }
            }

            body = cur.isa<stmts>() ? cur
                                    : make_stmt<stmts_node_t>(
                                            std::vector<stmt> {std::move(cur)});
            cur = make_stmt<for_loop_node_t>(std::move(iter_vars.at(i)),
                    expr(0), iter_end.remove_const(), cur_step, std::move(body),
                    true, for_type::NORMAL);
        }
        cur->attr()[stmt_attr_key::merge_loop] = true;
        bld->emit(cur);
    }
}

void compute_reorder_block(sc_graph_t &graph, const context_ptr &ctx,
        const tensor_slice &src, tensor_slice &dst,
        const sc_data_format_t &input_format,
        const sc_data_format_t &output_format, sc_data_type_t dtype,
        const sc_dims &plain_dims, bool output_loop, any_map_t &attrs,
        size_t wkld = 0UL, bool is_innermost_dim_strided = false,
        bool is_dynamic = false, int impl_alg = 0) {
    COMPILE_ASSERT(input_format.is_convertible(output_format),
            "Can not convert input format "
                    << input_format << " to output format " << output_format
                    << ".");

    std::vector<int> inp_a_axis, inp_b_axis, out_a_axis, out_b_axis;
    bool is_vnni_reorder = false;
    bool dynamic_no_padding = impl_alg & impl_kind_t::no_padding;
    if (!is_innermost_dim_strided
            && can_be_vnni_reorder(ctx, inp_a_axis, inp_b_axis, out_a_axis,
                    out_b_axis, plain_dims, input_format, output_format, src,
                    dst, dtype, is_vnni_reorder, is_dynamic,
                    dynamic_no_padding)) {
        compute_vnni_reorder(graph, ctx, src, dst, input_format, output_format,
                dtype, plain_dims, output_loop, attrs, inp_a_axis, inp_b_axis,
                out_a_axis, out_b_axis, wkld, is_vnni_reorder, is_dynamic,
                dynamic_no_padding);
    } else if (!is_innermost_dim_strided
            && can_be_fast_transpose(ctx, inp_a_axis, inp_b_axis, out_a_axis,
                    out_b_axis, plain_dims, input_format, output_format, src,
                    dst, dtype, is_dynamic, dynamic_no_padding)) {
        compute_fast_transpose(graph, ctx, src, dst, input_format,
                output_format, dtype, plain_dims, output_loop, attrs,
                inp_a_axis, inp_b_axis, out_a_axis, out_b_axis, wkld,
                is_dynamic, dynamic_no_padding);
    } else if (is_not_blocking(input_format)
            && is_not_blocking(output_format)) {
        compute_reorder_stride2stride(graph, ctx, src, dst, input_format,
                output_format, dtype, plain_dims, output_loop, attrs, wkld,
                is_innermost_dim_strided, is_dynamic, dynamic_no_padding);
    } else if (is_not_blocking(input_format) && output_format.is_blocking()) {
        compute_reorder_stride2block(graph, ctx, src, dst, input_format,
                output_format, dtype, plain_dims, output_loop, attrs, wkld,
                is_innermost_dim_strided, is_dynamic, dynamic_no_padding);
    } else if (input_format.is_blocking() && is_not_blocking(output_format)) {
        compute_reorder_block2stride(graph, ctx, src, dst, input_format,
                output_format, dtype, plain_dims, attrs, wkld,
                is_innermost_dim_strided, is_dynamic, dynamic_no_padding);
    } else if (input_format.is_blocking() && output_format.is_blocking()) {
        compute_reorder_block2block(graph, ctx, src, dst, input_format,
                output_format, dtype, plain_dims, output_loop, attrs, wkld,
                is_innermost_dim_strided, is_dynamic, dynamic_no_padding);
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
    if (auto inp = get_inputs()[0]->producer_owner_->dyn_cast<input_op>()) {
        return inp->is_arg_input();
    }
    return false;
}

bool reorder_op_t::support_output_loop() const {
    return get_output_format().is_blocking();
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
    bool is_vnni_reorder = false;

bool reorder_op_t::support_optimized_kernel(const context_ptr &ctx) const {
    INIT_REORDER_OP_INFO()
    return !is_innermost_dim_strided
            && (can_be_fast_transpose(ctx, inp_a_axis, inp_b_axis, out_a_axis,
                        out_b_axis, plain_dims_, input_format, output_format,
                        src, dst, dtype, is_dynamic(),
                        info_.cur_impl_ & impl_kind_t::no_padding)
                    || can_be_vnni_reorder(ctx, inp_a_axis, inp_b_axis,
                            out_a_axis, out_b_axis, plain_dims_, input_format,
                            output_format, src, dst, dtype, is_vnni_reorder,
                            is_dynamic(),
                            info_.cur_impl_ & impl_kind_t::no_padding));
}

bool reorder_op_t::meet_vnni_reorder_require(const context_ptr &ctx) const {
    INIT_REORDER_OP_INFO()
    return !is_innermost_dim_strided
            && can_be_vnni_reorder(ctx, inp_a_axis, inp_b_axis, out_a_axis,
                    out_b_axis, plain_dims_, input_format, output_format, src,
                    dst, dtype, is_vnni_reorder, is_dynamic(),
                    info_.cur_impl_ & impl_kind_t::no_padding);
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
    compute_reorder_block(get_owner_graph(), ctx, *inputs[0], *dst[0],
            input_format, output_format, info_.inputs_[0]->details_.dtype_,
            plain_dims_, use_output_loop(), attrs_, wkld,
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
