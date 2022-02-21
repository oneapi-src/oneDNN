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

#include <assert.h>

#include <utility>
#include "memory_movement.hpp"
#include <compiler/ir/builder.hpp>
#include <compiler/ir/graph/fusible_op_utils.hpp>
#include <compiler/ir/graph/fusion_mgr.hpp>
#include <compiler/ir/graph/outer_loop_generator.hpp>
#include <compiler/ir/transform/auto_cast.hpp>
#include <compiler/ir/transform/constant_fold.hpp>
#include <util/exceptions.hpp>
#include <util/utils.hpp>

namespace sc {

ir_module_ptr reorder_op_t::get_func(context_ptr ctx) {
    top_level_anchor_generator_t gen;
    attrs_.set(op_attr_key::no_fuse, true);
    auto ret = fusible_op_get_func(this, gen, ctx, false);
    auto func = ret->get_entry_func();
    auto body = func->body_.as<stmts>();
    COMPILE_ASSERT(body.defined(), "Expecting a body");
    COMPILE_ASSERT(body->seq_.size() == 2, "Expecting 2 stmt in reorder body");
    auto loop = body->seq_[0].as<for_loop>();
    COMPILE_ASSERT(loop.defined(), "Expecting a for loop in reorder body");
    loop->kind_ = for_type::PARALLEL;
    return ret;
}

void reorder_op_t::query_format(context_ptr ctx,
        std::vector<std::vector<sc_data_format_t>> &in_formats,
        std::vector<std::vector<sc_data_format_t>> &out_formats) {
    if (!attrs_.get_or_else("internal", false)) {
        in_formats.push_back(std::vector<sc_data_format_t> {
                info_.inputs_[0]->details_.get_format()});
        out_formats.push_back(std::vector<sc_data_format_t> {
                info_.inputs_[0]->details_.get_format()});
    }
}

reorder_op_t::reorder_op_t(const std::vector<graph_tensor_ptr> &ins,
        const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs) {
    for (auto &in : ins) {
        info_.inputs_.emplace_back(in);
    }
    if (outs.empty()) {
        info_.outputs_.emplace_back(std::make_shared<graph_tensor>(
                this, info_.inputs_[0]->details_));
        info_.outputs_[0]->details_.set_format(
                attrs.get<sc_data_format_t>("out_format"));
    } else {
        info_.outputs_ = outs;
    }
    op_name_ = "reorder";
    attrs_ = attrs;
    plain_dims_ = ins[0]->details_.get_plain_dims();
    input_format_ = info_.inputs_[0]->details_.get_format();
    output_format_ = info_.outputs_[0]->details_.get_format();
    COMPILE_ASSERT(info_.inputs_[0]->details_.get_format().is_convertible(
                           info_.outputs_[0]->details_.get_format()),
            "input format " << info_.inputs_[0]->details_.get_format()
                            << " can not convert to "
                            << info_.outputs_[0]->details_.get_format() << ".");
}

reorder_op_t::reorder_op_t(graph_tensor_ptr v, sc_data_format_t input_format,
        sc_data_format_t output_format)
    : input_format_(input_format), output_format_(output_format) {
    COMPILE_ASSERT(input_format.is_convertible(output_format),
            "input format " << input_format << " can not convert to "
                            << output_format << ".");
    plain_dims_ = v->details_.get_plain_dims();
    info_.inputs_.emplace_back(std::move(v));
    info_.outputs_.emplace_back(
            std::make_shared<graph_tensor>(this, info_.inputs_[0]->details_));
    info_.outputs_[0]->details_.set_format(output_format);
    op_name_ = "reorder";
}

void reorder_op_t::prepare_fusion_data(fdata_map &fdmap) {
    auto &in_detail0 = fdmap.get(info_.inputs_[0]);
    in_detail0.use_count_++;
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
    COMPILE_ASSERT(block_num_length.isa<constant_c>()
                    && block_size_length.isa<constant_c>(),
            "constant length is expected, but got "
                    << block_num_length << " and " << block_size_length);
    int block_num_length_int
            = get_const_as_int(block_num_length.checked_as<constant_c>());
    int block_size_length_int
            = get_const_as_int(block_size_length.checked_as<constant_c>());

    std::vector<std::pair<expr, expr>> plain_range_list;
    if (block_size_length_int == blocks) {
        // when block size is equal to blocks, reorder will generate
        // consequent slice in output
        auto plain_range
                = std::make_pair(block_num_start * blocks + block_size_start,
                        expr(block_num_length_int * block_size_length_int));
        plain_range_list = {plain_range};
    } else {
        // multi plain ranges
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

// get greatest common divisor of block_in and block_out
inline int get_gcd(int a, int b) {
    COMPILE_ASSERT(a * b != 0, "non-zero number is expected");
    int i = std::min(a, b);
    while (a % i != 0 || b % i != 0) {
        i--;
        if (i == 0) return 1;
    }
    return i;
}

// Get plain to block ranges
std::vector<std::pair<std::pair<expr, expr>, std::pair<expr, expr>>>
get_plain2block_ranges(const expr &start, const expr &length, int blocks) {
    std::vector<std::pair<std::pair<expr, expr>, std::pair<expr, expr>>> ret;
    COMPILE_ASSERT(length.isa<constant_c>(),
            "constant length is expected, but got " << length);
    int ilength = get_const_as_int(length.checked_as<constant_c>());
    expr folded_start
            = constant_folder_t()(auto_caster_t()(start)).remove_const();

    // Case 1: the most commone case.
    if (folded_start.isa<constant>() && get_expr_as_int(folded_start) == 0) {
        if (ilength >= blocks) {
            auto block_num_range = std::make_pair(0, ilength / blocks);
            auto block_size_range = std::make_pair(0, blocks);
            ret.emplace_back(std::make_pair(
                    std::move(block_num_range), std::move(block_size_range)));
        }
        if (ilength % blocks != 0) {
            auto block_num_range = std::make_pair(ilength / blocks, 1);
            auto block_size_range = std::make_pair(0, ilength % blocks);
            ret.emplace_back(std::make_pair(
                    std::move(block_num_range), std::move(block_size_range)));
        }
    } else {
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
                    int gcd = get_gcd(multiple, blocks);
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
    int ndim_begin = static_cast<int>(input_slice.size())
            - input_format.format_code_.norig_dims();
    output_slice = input_slice;
    for (int i = 0; i < input_format.format_code_.norig_dims(); i++) {
        int plain_axis = input_format.format_code_.get(i);
        output_slice[plain_axis + ndim_begin] = input_slice[i + ndim_begin];
    }
}

void infer_plain2stride_reorder(slice_range &input_slice,
        sc_data_format_t input_format, sc_data_format_t output_format,
        slice_range &output_slice) {
    int ndim_begin = static_cast<int>(input_slice.size())
            - input_format.format_code_.norig_dims();
    output_slice = input_slice;
    for (int i = 0; i < output_format.format_code_.norig_dims(); i++) {
        int plain_axis = output_format.format_code_.get(i);
        output_slice[i + ndim_begin] = input_slice[plain_axis + ndim_begin];
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

// infer plain format to blocking format generally
void infer_stride2block_reorder(slice_range_list &input_slice_list,
        sc_data_format_t input_format, sc_data_format_t output_format,
        slice_range_list &output_slice_list) {
    auto plain_format = input_format.to_plain();
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
        int ndim_begin = static_cast<int>(plain_slice.size())
                - plain_format.format_code_.ndims();

        for (int i = 0; i < out_kind.ndims(); i++) {
            int plain_pos = out_kind.get(i);
            block_cnt_dict[plain_pos].emplace_back(i);
            if (block_slice_dict[plain_pos].empty()) {
                block_slice_dict[plain_pos].emplace_back(
                        slice_range {plain_slice[ndim_begin + plain_pos]});
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

        std::vector<slice_range> total_range_list(out_kind.ndims());
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

        if (ndim_begin) {
            for (auto &range : reorder_ranges_list) {
                range.insert(range.begin(), input_slice.begin(),
                        input_slice.begin() + ndim_begin);
            }
        }

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
        int ndim_begin = static_cast<int>(input_slice.size())
                - input_format.format_code_.ndims();
        // from right to left
        for (int i = in_kind.ndims() - 1; i >= 0; i--) {
            int plain_pos = in_kind.get(i);
            if (plain_slice_dict[plain_pos].empty()) {
                plain_slice_dict[plain_pos].emplace_back(
                        slice_range {input_slice[i + ndim_begin]});
            } else {
                std::pair<expr, expr> cur_block_num_range
                        = input_slice[i + ndim_begin];
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

        if (ndim_begin) {
            for (auto &range : reorder_ranges_list) {
                // plain -> stride
                slice_range stride_range;
                infer_plain2stride_reorder(range, output_format.to_plain(),
                        output_format, stride_range);
                stride_range.insert(stride_range.begin(), input_slice.begin(),
                        input_slice.begin() + ndim_begin);
                range = std::move(stride_range);
            }
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
        SC_WARN << ss.str();
        throw tuner_recoverable_exception_t(ss.str());
    }
    constant_folder_t f;
    auto_caster_t ca;
    for (auto &reorder_range : output_slice_list) {
        for (auto &r : reorder_range) {
            r.first = f.expand_polynomial(ca(r.first)).remove_const();
        }
    }
}

// infer reorder slice according input_slice
void reorder_op_t::infer_slice_ranges(
        fslice_map &fsmap, infer_status_map_t &stat_map) {
    // has been pre-inferred, skip
    if (!fsmap.get(get_outputs()[0]).empty()) return;
    COMPILE_ASSERT(input_format_.is_convertible(output_format_),
            "Can not convert input format "
                    << input_format_ << " to output format " << output_format_
                    << ".");
    // search known ranges from any input of cur fusbile op
    slice_range_map known_ranges_map = search_known_slice_ranges(this, fsmap);
    auto input_slice_list = known_ranges_map[0];
    slice_range_list reorder_ranges_list;

    infer_reorder_slice(input_slice_list, input_format_, output_format_,
            reorder_ranges_list);
    if (reorder_ranges_list.empty()) {
        for (auto &user : get_outputs()[0]->uses_) {
            if (user.second->isa<output_op>()) {
                continue;
            } else {
                user.second->attrs_.set(op_attr_key::fused_mode_hint,
                        op_attr_key::break_pre_fuse);
                stat_map.get_ops_by_status(infer_status_code::FAIL)
                        .emplace_back(user.second);
                return;
            }
        }
    }
    fsmap.get(get_outputs()[0]) = reorder_ranges_list;
}

// pre-infer reorder slice according output_slice
void reorder_op_t::pre_slice_ranges(
        fslice_map &fsmap, infer_status_map_t &stat_map) {
    if (fsmap.get(get_inputs()[0]).empty()) {
        slice_range_list known_ranges_list = fsmap.get(get_outputs()[0]);
        slice_range_list input_slice_list;

        infer_reorder_slice(known_ranges_list, output_format_, input_format_,
                input_slice_list);
        if (input_slice_list.empty()) {
            stat_map.append_ops_by_status(this, infer_status_code::FAIL);
            return;
        }
        fsmap.get(get_inputs()[0]) = input_slice_list;
        // recursively pre-infer
        info_.inputs_[0]
                ->producer_owner_->dyn_cast<fusible_op_t>()
                ->pre_slice_ranges(fsmap, stat_map);
    }
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
    if (in_format.format_code_.is_batch_format()) {
        COMPILE_ASSERT(in_indexes.size() >= num_plain_dims,
                "Wrong number of dimensions for batch format: "
                        << in_format << ", real shape = "
                        << utils::print_vector(in_indexes));
        base_out_dim = in_indexes.size() - num_plain_dims;
        num_out_dims = base_out_dim + num_plain_dims;
        ret.resize(num_out_dims, 0);
        for (size_t i = 0; i < base_out_dim; i++) {
            ret[i] = (in_indexes[i]);
        }
    } else {
        COMPILE_ASSERT(in_indexes.size() == num_plain_dims,
                "Wrong number of dimensions for format: "
                        << in_format << ", real shape = "
                        << utils::print_vector(in_indexes));
    };

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

static std::vector<expr> get_reorder_block2plain_indexes(
        const std::vector<expr> &in_indexes, const sc_data_format_t &format,
        const sc_dims &plain_dims, expr &condition) {
    if (in_indexes.empty()) { return std::vector<expr>(); }
    COMPILE_ASSERT(format.format_code_ != format_kinds::any,
            "format can not be any_t in reorder op, please check it in layout "
            "propagation.");
    size_t base_out_dim = 0;
    size_t num_plain_dims = format.format_code_.norig_dims();
    size_t num_format_dims = format.format_code_.ndims();
    size_t num_out_dims = num_plain_dims;
    std::vector<expr> ret(num_out_dims, 0);
    if (format.format_code_.is_batch_format()) {
        COMPILE_ASSERT(in_indexes.size() >= num_format_dims,
                "Wrong number of dimensions for batch format: "
                        << format << ", real shape = "
                        << utils::print_vector(in_indexes));
        base_out_dim = in_indexes.size() - num_format_dims;
        num_out_dims = base_out_dim + num_plain_dims;
        ret.resize(num_out_dims, 0);
        for (size_t i = 0; i < base_out_dim; i++) {
            ret[i] = (in_indexes[i]);
        }
    } else {
        COMPILE_ASSERT(in_indexes.size() == num_format_dims,
                "Wrong number of dimensions for format: "
                        << format << ", real shape = "
                        << utils::print_vector(in_indexes));
    };

    COMPILE_ASSERT(in_indexes.size() <= sc_data_format_kind_t::MAX_DIMS,
            "Too many dims in plain shapes");
    condition = true;
    std::unordered_map<int, int>
            axis2blocks; // plain_axis to block idx, idx++ after an access
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
                condition = condition
                        && ret[base_out_dim + orig_axis] < dim2unsigned(
                                   plain_dims[base_out_dim + orig_axis]);
            } else {
                condition = condition
                        && ret[base_out_dim + orig_axis]
                                < format.blocks_[blocks[axis2blocks[orig_axis]
                                        - 1]];
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
    size_t num_plain_dims = format.format_code_.norig_dims();
    size_t num_format_dims = format.format_code_.ndims();
    size_t num_out_dims = num_format_dims;
    std::vector<expr> ret(num_out_dims, 0);
    if (format.format_code_.is_batch_format()) {
        COMPILE_ASSERT(in_indexes.size() >= num_plain_dims,
                "Wrong number of dimensions for batch format: "
                        << format << ", real shape = "
                        << utils::print_vector(in_indexes));
        base_out_dim = in_indexes.size() - num_plain_dims;
        num_out_dims = base_out_dim + num_format_dims;
        ret.resize(num_out_dims, 0);
        for (size_t i = 0; i < base_out_dim; i++) {
            ret[i] = in_indexes[i];
        }
    } else {
        COMPILE_ASSERT(in_indexes.size() == num_plain_dims,
                "Wrong number of dimensions for format: "
                        << format << ", real shape = "
                        << utils::print_vector(in_indexes));
    };

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

void compute_reorder_stride2stride(const context_ptr &ctx,
        const tensor_slice &src, tensor_slice &dst,
        const sc_data_format_t &input_format,
        const sc_data_format_t &output_format, sc_data_type_t dtype,
        const sc_dims &plain_dims, any_map_t &attrs, size_t wkld = 0UL) {
    auto input = src.get_real_tensor();
    auto output = dst.get_real_tensor();
    int step = static_cast<int>(vectorize_step(ctx, dtype.type_code_));
    auto bld = builder::get_current_builder();
    std::vector<expr> iter_vars;
    std::vector<expr> in_indexes;
    std::vector<expr> loop_indexes;
    for (size_t i = 0; i < plain_dims.size(); i++) {
        iter_vars.emplace_back(builder::make_var(datatypes::index,
                std::string("_fuseiter") + fusion_create_idx()));
        loop_indexes.emplace_back(iter_vars[i]);
        in_indexes.emplace_back(iter_vars[i] + src.get_offset()[i]);
    }
    std::vector<expr> out_indexes = get_reorder_stride2stride_indexes(
            in_indexes, input_format, output_format, plain_dims);

    auto cur = builder::make_stmts_unattached({builder::make_assign_unattached(
            builder::make_indexing(output, out_indexes),
            builder::make_indexing(src.tptr_, loop_indexes))});
    cur->attr()[op_traits::workload_computable_t::workload_number] = wkld;
    stmt body;
    std::vector<stmt> loops;
    for (int i = static_cast<int>(plain_dims.size()) - 1; i >= 0; i--) {
        body = cur.isa<stmts>()
                ? cur
                : make_stmt<stmts_node_t>(std::vector<stmt> {std::move(cur)});
        cur = make_stmt<for_loop_node_t>(std::move(iter_vars.at(i)), expr(0),
                src.get_shape()[i], expr(1), std::move(body), true,
                for_type::NORMAL);
        loops.push_back(cur);
    }
    std::reverse(loops.begin(), loops.end());
    for (size_t i = 0; i < plain_dims.size() - 2; i++) {
        loops[0].checked_as<for_loop>()->fuse(loops[i].checked_as<for_loop>());
    }
    bld->emit(cur);
}

void compute_reorder_block2stride(const context_ptr &ctx,
        const tensor_slice &src, tensor_slice &dst,
        const sc_data_format_t &input_format,
        const sc_data_format_t &output_format, sc_data_type_t dtype,
        const sc_dims &plain_dims, any_map_t &attrs, size_t wkld = 0UL) {
    auto input = src.get_real_tensor();
    auto output = dst.get_real_tensor();
    int step = static_cast<int>(vectorize_step(ctx, dtype.type_code_));
    auto bld = builder::get_current_builder();
    auto input_blocking_dims
            = sc_data_format_t::get_blocking_shapes(plain_dims, input_format);
    auto output_blocking_dims
            = sc_data_format_t::get_blocking_shapes(plain_dims, output_format);
    assert(output_format.format_code_.ndims()
            == output_format.format_code_.norig_dims());
    auto output_last_origin_axis = output_format.format_code_.get(
            output_format.format_code_.ndims() - 1);
    auto block_axis = input_format.format_code_.collect_blocking_index(
            output_last_origin_axis); // {block_idx1, block_idx2,...}
    bool can_vectorize = !block_axis.empty()
            && block_axis.at(block_axis.size() - 1)
                    == input_format.get_blocks_size() - 1
            && input_blocking_dims[input_blocking_dims.size() - 1] % step == 0
            && plain_dims[plain_dims.size() - 1] % step == 0;
    if (can_vectorize && attrs.get_or_else(op_attr_key::no_fuse, false)) {
        int max_step = ctx->get_max_vector_lanes(dtype.type_code_);
        while (step < max_step
                && input_blocking_dims[input_blocking_dims.size() - 1]
                                % (2 * step)
                        == 0
                && plain_dims[plain_dims.size() - 1] % (2 * step) == 0) {
            step = 2 * step;
        }
    }
    bool no_padding = sc_data_format_t::get_padded_plain_shapes(
                              input_blocking_dims, input_format)
            == sc_data_format_t::get_padded_plain_shapes(
                    output_blocking_dims, output_format);

    // For no padding case, vectorize judgement should be more strictly for src
    // due to input maybe fused by previous op
    if (no_padding)
        can_vectorize = can_vectorize && src.get_shape().back().isa<constant>()
                && get_expr_as_int(src.get_shape().back()) % step == 0;

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
    std::vector<expr> tmp_out_indexes = get_reorder_block2plain_indexes(
            in_indexes, input_format, plain_dims, condition);
    std::vector<expr> out_indexes
            = get_reorder_stride2stride_indexes(tmp_out_indexes,
                    output_format.to_plain(), output_format, plain_dims);

    auto assign
            = builder::make_stmts_unattached({builder::make_assign_unattached(
                    builder::make_indexing(output, out_indexes, step),
                    // here, use src.tptr instead of input is aimed to avoid
                    // input is tensor_view_op. Oherwisw, it will throw
                    // illegal exception in index_flatten
                    builder::make_indexing(
                            expr(src.tptr_), loop_indexes, step))});
    assign->attr()[op_traits::workload_computable_t::workload_number] = wkld;
    auto cur = no_padding
            ? assign
            : builder::make_if_else_unattached(condition, assign, stmt());
    stmt body;
    for (int i = static_cast<int>(input_blocking_dims.size()) - 1; i >= 0;
            i--) {
        body = cur.isa<stmts>()
                ? cur
                : make_stmt<stmts_node_t>(std::vector<stmt> {std::move(cur)});
        cur = make_stmt<for_loop_node_t>(std::move(iter_vars.at(i)), expr(0),
                src.get_shape()[i],
                i == static_cast<int>(input_blocking_dims.size()) - 1
                                && can_vectorize
                        ? expr(static_cast<int>(step))
                        : expr(1),
                std::move(body), true, for_type::NORMAL);
    }
    cur->attr()[stmt_attr_key::merge_loop] = true;
    bld->emit(cur);
}

void compute_reorder_stride2block(const context_ptr &ctx,
        const tensor_slice &src, tensor_slice &dst,
        const sc_data_format_t &input_format,
        const sc_data_format_t &output_format, sc_data_type_t dtype,
        const sc_dims &plain_dims, bool output_loop, any_map_t &attrs,
        size_t wkld = 0UL) {
    auto input = src.get_real_tensor();
    auto output = dst.get_real_tensor();
    int step = static_cast<int>(vectorize_step(ctx, dtype.type_code_));
    auto bld = builder::get_current_builder();
    auto input_blocking_dims
            = sc_data_format_t::get_blocking_shapes(plain_dims, input_format);
    auto output_blocking_dims
            = sc_data_format_t::get_blocking_shapes(plain_dims, output_format);
    assert(input_format.format_code_.ndims()
            == input_format.format_code_.norig_dims());
    auto input_last_origin_axis = input_format.format_code_.get(
            input_format.format_code_.ndims() - 1);
    auto block_axis = output_format.format_code_.collect_blocking_index(
            input_last_origin_axis); // {block_idx1, block_idx2,...}
    bool can_vectorize = !block_axis.empty()
            && block_axis.at(block_axis.size() - 1)
                    == output_format.get_blocks_size() - 1
            && output_blocking_dims[output_blocking_dims.size() - 1] % step == 0
            && plain_dims[plain_dims.size() - 1] % step == 0;
    bool no_padding = sc_data_format_t::get_padded_plain_shapes(
                              output_blocking_dims, output_format)
            == sc_data_format_t::get_padded_plain_shapes(
                    input_blocking_dims, input_format);

    // For no padding case, vectorize judgement should be more strictly for src
    // due to input maybe fused by previous op
    if (no_padding) {
        can_vectorize = can_vectorize && src.get_shape().back().isa<constant>()
                && get_expr_as_int(src.get_shape().back()) % step == 0;
    }

    if (can_vectorize && attrs.get_or_else(op_attr_key::no_fuse, false)) {
        int max_step = ctx->get_max_vector_lanes(dtype.type_code_);
        while (step < max_step
                && output_blocking_dims[output_blocking_dims.size() - 1]
                                % (2 * step)
                        == 0
                && plain_dims[plain_dims.size() - 1] % (2 * step) == 0) {
            step = 2 * step;
        }
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
                    i == static_cast<int>(input_blocking_dims.size()) - 1
                                    && can_vectorize
                            ? expr(static_cast<int>(step))
                            : expr(1),
                    std::move(body), true, for_type::NORMAL);
        }
        cur->attr()[stmt_attr_key::merge_loop] = true;
        bld->emit(cur);
    } else {
        std::vector<expr> out_indexes;
        for (size_t i = 0; i < output_blocking_dims.size(); i++) {
            iter_vars.emplace_back(builder::make_var(datatypes::index,
                    std::string("_fuseiter") + fusion_create_idx()));
            out_indexes.emplace_back(iter_vars[i] + dst.get_offset()[i]);
        }
        expr condition;
        std::vector<expr> tmp_in_indexes = get_reorder_block2plain_indexes(
                out_indexes, output_format, plain_dims, condition);
        std::vector<expr> in_indexes
                = get_reorder_stride2stride_indexes(tmp_in_indexes,
                        input_format.to_plain(), input_format, plain_dims);

        auto assign = builder::make_stmts_unattached(
                {builder::make_assign_unattached(
                        builder::make_indexing(expr(output), out_indexes, step),
                        builder::make_indexing(
                                expr(input), in_indexes, step))});
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
                    no_padding ? dst.get_shape()[i]
                               : dim2unsigned(output_blocking_dims[i]),
                    i == static_cast<int>(output_blocking_dims.size()) - 1
                                    && can_vectorize
                            ? expr(static_cast<int>(step))
                            : expr(1),
                    std::move(body), true, for_type::NORMAL);
            loops.push_back(cur);
        }
        bld->emit(cur);
    }
}

void compute_reorder_block2block(const context_ptr &ctx,
        const tensor_slice &src, tensor_slice &dst,
        const sc_data_format_t &input_format,
        const sc_data_format_t &output_format, sc_data_type_t dtype,
        const sc_dims &plain_dims1, bool output_loop, any_map_t &attrs,
        size_t wkld = 0UL) {
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
    auto input_padded_plain_dims = sc_data_format_t::get_padded_plain_shapes(
            input_blocking_dims, input_format);
    auto output_padded_plain_dims = sc_data_format_t::get_padded_plain_shapes(
            output_blocking_dims, output_format);
    // plain axis of last block
    auto input_block_axis = input_format.format_code_.get(
            input_format.format_code_.ndims() - 1);
    auto output_block_axis = output_format.format_code_.get(
            output_format.format_code_.ndims() - 1);
    bool no_padding = input_padded_plain_dims == output_padded_plain_dims;
    bool can_vectorize = input_block_axis == output_block_axis
            && output_blocking_dims[output_blocking_dims.size() - 1] % step == 0
            && input_blocking_dims[input_blocking_dims.size() - 1] % step == 0
            && no_padding;

    // For no padding case, vectorize judgement should be more strictly for src
    // due to input maybe fused by previous op
    if (no_padding)
        can_vectorize = can_vectorize && src.get_shape().back().isa<constant>()
                && get_expr_as_int(src.get_shape().back()) % step == 0;

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
        std::vector<expr> tmp_indexes = get_reorder_block2plain_indexes(
                in_indexes, input_format, plain_dims, condition);
        std::vector<expr> out_indexes
                = get_reorder_plain2block_indexes(tmp_indexes, output_format);
        auto cur = builder::make_stmts_unattached(
                {builder::make_assign_unattached(
                        builder::make_indexing(output, out_indexes, step),
                        // here, use src.tptr instead of input is aimed to
                        // avoid input is tensor_view_op. Oherwisw, it will
                        // throw illegal exception in index_flatten
                        builder::make_indexing(
                                src.tptr_, loop_indexes, step))});
        cur->attr()[op_traits::workload_computable_t::workload_number] = wkld;
        stmt body;
        for (int i = static_cast<int>(input_blocking_dims.size()) - 1; i >= 0;
                i--) {
            body = cur.isa<stmts>() ? cur
                                    : make_stmt<stmts_node_t>(
                                            std::vector<stmt> {std::move(cur)});
            cur = make_stmt<for_loop_node_t>(std::move(iter_vars.at(i)),
                    expr(0), src.get_shape()[i],
                    i == static_cast<int>(input_blocking_dims.size()) - 1
                                    && can_vectorize
                            ? expr(static_cast<int>(step))
                            : expr(1),
                    std::move(body), true, for_type::NORMAL);
        }
        cur->attr()[stmt_attr_key::merge_loop] = true;
        bld->emit(cur);
    } else {
        std::vector<expr> out_indexes;
        for (size_t i = 0; i < output_blocking_dims.size(); i++) {
            iter_vars.emplace_back(builder::make_var(datatypes::index,
                    std::string("_fuseiter") + fusion_create_idx()));
            out_indexes.emplace_back(iter_vars[i] + dst.get_offset()[i]);
        }
        expr condition;
        std::vector<expr> tmp_indexes = get_reorder_block2plain_indexes(
                out_indexes, output_format, plain_dims, condition);
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
                        builder::make_indexing(output, out_indexes),
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
                    no_padding ? dst.get_shape()[i]
                               : dim2unsigned(output_blocking_dims[i]),
                    i == static_cast<int>(output_blocking_dims.size()) - 1
                                    && can_vectorize
                            ? expr(static_cast<int>(step))
                            : expr(1),
                    std::move(body), true, for_type::NORMAL);
            loops.push_back(cur);
        }
        std::reverse(loops.begin(), loops.end());
        for (size_t i = 0; i < output_blocking_dims.size() - 2; i++) {
            loops[0].checked_as<for_loop>()->fuse(
                    loops[i].checked_as<for_loop>());
        }
        bld->emit(cur);
    }
}

void compute_reorder_block(const context_ptr &ctx, const tensor_slice &src,
        tensor_slice &dst, const sc_data_format_t &input_format,
        const sc_data_format_t &output_format, sc_data_type_t dtype,
        const sc_dims &plain_dims, bool output_loop, any_map_t &attrs,
        size_t wkld = 0UL) {
    COMPILE_ASSERT(input_format.is_convertible(output_format),
            "Can not convert input format "
                    << input_format << " to output format " << output_format
                    << ".");
    if (is_not_blocking(input_format) && is_not_blocking(output_format)) {
        compute_reorder_stride2stride(ctx, src, dst, input_format,
                output_format, dtype, plain_dims, attrs, wkld);
    } else if (is_not_blocking(input_format) && output_format.is_blocking()) {
        compute_reorder_stride2block(ctx, src, dst, input_format, output_format,
                dtype, plain_dims, output_loop, attrs, wkld);
    } else if (input_format.is_blocking() && is_not_blocking(output_format)) {
        compute_reorder_block2stride(ctx, src, dst, input_format, output_format,
                dtype, plain_dims, attrs, wkld);
    } else if (input_format.is_blocking() && output_format.is_blocking()) {
        compute_reorder_block2block(ctx, src, dst, input_format, output_format,
                dtype, plain_dims, output_loop, attrs, wkld);
    } else {
        std::ostringstream ss;
        ss << "Unsupported data format. in = " << input_format
           << ", out = " << output_format;
        SC_WARN << ss.str();
        throw tuner_recoverable_exception_t(ss.str());
    }
}

size_t reorder_op_t::compute_workload(const std::vector<shape_dtype_pair> &ins,
        const std::vector<shape_dtype_pair> &outs) {
    return fusible_op_t::compute_workload(ins, outs)
            * workload_penalty_coefficient;
}

bool reorder_op_t::check_padding() const {
    auto input_blocking_dims
            = sc_data_format_t::get_blocking_shapes(plain_dims_, input_format_);
    auto output_blocking_dims = sc_data_format_t::get_blocking_shapes(
            plain_dims_, output_format_);
    return sc_data_format_t::get_padded_plain_shapes(
                   input_blocking_dims, input_format_)
            != sc_data_format_t::get_padded_plain_shapes(
                    output_blocking_dims, output_format_);
}

bool reorder_op_t::use_output_loop() const {
    if (check_padding()) return true;
    if (attrs_.get_or_else(op_attr_key::no_fuse, false)) {
        if (!get_input_format().is_blocking()) return true;
    }
    if (attrs_.get_or_else(op_attr_key::break_pre_fuse, false)) return true;
    return false;
}

void reorder_op_t::compute_block(context_ptr ctx,
        const std::vector<tensor_slice *> &dst,
        const std::vector<const tensor_slice *> &inputs) {
    size_t wkld = compute_fusible_workload(ctx, dst, inputs);
    compute_reorder_block(ctx, *inputs[0], *dst[0], input_format_,
            output_format_, info_.inputs_[0]->details_.dtype_, plain_dims_,
            use_output_loop(), attrs_, wkld);
}

} // namespace sc
