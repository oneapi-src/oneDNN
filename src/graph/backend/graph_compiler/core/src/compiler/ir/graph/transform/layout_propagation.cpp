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

#include <utility>

#include <functional>
#include <limits>
#include "../dynamic_dispatch_key.hpp"
#include "../fusible_op.hpp"
#include "../graph_op.hpp"
#include "../pass/pass.hpp"
#include "../tunable_op.hpp"
#include "../visitor.hpp"
#include <ops/convolution.hpp>
#include <ops/fusible/broadcast.hpp>
#include <ops/fusible/memory_movement.hpp>
#include <ops/fusible/reduce.hpp>
#include <ops/fusible/ternary_elemwise.hpp>
#include <ops/reshape.hpp>
namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

SC_MODULE(graph.layout_propagation);

using reorder_callback_type = std::function<void(
        const graph_tensor_ptr &in, const sc_data_format_t &target_formats)>;
using reorder_map_t = std::unordered_map<graph_tensor_ptr,
        std::unordered_map<sc_op_ptr, sc_op_ptr>>;

static constexpr const char *reorder_format_set_key = "temp.reorder_format_set";
static void insert_reorder_op(sc_graph_t &graph, reorder_map_t &reorder_map,
        const graph_tensor_ptr &in, size_t in_index,
        const format_stride_pair &out_format_stride, const sc_op_ptr &cur_op,
        bool is_input_plain, reorder_callback_type &on_insert_reorder) {
    // if we don't need to keep plain format for input op and input op tensor is
    // currently not blocking, copy the blocking format
    // If the output has multiply uses, choose one format for all that now to
    // minimize the number of reorder, mainly for bottleneck.
    if (!is_input_plain && !cur_op->is_dynamic()
            && in->producer_owner_->isa<input_op>()
            && in->details_.get_format().is_plain()
            && !in->producer_owner_->attrs_.get_or_else("keep_plain", false)) {
        in->details_.set_format_and_stride(
                out_format_stride.first, out_format_stride.second);
        return;
    }
    bool is_graph_dynamic = graph.is_dynamic();
    auto &in_format = in->details_.get_format();
    auto dispatch_key = op_dispatch_key_t(
            std::vector<sc_data_format_t> {in_format, out_format_stride.first});
    auto is_key_valid = [](const op_dispatch_key_t &key) {
        return !(key.in_out_formats_[0].is_blocking()
                       && key.in_out_formats_[1].is_blocking())
                || key.in_out_formats_[0] == key.in_out_formats_[1];
    }(dispatch_key);
    sc_op_ptr ret;
    // find reorder in map, if not insert one.
    auto tsr_it = reorder_map.find(in);
    if (tsr_it != reorder_map.end()) {
        auto &op_map = tsr_it->second;
        auto op_it = op_map.find(cur_op);
        if (op_it != op_map.end()) {
            ret = op_it->second;
            ret->get_outputs()[0]->details_.set_format_and_stride(
                    out_format_stride.first, out_format_stride.second);
            ret->attrs_.set("out_format", out_format_stride.first);
            ret->attrs_.set("out_stride", out_format_stride.second);
            // update fuse attr for latest format
            ret->stc_cast<reorder_op_t>()->update_fuse_attr();
            // map reorder's in/out
            if (is_graph_dynamic && is_key_valid) {
                auto &dynamic_formats
                        = ret->get_dispatch_key_set()->get_inner_set();
                // todo: remove internal reorder by following last tunable op
                // output layout.
                if (dynamic_formats.find(dispatch_key)
                        == dynamic_formats.end()) {
                    ret->get_outputs()[0]->details_.add_format_candidate(
                            out_format_stride.first);
                    dynamic_formats.insert(dispatch_key);
                }
                ret->get_outputs()[0]->details_.add_format_candidate(
                        out_format_stride.first);
            }
        }
    }
    bool create_reorder = !ret;
    if (create_reorder) {
        ret = graph.make("reorder", {in}, {},
                {{"internal", true}, {"out_format", out_format_stride.first},
                        {"out_stride", out_format_stride.second},
                        {op_attr_key::no_fuse, // work around for conv
                                // graph. will be dropped
                                // after yijie's refactor
                                graph.attrs_.get_or_else(
                                        "reorder_not_to_fuse", false)}});
        // map reorder's in/out
        if (is_graph_dynamic && is_key_valid) {
            ret->get_dispatch_key_set()->get_inner_set().insert(dispatch_key);
        }
        ret->get_outputs()[0]->details_.add_format_candidate(
                out_format_stride.first);
        reorder_map[in][cur_op] = ret;
    }
    // if we are in trying-mode, don't insert an reorder. Instead,
    // notify that there will be a reorder to be inerted
    if (on_insert_reorder) {
        in->details_.set_format_and_stride(
                out_format_stride.first, out_format_stride.second);
        on_insert_reorder(in, out_format_stride.first);
    } else {
        cur_op->replace_input(in_index, ret->get_outputs()[0]);
    }
}

static void remove_inserted_reorder_op(sc_graph_t &graph,
        reorder_map_t &reorder_map, const graph_tensor_ptr &in,
        const sc_op_ptr &cur_op) {
    auto tsr_it = reorder_map.find(in);
    if (tsr_it != reorder_map.end()) {
        auto &op_map = tsr_it->second;
        auto op_it = op_map.find(cur_op);
        if (op_it != op_map.end()) { op_it->second->remove(); }
    }
}

static void update_output_formats(std::vector<graph_tensor_ptr> &outs,
        const std::vector<std::vector<format_stride_pair>> &out_supported_pairs,
        size_t layout_choice) {
    for (size_t i = 0; i < outs.size(); ++i) {
        if (!out_supported_pairs.empty()) {
            auto &fs_pair = out_supported_pairs[i][layout_choice];
            outs[i]->details_.set_format_and_stride(
                    fs_pair.first, fs_pair.second);
        } else if (outs[i]->details_.get_format().is_any()) {
            outs[i]->details_.set_format({sc_data_format_t::get_plain_by_dims(
                    (int)outs[i]->details_.get_plain_dims().size())});
        }
        outs[i]->details_.add_format_candidate(outs[i]->details_.get_format());
    }
}

static void check_input_format(const std::vector<graph_tensor_ptr> &ins) {
    for (auto &in : ins) {
        COMPILE_ASSERT(!in->details_.get_format().is_any(),
                "input format don't allow any format");
    }
}

static void insert_reorder_for_output_op(reorder_map_t &reorder_map,
        const sc_op_ptr &node, bool use_channel_last_format, bool is_out_plain,
        bool is_input_plain, bool is_graph_dynamic, sc_graph_t &graph,
        reorder_callback_type &insert_reorder_callback) {
    auto given_target_formats
            = node->attrs_.get_or_null<std::vector<sc_data_format_t>>(
                    "target_formats");
    auto given_target_strides
            = node->attrs_.get_or_null<std::vector<sc_dims>>("target_strides");
    if (given_target_formats) {
        COMPILE_ASSERT(
                given_target_formats->size() == node->get_inputs().size(),
                "Output op's target_formats' size should be equal to "
                "number of tensors");
    }

    if (given_target_strides) {
        COMPILE_ASSERT(
                given_target_strides->size() == node->get_inputs().size(),
                "Output op's target_strides' size should be equal to "
                "number of tensors");
    }
    for (size_t i = 0; i < node->get_inputs().size(); ++i) {
        auto &in_detail = node->get_inputs()[i]->details_;
        sc_data_format_t target_format;
        sc_dims target_stride;
        if (given_target_formats) {
            target_format = (*given_target_formats)[i];
        } else if (is_out_plain) {
            if (use_channel_last_format) {
                target_format = in_detail.get_format().to_channel_last();
            } else {
                target_format = in_detail.get_format().to_plain();
            }
        } else {
            target_format = in_detail.get_format();
        }
        if (given_target_strides) {
            target_stride = (*given_target_strides)[i];
        } else if (is_out_plain) {
            // here stride is calculated according to plain dims & format
            if (use_channel_last_format) {
                // permute dense stride to channel last stride
                sc_dims permuted_dims = in_detail.get_plain_dims();
                size_t ndims = permuted_dims.size();
                sc_dim channel = permuted_dims[1];
                for (size_t d = 1; d < ndims - 1; ++d) {
                    permuted_dims[d] = permuted_dims[d + 1];
                }
                permuted_dims[ndims - 1] = channel;
                target_stride
                        = logical_tensor_t::compute_dense_stride(permuted_dims);
            } else {
                target_stride = logical_tensor_t::compute_dense_stride(
                        in_detail.get_plain_dims());
            }
        } else {
            if (given_target_formats) {
                auto dims = logical_tensor_t(target_format,
                        in_detail.get_plain_dims(), in_detail.dtype_)
                                    .get_blocking_dims();
                target_stride = logical_tensor_t::compute_dense_stride(dims);
            } else {
                target_stride = in_detail.get_strides();
            }
        }

        auto in = node->get_inputs()[i];
        COMPILE_ASSERT(!in_detail.get_format().is_any(),
                "output op's input format should have a concrete "
                "format, instead of any format");
        bool plain_check_failed = is_out_plain && target_format.is_blocking();
        COMPILE_ASSERT(!target_format.is_any() && !plain_check_failed,
                "output op's target format should be plain or "
                "permuted.")

        if (is_graph_dynamic) {
            auto old_format = in_detail.get_format();
            format_stride_pair target_fs_pair(
                    target_format, std::move(target_stride));
            for (auto &candidate : in_detail.get_format_candidates()) {
                in_detail.set_format(candidate);
                insert_reorder_op(graph, reorder_map, in, i, target_fs_pair,
                        node, is_input_plain, insert_reorder_callback);
            }
            in_detail.set_format(old_format);
        } else if (target_format != in_detail.get_format()
                || target_stride != in_detail.get_strides()) {
            format_stride_pair target_fs_pair(
                    target_format, std::move(target_stride));
            insert_reorder_op(graph, reorder_map, in, i, target_fs_pair, node,
                    is_input_plain, insert_reorder_callback);
        } else if (!insert_reorder_callback) {
            // if static and no need of reorder
            remove_inserted_reorder_op(graph, reorder_map, in, node);
        }
    }
}

static void combine_layout_and_impl_dispatch(
        const context_ptr &ctx, sc_graph_t &graph) {
    auto &ops = graph.ops_;
    for (auto &op : ops) {
        if (!op->is_dynamic()) { continue; }
        auto impl_candidates = op->get_impl_dispatch_candidates(ctx);
        if (impl_candidates.empty() || !op->is_dynamic()) { continue; }
        auto &key_set = op->get_dispatch_key_set()->get_inner_set();
        dispatch_key_set_t::inner_set_t new_set(key_set);
        for (auto key : key_set) {
            for (auto &impl : impl_candidates) {
                if (impl) {
                    key.impl_ = impl;
                    new_set.insert(key);
                }
            }
        }
        op->get_dispatch_key_set()->get_inner_set() = new_set;
    }
}

static bool has_channel_last_input(sc_graph_t &graph) {
    for (const auto &inputs : graph.get_input_ops()) {
        for (const auto &in_gt : inputs->get_outputs()) {
            if (in_gt->details_.get_format().is_channel_last()
                    && !in_gt->details_.get_format().is_blocking()) {
                return true;
            }
        }
    }
    return false;
}

static bool has_conv_op(sc_graph_t &graph) {
    auto pos = std::find_if(
            graph.ops_.begin(), graph.ops_.end(), [](const sc_op_ptr &op) {
                if (op->dyn_cast<ops::conv_fwd_core_op_t>()) {
                    return true;
                } else {
                    return false;
                }
            });
    return pos != graph.ops_.end();
}

// max times of layout tries with static shape
constexpr int STATIC_MAX_LAYOUT_TRIES = 64;

SC_INTERNAL_API void layout_propagation(
        sc_graph_t &graph, const context_ptr &ctx) {
    bool use_channel_last_format = has_channel_last_input(graph)
            && has_conv_op(graph)
            && graph.attrs_.get_or_else(
                    sc_graph_t::attr_key_t::allow_channel_last_output, false);
    bool is_input_plain = graph.attrs_.get_or_else(
            sc_graph_t::attr_key_t::is_input_plain, true);
    bool is_graph_dynamic = graph.is_dynamic()
            && graph.attrs_.get_or_else("insert_reorder", true);
    std::vector<sc_op_ptr> sorted_ops;
    sorted_ops.reserve(graph.ops_.size());
    std::vector<size_t> num_choices;
    num_choices.resize(graph.ops_.size(), 1);
    size_t total_choices = 1;
    std::vector<size_t> cur_choice;
    cur_choice.resize(graph.ops_.size(), 0);
    // the try-run will reset the input op's format. need to remember them
    std::vector<std::vector<format_stride_pair>> format_backup;
    std::vector<sc_op *> input_ops;

    op_visitor_t vis = op_visitor_t::dfs_topology_sort(graph.ops_.size());
    // stage 1, collect all ops and the number of choices of formats
    vis.visit_graph(graph, [&](op_visitor_t *vis, const sc_op_ptr &node) {
        sorted_ops.emplace_back(node);
        if (is_graph_dynamic) { return; }
        if (auto input_node = node->dyn_cast<input_op>()) {
            // backup the input's format
            std::vector<format_stride_pair> backup_info;
            for (auto &in : input_node->get_outputs()) {
                backup_info.emplace_back(std::make_pair(
                        in->details_.get_format(), in->details_.get_strides()));
            }
            format_backup.emplace_back(backup_info);
            input_ops.push_back(input_node);
        }
        if (auto tunable_node = node->dyn_cast<tunable_op_t>()) {
            std::vector<std::vector<format_stride_pair>> in_supported_pairs,
                    out_supported_pairs;
            bool has_config = bool(tunable_node->get_config());
            node->query_format(ctx, in_supported_pairs, out_supported_pairs);
            if (!has_config) { tunable_node->set_config(nullptr); }
            if (in_supported_pairs.empty()) {
                num_choices[node->logical_op_id_] = 1;
            } else {
                num_choices[node->logical_op_id_]
                        = in_supported_pairs[0].size();
                if (!graph.is_dynamic()
                        && total_choices <= STATIC_MAX_LAYOUT_TRIES)
                    total_choices
                            = total_choices * num_choices[node->logical_op_id_];
            }
            SC_MODULE_INFO << node->op_name_ << '_' << node->logical_op_id_
                           << " has num_choices="
                           << num_choices[node->logical_op_id_];
        } else {
            num_choices[node->logical_op_id_] = 1;
        }
    });

    auto reset_input_fmt = [&]() {
        // restore input formats
        for (size_t input_cnt = 0; input_cnt < input_ops.size(); input_cnt++) {
            auto input_node = input_ops[input_cnt];
            // backup the input's format
            auto &fmts = format_backup[input_cnt];
            auto &outs = input_node->get_outputs();
            for (size_t i = 0; i < outs.size(); i++) {
                outs[i]->details_.set_format_and_stride(
                        fmts[i].first, fmts[i].second);
            }
        }
    };
    reorder_callback_type insert_reorder_callback;
    // the map is used to record the reorder between original tensor and op. In
    // try mode, we insert an reorder but does not change the connection. If we
    // work on dynamic shape or really need the reorder in static, we reserve it
    // and insert it between graph tensor and op. If not, we delete it.
    reorder_map_t reorder_map;
    bool is_out_plain = graph.attrs_.get_or_else(
            sc_graph_t::attr_key_t::is_output_plain, true);
    auto do_visit = [&](const sc_op_ptr &node) {
        if (node->isa<output_op>()) {
            insert_reorder_for_output_op(reorder_map, node,
                    use_channel_last_format, is_out_plain, is_input_plain,
                    is_graph_dynamic, graph, insert_reorder_callback);
        } else if (node->isa<input_op>() || node->isa<constant_op_t>()) {
            update_output_formats(node->info_.outputs_, {}, 0);
        } else {
            std::vector<std::vector<format_stride_pair>> in_supported_pairs,
                    out_supported_pairs;
            auto reset_in_out_supported_pairs = [&]() {
                in_supported_pairs.clear();
                out_supported_pairs.clear();
            };
            // we need to reset the config after query_format if we need to use
            // the default config for tunable_ops
            bool reset_config = false;

            if (insert_reorder_callback) {
                // we only need to reset the config when we are in "try" mode
                auto tunable_node = node->dyn_cast<tunable_op_t>();
                if (tunable_node && !tunable_node->get_config()) {
                    // if it is tunable and has no user-defined config
                    reset_config = true;
                }
            }
            node->query_format(ctx, in_supported_pairs, out_supported_pairs);

            if (reset_config) {
                node->stc_cast<tunable_op_t>()->set_config(nullptr);
            }
            // as we will insert reorder in dynamic mode, we should record
            // previous inputs.
            auto inputs_backup = node->info_.inputs_;
            auto &inputs
                    = is_graph_dynamic ? inputs_backup : node->info_.inputs_;
            auto &outputs = node->info_.outputs_;
            check_input_format(inputs);
            size_t cur_layout_choice = cur_choice[node->logical_op_id_];
            if (node->isa<tunable_op_t>()
                    || node->isa<binary_elementwise_op_t>()
                    || node->isa<tensor_view_op_t>()
                    || node->isa<ops::dynamic_reshape_op>()
                    || node->isa<concat_op_t>() || node->isa<select_op_t>()
                    || node->isa<broadcast_op_t>() || node->isa<reshape_op_t>()
                    || node->isa<reduce_op_t>()) {
                std::vector<sc_data_format_t> old_formats;
                std::vector<std::vector<sc_data_format_t>>
                        input_format_candidates;
                std::vector<size_t> format_choice;
                size_t total_format_choice = 1;
                old_formats.reserve(inputs.size());
                input_format_candidates.reserve(inputs.size());
                for (size_t i = 0; i < inputs.size(); ++i) {
                    old_formats.push_back(inputs[i]->details_.get_format());
                    std::set<sc_data_format_t, sc_data_format_cmper_t>
                            format_candidates;
                    if (is_graph_dynamic) {
                        auto &inp_candidates
                                = inputs[i]->details_.get_format_candidates();
                        format_candidates.insert(
                                inp_candidates.begin(), inp_candidates.end());
                    } else {
                        format_candidates.insert(
                                inputs[i]->details_.get_format());
                    }
                    input_format_candidates.emplace_back(
                            format_candidates.begin(), format_candidates.end());
                    format_choice.push_back(format_candidates.size());
                    total_format_choice *= format_choice.back();
                }
                for (size_t c = 0; c < total_format_choice; c++) {
                    size_t format_idx = c;
                    // for dynamic dispatch of fusible op.
                    std::vector<sc_data_format_t> dispatch_format;
                    dispatch_format.reserve(inputs.size() + outputs.size());
                    if (is_graph_dynamic) {
                        for (size_t i = input_format_candidates.size(); i > 0;
                                i--) {
                            inputs[i - 1]->details_.set_format(
                                    input_format_candidates[i - 1][format_idx
                                            % format_choice[i - 1]]);
                            format_idx /= format_choice[i - 1];
                        }
                    } else {
                        COMPILE_ASSERT(total_format_choice == 1,
                                "Static traverses one format at a time.");
                    }
                    // tunable op should return same formats in dynamic
                    // shape.
                    if (is_graph_dynamic) {
                        // tuanble op also need re-query as the internal layer
                        // rely on previous layer's output layout.
                        assert(cur_layout_choice == 0);
                        reset_in_out_supported_pairs();
                        if (node->isa<tunable_op_t>()) {
                            node->stc_cast<tunable_op_t>()->set_config(nullptr);
                        }
                        node->query_format(
                                ctx, in_supported_pairs, out_supported_pairs);
                    }
                    // need to unify input formats
                    // todo: should check add_op input shape, output shape
                    // size = max(input size), so need to enhance
                    if (!in_supported_pairs.empty()
                            && !out_supported_pairs.empty()) {
                        for (size_t i = 0; i < inputs.size(); i++) {
                            auto &target_fs_pair
                                    = in_supported_pairs[i][cur_layout_choice];
                            format_stride_pair in_fs_pair(
                                    inputs[i]->details_.get_format(),
                                    inputs[i]->details_.get_strides());
                            auto &format_candidates
                                    = inputs[i]
                                              ->details_
                                              .get_format_candidates();
                            auto &target_format = target_fs_pair.first;
                            // tunable op always need reorder before op in
                            // dynamic.
                            if (is_graph_dynamic) {
                                if (node->isa<tunable_op_t>()) {
                                    // Reverse order here to match tunable op's
                                    // config.
                                    for (size_t k
                                            = in_supported_pairs[i].size();
                                            k > 0; k--) {
                                        // ensure that each internal tunable op
                                        // follows last format
                                        // clang-format off
                                        if (!in_supported_pairs[i][k - 1]
                                                        .first.is_blocking()
                                                || (format_candidates.find(
                                                            in_supported_pairs
                                                                    [i][k - 1]
                                                                        .first)
                                                        == format_candidates
                                                                   .end())) {
                                            insert_reorder_op(graph,
                                                    reorder_map, inputs[i], i,
                                                    in_supported_pairs[i]
                                                                      [k - 1],
                                                    node, is_input_plain,
                                                    insert_reorder_callback);
                                        }
                                        // clang-format on
                                    }
                                } else {
                                    // binary fusible, if input format
                                    // candidates contains the target format, do
                                    // not need reorder
                                    if (format_candidates.find(target_format)
                                            == format_candidates.end()) {
                                        insert_reorder_op(graph, reorder_map,
                                                inputs[i], i, target_fs_pair,
                                                node, is_input_plain,
                                                insert_reorder_callback);
                                    }
                                }
                            } else if (in_fs_pair != target_fs_pair) {
                                if (graph.attrs_.get_or_else(
                                            "insert_reorder", true)) {
                                    // static update
                                    insert_reorder_op(graph, reorder_map,
                                            inputs[i], i, target_fs_pair, node,
                                            is_input_plain,
                                            insert_reorder_callback);
                                } else {
                                    // For dynamic fused op internal format
                                    // update.
                                    inputs[i]->details_.set_format_and_stride(
                                            target_fs_pair.first,
                                            target_fs_pair.second);
                                }
                            } else if (!insert_reorder_callback) {
                                // if static and no need of reorder
                                remove_inserted_reorder_op(
                                        graph, reorder_map, inputs[i], node);
                            }
                            if (is_graph_dynamic) {
                                dispatch_format.push_back(target_fs_pair.first);
                            }
                        }
                        if (is_graph_dynamic) {
                            assert(!out_supported_pairs[0].empty());
                            for (size_t j = out_supported_pairs[0].size();
                                    j > 0; j--) {
                                update_output_formats(node->info_.outputs_,
                                        out_supported_pairs, j - 1);
                            }
                            if (!node->isa<tunable_op_t>()) {
                                // dispatch_key of tunable op is decided by
                                // itself.
                                dispatch_format.push_back(
                                        out_supported_pairs[0][0].first);
                                // update fusible_op's dispatch key in pass as
                                // it follows format of tunable op.
                                node->get_dispatch_key_set()
                                        ->get_inner_set()
                                        .insert(dispatch_format);
                            }
                        }
                    } else {
                        COMPILE_ASSERT(0,
                                "The op must support query_format: "
                                        << node->op_name_);
                    }
                }
                // reset old format
                if (is_graph_dynamic) {
                    for (size_t i = 0; i < inputs.size(); ++i) {
                        inputs[i]->details_.set_format(old_formats[i]);
                    }
                    reset_in_out_supported_pairs();
                    if (node->isa<tunable_op_t>()) {
                        node->stc_cast<tunable_op_t>()->set_config(nullptr);
                    }
                    node->query_format(
                            ctx, in_supported_pairs, out_supported_pairs);
                    // if (node->isa<tunable_op_t>()) {
                    // reset one unified layout for fusion.
                    for (size_t i = 0; i < node->get_inputs().size(); ++i) {
                        node->get_inputs()[i]->details_.set_format(
                                in_supported_pairs[i][0].first);
                    }
                    // }
                }
                update_output_formats(node->info_.outputs_, out_supported_pairs,
                        cur_layout_choice);
            } else if (node->isa<fusible_op_t>()) {
                // split/flatten/reshape/concat/matmul/reduce/reorder/trans2d/transpose
                // has itself query_format func
                // relu/exp/tanh/erf/squared_root/triangle has utility
                // query_format func
                if (is_graph_dynamic) {
                    auto &format_candidates
                            = inputs[0]->details_.get_format_candidates();
                    auto old_format = inputs[0]->details_.get_format();
                    for (auto &candidate : format_candidates) {
                        std::vector<sc_data_format_t> dispatch_format;
                        dispatch_format.reserve(inputs.size() + outputs.size());
                        inputs[0]->details_.set_format(candidate);
                        dispatch_format.push_back(candidate);
                        reset_in_out_supported_pairs();
                        node->query_format(
                                ctx, in_supported_pairs, out_supported_pairs);
                        assert(out_supported_pairs[0].size() == 1);
                        update_output_formats(node->info_.outputs_,
                                out_supported_pairs, cur_layout_choice);
                        dispatch_format.push_back(
                                out_supported_pairs[0][0].first);
                        node->get_dispatch_key_set()->get_inner_set().insert(
                                dispatch_format);
                    }
                    inputs[0]->details_.set_format(old_format);
                }
                reset_in_out_supported_pairs();
                node->query_format(
                        ctx, in_supported_pairs, out_supported_pairs);
                update_output_formats(node->info_.outputs_, out_supported_pairs,
                        cur_layout_choice);
            } else {
                COMPILE_ASSERT(0,
                        "Only support fusible op/tunable op/in op/out op "
                        "in "
                        "the layout_propagation pass");
            }
        }
    };

    // try all combinations of possible layouts and select the least cost
    // one
    if (total_choices <= STATIC_MAX_LAYOUT_TRIES && total_choices > 1) {
        size_t best_cost = std::numeric_limits<size_t>::max();
        std::vector<size_t> best_choice = cur_choice;
        for (size_t tr = 0; tr < total_choices; tr++) {
            reset_input_fmt();
            size_t cost = 0;
            insert_reorder_callback =
                    [&cost](const graph_tensor_ptr &in,
                            const sc_data_format_t &target_format) {
                        size_t cur_cost = in->details_.get_blocking_byte_size();
                        // give blocking format a discount as it may
                        // benefit performance on other ops
                        if (target_format.is_blocking()) {
                            cur_cost = cur_cost * 0.9;
                        }
                        cost += cur_cost;
                    };
            size_t cur_idx = tr;
            for (size_t i = 0; i < cur_choice.size(); i++) {
                cur_choice[i] = cur_idx % num_choices[i];
                cur_idx /= num_choices[i];
            }
            for (auto &op : sorted_ops) {
                do_visit(op);
            }
            SC_MODULE_INFO << "cost=" << cost << " "
                           << utils::print_vector(cur_choice);
            if (!is_graph_dynamic && cost < best_cost) {
                best_cost = cost;
                best_choice = cur_choice;
            }
        }
        cur_choice = std::move(best_choice);
    } else {
        // if there are too many choices, we can choose the default all-zero
        // choice
        if (total_choices > STATIC_MAX_LAYOUT_TRIES) {
            SC_MODULE_WARN << "Too many choices, using default";
        }
    }

    // clear the callback to let insert_reorder_op really work
    insert_reorder_callback = reorder_callback_type();
    reset_input_fmt();
    // visit again to insert the reorder ops
    for (auto &op : sorted_ops) {
        do_visit(op);
    }

    // it should be refactor to one standalone pass to finally fix constant
    // value
    auto vis2 = op_visitor_t::bfs();
    vis2.visit_graph(graph, [&](op_visitor_t *vis2, const sc_op_ptr &node) {
        if (node->isa<constant_op_t>() && node->attrs_.has_key("temp.var")) {
            auto const_op = node->dyn_cast<constant_op_t>();
            const_op->reset_const_values();
        }
    });
    if (is_graph_dynamic) { combine_layout_and_impl_dispatch(ctx, graph); }
    graph.reset_op_ids();
}
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
