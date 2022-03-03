/*******************************************************************************
 * Copyright 2021-2022 Intel Corporation
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
#include "../fusible_op.hpp"
#include "../graph_op.hpp"
#include "../tunable_op.hpp"
#include "../visitor.hpp"
#include <ops/fusible/memory_movement.hpp>
namespace sc {

SC_MODULE(graph.layout_propagation);

using reorder_callback_type = std::function<void(
        const graph_tensor_ptr &in, const sc_data_format_t &target_formats)>;

static void insert_reorder_op(sc_graph_t &graph, const graph_tensor_ptr &in,
        size_t in_index, const sc_data_format_t &out_format,
        const sc_op_ptr &cur_op, bool is_input_plain,
        reorder_callback_type &on_insert_reorder) {
    // if we don't need to keep plain format for input op and input op tensor is
    // currently not blocking, copy the blocking format
    if (!is_input_plain && in->uses_.size() == 1
            && in->producer_owner_->isa<input_op>()
            && in->details_.get_format().is_plain()
            && !in->producer_owner_->attrs_.get_or_else("keep_plain", false)) {
        in->details_.set_format(out_format);
        return;
    }

    // if we are in trying-mode, don't insert an reorder. Instead, notify that
    // there will be a reorder to be inerted
    if (on_insert_reorder) {
        in->details_.set_format(out_format);
        on_insert_reorder(in, out_format);
    } else {
        auto ret = graph.make("reorder", {in}, {},
                {{"out_format", out_format},
                        {op_attr_key::no_fuse, // work around for conv graph.
                                // will be dropped after yijie's
                                // refactor
                                graph.attrs_.get_or_else(
                                        "reorder_not_to_fuse", false)}});
        cur_op->replace_input(in_index, ret->get_outputs()[0]);
    }
}

static void update_output_formats(std::vector<graph_tensor_ptr> &outs,
        const std::vector<std::vector<sc_data_format_t>> &out_supported_format,
        size_t layout_choice) {
    for (size_t i = 0; i < outs.size(); ++i) {
        if (!out_supported_format.empty()) {
            outs[i]->details_.set_format(
                    out_supported_format[i][layout_choice]);
        } else if (outs[i]->details_.get_format().is_any()) {
            outs[i]->details_.set_format(sc_data_format_t::get_plain_by_dims(
                    (int)outs[i]->details_.get_plain_dims().size()));
        }
    }
}

static void check_input_format(const std::vector<sc::graph_tensor_ptr> &ins) {
    for (auto &in : ins) {
        COMPILE_ASSERT(!in->details_.get_format().is_any(),
                "input format don't allow any format");
    }
}

constexpr int MAX_LAYOUT_TRIES = 64;

SC_INTERNAL_API void layout_propagation(
        sc_graph_t &graph, const context_ptr &ctx) {
    bool is_input_plain = graph.attrs_.get_or_else(
            sc_graph_t::attr_key_t::is_input_plain, true);
    std::vector<sc_op_ptr> sorted_ops;
    sorted_ops.reserve(graph.ops_.size());
    std::vector<size_t> num_choices;
    num_choices.reserve(graph.ops_.size());
    size_t total_choices = 1;
    std::vector<size_t> cur_choice;
    cur_choice.resize(graph.ops_.size());
    // the try-run will reset the input op's format. need to remember them
    std::vector<std::vector<sc_data_format_t>> format_backup;
    std::vector<sc_op *> input_ops;

    op_visitor_t vis {op_visitor_t::pop_back_selector,
            op_visitor_t::create_DAG_updater(graph.ops_.size())};
    // stage 1, collect all ops and the number of choices of formats
    vis.visit_graph(graph, [&](const sc_op_ptr &node) {
        sorted_ops.emplace_back(node);
        if (auto input_node = node->dyn_cast<input_op>()) {
            // backup the input's format
            std::vector<sc_data_format_t> fmts;
            for (auto &in : input_node->get_outputs()) {
                fmts.emplace_back(in->details_.get_format());
            }
            format_backup.emplace_back(std::move(fmts));
            input_ops.push_back(input_node);
        }
        if (auto tunable_node = node->dyn_cast<tunable_op_t>()) {
            std::vector<std::vector<sc_data_format_t>> in_supported_formats,
                    out_supported_formats;
            bool has_config = bool(tunable_node->get_config());
            node->query_format(
                    ctx, in_supported_formats, out_supported_formats);
            if (!has_config) { tunable_node->set_config(nullptr); }
            if (in_supported_formats.empty()) {
                num_choices[node->logical_op_id_] = 1;
            } else {
                num_choices[node->logical_op_id_]
                        = in_supported_formats[0].size();
                if (total_choices <= MAX_LAYOUT_TRIES)
                    total_choices *= num_choices[node->logical_op_id_];
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
                outs[i]->details_.set_format(fmts[i]);
            }
        }
    };
    reorder_callback_type insert_reorder_callback;
    auto do_visit = [&](const sc_op_ptr &node) {
        if (node->isa<output_op>()) {
            if (graph.attrs_.get_or_else(
                        sc_graph_t::attr_key_t::is_output_plain, true)) {
                // if is not plain format, will insert reorder.
                std::vector<sc_data_format_t> plain_formats(
                        node->get_inputs().size());
                for (size_t i = 0; i < node->get_inputs().size(); ++i) {
                    plain_formats[i] = node->get_inputs()[i]
                                               ->details_.get_format()
                                               .to_plain();
                }
                const auto &target_formats = node->attrs_.get_or_else(
                        "target_formats", plain_formats);
                COMPILE_ASSERT(
                        target_formats.size() == node->get_inputs().size(),
                        "Output op's target_formats' size should be equal to "
                        "number of tensors");
                for (size_t i = 0; i < node->get_inputs().size(); ++i) {
                    auto in = node->get_inputs()[i];
                    auto target_format = target_formats[i];
                    COMPILE_ASSERT(!in->details_.get_format().is_any(),
                            "output op's input format should have a concrete "
                            "format, instead of any format");
                    COMPILE_ASSERT(!target_format.is_any()
                                    && !target_format.is_blocking(),
                            "output op's target format should be plain or "
                            "permuted.")
                    if (in->details_.get_format() != target_format) {
                        insert_reorder_op(graph, in, i, target_format, node,
                                is_input_plain, insert_reorder_callback);
                    }
                }
            }
        } else if (node->isa<input_op>() || node->isa<constant_op_t>()) {
            update_output_formats(node->info_.outputs_, {}, 0);
        } else {
            std::vector<std::vector<sc_data_format_t>> in_supported_formats,
                    out_supported_formats;
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
            node->query_format(
                    ctx, in_supported_formats, out_supported_formats);
            if (reset_config) {
                node->stc_cast<tunable_op_t>()->set_config(nullptr);
            }
            auto &inputs = node->info_.inputs_;
            auto &outputs = node->info_.outputs_;
            check_input_format(inputs);
            size_t cur_layout_choice = cur_choice[node->logical_op_id_];
            if (node->isa<binary_elementwise_op_t>()
                    || node->isa<tunable_op_t>()) {
                // need to unify input formats
                // todo: should check add_op input shape, output shape size =
                // max(input size), so need to enhance
                if (!in_supported_formats.empty()
                        && !out_supported_formats.empty()) {
                    for (size_t i = 0; i < inputs.size(); ++i) {
                        auto &in_fmt
                                = in_supported_formats[i][cur_layout_choice];
                        if (inputs[i]->details_.get_format() != in_fmt) {
                            insert_reorder_op(graph, inputs[i], i, in_fmt, node,
                                    is_input_plain, insert_reorder_callback);
                        }
                    }
                    update_output_formats(node->info_.outputs_,
                            out_supported_formats, cur_layout_choice);
                } else {
                    COMPILE_ASSERT(0,
                            "The op must support query_format: "
                                    << node->op_name_);
                }
            } else if (node->isa<tensor_view_op_t>()) {
                auto &input_format = in_supported_formats[0][0];
                auto &in0_fmt = inputs[0]->details_.get_format();
                if (input_format.format_code_ != format_kinds::any
                        && (in0_fmt.get_format_category()
                                        != input_format.get_format_category()
                                || ((in0_fmt.format_code_
                                            == input_format.format_code_)
                                        && (in0_fmt.blocks_
                                                != input_format.blocks_)))) {
                    insert_reorder_op(graph, inputs[0], 0, input_format, node,
                            is_input_plain, insert_reorder_callback);
                }
                update_output_formats(node->info_.outputs_,
                        out_supported_formats, cur_layout_choice);

            } else if (node->isa<fusible_op_t>()) {
                // split/flatten/reshape/concat/matmul/reduce/reorder/trans2d/transpose
                // has itself query_format func
                // relu/exp/tanh/erf/squared_root/triangle has utility
                // query_format func
                update_output_formats(node->info_.outputs_,
                        out_supported_formats, cur_layout_choice);
            } else {
                COMPILE_ASSERT(0,
                        "Only support fusible op/tunable op/in op/out op in "
                        "the layout_propagation pass");
            }
        }
    };

    // try all combinations of possible layouts and select the least cost one
    if (total_choices <= MAX_LAYOUT_TRIES && total_choices > 1) {
        size_t best_cost = std::numeric_limits<size_t>::max();
        std::vector<size_t> best_choice;
        for (size_t tr = 0; tr < total_choices; tr++) {
            reset_input_fmt();
            size_t cost = 0;
            insert_reorder_callback
                    = [&cost](const graph_tensor_ptr &in,
                              const sc_data_format_t &target_format) {
                          size_t cur_cost = in->details_.size();
                          // give blocking format a discount as it may benefit
                          // performance on other ops
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
            if (cost < best_cost) {
                best_cost = cost;
                best_choice = cur_choice;
            }
        }
        cur_choice = std::move(best_choice);
    } else {
        // if there are too many choices, we can choose the default all-zero
        // choice
        if (total_choices > MAX_LAYOUT_TRIES) {
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
    vis2.visit_graph(graph, [&](const sc_op_ptr &node) {
        if (node->isa<constant_op_t>() && node->attrs_.has_key("temp.var")) {
            auto const_op = node->dyn_cast<constant_op_t>();
            const_op->reset_const_values();
        }
    });
}
} // namespace sc
