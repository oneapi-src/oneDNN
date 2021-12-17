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

#include <utility>

#include "../fusible_op.hpp"
#include "../graph_op.hpp"
#include "../tunable_op.hpp"
#include "../visitor.hpp"
namespace sc {

static void insert_reorder_op(sc_graph_t &graph, const graph_tensor_ptr &in,
        size_t in_index, const sc_data_format_t &out_format,
        const sc_op_ptr &cur_op, bool is_input_plain) {
    // if we don't need to keep plain format for input op and input op tensor is
    // currently not blocking, copy the blocking format
    if (!is_input_plain && in->uses_.size() == 1
            && in->producer_owner_->isa<input_op>()
            && in->details_.get_format().is_plain()
            && !in->producer_owner_->attrs_.get_or_else("keep_plain", false)) {
        in->details_.set_format(out_format);
        return;
    }
    auto ret = graph.make("reorder", {in}, {},
            {{"out_format", out_format},
                    {op_attr_key::no_fuse, // walk around for conv graph. will
                            // be dropped after yijie's refactor
                            graph.attrs_.get_or_else(
                                    "reorder_not_to_fuse", false)}});
    cur_op->replace_input(in_index, ret->get_outputs()[0]);
}

static void update_output_formats(std::vector<graph_tensor_ptr> &outs,
        const std::vector<std::vector<sc_data_format_t>> &out_supported_format
        = {}) {
    for (size_t i = 0; i < outs.size(); ++i) {
        if (outs[i]->details_.get_format().is_any()) {
            outs[i]->details_.set_format(sc_data_format_t::get_plain_by_dims(
                    (int)outs[i]->details_.get_plain_dims().size()));
        }
        if (!out_supported_format.empty()) {
            outs[i]->details_.set_format(out_supported_format[i][0]);
        }
    }
}

static void check_input_format(const std::vector<sc::graph_tensor_ptr> &ins) {
    for (auto &in : ins) {
        COMPILE_ASSERT(!in->details_.get_format().is_any(),
                "input format don't allow any format");
    }
}

static void fusible_layout_propagation(sc_graph_t &graph, context_ptr ctx,
        const sc_op_ptr &cur_node, bool is_input_plain) {
    std::vector<std::vector<sc_data_format_t>> in_supported_formats,
            out_supported_formats;
    cur_node->query_format(
            std::move(ctx), in_supported_formats, out_supported_formats);
    auto &inputs = cur_node->info_.inputs_;
    auto &outputs = cur_node->info_.outputs_;
    check_input_format(inputs);
    if (dynamic_cast<binary_elementwise_op_t *>(cur_node.get())
            || dynamic_cast<transpose_op_t *>(cur_node.get())) {
        // need to unify input formats
        // todo: should check add_op input shape, output shape size =
        // max(input size), so need to enhance
        if (!in_supported_formats.empty() && !out_supported_formats.empty()) {
            for (size_t i = 0; i < inputs.size(); ++i) {
                if (inputs[i]->details_.get_format()
                        != in_supported_formats[i][0]) {
                    insert_reorder_op(graph, inputs[i], i,
                            in_supported_formats[i][0], cur_node,
                            is_input_plain);
                }
            }
            update_output_formats(
                    cur_node->info_.outputs_, out_supported_formats);
        } else {
            COMPILE_ASSERT(0,
                    "Binary op and broadcast op must have supported "
                    "input/output format");
        }
        update_output_formats(cur_node->info_.outputs_, out_supported_formats);
    } else if (dynamic_cast<tensor_view_op_t *>(cur_node.get())) {
        auto &input_format = in_supported_formats[0][0];
        if (input_format.format_code_ != format_kinds::any
                && (inputs[0]->details_.get_format().get_format_category()
                                != input_format.get_format_category()
                        || ((inputs[0]->details_.get_format().format_code_
                                    == input_format.format_code_)
                                && (inputs[0]->details_.get_format().blocks_
                                        != input_format.blocks_)))) {
            insert_reorder_op(graph, inputs[0], 0, input_format, cur_node,
                    is_input_plain);
        }
        update_output_formats(cur_node->info_.outputs_, out_supported_formats);

    } else {
        // split/flatten/reshape/concat/matmul/reduce/reorder/trans2d/transpose
        // has itself query_format func
        // relu/exp/tanh/erf/squared_root/triangle has utility query_format
        // func
        update_output_formats(cur_node->info_.outputs_, out_supported_formats);
    }
}

static void tunable_layout_propagation(sc_graph_t &graph, context_ptr ctx,
        const sc_op_ptr &cur_node, bool is_input_plain) {
    std::vector<std::vector<sc_data_format_t>> in_supported_formats,
            out_supported_formats;
    cur_node->query_format(
            std::move(ctx), in_supported_formats, out_supported_formats);
    auto &inputs = cur_node->info_.inputs_;
    auto &outputs = cur_node->info_.outputs_;
    check_input_format(inputs);

    // situation one: conv/gemm/pooling/batch_mm/conv_block/interaction ops have
    // certain and complete support input and output formats
    if (!in_supported_formats.empty() && !out_supported_formats.empty()) {
        for (size_t i = 0; i < inputs.size(); ++i) {
            if (inputs[i]->details_.get_format()
                    != in_supported_formats[i][0]) {
                insert_reorder_op(graph, inputs[i], i,
                        in_supported_formats[i][0], cur_node, is_input_plain);
            }
        }
        update_output_formats(cur_node->info_.outputs_, out_supported_formats);
    } else {
        // situation two: blocking_concat/layernorm/gelu/softmax ops can support
        // any_t input and output format.

        // situation two : bertBMM/mlp complex ops have complex query
        // format logic.
        // todo: implement
        COMPILE_ASSERT(
                0, "not implemented for now, will support in the future");
    }
}

SC_INTERNAL_API void layout_propagation(
        sc_graph_t &graph, const context_ptr &ctx) {
    bool is_input_plain = graph.attrs_.get_or_else(
            sc_graph_t::attr_key_t::is_input_plain, true);
    op_visitor_t vis {op_visitor_t::pop_back_selector,
            op_visitor_t::create_DAG_updater(graph.ops_.size())};
    vis.visit_graph(graph, [&](const sc_op_ptr &node) {
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
                                is_input_plain);
                    }
                }
            }
        } else if (node->isa<input_op>() || node->isa<constant_op_t>()) {
            update_output_formats(node->info_.outputs_);
        } else {
            if (node->isa<fusible_op_t>()) {
                fusible_layout_propagation(graph, ctx, node, is_input_plain);
            } else if (node->isa<tunable_op_t>()) {
                tunable_layout_propagation(graph, ctx, node, is_input_plain);
            } else {
                COMPILE_ASSERT(0,
                        "Only support fusible op/tunable op/in op/out op in "
                        "the layout_propagation pass");
            }
        }
    });
}
} // namespace sc
