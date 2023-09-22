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

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>
#include "../fusible_op.hpp"
#include "../pass/pass.hpp"
#include "../visitor.hpp"
#include <compiler/ir/graph/dynamic_utils.hpp>
#include <compiler/ir/graph/quantization/quantize_op.hpp>
#include <ops/convolution.hpp>
#include <ops/fusible/binary_elemwise.hpp>
#include <ops/fusible/memory_movement.hpp>
#include <ops/fusible/unary_elemwise.hpp>
#include <runtime/config.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

sc_dims get_conv1d_flatten_shape(const sc_data_format_t &format,
        const sc_dims &origin_shape, int merge_bs = 1) {
    COMPILE_ASSERT(
            origin_shape.size() == 4, "Conv1d flatten only support 2d case");
    sc_dims out_shape;
    out_shape.push_back(origin_shape[0]);
    if (format == sc_data_format_t::NCHW()) {
        out_shape.push_back(origin_shape[1]);
        out_shape.push_back(origin_shape[2] * origin_shape[3]);
    } else if (format == sc_data_format_t::NHWC()) {
        out_shape.push_back(origin_shape[3]);
        out_shape.push_back(origin_shape[1] * origin_shape[2]);
    }
    COMPILE_ASSERT(out_shape[0] % merge_bs == 0,
            "N % merge_bs should be equal to zero but get "
                    << out_shape[0] << " % " << merge_bs << " = "
                    << out_shape[0] % merge_bs);
    out_shape[2] *= merge_bs;
    out_shape[0] /= merge_bs;

    return out_shape;
}

// Whether to flatten N axis, which will break bs fusion but may bring perf
// benefit to single layer
int get_minibatch(const int &bs, const int &min_os) {
    auto num_threads = runtime_config_t::get().get_num_threads();
    int minibatch = std::max(sc_dim(1), sc_dim(28) / sc_dim(std::sqrt(min_os)));
    if ((bs / minibatch % num_threads != 0
                && bs / minibatch < 4 * num_threads)) {
        // TODO(zhicong): use a more general way for minibatch image affinity
        if (bs % num_threads == 0) {
            return 1;
        } else {
            return bs;
        }
    }
    return bs % minibatch == 0 ? minibatch : 1;
}

int minimum_spatial_shape(sc_graph_t &graph, bool &is_support) {
    auto vis = op_visitor_t::bfs();
    int min_spatial = std::numeric_limits<int>::max();
    vis.visit_graph(graph, [&](op_visitor_t *vis, const sc_op_ptr &node) {
        if (auto op = node->dyn_cast<ops::conv_fwd_core_op_t>()) {
            int spatial_size = 1;
            auto plain_shape = op->get_inputs()[0]->details_.get_plain_dims();
            if (plain_shape.size() != 4) { is_support = false; }

            for (auto i = 2UL; i < plain_shape.size(); i++) {
                spatial_size *= plain_shape[i];
            }
            if (spatial_size < min_spatial) { min_spatial = spatial_size; }
        }
    });
    return min_spatial;
}

void conv1d_flatten(sc_graph_t &graph, const context_ptr &ctx) {
    auto vis = op_visitor_t::bfs();
    bool is_support = true;
    auto minimum_os = minimum_spatial_shape(graph, is_support);
    if (!is_support) return;
    vis.visit_graph(graph, [&](op_visitor_t *vis, const sc_op_ptr &node) {
        if (auto op = node->dyn_cast<ops::conv_fwd_core_op_t>()) {
            auto data_plain_shape
                    = op->get_inputs()[0]->details_.get_plain_dims();
            // do not support dynamic shape
            if (op->get_inputs()[0]->is_dynamic()) return;
            auto data_origin_shape
                    = op->get_inputs()[0]->details_.get_blocking_dims();
            auto weight_origin_shape
                    = op->get_inputs()[1]->details_.get_blocking_dims();
            auto ndims = data_plain_shape.size();
            auto &stride = op->attrs_.get<sc_dims>("strides");
            auto &pads_begin = op->attrs_.has_key("pads_begin")
                    ? op->attrs_.get<sc_dims>("pads_begin")
                    : op->attrs_.get<sc_dims>("paddings");
            sc_dim groups = op->attrs_.get_or_else("groups", 1);
            if (groups > 1) { return; }
            auto weight_plain_dims
                    = op->get_inputs()[1]->details_.get_plain_dims();
            auto kh = weight_plain_dims[ndims - 2];
            auto kw = weight_plain_dims[ndims - 1];
            auto data_format = op->get_inputs()[0]->details_.get_format();
            auto weight_format = op->get_inputs()[1]->details_.get_format();
            int merge_bs = get_minibatch(data_plain_shape[0], minimum_os);
            auto iw = op->get_inputs()[0]->details_.get_plain_dims()[3];
            auto ih = op->get_inputs()[0]->details_.get_plain_dims()[2];
            auto ow = op->get_outputs()[0]->details_.get_plain_dims()[3];
            auto oh = op->get_outputs()[0]->details_.get_plain_dims()[2];
            if (op->use_conv1d()) {
                { // pre tensor_view(data)
                    auto shape
                            = get_conv1d_flatten_shape(sc_data_format_t::NCHW(),
                                    sc_data_format_t::get_padded_plain_shapes(
                                            data_origin_shape, data_format),
                                    merge_bs);
                    auto reorder_op
                            = graph.make("reorder", {op->get_inputs()[0]}, {},
                                    {{"out_format", sc_data_format_t::NHWC()},
                                            {"internal", true}});
                    auto view = graph.make("tensor_view",
                            reorder_op->get_outputs(), {},
                            {{"shape", shape},
                                    {"format", sc_data_format_t::NSC()},
                                    {"expand_dim", std::vector<int> {}}});
                    op->replace_input(0, view->get_outputs()[0]);
                    vis->update_state_for_visited(reorder_op);
                    vis->update_state_for_visited(view);
                }
                { // pre tensor_view(weight)
                    auto shape = get_conv1d_flatten_shape(
                            weight_format, weight_origin_shape);
                    auto reorder_op
                            = graph.make("reorder", {op->get_inputs()[1]}, {},
                                    {{"out_format", sc_data_format_t::KCRS()},
                                            {"internal", true}});
                    auto view = graph.make("tensor_view",
                            reorder_op->get_outputs(), {},
                            {{"shape", shape},
                                    {"format", sc_data_format_t::KCS()},
                                    {"expand_dim", std::vector<int> {}}});
                    op->replace_input(1, view->get_outputs()[0]);
                    vis->update_state_for_visited(reorder_op);
                    vis->update_state_for_visited(view);
                }
                { // post tensor view(output)
                    auto origin_out = op->get_outputs()[0]->copy();
                    op->get_outputs()[0]->replace_with(origin_out);
                    origin_out->producer_owner_ = nullptr;

                    auto dtype = origin_out->details_.dtype_;
                    auto new_conv_out = std::make_shared<graph_tensor>(&(*node),
                            sc_data_format_t::NCS(),
                            get_conv1d_flatten_shape(sc_data_format_t::NCHW(),
                                    op->get_outputs()[0]
                                            ->details_.get_blocking_dims(),
                                    merge_bs),
                            dtype);
                    op->info_.outputs_[0] = new_conv_out;

                    auto reorder_op = graph.make("reorder", op->get_outputs(),
                            {}, {{"out_format", sc_data_format_t::NSC()}});
                    auto view = graph.make("tensor_view",
                            reorder_op->get_outputs(), {},
                            {{"shape", origin_out->details_.get_plain_dims()},
                                    {"format", sc_data_format_t::NHWC()},
                                    {"expand_dim", std::vector<int> {}},
                                    {"push_back", true}});
                    origin_out->replace_with(view->get_outputs()[0]);
                    view->copy_dispatch_key_set_from_op(node);
                    vis->update_state_for_visited(reorder_op);
                    vis->update_state_for_visited(view);
                }
                if (op->attrs_.has_key("pads_begin")) {
                    op->attrs_.get<sc_dims>("pads_begin") = {0};
                    op->attrs_.get<sc_dims>("pads_end") = {0};
                } else {
                    op->attrs_.get<sc_dims>("paddings") = {0};
                }
                op->attrs_["origin_ih"] = ih;
                op->attrs_["origin_iw"] = iw;
                op->attrs_["origin_ow"] = ow;
                op->attrs_["origin_oh"] = oh;
            }
        }
    });
    graph.reset_op_ids();
}

void flatten_conv(sc_graph_t &graph, const context_ptr &ctx) {
    if (graph.attrs_.get_or_else("no_conv1d", false)) { return; }
    conv1d_flatten(graph, ctx);
}
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
