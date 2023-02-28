/*******************************************************************************
 * Copyright 2020-2023 Intel Corporation
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

#include "graph_op.hpp"
#include "lowering.hpp"
#include "util/utils.hpp"
#include "visitor.hpp"
#include <compiler/ir/graph/pass/pass.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

std::vector<graph_tensor_ptr> graph_op_t::remake_logical_tensors(
        const std::vector<graph_tensor_ptr> &flts) {
    std::vector<graph_tensor_ptr> new_flts(flts.size());
    for (size_t i = 0; i < flts.size(); ++i) {
        new_flts[i] = std::make_shared<graph_tensor>(nullptr,
                flts[i]->details_.get_format(),
                flts[i]->details_.get_plain_dims(), flts[i]->details_.dtype_);
    }
    return new_flts;
}

ir_module_ptr graph_op_t::get_func(context_ptr ctx) {
    auto graph = get_graph();
    return lower_graph(
            ctx, *graph, {graph->ops_.back(), graph->ops_[0]}, false);
}

std::shared_ptr<sc_graph_t> graph_op_t::get_graph() {
    auto g = std::make_shared<sc_graph_t>();
    g->sync_dynamic_info_with_graph(get_owner_graph());
    get_graph_impl(g);
    return g;
}

std::shared_ptr<sc_graph_t> configurable_graph_op_t::get_graph() {
    auto g = std::make_shared<sc_graph_t>();
    g->sync_dynamic_info_with_graph(get_owner_graph());
    get_graph_impl(g);
    // set config space;
    return g;
}

config_ptr configurable_graph_op_t::get_config() {
    return reflection::general_object_t::make(config_data_);
}

void configurable_graph_op_t::set_config(const config_ptr &config) {
    config_data_ = *config.get_as<graph_config>();
}

config_ptr configurable_graph_op_t::get_default_config(context_ptr ctx) {
    auto op_graph = this->get_graph();
    return reflection::general_object_t::make(
            graph::get_graph_default_config(ctx, *op_graph));
}

nested_graph_op_t::nested_graph_op_t(const std::string &op_name,
        const std::vector<graph_tensor_ptr> &ins,
        const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs,
        sc_graph_t &&graph)
    : configurable_graph_op_t(op_name, ins, outs, attrs)
    , graph_(std::move(graph)) {
    if (outs.empty()) {
        for (auto &out_op : graph_.get_output_ops()) {
            info_.outputs_.insert(info_.outputs_.end(),
                    out_op->get_inputs().begin(), out_op->get_inputs().end());
        }
        info_.outputs_ = remake_logical_tensors(info_.outputs_);
        for (auto &op : info_.outputs_) {
            op->producer_owner_ = this;
        }
    }

    auto required_input_tsenor_num = 0ul;
    for (auto &in_op : graph_.get_input_ops()) {
        required_input_tsenor_num += in_op->get_outputs().size();
    }
    COMPILE_ASSERT(required_input_tsenor_num == ins.size(),
            "The number of input tensor "
                    << ins.size() << " is incorrect. The required number is "
                    << required_input_tsenor_num);

    // create new input logical tensors
    std::vector<graph_tensor_ptr> inputs, outputs;
    inputs = remake_logical_tensors(info_.inputs_);

    // combine all input op into one
    int counter = 0;
    for (auto &op : graph_.get_input_ops()) {
        for (size_t idx = 0; idx < op->get_outputs().size(); ++idx) {
            op->get_outputs()[idx]->replace_with(inputs.at(counter++));
        }
        op->remove();
    }
    auto in_op = graph_.make_input(inputs);

    // combine all output op into one
    counter = 0;
    for (auto &op : graph_.get_output_ops()) {
        outputs.insert(outputs.end(), op->get_inputs().begin(),
                op->get_inputs().end());
        op->remove();
    }
    auto out_op = graph_.make_output(outputs);

    // delete those removed op
    graph_.reset_op_ids();
}

void nested_graph_op_t::get_graph_impl(std::shared_ptr<sc_graph_t> &graph) {
    graph = std::make_shared<sc_graph_t>(copy_graph(graph_));
}

// linter has a false alarm to treat copy here as a STL function
sc_op_ptr nested_graph_op_t::copy( // NOLINT
        const std::vector<graph_tensor_ptr> &ins,
        const std::vector<graph_tensor_ptr> &outs, sc_graph_t &mgr) {
    auto ret = mgr.make<nested_graph_op_t>(
            this->op_name_, ins, outs, attrs_, copy_graph(graph_));
    return ret;
}
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
