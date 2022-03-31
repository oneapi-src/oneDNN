/*******************************************************************************
 * Copyright 2020-2022 Intel Corporation
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

namespace sc {

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
    return lower_graph(ctx, *graph, {graph->ops_.back(), graph->ops_[0]});
}

std::shared_ptr<sc_graph_t> graph_op_t::get_graph() {
    return get_graph_impl();
}

std::shared_ptr<sc_graph_t> configurable_graph_op_t::get_graph() {
    auto g = get_graph_impl();
    // set config space;
    return g;
}

std::shared_ptr<void> configurable_graph_op_t::get_config() {
    auto cfg = std::make_shared<graph_config>();
    *cfg = config_data_;
    return cfg;
}

void configurable_graph_op_t::set_config(const std::shared_ptr<void> &config) {
    config_data_ = *reinterpret_cast<graph_config *>(config.get());
}

} // namespace sc
