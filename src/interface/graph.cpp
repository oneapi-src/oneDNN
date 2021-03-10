/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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

#include <cstring>
#include <string>
#include <vector>
#include <unordered_map>

#include "backend/pass/pass_manager.hpp"
#include "graph.hpp"
#include "partition.hpp"

using namespace dnnl::graph::impl;

namespace {
std::vector<size_t> get_ids(
        const std::vector<std::shared_ptr<value_t>> &values) {
    std::vector<size_t> ids;
    ids.reserve(values.size());
    for (const auto &t : values) {
        ids.push_back(t->get_logical_tensor().id);
    }
    return ids;
}

bool logical_tensor_sanity_check(
        std::unordered_map<size_t, dnnl::graph::impl::logical_tensor_t>
                &id_to_lts,
        const std::vector<std::shared_ptr<value_t>> &values) {
    for (const auto &v : values) {
        auto lt = v->get_logical_tensor();
        auto id_search = id_to_lts.find(lt.id);
        if (id_search == id_to_lts.end()) {
            id_to_lts[lt.id] = lt;
        } else {
            // compare two tensors with the same id;
            if (dnnl::graph::impl::logical_tensor_wrapper(id_search->second)
                    != dnnl::graph::impl::logical_tensor_wrapper(lt)) {
                return false;
            }
        }
    }
    return true;
}
} // namespace

status_t dnnl_graph_graph::run_pass(partition_policy_t policy) {
    // test pass registration
    UNUSED(policy);

    // we cannot run passes on a un-built graph.
    if (!is_built_) return status::invalid_graph;

    auto pm = pass::pass_manager();
    auto &passes = pm.get_passes();

    char *val = std::getenv("DNNL_GRAPH_DUMP");
    if (val != nullptr && std::strcmp(val, "1") == 0) {
        std::cout << "number of registered passes: " << passes.size() << "\n";
        for (auto &pass : passes) {
            std::cout << "pass_name: " << pass->get_pass_name() << "\n";
        }
        std::cout << "visualize un-fused graph to a dot file" << std::endl;
        visualize("_backend.dot");
    }

    pm.run_passes(*this, "fake_config.json");

    if (val != nullptr && std::strcmp(val, "1") == 0) {
        std::cout << "visualize fused graph to a dot file" << std::endl;
        visualize("_backend_opt.dot");
    }

    return status::success;
}

void dnnl_graph_graph::get_partitions(std::vector<partition_t *> &partitions) {
    size_t count = 0;
    topo_order_visit(this->get_output_ops(), [&](op_t *n) {
        if (n->has_attr("backend")) {
            partitions[count]->init(n, engine_kind_);
            count++;
        }
    });
}

status_t dnnl_graph_graph::build_graph() {
    // if the graph is already built, return directly.
    // TODO(xxx): actually we may need to a verification here.
    if (is_built_) return status::success;

    // map of {backend op: vector of input tensor_ids}
    std::unordered_map<graph_t::op_t *, std::vector<size_t>> op_to_tensor_id;
    // map of {tensor_id, (producer_op, producer_offset)}
    std::unordered_map<size_t, std::pair<graph_t::op_t *, size_t>>
            tensor_id_to_producer;
    // map of {tensor_id: tensor}
    std::unordered_map<size_t, logical_tensor_t> id_to_tensor;

    for (const auto &op : ops_) {
        auto in_values = op->get_input_values();
        auto out_values = op->get_output_values();
        if (!logical_tensor_sanity_check(id_to_tensor, in_values)
                || !logical_tensor_sanity_check(id_to_tensor, out_values)) {
            return status::invalid_graph;
        }

        // save the input tensor ids of current op
        op_to_tensor_id[op.get()] = get_ids(in_values);
        // save the outputs of current op
        for (size_t i = 0; i < out_values.size(); ++i) {
            const logical_tensor_t out = out_values[i]->get_logical_tensor();
            tensor_id_to_producer[out.id] = std::make_pair(op.get(), i);
        }
    }

    for (const op_ptr &op : ops_) {
        // find the input tensor ids of current op
        const std::vector<size_t> &input_tensor_ids = op_to_tensor_id[op.get()];

        for (size_t i = 0; i < input_tensor_ids.size(); ++i) {
            // find the producer of the input tensor
            auto id_search = tensor_id_to_producer.find(input_tensor_ids[i]);
            // if the producer of the input tensor does not exist in the graph,
            // the input tensor is not produced by another op,
            // it's the input of the whole graph
            if (id_search != tensor_id_to_producer.end()) {
                std::pair<op_t *, size_t> producer = id_search->second;
                // set input of current op
                op->connect_input(i, *(producer.first), producer.second);
            }
        }
    }

    is_built_ = true;
    return status::success;
}

void dnnl_graph_graph::visualize(const std::string &filename) {
    std::ofstream out;
    static size_t i = 0;
    out.open(std::to_string(i++).append(filename));
    out << "digraph G {\n";
    topo_order_visit(this->get_output_ops(), [&](op_t *op) {
        auto current_op_name = op->get_name();
        for (size_t i = 0; i < op->num_inputs(); ++i) {
            op_t *input_op = op->get_input_op(i);
            if (input_op) {
                const std::string &input_op_name = input_op->get_name();
                out << input_op_name << " -> " << current_op_name << ";\n";
            }
        }
    });
    out << "}\n";
    out.close();
}

status_t DNNL_GRAPH_API dnnl_graph_graph_create(
        graph_t **created_graph, engine_kind_t engine_kind) {
    *created_graph = new graph_t(engine_kind);
    return status::success;
}

status_t DNNL_GRAPH_API dnnl_graph_graph_destroy(graph_t *graph) {
    delete graph;
    return status::success;
}

status_t DNNL_GRAPH_API dnnl_graph_add_op(graph_t *graph, op_t *op) {
    if (graph == nullptr || op == nullptr) { return status::invalid_argument; }
    return graph->add_op(op);
}

status_t DNNL_GRAPH_API dnnl_graph_graph_filter(
        graph_t *graph, partition_policy_t policy) {
    if (graph == nullptr) { return status::invalid_graph; }
    auto status = graph->build_graph();
    if (status != status::success) return status::invalid_graph;
    status = graph->run_pass(policy);
    if (status != status::success) {
        return status::invalid_graph;
    } else {
        return status::success;
    }
}

status_t DNNL_GRAPH_API dnnl_graph_graph_get_partition_num(
        const graph_t *graph, uint64_t *num) {
    if (graph == nullptr) { return status::invalid_graph; }
    *num = graph->get_num_partitions();
    return status::success;
}

status_t DNNL_GRAPH_API dnnl_graph_graph_get_partitions(
        graph_t *graph, uint64_t num, partition_t **partition) {
    if (graph == nullptr) { return status::invalid_graph; }
    std::vector<partition_t *> partitions {partition, partition + num};
    graph->get_partitions(partitions);
    return status::success;
}
