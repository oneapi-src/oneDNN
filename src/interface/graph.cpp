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
        const std::vector<dnnl::graph::impl::logical_tensor_t> &tensors) {
    std::vector<size_t> ids;
    ids.reserve(tensors.size());
    for (const auto &t : tensors) {
        ids.push_back(t.id);
    }
    return ids;
}

bool logical_tensor_sanity_check(
        std::unordered_map<size_t, dnnl::graph::impl::logical_tensor_t>
                &id_to_lts,
        const std::vector<dnnl::graph::impl::logical_tensor_t> &lts) {
    for (const auto &t : lts) {
        auto id_search = id_to_lts.find(t.id);
        if (id_search == id_to_lts.end()) {
            id_to_lts[t.id] = t;
        } else {
            // compare two tensors with the same id;
            if (dnnl::graph::impl::logical_tensor_wrapper(id_search->second)
                    != dnnl::graph::impl::logical_tensor_wrapper(t)) {
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
    dfs_visit(this->get_outputs(), [&](node_t *n) {
        if (n->has_attr("backend")) {
            partitions[count]->init(n, engine_kind_);
            count++;
        }
    });
}

status_t dnnl_graph_graph::build_graph() {
    // map of {backend_node: vector of input tensor_ids}
    std::unordered_map<node_t *, std::vector<size_t>> node_to_tensor_id;
    // map of {tensor_id, (producer_node, producer_offset)}
    std::unordered_map<size_t, std::pair<node_t *, size_t>>
            tensor_id_to_producer;
    // map of {tensor_id: tensor}
    std::unordered_map<size_t, logical_tensor_t> id_to_tensor;

    for (const op_t &l_n : ops_) {
        if (!logical_tensor_sanity_check(id_to_tensor, l_n.inputs())
                || !logical_tensor_sanity_check(id_to_tensor, l_n.outputs())) {
            return status::invalid_graph;
        }
        // create backend node for each op
        dnnl::graph::impl::node_t *lbk_n = create_node(l_n);
        // save the input tensor ids of current op
        node_to_tensor_id[lbk_n] = get_ids(l_n.inputs());
        // save the outputs of current op
        const std::vector<logical_tensor_t> &l_n_outputs = l_n.outputs();
        for (size_t i = 0; i < l_n_outputs.size(); ++i) {
            const logical_tensor_t &out = l_n_outputs[i];
            tensor_id_to_producer[out.id] = std::make_pair(lbk_n, i);
        }
    }

    // set connections between backend nodes
    for (const node_ptr &anode : nodes_) {
        // find the input tensor ids of current node
        const std::vector<size_t> &input_tensor_ids
                = node_to_tensor_id[anode.get()];

        for (size_t i = 0; i < input_tensor_ids.size(); ++i) {
            // find the producer of the input tensor
            auto id_search = tensor_id_to_producer.find(input_tensor_ids[i]);
            // if the producer of the input tensor does not exist in the graph,
            // the input tensor is not produced by another node,
            // it's the input of the whole graph
            if (id_search != tensor_id_to_producer.end()) {
                std::pair<node_t *, size_t> producer = id_search->second;
                // set input of current node
                anode->set_input(
                        anode->num_inputs(), producer.first, producer.second);
            }
        }
    }
    return status::success;
}

void dnnl_graph_graph::visualize(std::string filename) {
    std::ofstream out;
    static size_t i = 0;
    out.open(std::to_string(i++).append(filename));
    out << "digraph G {\n";
    dfs_visit(this->get_outputs(), [&](node_t *node) {
        auto current_node_name = node->get_name();
        for (size_t i = 0; i < node->num_inputs(); ++i) {
            auto input_node = node->get_input_node(i);
            auto input_node_name = input_node->get_name();
            out << input_node_name << " -> " << current_node_name << ";\n";
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

status_t DNNL_GRAPH_API dnnl_graph_add_op(graph_t *graph, dnnl_graph_op_t *op) {
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
        graph_t *graph, uint64_t num, dnnl_graph_partition_t **partition) {
    if (graph == nullptr) { return status::invalid_graph; }
    std::vector<partition_t *> partitions {partition, partition + num};
    graph->get_partitions(partitions);
    return status::success;
}
