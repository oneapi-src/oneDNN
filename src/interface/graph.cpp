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
#include <unordered_set>

#include "backend.hpp"
#include "c_types_map.hpp"
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

// function to do graph rewriting
static void rewrite(dnnl::graph::impl::graph_t &agraph,
        std::vector<std::vector<op_t *>> &fusion_ops) {
    std::unordered_set<op_t *> visited;
    std::unordered_set<op_t *> fusion_ops_set;

    for (auto &ops : fusion_ops) {
        visited.clear();
        fusion_ops_set.clear();

        for (size_t i = 0; i < ops.size(); ++i) {
            fusion_ops_set.insert(ops[i]);
        }

        op_t *fused_op = agraph.create_op(op_kind::Wildcard);
        fused_op->set_partition(ops[0]->get_partition());

        for (size_t i = 0; i < ops.size(); ++i) {
            op_t *cur_op = ops[i];
            visited.insert(cur_op);
            fused_op->merge_attributes(cur_op->get_attributes());
            fused_op->add_op_ids(cur_op->get_op_ids());

            // if cur_node has input node which isn't in pattern,
            // update value's connection. if cur_node has input node
            // which is in pattern, add its output_tensor into visited
            for (size_t j = 0; j < cur_op->num_inputs(); ++j) {
                auto in_value = cur_op->get_input_value(j);
                //if in_op isn't in pattern,
                //set it as a input op of fused_op
                if (!in_value->has_producer()
                        || !visited.count(&in_value->get_producer())) {
                    in_value->remove_consumer(*cur_op, j);
                    in_value->add_consumer(*fused_op, fused_op->num_inputs());
                    fused_op->add_input(in_value);
                }
            }

            for (size_t k = 0; k < cur_op->num_outputs(); ++k) {
                auto out_value = cur_op->get_output_value(k);
                auto out_cons = out_value->get_consumers();

                bool cons_all_in_pattern = true;
                for (auto &con : out_cons) {
                    if (!fusion_ops_set.count(&(con.get_op()))) {
                        cons_all_in_pattern = false;
                        break;
                    }
                }

                if (out_cons.empty() || !cons_all_in_pattern) {
                    // it's a end node of pattern, need to update
                    // node connection of it's output nodes
                    out_value->set_producer(*fused_op);
                    fused_op->add_output(out_value);
                }
            }
        }

        for (size_t i = 0; i < ops.size(); ++i) {
            agraph.delete_op(ops[i]);
        }
    }
}

void dnnl_graph_graph::get_ordered_partitions(
        std::vector<partition_t *> &partitions) {
    dnnl_graph_graph copied_graph(*this); // deep copy

    // Cluster nodes that belong to same partition
    std::vector<std::vector<op_t *>> fusion_nodes;
    topo_order_visit(copied_graph.get_output_ops(), [&](op_t *n) {
        partition_impl_t *part = n->get_partition();
        if (!part) return;
        auto pos = std::find_if(fusion_nodes.begin(), fusion_nodes.end(),
                [&](std::vector<op_t *> &tmp) -> bool {
                    return tmp[0]->get_partition() == part;
                });
        if (pos != fusion_nodes.end()) {
            pos->emplace_back(n);
        } else {
            std::vector<op_t *> tmp {n};
            fusion_nodes.emplace_back(tmp);
        }
    });

    // Fuse nodes that belong to same partition
    rewrite(copied_graph, fusion_nodes);

    // Get partitions out according to the order of fused node
    // TODO(qun) Here is a workaround. Dfs order of unfused nodes
    // and fused nodes is not exactly same, which will break the
    // tests and examples
    size_t count = 0;
    topo_order_visit(copied_graph.get_output_ops(), [&](op_t *n) {
        partition_impl_t *part = n->get_partition();
        if (part) {
            partitions[count]->init(part->shared_from_this());
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

// Deep copy a graph
std::vector<dnnl_graph_graph::op_ptr> dnnl_graph_graph::deep_copy(
        const std::vector<dnnl_graph_graph::op_ptr> &ops) {
    using op_t = dnnl::graph::impl::op_t;
    using op_ptr = dnnl_graph_graph::op_ptr;
    using value_ptr = std::shared_ptr<value_t>;

    std::vector<op_ptr> copied_ops;

    // Create org_node to new_node map
    std::unordered_map<op_ptr, op_ptr> op_map;
    for (const op_ptr &cur_op : ops) {
        // copy the node
        op_ptr copied_op = std::make_shared<op_t>(
                cur_op->get_id(), cur_op->get_kind(), cur_op->get_name());
        copied_op->merge_attributes(cur_op->get_attributes());
        copied_op->set_partition(cur_op->get_partition());

        op_map[cur_op] = copied_op;
        copied_ops.emplace_back(copied_op);
    }

    // Connect the new nodes according to org nodes
    std::unordered_map<value_ptr, value_ptr> value_map;
    for (const op_ptr &cur_op : ops) {
        op_ptr copied_op = op_map[cur_op];

        for (size_t i = 0; i < cur_op->num_outputs(); i++) {
            auto value = cur_op->get_output_value(i);

            value_ptr copied_value;
            if (value_map.count(value) == 0) {
                copied_value = std::make_shared<value_t>(
                        value->get_logical_tensor(), value->is_internal());
                value_map[value] = copied_value;
            } else {
                copied_value = value_map[value];
            }

            copied_op->add_output(copied_value);
        }

        for (size_t i = 0; i < cur_op->num_inputs(); i++) {
            auto value = cur_op->get_input_value(i);

            value_ptr copied_value;
            if (value_map.count(value) == 0) {
                copied_value = std::make_shared<value_t>(
                        value->get_logical_tensor(), value->is_internal());
                value_map[value] = copied_value;
            } else {
                copied_value = value_map[value];
            }

            copied_op->add_input(copied_value);
            copied_value->add_consumer(*copied_op, i);
        }
    }

    return copied_ops;
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

    char *val = std::getenv("DNNL_GRAPH_DUMP");
    if (val != nullptr && std::strcmp(val, "1") == 0) {
        std::cout << "visualize un-fused graph to a dot file" << std::endl;
        graph->visualize("_backend.dot");
    }

    // Get partition_impl by calling each backends
    std::vector<const backend *> &backends
            = backend_registry::get_singleton().get_registered_backends();
    for (auto cbkd : backends) {
        backend *bkd = const_cast<backend *>(cbkd);
        status = bkd->get_partitions(*graph, policy);
        if (status != status::success) return status::invalid_graph;
    }

    // Check the partition_impl
    auto &partition_vec = graph->get_partitions();
    for (auto &p : partition_vec) {
        if (p->get_assigned_backend() == nullptr) {
            return status::invalid_graph;
        }
    }

    if (val != nullptr && std::strcmp(val, "1") == 0) {
        std::cout << "visualize fused graph to a dot file" << std::endl;
        graph->visualize("_backend_opt.dot");
    }

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
    graph->get_ordered_partitions(partitions);
    return status::success;
}
