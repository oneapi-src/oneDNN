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

#include <cstring>
#include <fstream>
#include <iostream>
#include <list>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>

#include "interface/backend.hpp"
#include "interface/c_types_map.hpp"
#include "interface/graph.hpp"
#include "interface/partition.hpp"

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
            if (dnnl::graph::impl::logical_tensor_wrapper_t(id_search->second)
                    != dnnl::graph::impl::logical_tensor_wrapper_t(lt)) {
                return false;
            }
        }
    }
    return true;
}
} // namespace

namespace dnnl {
namespace graph {
namespace impl {
fpmath_mode_t get_default_fpmath_mode() {
    static fpmath_mode_t default_fpmath_mode = fpmath_mode::strict;
    static std::string val = utils::getenv_string_user("DEFAULT_FPMATH_MODE");
    if (!val.empty()) {
        if (val.compare("strict") == 0)
            default_fpmath_mode = fpmath_mode::strict;
        if (val.compare("bf16") == 0) default_fpmath_mode = fpmath_mode::bf16;
        if (val.compare("f16") == 0) default_fpmath_mode = fpmath_mode::f16;
        if (val.compare("tf32") == 0) default_fpmath_mode = fpmath_mode::tf32;
        if (val.compare("any") == 0) default_fpmath_mode = fpmath_mode::any;
    }
    return default_fpmath_mode;
}

// function to do graph rewriting
void rewrite(impl::graph_t &agraph,
        const std::vector<std::vector<op_t *>> &fusion_ops) {
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

        // the values that will be connected to fused op
        std::vector<std::pair<std::shared_ptr<value_t>, value_t::consumer_t>>
                fused_inputs;
        std::vector<std::shared_ptr<value_t>> fused_outputs;

        for (size_t i = 0; i < ops.size(); ++i) {
            op_t *cur_op = ops[i];
            visited.insert(cur_op);
            fused_op->merge_attributes(cur_op->get_attributes());
            fused_op->add_op_ids(cur_op->get_op_ids());

            // if cur_op has input op which isn't in pattern,
            // update value's connection. if cur_op has input op
            // which is in pattern, add its output_tensor into visited
            for (size_t j = 0; j < cur_op->num_inputs(); ++j) {
                auto in_value = cur_op->get_input_value(j);
                //if in_op isn't in pattern,
                //set it as a input op of fused_op
                if (!in_value->has_producer()
                        || !visited.count(&in_value->get_producer())) {
                    fused_inputs.emplace_back(
                            in_value, value_t::consumer_t(*cur_op, j));
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
                    // it's a end op of pattern, need to update
                    // op connection of it's output ops
                    fused_outputs.emplace_back(out_value);
                }
            }
        }

        // connect inputs to fused op
        for (const auto &fused_ins : fused_inputs) {
            fused_ins.first->remove_consumer(
                    fused_ins.second.get_op(), fused_ins.second.get_offset());
            fused_ins.first->add_consumer(*fused_op, fused_op->num_inputs());
            fused_op->add_input(fused_ins.first);
        }

        // connect outputs to fused op
        for (const auto &fused_outs : fused_outputs) {
            fused_outs->set_producer(*fused_op);
            fused_op->add_output(fused_outs);
        }

        for (size_t i = 0; i < ops.size(); ++i) {
            agraph.delete_op(ops[i]);
        }
    }
}
} // namespace impl
} // namespace graph
} // namespace dnnl

void dnnl_graph_graph::get_ordered_partitions(
        std::vector<partition_t *> &partitions) {
    dnnl_graph_graph copied_graph(*this); // deep copy

    // Cluster ops that belong to same partition
    std::vector<std::vector<op_t *>> fusion_ops;
    topo_order_visit(copied_graph.get_output_ops(), [&](op_t *n) {
        partition_impl_t *part = n->get_partition();
        if (!part) return impl::status::success;
        auto pos = std::find_if(fusion_ops.begin(), fusion_ops.end(),
                [&](std::vector<op_t *> &tmp) -> bool {
                    return tmp[0]->get_partition() == part;
                });
        if (pos != fusion_ops.end()) {
            pos->emplace_back(n);
        } else {
            std::vector<op_t *> tmp {n};
            fusion_ops.emplace_back(tmp);
        }
        return impl::status::success;
    });

    // Fuse ops that belong to same partition
    impl::rewrite(copied_graph, fusion_ops);

    // Get partitions out according to the order of fused op
    // TODO(qun) Here is a workaround. Dfs order of unfused ops
    // and fused ops is not exactly same, which will break the
    // tests and examples
    size_t count = 0;
    topo_order_visit(copied_graph.get_output_ops(), [&](op_t *n) {
        partition_impl_t *part = n->get_partition();
        if (part) {
            partitions[count]->init(part->shared_from_this());
            count++;
        }
        return impl::status::success;
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

    std::list<op_t> dummy_ops; // use list to avoid re-allocation
    std::unordered_map<size_t, std::pair<graph_t::op_t *, size_t>>
            tensor_id_to_dummy_producer;

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
            } else {
                auto id_search
                        = tensor_id_to_dummy_producer.find(input_tensor_ids[i]);
                if (id_search != tensor_id_to_dummy_producer.end()) {
                    std::pair<op_t *, size_t> dummy_producer
                            = id_search->second;
                    op->connect_input(
                            i, *(dummy_producer.first), dummy_producer.second);
                    op->get_input_value(i)->reset_producer();
                } else { // create a dummy input op
                    dummy_ops.emplace_back(op_kind::Wildcard);
                    dummy_ops.back().add_output(op->get_input_value(i));
                    tensor_id_to_dummy_producer[input_tensor_ids[i]]
                            = std::make_pair(&dummy_ops.back(), 0);
                    op->connect_input(i, op->get_input_value(i));
                    op->get_input_value(i)->reset_producer();
                }
            }
        }
    }

    is_built_ = true;
    return status::success;
}

// Deep copy a graph
std::vector<dnnl_graph_graph::op_ptr> dnnl_graph_graph::deep_copy(
        const std::vector<dnnl_graph_graph::op_ptr> &ops) {
    using op_t = dnnl::graph::impl::op_t;
    using op_ptr = dnnl_graph_graph::op_ptr;
    using value_ptr = std::shared_ptr<value_t>;

    std::vector<op_ptr> copied_ops;

    // Create org_op to new_op map
    std::unordered_map<op_ptr, op_ptr> op_map;
    for (const op_ptr &cur_op : ops) {
        // copy the op
        op_ptr copied_op = std::make_shared<op_t>(
                cur_op->get_id(), cur_op->get_kind(), cur_op->get_name());
        copied_op->merge_attributes(cur_op->get_attributes());
        copied_op->set_partition(cur_op->get_partition());

        op_map[cur_op] = copied_op;
        copied_ops.emplace_back(copied_op);
    }

    // Connect the new ops according to org ops
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
        graph_t **graph, engine_kind_t engine_kind) {
    *graph = new graph_t(engine_kind);
    return status::success;
}

status_t DNNL_GRAPH_API dnnl_graph_graph_create_with_fpmath_mode(
        graph_t **graph, engine_kind_t engine_kind, fpmath_mode_t fpmath_mode) {
    if (graph == nullptr) return status::invalid_arguments;
    *graph = new graph_t(engine_kind, fpmath_mode);
    return status::success;
}

status_t DNNL_GRAPH_API dnnl_graph_graph_destroy(graph_t *graph) {
    delete graph;
    return status::success;
}

status_t DNNL_GRAPH_API dnnl_graph_add_op(graph_t *graph, op_t *op) {
    if (graph == nullptr || op == nullptr) { return status::invalid_arguments; }
    return graph->add_op(op);
}

status_t DNNL_GRAPH_API dnnl_graph_graph_filter(
        graph_t *graph, partition_policy_t policy) {
    if (graph == nullptr) { return status::invalid_graph; }
    auto status = graph->build_graph();
    if (status != status::success) return status::invalid_graph;

#ifdef DNNL_GRAPH_ENABLE_DUMP
    if (utils::getenv_int_user("DUMP", 0) > 0
            || utils::check_verbose_string_user("DUMP", "graph")) {
        // deep copy for graph serialization. note that this is for
        // visualization purpose
        graph_t agraph(*graph);
        std::stringstream filename;
        filename << "graph-" << agraph.id() << ".json";
        agraph.serialize(filename.str());
    }
#endif

    // Get partition_impl by calling each backends
    std::vector<const backend *> &backends
            = backend_registry_t::get_singleton().get_registered_backends();
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

    if (status != status::success) {
        return status::invalid_graph;
    } else {
        return status::success;
    }
}

status_t DNNL_GRAPH_API dnnl_graph_graph_get_partition_num(
        const graph_t *graph, size_t *num) {
    if (graph == nullptr) { return status::invalid_graph; }
    *num = graph->get_num_partitions();
    return status::success;
}

status_t DNNL_GRAPH_API dnnl_graph_graph_get_partitions(
        graph_t *graph, size_t num, partition_t **partition) {
    if (graph == nullptr) { return status::invalid_graph; }
    std::vector<partition_t *> partitions {partition, partition + num};
    graph->get_ordered_partitions(partitions);
#ifdef DNNL_GRAPH_ENABLE_DUMP
    if (utils::getenv_int_user("DUMP", 0) > 0
            || utils::check_verbose_string_user("DUMP", "graph")) {
        // graph serialization after partitioning. note that this is for
        // visualization purpose
        graph_t agraph(*graph);
        for (auto &aop : agraph.get_ops()) {
            // p_impl is shallow copy
            const auto p_impl = aop->get_partition();
            const auto p_id = p_impl->id();
            auto *bkd = p_impl->get_assigned_backend();
            auto bkd_name = bkd->get_name();
            aop->set_attr<std::string>(
                    op_attr::partition_id, std::to_string(p_id));
            aop->set_attr<std::string>(op_attr::backend, bkd_name);
        }

        std::stringstream filename;
        filename << "graph-" << agraph.id() << "-partitioning.json";
        agraph.serialize(filename.str());
    }
#endif
    return status::success;
}
