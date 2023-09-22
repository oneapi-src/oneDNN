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

#include <memory>

#include "graph/interface/op.hpp"

#include "graph/utils/pm/pbuilder.hpp"

using namespace dnnl::impl::graph::utils::pm;
using std::dynamic_pointer_cast;

std::shared_ptr<consumers_t> pb_node_t::get_consumers(oport_t p_port) {
    if (outs_.size() <= p_port) { return nullptr; }
    return outs_[p_port];
}

std::shared_ptr<producer_t> pb_node_t::get_producer(iport_t p_port) {
    if (ins_.size() <= p_port) { return nullptr; }
    return ins_[p_port];
}

std::vector<std::pair<iport_t, producer_t>> pb_node_t::get_inputs() {
    std::vector<std::pair<iport_t, producer_t>> inputs;
    size_t s = ins_.size();
    for (size_t i = 0; i < s; i++) {
        if (ins_[i] != nullptr) inputs.emplace_back(i, *ins_[i]);
    }
    return inputs;
}

std::vector<std::pair<oport_t, consumers_t>> pb_node_t::get_outputs() {
    std::vector<std::pair<oport_t, consumers_t>> outputs;
    size_t s = outs_.size();
    for (size_t i = 0; i < s; i++) {
        if (outs_[i] != nullptr) outputs.emplace_back(i, *outs_[i]);
    }
    return outputs;
}

bool pb_node_t::set_producer(
        iport_t p_port, std::shared_ptr<producer_t> p_producer) {
    if (ins_.size() <= p_port) { ins_.resize(p_port + 1, nullptr); }
    ins_[p_port] = std::move(p_producer);
    return true;
}

bool pb_node_t::add_consumer(
        oport_t p_port, const std::shared_ptr<consumer_t> &p_consumer) {
    if (outs_.size() <= p_port) { outs_.resize(p_port + 1, nullptr); }
    std::shared_ptr<consumers_t> con = get_consumers(p_port);
    if (con == nullptr) {
        con = std::make_shared<consumers_t>();
        outs_[p_port] = con;
    }
    con->push_back(p_consumer);
    return true;
}

std::shared_ptr<consumer_t> dnnl::impl::graph::utils::pm::consumer(
        pb_node_t *p_node, iport_t i_t) {
    return std::make_shared<consumer_t>(p_node, i_t);
}

std::shared_ptr<consumer_t> dnnl::impl::graph::utils::pm::producer(
        pb_node_t *p_node, oport_t o_t) {
    return std::make_shared<producer_t>(p_node, o_t);
}

std::shared_ptr<in_edge_t> dnnl::impl::graph::utils::pm::in_edge(
        iport_t i_t, pb_node_t *p_node, oport_t o_t) {
    auto prod = std::make_shared<producer_t>(p_node, o_t);
    auto edge = std::make_shared<in_edge_t>(i_t, prod);
    return edge;
}

decision_function dnnl::impl::graph::utils::pm::kind(
        dnnl::impl::graph::op_kind_t okind) {
    return [okind](op_t *p_op) -> bool {
        return okind == p_op->get_kind() || okind == op_kind::Wildcard;
    };
}

decision_function dnnl::impl::graph::utils::pm::one_of_kind(
        const std::vector<dnnl::impl::graph::op_kind_t> &okind) {
    return [okind](op_t *p_op) -> bool {
        for (auto k : okind) {
            if (k == p_op->get_kind()) return true;
        }
        return false;
    };
}

bool pb_op_t::append_decision_function(const decision_function &p_fn) {
    decision_functions_.emplace_back(p_fn);
    return true;
}

size_t pb_node_t::get_num_decision_functions() {
    return decision_functions_.size();
}

decision_function pb_node_t::get_decision_function(size_t index) {
    if (index > get_num_decision_functions()) {
        decision_function foo;
        return foo;
    }
    return decision_functions_[index];
}

pb_op_t::pb_op_t(const decision_function &p_fn) {
    node_kind_ = pb_node_kind::PB_NODE_KIND_OP;
    p_ops_.insert(this);
    if (p_fn) { decision_functions_.emplace_back(p_fn); }
}

std::vector<std::pair<iport_t, consumers_t>> pb_graph_t::get_inner_consumers() {
    std::vector<std::pair<iport_t, consumers_t>> consumers;
    size_t s = inner_consumers_.size();
    for (size_t i = 0; i < s; i++) {
        if (inner_consumers_[i] != nullptr)
            consumers.emplace_back(i, *inner_consumers_[i]);
    }
    return consumers;
}

std::vector<std::pair<oport_t, producer_t>> pb_graph_t::get_inner_producers() {
    std::vector<std::pair<oport_t, producer_t>> producers;
    size_t s = inner_producers_.size();
    for (size_t i = 0; i < s; i++) {
        if (inner_producers_[i] != nullptr)
            producers.emplace_back(i, *inner_producers_[i]);
    }
    return producers;
}

std::shared_ptr<consumers_t> pb_graph_t::get_inner_consumer(iport_t i_t) {
    if (inner_consumers_.size() <= i_t) { return nullptr; }
    return inner_consumers_[i_t];
}

std::shared_ptr<producer_t> pb_graph_t::get_inner_producer(oport_t o_t) {
    if (inner_producers_.size() <= o_t) { return nullptr; }
    return inner_producers_[o_t];
}

bool pb_graph_t::create_input_port(
        iport_t p_port, const std::shared_ptr<consumer_t> &p_consumer) {
    for (auto const &con_set : inner_consumers_) {
        if (con_set != nullptr) {
            for (auto const &con : *con_set) {
                if (con->first == p_consumer->first
                        && con->second == p_consumer->second)
                    return false;
            }
        }
    }
    if (inner_consumers_.size() <= p_port) {
        inner_consumers_.resize(p_port + 1, nullptr);
    }
    if (inner_consumers_[p_port] == nullptr) {
        inner_consumers_[p_port] = std::make_shared<consumers_t>();
    }
    inner_consumers_[p_port]->push_back(p_consumer);
    return true;
}

bool pb_graph_t::create_output_port(
        oport_t p_port, std::shared_ptr<producer_t> p_producer) {
    if (inner_producers_.size() <= p_port) {
        inner_producers_.resize(p_port + 1, nullptr);
    }
    if (inner_producers_[p_port] != nullptr) return false;
    inner_producers_[p_port] = std::move(p_producer);
    return true;
}

bool pb_graph_t::create_input_port(
        iport_t p_port, pb_node_t *p_int_node, iport_t p_int_port) {
    return create_input_port(p_port, consumer(p_int_node, p_int_port));
}

bool pb_graph_t::create_output_port(
        oport_t p_port, pb_node_t *p_int_node, oport_t p_int_port) {
    return create_output_port(p_port, producer(p_int_node, p_int_port));
}

bool pb_graph_t::connect_edges(
        pb_node_t *p_node, const in_edges_t &p_in_edges) {
    if (!p_in_edges.empty()) {
        for (auto const &i : p_in_edges) {
            auto con = std::make_shared<consumer_t>(p_node, i->first);
            set_edge(con, i->second);
        }
    }
    return true;
}

bool pb_graph_t::set_edge(const std::shared_ptr<consumer_t> &p_consumer,
        const std::shared_ptr<producer_t> &p_producer) {
    auto con = p_consumer->first;
    con->set_producer(p_consumer->second, p_producer);
    auto pro = p_producer->first;
    pro->add_consumer(p_producer->second, p_consumer);
    return true;
}

std::vector<pb_node_t *> pb_graph_t::get_nodes() {
    std::vector<pb_node_t *> retval;
    for (auto const &i : nodes_) {
        retval.push_back(i.get());
    }
    return retval;
}

pb_graph_t::pb_graph_t() {
    debug_string_ = "pgraph";
}

pb_op_t *pb_graph_t::append_op(const decision_function &p_fn,
        const in_edges_t &p_in_edges, std::string name) {
    std::shared_ptr<pb_op_t> p_op(new pb_op_t(p_fn));
    p_op->set_name(std::move(name));
    connect_edges(p_op.get(), p_in_edges);
    nodes_.push_back(dynamic_pointer_cast<pb_node_t>(p_op));
    p_ops_.insert(p_op.get());
    return p_op.get();
}

pb_op_t *pb_graph_t::append_op(
        const decision_function &p_fn, std::string name) {
    return append_op(p_fn, {}, std::move(name));
}

pb_op_t *pb_graph_t::append_op(
        dnnl::impl::graph::op_kind_t p_kind, const in_edges_t &p_in_edges) {
    return append_op(kind(p_kind), p_in_edges,
            dnnl::impl::graph::op_t::kind2str(p_kind)
                    + std::to_string(nodes_.size()));
}

pb_op_t *pb_graph_t::append_op(dnnl::impl::graph::op_kind_t p_kind) {
    return append_op(kind(p_kind), {},
            dnnl::impl::graph::op_t::kind2str(p_kind)
                    + std::to_string(nodes_.size()));
}

pb_op_t *pb_graph_t::append_alternation(
        const std::vector<dnnl::impl::graph::op_kind_t> &p_kind,
        const in_edges_t &p_in_edges) {
    return append_op(one_of_kind(p_kind), p_in_edges,
            "alternation" + std::to_string(nodes_.size()));
}

pb_op_t *pb_graph_t::append_alternation(
        const std::vector<dnnl::impl::graph::op_kind_t> &p_kind) {
    return append_op(one_of_kind(p_kind), {},
            "alternation" + std::to_string(nodes_.size()));
}

alternation_t *pb_graph_t::append_alternation(
        const std::vector<std::shared_ptr<pb_graph_t>> &p_nodes,
        const in_edges_t &p_in_edges) {
    for (size_t i = 0; i < p_nodes.size(); ++i) {
        p_nodes[i]->set_name("alternation" + std::to_string(nodes_.size())
                + "_pgraph" + std::to_string(i));
    }
    std::shared_ptr<alternation_t> p_alternation(new alternation_t(p_nodes));
    p_alternation->set_name("alternation" + std::to_string(nodes_.size()));
    connect_edges(p_alternation.get(), p_in_edges);
    nodes_.push_back(dynamic_pointer_cast<pb_node_t>(p_alternation));
    auto contained_ops = p_alternation->get_contained_ops();
    p_ops_.insert(contained_ops.begin(), contained_ops.end());
    return p_alternation.get();
}

alternation_t *pb_graph_t::append_alternation(
        const std::vector<std::shared_ptr<pb_graph_t>> &p_nodes) {
    return append_alternation(p_nodes, {});
}

repetition_t *pb_graph_t::append_repetition(
        const std::shared_ptr<pb_graph_t> &p_node, const port_map &p_map,
        size_t min_rep, size_t max_rep, const in_edges_t &p_in_edges) {
    assertm(p_map.first == 0, "repetition only supports 1 output port");
    p_node->set_name("repetition" + std::to_string(nodes_.size()) + "_pgraph");
    std::shared_ptr<repetition_t> p_repetition(
            new repetition_t(p_node, p_map, min_rep, max_rep));
    p_repetition->set_name("repetition" + std::to_string(nodes_.size()));
    connect_edges(p_repetition.get(), p_in_edges);
    nodes_.push_back(dynamic_pointer_cast<pb_node_t>(p_repetition));
    auto contained_ops = p_repetition->get_contained_ops();
    p_ops_.insert(contained_ops.begin(), contained_ops.end());
    return p_repetition.get();
}

repetition_t *pb_graph_t::append_repetition(
        const std::shared_ptr<pb_graph_t> &p_node, const port_map &p_map,
        size_t min_rep, size_t max_rep) {
    return append_repetition(p_node, p_map, min_rep, max_rep, {});
}

repetition_t *pb_graph_t::append_optional(
        const std::shared_ptr<pb_graph_t> &p_node,
        const in_edges_t &p_in_edges) {
    // When append optional consumer B to a producer A, some conditions need
    // to be met:
    // A -> B*
    // 1. for the optional consumer B, it should have only 1 producer (A in
    // this case)
    // 2. for the producer A, it should have only 1 output, and the output
    // should have only 1 consumer (B in this case)
    //
    assertm(p_in_edges.size() <= 1, "optional graph can only have 0/1 input");
    if (p_in_edges.size() == 1) {
        assertm(p_in_edges[0]->second->first->get_outputs().empty(),
                "optional graph's producer can only have 1 output and 1 "
                "consumer");
    }
    p_node->set_name("optional" + std::to_string(nodes_.size()) + "_pgraph");
    std::shared_ptr<repetition_t> p_repetition(new repetition_t(p_node));
    p_repetition->set_name("optional" + std::to_string(nodes_.size()));
    connect_edges(p_repetition.get(), p_in_edges);
    nodes_.push_back(dynamic_pointer_cast<pb_node_t>(p_repetition));
    return p_repetition.get();
}

repetition_t *pb_graph_t::append_optional(
        const std::shared_ptr<pb_graph_t> &p_node) {
    return append_optional(p_node, {});
}

alternation_t::alternation_t(std::vector<std::shared_ptr<pb_graph_t>> p_nodes)
    : alternatives_ {std::move(p_nodes)} {
    node_kind_ = pb_node_kind::PB_NODE_KIND_ALTERNATION;
    for (const auto &node : alternatives_) {
        auto contained_ops = node->get_contained_ops();
        p_ops_.insert(contained_ops.begin(), contained_ops.end());
    }
}

std::vector<pb_graph_t *> alternation_t::get_alternatives() {
    std::vector<pb_graph_t *> retval;
    for (auto const &i : alternatives_) {
        retval.push_back(i.get());
    }
    return retval;
}

repetition_t::repetition_t(std::shared_ptr<pb_graph_t> p_node, port_map p_map,
        size_t min_rep, size_t max_rep)
    : body_ {std::move(p_node)}
    , port_map_ {std::move(p_map)}
    , min_rep_ {min_rep}
    , max_rep_ {max_rep} {
    node_kind_ = pb_node_kind::PB_NODE_KIND_REPETITION;
    auto contained_ops = body_->get_contained_ops();
    p_ops_.insert(contained_ops.begin(), contained_ops.end());
}

repetition_t::repetition_t(std::shared_ptr<pb_graph_t> p_node)
    : body_ {std::move(p_node)}, min_rep_ {0}, max_rep_ {2} {
    node_kind_ = pb_node_kind::PB_NODE_KIND_REPETITION;
    port_map_ = {0, 0};
    auto contained_ops = body_->get_contained_ops();
    p_ops_.insert(contained_ops.begin(), contained_ops.end());
}

pb_graph_t *repetition_t::get_body() {
    return body_.get();
}

port_map repetition_t::get_port_map() {
    return port_map_;
}
