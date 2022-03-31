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

#include <memory>

#include "interface/op.hpp"
#include "utils/pm/pbuilder.hpp"

using namespace dnnl::graph::impl::utils::pm;
using std::dynamic_pointer_cast;

std::shared_ptr<consumers_t> pb_node::get_consumers(oport_t p_port) {
    if (p_port < 0) { return nullptr; }
    if (m_outs.size() <= p_port) { return nullptr; }
    return m_outs[static_cast<uint64_t>(p_port)];
}

std::shared_ptr<producer_t> pb_node::get_producer(iport_t p_port) {
    if (p_port < 0) { return nullptr; }
    if (m_ins.size() <= p_port) { return nullptr; }
    return m_ins[static_cast<uint64_t>(p_port)];
}

std::vector<std::pair<iport_t, producer_t>> pb_node::get_inputs() {
    std::vector<std::pair<iport_t, producer_t>> inputs;
    size_t s = m_ins.size();
    for (size_t i = 0; i < s; i++) {
        if (m_ins[i] != nullptr)
            inputs.emplace_back(static_cast<iport_t>(i), *m_ins[i]);
    }
    return inputs;
}

std::vector<std::pair<oport_t, consumers_t>> pb_node::get_outputs() {
    std::vector<std::pair<oport_t, consumers_t>> outputs;
    size_t s = m_outs.size();
    for (size_t i = 0; i < s; i++) {
        if (m_outs[i] != nullptr)
            outputs.emplace_back(static_cast<oport_t>(i), *m_outs[i]);
    }
    return outputs;
}

bool pb_node::set_producer(
        iport_t p_port, std::shared_ptr<producer_t> p_producer) {
    if (p_port < 0) { return false; }
    uint64_t index = static_cast<uint64_t>(p_port);
    if (m_ins.size() <= index) { m_ins.resize(index + 1, nullptr); }
    m_ins[index] = move(p_producer);
    return true;
}

bool pb_node::set_consumers(
        oport_t p_port, std::shared_ptr<consumers_t> p_consumers) {
    if (p_port < 0) { return false; }
    uint64_t index = static_cast<uint64_t>(p_port);
    if (m_outs.size() <= index) { m_outs.resize(index + 1, nullptr); }
    m_outs[index] = move(p_consumers);
    return true;
}

bool pb_node::add_consumer(
        oport_t p_port, const std::shared_ptr<consumer_t> &p_consumer) {
    if (p_port < 0) { return false; }
    uint64_t index = static_cast<uint64_t>(p_port);
    if (m_outs.size() <= index) { m_outs.resize(index + 1, nullptr); }
    std::shared_ptr<consumers_t> con = get_consumers(p_port);
    if (con == nullptr) {
        con = std::make_shared<consumers_t>();
        m_outs[index] = con;
    }
    con->push_back(p_consumer);
    return true;
}

std::shared_ptr<consumer_t> dnnl::graph::impl::utils::pm::consumer(
        pb_node *p_node, iport_t i_t) {
    return std::make_shared<consumer_t>(p_node, i_t);
}

std::shared_ptr<consumer_t> dnnl::graph::impl::utils::pm::producer(
        pb_node *p_node, oport_t o_t) {
    return std::make_shared<producer_t>(p_node, o_t);
}

std::shared_ptr<in_edge_t> dnnl::graph::impl::utils::pm::in_edge(
        iport_t i_t, pb_node *p_node, oport_t o_t) {
    auto prod = std::make_shared<producer_t>(p_node, o_t);
    auto edge = std::make_shared<in_edge_t>(i_t, prod);
    return edge;
}

decision_function dnnl::graph::impl::utils::pm::kind(
        dnnl::graph::impl::op_kind_t okind) {
    return [okind](op_t *p_op) -> bool {
        return okind == p_op->get_kind() || okind == op_kind::Wildcard;
    };
}

decision_function dnnl::graph::impl::utils::pm::one_of_kind(
        const std::vector<dnnl::graph::impl::op_kind_t> &okind) {
    return [okind](op_t *p_op) -> bool {
        for (auto k : okind) {
            if (k == p_op->get_kind()) return true;
        }
        return false;
    };
}

bool pb_op::append_decision_function(const decision_function &p_fn) {
    m_decision_functions.emplace_back(p_fn);
    return true;
}

size_t pb_node::get_num_decision_functions() {
    return m_decision_functions.size();
}

decision_function pb_node::get_decision_function(size_t index) {
    if (index > get_num_decision_functions()) {
        decision_function foo;
        return foo;
    }
    return m_decision_functions[index];
}

pb_op::pb_op(const decision_function &p_fn) {
    m_node_kind = pb_node_kind::PB_NODE_KIND_OP;
    if (p_fn) { m_decision_functions.emplace_back(p_fn); }
}

void pb_op::allow_external_output(oport_t p_port) {
    m_external_outputs.insert(p_port);
}

void pb_op::allow_internal_input(iport_t p_port) {
    m_internal_inputs.insert(p_port);
}

std::vector<std::pair<oport_t, consumers_t>> pb_graph_t::get_inner_consumers() {
    std::vector<std::pair<oport_t, consumers_t>> consumers;
    size_t s = m_inner_consumers.size();
    for (size_t i = 0; i < s; i++) {
        if (m_inner_consumers[i] != nullptr)
            consumers.emplace_back(
                    static_cast<oport_t>(i), *m_inner_consumers[i]);
    }
    return consumers;
}

std::vector<std::pair<iport_t, producer_t>> pb_graph_t::get_inner_producers() {
    std::vector<std::pair<iport_t, producer_t>> producers;
    size_t s = m_inner_producers.size();
    for (size_t i = 0; i < s; i++) {
        if (m_inner_producers[i] != nullptr)
            producers.emplace_back(
                    static_cast<iport_t>(i), *m_inner_producers[i]);
    }
    return producers;
}

std::shared_ptr<consumers_t> pb_graph_t::get_inner_consumer(iport_t i_t) {
    if (i_t < 0) { return nullptr; }
    uint64_t index = static_cast<uint64_t>(i_t);
    if (m_inner_consumers.size() <= index) { return nullptr; }
    return m_inner_consumers[index];
}

std::shared_ptr<producer_t> pb_graph_t::get_inner_producer(oport_t o_t) {
    if (o_t < 0) { return nullptr; }
    uint64_t index = static_cast<uint64_t>(o_t);
    if (m_inner_producers.size() <= index) { return nullptr; }
    return m_inner_producers[index];
}

bool pb_graph_t::create_input_port(
        iport_t p_port, const std::shared_ptr<consumer_t> &p_consumer) {
    if (p_port < 0) { return false; }
    uint64_t index = static_cast<uint64_t>(p_port);
    for (auto const &con_set : m_inner_consumers) {
        if (con_set != nullptr) {
            for (auto const &con : *con_set) {
                if (con->first == p_consumer->first
                        && con->second == p_consumer->second)
                    return false;
            }
        }
    }
    if (m_inner_consumers.size() <= index) {
        m_inner_consumers.resize(index + 1, nullptr);
    }
    if (m_inner_consumers[index] == nullptr) {
        m_inner_consumers[index] = std::make_shared<consumers_t>();
    }
    m_inner_consumers[index]->push_back(p_consumer);
    return true;
}

bool pb_graph_t::create_output_port(
        oport_t p_port, std::shared_ptr<producer_t> p_producer) {
    if (p_port < 0) { return false; }
    uint64_t index = static_cast<uint64_t>(p_port);
    if (m_inner_producers.size() <= index) {
        m_inner_producers.resize(index + 1, nullptr);
    }
    if (m_inner_producers[index] != nullptr) return false;
    m_inner_producers[index] = move(p_producer);
    return true;
}

bool pb_graph_t::create_input_port(
        iport_t p_port, pb_node *p_int_node, iport_t p_int_port) {
    return create_input_port(p_port, consumer(p_int_node, p_int_port));
}

bool pb_graph_t::create_output_port(
        oport_t p_port, pb_node *p_int_node, oport_t p_int_port) {
    return create_output_port(p_port, producer(p_int_node, p_int_port));
}

bool pb_graph_t::connect_edges(pb_node *p_node, const in_edges_t &p_in_edges) {
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

std::vector<pb_node *> pb_graph_t::get_nodes() {
    std::vector<pb_node *> retval;
    for (auto const &i : m_nodes) {
        retval.push_back(i.get());
    }
    return retval;
}

pb_graph_t::pb_graph_t(std::string name) {
    m_debug_string = move(name);
}

pb_op *pb_graph_t::append_op(const decision_function &p_fn,
        const in_edges_t &p_in_edges, std::string name) {
    std::shared_ptr<pb_op> p_op(new pb_op(p_fn));
    p_op->set_name(move(name));
    connect_edges(p_op.get(), p_in_edges);
    m_nodes.push_back(dynamic_pointer_cast<pb_node>(p_op));
    return p_op.get();
}

pb_op *pb_graph_t::append_op(const decision_function &p_fn, std::string name) {
    return append_op(p_fn, {}, move(name));
}

pb_op *pb_graph_t::append_op(dnnl::graph::impl::op_kind_t p_kind,
        const in_edges_t &p_in_edges, std::string name) {
    return append_op(kind(p_kind), p_in_edges, move(name));
}

pb_op *pb_graph_t::append_op(
        dnnl::graph::impl::op_kind_t p_kind, std::string name) {
    return append_op(kind(p_kind), {}, move(name));
}

pb_op *pb_graph_t::append_alternation(
        const std::vector<dnnl::graph::impl::op_kind_t> &p_kind,
        const in_edges_t &p_in_edges, std::string name) {
    return append_op(one_of_kind(p_kind), p_in_edges, move(name));
}

pb_op *pb_graph_t::append_alternation(
        const std::vector<dnnl::graph::impl::op_kind_t> &p_kind,
        std::string name) {
    return append_op(one_of_kind(p_kind), {}, move(name));
}

alternation_t *pb_graph_t::append_alternation(
        std::vector<std::shared_ptr<pb_graph_t>> p_nodes,
        const in_edges_t &p_in_edges, std::string name) {
    std::shared_ptr<alternation_t> p_alternation(
            new alternation_t(move(p_nodes)));
    p_alternation->set_name(move(name));
    connect_edges(p_alternation.get(), p_in_edges);
    m_nodes.push_back(dynamic_pointer_cast<pb_node>(p_alternation));
    return p_alternation.get();
}

alternation_t *pb_graph_t::append_alternation(
        std::vector<std::shared_ptr<pb_graph_t>> p_nodes, std::string name) {
    return append_alternation(move(p_nodes), {}, move(name));
}

repetition_t *pb_graph_t::append_repetition(std::shared_ptr<pb_graph_t> p_node,
        port_map p_map, int64_t min_rep, int64_t max_rep,
        const in_edges_t &p_in_edges, std::string name) {
    assertm(p_map.first == 0, "repetition only supports 1 output port");
    std::shared_ptr<repetition_t> p_repetition(
            new repetition_t(move(p_node), move(p_map), min_rep, max_rep));
    p_repetition->set_name(move(name));
    connect_edges(p_repetition.get(), p_in_edges);
    m_nodes.push_back(dynamic_pointer_cast<pb_node>(p_repetition));
    return p_repetition.get();
}

repetition_t *pb_graph_t::append_repetition(std::shared_ptr<pb_graph_t> p_node,
        port_map p_map, int64_t min_rep, int64_t max_rep, std::string name) {
    return append_repetition(
            move(p_node), move(p_map), min_rep, max_rep, {}, move(name));
}

repetition_t *pb_graph_t::append_optional(std::shared_ptr<pb_graph_t> p_node,
        const in_edges_t &p_in_edges, std::string name) {
    std::shared_ptr<repetition_t> p_repetition(new repetition_t(move(p_node)));
    p_repetition->set_name(move(name));
    connect_edges(p_repetition.get(), p_in_edges);
    m_nodes.push_back(dynamic_pointer_cast<pb_node>(p_repetition));
    return p_repetition.get();
}

repetition_t *pb_graph_t::append_optional(
        std::shared_ptr<pb_graph_t> p_node, std::string name) {
    return append_optional(move(p_node), {}, move(name));
}

alternation_t::alternation_t(std::vector<std::shared_ptr<pb_graph_t>> p_nodes)
    : m_alternatives {move(p_nodes)} {
    m_node_kind = pb_node_kind::PB_NODE_KIND_ALTERNATION;
}

std::vector<pb_graph_t *> alternation_t::get_alternatives() {
    std::vector<pb_graph_t *> retval;
    for (auto const &i : m_alternatives) {
        retval.push_back(i.get());
    }
    return retval;
}

repetition_t::repetition_t(std::shared_ptr<pb_graph_t> p_node, port_map p_map,
        int64_t min_rep, int64_t max_rep)
    : m_body {move(p_node)}
    , m_port_map {move(p_map)}
    , m_min_rep {min_rep}
    , m_max_rep {max_rep} {
    m_node_kind = pb_node_kind::PB_NODE_KIND_REPETITION;
}

repetition_t::repetition_t(std::shared_ptr<pb_graph_t> p_node)
    : m_body {move(p_node)}, m_min_rep {0}, m_max_rep {2} {
    m_node_kind = pb_node_kind::PB_NODE_KIND_REPETITION;
    m_port_map = {0, 0};
}

pb_graph_t *repetition_t::get_body() {
    return m_body.get();
}

port_map repetition_t::get_port_map() {
    return m_port_map;
}
