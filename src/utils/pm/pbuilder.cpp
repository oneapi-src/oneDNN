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

#include <memory>

#include "interface/op.hpp"
#include "utils/pm/pbuilder.hpp"

using namespace dnnl::graph::impl::utils::pm;
using std::dynamic_pointer_cast;

shared_ptr<consumers_t> pb_node::get_consumers(oport_t p_port) {
    if (p_port < 0) { return nullptr; }
    if (m_outs.size() <= p_port) { return nullptr; }
    return m_outs[static_cast<uint64_t>(p_port)];
}

shared_ptr<producer_t> pb_node::get_producer(iport_t p_port) {
    if (p_port < 0) { return nullptr; }
    if (m_ins.size() <= p_port) { return nullptr; }
    return m_ins[static_cast<uint64_t>(p_port)];
}

vector<pair<iport_t, producer_t>> pb_node::get_inputs() {
    vector<pair<iport_t, producer_t>> inputs;
    size_t s = m_ins.size();
    for (size_t i = 0; i < s; i++) {
        if (m_ins[i] != nullptr)
            inputs.emplace_back(static_cast<iport_t>(i), *m_ins[i]);
    }
    return inputs;
}

vector<pair<oport_t, consumers_t>> pb_node::get_outputs() {
    vector<pair<oport_t, consumers_t>> outputs;
    size_t s = m_outs.size();
    for (size_t i = 0; i < s; i++) {
        if (m_outs[i] != nullptr)
            outputs.emplace_back(static_cast<oport_t>(i), *m_outs[i]);
    }
    return outputs;
}

bool pb_op::set_commutative_pair(iport_pair p_port_pair) {
    if (p_port_pair.first < p_port_pair.second) {
        m_commutative_pair = p_port_pair;
    } else {
        return false;
    }
    return true;
}

iport_pair pb_node::get_commutative_pair() {
    return m_commutative_pair;
}

bool pb_node::set_producer(iport_t p_port, shared_ptr<producer_t> p_producer) {
    if (p_port < 0) { return false; }
    uint64_t index = static_cast<uint64_t>(p_port);
    if (m_ins.size() <= index) { m_ins.resize(index + 1, nullptr); }
    m_ins[index] = move(p_producer);
    return true;
}

bool pb_node::set_consumers(
        oport_t p_port, shared_ptr<consumers_t> p_consumers) {
    if (p_port < 0) { return false; }
    uint64_t index = static_cast<uint64_t>(p_port);
    if (m_outs.size() <= index) { m_outs.resize(index + 1, nullptr); }
    m_outs[index] = move(p_consumers);
    return true;
}

bool pb_node::add_consumer(
        oport_t p_port, const shared_ptr<consumer_t> &p_consumer) {
    if (p_port < 0) { return false; }
    uint64_t index = static_cast<uint64_t>(p_port);
    if (m_outs.size() <= index) { m_outs.resize(index + 1, nullptr); }
    shared_ptr<consumers_t> con = get_consumers(p_port);
    if (con == nullptr) {
        con = make_shared<consumers_t>();
        m_outs[index] = con;
    }
    con->push_back(p_consumer);
    return true;
}

shared_ptr<consumer_t> dnnl::graph::impl::utils::pm::consumer(
        const pb_node_ptr &p_node, iport_t i_t) {
    return make_shared<consumer_t>(p_node, i_t);
}

shared_ptr<consumer_t> dnnl::graph::impl::utils::pm::producer(
        const pb_node_ptr &p_node, oport_t o_t) {
    return make_shared<producer_t>(p_node, o_t);
}

shared_ptr<in_edge_t> dnnl::graph::impl::utils::pm::in_edge(
        iport_t i_t, const pb_node_ptr &p_node, oport_t o_t) {
    auto prod = make_shared<producer_t>(p_node, o_t);
    auto edge = make_shared<in_edge_t>(i_t, prod);
    return edge;
}

decision_function dnnl::graph::impl::utils::pm::kind(
        dnnl::graph::impl::op_kind_t okind) {
    return [okind](op_t *p_op) -> bool {
        return okind == p_op->get_kind() || okind == op_kind::Wildcard;
    };
}

decision_function dnnl::graph::impl::utils::pm::one_of_kind(
        const vector<dnnl::graph::impl::op_kind_t> &okind) {
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

void pb_op::allow_internal_inputs(const std::vector<iport_t> &p_ports) {
    for (auto &port : p_ports)
        allow_internal_input(port);
}

vector<shared_ptr<unordered_set<shared_ptr<consumer_t>>>>
pb_graph::get_inner_consumers() {
    return m_inner_consumers;
}

vector<shared_ptr<producer_t>> pb_graph::get_inner_producers() {
    return m_inner_producers;
}

shared_ptr<unordered_set<shared_ptr<consumer_t>>> pb_graph::get_inner_consumer(
        iport_t i_t) {
    if (i_t < 0) { return nullptr; }
    uint64_t index = static_cast<uint64_t>(i_t);
    if (m_inner_consumers.size() <= index) { return nullptr; }
    return m_inner_consumers[index];
}

shared_ptr<producer_t> pb_graph::get_inner_producer(oport_t o_t) {
    if (o_t < 0) { return nullptr; }
    uint64_t index = static_cast<uint64_t>(o_t);
    if (m_inner_producers.size() <= index) { return nullptr; }
    return m_inner_producers[index];
}

bool pb_graph::create_input_port(
        iport_t p_port, const shared_ptr<consumer_t> &p_consumer) {
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
        m_inner_consumers[index]
                = make_shared<unordered_set<shared_ptr<consumer_t>>>();
    }
    m_inner_consumers[index]->emplace(p_consumer);
    return true;
}

bool pb_graph::create_output_port(
        oport_t p_port, shared_ptr<producer_t> p_producer) {
    if (p_port < 0) { return false; }
    uint64_t index = static_cast<uint64_t>(p_port);
    if (m_inner_producers.size() <= index) {
        m_inner_producers.resize(index + 1, nullptr);
    }
    if (m_inner_producers[index] != nullptr) return false;
    m_inner_producers[index] = move(p_producer);
    return true;
}

bool pb_graph::create_input_port(
        iport_t p_port, const pb_node_ptr &p_int_node, iport_t p_int_port) {
    return create_input_port(p_port, consumer(p_int_node, p_int_port));
}

bool pb_graph::create_output_port(
        oport_t p_port, const pb_node_ptr &p_int_node, oport_t p_int_port) {
    return create_output_port(p_port, producer(p_int_node, p_int_port));
}

bool pb_graph::connect_edges(
        const pb_node_ptr &p_node, const in_edges_t &p_in_edges) {
    if (!p_in_edges.empty()) {
        for (auto const &i : p_in_edges) {
            auto con = make_shared<consumer_t>(p_node, i->first);
            set_edge(con, i->second);
        }
    }
    return true;
}

vector<pb_node_ptr> pb_graph::get_nodes() {
    vector<pb_node_ptr> retval;
    for (auto const &i : m_nodes) {
        retval.push_back(i.get());
    }
    return retval;
}

pb_graph::pb_graph(string name) {
    m_debug_string = move(name);
}

pb_op *pb_graph::append_op(const decision_function &p_fn,
        const in_edges_t &p_in_edges, string name) {
    shared_ptr<pb_op> p_op(new pb_op(p_fn));
    p_op->set_name(move(name));
    connect_edges(p_op.get(), p_in_edges);
    m_nodes.push_back(dynamic_pointer_cast<pb_node>(p_op));
    return p_op.get();
}

pb_op *pb_graph::append_op(const decision_function &p_fn, string name) {
    return append_op(p_fn, {}, move(name));
}

pb_op *pb_graph::append_op(dnnl::graph::impl::op_kind_t p_kind,
        const in_edges_t &p_in_edges, string name) {
    return append_op(kind(p_kind), p_in_edges, move(name));
}

pb_op *pb_graph::append_op(dnnl::graph::impl::op_kind_t p_kind, string name) {
    return append_op(kind(p_kind), {}, move(name));
}

pb_op *pb_graph::append_alternation(
        const vector<dnnl::graph::impl::op_kind_t> &p_kind,
        const in_edges_t &p_in_edges, string name) {
    return append_op(one_of_kind(p_kind), p_in_edges, move(name));
}

pb_op *pb_graph::append_alternation(
        const vector<dnnl::graph::impl::op_kind_t> &p_kind, string name) {
    return append_op(one_of_kind(p_kind), {}, move(name));
}

alternation *pb_graph::append_alternation(vector<shared_ptr<pb_graph>> p_nodes,
        const in_edges_t &p_in_edges, string name) {
    shared_ptr<alternation> p_alternation(new alternation(move(p_nodes)));
    p_alternation->set_name(move(name));
    connect_edges(p_alternation.get(), p_in_edges);
    m_nodes.push_back(dynamic_pointer_cast<pb_node>(p_alternation));
    return p_alternation.get();
}

alternation *pb_graph::append_alternation(
        vector<shared_ptr<pb_graph>> p_nodes, string name) {
    return append_alternation(move(p_nodes), {}, move(name));
}

repetition *pb_graph::append_repetition(shared_ptr<pb_graph> p_node,
        port_maps p_maps, int64_t min_rep, int64_t max_rep,
        const in_edges_t &p_in_edges, string name) {
    shared_ptr<repetition> p_repetition(
            new repetition(move(p_node), move(p_maps), min_rep, max_rep));
    p_repetition->set_name(move(name));
    connect_edges(p_repetition.get(), p_in_edges);
    m_nodes.push_back(dynamic_pointer_cast<pb_node>(p_repetition));
    return p_repetition.get();
}

repetition *pb_graph::append_repetition(shared_ptr<pb_graph> p_node,
        port_maps p_maps, int64_t min_rep, int64_t max_rep, string name) {
    return append_repetition(
            move(p_node), move(p_maps), min_rep, max_rep, {}, move(name));
}

repetition *pb_graph::append_optional(shared_ptr<pb_graph> p_node,
        const in_edges_t &p_in_edges, string name) {
    shared_ptr<repetition> p_repetition(new repetition(move(p_node)));
    p_repetition->set_name(move(name));
    connect_edges(p_repetition.get(), p_in_edges);
    m_nodes.push_back(dynamic_pointer_cast<pb_node>(p_repetition));
    return p_repetition.get();
}

repetition *pb_graph::append_optional(
        shared_ptr<pb_graph> p_node, string name) {
    return append_optional(move(p_node), {}, move(name));
}

bool pb_graph::set_edge(const shared_ptr<consumer_t> &p_consumer,
        const shared_ptr<producer_t> &p_producer) {
    auto con = p_consumer->first;
    con->set_producer(p_consumer->second, p_producer);
    auto pro = p_producer->first;
    pro->add_consumer(p_producer->second, p_consumer);
    return true;
}

bool pb_graph::has_edge(const shared_ptr<consumer_t> &p_consumer,
        const shared_ptr<producer_t> &p_producer) {
    bool retval = true;
    auto con = p_consumer->first;
    auto prod1 = con->get_producer(p_consumer->second);
    retval = retval && (prod1 == p_producer);
    auto prod2 = p_producer->first;
    auto cons = prod2->get_consumers(p_producer->second);
    bool has_con_match = false;
    for (auto const &icon : *cons) {
        has_con_match = has_con_match || (icon == p_consumer);
    }
    retval = retval && has_con_match;
    return retval;
}

alternation::alternation(vector<shared_ptr<pb_graph>> p_nodes)
    : m_alternatives {move(p_nodes)} {
    m_node_kind = pb_node_kind::PB_NODE_KIND_ALTERNATION;
}

vector<pb_graph *> alternation::get_alternatives() {
    vector<pb_graph *> retval;
    for (auto const &i : m_alternatives) {
        retval.push_back(i.get());
    }
    return retval;
}

repetition::repetition(shared_ptr<pb_graph> p_node, port_maps p_maps,
        int64_t min_rep, int64_t max_rep)
    : m_body {move(p_node)}
    , m_port_maps {move(p_maps)}
    , m_min_rep {min_rep}
    , m_max_rep {max_rep} {
    m_node_kind = pb_node_kind::PB_NODE_KIND_REPETITION;
}

repetition::repetition(shared_ptr<pb_graph> p_node)
    : m_body {move(p_node)}, m_min_rep {0}, m_max_rep {2} {
    m_node_kind = pb_node_kind::PB_NODE_KIND_REPETITION;
}

pb_graph *repetition::get_body() {
    return m_body.get();
}

port_maps repetition::get_port_maps() {
    return m_port_maps;
}
