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

#ifndef UTILS_PM_PBUILDER_HPP
#define UTILS_PM_PBUILDER_HPP

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <unordered_set>

#include "interface/c_types_map.hpp"

using std::make_shared;
using std::pair;
using std::shared_ptr;
using std::string;
using std::unordered_set;
using std::vector;

namespace dnnl {
namespace graph {
namespace impl {
namespace utils {
namespace pm {
class pb_op;
class pb_node;
class pb_graph;
// Helper types
using iport_t = int64_t;
using oport_t = int64_t;
const iport_t INVALID_IN_PORT = -1;
const oport_t INVALID_OUT_PORT = -1;
const oport_t ANY_OUT_PORT = -2;
using iport_pair = pair<iport_t, iport_t>;
using pb_node_ptr = pb_node *;
using producer_t = pair<pb_node_ptr, oport_t>;
using consumer_t = pair<pb_node_ptr, iport_t>;
using consumers_t = vector<shared_ptr<consumer_t>>;
using in_edge_t = pair<iport_t, shared_ptr<producer_t>>;
using in_edges_t = vector<shared_ptr<in_edge_t>>;
using port_map = pair<oport_t, iport_t>;
using port_maps = vector<port_map>;
using pattern_pair = std::pair<op_t *, impl::utils::pm::pb_op *>;

//
// Part 1:
// Structures for representing basic topological patterns
// and attribute patterns
//

// Represents any backend defined function that takes a pointer to dnnl graph op
// and check some attribute(op type, attributes, input shapes ...)
using decision_function = std::function<bool(op_t *)>;

enum class pb_node_kind {
    PB_NODE_KIND_OP,
    PB_NODE_KIND_GRAPH,
    PB_NODE_KIND_ALTERNATION,
    PB_NODE_KIND_REPETITION,
};

// Base class for pattern graph with input and output ports (placeholders)
// Only implements traversal methods and setting commutative input pairs.
// Suitable for representing topological patterns
class pb_node {
public:
    virtual ~pb_node() = default;
    // API for traversing
    shared_ptr<producer_t> get_producer(iport_t p_port);
    shared_ptr<consumers_t> get_consumers(oport_t p_port);

    vector<pair<iport_t, producer_t>> get_inputs();
    vector<pair<oport_t, consumers_t>> get_outputs();

    iport_pair get_commutative_pair();
    size_t get_num_decision_functions();
    decision_function get_decision_function(size_t index);
    pb_node_kind get_node_kind() { return m_node_kind; };
    virtual string get_name() { return m_debug_string; };
    virtual void set_name(string name) { m_debug_string = name; };

protected:
    friend class pb_graph;
    pb_node() = default;
    bool set_producer(iport_t p_port, shared_ptr<producer_t> p_producer);
    bool set_consumers(oport_t p_port, shared_ptr<consumers_t> p_consumers);
    bool add_consumer(oport_t p_port, const shared_ptr<consumer_t> &p_consumer);
    vector<shared_ptr<producer_t>> m_ins;
    vector<shared_ptr<consumers_t>> m_outs;
    iport_pair m_commutative_pair {INVALID_IN_PORT, INVALID_IN_PORT};
    vector<decision_function> m_decision_functions;
    string m_debug_string = "PB_NODE: ";
    pb_node_kind m_node_kind = pb_node_kind::PB_NODE_KIND_GRAPH;
};

shared_ptr<consumer_t> consumer(const pb_node_ptr &p_node, iport_t i_t);

shared_ptr<consumer_t> producer(const pb_node_ptr &p_node, oport_t o_t);

shared_ptr<in_edge_t> in_edge(
        iport_t i_t, const pb_node_ptr &p_node, oport_t o_t);

// Helper function for op kind check
decision_function kind(dnnl::graph::impl::op_kind_t okind);
decision_function one_of_kind(
        const vector<dnnl::graph::impl::op_kind_t> &okind);

// pb_op represents a single dnnl graph  op (and future sub-class) operation
// No public constructor
// Always created by a pb_graph
// pb_op has type and attributes
// Type and attribute contraint checkers are registered in pb_op
// Extends "pb_node" to enable attribute matching including op type check.
class pb_op : public pb_node {
public:
    pb_op() = delete;
    // like is_commutative by callback
    bool append_decision_function(const decision_function &p_fn);
    bool set_commutative_pair(iport_pair p_port_pair);

    // mark pattern node can only match ops with specific attrs
    template <typename T>
    void set_attr(std::string attr_name, T value) {
        this->append_decision_function([=](op_t *op) -> bool {
            if (op->has_attr(attr_name)) {
                return op->get_attr<T>(attr_name) == value;
            }
            return false;
        });
    }

    // For overriding default side output control
    void allow_external_output(oport_t);
    // For overriding default unmatched input control
    void allow_internal_input(iport_t);
    void allow_internal_inputs(const std::vector<iport_t> &p_ports);

    unordered_set<oport_t> get_allowed_external_outputs() {
        return m_external_outputs;
    };
    unordered_set<oport_t> get_allowed_internal_inputs() {
        return m_internal_inputs;
    };

protected:
    friend class pb_graph;
    pb_op(const decision_function &p_fn);
    // For overriding default side output control
    unordered_set<oport_t> m_external_outputs;
    // For overriding default unmatched input control
    unordered_set<iport_t> m_internal_inputs;
};

//
// Part 2:
// Strutures for extended patterns
// API may change
//
class alternation : public pb_node {
public:
    alternation() = delete;
    vector<pb_graph *> get_alternatives();

protected:
    friend class pb_graph;
    alternation(vector<shared_ptr<pb_graph>> p_nodes);
    vector<shared_ptr<pb_graph>> m_alternatives;
};

class repetition : public pb_node {
public:
    repetition() = delete;
    pb_graph *get_body();
    port_maps get_port_maps();
    int64_t get_min_rep() const { return m_min_rep; }
    int64_t get_max_rep() const { return m_max_rep; }

protected:
    friend class pb_graph;
    // Represents p_node repeated [min_rep, max_rep) times with p_map for
    // output to input binding
    // [n, n+1) means exactly n repetitions
    // [0, n+1) means at most n repetitions
    // [n, INT64_MAX) means at least n repetitions
    repetition(shared_ptr<pb_graph> p_node, port_maps p_maps, int64_t min_rep,
            int64_t max_rep);
    // Usage case for Optional does not need a port map
    repetition(shared_ptr<pb_graph> p_node);
    shared_ptr<pb_graph> m_body;
    port_maps m_port_maps;
    int64_t m_min_rep;
    int64_t m_max_rep;
};

// "pb_graph" represents a group of pb_ops and also serves as a pb_node anywhere
// And provides a way to limit interface by limiting ports(placeholders)
// to outside of pb_graph.
// Nested/Hierarchical pb_nodes are useful for expressing patterns beyond fixed
// pb_graph. Regular expression like extension may works on a unit larger than
// a single pb_node.
// So a concept that represent grouping is going to be useful.
// pb_graph defines a way to forward input/output of the group
// to input/output of individual pb_nodes.
// For example, pb_graph "G" below wraps two connected pb_nodes "MUL" and "ADD"
// Collectively, G defines three inputs and one output. The three inputs of "G"
// are mapped to (pb_graph inner) inputs of "MUL" and "ADD"
// The single output of "G" maps to the single output of "ADD"
// Now, this "G" can used as part of a bigger pattern by connecting through
// the three inputs and one output just defined.
// Also, "G" can declare output ports which provides a way for backends to
// declare which outputs can be produced by compiled kernels for the pattern.
// Declaring output ports is important for exposing backend's ability to handle
// side outputs.
//    ----------------------
//    |   ------   -----   |
// 0- | 0-| MUL|---|ADD|   |
//    | 1-|    |  0|   |-0 |
// 1- |   ------   |   |   |-0
//    |          1-|   |   |
// 2- |            -----   |
//    ----------------------
//          pb_graph "G"
//
// G:IN0->MUL:IN0, G:IN1->MUL:IN1, G:IN2->ADD:IN1
// G:OUT0->ADD:OUT0
// G:OUTPUT PORTS = {OUT0}

class pb_graph : public pb_node {
public:
    pb_graph(string name = "");
    // Restrict "pb_op" create to a pb_graph to avoid dangling "pb_op"s
    pb_op *append_op(const decision_function &type_checker,
            const in_edges_t &p_in_edges, string name = "");
    pb_op *append_op(const decision_function &type_checker, string name = "");
    pb_op *append_op(dnnl::graph::impl::op_kind_t p_kind,
            const in_edges_t &p_in_edges, string name = "");
    pb_op *append_op(dnnl::graph::impl::op_kind_t p_kind, string name = "");
    pb_op *append_alternation(
            const vector<dnnl::graph::impl::op_kind_t> &p_kind,
            const in_edges_t &p_in_edges, string name = "");
    pb_op *append_alternation(
            const vector<dnnl::graph::impl::op_kind_t> &p_kind,
            string name = "");

    alternation *append_alternation(vector<shared_ptr<pb_graph>> p_nodes,
            const in_edges_t &p_in_edges, string name = "");
    alternation *append_alternation(
            vector<shared_ptr<pb_graph>> p_nodes, string name = "");
    repetition *append_repetition(shared_ptr<pb_graph> p_node, port_maps p_maps,
            int64_t min_rep, int64_t max_rep, const in_edges_t &p_in_edges,
            string name = "");
    repetition *append_repetition(shared_ptr<pb_graph> p_node, port_maps p_maps,
            int64_t min_rep, int64_t max_rep, string name = "");
    repetition *append_optional(shared_ptr<pb_graph> p_node,
            const in_edges_t &p_in_edges, string name = "");
    repetition *append_optional(shared_ptr<pb_graph> p_node, string name = "");
    bool set_edge(
            const shared_ptr<consumer_t> &, const shared_ptr<producer_t> &);
    bool has_edge(
            const shared_ptr<consumer_t> &, const shared_ptr<producer_t> &);
    bool connect_edges(const pb_node_ptr &p_node, const in_edges_t &p_in_edges);
    vector<shared_ptr<unordered_set<shared_ptr<consumer_t>>>>
    get_inner_consumers();
    vector<shared_ptr<producer_t>> get_inner_producers();
    shared_ptr<unordered_set<shared_ptr<consumer_t>>> get_inner_consumer(
            iport_t);
    shared_ptr<producer_t> get_inner_producer(oport_t);
    bool create_input_port(iport_t, const shared_ptr<consumer_t> &);
    bool create_output_port(oport_t, shared_ptr<producer_t>);
    bool create_input_port(iport_t, const pb_node_ptr &, iport_t);
    bool create_output_port(oport_t, const pb_node_ptr &, oport_t);

    vector<pb_node *> get_nodes();

protected:
    // Reference to all internal pb_nodes
    vector<shared_ptr<pb_node>> m_nodes;
    unordered_set<oport_t> m_output_ports;
    vector<shared_ptr<unordered_set<shared_ptr<consumer_t>>>>
            m_inner_consumers {nullptr};
    vector<shared_ptr<producer_t>> m_inner_producers {nullptr};
};

} // namespace pm
} // namespace utils
} // namespace impl
} // namespace graph
} // namespace dnnl
#endif
