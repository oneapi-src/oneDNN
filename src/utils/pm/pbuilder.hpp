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

#ifndef UTILS_PM_PBUILDER_HPP
#define UTILS_PM_PBUILDER_HPP

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <unordered_set>

#include "interface/c_types_map.hpp"

namespace dnnl {
namespace graph {
namespace impl {
namespace utils {
namespace pm {
class pb_op;
class pb_node;
class pb_graph_t;
// Helper types
using iport_t = int64_t;
using oport_t = int64_t;
using producer_t = std::pair<pb_node *, oport_t>;
using consumer_t = std::pair<pb_node *, iport_t>;
using consumers_t = std::vector<std::shared_ptr<consumer_t>>;
using in_edge_t = std::pair<iport_t, std::shared_ptr<producer_t>>;
using in_edges_t = std::vector<std::shared_ptr<in_edge_t>>;
using port_map = std::pair<oport_t, iport_t>;
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
    std::shared_ptr<producer_t> get_producer(iport_t p_port);
    std::shared_ptr<consumers_t> get_consumers(oport_t p_port);

    std::vector<std::pair<iport_t, producer_t>> get_inputs();
    std::vector<std::pair<oport_t, consumers_t>> get_outputs();

    size_t get_num_decision_functions();
    decision_function get_decision_function(size_t index);
    pb_node_kind get_node_kind() { return m_node_kind; };
    virtual std::string get_name() { return m_debug_string; };
    virtual void set_name(std::string &&name) {
        m_debug_string = std::move(name);
    };

protected:
    friend class pb_graph_t;
    pb_node() = default;
    bool set_producer(iport_t p_port, std::shared_ptr<producer_t> p_producer);
    bool set_consumers(
            oport_t p_port, std::shared_ptr<consumers_t> p_consumers);
    bool add_consumer(
            oport_t p_port, const std::shared_ptr<consumer_t> &p_consumer);
    std::vector<std::shared_ptr<producer_t>> m_ins;
    std::vector<std::shared_ptr<consumers_t>> m_outs;
    std::vector<decision_function> m_decision_functions;
    std::string m_debug_string = "PB_NODE: ";
    pb_node_kind m_node_kind = pb_node_kind::PB_NODE_KIND_GRAPH;
};

std::shared_ptr<consumer_t> consumer(pb_node *p_node, iport_t i_t);

std::shared_ptr<consumer_t> producer(pb_node *p_node, oport_t o_t);

std::shared_ptr<in_edge_t> in_edge(iport_t i_t, pb_node *p_node, oport_t o_t);

// Helper function for op kind check
decision_function kind(dnnl::graph::impl::op_kind_t okind);
decision_function one_of_kind(
        const std::vector<dnnl::graph::impl::op_kind_t> &okind);

// pb_op represents a single dnnl graph  op (and future sub-class) operation
// No public constructor
// Always created by a pb_graph_t
// pb_op has type and attributes
// Type and attribute contraint checkers are registered in pb_op
// Extends "pb_node" to enable attribute matching including op type check.
class pb_op : public pb_node {
public:
    pb_op() = delete;
    // like is_commutative by callback
    bool append_decision_function(const decision_function &p_fn);

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

    std::unordered_set<oport_t> get_allowed_external_outputs() {
        return m_external_outputs;
    };
    std::unordered_set<oport_t> get_allowed_internal_inputs() {
        return m_internal_inputs;
    };

protected:
    friend class pb_graph_t;
    pb_op(const decision_function &p_fn);
    // For overriding default side output control
    std::unordered_set<oport_t> m_external_outputs;
    // For overriding default unmatched input control
    std::unordered_set<iport_t> m_internal_inputs;
};

//
// Part 2:
// Strutures for extended patterns
// API may change
//
class alternation_t : public pb_node {
public:
    alternation_t() = delete;
    std::vector<pb_graph_t *> get_alternatives();

protected:
    friend class pb_graph_t;
    alternation_t(std::vector<std::shared_ptr<pb_graph_t>> p_nodes);
    std::vector<std::shared_ptr<pb_graph_t>> m_alternatives;
};

class repetition_t : public pb_node {
public:
    repetition_t() = delete;
    pb_graph_t *get_body();
    port_map get_port_map(); // only support single port binding
    int64_t get_min_rep() const { return m_min_rep; }
    int64_t get_max_rep() const { return m_max_rep; }

protected:
    friend class pb_graph_t;
    // Represents p_node repeated [min_rep, max_rep) times with p_map for
    // output to input binding
    // [n, n+1) means exactly n repetitions
    // [0, n+1) means at most n repetitions
    // [n, INT64_MAX) means at least n repetitions
    repetition_t(std::shared_ptr<pb_graph_t> p_node, port_map p_map,
            int64_t min_rep, int64_t max_rep);
    // Usage case for Optional does not need a port map
    repetition_t(std::shared_ptr<pb_graph_t> p_node);
    std::shared_ptr<pb_graph_t> m_body;
    port_map m_port_map;
    int64_t m_min_rep;
    int64_t m_max_rep;
};

// "pb_graph_t" represents a group of pb_ops and also serves as a pb_node
// anywhere And provides a way to limit interface by limiting ports
// (placeholders) to outside of pb_graph_t.
// Nested/Hierarchical pb_nodes are useful for expressing patterns beyond fixed
// pb_graph_t. Regular expression like extension may works on a unit larger
// than a single pb_node.
// So a concept that represent grouping is going to be useful.
// pb_graph_t defines a way to forward input/output of the group
// to input/output of individual pb_nodes.
// For example, pb_graph_t "G" below wraps two connected pb_nodes "MUL" and
// "ADD" Collectively, G defines three inputs and one output. The three inputs
// of "G" are mapped to (pb_graph_t inner) inputs of "MUL" and "ADD"
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
//          pb_graph_t "G"
//
// G:IN0->MUL:IN0, G:IN1->MUL:IN1, G:IN2->ADD:IN1
// G:OUT0->ADD:OUT0
// G:OUTPUT PORTS = {OUT0}

class pb_graph_t : public pb_node {
public:
    pb_graph_t(std::string name = "");
    // Restrict "pb_op" create to a pb_graph_t to avoid dangling "pb_op"s
    pb_op *append_op(const decision_function &type_checker,
            const in_edges_t &p_in_edges, std::string name = "");
    pb_op *append_op(
            const decision_function &type_checker, std::string name = "");
    pb_op *append_op(dnnl::graph::impl::op_kind_t p_kind,
            const in_edges_t &p_in_edges, std::string name = "");
    pb_op *append_op(
            dnnl::graph::impl::op_kind_t p_kind, std::string name = "");
    pb_op *append_alternation(
            const std::vector<dnnl::graph::impl::op_kind_t> &p_kind,
            const in_edges_t &p_in_edges, std::string name = "");
    pb_op *append_alternation(
            const std::vector<dnnl::graph::impl::op_kind_t> &p_kind,
            std::string name = "");

    alternation_t *append_alternation(
            std::vector<std::shared_ptr<pb_graph_t>> p_nodes,
            const in_edges_t &p_in_edges, std::string name = "");
    alternation_t *append_alternation(
            std::vector<std::shared_ptr<pb_graph_t>> p_nodes,
            std::string name = "");
    repetition_t *append_repetition(std::shared_ptr<pb_graph_t> p_node,
            port_map p_map, int64_t min_rep, int64_t max_rep,
            const in_edges_t &p_in_edges, std::string name = "");
    repetition_t *append_repetition(std::shared_ptr<pb_graph_t> p_node,
            port_map p_map, int64_t min_rep, int64_t max_rep,
            std::string name = "");
    repetition_t *append_optional(std::shared_ptr<pb_graph_t> p_node,
            const in_edges_t &p_in_edges, std::string name = "");
    repetition_t *append_optional(
            std::shared_ptr<pb_graph_t> p_node, std::string name = "");
    bool set_edge(const std::shared_ptr<consumer_t> &,
            const std::shared_ptr<producer_t> &);
    bool has_edge(const std::shared_ptr<consumer_t> &,
            const std::shared_ptr<producer_t> &);
    bool connect_edges(pb_node *p_node, const in_edges_t &p_in_edges);
    std::vector<std::pair<oport_t, consumers_t>> get_inner_consumers();
    std::vector<std::pair<iport_t, producer_t>> get_inner_producers();
    std::shared_ptr<consumers_t> get_inner_consumer(iport_t);
    std::shared_ptr<producer_t> get_inner_producer(oport_t);
    bool create_input_port(iport_t, const std::shared_ptr<consumer_t> &);
    bool create_output_port(oport_t, std::shared_ptr<producer_t>);
    bool create_input_port(iport_t, pb_node *, iport_t);
    bool create_output_port(oport_t, pb_node *, oport_t);

    std::vector<pb_node *> get_nodes();

protected:
    // Reference to all internal pb_nodes
    std::vector<std::shared_ptr<pb_node>> m_nodes;
    std::unordered_set<oport_t> m_output_ports;
    std::vector<std::shared_ptr<consumers_t>> m_inner_consumers {nullptr};
    std::vector<std::shared_ptr<producer_t>> m_inner_producers {nullptr};
};

} // namespace pm
} // namespace utils
} // namespace impl
} // namespace graph
} // namespace dnnl
#endif
