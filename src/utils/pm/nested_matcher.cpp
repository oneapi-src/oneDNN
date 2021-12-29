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

#include <algorithm>
#include <iterator>

#include "utils/pm/nested_matcher.hpp"

namespace dnnl {
namespace graph {
namespace impl {
namespace utils {
namespace pm {

using std::find;

binding_t::binding_t(node_bind_kind p_kind, op_ptr p_op, int64_t p_op_port,
        pb_node_ptr p_node, int64_t p_port, int64_t p_idx)
    : bind_op {p_op}
    , bind_node {p_node}
    , bind_kind {p_kind}
    , bind_port {p_port}
    , bind_op_port {p_op_port}
    , bind_port_user_idx {p_idx} {}

node_tracker_t::node_tracker_t(const binding_t &bind_arg)
    : m_node {bind_arg.bind_node}, m_op {bind_arg.bind_op} {
    vector<pair<iport_t, producer_t>> vinputs = m_node->get_inputs();
    iport_pair ipair = m_node->get_commutative_pair();
    bool has_commutative_input = (ipair.first != -1) && (ipair.second != -1);
    if (has_commutative_input) {
        deque<iport_t> inputs;
        for (auto const &i : vinputs) {
            inputs.push_back(i.first);
        }
        while (!inputs.empty()) {
            iport_t port = inputs.front();
            inputs.pop_front();
            input_match_task_t itask;
            if (port == ipair.first) {
                itask.port = ipair.first;
                itask.additional_port = ipair.second;
                auto f = find(inputs.begin(), inputs.end(), ipair.second);
                if (f == inputs.end()) {
                    itask.match_kind
                            = INPUT_MATCH_KIND_COMMUTATIVE_ONE_CONSTRAINT;
                } else {
                    inputs.erase(f);
                    itask.match_kind
                            = INPUT_MATCH_KIND_COMMUTATIVE_TWO_CONSTRAINT;
                }
                src_to_visit.push_back(itask);
            } else {
                itask.match_kind = INPUT_MATCH_KIND_NORMAL;
                itask.port = port;
                itask.additional_port = -1;
                src_to_visit.push_back(itask);
            }
        }
    } else {
        for (auto const &i : vinputs) {
            input_match_task_t itask;
            itask.match_kind = INPUT_MATCH_KIND_NORMAL;
            itask.port = i.first;
            itask.additional_port = -1;
            src_to_visit.push_back(itask);
        }
    }

    vector<pair<oport_t, consumers_t>> outputs = m_node->get_outputs();
    for (auto const &j : outputs) {
        output_match_task_t otask;
        otask.port = j.first;
        otask.num_consumers = static_cast<int64_t>(j.second.size());
        dst_to_visit.push_back(otask);
    }

    // Setup unhandled inputs/outputs of op.
    size_t op_num_inputs = m_op->num_inputs();
    for (size_t i = 0; i < op_num_inputs; i++) {
        op_unhandled_input.push_back(true);
    }
    size_t op_num_outputs = m_op->num_outputs();
    for (size_t i = 0; i < op_num_outputs; i++) {
        size_t op_num_output_users = m_op->num_output_consumers(i);
        vector<bool> v;
        v.clear();
        for (size_t j = 0; j < op_num_output_users; j++) {
            v.push_back(true);
        }
        op_unhandled_output.emplace_back(v);
    }
}
//
// Part 1.
// match functions for pb_op's
//

bool match_node_attributes(op_ptr op, const pb_node_ptr &node) {
    size_t n_func = node->get_num_decision_functions();
    for (size_t i = 0; i < n_func; i++) {
        if (!(node->get_decision_function(i)(op))) { return false; }
    }
    return true;
}

static node_tracker_ptr find_node_tracker(op_ptr op, match_context_ptr ctx) {
    match_context_ptr curr_ctx = ctx;
    while (curr_ctx != nullptr) {
        auto it = curr_ctx->node_tracker_map.find(op);
        if (it != curr_ctx->node_tracker_map.end()) { return it->second; }
        curr_ctx = curr_ctx->get_parent_context();
    }
    return nullptr;
}

bool register_node_tracker(const binding_t &bind_arg, match_context_ptr ctx) {
    // Workflow
    // Check if op bind_arg.bind_op is already matched by some other pattern.
    // Find op in node_tracker_map up the context chain.
    // If op is found.
    //     If op is blacklisted return false.
    //     If op has a node tracker, check if it matches the bind_node
    //     otherwise return false.
    // Else, create a node tracker, register and put it in visit queue.
    // Update unhandled edges of op according to binding.
    if (bind_arg.bind_op->get_partition() != nullptr
            || bind_arg.bind_op->has_attr("matched_pattern")) {
        return false;
    }
    node_tracker_ptr n_tracker = find_node_tracker(bind_arg.bind_op, ctx);
    if (n_tracker == nullptr) {
        n_tracker = make_shared<node_tracker_t>(bind_arg);
        ctx->ops_to_visit.push_back(bind_arg.bind_op);
        ctx->node_tracker_map.insert(
                pair<op_ptr, node_tracker_ptr> {bind_arg.bind_op, n_tracker});
    } else {
        if (n_tracker->get_node() != bind_arg.bind_node) { return false; }
    }
    // Update unhandled edges of op according to binding.
    switch (bind_arg.bind_kind) {
        case BIND_IN: {
            int64_t task_to_remove = -1;
            // pre mark op port as handled
            // Addition checking will happen below and return false
            // if there is any violation
            n_tracker->op_unhandled_input[static_cast<size_t>(
                    bind_arg.bind_op_port)]
                    = false;
            for (size_t i = 0; i < n_tracker->src_to_visit.size(); i++) {
                input_match_task_t &itask = n_tracker->src_to_visit[i];
                switch (itask.match_kind) {
                    case INPUT_MATCH_KIND_NORMAL:
                        // Check if task is for this bind port
                        if (itask.port == bind_arg.bind_port) {
                            // op port has to match the bind port
                            if (bind_arg.bind_op_port != bind_arg.bind_port) {
                                return false;
                            } else {
                                // mark to remove task
                                task_to_remove = static_cast<int64_t>(i);
                            }
                        }
                        break;
                    case INPUT_MATCH_KIND_COMMUTATIVE_ONE_CONSTRAINT:
                        // Check if task is for this bind port by
                        // checking bind port against the two commutatitve
                        // ports of the task.
                        if (itask.port == bind_arg.bind_port
                                || itask.additional_port
                                        == bind_arg.bind_port) {
                            // op port has to match one of the commutative
                            // ports
                            if (bind_arg.bind_op_port == itask.port
                                    || bind_arg.bind_op_port
                                            == itask.additional_port) {
                                // mark to remove task
                                task_to_remove = static_cast<int64_t>(i);
                            } else {
                                return false;
                            }
                        }
                        break;
                    case INPUT_MATCH_KIND_COMMUTATIVE_TWO_CONSTRAINT:
                        if (itask.port == bind_arg.bind_port
                                || itask.additional_port
                                        == bind_arg.bind_port) {
                            if (itask.port == bind_arg.bind_op_port) {
                                if (itask.port == bind_arg.bind_port) {
                                    itask.port = itask.additional_port;
                                } else {
                                    iport_t temp = itask.port;
                                    itask.port = itask.additional_port;
                                    itask.additional_port = temp;
                                }
                                itask.match_kind
                                        = INPUT_MATCH_KIND_COMMUTATIVE_PINNED;
                            } else if (itask.additional_port
                                    == bind_arg.bind_op_port) {
                                if (itask.additional_port
                                        == bind_arg.bind_port) {
                                    itask.additional_port = itask.port;
                                }
                                itask.match_kind
                                        = INPUT_MATCH_KIND_COMMUTATIVE_PINNED;
                            } else {
                                return false;
                            }
                        }
                        break;
                    case INPUT_MATCH_KIND_COMMUTATIVE_PINNED:
                        if (itask.port == bind_arg.bind_port) {
                            if (bind_arg.bind_op_port
                                    == itask.additional_port) {
                                task_to_remove = static_cast<int64_t>(i);
                            } else {
                                return false;
                            }
                        }
                        break;
                    default: break;
                }
            }
            // remove task if needed.
            if (task_to_remove != -1) {
                n_tracker->src_to_visit.erase(
                        n_tracker->src_to_visit.begin() + task_to_remove);
            }
        } break;
        case BIND_OUT: {
            int64_t task_to_remove = -1;
            // pre clear unhandled output
            vector<bool> v = n_tracker->op_unhandled_output[static_cast<size_t>(
                    bind_arg.bind_op_port)];
            v[static_cast<size_t>(bind_arg.bind_port_user_idx)] = false;
            n_tracker->op_unhandled_output[static_cast<size_t>(
                    bind_arg.bind_op_port)]
                    = v;
            for (size_t i = 0; i < n_tracker->dst_to_visit.size(); i++) {
                output_match_task_t &otask = n_tracker->dst_to_visit[i];
                // If match task is for this bind_port
                if (otask.port == bind_arg.bind_port) {
                    if (bind_arg.bind_op_port != bind_arg.bind_port) {
                        return false;
                    }
                    otask.num_consumers -= 1;
                    if (otask.num_consumers < 1) {
                        task_to_remove = static_cast<int64_t>(i);
                    }
                    break;
                }
            }
            if (task_to_remove != -1) {
                n_tracker->dst_to_visit.erase(
                        n_tracker->dst_to_visit.begin() + task_to_remove);
            }
        } break;
        default: break;
    }
    return true;
}

// For a consumer op o and input port in_offset.
// return the input value's index that matches this consumer
static int64_t get_output_consumer_index(op_ptr op, size_t in_offset) {
    // Get input value
    auto in_value = op->get_input_value(in_offset);
    // Get value consumers
    auto in_value_cons = in_value->get_consumers();
    value_t::consumer_t expected_con {*op, in_offset};
    // Find current op at pattern in value consumers.
    // Would fail if value is from different output port
    auto it = find(in_value_cons.begin(), in_value_cons.end(), expected_con);
    if (it == in_value_cons.end()) { return -1; }
    return distance(in_value_cons.begin(), it);
}

static bool bind_node_input(op_ptr op, int64_t in_offset, op_ptr prod_op,
        pb_node_ptr prod_node, oport_t out_offset, match_context_ptr ctx) {
    int64_t idx = get_output_consumer_index(op, static_cast<size_t>(in_offset));
    if (idx == -1) { return false; }
    auto val = op->get_input_value(static_cast<size_t>(in_offset));
    auto vals = prod_op->get_output_values();
    auto it = find(vals.begin(), vals.end(), val);
    int64_t prod_op_port = distance(vals.begin(), it);
    binding_t b {BIND_OUT, prod_op, prod_op_port, prod_node, out_offset, idx};
    ctx->node_tracker_map[op]
            ->op_unhandled_input[static_cast<size_t>(in_offset)]
            = false;
    if (!register_node_tracker(b, ctx)) { return false; }
    return true;
}

// Things needed in the loop body
// 1. current op
// 2. ctx
//       get node_tracker_ptr - for updating unhandled ops
//       op matched node (pb_op) can get it from ctx and op
// 3. input_match_task (matched node in port and alt in port)
// 4. producer_t prod and alt_prod (if input is commutative)
static bool match_input(op_ptr op, match_context_ptr ctx,
        input_match_task_t itask, const shared_ptr<producer_t> &prod,
        const shared_ptr<producer_t> &alt_prod) {
    node_tracker_ptr n_tracker = ctx->node_tracker_map[op];
    auto o_num_inputs = op->num_inputs();
    if (itask.port >= o_num_inputs) {
        if (itask.match_kind == INPUT_MATCH_KIND_NORMAL
                && prod->first->get_node_kind()
                        == pb_node_kind::PB_NODE_KIND_REPETITION) {
            return true;
        }
        return false;
    }
    // Handle commutative inputs with look ahead
    switch (itask.match_kind) {
        case INPUT_MATCH_KIND_NORMAL: {
            pb_node_ptr prod_node = prod->first;
            int64_t idx = get_output_consumer_index(
                    op, static_cast<size_t>(itask.port));
            if (idx == -1) { return false; }
            auto val = op->get_input_value(static_cast<size_t>(itask.port));
            op_ptr prod_op = &(val->get_producer());
            auto vals = prod_op->get_output_values();
            int64_t prod_op_port = distance(
                    vals.begin(), find(vals.begin(), vals.end(), val));
            binding_t b {BIND_OUT, prod_op, prod_op_port, prod_node,
                    prod->second, idx};
            n_tracker->op_unhandled_input[static_cast<size_t>(itask.port)]
                    = false;
            if (prod_node->get_node_kind() == pb_node_kind::PB_NODE_KIND_OP) {
                if (!register_node_tracker(b, ctx)) { return false; }
            } else {
                if (!resolve_node(b, ctx)) { return false; }
            }
        } break;
        case INPUT_MATCH_KIND_COMMUTATIVE_ONE_CONSTRAINT: {
            pb_node_ptr prod_node = prod->first;
            if (prod_node->get_node_kind() != pb_node_kind::PB_NODE_KIND_OP) {
                return false;
            }
            // Need to check target op attribute for matching.
            // Try producers of both commutative ports for matching.
            op_ptr prod_op
                    = &(op->get_input_value(static_cast<size_t>(itask.port))
                                    ->get_producer());
            if (match_node_attributes(prod_op, prod_node)) {
                if (!bind_node_input(op, itask.port, prod_op, prod_node,
                            prod->second, ctx)) {
                    return false;
                }
            } else {
                if (o_num_inputs <= itask.additional_port) { return false; }
                op_ptr alt_prod_op = &(
                        op->get_input_value(
                                  static_cast<size_t>(itask.additional_port))
                                ->get_producer());
                if (!match_node_attributes(alt_prod_op, prod_node)) {
                    return false;
                }
                if (!bind_node_input(op, itask.additional_port, alt_prod_op,
                            prod_node, prod->second, ctx)) {
                    return false;
                }
            }
        } break;
        case INPUT_MATCH_KIND_COMMUTATIVE_TWO_CONSTRAINT: {
            op_ptr prod_op
                    = &(op->get_input_value(static_cast<size_t>(itask.port))
                                    ->get_producer());
            if (o_num_inputs <= itask.additional_port) { return false; }
            op_ptr alt_prod_op
                    = &(op->get_input_value(
                                  static_cast<size_t>(itask.additional_port))
                                    ->get_producer());
            // One of the constraints may have been matched.
            // Check op unhandled inputs and see if one op input is matched,
            // then the other op input is matched with the other producer node.
            // Need decision functions from producer nodes of both ports
            // if none of the op inputs are handled.
            pb_node_ptr prod_node = prod->first;
            if (prod_node->get_node_kind() != pb_node_kind::PB_NODE_KIND_OP) {
                return false;
            }
            pb_node_ptr alt_prod_node = alt_prod->first;
            if (alt_prod_node->get_node_kind()
                    != pb_node_kind::PB_NODE_KIND_OP) {
                return false;
            }
            if (n_tracker->op_unhandled_input[static_cast<size_t>(itask.port)]
                    == false) {
                node_tracker_ptr nt = ctx->node_tracker_map[prod_op];
                if (n_tracker->get_node() == prod_node) {
                    // match alt_prod_node with alt_prod_op
                    if (!bind_node_input(op, itask.additional_port, alt_prod_op,
                                alt_prod_node, alt_prod->second, ctx)) {
                        return false;
                    }
                } else {
                    // match prod_node with alt_prod_op
                    if (!bind_node_input(op, itask.additional_port, alt_prod_op,
                                prod_node, prod->second, ctx)) {
                        return false;
                    }
                }
            } else if (n_tracker->op_unhandled_input[static_cast<size_t>(
                               itask.additional_port)]
                    == false) {
                node_tracker_ptr n_tracker = ctx->node_tracker_map[alt_prod_op];
                if (n_tracker->get_node() == prod_node) {
                    // match alt_prod_node with prod_op
                    if (!bind_node_input(op, itask.port, prod_op, alt_prod_node,
                                alt_prod->second, ctx)) {
                        return false;
                    }
                } else {
                    // match prod_node with prod_op
                    if (!bind_node_input(op, itask.port, prod_op, prod_node,
                                prod->second, ctx)) {
                        return false;
                    }
                }
            } else {
                if (match_node_attributes(prod_op, prod_node)
                        && match_node_attributes(alt_prod_op, alt_prod_node)) {
                    if (!bind_node_input(op, itask.port, prod_op, prod_node,
                                prod->second, ctx)) {
                        return false;
                    }
                    if (!bind_node_input(op, itask.additional_port, alt_prod_op,
                                alt_prod_node, alt_prod->second, ctx)) {
                        return false;
                    }
                } else if (match_node_attributes(alt_prod_op, prod_node)
                        && match_node_attributes(prod_op, alt_prod_node)) {
                    if (!bind_node_input(op, itask.additional_port, alt_prod_op,
                                prod_node, prod->second, ctx)) {
                        return false;
                    }
                    if (!bind_node_input(op, itask.port, prod_op, alt_prod_node,
                                alt_prod->second, ctx)) {
                        return false;
                    }
                } else {
                    return false;
                }
            }
        } break;
        case INPUT_MATCH_KIND_COMMUTATIVE_PINNED: {
            op_ptr prod_op
                    = &(op->get_input_value(static_cast<size_t>(itask.port))
                                    ->get_producer());
            if (!bind_node_input(op, itask.port, prod_op, alt_prod->first,
                        alt_prod->second, ctx)) {
                return false;
            }
        } break;
        default: break;
    }
    return true;
}
bool match_node_inputs(op_ptr op, match_context_ptr ctx) {
    node_tracker_ptr n_tracker = ctx->node_tracker_map[op];
    pb_node_ptr node = n_tracker->get_node();
    if (node->get_node_kind() != pb_node_kind::PB_NODE_KIND_OP) {
        return false;
    }
    auto inputs = node->get_inputs();
    if (inputs.empty()) { return true; }

    while (!n_tracker->src_to_visit.empty()) {
        input_match_task_t itask = n_tracker->src_to_visit.front();
        n_tracker->src_to_visit.pop_front();
        iport_t node_port = itask.port;
        iport_t node_additional_port = itask.additional_port;
        if (!match_input(op, ctx, itask, node->get_producer(node_port),
                    node_additional_port < 0
                            ? nullptr
                            : node->get_producer(node_additional_port))) {
            return false;
        }
    }
    return true;
}

// Things needed in the loop body
// 1. current op
// 2. ctx
//       get node_tracker_ptr - for updating unhandled ops
//       op matched node (pb_op) can get it from ctx and op
// 3. output_match_task (matched node out port and prebinded consumer idx)
// 4. consumers_t
//
static bool match_output(op_ptr op, match_context_ptr ctx,
        output_match_task_t otask, const shared_ptr<consumers_t> &cons) {
    if (otask.port >= op->num_outputs()
            || op->get_output_value(static_cast<size_t>(otask.port))
                       ->get_consumers()
                       .empty()) {
        if (cons->size() == 1
                && cons->at(0)->first->get_node_kind()
                        == pb_node_kind::PB_NODE_KIND_REPETITION) {
            return true;
        }
        return false;
    }
    node_tracker_ptr n_tracker = find_node_tracker(op, ctx);
    auto val = op->get_output_value(static_cast<size_t>(otask.port));
    auto con_ops = val->get_consumers();
    vector<bool> v
            = n_tracker->op_unhandled_output[static_cast<size_t>(otask.port)];
    // common case: one output logical tensor consumer
    // and one output port consumer.
    if ((cons->size() == 1) && (con_ops.size() == 1)) {
        shared_ptr<consumer_t> con = cons->at(0);
        pb_node_ptr con_node = con->first;
        op_ptr con_op = &(con_ops[0].get_op());
        size_t con_op_port = con_ops[0].get_offset();
        v[0] = false;
        binding_t b {BIND_IN, con_op, static_cast<int64_t>(con_op_port),
                con_node, con->second, 0};
        n_tracker->op_unhandled_output[static_cast<size_t>(otask.port)] = v;
        if (con_node->get_node_kind() == pb_node_kind::PB_NODE_KIND_OP) {
            if (!register_node_tracker(b, ctx)) { return false; }
        } else {
            if (!resolve_node(b, ctx)) { return false; }
        }
    } else {
        // For multiple output consumers (n) and op output consumers (k)
        // Loop n x k combinations to match all output consumers
        if (cons->size() > con_ops.size()) { return false; }
        for (const shared_ptr<consumer_t> &con : *cons) {
            bool matched_node = false;
            pb_node_ptr con_node = con->first;
            // Limitation: consumer node must be pb_op to match node attributes
            if (con_node->get_node_kind() != pb_node_kind::PB_NODE_KIND_OP) {
                return false;
            }
            for (auto c : con_ops) {
                op_ptr con_op = &(c.get_op());
                size_t con_op_port = c.get_offset();
                if (match_node_attributes(con_op, con_node)) {
                    node_tracker_ptr tcon = find_node_tracker(con_op, ctx);
                    if (tcon != nullptr) {
                        // if consumer input port is already matched,
                        // mark as matched.
                        if (!tcon->op_unhandled_input[con_op_port]) {
                            if (con_op_port != con->second) { continue; }
                        }
                    }
                    matched_node = true;
                    auto idx = get_output_consumer_index(con_op, con_op_port);
                    v[static_cast<size_t>(idx)] = false;
                    binding_t b {BIND_IN, con_op,
                            static_cast<int64_t>(con_op_port), con_node,
                            con->second, idx};
                    if (!register_node_tracker(b, ctx)) { return false; }
                    break;
                }
            }
            if (!matched_node) { return false; }
        }
    }
    n_tracker->op_unhandled_output[static_cast<size_t>(otask.port)] = v;
    return true;
}

bool match_node_outputs(op_ptr op, match_context_ptr ctx) {
    node_tracker_ptr n_tracker = ctx->node_tracker_map[op];
    pb_node_ptr node = n_tracker->get_node();
    if (node->get_node_kind() != pb_node_kind::PB_NODE_KIND_OP) {
        return false;
    }
    auto outputs = node->get_outputs();
    if (outputs.empty()) { return true; }
    // Get output logical tensors of an op.
    while (!n_tracker->dst_to_visit.empty()) {
        output_match_task_t otask = n_tracker->dst_to_visit.front();
        n_tracker->dst_to_visit.pop_front();
        oport_t node_port = otask.port;
        if (!match_output(op, ctx, otask, node->get_consumers(node_port))) {
            return false;
        }
    }
    return true;
}

bool match_node(op_ptr op, match_context_ptr ctx) {
    pb_node_ptr node = ctx->node_tracker_map[op]->get_node();
    if (node->get_node_kind() != pb_node_kind::PB_NODE_KIND_OP) {
        return false;
    }
    if (!match_node_attributes(op, node)) { return false; }
    if (!match_node_inputs(op, ctx)) { return false; }
    if (!match_node_outputs(op, ctx)) { return false; }
    // Check if this node is an i/o pad of current graph
    // And update i/o pad mapping.
    auto inner_cons = ctx->get_graph()->get_inner_consumers();
    for (size_t i = 0; i < inner_cons.size(); i++) {
        auto con_set = inner_cons[i];
        if (con_set == nullptr) continue;
        // Limitation: inner port forwarding only works for single consumer.
        if (con_set->size() > 1) { return false; }
        auto con = *(con_set->begin());
        pb_node_ptr con_node = con->first;
        if (con_node == node) {
            ctx->in_port_map.emplace(
                    i, pair<op_ptr, int64_t> {op, con->second});
        }
    }
    auto inner_prods = ctx->get_graph()->get_inner_producers();
    for (size_t i = 0; i < inner_prods.size(); i++) {
        auto prod = inner_prods[i];
        if (prod == nullptr) continue;
        pb_node_ptr prod_node = prod->first;
        if (prod_node == node) {
            ctx->out_port_map.emplace(
                    i, pair<op_ptr, int64_t> {op, prod->second});
        }
    }
    ctx->unhandled_nodes.erase(node);
    return true;
}

//
// Part 2.
// match functions for nested pattern nodes.
//

//
// If pb_op, put in work deque and return pb_op
// Else call nested matchers depending on node type
//
bool resolve_node(const binding_t &bind_arg, match_context_ptr ctx) {
    bool success = false;
    switch (bind_arg.bind_node->get_node_kind()) {
        case pb_node_kind::PB_NODE_KIND_GRAPH:
            success = match_graph(bind_arg, ctx, nullptr);
            break;
        case pb_node_kind::PB_NODE_KIND_ALTERNATION:
            success = match_alternation(bind_arg, ctx);
            break;
        case pb_node_kind::PB_NODE_KIND_REPETITION:
            success = match_repetition(bind_arg, ctx);
            break;
        default: break;
    }
    return success;
}

bool match_pattern(op_ptr first_op, const shared_ptr<pb_graph_t> &pattern,
        match_t &m, bool auto_export_externals, bool match_forward) {
    match_context_t global_ctx {nullptr, nullptr};
    binding_t init_bind {BIND_NONE, first_op, -1, pattern.get(),
            (match_forward ? 0 : -1), 0};
    if (!match_graph(init_bind, &global_ctx, nullptr)) { return false; }
    vector<op_ptr> matched_ops;
    for (auto const &i : global_ctx.node_tracker_map) {
        matched_ops.push_back(i.first);
    }
    for (auto const &i : global_ctx.node_tracker_map) {
        pb_op *p_op = static_cast<pb_op *>(i.second->get_node());
        m.op_pb_op_pairs.emplace_back(i.first, p_op);
        for (size_t j = 0; j < i.second->op_unhandled_input.size(); j++) {
            op_ptr op = i.first;
            op_ptr prod_op = nullptr;
            if (i.second->op_unhandled_input[j] == true) {
                auto input_value = op->get_input_value(j);
                bool found = false;
                if (input_value->has_producer()) {
                    prod_op = &(input_value->get_producer());
                    // check if prod_op is in matched_ops
                    for (size_t k = 0; k < matched_ops.size(); k++) {
                        if (matched_ops[k] == prod_op) {
                            found = true;
                            break;
                        }
                    }
                }
                if (!found) {
                    // External input
                    m.inputs.push_back(i.first->get_input_value(j));
                } else {
                    // Internal input
                    if (!auto_export_externals) {
                        auto inter_ins = p_op->get_allowed_internal_inputs();
                        auto inter = inter_ins.find(static_cast<int64_t>(j));
                        if (inter == inter_ins.end()) {
                            bool has_unallowed = true;
                            auto comm_pair = p_op->get_commutative_pair();
                            if (j == comm_pair.first) {
                                if (inter_ins.end()
                                        != inter_ins.find(comm_pair.second)) {
                                    has_unallowed = false;
                                }
                            } else if (j == comm_pair.second) {
                                if (inter_ins.end()
                                        != inter_ins.find(comm_pair.first)) {
                                    has_unallowed = false;
                                }
                            }
                            if (has_unallowed) { return false; }
                        }
                    }
                }
            }
        }
        for (size_t j = 0; j < i.second->op_unhandled_output.size(); j++) {
            op_ptr op = i.first;
            auto cons = op->get_output_value(j)->get_consumers();
            vector<bool> outs = i.second->op_unhandled_output[j];
            // graph op has an output logical tensor without a consumer
            if (outs.empty()) {
                m.outputs.push_back(i.first->get_output_value(j));
                continue;
            }
            bool unhandled = false;
            bool external_allowed = true;
            bool root_node = (p_op->get_outputs().empty());
            if (!auto_export_externals) {
                auto ext_outs = p_op->get_allowed_external_outputs();
                auto ext = ext_outs.find(static_cast<int64_t>(j));
                if (ext == ext_outs.end()) { external_allowed = false; }
            }
            for (size_t k = 0; k < outs.size(); k++) {
                bool b = outs[k];
                // port k has unhandled output
                if (b) {
                    // Check if op consumer is external
                    op_ptr con_op = &(cons[k].get_op());
                    bool found = false;
                    for (size_t n = 0; n < matched_ops.size(); n++) {
                        auto m = matched_ops[n];
                        if (m == con_op) {
                            b = false;
                            found = true;
                            break;
                        }
                    }
                    if (!found && !(root_node || external_allowed)) {
                        return false;
                    }
                }
                unhandled = unhandled || b;
            }
            if (unhandled) {
                m.outputs.push_back(i.first->get_output_value(j));
            }
        }
    }

    std::vector<op_ptr> candidate_fusion;
    for (auto &pair : m.op_pb_op_pairs) {
        candidate_fusion.push_back(pair.first);
    }

    return true;
}

match_context_t::match_context_t(
        match_context_ptr p_ctx, const pb_node_ptr &p_graph)
    : parent_ctx {p_ctx}, m_node {p_graph} {
    m_graph = dynamic_cast<pb_graph_t *>(p_graph);
    if (m_graph != nullptr) {
        for (auto const &node : m_graph->get_nodes()) {
            unhandled_nodes.insert(node);
        }
    }
}

static bool fill_parent_io_map(match_context_ptr local_ctx) {
    if (local_ctx == nullptr) { return false; }
    auto parent_ctx = local_ctx->get_parent_context();
    if (local_ctx->get_parent_context() != nullptr) {
        auto pgraph = parent_ctx->get_graph();
        if (!pgraph) {
            parent_ctx->in_port_map.insert(local_ctx->in_port_map.begin(),
                    local_ctx->in_port_map.end());
            parent_ctx->out_port_map.insert(local_ctx->out_port_map.begin(),
                    local_ctx->out_port_map.end());
        } else {
            auto inner_cons = pgraph->get_inner_consumers();
            for (size_t i = 0; i < inner_cons.size(); i++) {
                auto con_set = inner_cons[i];
                if (con_set == nullptr) continue;
                // Limitation: inner port forwarding only works for
                // single consumer.
                if (con_set->size() > 1) { return false; }
                auto con = *(con_set->begin());
                // this graph, i -> con->first, con->second
                int64_t si = static_cast<int64_t>(i);
                pb_node_ptr con_node = con->first;
                if (con_node == local_ctx->get_node()) {
                    parent_ctx->in_port_map.emplace(si,
                            pair<op_ptr, int64_t> {
                                    local_ctx->in_port_map[si].first,
                                    local_ctx->in_port_map[si].second});
                }
            }
            auto inner_prods = pgraph->get_inner_producers();
            for (size_t i = 0; i < inner_prods.size(); i++) {
                auto prod = inner_prods[i];
                if (prod == nullptr) continue;
                pb_node_ptr prod_node = prod->first;
                // this graph, i -> prod->first, prod->second
                int64_t si = static_cast<int64_t>(i);
                if (prod_node == local_ctx->get_node()) {
                    parent_ctx->out_port_map.emplace(si,
                            pair<op_ptr, int64_t> {
                                    local_ctx->out_port_map[si].first,
                                    local_ctx->out_port_map[si].second});
                }
            }
        }
    }
    return true;
}

//
// match nested pattern starting from initial binding
//
bool match_graph(const binding_t &bind_arg, match_context_ptr parent_ctx,
        pair<graph_port_map, graph_port_map> *io_map) {
    // Create local match context
    match_context_t local_ctx {parent_ctx, bind_arg.bind_node};
    binding_t local_bind = bind_arg;
    // Get initial internal node to bind
    switch (bind_arg.bind_kind) {
        case BIND_NONE:
            if (local_bind.bind_port == 0) {
                local_bind.bind_node
                        = local_ctx.get_graph()->get_nodes().front();
            } else {
                local_bind.bind_node
                        = local_ctx.get_graph()->get_nodes().back();
            }
            break;
        case BIND_IN: {
            // Generally a graph can have more than one inner consumers
            // But is restricted to one in a BIND_IN case.
            shared_ptr<unordered_set<shared_ptr<consumer_t>>> cons
                    = local_ctx.get_graph()->get_inner_consumer(
                            bind_arg.bind_port);
            if (cons->size() != 1) { return false; }
            shared_ptr<consumer_t> con = *(cons->begin());
            local_bind.bind_node = con->first;
            local_bind.bind_port = con->second;
        } break;
        case BIND_OUT: {
            shared_ptr<producer_t> prod
                    = local_ctx.get_graph()->get_inner_producer(
                            bind_arg.bind_port);
            local_bind.bind_node = prod->first;
            local_bind.bind_port = prod->second;
        } break;
        default: break;
    }
    // initial internal node may not be a pb_op
    if (local_bind.bind_node->get_node_kind()
            != pb_node_kind::PB_NODE_KIND_OP) {
        if (!resolve_node(local_bind, &local_ctx)) { return false; }
    } else {
        if (!register_node_tracker(local_bind, &local_ctx)) { return false; }
    }
    while (!local_ctx.ops_to_visit.empty()) {
        op_ptr op = local_ctx.ops_to_visit.front();
        local_ctx.ops_to_visit.pop_front();
        if (!match_node(op, &local_ctx)) { return false; }
    }

    if (!local_ctx.unhandled_nodes.empty()) {
        for (auto const &n : local_ctx.unhandled_nodes) {
            if (n->get_node_kind() != pb_node_kind::PB_NODE_KIND_REPETITION) {
                return false;
            }
            repetition_t *rep = dynamic_cast<repetition_t *>(n);
            if (rep->get_min_rep() != 0) { return false; }
        }
    }
    // ctx: Fill in input port, output port map to op
    // of parent context if this graph is an i/o pad.
    // i/o pad from ops in the graph is already filled in at this point.
    if (!fill_parent_io_map(&local_ctx)) { return false; }
    // merge current node tracker map to parent map
    // Need to do this before match_graph_inputs and match_graph_outputs
    parent_ctx->node_tracker_map.insert(local_ctx.node_tracker_map.begin(),
            local_ctx.node_tracker_map.end());
    // similar to match_input
    if (!match_graph_inputs(&local_ctx, local_ctx.get_node(), bind_arg,
                &(local_ctx.in_port_map))) {
        return false;
    }
    if (!match_graph_outputs(
                &local_ctx, local_ctx.get_node(), &(local_ctx.out_port_map))) {
        return false;
    }

    // This code should be just before returning.
    if (io_map != nullptr) {
        io_map->first.insert(
                local_ctx.in_port_map.begin(), local_ctx.in_port_map.end());
        io_map->second.insert(
                local_ctx.out_port_map.begin(), local_ctx.out_port_map.end());
    }
    return true;
}

bool match_graph_inputs(match_context_ptr ctx, const pb_node_ptr &graph_node,
        const binding_t &graph_binding, graph_port_map *in_port_map) {
    // put neighbor ops into work deque from i/o pad mapping
    // update i/o pad ops unhandled ports
    for (auto inp : *in_port_map) {
        iport_t graph_iport = inp.first;
        op_ptr op = inp.second.first;
        iport_t node_in_port = inp.second.second;
        node_tracker_ptr nt_ptr = ctx->node_tracker_map[op];
        // filtering before creating input match task
        shared_ptr<producer_t> prod = graph_node->get_producer(graph_iport);
        if (prod == nullptr) continue;
        // if this input port was used for binding_t to this graph
        if (graph_binding.bind_kind == BIND_IN
                && graph_binding.bind_port == graph_iport) {
            nt_ptr->op_unhandled_input[static_cast<size_t>(node_in_port)]
                    = false;
            continue;
        }
        iport_pair ipair = nt_ptr->get_node()->get_commutative_pair();
        bool has_commutative_input = (ipair.first >= 0) && (ipair.second >= 0);
        input_match_kind mkind = has_commutative_input
                ? INPUT_MATCH_KIND_COMMUTATIVE_ONE_CONSTRAINT
                : INPUT_MATCH_KIND_NORMAL;
        input_match_task_t itask {mkind, node_in_port, ipair.second};
        if (!match_input(op, ctx, itask, graph_node->get_producer(graph_iport),
                    nullptr)) {
            return false;
        }
    }
    return true;
}

bool match_graph_outputs(match_context_ptr ctx, const pb_node_ptr &graph_node,
        graph_port_map *out_port_map) {
    // put neighbor ops into work deque from i/o pad mapping
    // update i/o pad ops unhandled ports
    for (auto outp : *out_port_map) {
        oport_t graph_oport = outp.first;
        op_ptr op = outp.second.first;
        oport_t node_out_port = outp.second.second;
        shared_ptr<consumers_t> cons = graph_node->get_consumers(graph_oport);
        if (cons == nullptr) continue;
        output_match_task_t otask;
        otask.port = node_out_port;
        // otask.num_consumers is not used in this work flow
        if (!match_output(
                    op, ctx, otask, graph_node->get_consumers(graph_oport))) {
            return false;
        }
    }
    return true;
}

//
// match_graph_inputs/outputs do not apply to body of alternation
// and repetition so we don't encounter a case where we have to
// rollback op unhandled edge states.
//

bool match_alternation(
        const binding_t &bind_arg, match_context_ptr parent_ctx) {
    bool success = false;
    alternation_t *altnode = dynamic_cast<alternation_t *>(bind_arg.bind_node);
    vector<pb_graph_t *> alts = altnode->get_alternatives();

    pair<graph_port_map, graph_port_map> io_map;
    // binding_t can be created with real parent ctx g
    // since we settle with first matched alternative.
    // but we need to connect the inputs and outputs
    // body manually since the body is not directly connect to
    // parent. (update i/o op edges)
    // Alternatively, we may use a null context for simplicity.
    for (pb_graph_t *gra : alts) {
        binding_t local_bind = bind_arg;
        local_bind.bind_node = gra;
        success = match_graph(local_bind, parent_ctx, &io_map);
        if (success) break;
    }
    if (!success) { return false; }
    // connect global inputs/outputs and merge with g
    pb_node_ptr graph_node = bind_arg.bind_node;
    if (!match_graph_inputs(
                parent_ctx, graph_node, bind_arg, &(io_map.first))) {
        return false;
    }
    if (!match_graph_outputs(parent_ctx, graph_node, &(io_map.second))) {
        return false;
    }
    parent_ctx->unhandled_nodes.erase(altnode);
    return true;
}

bool match_repetition(const binding_t &bind_arg, match_context_ptr parent_ctx) {
    repetition_t *repnode = dynamic_cast<repetition_t *>(bind_arg.bind_node);
    pb_node_ptr body = repnode->get_body();
    port_maps pmap = repnode->get_port_maps();
    int64_t min_rep = repnode->get_min_rep();
    int64_t max_rep = repnode->get_max_rep();

    // binding_t has to be created with fake parent if min_rep is
    // greater than 1. Since a match may need to rollback.
    // We use a null context for simplicity.
    // While looping, ff trip count becomes max_rep. stop iteration
    // and merge null context with parent, update i/o op edges
    // and return true.
    // If loop ends and trip count is less than rep_min,
    // don't merge null context with parent and return false.
    // Else, merge null context with parent, update i/o op edges
    // and return true.

    // binding_t for first iteration.
    // all iterations have same body, bind_kind and bind_port
    // but they have different bind_op.
    // First iteration has same bind_op as the repetition node.
    binding_t local_bind = bind_arg;
    local_bind.bind_node = body;

    // Create reverse port map iport_t -> oport_t
    vector<pair<iport_t, oport_t>> rev_pmap;
    for (auto e : pmap) {
        rev_pmap.emplace_back(e.second, e.first);
    }

    // Various match_context
    // Need a confirmed context since exploring iteration is speculative
    match_context_t confirmed_ctx {parent_ctx, bind_arg.bind_node};
    // Also need a merge context to tag on incremental iterations.
    // This context is speculative since matching edges across neighboring
    // iterations can fail.
    match_context_t speculative_ctx {parent_ctx, bind_arg.bind_node};
    // Both context has the same binding_t (local_bind) and same parent context
    // (parent_ctx) to refer to outside ops for alias checking
    // Every iteration needs a separate context and they will be used as
    // a unit for:
    // 1. (Auto) Adding to speculative_ctx for checking cross edges
    // 2. (Manual) Commiting to the confirmed_ctx.
    int i = 0;
    binding_t temp_bind = local_bind;
    // Loop appears to be moving forward, but matching may happen
    // backward if bind_kind os BIND_OUT
    bool forward_match
            = (bind_arg.bind_kind == BIND_NONE && bind_arg.bind_port == 0)
            || bind_arg.bind_kind == BIND_IN;
    for (; i < max_rep - 1; i++) {
        if (!temp_bind.bind_op) break;
        match_context_t temp_ctx {&speculative_ctx, nullptr};
        if (!match_graph(temp_bind, &temp_ctx, nullptr)) { break; }
        // update binding_t temp_bind for next rep;
        if (i < max_rep - 2) {
            // Forward matching
            if (forward_match) {
                // Forward (BIND_IN):
                oport_t oport = pmap[0].first;
                op_ptr out_op = temp_ctx.out_port_map[oport].first;
                if (oport >= out_op->num_outputs()) { break; }
                auto con_ops
                        = out_op->get_output_value(static_cast<size_t>(oport))
                                  ->get_consumers();
                op_ptr in_op = nullptr;
                if (con_ops.size() == 1) { in_op = &(con_ops.at(0).get_op()); }
                temp_bind.bind_op = in_op;
            } else { // backward matching
                // Backward:(BIND_OUT):
                forward_match = false;
                iport_t iport = pmap[0].second;
                op_ptr in_op = temp_ctx.in_port_map[iport].first;
                if (iport >= in_op->num_inputs()) { break; }
                op_ptr out_op
                        = &(in_op->get_input_value(static_cast<size_t>(iport))
                                        ->get_producer());
                temp_bind.bind_op = out_op;
            }
        }
        // connect other edges across prev and this rep
        // with match_graph_inputs and match_graph_outputs
        if (i > 0) {
            match_context_ptr out_ctx
                    = forward_match ? &confirmed_ctx : &temp_ctx;
            match_context_ptr in_ctx
                    = forward_match ? &temp_ctx : &confirmed_ctx;
            for (auto i : pmap) {
                auto j = out_ctx->out_port_map[i.first];
                op_ptr prod_op = j.first;
                int64_t prod_port = j.second;
                auto k = in_ctx->in_port_map[i.second];
                op_ptr con_op = k.first;
                int64_t con_port = k.second;
                op_ptr expected = &(
                        con_op->get_input_value(static_cast<size_t>(con_port))
                                ->get_producer());
                if (expected != prod_op) { return false; }
                int64_t idx = get_output_consumer_index(
                        con_op, static_cast<size_t>(con_port));
                if (idx == -1) { return false; }
                node_tracker_ptr n_prod = out_ctx->node_tracker_map[prod_op];
                vector<bool> v
                        = n_prod->op_unhandled_output[static_cast<size_t>(
                                prod_port)];
                v[static_cast<size_t>(idx)] = false;
                n_prod->op_unhandled_output[static_cast<size_t>(prod_port)] = v;
                node_tracker_ptr n_con = in_ctx->node_tracker_map[con_op];
                n_con->op_unhandled_input[static_cast<size_t>(con_port)]
                        = false;
            }
        }
        // merge temp_ctx's node_track_map to confirmed_ctx
        confirmed_ctx.node_tracker_map.insert(temp_ctx.node_tracker_map.begin(),
                temp_ctx.node_tracker_map.end());
        // update in_port_map and/or out_port_map of
        // confirmed_ctx
        if (forward_match) {
            if (i == 0) {
                // temp_ctx's in_port_map becomes confirmed_ctx's
                // in_port_map
                confirmed_ctx.in_port_map.insert(temp_ctx.in_port_map.begin(),
                        temp_ctx.in_port_map.end());
            }
            // "update" confirmed_ctx's out_port_map with temp_ctx's
            // out_port_map
            confirmed_ctx.out_port_map.clear();
            confirmed_ctx.out_port_map.insert(
                    temp_ctx.out_port_map.begin(), temp_ctx.out_port_map.end());
        } else {
            if (i == 0) {
                // temp_ctx's out_port_map becomes confirmed_ctx's
                // out_port_map
                confirmed_ctx.out_port_map.insert(temp_ctx.out_port_map.begin(),
                        temp_ctx.out_port_map.end());
            }
            // "update" confirmed_ctx's in_port_map with temp_ctx's
            // in_port_map
            confirmed_ctx.in_port_map.clear();
            confirmed_ctx.in_port_map.insert(
                    temp_ctx.in_port_map.begin(), temp_ctx.in_port_map.end());
        }
    }
    if (i < min_rep) { return false; }
    if (i == min_rep && i == 0) {
        // Zero trip match
        // need to forward binding_t request to neighboring nodes
        if (forward_match
                && (bind_arg.bind_node->get_consumers(0) != nullptr)) {
            auto cons = bind_arg.bind_node->get_consumers(0);
            if (bind_arg.bind_kind == BIND_NONE) {
                // resolve_node with ... if BIND_NONE
                // get next consumer
                if (cons->size() != 1) { return false; }
                auto con = *(cons->begin());
                binding_t optional_bind = bind_arg;
                optional_bind.bind_node = con->first;
                if (!resolve_node(optional_bind, parent_ctx)) { return false; }
            } else {
                // get input port from binding
                // get predecessor graph op
                op_ptr op = &(bind_arg.bind_op
                                      ->get_input_value(static_cast<size_t>(
                                              bind_arg.bind_port))
                                      ->get_producer());
                // do a match_output with a new consumers
                output_match_task_t otask {0, 0};
                if (!match_output(op, parent_ctx, otask, cons)) {
                    return false;
                }
            }
        } else if (!forward_match
                && (bind_arg.bind_node->get_producer(0) != nullptr)) {
            auto prod = bind_arg.bind_node->get_producer(0);
            if (bind_arg.bind_kind == BIND_NONE) {
                // resolve_node with ... if BIND_NONE
                // get next producer
                binding_t optional_bind = bind_arg;
                optional_bind.bind_node = prod->first;
                if (!resolve_node(optional_bind, parent_ctx)) { return false; }
            } else {
                // get output port and idx from binding.
                // get successor graph op
                op_ptr op = &(bind_arg.bind_op
                                      ->get_output_value(static_cast<size_t>(
                                              bind_arg.bind_port))
                                      ->get_consumers()
                                      .at(static_cast<size_t>(
                                              bind_arg.bind_port_user_idx))
                                      .get_op());
                // do a match input with a new producer
                input_match_task_t itask {INPUT_MATCH_KIND_NORMAL, 0, -1};
                if (!match_input(op, parent_ctx, itask, prod, nullptr)) {
                    return false;
                }
            }
        }
    } else {
        // From confirmed_ctx, merge node_tracker_map onto parent_ctx
        parent_ctx->node_tracker_map.insert(
                confirmed_ctx.node_tracker_map.begin(),
                confirmed_ctx.node_tracker_map.end());
        // connect global inputs/outputs with match_graph_inputs and
        // match_graph_outputs.
        pb_node_ptr graph_node = bind_arg.bind_node;
        if (!match_graph_inputs(parent_ctx, graph_node, bind_arg,
                    &(confirmed_ctx.in_port_map))) {
            return false;
        }
        if (!match_graph_outputs(
                    parent_ctx, graph_node, &(confirmed_ctx.out_port_map))) {
            return false;
        }
        // update parent_ctx's in_port_maps and out_port_map if needed
        if (!fill_parent_io_map(&confirmed_ctx)) { return false; }
    }
    parent_ctx->unhandled_nodes.erase(bind_arg.bind_node);
    return true;
}

} // namespace pm
} // namespace utils
} // namespace impl
} // namespace graph
} // namespace dnnl
