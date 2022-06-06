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

#include <algorithm>
#include <deque>
#include <iterator>
#include <unordered_map>
#include <unordered_set>

#include "utils/pm/nested_matcher.hpp"

namespace dnnl {
namespace graph {
namespace impl {
namespace utils {
namespace pm {

bool has_commutative_inputs(op_t *op) {
    static const std::unordered_set<op_kind_t> commutative_kinds {op_kind::Add,
            op_kind::Multiply, op_kind::Maximum, op_kind::Minimum};
    return commutative_kinds.count(op->get_kind());
}

bool has_variadic_inputs(op_t *op) {
    static const std::unordered_set<op_kind_t> variadic_kinds {op_kind::Concat};
    return variadic_kinds.count(op->get_kind());
}

binding_t::binding_t(node_bind_kind p_kind, op_t *p_op, int64_t p_op_port,
        pb_node *p_node, int64_t p_port)
    : bind_op {p_op}
    , bind_node {p_node}
    , bind_kind {p_kind}
    , bind_port {p_port}
    , bind_op_port {p_op_port} {}

//
// Part 1.
// match functions for pb_op's
//

bool match_node_attributes(op_t *op, pb_node *node) {
    size_t n_func = node->get_num_decision_functions();
    for (size_t i = 0; i < n_func; i++) {
        if (!(node->get_decision_function(i)(op))) { return false; }
    }
    return true;
}

bool match_node_inputs(op_t *op, pb_node *node, match_context_t *ctx,
        std::unordered_map<op_t *, pb_op *> &matched_op_map) {
    std::vector<std::pair<iport_t, producer_t>> node_inputs
            = node->get_inputs();
    if (node_inputs.empty()) return true;

    std::unordered_map<op_t *, pb_op *> copied_op_map = matched_op_map;
    if (!has_commutative_inputs(op)) {
        for (size_t i = 0; i < node_inputs.size(); ++i) {
            if (op->num_inputs() < i + 1) {
                if (has_variadic_inputs(op))
                    break;
                else
                    return false;
            }
            iport_t node_iport = node_inputs[i].first;
            if (op->num_inputs() < node_iport + 1) return false;
            std::shared_ptr<value_t> op_in_value
                    = op->get_input_value(node_iport);
            pb_node *in_node = node_inputs[i].second.first;
            if (!op_in_value->has_producer()) {
                // in this case, only optional can survive
                if (in_node->get_node_kind()
                        != pb_node_kind::PB_NODE_KIND_REPETITION)
                    return false;
                repetition_t *rep_node = dynamic_cast<repetition_t *>(in_node);
                if (rep_node->get_min_rep() != 0) return false;
            } else {
                op_t *in_op = op->get_input_op(node_iport);
                size_t in_op_oport = op_in_value->get_offset();
                oport_t in_node_oport = node_inputs[i].second.second;
                binding_t in_bind(
                        BIND_OUT, in_op, in_op_oport, in_node, in_node_oport);
                if (!match_graph_helper(in_bind, ctx, copied_op_map)) {
                    return false;
                }
            }
        }
    } else { // commutative ops need to consider switching inputs
        size_t matched_op_input_offset = op->num_inputs(); // init with illegal
        for (size_t node_input_offset = 0;
                node_input_offset < node_inputs.size(); ++node_input_offset) {
            for (size_t op_input_offset = 0; op_input_offset < op->num_inputs();
                    ++op_input_offset) {
                if (op_input_offset == matched_op_input_offset) {
                    matched_op_input_offset = op->num_inputs();
                    continue;
                }
                std::shared_ptr<value_t> op_in_value
                        = op->get_input_value(op_input_offset);
                pb_node *in_node = node_inputs[node_input_offset].second.first;
                oport_t in_node_oport
                        = node_inputs[node_input_offset].second.second;
                if (op_in_value->has_producer()) {
                    op_t *in_op = op->get_input_op(op_input_offset);
                    size_t in_op_oport = op_in_value->get_offset();
                    binding_t in_bind(BIND_OUT, in_op, in_op_oport, in_node,
                            in_node_oport);
                    if (match_graph_helper(in_bind, ctx, copied_op_map)) {
                        matched_op_input_offset = op_input_offset;
                        break;
                    }
                }
            }
            if (matched_op_input_offset == op->num_inputs()) return false;
        }
    }

    matched_op_map = copied_op_map;
    return true;
}

bool match_node_outputs(op_t *op, pb_node *node, match_context_t *ctx,
        std::unordered_map<op_t *, pb_op *> &matched_op_map) {
    std::vector<std::pair<oport_t, consumers_t>> node_outputs
            = node->get_outputs();
    if (node_outputs.empty()) return true;

    std::unordered_map<op_t *, pb_op *> copied_op_map = matched_op_map;

    //match output for node and op
    for (auto &node_output : node_outputs) {
        size_t node_output_offset = node_output.first;
        if (op->num_outputs() < node_output_offset + 1) return false;
        std::shared_ptr<value_t> op_out_value
                = op->get_output_value(node_output_offset);
        std::unordered_set<size_t> matched_node_offsets;
        std::unordered_map<op_t *, pb_op *> op_map_for_current_node_output
                = copied_op_map;
        // match the consumers one by one
        for (size_t j = 0; j < op_out_value->get_consumers().size(); j++) {
            auto op_consumer = op_out_value->get_consumers()[j];
            op_t *out_op = &(op_consumer.get_op());
            bool consumer_matched = false;

            for (size_t k = 0; k < node_output.second.size(); k++) {
                auto node_consumer = node_output.second[k];
                pb_node *out_node = node_consumer->first;
                // check if the out_node has been matched by previous out_ops
                if (matched_node_offsets.count(k)) continue;
                binding_t out_bind(BIND_IN, out_op,
                        int64_t(op_consumer.get_offset()), out_node,
                        node_consumer->second);
                if (!match_graph_helper(
                            out_bind, ctx, op_map_for_current_node_output)) {
                    continue;
                } else {
                    consumer_matched = true;
                    matched_node_offsets.insert(k);
                    break;
                }
            }
            // find coupled node_output
            if (!consumer_matched) {
                // TODO(Yixin): temporary fix sigmoid + multiply = swish
                // After successfully matching sigmoid, multiply is also
                // matched because multiply is sigmoid's out_op.
                // When matching multiply, check if it is already in the
                // matched op_map, if yes, the match is OK
                if (node_output.second.size() == 1
                        && node_output.second[0]->first->get_node_kind()
                                != pb_node_kind::PB_NODE_KIND_OP
                        && op_map_for_current_node_output.count(out_op))
                    continue;
                // check if allow external output
                if (node->get_node_kind() == pb_node_kind::PB_NODE_KIND_OP) {
                    pb_op *p_op = dynamic_cast<pb_op *>(node);
                    std::unordered_set<oport_t> external_outputs
                            = p_op->get_allowed_external_outputs();
                    if (!external_outputs.empty()
                            && external_outputs.find(node_output_offset)
                                    != external_outputs.end()) {
                        continue;
                    } else {
                        // the current node_output_offset match failed, clear
                        // the matched_node_offsets, clear current node output's
                        // matched_op_map;
                        matched_node_offsets.clear();
                        op_map_for_current_node_output = copied_op_map;
                        break;
                    }
                }
            }
        }

        // check if there are unmatched node outputs
        for (size_t k = 0; k < node_output.second.size(); k++) {
            if (!matched_node_offsets.count(k)) {
                // in this case, only optional can survive
                pb_node *out_node = node_output.second[k]->first;
                bool is_optional = check_is_optional(out_node);
                if (!is_optional) return false;
            }
        }
        copied_op_map = op_map_for_current_node_output;
    }
    matched_op_map = copied_op_map;
    return true;
}

bool check_is_optional(pb_node *n) {
    if (n->get_node_kind() != pb_node_kind::PB_NODE_KIND_REPETITION)
        return false;
    repetition_t *rep_node = dynamic_cast<repetition_t *>(n);
    if (rep_node->get_min_rep() != 0) return false;

    std::vector<std::pair<oport_t, consumers_t>> node_outputs
            = n->get_outputs();
    if (!node_outputs.empty()) {
        for (auto &node_output : node_outputs) {
            for (size_t i = 0; i < node_output.second.size(); i++) {
                bool is_optional
                        = check_is_optional(node_output.second[i]->first);
                if (!is_optional) return false;
            }
        }
    }
    return true;
}

bool match_node(const binding_t &b, match_context_t *ctx,
        std::unordered_map<op_t *, pb_op *> &matched_op_map) {
    if (b.bind_op == nullptr) return false;
    if (b.bind_node == nullptr) return false;
    if (b.bind_op->get_partition() != nullptr) return false;
    if (b.bind_op->has_attr("matched_pattern")) return false;

    if (!match_node_attributes(b.bind_op, b.bind_node)) return false;

    if (!match_node_inputs(b.bind_op, b.bind_node, ctx, matched_op_map)) {
        return false;
    }

    if (!match_node_outputs(b.bind_op, b.bind_node, ctx, matched_op_map)) {
        return false;
    }

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
bool resolve_node(const binding_t &bind_arg, match_context_t *ctx,
        std::unordered_map<op_t *, pb_op *> &matched_op_map) {
    bool success = false;
    switch (bind_arg.bind_node->get_node_kind()) {
        case pb_node_kind::PB_NODE_KIND_ALTERNATION:
            success = match_alternation(bind_arg, ctx, matched_op_map);
            break;
        case pb_node_kind::PB_NODE_KIND_REPETITION:
            success = match_repetition(bind_arg, ctx, matched_op_map);
            break;
        default: break;
    }
    return success;
}

bool match_pattern(op_t *first_op, const std::shared_ptr<pb_graph_t> &pattern,
        std::vector<op_t *> &fusion_ops) {
    match_context_t global_ctx {nullptr, nullptr};
    match_context_t init_ctx {&global_ctx, pattern.get()};
    binding_t init_bind {BIND_NONE, first_op, 0, pattern.get(), 0};
    std::unordered_map<op_t *, pb_op *> matched_op_map;
    if (!match_graph(init_bind, &init_ctx, matched_op_map)) { return false; }

    fusion_ops = reorder_matched_list(matched_op_map);

    return true;
}

// function to reorder the matched list into topo order
// fuse function need op_list is ordered by topo order,
// which is not confirmed by v2 matcher, so need to reorder
// the op_list firstly.
// can be removed after subgraph mode finished.
inline std::vector<op_t *> reorder_matched_list(
        const std::unordered_map<op_t *, pb_op *> &matched_op_map) {
    // split ops and pb_ops
    std::vector<op_t *> fusion_ops;
    std::vector<pb_op *> pb_ops;
    for (auto kv : matched_op_map) {
        fusion_ops.push_back(kv.first);
        pb_ops.push_back(kv.second);
    }

    // a deque of op indexes in fusion_ops
    std::deque<op_t *> dq;
    // find an end_op to start reorder, find an op whose output isn't in
    // fusion_ops
    for (auto &aop : fusion_ops) {
        bool is_end = true;
        for (auto &output : aop->get_output_values()) {
            for (auto &consumer : output->get_consumers()) {
                if (std::find(fusion_ops.begin(), fusion_ops.end(),
                            &(consumer.get_op()))
                        != fusion_ops.end()) {
                    is_end = false;
                    break;
                }
            }
            if (!is_end) { break; }
        }
        if (is_end) {
            dq.push_back(aop);
            break;
        }
    }
    std::unordered_set<op_t *> visited;
    std::vector<op_t *> reordered_fusion_ops;
    // sort the fusion_ops to topo order
    while (!dq.empty()) {
        op_t *op = dq.front();
        if (visited.find(op) != visited.end()) {
            dq.pop_front();
            continue;
        }
        bool ready = true;
        auto &inputs = op->get_input_values();
        for (auto it = inputs.rbegin(); it != inputs.rend(); ++it) {
            if ((*it)->has_producer()) {
                op_t &producer = (*it)->get_producer();
                // if current op's producers have not been visited
                // current op is not ready, need to visit its
                // producers first
                if (visited.find(&producer) == visited.end()
                        && std::find(fusion_ops.begin(), fusion_ops.end(),
                                   &producer)
                                != fusion_ops.end()) {
                    dq.push_front(&producer);
                    ready = false;
                }
            }
        }
        auto &outputs = op->get_output_values();
        for (auto it = outputs.begin(); it != outputs.end(); ++it) {
            auto cons = (*it)->get_consumers();
            for (auto &con : (*it)->get_consumers()) {
                op_t &con_op = con.get_op();
                if (std::find(dq.begin(), dq.end(), &con_op) == dq.end()
                        && std::find(fusion_ops.begin(), fusion_ops.end(),
                                   &con_op)
                                != fusion_ops.end()) {
                    dq.push_back(&con_op);
                }
            }
        }
        // all producers of current op have been visited,
        // then current op is ready
        if (ready) {
            // need to check if the corresponding pb_op is
            // wildcard before adding it to reordered_fusion_ops
            size_t op_offset = 0;
            while (op_offset < fusion_ops.size()) {
                if (fusion_ops[op_offset] == op) break;
                op_offset++;
            }
            pb_op *corresponding_pb_op = pb_ops[op_offset];
            // create a temp_op to match, only wildcard can match wildcard
            op_t temp_op {op_kind::Wildcard};
            if (fusion_ops.size() == 1 // single op partition
                    || !match_node_attributes(&temp_op, corresponding_pb_op)) {
                // pb_op is not a wildcard
                op->set_attr<bool>("matched_pattern", true);
                reordered_fusion_ops.emplace_back(op);
            }
            visited.insert(op);
        }
    }
    return reordered_fusion_ops;
}

match_context_t::match_context_t(match_context_t *p_ctx, pb_node *p_graph)
    : parent_ctx {p_ctx} {
    m_graph = dynamic_cast<pb_graph_t *>(p_graph);
}

match_context_t::match_context_t(match_context_t *other_ctx) {
    parent_ctx = other_ctx->get_parent_context();
    m_graph = other_ctx->get_graph();
    in_port_map = other_ctx->in_port_map;
    out_port_map = other_ctx->out_port_map;
}

void fill_parent_io_map(
        match_context_t *local_ctx, const binding_t &local_bind) {
    auto parent_ctx = local_ctx->get_parent_context();
    auto pgraph = parent_ctx->get_graph();
    if (!pgraph) return; // pgraph is the toplevel graph (nullptr)

    auto inner_cons = pgraph->get_inner_consumers();
    if (inner_cons.empty()) { // pgraph is the main graph
        parent_ctx->in_port_map.insert(
                local_ctx->in_port_map.begin(), local_ctx->in_port_map.end());
    }
    for (size_t i = 0; i < inner_cons.size(); i++) {
        auto con_set = inner_cons[i].second;
        if (con_set.empty()) continue;
        int64_t si = static_cast<int64_t>(i);
        pb_node *con_node = con_set[0]->first;
        if (con_node == local_bind.bind_node) {
            parent_ctx->in_port_map[si] = {local_ctx->in_port_map[si].first,
                    local_ctx->in_port_map[si].second};
        }
    }
    auto inner_prods = pgraph->get_inner_producers();
    if (inner_prods.empty()) { // pgraph is the main graph
        parent_ctx->out_port_map.insert(
                local_ctx->out_port_map.begin(), local_ctx->out_port_map.end());
    }
    for (size_t i = 0; i < inner_prods.size(); i++) {
        auto prod = inner_prods[i];
        pb_node *prod_node = prod.second.first;
        int64_t si = static_cast<int64_t>(i);
        if (prod_node == local_bind.bind_node) {
            parent_ctx->out_port_map[si] = {local_ctx->out_port_map[si].first,
                    local_ctx->out_port_map[si].second};
        }
    }
}

bool match_graph_helper(const binding_t &local_bind, match_context_t *ctx,
        std::unordered_map<op_t *, pb_op *> &matched_op_map) {
    if (local_bind.bind_node->get_node_kind()
            != pb_node_kind::PB_NODE_KIND_OP) {
        if (matched_op_map.count(local_bind.bind_op)) return true;
        if (!resolve_node(local_bind, ctx, matched_op_map)) return false;
    } else {
        pb_op *bind_pb_op = dynamic_cast<pb_op *>(local_bind.bind_node);
        // if current op hasn't been visited
        if (!matched_op_map.count(local_bind.bind_op)) {
            matched_op_map[local_bind.bind_op] = bind_pb_op;
            if (!match_node(local_bind, ctx, matched_op_map)) {
                matched_op_map.erase(local_bind.bind_op);
                return false;
            } else {
                // match node success, fill local_context's io ports
                pb_graph_t *graph = ctx->get_graph();
                auto inner_cons = graph->get_inner_consumers();
                for (size_t i = 0; i < inner_cons.size(); i++) {
                    auto con_set = inner_cons[i].second;
                    for (auto &con : con_set) {
                        if (con->first == local_bind.bind_node)
                            ctx->in_port_map[i]
                                    = {local_bind.bind_op, con->second};
                    }
                }
                auto inner_pros = graph->get_inner_producers();
                for (size_t i = 0; i < inner_pros.size(); i++) {
                    auto pro = inner_pros[i].second;
                    if (pro.first == local_bind.bind_node)
                        ctx->out_port_map[i] = {local_bind.bind_op, pro.second};
                }
            }
        } else { // if current op has been visited
            if (matched_op_map[local_bind.bind_op] != bind_pb_op) {
                return false;
            } else {
                // find io ports info from history context
                match_context_t copied_ctx {ctx};
                while (copied_ctx.get_parent_context()) {
                    auto parent_ctx = copied_ctx.get_parent_context();
                    if (parent_ctx->in_port_map.empty()
                            || parent_ctx->out_port_map.empty()) {
                        copied_ctx = *parent_ctx;
                        continue;
                    }
                    if (parent_ctx->in_port_map[0].first == local_bind.bind_op
                            || parent_ctx->out_port_map[0].first
                                    == local_bind.bind_op) {
                        ctx->in_port_map.insert(parent_ctx->in_port_map.begin(),
                                parent_ctx->in_port_map.end());
                        ctx->out_port_map.insert(
                                parent_ctx->out_port_map.begin(),
                                parent_ctx->out_port_map.end());
                        break;
                    }
                    copied_ctx = *parent_ctx;
                }
            }
        }
    }

    return true;
}
//
// match nested pattern starting from initial binding
//
bool match_graph(const binding_t &bind_arg, match_context_t *ctx,
        std::unordered_map<op_t *, pb_op *> &matched_op_map) {
    binding_t local_bind = bind_arg;
    // Get initial internal node to bind
    switch (bind_arg.bind_kind) {
        case BIND_NONE: {
            local_bind.bind_node = ctx->get_graph()->get_nodes().front();
            if (!match_graph_helper(local_bind, ctx, matched_op_map))
                return false;
        } break;
        case BIND_IN: {
            auto consumers
                    = ctx->get_graph()->get_inner_consumer(bind_arg.bind_port);
            if (consumers == nullptr) return true;

            // TODO(Yixin) Currently support more than 1 consumer for in_ports
            // But will only traverse from the first consumer
            local_bind.bind_node = (*consumers)[0]->first;
            local_bind.bind_port = (*consumers)[0]->second;
            if (!match_graph_helper(local_bind, ctx, matched_op_map))
                return false;
        } break;
        case BIND_OUT: {
            std::shared_ptr<producer_t> prod
                    = ctx->get_graph()->get_inner_producer(bind_arg.bind_port);
            local_bind.bind_node = prod->first;
            local_bind.bind_port = prod->second;
            if (!match_graph_helper(local_bind, ctx, matched_op_map))
                return false;
        } break;
        default: break;
    }

    return true;
}

bool match_alternation(const binding_t &bind_arg, match_context_t *ctx,
        std::unordered_map<op_t *, pb_op *> &matched_op_map) {
    alternation_t *alt_nodes
            = dynamic_cast<alternation_t *>(bind_arg.bind_node);
    for (pb_graph_t *alt_node : alt_nodes->get_alternatives()) {
        std::unordered_map<op_t *, pb_op *> temp_op_map = matched_op_map;
        binding_t temp_bind = bind_arg;
        temp_bind.bind_node = alt_node;
        match_context_t local_ctx {ctx, temp_bind.bind_node};
        if (match_graph(temp_bind, &local_ctx, temp_op_map)) {
            matched_op_map = temp_op_map;
            fill_parent_io_map(&local_ctx, bind_arg);
            return true;
        }
    }
    return false;
}

bool match_repetition(const binding_t &bind_arg, match_context_t *parent_ctx,
        std::unordered_map<op_t *, pb_op *> &matched_op_map) {
    repetition_t *rep_node = dynamic_cast<repetition_t *>(bind_arg.bind_node);
    port_map pmap = rep_node->get_port_map();
    int64_t min_rep = rep_node->get_min_rep();
    int64_t max_rep = rep_node->get_max_rep() - 1;

    // binding_t for first iteration.
    // all iterations have same body_graph, bind_kind and bind_port
    // but they have different bind_op.
    // First iteration has the same bind_op as the repetition node.
    binding_t temp_bind = bind_arg;
    temp_bind.bind_node = rep_node->get_body();
    std::unordered_map<op_t *, pb_op *> temp_op_map = matched_op_map;

    // a merge context to tag on incremental iterations.
    match_context_t speculative_ctx {parent_ctx, temp_bind.bind_node};
    bool forward_match = temp_bind.bind_kind != BIND_OUT;

    // num of repetition blocks matched
    int64_t num_rep = 0;
    while (true) {
        match_context_t temp_ctx {speculative_ctx};
        if (!match_graph(temp_bind, &temp_ctx, temp_op_map)) break;
        ++num_rep;

        // connect previous repetition's out_port_map to
        // current repetition's in_port_map
        if (forward_match) {
            if (num_rep == 1) {
                speculative_ctx.in_port_map.insert(temp_ctx.in_port_map.begin(),
                        temp_ctx.in_port_map.end());
            }
            speculative_ctx.out_port_map.clear();
            speculative_ctx.out_port_map.insert(
                    temp_ctx.out_port_map.begin(), temp_ctx.out_port_map.end());

        } else {
            if (num_rep == 1) {
                speculative_ctx.out_port_map.insert(
                        temp_ctx.out_port_map.begin(),
                        temp_ctx.out_port_map.end());
            }
            speculative_ctx.in_port_map.clear();
            speculative_ctx.in_port_map.insert(
                    temp_ctx.in_port_map.begin(), temp_ctx.in_port_map.end());
        }

        if (num_rep == max_rep) break;

        // prepare for the next round of matching
        // Forward matching
        if (forward_match) {
            oport_t oport = pmap.first;
            op_t *current_op = temp_ctx.out_port_map[oport].first;
            if (oport >= current_op->num_outputs()) break;
            auto cons = current_op->get_output_value(static_cast<size_t>(oport))
                                ->get_consumers();
            if (cons.size() != 1) break;
            op_t *next_op = &(cons[0].get_op());
            temp_bind.bind_op = next_op;

        } else { // backward matching
            iport_t iport = pmap.second;
            op_t *current_op = temp_ctx.in_port_map[iport].first;
            if (iport >= current_op->num_inputs()) break;
            op_t *next_op
                    = &(current_op->get_input_value(static_cast<size_t>(iport))
                                    ->get_producer());
            temp_bind.bind_op = next_op;
        }
    }

    if (num_rep < min_rep) return false;
    if (num_rep == 0 && min_rep == 0) {
        // Zero trip match
        // need to forward binding to neighboring nodes
        if (forward_match) {
            // nothing needs to be matched, success
            if (bind_arg.bind_node->get_outputs().empty()) return true;
            assertm(bind_arg.bind_node->get_outputs().size() == 1,
                    "repetition is restricted to have only 1 output");
            assertm(bind_arg.bind_node->get_consumers(0)->size() == 1,
                    "repetition is restricted to have only 1 output with "
                    "only 1 consumer");
            auto cons = bind_arg.bind_node->get_consumers(pmap.first);
            if (cons) {
                binding_t con_bind = bind_arg;
                con_bind.bind_node = (*cons)[0]->first;
                if (!match_graph_helper(con_bind, parent_ctx, temp_op_map))
                    return false;
            }
        } else {
            if (bind_arg.bind_node->get_inputs().empty()) return true;
            binding_t b = bind_arg;

            auto prod = b.bind_node->get_producer(pmap.second);
            if (prod) {
                b.bind_node = prod->first;
                if (!match_graph_helper(b, parent_ctx, temp_op_map))
                    return false;
            }
        }
    } else { // num_rep > 0
        fill_parent_io_map(&speculative_ctx, bind_arg);
        if (forward_match) {
            assertm(bind_arg.bind_node->get_outputs().size() <= 1,
                    "repetition is restricted to have only 1 output");

            if (bind_arg.bind_node->get_outputs().size() == 1) {
                assertm(bind_arg.bind_node->get_consumers(0)->size() == 1,
                        "repetition is restricted to have only 1 output with "
                        "only 1 consumer");
                op_t *current_op
                        = speculative_ctx.out_port_map[pmap.first].first;
                if (!match_node_outputs(
                            current_op, rep_node, parent_ctx, temp_op_map))
                    return false;
            }

        } else {
            assertm(bind_arg.bind_node->get_outputs().size() <= 1,
                    "repetition is restricted to have only 1 output");
            if (bind_arg.bind_node->get_outputs().size() == 1) {
                op_t *current_op
                        = speculative_ctx.in_port_map[pmap.second].first;
                if (!match_node_inputs(
                            current_op, rep_node, parent_ctx, temp_op_map))
                    return false;
            }
        }
    }

    matched_op_map = temp_op_map;
    return true;
}

} // namespace pm
} // namespace utils
} // namespace impl
} // namespace graph
} // namespace dnnl
