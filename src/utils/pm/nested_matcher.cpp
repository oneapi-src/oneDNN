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

namespace {
// check if an op's inputs are commutative
bool has_commutative_inputs(op_t *op) {
    static const std::unordered_set<op_kind_t> commutative_kinds {op_kind::Add,
            op_kind::Multiply, op_kind::Maximum, op_kind::Minimum};
    return commutative_kinds.count(op->get_kind());
}

// check if a pb_node is optional and its all consumers are optional
// also record the nodes that are likely to be inner producers
bool get_optional_outputs(
        pb_node_t *n, std::unordered_set<pb_node_t *> &opt_nodes) {
    if (n->get_node_kind() != pb_node_kind::PB_NODE_KIND_REPETITION)
        return false;
    repetition_t *rep_node = dynamic_cast<repetition_t *>(n);
    if (rep_node->get_min_rep() != 0) return false;

    std::vector<std::pair<oport_t, consumers_t>> node_outputs
            = n->get_outputs();
    if (!node_outputs.empty()) {
        for (auto &node_output : node_outputs) {
            for (size_t i = 0; i < node_output.second.size(); i++) {
                bool is_optional = get_optional_outputs(
                        node_output.second[i]->first, opt_nodes);
                if (!is_optional) return false;
            }
        }
    } else
        // only a node without outputs is likely to be a inner producer
        opt_nodes.emplace(n);
    return true;
}
} // namespace

binding_t::binding_t(node_bind_kind p_kind, op_t *p_op, int64_t p_op_port,
        pb_node_t *p_node, int64_t p_port)
    : bind_op {p_op}
    , bind_node {p_node}
    , bind_kind {p_kind}
    , bind_port {p_port}
    , bind_op_port {p_op_port} {}

match_context_t::match_context_t(match_context_t *p_ctx, pb_node_t *p_graph)
    : parent_ctx {p_ctx} {
    graph_ = dynamic_cast<pb_graph_t *>(p_graph);
}

bool match_node_attributes(op_t *op, pb_node_t *node) {
    size_t n_func = node->get_num_decision_functions();
    for (size_t i = 0; i < n_func; i++) {
        if (!(node->get_decision_function(i)(op))) { return false; }
    }
    return true;
}

bool match_node_inputs(op_t *op, pb_node_t *node, match_context_t *ctx,
        std::unordered_map<op_t *, pb_op_t *> &matched_op_map) {
    std::vector<std::pair<iport_t, producer_t>> node_inputs
            = node->get_inputs();
    if (node_inputs.empty()) return true;

    std::unordered_map<op_t *, pb_op_t *> copied_op_map = matched_op_map;

    // a lambda function to match each of the inputs in op and node
    auto match_input
            = [&](size_t op_input_offset, size_t node_input_offset) -> bool {
        pb_node_t *in_node = node_inputs[node_input_offset].second.first;
        std::shared_ptr<value_t> op_in_value
                = op->get_input_value(op_input_offset);
        if (!op_in_value->has_producer()) {
            // pattern node has producer while graph op
            // doesn't have. In this case, only optional
            // can survive
            if (in_node->get_node_kind()
                    != pb_node_kind::PB_NODE_KIND_REPETITION)
                return false;
            repetition_t *rep_node = dynamic_cast<repetition_t *>(in_node);
            if (rep_node->get_min_rep() != 0) return false;
        } else {
            op_t *in_op = op->get_input_op(op_input_offset);
            size_t in_op_oport = op_in_value->get_offset();
            oport_t in_node_oport
                    = node_inputs[node_input_offset].second.second;
            binding_t in_bind(
                    BIND_OUT, in_op, in_op_oport, in_node, in_node_oport);
            if (!match_graph_helper(in_bind, ctx, copied_op_map)) {
                // match failure, check if the node is optional
                // TODO(Zitian): check not only the direct producer
                if (in_node->get_node_kind()
                        == pb_node_kind::PB_NODE_KIND_REPETITION) {
                    repetition_t *rep_node
                            = dynamic_cast<repetition_t *>(in_node);
                    if (rep_node->get_min_rep() == 0) return true;
                }
                return false;
            }
        }
        return true;
    };

    if (node->get_inputs().size() == VARIADIC_INPUT_NUM) {
        assertm(op->num_inputs() < VARIADIC_INPUT_NUM,
                "variadic input num should be larger than actual op's num of "
                "inputs");
        for (size_t i = 0; i < node_inputs.size(); ++i) {
            if (op->num_inputs() < i + 1) break;
            iport_t node_iport = node_inputs[i].first;
            if (!match_input(node_iport, i)) return false;
        }
    } else if (!has_commutative_inputs(op)) {
        for (size_t i = 0; i < node_inputs.size(); ++i) {
            if (op->num_inputs() < i + 1) return false;
            iport_t node_iport = node_inputs[i].first;
            if (op->num_inputs() < node_iport + 1) return false;
            if (!match_input(node_iport, i)) return false;
        }
    } else { // commutative ops need to consider switching inputs
        // TODO(Zitian): fill input port map
        // if an optional subpattern is not matched
        size_t matched_op_input_offset = op->num_inputs(); // init with illegal
        for (size_t node_input_offset = 0;
                node_input_offset < node_inputs.size(); ++node_input_offset) {
            for (size_t op_input_offset = 0; op_input_offset < op->num_inputs();
                    ++op_input_offset) {
                if (op_input_offset == matched_op_input_offset) {
                    matched_op_input_offset = op->num_inputs();
                    continue;
                }
                if (match_input(op_input_offset, node_input_offset)) {
                    matched_op_input_offset = op_input_offset;
                    break;
                }
            }
            if (matched_op_input_offset == op->num_inputs()) return false;
        }
    }

    matched_op_map = copied_op_map;
    return true;
}

bool match_node_outputs(op_t *op, pb_node_t *node, match_context_t *ctx,
        std::unordered_map<op_t *, pb_op_t *> &matched_op_map) {
    std::vector<std::pair<oport_t, consumers_t>> node_outputs
            = node->get_outputs();
    if (node_outputs.empty()) return true;

    // the worst situation for matching node output is that pattern node
    // output and graph op output cannot be matched, in this case,
    // only optional can survive.
    std::vector<std::pair<bool, std::unordered_set<pb_node_t *>>>
            all_opt_out_nodes {};
    bool support_optional = true;
    std::vector<size_t> output_ranges {};
    size_t output_count = 0;
    for (const auto &node_output : node_outputs) {
        output_ranges.emplace_back(output_count);
        for (const auto &con : node_output.second) {
            pb_node_t *out_node = con->first;
            all_opt_out_nodes.emplace_back(
                    false, std::initializer_list<pb_node_t *> {});
            bool is_optional = get_optional_outputs(
                    out_node, all_opt_out_nodes.back().second);
            if (!is_optional) {
                support_optional = false;
                all_opt_out_nodes.back().second.clear();
            } else
                all_opt_out_nodes.back().first = true;
            ++output_count;
        }
    }

    pb_graph_t *graph = ctx->get_graph();
    auto inner_pros = graph->get_inner_producers();
    std::unordered_map<pb_node_t *, std::pair<size_t, oport_t>> inner_outputs;
    for (size_t i = 0; i < inner_pros.size(); i++) {
        auto pro = inner_pros[i].second;
        inner_outputs.emplace(
                pro.first, std::pair<size_t, oport_t>(i, pro.second));
    }

    std::unordered_map<op_t *, pb_op_t *> copied_op_map = matched_op_map;

    auto fill_consumer_outportmaps
            = [&](op_t *current_op, match_context_t *current_ctx,
                      const std::unordered_set<pb_node_t *> &out_nodes,
                      const std::unordered_map<pb_node_t *,
                              std::pair<size_t, oport_t>> &inner_outputs)
            -> void {
        // fill the output map to all the optional subgraphs in out_nodes
        for (const auto &on : out_nodes)
            if (inner_outputs.find(on) != inner_outputs.end())
                current_ctx->out_port_map[inner_outputs.at(on).first]
                        = {current_op, inner_outputs.at(on).second};
    };

    // match output for node and op
    for (size_t i = 0; i < node_outputs.size(); ++i) {
        auto node_output = node_outputs[i];
        size_t node_oport = node_output.first;
        if (op->num_outputs() < node_oport + 1) {
            if (support_optional) {
                for (const auto &opt_nodes : all_opt_out_nodes)
                    fill_consumer_outportmaps(
                            op, ctx, opt_nodes.second, inner_outputs);
            }
            return support_optional;
        }

        std::shared_ptr<value_t> op_out_value
                = op->get_output_value(node_oport);
        std::unordered_set<size_t> node_oport_matched_cons;
        // match the op consumers one by one
        for (size_t j = 0; j < op_out_value->get_consumers().size(); j++) {
            auto op_consumer = op_out_value->get_consumers()[j];
            op_t *out_op = &(op_consumer.get_op());
            bool consumer_matched = false;

            for (size_t k = 0; k < node_output.second.size(); k++) {
                auto node_consumer = node_output.second[k];
                pb_node_t *out_node = node_consumer->first;
                // check if the out_node has been matched by previous out_ops
                if (node_oport_matched_cons.count(k)) continue;
                binding_t out_bind(BIND_IN, out_op,
                        int64_t(op_consumer.get_offset()), out_node,
                        node_consumer->second);
                if (!match_graph_helper(out_bind, ctx, copied_op_map)) {
                    continue;
                } else {
                    consumer_matched = true;
                    node_oport_matched_cons.insert(k);
                    break;
                }
            }

            if (!consumer_matched) {
                // TODO(Yixin): temporary fix sigmoid + multiply = swish
                // After successfully matching sigmoid, multiply is also
                // matched because multiply is sigmoid's out_op.
                // When matching multiply, check if it is already in the
                // matched op_map, if yes, the match is OK
                if (node_output.second.size() == 1
                        && node_output.second[0]->first->get_node_kind()
                                != pb_node_kind::PB_NODE_KIND_OP
                        && copied_op_map.count(out_op))
                    continue;
                //if it's the allow_external_output case, then it's fine
                pb_op_t *p_op = copied_op_map[op];
                const std::unordered_set<oport_t> &external_outputs
                        = p_op->get_allowed_external_outputs();
                if (!external_outputs.empty()
                        && external_outputs.find(node_oport)
                                != external_outputs.end()) {
                    continue;
                }

                // the consumer op not matched,
                // success if all the consumer nodes are optional
                if (support_optional) {
                    // success matched, should also fill the outport map
                    for (const auto &opt_nodes : all_opt_out_nodes)
                        fill_consumer_outportmaps(
                                op, ctx, opt_nodes.second, inner_outputs);
                }
                return support_optional;
            }
        }

        // check if not all consumers of node output are matched
        if (node_oport_matched_cons.size() != node_output.second.size()) {
            // if an output node is not matched, check whether it is optional
            std::vector<size_t> unmatched_opt_nodes {};
            for (size_t k = 0; k < node_output.second.size(); ++k) {
                auto node_consumer = node_output.second[k];
                if (node_oport_matched_cons.find(k)
                        == node_oport_matched_cons.end()) {
                    if (all_opt_out_nodes[k].first)
                        unmatched_opt_nodes.emplace_back(output_ranges[i] + k);
                    else
                        return false;
                }
            }

            // all the unmatched nodes are optional, the match is still success
            // so we can fill the outport map
            for (auto out_idx : unmatched_opt_nodes)
                fill_consumer_outportmaps(op, ctx,
                        all_opt_out_nodes[out_idx].second, inner_outputs);
        }
    }

    matched_op_map = copied_op_map;
    return true;
}

bool check_cyclic(
        op_t *op, const std::unordered_map<op_t *, pb_op_t *> &matched_op_map) {
    // internal_ops: ops that are in the matched graph
    std::unordered_set<op_t *> internal_ops;
    for (const auto &kv : matched_op_map)
        internal_ops.insert(kv.first);

    for (size_t op_input_offset = 0; op_input_offset < op->num_inputs();
            ++op_input_offset) {
        std::shared_ptr<value_t> op_in_value
                = op->get_input_value(op_input_offset);
        if (op_in_value->has_producer()) {
            op_t *in_op = op->get_input_op(op_input_offset);
            if (internal_ops.find(in_op) == internal_ops.end()) {
                // in_op is an external_op.
                // recursively traverse the producers of in_op
                // check if any producer is among internal_ops
                // if yes, a cycle exists.
                status_t ret = topo_order_visit({in_op}, [&](op_t *temp_op) {
                    if (internal_ops.find(temp_op) != internal_ops.end())
                        return status::invalid_graph;
                    return status::success;
                });
                // there's a cycle
                if (ret != status::success) return true;
            }
        }
    }

    // no cycle
    return false;
}

bool match_node(const binding_t &b, match_context_t *ctx,
        std::unordered_map<op_t *, pb_op_t *> &matched_op_map) {
    if (b.bind_op == nullptr) return false;
    if (b.bind_node == nullptr) return false;
    if (b.bind_op->get_partition() != nullptr) return false;
    if (b.bind_op->has_attr(op_attr::matched)) return false;
    if (!has_commutative_inputs(b.bind_op) && b.bind_op_port != b.bind_port)
        return false;

    if (!match_node_attributes(b.bind_op, b.bind_node)) return false;

    if (!match_node_inputs(b.bind_op, b.bind_node, ctx, matched_op_map))
        return false;

    if (check_cyclic(b.bind_op, matched_op_map)) return false;

    if (!match_node_outputs(b.bind_op, b.bind_node, ctx, matched_op_map))
        return false;

    return true;
}

bool resolve_node(const binding_t &bind_arg, match_context_t *ctx,
        std::unordered_map<op_t *, pb_op_t *> &matched_op_map) {
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
    std::unordered_map<op_t *, pb_op_t *> matched_op_map;
    if (!match_graph(init_bind, &init_ctx, matched_op_map)) { return false; }

    fusion_ops = reorder_matched_list(matched_op_map);

    return true;
}

inline std::vector<op_t *> reorder_matched_list(
        const std::unordered_map<op_t *, pb_op_t *> &matched_op_map) {
    // split ops and pb_op_ts
    std::vector<op_t *> fusion_ops;
    std::vector<pb_op_t *> pb_op_ts;
    for (auto kv : matched_op_map) {
        fusion_ops.push_back(kv.first);
        pb_op_ts.push_back(kv.second);
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
            // need to check if the corresponding pb_op_t is
            // wildcard before adding it to reordered_fusion_ops
            auto iter = std::find(fusion_ops.begin(), fusion_ops.end(), op);
            size_t index = iter - fusion_ops.begin();
            pb_op_t *corresponding_pb_op_t = pb_op_ts[index];
            // create a temp_op to match, only wildcard can match wildcard
            op_t temp_op {op_kind::Wildcard};
            if (fusion_ops.size() == 1 // single op partition
                    || !match_node_attributes(
                            &temp_op, corresponding_pb_op_t)) {
                // pb_op_t is not a wildcard
                op->set_attr<bool>(op_attr::matched, true);
                reordered_fusion_ops.emplace_back(op);
            }
            visited.insert(op);
        }
    }
    return reordered_fusion_ops;
}

void fill_parent_io_map(
        match_context_t *local_ctx, const binding_t &local_bind) {
    auto parent_ctx = local_ctx->get_parent_context();
    auto pgraph = parent_ctx->get_graph();
    if (!pgraph) return; // pgraph is the toplevel graph (nullptr)

    auto inner_cons = pgraph->get_inner_consumers();
    for (size_t i = 0; i < inner_cons.size(); i++) {
        auto con_set = inner_cons[i].second;
        if (con_set.empty()) continue;
        int64_t si = static_cast<int64_t>(i);
        pb_node_t *con_node = con_set[0]->first;
        if (con_node == local_bind.bind_node) {
            parent_ctx->in_port_map[si] = {local_ctx->in_port_map[si].first,
                    local_ctx->in_port_map[si].second};
        }
    }
    auto inner_prods = pgraph->get_inner_producers();
    for (size_t i = 0; i < inner_prods.size(); i++) {
        auto prod = inner_prods[i];
        pb_node_t *prod_node = prod.second.first;
        int64_t si = static_cast<int64_t>(i);
        if (prod_node == local_bind.bind_node) {
            parent_ctx->out_port_map[si] = {local_ctx->out_port_map[si].first,
                    local_ctx->out_port_map[si].second};
        }
    }
}

bool match_graph_helper(const binding_t &local_bind, match_context_t *ctx,
        std::unordered_map<op_t *, pb_op_t *> &matched_op_map) {
    if (local_bind.bind_node->get_node_kind()
            != pb_node_kind::PB_NODE_KIND_OP) {
        if (matched_op_map.count(local_bind.bind_op)) return true;
        if (!resolve_node(local_bind, ctx, matched_op_map)) return false;
    } else {
        pb_op_t *bind_pb_op = dynamic_cast<pb_op_t *>(local_bind.bind_node);
        // if current op has been visited
        if (matched_op_map.count(local_bind.bind_op)) {
            return matched_op_map[local_bind.bind_op] == bind_pb_op;
        }
        // if current op hasn't been visited
        matched_op_map[local_bind.bind_op] = bind_pb_op;
        if (!match_node(local_bind, ctx, matched_op_map)) {
            matched_op_map.erase(local_bind.bind_op);
            return false;
        }
        // match node success, fill local_context's io ports
        pb_graph_t *graph = ctx->get_graph();
        auto inner_cons = graph->get_inner_consumers();
        for (size_t i = 0; i < inner_cons.size(); i++) {
            auto con_set = inner_cons[i].second;
            for (auto &con : con_set) {
                if (con->first == local_bind.bind_node)
                    ctx->in_port_map[i] = {local_bind.bind_op, con->second};
            }
        }
        auto inner_pros = graph->get_inner_producers();
        for (size_t i = 0; i < inner_pros.size(); i++) {
            auto pro = inner_pros[i].second;
            if (pro.first == local_bind.bind_node)
                ctx->out_port_map[i] = {local_bind.bind_op, pro.second};
        }
    }
    return true;
}

bool match_graph(const binding_t &bind_arg, match_context_t *ctx,
        std::unordered_map<op_t *, pb_op_t *> &matched_op_map) {
    binding_t local_bind = bind_arg;
    // Get initial internal node to bind
    switch (bind_arg.bind_kind) {
        case BIND_NONE: {
            local_bind.bind_node = ctx->get_graph()->get_nodes().front();
        } break;
        case BIND_IN: {
            auto consumers
                    = ctx->get_graph()->get_inner_consumer(bind_arg.bind_port);
            // TODO(Yixin) Currently support more than 1 consumer for in_ports
            // But will only traverse from the first consumer
            local_bind.bind_node = (*consumers)[0]->first;
            local_bind.bind_port = (*consumers)[0]->second;
        } break;
        case BIND_OUT: {
            std::shared_ptr<producer_t> prod
                    = ctx->get_graph()->get_inner_producer(bind_arg.bind_port);
            local_bind.bind_node = prod->first;
            local_bind.bind_port = prod->second;
        } break;
        default: {
            return false;
        } break;
    }

    return match_graph_helper(local_bind, ctx, matched_op_map);
}

bool match_alternation(const binding_t &bind_arg, match_context_t *ctx,
        std::unordered_map<op_t *, pb_op_t *> &matched_op_map) {
    alternation_t *alt_nodes
            = dynamic_cast<alternation_t *>(bind_arg.bind_node);
    for (pb_graph_t *alt_node : alt_nodes->get_alternatives()) {
        std::unordered_map<op_t *, pb_op_t *> temp_op_map = matched_op_map;
        binding_t temp_bind = bind_arg;
        temp_bind.bind_node = alt_node;
        match_context_t local_ctx {ctx, temp_bind.bind_node};
        if (match_graph(temp_bind, &local_ctx, temp_op_map)) {
            matched_op_map = temp_op_map;
            fill_parent_io_map(&local_ctx, bind_arg);
            if (bind_arg.bind_kind != BIND_OUT) {
                // alternation is restricted to have only 1 out port
                if (local_ctx.out_port_map.size() != 1) return false;
                op_t *current_op = local_ctx.out_port_map[0].first;
                return match_node_outputs(
                        current_op, bind_arg.bind_node, ctx, matched_op_map);
            } else {
                // alternation is restricted to have only 1 in port
                if (local_ctx.in_port_map.size() != 1) return false;
                op_t *current_op = local_ctx.in_port_map[0].first;
                return match_node_inputs(
                        current_op, bind_arg.bind_node, ctx, matched_op_map);
            }
        }
    }
    return false;
}

bool match_repetition(const binding_t &bind_arg, match_context_t *parent_ctx,
        std::unordered_map<op_t *, pb_op_t *> &matched_op_map) {
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
    std::unordered_map<op_t *, pb_op_t *> temp_op_map = matched_op_map;

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
            temp_bind.bind_kind = BIND_IN;
            oport_t oport = pmap.first;
            op_t *current_op = temp_ctx.out_port_map[oport].first;
            if (oport >= current_op->num_outputs()) break;
            auto cons = current_op->get_output_value(static_cast<size_t>(oport))
                                ->get_consumers();
            if (cons.empty()) break;
            if (cons.size() == 1) {
                op_t *next_op = &(cons[0].get_op());
                temp_bind.bind_op = next_op;
                temp_bind.bind_op_port = cons[0].get_offset();
            } else {
                // More than 1 consumers. In this case, needs to check
                // if the last node of previous match accepts external
                // output. If no, break
                pb_op_t *current_pb_op = temp_op_map[current_op];
                const std::unordered_set<oport_t> &external_outputs
                        = current_pb_op->get_allowed_external_outputs();
                if (external_outputs.empty()
                        || external_outputs.find(oport)
                                == external_outputs.end()) {
                    break;
                }
                // If yes, decide which one of the consumers will be used
                // for next round's match
                iport_t iport = pmap.second;
                // start op for last round's match
                op_t *start_op = temp_ctx.in_port_map[iport].first;
                pb_op_t *start_pb_op = temp_op_map[start_op];
                op_t *next_op = nullptr;
                size_t next_op_iport = 0;
                for (auto &con : cons) {
                    if (match_node_attributes(&con.get_op(), start_pb_op)) {
                        next_op = &(con.get_op());
                        next_op_iport = con.get_offset();
                        break;
                    }
                }
                if (!next_op) break;
                temp_bind.bind_op = next_op;
                temp_bind.bind_op_port = next_op_iport;
            }
        } else { // backward matching
            temp_bind.bind_kind = BIND_OUT;
            iport_t iport = pmap.second;
            op_t *current_op = temp_ctx.in_port_map[iport].first;
            if (iport >= current_op->num_inputs()) break;
            auto in_value
                    = current_op->get_input_value(static_cast<size_t>(iport));
            temp_bind.bind_op = &(in_value->get_producer());
            temp_bind.bind_op_port = in_value->get_offset();
        }
    }

    if (num_rep < min_rep) return false;
    if (num_rep == 0 && min_rep == 0) {
        // Zero trip match
        // need to forward binding to neighboring nodes
        if (forward_match) {
            // nothing matched and nothing needs to be matched, failed
            if (bind_arg.bind_node->get_outputs().empty()) return false;
            assertm(bind_arg.bind_node->get_outputs().size() == 1,
                    "repetition is restricted to have only 1 output");
            assertm(bind_arg.bind_node->get_consumers(0)->size() == 1,
                    "repetition is restricted to have only 1 output with "
                    "only 1 consumer");
            auto cons = bind_arg.bind_node->get_consumers(pmap.first);
            binding_t con_bind = bind_arg;
            con_bind.bind_node = (*cons)[0]->first;
            if (!match_graph_helper(con_bind, parent_ctx, temp_op_map))
                return false;
        } else {
            if (bind_arg.bind_node->get_inputs().empty()) return false;
            binding_t b = bind_arg;
            b.bind_node = b.bind_node->get_producer(pmap.second)->first;
            if (!match_graph_helper(b, parent_ctx, temp_op_map)) return false;
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
