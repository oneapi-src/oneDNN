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

#ifndef UTILS_PM_NESTED_MATCHER_HPP
#define UTILS_PM_NESTED_MATCHER_HPP

#include <map>
#include <memory>
#include <utility>
#include <vector>
#include <unordered_map>

#include "interface/op.hpp"
#include "utils/pm/pbuilder.hpp"

namespace dnnl {
namespace graph {
namespace impl {
namespace utils {
namespace pm {

//
// Nested pattern matcher is greedy graph pattern matcher that is based
// on limited look ahead.
// Matching start with a seed graph op and pattern graph.
// Pattern graph serves as a map for guiding the matching process and
// while matching, individual graph ops will be paired up wil a pb_op
// in the pattern graph. pb_op encodes which edges need to be matched
// (topologicial constraints) and what other properties a a paired graph
// op should possess(attribute constraints).
// The pattern graph itself is a nested graph where nesting happens
// at alternation and repetition nodes.
// Each pattern graph level has a match_context associated with it.
// A match_context holds graph op to pb_op pairing map along with
// a work queue of graph ops to visit for matching at this level.
// A "binding" is pre step for graph op to pb_op pairing. At any pattern
// level, some pattern nodes are not pb_op. binding is between a
// graph op and a pattern node (not necessarily a pb_op).
// Matching is driven by binding. If binding is not to a pb_op,
// resolve_node() will be called which invokes matching at the nesting
// pattern level. This process will invoke match_graph, match_alternation
// and match_repetition. At the end of those functions, matcher will
// will treat the local match like a fused graph op and bind neighboring
// ops of the conceptually fused op and corresponding pattern nodes.
// graph op binded with a pb_op will be handled by match_node()
// which checks graph op attributes with the pb_op's decision_functions
// and bind neighboring ops and corresponding pattern nodes.
// node_tracker_t is used by detailed implementation and is used to track
// which pb_op a graph op is paired with, how graph op edges need to
// be matched and tracks unmatched input and output edges of a graph op.
// How a graph op need to be matched is described by input_match_task_t
// and output_match_task_t. input_match_task_t tells whether the input is
// commutative and whether both edges of a commutative pair is
// constrained. output_match_task_t tells which output value's consumers
// need to be match and which one has already been matched.
// The matcher is predictive since it has to decide how a graph op and
// pattern node will be paired without looking at the entire graph and
// only relying on local heuristics. graph op input matching becomes
// predictive if the paired pb_op's input is commutative.
// The matcher in this case, relies on the input to that pattern to
// be a pb_op. Then decision_functions of the the input pb_op is used
// for deciding how the commutative inputs should be wired.
// Similar situation happens when a graph op output value has multiple
// consumer graph ops. The consumer pattern nodes need to be pb_ops.
// those pb_ops' decision functions will be used to pair graph op
// consumers and pattern node consumers.
// Input prediction will mostly impact backward matching and Output
// prediction will mostly impact forward matching.
// In case of training network where there are side output edges not
// constrained by a pattern, backward matching may give a better
// matching result.
// Overall, the matcher can be improved with better prediction
// heuristics.
// Note that incorrect prediction does not result in false positive
// matching. It just guides the matching process in the wrong direction
// and may result in not matching a graph that satisfies the pattern.
//
//
// Initial binding for nested graph
// op, pb_node, binding kind, binding port
//
// Tells how a pattern node was brought in (reached)
// during matching process.
// BIND_IN: node reached by in port
// BIND_OUT: node reached by out port
// BIND_NONE: Start of matching. (First of last in list of nodes.)
//
enum node_bind_kind {
    BIND_IN,
    BIND_OUT,
    BIND_NONE,
};

//
// Bind a graph op and pattern node for matching
// pattern node may not be a pb_op. In that case, nested(recursive)
// matching happens.
//
class binding_t {
public:
    binding_t(node_bind_kind p_kind, op_t *p_op, int64_t p_op_port,
            pb_node *p_node, int64_t p_port);

    op_t *bind_op;
    pb_node *bind_node;
    node_bind_kind bind_kind;
    int64_t bind_port;
    int64_t bind_op_port;
};

using graph_port_map = std::unordered_map<int64_t, std::pair<op_t *, int64_t>>;

//
// match context tracks a pattern graph match progress
// it consists of
// - pointer to parent match context
// - pointer to pattern graph tracked by this match context
// - a deque of graph ops to visit at this local nesting
// - a map from graph op to node_tracker_t
// - a list of unmatched pb_nodes
// - a map of pattern graph input port to matched graph op and
// input port
// - a map of pattern graph output port to matched graph op and
// output port
//
class match_context_t {
public:
    match_context_t(match_context_t *p_parent_ctx, pb_node *p_graph);
    match_context_t(match_context_t *other_ctx);
    match_context_t *get_parent_context() { return parent_ctx; };
    pb_graph_t *get_graph() { return m_graph; };

    // This pb_graph_t i/o pads to actual op i/o pad mapping.
    graph_port_map in_port_map;
    graph_port_map out_port_map;

protected:
    match_context_t *parent_ctx;
    // Can be a nullptr if context is a global for holding graph(s)
    pb_graph_t *m_graph;
};

bool has_commutative_inputs(op_t *op);

bool has_variadic_inputs(op_t *op);

// check if a pb_node is optional and its all consumers are optional
bool check_is_optional(pb_node *n);
//
// match a pattern node by checking paired graph op's attributes
// and pair pattern node neighboring nodes with graph op's neighboring
// graph ops
//
bool match_node(const binding_t &b, match_context_t *context,
        std::unordered_map<op_t *, pb_op *> &matched_op_map);

//
// pair pattern node input nodes (producers) with the paired op's
// input ops. If node has commutative inputs, input ops are paired by using
// a heuristic that assuming producer node is a pb_op and use it's attribute
// checker for finding an input op to pair.
// This puts limitation on the type of patterns that can be matched by
// backward matching.
// More advanced heuristics or a manually provided predicate set for
// non pb_op pattern nodes can be used to relax this limitation
// If producer is nested node, "resolve_node" is called to dispatch
// to one of the proper handlers "match_graph", "match_alternation"
// and "match_repetition" based on node type. This triggers a nested/recursive
// matching.
//
bool match_node_inputs(op_t *o, pb_node *node, match_context_t *context,
        std::unordered_map<op_t *, pb_op *> &matched_op_map);
//
// pair pattern node output nodes (consumers) with the paired op's
// output ops. If node output has multiple consumers. output ops are paired
// by using a heuristic that assumes consumer node is a pbb_op and use it's
// attribute checker for finding an output op to pair.
// This puts limitation on the type of patterns that can be matched by
// forward matching.
// More advanced heuristics or a manually provided predicate set for
// non pb_op pattern nodes can be used to relax this limitation
// If producer is nested node, "resolve_node" is called to dispatch
// to one of the proper handlers "match_graph", "match_alternation"
// and "match_repetition" based on node type. This triggers a nested/recursive
// matching.
//
bool match_node_outputs(op_t *o, pb_node *node, match_context_t *context,
        std::unordered_map<op_t *, pb_op *> &matched_op_map);

//
// match a graph op's attributes using decision_functions of a pb_op node
//
bool match_node_attributes(op_t *o, pb_node *n);

//
// Trigger nested matching for non pb_op nodes
//
bool resolve_node(const binding_t &b, match_context_t *context,
        std::unordered_map<op_t *, pb_op *> &matched_op_map);

//
// Match a graph
//
bool match_graph(const binding_t &b, match_context_t *context,
        std::unordered_map<op_t *, pb_op *> &matched_op_map);

bool match_graph_helper(const binding_t &local_bind, match_context_t *ctx,
        std::unordered_map<op_t *, pb_op *> &matched_op_map);
//
// match an alternation
// iterates alternatives and apply match_graph until a matching one is found.
//
bool match_alternation(const binding_t &b, match_context_t *context,
        std::unordered_map<op_t *, pb_op *> &matched_op_map);

//
// match a repetition including optional
// matches one iteration of repeating body at a time and matches edges
// across iterations of matched bodies.
//
bool match_repetition(const binding_t &b, match_context_t *context,
        std::unordered_map<op_t *, pb_op *> &matched_op_map);

//
// Find a match for given a graph op first_op from an input graph
// and pattern.
bool match_pattern(op_t *first_op, const std::shared_ptr<pb_graph_t> &pattern,
        std::vector<op_t *> &fusion_ops);

inline std::vector<op_t *> reorder_matched_list(
        const std::unordered_map<op_t *, pb_op *> &matched_op_map);

void fill_parent_io_map(match_context_t *local_ctx);

} // namespace pm
} // namespace utils
} // namespace impl
} // namespace graph
} // namespace dnnl
#endif
