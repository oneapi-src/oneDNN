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

#ifndef UTILS_PM_NESTED_MATCHER_HPP
#define UTILS_PM_NESTED_MATCHER_HPP

#include <deque>
#include <map>
#include <memory>
#include <utility>
#include <vector>
#include <unordered_set>

#include "interface/op.hpp"
#include "utils/pm/pbuilder.hpp"

using std::deque;
using std::make_shared;
using std::map;
using std::pair;
using std::shared_ptr;
using std::unordered_set;
using std::vector;

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
// node_tracker is used by detailed implementation and is used to track
// which pb_op a graph op is paired with, how graph op edges need to
// be matched and tracks unmatched input and output edges of a graph op.
// How a graph op need to be matched is described by input_match_task
// and output_match_task. input_match_task tells whether the input is
// commutative and whether both edges of a commutative pair is
// constrained. output_match_task tells which output value's consumers
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

using op_ptr = op_t *;

//
// The type of input constraint the pattern node imposed
// on the pattern input edge.
// No constraint, constraint on just one of the commutative pair
// or constraint on both.
// Or originally constrained on both but one of them matched.
//
enum input_match_kind {
    INPUT_MATCH_KIND_NORMAL,
    INPUT_MATCH_KIND_COMMUTATIVE_ONE_CONSTRAINT,
    INPUT_MATCH_KIND_COMMUTATIVE_TWO_CONSTRAINT,
    INPUT_MATCH_KIND_COMMUTATIVE_PINNED,
};

//
// Instructs how a pattern input edge should be matched.
// There are two ports given to cover commutative pair
//
struct input_match_task {
    input_match_kind match_kind;
    iport_t port;
    iport_t additional_port;
};

//
// Instructs how a pattern output edge should be matched
// port is the output port for the edge.
// num_consumers is the number of consumers to match
//
struct output_match_task {
    oport_t port;
    int64_t num_consumers;
};

//
// Initial binding for nested graph
// op, pb_node, binding kind, binding port
//

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
class binding {
public:
    binding(node_bind_kind p_kind, op_ptr p_op, int64_t p_op_port,
            pb_node_ptr p_node, int64_t p_port, int64_t p_idx);

    op_ptr bind_op;
    pb_node_ptr bind_node;
    node_bind_kind bind_kind;
    int64_t bind_port;
    int64_t bind_op_port;
    int64_t bind_port_user_idx;
};

//
// node_tracker keeps track of a graph op's matched state
// It consists of
// - the graph op
// - paired pattern node
// - tasks for matching graph op input edges
// - tasks for matching graph op output edges
// - graph op in edges not matched by pattern
// - graph op out edges not matched by pattern
//
class node_tracker {
public:
    node_tracker(const binding &b);
    pb_node_ptr get_node() { return m_node; };
    op_ptr get_op() { return m_op; };
    // work items for node
    // matched input edges, could be commutative
    // initialize from m_node
    deque<input_match_task> src_to_visit;
    // matched output edges, output consumers are commutative
    deque<output_match_task> dst_to_visit;
    // edges matched for an op
    // other unconstrained edges
    // initialize from m_op
    vector<bool> op_unhandled_input;
    vector<vector<bool>> op_unhandled_output;

protected:
    // node for matching
    pb_node_ptr m_node;
    // op that is tied to node
    op_ptr m_op;
};

using node_tracker_ptr = shared_ptr<node_tracker>;

using graph_port_map = map<int64_t, pair<op_ptr, int64_t>>;

//
// match context tracks a pattern graph match progress
// it consists of
// - pointer to parent match context
// - pointer to pattern graph tracked by this match context
// - a deque of graph ops to visit at this local nesting
// - a map from graph op to node_tracker
// - a list of unmatched pb_nodes
// - a map of pattern graph input port to matched graph op and
// input port
// - a map of pattern graph output port to matched graph op and
// output port
//
class match_context {
public:
    match_context(match_context *p_parent_ctx, const pb_node_ptr &p_graph);
    match_context *get_parent_context() { return parent_ctx; };
    pb_graph *get_graph() { return m_graph; };
    pb_node_ptr get_node() { return m_node; };

    // Local and Global : track alias
    // repetition needs backtracking so need a local vs global map
    map<op_ptr, node_tracker_ptr> node_tracker_map;

    // Work items
    deque<op_ptr> ops_to_visit;
    unordered_set<pb_node_ptr> unhandled_nodes;

    // This pb_graph i/o pads to actual op i/o pad mapping.
    graph_port_map in_port_map;
    graph_port_map out_port_map;

protected:
    match_context *parent_ctx;
    // Can be a nullptr if context is a global for holding graph(s)
    pb_graph *m_graph;
    pb_node_ptr m_node;
};

using match_context_ptr = match_context *;

//
// match a pattern node by checking paired graph op's attributes
// and pair pattern node neighboring nodes with graph op's neighboring
// graph ops
//
bool match_node(op_ptr o, match_context_ptr context);

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
bool match_node_inputs(op_ptr o, match_context_ptr context);

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
bool match_node_outputs(op_ptr o, match_context_ptr context);

//
// match a graph op's attributes using decision_functions of a pb_op node
//
bool match_node_attributes(op_ptr o, const pb_node_ptr &n);

//
// Trigger nested matching for non pb_op nodes
//
bool resolve_node(const binding &b, match_context_ptr context);

//
// Match a graph
//
bool match_graph(const binding &b, match_context_ptr context,
        pair<graph_port_map, graph_port_map> *io_map);

//
// similar to match_node_inputs, after a nested pattern graph is matched,
// this function pairs pattern node's input nodes with input graph ops
// to graph op tied to the pattern graphs input port.
// in_port_map is passed to look up graph op tied to graph input port.
//
bool match_graph_inputs(match_context_ptr context,
        const pb_node_ptr &graph_node, const binding &graph_binding,
        graph_port_map *in_port_map);

//
// similar to match_node_outputs, after a nested pattern graph is matched,
// this function pairs pattern node's output nodes with output graph ops
// to graph op tied to the pattern graphs output port.
// out_port_map is passed to look up graph op tied to graph output port.
//
bool match_graph_outputs(match_context_ptr context,
        const pb_node_ptr &graph_node, graph_port_map *out_port_map);

//
// match an alternation
// iterates alternatives and apply match_graph until a matching one is found.
//
bool match_alternation(const binding &b, match_context_ptr context);

//
// match a repetition including optional
// matches one iteration of repeating body at a time and matches edges
// across iterations of matched bodies.
//
bool match_repetition(const binding &b, match_context_ptr context);

//
// A match returns
// a list of graph op to pb_op pair
// And
// a list of input value_t (logical_tensor) from input graph
// a list of output value_t (logical_tensor) from intput graph
// for interfacing with the match
//
struct match {
    vector<pair<op_ptr, pb_op *>> op_pb_op_pairs;
    vector<shared_ptr<value_t>> inputs;
    vector<shared_ptr<value_t>> outputs;
};

//
// Find a match for given a graph op first_op from an input graph
// and pattern.
// auto_export_externals controls how unmatched edges after the
// matching should be interpreted.
// match_forward controls matching direction (forward, backward)
//
bool match_pattern(op_ptr first_op, const shared_ptr<pb_graph> &pattern,
        match &m, bool auto_export_externals = false,
        bool match_forward = true);

bool check_inputs_alias(std::vector<op_t *> &candidate_fusion);

} // namespace pm
} // namespace utils
} // namespace impl
} // namespace graph
} // namespace dnnl
#endif
