/*******************************************************************************
* Copyright 2020 Intel Corporation
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
#ifndef LLGA_BACKEND_PASS_PATTERN_UTILS_HPP
#define LLGA_BACKEND_PASS_PATTERN_UTILS_HPP

#include <deque>
#include <iostream>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>
#include <unordered_set>

#include "interface/graph.hpp"
#include "interface/ir.hpp"

namespace dnnl {
namespace graph {
namespace impl {
// FRequirement: A function to check if graph node can meet
// the requirement of pattern node
// Should be defined when register passes
using FRequirement = pass::FRequirement;

/**
 * Operators set for checking number of op inputs
 */
struct op_set {
    /**
     * Check if operator is binary (has two swapable inputs).
     *
     * @param kind operator kind
     * @return whether the operator is binary
     */
    static bool is_binary(op_kind_t op_kind) {
        static const std::set<op_kind_t> supported_binary_ops {op_kind::Add,
                op_kind::Multiply, op_kind::Maximum, op_kind::Minimum};
        return supported_binary_ops.count(op_kind);
    }
};

//check if the input order of the "binary" node in pattern need to be swapped
template <typename getinput>
bool should_swap_inputs(
        node_t *graph_node, node_t *pattern_node, getinput get_input) {
    const auto get_op_kind_ = [&](node_t *node_, int idx) {
        return get_input(node_, idx)->get_op_kind();
    };
    const auto pin_0 = get_op_kind_(pattern_node, 0);
    const auto pin_1 = get_op_kind_(pattern_node, 1);
    const auto gin_0 = get_op_kind_(graph_node, 0);
    const auto gin_1 = get_op_kind_(graph_node, 1);

    if (pin_0 == op_kind::any) { // if pin_0 accepts any
        // check if the corresponding inputs of pattern and graph match,
        // if they do, no swap. If they don't, then check if the opposite
        // inputs match or not
        if (pin_1 != gin_1 && pin_1 == gin_0)
            return true;
        else
            return false;
    } else if (pin_1 == op_kind::any) { // if pin_1 accepts any
        if (pin_0 != gin_0 && pin_0 == gin_1)
            return true;
        else
            return false;
    } else { // if no any inputs in pattern
        if ((pin_0 != gin_0 || pin_1 != gin_1)
                && (pin_0 == gin_1 && pin_1 == gin_0))
            return true;
        else
            return false;
    }
}

//handles special cases in binary ops - allows any order of op inputs
template <typename indegree, typename getinput>
void special_case_handle(node_t *graph_node, node_t *pattern_node,
        indegree in_degree, getinput get_input) {
    const auto pin_deg = in_degree(pattern_node);
    const auto gin_deg = in_degree(graph_node);
    const auto expected_deg = 2;

    if (op_set::is_binary(pattern_node->get_op_kind())
            && expected_deg == pin_deg && pin_deg == gin_deg) {
        if (should_swap_inputs(graph_node, pattern_node, get_input)) {
            pattern_node->swap_input_value(0, 1);
        }
    }
}

/*!
* \brief Function to do comparison between a graph
         and a pattern. It will search from a graph node,
         and compare its inputs / outputs with the nodes in
         the pattern, until all the nodes in the pattern are
         exhausted.
* \param graph_node the node in the graph to compare from
* \param pattern_node the node in the pattern to compare from
* \param candidate_fusion the vector stores the matched nodes
* \param selected the set stores the nodes have been selected
* \param hash_func the hash function
* \param out_degree the output degree
* \param get_output the function to get a specific output
* \param in_degree the input degree
* \param get_input the function to get a specific input
* \tparam bool whether the pattern is matched
*/
template <typename hashtype, typename hashfunc, typename outdegree,
        typename getoutput, typename indegree, typename getinput>
bool per_node_comp_(node_t *graph_node, node_t *pattern_node,
        std::vector<node_t *> &candidate_fusion,
        std::unordered_set<node_t *> &selected, hashfunc hash_func,
        outdegree out_degree, getoutput get_output, indegree in_degree,
        getinput get_input) {
    std::deque<std::pair<node_t *, std::pair<uint32_t, uint32_t>>>
            pattern_queue;
    std::deque<node_t *> node_queue;
    //if a node have been visited
    std::unordered_set<hashtype> visited;
    bool graph_is_pattern = graph_node == pattern_node;
    std::set<std::string> expected {"FRequirement"};

    hashtype pattern_starter_hash = hash_func(pattern_node);
    pattern_queue.push_back(std::make_pair(
            pattern_node, std::make_pair(0, 0))); //node, output_idx, input_idx
    visited.insert(pattern_starter_hash);
    node_queue.push_back(graph_node);

    while (!pattern_queue.empty()) {
        std::pair<node_t *, std::pair<uint32_t, uint32_t>> &pfront
                = pattern_queue.front();
        node_t *nfront = node_queue.front();

        // check if graph node is an unvisited node and can meet the
        // requirement of pattern node
        if (pfront.first->get_op_kind() != op_kind::any
                && (selected.count(nfront) != 0 || nfront->has_attr("backend")
                        || nfront->get_op_kind() != pfront.first->get_op_kind()
                        || !pfront.first->has_same_attr_values(
                                *nfront, expected))) {
            return false;
        }
        if (!graph_is_pattern && pfront.first->has_attr("FRequirement")) {
            auto req = pfront.first->get_attr<FRequirement>("FRequirement");
            if (!req(nfront)) return false;
        }

        // handle the case that "Add" node's input tensor's order
        // is different from it in graph
        special_case_handle(nfront, pfront.first, in_degree, get_input);

        if (pfront.first->get_op_kind() == op_kind::any) {
            node_queue.pop_front();
            pattern_queue.pop_front();
        } else if (pfront.second.first == out_degree(pfront.first)
                && pfront.second.second == in_degree(pfront.first)) {
            candidate_fusion.emplace_back(nfront);
            node_queue.pop_front();
            pattern_queue.pop_front();
        } else if (pfront.second.first != out_degree(pfront.first)) {
            if (out_degree(pfront.first) != out_degree(nfront)) {
                return false;
            }
            node_t *poutput = get_output(pfront.first, pfront.second.first);
            // output nodes have no order, so need to match pattern to get the
            // corresponding node
            size_t corresponding_offset;
            bool flag = nfront->find_output_node(
                    poutput, &corresponding_offset, pfront.second.first);
            if (!flag) { return false; }
            nfront->swap_output_node(pfront.second.first, corresponding_offset);
            node_t *noutput = get_output(nfront, pfront.second.first++);

            hashtype poutput_hash = hash_func(poutput);
            if (visited.count(poutput_hash) == 0) {
                pattern_queue.push_back(
                        std::make_pair(poutput, std::make_pair(0, 0)));
                node_queue.push_back(noutput);
                visited.insert(poutput_hash);
            }
        } else { // pfront.second.second != in_degree(pfront.first)
            if (in_degree(nfront) > in_degree(pfront.first)) return false;

            // two cases for any as input:
            // 1. any matches to a tensor
            //    e.g. conv   tensor
            //            \   /
            //             add
            // 2. any matches to an arbitrary node
            //    e.g. conv   any_node
            //             \  /
            //              add
            //
            node_t *pinput = get_input(pfront.first, pfront.second.second);
            if (in_degree(nfront) == in_degree(pfront.first) // case #2
                    || (pinput->get_op_kind() != op_kind::any // maybe case #1
                            && pfront.second.second < in_degree(nfront))) {
                node_t *ninput = get_input(nfront, pfront.second.second++);
                hashtype pinput_hash = hash_func(pinput);
                if (visited.count(pinput_hash) == 0) {
                    pattern_queue.push_front(
                            std::make_pair(pinput, std::make_pair(0, 0)));
                    node_queue.push_front(ninput);
                    visited.insert(pinput_hash);
                }
            } else if (pinput->get_op_kind() == op_kind::any) {
                // case #1
                hashtype pinput_hash = hash_func(pinput);
                if (visited.count(pinput_hash) == 0) {
                    visited.insert(pinput_hash);
                }
                pfront.second.second++;
            } else {
                return false;
            }
        }
    }
    return true;
}

// function to do per node comparison
inline bool per_node_comp(node_t *graph_node, node_t *pattern_node,
        std::vector<node_t *> &candidate_fusion,
        std::unordered_set<node_t *> &selected) {
    return per_node_comp_<size_t>(
            graph_node, pattern_node, candidate_fusion, selected,
            [](node_t *n) -> size_t { // hashfunc
                return n->id();
            },
            [](node_t *n) -> size_t { // outdegree
                return n->num_outputs();
            },
            [](node_t *n, size_t index) -> node_t * { // getoutput
                return n->get_output_node(index);
            },
            [](node_t *n) -> size_t { // indegree
                return n->num_inputs();
            },
            [](node_t *n, size_t index) -> node_t * { // getinput
                return n->get_input_node(index);
            });
}

class pattern_utils {
public:
    inline void match(dnnl::graph::impl::graph_t &backend_graph,
            node_t *op_pattern,
            std::vector<std::vector<node_t *>> &fusion_nodes);
    inline void rewrite(dnnl::graph::impl::graph_t &backend_graph,
            node_t *origin_pattern, node_t *optimized_pattern,
            std::vector<std::vector<node_t *>> &fusion_nodes);
    // function to convert pattern to a vector based on search order
    std::vector<node_t *> pattern2vector(node_t *op_pattern) {
        std::unordered_set<node_t *> selected;
        std::vector<node_t *> pattern_vector;
        per_node_comp(op_pattern, op_pattern, pattern_vector, selected);
        return pattern_vector;
    }
    pattern_utils() = default;
    pattern_utils(const pattern_utils &) = delete;
    pattern_utils(pattern_utils &&) = delete;
    pattern_utils &operator=(const pattern_utils &) = delete;
};

// function to do pattern matching
inline void pattern_utils::match(dnnl::graph::impl::graph_t &backend_graph,
        node_t *op_pattern, std::vector<std::vector<node_t *>> &fusion_nodes) {
    std::unordered_set<node_t *> selected;
    // dfs_visit graph, do pattern matching
    dfs_visit(backend_graph.get_outputs(), [&](node_t *cur_node) {
        std::vector<node_t *> candidate_fusion;
        if (!per_node_comp(cur_node, op_pattern, candidate_fusion, selected)) {
            return;
        }
        fusion_nodes.emplace_back(candidate_fusion);
        for (auto &anode : candidate_fusion) {
            selected.insert(anode);
        }
    });
}

// function to do graph rewriting
inline void pattern_utils::rewrite(dnnl::graph::impl::graph_t &backend_graph,
        node_t *origin_pattern, node_t *optimized_pattern,
        std::vector<std::vector<node_t *>> &fusion_nodes) {
    std::vector<node_t *> pattern_vec = pattern2vector(origin_pattern);
    std::unordered_set<node_t *> visited;
    std::unordered_set<size_t> visited_tensor;
    for (auto &nodes : fusion_nodes) {
        visited.clear();
        visited_tensor.clear();
        node_t *fused_node
                = backend_graph.create_node(optimized_pattern->get_op_kind());
        fused_node->merge_attrs_map(optimized_pattern->get_attrs_map());
        for (size_t i = 0; i < nodes.size(); ++i) {
            node_t *cur_node = nodes[i];
            visited.insert(cur_node);
            fused_node->merge_attrs_map(cur_node->get_attrs_map());
            fused_node->add_op_ids(cur_node->get_op_ids());
            const node_t *pattern_node = pattern_vec[i];

            // if cur_node has input node which isn't in pattern,
            // update value's connection. if cur_node has input node
            // which is in pattern, add its output_tensor into visited
            for (size_t j = 0; j < cur_node->num_inputs(); ++j) {
                auto in_node = cur_node->get_input_node(j);

                std::vector<size_t> in_offsets;
                cur_node->get_input_offsets(in_node, in_offsets);
                //if in_node isn't in pattern,
                //set it as a input node of fused_node
                if (!visited.count(in_node)) {
                    in_node->remove_output(cur_node);
                    in_node->add_output(fused_node);
                    for (auto &offset : in_offsets) {
                        fused_node->set_input(
                                fused_node->num_inputs(), in_node, offset);
                    }
                } else { //else, add it's output tensors into visited
                    for (size_t k = 0; k < in_node->num_outputs_tensor(); ++k) {
                        visited_tensor.insert(in_node->get_output_tensor(k).id);
                    }
                }
            }

            //add cur_node's input_tensors which isn't visited into fused_node
            for (size_t k = 0; k < cur_node->num_inputs_tensor(); ++k) {
                auto in_tensor = cur_node->get_input_tensor(k);
                if (!visited_tensor.count(in_tensor.id)) {
                    fused_node->add_input_tensors(in_tensor, cur_node, k);
                }
            }

            if (pattern_node->num_outputs() == 0) {
                // it's a end node of pattern, need to update
                // node connection of it's output nodes
                for (size_t k = 0; k < cur_node->num_outputs(); ++k) {
                    auto out_node = cur_node->get_output_node(k);
                    std::vector<size_t> offsets;
                    out_node->find_input_nodes(cur_node, offsets);
                    for (auto &offset : offsets) {
                        auto input_offset = out_node->get_input_offset(offset);
                        out_node->set_input(offset, fused_node, input_offset);
                        fused_node->add_output(out_node);
                    }
                }

                for (size_t k = 0; k < cur_node->num_outputs_tensor(); ++k) {
                    auto out_tensor = cur_node->get_output_tensor(k);
                    fused_node->add_output_tensors(out_tensor, cur_node, k);
                }
            }
        }

        for (size_t i = 0; i < nodes.size(); ++i) {
            backend_graph.delete_node(nodes[i]);
        }
    }
}
} // namespace impl
} // namespace graph
} // namespace dnnl

#endif
