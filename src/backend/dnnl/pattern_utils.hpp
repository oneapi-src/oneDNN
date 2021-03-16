/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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
#ifndef BACKEND_DNNL_PATTERN_UTILS_HPP
#define BACKEND_DNNL_PATTERN_UTILS_HPP

#include <deque>
#include <iostream>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>
#include <unordered_set>

#include "interface/graph.hpp"
#include "interface/partition.hpp"

#include "backend/dnnl/dnnl_backend.hpp"
#include "backend/dnnl/dnnl_partition_impl.hpp"

namespace dnnl {
namespace graph {
namespace impl {
namespace dnnl_impl {

// FRequirement: A function to check if graph node can meet
// the requirement of pattern node
// Should be defined when register passes
using FRequirement = impl::pass::FRequirement;

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

//check if the input order of the "binary" op in pattern need to be swapped
template <typename getinput>
bool should_swap_inputs(op_t *graph_op, op_t *pattern_op, getinput get_input) {
    const auto get_op_kind_ = [&](op_t *op_, size_t idx) {
        value_t *invalue = get_input(op_, idx);
        if (invalue->has_producer()) {
            return invalue->get_producer().get_kind();
        } else {
            return op_kind::any;
        }
    };
    const auto pin_0 = get_op_kind_(pattern_op, 0);
    const auto pin_1 = get_op_kind_(pattern_op, 1);
    const auto gin_0 = get_op_kind_(graph_op, 0);
    const auto gin_1 = get_op_kind_(graph_op, 1);

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
void special_case_handle(op_t *graph_op, op_t *pattern_op, indegree in_degree,
        getinput get_input) {
    const auto pin_deg = in_degree(pattern_op);
    const auto gin_deg = in_degree(graph_op);
    const auto expected_deg = 2;

    if (op_set::is_binary(pattern_op->get_kind()) && expected_deg == pin_deg
            && pin_deg == gin_deg) {
        if (should_swap_inputs(graph_op, pattern_op, get_input)) {
            pattern_op->swap_input_values(0, 1);
        }
    }
}

/*!
* \brief Function to do comparison between a graph
         and a pattern. It will search from a graph op,
         and compare its inputs / outputs with the ops in
         the pattern, until all the ops in the pattern are
         exhausted.
* \param graph_op the op in the graph to compare from
* \param pattern_op the op in the pattern to compare from
* \param candidate_fusion the vector stores the matched ops
* \param selected the set stores the ops have been selected
* \param hash_func the hash function
* \param out_degree the output degree
* \param get_output the function to get a specific output
* \param in_degree the input degree
* \param get_input the function to get a specific input
* \tparam bool whether the pattern is matched
*/
template <typename hashtype, typename hashfunc, typename outdegree,
        typename getoutput, typename indegree, typename getinput>
bool per_op_comp_(op_t *graph_op, op_t *pattern_op,
        std::vector<op_t *> &candidate_fusion,
        std::unordered_set<op_t *> &selected, hashfunc hash_func,
        outdegree out_degree, getoutput get_output, indegree in_degree,
        getinput get_input) {
    using consumer_t = value_t::consumer_t;
    std::deque<std::pair<op_t *, std::pair<uint32_t, uint32_t>>> pattern_queue;
    std::deque<op_t *> op_queue;
    //if a op have been visited
    std::unordered_set<hashtype> visited;
    std::set<std::string> excepted {"num_inputs"};
    bool pattern_is_graph = graph_op == pattern_op;
    hashtype pattern_starter_hash = hash_func(pattern_op);
    pattern_queue.push_back(std::make_pair(pattern_op,
            std::make_pair(
                    0, 0))); //op, matched_input_value, matched_output_value
    visited.insert(pattern_starter_hash);
    op_queue.push_back(graph_op);

    while (!pattern_queue.empty()) {
        std::pair<op_t *, std::pair<uint32_t, uint32_t>> &pfront
                = pattern_queue.front();
        op_t *nfront = op_queue.front();

        // check if graph node is an unvisited node and can meet the
        // requirement of pattern node
        if (pfront.first->get_kind() != op_kind::any
                && (selected.count(nfront) != 0
                        || nfront->get_partition() != nullptr
                        || nfront->get_kind() != pfront.first->get_kind()
                        || !pfront.first->has_same_attr_values(
                                *nfront, excepted))) {
            return false;
        }

        if (!pattern_is_graph && pfront.first->has_attr("num_inputs")
                && pfront.first->get_attr<int64_t>("num_inputs")
                        != in_degree(nfront)) {
            return false;
        }

        // handle the case that "Add" op's input tensor's order
        // is different from it in graph
        special_case_handle(nfront, pfront.first, in_degree, get_input);

        if (pfront.first->get_kind() == op_kind::any) {
            op_queue.pop_front();
            pattern_queue.pop_front();
        } else if (pfront.second.first == out_degree(pfront.first)
                && pfront.second.second == in_degree(pfront.first)) {
            candidate_fusion.emplace_back(nfront);
            op_queue.pop_front();
            pattern_queue.pop_front();
        } else if (pfront.second.first != out_degree(pfront.first)) {
            if (out_degree(pfront.first) != out_degree(nfront)) {
                return false;
            }
            value_t *poutput = get_output(pfront.first, pfront.second.first);
            value_t *noutput = get_output(nfront, pfront.second.first);
            std::vector<consumer_t> pconsumers = poutput->get_consumers();
            std::vector<consumer_t> nconsumers = noutput->get_consumers();
            if (pconsumers.size() != nconsumers.size()) { return false; }
            // output ops have no order, so need to match pattern to get the
            // corresponding op
            size_t corresponding_offset;
            for (size_t i = 0; i < pconsumers.size(); i++) {
                bool flag = noutput->find_consumer(
                        pconsumers[i].get_op().get_kind(),
                        corresponding_offset);
                if (!flag) { return false; }
                noutput->swap_consumer(i, corresponding_offset);
            }
            pfront.second.first++;
            nconsumers = noutput->get_consumers();
            for (size_t i = 0; i < nconsumers.size(); i++) {
                op_t &pout = pconsumers[i].get_op();
                op_t &nout = nconsumers[i].get_op();
                hashtype poutput_hash = hash_func(&pout);
                if (visited.count(poutput_hash) == 0) {
                    pattern_queue.push_back(
                            std::make_pair(&pout, std::make_pair(0, 0)));
                    op_queue.push_back(&nout);
                    visited.insert(poutput_hash);
                }
            }
        } else { // pfront.second.second != in_degree(pfront.first)
            if (in_degree(nfront) < in_degree(pfront.first)) { return false; }
            // two cases for any as input:
            // 1. any matches to a tensor
            //    e.g. conv   tensor
            //            \   /
            //             add
            // 2. any matches to an arbitrary op
            //    e.g. conv   any_op
            //             \  /
            //              add
            //

            value_t *pvalue = get_input(pfront.first, pfront.second.second);
            op_t *pinput = &pvalue->get_producer();
            value_t *nvalue = get_input(nfront, pfront.second.second++);
            op_t *ninput = &nvalue->get_producer();
            hashtype pinput_hash = hash_func(pinput);
            if (nvalue->has_producer()) { // maybe case #2
                if (visited.count(pinput_hash) == 0) {
                    pattern_queue.push_front(
                            std::make_pair(pinput, std::make_pair(0, 0)));
                    op_queue.push_front(ninput);
                    visited.insert(pinput_hash);
                }
            } else if (pinput->get_kind() == op_kind::any) {
                // case #1
                hashtype pinput_hash = hash_func(pinput);
                if (visited.count(pinput_hash) == 0) {
                    visited.insert(pinput_hash);
                }
            } else {
                return false;
            }
        }
    }
    return true;
}

// function to do per op comparison
inline bool per_op_comp(op_t *graph_op, op_t *pattern_op,
        std::vector<op_t *> &candidate_fusion,
        std::unordered_set<op_t *> &selected) {
    return per_op_comp_<op_t *>(
            graph_op, pattern_op, candidate_fusion, selected,
            [](op_t *n) -> op_t * { // hashfunc
                return n;
            },
            [](op_t *n) -> size_t { // outdegree
                return n->num_outputs();
            },
            [](op_t *n, size_t index) -> value_t * { // getoutput
                return (n->get_output_value(index).get());
            },
            [](op_t *n) -> size_t { // indegree
                return n->num_inputs();
            },
            [](op_t *n, size_t index) -> value_t * { // getinput
                return n->get_input_value(index).get();
            });
}

class pattern_utils {
public:
    inline void match(dnnl::graph::impl::graph_t &backend_graph,
            op_t *op_pattern, std::vector<std::vector<op_t *>> &fusion_ops);
    inline void rewrite(dnnl::graph::impl::graph_t &backend_graph,
            op_t *origin_pattern, op_t *optimized_pattern,
            std::vector<std::vector<op_t *>> &fusion_ops);
    inline void fuse(dnnl::graph::impl::graph_t &backend_graph,
            op_t *origin_pattern, op_t *optimized_pattern,
            std::vector<std::vector<op_t *>> &fusion_ops);
    // function to convert pattern to a vector based on search order
    std::vector<op_t *> pattern2vector(op_t *op_pattern) {
        std::unordered_set<op_t *> selected;
        std::vector<op_t *> pattern_vector;
        per_op_comp(op_pattern, op_pattern, pattern_vector, selected);
        return pattern_vector;
    }
    pattern_utils() = default;
    pattern_utils(const pattern_utils &) = delete;
    pattern_utils(pattern_utils &&) = delete;
    pattern_utils &operator=(const pattern_utils &) = delete;
};

// function to do pattern matching
inline void pattern_utils::match(dnnl::graph::impl::graph_t &backend_graph,
        op_t *op_pattern, std::vector<std::vector<op_t *>> &fusion_ops) {
    std::unordered_set<op_t *> selected;
    // dfs_visit graph, do pattern matching
    topo_order_visit(backend_graph.get_output_ops(), [&](op_t *cur_op) {
        std::vector<op_t *> candidate_fusion;
        if (!per_op_comp(cur_op, op_pattern, candidate_fusion, selected)) {
            return;
        }
        fusion_ops.emplace_back(candidate_fusion);
        for (auto &aop : candidate_fusion) {
            selected.insert(aop);
        }
    });
}

// function to do graph rewriting
inline void pattern_utils::rewrite(dnnl::graph::impl::graph_t &backend_graph,
        op_t *origin_pattern, op_t *optimized_pattern,
        std::vector<std::vector<op_t *>> &fusion_ops) {
    std::vector<op_t *> pattern_vec = pattern2vector(origin_pattern);
    std::unordered_set<op_t *> visited;
    std::unordered_set<value_t *> visited_value;

    for (auto &ops : fusion_ops) {
        visited.clear();
        visited_value.clear();
        op_t *fused_op = backend_graph.create_op(
                optimized_pattern->get_kind(), "fused_op");
        //need discuss: how to add into graph
        fused_op->merge_attributes(optimized_pattern->get_attributes());
        for (size_t i = 0; i < ops.size(); ++i) {
            op_t *cur_op = ops[i];
            visited.insert(cur_op);
            fused_op->merge_attributes(cur_op->get_attributes());
            fused_op->add_op_ids(cur_op->get_id());
            const op_t *pattern_op = pattern_vec[i];
            // if cur_op has input op which isn't in pattern,
            // update value's connection. if cur_op has input op
            // which is in pattern, add its output_tensor into visited
            for (size_t j = 0; j < cur_op->num_inputs(); ++j) {
                auto in_value = cur_op->get_input_value(j);
                //if in_op isn't in pattern,
                //set it as a input op of fused_op
                if (!in_value->has_producer()
                        || !visited.count(&in_value->get_producer())) {
                    in_value->remove_consumer(*cur_op, j);
                    in_value->add_consumer(*fused_op, fused_op->num_inputs());
                    fused_op->add_input(in_value);
                }
            }
            if (pattern_op->num_outputs() == 0) {
                // it's a end op of pattern, need to update
                // op connection of it's output ops
                for (size_t k = 0; k < cur_op->num_outputs(); ++k) {
                    auto out_value = cur_op->get_output_value(k);
                    out_value->set_producer(*fused_op);
                    fused_op->add_output(out_value);
                }
            }
        }

        for (size_t i = 0; i < ops.size(); ++i) {
            backend_graph.delete_op(ops[i]);
        }
    }
}

// function to do fusion but not rewrite the graph
inline void pattern_utils::fuse(dnnl::graph::impl::graph_t &backend_graph,
        op_t *origin_pattern, op_t *opt_pattern,
        std::vector<std::vector<op_t *>> &fusion_ops) {
    std::vector<op_t *> pattern_vec = pattern2vector(origin_pattern);
    std::unordered_set<op_t *> fusion_ops_set;
    for (auto &ops : fusion_ops) {
        fusion_ops_set.clear();

        std::shared_ptr<op_t> fused_op(new op_t(opt_pattern->get_kind()));
        fused_op->merge_attributes(opt_pattern->get_attributes());

        for (size_t i = 0; i < ops.size(); ++i) {
            fusion_ops_set.insert(ops[i]);
        }

        for (size_t i = 0; i < ops.size(); ++i) {
            op_t *cur_op = ops[i];
            const op_t *pattern_node = pattern_vec[i];

            // merge the attrs and op ids
            fused_op->merge_attributes(cur_op->get_attributes());
            fused_op->add_op_ids(cur_op->get_op_ids());

            // merge the input tensor
            // FIXME(qun) Here is a potential bug: We assume that the input
            // tensors which have producer will be in prior to the input
            // tensors which have no producer, but this assumption is not
            // always true. However, Above buggy pattern will not be matched
            // by pattern matcher now, because of another bug in our current
            // pattern matcher. We will fix all these bugs in new pattern
            // matcher
            for (size_t j = 0; j < cur_op->num_inputs(); ++j) {
                auto in_value = cur_op->get_input_value(j);
                // if in_value has no producer or its producer isn't in pattern,
                // add this input value to fused op
                if (!in_value->has_producer()
                        || !fusion_ops_set.count(&in_value->get_producer())) {
                    auto copied_in_value = std::make_shared<value_t>(
                            in_value->get_logical_tensor(), /*internal*/ true);
                    fused_op->add_input(copied_in_value);
                }
            }

            // merge the output tensor
            if (pattern_node->num_outputs() == 0) {
                // it's a end node of pattern
                for (size_t k = 0; k < cur_op->num_outputs(); ++k) {
                    auto out_value = cur_op->get_output_value(k);
                    auto copied_out_value = std::make_shared<value_t>(
                            out_value->get_logical_tensor(), /*internal*/ true);
                    fused_op->add_output(copied_out_value);
                }
            }
        }

        auto pimpl = std::make_shared<dnnl_partition_impl_t>(
                backend_graph.get_engine_kind());

        // use the fused node to initialize the partition_impl, and merge the
        // informations to it.
        pimpl->init(fused_op.get());

        // transfer the ownership of fusion node from graph to partition
        // note: the fusion node will not be removed from the graph
        for (size_t i = 0; i < ops.size(); ++i) {
            pimpl->add_op(ops[i]->shared_from_this());
            // claim the op belong to the partition
            ops[i]->set_partition(pimpl.get());
        }

        backend_graph.add_partition(pimpl);
    }
}

} // namespace dnnl_impl
} // namespace impl
} // namespace graph
} // namespace dnnl

#endif
