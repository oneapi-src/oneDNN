/*******************************************************************************
* Copyright 2020-2022 Intel Corporation
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
#ifndef BACKEND_DNNL_PATTERNS_PATTERN_UTILS_HPP
#define BACKEND_DNNL_PATTERNS_PATTERN_UTILS_HPP

#include <deque>
#include <iostream>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>
#include <unordered_set>

#include "utils/compatible.hpp"

#include "interface/graph.hpp"
#include "interface/partition.hpp"

#include "backend/dnnl/dnnl_backend.hpp"
#include "backend/dnnl/dnnl_partition_impl.hpp"

#include "utils/pm/nested_matcher.hpp"
#include "utils/pm/pbuilder.hpp"

namespace dnnl {
namespace graph {
namespace impl {
namespace dnnl_impl {
/**
 * Operators set for checking number of op inputs
 */
struct op_set_t {
    /**
     * Check if operator inputs are commutative.
     *
     * @param kind operator kind
     * @return whether the operator inputs are commutative
     */
    static bool check_inputs_commutativity(op_kind_t op_kind) {
        static const std::set<op_kind_t> supported_ops {impl::op_kind::Add,
                impl::op_kind::Subtract, impl::op_kind::SquaredDifference,
                impl::op_kind::Multiply, impl::op_kind::Maximum,
                impl::op_kind::Minimum};
        return supported_ops.count(op_kind);
    }
};

//check if the inputs of the op in pattern need to be swapped
template <typename getinput>
bool should_swap_inputs(op_t *graph_op, op_t *pattern_op, getinput get_input) {
    const auto get_op_kind_ = [&](op_t *op_, size_t idx) {
        value_t *invalue = get_input(op_, idx);
        if (invalue->has_producer()) {
            return invalue->get_producer().get_kind();
        } else {
            return impl::op_kind::Wildcard;
        }
    };
    const auto pin_0 = get_op_kind_(pattern_op, 0);
    const auto pin_1 = get_op_kind_(pattern_op, 1);
    const auto gin_0 = get_op_kind_(graph_op, 0);
    const auto gin_1 = get_op_kind_(graph_op, 1);

    if (pin_0 == impl::op_kind::Wildcard) { // if pin_0 accepts Wildcard
        // check if the corresponding inputs of pattern and graph match,
        // if they do, no swap. If they don't, then check if the opposite
        // inputs match or not
        if (pin_1 != gin_1 && pin_1 == gin_0)
            return true;
        else
            return false;
    } else if (pin_1 == impl::op_kind::Wildcard) { // if pin_1 accepts Wildcard
        if (pin_0 != gin_0 && pin_0 == gin_1)
            return true;
        else
            return false;
    } else { // if no Wildcard inputs in pattern
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

    if (op_set_t::check_inputs_commutativity(pattern_op->get_kind())
            && expected_deg == pin_deg && pin_deg == gin_deg) {
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
    std::set<std::string> excepted {"num_inputs", "s8_check", "broadcast_check",
            "out_bf16_check", "in_bf16_check"};
    bool pattern_is_graph = graph_op == pattern_op;
    hashtype pattern_starter_hash = hash_func(pattern_op);
    pattern_queue.emplace_back(std::make_pair(pattern_op,
            std::make_pair(
                    0, 0))); //op, matched_input_value, matched_output_value
    visited.insert(pattern_starter_hash);
    op_queue.push_back(graph_op);

    while (!pattern_queue.empty()) {
        std::pair<op_t *, std::pair<uint32_t, uint32_t>> &pfront
                = pattern_queue.front();
        op_t *nfront = op_queue.front();

        // check if graph op is an unvisited op and can meet the
        // requirement of pattern op
        if (pfront.first->get_kind() != impl::op_kind::Wildcard
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

        if (!pattern_is_graph && pfront.first->has_attr("s8_check")
                && pfront.first->get_attr<bool>("s8_check") == true) {
            for (size_t i = 0; i < in_degree(nfront); ++i) {
                logical_tensor_t inport
                        = nfront->get_input_value(i)->get_logical_tensor();
                if (inport.data_type != impl::data_type::s8) return false;
            }
        }

        if (!pattern_is_graph && pfront.first->has_attr("out_bf16_check")
                && pfront.first->get_attr<bool>("out_bf16_check") == true) {
            for (size_t i = 0; i < out_degree(nfront); ++i) {
                logical_tensor_t outport
                        = nfront->get_output_value(i)->get_logical_tensor();
                if (outport.data_type != impl::data_type::bf16) return false;
            }
        }

        if (!pattern_is_graph && pfront.first->has_attr("in_bf16_check")
                && pfront.first->get_attr<bool>("in_bf16_check") == true) {
            for (size_t i = 0; i < in_degree(nfront); ++i) {
                logical_tensor_t inport
                        = nfront->get_input_value(i)->get_logical_tensor();
                if (inport.data_type != impl::data_type::bf16) return false;
            }
        }

        // handle the case that "Add" op's input tensor's order
        // is different from it in graph
        special_case_handle(nfront, pfront.first, in_degree, get_input);

        if (!pattern_is_graph && pfront.first->has_attr("broadcast_check")
                && pfront.first->get_attr<bool>("broadcast_check") == true) {
            if (pfront.first->get_kind() == impl::op_kind::Add) {
                // find the input that will NOT be mapped to post-ops's src1
                op_t &pin0_producer
                        = pfront.first->get_input_value(0)->get_producer();

                size_t no_post_src_index
                        = pin0_producer.get_kind() == impl::op_kind::Wildcard
                        ? 1
                        : 0;

                logical_tensor_t no_post_src
                        = nfront->get_input_value(no_post_src_index)
                                  ->get_logical_tensor();
                logical_tensor_t dst
                        = nfront->get_output_value(0)->get_logical_tensor();
                auto no_post_src_ltw = logical_tensor_wrapper_t(no_post_src);
                auto dst_ltw = logical_tensor_wrapper_t(dst);

                // unsupported case1: no ndims
                if (no_post_src_ltw.ndims() == -1 || dst_ltw.ndims() == -1)
                    return false;

                // expand expand to same ndims with dst
                auto no_post_src_dims = no_post_src_ltw.vdims();
                for (size_t i = no_post_src_ltw.ndims(); i < dst_ltw.ndims();
                        i++) {
                    no_post_src_dims.insert(no_post_src_dims.begin(), 1);
                }

                // unsupported case2: have ndims, but no concret shape
                for (size_t i = 0; i < no_post_src_dims.size(); i++) {
                    if (no_post_src_dims[i] == -1 || dst_ltw.vdims()[i] == -1)
                        return false;
                }

                // unsupported case3: have shape, but post ops will affect
                // output shape binary post-ops shouldn't affect the primitive
                // output shape, so the on_post_src shape should be exactly same
                // with dst
                if (no_post_src_ltw.vdims() != dst_ltw.vdims()) return false;
            }
        }

        if (pfront.first->get_kind() == impl::op_kind::Wildcard) {
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
            for (size_t i = 0; i < pconsumers.size(); i++) {
                impl::utils::optional<size_t> corresponding_offset
                        = noutput->find_consumer(i,
                                pconsumers[i].get_op().get_kind(),
                                pconsumers[i].get_offset(),
                                op_set_t::check_inputs_commutativity(
                                        pconsumers[i].get_op().get_kind()));
                if (!corresponding_offset.has_value()) { return false; }
                noutput->swap_consumer(i, corresponding_offset.value());
            }
            pfront.second.first++;
            nconsumers = noutput->get_consumers();
            for (size_t i = 0; i < nconsumers.size(); i++) {
                op_t &pout = pconsumers[i].get_op();
                op_t &nout = nconsumers[i].get_op();
                hashtype poutput_hash = hash_func(&pout);
                if (visited.count(poutput_hash) == 0) {
                    pattern_queue.emplace_back(
                            std::make_pair(&pout, std::make_pair(0, 0)));
                    op_queue.push_back(&nout);
                    visited.insert(poutput_hash);
                }
            }
        } else { // pfront.second.second != in_degree(pfront.first)
            if (in_degree(nfront) < in_degree(pfront.first)) { return false; }
            // two cases for Wildcard as input:
            // 1. Wildcard matches to a tensor
            //    e.g. conv   tensor
            //            \   /
            //             add
            // 2. Wildcard matches to an arbitrary op
            //    e.g. conv   Wildcard
            //             \  /
            //              add
            //

            value_t *pvalue = get_input(pfront.first, pfront.second.second);
            op_t *pinput = &pvalue->get_producer();
            value_t *nvalue = get_input(nfront, pfront.second.second++);
            hashtype pinput_hash = hash_func(pinput);
            if (nvalue->has_producer()) { // maybe case #2
                op_t *ninput = &nvalue->get_producer();
                if (visited.count(pinput_hash) == 0) {
                    pattern_queue.push_front(
                            std::make_pair(pinput, std::make_pair(0, 0)));
                    op_queue.push_front(ninput);
                    visited.insert(pinput_hash);
                }
            } else if (pinput->get_kind() == impl::op_kind::Wildcard) {
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

class pattern_utils_t {
public:
    inline void match(dnnl::graph::impl::graph_t &backend_graph,
            op_t *op_pattern, std::vector<std::vector<op_t *>> &fusion_ops);
    inline void match(dnnl::graph::impl::graph_t &backend_graph,
            std::shared_ptr<impl::utils::pm::pb_graph_t> pgraph,
            std::vector<std::vector<op_t *>> &fusion_ops);

    inline void rewrite(dnnl::graph::impl::graph_t &backend_graph,
            op_t *origin_pattern, op_t *optimized_pattern,
            std::vector<std::vector<op_t *>> &fusion_ops);

    inline void fuse(dnnl::graph::impl::graph_t &backend_graph,
            op_t *origin_pattern, op_t *optimized_pattern,
            std::vector<std::vector<op_t *>> &fusion_ops);
    inline void fuse(dnnl::graph::impl::graph_t &backend_graph,
            std::vector<std::vector<op_t *>> &fusion_ops,
            op_t &op_with_backend);

    // function to convert pattern to a vector based on search order
    std::vector<op_t *> pattern2vector(op_t *op_pattern) {
        std::unordered_set<op_t *> selected;
        std::vector<op_t *> pattern_vector;
        per_op_comp(op_pattern, op_pattern, pattern_vector, selected);
        return pattern_vector;
    }
    pattern_utils_t() = default;
    pattern_utils_t(const pattern_utils_t &) = delete;
    pattern_utils_t(pattern_utils_t &&) = delete;
    pattern_utils_t &operator=(const pattern_utils_t &) = delete;
};

// function to do pattern matching
inline void pattern_utils_t::match(dnnl::graph::impl::graph_t &backend_graph,
        op_t *op_pattern, std::vector<std::vector<op_t *>> &fusion_ops) {
    std::unordered_set<op_t *> selected;
    // dfs_visit graph, do pattern matching
    topo_order_visit(backend_graph.get_output_ops(), [&](op_t *cur_op) {
        std::vector<op_t *> candidate_fusion;
        if (!per_op_comp(cur_op, op_pattern, candidate_fusion, selected)) {
            return status::success;
        }
        fusion_ops.emplace_back(candidate_fusion);
        for (auto &aop : candidate_fusion) {
            selected.insert(aop);
        }
        return status::success;
    });
}

// function to do v2 pattern matching
inline void pattern_utils_t::match(dnnl::graph::impl::graph_t &backend_graph,
        std::shared_ptr<impl::utils::pm::pb_graph_t> pgraph,
        std::vector<std::vector<op_t *>> &fusion_ops) {
    // dfs_visit graph, do pattern matching
    topo_order_visit(backend_graph.get_output_ops(), [&](op_t *cur_op) {
        std::vector<op_t *> candidate_fusion;
        if (!impl::utils::pm::match_pattern(cur_op, pgraph, candidate_fusion)) {
            return status::success;
        }
        fusion_ops.emplace_back(candidate_fusion);
        return status::success;
    });
}

// function to do fusion but not rewrite the graph
inline void pattern_utils_t::fuse(dnnl::graph::impl::graph_t &backend_graph,
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
            const op_t *pattern_op = pattern_vec[i];

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
            if (pattern_op->num_outputs() == 0) {
                // it's a end op of pattern
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

        // use the fused op to initialize the partition_impl, and merge the
        // informations to it.
        pimpl->init(fused_op.get());

        // transfer the ownership of fusion op from graph to partition
        // note: the fusion op will not be removed from the graph
        for (size_t i = 0; i < ops.size(); ++i) {
            pimpl->add_op(ops[i]->shared_from_this());
            // claim the op belong to the partition
            ops[i]->set_partition(pimpl.get());
        }

        backend_graph.add_partition(pimpl);
    }
}

//do fuse with v2 pattern language
inline void pattern_utils_t::fuse(dnnl::graph::impl::graph_t &backend_graph,
        std::vector<std::vector<op_t *>> &fusion_ops, op_t &fused_op) {
    std::vector<op_t *> fusion_ops_set;
    std::unordered_set<op_t *> visit;

    for (auto &pairs : fusion_ops) {
        fusion_ops_set.clear();
        visit.clear();
        std::shared_ptr<op_t> partition_fused_op(new op_t(fused_op.get_kind()));
        partition_fused_op->merge_attributes(fused_op.get_attributes());
        for (size_t i = 0; i < pairs.size(); ++i) {
            visit.insert(pairs[i]);
            fusion_ops_set.push_back(pairs[i]);
        }

        for (auto &cur_op : fusion_ops_set) {
            // merge the attrs and op ids
            partition_fused_op->merge_attributes(cur_op->get_attributes());
            partition_fused_op->add_op_ids(cur_op->get_op_ids());

            for (size_t j = 0; j < cur_op->num_inputs(); ++j) {
                std::shared_ptr<value_t> in_value = cur_op->get_input_value(j);
                // if in_value has no producer or its producer isn't in pattern,
                // add this input value to fused op
                if (!in_value->has_producer()
                        || !visit.count(&in_value->get_producer())) {
                    std::shared_ptr<value_t> copied_in_value
                            = std::make_shared<value_t>(
                                    in_value->get_logical_tensor(),
                                    /*internal*/ true);
                    partition_fused_op->add_input(copied_in_value);
                }
            }

            // find an end_op to start reorder, find an op whose output isn't in
            // matched_pairs, can't use pb_op's output == 0, because ops defined
            // in subgraph also has 0 output but isn't the end of the pattern.
            bool is_end = true;
            for (auto &output : cur_op->get_output_values()) {
                for (auto &consumer : output->get_consumers()) {
                    if (visit.count(&(consumer.get_op()))) {
                        is_end = false;
                        break;
                    }
                }
                if (!is_end) { break; }
            }

            // merge the output tensor if current op is an end op
            if (is_end) {
                // it's the end op of pattern
                for (size_t k = 0; k < cur_op->num_outputs(); ++k) {
                    std::shared_ptr<value_t> out_value
                            = cur_op->get_output_value(k);
                    std::shared_ptr<value_t> copied_out_value
                            = std::make_shared<value_t>(
                                    out_value->get_logical_tensor(),
                                    /*internal*/ true);
                    partition_fused_op->add_output(copied_out_value);
                }
            }
        }

        std::shared_ptr<dnnl_partition_impl_t> pimpl
                = std::make_shared<dnnl_partition_impl_t>(
                        backend_graph.get_engine_kind());

        // use the fused op to initialize the partition_impl, and merge the
        // informations to it.
        pimpl->init(partition_fused_op.get());

        // transfer the ownership of fusion op from graph to partition
        // note: the fusion op will not be removed from the graph
        for (size_t i = 0; i < pairs.size(); ++i) {
            pimpl->add_op(pairs[i]->shared_from_this());
            // claim the op belong to the partition
            pairs[i]->set_partition(pimpl.get());
        }

        backend_graph.add_partition(pimpl);
    }
}

} // namespace dnnl_impl
} // namespace impl
} // namespace graph
} // namespace dnnl

#endif
