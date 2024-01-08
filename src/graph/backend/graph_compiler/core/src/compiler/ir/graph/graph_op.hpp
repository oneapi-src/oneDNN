/*******************************************************************************
 * Copyright 2020-2023 Intel Corporation
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
#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_GRAPH_GRAPH_OP_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_GRAPH_GRAPH_OP_HPP

#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "graph.hpp"
#include "util/general_object.hpp"
#include <compiler/ir/graph/graph_config.hpp>
#include <compiler/ir/graph/trait/configurable.hpp>
#include <compiler/ir/graph/tunable_op.hpp>
#include <util/reflection.hpp>
#include <util/utils.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

class graph_op_t : public sc_op {
public:
    graph_op_t() = default;

    graph_op_t(const std::string &op_name,
            const std::vector<graph_tensor_ptr> &producer_lt,
            const std::vector<graph_tensor_ptr> &consumer_lt,
            const any_map_t &attrs)
        : sc_op(op_name, producer_lt, consumer_lt, attrs) {}

    ir_module_ptr get_func(context_ptr ctx) override;

    void query_format(context_ptr ctx,
            std::vector<std::vector<format_stride_pair>> &supported_ins,
            std::vector<std::vector<format_stride_pair>> &supported_outs)
            override {};

    // the param graph is created by upper function and passed to this function.
    // It should be an empty graph and already synced with external graph.
    // For the alignment of outer and inner dynamic graph.
    virtual void get_graph_impl(std::shared_ptr<sc_graph_t> &graph) = 0;

    virtual std::shared_ptr<sc_graph_t> get_graph();

    static std::vector<graph_tensor_ptr> remake_logical_tensors(
            const std::vector<graph_tensor_ptr> &flts);

    float get_gflop() override { return get_graph()->get_gflop(); }

    static graph_tensor_ptr cast_input_dtype(graph_tensor_ptr &inp,
            std::shared_ptr<sc_graph_t> &graph, const any_map_t &attrs = {});

    static std::shared_ptr<sc_op> cast_output_dtype(graph_tensor_ptr &inp,
            std::shared_ptr<sc_graph_t> &graph, std::shared_ptr<sc_op> &last_op,
            const any_map_t &attrs = {});
};

class configurable_graph_op_t : public graph_op_t,
                                public op_traits::configurable_t {
public:
    configurable_graph_op_t() = default;

    configurable_graph_op_t(const std::string &op_name,
            const std::vector<graph_tensor_ptr> &producer_lt,
            const std::vector<graph_tensor_ptr> &consumer_lt,
            const any_map_t &attrs)
        : graph_op_t(op_name, producer_lt, consumer_lt, attrs) {}

    std::shared_ptr<sc_graph_t> get_graph() override;

    config_ptr get_config() override;

    void set_config(const config_ptr &config) override;

    reflection::shared_general_object_t get_default_config(
            context_ptr ctx) override;

    config_ptr_vec get_dynamic_config_candidates(
            const context_ptr &ctx) override {
        throw std::runtime_error("Unimplement.");
    }
    impl_kind_map convert_config_candidates_to_impl_map(
            const config_ptr_vec &configs) override {
        throw std::runtime_error("Unimplement.");
    }

protected:
    graph_config config_data_;
};

/**
 * The nested graph op
 * Used to convert a graph to a graph op which could be
 * reused in other graph.
 * Ins:
 *  - The corresponding input tensors of the nested graph
 * Outs:
 *  - The corresponding output tensors of the nested graph
 * Attrs:
 * - The attribute of the op
 * Graph:
 * - The graph used to convert a graph to a nested graph op.
 * */
class SC_INTERNAL_API nested_graph_op_t : public configurable_graph_op_t,
                                          public op_traits::copyable_t {
public:
    nested_graph_op_t(const std::string &op_name,
            const std::vector<graph_tensor_ptr> &ins,
            const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs,
            sc_graph_t &&graph);

    void get_graph_impl(std::shared_ptr<sc_graph_t> &graph) override;

    // linter has a false alarm to treat copy here as a STL function
    sc_op_ptr copy(const std::vector<graph_tensor_ptr> &ins, // NOLINT
            const std::vector<graph_tensor_ptr> &outs,
            sc_graph_t &mgr) override;

protected:
    sc_graph_t graph_;
};

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
