/*******************************************************************************
 * Copyright 2023 Intel Corporation
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
#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_OPS_REDUCE_GRAPH_OP_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_OPS_REDUCE_GRAPH_OP_HPP

#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "fusible/reduce.hpp"
#include <compiler/ir/graph/graph_op.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

class graph_reduce_base_t : public graph_op_t,
                            public op_traits::auto_copyable_t {
public:
    graph_reduce_base_t(graph_tensor_ptr v, const std::vector<int> &rd_axis,
            const std::string &op_name, bool keep_dims = false);
    graph_reduce_base_t(const std::vector<graph_tensor_ptr> &ins,
            const std::vector<graph_tensor_ptr> &outs,
            const std::string &op_name, const any_map_t &attrs);
    void query_format(context_ptr ctx,
            std::vector<std::vector<format_stride_pair>> &supported_ins,
            std::vector<std::vector<format_stride_pair>> &supported_outs)
            override;
};

class reduce_mean_op_t : public graph_reduce_base_t {
public:
    reduce_mean_op_t(graph_tensor_ptr v, const std::vector<int> &rd_axis,
            bool keep_dims = false)
        : graph_reduce_base_t(
                std::move(v), rd_axis, "reduce_mean", keep_dims) {};
    reduce_mean_op_t(const std::vector<graph_tensor_ptr> &ins,
            const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs)
        : graph_reduce_base_t(ins, outs, "reduce_mean", attrs) {};
    void get_graph_impl(std::shared_ptr<sc_graph_t> &graph) override;
};

class reduce_l1_op_t : public graph_reduce_base_t {
public:
    reduce_l1_op_t(graph_tensor_ptr v, const std::vector<int> &rd_axis,
            bool keep_dims = false)
        : graph_reduce_base_t(std::move(v), rd_axis, "reduce_l1", keep_dims) {};
    reduce_l1_op_t(const std::vector<graph_tensor_ptr> &ins,
            const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs)
        : graph_reduce_base_t(ins, outs, "reduce_l1", attrs) {};
    void get_graph_impl(std::shared_ptr<sc_graph_t> &graph) override;
};

class reduce_l2_op_t : public graph_reduce_base_t {
public:
    reduce_l2_op_t(graph_tensor_ptr v, const std::vector<int> &rd_axis,
            bool keep_dims = false)
        : graph_reduce_base_t(std::move(v), rd_axis, "reduce_l2", keep_dims) {};
    reduce_l2_op_t(const std::vector<graph_tensor_ptr> &ins,
            const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs)
        : graph_reduce_base_t(ins, outs, "reduce_l2", attrs) {};
    void get_graph_impl(std::shared_ptr<sc_graph_t> &graph) override;
};

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
