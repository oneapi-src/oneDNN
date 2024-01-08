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
#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_OPS_SOFTMAX_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_OPS_SOFTMAX_HPP

#include <memory>
#include <utility>
#include <vector>
#include <compiler/ir/graph/graph_op.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace ops {
/**
 * The softmax operator: exp(x)/sum(exp(x))
 * Inputs:
 *  - A single tensor
 * Outputs:
 *  - The result tensor
 * Attrs:
 *  - axis: vector<int> - The reduce axis for the "sum", see "reduce" op
 * */
class softmax_base_t : public graph_op_t, public op_traits::auto_copyable_t {
public:
    softmax_base_t(const std::vector<graph_tensor_ptr> &ins,
            const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs);
    void query_format(context_ptr ctx,
            std::vector<std::vector<format_stride_pair>> &supported_ins,
            std::vector<std::vector<format_stride_pair>> &supported_outs)
            override;
    std::vector<int> &get_axis() { return axis_; }
    void set_axis(const std::vector<int> &axis) { this->axis_ = axis; }
    graph_tensor_ptr get_stable_exp_inp(const graph_tensor_ptr &input,
            const std::vector<int> &axis, std::shared_ptr<sc_graph_t> &graph);

private:
    std::vector<int> axis_;
};

class softmax_op_t : public softmax_base_t {
public:
    softmax_op_t(const std::vector<graph_tensor_ptr> &ins,
            const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs)
        : softmax_base_t(ins, outs, attrs) {
        op_name_ = "softmax";
        COMPILE_ASSERT(info_.inputs_.size() == 1,
                "softmax op shall have only 1 inputs.")
    };
    void make_logical_tensor(std::vector<graph_tensor_ptr> &inputs,
            std::vector<graph_tensor_ptr> &outputs);
    std::pair<std::shared_ptr<sc_op>, std::shared_ptr<sc_op>> get_exp_reduce(
            std::shared_ptr<sc_graph_t> &graph, const graph_tensor_ptr &input,
            const std::vector<int> &axis);
    std::shared_ptr<sc_op> get_softmax_result(
            std::shared_ptr<sc_graph_t> &graph, const graph_tensor_ptr &input,
            const std::vector<int> &axis);
    void get_graph_impl(std::shared_ptr<sc_graph_t> &graph) override;
};

class softmax_bwd_op_t : public softmax_base_t {
public:
    softmax_bwd_op_t(const std::vector<graph_tensor_ptr> &ins,
            const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs)
        : softmax_base_t(ins, outs, attrs) {
        op_name_ = "softmax_bwd";
        COMPILE_ASSERT(info_.inputs_.size() == 2,
                "softmax backward op shall have only 2 inputs.")
    };
    void get_graph_impl(std::shared_ptr<sc_graph_t> &graph) override;
};

class log_softmax_op_t : public softmax_op_t {
public:
    log_softmax_op_t(const std::vector<graph_tensor_ptr> &ins,
            const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs)
        : softmax_op_t(ins, outs, attrs) {
        op_name_ = "log_softmax";
        COMPILE_ASSERT(info_.inputs_.size() == 1,
                "log softmax op shall have only 1 inputs.")
    };
    void get_graph_impl(std::shared_ptr<sc_graph_t> &graph) override;
};

class log_softmax_backward_op_t : public softmax_base_t {
public:
    log_softmax_backward_op_t(const std::vector<graph_tensor_ptr> &ins,
            const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs)
        : softmax_base_t(ins, outs, attrs) {
        op_name_ = "log_softmax_bwd";
        COMPILE_ASSERT(info_.inputs_.size() == 2,
                "log softmax backward op shall have only 2 inputs.")
    };
    void get_graph_impl(std::shared_ptr<sc_graph_t> &graph) override;
};

} // namespace ops
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
