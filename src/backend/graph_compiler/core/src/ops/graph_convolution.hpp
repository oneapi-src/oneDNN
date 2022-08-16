/*******************************************************************************
 * Copyright 2022 Intel Corporation
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
 ******************************************************************************/

#ifndef BACKEND_GRAPH_COMPILER_CORE_SRC_OPS_GRAPH_CONVOLUTION_HPP
#define BACKEND_GRAPH_COMPILER_CORE_SRC_OPS_GRAPH_CONVOLUTION_HPP

#include <memory>
#include <string>
#include <vector>

#include "compiler/ir/graph/graph.hpp"
#include "compiler/ir/graph/graph_op.hpp"
#include "compiler/ir/graph/traits.hpp"

namespace sc {
namespace ops {

class conv_fwd_op_t : public configurable_graph_op_t,
                      public op_traits::auto_copyable_t {
public:
    conv_fwd_op_t(const std::vector<graph_tensor_ptr> &ins,
            const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs);
    void get_graph_impl(std::shared_ptr<sc_graph_t> &graph) override;
    void query_format(context_ptr ctx,
            std::vector<std::vector<format_stride_pair>> &supported_ins,
            std::vector<std::vector<format_stride_pair>> &supported_outs)
            override;
    static sc_dims infer_out_dims(sc_graph_t &owner_graph,
            const sc_dims &input_dims, const sc_dims &filter_dims,
            const sc_dims &pads_begin, const sc_dims &pads_end,
            const sc_dims &strides, const std::string &data_format,
            const std::string &filter_format, bool is_same);
};

class conv_bwd_data_op_t : public configurable_graph_op_t,
                           public op_traits::auto_copyable_t {
public:
    conv_bwd_data_op_t(const std::vector<graph_tensor_ptr> &ins,
            const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs);
    void get_graph_impl(std::shared_ptr<sc_graph_t> &graph) override;
    void query_format(context_ptr ctx,
            std::vector<std::vector<format_stride_pair>> &supported_ins,
            std::vector<std::vector<format_stride_pair>> &supported_outs)
            override;
};

class conv_bwd_weight_op_t : public configurable_graph_op_t,
                             public op_traits::auto_copyable_t {
public:
    conv_bwd_weight_op_t(const std::vector<graph_tensor_ptr> &ins,
            const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs);
    void get_graph_impl(std::shared_ptr<sc_graph_t> &graph) override;
    void query_format(context_ptr ctx,
            std::vector<std::vector<format_stride_pair>> &supported_ins,
            std::vector<std::vector<format_stride_pair>> &supported_outs)
            override;
};

} // namespace ops
} // namespace sc

#endif
